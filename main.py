
import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFormLayout, QGroupBox, QFrame, 
                             QScrollArea, QMessageBox, QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

matplotlib.use('QtAgg')

from bishop import BishopAnalyzer
from fellenius import FelleniusAnalyzer
from taylor import TaylorSolver
from janbu import GeometryBuilder, find_critical_fos_circular_arc

# Fellenius / Taylor 的具体实现已移动到独立模块 `fellenius.py` / `taylor.py`，
# 这里不再重复定义，仅在后续 GUI 逻辑中直接实例化并调用这些 solver。

# ==========================================
# 3. GUI IMPLEMENTATION
# ==========================================

class ModernInput(QLineEdit):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLineEdit { background-color: #2b2b3b; color: white; border: 1px solid #3e3e50; border-radius: 4px; padding: 5px; font-size: 13px; }
            QLineEdit:focus { border: 1px solid #5a9fd4; }
            QLineEdit:disabled { background-color: #1e1e24; color: #555; border: 1px solid #2a2a35; }
        """)

class SlopeStabilityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SlopeStability Pro")
        self.resize(1300, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #3e3e50; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QComboBox { background-color: #2b2b3b; color: white; border: 1px solid #3e3e50; border-radius: 4px; padding: 5px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #2b2b3b; color: white; selection-background-color: #3e3e50; }
            QComboBox::item { color: white; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT SIDEBAR ---
        sidebar_frame = QFrame()
        sidebar_frame.setStyleSheet("background-color: #181825; border-radius: 8px;")
        sidebar_frame.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar_frame)

        # Header
        title = QLabel("SlopeStability Pro")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        sidebar_layout.addWidget(title)

        # Method Selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Calculation Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Fellenius Method",
            "Bishop Method",
            "Taylor Stability Method",
            "Janbu GPS Method",
        ])
        self.method_combo.currentIndexChanged.connect(self.on_method_change)
        method_layout.addWidget(self.method_combo)
        sidebar_layout.addLayout(method_layout)

        # Inputs Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll_content = QWidget()
        self.form_layout = QVBoxLayout(scroll_content)
        self.form_layout.setSpacing(15)

        self.inputs = {}
        self.field_groups = {} # To toggle visibility

        # Group 1: Soil Properties (Common)
        g1 = self.create_group("Soil Properties", self.form_layout)
        f1 = QFormLayout()
        self.add_input(f1, "Cohesion (c') [kPa]", "20.0", "c")
        self.add_input(f1, "Friction Angle (φ') [°]", "20.0", "phi")
        self.add_input(f1, "Unit Weight (γ) [kN/m³]", "18.0", "gamma")
        self.add_input(f1, "Pore Pressure Ratio (ru)", "0.0", "ru") # Fellenius only
        g1.setLayout(f1)
        self.field_groups['ru'] = (f1.labelForField(self.inputs['ru']), self.inputs['ru'])

        # Group 2: Slope Geometry (Mixed)
        g2 = self.create_group("Slope Geometry", self.form_layout)
        f2 = QFormLayout()
        self.add_input(f2, "Slope Height (H) [m]", "5.0", "height")
        
        # Fellenius specific geometry
        self.add_input(f2, "Slope Ratio (1V:mH)", "0.363", "ratio") 
        self.add_input(f2, "Toe Extension [m]", "5.0", "toe_ext")
        self.add_input(f2, "Crest Extension [m]", "30.0", "crest_ext")
        
        # Taylor specific geometry
        self.add_input(f2, "Slope Angle (β) [°]", "30.0", "beta")
        self.add_input(f2, "Depth Factor (D)", "1.0", "depth_factor")
        
        g2.setLayout(f2)
        
        # Store for toggling
        self.field_groups['ratio'] = (f2.labelForField(self.inputs['ratio']), self.inputs['ratio'])
        self.field_groups['toe_ext'] = (f2.labelForField(self.inputs['toe_ext']), self.inputs['toe_ext'])
        self.field_groups['crest_ext'] = (f2.labelForField(self.inputs['crest_ext']), self.inputs['crest_ext'])
        self.field_groups['beta'] = (f2.labelForField(self.inputs['beta']), self.inputs['beta'])
        self.field_groups['D'] = (f2.labelForField(self.inputs['depth_factor']), self.inputs['depth_factor'])

        # Group 3: Analysis Settings (Fellenius only)
        self.sett_group = self.create_group("Analysis Settings", self.form_layout)
        f3 = QFormLayout()
        self.add_input(f3, "Number of Slices", "10", "slices")
        self.sett_group.setLayout(f3)

        # Group 4: Search Grid (Fellenius only)
        self.grid_group = self.create_group("Search Grid (Center)", self.form_layout)
        f4 = QFormLayout()
        self.add_input(f4, "X Start [m]", "0.0", "grid_x_start")
        self.add_input(f4, "X End [m]", "30.0", "grid_x_end")
        self.add_input(f4, "Y Start [m]", "0.0", "grid_y_start")
        self.add_input(f4, "Y End [m]", "30.0", "grid_y_end")
        self.add_input(f4, "Resolution", "20", "grid_res")
        self.grid_group.setLayout(f4)

        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)

        # Run Button
        self.run_btn = QPushButton("RUN ANALYSIS")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; font-weight: bold; font-size: 14px; border-radius: 6px; }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        sidebar_layout.addWidget(self.run_btn)
        main_layout.addWidget(sidebar_frame)

        # --- RIGHT SIDE ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        self.result_label = QLabel("Select parameters and run analysis.")
        self.result_label.setStyleSheet("background-color: rgba(30, 30, 46, 0.8); color: #5a9fd4; font-size: 14px; padding: 10px; border: 1px solid #3e3e50; border-radius: 4px;")
        right_layout.addWidget(self.result_label)
        
        self.figure = Figure(figsize=(10, 8), facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel)
        
        self.ax = self.figure.add_subplot(111)
        self.setup_plot_style()
        
        # Initialize visibility
        self.on_method_change(0)

    def create_group(self, title, parent_layout):
        group = QGroupBox(title)
        parent_layout.addWidget(group)
        return group

    def add_input(self, layout, label_text, default_val, key):
        inp = ModernInput(default_val)
        layout.addRow(label_text, inp)
        self.inputs[key] = inp

    def toggle_field(self, key, visible):
        lbl, widget = self.field_groups[key]
        lbl.setVisible(visible)
        widget.setVisible(visible)

    def on_method_change(self, index):
        # index: 0=Fellenius, 1=Bishop, 2=Taylor, 3=Janbu
        is_taylor = (index == 2)
        # Fellenius & Bishop: slices + grid + geometry
        self.sett_group.setVisible(not is_taylor)
        self.grid_group.setVisible(not is_taylor)
        self.toggle_field('ru', not is_taylor)
        self.toggle_field('ratio', not is_taylor)
        self.toggle_field('toe_ext', not is_taylor)
        self.toggle_field('crest_ext', not is_taylor)
        
        # Taylor specific
        self.toggle_field('beta', is_taylor)
        self.toggle_field('D', is_taylor)
        
        # Initialize right side plot area
        self.reset_figure()
        method_names = [
            "Fellenius Method",
            "Bishop Method",
            "Taylor Stability Method",
            "Janbu GPS Method",
        ]
        method_name = method_names[index]
        self.result_label.setText(f"{method_name} | Select parameters and run analysis.")
        self.result_label.setStyleSheet("background-color: rgba(30, 30, 46, 0.8); color: #5a9fd4; font-size: 14px; padding: 10px; border: 1px solid #3e3e50; border-radius: 4px;")

    def reset_figure(self):
        """Reset figure to single subplot state"""
        # Remove colorbar before clearing figure
        if hasattr(self, 'cbar') and self.cbar:
            try:
                self.cbar.remove()
            except (KeyError, AttributeError):
                pass  # Colorbar may already be removed
            self.cbar = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.setup_plot_style()

    def setup_plot_style(self):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e2e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        for spine in self.ax.spines.values(): spine.set_edgecolor('#555555')
        self.ax.grid(True, linestyle=':', alpha=0.3, color='white')
        self.ax.set_aspect('equal')
        self.canvas.draw()

    def get_float(self, key):
        try: return float(self.inputs[key].text())
        except: return 0.0

    def run_analysis(self):
        method_idx = self.method_combo.currentIndex()
        # Reset figure to ensure clean state (especially when switching from Taylor to Fellenius)
        self.reset_figure()

        try:
            c = self.get_float('c')
            phi = self.get_float('phi')
            gamma = self.get_float('gamma')
            H = self.get_float('height')

            if method_idx == 0:
                self.run_fellenius(c, phi, gamma, H)
            elif method_idx == 1:
                self.run_bishop(c, phi, gamma, H)
            elif method_idx == 2:
                self.run_taylor(c, phi, gamma, H)
            else:
                self.run_janbu(c, phi, gamma, H)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_fellenius(self, c, phi, gamma, H):
        ru = self.get_float('ru')
        ratio = self.get_float('ratio')
        toe = self.get_float('toe_ext')
        crest = self.get_float('crest_ext')
        slices = int(self.get_float('slices'))
        
        analyzer = FelleniusAnalyzer(c, phi, gamma, ru)
        analyzer.define_slope(H, ratio, toe, crest)
        
        gx = np.linspace(self.get_float('grid_x_start'), self.get_float('grid_x_end'), int(self.get_float('grid_res')))
        gy = np.linspace(self.get_float('grid_y_start'), self.get_float('grid_y_end'), int(self.get_float('grid_res')))
        
        best, results = analyzer.find_critical_fos(slices, gx, gy)

        # Plot Heatmap
        if results:
            Z = np.array([r[2] for r in results]).reshape(len(gx), len(gy)).T
            contour = self.ax.contourf(gx, gy, Z, levels=20, cmap="viridis_r", alpha=0.8)
            self.cbar = self.figure.colorbar(contour, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("FoS", color='white')
            self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # Plot Geometry
        poly = analyzer.surface_poly
        self.ax.plot(poly[:,0], poly[:,1], 'w-', linewidth=3)
        self.ax.fill_between(poly[:,0], -10, poly[:,1], color='#4a4a5a', alpha=0.5)

        if best:
            circ = Circle(best['center'], best['radius'], fill=False, edgecolor='#ff4444', linewidth=2, linestyle='--')
            self.ax.add_patch(circ)
            self.ax.plot(best['center'][0], best['center'][1], 'rx')
            self.result_label.setText(f"Fellenius Method | Min FoS: {best['fos']:.3f}")
        else:
            self.result_label.setText("Fellenius Method | No valid slip surface found.")
        
        self.canvas.draw()

    def run_bishop(self, c, phi, gamma, H):
        ru = self.get_float('ru')
        ratio = self.get_float('ratio')
        toe = self.get_float('toe_ext')
        crest = self.get_float('crest_ext')
        slices = int(self.get_float('slices'))
        
        analyzer = BishopAnalyzer(c, phi, gamma, ru)
        analyzer.define_slope(H, ratio, toe_width=toe, crest_width=crest)
        
        gx = np.linspace(self.get_float('grid_x_start'), self.get_float('grid_x_end'), int(self.get_float('grid_res')))
        gy = np.linspace(self.get_float('grid_y_start'), self.get_float('grid_y_end'), int(self.get_float('grid_res')))
        
        best, results = analyzer.find_critical_fos(slices, gx, gy, plot=False)
        
        if results:
            Z = np.array([r[2] for r in results]).reshape(len(gx), len(gy)).T
            contour = self.ax.contourf(gx, gy, Z, levels=20, cmap="viridis_r", alpha=0.8)
            self.cbar = self.figure.colorbar(contour, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("FoS", color='white')
            self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        
        poly = analyzer.surface_poly
        self.ax.plot(poly[:, 0], poly[:, 1], 'w-', linewidth=3)
        self.ax.fill_between(poly[:, 0], -10, poly[:, 1], color='#4a4a5a', alpha=0.5)
        
        if best:
            circ = Circle(best['center'], best['radius'], fill=False, edgecolor='#ff4444', linewidth=2, linestyle='--')
            self.ax.add_patch(circ)
            self.ax.plot(best['center'][0], best['center'][1], 'rx')
            self.result_label.setText(f"Bishop Method | Min FoS: {best['fos']:.3f}")
        else:
            self.result_label.setText("Bishop Method | No valid slip surface found.")
        
        self.canvas.draw()

    def plot_taylor_charts(self, solver, phi_in, beta_in, D_in, N_res):
        """
        Plot two Taylor charts
        """
        self.figure.clear()

        text_color = 'white'
        grid_color = '#555555'
        line_color_inactive = '#666666'
        line_color_active = '#5a9fd4'
        highlight_color = '#ff4444' 
        
        ax1 = self.figure.add_subplot(121) # Chart 1: Phi > 0
        ax2 = self.figure.add_subplot(122) # Chart 2: Phi = 0 (Depth Factor)
        
        # Determine which mode we're in
        is_depth_mode = (phi_in == 0 and beta_in < 53)
        active_ax = ax2 if is_depth_mode else ax1

        # Plot Chart 1 
        ax1.set_title("Chart 1: Standard", color=text_color, pad=15)
        ax1.set_xlabel("Slope Angle β (deg)", color=text_color)
        ax1.set_ylabel("Stability Number N", color=text_color)
        chart1_data = solver.chart.chart1_data
        sorted_phis = sorted(chart1_data.keys())
        
        for p in sorted_phis:
            bs, ns = chart1_data[p]
            # Bold if near the current input phi
            lw = 2 if abs(p - phi_in) < 2.5 else 1
            alpha = 1.0 if not is_depth_mode else 0.3
            color = line_color_active if not is_depth_mode else line_color_inactive
            
            ax1.plot(bs, ns, color=color, alpha=alpha, linewidth=lw, label=f"φ={p}°")
            # End-of-line label
            if not is_depth_mode:
                ax1.text(bs[-1], ns[-1], f" {p}°", color=color, fontsize=8, verticalalignment='center')

        ax1.set_xlim(0, 90)
        ax1.set_ylim(0, 0.30)
        ax1.grid(True, color=grid_color, linestyle=':', alpha=0.5)

        # Plot Chart 2 
        ax2.set_title("Chart 2: Depth Factor (φ=0, β<53°)", color=text_color, pad=15)
        ax2.set_xlabel("Depth Factor D", color=text_color)
        ax2.set_ylabel("Stability Number N", color=text_color)
        
        d_range = np.linspace(1.0, 4.0, 50)
        coeffs = solver.chart.depth_coeffs
        sorted_betas = sorted(coeffs.keys(), reverse=True) 
        
        for b in sorted_betas:
            # Plot fitted curve
            A, k = coeffs[b]
            n_vals = [max(0, 0.181 - A * (d**-k)) for d in d_range]
            
            lw = 2 if abs(b - beta_in) < 5 else 1
            alpha = 1.0 if is_depth_mode else 0.3
            color = line_color_active if is_depth_mode else line_color_inactive
            
            ax2.plot(d_range, n_vals, color=color, alpha=alpha, linewidth=lw)
            # Label
            if is_depth_mode:
                ax2.text(d_range[-1], n_vals[-1], f" β={b}°", color=color, fontsize=8, verticalalignment='center')

        ax2.set_xlim(1, 4)
        ax2.set_ylim(0.10, 0.20)
        ax2.grid(True, color=grid_color, linestyle=':', alpha=0.5)

        # Highlight active chart
        for spine in active_ax.spines.values():
            spine.set_edgecolor(highlight_color)
            spine.set_linewidth(2.5)   
        # Annotate result point
        if is_depth_mode:
            # Annotate point (D, N)
            target_x = D_in
            target_y = N_res
            ax2.plot(target_x, target_y, 'x', color=highlight_color, markersize=12, markeredgewidth=3, zorder=10)
            ax2.annotate(f"N={N_res:.3f}\n(D={target_x})", xy=(target_x, target_y), xytext=(target_x+0.2, target_y+0.01),
                         arrowprops=dict(facecolor=highlight_color, shrink=0.05, headwidth=8, width=2),
                         color=highlight_color, fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e2e", ec=highlight_color, alpha=0.9))
        else:
            # Annotate point (Beta, N)
            target_x = beta_in
            target_y = N_res
            ax1.plot(target_x, target_y, 'x', color=highlight_color, markersize=12, markeredgewidth=3, zorder=10)
            # Add arrow
            ax1.annotate(f"N={N_res:.3f}\n(β={target_x}°)", xy=(target_x, target_y), xytext=(target_x-20, target_y+0.03),
                         arrowprops=dict(facecolor=highlight_color, shrink=0.05, headwidth=8, width=2),
                         color=highlight_color, fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e2e", ec=highlight_color, alpha=0.9))

        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e1e2e')
            ax.tick_params(colors=text_color)
            if ax != active_ax:
                # Dim border of inactive chart
                for spine in ax.spines.values():
                    spine.set_edgecolor(grid_color)
                    
        self.figure.tight_layout()
        self.canvas.draw()

    def run_taylor(self, c, phi, gamma, H):
        beta = self.get_float('beta')
        D = self.get_float('depth_factor')
        
        solver = TaylorSolver()
        Fs, N = solver.solve(c, phi, gamma, beta, H, D)
        self.plot_taylor_charts(solver, phi, beta, D, N)
        self.result_label.setText(f"Taylor Method | Stability Number N: {N:.3f} | Factor of Safety: {Fs:.3f}")
        if N < 1e-4 or Fs > 100:
             self.result_label.setStyleSheet("color: #5a9fd4; background-color: rgba(30, 30, 46, 0.9); padding: 10px; border: 1px solid #5a9fd4; border-radius: 4px;")
        else:
             self.result_label.setStyleSheet("color: #2ecc71; background-color: rgba(30, 30, 46, 0.9); font-weight: bold; padding: 10px; border: 1px solid #2ecc71; border-radius: 4px;")

    def run_janbu(self, c, phi, gamma, H):
        """
        使用 Janbu GPS 圆弧搜索接口，在给定圆心搜索网格内寻找最小 FoS。
        几何与输入参数风格与 Bishop/Fellenius 保持一致：
        - H: 坡高
        - ratio: 坡度比 1V:mH
        - toe_ext: 坡脚前平台长度 -> GeometryBuilder.bottom_extension
        - crest_ext: 坡顶后平台长度 -> GeometryBuilder.top_extension
        - slices: 条分数
        - grid_x/y_*: 圆心搜索范围
        """
        ru = self.get_float('ru')
        ratio = self.get_float('ratio')
        toe = self.get_float('toe_ext')
        crest = self.get_float('crest_ext')
        slices = int(self.get_float('slices'))

        # 1. 构建 Janbu 几何（ground_profile 供计算，region 可用于填充）
        gb = GeometryBuilder(slope_height=H, slope_ratio=ratio, bottom_extension=toe, top_extension=crest)
        ground_profile, _region = gb.build()
        gp = np.asarray(ground_profile, dtype=float)

        # 2. 圆心搜索网格（与 Bishop/Fellenius 一样）
        gx = np.linspace(self.get_float('grid_x_start'), self.get_float('grid_x_end'), int(self.get_float('grid_res')))
        gy = np.linspace(self.get_float('grid_y_start'), self.get_float('grid_y_end'), int(self.get_float('grid_res')))

        # 3. Janbu GPS 圆弧搜索（入口默认取坡脚点，require_exit_at_crest=True）
        best, results = find_critical_fos_circular_arc(
            ground_profile=ground_profile,
            gamma=gamma,
            c_prime=c,
            phi_prime=phi,
            ru=ru,
            n_slices=slices,
            center_grid_x=gx,
            center_grid_y=gy,
            entry_x_range=None,        # 默认仅从坡脚出发
            q=0.0,
            use_gps=True,
            gps_tolerance=1e-6,
            gps_max_iter=80,
            require_exit_at_crest=True,
        )

        # 4. 画 FoS 等高线
        if results:
            Z = np.array([r[2] for r in results]).reshape(len(gx), len(gy)).T
            contour = self.ax.contourf(gx, gy, Z, levels=20, cmap="viridis_r", alpha=0.8)
            self.cbar = self.figure.colorbar(contour, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("FoS", color='white')
            self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # 5. 画几何（仅地表线，填充至 y=0）
        self.ax.plot(gp[:, 0], gp[:, 1], 'w-', linewidth=3)
        self.ax.fill_between(gp[:, 0], 0.0, gp[:, 1], color='#4a4a5a', alpha=0.5)

        # 6. 画最危险滑动圆弧
        if best:
            circ = Circle(best['center'], best['radius'], fill=False, edgecolor='#ff4444', linewidth=2, linestyle='--')
            self.ax.add_patch(circ)
            self.ax.plot(best['center'][0], best['center'][1], 'rx')
            self.result_label.setText(f"Janbu GPS Method | Min FoS: {best['fos']:.3f}")
        else:
            self.result_label.setText("Janbu GPS Method | No valid slip surface found.")

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)
    window = SlopeStabilityApp()
    window.show()
    sys.exit(app.exec())
