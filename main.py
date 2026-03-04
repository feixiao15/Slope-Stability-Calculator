
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

        # View Mode Selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View Mode"))
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Main",
            "Evaluation",
        ])
        self.view_combo.currentIndexChanged.connect(self.on_view_change)
        view_layout.addWidget(self.view_combo)
        sidebar_layout.addLayout(view_layout)

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
        self.add_input(f1, "Friction Angle (φ') [°]", "10", "phi")
        self.add_input(f1, "Unit Weight (γ) [kN/m³]", "19.0", "gamma")
        self.add_input(f1, "Pore Pressure Ratio (ru)", "0.0", "ru") # Fellenius only
        g1.setLayout(f1)
        self.field_groups['ru'] = (f1.labelForField(self.inputs['ru']), self.inputs['ru'])

        # Group 2: Slope Geometry (Mixed)
        g2 = self.create_group("Slope Geometry", self.form_layout)
        f2 = QFormLayout()
        self.add_input(f2, "Slope Height (H) [m]", "5.0", "height")
        
        # Fellenius / Bishop / Janbu common geometry
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

        # Group 2.1: Evaluation only – 滑动面圆心
        g_center = self.create_group("Slip Surface Center", self.form_layout)
        f_center = QFormLayout()
        self.add_input(f_center, "Circle Center X (xc) [m]", "1.58", "center_x")
        self.add_input(f_center, "Circle Center Y (yc) [m]", "11.05", "center_y")
        g_center.setLayout(f_center)
        self.center_group = g_center
        # 默认在 Main 视图下隐藏，Evaluation 时显示
        self.center_group.setVisible(False)

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

    def on_view_change(self, index):
        """
        View mode:
        0 - Main（单方法分析，默认，与原来行为一致）
        1 - Evaluation（同时对比 Fellenius/Bishop/Janbu 并做 slices / iterations 实验）
        """
        self.reset_figure()
        if index == 0:
            self.on_method_change(self.method_combo.currentIndex())
            self.run_btn.setText("RUN ANALYSIS")
            if hasattr(self, "center_group"):
                self.center_group.setVisible(False)
            if hasattr(self, "method_combo"):
                self.method_combo.setVisible(True)
        else:
            self.result_label.setText("Evaluation | Compare Fellenius, Bishop, Janbu under same input.")
            self.result_label.setStyleSheet(
                "background-color: rgba(30, 30, 46, 0.8); color: #5a9fd4; font-size: 14px; "
                "padding: 10px; border: 1px solid #3e3e50; border-radius: 4px;"
            )
            self.run_btn.setText("RUN EVALUATION")
            if hasattr(self, "center_group"):
                self.center_group.setVisible(True)
            # Evaluation 视图下隐藏 Analysis Settings 与 Search Grid
            if hasattr(self, "sett_group"):
                self.sett_group.setVisible(False)
            if hasattr(self, "grid_group"):
                self.grid_group.setVisible(False)
            if hasattr(self, "method_combo"):
                self.method_combo.setVisible(False)

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
        view_idx = self.view_combo.currentIndex() if hasattr(self, "view_combo") else 0
        # Reset figure to ensure clean state (especially when switching modes)
        self.reset_figure()

        try:
            c = self.get_float('c')
            phi = self.get_float('phi')
            gamma = self.get_float('gamma')
            H = self.get_float('height')

            if view_idx == 0:
                if method_idx == 0:
                    self.run_fellenius(c, phi, gamma, H)
                elif method_idx == 1:
                    self.run_bishop(c, phi, gamma, H)
                elif method_idx == 2:
                    self.run_taylor(c, phi, gamma, H)
                else:
                    self.run_janbu(c, phi, gamma, H)
            else:
                self.run_evaluation(c, phi, gamma, H)
                
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
        
        best, results = analyzer.find_critical_fos(slices, gx, gy, plot=False)

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
            self.result_label.setText(f"Fellenius Method | Min FoS: {best['fos']:.3f} | Center: ({best['center'][0]:.2f}, {best['center'][1]:.2f}) | Radius: {best['radius']:.2f}")
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

        # 默认不再强制“只过坡脚”，而是在整个坡脚平台宽度 [-toe_ext, 0] 上遍历入口点
        if toe > 0:
            # 以约 0.5m 步长离散入口点，至少 3 个点
            step = 0.5
            n_entry = max(3, int(toe / step) + 1)
            entry_x_range = np.linspace(-toe, 0.0, n_entry)
        else:
            # toe_ext 非正时退回只过坡脚
            entry_x_range = None
        
        best, results = analyzer.find_critical_fos(
            slices,
            gx,
            gy,
            plot=False,
            entry_x_range=entry_x_range,
        )
        
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
            self.result_label.setText(f"Bishop Method | Min FoS: {best['fos']:.3f} | Center: ({best['center'][0]:.2f}, {best['center'][1]:.2f}) | Radius: {best['radius']:.2f}")
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

        # entrance point
        if toe > 0:
            step = 0.5
            n_entry = max(3, int(toe / step) + 1)
            entry_x_range = np.linspace(-toe, 0.0, n_entry)
        else:
            # bottom_extension 非正时退回只在坡趾入口
            entry_x_range = None

        # 3. Janbu GPS circular arc search (entrance point on the slope bottom, require_exit_at_crest=True)
        best, results = find_critical_fos_circular_arc(
            ground_profile=ground_profile,
            gamma=gamma,
            c_prime=c,
            phi_prime=phi,
            ru=ru,
            n_slices=slices,
            center_grid_x=gx,
            center_grid_y=gy,
            entry_x_range=entry_x_range,
            q=0.0,
            use_gps=True,
            gps_tolerance=1e-6,
            gps_max_iter=80,
            require_exit_at_crest=True,
        )

        # 4. draw FoS contour
        if results:
            Z = np.array([r[2] for r in results]).reshape(len(gx), len(gy)).T
            contour = self.ax.contourf(gx, gy, Z, levels=20, cmap="viridis_r", alpha=0.8)
            self.cbar = self.figure.colorbar(contour, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("FoS", color='white')
            self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        # 5. draw geometry 
        self.ax.plot(gp[:, 0], gp[:, 1], 'w-', linewidth=3)
        self.ax.fill_between(gp[:, 0], 0.0, gp[:, 1], color='#4a4a5a', alpha=0.5)

        # 6. draw most dangerous slip surface
        if best:
            circ = Circle(best['center'], best['radius'], fill=False, edgecolor='#ff4444', linewidth=2, linestyle='--')
            self.ax.add_patch(circ)
            self.ax.plot(best['center'][0], best['center'][1], 'rx')
            self.result_label.setText(f"Janbu GPS Method | Min FoS: {best['fos']:.3f} | Center: ({best['center'][0]:.2f}, {best['center'][1]:.2f}) | Radius: {best['radius']:.2f}")
        else:
            self.result_label.setText("Janbu GPS Method | No valid slip surface found.")

        self.canvas.draw()

    def run_evaluation(self, c, phi, gamma, H):
        """
        Evaluation 界面：在相同土性与几何、圆心坐标下，对比 Fellenius / Bishop / Janbu 三种方法，
        并考察：
          1) 不同条分数 slices (1–25, max iterations=20) 下三种方法的 FoS 变化；
          2) 不同迭代次数 iterations (0–50, slices=15 固定) 下 Bishop / Janbu 的 FoS 变化，
             同时绘制 Fellenius 的对比参考线。
        """
        ru = self.get_float('ru')
        ratio = self.get_float('ratio')
        toe = self.get_float('toe_ext')
        crest = self.get_float('crest_ext')
        base_slices = max(1, int(self.get_float('slices')))
        xc = self.get_float('center_x')
        yc = self.get_float('center_y')

        # 公共几何：Fellenius / Bishop 共享 1:1 坐标；Janbu 使用 GeometryBuilder
        # 1) 基线计算：同一圆心 (xc, yc)、同一 slices、迭代上限 20
        # Fellenius
        fell = FelleniusAnalyzer(c, phi, gamma, ru)
        fell.define_slope(H, ratio, toe_width=toe, crest_width=crest)
        dummy = np.array([0.0])
        fell_best, _ = fell.find_critical_fos(base_slices, dummy, dummy, plot=False, center=(xc, yc))
        fos_fell_base = fell_best["fos"] if fell_best else np.nan
        R_fell = fell_best["radius"] if fell_best else np.nan

        # Bishop（iterations 固定为 20）
        bishop_iter_eval = 20
        bish = BishopAnalyzer(c, phi, gamma, ru, iterations=bishop_iter_eval)
        bish.define_slope(H, ratio, toe_width=toe, crest_width=crest)
        bish_best, _ = bish.find_critical_fos(base_slices, dummy, dummy, plot=False, center=(xc, yc))
        fos_bish_base = bish_best["fos"] if bish_best else np.nan
        R_bish = bish_best["radius"] if bish_best else np.nan

        # Janbu：使用 GeometryBuilder + 单中心模式的 find_critical_fos_circular_arc（max_iter=20）
        gb = GeometryBuilder(slope_height=H, slope_ratio=ratio, bottom_extension=toe, top_extension=crest)
        ground_profile, _region = gb.build()
        gx_dummy = np.array([0.0])
        gy_dummy = np.array([0.0])
        janbu_best, _ = find_critical_fos_circular_arc(
            ground_profile=ground_profile,
            gamma=gamma,
            c_prime=c,
            phi_prime=phi,
            ru=ru,
            n_slices=base_slices,
            center_grid_x=gx_dummy,
            center_grid_y=gy_dummy,
            entry_x_range=None,
            q=0.0,
            use_gps=True,
            gps_tolerance=1e-6,
            gps_max_iter=20,
            require_exit_at_crest=True,
            center=(xc, yc),
            x_entry_single=None,
        )
        fos_janbu_base = janbu_best["fos"] if janbu_best else np.nan
        R_janbu = janbu_best["radius"] if janbu_best else np.nan

        # 在结果标签中输出类似表格的对比
        table_text = (
            "Evaluation | Most Dangerous Circle (single given center)\n"
            f"  Fellenius: FoS = {fos_fell_base:.3f} | R = {R_fell:.3f}\n"
            f"  Bishop   : FoS = {fos_bish_base:.3f} | R = {R_bish:.3f}\n"
            f"  Janbu GPS: FoS = {fos_janbu_base:.3f} | R = {R_janbu:.3f}\n"
            f"  Center (xc, yc) = ({xc:.3f}, {yc:.3f}),  H = {H:.3f}"
        )
        self.result_label.setText(table_text)
        self.result_label.setStyleSheet(
            "background-color: rgba(30, 30, 46, 0.9); color: #f0f0f0; font-size: 13px; "
            "padding: 10px; border: 1px solid #5a9fd4; border-radius: 4px;"
        )

        # 2) 绘制两组实验图：上 slices 实验，下 iterations 实验
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        color_fell = "#42a5f5"   
        color_bish = "#ffb74d"   
        color_janbu = "#66bb6a"  
        color_ref = "#bdbdbd"    

        # ---- 实验 1：不同 slices (1–25)，iter 上限统一为 20 ----
        max_slices = 25
        slices_list = list(range(1, max_slices + 1))
        fos_fell_list = []
        fos_bish_list = []
        fos_janbu_list = []

        for n in slices_list:
            # Fellenius
            fell_best_n, _ = fell.find_critical_fos(n, dummy, dummy, plot=False, center=(xc, yc))
            fos_fell_list.append(fell_best_n["fos"] if fell_best_n else np.nan)

            # Bishop（同一迭代上限 20）
            bish_n = BishopAnalyzer(c, phi, gamma, ru, iterations=bishop_iter_eval)
            bish_n.define_slope(H, ratio, toe_width=toe, crest_width=crest)
            bish_best_n, _ = bish_n.find_critical_fos(n, dummy, dummy, plot=False, center=(xc, yc))
            fos_bish_list.append(bish_best_n["fos"] if bish_best_n else np.nan)

            # Janbu
            janbu_best_n, _ = find_critical_fos_circular_arc(
                ground_profile=ground_profile,
                gamma=gamma,
                c_prime=c,
                phi_prime=phi,
                ru=ru,
                n_slices=n,
                center_grid_x=gx_dummy,
                center_grid_y=gy_dummy,
                entry_x_range=None,
                q=0.0,
                use_gps=True,
                gps_tolerance=1e-6,
                gps_max_iter=20,
                require_exit_at_crest=True,
                center=(xc, yc),
                x_entry_single=None,
            )
            fos_janbu_list.append(janbu_best_n["fos"] if janbu_best_n else np.nan)

        ax1.plot(slices_list, fos_fell_list, marker="o", color=color_fell, linewidth=2.0, label="Fellenius")
        ax1.plot(slices_list, fos_bish_list, marker="s", color=color_bish, linewidth=2.0, label="Bishop")
        ax1.plot(slices_list, fos_janbu_list, marker="^", color=color_janbu, linewidth=2.0, label="Janbu GPS")
        ax1.set_xlabel("Number of slices n")
        ax1.set_ylabel("FoS")
        ax1.set_title("Experiment 1: FoS vs slices (max iterations = 20)")
        ax1.grid(True, color="#444444", linestyle=":", alpha=0.7)
        ax1.legend()

        # ---- 实验 2：不同迭代次数 (0–50)，slices 固定 15 ----
        fixed_slices = 15
        iter_max = 50
        iter_list = list(range(0, iter_max + 1))
        fos_fell_iter = []
        fos_bish_iter = []
        fos_janbu_iter = []

        # Fellenius 在该实验中不随迭代变化，只作为水平参考线
        fell_fixed_best, _ = fell.find_critical_fos(fixed_slices, dummy, dummy, plot=False, center=(xc, yc))
        fos_fell_fixed = fell_fixed_best["fos"] if fell_fixed_best else np.nan

        for it in iter_list:
            fos_fell_iter.append(fos_fell_fixed)

            # Bishop：iterations 直接等于 it
            bish_it = BishopAnalyzer(c, phi, gamma, ru, iterations=it)
            bish_it.define_slope(H, ratio, toe_width=toe, crest_width=crest)
            bish_best_it, _ = bish_it.find_critical_fos(fixed_slices, dummy, dummy, plot=False, center=(xc, yc))
            fos_bish_iter.append(bish_best_it["fos"] if bish_best_it else np.nan)

            # Janbu：gps_max_iter = it
            janbu_best_it, _ = find_critical_fos_circular_arc(
                ground_profile=ground_profile,
                gamma=gamma,
                c_prime=c,
                phi_prime=phi,
                ru=ru,
                n_slices=fixed_slices,
                center_grid_x=gx_dummy,
                center_grid_y=gy_dummy,
                entry_x_range=None,
                q=0.0,
                use_gps=True,
                gps_tolerance=1e-6,
                gps_max_iter=max(1, it),  # 至少做一次 GPS 步，it=0 时近似为一次更新
                require_exit_at_crest=True,
                center=(xc, yc),
                x_entry_single=None,
            )
            fos_janbu_iter.append(janbu_best_it["fos"] if janbu_best_it else np.nan)

        ax2.plot(iter_list, fos_fell_iter, linestyle="--", color=color_ref, linewidth=1.8, label="Fellenius (reference)")
        ax2.plot(iter_list, fos_bish_iter, marker="s", markevery=5, color=color_bish, linewidth=2.0, label="Bishop")
        ax2.plot(iter_list, fos_janbu_iter, marker="^", markevery=5, color=color_janbu, linewidth=2.0, label="Janbu GPS")
        ax2.set_xlabel("Number of iterations")
        ax2.set_ylabel("FoS")
        ax2.set_title(f"Experiment 2: FoS vs iterations (slices = {fixed_slices})")
        ax2.grid(True, color="#444444", linestyle=":", alpha=0.7)
        ax2.legend()

        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")  # 略浅的深蓝背景，提高对比度
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#777777")

        self.figure.tight_layout()
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
