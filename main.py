
import sys
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFormLayout, QGroupBox, QFrame, 
                             QScrollArea, QMessageBox, QStackedWidget, QDialog, QTextEdit)
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
from janbu import GeometryBuilder, find_critical_fos_circular_arc, calculate_fos_for_circular_arc

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

        # Group 2.2: Evaluation only – Experiment 1
        self.eval_groups = []
        g_eval_1 = self.create_group("Experiment 1: Slices", self.form_layout)
        f_eval_1 = QFormLayout()
        self.add_input(f_eval_1, "Slices Start", "1", "eval_slice_start")
        self.add_input(f_eval_1, "Slices End", "20", "eval_slice_end")
        g_eval_1.setLayout(f_eval_1)
        g_eval_1.setVisible(False)
        self.eval_groups.append(g_eval_1)

        # Group 2.3: Evaluation only – Experiment 2
        g_eval_2 = self.create_group("Experiment 2: Iterations", self.form_layout)
        f_eval_2 = QFormLayout()
        self.add_input(f_eval_2, "Iter Start", "0", "eval_iter_start")
        self.add_input(f_eval_2, "Iter End", "50", "eval_iter_end")
        self.add_input(f_eval_2, "Fixed Slices", "15", "eval_iter_fixed_slices")
        g_eval_2.setLayout(f_eval_2)
        g_eval_2.setVisible(False)
        self.eval_groups.append(g_eval_2)

        # Group 2.4: Evaluation only – Experiment 3
        g_eval_3 = self.create_group("Experiment 3: Cohesion Range", self.form_layout)
        f_eval_3 = QFormLayout()
        self.add_input(f_eval_3, "Cohesion Min [kPa]", "0.0", "eval_c_min")
        self.add_input(f_eval_3, "Cohesion Max [kPa]", "20.0", "eval_c_max")
        self.add_input(f_eval_3, "Cohesion Points", "21", "eval_c_points")
        g_eval_3.setLayout(f_eval_3)
        g_eval_3.setVisible(False)
        self.eval_groups.append(g_eval_3)

        # Group 2.5: Evaluation only – Experiment 4
        g_eval_4 = self.create_group("Experiment 4: Friction Angle Range", self.form_layout)
        f_eval_4 = QFormLayout()
        self.add_input(f_eval_4, "Friction Min [deg]", "0.0", "eval_phi_min")
        self.add_input(f_eval_4, "Friction Max [deg]", "50.0", "eval_phi_max")
        self.add_input(f_eval_4, "Friction Points", "26", "eval_phi_points")
        g_eval_4.setLayout(f_eval_4)
        g_eval_4.setVisible(False)
        self.eval_groups.append(g_eval_4)

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
        self.save_btn = QPushButton("SAVE EVALUATION")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedHeight(42)
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; font-weight: bold; font-size: 13px; border-radius: 6px; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #233246; color: #777; }
        """)
        self.save_btn.clicked.connect(self.save_evaluation_results)
        self.save_btn.setVisible(False)
        self.save_btn.setEnabled(False)
        sidebar_layout.addWidget(self.save_btn)
        main_layout.addWidget(sidebar_frame)

        # --- RIGHT SIDE ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        self.result_label = QLabel("Select parameters and run analysis.")
        self.result_label.setStyleSheet("background-color: rgba(30, 30, 46, 0.8); color: #5a9fd4; font-size: 14px; padding: 10px; border: 1px solid #3e3e50; border-radius: 4px;")
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.result_label, 1)
        self.details_btn = QPushButton("Calculation Details")
        self.details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_btn.setFixedHeight(32)
        self.details_btn.setStyleSheet("""
            QPushButton { background-color: #2b2b3b; color: #e0e0e0; border: 1px solid #3e3e50; border-radius: 4px; padding: 6px 10px; }
            QPushButton:hover { border: 1px solid #5a9fd4; color: white; }
            QPushButton:disabled { color: #777; border: 1px solid #2a2a35; background-color: #1e1e24; }
        """)
        self.details_btn.clicked.connect(self.show_calculation_details)
        top_bar.addWidget(self.details_btn, 0, Qt.AlignmentFlag.AlignRight)
        right_layout.addLayout(top_bar)
        
        self.figure = Figure(figsize=(10, 8), facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel)
        
        self.ax = self.figure.add_subplot(111)
        self.setup_plot_style()
        self.last_calc_details = None
        self.last_calc_method = ""
        self.last_evaluation_result = None
        
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
            if hasattr(self, "eval_groups"):
                for g in self.eval_groups:
                    g.setVisible(False)
            if hasattr(self, "method_combo"):
                self.method_combo.setVisible(True)
            if hasattr(self, "save_btn"):
                self.save_btn.setVisible(False)
        else:
            self.clear_calculation_details()
            self.last_evaluation_result = None
            self.result_label.setText("Evaluation | Compare Fellenius, Bishop, Janbu under same input.")
            self.result_label.setStyleSheet(
                "background-color: rgba(30, 30, 46, 0.8); color: #5a9fd4; font-size: 14px; "
                "padding: 10px; border: 1px solid #3e3e50; border-radius: 4px;"
            )
            self.run_btn.setText("RUN EVALUATION")
            if hasattr(self, "center_group"):
                self.center_group.setVisible(True)
            if hasattr(self, "eval_groups"):
                for g in self.eval_groups:
                    g.setVisible(True)
            # Evaluation 视图下隐藏 Analysis Settings 与 Search Grid
            if hasattr(self, "sett_group"):
                self.sett_group.setVisible(False)
            if hasattr(self, "grid_group"):
                self.grid_group.setVisible(False)
            if hasattr(self, "method_combo"):
                self.method_combo.setVisible(False)
            if hasattr(self, "save_btn"):
                self.save_btn.setVisible(True)
                self.save_btn.setEnabled(self.last_evaluation_result is not None)
        self.update_details_button_visibility()

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
        self.clear_calculation_details()
        self.update_details_button_visibility()

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

    def update_details_button_visibility(self):
        if not hasattr(self, "details_btn"):
            return
        view_idx = self.view_combo.currentIndex() if hasattr(self, "view_combo") else 0
        method_idx = self.method_combo.currentIndex() if hasattr(self, "method_combo") else 0
        supported = (view_idx == 0) and (method_idx in (0, 1, 3))
        self.details_btn.setVisible(supported)
        self.details_btn.setEnabled(supported and self.last_calc_details is not None)

    def clear_calculation_details(self):
        self.last_calc_details = None
        self.last_calc_method = ""
        self.update_details_button_visibility()

    def set_calculation_details(self, method_name, text):
        self.last_calc_method = str(method_name)
        self.last_calc_details = str(text)
        self.update_details_button_visibility()

    def show_calculation_details(self):
        if not self.last_calc_details:
            QMessageBox.information(self, "Calculation Details", "No calculation details available. Please run analysis first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Calculation Details - {self.last_calc_method}")
        dlg.resize(980, 720)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet("background-color: #151522; color: #e8e8e8; border: 1px solid #3e3e50; font-family: Consolas, 'Courier New', monospace; font-size: 12px;")
        txt.setPlainText(self.last_calc_details)
        lay.addWidget(txt)
        dlg.exec()

    def _format_fellenius_details(self, detail):
        lines = []
        lines.append("Method: Fellenius")
        lines.append("Key Formula:")
        lines.append("  FoS = Σ[c'l + N'tanφ'] / Σ[Wsinα],  N' = Wcosα - ul,  ul = ru*W/cosα")
        lines.append("")
        lines.append(f"Critical Circle: center=({detail['center'][0]:.4f}, {detail['center'][1]:.4f}), R={detail['radius']:.4f}")
        lines.append(f"FoS(min) = {detail['fos']:.6f}, Effective slices = {detail['n_slices']}")
        d = detail["detail"]
        lines.append(f"ΣResisting = {d['numerator']:.6f}, ΣSliding = {d['denominator']:.6f}")
        lines.append("")
        lines.append("Per-slice terms:")
        lines.append("  i | x_mid | h_mid | W | α(deg) | l | ul | N' | Cohesion | Friction | Resisting | Sliding")
        for s in d["slice_terms"]:
            lines.append(
                f"  {s['slice']:>2d} | {s['x_mid']:.4f} | {s['h_mid']:.4f} | {s['W']:.4f} | "
                f"{s['alpha_deg']:.4f} | {s['l']:.4f} | {s['u_l']:.4f} | {s['N_prime']:.4f} | "
                f"{s['cohesion_resistance']:.4f} | {s['friction_resistance']:.4f} | {s['resisting']:.4f} | {s['sliding']:.4f}"
            )
        return "\n".join(lines)

    def _format_bishop_details(self, detail):
        d = detail["detail"]
        lines = []
        lines.append("Method: Bishop")
        lines.append("Key Formula:")
        lines.append("  FoS = Σ{[c'b + (W(1-ru))tanφ'] * mα(F)} / Σ(Wsinα)")
        lines.append("  mα(F) = secα / [1 + tanφ*tanα/F]")
        lines.append("")
        lines.append(
            f"Critical Circle: center=({detail['center'][0]:.4f}, {detail['center'][1]:.4f}), "
            f"R={detail['radius']:.4f}, x_entry={detail['x_entry']:.4f}"
        )
        lines.append(
            f"FoS(min) = {detail['fos']:.6f}, Effective slices = {detail['n_slices']}, "
            f"Converged = {d.get('converged', False)}"
        )
        lines.append("")
        for it in d["iterations"]:
            lines.append(
                f"Iteration {it['iteration']:>2d}: F_old={it['fos_old']:.6f}, F_new={it['fos_new']:.6f}, "
                f"ΔF={it['delta']:.6e}, ΣResisting={it['numerator']:.6f}, ΣSliding={it['denominator']:.6f}"
            )
            lines.append("  i | W | α(deg) | b | mα | Resisting_i | Sliding_i")
            for s in it["slice_terms"]:
                lines.append(
                    f"  {s['slice']:>2d} | {s['W']:.4f} | {s['alpha_deg']:.4f} | {s['b']:.4f} | "
                    f"{s['m_alpha']:.6f} | {s['resisting']:.6f} | {s['sliding']:.6f}"
                )
            lines.append("")
        return "\n".join(lines)

    def _format_janbu_details(self, best, slices, meta):
        lines = []
        lines.append("Method: Janbu GPS")
        lines.append("Key Formula (iteration core):")
        lines.append("  F = Σ(A_i / nα_i) / Σ(B_i)")
        lines.append("  nα_i = cos²α_i * [1 + tanα_i*tanφ'_i/F]")
        lines.append("  A_i = [c'_i + (p_i + t_i - u_i)tanφ'_i]Δx_i,  B_i = ΔQ_i + (p_i + t_i)Δx_i tanα_i")
        lines.append("")
        lines.append(
            f"Critical Circle: center=({best['center'][0]:.4f}, {best['center'][1]:.4f}), "
            f"R={best['radius']:.4f}, x_entry={best.get('x_entry', 0.0):.4f}"
        )
        lines.append(
            f"FoS(min) = {best['fos']:.6f}, F0 = {meta.get('F0', np.nan):.6f}, "
            f"Converged = {meta.get('converged', False)}, Iterations = {meta.get('iterations', 0)}"
        )
        lines.append("")
        lines.append("Slice base data:")
        lines.append("  i | x_mid | width | α(deg) | y_top | y_base | h_mid | W | p")
        for i, s in enumerate(slices, start=1):
            lines.append(
                f"  {i:>2d} | {s['x_mid']:.4f} | {s['width']:.4f} | {s['alpha_deg']:.4f} | "
                f"{s['y_top']:.4f} | {s['y_base']:.4f} | {s['h_mid']:.4f} | {s['W']:.4f} | {s['p']:.4f}"
            )
        debug = meta.get("debug", None)
        if isinstance(debug, dict):
            F_hist = debug.get("F", [])
            t_hist = debug.get("t", [])
            E_hist = debug.get("E_interface", [])
            T_hist = debug.get("T_interface", [])
            lines.append("")
            lines.append("Iteration history:")
            if len(F_hist) == 0:
                lines.append("  (No GPS iteration detail returned)")
            prev = float(meta.get("F0", np.nan))
            for k, f_now in enumerate(F_hist, start=1):
                df = f_now - prev if np.isfinite(prev) else np.nan
                t_arr = np.asarray(t_hist[k - 1]) if k - 1 < len(t_hist) else np.array([])
                e_arr = np.asarray(E_hist[k - 1]) if k - 1 < len(E_hist) else np.array([])
                T_arr = np.asarray(T_hist[k - 1]) if k - 1 < len(T_hist) else np.array([])
                t_max = float(np.max(np.abs(t_arr))) if t_arr.size else np.nan
                e_end = float(e_arr[-1]) if e_arr.size else np.nan
                T_max = float(np.max(np.abs(T_arr))) if T_arr.size else np.nan
                lines.append(
                    f"  Iter {k:>2d}: F={float(f_now):.6f}, ΔF={float(df):.6e}, "
                    f"max|t|={t_max:.6f}, E_end={e_end:.6f}, max|T|={T_max:.6f}"
                )
                prev = float(f_now)
        return "\n".join(lines)

    def get_int(self, key, default=0):
        try:
            return int(float(self.inputs[key].text()))
        except Exception:
            return int(default)

    def _clamped_linspace(self, start, end, points, min_points=2):
        p = max(int(points), int(min_points))
        a, b = float(start), float(end)
        if b < a:
            a, b = b, a
        if abs(b - a) < 1e-12:
            b = a + 1e-6
        return np.linspace(a, b, p)

    def _apply_eval_plot_style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, color="#111111", pad=10)
        ax.set_xlabel(xlabel, color="#111111")
        ax.set_ylabel(ylabel, color="#111111")
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#d0d7de", linestyle="--", linewidth=0.8, alpha=0.9)
        ax.tick_params(colors="#1f2328")
        for spine in ax.spines.values():
            spine.set_edgecolor("#8c959f")

    def save_evaluation_results(self):
        if self.last_evaluation_result is None:
            QMessageBox.information(self, "Save Evaluation", "No evaluation results to save. Please run evaluation first.")
            return

        out_dir = Path(__file__).resolve().parent
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"evaluation_results_{stamp}"
        fig_path = base.with_suffix(".png")
        csv_path = base.with_suffix(".csv")
        meta_path = out_dir / f"evaluation_results_{stamp}_meta.txt"

        self.figure.savefig(fig_path, dpi=300, bbox_inches="tight")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "x", "fellenius", "bishop", "janbu"])
            for key in ("exp1", "exp2", "exp3", "exp4"):
                exp = self.last_evaluation_result.get(key, {})
                x = exp.get("x", [])
                fell = exp.get("fellenius", [])
                bish = exp.get("bishop", [])
                janbu = exp.get("janbu", [])
                for i in range(len(x)):
                    writer.writerow([
                        key,
                        x[i],
                        fell[i] if i < len(fell) else np.nan,
                        bish[i] if i < len(bish) else np.nan,
                        janbu[i] if i < len(janbu) else np.nan,
                    ])

        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("Evaluation Summary\n")
            f.write("==================\n")
            f.write(self.last_evaluation_result.get("summary", ""))
            f.write("\n")

        QMessageBox.information(
            self,
            "Save Evaluation",
            f"Saved successfully:\n- {fig_path}\n- {csv_path}\n- {meta_path}"
        )

    def run_analysis(self):
        method_idx = self.method_combo.currentIndex()
        view_idx = self.view_combo.currentIndex() if hasattr(self, "view_combo") else 0
        # Reset figure to ensure clean state (especially when switching modes)
        self.reset_figure()
        self.clear_calculation_details()

        try:
            c = self.get_float('c')
            phi = self.get_float('phi')
            gamma = self.get_float('gamma')
            H = self.get_float('height')

            if view_idx == 0:
                self.last_evaluation_result = None
                if hasattr(self, "save_btn"):
                    self.save_btn.setEnabled(False)
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
            detail = analyzer.calculate_circle_details(best['center'], slices)
            if detail is not None:
                self.set_calculation_details("Fellenius Method", self._format_fellenius_details(detail))
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
            x_entry_best = best.get("x_entry", 0.0)
            detail = analyzer.calculate_circle_details(best['center'], slices, x_entry=x_entry_best)
            if detail is not None:
                self.set_calculation_details("Bishop Method", self._format_bishop_details(detail))
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
        self.clear_calculation_details()

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
            fos_det, slices_det, meta_det = calculate_fos_for_circular_arc(
                ground_profile=ground_profile,
                gamma=gamma,
                c_prime=c,
                phi_prime=phi,
                ru=ru,
                n_slices=slices,
                center=best["center"],
                x_entry=best.get("x_entry", 0.0),
                q=0.0,
                require_exit_at_crest=True,
                use_gps=True,
                gps_tolerance=1e-6,
                gps_max_iter=80,
                print_iteration_table=False,
                return_debug=True,
            )
            if np.isfinite(fos_det) and slices_det is not None:
                self.set_calculation_details(
                    "Janbu GPS Method",
                    self._format_janbu_details(best, slices_det, meta_det),
                )
        else:
            self.result_label.setText("Janbu GPS Method | No valid slip surface found.")

        self.canvas.draw()

    def run_evaluation(self, c, phi, gamma, H):
        ru = self.get_float('ru')
        ratio = self.get_float('ratio')
        toe = self.get_float('toe_ext')
        crest = self.get_float('crest_ext')
        base_slices = max(1, int(self.get_float('slices')))
        xc = self.get_float('center_x')
        yc = self.get_float('center_y')

        slice_start = max(1, self.get_int("eval_slice_start", 1))
        slice_end = max(slice_start, self.get_int("eval_slice_end", 20))
        iter_start = max(0, self.get_int("eval_iter_start", 0))
        iter_end = max(iter_start, self.get_int("eval_iter_end", 50))
        iter_fixed_slices = max(1, self.get_int("eval_iter_fixed_slices", 15))
        c_vals = self._clamped_linspace(
            self.get_float("eval_c_min"),
            self.get_float("eval_c_max"),
            self.get_int("eval_c_points", 21),
            min_points=2,
        )
        phi_vals = self._clamped_linspace(
            self.get_float("eval_phi_min"),
            self.get_float("eval_phi_max"),
            self.get_int("eval_phi_points", 26),
            min_points=2,
        )

        iter_ref = max(1, iter_end)
        slices_list = list(range(slice_start, slice_end + 1))
        iter_list = list(range(iter_start, iter_end + 1))

        gb = GeometryBuilder(slope_height=H, slope_ratio=ratio, bottom_extension=toe, top_extension=crest)
        ground_profile, _region = gb.build()
        gx_dummy = np.array([0.0])
        gy_dummy = np.array([0.0])
        dummy = np.array([0.0])

        def calc_triplet(c_val, phi_val, n_slices, bishop_iter, janbu_iter):
            fell = FelleniusAnalyzer(c_val, phi_val, gamma, ru)
            fell.define_slope(H, ratio, toe_width=toe, crest_width=crest)
            fell_best, _ = fell.find_critical_fos(n_slices, dummy, dummy, plot=False, center=(xc, yc))
            fos_fell = float(fell_best["fos"]) if fell_best else np.nan

            bish = BishopAnalyzer(c_val, phi_val, gamma, ru, iterations=max(1, int(bishop_iter)))
            bish.define_slope(H, ratio, toe_width=toe, crest_width=crest)
            bish_best, _ = bish.find_critical_fos(n_slices, dummy, dummy, plot=False, center=(xc, yc))
            fos_bish = float(bish_best["fos"]) if bish_best else np.nan

            janbu_best, _ = find_critical_fos_circular_arc(
                ground_profile=ground_profile,
                gamma=gamma,
                c_prime=c_val,
                phi_prime=phi_val,
                ru=ru,
                n_slices=n_slices,
                center_grid_x=gx_dummy,
                center_grid_y=gy_dummy,
                entry_x_range=None,
                q=0.0,
                use_gps=True,
                gps_tolerance=1e-6,
                gps_max_iter=max(1, int(janbu_iter)),
                require_exit_at_crest=True,
                center=(xc, yc),
                x_entry_single=None,
            )
            fos_janbu = float(janbu_best["fos"]) if janbu_best else np.nan
            return fos_fell, fos_bish, fos_janbu

        # Base comparison
        fos_fell_base, fos_bish_base, fos_janbu_base = calc_triplet(
            c_val=c, phi_val=phi, n_slices=base_slices, bishop_iter=iter_ref, janbu_iter=iter_ref
        )

        summary_text = (
            "Evaluation | 4 experiments at fixed center\n"
            f"  Center (xc, yc)=({xc:.3f}, {yc:.3f}), H={H:.3f}\n"
            f"  Base FoS (slices={base_slices}, iter={iter_ref}): "
            f"Fellenius={fos_fell_base:.3f}, Bishop={fos_bish_base:.3f}, Janbu={fos_janbu_base:.3f}"
        )
        self.result_label.setText(summary_text)
        self.result_label.setStyleSheet(
            "background-color: rgba(30, 30, 46, 0.9); color: #f0f0f0; font-size: 13px; "
            "padding: 10px; border: 1px solid #5a9fd4; border-radius: 4px;"
        )

        # Experiment 1: FoS vs slices
        e1_f, e1_b, e1_j = [], [], []
        for n in slices_list:
            f, b, j = calc_triplet(c, phi, n_slices=n, bishop_iter=iter_ref, janbu_iter=iter_ref)
            e1_f.append(f); e1_b.append(b); e1_j.append(j)

        # Experiment 2: FoS vs iterations (slices fixed)
        e2_f, e2_b, e2_j = [], [], []
        f_ref, _, _ = calc_triplet(c, phi, n_slices=iter_fixed_slices, bishop_iter=iter_ref, janbu_iter=iter_ref)
        for it in iter_list:
            _, b, j = calc_triplet(c, phi, n_slices=iter_fixed_slices, bishop_iter=max(1, it), janbu_iter=max(1, it))
            e2_f.append(f_ref); e2_b.append(b); e2_j.append(j)

        # Experiment 3: FoS vs cohesion range
        e3_f, e3_b, e3_j = [], [], []
        for c_now in c_vals:
            f, b, j = calc_triplet(float(c_now), phi, n_slices=base_slices, bishop_iter=iter_ref, janbu_iter=iter_ref)
            e3_f.append(f); e3_b.append(b); e3_j.append(j)

        # Experiment 4: FoS vs friction angle range
        e4_f, e4_b, e4_j = [], [], []
        for phi_now in phi_vals:
            f, b, j = calc_triplet(c, float(phi_now), n_slices=base_slices, bishop_iter=iter_ref, janbu_iter=iter_ref)
            e4_f.append(f); e4_b.append(b); e4_j.append(j)

        # Plot 2x2 with light style (verification-like)
        self.figure.clear()
        self.figure.patch.set_facecolor("#ffffff")
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        ax1.plot(slices_list, e1_f, marker="o", linewidth=2.0, label="Fellenius")
        ax1.plot(slices_list, e1_b, marker="s", linewidth=2.0, label="Bishop")
        ax1.plot(slices_list, e1_j, marker="^", linewidth=2.0, label="Janbu GPS")
        self._apply_eval_plot_style(ax1, "Experiment 1: FoS vs slices", "Number of slices n", "FoS")
        tick_start = 5 * (slice_start // 5)
        if tick_start < slice_start:
            tick_start += 5
        ticks = np.arange(tick_start, slice_end + 1, 5, dtype=int)
        if ticks.size == 0 or ticks[0] != slice_start:
            ticks = np.unique(np.concatenate(([slice_start], ticks)))
        if ticks[-1] != slice_end:
            ticks = np.unique(np.concatenate((ticks, [slice_end])))
        ax1.set_xticks(ticks)
        ax1.legend()

        ax2.plot(iter_list, e2_f, linestyle="--", linewidth=1.8, label="Fellenius (reference)")
        ax2.plot(iter_list, e2_b, marker="s", markevery=max(1, len(iter_list)//10), linewidth=2.0, label="Bishop")
        ax2.plot(iter_list, e2_j, marker="^", markevery=max(1, len(iter_list)//10), linewidth=2.0, label="Janbu GPS")
        self._apply_eval_plot_style(ax2, f"Experiment 2: FoS vs iterations", "Number of iterations", "FoS")
        ax2.legend()

        ax3.plot(c_vals, e3_f, marker="o", linewidth=2.0, label="Fellenius")
        ax3.plot(c_vals, e3_b, marker="s", linewidth=2.0, label="Bishop")
        ax3.plot(c_vals, e3_j, marker="^", linewidth=2.0, label="Janbu GPS")
        self._apply_eval_plot_style(ax3, "Experiment 3: FoS vs cohesion", "Cohesion c' (kPa)", "FoS")
        ax3.legend()

        ax4.plot(phi_vals, e4_f, marker="o", linewidth=2.0, label="Fellenius")
        ax4.plot(phi_vals, e4_b, marker="s", linewidth=2.0, label="Bishop")
        ax4.plot(phi_vals, e4_j, marker="^", linewidth=2.0, label="Janbu GPS")
        self._apply_eval_plot_style(ax4, "Experiment 4: FoS vs friction angle", "Friction angle φ' (deg)", "FoS")
        ax4.legend()

        self.figure.tight_layout()
        self.canvas.draw()
        self.clear_calculation_details()

        self.last_evaluation_result = {
            "summary": summary_text,
            "exp1": {"x": slices_list, "fellenius": e1_f, "bishop": e1_b, "janbu": e1_j},
            "exp2": {"x": iter_list, "fellenius": e2_f, "bishop": e2_b, "janbu": e2_j},
            "exp3": {"x": [float(v) for v in c_vals], "fellenius": e3_f, "bishop": e3_b, "janbu": e3_j},
            "exp4": {"x": [float(v) for v in phi_vals], "fellenius": e4_f, "bishop": e4_b, "janbu": e4_j},
        }
        if hasattr(self, "save_btn"):
            self.save_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)
    window = SlopeStabilityApp()
    window.show()
    sys.exit(app.exec())
