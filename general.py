import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class FelleniusAnalyzer:
    """
    一个通用的 Fellenius（常规条分法）边坡稳定分析工具。

    核心假设：
    1. 边坡几何被定义为简单的 坡高(H) 和 坡比(m, 1V:mH)。
    2. 只分析通过坡脚(Toe)的圆弧滑动面。
    3. 假定滑动面在坡顶的平坦地面上"穿出"。
    4. 土壤性质均匀。
    """

    def __init__(self, c_prime, phi_prime, gamma, r_u=0.0):
        """
        初始化土壤参数

        参数:
        c_prime (float): 有效粘聚力 (kPa or kN/m^2)
        phi_prime (float): 有效内摩擦角 (degrees)
        gamma (float): 土体重度 (kN/m^3)
        r_u (float): 孔隙水压力系数
        """
        self.c_prime = c_prime
        self.phi_prime = phi_prime
        self.phi_rad = np.radians(phi_prime)
        self.tan_phi = np.tan(self.phi_rad)
        self.gamma = gamma
        self.r_u = r_u
        print(f"--- 分析器已初始化 ---")
        print(f"c' = {c_prime} kPa, φ' = {phi_prime}°, γ = {gamma} kN/m³, r_u = {r_u}")

    def define_slope(self, height, ratio, toe_width=10, crest_width=20):
        """
        (步骤 1) 定义边坡几何

        参数:
        height (float): 边坡高度 H
        ratio (float): 坡比 m (1V:mH, e.g., 1.5)
        toe_width (float): 坡脚前平坦地面的长度
        crest_width (float): 坡顶后平坦地面的长度
        """
        self.height = height
        self.ratio = ratio
        self.crest_x = height * ratio

        # 定义一个(x, y)坐标列表来表示地表
        # (0,0) 点被设为坡脚 (Toe)
        self.surface_points = [
            (-toe_width, 0),
            (0, 0),  # 坡脚 (Toe)
            (self.crest_x, self.height),  # 坡顶 (Crest)
            (self.crest_x + crest_width, self.height)
        ]
        self.surface_poly = np.array(self.surface_points)
        print(f"边坡几何: H = {height}m, 坡比 = 1:{ratio}")
        print(f"坡脚: (0, 0), 坡顶: ({self.crest_x}, {self.height})")

    def _get_y_on_surface(self, x):
        """(辅助) 根据x坐标获取地表y坐标"""
        # 使用np.interp进行线性插值
        xp = self.surface_poly[:, 0]
        fp = self.surface_poly[:, 1]
        return np.interp(x, xp, fp)

    def _calculate_fos(self, slice_data):
        """
        (步骤 5) Fellenius FoS 计算引擎

        参数:
        slice_data (list of dicts): 由 _slice_mass 计算出的土条数据
        """
        numerator = 0.0  # 总抗滑力
        denominator = 0.0  # 总滑动力

        for s in slice_data:
            W_i = s['W']
            alpha_rad = s['alpha_rad']
            l_i = s['l']

            # 1. 滑动力 (分母)
            denominator += W_i * np.sin(alpha_rad)

            # 2. 抗滑力 (分子)
            cohesion_resistance = self.c_prime * l_i

            # u*l = (r_u * W / b) * l = (r_u * W / b) * (b / cos(alpha))
            # u*l = r_u * W / cos(alpha)
            # (注意: 这种 ru 定义是基于书本的简化)
            ul_i = self.r_u * W_i / np.cos(alpha_rad)

            # N_i' = W*cos(alpha) - u*l
            N_prime_i = (W_i * np.cos(alpha_rad)) - ul_i

            if N_prime_i < 0:
                N_prime_i = 0.0

            friction_resistance = N_prime_i * self.tan_phi

            numerator += cohesion_resistance + friction_resistance

        if denominator <= 0:
            return np.inf

        return numerator / denominator

    def _slice_mass(self, center, radius, n_slices):
        """
        (步骤 3 & 4) 自动切片并计算土条属性

        返回: 一个包含土条属性的列表，或 None (如果圆弧无效)
        """
        xc, yc = center

        # 1. 找到圆弧的 "穿出点" (x_exit)
        H = self.height
        term_sq = radius ** 2 - (H - yc) ** 2

        if term_sq < 0:
            return None  # 圆弧没有高到坡顶

        x_exit = xc + np.sqrt(term_sq)

        if x_exit < self.crest_x:
            return None  # 圆弧在坡面上穿出

        x_entry = 0.0  # 强制坡脚破坏

        total_width = x_exit - x_entry
        if total_width <= 0:
            return None

        b_width = total_width / n_slices  # 每个土条的宽度

        slices_data = []

        for i in range(n_slices):
            x_left = x_entry + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            y_base_sq_term = radius ** 2 - (x_mid - xc) ** 2
            if y_base_sq_term < 0:
                continue

            y_base = yc - np.sqrt(y_base_sq_term)

            # --- (BUG FIX) ---
            #
            # 修正 alpha (底面倾角) 的计算
            #
            # 原始错误代码: m_tan = (x_mid - xc) / (y_base - yc)
            #
            # 修正后代码 (基于圆的导数 dy/dx = -(x-xc)/(y-yc)):
            #
            # (y_base - yc) 是负数
            m_tan = -(x_mid - xc) / (y_base - yc)
            alpha_rad = np.arctan(m_tan)

            # --- (End Bug Fix) ---

            y_top = self._get_y_on_surface(x_mid)
            h_mid = y_top - y_base

            if h_mid < 0:
                continue

            W = h_mid * b_width * self.gamma  # 重量 (kN/m)

            l = b_width / np.cos(alpha_rad)

            slices_data.append({
                'W': W,
                'alpha_rad': alpha_rad,
                'l': l,
                'b': b_width,
                'x_mid': x_mid,
                'h_mid': h_mid
            })

        # 确保我们真的有土条（有时圆弧可能无效）
        if not slices_data:
            return None

        return slices_data

    def find_critical_fos(self, n_slices, center_grid_x, center_grid_y, plot=True):
        """
        (步骤 2, 5, 6) 搜索、迭代并找到最小安全系数

        参数:
        n_slices (int): 每个圆弧的切片数
        center_grid_x (np.array): 试探圆心的 x 坐标范围
        center_grid_y (np.array): 试探圆心的 y 坐标范围
        """
        print(f"\n--- 开始搜索 (Fellenius)... ---")
        print(
            f"搜索网格: {len(center_grid_x)} (x) * {len(center_grid_y)} (y) = {len(center_grid_x) * len(center_grid_y)} 个圆心")

        min_fos = np.inf
        best_circle = None

        fos_results = []

        for xc in center_grid_x:
            for yc in center_grid_y:
                # (步骤 2) 强制圆弧通过坡脚 (0,0)
                radius = np.sqrt(xc ** 2 + yc ** 2)

                # (步骤 3 & 4) 自动切片
                slice_data = self._slice_mass((xc, yc), radius, n_slices)

                if not slice_data:
                    fos_results.append((xc, yc, np.nan))
                    continue

                # (步骤 5) 计算 FoS
                fos = self._calculate_fos(slice_data)
                fos_results.append((xc, yc, fos))

                # (步骤 6) 寻优
                if fos < min_fos:
                    min_fos = fos
                    best_circle = {'center': (xc, yc), 'radius': radius, 'fos': fos}

        print(f"--- 搜索完成 ---")
        print(f"最小安全系数 (FoS_min): {min_fos:.3f}")

        # --- (FIX) ---
        # 增加一个检查，看是否找到了有效的圆
        if best_circle is None:
            print("\n!!! 错误：在定义的网格中没有找到任何有效的滑动面。")
            print("    请尝试调整 'center_grid_x' 和 'center_grid_y' 的搜索范围。")
        else:
            # 只有在找到圆时才打印详细信息和绘图
            print(f"最危险圆心 O(x,y): ({best_circle['center'][0]:.2f}, {best_circle['center'][1]:.2f})")
            print(f"最危险半径 R: {best_circle['radius']:.2f}")

            if plot:
                self.plot_result(best_circle, fos_results, center_grid_x, center_grid_y)

        return best_circle

    def plot_result(self, best_circle, fos_results, grid_x, grid_y):
        """(辅助) 可视化搜索结果"""

        fig, ax = plt.subplots(figsize=(14, 8))

        # 1. 绘制等高线图 (Heatmap)
        Z = np.array([r[2] for r in fos_results]).reshape(len(grid_x), len(grid_y)).T
        # Z = np.nan_to_num(Z, nan=10.0) #
        contours = ax.contourf(grid_x, grid_y, Z, levels=20, cmap="viridis_r", alpha=0.7)
        fig.colorbar(contours, ax=ax, label="Factor of Safety (FoS)")

        # 2. 绘制边坡几何
        ax.plot(self.surface_poly[:, 0], self.surface_poly[:, 1], 'k-', linewidth=3, label="surface")

        # 3. 绘制最危险滑动面
        center = best_circle['center']
        radius = best_circle['radius']
        fos = best_circle['fos']

        slip_circle = Circle(center, radius, fill=False, edgecolor='red',
                             linewidth=2, linestyle='--', label=f"Critical Circle (FoS={fos:.3f})")
        ax.add_patch(slip_circle)

        # 4. 绘制圆心
        ax.plot(center[0], center[1], 'r+', markersize=15, label="most dangerous circle center")

        # 5. 设置图形
        ax.set_title("Fellenius ")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal')

        # 设置合理的显示范围
        ax.set_xlim(self.surface_poly[0][0], self.surface_poly[-1][0])
        ax.set_ylim(self.surface_poly[0][1] - self.height * 0.5,
                    np.max(grid_y) + self.height * 0.5)

        plt.show()

# --- 1. 设置分析参数 ---

# 土壤参数
c_prime = 20.0
phi_prime = 15.0
gamma = 19.0
r_u = 0.4  # !! 使用题目中的 r_u = 0.4 !!

# 几何参数
slope_height = 6.0
slope_ratio = 1.5 # 1.5V : 1H

# 分析参数
num_slices = 5 # 使用例题中的土条数

# (步骤 2) 定义搜索网格
# 坡顶 x 在 6*1.5 = 9m 处。
# 我们在 x=[0, 20], y=[8, 20] 的范围内搜索圆心
grid_x = np.linspace(0, 30, 20)
grid_y = np.linspace(0, 30, 20)


# --- 2. 运行分析 ---

# 初始化分析器
analyzer = FelleniusAnalyzer(c_prime, phi_prime, gamma, r_u)

# 定义边坡
analyzer.define_slope(slope_height, slope_ratio)

# 寻找最危险滑动面
# 这将自动完成 步骤 2, 3, 4, 5, 6
critical_circle = analyzer.find_critical_fos(
    n_slices=num_slices,
    center_grid_x=grid_x,
    center_grid_y=grid_y,
    plot=True
)