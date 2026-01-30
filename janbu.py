import numpy as np
import matplotlib.pyplot as plt


class GeometryBuilder:
    """
    Janbu GPS – 模块一：几何构建器 (Geometry Builder)

    根据设计文档：
    - 输入: 坡高 H、坡度比 m(1V:mH)、坡脚前/后水平延伸 L_bot, L_top
    - 坐标原点 O(0,0) 取在模型最左下角
    - 输出:
        * Ground_Profile: [(0,0), (L_bot,0), (C_x,H), (D_x,H)]
        * Slope_Region:   与 Ground_Profile 相同的四边形多边形，用于约束滑裂面不跑出土体
    """

    def __init__(self, slope_height, slope_ratio, bottom_extension, top_extension):
        """
        Parameters
        ----------
        slope_height : float
            坡高 H (m)
        slope_ratio : float
            坡度比 m (1V:mH，即竖向 1、水平 m)
        bottom_extension : float
            坡脚前水平延伸 L_bot (m)
        top_extension : float
            坡顶后水平延伸 L_top (m)
        """
        self.H = float(slope_height)
        self.m = float(slope_ratio)
        self.L_bot = float(bottom_extension)
        self.L_top = float(top_extension)

        if self.H <= 0 or self.m <= 0:
            raise ValueError("slope_height 和 slope_ratio 必须为正数。")

    def build(self):
        """
        构建地表轮廓与土体区域。

        Returns
        -------
        ground_profile : list[tuple[float, float]]
            A-B-C-D 四个顶点的坐标:
              A = (0, 0)
              B = (L_bot, 0)
              C = (L_bot + H * m, H)
              D = (L_bot + H * m + L_top, H)
        slope_region : np.ndarray
            形如 [[x0,y0], [x1,y1], ...] 的多边形顶点数组。
        """
        A = (0.0, 0.0)
        B = (self.L_bot, 0.0)
        C_x = self.L_bot + self.H * self.m
        C = (C_x, self.H)
        D_x = C_x + self.L_top
        D = (D_x, self.H)

        ground_profile = [A, B, C, D]

        # 土体区域多边形（用于可视化/约束）：沿地表走一圈，并在底部 y=0 封闭
        # 这样填充时不会出现 D->A 的对角线穿过土体内部。
        # A -> B -> C -> D -> (D_x, 0) -> A
        slope_region = np.array([A, B, C, D, (D_x, 0.0)])
        return ground_profile, slope_region


class JanbuPreprocessor:
    """
    Janbu GPS – 模块三：预处理 / 条分与几何计算 (Preprocessing)

    本类只负责几何与基本重力参数的计算，不做 FoS 计算。
    当前实现重点是得到每个土条的:
        - width  Δx_i
        - alpha  α_i  (底面倾角, 弧度 & 角度)
        - weight W_i
        - p_i = W_i / Δx_i (暂不考虑超载, 可在调用处自行加上 q)

    为简洁起见，暂按单一均质土层处理，分层扩展时可在计算 W_i 时按深度分段积分。
    """

    def __init__(self, gamma):
        """
        Parameters
        ----------
        gamma : float
            土体重度 γ (kN/m³)
        """
        self.gamma = float(gamma)

    @staticmethod
    def _interp_y(xs, ys, x):
        """在线性折线 (xs, ys) 上插值 y(x)。"""
        return float(np.interp(x, xs, ys))

    @staticmethod
    def _interp_y_and_slope(xs, ys, x):
        """
        在折线 (xs, ys) 上插值 y(x)，并返回该点所在线段的斜率 dy/dx。
        要求 xs 单调递增。
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
            raise ValueError("折线节点必须为长度>=2 的一维数组，且 x、y 数量相同。")

        # 找到 x 所在的线段 [x_i, x_{i+1}]
        idx = np.searchsorted(xs, x) - 1
        idx = int(np.clip(idx, 0, xs.size - 2))
        x1, x2 = xs[idx], xs[idx + 1]
        y1, y2 = ys[idx], ys[idx + 1]
        if x2 == x1:
            # 垂直段的情况，这里返回 0 斜率以避免数值问题
            slope = 0.0
            y = y1
        else:
            t = (x - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            slope = (y2 - y1) / (x2 - x1)
        return float(y), float(slope)

    def slice_along_poly_surface(
        self,
        ground_profile,
        slip_profile,
        n_slices,
        q=0.0,
    ):
        """
        使用“任意折线滑动面”进行垂直条分，计算每条的几何与 p、α。

        Parameters
        ----------
        ground_profile : list[tuple[float, float]]
            地表折线坐标（通常为 GeometryBuilder.build() 的 Ground_Profile）。
        slip_profile : list[tuple[float, float]]
            预设的一系列坐标点，按 x 递增依次连接形成滑动面多折线。
        n_slices : int
            条分数。
        q : float, optional
            均布超载 (kN/m²)，p_i = W_i/Δx_i + q。

        Returns
        -------
        slices : list[dict]
            每个元素包含:
                - 'x_mid'      : 条中点 x 坐标
                - 'width'      : Δx_i
                - 'alpha_rad'  : α_i (rad)
                - 'alpha_deg'  : α_i (deg)
                - 'W'          : W_i
                - 'p'          : p_i
        """
        if n_slices <= 0:
            raise ValueError("n_slices 必须为正整数。")

        gp = np.asarray(ground_profile, dtype=float)
        sp = np.asarray(slip_profile, dtype=float)

        xs_surface, ys_surface = gp[:, 0], gp[:, 1]
        xs_slip, ys_slip = sp[:, 0], sp[:, 1]

        # 只在地表与滑动面在 x 方向有公共投影的区间内条分
        x_min = max(xs_surface.min(), xs_slip.min())
        x_max = min(xs_surface.max(), xs_slip.max())

        total_width = x_max - x_min
        if total_width <= 0:
            raise ValueError("地表与滑动面在 x 方向没有有效重叠区间。")

        b_width = total_width / n_slices

        slices = []
        skipped_indices = []
        for i in range(n_slices):
            x_left = x_min + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            # 滑动面底部 y_base 及其斜率
            y_base, slope_slip = self._interp_y_and_slope(xs_slip, ys_slip, x_mid)
            alpha_rad = np.arctan(slope_slip)

            # 地表高度
            y_top = self._interp_y(xs_surface, ys_surface, x_mid)
            h_mid = y_top - y_base
            if h_mid <= 0:
                # 滑动面位于地表之上 / 重合，不形成有效土条
                skipped_indices.append(i + 1)  # 记录被跳过的土条编号（从1开始）
                continue

            W_i = h_mid * b_width * self.gamma
            p_i = W_i / b_width + q

            slices.append(
                {
                    "x_mid": x_mid,
                    "x_left": x_left,
                    "x_right": x_right,
                    "width": b_width,
                    "alpha_rad": alpha_rad,
                    "alpha_deg": np.degrees(alpha_rad),
                    "y_base": y_base,
                    "y_top": y_top,
                    "h_mid": h_mid,
                    "W": W_i,
                    "p": p_i,
                }
            )

        # 如果有土条被跳过，输出警告信息
        if skipped_indices:
            print(f"\n[警告] 条分时跳过了 {len(skipped_indices)} 个土条（滑动面在地表之上）：")
            print(f"       被跳过的土条编号: {skipped_indices}")
            print(f"       实际有效土条数: {len(slices)} / {n_slices}")
            print()

        return slices


def build_slip_profile_from_factors(
    ground_profile,
    factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """
    根据遗传算法的因子数组生成一个满足约束的折线滑动面。
    
    X 和 Y 坐标都可以由遗传算法控制，这样更灵活，能搜索到更优的滑动面形状。

    约束逻辑：
    1. x 方向：所有点的 x 都落在坡体内部的一个子区间内（默认 [10%, 90%] 宽度），
       这样至少有两点位于坡体内部。
    2. y 方向：所有点都在地表之下（或重合），可以延伸到 y=0 以下。

    这非常适合作为遗传算法的“解码”函数：
    - 染色体可以表示为一个一维数组 factors，支持两种格式：
      * 格式1: shape (2*n_points,)，前 n_points 个是 X 因子，后 n_points 个是 Y 因子
      * 格式2: shape (n_points, 2)，第一列是 X 因子，第二列是 Y 因子
    - 本函数负责把 factors 转为实际的折线坐标 slip_profile。

    Parameters
    ----------
    ground_profile : list[tuple[float, float]]
        地表折线坐标。
    factors : array-like
        控制点的因子数组，支持两种格式：
        - shape (2*n_points,)：前 n_points 个是 X 因子（0~1），后 n_points 个是 Y 因子（0~1 或更大）
        - shape (n_points, 2)：第一列是 X 因子（0~1），第二列是 Y 因子（0~1 或更大）
        - Y 因子：0~1 表示在地表与 y=0 之间，>1 表示可以延伸到 y=0 以下
    x_range_ratio : tuple(float, float)
        在坡体水平范围内，控制点 X 坐标的允许范围，例如 (0.1, 0.9)
        表示 X 因子映射到 [x_min + 10%宽度, x_min + 90%宽度]。
    min_relative_depth : float
        最小相对深度，用于避免与地表完全重合（可设很小）。
    max_depth_below_zero : float, optional
        允许滑动面延伸到 y=0 以下的最大深度（单位：m）。
        如果为 None，则自动设为坡高的 50%（即允许延伸到 y = -0.5*H）。

    Returns
    -------
    slip_profile : np.ndarray, shape (n_points, 2)
        满足“全部在坡体内、且位于地表之下”的滑动面折线坐标。
    """
    gp = np.asarray(ground_profile, dtype=float)
    xs_surface, ys_surface = gp[:, 0], gp[:, 1]

    factors = np.asarray(factors, dtype=float)
    
    # 解析输入格式，提取 X 和 Y 因子
    if factors.ndim == 1:
        # 格式1: (2*n_points,) -> 前 n_points 是 X，后 n_points 是 Y
        if factors.size % 2 != 0:
            raise ValueError("factors 一维数组的长度必须是偶数（2*n_points）。")
        n_points = factors.size // 2
        x_factors = factors[:n_points]
        y_factors = factors[n_points:]
    elif factors.ndim == 2:
        # 格式2: (n_points, 2) -> 第一列是 X，第二列是 Y
        if factors.shape[1] != 2:
            raise ValueError("factors 二维数组的第二维必须是 2（n_points, 2）。")
        n_points = factors.shape[0]
        x_factors = factors[:, 0]
        y_factors = factors[:, 1]
    else:
        raise ValueError("factors 必须是 1 维或 2 维数组。")
    
    if n_points < 2:
        raise ValueError("滑动面控制点数必须至少为 2。")

    # 1. 确定 X 坐标的允许范围
    x_min, x_max = xs_surface.min(), xs_surface.max()
    width = x_max - x_min
    if width <= 0:
        raise ValueError("ground_profile 的 x 坐标不合法。")

    left_ratio, right_ratio = x_range_ratio
    left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
    right_ratio = float(np.clip(right_ratio, 0.0, 1.0))
    if right_ratio <= left_ratio:
        raise ValueError("x_range_ratio 必须满足 left < right。")

    xa = x_min + left_ratio * width
    xb = x_min + right_ratio * width

    # 2. 将 X 因子（0~1）映射到实际 X 坐标 [xa, xb]
    x_factors_clipped = np.clip(x_factors, 0.0, 1.0)
    xs_ctrl = xa + x_factors_clipped * (xb - xa)
    
    # 确保 X 坐标按递增排序（滑动面应该是从左到右的）
    xs_ctrl = np.sort(xs_ctrl)

    # 3. 计算各控制点处的地表高度
    ys_ground = np.interp(xs_ctrl, xs_surface, ys_surface)

    # 4. 确定最大可延伸深度
    if max_depth_below_zero is None:
        H_max = ys_ground.max()  # 最大地表高度（近似坡高）
        max_depth_below_zero = 0.5 * H_max

    # 5. 将 Y 因子转为实际深度
    #    Y 因子 = 0~1：在地表与 y=0 之间
    #    Y 因子 > 1：可以延伸到 y=0 以下
    depths_to_zero = ys_ground
    min_depths = min_relative_depth * depths_to_zero
    
    depths = np.zeros_like(y_factors)
    for i in range(len(y_factors)):
        yf = y_factors[i]
        if yf <= 1.0:
            # 在地表与 y=0 之间
            depths[i] = min_depths[i] + yf * (depths_to_zero[i] - min_depths[i])
        else:
            # 超过 y=0，延伸到 y=0 以下
            depths[i] = depths_to_zero[i] + (yf - 1.0) * max_depth_below_zero

    # 6. 计算滑动面坐标（位于地表之下，可以延伸到 y=0 以下）
    ys_slip = ys_ground - depths

    slip_profile = np.column_stack([xs_ctrl, ys_slip])
    return slip_profile


# 为了向后兼容，保留旧函数名作为别名
def build_slip_profile_from_depth_factors(
    ground_profile,
    depth_factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """
    向后兼容的旧函数名。现在推荐使用 build_slip_profile_from_factors。
    
    如果只传入 Y 因子（一维数组），X 坐标会自动均匀分布。
    """
    depth_factors = np.asarray(depth_factors, dtype=float)
    
    # 如果是一维数组，说明只有 Y 因子，X 坐标自动均匀分布
    if depth_factors.ndim == 1:
        n_points = depth_factors.size
        # 构造完整的 factors：X 因子均匀分布（0, 0.2, 0.4, ..., 1.0），Y 因子用用户提供的
        x_factors = np.linspace(0.0, 1.0, n_points)
        factors = np.column_stack([x_factors, depth_factors])
        return build_slip_profile_from_factors(
            ground_profile, factors, x_range_ratio, min_relative_depth, max_depth_below_zero
        )
    else:
        # 如果已经是二维数组，直接调用新函数
        return build_slip_profile_from_factors(
            ground_profile, depth_factors, x_range_ratio, min_relative_depth, max_depth_below_zero
        )


def plot_slope_and_slip(
    ground_profile,
    slip_profile=None,
    slope_region=None,
    ax=None,
    show=True,
    title="Janbu - Slope & Slip Surface",
):
    """
    可视化：输出当前坡面与滑动破坏面（折线）。

    Parameters
    ----------
    ground_profile : list[tuple[float, float]]
        地表折线坐标。
    slip_profile : list[tuple[float, float]] | None
        滑动面折线坐标（可选）。
    slope_region : np.ndarray | None
        土体区域多边形顶点（可选）。传入时会填充显示。
    ax : matplotlib.axes.Axes | None
        传入则画在现有坐标轴上。
    show : bool
        是否 plt.show()。
    title : str
        图标题。

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    gp = np.asarray(ground_profile, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 填充土体区域（可选）
    if slope_region is not None:
        reg = np.asarray(slope_region, dtype=float)
        ax.fill(reg[:, 0], reg[:, 1], color="#c7c7c7", alpha=0.35, label="Soil Region")

    # 地表线
    ax.plot(gp[:, 0], gp[:, 1], "k-", linewidth=3, label="Ground Surface")

    # 滑动面线（可选）
    if slip_profile is not None:
        sp = np.asarray(slip_profile, dtype=float)
        ax.plot(sp[:, 0], sp[:, 1], "r--", linewidth=2.5, label="Slip Surface")
        ax.scatter(sp[:, 0], sp[:, 1], c="r", s=18, zorder=5)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    # 合理范围
    x_all = [gp[:, 0]]
    y_all = [gp[:, 1]]
    if slip_profile is not None:
        x_all.append(sp[:, 0])
        y_all.append(sp[:, 1])
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    pad_x = (x_all.max() - x_all.min()) * 0.08 if x_all.max() > x_all.min() else 1.0
    pad_y = (y_all.max() - y_all.min()) * 0.15 if y_all.max() > y_all.min() else 1.0
    ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
    ax.set_ylim(min(0.0, y_all.min() - pad_y), y_all.max() + pad_y)

    if show:
        plt.show()
    return ax


class JanbuSolver:
    """
    Janbu GPS – 模块二：核心解算器 (Core Solver)
    
    根据设计文档，实现 Janbu 方法的 FOS 计算。
    当前实现初始化阶段（简化估算），假设条间垂直剪力 t_i = 0。
    """

    def __init__(self, c_prime, phi_prime, ru=0.0, u_i=None, delta_Q_i=None):
        """
        Parameters
        ----------
        c_prime : float or list[float]
            有效粘聚力 c'_i (kPa)。可以是单一值（均质土）或每个土条的列表。
        phi_prime : float or list[float]
            有效内摩擦角 φ'_i (度)。可以是单一值（均质土）或每个土条的列表。
        ru : float, optional
            孔压系数 r_u，计算方法与 fellenius.py 一致：
              u*l = r_u * W / cos(alpha)
            换算到本模块所需的 u_i（kPa）：
              u_i = u*l / l = r_u * W / Δx
            仅在 u_i 未显式提供时生效。
        u_i : float or list[float], optional
            孔隙水压力 u_i (kPa)。默认 None 表示 u_i = 0。
            可以是单一值或每个土条的列表。
        delta_Q_i : float or list[float], optional
            外部荷载增量 ΔQ_i (kN/m)。默认 None 表示 ΔQ_i = 0。
            可以是单一值或每个土条的列表。
        """
        self.c_prime = c_prime
        self.phi_prime = phi_prime
        self.ru = float(ru)
        self.u_i = u_i if u_i is not None else 0.0
        self.delta_Q_i = delta_Q_i if delta_Q_i is not None else 0.0

    def _get_slice_param(self, param, index, n_slices):
        """
        辅助函数：从参数（可能是标量或列表）中获取第 i 个土条的值。
        """
        if isinstance(param, (list, np.ndarray)):
            if len(param) != n_slices:
                raise ValueError(f"参数列表长度 ({len(param)}) 必须等于土条数 ({n_slices})")
            return float(param[index])
        else:
            return float(param)

    def calculate_fos_initial(
        self,
        slices,
        F_init=1.0,
        tolerance=1e-6,
        max_iter=50,
    ):
        """
        计算初始安全系数 F0（t_i = 0）。
        
        根据公式：
            n_α,i = cos²α_i (1 + tan α_i tan φ'_i / F)
            A_i = [c'_i + (p_i - u_i) tan φ'_i] Δx_i (1 + tan²α_i)
            B_i = p_i Δx_i tan α_i + ΔQ_i
            F_0 = Σ(A_i / n_α,i) / Σ(B_i)
        
        由于 n_α,i 中包含 F，需要进行子迭代使 F 收敛。

        Parameters
        ----------
        slices : list[dict]
            来自 JanbuPreprocessor.slice_along_poly_surface() 的土条数据。
            每个元素需包含: 'p', 'alpha_rad', 'width'
        F_init : float, optional
            初始试算值，默认 1.0。
        tolerance : float, optional
            收敛容差，默认 1e-6。
        max_iter : int, optional
            最大迭代次数，默认 50。

        Returns
        -------
        F0 : float
            初始安全系数。
        converged : bool
            是否收敛。
        iterations : int
            实际迭代次数。
        """
        if not slices:
            raise ValueError("slices 列表不能为空。")

        n_slices = len(slices)
        
        c_list = [self._get_slice_param(self.c_prime, i, n_slices) for i in range(n_slices)]
        phi_list = [self._get_slice_param(self.phi_prime, i, n_slices) for i in range(n_slices)]
        if self.u_i is None:
            u_list = []
            for i in range(n_slices):
                W_i = float(slices[i]["W"])
                dx_i = float(slices[i]["width"])
                if dx_i <= 0:
                    raise ValueError("土条 width(Δx) 必须为正数。")
                u_list.append(self.ru * W_i / dx_i)  # kPa
        else:
            u_list = [self._get_slice_param(self.u_i, i, n_slices) for i in range(n_slices)]
        dQ_list = [self._get_slice_param(self.delta_Q_i, i, n_slices) for i in range(n_slices)]
        
        # 转换为弧度
        phi_rad_list = [np.radians(phi) for phi in phi_list]
        tan_phi_list = [np.tan(phi_rad) for phi_rad in phi_rad_list]

        # 提取土条数据
        p_list = [s['p'] for s in slices]
        alpha_rad_list = [s['alpha_rad'] for s in slices]
        dx_list = [s['width'] for s in slices]

        # 迭代求解 F0
        F = float(F_init)
        converged = False
        
        for iteration in range(max_iter):
            F_old = F
            
            # 计算 n_α,i, A_i, B_i
            sum_A_over_n = 0.0
            sum_B = 0.0
            
            for i in range(n_slices):
                p_i = p_list[i]
                alpha_rad = alpha_rad_list[i]
                dx_i = dx_list[i]
                c_i = c_list[i]
                tan_phi_i = tan_phi_list[i]
                u_i = u_list[i]
                dQ_i = dQ_list[i]
                
                cos_alpha = np.cos(alpha_rad)
                tan_alpha = np.tan(alpha_rad)
                
                # n_α,i = cos²α_i (1 + tan α_i tan φ'_i / F)
                n_alpha_i = (cos_alpha ** 2) * (1.0 + tan_alpha * tan_phi_i / F)
                
                # A_i = [c'_i + (p_i - u_i) tan φ'_i] Δx_i
                A_i = (c_i + (p_i - u_i) * tan_phi_i) * dx_i
                
                # B_i = p_i Δx_i tan α_i + ΔQ_i
                B_i = p_i * dx_i * tan_alpha + dQ_i
                
                sum_A_over_n += A_i / n_alpha_i
                sum_B += B_i
            
            # F_0 = Σ(A_i / n_α,i) / Σ(B_i)
            if abs(sum_B) < 1e-10:
                # 分母为0，返回无穷大
                return np.inf, False, iteration + 1
            
            F = sum_A_over_n / sum_B
            
            # 检查收敛
            if abs(F - F_old) < tolerance:
                converged = True
                break
        
        return F, converged, iteration + 1

    @staticmethod
    def _as_array(x, n, name):
        """将标量或列表转换为长度为 n 的数组。"""
        if x is None:
            return None
        if isinstance(x, (list, np.ndarray)):
            if len(x) != n:
                raise ValueError(f"{name} 长度 ({len(x)}) 必须等于土条数 ({n})")
            return np.asarray(x, dtype=float)
        return np.full(n, float(x), dtype=float)

    @staticmethod
    def _interp_poly_y(xs, ys, x):
        """在折线上插值 y 坐标。"""
        return float(np.interp(x, xs, ys))
    
    def _prepare_slice_arrays(self, slices):
        """从 slices 字典列表中提取数组数据。"""
        n = len(slices)
        return {
            'n': n,
            'dx': np.asarray([s["width"] for s in slices], dtype=float),
            'alpha': np.asarray([s["alpha_rad"] for s in slices], dtype=float),
            'p': np.asarray([s["p"] for s in slices], dtype=float),
            'W': np.asarray([s["W"] for s in slices], dtype=float),
            'h_mid': np.asarray([s.get("h_mid", 0.0) for s in slices], dtype=float),
            'x_left': np.asarray([s["x_left"] for s in slices], dtype=float),
            'x_right': np.asarray([s["x_right"] for s in slices], dtype=float),
        }
    
    def _prepare_parameters(self, n):
        """准备土壤参数数组（c, phi, u, dQ）"""
        c = self._as_array(self.c_prime, n, "c_prime")
        phi_deg = self._as_array(self.phi_prime, n, "phi_prime")
        tan_phi = np.tan(np.radians(phi_deg))
        
        dQ = self._as_array(self.delta_Q_i, n, "delta_Q_i")
        if dQ is None:
            dQ = np.zeros(n, dtype=float)
        
        return c, phi_deg, tan_phi, dQ
    
    def _calculate_pore_pressure(self, W, dx, n):
        """计算孔隙水压力 u """
        if self.u_i is None:
            return self.ru * (W / dx)
        else:
            return self._as_array(self.u_i, n, "u_i")
    
    def _build_ground_profile_from_slices(self, slices, xs_slip, ys_slip):
        """从 slices 构建 ground_profile（如果未提供）"""
        gp_x, gp_y = [], []
        for i, s in enumerate(slices):
            gp_x.append(s["x_left"])
            if "y_top" in s and s["y_top"] is not None:
                gp_y.append(s["y_top"])
            else:
                x_mid = s.get("x_mid", s["x_left"] + 0.5 * s["width"])
                y_base_est = np.interp(x_mid, xs_slip, ys_slip)
                h_mid_est = s.get("h_mid", 0.0)
                gp_y.append(y_base_est + h_mid_est)
        
        if slices:
            gp_x.append(slices[-1]["x_right"])
            if "y_top" in slices[-1] and slices[-1]["y_top"] is not None:
                gp_y.append(slices[-1]["y_top"])
            else:
                x_mid = slices[-1].get("x_mid", slices[-1]["x_right"])
                y_base_est = np.interp(x_mid, xs_slip, ys_slip)
                h_mid_est = slices[-1].get("h_mid", 0.0)
                gp_y.append(y_base_est + h_mid_est)
        
        return np.column_stack([gp_x, gp_y])
    
    def _calculate_initial_E0(self, c, p, t, u, tan_phi, dx, alpha, tan_phi_array, dQ, F_old, n):
        """计算初始 E0"""
        tan_alpha = np.tan(alpha)
        cos_alpha = np.cos(alpha)
        n_alpha = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi_array) / F_old)
        
        A = (c + (p + t - u) * tan_phi_array) * dx
        B = dQ + (p + t) * dx * tan_alpha
        delta_E = B - (A / n_alpha) / F_old
        
        E_interface = np.zeros(n + 1, dtype=float)
        E_interface[1:] = np.cumsum(delta_E)
        
        return E_interface, delta_E
    
    def _calculate_thrust_line_geometry(self, x_interface, xs_ground, ys_ground, 
                                        xs_slip, ys_slip, lambda_thrust, n):
        """计算h_t, tan_alpha_t"""
        # Step A: 计算界面高度 H_int
        y_ground_interface = np.interp(x_interface, xs_ground, ys_ground)
        y_slip_interface = np.interp(x_interface, xs_slip, ys_slip)
        H_int = y_ground_interface - y_slip_interface
        
        # Step B: 计算推力线高度和绝对高程
        h_t_interface = float(lambda_thrust) * H_int
        y_thrust = y_slip_interface + h_t_interface
        
        # Step C: 计算推力线斜率 tan_alpha_t
        tan_alpha_t = self._calculate_thrust_line_slope(x_interface, y_thrust, n)
        
        return h_t_interface, tan_alpha_t, H_int
    
    def _calculate_thrust_line_slope(self, x_interface, y_thrust, n):
        """计算推力线斜率 tan_alpha_t"""
        tan_alpha_t = np.zeros(n + 1, dtype=float)
        EPS = 1e-12
        
        # 内部节点：使用中心差分
        for i in range(1, n):
            dx_i = x_interface[i + 1] - x_interface[i - 1]
            if abs(dx_i) > EPS:
                tan_alpha_t[i] = (y_thrust[i + 1] - y_thrust[i - 1]) / dx_i
        
        # 边界处理
        if n > 0:
            # 首界面：前向差分
            dx_0 = x_interface[1] - x_interface[0]
            if abs(dx_0) > EPS:
                tan_alpha_t[0] = (y_thrust[1] - y_thrust[0]) / dx_0
            
            # 尾界面：后向差分
            dx_n = x_interface[n] - x_interface[n - 1]
            if abs(dx_n) > EPS:
                tan_alpha_t[n] = (y_thrust[n] - y_thrust[n - 1]) / dx_n
        
        return tan_alpha_t
    
    def _calculate_dE_dx(self, delta_E_for_dE_dx, dx, n, it, delta_E_prev):
        """计算 dE/dx（使用加权平均公式，Janbu 论文 Eq. 117）。"""
        dE_dx = np.zeros(n + 1, dtype=float)
        dE_dx[0] = 0.0  # 首界面
        
        # 确定用于计算的 delta_E
        if it == 0:
            delta_E_used = delta_E_prev.copy()
        else:
            delta_E_used = delta_E_for_dE_dx.copy()
        
        # 内部界面：加权平均
        EPS = 1e-12
        for i in range(1, n):
            slice_idx_L = i - 1
            slice_idx_R = i
            numerator = delta_E_used[slice_idx_L] + delta_E_used[slice_idx_R]
            denominator = dx[slice_idx_L] + dx[slice_idx_R]
            if abs(denominator) > EPS:
                dE_dx[i] = numerator / denominator
        
        # 尾界面：设为 0
        if n > 0:
            dE_dx[n] = 0.0
        
        return dE_dx
    
    def _calculate_interface_shear_T(self, E_for_T, tan_alpha_t, h_t_interface, dE_dx, n):
        """计算界面垂直剪力 T。"""
        T_interface = -E_for_T * tan_alpha_t + h_t_interface * dE_dx
        T_interface[0] = 0.0
        return T_interface
    
    def _update_slice_shear_t(self, T_interface, dx, n):
        """更新土条内部垂直剪力变化率 t。"""
        t_new = np.zeros(n, dtype=float)
        for i in range(1, n + 1):
            t_new[i - 1] = (T_interface[i] - T_interface[i - 1]) / dx[i - 1]
        return t_new
    
    def _calculate_new_F(self, c, p, t_new, u, tan_phi, dx, alpha, dQ, n_alpha, n):
        """用新的 t 计算新的安全系数 F_new。"""
        tan_alpha = np.tan(alpha)
        A2 = (c + (p + t_new - u) * tan_phi) * dx
        B2 = dQ + (p + t_new) * dx * tan_alpha
        
        denom = np.sum(B2)
        if abs(denom) < 1e-12:
            return None, A2, B2
        
        F_new = np.sum(A2 / n_alpha) / denom
        return F_new, A2, B2

    def calculate_fos_gps(
        self,
        slices,
        slip_profile,
        ground_profile=None,
        F_init=1.0,
        tolerance=1e-6,
        max_iter=100,
        lambda_thrust=0.33,
        t_init=None,
        return_debug=False,
        print_iteration_table=False,
    ):
        """
        Janbu GPS 完整迭代

        输入：上一轮 F（初始为 F0）和上一轮 t（初始为 0）。
        每轮：
          1) 计算 ΔE_i, 累加得到 E_interface,i
          2) 计算推力线几何：h_t,i = λ * H_int,i，得到 α_t,i 与 (dE/dx)_i
          3) 计算界面垂直剪力 T_interface,i
          4) 更新土条内部垂直剪力变化率 t_i
          5) 用新 t_i 更新 A_i、B_i，并按 F_old 计算 n_α,i，得到 F_new
          6) 收敛检查 |F_new - F_old|

        Parameters
        ----------
        slices : list[dict]
            土条数据列表。
        slip_profile : array-like
            滑动面折线坐标，shape (n_points, 2)。
        ground_profile : array-like, optional
            地表折线坐标，shape (n_points, 2)。如果为 None，将从 slices 构建。
        F_init : float, optional
            初始安全系数，默认 1.0。
        tolerance : float, optional
            收敛容差，默认 1e-6。
        max_iter : int, optional
            最大迭代次数，默认 100。
        lambda_thrust : float, optional
            推力线系数，默认 0.33。
        t_init : array-like, optional
            初始垂直剪力变化率，默认全为 0。
        return_debug : bool, optional
            是否返回调试信息，默认 False。
        print_iteration_table : bool, optional
            是否打印迭代表格，默认 False。

        Returns
        -------
        F : float
            最终安全系数。
        converged : bool
            是否收敛。
        iterations : int
            实际迭代次数。
        debug : dict, optional
            调试信息（仅当 return_debug=True 时返回）。
        """
        # 输入验证
        if not slices:
            raise ValueError("slices 不能为空。")

        n = len(slices)
        sp = np.asarray(slip_profile, dtype=float)
        if sp.ndim != 2 or sp.shape[1] != 2 or sp.shape[0] < 2:
            raise ValueError("slip_profile 必须是 shape (n_points, 2) 的折线坐标数组/列表。")

        # 准备数据数组
        arrays = self._prepare_slice_arrays(slices)
        dx = arrays['dx']
        alpha = arrays['alpha']
        p = arrays['p']
        W = arrays['W']
        x_left = arrays['x_left']
        x_right = arrays['x_right']

        # 准备参数
        c, phi_deg, tan_phi, dQ = self._prepare_parameters(n)
        u = self._calculate_pore_pressure(W, dx, n)

        # 初始 t
        if t_init is None:
            t = np.zeros(n, dtype=float)
        else:
            t = self._as_array(t_init, n, "t_init")

        # 界面 x 坐标
        x_interface = np.concatenate([[x_left[0]], x_right])  # 长度 n+1

        # 处理滑动面和地表
        xs_slip, ys_slip = sp[:, 0], sp[:, 1]
        if ground_profile is None:
            ground_profile = self._build_ground_profile_from_slices(slices, xs_slip, ys_slip)
        else:
            ground_profile = np.asarray(ground_profile, dtype=float)
        xs_ground, ys_ground = ground_profile[:, 0], ground_profile[:, 1]

        # 初始化
        debug = {"F": [], "t": [], "E_interface": [], "T_interface": []} if return_debug else None
        F_old = float(F_init)
        converged = False

        # 输出初始数据表格
        if print_iteration_table:
            self._print_initial_data_table(slices, c, phi_deg, u, dQ, dx, alpha, p)

        # 计算初始 E0（t=0）
        E_interface_prev, delta_E_prev = self._calculate_initial_E0(
            c, p, t, u, tan_phi, dx, alpha, tan_phi, dQ, F_old, n
        )

        # 主迭代循环
        for it in range(max_iter):
            # Step 1: 计算 ΔE 和 E_interface
            tan_alpha = np.tan(alpha)
            cos_alpha = np.cos(alpha)
            n_alpha = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)

            A = (c + (p + t - u) * tan_phi) * dx
            B = dQ + (p + t) * dx * tan_alpha
            delta_E = B - (A / n_alpha) / F_old

            E_interface = np.zeros(n + 1, dtype=float)
            E_interface[1:] = np.cumsum(delta_E)

            # 确定用于计算 T 的 E（第一次迭代用 E0，后续用上一轮的 E）
            E_for_T = E_interface_prev.copy()

            # Step 2: 计算推力线几何
            h_t_interface, tan_alpha_t, _ = self._calculate_thrust_line_geometry(
                x_interface, xs_ground, ys_ground, xs_slip, ys_slip, lambda_thrust, n
            )

            # Step 3: 计算 dE/dx
            dE_dx = self._calculate_dE_dx(delta_E, dx, n, it, delta_E_prev)

            # Step 4: 计算界面垂直剪力 T
            T_interface = self._calculate_interface_shear_T(
                E_for_T, tan_alpha_t, h_t_interface, dE_dx, n
            )

            # Step 5: 更新土条垂直剪力变化率 t
            t_new = self._update_slice_shear_t(T_interface, dx, n)

            # Step 6: 计算新的安全系数 F_new
            F_new, A2, B2 = self._calculate_new_F(
                c, p, t_new, u, tan_phi, dx, alpha, dQ, n_alpha, n
            )
            if F_new is None:
                return np.inf, False, it + 1, (debug if return_debug else None)

            # 输出迭代表格
            if print_iteration_table:
                self._print_iteration_table(
                    it, F_old, F_new,
                    slices, c, phi_deg, u, dQ, dx, alpha, p, t, t_new,
                    A, A2, B, B2, n_alpha, delta_E, E_interface,
                    dE_dx, tan_alpha_t, h_t_interface, T_interface,
                    E_for_display_T=E_for_T,
                )

            # 记录调试信息
            if return_debug:
                debug["F"].append(F_old)
                debug["t"].append(t.copy())
                debug["E_interface"].append(E_interface.copy())
                debug["T_interface"].append(T_interface.copy())

            # Step 7: 收敛检查
            if abs(F_new - F_old) < tolerance:
                converged = True
                F_old = float(F_new)
                t = t_new
                E_interface_prev = E_interface.copy()
                delta_E_prev = delta_E.copy()
                break

            # 更新状态，准备下一轮迭代
            F_old = float(F_new)
            t = t_new
            E_interface_prev = E_interface.copy()
            delta_E_prev = delta_E.copy()

        # 最终处理
        if return_debug:
            debug["F_final"] = F_old
            debug["t_final"] = t

        return F_old, converged, (it + 1), (debug if return_debug else None)

    def _print_initial_data_table(self, slices, c, phi_deg, u, dQ, dx, alpha, p):
        """输出初始数据表格"""
        n = len(slices)
        print("\n" + "=" * 100)
        print("Initial Step - Data from profile")
        print("=" * 100)
        header_fmt = "{:<6} {:<10} {:<10} {:<12} {:<12} {:<12} {:<12} {:<10}"
        print(header_fmt.format("Slice", "tan α", "Δx", "p (kPa)", "u (kPa)", "c' (kPa)", "tan φ'", "ΔQ"))
        print("-" * 100)
        for i in range(n):
            print(f"{i+1:<6} {np.tan(alpha[i]):<10.4f} {dx[i]:<10.3f} {p[i]:<12.3f} {u[i]:<12.3f} {c[i]:<12.3f} {np.tan(np.radians(phi_deg[i])):<12.4f} {dQ[i]:<10.3f}")
        print("=" * 100)

    def _print_iteration_table(
        self, it, F_old, F_new,
        slices, c, phi_deg, u, dQ, dx, alpha, p, t, t_new,
        A, A2, B, B2, n_alpha, delta_E, E_interface,
        dE_dx, tan_alpha_t, h_t_interface, T_interface,
        E_for_display_T=None,
    ):
        """输出每轮迭代的详细中间值表格"""
        n = len(slices)
        tan_alpha = np.tan(alpha)
        tan_phi = np.tan(np.radians(phi_deg))
        
        # 计算 A' (A_prime)
        A_prime = (c + (p + t - u) * tan_phi) * dx
        A2_prime = (c + (p + t_new - u) * tan_phi) * dx
        
        # 计算 ΔT
        delta_T = np.zeros(n)
        for i in range(1, n + 1):
            delta_T[i - 1] = T_interface[i] - T_interface[i - 1]
        

        # 迭代步骤：输出 T 和更新后的 F
        iteration_name = f"Iteration F{it + 1}"
        
        # 输出 T 计算部分
        E_display = E_for_display_T if E_for_display_T is not None else E_interface
        print("\n" + "=" * 120)
        print(f"{iteration_name} - Calculation of T{it + 1}")
        print("=" * 120)
        header_T = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(header_T.format("Slice", "E(i-1)", "dE/dx", "tan α_t", "ht", "T"))
        print("-" * 120)
        for i in range(n + 1):
            interface_label = f"i={i}" if i < n else "b"
            print(f"{interface_label:<6} {E_display[i]:<12.3f} {dE_dx[i]:<12.4f} {tan_alpha_t[i]:<12.4f} {h_t_interface[i]:<12.3f} {T_interface[i]:<12.3f}")
        
        # 输出 t 和更新后的 B, A
        print("\n" + "-" * 120)
        print(f"{iteration_name} - Calculation of F (with updated t)")
        print("-" * 120)
        header_iter = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(header_iter.format("Slice", "ΔT", "t", "B", "A'", "nα", "A", "ΔE", "E"))
        print("-" * 120)
        for i in range(n):
            print(f"{i+1:<6} {delta_T[i]:<12.3f} {t_new[i]:<12.4f} {B2[i]:<12.3f} {A2_prime[i]:<12.3f} {n_alpha[i]:<12.4f} {A2[i]/n_alpha[i]:<12.3f} {delta_E[i]:<12.3f} {E_interface[i+1]:<12.3f}")
        print(f"{'Σ':<6} {np.sum(delta_T):<12.3f} {np.sum(t_new):<12.4f} {np.sum(B2):<12.3f} {np.sum(A2_prime):<12.3f} {'':<12} {np.sum(A2/n_alpha):<12.3f} {np.sum(delta_E):<12.3f} {E_interface[-1]:<12.3f}")
        print(f"\nF{it} = Σ(A{it}) / Σ(B{it}) = {np.sum(A2/n_alpha):.3f} / {np.sum(B2):.3f} = {F_new:.3f}")
        print("=" * 120)




if __name__ == "__main__":
    
    # 1. 构建几何
    gb = GeometryBuilder(slope_height=12.5, slope_ratio=2.5, bottom_extension=5.0, top_extension=10.0)
    ground, region = gb.build()

    # 2. 生成滑动面
    n_points = 5
    factors_1d = np.array([
        0, 0.1, 0.5, 0.7, 0.9,  
        1, 1.1, 0.8, 0.5, 0.1   
    ])
    slip_profile = build_slip_profile_from_factors(
        ground_profile=ground,
        factors=factors_1d,
        x_range_ratio=(0.1, 0.9),
    )

    # 3. 预处理：条分并计算 p 和 α 
    pre = JanbuPreprocessor(gamma=19.0)
    slices = pre.slice_along_poly_surface(
        ground_profile=ground,
        slip_profile=slip_profile,
        n_slices=10,
        q=0.0,
    )

    plot_slope_and_slip(ground_profile=ground, slip_profile=slip_profile, slope_region=region, show=True)

    print("=" * 60)
    print("Janbu 方法 - 预处理结果（土条数据）")
    print("=" * 60)
    print("i\t x_mid\t  alpha(deg)\t   p(kPa)\t  W(kN/m)")
    for i, s in enumerate(slices, start=1):
        print(f"{i:02d}\t{s['x_mid']:.3f}\t{s['alpha_deg']:.3f}\t{s['p']:.3f}\t{s['W']:.3f}")
    
    # 4. 计算 FOS
    c_prime = 9.5  # kPa
    phi_prime = 33.8  # 度
    ru = 0.4  # u = ru * W/Δx 计算
    u_i = None  # 若想手动指定孔压（kPa），可改为数值或列表；None 表示用 ru 自动计算
    delta_Q_i = 0.0  # 外部荷载增量（kN/m），通常为0
    
    solver = JanbuSolver(c_prime=c_prime, phi_prime=phi_prime, ru=ru, u_i=u_i, delta_Q_i=delta_Q_i)

    # 先用初始化阶段得到 F0（t=0）
    F0, conv0, it0 = solver.calculate_fos_initial(slices, F_init=1.0, tolerance=1e-6, max_iter=50)

    # 再用 GPS 完整迭代
    F, converged, iterations, _ = solver.calculate_fos_gps(
        slices=slices,
        slip_profile=slip_profile,
        F_init=F0,
        tolerance=1e-6,
        max_iter=100,
        lambda_thrust=0.33,
        print_iteration_table=True,
    )
    
    print("\n" + "=" * 60)
    print("Janbu 方法 - FOS 计算结果")
    print("=" * 60)
    print(f"初始化安全系数 F0(t=0): {F0:.6f}")
    print(f"GPS 完整迭代安全系数 F: {F:.6f}")
    print(f"收敛状态: {'已收敛' if converged else '未收敛（达到最大迭代次数）'}")
    print(f"迭代次数: {iterations}")
    print("=" * 60)

