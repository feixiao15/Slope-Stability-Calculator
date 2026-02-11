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

    def slice_along_circular_arc(
        self,
        ground_profile,
        center,
        x_entry,
        n_slices,
        q=0.0,
        require_exit_at_crest=True,
    ):
        """
        直接基于“圆弧滑动面”（不离散成折线）进行垂直条分。

        与 bishop 的圆弧一致：对给定圆心 center=(xc,yc) 与坡底入口 x_entry（入口点为 (x_entry, 0)）
        定义圆弧滑动面为圆的下半支：y_base(x)=yc - sqrt(R^2-(x-xc)^2)。

        约束：
        - x_entry 必须在坡底宽度 [A.x, B.x] 内
        - 默认要求穿出点在坡顶高度 y=H 且 x_exit >= crest_x（即“必须过坡顶平台”）

        Returns
        -------
        slices : list[dict] 或 None（若该圆弧无效）
        meta : dict，包含 x_exit、radius 等信息
        """
        if n_slices <= 0:
            raise ValueError("n_slices 必须为正整数。")

        gp = np.asarray(ground_profile, dtype=float)
        xs_surface, ys_surface = gp[:, 0], gp[:, 1]
        # Janbu GeometryBuilder: A=(0,0), B=(L_bot,0), C=(crest_x,H)
        A = gp[0]
        B = gp[1]
        C = gp[2] if gp.shape[0] >= 3 else gp[-1]
        x_min_bot = float(min(A[0], B[0]))
        x_max_bot = float(max(A[0], B[0]))
        x_entry = float(x_entry)
        if x_entry < x_min_bot or x_entry > x_max_bot:
            return None, {"reason": "x_entry_out_of_toe_width"}

        xc, yc = map(float, center)
        radius = float(np.hypot(xc - x_entry, yc - 0.0))
        if radius <= 0:
            return None, {"reason": "radius_non_positive"}

        # 穿出点：默认强制在 y=H 穿出（坡顶高度），并要求落在坡顶平台（x_exit >= crest_x）
        H = float(np.max(ys_surface))
        crest_x = float(C[0])
        term_sq = radius * radius - (H - yc) ** 2
        if term_sq < 0:
            return None, {"reason": "arc_not_reach_crest_height"}
        x_exit = float(xc + np.sqrt(term_sq))
        if require_exit_at_crest and x_exit < crest_x:
            return None, {"reason": "exit_not_on_crest_platform"}
        if x_exit <= x_entry:
            return None, {"reason": "x_exit_not_right_of_entry"}

        total_width = x_exit - x_entry
        b_width = total_width / n_slices

        slices = []
        for i in range(n_slices):
            x_left = x_entry + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            base_term = radius * radius - (x_mid - xc) ** 2
            if base_term <= 0:
                continue
            y_base = float(yc - np.sqrt(base_term))  # 下半圆

            # 圆在该点的切线斜率 dy/dx = -(x-xc)/(y-yc)
            denom = (y_base - yc)
            if abs(denom) < 1e-12:
                slope_slip = 0.0
            else:
                slope_slip = float(-(x_mid - xc) / denom)
            alpha_rad = float(np.arctan(slope_slip))

            y_top = float(np.interp(x_mid, xs_surface, ys_surface))
            h_mid = y_top - y_base
            if h_mid <= 0:
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

        if not slices:
            return None, {"reason": "no_valid_slices", "x_exit": x_exit, "radius": radius}

        meta = {"x_exit": x_exit, "radius": radius, "crest_x": crest_x, "H": H}
        return slices, meta


def build_slip_profile_from_factors(
    ground_profile,
    factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """
    根据因子数组生成一个满足约束的滑动面。

    注意：本项目中“滑动面”现在推荐使用 **圆弧滑动面**（与 `bishop.py` 一致）的生成方式：
    - `build_slip_profile_circular_arc_from_factors`
    原先的“任意折线滑动面”生成逻辑已迁移到：
    - `build_slip_profile_polyline_from_factors`
    
    为兼容旧代码，本函数仍保留旧签名，但默认行为已调整为调用圆弧滑动面生成器：
    - 若 `factors` 形如长度为 3 的数组：视为圆弧模式 [fx_center, fy_center, fx_entry]
    - 否则：退回旧的折线模式（调用 `build_slip_profile_polyline_from_factors`）

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
    factors = np.asarray(factors, dtype=float).reshape(-1)
    if factors.size == 3:
        return build_slip_profile_circular_arc_from_factors(ground_profile, factors)
    return build_slip_profile_polyline_from_factors(
        ground_profile=ground_profile,
        factors=factors,
        x_range_ratio=x_range_ratio,
        min_relative_depth=min_relative_depth,
        max_depth_below_zero=max_depth_below_zero,
    )


def _circle_segment_intersections(center, radius, p1, p2):
    """返回圆与线段 p1->p2 的交点列表（每个为 (x,y)），可能为空/1个/2个。"""
    xc, yc = center
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    a, b = x1 - xc, y1 - yc

    A = dx * dx + dy * dy
    if A == 0:
        return []
    B = 2 * (a * dx + b * dy)
    C = a * a + b * b - radius * radius
    disc = B * B - 4 * A * C
    if disc < 0:
        return []

    sqrt_d = np.sqrt(disc)
    out = []
    for t in [(-B - sqrt_d) / (2 * A), (-B + sqrt_d) / (2 * A)]:
        if 0 <= t <= 1:
            out.append((x1 + t * dx, y1 + t * dy))
    return out


def _find_arc_exit_on_surface(ground_profile, center, radius, x_entry):
    """
    求圆弧与地表折线的穿出点：在入口右侧 (x > x_entry) 的交点中取 x 最大者。
    返回 (x_exit, y_exit) 或 None。
    """
    gp = np.asarray(ground_profile, dtype=float)
    best = None
    for k in range(len(gp) - 1):
        p1 = tuple(gp[k])
        p2 = tuple(gp[k + 1])
        for x, y in _circle_segment_intersections(center, radius, p1, p2):
            if x <= x_entry:
                continue
            if best is None or x > best[0]:
                best = (float(x), float(y))
    return best


def build_slip_profile_circular_arc(
    ground_profile,
    center,
    x_entry,
    n_points=80,
    eps=1e-6,
):
    """
    生成与 `bishop.py` 一致的圆弧滑动面（用折线离散表示）。

    - 入口点：(x_entry, 0)，且要求 x_entry 位于坡底宽度 [A.x, B.x] 内
    - 圆心：center = (xc, yc)
    - 半径：圆心到入口点距离
    - 穿出点：圆与地表折线的交点中，入口右侧 x 最大者（可在坡面/坡顶平台穿出）
    - 返回：shape (n_points, 2) 的折线坐标，x 单调递增
    """
    gp = np.asarray(ground_profile, dtype=float)
    A = gp[0]
    B = gp[1]
    x_min_bot = float(min(A[0], B[0]))
    x_max_bot = float(max(A[0], B[0]))
    x_entry = float(x_entry)
    if not (x_min_bot - eps <= x_entry <= x_max_bot + eps):
        raise ValueError(f"x_entry 必须在坡底宽度范围内 [{x_min_bot}, {x_max_bot}]，当前为 {x_entry}.")

    xc, yc = map(float, center)
    entry_pt = (x_entry, 0.0)
    radius = float(np.hypot(xc - entry_pt[0], yc - entry_pt[1]))
    if radius <= 0:
        raise ValueError("radius 必须为正。")

    exit_pt = _find_arc_exit_on_surface(ground_profile, (xc, yc), radius, x_entry)
    if exit_pt is None:
        raise ValueError("该圆与地表在入口右侧无有效穿出点，无法构造圆弧滑动面。")
    x_exit, y_exit = exit_pt

    # 以 x 均匀离散圆弧（取下半圆：y = yc - sqrt(...)）
    xs = np.linspace(x_entry, x_exit, int(n_points))
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        term = radius * radius - (x - xc) ** 2
        if term < 0:
            ys[i] = np.nan
        else:
            ys[i] = yc - np.sqrt(term)

    # 确保在地表之下（数值上略微下压 eps）
    y_ground = np.interp(xs, gp[:, 0], gp[:, 1])
    ys = np.minimum(ys, y_ground - eps)

    slip_profile = np.column_stack([xs, ys])
    slip_profile = slip_profile[~np.isnan(slip_profile[:, 1])]
    if slip_profile.shape[0] < 2:
        raise ValueError("圆弧离散点不足，无法形成有效滑动面。")
    return slip_profile


def build_slip_profile_circular_arc_from_factors(ground_profile, factors, n_points=80):
    """
    圆弧滑动面“解码器”：
    factors = [fx_center, fy_center, fx_entry]，均建议取 0~1。

    - fx_entry 映射到坡底宽度 [A.x, B.x]
    - fx_center 映射到 [A.x, D.x]
    - fy_center 映射到 [0, 3*H]（H 取地表最大高度）
    """
    gp = np.asarray(ground_profile, dtype=float)
    xs, ys = gp[:, 0], gp[:, 1]
    A, B, D = gp[0], gp[1], gp[-1]
    H = float(np.max(ys))
    fx_c, fy_c, fx_e = map(float, np.clip(np.asarray(factors, dtype=float).reshape(-1), 0.0, 1.0))

    x_entry = float(min(A[0], B[0]) + fx_e * (max(A[0], B[0]) - min(A[0], B[0])))
    xc = float(xs.min() + fx_c * (xs.max() - xs.min()))
    yc = float(0.0 + fy_c * (3.0 * H if H > 0 else 1.0))
    return build_slip_profile_circular_arc(ground_profile, (xc, yc), x_entry, n_points=n_points)


def find_critical_fos_circular_arc(
    ground_profile,
    gamma,
    c_prime,
    phi_prime,
    ru,
    n_slices,
    center_grid_x,
    center_grid_y,
    entry_x_range=None,
    q=0.0,
    n_arc_points=120,
    use_gps=True,
    gps_tolerance=1e-6,
    gps_max_iter=80,
    lambda_thrust=0.33,
):
    """
    像 bishop.py 一样：在约束条件下搜索 valid 的圆弧滑动面，并返回最小 FoS。

    搜索变量：
    - 圆心 (xc, yc) ∈ center_grid_x × center_grid_y
    - 坡底入口 x_entry ∈ entry_x_range（要求位于坡底宽度 [A.x, B.x] 内）

    圆弧构造：
    - 入口点固定为 (x_entry, 0)
    - 半径由圆心到入口点确定
    - 穿出点为圆与地表折线的交点中，入口右侧 x 最大者（可在坡面或坡顶平台穿出）
    - 圆弧离散为折线 slip_profile，供 JanbuPreprocessor 条分

    返回：
    - best: dict 或 None
    - fos_results: list[(xc, yc, fos_min_over_entry)]，用于画等高线
    """
    gp = np.asarray(ground_profile, dtype=float)
    A = gp[0]
    B = gp[1]
    x_min_bot = float(min(A[0], B[0]))
    x_max_bot = float(max(A[0], B[0]))

    if entry_x_range is None:
        entry_x_list = np.array([x_max_bot], dtype=float)  # 默认从坡脚点出发
    else:
        entry_x_list = np.atleast_1d(np.asarray(entry_x_range, dtype=float))

    pre = JanbuPreprocessor(gamma=gamma)
    solver = JanbuSolver(c_prime=c_prime, phi_prime=phi_prime, ru=ru, u_i=None, delta_Q_i=0.0)

    min_fos = np.inf
    best = None
    fos_results = []

    for xc in center_grid_x:
        for yc in center_grid_y:
            best_fos_here = np.inf
            best_here = None

            for x_entry in entry_x_list:
                x_entry = float(x_entry)
                if x_entry < x_min_bot or x_entry > x_max_bot:
                    continue

                try:
                    slip_profile = build_slip_profile_circular_arc(
                        ground_profile=ground_profile,
                        center=(float(xc), float(yc)),
                        x_entry=x_entry,
                        n_points=n_arc_points,
                    )
                    slices = pre.slice_along_poly_surface(
                        ground_profile=ground_profile,
                        slip_profile=slip_profile,
                        n_slices=n_slices,
                        q=q,
                    )
                    if not slices:
                        continue

                    F0, conv0, it0 = solver.calculate_fos_initial(
                        slices, F_init=1.0, tolerance=gps_tolerance, max_iter=50
                    )
                    if not np.isfinite(F0):
                        continue

                    if use_gps:
                        F, converged, iterations, _ = solver.calculate_fos_gps(
                            slices=slices,
                            slip_profile=slip_profile,
                            F_init=F0,
                            tolerance=gps_tolerance,
                            max_iter=gps_max_iter,
                            lambda_thrust=lambda_thrust,
                            print_iteration_table=False,
                        )
                        fos = float(F)
                    else:
                        fos = float(F0)

                    if np.isfinite(fos) and fos < best_fos_here:
                        radius = float(np.hypot(float(xc) - x_entry, float(yc) - 0.0))
                        best_fos_here = fos
                        best_here = {
                            "center": (float(xc), float(yc)),
                            "radius": radius,
                            "x_entry": x_entry,
                            "fos": fos,
                            "use_gps": bool(use_gps),
                        }
                except Exception:
                    # 构造/条分/求解失败都视为 invalid，继续搜索
                    continue

            fos_results.append((float(xc), float(yc), best_fos_here if np.isfinite(best_fos_here) else np.nan))

            if best_here is not None and best_here["fos"] < min_fos:
                min_fos = best_here["fos"]
                best = best_here

    return best, fos_results


def calculate_fos_for_circular_arc(
    ground_profile,
    gamma,
    c_prime,
    phi_prime,
    ru,
    n_slices,
    center,
    x_entry,
    q=0.0,
    require_exit_at_crest=True,
    use_gps=False,
    gps_tolerance=1e-6,
    gps_max_iter=10,
    lambda_thrust=0.33,
    print_iteration_table=False,
    plot_f_history=False,
):
    """
    计算“给定特定圆弧滑动面”的 FoS（不离散成折线）。

    - 先用 `JanbuPreprocessor.slice_along_circular_arc` 直接条分得到 slices
    - 再用 JanbuSolver 计算：
      * use_gps=False：只算初始化阶段 F0（t=0）
      * use_gps=True ：先算 F0，再做 GPS 完整迭代

    返回：
    - fos: float
    - slices: list[dict]
    - meta: dict（包含 x_exit / radius / reason 等）
    """
    pre = JanbuPreprocessor(gamma=gamma)
    slices, meta = pre.slice_along_circular_arc(
        ground_profile=ground_profile,
        center=center,
        x_entry=x_entry,
        n_slices=n_slices,
        q=q,
        require_exit_at_crest=require_exit_at_crest,
    )
    if slices is None:
        return np.nan, None, meta

    solver = JanbuSolver(c_prime=c_prime, phi_prime=phi_prime, ru=ru, u_i=None, delta_Q_i=0.0)

    # 如果不做 GPS，只需在 t=0 条件下用简化 Janbu 法求一次 F0
    if not use_gps:
        F0, conv0, it0 = solver.calculate_fos_initial(
            slices, F_init=1.0, tolerance=gps_tolerance, max_iter=50
        )
        if not np.isfinite(F0):
            meta = {**meta, "reason": "F0_non_finite", "F0": F0}
            return np.nan, slices, meta
        return float(F0), slices, {**meta, "F0": float(F0), "use_gps": False}

    # 做 GPS 时，F0 的计算已经在 calculate_fos_gps 内部完成（视为 GPS 的第 0 步）
    F, converged, iterations, debug = solver.calculate_fos_gps(
        slices=slices,
        slip_profile=None,
        ground_profile=ground_profile,
        F_init=1.0,
        tolerance=gps_tolerance,
        max_iter=gps_max_iter,
        lambda_thrust=lambda_thrust,
        t_init=None,
        return_debug=True,
        print_iteration_table=print_iteration_table,
        arc_center=center,
        arc_radius=meta.get("radius"),
    )

    # 可选：根据每轮迭代的 FoS 画收敛曲线 F-iteration
    if plot_f_history and debug is not None:
        F_hist = debug.get("F", [])
        if len(F_hist) > 0:
            it = np.arange(1, len(F_hist) + 1, dtype=int)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(it, F_hist, "o-", label="F (per iteration)")
            ax.axhline(float(F), color="r", linestyle="--", label=f"F_final = {float(F):.3f}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Factor of Safety F")
            ax.set_title("Janbu GPS - F vs Iteration (Circular Arc)")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            plt.show()

    # F0 信息由 GPS 调试数据给出
    F0_from_gps = debug.get("F0", np.nan) if debug is not None else np.nan
    return float(F), slices, {
        **meta,
        "F0": float(F0_from_gps),
        "use_gps": True,
        "converged": bool(converged),
        "iterations": int(iterations),
    }


def build_slip_profile_polyline_from_factors(
    ground_profile,
    factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """原先的“任意折线滑动面”生成逻辑（保留以便遗传算法继续使用）。"""
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

    def _calculate_thrust_line_geometry_arc(self, x_interface, xs_ground, ys_ground,
                                            center, radius, lambda_thrust, n):
        """
        基于圆弧（非离散）计算推力线几何：给定圆心与半径，直接用圆方程得到 y_slip。
        """
        xc, yc = map(float, center)
        R = float(radius)
        y_ground_interface = np.interp(x_interface, xs_ground, ys_ground)
        # 圆弧下半支：y = yc - sqrt(R^2 - (x-xc)^2)
        term = R * R - (x_interface - xc) ** 2
        term = np.maximum(term, 0.0)
        y_slip_interface = yc - np.sqrt(term)
        H_int = y_ground_interface - y_slip_interface

        h_t_interface = float(lambda_thrust) * H_int
        y_thrust = y_slip_interface + h_t_interface

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
    
    def _calculate_dE_dx(self, delta_E_used, dx, n):
        """计算 dE/dx（使用加权平均公式，Janbu 论文 Eq. 117）。

        注意：这里的 delta_E_used 必须与用于计算 E_for_T 的那一轮 E 保持同步，
        即对应“上一轮/当前用于受力平衡的 E”的 ΔE'。
        """
        dE_dx = np.zeros(n + 1, dtype=float)
        dE_dx[0] = 0.0  # 首界面

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
        slip_profile=None,
        ground_profile=None,
        F_init=1.0,
        tolerance=1e-6,
        max_iter=100,
        lambda_thrust=0.33,
        t_init=None,
        return_debug=False,
        print_iteration_table=False,
        arc_center=None,
        arc_radius=None,
    ):
        """
        Janbu GPS 完整迭代

        初始化（Step 0, t=0）：直接由 A,B 求 F0，并得到 ΔE0/E0，作为后续迭代的起点。

        每轮 GPS（k=1,2,...）：
          1) 用上一轮的 E_{k-1}（以及对应的 ΔE_{k-1}）计算 dE/dx、推力线几何、T_interface,k
          2) 由 T_interface,k 更新得到本轮 t_k
          3) 用 t_k 计算 A,B，并按上一轮 F_{k-1} 计算 n_α，得到 F_k
          4) 用 (F_k, t_k) 更新 ΔE_k 与 E_k，供下一轮使用
          5) 收敛检查 |F_k - F_{k-1}|

        Parameters
        ----------
        slices : list[dict]
            土条数据列表。
        slip_profile : array-like or None
            若提供，则按“折线滑动面”模式计算（与旧实现兼容）。
        ground_profile : array-like, optional
            地表折线坐标，shape (n_points, 2)。
            - 折线模式且为 None：将从 slices + slip_profile 构建。
            - 圆弧模式（非离散）下：必须显式提供 ground_profile。
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
        use_arc = arc_center is not None and arc_radius is not None
        if slip_profile is None and not use_arc:
            raise ValueError("必须提供 slip_profile（折线模式）或 arc_center+arc_radius（圆弧模式）之一。")
        if slip_profile is not None and use_arc:
            raise ValueError("slip_profile 与 arc_center/arc_radius 不能同时提供，请二选一。")

        if slip_profile is not None:
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
        if slip_profile is not None:
            xs_slip, ys_slip = sp[:, 0], sp[:, 1]
            if ground_profile is None:
                ground_profile = self._build_ground_profile_from_slices(slices, xs_slip, ys_slip)
            else:
                ground_profile = np.asarray(ground_profile, dtype=float)
        else:
            if ground_profile is None:
                raise ValueError("圆弧模式下 ground_profile 不能为空。")
            ground_profile = np.asarray(ground_profile, dtype=float)
            xs_slip = ys_slip = None  # 不使用
        xs_ground, ys_ground = ground_profile[:, 0], ground_profile[:, 1]

        # 初始化
        debug = {"F": [], "t": [], "E_interface": [], "T_interface": []} if return_debug else None
        converged = False

        # 输出土条几何/参数表（可选）
        if print_iteration_table:
            self._print_initial_data_table(slices, c, phi_deg, u, dQ, dx, alpha, p)

        # Step 0（GPS 第一步，t=0）：用 p, Δx, tanα 直接算 B, A', n, A，再算 F0 和 ΔE0，只算一次，不子迭代
        F_old = float(F_init)
        tan_alpha = np.tan(alpha)
        cos_alpha = np.cos(alpha)
        n_alpha_0 = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)
        A_0 = (c + (p - u) * tan_phi) * dx
        B_0 = dQ + p * dx * tan_alpha
        sum_A_over_n = np.sum(A_0 / n_alpha_0)
        sum_B_0 = np.sum(B_0)
        if abs(sum_B_0) < 1e-12:
            return np.inf, False, 0, (debug if return_debug else None)
        F0 = sum_A_over_n / sum_B_0
        F_old = float(F0)
        delta_E_prev = B_0 - (A_0 / n_alpha_0) / F_old
        E_interface_prev = np.zeros(n + 1, dtype=float)
        E_interface_prev[1:] = np.cumsum(delta_E_prev)
        if debug is not None:
            debug["F0"] = float(F0)

        if print_iteration_table:
            self._print_step0_table(
                n, c, phi_deg, u, dx, alpha, B_0, A_0, n_alpha_0, delta_E_prev, E_interface_prev, F0
            )

        # 主迭代：第二步起用上一步的 ΔE 算 T、t，再算 B、A'、n、A、F 和新的 ΔE
        t = np.zeros(n, dtype=float)
        it = -1
        for it in range(max_iter):
            # 每一轮 GPS：
            # 1) 用“上一轮 E”先求 T -> t（第一次迭代前，t=0 已用于得到 E0）
            # 2) 用求得的 t 计算 A,B -> F_new
            # 3) 用 (F_new, t) 更新 ΔE 与 E，供下一轮使用

            # Step 1: 计算推力线几何（与 E 无关）
            if use_arc:
                h_t_interface, tan_alpha_t, _ = self._calculate_thrust_line_geometry_arc(
                    x_interface, xs_ground, ys_ground, arc_center, arc_radius, lambda_thrust, n
                )
            else:
                h_t_interface, tan_alpha_t, _ = self._calculate_thrust_line_geometry(
                    x_interface, xs_ground, ys_ground, xs_slip, ys_slip, lambda_thrust, n
                )

            # Step 2: 用“上一轮 ΔE / E”计算 dE/dx 与 T_interface
            E_for_T = E_interface_prev
            dE_dx = self._calculate_dE_dx(delta_E_prev, dx, n)

            T_interface = self._calculate_interface_shear_T(
                E_for_T, tan_alpha_t, h_t_interface, dE_dx, n
            )

            # Step 3: 由 T 更新得到本轮用于计算 F 的 t
            t_new = self._update_slice_shear_t(T_interface, dx, n)

            # Step 4: 以 F_old 计算 nα（本轮外循环不做 F 子迭代），并用 t_new 计算 F_new
            tan_alpha = np.tan(alpha)
            cos_alpha = np.cos(alpha)
            n_alpha = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)
            F_new, A2, B2 = self._calculate_new_F(
                c, p, t_new, u, tan_phi, dx, alpha, dQ, n_alpha, n
            )
            if F_new is None:
                return np.inf, False, it + 1, (debug if return_debug else None)

            # Step 5: 用 (F_new, t_new) 更新 ΔE 与 E，作为下一轮的输入
            # 关键：本轮 ΔE_k 必须使用“算 F_k 时同一套 nα_k(F_{k-1})”，
            # 这样才能保证教材推导的 ΣΔE_k = 0 恒成立：
            #   F_k = Σ(A_k/nα_k) / Σ(B_k)
            #   ΔE_k = B_k - (A_k/nα_k) / F_k
            #   ⇒ ΣΔE_k = ΣB_k - (1/F_k) Σ(A_k/nα_k) = 0
            A_next = A2
            B_next = B2
            delta_E_next = B_next - (A_next / n_alpha) / F_new
            E_interface_next = np.zeros(n + 1, dtype=float)
            E_interface_next[1:] = np.cumsum(delta_E_next)

            # 输出迭代表格
            if print_iteration_table:
                self._print_iteration_table(
                    it, F_old, F_new,
                    slices, c, phi_deg, u, dQ, dx, alpha, p, t, t_new,
                    A2, A2, B2, B2, n_alpha, delta_E_next, E_interface_next,
                    dE_dx, tan_alpha_t, h_t_interface, T_interface,
                    E_for_display_T=E_for_T,
                )

            # 记录调试信息（记录本轮产出的新状态）
            if return_debug:
                debug["F"].append(float(F_new))
                debug["t"].append(t_new.copy())
                debug["E_interface"].append(E_interface_next.copy())
                debug["T_interface"].append(T_interface.copy())

            # 收敛检查（仍以 |F_new - F_old| 为准，这里的 F_old 是上一轮的值）
            if abs(F_new - F_old) < tolerance:
                converged = True
                F_old = float(F_new)
                t = t_new
                E_interface_prev = E_interface_next.copy()
                delta_E_prev = delta_E_next.copy()
                break

            # 未收敛：更新状态，准备下一轮
            F_old = float(F_new)
            t = t_new
            E_interface_prev = E_interface_next.copy()
            delta_E_prev = delta_E_next.copy()

        # 最终处理
        if return_debug:
            debug["F_final"] = F_old
            debug["t_final"] = t

        return F_old, converged, (it + 1), (debug if return_debug else None)

    def _print_step0_table(self, n, c, phi_deg, u, dx, alpha, B_0, A_0, n_alpha_0, delta_E, E_interface, F0):
        """输出 GPS 第一步（t=0）的 B, A', nα, A, ΔE, E 及 F0。"""
        A_prime = A_0
        A_over_n = A_0 / n_alpha_0
        print("\n" + "=" * 120)
        print("Step 0 (t=0) - B, A', nα, A, ΔE, E  (no T/t; used for next step)")
        print("=" * 120)
        header = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(header.format("Slice", "B", "A'", "nα", "A", "ΔE", "E"))
        print("-" * 120)
        for i in range(n):
            print(f"{i+1:<6} {B_0[i]:<12.3f} {A_prime[i]:<12.3f} {n_alpha_0[i]:<12.4f} {A_over_n[i]:<12.3f} {delta_E[i]:<12.3f} {E_interface[i+1]:<12.3f}")
        print(f"{'Σ':<6} {np.sum(B_0):<12.3f} {np.sum(A_prime):<12.3f} {'':<12} {np.sum(A_over_n):<12.3f} {np.sum(delta_E):<12.3f} {E_interface[-1]:<12.3f}")
        print(f"\nF0 = Σ(A) / Σ(B) = {np.sum(A_over_n):.3f} / {np.sum(B_0):.3f} = {F0:.3f}")
        print("=" * 120)

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
        

        # 计算 A' (A_prime)
        A_prime = (c + (p + t - u) * tan_phi) * dx
        A2_prime = (c + (p + t_new - u) * tan_phi) * dx
        
        # 计算 ΔT
        delta_T = np.zeros(n)
        for i in range(1, n + 1):
            delta_T[i - 1] = T_interface[i] - T_interface[i - 1]
        


        # 输出 t 和更新后的 B, A
        print("\n" + "-" * 120)
        print(f"{iteration_name} - Calculation of F (with updated t)")
        print("-" * 120)
        header_iter = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(header_iter.format("Slice", "ΔT", "t", "B", "A'", "nα", "A", "ΔE", "E"))
        print("-" * 120)
        for i in range(n):
            print(f"{i+1:<6} {delta_T[i]:<12.3f} {t_new[i]:<12.4f} {B2[i]:<12.3f} {A2_prime[i]:<12.3f} {n_alpha[i]:<12.4f} {A2[i]/n_alpha[i]:<12.3f} {delta_E[i]:<12.3f} {E_interface[i+1]:<12.3f}")
        print(f"{'Σ':<6} {np.sum(delta_T):<12.3f} {np.sum(t_new):<12.4f} {np.sum(B2):<12.3f} {np.sum(A2_prime):<12.3f} {'':<12} {np.sum(A2/n_alpha):<12.3f} {np.sum(delta_E):<12.3f} {E_interface[-1]:<12.3f}")
        print(f"\nF{it + 1} = Σ(A) / Σ(B) = {np.sum(A2/n_alpha):.3f} / {np.sum(B2):.3f} = {F_new:.3f}")
        print("=" * 120)




if __name__ == "__main__":
    """
    janbu.py 内置示例：
    - 目的是构造一个“强度和几何都比较温和、接近 janbu_verification.py 教材算例量级”的圆弧滑动面，
      用来单独检查 Janbu GPS 流程在圆弧模式下的收敛与中间量是否合理。
    - 为了便于在命令行反复运行，本示例默认**关闭所有绘图**，只打印条分与迭代表。
    """

    # 1. 构建几何（与之前示例类似的 1:1 边坡，只是高度略小一点）
    gb = GeometryBuilder(slope_height=8.0, slope_ratio=1.5, bottom_extension=5.0, top_extension=15.0)
    ground, region = gb.build()

    # 2. 圆弧滑动面与土性参数（量级参考 janbu_verification.py：c'≈1 kPa, φ'≈33.8°，不考虑孔压）
    gamma = 18.0          # 土重度，保持常见取值
    c_prime = 1.0         # kPa
    phi_prime = 33.8      # deg
    ru = 0.0              # 这里用 ru=0，简化为无孔压情形
    n_slices = 4          # 条数与 janbu_verification.py 验证算例相同
    q = 0.0               # 坡顶无均布超载

    # 估计一个“中等尺寸”的圆弧：
    # - 入口取在坡脚附近（x_entry ≈ L_bot）
    # - 圆心大致放在坡肩外侧上方，使弧线穿过坡趾并向坡后延伸
    x_entry = 5.0              # 必须在坡底宽度 [0, L_bot] 内（本例 L_bot=5）
    center = (11.0, 9.0)       # 经验估计的圆心 (xc, yc)，你可以在调试时微调

    fos, slices, meta = calculate_fos_for_circular_arc(
        ground_profile=ground,
        gamma=gamma,
        c_prime=c_prime,
        phi_prime=phi_prime,
        ru=ru,
        n_slices=n_slices,
        center=center,
        x_entry=x_entry,
        q=q,
        require_exit_at_crest=True,  # 约束：必须到达坡顶高度且穿出点在坡顶平台
        use_gps=True,
        print_iteration_table=True,  # 打印 F0、各轮 Fk 及 ΔE、E、T 等表格
        plot_f_history=True,        # 暂时关闭 F-iteration 曲线
        gps_max_iter=20,             # 允许 GPS 至多迭代 20 轮以观察收敛行为
    )

    if slices is None or not np.isfinite(fos):
        raise RuntimeError(f"该特定圆弧无效或计算失败：meta={meta}")

    # 3. 可视化
    x_exit = meta["x_exit"]
    xc, yc = center
    R = meta["radius"]
    xs_plot = np.linspace(x_entry, x_exit, 200)
    ys_plot = yc - np.sqrt(np.maximum(0.0, R * R - (xs_plot - xc) ** 2))
    slip_profile_plot = np.column_stack([xs_plot, ys_plot])
    plot_slope_and_slip(ground_profile=ground, slip_profile=slip_profile_plot, slope_region=region, show=True)

    print("=" * 60)
    print("Janbu 方法 - 预处理结果（土条数据）")
    print("=" * 60)
    print("i\t x_mid\t  alpha(deg)\t   p(kPa)\t  W(kN/m)")
    for i, s in enumerate(slices, start=1):
        print(f"{i:02d}\t{s['x_mid']:.3f}\t{s['alpha_deg']:.3f}\t{s['p']:.3f}\t{s['W']:.3f}")
    
    # 4. 输出该特定圆弧的 FoS（包含 F0 与 GPS 结果）
    print("\n" + "=" * 60)
    print("Janbu 方法 - FOS 计算结果")
    print("=" * 60)
    print(f"圆心 center: {center}")
    print(f"入口 x_entry: {x_entry:.3f}")
    print(f"穿出 x_exit: {meta['x_exit']:.3f} (crest_x={meta['crest_x']:.3f}, H={meta['H']:.3f})")
    print(f"半径 R: {meta['radius']:.3f}")
    if meta.get('use_gps'):
        F0 = meta.get('F0', np.nan)
        print(f"初始化安全系数 F0(t=0): {F0:.6f}")
        print(f"GPS 完整迭代安全系数 F: {fos:.6f}")
        print(f"收敛状态: {'已收敛' if meta.get('converged') else '未收敛（达到最大迭代次数）'}")
        print(f"迭代次数: {meta.get('iterations')}")
    else:
        print(f"FoS (F0, t=0): {fos:.6f}")
    print("=" * 60)

