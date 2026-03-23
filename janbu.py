import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class GeometryBuilder:
    """Build ground profile and closed slope region polygon."""

    def __init__(self, slope_height, slope_ratio, bottom_extension, top_extension):
        """Initialize geometric parameters."""
        self.H = float(slope_height)
        self.m = float(slope_ratio)
        self.L_bot = float(bottom_extension)
        self.L_top = float(top_extension)

        if self.H <= 0 or self.m <= 0:
            raise ValueError("slope_height and slope_ratio must be positive.")

    def build(self):
        """Return `ground_profile` and `slope_region`."""
        # Use toe as the origin to keep coordinates consistent with Fellenius/Bishop.
        A = (-self.L_bot, 0.0)
        B = (0.0, 0.0)
        C_x = self.H * self.m
        C = (C_x, self.H)
        D_x = C_x + self.L_top
        D = (D_x, self.H)

        ground_profile = [A, B, C, D]

        # Close the polygon at y=0 to avoid diagonal fill artifacts.
        slope_region = np.array([A, B, C, D, (D_x, 0.0)])
        return ground_profile, slope_region


class JanbuPreprocessor:
    """Precompute slice geometry and load terms for Janbu solver."""

    def __init__(self, gamma):
        """Set unit weight `gamma`."""
        self.gamma = float(gamma)

    @staticmethod
    def _interp_y(xs, ys, x):
        """Interpolate y(x) along a polyline."""
        return float(np.interp(x, xs, ys))

    @staticmethod
    def _interp_y_and_slope(xs, ys, x):
        """Interpolate y(x) and local slope dy/dx on monotonic `xs`."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size or xs.size < 2:
            raise ValueError("Polyline nodes must be 1D arrays (length >= 2) with matching x/y sizes.")

        idx = np.searchsorted(xs, x) - 1
        idx = int(np.clip(idx, 0, xs.size - 2))
        x1, x2 = xs[idx], xs[idx + 1]
        y1, y2 = ys[idx], ys[idx + 1]
        if x2 == x1:
            # Guard vertical segments to avoid numerical issues.
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
        """Create vertical slices along a polyline slip surface."""
        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer.")

        gp = np.asarray(ground_profile, dtype=float)
        sp = np.asarray(slip_profile, dtype=float)

        xs_surface, ys_surface = gp[:, 0], gp[:, 1]
        xs_slip, ys_slip = sp[:, 0], sp[:, 1]

        # Slice only where ground and slip share an x-range overlap.
        x_min = max(xs_surface.min(), xs_slip.min())
        x_max = min(xs_surface.max(), xs_slip.max())

        total_width = x_max - x_min
        if total_width <= 0:
            raise ValueError("Ground and slip surface have no valid overlap in x-range.")

        b_width = total_width / n_slices

        slices = []
        skipped_indices = []
        for i in range(n_slices):
            x_left = x_min + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            y_base, slope_slip = self._interp_y_and_slope(xs_slip, ys_slip, x_mid)
            alpha_rad = np.arctan(slope_slip)

            y_top = self._interp_y(xs_surface, ys_surface, x_mid)
            h_mid = y_top - y_base
            if h_mid <= 0:
                skipped_indices.append(i + 1)
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

        if skipped_indices:
            print(f"\n[Warning] Skipped {len(skipped_indices)} slices (slip surface above ground).")
            print(f"          Skipped slice indices: {skipped_indices}")
            print(f"          Effective slices: {len(slices)} / {n_slices}")
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
        """Create vertical slices directly from a circular slip arc."""
        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer.")

        gp = np.asarray(ground_profile, dtype=float)
        xs_surface, ys_surface = gp[:, 0], gp[:, 1]
        # GeometryBuilder convention: A=(-L_bot,0), B=(0,0), C=(crest_x,H).
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

        # Require exit at crest elevation; optionally enforce platform exit.
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
            y_base = float(yc - np.sqrt(base_term))

            # Arc tangent slope from circle geometry: dy/dx = -(x-xc)/(y-yc).
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
    """Decode factors into a slip profile (arc mode for 3 factors, else polyline mode)."""
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
    """Return intersections between a circle and segment p1->p2."""
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
    """Find rightmost circle-ground intersection to the right of entry."""
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
    """Build a circular-arc slip profile and return it as polyline points."""
    gp = np.asarray(ground_profile, dtype=float)
    A = gp[0]
    B = gp[1]
    x_min_bot = float(min(A[0], B[0]))
    x_max_bot = float(max(A[0], B[0]))
    x_entry = float(x_entry)
    if not (x_min_bot - eps <= x_entry <= x_max_bot + eps):
        raise ValueError(f"x_entry must be within toe-width range [{x_min_bot}, {x_max_bot}], got {x_entry}.")

    xc, yc = map(float, center)
    entry_pt = (x_entry, 0.0)
    radius = float(np.hypot(xc - entry_pt[0], yc - entry_pt[1]))
    if radius <= 0:
        raise ValueError("radius must be positive.")

    exit_pt = _find_arc_exit_on_surface(ground_profile, (xc, yc), radius, x_entry)
    if exit_pt is None:
        raise ValueError("No valid exit point exists to the right of entry; cannot build arc slip surface.")
    x_exit, y_exit = exit_pt

    # Uniformly discretize the lower arc branch in x.
    xs = np.linspace(x_entry, x_exit, int(n_points))
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        term = radius * radius - (x - xc) ** 2
        if term < 0:
            ys[i] = np.nan
        else:
            ys[i] = yc - np.sqrt(term)

    # Keep arc points strictly below ground using a small epsilon.
    y_ground = np.interp(xs, gp[:, 0], gp[:, 1])
    ys = np.minimum(ys, y_ground - eps)

    slip_profile = np.column_stack([xs, ys])
    slip_profile = slip_profile[~np.isnan(slip_profile[:, 1])]
    if slip_profile.shape[0] < 2:
        raise ValueError("Insufficient arc points after discretization; invalid slip surface.")
    return slip_profile


def build_slip_profile_circular_arc_from_factors(ground_profile, factors, n_points=80):
    """Decode `[fx_center, fy_center, fx_entry]` into a circular slip profile."""
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
    use_gps=True,
    gps_tolerance=1e-6,
    gps_max_iter=80,
    lambda_thrust=0.33,
    require_exit_at_crest=True,
    center=None,
    x_entry_single=None,
):
    """Search circular arcs and return the minimum valid FoS."""
    gp = np.asarray(ground_profile, dtype=float)
    A = gp[0]
    B = gp[1]
    x_min_bot = float(min(A[0], B[0]))
    x_max_bot = float(max(A[0], B[0]))

    if entry_x_range is None:
        entry_x_list = np.array([x_max_bot], dtype=float)
    else:
        entry_x_list = np.atleast_1d(np.asarray(entry_x_range, dtype=float))

    pre = JanbuPreprocessor(gamma=gamma)
    solver = JanbuSolver(c_prime=c_prime, phi_prime=phi_prime, ru=ru, u_i=None, delta_Q_i=0.0)

    # Single-center mode skips the full center grid search.
    if center is not None:
        xc, yc = map(float, center)

        if x_entry_single is None:
            x_entry_use = x_max_bot
        else:
            x_entry_use = float(x_entry_single)

        if x_entry_use < x_min_bot or x_entry_use > x_max_bot:
            return None, [(xc, yc, np.nan)]

        slices, meta = pre.slice_along_circular_arc(
            ground_profile=ground_profile,
            center=(xc, yc),
            x_entry=x_entry_use,
            n_slices=n_slices,
            q=q,
            require_exit_at_crest=require_exit_at_crest,
        )
        if slices is None:
            return None, [(xc, yc, np.nan)]

        radius = float(meta["radius"])
        F0, conv0, it0 = solver.calculate_fos_initial(
            slices, F_init=1.0, tolerance=gps_tolerance, max_iter=50
        )
        if not np.isfinite(F0):
            return None, [(xc, yc, np.nan)]

        if use_gps:
            F, converged, iterations, _ = solver.calculate_fos_gps(
                slices=slices,
                slip_profile=None,
                ground_profile=ground_profile,
                F_init=F0,
                tolerance=gps_tolerance,
                max_iter=gps_max_iter,
                lambda_thrust=lambda_thrust,
                print_iteration_table=False,
                arc_center=(xc, yc),
                arc_radius=radius,
            )
            fos_val = float(F)
        else:
            fos_val = float(F0)

        if not (np.isfinite(fos_val) and fos_val > 0):
            best_single = None
        else:
            best_single = {
                "center": (xc, yc),
                "radius": radius,
                "x_entry": x_entry_use,
                "fos": fos_val,
                "use_gps": bool(use_gps),
            }

        fos_results_single = [(xc, yc, fos_val if np.isfinite(fos_val) else np.nan)]
        return best_single, fos_results_single

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
                    # Use the same arc slicing path as single-arc evaluation for consistency.
                    slices, meta = pre.slice_along_circular_arc(
                        ground_profile=ground_profile,
                        center=(float(xc), float(yc)),
                        x_entry=x_entry,
                        n_slices=n_slices,
                        q=q,
                        require_exit_at_crest=require_exit_at_crest,
                    )
                    if slices is None:
                        continue

                    radius = float(meta["radius"])
                    F0, conv0, it0 = solver.calculate_fos_initial(
                        slices, F_init=1.0, tolerance=gps_tolerance, max_iter=50
                    )
                    if not np.isfinite(F0):
                        continue

                    if use_gps:
                        F, converged, iterations, _ = solver.calculate_fos_gps(
                            slices=slices,
                            slip_profile=None,
                            ground_profile=ground_profile,
                            F_init=F0,
                            tolerance=gps_tolerance,
                            max_iter=gps_max_iter,
                            lambda_thrust=lambda_thrust,
                            print_iteration_table=False,
                            arc_center=(float(xc), float(yc)),
                            arc_radius=radius,
                        )
                        fos = float(F)
                    else:
                        fos = float(F0)

                    # Accept only physically meaningful FoS values.
                    if np.isfinite(fos) and fos > 0 and fos < best_fos_here:
                        best_fos_here = fos
                        best_here = {
                            "center": (float(xc), float(yc)),
                            "radius": radius,
                            "x_entry": x_entry,
                            "fos": fos,
                            "use_gps": bool(use_gps),
                        }
                except Exception:
                    continue

            fos_results.append((float(xc), float(yc), best_fos_here if (np.isfinite(best_fos_here) and best_fos_here < np.inf) else np.nan))

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
    return_debug=False,
):
    """Compute FoS for a given circular arc using initial or GPS solver path."""
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

    # Without GPS, solve only the initial t=0 Janbu equation.
    if not use_gps:
        F0, conv0, it0 = solver.calculate_fos_initial(
            slices, F_init=1.0, tolerance=gps_tolerance, max_iter=50
        )
        if not np.isfinite(F0):
            meta = {**meta, "reason": "F0_non_finite", "F0": F0}
            return np.nan, slices, meta
        return float(F0), slices, {**meta, "F0": float(F0), "use_gps": False}

    # With GPS enabled, F0 is computed internally as step 0.
    F, converged, iterations, debug = solver.calculate_fos_gps(
        slices=slices,
        slip_profile=None,
        ground_profile=ground_profile,
        F_init=1.0,
        tolerance=gps_tolerance,
        max_iter=gps_max_iter,
        lambda_thrust=lambda_thrust,
        t_init=None,
        return_debug=bool(return_debug),
        print_iteration_table=print_iteration_table,
        arc_center=center,
        arc_radius=meta.get("radius"),
    )

    # Optional convergence plot for per-iteration FoS.
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

    # Pull F0 from GPS debug output when available.
    F0_from_gps = debug.get("F0", np.nan) if debug is not None else np.nan
    result_meta = {
        **meta,
        "F0": float(F0_from_gps),
        "use_gps": True,
        "converged": bool(converged),
        "iterations": int(iterations),
    }
    if return_debug:
        result_meta["debug"] = debug
    return float(F), slices, result_meta


def build_slip_profile_polyline_from_factors(
    ground_profile,
    factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """Legacy polyline slip-profile decoder (kept for GA compatibility)."""
    gp = np.asarray(ground_profile, dtype=float)
    xs_surface, ys_surface = gp[:, 0], gp[:, 1]

    factors = np.asarray(factors, dtype=float)

    if factors.ndim == 1:
        if factors.size % 2 != 0:
            raise ValueError("For 1D factors, length must be even (2*n_points).")
        n_points = factors.size // 2
        x_factors = factors[:n_points]
        y_factors = factors[n_points:]
    elif factors.ndim == 2:
        if factors.shape[1] != 2:
            raise ValueError("For 2D factors, second dimension must be 2 (n_points, 2).")
        n_points = factors.shape[0]
        x_factors = factors[:, 0]
        y_factors = factors[:, 1]
    else:
        raise ValueError("factors must be a 1D or 2D array.")

    if n_points < 2:
        raise ValueError("Slip-surface control points must be at least 2.")

    # 1) Build allowed X range.
    x_min, x_max = xs_surface.min(), xs_surface.max()
    width = x_max - x_min
    if width <= 0:
        raise ValueError("Invalid x-coordinates in ground_profile.")

    left_ratio, right_ratio = x_range_ratio
    left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
    right_ratio = float(np.clip(right_ratio, 0.0, 1.0))
    if right_ratio <= left_ratio:
        raise ValueError("x_range_ratio must satisfy left < right.")

    xa = x_min + left_ratio * width
    xb = x_min + right_ratio * width

    # 2) Map X factors from [0,1] to [xa, xb].
    x_factors_clipped = np.clip(x_factors, 0.0, 1.0)
    xs_ctrl = xa + x_factors_clipped * (xb - xa)

    # Enforce left-to-right ordering for slip control points.
    xs_ctrl = np.sort(xs_ctrl)

    # 3) Interpolate ground elevations at control points.
    ys_ground = np.interp(xs_ctrl, xs_surface, ys_surface)

    if max_depth_below_zero is None:
        H_max = ys_ground.max()
        max_depth_below_zero = 0.5 * H_max

    # 4) Convert Y factors to depth; allow extension below y=0 when >1.
    depths_to_zero = ys_ground
    min_depths = min_relative_depth * depths_to_zero

    depths = np.zeros_like(y_factors)
    for i in range(len(y_factors)):
        yf = y_factors[i]
        if yf <= 1.0:
            depths[i] = min_depths[i] + yf * (depths_to_zero[i] - min_depths[i])
        else:
            depths[i] = depths_to_zero[i] + (yf - 1.0) * max_depth_below_zero

    ys_slip = ys_ground - depths

    slip_profile = np.column_stack([xs_ctrl, ys_slip])
    return slip_profile


# Backward-compatible alias.
def build_slip_profile_from_depth_factors(
    ground_profile,
    depth_factors,
    x_range_ratio=(0.1, 0.9),
    min_relative_depth=0.02,
    max_depth_below_zero=None,
):
    """Backward-compatible alias for depth-factor based profile construction."""
    depth_factors = np.asarray(depth_factors, dtype=float)
    
    if depth_factors.ndim == 1:
        n_points = depth_factors.size
        x_factors = np.linspace(0.0, 1.0, n_points)
        factors = np.column_stack([x_factors, depth_factors])
        return build_slip_profile_from_factors(
            ground_profile, factors, x_range_ratio, min_relative_depth, max_depth_below_zero
        )
    else:
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
    """Plot ground profile, optional slip profile, and optional soil region."""
    gp = np.asarray(ground_profile, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if slope_region is not None:
        reg = np.asarray(slope_region, dtype=float)
        ax.fill(reg[:, 0], reg[:, 1], color="#c7c7c7", alpha=0.35, label="Soil Region")

    ax.plot(gp[:, 0], gp[:, 1], "k-", linewidth=3, label="Ground Surface")

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

    # Auto-scale plot limits with a small margin.
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


def plot_janbu_search_result(
    ground_profile,
    best_circle,
    fos_results,
    center_grid_x,
    center_grid_y,
    slope_region=None,
    title="Janbu - Critical Slip Surface Search",
):
    """Plot FoS contour, slope geometry, and critical circular slip surface."""
    fig, ax = plt.subplots(figsize=(14, 8))
    gp = np.asarray(ground_profile, dtype=float)

    # 1) FoS contour map (if valid data exists).
    if fos_results:
        Z = np.array([r[2] for r in fos_results], dtype=float).reshape(len(center_grid_x), len(center_grid_y)).T
        if np.any(np.isfinite(Z)):
            contours = ax.contourf(center_grid_x, center_grid_y, Z, levels=20, cmap="viridis_r", alpha=0.7)
            fig.colorbar(contours, ax=ax, label="Factor of Safety (FoS)")

    # 2) Slope surface and optional soil region.
    if slope_region is not None:
        reg = np.asarray(slope_region, dtype=float)
        ax.fill(reg[:, 0], reg[:, 1], color="#c7c7c7", alpha=0.35, label="Soil Region")
    ax.plot(gp[:, 0], gp[:, 1], "k-", linewidth=3, label="Ground Surface")

    # 3) Critical slip arc and center.
    if best_circle is not None:
        center = best_circle["center"]
        radius = best_circle["radius"]
        fos = best_circle["fos"]
        slip_circle = Circle(
            center, radius, fill=False, edgecolor="red",
            linewidth=2, linestyle="--", label=f"Critical Circle (FoS={fos:.3f})"
        )
        ax.add_patch(slip_circle)
        ax.plot(center[0], center[1], "r+", markersize=15, label="Critical circle center")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")

    x_min, x_max = gp[:, 0].min(), gp[:, 0].max()
    y_min, y_max = gp[:, 1].min(), gp[:, 1].max()
    if len(center_grid_y) > 0:
        y_max = max(y_max, np.max(center_grid_y))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min(0.0, y_min) - (y_max - y_min) * 0.2, y_max + (y_max - y_min) * 0.2)
    plt.show()


class JanbuSolver:
    """Core Janbu solver with initial and GPS iteration modes."""

    def __init__(self, c_prime, phi_prime, ru=0.0, u_i=None, delta_Q_i=None):
        """Set material parameters and optional per-slice loads."""
        self.c_prime = c_prime
        self.phi_prime = phi_prime
        self.ru = float(ru)
        self.u_i = u_i if u_i is not None else 0.0
        self.delta_Q_i = delta_Q_i if delta_Q_i is not None else 0.0

    def _get_slice_param(self, param, index, n_slices):
        """Return scalar parameter or per-slice value at `index`."""
        if isinstance(param, (list, np.ndarray)):
            if len(param) != n_slices:
                raise ValueError(f"Parameter list length ({len(param)}) must equal number of slices ({n_slices}).")
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
        """Compute initial FoS F0 at t=0 via fixed-point iteration."""
        if not slices:
            raise ValueError("slices list cannot be empty.")

        n_slices = len(slices)
        
        c_list = [self._get_slice_param(self.c_prime, i, n_slices) for i in range(n_slices)]
        phi_list = [self._get_slice_param(self.phi_prime, i, n_slices) for i in range(n_slices)]
        if self.u_i is None:
            u_list = []
            for i in range(n_slices):
                W_i = float(slices[i]["W"])
                dx_i = float(slices[i]["width"])
                if dx_i <= 0:
                    raise ValueError("Slice width (Delta-x) must be positive.")
                u_list.append(self.ru * W_i / dx_i)  # kPa
        else:
            u_list = [self._get_slice_param(self.u_i, i, n_slices) for i in range(n_slices)]
        dQ_list = [self._get_slice_param(self.delta_Q_i, i, n_slices) for i in range(n_slices)]
        
        phi_rad_list = [np.radians(phi) for phi in phi_list]
        tan_phi_list = [np.tan(phi_rad) for phi_rad in phi_rad_list]

        p_list = [s['p'] for s in slices]
        alpha_rad_list = [s['alpha_rad'] for s in slices]
        dx_list = [s['width'] for s in slices]

        # Fixed-point iteration for initial FoS F0.
        F = float(F_init)
        converged = False
        
        for iteration in range(max_iter):
            F_old = F
            
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
                
                n_alpha_i = (cos_alpha ** 2) * (1.0 + tan_alpha * tan_phi_i / F)
                
                # Clamp negative resistance terms for physical consistency.
                A_i = (c_i + (p_i - u_i) * tan_phi_i) * dx_i
                if A_i < 0.0:
                    A_i = 0.0
                A_i = max(A_i, 0.0)
                B_i = p_i * dx_i * tan_alpha + dQ_i
                
                sum_A_over_n += A_i / n_alpha_i
                sum_B += B_i
            
            if abs(sum_B) < 1e-10:
                return np.inf, False, iteration + 1
            
            F = sum_A_over_n / sum_B
            
            if abs(F - F_old) < tolerance:
                converged = True
                break
        
        return F, converged, iteration + 1

    @staticmethod
    def _as_array(x, n, name):
        """Convert scalar/list input to a length-`n` float array."""
        if x is None:
            return None
        if isinstance(x, (list, np.ndarray)):
            if len(x) != n:
                raise ValueError(f"{name} length ({len(x)}) must equal number of slices ({n}).")
            return np.asarray(x, dtype=float)
        return np.full(n, float(x), dtype=float)

    @staticmethod
    def _interp_poly_y(xs, ys, x):
        """Interpolate y on a polyline."""
        return float(np.interp(x, xs, ys))
    
    def _prepare_slice_arrays(self, slices):
        """Extract numpy arrays from slice dictionaries."""
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
        """Prepare parameter arrays: c, phi, tan(phi), dQ."""
        c = self._as_array(self.c_prime, n, "c_prime")
        phi_deg = self._as_array(self.phi_prime, n, "phi_prime")
        tan_phi = np.tan(np.radians(phi_deg))
        
        dQ = self._as_array(self.delta_Q_i, n, "delta_Q_i")
        if dQ is None:
            dQ = np.zeros(n, dtype=float)
        
        return c, phi_deg, tan_phi, dQ
    
    def _calculate_pore_pressure(self, W, dx, n):
        """Compute pore pressure `u` from `ru` or explicit `u_i`."""
        if self.u_i is None:
            return self.ru * (W / dx)
        else:
            return self._as_array(self.u_i, n, "u_i")
    
    def _build_ground_profile_from_slices(self, slices, xs_slip, ys_slip):
        """Reconstruct ground profile from slices when not provided."""
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
        """Compute initial interface force state E0 and Delta-E."""
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
        """Compute thrust-line height and slope for polyline slip."""
        # Step A: interface height.
        y_ground_interface = np.interp(x_interface, xs_ground, ys_ground)
        y_slip_interface = np.interp(x_interface, xs_slip, ys_slip)
        H_int = y_ground_interface - y_slip_interface
        
        # Step B: thrust-line elevation.
        h_t_interface = float(lambda_thrust) * H_int
        y_thrust = y_slip_interface + h_t_interface
        
        # Step C: thrust-line slope.
        tan_alpha_t = self._calculate_thrust_line_slope(x_interface, y_thrust, n)
        
        return h_t_interface, tan_alpha_t, H_int

    def _calculate_thrust_line_geometry_arc(self, x_interface, xs_ground, ys_ground,
                                            center, radius, lambda_thrust, n):
        """Compute thrust-line geometry directly from circle equation."""
        xc, yc = map(float, center)
        R = float(radius)
        y_ground_interface = np.interp(x_interface, xs_ground, ys_ground)
        # Lower branch of the circle.
        term = R * R - (x_interface - xc) ** 2
        term = np.maximum(term, 0.0)
        y_slip_interface = yc - np.sqrt(term)
        H_int = y_ground_interface - y_slip_interface

        h_t_interface = float(lambda_thrust) * H_int
        y_thrust = y_slip_interface + h_t_interface

        tan_alpha_t = self._calculate_thrust_line_slope(x_interface, y_thrust, n)
        return h_t_interface, tan_alpha_t, H_int
    
    def _calculate_thrust_line_slope(self, x_interface, y_thrust, n):
        """Compute thrust-line slope `tan_alpha_t` at interfaces."""
        tan_alpha_t = np.zeros(n + 1, dtype=float)
        EPS = 1e-12
        
        # Interior interfaces: central difference.
        for i in range(1, n):
            dx_i = x_interface[i + 1] - x_interface[i - 1]
            if abs(dx_i) > EPS:
                tan_alpha_t[i] = (y_thrust[i + 1] - y_thrust[i - 1]) / dx_i
        
        # Boundary interfaces.
        if n > 0:
            dx_0 = x_interface[1] - x_interface[0]
            if abs(dx_0) > EPS:
                tan_alpha_t[0] = (y_thrust[1] - y_thrust[0]) / dx_0
            
            dx_n = x_interface[n] - x_interface[n - 1]
            if abs(dx_n) > EPS:
                tan_alpha_t[n] = (y_thrust[n] - y_thrust[n - 1]) / dx_n
        
        return tan_alpha_t
    
    def _calculate_dE_dx(self, delta_E_used, dx, n):
        """Compute dE/dx using Janbu Eq. 117 weighted averaging."""
        dE_dx = np.zeros(n + 1, dtype=float)
        dE_dx[0] = 0.0

        # Interior interfaces: weighted average (Janbu Eq. 117).
        EPS = 1e-12
        for i in range(1, n):
            slice_idx_L = i - 1
            slice_idx_R = i
            numerator = delta_E_used[slice_idx_L] + delta_E_used[slice_idx_R]
            denominator = dx[slice_idx_L] + dx[slice_idx_R]
            if abs(denominator) > EPS:
                dE_dx[i] = numerator / denominator
        
        if n > 0:
            dE_dx[n] = 0.0
        
        return dE_dx
    
    def _calculate_interface_shear_T(self, E_for_T, tan_alpha_t, h_t_interface, dE_dx, n):
        """Compute interface vertical shear force T."""
        T_interface = -E_for_T * tan_alpha_t + h_t_interface * dE_dx 
        T_interface[0] = 0.0
        return T_interface
    
    def _update_slice_shear_t(self, T_interface, dx, n):
        """Update per-slice vertical shear gradient `t`."""
        t_new = np.zeros(n, dtype=float)
        for i in range(1, n + 1):
            t_new[i - 1] = (T_interface[i] - T_interface[i - 1]) / dx[i - 1]
        return t_new
    
    def _calculate_new_F(self, c, p, t_new, u, tan_phi, dx, alpha, dQ, n_alpha, n):
        """Compute updated FoS F_new from current `t_new`."""
        tan_alpha = np.tan(alpha)
        A2 = (c + (p + t_new - u) * tan_phi) * dx
        # Clamp per-slice resistance to non-negative values.
        A2 = np.maximum(A2, 0.0)
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
        """Run full Janbu GPS iteration and optionally return debug traces."""
        if not slices:
            raise ValueError("slices cannot be empty.")

        n = len(slices)
        use_arc = arc_center is not None and arc_radius is not None
        if slip_profile is None and not use_arc:
            raise ValueError("Provide either slip_profile (polyline mode) or arc_center+arc_radius (arc mode).")
        if slip_profile is not None and use_arc:
            raise ValueError("slip_profile and arc_center/arc_radius are mutually exclusive.")

        if slip_profile is not None:
            sp = np.asarray(slip_profile, dtype=float)
            if sp.ndim != 2 or sp.shape[1] != 2 or sp.shape[0] < 2:
                raise ValueError("slip_profile must be an array/list with shape (n_points, 2).")

        arrays = self._prepare_slice_arrays(slices)
        dx = arrays['dx']
        alpha = arrays['alpha']
        p = arrays['p']
        W = arrays['W']
        x_left = arrays['x_left']
        x_right = arrays['x_right']

        c, phi_deg, tan_phi, dQ = self._prepare_parameters(n)
        u = self._calculate_pore_pressure(W, dx, n)

        if t_init is None:
            t = np.zeros(n, dtype=float)
        else:
            t = self._as_array(t_init, n, "t_init")

        # Interface x-coordinates, length n+1.
        x_interface = np.concatenate([[x_left[0]], x_right])

        if slip_profile is not None:
            xs_slip, ys_slip = sp[:, 0], sp[:, 1]
            if ground_profile is None:
                ground_profile = self._build_ground_profile_from_slices(slices, xs_slip, ys_slip)
            else:
                ground_profile = np.asarray(ground_profile, dtype=float)
        else:
            if ground_profile is None:
                raise ValueError("ground_profile is required in arc mode.")
            ground_profile = np.asarray(ground_profile, dtype=float)
            xs_slip = ys_slip = None
        xs_ground, ys_ground = ground_profile[:, 0], ground_profile[:, 1]

        debug = {"F": [], "t": [], "E_interface": [], "T_interface": []} if return_debug else None
        converged = False

        if print_iteration_table:
            self._print_initial_data_table(slices, c, phi_deg, u, dQ, dx, alpha, p)

        # Step 0: compute F0 and Delta-E once at t=0.
        F_old = float(F_init)
        tan_alpha = np.tan(alpha)
        cos_alpha = np.cos(alpha)
        n_alpha_0 = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)
        A_0 = (c + (p - u) * tan_phi) * dx
        A_0 = np.maximum(A_0, 0.0)
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

        # Main GPS loop from step 1 onward.
        t = np.zeros(n, dtype=float)
        it = -1
        for it in range(max_iter):
            # Step 1: thrust-line geometry (independent of E).
            if use_arc:
                h_t_interface, tan_alpha_t, _ = self._calculate_thrust_line_geometry_arc(
                    x_interface, xs_ground, ys_ground, arc_center, arc_radius, lambda_thrust, n
                )
            else:
                h_t_interface, tan_alpha_t, _ = self._calculate_thrust_line_geometry(
                    x_interface, xs_ground, ys_ground, xs_slip, ys_slip, lambda_thrust, n
                )

            # Step 2: compute dE/dx and interface shear from previous E state.
            E_for_T = E_interface_prev
            dE_dx = self._calculate_dE_dx(delta_E_prev, dx, n)

            T_interface = self._calculate_interface_shear_T(
                E_for_T, tan_alpha_t, h_t_interface, dE_dx, n
            )

            # Step 3: update per-slice shear gradient t from interface shear.
            t_new = self._update_slice_shear_t(T_interface, dx, n)

            # Step 4: compute F_new using n_alpha based on F_old.
            tan_alpha = np.tan(alpha)
            cos_alpha = np.cos(alpha)
            n_alpha = (cos_alpha**2) * (1.0 + (tan_alpha * tan_phi) / F_old)
            F_new, A2, B2 = self._calculate_new_F(
                c, p, t_new, u, tan_phi, dx, alpha, dQ, n_alpha, n
            )
            if F_new is None:
                return np.inf, False, it + 1, (debug if return_debug else None)

            # Step 5: update Delta-E and E for next iteration.
            # Delta-E must use the same n_alpha set as the current F_k evaluation:
            #   F_k = Σ(A_k/nα_k) / Σ(B_k)
            #   ΔE_k = B_k - (A_k/nα_k) / F_k
            #   ⇒ ΣΔE_k = ΣB_k - (1/F_k) Σ(A_k/nα_k) = 0
            A_next = A2
            B_next = B2
            delta_E_next = B_next - (A_next / n_alpha) / F_new
            E_interface_next = np.zeros(n + 1, dtype=float)
            E_interface_next[1:] = np.cumsum(delta_E_next)

            if print_iteration_table:
                self._print_iteration_table(
                    it, F_old, F_new,
                    slices, c, phi_deg, u, dQ, dx, alpha, p, t, t_new,
                    A2, A2, B2, B2, n_alpha, delta_E_next, E_interface_next,
                    dE_dx, tan_alpha_t, h_t_interface, T_interface,
                    E_for_display_T=E_for_T,
                )

            if return_debug:
                debug["F"].append(float(F_new))
                debug["t"].append(t_new.copy())
                debug["E_interface"].append(E_interface_next.copy())
                debug["T_interface"].append(T_interface.copy())

            # Convergence check against previous FoS.
            if abs(F_new - F_old) < tolerance:
                converged = True
                F_old = float(F_new)
                t = t_new
                E_interface_prev = E_interface_next.copy()
                delta_E_prev = delta_E_next.copy()
                break

            F_old = float(F_new)
            t = t_new
            E_interface_prev = E_interface_next.copy()
            delta_E_prev = delta_E_next.copy()

        if return_debug:
            debug["F_final"] = F_old
            debug["t_final"] = t

        return F_old, converged, (it + 1), (debug if return_debug else None)

    def _print_step0_table(self, n, c, phi_deg, u, dx, alpha, B_0, A_0, n_alpha_0, delta_E, E_interface, F0):
        """Print detailed table for GPS step 0 (t=0)."""
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
        """Print initial slice data table."""
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
        """Print detailed intermediate values for one GPS iteration."""
        n = len(slices)
        tan_alpha = np.tan(alpha)
        tan_phi = np.tan(np.radians(phi_deg))
        
        iteration_name = f"Iteration F{it + 1}"
        
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
        

        A_prime = (c + (p + t - u) * tan_phi) * dx
        A2_prime = (c + (p + t_new - u) * tan_phi) * dx
        
        delta_T = np.zeros(n)
        for i in range(1, n + 1):
            delta_T[i - 1] = T_interface[i] - T_interface[i - 1]
        


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
    """Built-in demo: search critical circular slip surface and report minimum FoS."""

    # 1) Build geometry.
    gb = GeometryBuilder(slope_height=8.0, slope_ratio=1.5, bottom_extension=5.0, top_extension=15.0)
    ground, region = gb.build()

    # 2) Soil parameters.
    gamma = 18.0
    c_prime = 1.0
    phi_prime = 33.8
    ru = 0.0
    n_slices = 4
    q = 0.0

    # 3) Search ranges for center and entry point.
    center_grid_x = np.linspace(0, 25.0, 30)
    center_grid_y = np.linspace(1.0, 15.0, 20)
    entry_x_range = np.linspace(0.0, 5.0, 6)
    plot = True

    print("\n--- Starting Janbu search (arc through crest) ---")
    print(f"Search grid: {len(center_grid_x)} (x) × {len(center_grid_y)} (y) × {len(entry_x_range)} (entry_x)")

    best, fos_results = find_critical_fos_circular_arc(
        ground_profile=ground,
        gamma=gamma,
        c_prime=c_prime,
        phi_prime=phi_prime,
        ru=ru,
        n_slices=n_slices,
        center_grid_x=center_grid_x,
        center_grid_y=center_grid_y,
        entry_x_range=entry_x_range,
        q=q,
        use_gps=True,
        gps_tolerance=1e-6,
        gps_max_iter=80,
        require_exit_at_crest=True,
    )

    print("--- Search complete ---")
    if best is None:
        print("No valid slip surface found in the defined grid.")
        print("Try adjusting center_grid_x, center_grid_y, or entry_x_range.")
    else:
        print(f"Minimum safety factor (FoS_min): {best['fos']:.3f}")
        print(f"Most dangerous center O(x,y): ({best['center'][0]:.2f}, {best['center'][1]:.2f})")
        print(f"Most dangerous radius R: {best['radius']:.2f}")
        print(f"Entry point x_entry: {best['x_entry']:.2f}")
        if plot and fos_results:
            plot_janbu_search_result(
                ground_profile=ground,
                best_circle=best,
                fos_results=fos_results,
                center_grid_x=center_grid_x,
                center_grid_y=center_grid_y,
                slope_region=region,
                title="Janbu - Critical Slip Surface Search",
            )

