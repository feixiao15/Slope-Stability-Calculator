"""
Janbu verification script using tabulated reference values.

Reference target:
- F0 ~= 1.385
- F1 ~= 1.485
- F2 ~= 1.480
"""

import numpy as np

from janbu import JanbuSolver


def create_reference_slices():
    tan_alpha = np.array([1.13, 0.50, 0.18, -0.04], dtype=float)
    dx = np.array([4.4, 11.0, 11.0, 6.0], dtype=float)
    p = np.array([5.3, 10.1, 8.6, 2.9], dtype=float)
    u = np.array([2.12, 4.04, 3.44, 1.16], dtype=float)
    c = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    tan_phi = np.array([0.67, 0.67, 0.67, 0.67], dtype=float)
    dQ = np.zeros_like(dx)

    slices = []
    for i in range(dx.size):
        alpha_rad = float(np.arctan(tan_alpha[i]))
        slices.append(
            {
                "width": float(dx[i]),
                "alpha_rad": alpha_rad,
                "alpha_deg": float(np.degrees(alpha_rad)),
                "p": float(p[i]),
                "W": float(p[i] * dx[i]),
                "h_mid": 1.0,
                "x_mid": float(np.sum(dx[:i]) + 0.5 * dx[i]),
                "x_left": float(np.sum(dx[:i])),
                "x_right": float(np.sum(dx[: i + 1])),
            }
        )
    return slices, u, c, tan_phi, dQ


def build_polyline_from_slices(slices):
    pts = [[0.0, 0.0]]
    x_cur, y_cur = 0.0, 0.0
    for s in slices:
        dx = float(s["width"])
        tan_a = float(np.tan(float(s["alpha_rad"])))
        x_cur += dx
        y_cur += dx * tan_a
        pts.append([x_cur, y_cur])
    return np.asarray(pts, dtype=float)


def main():
    print("=" * 90)
    print("Janbu verification with reference table data")
    print("=" * 90)

    slices, u_table, c_table, tan_phi_table, dQ_table = create_reference_slices()
    phi_table = [float(np.degrees(np.arctan(t))) for t in tan_phi_table]

    solver = JanbuSolver(
        c_prime=c_table.tolist(),
        phi_prime=phi_table,
        ru=0.0,
        u_i=u_table.tolist(),
        delta_Q_i=dQ_table.tolist(),
    )

    F0, conv0, it0 = solver.calculate_fos_initial(slices=slices, F_init=1.0, tolerance=1e-6, max_iter=50)
    print(f"F0 = {F0:.6f} | expected ~1.385 | converged={conv0} | iterations={it0}")

    slip_profile = build_polyline_from_slices(slices)
    F, converged, iterations, debug = solver.calculate_fos_gps(
        slices=slices,
        slip_profile=slip_profile.tolist(),
        F_init=F0,
        tolerance=1e-6,
        max_iter=100,
        lambda_thrust=0.33,
        print_iteration_table=True,
        return_debug=True,
    )

    F1 = debug["F"][0] if debug and debug.get("F") else np.nan
    F2 = debug["F"][1] if debug and len(debug.get("F", [])) > 1 else np.nan

    print("-" * 90)
    print(f"F0 = {F0:.6f} (expected 1.385)")
    print(f"F1 = {F1:.6f} (expected 1.485)")
    print(f"F2 = {F2:.6f} (expected 1.480)")
    print(f"F  = {F:.6f} (expected 1.480)")
    print(f"converged={converged}, iterations={iterations}")
    print("=" * 90)


if __name__ == "__main__":
    main()

