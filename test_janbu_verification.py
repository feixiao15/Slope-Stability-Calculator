import numpy as np


class JanbuReproductionSolver:
    def calculate_fos_gps_reproduction(self, slices, geometry_override, F_init=1.4, max_iter=3):
        n = len(slices)
        dx = np.array([s["dx"] for s in slices], dtype=float)
        tan_alpha = np.array([s["tan_alpha"] for s in slices], dtype=float)
        alpha = np.arctan(tan_alpha)
        cos_alpha = np.cos(alpha)
        p = np.array([s["p"] for s in slices], dtype=float)
        u = np.array([s["u"] for s in slices], dtype=float)
        c = np.array([s["c"] for s in slices], dtype=float)
        tan_phi = np.array([s["tan_phi"] for s in slices], dtype=float)
        dQ = np.zeros(n, dtype=float)
        h_t_interface = np.asarray(geometry_override["ht"], dtype=float)
        tan_alpha_t = np.asarray(geometry_override["tan_alpha_t"], dtype=float)
        F_old = float(F_init)
        t = np.zeros(n, dtype=float)

        print("=" * 80)
        print(f"Start Janbu GPS reproduction (F_init={F_init:.3f})")
        print("=" * 80)

        for it in range(max_iter):
            print(f"\n>>> Iteration step {it} (solve F_{it})")
            n_alpha = (cos_alpha ** 2) * (1.0 + (tan_alpha * tan_phi) / F_old)
            sigma_term = p + t - u
            A_prime = (c + sigma_term * tan_phi) * dx
            A = A_prime / n_alpha
            B = dQ + (p + t) * dx * tan_alpha
            delta_E = B - A / F_old
            E_interface = np.zeros(n + 1, dtype=float)
            E_interface[1:] = np.cumsum(delta_E)
            denom = np.sum(B) - E_interface[-1]
            F_new = float(np.sum(A) / denom)
            dE_dx = np.zeros(n + 1, dtype=float)

            for i in range(1, n):
                dE_dx[i] = (delta_E[i - 1] + delta_E[i]) / (dx[i - 1] + dx[i])

            T_interface = -E_interface * tan_alpha_t + h_t_interface * dE_dx
            T_interface[0] = 0.0
            T_interface[-1] = 0.0
            delta_T = np.diff(T_interface)
            t_new = delta_T / dx
            self._print_table_row(
                it,
                B,
                A_prime,
                n_alpha,
                A,
                delta_E,
                E_interface,
                F_old,
                F_new,
                dE_dx,
                tan_alpha_t,
                h_t_interface,
                T_interface,
                t_new,
            )
            F_old = F_new
            t = t_new

        return F_old

    def _print_table_row(self, it, B, A_prime, n_alpha, A, dE, E, F_in, F_out, dE_dx, tan_at, ht, T, t):
        print(f"\n--- Iteration {it} check ---")
        print(f"Input F = {F_in:.3f} -> Calculated F = {F_out:.3f}")
        print(
            f"{'Slice':<6} {'B':<8} {'A_prime':<10} {'n_a':<8} {'A':<8} {'dE':<8} {'E_int':<8} | "
            f"{'dE/dx':<8} {'tan_at':<8} {'ht':<6} {'T_int':<8} {'t_slice':<8}"
        )
        print("-" * 120)

        for i in range(B.size):
            idx_int = i + 1
            print(
                f"{i+1:<6} {B[i]:<8.1f} {A_prime[i]:<10.1f} {n_alpha[i]:<8.2f} {A[i]:<8.1f} {dE[i]:<8.1f} {E[idx_int]:<8.1f} | "
                f"{dE_dx[idx_int]:<8.2f} {tan_at[idx_int]:<8.2f} {ht[idx_int]:<6.1f} {T[idx_int]:<8.2f} {t[i]:<8.2f}"
            )

        print(f"{'Sum':<6} {np.sum(B):<8.1f} {'-':<10} {'-':<8} {np.sum(A):<8.1f} {np.sum(dE):<8.1f}")


slices_data = [
    {"id": 1, "tan_alpha": 1.13, "dx": 4.4, "p": 5.3, "u": 2.12, "c": 1.0, "tan_phi": 0.67},
    {"id": 2, "tan_alpha": 0.50, "dx": 11.0, "p": 10.1, "u": 4.04, "c": 1.0, "tan_phi": 0.67},
    {"id": 3, "tan_alpha": 0.18, "dx": 11.0, "p": 8.6, "u": 3.44, "c": 1.0, "tan_phi": 0.67},
    {"id": 4, "tan_alpha": -0.04, "dx": 6.0, "p": 2.9, "u": 1.16, "c": 1.0, "tan_phi": 0.67},
]

ht_input = np.array([0.0, 1.2, 1.8, 1.1, 0.0], dtype=float)
tan_at_input = np.array([0.0, 0.63, 0.33, 0.16, 0.0], dtype=float)
geo_override = {"ht": ht_input, "tan_alpha_t": tan_at_input}


if __name__ == "__main__":
    solver = JanbuReproductionSolver()
    solver.calculate_fos_gps_reproduction(slices_data, geo_override, F_init=1.4, max_iter=3)
