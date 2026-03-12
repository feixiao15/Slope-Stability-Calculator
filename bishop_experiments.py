import numpy as np
import matplotlib.pyplot as plt

from bishop import BishopAnalyzer


def evaluate_fos(
    c_prime: float,
    phi_prime: float,
    gamma: float,
    r_u: float,
    height: float,
    ratio: float,
    toe_width: float,
    crest_width: float,
    n_slices: int,
    center_grid_x: np.ndarray,
    center_grid_y: np.ndarray,
) -> float:
    analyzer = BishopAnalyzer(c_prime, phi_prime, gamma, r_u)
    analyzer.define_slope(height, ratio, toe_width=toe_width, crest_width=crest_width)
    best_circle, _ = analyzer.find_critical_fos(
        n_slices=n_slices,
        center_grid_x=center_grid_x,
        center_grid_y=center_grid_y,
        plot=False,
    )
    if best_circle is None:
        return np.nan
    return float(best_circle["fos"])


def run_experiments():
    # Parameters from the provided UI screenshot
    c_base = 3.0
    phi_base = 19.6
    gamma = 20.0
    r_u = 0.0

    height = 10.0
    ratio = 2.0
    toe_width = 10.0
    crest_width = 20.0

    n_slices = 10
    center_grid_x = np.linspace(0.0, 30.0, 61)
    center_grid_y = np.linspace(0.0, 30.0, 61)

    # Experiment 1: FoS vs Cohesion (0-20 kPa)
    cohesion_values = np.linspace(0.0, 20.0, 41)
    fos_vs_cohesion = []
    for c_val in cohesion_values:
        fos = evaluate_fos(
            c_prime=float(c_val),
            phi_prime=phi_base,
            gamma=gamma,
            r_u=r_u,
            height=height,
            ratio=ratio,
            toe_width=toe_width,
            crest_width=crest_width,
            n_slices=n_slices,
            center_grid_x=center_grid_x,
            center_grid_y=center_grid_y,
        )
        fos_vs_cohesion.append(fos)

    # Experiment 2: FoS vs Friction Angle (0-50 deg)
    friction_values = np.linspace(0.0, 50.0, 51)
    fos_vs_friction = []
    for phi_val in friction_values:
        fos = evaluate_fos(
            c_prime=c_base,
            phi_prime=float(phi_val),
            gamma=gamma,
            r_u=r_u,
            height=height,
            ratio=ratio,
            toe_width=toe_width,
            crest_width=crest_width,
            n_slices=n_slices,
            center_grid_x=center_grid_x,
            center_grid_y=center_grid_y,
        )
        fos_vs_friction.append(fos)

    # Plot and save charts
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(cohesion_values, fos_vs_cohesion, marker="o", linewidth=2)
    ax1.set_xlabel("Cohesion c' (kPa)")
    ax1.set_ylabel("Factor of Safety (FoS)")
    ax1.set_title("Experiment 1: FoS vs Cohesion")
    ax1.grid(True, linestyle=":", alpha=0.7)
    fig1.tight_layout()
    fig1.savefig("experiment1_fos_vs_cohesion.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(friction_values, fos_vs_friction, marker="s", linewidth=2, color="tab:orange")
    ax2.set_xlabel("Friction Angle phi' (deg)")
    ax2.set_ylabel("Factor of Safety (FoS)")
    ax2.set_title("Experiment 2: FoS vs Friction Angle")
    ax2.grid(True, linestyle=":", alpha=0.7)
    fig2.tight_layout()
    fig2.savefig("experiment2_fos_vs_friction_angle.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    run_experiments()
