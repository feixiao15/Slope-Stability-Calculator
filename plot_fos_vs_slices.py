import numpy as np
import matplotlib.pyplot as plt

from fellenius import FelleniusAnalyzer


def main():
    plt.rcParams["font.family"] = "Times New Roman"

    # Soil parameters
    c = 20.0
    soil_type = 1  # 1 is drained, 0 is undrained
    phi_prime = 15.0 if soil_type else 0.0
    gamma = 19.0
    r_u = 0.4

    # Geometric parameters
    slope_height = 6.0
    slope_ratio = 1.5  # V : mH

    # Search grid
    grid_x = np.linspace(0, 30, 20)
    grid_y = np.linspace(0, 30, 20)

    analyzer = FelleniusAnalyzer(c, phi_prime, gamma, r_u, soil_type)
    analyzer.define_slope(slope_height, slope_ratio)

    slice_options = [5, 10, 15, 20]
    fos_values = []

    for n_slices in slice_options:
        best_circle = analyzer.find_critical_fos(
            n_slices=n_slices,
            center_grid_x=grid_x,
            center_grid_y=grid_y,
            plot=False,
        )

        fos = best_circle["fos"] if best_circle is not None else np.nan
        fos_values.append(fos)

    fig, ax = plt.subplots()
    ax.plot(slice_options, fos_values, color="red", marker="o")
    ax.set_xlabel("i slices (n)")
    ax.set_ylabel("Factor of Safety")
    ax.set_title("FoS vs Number of Slices")
    ax.grid(True, linestyle=":")

    plt.show()


if __name__ == "__main__":
    main()

