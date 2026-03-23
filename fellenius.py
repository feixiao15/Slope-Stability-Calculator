import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class FelleniusAnalyzer:
    """Fellenius (ordinary method of slices) slope stability analyzer."""

    def __init__(self, c_prime, phi_prime, gamma, r_u=0.0, type=1):
        """Initialize soil strength and pore pressure parameters."""
        self.c_prime = c_prime
        self.phi_prime = phi_prime
        self.phi_rad = np.radians(phi_prime)
        self.tan_phi = np.tan(self.phi_rad)
        self.gamma = gamma
        self.r_u = r_u
        self.type = type
        print(f"--- Analyzer initialized ---")
        print(f"c' = {c_prime} kPa, φ' = {phi_prime}°, γ = {gamma} kN/m³, r_u = {r_u}, soil type = {type}")

    def define_slope(self, height, ratio, toe_width=10, crest_width=20):
        """Define slope geometry with toe and crest platforms."""
        self.height = height
        self.ratio = ratio
        self.crest_x = height * ratio

        self.surface_points = [
            (-toe_width, 0),
            (0, 0),
            (self.crest_x, self.height),
            (self.crest_x + crest_width, self.height)
        ]
        self.surface_poly = np.array(self.surface_points)
        print(f"Slope Geometry: H = {height}m, Slope Ratio = 1:{ratio}")
        print(f"Toe: (0, 0), Crest: ({self.crest_x}, {self.height})")

    def _get_y_on_surface(self, x):
        """Return surface elevation by linear interpolation."""
        xp = self.surface_poly[:, 0]
        fp = self.surface_poly[:, 1]
        return np.interp(x, xp, fp)

    def _calculate_fos(self, slice_data, return_details=False):
        """Calculate FoS from slice forces; optionally return term details."""
        numerator = 0.0
        denominator = 0.0

        slice_terms = []
        for idx, s in enumerate(slice_data, start=1):
            W_i = s['W']
            alpha_rad = s['alpha_rad']
            l_i = s['l']

            sliding_i = W_i * np.sin(alpha_rad)
            denominator += sliding_i

            cohesion_resistance = self.c_prime * l_i

            ul_i = self.r_u * W_i / np.cos(alpha_rad)

            # Effective normal force on the base.
            N_prime_i = (W_i * np.cos(alpha_rad)) - ul_i

            if N_prime_i < 0:
                N_prime_i = 0.0

            friction_resistance = N_prime_i * self.tan_phi

            resisting_i = cohesion_resistance + friction_resistance
            numerator += resisting_i

            if return_details:
                slice_terms.append({
                    "slice": idx,
                    "W": float(W_i),
                    "alpha_rad": float(alpha_rad),
                    "alpha_deg": float(np.degrees(alpha_rad)),
                    "l": float(l_i),
                    "u_l": float(ul_i),
                    "N_prime": float(N_prime_i),
                    "cohesion_resistance": float(cohesion_resistance),
                    "friction_resistance": float(friction_resistance),
                    "resisting": float(resisting_i),
                    "sliding": float(sliding_i),
                    "x_mid": float(s.get("x_mid", np.nan)),
                    "h_mid": float(s.get("h_mid", np.nan)),
                })

        if denominator <= 0:
            if return_details:
                return np.inf, {
                    "numerator": float(numerator),
                    "denominator": float(denominator),
                    "slice_terms": slice_terms,
                    "formula": "FoS = Σ[c'l + N' tanφ'] / Σ[W sinα], N' = W cosα - u l",
                }
            return np.inf

        fos = numerator / denominator
        if return_details:
            return fos, {
                "numerator": float(numerator),
                "denominator": float(denominator),
                "slice_terms": slice_terms,
                "formula": "FoS = Σ[c'l + N' tanφ'] / Σ[W sinα], N' = W cosα - u l",
            }
        return fos

    def calculate_circle_details(self, center, n_slices):
        """Compute FoS and slice details for one toe-passing circle."""
        xc, yc = map(float, center)
        radius = float(np.sqrt(xc ** 2 + yc ** 2))
        slice_data = self._slice_mass((xc, yc), radius, n_slices)
        if not slice_data:
            return None
        fos, detail = self._calculate_fos(slice_data, return_details=True)
        return {
            "center": (xc, yc),
            "radius": radius,
            "fos": float(fos),
            "n_slices": len(slice_data),
            "detail": detail,
        }

    def _slice_mass(self, center, radius, n_slices):
        """Build slices for one circle; return None for invalid geometry."""
        xc, yc = center

        H = self.height
        term_sq = radius ** 2 - (H - yc) ** 2

        if term_sq < 0:
            return None

        x_exit = xc + np.sqrt(term_sq)

        if x_exit < self.crest_x:
            return None

        x_entry = 0.0

        total_width = x_exit - x_entry
        if total_width <= 0:
            return None

        b_width = total_width / n_slices

        slices_data = []

        for i in range(n_slices):
            x_left = x_entry + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            y_base_sq_term = radius ** 2 - (x_mid - xc) ** 2
            if y_base_sq_term < 0:
                continue

            y_base = yc - np.sqrt(y_base_sq_term)


            # Tangent slope from circle derivative.
            m_tan = -(x_mid - xc) / (y_base - yc)
            alpha_rad = np.arctan(m_tan)

            y_top = self._get_y_on_surface(x_mid)
            h_mid = y_top - y_base

            if h_mid < 0:
                continue

            W = h_mid * b_width * self.gamma

            l = b_width / np.cos(alpha_rad)

            slices_data.append({
                'W': W,
                'alpha_rad': alpha_rad,
                'l': l,
                'b': b_width,
                'x_mid': x_mid,
                'h_mid': h_mid
            })

        if not slices_data:
            return None

        return slices_data

    def find_critical_fos(self, n_slices, center_grid_x, center_grid_y, plot=True, center=None):
        """Run single-center or grid search and return best circle with FoS map."""
        if center is not None:
            xc, yc = map(float, center)
            radius = float(np.sqrt(xc ** 2 + yc ** 2))
            slice_data = self._slice_mass((xc, yc), radius, n_slices)
            if not slice_data:
                best_circle = None
                fos_val = np.nan
            else:
                fos_val = float(self._calculate_fos(slice_data))
                best_circle = {"center": (xc, yc), "radius": radius, "fos": fos_val}

            fos_results = [(xc, yc, fos_val)]

            print(f"\n--- Fellenius single-center evaluation ---")
            print(f"Center O(x,y) = ({xc:.3f}, {yc:.3f}), R = {radius:.3f}, FoS = {fos_val:.6f}")

            if plot and best_circle is not None:
                self.plot_result(best_circle, fos_results, np.array([xc]), np.array([yc]))

            return best_circle, fos_results

        print(f"\n--- Starting search (Fellenius)... ---")
        print(
            f"Search grid: {len(center_grid_x)} (x) * {len(center_grid_y)} (y) = {len(center_grid_x) * len(center_grid_y)} circle centers")

        min_fos = np.inf
        best_circle = None

        fos_results = []

        for xc in center_grid_x:
            for yc in center_grid_y:
                radius = np.sqrt(xc ** 2 + yc ** 2)
                slice_data = self._slice_mass((xc, yc), radius, n_slices)

                if not slice_data:
                    fos_results.append((xc, yc, np.nan))
                    continue

                fos = self._calculate_fos(slice_data)
                fos_results.append((xc, yc, fos))

                if fos < min_fos:
                    min_fos = fos
                    best_circle = {'center': (xc, yc), 'radius': radius, 'fos': fos}

        print(f"--- Search complete ---")
        print(f"Minimum safety factor (FoS_min): {min_fos:.3f}")

        if best_circle is None:
            print("\n!!! Error: No valid slip surface was found in the defined grid.")
            print("    Try adjusting the search range for 'center_grid_x' and 'center_grid_y'.")
        else:
            print(f"Most dangerous center O(x,y): ({best_circle['center'][0]:.2f}, {best_circle['center'][1]:.2f})")
            print(f"Most dangerous radius R: {best_circle['radius']:.2f}")

            if plot:
                self.plot_result(best_circle, fos_results, center_grid_x, center_grid_y)

        return best_circle, fos_results

    def plot_result(self, best_circle, fos_results, grid_x, grid_y):
        """Plot FoS contour, slope geometry, and critical circle."""

        fig, ax = plt.subplots(figsize=(14, 8))

        Z = np.array([r[2] for r in fos_results]).reshape(len(grid_x), len(grid_y)).T
        contours = ax.contourf(grid_x, grid_y, Z, levels=20, cmap="viridis_r", alpha=0.7)
        fig.colorbar(contours, ax=ax, label="Factor of Safety (FoS)")

        ax.plot(self.surface_poly[:, 0], self.surface_poly[:, 1], 'k-', linewidth=3, label="surface")

        center = best_circle['center']
        radius = best_circle['radius']
        fos = best_circle['fos']

        slip_circle = Circle(center, radius, fill=False, edgecolor='red',
                             linewidth=2, linestyle='--', label=f"Critical Circle (FoS={fos:.3f})")
        ax.add_patch(slip_circle)

        ax.plot(center[0], center[1], 'r+', markersize=15, label="most dangerous circle center")

        ax.set_title("Fellenius ")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal')

        ax.set_xlim(self.surface_poly[0][0], self.surface_poly[-1][0])
        ax.set_ylim(self.surface_poly[0][1] - self.height * 0.5,
                    np.max(grid_y) + self.height * 0.5)

        plt.show()

if __name__ == "__main__":
    # Example CLI run.
    c = 20.0
    soil_type = 1  # 1 is drained and 0 is undrained
    phi_prime = 10 if soil_type else 0
    gamma = 19.0
    r_u = 0

    slope_height = 6.0
    slope_ratio = np.tan(np.radians(90 - 70))  # V : mH

    num_slices = 20

    grid_x = np.linspace(0, 30, 20)
    grid_y = np.linspace(0, 30, 20)

    analyzer = FelleniusAnalyzer(c, phi_prime, gamma, r_u)
    analyzer.define_slope(slope_height, slope_ratio)
    critical_circle, fos_results = analyzer.find_critical_fos(
        n_slices=num_slices,
        center_grid_x=grid_x,
        center_grid_y=grid_y,
        plot=True,
    )
