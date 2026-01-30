import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class FelleniusAnalyzer:
    """
    A general Fellenius (conventional slice method) slope stability analysis tool.
    """

    def __init__(self, c_prime, phi_prime, gamma, r_u=0.0, type = 1):
        """
        Initialize soil parameters

        Parameters:
        c_prime (float): Effective cohesion (kPa or kN/m^2)
        phi_prime (float): Effective internal friction angle (degrees)
        gamma (float): Unit weight of soil (kN/m^3)
        r_u (float): Pore water pressure coefficient
        """
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
        """
        Parameters:
        height (float): Slope height H
        ratio (float): Slope ratio m (1V:mH)
        toe_width (float): Length of flat ground at the toe of the slope
        crest_width (float): Length of flat ground at the crest of the slope
        """
        self.height = height
        self.ratio = ratio
        self.crest_x = height * ratio

        # Define a list of (x, y) coordinates to represent the surface
        # The (0,0) point is set as the toe
        self.surface_points = [
            (-toe_width, 0),
            (0, 0),  # Toe
            (self.crest_x, self.height),  # Crest
            (self.crest_x + crest_width, self.height)
        ]
        self.surface_poly = np.array(self.surface_points)
        print(f"Slope Geometry: H = {height}m, Slope Ratio = 1:{ratio}")
        print(f"Toe: (0, 0), Crest: ({self.crest_x}, {self.height})")

    def _get_y_on_surface(self, x):
        """(Helper) Get the surface y-coordinate based on x"""
        # Use np.interp for linear interpolation
        xp = self.surface_poly[:, 0]
        fp = self.surface_poly[:, 1]
        return np.interp(x, xp, fp)

    def _calculate_fos(self, slice_data):
        """
        Parameters:
        slice_data (list of dicts): Soil slice data calculated by _slice_mass
        """
        numerator = 0.0  # Total resisting force
        denominator = 0.0  # Total sliding force

        for s in slice_data:
            W_i = s['W']
            alpha_rad = s['alpha_rad']
            l_i = s['l']

            # 1. Sliding force (denominator)
            denominator += W_i * np.sin(alpha_rad)

            # 2. Resisting force (numerator)
            cohesion_resistance = self.c_prime * l_i

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
        Returns: A list of slice properties or None (if arc is invalid)
        """
        xc, yc = center

        # 1. Find the "exit point" (x_exit) of the arc
        H = self.height
        term_sq = radius ** 2 - (H - yc) ** 2

        if term_sq < 0:
            return None  # Arc does not reach the top of the slope

        x_exit = xc + np.sqrt(term_sq)

        if x_exit < self.crest_x:
            return None  # Arc exits at the slope surface

        x_entry = 0.0  # Force slope toe failure

        total_width = x_exit - x_entry
        if total_width <= 0:
            return None

        b_width = total_width / n_slices  # Width of each soil slice

        slices_data = []

        for i in range(n_slices):
            x_left = x_entry + i * b_width
            x_right = x_left + b_width
            x_mid = x_left + 0.5 * b_width

            y_base_sq_term = radius ** 2 - (x_mid - xc) ** 2
            if y_base_sq_term < 0:
                continue

            y_base = yc - np.sqrt(y_base_sq_term)


            # (y_base - yc) is negative
            m_tan = -(x_mid - xc) / (y_base - yc)
            alpha_rad = np.arctan(m_tan)

            y_top = self._get_y_on_surface(x_mid)
            h_mid = y_top - y_base

            if h_mid < 0:
                continue

            W = h_mid * b_width * self.gamma  # Weight (kN/m)

            l = b_width / np.cos(alpha_rad)

            slices_data.append({
                'W': W,
                'alpha_rad': alpha_rad,
                'l': l,
                'b': b_width,
                'x_mid': x_mid,
                'h_mid': h_mid
            })

        # Ensure we really have slices (sometimes the arc may be invalid)
        if not slices_data:
            return None

        return slices_data

    def find_critical_fos(self, n_slices, center_grid_x, center_grid_y, plot=True):
        """
        Parameters:
        n_slices (int): Number of slices for each arc
        center_grid_x (np.array): Range of x-coordinates for the trial circle centers
        center_grid_y (np.array): Range of y-coordinates for the trial circle centers
        """
        print(f"\n--- Starting search (Fellenius)... ---")
        print(
            f"Search grid: {len(center_grid_x)} (x) * {len(center_grid_y)} (y) = {len(center_grid_x) * len(center_grid_y)} circle centers")

        min_fos = np.inf
        best_circle = None

        fos_results = []

        for xc in center_grid_x:
            for yc in center_grid_y:
                # (Step 2) Force the arc to pass through the toe (0,0)
                radius = np.sqrt(xc ** 2 + yc ** 2)

                # (Steps 3 & 4) Automatically slice
                slice_data = self._slice_mass((xc, yc), radius, n_slices)

                if not slice_data:
                    fos_results.append((xc, yc, np.nan))
                    continue

                # (Step 5) Calculate FoS
                fos = self._calculate_fos(slice_data)
                fos_results.append((xc, yc, fos))

                # (Step 6) Optimization
                if fos < min_fos:
                    min_fos = fos
                    best_circle = {'center': (xc, yc), 'radius': radius, 'fos': fos}

        print(f"--- Search complete ---")
        print(f"Minimum safety factor (FoS_min): {min_fos:.3f}")

        if best_circle is None:
            print("\n!!! Error: No valid slip surface was found in the defined grid.")
            print("    Try adjusting the search range for 'center_grid_x' and 'center_grid_y'.")
        else:
            # Only print detailed information and plot if a circle is found
            print(f"Most dangerous center O(x,y): ({best_circle['center'][0]:.2f}, {best_circle['center'][1]:.2f})")
            print(f"Most dangerous radius R: {best_circle['radius']:.2f}")

            if plot:
                self.plot_result(best_circle, fos_results, center_grid_x, center_grid_y)

        return best_circle

    def plot_result(self, best_circle, fos_results, grid_x, grid_y):
        """(Helper) Visualize the search results"""

        fig, ax = plt.subplots(figsize=(14, 8))

        # 1. Draw contour plot (Heatmap)
        Z = np.array([r[2] for r in fos_results]).reshape(len(grid_x), len(grid_y)).T
        contours = ax.contourf(grid_x, grid_y, Z, levels=20, cmap="viridis_r", alpha=0.7)
        fig.colorbar(contours, ax=ax, label="Factor of Safety (FoS)")

        # 2. Draw slope geometry
        ax.plot(self.surface_poly[:, 0], self.surface_poly[:, 1], 'k-', linewidth=3, label="surface")

        # 3. Draw the most dangerous slip surface
        center = best_circle['center']
        radius = best_circle['radius']
        fos = best_circle['fos']

        slip_circle = Circle(center, radius, fill=False, edgecolor='red',
                             linewidth=2, linestyle='--', label=f"Critical Circle (FoS={fos:.3f})")
        ax.add_patch(slip_circle)

        # 4. Draw the circle center
        ax.plot(center[0], center[1], 'r+', markersize=15, label="most dangerous circle center")

        # 5. Set up the plot
        ax.set_title("Fellenius ")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal')

        # Set reasonable display limits
        ax.set_xlim(self.surface_poly[0][0], self.surface_poly[-1][0])
        ax.set_ylim(self.surface_poly[0][1] - self.height * 0.5,
                    np.max(grid_y) + self.height * 0.5)

        plt.show()

# --- 1. Set analysis parameters ---

# Soil parameters
c = 20.0
soil_type = 1  # 1 is drained and 0 is undrained
phi_prime = 10 if soil_type else 0
gamma = 19.0
r_u = 0

# Geometric parameters
slope_height = 6.0
slope_ratio = np.tan(np.radians(90-70)) # V : mH

# Analysis parameters
num_slices = 20 # Number of slices as per the example problem

# (Step 2) Define search grid
grid_x = np.linspace(0, 30, 20)
grid_y = np.linspace(0, 30, 20)


# --- 2. Run analysis ---

# Initialize analyzer
analyzer = FelleniusAnalyzer(c, phi_prime, gamma, r_u)

# Define slope
analyzer.define_slope(slope_height, slope_ratio)

# Find the most dangerous slip surface
# This will automatically complete Steps 2, 3, 4, 5, and 6
critical_circle = analyzer.find_critical_fos(
    n_slices=num_slices,
    center_grid_x=grid_x,
    center_grid_y=grid_y,
    plot=True
)
