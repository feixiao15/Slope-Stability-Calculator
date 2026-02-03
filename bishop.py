import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class BishopAnalyzer:
    def __init__(self, c_prime, phi_prime, gamma, r_u=0.0, iterations=5):
        self.c_prime = c_prime
        self.phi_prime = phi_prime
        self.phi_rad = np.radians(phi_prime)
        self.tan_phi = np.tan(self.phi_rad)
        self.gamma = gamma
        self.r_u = r_u
        self.iterations = iterations
        print(f"--- Analyzer initialized ---")
        print(f"c' = {c_prime} kPa, φ' = {phi_prime}°, γ = {gamma} kN/m³, r_u = {r_u}")

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
        fos = 1
        for i in range(self.iterations):
            numerator = 0.0  
            denominator = 0.0  
            for s in slice_data:
                W_i = s['W']
                alpha_rad = s['alpha_rad']
                b_i = s['b']

                denominator += W_i * np.sin(alpha_rad)

                numerator += (self.c_prime * b_i + self.tan_phi * (W_i * (1 - self.r_u))) * ((1/np.cos(alpha_rad))/(1+self.tan_phi * np.tan(alpha_rad)/fos))

            if denominator <= 0:
                return np.inf
            fos_new = numerator / denominator
            if abs(fos_new - fos) < 0.0001:
                break
            fos = fos_new
        return fos

    def _slice_mass(self, center, radius, n_slices, x_entry=0.0):
        """
        Returns: A list of slice properties or None (if arc is invalid)
        x_entry: 滑动弧在坡底侧的入口 x 坐标；默认 0 表示过坡脚 (slope toe)。
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

        # 入口点：可由调用方指定，不再强制过坡脚
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

    def find_critical_fos(self, n_slices, center_grid_x, center_grid_y, plot=True, entry_x_range=None):
        """
        Parameters:
        n_slices (int): Number of slices for each arc
        center_grid_x (np.array): Range of x-coordinates for the trial circle centers
        center_grid_y (np.array): Range of y-coordinates for the trial circle centers
        plot (bool): Whether to plot the result
        entry_x_range (array-like or None): 坡底入口点 x 的遍历序列；None 表示仅过坡脚 (0,0)。
            例如 np.arange(-toe_width, 0.5, interval) 表示在坡底宽度上按 interval 步长遍历。
        """
        # 未指定时保持“过坡脚”约束；指定后在该序列上遍历入口点，解除“过坡脚”约束
        if entry_x_range is None:
            entry_x_list = np.array([0.0])
            print(f"\n--- Starting search (arc through toe) ---")
        else:
            entry_x_list = np.atleast_1d(entry_x_range)
            print(f"\n--- Starting search (entry points along toe width: {len(entry_x_list)} points) ---")
        print(
            f"Search grid: {len(center_grid_x)} (x) * {len(center_grid_y)} (y) * {len(entry_x_list)} (entry_x)")

        min_fos = np.inf
        best_circle = None

        fos_results = []  # 每个 (xc, yc) 存该组合下所有 entry 中的最小 FoS，用于等高线

        for xc in center_grid_x:
            for yc in center_grid_y:
                best_fos_here = np.inf
                best_radius_here = None
                best_x_entry_here = None

                for x_entry in entry_x_list:
                    y_entry = self._get_y_on_surface(x_entry)
                    radius = np.sqrt((xc - x_entry) ** 2 + (yc - y_entry) ** 2)

                    slice_data = self._slice_mass((xc, yc), radius, n_slices, x_entry=x_entry)

                    if not slice_data:
                        continue

                    fos = self._calculate_fos(slice_data)
                    if fos < best_fos_here:
                        best_fos_here = fos
                        best_radius_here = radius
                        best_x_entry_here = x_entry

                fos_results.append((xc, yc, best_fos_here if best_radius_here is not None else np.nan))

                if best_radius_here is not None and best_fos_here < min_fos:
                    min_fos = best_fos_here
                    best_circle = {
                        'center': (xc, yc),
                        'radius': best_radius_here,
                        'fos': best_fos_here,
                        'x_entry': best_x_entry_here,
                    }

        print(f"--- Search complete ---")
        print(f"Minimum safety factor (FoS_min): {min_fos:.3f}")

        if best_circle is None:
            print("\n!!! Error: No valid slip surface was found in the defined grid.")
            print("    Try adjusting the search range for 'center_grid_x', 'center_grid_y', or 'entry_x_range'.")
        else:
            print(f"Most dangerous center O(x,y): ({best_circle['center'][0]:.2f}, {best_circle['center'][1]:.2f})")
            print(f"Most dangerous radius R: {best_circle['radius']:.2f}")
            if best_circle.get('x_entry') is not None and entry_x_range is not None:
                print(f"Entry point x_entry: {best_circle['x_entry']:.2f}")

            if plot:
                self.plot_result(best_circle, fos_results, center_grid_x, center_grid_y)

        return best_circle, fos_results

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
        ax.set_title("Bishop ")
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

if __name__ == "__main__":
    c_prime = 20.0
    phi_prime = 10.0
    gamma = 19.0
    r_u = 0.0
    height = 6
    ratio = 1  # 1V:mH
    toe_width = 10
    n_slices = 20
    center_grid_x = np.linspace(-10, 30, 100)
    center_grid_y = np.linspace(0, 30, 20)
    plot = True

    analyzer = BishopAnalyzer(c_prime, phi_prime, gamma, r_u)
    analyzer.define_slope(height, ratio, toe_width=toe_width)

    # 方式一：仅过坡脚 (0,0)，与原来一致
    # analyzer.find_critical_fos(n_slices, center_grid_x, center_grid_y, plot)

    # 方式二：去除“过坡脚”约束，对坡底宽度按 interval 遍历入口点
    interval = 2.0  # 坡底 x 步长 (m)
    entry_x_range = np.arange(-toe_width, 0.5, interval)
    best, results = analyzer.find_critical_fos(n_slices, center_grid_x, center_grid_y, plot, entry_x_range=entry_x_range)