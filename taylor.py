import numpy as np
import matplotlib.pyplot as plt


class TaylorStabilityChart:
    def __init__(self):
        """
        Initialize Taylor chart data.
        No longer using polynomial functions, but using high-precision data points manually extracted from the original chart for piecewise linear interpolation.
        """
        
        # ==========================================
        # Chart 1 Data: N vs Beta (different Phi)
        # ==========================================
        # Data format: { phi_value: (beta_points_array, n_values_array) }
        # Data points taken from key turning points and grid intersections of the original chart to ensure exact shape match
        self.chart1_data = {
            0: (
                np.array([53.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]),
                np.array([0.181, 0.184, 0.193, 0.205, 0.218, 0.229, 0.241, 0.252, 0.261])
            ),
            5: (
                np.array([8.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
                np.array([0, 0.045, 0.060, 0.082, 0.114, 0.137, 0.156, 0.173, 0.189, 0.204, 0.218])
            ),
            10: (
                np.array([10, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
                np.array([0, 0.045, 0.060, 0.088, 0.110, 0.129, 0.147, 0.163, 0.178, 0.192])
            ),
            15: (
                np.array([15,22.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
                np.array([0, 0.040, 0.050, 0.066, 0.090, 0.109, 0.125, 0.141, 0.155, 0.169])
            ),
            20: (
                np.array([20, 28.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
                np.array([0, 0.040, 0.048, 0.075, 0.094, 0.111, 0.126, 0.140, 0.152])
            ),
            25: (
                np.array([25, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
                np.array([0, 0.040, 0.055, 0.081, 0.098, 0.113, 0.126, 0.138])
            )
        }

        # ==========================================
        # Chart 2 Parameters: Phi=0, Beta < 53
        # ==========================================
        # Keep the original power function decay model, as the previous fitting was good
        # N = 0.181 - A * D^(-k)
        self.depth_coeffs = {
            53.0: [0.000, 1.0], # Limit
            45.0: [0.011, 2.5],
            30.0: [0.066, 2.7],
            22.5: [0.119, 2.5],
            15.0: [0.226, 2.5],
            7.5:  [0.324, 2.0]
        }

    def _get_N_from_chart1(self, phi, beta):
        """
        Calculate N value from Chart 1 using bilinear interpolation
        """
        # 1. Find two adjacent Phi curves (e.g., Phi=12 falls between 10 and 15)
        known_phis = sorted(self.chart1_data.keys()) # [0, 5, 10, 15, 20, 25]
        
        # Boundary handling
        if phi >= 25: return self._interp_beta(25, beta)
        if phi < 0: return self._interp_beta(0, beta) # Unlikely to happen

        # Find upper and lower bounds
        p_low = max([p for p in known_phis if p <= phi], default=0)
        p_high = min([p for p in known_phis if p >= phi], default=25)

        # 2. Interpolate in Beta direction
        N_low = self._interp_beta(p_low, beta)
        
        if p_low == p_high:
            return N_low
        
        N_high = self._interp_beta(p_high, beta)

        # 3. Interpolate in Phi direction
        # Check if N is invalid due to Beta being too small (e.g., Phi=25, Beta=20 has no solution)
        if np.isnan(N_low) or np.isnan(N_high):
            # If at the edge of curve definition, try to return valid value, or determine no solution
            if not np.isnan(N_low): return N_low
            if not np.isnan(N_high): return N_high
            return 0.0 # Theoretically, Beta too gentle for large Phi is absolutely stable

        ratio = (phi - p_low) / (p_high - p_low)
        return N_low + ratio * (N_high - N_low)

    def _interp_beta(self, phi_key, beta):
        """Helper function: linear interpolation of Beta on a single Phi curve"""
        beta_arr, n_arr = self.chart1_data[phi_key]
        
        # Special handling for Phi=0 horizontal segment
        if phi_key == 0 and beta < 53:
            return 0.181
        
        # If Beta is less than the starting Beta of this curve (e.g., Phi=25 when Beta=10)
        if beta < beta_arr[0]:
            return np.nan # or 0, indicating extremely stable
            
        return np.interp(beta, beta_arr, n_arr)

    def _calculate_depth_N(self, beta_key, D):
        """Chart 2 calculation formula"""
        if beta_key not in self.depth_coeffs: return 0.181
        if D < 1.0: D = 1.0
        A, k = self.depth_coeffs[beta_key]
        return max(0, 0.181 - A * (D ** -k))

    def get_stability_number(self, phi, beta, D=1.0):
        """
        Core logic: automatically select chart based on input conditions
        """
        # =============================================
        # Logic branch: Chart 2 (Phi=0 and Beta < 53)
        # =============================================
        if phi == 0 and beta < 53:
            # Chart 2 interpolation logic
            beta_keys = sorted(self.depth_coeffs.keys()) # [7.5, ..., 53]
            
            b_low = max([b for b in beta_keys if b <= beta], default=7.5)
            b_high = min([b for b in beta_keys if b >= beta], default=53)
            
            N_low = self._calculate_depth_N(b_low, D)
            N_high = self._calculate_depth_N(b_high, D)
            
            if b_low == b_high: return N_low
            
            ratio = (beta - b_low) / (b_high - b_low)
            return N_low + ratio * (N_high - N_low)

        # =============================================
        # Logic branch: Chart 1 (all other cases)
        # =============================================
        else:
            return self._get_N_from_chart1(phi, beta)

    def plot_verification(self):
        """
        Visualization verification: output fitting results of both charts
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # --- Verify Chart 1 ---
        betas = np.linspace(5, 90, 400) # Subdivide into 400 points to check smoothness
        phis = [0, 5, 10, 15, 20, 25]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        for i, phi in enumerate(phis):
            Ns = [self.get_stability_number(phi, b, D=10) for b in betas]
            ax1.plot(betas, Ns, label=f'φ = {phi}°', color=colors[i], linewidth=2)
            
        ax1.set_title("Chart 1 Verification: Piecewise Linear (Exact Shape)")
        ax1.set_xlabel("Slope Angle β (degrees)")
        ax1.set_ylabel("Stability Number N")
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()
        ax1.set_ylim(0, 0.27)
        ax1.set_xlim(0, 90)
        ax1.axvline(53, color='k', linestyle=':', alpha=0.3)

        # --- Verify Chart 2 ---
        Ds = np.linspace(1.0, 4.0, 100)
        target_betas = [53, 45, 30, 22.5, 15, 7.5]
        
        for beta in target_betas:
            # Force Chart 2 logic (phi=0, beta<53)
            # Note: when beta=53, code logic usually goes to Chart 1's 0.181, but here we slightly adjust beta to test Chart 2 formula
            test_beta = beta - 0.01 if beta == 53 else beta
            Ns = [self.get_stability_number(0, test_beta, D=d) for d in Ds]
            ax2.plot(Ds, Ns, label=f'β = {beta}°')

        ax2.set_title("Chart 2 Verification: Depth Factor (φ=0, β<53°)")
        ax2.set_xlabel("Depth Factor D")
        ax2.set_ylabel("Stability Number N")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.set_ylim(0.09, 0.19)

        plt.tight_layout()
        plt.show()


class TaylorSolver:
    """
    A unified Taylor stability analysis solver with interface style similar to `FelleniusAnalyzer`.

    - Internally contains a `TaylorStabilityChart`, responsible for finding stability number N based on φ, β, D;
    - Externally provides methods to calculate factor of safety Fs based on (c, γ, H, N) or (c, φ, γ, H, β, D).
    """

    def __init__(self):
        # Currently Taylor chart itself is independent of soil parameters, so c, φ, γ are not passed in __init__
        self.chart = TaylorStabilityChart()

    # --------- N calculation related to Taylor chart ---------
    def get_stability_number(self, phi: float, beta: float, D: float = 1.0) -> float:
        """
        Calculate stability number N from Taylor chart based on φ, β, D.
        """
        return self.chart.get_stability_number(phi, beta, D)

    # --------- Unified Fs calculation interface (internal tool) ---------
    def _calculate_fos_from_N(self, c: float, gamma: float, H: float, N: float) -> float:
        """
        Calculate factor of safety using formula Fs = c / (N * γ * H).

        Parameters:
          c     : Cohesion (kPa)
          gamma : Unit Weight (kN/m³)
          H     : Slope Height (m)
          N     : Taylor stability number obtained from lookup table or other methods
        """
        # 1. Boundary condition check
        if N <= 1e-5:
            return 999.0  # Return a large value representing "infinite stability"

        # 2. Physical parameter validity check
        if gamma <= 0 or H <= 0:
            print(f"Error: Gamma ({gamma}) and H ({H}) must be greater than 0")
            return None

        # 3. Core formula calculation
        Fs = c / (N * gamma * H)
        return Fs

    # --------- Main solving interface: only receives physical parameters, internally auto-selects chart and calculates N ---------
    def solve(self,
              c: float,
              phi: float,
              gamma: float,
              beta: float,
              H: float,
              D: float = 1.0) -> float:
        """
        Based on inputs c, φ, γ, β, H, D:
        1) Automatically determine whether to use Chart 1 or Chart 2 based on φ, β, D and calculate N;
        2) Then calculate factor of safety using Fs = c / (N * γ * H).
        """
        # 1. First get N from Taylor chart based on φ, β, D (internally automatically distinguishes Chart 1 / Chart 2)
        N = self.get_stability_number(phi, beta, D)

        # For debugging output only: indicate which chart is being used
        used_chart = "Chart 2 (Depth)" if (phi == 0 and beta < 53) else "Chart 1 (Standard)"
        print(f"   [TaylorSolver] Mode: {used_chart}")
        print(f"   [TaylorSolver] Inputs: Phi={phi}, Beta={beta}, D={D}")
        print(f"   [TaylorSolver] Stability Number N = {N:.4f}")

        # 2. Boundary check
        if N is None or np.isnan(N) or N <= 0:
            return 999.0  # Indicates extremely stable

        # 3. Calculate Fs based on N
        return self._calculate_fos_from_N(c, gamma, H, N)

    # Optional: keep old name, internally call solve (to avoid breaking existing external code)
    def calculate_fos_from_chart(self,
                                 c: float,
                                 phi: float,
                                 gamma: float,
                                 H: float,
                                 beta: float,
                                 D: float = 1.0) -> float:
        """Compatible with old interface, equivalent to solve."""
        return self.solve(c, phi, gamma, beta, H, D)

    # --------- Optional: wrap original verification plotting ---------
    def plot_verification(self):
        """Directly reuse internal chart's verification plotting."""
        self.chart.plot_verification()


# For backward compatibility, keep original function interface, but internally all call new TaylorSolver
def calculate_fos_taylor(c, phi, gamma, H, beta, D=1.0, analyzer=None):
    solver = TaylorSolver()
    # If user passes custom chart, replace internal chart
    if analyzer is not None:
        solver.chart = analyzer
    # Here strictly follows the required parameter set: c, phi, gamma, beta, H, D
    return solver.solve(c, phi, gamma, beta, H, D)


def calculate_fos_from_formula(c, gamma, H, N):
    solver = TaylorSolver()
    return solver._calculate_fos_from_N(c, gamma, H, N)


# --- Main program execution (example) ---
if __name__ == "__main__":
    solver = TaylorSolver()

    # 1. First output chart for inspection
    print("Generating verification chart...")
    solver.plot_verification()

    # --- Input known conditions ---
    c = 20.0       # kPa
    gamma = 18.0   # kN/m³
    H = 5.0        # m
    beta = 30.0    # degrees
    phi = 20        # degrees
    D = 1.0        # dimensionless

    # First calculate N based on chart
    N = solver.get_stability_number(phi, beta, D)

    # --- Call class method to calculate Fs ---
    final_fos = solver.solve(c, phi, gamma, beta, H, D)

    print(f"Known parameters:")
    print(f"  c (Cohesion) = {c} kPa")
    print(f"  γ (Unit Weight)   = {gamma} kN/m³")
    print(f"  H (Slope Height)   = {H} m")
    print(f"  β (Slope Angle)   = {beta} degrees")
    print(f"  φ (Friction Angle) = {phi} degrees")
    print(f"  D (Depth Factor) = {D}")
    print(f"  N (Stability Number) = {N:.4f}")
    print("-" * 30)
    print(f"Calculated result Fs = {final_fos:.3f}")
