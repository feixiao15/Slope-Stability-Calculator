import time
import numpy as np

from fellenius import FelleniusAnalyzer
from bishop import BishopAnalyzer
from janbu import GeometryBuilder, find_critical_fos_circular_arc


def run_benchmark():
    # Parameters reused from bishop_experiments.py
    c_prime = 3.0
    phi_prime = 19.6
    gamma = 20.0
    r_u = 0.0

    height = 10.0
    ratio = 2.0
    toe_width = 10.0
    crest_width = 20.0

    # Requested settings
    n_slices = 20
    n_iterations = 20

    # Same center search grid style as bishop_experiments.py
    center_grid_x = np.linspace(0.0, 30.0, 61)
    center_grid_y = np.linspace(0.0, 30.0, 61)

    timings_s = {}

    # 1) Fellenius
    t0 = time.perf_counter()
    fell = FelleniusAnalyzer(c_prime, phi_prime, gamma, r_u)
    fell.define_slope(height, ratio, toe_width=toe_width, crest_width=crest_width)
    fell.find_critical_fos(
        n_slices=n_slices,
        center_grid_x=center_grid_x,
        center_grid_y=center_grid_y,
        plot=False,
    )
    t1 = time.perf_counter()
    timings_s["Fellenius"] = (t1 - t0) 

    # 2) Bishop
    t0 = time.perf_counter()
    bish = BishopAnalyzer(c_prime, phi_prime, gamma, r_u, iterations=n_iterations)
    bish.define_slope(height, ratio, toe_width=toe_width, crest_width=crest_width)
    bish.find_critical_fos(
        n_slices=n_slices,
        center_grid_x=center_grid_x,
        center_grid_y=center_grid_y,
        plot=False,
    )
    t1 = time.perf_counter()
    timings_s["Bishop"] = (t1 - t0) 

    # 3) Janbu GPS
    t0 = time.perf_counter()
    gb = GeometryBuilder(
        slope_height=height,
        slope_ratio=ratio,
        bottom_extension=toe_width,
        top_extension=crest_width,
    )
    ground_profile, _ = gb.build()
    find_critical_fos_circular_arc(
        ground_profile=ground_profile,
        gamma=gamma,
        c_prime=c_prime,
        phi_prime=phi_prime,
        ru=r_u,
        n_slices=n_slices,
        center_grid_x=center_grid_x,
        center_grid_y=center_grid_y,
        entry_x_range=None,
        q=0.0,
        use_gps=True,
        gps_tolerance=1e-4,
        gps_max_iter=n_iterations,
        require_exit_at_crest=True,
    )
    t1 = time.perf_counter()
    timings_s["Janbu"] = (t1 - t0) 

    print("Runtime Benchmark (unit: s)")
    print(f"Fellenius: {timings_s['Fellenius']:.3f}")
    print(f"Bishop   : {timings_s['Bishop']:.3f}")
    print(f"Janbu    : {timings_s['Janbu']:.3f}")


if __name__ == "__main__":
    run_benchmark()
