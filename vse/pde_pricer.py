"""
Thin Python-side wrapper around the compiled C++ `cn_pricer` module: build
the spatial/time grids and the local-vol grid the C++ side expects, call
across the pybind11 boundary once, and interpolate the returned price curve
to the actual spot S0.
"""
import sys
import os
import numpy as np

_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

import cn_pricer  # noqa: E402  (compiled pybind11 module, built via cpp/CMakeLists.txt)

from vse.dupire import build_local_vol_grid


def price_down_and_out(surface, S0, K, barrier, r, q, T,
                        is_call=True, N_S=400, N_t=400, S_max_mult=3.0):
    """
    S_max_mult sets how far above spot the upper grid boundary goes, as a
    multiple of S0 -- needs to be comfortably past where the deep-ITM linear
    boundary condition is a good approximation, several sigma*sqrt(T) above
    spot; too tight and the boundary condition contaminates the price near
    S0, too loose just wastes grid points.
    """
    if min(S0, K, barrier, T) <= 0 or N_S < 3 or N_t < 1 or S_max_mult <= 1:
        raise ValueError("Invalid prices, maturity, or PDE grid")
    if S0 <= barrier:
        return 0.0, np.array([barrier]), np.array([0.0])
    S_max = S0 * S_max_mult
    S_grid = np.linspace(barrier, S_max, N_S)
    t_grid = np.linspace(0.0, T, N_t + 1)

    local_vol = build_local_vol_grid(surface, S_grid, t_grid)

    V = cn_pricer.price_barrier_cn(S_grid, local_vol, r, q, K, T, is_call)
    price = float(np.interp(S0, S_grid, V))
    return price, S_grid, V
