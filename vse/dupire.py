"""Local variance using k=log(K/F(T)) and w=implied_vol**2*T.

The maturity derivative holds k fixed, as does the interpolation below.
See Gatheral-Jacquier, arXiv:1204.0646, section 2. Slice checks do not prove
all interpolated smiles are valid; invalid local variance raises an error.
"""
import numpy as np
from vse.svi import _svi_derivatives


class CalibratedSurface:
    def __init__(self, slices):
        self.slices = sorted(slices, key=lambda s: s["T"])
        self.T_grid = np.array([s["T"] for s in self.slices])
        if len(slices) < 2 or self.T_grid[0] <= 0 or np.any(np.diff(self.T_grid) <= 0):
            raise ValueError("Need at least two distinct positive maturities")

    def forward(self, T):
        """Log-linear interpolation/extrapolation preserves constant carry.

        log(F)=log(S0)+(r-q)*T, so this recovers spot at zero in our examples.
        """
        i = np.clip(np.searchsorted(self.T_grid, T, side="right"), 1, len(self.slices)-1)
        lo, hi = self.slices[i-1], self.slices[i]
        weight = (T-lo["T"])/(hi["T"]-lo["T"])
        return np.exp((1-weight)*np.log(lo["F"]) + weight*np.log(hi["F"]))

    def variance_terms(self, k, T):
        """Return w, w_k, w_kk, w_T analytically within each time interval."""
        if T <= 0:
            raise ValueError("Maturity must be positive")
        k = np.asarray(k, dtype=float)
        if T < self.T_grid[0] or T >= self.T_grid[-1]:
            # Constant implied-vol extrapolation: w grows with time.
            # Freezing w would instead make its time derivative zero.
            edge = self.slices[0] if T < self.T_grid[0] else self.slices[-1]
            w, wk, wkk = _svi_derivatives(k, edge["params"])
            scale = T / edge["T"]
            return scale*w, scale*wk, scale*wkk, w/edge["T"]
        i = np.searchsorted(self.T_grid, T, side="right")
        lo, hi = self.slices[i-1], self.slices[i]
        gap = hi["T"] - lo["T"]
        weight = (T-lo["T"])/gap
        low = _svi_derivatives(k, lo["params"])
        high = _svi_derivatives(k, hi["params"])
        terms = [(1-weight)*a + weight*b for a, b in zip(low, high)]
        # At a quoted maturity take the right-hand interval derivative.
        return *terms, (high[0]-low[0])/gap

    def total_variance(self, K, T):
        return self.variance_terms(np.log(np.asarray(K)/self.forward(T)), T)[0]

    def implied_vol(self, K, T):
        return np.sqrt(self.total_variance(K, T)/T)


def local_variance(surface, K, T, dT=1e-3, dk=1e-3):
    """Dupire's formula; dT/dk retained for old callers, no differences needed.

    Flat implied vol: w=sigma²*T, w_k=w_kk=0. The denominator is one,
    so local variance must equal sigma². This is an independent unit test.
    """
    K = np.asarray(K, dtype=float)
    if np.any(K <= 0) or not np.all(np.isfinite(K)):
        raise ValueError("Strikes/spots must be finite and positive")
    k = np.log(K/surface.forward(T))
    w, wk, wkk, wt = surface.variance_terms(k, T)
    if np.any(w <= 0):
        raise ValueError("Nonpositive implied total variance")
    denominator = (1-k*wk/(2*w))**2 - wk**2/4*(1/w+.25) + wkk/2
    if np.any(denominator <= 0) or np.any(wt < -1e-7):
        raise ValueError("Invalid local variance: check interpolation, wings, and calibration")
    result = np.maximum(wt, 0)/denominator  # only absorb the explicit fit tolerance
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite local variance")
    return result


def build_local_vol_grid(surface, S_grid, t_grid):
    # Cross the Python/C++ boundary once, not on every time/space node.
    return np.array([np.sqrt(local_variance(surface, S_grid, max(t, 1e-6))) for t in t_grid])
