"""
Builds a synthetic "SPX-like" option chain: 12 maturities, a strike grid per
maturity, and market-observed implied vols that actually look like an
equity index smile (downward skew: OTM puts trade at higher implied vol
than OTM calls -- the classic "leverage effect" / crash-insurance skew).

There's no real data feed here. We construct the "true" market by picking a
believable SVI shape for the whole surface, evaluating implied vols off it,
and adding a bit of bid-ask-style noise -- so downstream calibration has a
genuine (if synthetic) target to fit, and genuine noise to occasionally
produce an arbitrage-violating unconstrained fit.
"""
import numpy as np

# 12 maturities in years, roughly weekly-to-LEAPS like a real SPX chain
MATURITIES = np.array([
    7, 14, 30, 60, 90, 120, 180, 270, 365, 545, 730, 1095
]) / 365.0

SPOT = 5000.0
RATE = 0.045      # risk-free rate (continuous comp.)
DIV_YIELD = 0.013  # SPX dividend yield


def forward(spot, r, q, T):
    """Forward price for maturity T under continuous rate r and dividend yield q."""
    return spot * np.exp((r - q) * T)


def _true_svi_params(T):
    """A modest synthetic skew with a finite short-time volatility limit.

    Every variance term scales with T. A nonzero intercept in w as T->0
    would imply exploding implied volatility and an artificial short-time
    layer that a coarse simulation can miss. Keep this teaching input smooth.
    The slight short-end vol elevation decays with maturity; it is not a
    claim to reproduce any particular SPX date or observed term structure.
    """
    b, rho, m, sigma = .015*T, -.65, 0., .2
    w_atm = T * (.18**2 + .008*np.exp(-T/.6))
    a = w_atm - b*sigma
    return a, b, rho, m, sigma


def _svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI total-variance formula, w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+sigma^2))."""
    y = k - m
    return a + b * (rho * y + np.sqrt(y ** 2 + sigma ** 2))


def build_option_chain(n_strikes=25, noise_bps=15, seed=7):
    """
    Returns a list of dicts, one per maturity:
      {T, F, log_moneyness (k), strikes (K), iv_mid (market implied vol)}

    `noise_bps` is the std-dev of the vol noise added to each quote, in vol
    points *100 (e.g. 15 -> 0.15 vol points = 15bps of vol -- deliberately
    small, real bid-ask on SPX options is tighter than what blows up a fit;
    the point is to have *some* noise, not to manufacture arbitrage).
    """
    rng = np.random.default_rng(seed)
    chain = []

    for T in MATURITIES:
        F = forward(SPOT, RATE, DIV_YIELD, T)
        # strikes spanning roughly 70%-130% moneyness, denser near the money
        moneyness_grid = np.concatenate([
            np.linspace(-0.35, -0.05, n_strikes // 2, endpoint=False),
            np.linspace(-0.05, 0.30, n_strikes - n_strikes // 2),
        ])
        k = moneyness_grid  # log-moneyness ln(K/F), already roughly in this range
        strikes = F * np.exp(k)

        a, b, rho, m, sigma = _true_svi_params(T)
        w = _svi_total_variance(k, a, b, rho, m, sigma)
        w = np.maximum(w, 1e-6)
        true_iv = np.sqrt(w / T)

        noisy_iv = true_iv + rng.normal(0, noise_bps / 1e4, size=len(k))
        noisy_iv = np.maximum(noisy_iv, 0.02)  # floor so nothing goes negative/absurd

        chain.append({
            "T": T, "F": F, "k": k, "strikes": strikes, "iv_mid": noisy_iv,
        })

    return chain
