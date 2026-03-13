"""
Raw SVI (Stochastic Volatility Inspired) per-maturity slice calibration, and
the two static no-arbitrage checks a real vol surface has to satisfy:

  - butterfly (no negative implied risk-neutral density in strike, at fixed T)
  - calendar (total variance must not decrease as T increases, at fixed log-forward moneyness)

Reference for the raw-SVI parameterization and the butterfly condition g(k):
Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014).
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# The SVI formula itself
# ---------------------------------------------------------------------------

def svi_total_variance(k, params):
    """
    Raw SVI: total implied variance (= implied_vol^2 * T) as a function of
    log-moneyness k = ln(strike / forward).

        w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

    Five params per maturity slice:
      a      overall level (vertical shift of the whole smile)
      b>=0   overall slope/steepness (how fast variance rises away from ATM)
      rho    in (-1,1), controls skew: rho<0 tilts the smile so OTM puts
             (low k) have higher variance than OTM calls -- the standard
             equity-index skew
      m      horizontal shift; minimum is m-rho*sigma/sqrt(1-rho**2) for b>0
      sigma>0 controls how rounded vs. V-shaped the smile is at the minimum

    Why this form and not "just spline the vols": SVI is a *parametric*
    curve with exactly the right asymptotic behaviour (total variance grows
    linearly in |k| as k -> +-infinity, matching Roger Lee's moment formula
    for how implied vol can behave at extreme strikes without admitting
    arbitrage), so extrapolating a few strikes beyond the quoted range is
    far safer than extrapolating a spline.
    """
    a, b, rho, m, sigma = params
    y = k - m
    s = np.sqrt(y ** 2 + sigma ** 2)
    return a + b * (rho * y + s)


def _svi_derivatives(k, params):
    """
    Analytic first and second derivatives of w(k) w.r.t. k. Needed for the
    butterfly-arbitrage check (g(k) below) and, later, for the Dupire local
    vol formula. Doing these by hand (rather than finite-differencing the
    fitted curve) keeps the arbitrage check numerically clean -- it's
    testing the *model's* shape, not finite-difference noise.

    y = k - m, s = sqrt(y^2 + sigma^2)
    w  = a + b*(rho*y + s)
    w' = b*(rho + y/s)                      [d(s)/dy = y/s]
    w''= b*sigma^2 / s^3                    [d(y/s)/dy = sigma^2/s^3]
    """
    a, b, rho, m, sigma = params
    y = k - m
    s = np.sqrt(y ** 2 + sigma ** 2)
    w = a + b * (rho * y + s)
    w1 = b * (rho + y / s)
    w2 = b * sigma ** 2 / s ** 3
    return w, w1, w2


# ---------------------------------------------------------------------------
# Butterfly (strike-arbitrage) condition
# ---------------------------------------------------------------------------

def butterfly_g(k, params):
    """
    Gatheral-Jacquier's g(k) function. A slice is free of butterfly
    arbitrage (i.e. the implied risk-neutral density stays non-negative
    on the tested domain) requires g(k) >= 0. Global absence also requires the wing limit. This comes from writing the
    Black-Scholes-with-a-smile density in terms of total variance w(k) and
    its derivatives and requiring it to stay non-negative -- see
    Gatheral & Jacquier (2014), eq. 2.9. We don't re-derive it here, just
    use it as the standard practitioner check.
    """
    w, w1, w2 = _svi_derivatives(k, params)
    term1 = (1 - k * w1 / (2 * w)) ** 2
    term2 = (w1 ** 2 / 4) * (1 / w + 0.25)
    term3 = w2 / 2
    return term1 - term2 + term3


def check_butterfly_arbitrage(params, k_range=(-1.5, 1.5), n=400, tol=-1e-8):
    """Scan g(k) on a fine grid; if it dips below `tol` anywhere, the slice
    admits butterfly arbitrage. Returns (is_clean, min_g, worst_k)."""
    k_grid = np.linspace(*k_range, n)
    g = butterfly_g(k_grid, params)
    i = np.argmin(g)
    return bool(g[i] >= tol), float(g[i]), float(k_grid[i])


# ---------------------------------------------------------------------------
# Calendar-spread (maturity-arbitrage) condition
# ---------------------------------------------------------------------------

def check_calendar_arbitrage(params_near, F_near, T_near, params_far, F_far, T_far,
                              K_range=None, n=400, tol=-1e-7):
    """Compare total variance at FIXED k=log(K/F), under proportional dividends.

    Equal k means different strikes when forwards differ. Comparing equal
    strikes instead is not the Gatheral-Jacquier calendar condition.
    K_range is retained for callers; it selects k using the near forward.
    The returned worst location is k, not a currency strike.
    """
    if not 0 < T_near < T_far or min(F_near, F_far) <= 0:
        raise ValueError("Need positive forwards and strictly increasing maturities")
    limits = (-1.5, 1.5) if K_range is None else np.log(np.asarray(K_range) / F_near)
    k = np.linspace(*limits, n)
    difference = svi_total_variance(k, params_far) - svi_total_variance(k, params_near)
    i = np.argmin(difference)
    return bool(difference[i] >= tol), float(difference[i]), float(k[i])


_BOUNDS = [(-1., 1.), (1e-4, 3.), (-.999, .999), (-1., 1.), (.001, 3.)]


def _initial_guess(k, w_mkt):
    return [max(w_mkt.min() * .9, 1e-4), .1, -.5, k[np.argmin(w_mkt)], .1]


def calibrate_slice_unconstrained(k, w_mkt, x0=None, robust=False):
    """Fit quotes with parameter bounds only; a small SSE is not a safety check."""
    if x0 is None:
        x0 = _initial_guess(k, w_mkt)
    scale = max(float(np.sum(w_mkt*w_mkt)),1e-12) if robust else 1.
    options = {'maxiter':1500,'ftol':1e-12,'gtol':1e-8} if robust else {'maxiter':500}
    res = minimize(lambda p: np.sum((svi_total_variance(k, p) - w_mkt)**2)/scale,
                   x0, method="L-BFGS-B", bounds=_BOUNDS, options=options)
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"SVI least-squares fit failed: {res.message}")
    return res.x, float(np.sum((svi_total_variance(k,res.x)-w_mkt)**2))


def calibrate_slice_arbitrage_free(k, w_mkt, F_this, T_this, prev_slice=None,
                                  x0=None, n_constraint_grid=1200, robust=False):
    """Constrained SVI fit on a finite grid; this is not a global proof.

    Positive minimum variance makes square roots meaningful. Wing slopes
    below two enforce the asymptotic bound; g(k) checks density on the grid.
    SLSQP's returned parameters are accepted only after checking feasibility.
    """
    if x0 is None:
        x0 = _initial_guess(k, w_mkt)
        if prev_slice is not None:
            # Warm-start from the prior smile, shifted upward in variance.
            # Unlike an unrelated default, this is close to calendar-feasible.
            x0 = np.array(prev_slice[0], dtype=float).copy()
            x0[0] += max(float(np.mean(w_mkt-svi_total_variance(k, x0))), 1e-5)
    k_grid = np.linspace(-1.5, 1.5, n_constraint_grid)

    def margins(p):
        a, b, rho, m, sigma = p
        # A small positive density margin in robust mode reduces crossings
        # between constraint nodes. The independent denser scan still decides.
        values = [butterfly_g(k_grid, p)-(1e-5 if robust else 0.),
                  np.array([a + b * sigma * np.sqrt(1-rho*rho) - 1e-8,
                            2 - b*(1+rho) - 1e-6, 2 - b*(1-rho) - 1e-6])]
        if prev_slice is not None:
            prev_params, prev_F, prev_T = prev_slice
            if prev_T >= T_this:
                raise ValueError("Fit shorter maturities first")
            # A small interior margin avoids accepting a crossing between
            # adjacent constraint nodes. The independent scan is still required.
            values.append(svi_total_variance(k_grid, p) - svi_total_variance(k_grid, prev_params) - 1e-6)
        return np.concatenate(values)

    res = minimize(lambda p: np.sum((svi_total_variance(k, p)-w_mkt)**2),
                   x0, method="SLSQP", bounds=_BOUNDS,
                   constraints=[{"type": "ineq", "fun": margins}],
                   options={"maxiter": 1500, "ftol": 1e-11})
    if robust or not res.success or not np.all(np.isfinite(res.x)) or np.min(margins(res.x)) < -1e-7:
        # Raw a/b/rho/sigma trial steps can pass through w=0, where g divides
        # by w. Reparameterize a via the strictly positive minimum variance
        # for a numerical fallback; keep the original economic constraints.
        # Scale the objective because short-expiry w errors can be tiny.
        scale = max(float(np.mean(w_mkt*w_mkt)), 1e-12)
        bounds = [(1e-8, 3.), *_BOUNDS[1:]]

        def unpack(z):
            minimum, b, rho, m, sigma = z
            return np.array([minimum-b*sigma*np.sqrt(1-rho*rho), b, rho, m, sigma])

        def pack(p):
            a,b,rho,m,sigma = p
            return np.array([max(a+b*sigma*np.sqrt(1-rho*rho),1e-7),b,rho,m,sigma])

        def safe_margins(z):
            p = unpack(z)
            # Retain the original a bounds as explicit inequalities too.
            return np.r_[margins(p), p[0]+1., 1.-p[0]]

        flat_level = max(float(np.mean(w_mkt)), 1e-6)
        if prev_slice is not None:
            flat_level = max(flat_level, float(np.max(svi_total_variance(k_grid, prev_slice[0])))+1e-5)
        starts = [pack(x0), np.array([flat_level,1e-4,0.,0.,.2])]
        candidates = []
        if res.success and np.all(np.isfinite(res.x)) and np.min(margins(res.x)) >= -1e-7:
            candidates.append((res.x,float(res.fun)))
        messages = [str(res.message)]
        for start in starts:
            trial = minimize(lambda z: np.mean((svi_total_variance(k,unpack(z))-w_mkt)**2)/scale,
                             start, method='SLSQP', bounds=bounds,
                             constraints=[{'type':'ineq','fun':safe_margins}],
                             options={'maxiter':1500,'ftol':1e-11})
            p = unpack(trial.x)
            if trial.success and np.all(np.isfinite(p)) and np.min(safe_margins(trial.x)) >= -1e-7:
                candidates.append((p,float(np.sum((svi_total_variance(k,p)-w_mkt)**2))))
            messages.append(str(trial.message))
        if not candidates:
            raise RuntimeError(f"Constrained SVI fit failed: {'; '.join(messages)}")
        return min(candidates,key=lambda item:item[1])
    return res.x, res.fun


def calibrate_surface(chain, arbitrage_free=True, robust=False):
    """Fit short to long, then independently check a denser diagnostic grid."""
    results, prev = [], None
    for s in sorted(chain, key=lambda s: s["T"]):
        w_mkt = s["iv_mid"]**2 * s["T"]
        try:
            if arbitrage_free:
                params, sse = calibrate_slice_arbitrage_free(s["k"], w_mkt, s["F"], s["T"], prev,robust=robust)
            else:
                params, sse = calibrate_slice_unconstrained(s["k"], w_mkt,robust=robust)
        except RuntimeError as exc:
            raise RuntimeError(f"T={s['T']}: {exc}") from exc
        clean_bf, min_g, _ = check_butterfly_arbitrage(params, n=2400, tol=-1e-7)
        clean_cal, min_diff = True, None
        if prev is not None:
            clean_cal, min_diff, _ = check_calendar_arbitrage(*prev, params, s["F"], s["T"], n=2400)
        if arbitrage_free and not (clean_bf and clean_cal):
            raise RuntimeError(f"SVI dense-grid failure T={s['T']}: min_g={min_g}, calendar_gap={min_diff}")
        results.append({**s, "params": params, "sse": sse,
                        "butterfly_clean": clean_bf, "butterfly_min_g": min_g,
                        "calendar_clean": clean_cal, "calendar_min_diff": min_diff})
        prev = params, s["F"], s["T"]
    return results
