"""
Monte Carlo pricer for the same down-and-out barrier option under the same
local volatility surface, used purely as an independent check on the C++
PDE engine -- two completely different numerical methods (finite
differences vs. simulation) that should agree if both are implemented
correctly, since they're solving for the same risk-neutral expectation.

Euler-Maruyama discretization of dS = (r-q)*S*dt + sigma_local(S,t)*S*dW,
simulated in log-space (dX = d(ln S)) so S stays positive by construction --
a plain Euler step on S itself can occasionally send a path negative for a
large local vol and a coarse time step, which is a well-known Euler-on-S
pitfall in a local-vol simulation, so we sidestep it.
"""
import numpy as np
from vse.dupire import local_variance


def simulate_barrier_price(surface, S0, K, barrier, r, q, T,
                            n_paths=200_000, n_steps=252, is_call=True, seed=11,
                            continuous_monitoring=True):
    """
    Full-precision (double), single-threaded numpy Monte Carlo. Not the
    fastest possible implementation (a batched/vectorized-in-time version
    or GPU would be much faster for production use) but simple enough to
    read top to bottom and trust.
    """
    if min(S0, K, barrier, T) <= 0 or n_paths < 2 or n_steps < 1:
        raise ValueError("Use positive prices/maturity, at least two paths and one step")
    if S0 <= barrier:
        return 0.0, 0.0  # knocked out at inception, even if the next step rises
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    log_S = np.full(n_paths, np.log(S0))
    survival = np.ones(n_paths)

    for step in range(n_steps):
        t = step * dt
        S = np.exp(log_S)

        # local vol needs a scalar-ish call per path; surface.total_variance
        # is already vectorized over strike, so batch all paths through it
        # at this single time point.
        v_L = local_variance(surface, S, max(t, 1e-4))
        sigma = np.sqrt(v_L)

        dW = rng.normal(0, sqrt_dt, n_paths)
        # Ito's lemma on X = ln(S): dX = (r - q - 0.5*sigma^2)*dt + sigma*dW
        log_S = log_S + (r - q - 0.5 * sigma ** 2) * dt + sigma * dW

        S_new = np.exp(log_S)
        above = S_new > barrier
        if continuous_monitoring:
            # Endpoint checks miss crossings between observations. For a
            # frozen log-vol step, Brownian-bridge crossing probability is
            # exp(-2*log(S/B)*log(S_new/B)/(variance*dt)). Multiplying by
            # survival probability avoids an extra Bernoulli random draw.
            # Exact for constant vol per step; approximate for local vol.
            distance = np.maximum(np.log(S/barrier), 0) * np.maximum(np.log(S_new/barrier), 0)
            crossing = np.exp(-2*distance / np.maximum(v_L*dt, 1e-20))
            survival *= above * (1-crossing)
        else:
            survival *= above

    S_T = np.exp(log_S)
    payoff = np.where(is_call, np.maximum(S_T - K, 0.0), np.maximum(K - S_T, 0.0))
    payoff *= survival

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, std_error
