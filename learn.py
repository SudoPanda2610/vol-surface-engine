"""Reproducible options experiment: calibration, known-answer check, barrier.

Every option quote here is synthetic. First validate the numerical engine
on flat volatility, where a Black-Scholes answer exists; then price under
calibrated local vol, where PDE/MC agreement is a weaker consistency check.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from vse import market, svi, dupire, pde_pricer, montecarlo


def flat_surface(spot=100., sigma=.2, r=.03, q=.01):
    # b=0 removes skew and curvature from raw SVI, leaving w=sigma²*T.
    return dupire.CalibratedSurface([
        dict(T=t, F=spot*np.exp((r-q)*t), params=[sigma*sigma*t, 0., 0., 0., .1])
        for t in [.1, 1., 2.]])


def black_scholes_call(spot, strike, r, q, sigma, t):
    d1 = (np.log(spot/strike)+(r-q+.5*sigma*sigma)*t)/(sigma*np.sqrt(t))
    d2 = d1-sigma*np.sqrt(t)
    return spot*np.exp(-q*t)*norm.cdf(d1)-strike*np.exp(-r*t)*norm.cdf(d2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=int, default=20000)
    parser.add_argument('--steps', type=int, default=126)
    parser.add_argument('--nodes', type=int, default=400)
    parser.add_argument('--barrier-ratio', type=float, default=.8)
    args = parser.parse_args()
    if args.paths < 2 or args.steps < 1 or args.nodes < 3 or not 0 < args.barrier_ratio < 1:
        parser.error('Invalid simulation size or barrier ratio')

    # A barrier very far below spot approximates a vanilla call. Compare to
    # an analytic value: two methods agreeing with each other is not enough
    # if both consume the same incorrectly constructed volatility surface.
    flat = flat_surface()
    exact = black_scholes_call(100, 100, .03, .01, .2, 1)
    vanilla_pde, _, _ = pde_pricer.price_down_and_out(flat, 100, 100, 1, .03, .01, 1,
                                                    N_S=801, N_t=400)
    print(f'Known-answer check: Black-Scholes={exact:.6f}, PDE={vanilla_pde:.6f}')

    # These fits use a full 12-maturity chain, but a single random draw.
    # The diagnostic percentage is for THIS draw, not a fabricated study.
    chain = market.build_option_chain(seed=0)
    naive = svi.calibrate_surface(chain, arbitrage_free=False)
    fitted = svi.calibrate_surface(chain, arbitrage_free=True)
    surface = dupire.CalibratedSurface(fitted)
    violations = sum(not(s['butterfly_clean'] and s['calendar_clean']) for s in naive)
    print(f'Naive diagnostic violations: {violations}/{len(naive)} slices')

    s0, strike, t = market.SPOT, market.SPOT*1.02, .5
    barrier = s0*args.barrier_ratio
    pde, _, _ = pde_pricer.price_down_and_out(surface, s0, strike, barrier, market.RATE,
                                             market.DIV_YIELD, t, N_S=args.nodes, N_t=args.steps*2)
    mc, se = montecarlo.simulate_barrier_price(surface, s0, strike, barrier, market.RATE,
                                              market.DIV_YIELD, t, n_paths=args.paths, n_steps=args.steps)
    # Standard error measures sampling noise ONLY. Grid, time-discretization,
    # calibration and bridge-approximation errors remain even at many paths.
    report = dict(data='synthetic', maturities=len(fitted), seed=0,
                  naive_violations=violations, paths=args.paths, steps=args.steps, nodes=args.nodes,
                  barrier_ratio=args.barrier_ratio, black_scholes=exact, flat_pde=vanilla_pde,
                  local_vol_pde=pde, local_vol_mc=float(mc), mc_standard_error=float(se),
                  mc_95_interval=[float(mc-1.96*se), float(mc+1.96*se)],
                  difference_bps_of_spot=float(abs(pde-mc)/s0*10000))
    destination = Path(__file__).parent/'output'/'learning_report.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print('Saved:', destination)


if __name__ == '__main__':
    main()
