"""
End-to-end run: synthetic SPX-like option chain -> arbitrage-free SVI
calibration per maturity -> Dupire local vol surface -> price a down-and-out
barrier call via the C++ Crank-Nicolson PDE engine -> validate against a
200k-path Monte Carlo simulation. Prints a report and saves charts.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vse import market, svi, dupire, pde_pricer, montecarlo

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Synthetic SVI/local-vol pricing experiment")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--paths", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=126)
    args = parser.parse_args()
    if args.trials < 1 or args.paths < 2 or args.steps < 1:
        parser.error("Use at least one trial, two paths and one step")
    print("=" * 70)
    print("1. UNCONSTRAINED VS ARBITRAGE-FREE SVI CALIBRATION")
    print("=" * 70)
    print(f"Calibrating synthetic 12-maturity chains across {args.trials} noise draws,")
    print("both without and with the butterfly/calendar constraints, to")
    print("measure how often the naive fit actually admits arbitrage...")

    n_trials = args.trials
    unc_bad = unc_total = af_bad = af_total = 0
    for seed in range(n_trials):
        ch = market.build_option_chain(seed=seed)
        unc = svi.calibrate_surface(ch, arbitrage_free=False)
        af = svi.calibrate_surface(ch, arbitrage_free=True)
        for r_ in unc:
            unc_total += 1
            if not (r_["butterfly_clean"] and r_["calendar_clean"]):
                unc_bad += 1
        for r_ in af:
            af_total += 1
            if not (r_["butterfly_clean"] and r_["calendar_clean"]):
                af_bad += 1

    print(f"  unconstrained: {unc_bad}/{unc_total} slice-fits violate an arbitrage "
          f"condition ({unc_bad/unc_total*100:.1f}%)")
    print(f"  constrained (SLSQP, grid diagnostics): {af_bad}/{af_total} violate "
          f"({af_bad/af_total*100:.1f}%)")

    # ---- 2. Calibrate one specific chain to build the rest of the pipeline on ----
    chain = market.build_option_chain(seed=0)
    slices = svi.calibrate_surface(chain, arbitrage_free=True)
    surface = dupire.CalibratedSurface(slices)

    print("\n" + "=" * 70)
    print("2. CALIBRATED SVI SLICES (this run's surface)")
    print("=" * 70)
    for s in slices:
        a, b, rho, m, sigma = s["params"]
        print(f"  T={s['T']:6.3f}  a={a:7.4f} b={b:6.4f} rho={rho:6.3f} "
              f"m={m:6.3f} sigma={sigma:6.4f}  fit SSE={s['sse']:.2e}")

    # ---- 3. Dupire local vol surface ----
    print("\n" + "=" * 70)
    print("3. DUPIRE LOCAL VOLATILITY SURFACE")
    print("=" * 70)
    S_grid_plot = np.linspace(3500, 7000, 80)
    t_grid_plot = np.linspace(0.03, 2.0, 60)
    lv_grid = dupire.build_local_vol_grid(surface, S_grid_plot, t_grid_plot)
    print(f"  grid: {lv_grid.shape[0]} maturities x {lv_grid.shape[1]} strikes")
    print(f"  local vol range: {lv_grid.min()*100:.1f}% - {lv_grid.max()*100:.1f}%, "
          f"median {np.median(lv_grid)*100:.1f}%")

    # ---- 4. Price a down-and-out barrier call: PDE vs Monte Carlo ----
    S0 = market.SPOT
    K = S0 * 1.02       # slightly OTM call
    barrier = S0 * 0.80  # 20% down-and-out barrier
    T = 0.5
    r, q = market.RATE, market.DIV_YIELD

    print("\n" + "=" * 70)
    print(f"4. DOWN-AND-OUT BARRIER CALL: S0={S0:.0f} K={K:.0f} "
          f"barrier={barrier:.0f} T={T}y")
    print("=" * 70)

    pde_price, S_grid_pde, V_pde = pde_pricer.price_down_and_out(
        surface, S0, K, barrier, r, q, T, is_call=True, N_S=400, N_t=400)
    print(f"  PDE (Crank-Nicolson, C++):  {pde_price:.4f}")

    mc_price, mc_se = montecarlo.simulate_barrier_price(
        surface, S0, K, barrier, r, q, T, n_paths=args.paths, n_steps=args.steps)
    print(f"  Monte Carlo ({args.paths} paths): {mc_price:.4f}  (std error {mc_se:.4f})")

    diff_bps = abs(pde_price - mc_price) / S0 * 1e4
    print(f"  |difference|: {abs(pde_price-mc_price):.4f}  "
          f"= {diff_bps:.2f} bp of spot  ({abs(pde_price-mc_price)/mc_se:.1f} MC std errors)")

    _make_charts(chain, slices, surface, S_grid_plot, t_grid_plot, lv_grid,
                 S_grid_pde, V_pde, K, barrier, pde_price, mc_price, mc_se)
    print(f"\nCharts saved to {OUT_DIR}/")


def _make_charts(chain, slices, surface, S_grid_plot, t_grid_plot, lv_grid,
                  S_grid_pde, V_pde, K, barrier, pde_price, mc_price, mc_se):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # smile fit for a few maturities: market points vs calibrated SVI curve
    ax = axes[0, 0]
    plot_slices = [slices[i] for i in (1, 4, 7, 10)]
    for s in plot_slices:
        k_fine = np.linspace(s["k"].min(), s["k"].max(), 200)
        iv_fit = np.sqrt(np.maximum(svi.svi_total_variance(k_fine, s["params"]), 1e-10) / s["T"])
        line, = ax.plot(k_fine, iv_fit * 100, label=f"T={s['T']:.2f}")
        ax.plot(s["k"], s["iv_mid"] * 100, "o", ms=3, color=line.get_color())
    ax.set_title("SVI fit vs market smile (dots=market, lines=fit)")
    ax.set_xlabel("log-moneyness k"); ax.set_ylabel("implied vol (%)"); ax.legend(fontsize=8)

    # local vol heatmap
    ax = axes[0, 1]
    im = ax.pcolormesh(S_grid_plot, t_grid_plot, lv_grid * 100, shading="auto", cmap="viridis")
    ax.set_title("Dupire local volatility (%)")
    ax.set_xlabel("S"); ax.set_ylabel("t (years)")
    plt.colorbar(im, ax=ax)

    # PDE price curve vs payoff
    ax = axes[1, 0]
    payoff = np.maximum(S_grid_pde - K, 0.0)
    ax.plot(S_grid_pde, payoff, "--", color="gray", label="payoff at T")
    ax.plot(S_grid_pde, V_pde, color="tab:blue", label="PDE price at t=0")
    ax.axvline(barrier, color="crimson", linestyle=":", label=f"barrier={barrier:.0f}")
    ax.set_title("Down-and-out call: price vs payoff")
    ax.set_xlabel("S"); ax.set_ylabel("value"); ax.legend(fontsize=8)
    ax.set_xlim(barrier * 0.9, K * 1.6)

    # PDE vs MC comparison
    ax = axes[1, 1]
    ax.bar(["PDE (C++)", "Monte Carlo"], [pde_price, mc_price],
           yerr=[0, mc_se], capsize=5, color=["tab:blue", "tab:orange"])
    ax.set_title("PDE vs Monte Carlo validation")
    ax.set_ylabel("price")
    for i, v in enumerate([pde_price, mc_price]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "report.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
