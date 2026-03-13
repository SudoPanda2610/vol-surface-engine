# Options calibration and pricing: a four-hour working session

**Goal:** understand what is being calibrated, follow a numerical option price, and explain validation errors. Four hours is an introduction to the implemented pipeline. Deriving SVI no-arbitrage theory and Dupire from first principles takes longer.

## Run and inspect a known answer — 0–20 minutes

From this folder, using the prepared environment and C++ build:

```bash
venv/bin/python learn.py
venv/bin/python -m unittest discover -s tests -v
```

On a fresh machine, create `venv`, install `requirements.txt`, then build:

```bash
cmake -S cpp -B build -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)" -DPYTHON_EXECUTABLE="$PWD/venv/bin/python"
cmake --build build -j4
```

Read `learn.py`. The first comparison uses a flat 20% volatility surface and a very distant lower barrier to approximate a vanilla call. Black-Scholes supplies an independent analytic answer. Later it calibrates synthetic quotes and compares local-vol PDE and Monte Carlo prices. Results go to `output/learning_report.json`; no real SPX dataset is used.

## Implied volatility, total variance and SVI — 20–60 minutes

Read `vse/market.py` and the formulas near the beginning of `vse/svi.py`.

An implied volatility is the volatility input making Black-Scholes match an observed option price. Different strikes can imply different volatilities; that is the smile or skew. Here we generate those implied-volatility observations directly, rather than invert real option prices.

Use `k=log(K/F(T))` for log-forward moneyness. At-the-money-forward has k=0. Use `w=implied_vol²*T` for total variance. For 20% volatility over half a year, w is `.2²*.5=.02`, not `.2*.5`.

Raw SVI fits `w(k)=a+b*(rho*(k-m)+sqrt((k-m)²+sigma²))`. It has five parameters: vertical shift, wing steepness, skew, horizontal shift, and roundness. Its parameter named `sigma` is a shape parameter; it is **not** directly the option's implied volatility. With b=0 the smile becomes flat, which is useful for tests.

**Do:** calculate w for 20% and 30% vol at T=.5. Set rho to zero in a scratch example and compare symmetric k values around m. Explain which parameter controls the skew.

## Fit quality versus valid prices — 60–95 minutes

Read the constrained calibration and diagnostic functions in `vse/svi.py`.

A low sum of squared quote errors does not imply economically valid prices. Butterfly conditions relate to nonnegative risk-neutral density. Calendar consistency, in these coordinates and under proportional dividends, requires total variance to be nondecreasing at **fixed k**. Equal k generally means different currency strikes because forwards change with maturity.

This was a substantive correction: the old implementation compared fixed currency strikes and described it as the Gatheral-Jacquier condition. The corrected implementation uses fixed k in both calendar checks and Dupire interpolation. The source is [Gatheral and Jacquier, section 2](https://arxiv.org/abs/1204.0646).

We enforce constraints on a finite grid, positive minimum variance and wing-slope bounds, then check a denser grid independently. This is numerical screening, not a theorem proving all strikes and interpolated times are arbitrage-free. Optimizer or diagnostic failure raises an error. A single seeded chain is not evidence for the resume's claimed 31% violation frequency.

**Do:** explain why positive variance alone does not guarantee positive density. Read the test with identical smiles but different forwards: fixed-k comparison must find zero difference.

## Local volatility — 95–125 minutes

Read `vse/dupire.py`. Implied vol summarizes a whole option's price; local vol is the instantaneous volatility assigned to a particular spot and time in a diffusion.

The key structure is `local_variance = w_T / g`, where g depends on strike derivatives. All derivatives must use the same coordinates. In a flat-vol model, w=sigma²*T, its strike derivatives are zero, g=1, and local variance must recover sigma². This is the most useful sanity check to explain without memorizing the full formula.

Interpolation is linear in total variance at fixed k. Its time derivative is piecewise constant, with a right-hand convention at knots. Outside the quoted maturities, this implementation keeps implied vol constant at fixed k. Clamping total variance instead would force its time derivative to zero. The code raises on materially negative local variance rather than masking it with a small positive floor.

**Limit:** valid individual SVI slices do not guarantee every interpolated smile or extrapolated wing is valid. PDE and MC sharing the same surface can agree while that surface is wrong.

## PDE mechanics — 125–165 minutes

Read `vse/pde_pricer.py`, then the equation, coefficient construction and time loop in `cpp/crank_nicolson.cpp`. Leave pybind11 plumbing until last.

At expiry the payoff is known: `max(S-K,0)` for a call. The PDE works backwards to today's price. Each spot point depends on its two neighbours through central differences. Crank-Nicolson averages the spatial operator over the two time levels. Every step solves a tridiagonal system; the Thomas algorithm costs O(number of spot nodes), so the solve costs O(time steps × spot nodes).

A down-and-out option becomes worthless when spot touches its lower barrier. Set the grid's lower edge to that barrier and hold the value at zero there. The upper edge uses a far-in-the-money call approximation; moving it too close contaminates the price.

**Do:** point to the payoff, barrier boundary and upper boundary in the C++ code. Explain why Python precomputes a dense local-vol grid and crosses into C++ once. Read the known-answer grid-refinement test. Stability does not imply accuracy or absence of payoff-kink oscillations.

## Monte Carlo and experiments — 165–210 minutes

Read `vse/montecarlo.py`. Simulate log spot to keep prices positive; the drift includes `-.5*sigma²` from Ito's lemma. Average discounted surviving payoffs.

Checking the barrier only at time-step endpoints misses crossings. A Brownian-bridge survival weight corrects this under frozen step volatility. It is exact for a constant-vol step and approximate when local volatility changes. Raising a down-and-out barrier should reduce a nonnegative option payoff under common paths.

```bash
venv/bin/python learn.py --paths 20000 --steps 126 --nodes 400
venv/bin/python learn.py --paths 80000 --steps 126 --nodes 400
venv/bin/python learn.py --paths 20000 --steps 252 --nodes 800
```

Record the JSON after each run before it is overwritten. Four times the paths should roughly halve standard error; finer time/space grids address different numerical errors. A 95% MC interval captures sampling uncertainty, not calibration or time-step error. Compare both absolute currency error and basis points **of spot**, specifying the denominator. Eight bps of a 5,000 spot is 4 price units; it is not eight bps of a roughly 200-unit option premium.

## Explain and modify — 210–240 minutes

Implement a small table comparing bridge monitoring on/off at the same seed, paths and steps. Predict which price is higher and explain why. Then tell a two-minute story: input quotes → SVI → diagnostics → local volatility → PDE → independent known-answer check and MC comparison → limits.

## Interview questions and answer outlines

1. **Implied versus local volatility?** Implied vol is a parameter inferred from an option price; local vol is a spot/time-dependent diffusion coefficient.
2. **Why total variance?** It accumulates over time and gives convenient calendar conditions and Dupire derivatives.
3. **Is the SVI sigma parameter the implied vol?** No; it controls roundness. Implied vol is sqrt(w/T).
4. **Why fixed k for calendars?** The condition uses normalized forward moneyness under proportional dividends. Fixing currency strike moves k when carry changes.
5. **Does SLSQP success prove no arbitrage?** No. Check returned constraints, a denser grid, wings and interpolation. Numerical feasibility is finite-domain evidence.
6. **Why can local vol become negative?** Inconsistent time evolution or invalid smile curvature can invalidate the numerator/denominator. Flooring it hides model problems.
7. **Why Crank-Nicolson?** Second-order time accuracy for smooth solutions and good stability properties with a tridiagonal solve. Kinks and coarse grids still need attention.
8. **Why C++?** An explicit time/space loop fits compiled execution; Python handles calibration and orchestration. Do not claim a speedup without a benchmark.
9. **What does pybind11 need to know about arrays?** Shape, type and memory layout. Noncontiguous arrays cannot be read as row-major without conversion; the binding now requests contiguous data.
10. **Why might MC price exceed PDE?** Endpoint-only monitoring misses knockouts; sampling noise and discretization differ. Check matched contracts and uncertainty before blaming the PDE.
11. **Why isn't PDE/MC agreement sufficient?** They share local-vol input. Also validate the flat-vol limit against an analytic answer and study convergence.
12. **What remains unproven?** Global surface no-arbitrage, fidelity to real SPX quotes and price accuracy across a wide contract set. The new JSON supports only its explicit run.

**Ready-to-discuss check:** calculate w, distinguish k from K, explain the barrier boundary, describe the MC interval, and change monitoring yourself. Treat the full Dupire derivation as further study rather than claiming mastery.
