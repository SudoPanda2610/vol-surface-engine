# Volatility surface and barrier pricing

Fit implied-volatility smiles with SVI, screen their shape, construct local volatility, and price a down-and-out option in C++ through pybind11. A Python Monte Carlo implementation provides a second numerical method. The original demonstrations use synthetic quotes; a separate historical runner reads a Cboe SPXW sample dated 25 August 2023.

Start with [the four-hour study guide](STUDY_GUIDE.md) and the commented `learn.py`. The guide includes 12 interview questions and a known-answer-to-local-vol learning sequence.

```bash
venv/bin/python learn.py
venv/bin/python -m unittest discover -s tests -v
venv/bin/python main.py
# Optional larger comparison; records these exact settings in the JSON:
venv/bin/python learn.py --paths 200000 --steps 252 --nodes 800
```

Fresh setup: create `venv`, install `requirements.txt`, then:

```bash
cmake -S cpp -B build -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)" -DPYTHON_EXECUTABLE="$PWD/venv/bin/python"
cmake --build build -j4
```

`learn.py` writes `output/learning_report.json`. `main.py` writes `output/report.png`; it defaults to one noise draw and 20,000 paths. `--trials 60 --paths 200000 --steps 252` is a larger experiment, not a prerequisite to studying the code.

## Historical SPXW experiment

```bash
# 5% is an explicit constant discount-rate ASSUMPTION, not an observed curve.
venv/bin/python historical.py --rate 0.05 --price
```

`historical.py` reads the unchanged `cgi_or_historical` CSV from [Cboe's EOD sample](https://datashop.cboe.com/option-eod-summary). Download the sample from Cboe and place `UnderlyingOptionsEODCalcs_2023-08-25_cgi_or_historical.csv` in `data/historical/`, or pass its path with `--csv`. The CSV is excluded from this repository; the provider caveat and download/hash metadata are in `data/historical/`. Cboe describes this as a demonstration sample which may contain only a subset of actual data. The quote date is 2023-08-25; the loader rejects dates later than 2023 and never falls back to synthetic data.

The experiment uses only PM-settled SPXW contracts and the 15:45 snapshot, with expiry at 16:00 Eastern and ACT/365 year fractions. It excludes crossed, nonpositive and wide-spread quotes, then selects 12 evenly spaced indices from eligible sorted expiries between 7 and 730 days, before calibration. It infers a constant dividend yield from near-ATM put-call parity given the explicit rate assumption, and inverts OTM bid/ask midpoints under that same carry model. Vendor IVs are not reused. This approximation does not supply a historical discount/dividend term structure.

Historical calibration uses scaled total-variance losses to avoid premature convergence at short maturities. Constrained SLSQP also searches in minimum-variance coordinates, keeping variance positive during trial steps, with two deterministic starting points and the original fit as an additional candidate when feasible. It retains the original economic constraints, applies a small positive density margin, and checks a separate denser grid. The synthetic demonstrations retain their original mode, with the positive-variance parameterization available as a fallback on failure.

`output/historical_report.json` records selected dates, filters, carry assumptions, input/code hashes, runtime versions, every fitted slice, the measured diagnostic percentage, and optional pricing results. The diagnostic range is k in [-1.5,1.5], wider than the selected quote range [-0.35,0.30]; failures can concern extrapolated strikes. Failed optimizations or invalid local variance are reported as failures, rather than replaced with invented results. A single date and one barrier contract cannot establish a universal violation frequency or pricing tolerance.

## Numerical conventions and validation

Calendar comparisons and maturity interpolation use fixed `k=log(K/F(T))`. This corrects the previous fixed-currency-strike implementation. The Dupire variance formula uses analytic SVI strike derivatives and a piecewise time derivative. Its flat-vol limit is independently tested. See [Gatheral and Jacquier, section 2](https://arxiv.org/abs/1204.0646).

Constrained fits require positive variance, bounded wing slopes and density/calendar checks on a finite grid. A separate denser check can reject a fit. This is not a guarantee of global no-arbitrage or validity of every interpolated/extrapolated point.

The synthetic smile generator has a finite volatility limit as maturity tends to zero. The previous generator had a nonzero short-time total-variance intercept; after correcting surface coordinates, the coarse PDE/MC comparison on that input differed by about 47.5 bps of spot. The generator was simplified to make the learning experiment smooth; that earlier discrepancy is not evidence of validation for difficult short-time surfaces. The fixed seed was retained. The new default draw has zero unconstrained diagnostic violations; no 31% figure is assumed or manufactured.

Monte Carlo uses Brownian-bridge survival weights for continuous monitoring under frozen step volatility. The bridge is approximate for a variable local-vol model. The binding accepts strided arrays by converting to contiguous memory and validates array dimensions before reading them.

The core tests cover the Black-Scholes limit and grid refinement, flat local-vol recovery including short maturities, calendar coordinates, invalid local variance, barrier behaviour, bridge versus endpoint monitoring and C++ array handling. Sampling error, numerical bias and model error remain distinct. Read the generated JSON for a measured price difference and its MC interval.

## Scope

`learn.py` and `main.py` use synthetic quotes; `historical.py` imports a dated Cboe SPXW sample. The tested contract is a European down-and-out option. No production claim is made about global no-arbitrage, universal 8bp accuracy or speedup versus Python. The two numerical methods share local-vol inputs, so their agreement alone cannot validate calibration. The original study guide describes the synthetic path.
