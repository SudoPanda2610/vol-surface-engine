"""Read an older Cboe 15:45 SPXW snapshot; never substitute synthetic quotes.

Only PM-settled SPXW contracts are used. A constant discount-rate assumption
is supplied explicitly and constant q is estimated from near-ATM put-call
parity. Midpoint IVs are recomputed under the same carry approximation used
for calibration and pricing. This is not an observed rate/dividend curve.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


def maturity(quote_date, expiration):
    eastern = ZoneInfo('America/New_York')
    start = datetime.combine(date.fromisoformat(quote_date), time(15, 45), eastern)
    end = datetime.combine(date.fromisoformat(expiration), time(16), eastern)
    return (end.astimezone(timezone.utc)-start.astimezone(timezone.utc)).total_seconds()/(365*86400)


def option_price(spot, strike, T, r, q, sigma, is_call):
    d1 = (np.log(spot/strike)+(r-q+.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if is_call:
        return spot*np.exp(-q*T)*ndtr(d1)-strike*np.exp(-r*T)*ndtr(d2)
    return strike*np.exp(-r*T)*ndtr(-d2)-spot*np.exp(-q*T)*ndtr(-d1)


def implied_vol(mid, spot, strike, T, r, q, is_call):
    if not np.all(np.isfinite([mid, spot, strike, T, r, q])) or min(mid, spot, strike, T) <= 0:
        raise ValueError('Invalid option inputs')
    underlying, discounted_strike = spot*np.exp(-q*T), strike*np.exp(-r*T)
    lower = max(underlying-discounted_strike if is_call else discounted_strike-underlying, 0.)
    upper = underlying if is_call else discounted_strike
    if not lower < mid < upper:
        raise ValueError('Midpoint outside European option price bounds')
    return brentq(lambda v: option_price(spot, strike, T, r, q, v, is_call)-mid,
                  1e-7, 5., xtol=1e-12)


def load_chain(path, quote_date='2023-08-25', n_maturities=12, min_days=7,
               max_days=730, max_relative_spread=.5, min_quotes=10, *, rate=None):
    """Choose expiries before fitting, by evenly spaced sorted expiry indices.

    Returns a chain compatible with svi.calibrate_surface and full selection
    metadata. It fails if the requested number of usable expiries is absent.
    Quote selection and expiry choice never depend on calibration outcomes.
    """
    if date.fromisoformat(quote_date) > date(2023, 12, 31):
        raise ValueError('Historical experiment accepts quote dates through 2023 only')
    if n_maturities < 2 or min_quotes < 5 or not 0 < min_days < max_days or not 0 < max_relative_spread <= 2:
        raise ValueError('Invalid historical selection settings')
    if rate is None or not np.isfinite(rate) or not -.02 <= rate <= .2:
        raise ValueError('Supply an explicit constant discount-rate assumption between -0.02 and 0.2')
    path = Path(path)
    groups = defaultdict(dict)
    counts = Counter()
    spots = []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'underlying_symbol', 'root', 'quote_date', 'expiration', 'strike',
                    'option_type', 'bid_1545', 'ask_1545', 'active_underlying_price_1545'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Not the expected Cboe EOD-with-calcs CSV schema')
        for row in reader:
            counts['input_rows'] += 1
            if row['underlying_symbol'] != '^SPX' or row['root'] != 'SPXW' or row['quote_date'] != quote_date:
                continue
            counts['matching_spxw_rows'] += 1
            try:
                T = maturity(quote_date, row['expiration'])
                K, bid, ask, S = (float(row[key]) for key in
                                  ['strike', 'bid_1545', 'ask_1545', 'active_underlying_price_1545'])
                if not np.all(np.isfinite([K, bid, ask, S])) or min(K, S) <= 0:
                    raise ValueError('Nonfinite or nonpositive price')
            except (ValueError, TypeError):
                counts['invalid_rows'] += 1
                continue
            if not min_days <= T*365 <= max_days:
                counts['outside_maturity_window'] += 1
                continue
            if row['option_type'] not in ('C', 'P') or bid <= 0 or ask < bid or (ask-bid)/((ask+bid)/2) > max_relative_spread:
                counts['rejected_quotes'] += 1
                continue
            key = (K, row['option_type'])
            if key in groups[row['expiration']]:
                raise ValueError('Duplicate contract in selected snapshot')
            groups[row['expiration']][key] = dict(K=K, T=T, bid=bid, ask=ask, mid=(bid+ask)/2)
            spots.append(S)
    if not spots:
        raise ValueError('No eligible historical SPXW quotes found')
    spot = float(np.median(spots))
    if max(spots)-min(spots) > .001*spot:
        raise ValueError('Inconsistent underlying prices within snapshot')
    eligible = sorted(e for e, quotes in groups.items() if len(quotes) >= 2*min_quotes)
    if len(eligible) < n_maturities:
        raise ValueError(f'Need {n_maturities} expiries; found {len(eligible)}')
    indices = np.rint(np.linspace(0, len(eligible)-1, n_maturities)).astype(int)
    selected = [eligible[i] for i in indices]

    # C-P = S exp(-qT) - K exp(-rT). Use paired near-ATM quotes with robust
    # spread-scaled residuals; report residuals instead of assuming exact fit.
    # Fit only q: narrow ATM strike ranges weakly identify r and q jointly.
    pairs = []
    for expiry in selected:
        quotes = groups[expiry]
        for (K, side), call in quotes.items():
            put = quotes.get((K, 'P'))
            if side == 'C' and put is not None and abs(np.log(K/spot)) <= .02:
                scale = max((call['ask']-call['bid']+put['ask']-put['bid'])/2, .1)
                pairs.append((K, call['T'], call['mid']-put['mid'], scale))
    if len(pairs) < 10 or len(set(p[1] for p in pairs)) < 2:
        raise ValueError('Insufficient paired quotes for constant-carry estimation')
    strikes, times, differences, scales = np.asarray(pairs).T
    def parity_residual(parameters):
        q = parameters[0]
        return (spot*np.exp(-q*times)-strikes*np.exp(-rate*times)-differences)/scales
    fit = least_squares(parity_residual, [.015], bounds=([-.02], [.2]),
                        loss='soft_l1', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('Constant-carry inference failed or reached parameter bounds')
    r, q = float(rate), float(fit.x[0])
    chain = []
    for expiry in selected:
        T = maturity(quote_date, expiry)
        F = spot*np.exp((r-q)*T)
        quotes = []
        for (K, side), row in sorted(groups[expiry].items()):
            k = np.log(K/F)
            if not -.35 <= k <= .30 or side != ('P' if K < F else 'C'):
                continue
            try:
                iv = implied_vol(row['mid'], spot, K, T, r, q, side == 'C')
            except ValueError:
                counts['iv_inversion_failures'] += 1
                continue
            quotes.append((K, k, iv, row['mid']))
        if len(quotes) < min_quotes:
            raise ValueError(f'{expiry}: only {len(quotes)} usable OTM quotes after IV inversion')
        K, k, iv, mid = np.asarray(quotes).T
        chain.append(dict(T=T, F=F, k=k, strikes=K, iv_mid=iv, price_mid=mid, expiration=expiry))
    metadata = dict(data='historical_cboe_sample', quote_date=quote_date, root='SPXW',
                    csv_path=str(path.resolve()), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    selection='Evenly spaced indices across eligible sorted expiries; before fitting',
                    eligible_expiries=eligible, selected_expiries=selected,
                    settings=dict(n_maturities=n_maturities, min_days=min_days, max_days=max_days,
                                  max_relative_spread=max_relative_spread, min_quotes=min_quotes,
                                  k_range=[-.35,.30], quote_time='15:45 America/New_York',
                                  expiry_time='16:00 America/New_York', day_count='ACT/365'),
                    counts=dict(counts), quotes_per_expiry=[len(s['k']) for s in chain], spot=spot,
                    rate=r, dividend_yield=q, carry_method='User-supplied constant rate assumption; robust constant q fit to near-ATM put-call parity',
                    parity_pairs=len(pairs), parity_rmse_price=float(np.sqrt(np.mean((parity_residual(fit.x)*scales)**2))),
                    iv_method='Invert OTM 15:45 bid/ask midpoint using inferred constant carry; ignore vendor IV',
                    limitations=['Provider demonstration sample, not a full historical study',
                                 'Constant carry approximation, not a fitted discount/dividend term structure'])
    return chain, metadata
