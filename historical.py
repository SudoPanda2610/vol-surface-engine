"""Reproducible historical SPXW calibration and optional barrier comparison.

Print and save actual measurements, including failures. No target percentage
or price tolerance is used to choose data, optimizer inputs or contracts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from vse import svi, dupire, pde_pricer, montecarlo
from vse.historical_market import load_chain

ROOT = Path(__file__).resolve().parent


def summarize(slices):
    rows = []
    for s in slices:
        a,b,rho,m,sigma = s['params']
        rows.append(dict(expiration=s['expiration'], T=float(s['T']), quotes=len(s['k']),
                         params=s['params'].tolist(), sse=float(s['sse']),
                         iv_rmse=float(np.sqrt(np.mean((np.sqrt(np.maximum(svi.svi_total_variance(s['k'],s['params']),0)/s['T'])-s['iv_mid'])**2))),
                         butterfly_clean=s['butterfly_clean'], calendar_clean=s['calendar_clean'],
                         minimum_g=s['butterfly_min_g'], minimum_calendar_gap=s['calendar_min_diff'],
                         minimum_variance=float(a+b*sigma*np.sqrt(1-rho*rho))))
    bad = sum(not(s['butterfly_clean'] and s['calendar_clean']) for s in rows)
    return dict(slices=rows, violating_slices=bad, total_slices=len(rows),
                violation_percent=100*bad/len(rows),
                definition='A fitted expiry fails butterfly or preceding-expiry calendar diagnostic on k=[-1.5,1.5], 2400 points; first expiry has no calendar predecessor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT/'data/historical/UnderlyingOptionsEODCalcs_2023-08-25_cgi_or_historical.csv')
    parser.add_argument('--quote-date', default='2023-08-25')
    parser.add_argument('--maturities', type=int, default=12)
    parser.add_argument('--rate', type=float, required=True, help='Explicit constant continuously compounded rate assumption; not a market rate curve')
    parser.add_argument('--price', action='store_true')
    parser.add_argument('--paths', type=int, default=200000)
    parser.add_argument('--steps', type=int, default=252)
    parser.add_argument('--nodes', type=int, default=800)
    parser.add_argument('--pde-steps', type=int, default=504)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, default=ROOT/'output/historical_report.json')
    args = parser.parse_args()
    if args.paths < 2 or args.steps < 1 or args.nodes < 3 or args.pde_steps < 1:
        parser.error('Invalid pricing grid or path count')
    report = dict(run_utc=datetime.now(timezone.utc).isoformat(), status='started',
                  runtime=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__),
                  code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT/'historical.py',ROOT/'vse/historical_market.py',ROOT/'vse/svi.py',
                                         ROOT/'vse/dupire.py',ROOT/'vse/pde_pricer.py',ROOT/'vse/montecarlo.py',
                                         ROOT/'cpp/crank_nicolson.cpp']},
                  calibration_settings=dict(robust=True,loss='Unweighted squared total variance error; numerically scaled',
                                            constraints_grid=1200,diagnostics_grid=2400,k_range=[-1.5,1.5],
                                            density_margin=1e-5))
    try:
        chain, metadata = load_chain(args.csv, args.quote_date, args.maturities, rate=args.rate)
        report['market'] = metadata
        print(f"Historical quote date: {args.quote_date}; selected {len(chain)} SPXW expiries", flush=True)
        for label, constrained in [('unconstrained', False), ('constrained', True)]:
            try:
                fitted = svi.calibrate_surface(chain, arbitrage_free=constrained, robust=True)
            except (RuntimeError, ValueError) as exc:
                report[label] = dict(status='failed', error=str(exc))
                if constrained:
                    raise
                continue
            report[label] = dict(status='ok', **summarize(fitted))
            print(f"{label}: {report[label]['violating_slices']}/{len(fitted)} violating slices ({report[label]['violation_percent']:.2f}%)", flush=True)
            if constrained:
                surface = dupire.CalibratedSurface(fitted)
        report['status'] = 'calibration_complete'
        if args.price:
            S = metadata['spot']
            r,q = metadata['rate'], metadata['dividend_yield']
            K,B,T = 1.02*S,.8*S,.5
            report['pricing'] = dict(status='started', spot=S,strike=K,barrier=B,T=T,
                                     paths=args.paths,steps=args.steps,nodes=args.nodes,pde_steps=args.pde_steps,
                                     seed=args.seed,monitoring='continuous; frozen-vol bridge approximation in MC')
            pde,_,_ = pde_pricer.price_down_and_out(surface,S,K,B,r,q,T,N_S=args.nodes,N_t=args.pde_steps)
            report['pricing']['pde'] = pde
            mc,se = montecarlo.simulate_barrier_price(surface,S,K,B,r,q,T,n_paths=args.paths,n_steps=args.steps,seed=args.seed)
            low,high = float(mc-1.96*se),float(mc+1.96*se)
            report['pricing'].update(status='complete',mc=float(mc),mc_standard_error=float(se),
                                     mc_95_interval=[low,high],difference_bps_of_spot=float(abs(pde-mc)/S*10000),
                                     pde_within_mc95=bool(low<=pde<=high))
            report['status'] = 'complete'
        if report.get('unconstrained',{}).get('status')=='failed':
            report['status']='partial'
    except (ValueError, RuntimeError, OSError) as exc:
        report.update(status='failed',error=str(exc))
        if report.get('pricing',{}).get('status')=='started':
            report['pricing'].update(status='failed',error=str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['status','error','pricing']},indent=2))
    print('Saved:',args.output)
    return 1 if report['status'] in ('failed','partial') else 0


if __name__=='__main__':
    raise SystemExit(main())
