"""Historical data boundary tests with explicitly synthetic CSV fixtures."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from vse.historical_market import implied_vol, load_chain, maturity, option_price
from vse import svi, market


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'synthetic_fixture.csv'
        self.rows = []
        for expiry in ['2023-09-15', '2023-12-15', '2024-06-21']:
            T = maturity('2023-08-25', expiry)
            for K in np.linspace(97,103,31):
                for side in ['C','P']:
                    mid = option_price(100,K,T,.05,.01,.2,side=='C')
                    self.rows.append(dict(underlying_symbol='^SPX',root='SPXW',quote_date='2023-08-25',
                                          expiration=expiry,strike=K,option_type=side,
                                          bid_1545=mid*.995,ask_1545=mid*1.005,
                                          active_underlying_price_1545=100))
        self.write()

    def write(self):
        with self.path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def test_known_carry_and_iv_recovered_and_expiries_selected_before_fits(self):
        chain,meta=load_chain(self.path,n_maturities=2,rate=.05)
        self.assertEqual(meta['selected_expiries'],['2023-09-15','2024-06-21'])
        self.assertAlmostEqual(meta['dividend_yield'],.01,places=8)
        for s in chain:
            np.testing.assert_allclose(s['iv_mid'],.2,atol=1e-9)
            self.assertTrue(np.all(np.diff(s['strikes'])>0))

    def test_recent_date_and_missing_rate_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'through 2023'):
            load_chain(self.path,quote_date='2026-01-01',rate=.05)
        with self.assertRaisesRegex(ValueError,'explicit constant'):
            load_chain(self.path,n_maturities=2)

    def test_spy_cannot_be_mislabeled_spx(self):
        for row in self.rows:
            row['underlying_symbol']='SPY'
        self.write()
        with self.assertRaisesRegex(ValueError,'No eligible'):
            load_chain(self.path,n_maturities=2,rate=.05)

    def test_duplicate_contract_is_rejected(self):
        self.rows.append(self.rows[0].copy())
        self.write()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            load_chain(self.path,n_maturities=2,rate=.05)

    def test_invalid_quote_is_excluded_and_counted(self):
        self.rows[0]['bid_1545']=float('nan')
        self.rows[1]['ask_1545']=0
        self.write()
        _,meta=load_chain(self.path,n_maturities=2,rate=.05)
        self.assertEqual(meta['counts']['invalid_rows'],1)
        self.assertEqual(meta['counts']['rejected_quotes'],1)

    def test_insufficient_maturities_do_not_fall_back_to_synthetic_data(self):
        with self.assertRaisesRegex(ValueError,'Need 12 expiries'):
            load_chain(self.path,rate=.05)

    def test_price_bounds_and_roundtrip(self):
        for call in [False,True]:
            p=option_price(100,110,.7,.05,.01,.3,call)
            self.assertAlmostEqual(implied_vol(p,100,110,.7,.05,.01,call),.3,places=9)
        with self.assertRaisesRegex(ValueError,'bounds'):
            implied_vol(101,100,100,1,0,0,True)

    def test_pm_maturity_keeps_intraday_and_dst_conventions(self):
        self.assertAlmostEqual(maturity('2023-08-25','2023-08-25')*365*86400,15*60)
        # The November expiry is one UTC hour later after US daylight saving ends.
        self.assertAlmostEqual(maturity('2023-08-25','2023-11-17')*365*86400,84*86400+75*60)

    def test_short_expiry_fit_has_small_iv_error_and_positive_density(self):
        # Small raw total-variance losses can cause premature convergence;
        # assess in IV units and scan more finely than the constraint grid.
        k=np.linspace(-.2,.2,31)
        T=.02
        target=svi.svi_total_variance(k,market._true_svi_params(T))
        params,_=svi.calibrate_slice_arbitrage_free(k,target,100,T,robust=True)
        iv_error=np.sqrt(np.mean((np.sqrt(svi.svi_total_variance(k,params)/T)-np.sqrt(target/T))**2))
        self.assertLess(iv_error,.001)
        self.assertTrue(svi.check_butterfly_arbitrage(params,n=4800)[0])


if __name__=='__main__':
    unittest.main()
