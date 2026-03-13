"""Known-answer and boundary tests for the teaching path."""
import unittest
import numpy as np
from learn import flat_surface, black_scholes_call
from vse import dupire, svi, montecarlo, pde_pricer, market
import cn_pricer


class NumericsTests(unittest.TestCase):
    def test_flat_local_vol_at_short_interior_and_long_maturities(self):
        surface = flat_surface()
        for t in [.001, .1, .5, 1., 3.]:
            np.testing.assert_allclose(dupire.local_variance(surface, np.array([70, 100, 140]), t), .04, atol=1e-12)
        self.assertAlmostEqual(surface.forward(0), 100)

    def test_calendar_comparison_is_fixed_moneyness(self):
        # Identical smiles with different forwards have equal variance at
        # equal k; equal-strike comparisons would invent a difference.
        p = [.04, .1, -.5, 0, .2]
        clean, gap, _ = svi.check_calendar_arbitrage(p, 100, .5, p, 150, 1.)
        self.assertTrue(clean)
        self.assertEqual(gap, 0.)

    def test_invalid_local_variance_is_not_silently_floored(self):
        surface = dupire.CalibratedSurface([
            dict(T=.1, F=100, params=[.08, 0, 0, 0, .1]),
            dict(T=1., F=100, params=[.04, 0, 0, 0, .1])])
        with self.assertRaises(ValueError):
            dupire.local_variance(surface, 100, .5)

    def test_pde_matches_black_scholes_and_refines(self):
        exact = black_scholes_call(100, 100, .03, .01, .2, 1)
        errors = []
        for nodes in [201, 801]:
            price, _, _ = pde_pricer.price_down_and_out(flat_surface(), 100, 100, 1, .03, .01, 1,
                                                       N_S=nodes, N_t=400)
            errors.append(abs(price-exact))
        self.assertLess(errors[1], .01)  # 1bp of spot, not of option premium
        self.assertLess(errors[1], errors[0])

    def test_barrier_inception_and_monotonicity(self):
        surface = flat_surface()
        knocked = montecarlo.simulate_barrier_price(surface, 80, 100, 80, .03, .01, 1, n_paths=100)
        self.assertEqual(knocked, (0., 0.))
        low = pde_pricer.price_down_and_out(surface, 100, 100, 70, .03, .01, 1)[0]
        high = pde_pricer.price_down_and_out(surface, 100, 100, 95, .03, .01, 1)[0]
        self.assertGreater(low, high)
        self.assertGreaterEqual(high, 0)

    def test_bridge_price_no_higher_than_discrete(self):
        kw = dict(n_paths=10000, n_steps=30, seed=11)
        continuous, _ = montecarlo.simulate_barrier_price(flat_surface(), 100, 100, 95, .03, .01, 1, **kw)
        discrete, _ = montecarlo.simulate_barrier_price(flat_surface(), 100, 100, 95, .03, .01, 1,
                                                       continuous_monitoring=False, **kw)
        self.assertLessEqual(continuous, discrete)

    def test_default_calibration_passes_independent_checks(self):
        fitted = svi.calibrate_surface(market.build_option_chain(seed=0))
        self.assertEqual(len(fitted), 12)
        self.assertTrue(all(s['butterfly_clean'] and s['calendar_clean'] for s in fitted))
        # The generator must have finite short-time implied variance.
        t = 1e-8
        self.assertLess(svi.svi_total_variance(np.array([0.]), market._true_svi_params(t))[0]/t, .1)

    def test_flat_barrier_mc_agrees_with_pde_within_sampling_error(self):
        surface = flat_surface()
        pde = pde_pricer.price_down_and_out(surface, 100, 100, 95, .03, .01, 1, N_S=801, N_t=400)[0]
        mc, se = montecarlo.simulate_barrier_price(surface, 100, 100, 95, .03, .01, 1,
                                                  n_paths=40000, n_steps=100, seed=11)
        self.assertLess(abs(pde-mc), 4*se+.01)

    def test_cpp_accepts_strided_input_and_rejects_bad_grid(self):
        grid = np.linspace(50, 300, 101)
        lv = np.full((41, 101), .2)
        expected = cn_pricer.price_barrier_cn(grid, lv, .03, .01, 100, 1, True)
        storage = np.empty((41, 202))
        storage[:, ::2] = lv
        actual = cn_pricer.price_barrier_cn(grid, storage[:, ::2], .03, .01, 100, 1, True)
        np.testing.assert_allclose(actual, expected)
        with self.assertRaises(RuntimeError):
            cn_pricer.price_barrier_cn(np.array([50.]), np.ones((2, 1)), .03, .01, 100, 1, True)
        with self.assertRaises(RuntimeError):
            cn_pricer.price_barrier_cn(np.array([50., 60., 90.]), np.ones((2, 3)), .03, .01, 100, 1, True)


if __name__ == '__main__':
    unittest.main()
