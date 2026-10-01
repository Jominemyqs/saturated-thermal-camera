import unittest

import numpy as np

from src.projection_audit import (area_weights, lift_grid, controlled_forecasts,
                                  bridge_components, error_summary)


class ProjectionAuditTests(unittest.TestCase):
    def test_area_and_bilinear_lift(self):
        xs, ys = np.array([0., .3, 1.]), np.array([0., .8, 2.])
        x, y = np.meshgrid(xs, ys)
        z = 2 + x + 3*y + x*y
        weights = area_weights(xs, ys)
        self.assertAlmostEqual(weights.sum(), 2.)
        self.assertAlmostEqual(np.sum(weights*z), 12.)
        xx, yy = np.linspace(0, 1, 7), np.linspace(0, 2, 9)
        X, Y = np.meshgrid(xx, yy)
        np.testing.assert_allclose(lift_grid(z, xs, ys, xx, yy), 2+X+3*Y+X*Y)

    def test_no_projection_effect_and_exact_source_split(self):
        xs = ys = np.linspace(0, 1, 9)
        previous = np.ones((9, 9)) * 3
        flux = np.ones_like(previous)
        config = dict(diffusivity=.02, cooling_rate=.1, source_coupling=.3,
                      source_flux_threshold=.5)
        result = controlled_forecasts(previous, flux, previous, flux, xs, ys, 2., .1, config)
        np.testing.assert_array_equal(result['native_inputs'], result['project_both_inputs'])
        result = controlled_forecasts(previous, flux, previous*.9, flux*2, xs, ys, 2., .1, config)
        np.testing.assert_allclose(result['project_both_inputs']-result['native_inputs'],
                                   result['previous_projection_effect']+result['source_projection_effect'], atol=1e-14)

    def test_exact_bridge(self):
        x = np.arange(9.).reshape(3, 3)
        r = bridge_components(x, x+1, x-2, x-3, x+4)
        np.testing.assert_allclose(r['legacy_error'], sum(r[k] for k in r if k != 'legacy_error'))
        self.assertEqual(error_summary(r['input_projection'], np.ones_like(x))['bias_K'], 1.)


if __name__ == '__main__':
    unittest.main()
