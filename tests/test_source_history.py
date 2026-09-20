from types import SimpleNamespace
import unittest
import numpy as np
from src.source_history import source_history_mean
from src.thermal_posterior_physics import posterior_physics_means


class SourceHistoryTest(unittest.TestCase):
    def setUp(self):
        self.prepared = SimpleNamespace(times=np.array([0., .2, .7]),
            ambient=290., xs=np.arange(3.), ys=np.arange(3.),
            heat_flux=np.ones((3,3,3))*2.)
        self.options = dict(diffusivity=0., cooling_rate=.3,
            source_coupling=.5, source_flux_threshold=1.)

    def test_nonuniform_times_source_not_double_counted(self):
        result = source_history_mean(self.prepared, np.ones((3,3))*293.,
            start_index=0, current_index=2, **self.options)
        expected = 290. + (3.*np.exp(-.3*.2)+.2)*np.exp(-.3*.5)+.5
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_one_step_exactly_matches_inherited_forecast(self):
        initial = np.arange(9.).reshape(3,3)+287.
        result = source_history_mean(self.prepared, initial,
            start_index=1, current_index=2, **self.options)
        baseline, _, _ = posterior_physics_means(self.prepared, initial,
            previous_index=1, current_index=2, **self.options)
        np.testing.assert_array_equal(result, baseline)

    def test_no_source_returns_cooled_initial_excess(self):
        self.prepared.heat_flux[:] = 0.
        result = source_history_mean(self.prepared, np.ones((3,3))*293.,
            start_index=0, current_index=2, **self.options)
        np.testing.assert_allclose(result, 290.+3.*np.exp(-.3*.7))


if __name__ == "__main__":
    unittest.main()
