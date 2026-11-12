import unittest
import numpy as np

from src.source_off_audit import remove_current_increment
from src.native_mask_pilot import sample_mixed_noisy_blocks


class SourceOffAuditTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([2, 0])
        k = np.array([[2., .3, .5], [.3, 1.5, .4], [.5, .4, 2.5]])
        m = np.array([4., 3., 5.])
        self.b = dict(prediction_mean=m, observation_mean=m[self.ids],
            observed_covariance=k[np.ix_(self.ids, self.ids)],
            pred_observed_covariance=k[:, self.ids], prediction_variance=np.diag(k))
        self.args = dict(ids=self.ids, kinds=np.array([1, 0]), numerical_values=np.array([np.nan, 4.2]),
            thresholds=4.5, noise_sd=.5, n_samples=100, burn_in=20, thin=1, seed=73)

    def test_only_means_change_without_mutation_or_refloor(self):
        original = {k: v.copy() for k, v in self.b.items()}
        off = remove_current_increment(self.b, self.ids, np.array([6., 0., 1.]))
        np.testing.assert_array_equal(off['prediction_mean'], [-2., 3., 4.])
        np.testing.assert_array_equal(off['observation_mean'], [4., -2.])
        for k in original:
            np.testing.assert_array_equal(self.b[k], original[k])
            if 'mean' not in k:
                self.assertIs(off[k], self.b[k])

    def test_zero_increment_reproduces_sampler_exactly(self):
        off = remove_current_increment(self.b, self.ids, np.zeros(3))
        a, za = sample_mixed_noisy_blocks(**self.b, **self.args)
        b, zb = sample_mixed_noisy_blocks(**off, **self.args)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(za, zb)

    def test_gaussian_observation_update_matches_analytic_mean_change(self):
        source = np.array([.6, .1, 1.2])
        off = remove_current_increment(self.b, self.ids, source)
        args = dict(self.args, kinds=np.zeros(2, int), numerical_values=np.array([4.8, 4.2]))
        a, _ = sample_mixed_noisy_blocks(**self.b, **args)
        b, _ = sample_mixed_noisy_blocks(**off, **args)
        noisy = self.b['observed_covariance'] + (.25 + 1e-7*2.5)*np.eye(2)
        change = -source+self.b['pred_observed_covariance']@np.linalg.solve(noisy, source[self.ids])
        np.testing.assert_allclose(b[0]-a[0], change, atol=1e-12)
        np.testing.assert_allclose(b[1], a[1], atol=1e-12)

    def test_invalid_input_rejected(self):
        for source in [np.zeros(2), np.array([0., np.nan, 0.])]:
            with self.assertRaises(ValueError):
                remove_current_increment(self.b, self.ids, source)
        with self.assertRaises(ValueError):
            remove_current_increment(self.b, [0, 0], np.zeros(3))


if __name__ == '__main__':
    unittest.main()
