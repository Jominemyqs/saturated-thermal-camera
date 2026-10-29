import unittest

import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad

from src.native_mask_pilot import (sample_mixed_noisy_blocks, sign_transform,
                                   likelihood_layout, spread_nodes)
from src.dense_censored_gp import sample_censored_gaussian_blocks


class NativeMaskTests(unittest.TestCase):
    def sample(self, k, kinds, values, threshold=0., n=12000, seed=73):
        k = np.asarray(k, dtype=float)
        args = dict(prediction_mean=np.zeros(len(k)), observation_mean=np.zeros(len(k)),
                    observed_covariance=k, pred_observed_covariance=k,
                    prediction_variance=np.diag(k), noise_sd=.5,
                    n_samples=n, burn_in=500, thin=1, seed=seed)
        return sample_mixed_noisy_blocks(ids=np.arange(len(k)), kinds=kinds,
            numerical_values=values, thresholds=threshold, **args), args

    def test_zero_upper_exact_old_path(self):
        (new, _), args = self.sample([[1., .6], [.6, 1.]], [0, 1], [.2, np.nan], n=80)
        obs = dict(y_obs=np.array([.2, 0.]), sat_mask=np.array([False, True]), threshold=0.)
        old = sample_censored_gaussian_blocks(obs, **args, truncated_sampler='hmc')
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)

    def test_noisy_univariate_upper_and_lower(self):
        for sign in [-1, 1]:
            (p, z), _ = self.sample([[1.]], [sign], [np.nan], threshold=.4)
            density = lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)*ndtr(sign*(t-.4)/.5)
            mass = quad(density, -10, 10)[0]
            mean = quad(lambda t:t*density(t), -10, 10)[0]/mass
            variance = quad(lambda t:(t-mean)**2*density(t), -10, 10)[0]/mass
            self.assertAlmostEqual(p[0][0], mean, delta=.035)
            self.assertAlmostEqual(p[1][0]**2, variance, delta=.04)
            self.assertTrue(np.all(sign*(z[:, 0]-.4)>=-1e-7))
            # The constraint is on Z, not T: latent samples cross the ceiling.
            self.assertTrue(np.any(sign*(p[4][:, 0]-.4)<0))

    def test_correlated_mixed_against_rejection(self):
        k = np.array([[1., .65, .3], [.65, 1., .4], [.3, .4, 1.]])
        (p, _), _ = self.sample(k, [1, -1, 0], [np.nan, np.nan, .25], n=16000)
        # Independent check: condition T on the numerical third signal, then
        # rejection-sample T and independent noisy first/second signals.
        mean = k[:, 2]*.25/1.25
        cov = k-np.outer(k[:, 2], k[2])/1.25
        rng = np.random.default_rng(551)
        t = mean + np.einsum('ij,kj->ik', rng.normal(size=(500000, 3)),
                             np.linalg.cholesky(cov), optimize=False)
        z = t[:, :2]+rng.normal(0, .5, (len(t), 2))
        accepted = t[(z[:, 0]>=0)&(z[:, 1]<0)]
        self.assertGreater(len(accepted), 40000)
        np.testing.assert_allclose(p[0], accepted.mean(axis=0), atol=.04, rtol=0)
        np.testing.assert_allclose(p[1]**2, accepted.var(axis=0), atol=.045, rtol=0)

    def test_sign_transform_including_cross_covariance(self):
        m = np.array([.3, -.2]); k = np.array([[1., .6], [.6, 2.]])
        cross = np.array([[.2, .7], [.4, -.1]])
        s = np.array([1., -1.]); bounds = np.array([.5, .8])
        wm, wk, wc, wb = sign_transform(m, k, cross, s, bounds)
        d = np.diag(s)
        np.testing.assert_array_equal(wm, d@m)
        np.testing.assert_array_equal(wk, d@k@d)
        np.testing.assert_array_equal(wc, cross@d)
        np.testing.assert_array_equal(wb, d@bounds)
        z = np.array([.9, -.4])
        np.testing.assert_allclose(cross@np.linalg.solve(k, z-m),
                                   wc@np.linalg.solve(wk, d@z-wm))
        np.testing.assert_allclose(cross@np.linalg.solve(k, cross.T),
                                   wc@np.linalg.solve(wk, wc.T))

    def test_remote_upper_negligible(self):
        (p, _), _ = self.sample([[1.]], [-1], [np.nan], threshold=8.)
        self.assertAlmostEqual(p[0][0], 0., delta=.05)
        self.assertAlmostEqual(p[1][0], 1., delta=.05)

    def test_disjoint_and_no_numeric_leakage(self):
        with self.assertRaises(ValueError):
            likelihood_layout([1, 1], [0, -1], [.1, np.nan])
        with self.assertRaises(ValueError):
            likelihood_layout([1], [-1], [.1])
        with self.assertRaises(ValueError):
            likelihood_layout([1], [2], [np.nan])
        numeric, flags = likelihood_layout([1, 2, 3], [0, 1, -1], [.1, np.nan, np.nan])
        self.assertFalse(np.any(numeric & flags))

    def test_geometry_only_selection(self):
        x, y = np.meshgrid(np.linspace(0, 2, 31), np.linspace(0, 1, 11))
        xy = np.c_[x.ravel(), y.ravel()]
        eligible = xy[:, 0]>.2
        ids = spread_nodes(xy, eligible, 25)
        np.testing.assert_array_equal(ids, spread_nodes(xy*1000, eligible, 25))
        self.assertEqual(len(np.unique(ids)), 25)
        self.assertTrue(eligible[ids].all())


if __name__ == '__main__':
    unittest.main()
