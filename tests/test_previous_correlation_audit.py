import unittest
from unittest.mock import patch
import numpy as np

from src.censored_gp import RBFConfig, sample_censored_ess_fast, rbf_covariance
from src.previous_correlation_audit import (
    normal_chains, product, propagated_variances, psd_factor,
    recover_conditional_means, residual_covariance,
)


class PreviousCorrelationAuditTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0., 0.], [.3, 0., 0.], [.7, 0., 0.]])
        self.obs = dict(x_obs=self.points, x_pred=self.points[1:],
                        y_obs=np.array([.2, 1., 1.]),
                        sat_mask=np.array([False, True, True]), threshold=1.)
        self.config = RBFConfig(mean_temp=0., signal_sd=1.2,
                                lengthscale=.4, noise_sd=.25)

    def test_recovery_matches_direct_conditioning_and_legacy_rng(self):
        zobs = np.array([[1.1, 1.2], [1.5, 1.4], [1.2, 1.7], [1.8, 1.1]])
        with patch('src.censored_gp.sample_lower_truncated_gaussian_hmc', return_value=zobs):
            old = sample_censored_ess_fast(self.obs, self.config, n_samples=4,
                                          burn_in=2, thin=1, seed=17,
                                          truncated_sampler='hmc')
        c = residual_covariance(self.obs, self.config)
        means, z = recover_conditional_means(old[4], c, seed=17, chains=1,
                                             samples=4, truncated_sampler='hmc')
        k = rbf_covariance(self.points, self.points, self.config)
        v = k + np.eye(3) * (.25**2 + self.config.relative_jitter * 1.2**2)
        full_obs = np.column_stack([np.full(4, .2), zobs])
        np.testing.assert_allclose(means, full_obs @ np.linalg.solve(v, k[:, 1:]), atol=1e-12)
        np.testing.assert_array_equal(z, np.random.default_rng(17).normal(size=(4, 2)))
        np.testing.assert_allclose(old[0], means.mean(axis=0), atol=1e-12)

    def test_non_hmc_recovery_rejected(self):
        with self.assertRaises(ValueError):
            recover_conditional_means(np.ones((4, 2)), np.eye(2), seed=1,
                                      chains=1, samples=4, truncated_sampler='ess')

    def test_identity_preserves_marginals_not_cross_covariance(self):
        c = np.array([[1., .8], [.8, 1.]])
        means = np.array([[0., 0.], [1., 2.], [2., 1.]])
        shared, ind, joint = propagated_variances(np.eye(2), means, c)
        np.testing.assert_array_equal(ind, joint)
        np.testing.assert_allclose(shared, np.diag(np.cov(means.T)))
        factor, _ = psd_factor(c)
        np.testing.assert_allclose(factor @ factor.T, c, atol=1e-14)

    def test_positive_and_negative_correlations_have_different_effects(self):
        a = np.array([[.5, .5]])
        for rho in [.8, -.8, 0.]:
            c = np.array([[1., rho], [rho, 1.]])
            _, ind, joint = propagated_variances(a, np.zeros((4, 2)), c)
            self.assertAlmostEqual(float(joint[0] - ind[0]), rho / 2)

    def test_total_covariance_matches_direct_matrix_formula(self):
        rng = np.random.default_rng(31)
        means = rng.normal(size=(30, 5))
        f = rng.normal(size=(5, 5)); c = f @ f.T
        a = rng.normal(size=(4, 5))
        shared, ind, joint = propagated_variances(a, means, c)
        np.testing.assert_allclose(shared + joint,
            np.diag(a @ (np.cov(means.T) + c) @ a.T), atol=1e-12)
        np.testing.assert_allclose(shared + ind,
            np.diag(a @ (np.cov(means.T) + np.diag(np.diag(c))) @ a.T), atol=1e-12)

    def test_mc_joint_covariance_and_pinned_embedding(self):
        c = np.array([[.5, .35], [.35, .8]])
        factor, _ = psd_factor(c)
        z = normal_chains(58, 4, 10000, 2)
        r = product(z, factor.T)
        # Four standard errors for each Gaussian sample-covariance entry.
        se = np.sqrt((c**2 + np.diag(c)[:, None]*np.diag(c)[None, :])/(len(z)-1))
        self.assertTrue(np.all(abs(np.cov(r.T)-c) < 4*se))
        full = np.full((len(z), 4), 293.)
        full[:, [1, 3]] += r
        np.testing.assert_array_equal(full[:, [0, 2]], 293.)

    def test_non_psd_rejected(self):
        with self.assertRaises(ValueError):
            psd_factor(np.array([[1., 2.], [2., 1.]]))


if __name__ == '__main__':
    unittest.main()
