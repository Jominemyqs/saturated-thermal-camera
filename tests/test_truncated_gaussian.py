from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import norm

from src.censored_gp import RBFConfig, sample_censored_ess_fast
from src.truncated_gaussian import sample_lower_truncated_gaussian_hmc


class TruncatedGaussianHmcTest(unittest.TestCase):
    def test_one_dimensional_moments_match_truncated_normal(self) -> None:
        lower = 0.35
        draws = sample_lower_truncated_gaussian_hmc(
            np.array([0.0]),
            np.array([[1.0]]),
            lower,
            n_samples=12000,
            burn_in=500,
            thin=1,
            seed=8,
        )[:, 0]
        inverse_mills = norm.pdf(lower) / norm.sf(lower)
        expected_mean = inverse_mills
        expected_variance = 1.0 + lower * inverse_mills - inverse_mills**2
        self.assertTrue(np.all(draws >= lower))
        self.assertAlmostEqual(float(np.mean(draws)), expected_mean, delta=0.035)
        self.assertAlmostEqual(float(np.var(draws)), expected_variance, delta=0.035)

    def test_correlated_draws_respect_every_constraint(self) -> None:
        covariance = np.array(
            [[1.0, 0.65, 0.2], [0.65, 1.2, 0.4], [0.2, 0.4, 0.8]]
        )
        lower = np.array([-0.2, 0.1, 0.5])
        draws = sample_lower_truncated_gaussian_hmc(
            np.zeros(3),
            covariance,
            lower,
            n_samples=1000,
            burn_in=100,
            thin=1,
            seed=12,
        )
        self.assertEqual(draws.shape, (1000, 3))
        self.assertTrue(np.all(draws >= lower[None, :]))
        self.assertTrue(np.all(np.std(draws, axis=0) > 0.1))

    def test_hmc_matches_elliptical_sampler_on_small_censored_gp(self) -> None:
        x_obs = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
        observations = {
            "x_obs": x_obs,
            "y_obs": np.array([0.1, 0.5, 0.5]),
            "sat_mask": np.array([False, True, True]),
            "threshold": 0.5,
            "x_pred": x_obs,
        }
        config = RBFConfig(
            mean_temp=0.0,
            signal_sd=1.0,
            lengthscale=0.6,
            noise_sd=0.15,
        )
        ess = sample_censored_ess_fast(
            observations,
            config,
            n_samples=6000,
            burn_in=1000,
            thin=1,
            seed=23,
            truncated_sampler="ess",
        )
        hmc = sample_censored_ess_fast(
            observations,
            config,
            n_samples=6000,
            burn_in=1000,
            thin=1,
            seed=29,
            truncated_sampler="hmc",
        )
        np.testing.assert_allclose(hmc[0], ess[0], atol=0.05)
        np.testing.assert_allclose(hmc[1], ess[1], atol=0.05)


if __name__ == "__main__":
    unittest.main()
