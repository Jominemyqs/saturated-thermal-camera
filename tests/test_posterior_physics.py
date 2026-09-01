from __future__ import annotations

import unittest

import numpy as np

from src.censored_gp import RBFConfig, rbf_covariance, sample_censored_ess_fast
from src.dense_censored_gp import (
    censored_gaussian_mixture_component_weights,
    sample_censored_gaussian_blocks,
    sample_censored_gaussian_mixture_blocks,
)
from src.stochastic_heat_gp import (
    StochasticHeatConfig,
    finite_step_innovation_covariance,
    sequential_moment_matched_covariance,
    stochastic_heat_covariance,
)
from src.thermal_posterior_physics import (
    PreparedTrajectory,
    infer_previous_coherent_posterior,
    translate_field,
)


class TranslationConventionTest(unittest.TestCase):
    def test_positive_x_displacement_moves_peak_to_positive_x(self):
        field = np.zeros((7, 7))
        field[3, 3] = 1.0
        translated = translate_field(
            field,
            displacement=np.array([0.25, 0.0]),
            dx=0.25,
            dy=0.25,
        )
        self.assertEqual(np.unravel_index(np.argmax(translated), translated.shape), (3, 4))

    def test_positive_y_displacement_moves_peak_to_positive_y(self):
        field = np.zeros((7, 7))
        field[3, 3] = 1.0
        translated = translate_field(
            field,
            displacement=np.array([0.0, 0.25]),
            dx=0.25,
            dy=0.25,
        )
        self.assertEqual(np.unravel_index(np.argmax(translated), translated.shape), (4, 3))


class FixedResidualTest(unittest.TestCase):
    def test_mean_function_does_not_change_rbf_covariance(self):
        points = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, -0.25]])
        first = RBFConfig(
            mean_temp=0.0,
            mean_function=lambda values: np.zeros(len(values)),
            signal_sd=2.0,
            lengthscale=0.7,
            noise_sd=0.2,
        )
        second = RBFConfig(
            mean_temp=0.0,
            mean_function=lambda values: np.ones(len(values)),
            signal_sd=2.0,
            lengthscale=0.7,
            noise_sd=0.2,
        )
        np.testing.assert_array_equal(
            rbf_covariance(points, points, first),
            rbf_covariance(points, points, second),
        )


class CoherentPreviousPosteriorTest(unittest.TestCase):
    def test_unsaturated_latent_pixels_retain_posterior_uncertainty(self):
        xs = np.array([0.0, 0.5])
        ys = np.array([0.0, 0.5])
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        history = np.array([[[0.1, 0.2], [0.3, 1.4]]])
        prepared = PreparedTrajectory(
            name="toy",
            xs=xs,
            ys=ys,
            points=points,
            times=np.array([0.0]),
            history=history,
            heat_flux=np.zeros_like(history),
            truth=history[0],
            age=np.zeros((2, 2)),
            ambient=0.0,
            source_lengthscale=0.5,
            signal_scale=1.0,
        )
        frame = {
            "time_index": 0,
            "time": 0.0,
            "clipped_full": np.minimum(history[0], 1.0),
            "saturated_full": history[0] >= 1.0,
        }

        mean, draws, diagnostics = infer_previous_coherent_posterior(
            prepared,
            frame=frame,
            fixed_mask=np.ones((2, 2), dtype=bool),
            threshold=1.0,
            signal_sd=1.0,
            lengthscale=0.6,
            noise_sd=0.1,
            n_samples=24,
            burn_in=12,
            thin=1,
            seed=5,
        )

        self.assertEqual(mean.shape, (2, 2))
        self.assertEqual(draws.shape, (24, 2, 2))
        self.assertGreater(np.min(np.std(draws[:, :1, :], axis=0)), 0.0)
        self.assertGreater(diagnostics["previous_unsaturated_posterior_sd_K"], 0.0)


class SequentialStochasticHeatTest(unittest.TestCase):
    @staticmethod
    def path(times):
        values = np.asarray(times, dtype=float)
        return np.column_stack([0.3 * values, np.zeros_like(values)])

    def config(self, *, advective: bool) -> StochasticHeatConfig:
        return StochasticHeatConfig(
            signal_sd=2.0,
            forcing_lengthscale=0.45,
            diffusivity=0.04,
            cooling_rate=1.1,
            quadrature_order=32,
            advection_path=self.path if advective else None,
        )

    def test_advective_cross_covariance_has_forward_transport_sign(self):
        previous = np.array([[0.0, 0.0, 0.0]])
        aligned_current = np.array([[0.3, 0.0, 1.0]])
        stationary_current = np.array([[0.0, 0.0, 1.0]])
        config = self.config(advective=True)
        aligned = stochastic_heat_covariance(
            aligned_current, previous, config
        )[0, 0]
        stationary = stochastic_heat_covariance(
            stationary_current, previous, config
        )[0, 0]
        self.assertGreater(aligned, stationary)

    def test_total_covariance_recovers_marginal_for_prior_draws(self):
        previous = np.array(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]
        )
        current = previous.copy()
        current[:, 2] = 0.2
        config = self.config(advective=False)
        covariance = stochastic_heat_covariance(previous, previous, config)
        factor = np.linalg.cholesky(covariance + 1e-12 * np.eye(len(previous)))
        scale = np.sqrt((2 * len(previous) - 1) / 2.0)
        centered_draws = np.vstack([scale * factor.T, -scale * factor.T])
        predicted, _, _ = sequential_moment_matched_covariance(
            previous,
            current,
            centered_draws,
            config,
            relative_jitter=0.0,
        )
        marginal = stochastic_heat_covariance(current, current, config)
        np.testing.assert_allclose(predicted, marginal, rtol=1e-9, atol=2e-11)

    def test_finite_step_innovation_is_zero_at_zero_lag(self):
        points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        innovation = finite_step_innovation_covariance(
            points, points, self.config(advective=False), 0.0
        )
        np.testing.assert_allclose(innovation, 0.0, atol=1e-12)


class GaussianBlockSamplerTest(unittest.TestCase):
    def test_rbf_blocks_match_fast_sampler(self):
        points = np.array(
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0]]
        )
        config = RBFConfig(
            mean_temp=1.5,
            signal_sd=1.2,
            lengthscale=0.55,
            noise_sd=0.2,
        )
        observations = {
            "x_obs": points[:3],
            "y_obs": np.array([1.2, 1.7, 2.1]),
            "sat_mask": np.array([False, False, True]),
            "x_pred": points,
            "threshold": 2.0,
        }
        expected = sample_censored_ess_fast(
            observations,
            config,
            n_samples=30,
            burn_in=20,
            thin=1,
            seed=7,
        )
        actual = sample_censored_gaussian_blocks(
            observations,
            prediction_mean=np.full(len(points), config.mean_temp),
            observation_mean=np.full(3, config.mean_temp),
            observed_covariance=rbf_covariance(points[:3], points[:3], config),
            pred_observed_covariance=rbf_covariance(points, points[:3], config),
            prediction_variance=np.full(len(points), config.signal_sd**2),
            noise_sd=config.noise_sd,
            n_samples=30,
            burn_in=20,
            thin=1,
            seed=7,
        )
        for expected_item, actual_item in zip(expected, actual):
            np.testing.assert_allclose(actual_item, expected_item, atol=1e-12)

    def test_mixture_weights_match_uncensored_gaussian_likelihood(self):
        observations = {
            "y_obs": np.array([0.8]),
            "sat_mask": np.array([False]),
            "threshold": 2.0,
        }
        component_means = np.array([[-1.0], [1.0]])
        covariance = np.array([[0.5]])
        noise_sd = 0.5
        actual = censored_gaussian_mixture_component_weights(
            observations,
            component_observation_means=component_means,
            observed_covariance=covariance,
            noise_sd=noise_sd,
            relative_jitter=0.0,
        )
        variance = covariance[0, 0] + noise_sd**2
        log_expected = -0.5 * (observations["y_obs"][0] - component_means[:, 0]) ** 2 / variance
        expected = np.exp(log_expected - np.max(log_expected))
        expected /= np.sum(expected)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_saturation_reweights_mixture_toward_hot_component(self):
        observations = {
            "y_obs": np.array([2.0]),
            "sat_mask": np.array([True]),
            "threshold": 2.0,
        }
        actual = censored_gaussian_mixture_component_weights(
            observations,
            component_observation_means=np.array([[0.0], [3.0]]),
            observed_covariance=np.array([[0.25]]),
            noise_sd=0.25,
            relative_jitter=0.0,
        )
        self.assertGreater(actual[1], 0.99)

    def test_mixture_sampler_returns_calibrated_shapes_and_diagnostics(self):
        observations = {
            "y_obs": np.array([0.2, 1.0]),
            "sat_mask": np.array([False, True]),
            "threshold": 1.0,
        }
        component_prediction_means = np.array(
            [[0.0, 0.7, 0.4], [0.3, 1.2, 0.8]]
        )
        prediction, diagnostics = sample_censored_gaussian_mixture_blocks(
            observations,
            component_prediction_means=component_prediction_means,
            component_observation_means=component_prediction_means[:, :2],
            observed_covariance=np.array([[0.4, 0.1], [0.1, 0.5]]),
            pred_observed_covariance=np.array(
                [[0.4, 0.1], [0.1, 0.5], [0.2, 0.15]]
            ),
            prediction_variance=np.array([0.4, 0.5, 0.6]),
            noise_sd=0.2,
            n_samples=24,
            burn_in=12,
            thin=1,
            seed=9,
        )
        mean, sd, lower, upper, draws = prediction
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(sd.shape, (3,))
        self.assertEqual(lower.shape, (3,))
        self.assertEqual(upper.shape, (3,))
        self.assertEqual(draws.shape, (24, 3))
        self.assertTrue(np.all(np.isfinite(draws)))
        self.assertGreaterEqual(diagnostics["mixture_component_weight_ess"], 1.0)
        self.assertLessEqual(diagnostics["mixture_component_weight_ess"], 2.0)


if __name__ == "__main__":
    unittest.main()
