from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_thermal_module():
    module_path = ROOT / "scripts" / "16_thermal_diffusion_kernel.py"
    spec = importlib.util.spec_from_file_location("thermal_diffusion_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_diffusion_metrics"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


thermal = load_thermal_module()


class EmpiricalCrpsTest(unittest.TestCase):
    def test_matches_unbiased_pairwise_definition(self):
        draws = np.array(
            [
                [0.0, 1.0, 5.0],
                [2.0, 1.5, 3.0],
                [4.0, 3.0, 4.0],
                [1.0, 2.5, 7.0],
            ]
        )
        truth = np.array([1.5, 2.0, 4.5])
        n_samples = len(draws)
        first_term = np.mean(np.abs(draws - truth[None, :]), axis=0)
        pairwise = np.abs(draws[:, None, :] - draws[None, :, :])
        second_term = np.sum(pairwise, axis=(0, 1)) / (
            2.0 * n_samples * (n_samples - 1)
        )

        np.testing.assert_allclose(
            thermal.empirical_crps(draws, truth),
            first_term - second_term,
            rtol=0.0,
            atol=1e-12,
        )

    def test_requires_at_least_two_draws(self):
        with self.assertRaisesRegex(ValueError, "at least two draws"):
            thermal.empirical_crps(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

    def test_rejects_mismatched_targets(self):
        with self.assertRaisesRegex(ValueError, "same targets"):
            thermal.empirical_crps(np.ones((2, 3)), np.ones(2))


class CovarianceAdjustmentTest(unittest.TestCase):
    def test_low_rank_adjustment_updates_kernel_and_diagonal(self):
        points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        features = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

        def indices(values):
            return np.rint(2.0 * np.asarray(values)[:, 0]).astype(int)

        def covariance(values1, values2):
            return features[indices(values1)].dot(features[indices(values2)].T)

        def variance(values):
            selected = features[indices(values)]
            return np.sum(selected**2, axis=1)

        base = thermal.gp2d.GP2DConfig(signal_sd=2.0, lengthscale=0.7)
        adjusted = thermal.gp2d.GP2DConfig(
            signal_sd=2.0,
            lengthscale=0.7,
            covariance_adjustment=covariance,
            variance_adjustment=variance,
        )
        base_matrix = thermal.gp2d.rbf_kernel(points, points, base)
        adjusted_matrix = thermal.gp2d.rbf_kernel(points, points, adjusted)

        np.testing.assert_allclose(
            adjusted_matrix,
            base_matrix + features.dot(features.T),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            thermal.gp2d.kernel_diagonal(points, adjusted),
            np.diag(adjusted_matrix),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(adjusted_matrix)), -1e-12)


class StationaryAdvectiveSpdeTest(unittest.TestCase):
    @staticmethod
    def path(times):
        values = np.asarray(times, dtype=float)
        return np.column_stack([0.3 * values, np.zeros_like(values)])

    def make_config(self):
        return thermal.gp2d.GP2DConfig(
            signal_sd=3.0,
            lengthscale=0.5,
            diffusivity=0.05,
            cooling_rate=1.2,
            kernel="spatiotemporal_advective_forced_heat",
            advection_path=self.path,
            forcing_lengthscale=0.5,
            forcing_quadrature_order=32,
        )

    def test_stationary_covariance_is_normalized_and_psd(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 1.0],
                [0.2, 0.4, 0.5],
                [-0.1, 0.2, 1.3],
            ]
        )
        matrix = thermal.gp2d.rbf_kernel(points, points, self.make_config())

        np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.diag(matrix), np.full(len(points), 9.0), rtol=0.0, atol=1e-12
        )
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(matrix)), -1e-12)

    def test_transport_sign_centers_correlation_at_forward_displacement(self):
        earlier = np.array([[0.0, 0.0, 0.0]])
        aligned_later = np.array([[0.3, 0.0, 1.0]])
        unshifted_later = np.array([[0.0, 0.0, 1.0]])
        config = self.make_config()

        aligned = thermal.gp2d.rbf_kernel(aligned_later, earlier, config)[0, 0]
        unshifted = thermal.gp2d.rbf_kernel(unshifted_later, earlier, config)[0, 0]
        self.assertGreater(aligned, unshifted)


if __name__ == "__main__":
    unittest.main()
