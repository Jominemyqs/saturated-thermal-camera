from __future__ import annotations

import unittest

import numpy as np

from src.pseudo_censoring_calibration import (
    FEATURE_NAMES,
    calibration_features,
    equal_group_weights,
    fit_scale_calibrators,
    inflate_draws,
)


class PseudoCensoringCalibrationTest(unittest.TestCase):
    def test_features_use_observable_fields_and_have_expected_shape(self) -> None:
        shape = (3, 4)
        features = calibration_features(
            posterior_mean=np.full(shape, 4.0),
            posterior_sd=np.full(shape, 2.0),
            prior_mean=np.full(shape, 3.0),
            prior_sd=np.ones(shape),
            observed_censored=np.eye(3, 4, dtype=bool),
            heat_flux=np.arange(12).reshape(shape),
            ceiling=2.0,
        )
        self.assertEqual(features.shape, (12, len(FEATURE_NAMES)))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_group_weights_give_each_group_equal_mass(self) -> None:
        groups = np.array(["a", "a", "b", "b", "b"])
        weights = equal_group_weights(groups)
        self.assertAlmostEqual(float(np.sum(weights[groups == "a"])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[groups == "b"])), 0.5)

    def test_calibrators_learn_missing_scale_and_preserve_mean(self) -> None:
        rng = np.random.default_rng(4)
        n_examples = 120
        centered = rng.normal(0.0, 1.0, size=(300, n_examples))
        target_offsets = rng.normal(0.0, 3.0, size=n_examples)
        features = np.zeros((n_examples, len(FEATURE_NAMES)))
        features[:, 0] = np.repeat([0.05, 0.10, 0.15], 40)
        groups = np.repeat(["a", "b", "c"], 40)
        calibrators = fit_scale_calibrators(
            features=features,
            centered_draws=centered,
            target_offsets=target_offsets,
            group_ids=groups,
        )
        self.assertGreater(calibrators["global inflation"].global_scale, 1.2)
        draws = np.array([[-1.0, 3.0], [1.0, 5.0]])
        mean = np.array([0.0, 4.0])
        inflated = inflate_draws(
            draws,
            mean,
            scales=np.array([2.0, 2.0]),
            apply_mask=np.array([True, False]),
        )
        np.testing.assert_allclose(np.mean(inflated, axis=0), mean)
        np.testing.assert_allclose(inflated[:, 1], draws[:, 1])


if __name__ == "__main__":
    unittest.main()
