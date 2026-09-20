from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.nested_ceiling_response import (
    RESPONSE_FEATURES,
    build_nested_response_table,
    fit_grouped_ridge_model,
    grouped_ridge_predictions,
    grouped_ridge_transfer_predictions,
)


class NestedCeilingResponseTest(unittest.TestCase):
    def test_nested_response_slopes_are_constructed_in_ceiling_order(self) -> None:
        rows = []
        for level, ceiling, mean, sd in (
            (0.05, 9.0, 9.8, 1.0),
            (0.075, 8.0, 9.2, 1.2),
            (0.10, 6.0, 8.4, 1.6),
        ):
            rows.append(
                {
                    "trajectory": "a",
                    "pixel_index": 1,
                    "target_censored_fraction": level,
                    "ceiling_K": ceiling,
                    "known_reference_measurement_K": 9.5,
                    "latent_truth_K_evaluation_only": 9.7,
                    "posterior_mean_K": mean,
                    "posterior_sd_K": sd,
                    "local_censored_fraction": 0.5,
                    "posterior_standardized_gap": 0.8,
                    "prior_standardized_gap": 0.4,
                    "posterior_exceedance_probability": 0.8,
                    "log_posterior_sd": np.log(sd),
                    "normalized_heat_flux": 0.2,
                }
            )
        result = build_nested_response_table(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["dmu_dc_near"], 0.6)
        self.assertAlmostEqual(result.iloc[0]["dmu_dc_far"], 0.4)
        self.assertAlmostEqual(result.iloc[0]["measurement_exceedance_K"], 0.5)

    def test_grouped_predictions_do_not_train_on_held_out_group(self) -> None:
        rng = np.random.default_rng(7)
        groups = np.repeat(["a", "b", "c"], 20)
        feature = rng.normal(size=(60, 2))
        target = 1.5 * feature[:, 0] - 0.4 * feature[:, 1]
        result = grouped_ridge_predictions(feature, target, groups)
        self.assertEqual(result.prediction.shape, target.shape)
        self.assertEqual(set(result.selected_penalty), {"a", "b", "c"})
        self.assertLess(np.sqrt(np.mean((result.prediction - target) ** 2)), 0.05)
        self.assertEqual(len(RESPONSE_FEATURES), 6)

    def test_group_tuned_model_can_predict_new_rows(self) -> None:
        rng = np.random.default_rng(13)
        groups = np.repeat(["a", "b", "c"], 20)
        features = rng.normal(size=(60, 2))
        target = 1.0 + 0.4 * features[:, 0] - 0.2 * features[:, 1]
        model = fit_grouped_ridge_model(features, target, groups)
        prediction = model.predict(features)
        self.assertEqual(prediction.shape, target.shape)
        self.assertLess(np.sqrt(np.mean((prediction - target) ** 2)), 0.05)

    def test_grouped_transfer_trains_only_on_moderate_rows(self) -> None:
        rng = np.random.default_rng(17)
        groups = np.repeat(["a", "b", "c"], 20)
        training = rng.normal(size=(60, 2))
        evaluation = rng.normal(size=(90, 2))
        evaluation_groups = np.repeat(["a", "b", "c"], 30)
        target = 0.7 * training[:, 0] + 0.2 * training[:, 1]
        expected = 0.7 * evaluation[:, 0] + 0.2 * evaluation[:, 1]
        result = grouped_ridge_transfer_predictions(
            training,
            target,
            groups,
            evaluation,
            evaluation_groups,
        )
        self.assertEqual(result.prediction.shape, expected.shape)
        self.assertLess(
            np.sqrt(np.mean((result.prediction - expected) ** 2)), 0.05
        )


if __name__ == "__main__":
    unittest.main()
