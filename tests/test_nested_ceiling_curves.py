from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.nested_ceiling_curves import build_curve_examples


class NestedCeilingCurveTest(unittest.TestCase):
    def test_quadratic_curve_recovers_anchor_slope(self) -> None:
        ceilings = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        rows = []
        for index, ceiling in enumerate(ceilings):
            rows.append(
                {
                    "trajectory": "path",
                    "family": "Diagonal",
                    "pixel_index": 4,
                    "level_index": index,
                    "target_censored_fraction": 0.05 + 0.01 * index,
                    "ceiling_K": ceiling,
                    "reference_ceiling_K": 12.0,
                    "reference_measurement_K": 11.0,
                    "latent_truth_K_evaluation_only": 11.2,
                    "posterior_mean_K": 5.0 + 0.4 * ceiling + 0.02 * ceiling**2,
                    "posterior_sd_K": 1.0 + 0.1 * ceiling,
                    "local_censored_fraction": 0.5,
                    "posterior_standardized_gap": 0.8,
                    "prior_standardized_gap": 0.3,
                    "posterior_exceedance_probability": 0.8,
                    "log_posterior_sd": 0.1,
                    "normalized_heat_flux": 0.2,
                }
            )
        result = build_curve_examples(
            pd.DataFrame(rows), anchor_fractions=(0.05,)
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["mu_slope_at_anchor"], 0.8)
        self.assertAlmostEqual(result.iloc[0]["sd_slope_at_anchor"], 0.1)
        self.assertAlmostEqual(result.iloc[0]["mu_curvature"], 0.04)
        self.assertEqual(result.iloc[0]["response_n_levels"], 5)

    def test_reference_saturated_curve_has_truth_but_no_measurement_target(self) -> None:
        rows = []
        for index, ceiling in enumerate([12.0, 11.0, 10.0, 9.0, 8.0]):
            rows.append(
                {
                    "trajectory": "path",
                    "family": "Diagonal",
                    "pixel_index": 4,
                    "level_index": index,
                    "target_censored_fraction": 0.03 + 0.01 * index,
                    "ceiling_K": ceiling,
                    "reference_ceiling_K": 12.0,
                    "reference_measurement_K": 12.0,
                    "latent_truth_K_evaluation_only": 13.0,
                    "posterior_mean_K": ceiling + 0.5,
                    "posterior_sd_K": 1.0,
                    "local_censored_fraction": 0.5,
                    "posterior_standardized_gap": 0.5,
                    "prior_standardized_gap": 0.3,
                    "posterior_exceedance_probability": 0.7,
                    "log_posterior_sd": 0.0,
                    "normalized_heat_flux": 0.2,
                }
            )
        result = build_curve_examples(
            pd.DataFrame(rows),
            anchor_fractions=(0.03,),
            include_reference_saturated=True,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result.iloc[0]["is_reference_saturated"])
        self.assertTrue(np.isnan(result.iloc[0]["measurement_residual_K"]))
        self.assertAlmostEqual(
            result.iloc[0]["truth_residual_K_evaluation_only"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
