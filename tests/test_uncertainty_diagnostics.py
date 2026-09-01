import numpy as np
import unittest

from src.uncertainty_diagnostics import (
    percentile_diagnostic_rows,
    percentile_rank,
    posterior_diagnostic_metrics,
)


class UncertaintyDiagnosticTest(unittest.TestCase):
    def test_percentile_rank_is_stable_and_spans_open_interval(self):
        values = np.array([3.0, 1.0, 1.0, 2.0])

        ranks = percentile_rank(values)

        np.testing.assert_allclose(ranks, [87.5, 12.5, 37.5, 62.5])
        self.assertTrue(np.all((ranks > 0.0) & (ranks < 100.0)))

    def test_zero_sd_errors_are_reported_without_epsilon_regularization(self):
        truth = np.array([1.0, 2.0])
        mean = np.array([0.0, 2.0])
        sd = np.zeros(2)

        metrics = posterior_diagnostic_metrics(
            truth,
            mean,
            sd,
            lower=mean,
            upper=mean,
        )

        self.assertEqual(metrics["positive_sd_fraction"], 0.0)
        self.assertEqual(metrics["zero_sd_nonzero_error_fraction"], 0.5)
        self.assertTrue(np.isnan(metrics["mean_z"]))
        self.assertTrue(np.isnan(metrics["mean_abs_error_over_sd"]))
        self.assertEqual(metrics["coverage_95"], 0.5)

    def test_standardized_error_and_unbiased_crps_use_positive_sd(self):
        truth = np.array([0.0, 2.0])
        mean = np.array([1.0, 1.0])
        sd = np.array([2.0, 0.5])
        draws = np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])

        metrics = posterior_diagnostic_metrics(
            truth,
            mean,
            sd,
            lower=np.array([-3.0, 0.0]),
            upper=np.array([5.0, 2.0]),
            draws=draws,
        )

        np.testing.assert_allclose(metrics["mean_z"], 0.75)
        np.testing.assert_allclose(metrics["mean_abs_error_over_sd"], 1.25)
        self.assertTrue(np.isfinite(metrics["crps_K"]))
        self.assertEqual(metrics["coverage_95"], 1.0)

    def test_percentile_rows_partition_every_pixel_once(self):
        truth = np.arange(100.0)
        rows = percentile_diagnostic_rows(
            truth,
            mean=truth,
            sd=np.ones(100),
            lower=truth - 2.0,
            upper=truth + 2.0,
        )

        self.assertEqual(sum(row["n_pixels"] for row in rows), 100)
        self.assertEqual(rows[-1]["percentile_bin"], "99-100%")
        self.assertEqual(rows[-1]["n_pixels"], 1)

    def test_diagnostic_rejects_mismatched_shapes_and_empty_mask(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            posterior_diagnostic_metrics(
                np.ones(2), np.ones(3), np.ones(2), np.ones(2), np.ones(2)
            )
        with self.assertRaisesRegex(ValueError, "selects no pixels"):
            posterior_diagnostic_metrics(
                np.ones(2),
                np.ones(2),
                np.ones(2),
                np.ones(2),
                np.ones(2),
                mask=np.zeros(2, dtype=bool),
            )
