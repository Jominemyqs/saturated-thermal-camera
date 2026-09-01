from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.metrics import empirical_crps


PERCENTILE_BINS = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
    (50.0, 60.0),
    (60.0, 70.0),
    (70.0, 80.0),
    (80.0, 90.0),
    (90.0, 95.0),
    (95.0, 99.0),
    (99.0, 100.0),
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return stable within-array percentile ranks in the open interval (0, 100)."""
    flattened = np.asarray(values, dtype=float).ravel()
    order = np.argsort(flattened, kind="mergesort")
    percentiles = np.empty(len(flattened), dtype=float)
    percentiles[order] = 100.0 * (np.arange(len(flattened)) + 0.5) / len(
        flattened
    )
    return percentiles


def posterior_diagnostic_metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    draws: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    sd_tolerance: float = 1e-10,
    error_tolerance: float = 1e-10,
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=float).ravel()
    mean_values = np.asarray(mean, dtype=float).ravel()
    sd_values = np.asarray(sd, dtype=float).ravel()
    lower_values = np.asarray(lower, dtype=float).ravel()
    upper_values = np.asarray(upper, dtype=float).ravel()
    arrays = (mean_values, sd_values, lower_values, upper_values)
    if any(len(values) != len(truth_values) for values in arrays):
        raise ValueError("Posterior summaries and truth must have matching lengths")
    selected = (
        np.ones(len(truth_values), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).ravel()
    )
    if len(selected) != len(truth_values):
        raise ValueError("Diagnostic mask must match the truth length")
    if not np.any(selected):
        raise ValueError("Diagnostic mask selects no pixels")

    target = truth_values[selected]
    prediction = mean_values[selected]
    uncertainty = sd_values[selected]
    lo = lower_values[selected]
    hi = upper_values[selected]
    error = prediction - target
    valid_sd = uncertainty > sd_tolerance
    nonzero_error = np.abs(error) > error_tolerance
    standardized = (target[valid_sd] - prediction[valid_sd]) / uncertainty[valid_sd]

    metrics: dict[str, float | int] = {
        "n_pixels": int(np.sum(selected)),
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "signed_error_K": float(np.mean(error)),
        "posterior_sd_K": float(np.mean(uncertainty)),
        "coverage_95": float(np.mean((lo <= target) & (target <= hi))),
        "interval_width_95_K": float(np.mean(hi - lo)),
        "positive_sd_fraction": float(np.mean(valid_sd)),
        "zero_sd_nonzero_error_fraction": float(
            np.mean((~valid_sd) & nonzero_error)
        ),
        "mean_z": (
            float(np.mean(standardized)) if len(standardized) else np.nan
        ),
        "rms_z": (
            float(np.sqrt(np.mean(standardized**2)))
            if len(standardized)
            else np.nan
        ),
        "mean_abs_error_over_sd": (
            float(np.mean(np.abs(standardized)))
            if len(standardized)
            else np.nan
        ),
        "median_abs_error_over_sd": (
            float(np.median(np.abs(standardized)))
            if len(standardized)
            else np.nan
        ),
        "fraction_abs_z_gt_1_96": (
            float(np.mean(np.abs(standardized) > 1.96))
            if len(standardized)
            else np.nan
        ),
    }
    if draws is not None:
        samples = np.asarray(draws, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(truth_values):
            raise ValueError("Posterior draws must have shape (draw, pixel)")
        metrics["crps_K"] = float(
            np.mean(empirical_crps(samples[:, selected], target))
        )
    return metrics


def percentile_diagnostic_rows(
    truth: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    draws: np.ndarray | None = None,
    bins: Sequence[tuple[float, float]] = PERCENTILE_BINS,
) -> list[dict[str, float | int | str]]:
    truth_values = np.asarray(truth, dtype=float).ravel()
    percentiles = percentile_rank(truth_values)
    rows: list[dict[str, float | int | str]] = []
    for lower_pct, upper_pct in bins:
        selected = (percentiles >= lower_pct) & (percentiles < upper_pct)
        row: dict[str, float | int | str] = {
            "percentile_bin": f"{lower_pct:g}-{upper_pct:g}%",
            "percentile_lower": lower_pct,
            "percentile_upper": upper_pct,
        }
        row.update(
            posterior_diagnostic_metrics(
                truth_values,
                mean,
                sd,
                lower,
                upper,
                draws=draws,
                mask=selected,
            )
        )
        rows.append(row)
    return rows


def region_diagnostic_rows(
    truth: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    draws: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for region, mask in masks.items():
        row: dict[str, float | int | str] = {"region": region}
        row.update(
            posterior_diagnostic_metrics(
                truth,
                mean,
                sd,
                lower,
                upper,
                draws=draws,
                mask=mask,
            )
        )
        rows.append(row)
    return rows
