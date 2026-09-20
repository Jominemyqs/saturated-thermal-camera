from __future__ import annotations

import numpy as np
import pandas as pd


SINGLE_CURVE_BASELINE_FEATURES = (
    "posterior_exceedance_K",
    "posterior_sd_K",
    "local_censored_fraction",
    "posterior_standardized_gap",
    "prior_standardized_gap",
    "posterior_exceedance_probability",
    "log_posterior_sd",
    "normalized_heat_flux",
    "anchor_drop_from_reference_K",
)

SMOOTH_RESPONSE_FEATURES = (
    "mu_slope_at_anchor",
    "sd_slope_at_anchor",
    "mu_curvature",
    "sd_curvature",
    "mu_total_response_K",
    "sd_total_response_K",
    "mu_integrated_response_K",
    "sd_integrated_response_K",
    "response_ceiling_span_K",
)


def _smooth_curve_features(
    ceilings: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_sd: np.ndarray,
) -> dict[str, float]:
    ceiling = np.asarray(ceilings, dtype=float)
    mean = np.asarray(posterior_mean, dtype=float)
    sd = np.asarray(posterior_sd, dtype=float)
    anchor = float(np.max(ceiling))
    span = anchor - float(np.min(ceiling))
    if len(ceiling) < 5 or span <= 0.0:
        raise ValueError("A smooth response curve requires five levels and positive span")
    coordinate = (ceiling - anchor) / span
    design = np.column_stack(
        [np.ones(len(coordinate)), coordinate, coordinate**2]
    )
    mean_coef = np.linalg.lstsq(design, mean, rcond=None)[0]
    sd_coef = np.linalg.lstsq(design, sd, rcond=None)[0]
    mean_anchor = mean_coef[0]
    sd_anchor = sd_coef[0]
    mean_low = mean_coef[0] - mean_coef[1] + mean_coef[2]
    sd_low = sd_coef[0] - sd_coef[1] + sd_coef[2]
    return {
        "mu_slope_at_anchor": float(mean_coef[1] / span),
        "sd_slope_at_anchor": float(sd_coef[1] / span),
        "mu_curvature": float(2.0 * mean_coef[2] / span**2),
        "sd_curvature": float(2.0 * sd_coef[2] / span**2),
        "mu_total_response_K": float(mean_anchor - mean_low),
        "sd_total_response_K": float(sd_anchor - sd_low),
        "mu_integrated_response_K": float(-0.5 * mean_coef[1] + mean_coef[2] / 3.0),
        "sd_integrated_response_K": float(-0.5 * sd_coef[1] + sd_coef[2] / 3.0),
        "response_ceiling_span_K": span,
    }


def build_curve_examples(
    response_rows: pd.DataFrame,
    *,
    anchor_fractions: tuple[float, ...],
    minimum_curve_levels: int = 5,
    include_reference_saturated: bool = False,
) -> pd.DataFrame:
    """Create one residual-prediction example per pixel and anchor ceiling."""
    rows = []
    anchors = set(float(value) for value in anchor_fractions)
    for (trajectory, pixel_index), pixel in response_rows.groupby(
        ["trajectory", "pixel_index"], sort=False
    ):
        pixel = pixel.sort_values("level_index")
        reference_value = float(pixel["reference_measurement_K"].iloc[0])
        reference_ceiling = float(pixel["reference_ceiling_K"].iloc[0])
        for anchor in pixel.itertuples(index=False):
            fraction = float(anchor.target_censored_fraction)
            if fraction not in anchors:
                continue
            curve = pixel[pixel["level_index"] >= int(anchor.level_index)]
            if len(curve) < minimum_curve_levels:
                continue
            reference_saturated = reference_value >= reference_ceiling - 1e-10
            observable_target = (
                not reference_saturated
                and reference_value > float(anchor.ceiling_K)
            )
            if not observable_target and not (
                include_reference_saturated and reference_saturated
            ):
                continue
            response = _smooth_curve_features(
                curve["ceiling_K"].to_numpy(),
                curve["posterior_mean_K"].to_numpy(),
                curve["posterior_sd_K"].to_numpy(),
            )
            row = {
                "trajectory": trajectory,
                "family": anchor.family,
                "pixel_index": int(pixel_index),
                "anchor_fraction": fraction,
                "anchor_level_index": int(anchor.level_index),
                "anchor_ceiling_K": float(anchor.ceiling_K),
                "reference_ceiling_K": reference_ceiling,
                "reference_measurement_K": reference_value,
                "is_reference_saturated": reference_saturated,
                "latent_truth_K_evaluation_only": float(
                    anchor.latent_truth_K_evaluation_only
                ),
                "measurement_exceedance_K": (
                    reference_value - float(anchor.ceiling_K)
                    if observable_target
                    else np.nan
                ),
                "measurement_residual_K": (
                    reference_value - float(anchor.posterior_mean_K)
                    if observable_target
                    else np.nan
                ),
                "truth_exceedance_K_evaluation_only": float(
                    anchor.latent_truth_K_evaluation_only
                )
                - float(anchor.ceiling_K),
                "truth_residual_K_evaluation_only": float(
                    anchor.latent_truth_K_evaluation_only
                )
                - float(anchor.posterior_mean_K),
                "posterior_exceedance_K": float(anchor.posterior_mean_K)
                - float(anchor.ceiling_K),
                "posterior_sd_K": float(anchor.posterior_sd_K),
                "anchor_drop_from_reference_K": reference_ceiling
                - float(anchor.ceiling_K),
                "response_n_levels": len(curve),
                **response,
            }
            for feature in SINGLE_CURVE_BASELINE_FEATURES[2:-1]:
                row[feature] = float(getattr(anchor, feature))
            rows.append(row)
    return pd.DataFrame(rows)
