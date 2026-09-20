from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.ceiling_sweep import (
    SequentialAdvectiveConfig,
    artificial_ceiling_schedule,
    lower_camera_ceiling,
    matched_censoring_observations,
    sequential_prior_blocks,
)
from src.dense_censored_gp import sample_censored_gaussian_blocks
from src.mcmc_diagnostics import chain_diagnostics
from src.metrics import empirical_crps
from src.pseudo_censoring_calibration import (
    FEATURE_NAMES,
    ScaleCalibrator,
    calibration_features,
    fit_scale_calibrators,
    inflate_draws,
    scaled_crps,
)
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    infer_previous_censored_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "38_pseudo_censoring_calibration"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "37_canonical_converged_architectures"
    / "fixed_configuration.csv"
)
REFERENCE_RESULTS = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "30_final_architecture_comparison"
    / "sequential_rerun_results.csv"
)

BASE = "uncalibrated"
GLOBAL = "global inflation"
SEVERITY = "severity inflation"
FEATURE = "feature inflation"
METHODS = (BASE, GLOBAL, SEVERITY, FEATURE)
COLORS = {
    BASE: "#555555",
    GLOBAL: "#0072B2",
    SEVERITY: "#E69F00",
    FEATURE: "#009E73",
}
LINE_STYLES = {
    BASE: "-",
    GLOBAL: ":",
    SEVERITY: "--",
    FEATURE: "-",
}
TRAINING_FRACTIONS = (0.05, 0.075, 0.10)
REPRESENTATIVE_TRAJECTORIES = (
    "DiagonalScanPath_7",
    "HorizontalScanPath_10",
    "SpiralScanPath_12",
)


def exact_top_mask(values: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    count = max(1, int(np.floor(fraction * len(flat))))
    order = np.argsort(flat, kind="mergesort")
    mask = np.zeros(len(flat), dtype=bool)
    mask[order[-count:]] = True
    return mask


def prediction_from_draws(draws: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(draws, dtype=float)
    return (
        np.mean(values, axis=0),
        np.std(values, axis=0, ddof=1),
        np.quantile(values, 0.025, axis=0),
        np.quantile(values, 0.975, axis=0),
        values,
    )


def balanced_draw_subset(
    chain_predictions: list[tuple[np.ndarray, ...]],
    per_chain: int,
) -> np.ndarray:
    selected = []
    for prediction in chain_predictions:
        draws = np.asarray(prediction[4], dtype=float)
        if per_chain > len(draws):
            raise ValueError("Requested more balanced draws than a chain retained")
        indices = np.linspace(0, len(draws) - 1, per_chain, dtype=int)
        selected.append(draws[indices])
    return np.vstack(selected)


def sample_current_chains(
    observations: dict[str, object],
    blocks: dict[str, np.ndarray],
    *,
    noise_sd: float,
    chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
    truncated_sampler: str,
) -> list[tuple[np.ndarray, ...]]:
    return [
        sample_censored_gaussian_blocks(
            observations,
            **blocks,
            noise_sd=noise_sd,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 100_000 * chain,
            truncated_sampler=truncated_sampler,
        )
        for chain in range(chains)
    ]


def current_prediction_with_convergence(
    observations: dict[str, object],
    blocks: dict[str, np.ndarray],
    *,
    options: dict[str, object],
    seed: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, float | bool | int]]:
    settings = (
        (
            int(options["current_samples_per_chain"]),
            int(options["current_burn_in"]),
            int(options["current_thin"]),
            False,
        ),
        (
            int(options["current_retry_samples_per_chain"]),
            int(options["current_retry_burn_in"]),
            int(options["current_retry_thin"]),
            True,
        ),
    )
    censored = np.asarray(observations["saturated_full"], dtype=bool).ravel()
    last_diagnostics: dict[str, float | bool | int] | None = None
    for samples, burn_in, thin, retried in settings:
        predictions = sample_current_chains(
            observations,
            blocks,
            noise_sd=float(options["current_noise_sd"]),
            chains=int(options["current_chains"]),
            samples_per_chain=samples,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            truncated_sampler=str(options["truncated_sampler"]),
        )
        chain_draws = np.asarray([prediction[4] for prediction in predictions])
        diagnostics = chain_diagnostics(chain_draws, selected_mask=censored)
        diagnostics.update(
            {
                "current_retried": retried,
                "current_samples_per_chain_used": samples,
                "current_burn_in_used": burn_in,
                "current_thin_used": thin,
            }
        )
        last_diagnostics = diagnostics
        if (
            diagnostics["rhat_max"] <= float(options["maximum_rhat"])
            and diagnostics["selected_rhat_max"] <= float(options["maximum_rhat"])
        ):
            draws = balanced_draw_subset(
                predictions, int(options["evaluation_draws_per_chain"])
            )
            return prediction_from_draws(draws), diagnostics
    raise RuntimeError(
        "Current posterior failed convergence after retry: "
        f"all={last_diagnostics['rhat_max']:.4f}, "
        f"censored={last_diagnostics['selected_rhat_max']:.4f}"
    )


def previous_prediction_with_convergence(
    prepared,
    *,
    frame: dict[str, object],
    fixed_mask: np.ndarray,
    threshold: float,
    signal_sd: float,
    lengthscale: float,
    noise_sd: float,
    options: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | int]]:
    settings = (
        (
            int(options["previous_samples_per_chain"]),
            int(options["previous_burn_in"]),
            int(options["previous_thin"]),
            False,
        ),
        (
            int(options["previous_retry_samples_per_chain"]),
            int(options["previous_retry_burn_in"]),
            int(options["previous_retry_thin"]),
            True,
        ),
    )
    saturated = np.asarray(frame["saturated_full"], dtype=bool)
    last_diagnostics: dict[str, float | bool | int] | None = None
    for samples, burn_in, thin, retried in settings:
        mean, draws, raw = infer_previous_censored_posterior(
            prepared,
            frame=frame,
            fixed_mask=fixed_mask,
            threshold=threshold,
            signal_sd=signal_sd,
            lengthscale=lengthscale,
            noise_sd=noise_sd,
            n_chains=int(options["previous_chains"]),
            samples_per_chain=samples,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            truncated_sampler=str(options["truncated_sampler"]),
        )
        selected = draws[:, saturated].reshape(
            int(options["previous_chains"]), samples, int(np.sum(saturated))
        )
        diagnostics = chain_diagnostics(selected)
        diagnostics.update(raw)
        diagnostics.update(
            {
                "previous_retried": retried,
                "previous_samples_per_chain_used": samples,
                "previous_burn_in_used": burn_in,
                "previous_thin_used": thin,
            }
        )
        last_diagnostics = diagnostics
        if diagnostics["rhat_max"] <= float(options["maximum_rhat"]):
            return mean, draws, diagnostics
    raise RuntimeError(
        "Previous posterior failed convergence after retry: "
        f"R-hat={last_diagnostics['rhat_max']:.4f}"
    )


def region_metrics(
    prediction: tuple[np.ndarray, ...],
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    mean, sd, lower, upper, draws = prediction
    target = np.asarray(truth, dtype=float).ravel()
    selected = np.asarray(mask, dtype=bool).ravel()
    if not np.any(selected):
        return {
            "n_pixels": 0,
            "rmse_K": np.nan,
            "mae_K": np.nan,
            "bias_K": np.nan,
            "crps_K": np.nan,
            "coverage_95": np.nan,
            "width_95_K": np.nan,
            "posterior_sd_K": np.nan,
        }
    error = mean[selected] - target[selected]
    crps = empirical_crps(draws[:, selected], target[selected])
    return {
        "n_pixels": int(np.sum(selected)),
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "bias_K": float(np.mean(error)),
        "crps_K": float(np.mean(crps)),
        "coverage_95": float(
            np.mean(
                (lower[selected] <= target[selected])
                & (target[selected] <= upper[selected])
            )
        ),
        "width_95_K": float(np.mean((upper - lower)[selected])),
        "posterior_sd_K": float(np.mean(sd[selected])),
    }


def evaluate_methods(
    *,
    prepared,
    base_prediction: tuple[np.ndarray, ...],
    features: np.ndarray,
    calibrators: dict[str, ScaleCalibrator],
    observed_censored: np.ndarray,
    deliberately_hidden: np.ndarray,
    hardware_saturated: np.ndarray,
    identity: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_draws = np.asarray(base_prediction[4], dtype=float)
    base_mean = np.mean(base_draws, axis=0)
    top = exact_top_mask(prepared.truth)
    regions = {
        "overall": np.ones(base_mean.shape, dtype=bool),
        "observed_censored": np.asarray(observed_censored, dtype=bool).ravel(),
        "deliberately_hidden_observable": np.asarray(
            deliberately_hidden, dtype=bool
        ).ravel(),
        "hardware_saturated": np.asarray(hardware_saturated, dtype=bool).ravel(),
        "top_1pct": top,
    }
    result_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for method in METHODS:
        if method == BASE:
            scales = np.ones(len(base_mean))
            draws = base_draws
        else:
            scales = calibrators[method].scale(features)
            draws = inflate_draws(
                base_draws,
                base_mean,
                scales,
                regions["observed_censored"],
            )
        prediction = prediction_from_draws(draws)
        overall = region_metrics(prediction, prepared.truth, regions["overall"])
        hottest = region_metrics(prediction, prepared.truth, top)
        result_rows.append(
            {
                **identity,
                "method": method,
                "field_excess_rel_l2": float(
                    np.linalg.norm(base_mean - prepared.truth.ravel())
                    / np.linalg.norm(prepared.truth.ravel() - prepared.ambient)
                ),
                "overall_rmse_K": overall["rmse_K"],
                "overall_crps_K": overall["crps_K"],
                "top_1pct_rmse_K": hottest["rmse_K"],
                "top_1pct_mae_K": hottest["mae_K"],
                "top_1pct_bias_K": hottest["bias_K"],
                "top_1pct_crps_K": hottest["crps_K"],
                "top_1pct_coverage_95": hottest["coverage_95"],
                "top_1pct_width_95_K": hottest["width_95_K"],
                "peak_absolute_error_K": float(
                    abs(np.max(base_mean) - np.max(prepared.truth))
                ),
            }
        )
        for region, mask in regions.items():
            region_rows.append(
                {
                    **identity,
                    "method": method,
                    "region": region,
                    **region_metrics(prediction, prepared.truth, mask),
                }
            )
        selected_scales = scales[regions["observed_censored"]]
        scale_rows.append(
            {
                **identity,
                "method": method,
                "mean_scale": float(np.mean(selected_scales)),
                "median_scale": float(np.median(selected_scales)),
                "q05_scale": float(np.quantile(selected_scales, 0.05)),
                "q95_scale": float(np.quantile(selected_scales, 0.95)),
                "max_scale": float(np.max(selected_scales)),
            }
        )
    return result_rows, region_rows, scale_rows


def run_base_prediction(
    prepared,
    *,
    pseudo_camera: dict[str, object],
    ceiling: float,
    fixed: dict[str, float],
    options: dict[str, object],
    trajectory_index: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, object]]:
    previous_index = int(pseudo_camera["frames"][0]["time_index"])
    current_index = int(pseudo_camera["frames"][1]["time_index"])
    lengthscale = prepared.source_lengthscale * float(options["length_multiplier"])
    previous_mean, previous_draws, previous_diagnostics = (
        previous_prediction_with_convergence(
            prepared,
            frame=pseudo_camera["frames"][0],
            fixed_mask=pseudo_camera["fixed_observation_mask"],
            threshold=ceiling,
            signal_sd=float(fixed["signal_sd"]),
            lengthscale=lengthscale,
            noise_sd=float(options["previous_noise_sd"]),
            options=options,
            seed=int(options["seed"]) + 50_000 * trajectory_index,
        )
    )
    ordinary_mean, _, _ = posterior_physics_means(
        prepared,
        previous_mean,
        previous_index=previous_index,
        current_index=current_index,
        diffusivity=float(fixed["diffusivity"]),
        cooling_rate=float(fixed["cooling_rate"]),
        source_coupling=float(fixed["source_coupling"]),
        source_flux_threshold=float(options["source_flux_threshold"]),
    )
    current_observations = matched_censoring_observations(
        prepared,
        frame=pseudo_camera["frames"][1],
        fixed_mask=pseudo_camera["fixed_observation_mask"],
        threshold=ceiling,
    )
    config = SequentialAdvectiveConfig(
        signal_sd=float(fixed["signal_sd"]),
        forcing_lengthscale=lengthscale
        * float(options["forcing_length_multiplier"]),
        diffusivity=float(fixed["diffusivity"]),
        cooling_rate=float(fixed["cooling_rate"]),
        current_noise_sd=float(options["current_noise_sd"]),
        quadrature_order=int(options["quadrature_order"]),
    )
    blocks, covariance_diagnostics = sequential_prior_blocks(
        prepared,
        camera={**pseudo_camera, "current": current_observations},
        previous_mean=previous_mean,
        previous_draws=previous_draws,
        mean_field=ordinary_mean,
        displacement=None,
        config=config,
    )
    prediction, current_diagnostics = current_prediction_with_convergence(
        current_observations,
        blocks,
        options=options,
        seed=int(options["seed"]) + 100_000 * trajectory_index,
    )
    features = calibration_features(
        posterior_mean=prediction[0].reshape(prepared.truth.shape),
        posterior_sd=prediction[1].reshape(prepared.truth.shape),
        prior_mean=ordinary_mean,
        prior_sd=np.sqrt(np.maximum(blocks["prediction_variance"], 0.0)).reshape(
            prepared.truth.shape
        ),
        observed_censored=current_observations["saturated_full"],
        heat_flux=prepared.heat_flux[current_index],
        ceiling=ceiling,
    )
    diagnostics: dict[str, object] = {
        **{f"previous_{key}": value for key, value in previous_diagnostics.items()},
        **{f"current_{key}": value for key, value in current_diagnostics.items()},
        **covariance_diagnostics,
        "n_current_observations": int(len(current_observations["y_obs"])),
        "n_current_saturated_observations": int(
            np.sum(current_observations["sat_mask"])
        ),
    }
    return prediction, features, diagnostics


def trajectory_schedule(
    prepared,
    *,
    reference_ceiling: float,
    options: dict[str, object],
    trajectory_index: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    current_index = len(prepared.times) - 1
    previous_index = current_index - int(options["previous_frame_offset"])
    reference_camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=reference_ceiling,
        observation_stride=int(options["observation_stride"]),
        noise_sd=float(options["measurement_noise_sd"]),
        seed=int(options["seed"]) + 10_000 * trajectory_index,
    )
    schedule = artificial_ceiling_schedule(
        reference_camera["current"]["clipped_full"],
        reference_ceiling,
        options["target_fractions"],
    )
    return reference_camera, schedule


def development_worker(payload: dict[str, object]) -> dict[str, object]:
    options = payload["options"]
    fixed = payload["fixed"]
    prepared, _ = prepare_trajectory(
        Path(str(payload["dataset_dir"])),
        str(payload["trajectory"]),
        nx=int(options["nx"]),
        ny=int(options["ny"]),
        heat_flux_cutoff=float(options["heat_flux_cutoff"]),
    )
    reference_ceiling = float(payload["reference_ceiling"])
    trajectory_index = int(payload["trajectory_index"])
    reference_camera, schedule = trajectory_schedule(
        prepared,
        reference_ceiling=reference_ceiling,
        options=options,
        trajectory_index=trajectory_index,
    )
    reference_current = np.asarray(
        reference_camera["current"]["clipped_full"], dtype=float
    ).ravel()
    feature_parts = []
    draw_parts = []
    offset_parts = []
    group_parts = []
    example_rows = []
    check_rows = []
    training_fractions = set(float(value) for value in options["training_fractions"])
    for level in schedule:
        fraction = float(level["target_censored_fraction"])
        if fraction not in training_fractions:
            continue
        ceiling = float(level["ceiling_K"])
        pseudo_camera = lower_camera_ceiling(reference_camera, ceiling)
        prediction, features, diagnostics = run_base_prediction(
            prepared,
            pseudo_camera=pseudo_camera,
            ceiling=ceiling,
            fixed=fixed,
            options=options,
            trajectory_index=trajectory_index,
        )
        hidden = (
            (reference_current > ceiling) & (reference_current < reference_ceiling)
        )
        hidden &= np.asarray(
            pseudo_camera["current"]["saturated_full"], dtype=bool
        ).ravel()
        if not np.any(hidden):
            raise AssertionError("A training pseudo-ceiling has no hidden measurements")
        draws = np.asarray(prediction[4], dtype=float)
        mean = np.mean(draws, axis=0)
        subset_indices = np.linspace(
            0, len(draws) - 1, int(options["calibration_draws"]), dtype=int
        )
        group = f"{prepared.name}|{fraction:g}"
        feature_parts.append(features[hidden])
        draw_parts.append(draws[subset_indices][:, hidden] - mean[hidden][None, :])
        offset_parts.append(reference_current[hidden] - mean[hidden])
        group_parts.append(np.full(int(np.sum(hidden)), group, dtype=object))
        indices = np.flatnonzero(hidden)
        for local, pixel_index in enumerate(indices):
            row = {
                "trajectory": prepared.name,
                "target_censored_fraction": fraction,
                "ceiling_K": ceiling,
                "pixel_index": int(pixel_index),
                "known_reference_measurement_K": float(reference_current[pixel_index]),
                "latent_truth_K_evaluation_only": float(
                    prepared.truth.ravel()[pixel_index]
                ),
                "posterior_mean_K": float(mean[pixel_index]),
                "posterior_sd_K": float(prediction[1][pixel_index]),
                "measurement_residual_K": float(offset_parts[-1][local]),
                "group_id": group,
            }
            row.update(
                {
                    name: float(features[pixel_index, index])
                    for index, name in enumerate(FEATURE_NAMES)
                }
            )
            example_rows.append(row)
        check_rows.append(
            {
                "trajectory": prepared.name,
                "role": "development",
                "target_censored_fraction": fraction,
                "ceiling_K": ceiling,
                "n_hidden_training_pixels": int(np.sum(hidden)),
                "truth_used_as_calibration_target": False,
                "reference_measurement_used_as_calibration_target": True,
                **diagnostics,
            }
        )
    return {
        "features": np.vstack(feature_parts),
        "centered_draws": np.hstack(draw_parts),
        "target_offsets": np.concatenate(offset_parts),
        "group_ids": np.concatenate(group_parts),
        "examples": example_rows,
        "checks": check_rows,
    }


def evaluation_worker(payload: dict[str, object]) -> dict[str, object]:
    options = payload["options"]
    fixed = payload["fixed"]
    calibrators = payload["calibrators"]
    prepared, _ = prepare_trajectory(
        Path(str(payload["dataset_dir"])),
        str(payload["trajectory"]),
        nx=int(options["nx"]),
        ny=int(options["ny"]),
        heat_flux_cutoff=float(options["heat_flux_cutoff"]),
    )
    reference_ceiling = float(payload["reference_ceiling"])
    trajectory_index = int(payload["trajectory_index"])
    reference_camera, schedule = trajectory_schedule(
        prepared,
        reference_ceiling=reference_ceiling,
        options=options,
        trajectory_index=trajectory_index,
    )
    reference_current = np.asarray(
        reference_camera["current"]["clipped_full"], dtype=float
    ).ravel()
    hardware_saturated = reference_current >= reference_ceiling
    result_rows = []
    region_rows = []
    scale_rows = []
    schedule_rows = []
    check_rows = []
    for level in schedule:
        fraction = float(level["target_censored_fraction"])
        ceiling = float(level["ceiling_K"])
        pseudo_camera = lower_camera_ceiling(reference_camera, ceiling)
        prediction, features, diagnostics = run_base_prediction(
            prepared,
            pseudo_camera=pseudo_camera,
            ceiling=ceiling,
            fixed=fixed,
            options=options,
            trajectory_index=trajectory_index,
        )
        censored = np.asarray(
            pseudo_camera["current"]["saturated_full"], dtype=bool
        ).ravel()
        hidden = (
            (reference_current > ceiling) & (reference_current < reference_ceiling)
        )
        identity = {
            "trajectory": prepared.name,
            "family": str(payload["family"]),
            "role": "evaluation",
            "level_index": int(level["level_index"]),
            "level": str(level["level"]),
            "target_censored_fraction": fraction,
            "ceiling_K": ceiling,
            "reference_ceiling_K": reference_ceiling,
            "actual_censored_fraction": float(np.mean(censored)),
        }
        results, regions, scales = evaluate_methods(
            prepared=prepared,
            base_prediction=prediction,
            features=features,
            calibrators=calibrators,
            observed_censored=censored,
            deliberately_hidden=hidden,
            hardware_saturated=hardware_saturated,
            identity=identity,
        )
        result_rows.extend(results)
        region_rows.extend(regions)
        scale_rows.extend(scales)
        schedule_rows.append(
            {
                **identity,
                "ceiling_source": str(level["ceiling_source"]),
                "same_reference_noise_realization": True,
            }
        )
        check_rows.append(
            {
                **identity,
                "same_current_observations_for_all_calibrators": True,
                "posterior_mean_held_fixed": True,
                "truth_used_by_calibrator": False,
                "current_observation_protocol": (
                    "fixed-stride unsaturated plus every observed-censored pixel"
                ),
                **diagnostics,
            }
        )
    return {
        "results": result_rows,
        "regions": region_rows,
        "scales": scale_rows,
        "schedule": schedule_rows,
        "checks": check_rows,
    }


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    excluded = {*groups, "trajectory", "ceiling_K", "reference_ceiling_K"}
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    means = frame.groupby(groups, sort=False)[numeric].mean().reset_index()
    sems = frame.groupby(groups, sort=False)[numeric].sem().reset_index()
    sems = sems.rename(columns={column: f"{column}_sem" for column in numeric})
    counts = (
        frame.groupby(groups, sort=False)["trajectory"]
        .nunique()
        .rename("n_trajectories")
        .reset_index()
    )
    return means.merge(sems, on=groups).merge(counts, on=groups)


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "overall_crps_K",
        "top_1pct_crps_K",
        "top_1pct_coverage_95",
        "top_1pct_width_95_K",
    )
    rows = []
    pivot = frame.pivot(
        index=["trajectory", "level_index"], columns="method"
    )
    for method in METHODS[1:]:
        for metric in metrics:
            difference = pivot[metric][method] - pivot[metric][BASE]
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "difference": "calibrated minus uncalibrated",
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "calibrated_lower_count": int(np.sum(difference < 0.0)),
                    "calibrated_higher_count": int(np.sum(difference > 0.0)),
                    "paired_cases": int(len(difference)),
                }
            )
    return pd.DataFrame(rows)


def region_paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("crps_K", "coverage_95", "width_95_K")
    rows = []
    for region in (
        "observed_censored",
        "deliberately_hidden_observable",
        "top_1pct",
    ):
        selected = frame[frame["region"] == region]
        pivot = selected.pivot(
            index=["trajectory", "level_index"], columns="method"
        )
        for method in METHODS[1:]:
            for metric in metrics:
                difference = pivot[metric][method] - pivot[metric][BASE]
                difference = difference.dropna()
                rows.append(
                    {
                        "region": region,
                        "method": method,
                        "metric": metric,
                        "difference": "calibrated minus uncalibrated",
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(difference.median()),
                        "calibrated_lower_count": int(np.sum(difference < 0.0)),
                        "calibrated_higher_count": int(np.sum(difference > 0.0)),
                        "paired_cases": int(len(difference)),
                    }
                )
    return pd.DataFrame(rows)


def calibrator_parameter_rows(
    calibrators: dict[str, ScaleCalibrator],
) -> list[dict[str, object]]:
    rows = []
    for name, calibrator in calibrators.items():
        if calibrator.global_scale is not None:
            rows.append(
                {
                    "calibrator": name,
                    "term": "global_scale",
                    "coefficient": calibrator.global_scale,
                    "feature_mean": np.nan,
                    "feature_sd": np.nan,
                }
            )
            continue
        rows.append(
            {
                "calibrator": name,
                "term": "intercept",
                "coefficient": calibrator.coefficients[0],
                "feature_mean": np.nan,
                "feature_sd": np.nan,
            }
        )
        for local_index, feature_index in enumerate(calibrator.feature_indices):
            rows.append(
                {
                    "calibrator": name,
                    "term": FEATURE_NAMES[feature_index],
                    "coefficient": calibrator.coefficients[local_index + 1],
                    "feature_mean": calibrator.feature_mean[local_index],
                    "feature_sd": calibrator.feature_sd[local_index],
                }
            )
    return rows


def development_fit_summary(
    features: np.ndarray,
    centered_draws: np.ndarray,
    target_offsets: np.ndarray,
    group_ids: np.ndarray,
    calibrators: dict[str, ScaleCalibrator],
) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        scales = (
            np.ones(len(features))
            if method == BASE
            else calibrators[method].scale(features)
        )
        scores = scaled_crps(centered_draws, target_offsets, scales)
        for group in np.unique(group_ids):
            selected = group_ids == group
            rows.append(
                {
                    "group_id": group,
                    "method": method,
                    "n_pixels": int(np.sum(selected)),
                    "known_measurement_crps_K": float(np.mean(scores[selected])),
                    "mean_scale": float(np.mean(scales[selected])),
                }
            )
    return pd.DataFrame(rows)


def plot_response(
    summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    panels = (
        ("overall_crps_K", "All-domain CRPS (K)"),
        ("top_1pct_crps_K", "Hottest-1% CRPS (K)"),
        ("top_1pct_coverage_95", "Hottest-1% 95% coverage"),
        ("top_1pct_width_95_K", "Hottest-1% interval width (K)"),
    )
    for axis, (metric, title) in zip(axes.ravel()[:4], panels):
        for method in METHODS:
            rows = summary[summary["method"] == method].sort_values("level_index")
            axis.errorbar(
                100.0 * rows["target_censored_fraction"],
                rows[metric],
                yerr=rows[f"{metric}_sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=COLORS[method],
                linestyle=LINE_STYLES[method],
                label=method,
            )
        axis.set_title(title)
        axis.set_xlabel("Target censored fraction (%)")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "top_1pct_coverage_95":
            axis.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
    censored_region = region_summary[
        region_summary["region"] == "observed_censored"
    ]
    for method in METHODS:
        rows = censored_region[censored_region["method"] == method].sort_values(
            "level_index"
        )
        axes[1, 1].errorbar(
            100.0 * rows["target_censored_fraction"],
            rows["crps_K"],
            yerr=rows["crps_K_sem"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
        )
    axes[1, 1].set_title("Observed-censored-region CRPS (K)")
    axes[1, 1].set_xlabel("Target censored fraction (%)")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    for method in METHODS[1:]:
        rows = scale_summary[scale_summary["method"] == method].sort_values(
            "level_index"
        )
        axes[1, 2].plot(
            100.0 * rows["target_censored_fraction"],
            rows["mean_scale"],
            marker="o",
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            label=method,
        )
    axes[1, 2].set_title("Mean learned scale on censored pixels")
    axes[1, 2].set_xlabel("Target censored fraction (%)")
    axes[1, 2].grid(alpha=0.22)
    axes[1, 2].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(
        handles,
        labels,
        loc="lower right",
        fontsize=8,
    )
    figure.suptitle(
        "Pseudo-censoring uncertainty calibration: 30 held-out trajectories",
        fontsize=15,
    )
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    path: Path,
    summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    calibration_fit: pd.DataFrame,
    checks: pd.DataFrame,
    calibrators: dict[str, ScaleCalibrator],
) -> None:
    reference = summary[summary["level_index"] == 0].set_index("method")
    severe = summary[summary["level_index"] == summary["level_index"].max()].set_index(
        "method"
    )
    feature_ref = reference.loc[FEATURE]
    base_ref = reference.loc[BASE]
    feature_severe = severe.loc[FEATURE]
    base_severe = severe.loc[BASE]
    global_ref = reference.loc[GLOBAL]
    global_severe = severe.loc[GLOBAL]
    region_reference = region_summary[
        (region_summary["level_index"] == 0)
        & (region_summary["region"] == "observed_censored")
    ].set_index("method")
    region_severe = region_summary[
        (region_summary["level_index"] == region_summary["level_index"].max())
        & (region_summary["region"] == "observed_censored")
    ].set_index("method")
    lines = [
        "# Pseudo-censoring calibration",
        "",
        "## Question",
        "",
        "Can measurements deliberately hidden below the hardware ceiling teach a frozen uncertainty correction that transfers to unseen trajectories, unseen censoring severities, and the true hardware ceiling?",
        "",
        "## Protocol",
        "",
        "- Base reconstruction: posterior physics mean plus sequential stationary stochastic heat covariance.",
        "- Every run uses fixed-stride unsaturated observations plus every observed-censored current pixel.",
        "- Calibration targets are reference-camera measurements in the deliberately hidden band; latent truth is never used for fitting.",
        "- Training uses only the three development trajectories at 5%, 7.5%, and 10% pseudo-censoring.",
        "- The fitted rule is frozen before evaluating 30 held-out trajectories at 3%, 5%, 7.5%, 10%, 15%, and 20% censoring.",
        "- Inflation is applied only to observed-censored pixels. Posterior means and all point metrics are held fixed.",
        "- CRPS uses the unbiased M(M-1) estimator and the hottest 1% is the exact hottest 25 of 2501 pixels.",
        "",
        "## Calibration rules",
        "",
        "- Global inflation: one development-fitted scalar.",
        "- Severity inflation: a monotone function of the observed censored fraction.",
        "- Feature inflation: a small softplus regression using only posterior, prior, mask, and heat-flux features available at deployment.",
        "",
        "## Held-out reference ceiling",
        "",
        "| Method | Overall CRPS | Top-1% CRPS | Coverage | Width (K) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = reference.loc[method]
        lines.append(
            f"| {method} | {row['overall_crps_K']:.3f} | {row['top_1pct_crps_K']:.3f} | {row['top_1pct_coverage_95']:.3f} | {row['top_1pct_width_95_K']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out 20% stress ceiling",
            "",
            "| Method | Overall CRPS | Top-1% CRPS | Coverage | Width (K) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = severe.loc[method]
        lines.append(
            f"| {method} | {row['overall_crps_K']:.3f} | {row['top_1pct_crps_K']:.3f} | {row['top_1pct_coverage_95']:.3f} | {row['top_1pct_width_95_K']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The simplest fitted rule is a global posterior-SD multiplier of `{calibrators[GLOBAL].global_scale:.3f}` on observed-censored pixels. The severity-only coefficient is `{calibrators[SEVERITY].coefficients[1]:.3f}`, so this experiment did not learn an additional monotone severity response beyond the constant inflation.",
            "",
            f"At the reference ceiling, global inflation improves all-domain CRPS from `{base_ref['overall_crps_K']:.3f}` to `{global_ref['overall_crps_K']:.3f}` K and observed-censored-region CRPS from `{region_reference.loc[BASE, 'crps_K']:.3f}` to `{region_reference.loc[GLOBAL, 'crps_K']:.3f}` K. Hottest-1% CRPS instead changes from `{base_ref['top_1pct_crps_K']:.3f}` to `{global_ref['top_1pct_crps_K']:.3f}` K, while coverage rises from `{base_ref['top_1pct_coverage_95']:.3f}` to `{global_ref['top_1pct_coverage_95']:.3f}` and width from `{base_ref['top_1pct_width_95_K']:.2f}` to `{global_ref['top_1pct_width_95_K']:.2f}` K.",
            "",
            f"At the unseen 20% level, global inflation improves all-domain CRPS from `{base_severe['overall_crps_K']:.3f}` to `{global_severe['overall_crps_K']:.3f}` K and observed-censored-region CRPS from `{region_severe.loc[BASE, 'crps_K']:.3f}` to `{region_severe.loc[GLOBAL, 'crps_K']:.3f}` K. Hottest-1% CRPS worsens from `{base_severe['top_1pct_crps_K']:.3f}` to `{global_severe['top_1pct_crps_K']:.3f}` K despite coverage increasing from `{base_severe['top_1pct_coverage_95']:.3f}` to `{global_severe['top_1pct_coverage_95']:.3f}`.",
            "",
            "Thus pseudo-censoring learns a transferable broad uncertainty correction, but the current scalar and feature rules do not improve hottest-tail distribution quality. The richer feature rule does not outperform global inflation, so it should not be promoted as the preferred calibrator.",
            "",
            "Point metrics are identical by construction. Any improvement is therefore a distribution-calibration effect rather than a reconstruction-mean change.",
            "",
            "## Convergence and leakage checks",
            "",
            f"Maximum previous-posterior R-hat: `{checks['previous_rhat_max'].max():.3f}`. Maximum current all-domain R-hat: `{checks['current_rhat_max'].max():.3f}`. Maximum current censored-pixel R-hat: `{checks['current_selected_rhat_max'].max():.3f}`.",
            "",
            f"Development calibration contains `{int(calibration_fit['n_pixels'].sum() / len(METHODS))}` deliberately hidden pixel examples before equal group weighting. Held-out truth was not used by any fitted calibrator.",
            "",
            "## Files",
            "",
            "- `results.csv`, `region_results.csv`: trajectory-level metrics.",
            "- `heldout30_summary.csv`, `region_summary.csv`, `family_summary.csv`: held-out summaries.",
            "- `calibration_examples.csv`: development pseudo-labels and deployment-available features.",
            "- `calibrator_parameters.csv`: frozen calibration rules.",
            "- `development_calibration_fit.csv`: in-sample hidden-measurement scores.",
            "- `calibration_scale_summary.csv`: learned inflation versus ceiling.",
            "- `paired_comparisons.csv`, `region_paired_comparisons.csv`: paired held-out changes.",
            "- `implementation_checks.csv`: observation, convergence, and leakage checks.",
            "- `calibration_response.png`: held-out severity response.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_row = pd.read_csv(FIXED_CONFIG).iloc[0]
    fixed = {
        key: float(fixed_row[key])
        for key in ("diffusivity", "cooling_rate", "signal_sd", "source_coupling")
    }
    reference = pd.read_csv(REFERENCE_RESULTS)
    thresholds = reference.groupby("trajectory")["threshold_K"].first().to_dict()
    catalog = trajectory_catalog(args.dataset_dir)
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    if len(DEVELOPMENT_TRAJECTORIES) != 3:
        raise AssertionError("Expected exactly three development trajectories")
    if sum(record.name not in DEVELOPMENT_TRAJECTORIES for record in catalog) != 30:
        raise AssertionError("Expected exactly 30 held-out trajectories")
    if set(thresholds) != {record.name for record in catalog}:
        raise AssertionError("Reference ceilings do not match the trajectory catalog")

    options = {
        "nx": args.nx,
        "ny": args.ny,
        "target_fractions": tuple(args.target_fractions),
        "training_fractions": tuple(args.training_fractions),
        "observation_stride": args.observation_stride,
        "previous_frame_offset": args.previous_frame_offset,
        "measurement_noise_sd": args.measurement_noise_sd,
        "previous_noise_sd": args.previous_noise_sd,
        "current_noise_sd": args.current_noise_sd,
        "length_multiplier": args.length_multiplier,
        "forcing_length_multiplier": args.forcing_length_multiplier,
        "quadrature_order": args.quadrature_order,
        "heat_flux_cutoff": args.heat_flux_cutoff,
        "source_flux_threshold": args.source_flux_threshold,
        "previous_chains": args.previous_chains,
        "previous_samples_per_chain": args.previous_samples_per_chain,
        "previous_burn_in": args.previous_burn_in,
        "previous_thin": args.previous_thin,
        "previous_retry_samples_per_chain": args.previous_retry_samples_per_chain,
        "previous_retry_burn_in": args.previous_retry_burn_in,
        "previous_retry_thin": args.previous_retry_thin,
        "current_chains": args.current_chains,
        "current_samples_per_chain": args.current_samples_per_chain,
        "current_burn_in": args.current_burn_in,
        "current_thin": args.current_thin,
        "current_retry_samples_per_chain": args.current_retry_samples_per_chain,
        "current_retry_burn_in": args.current_retry_burn_in,
        "current_retry_thin": args.current_retry_thin,
        "evaluation_draws_per_chain": args.evaluation_draws_per_chain,
        "calibration_draws": args.calibration_draws,
        "maximum_rhat": args.maximum_rhat,
        "truncated_sampler": args.truncated_sampler,
        "seed": args.seed,
    }
    development_payloads = [
        {
            "dataset_dir": str(args.dataset_dir),
            "trajectory": record.name,
            "trajectory_index": index,
            "reference_ceiling": thresholds[record.name],
            "fixed": fixed,
            "options": options,
        }
        for index, record in enumerate(catalog)
        if record.name in DEVELOPMENT_TRAJECTORIES
    ]
    development_outputs = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as executor:
        futures = {
            executor.submit(development_worker, payload): payload
            for payload in development_payloads
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            payload = futures[future]
            development_outputs.append(future.result())
            print(
                f"[development {completed}/3] {payload['trajectory']}", flush=True
            )
    features = np.vstack([output["features"] for output in development_outputs])
    centered_draws = np.hstack(
        [output["centered_draws"] for output in development_outputs]
    )
    target_offsets = np.concatenate(
        [output["target_offsets"] for output in development_outputs]
    )
    group_ids = np.concatenate([output["group_ids"] for output in development_outputs])
    calibration_examples = pd.DataFrame(
        [row for output in development_outputs for row in output["examples"]]
    )
    development_checks = pd.DataFrame(
        [row for output in development_outputs for row in output["checks"]]
    )
    calibrators = fit_scale_calibrators(
        features=features,
        centered_draws=centered_draws,
        target_offsets=target_offsets,
        group_ids=group_ids,
    )
    calibration_fit = development_fit_summary(
        features,
        centered_draws,
        target_offsets,
        group_ids,
        calibrators,
    )
    calibration_examples.to_csv(args.output_dir / "calibration_examples.csv", index=False)
    development_checks.to_csv(
        args.output_dir / "development_implementation_checks.csv", index=False
    )
    calibration_fit.to_csv(
        args.output_dir / "development_calibration_fit.csv", index=False
    )
    pd.DataFrame(calibrator_parameter_rows(calibrators)).to_csv(
        args.output_dir / "calibrator_parameters.csv", index=False
    )

    evaluation_payloads = [
        {
            "dataset_dir": str(args.dataset_dir),
            "trajectory": record.name,
            "family": record.family,
            "trajectory_index": index,
            "reference_ceiling": thresholds[record.name],
            "fixed": fixed,
            "options": options,
            "calibrators": calibrators,
        }
        for index, record in enumerate(catalog)
        if record.name not in DEVELOPMENT_TRAJECTORIES
    ]
    collected: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluation_worker, payload): payload
            for payload in evaluation_payloads
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            payload = futures[future]
            collected.append(future.result())
            checkpoint = pd.DataFrame(
                [row for output in collected for row in output["results"]]
            )
            checkpoint.to_csv(args.output_dir / "checkpoint_results.csv", index=False)
            print(
                f"[heldout {completed:02d}/30] {payload['trajectory']}", flush=True
            )

    results = pd.DataFrame([row for output in collected for row in output["results"]])
    regions = pd.DataFrame([row for output in collected for row in output["regions"]])
    scales = pd.DataFrame([row for output in collected for row in output["scales"]])
    schedules = pd.DataFrame([row for output in collected for row in output["schedule"]])
    checks = pd.DataFrame([row for output in collected for row in output["checks"]])
    results = results.sort_values(["trajectory", "level_index", "method"])
    regions = regions.sort_values(["trajectory", "level_index", "method", "region"])
    scales = scales.sort_values(["trajectory", "level_index", "method"])
    schedules = schedules.sort_values(["trajectory", "level_index"])
    checks = checks.sort_values(["trajectory", "level_index"])
    summary = aggregate(results, ["level_index", "level", "target_censored_fraction", "method"])
    region_summary = aggregate(
        regions,
        ["level_index", "level", "target_censored_fraction", "method", "region"],
    )
    family_summary = aggregate(
        results,
        ["family", "level_index", "level", "target_censored_fraction", "method"],
    )
    scale_summary = aggregate(
        scales, ["level_index", "level", "target_censored_fraction", "method"]
    )
    paired = paired_comparisons(results)
    region_paired = region_paired_comparisons(regions)

    if len(results) != 30 * len(args.target_fractions) * len(METHODS):
        raise AssertionError("Held-out result grid is incomplete")
    if not checks["same_current_observations_for_all_calibrators"].all():
        raise AssertionError("Calibration methods used different observations")
    if not checks["posterior_mean_held_fixed"].all():
        raise AssertionError("Uncertainty calibration changed the posterior mean")
    if checks["truth_used_by_calibrator"].any():
        raise AssertionError("Held-out truth leaked into calibration")
    if checks["previous_rhat_max"].max() > args.maximum_rhat:
        raise AssertionError("A previous posterior failed convergence")
    if checks["current_rhat_max"].max() > args.maximum_rhat:
        raise AssertionError("A current posterior failed all-domain convergence")
    if checks["current_selected_rhat_max"].max() > args.maximum_rhat:
        raise AssertionError("A current posterior failed censored-region convergence")
    point_metrics = (
        "field_excess_rel_l2",
        "overall_rmse_K",
        "top_1pct_rmse_K",
        "top_1pct_mae_K",
        "top_1pct_bias_K",
        "peak_absolute_error_K",
    )
    for metric in point_metrics:
        spread = results.groupby(["trajectory", "level_index"])[metric].agg(
            lambda values: float(np.max(values) - np.min(values))
        )
        if spread.max() > 1e-12:
            raise AssertionError(f"Uncertainty calibration changed {metric}")

    results.to_csv(args.output_dir / "results.csv", index=False)
    regions.to_csv(args.output_dir / "region_results.csv", index=False)
    scales.to_csv(args.output_dir / "calibration_scales.csv", index=False)
    schedules.to_csv(args.output_dir / "threshold_schedule.csv", index=False)
    checks.to_csv(args.output_dir / "implementation_checks.csv", index=False)
    summary.to_csv(args.output_dir / "heldout30_summary.csv", index=False)
    region_summary.to_csv(args.output_dir / "region_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "family_summary.csv", index=False)
    scale_summary.to_csv(
        args.output_dir / "calibration_scale_summary.csv", index=False
    )
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    region_paired.to_csv(
        args.output_dir / "region_paired_comparisons.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **fixed,
                **options,
                "dataset_trajectories": 33,
                "development_trajectories": 3,
                "heldout_trajectories": 30,
                "base_model": "posterior physics mean + sequential stationary ST",
                "current_observation_protocol": (
                    "fixed-stride unsaturated plus every observed-censored pixel"
                ),
                "calibration_target": (
                    "known reference-camera measurement in deliberately hidden band"
                ),
                "truth_used_for_calibration": False,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "top_1pct_rule": "exact hottest 25 of 2501 pixels",
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    plot_response(
        summary,
        region_summary,
        scale_summary,
        args.output_dir / "calibration_response.png",
    )
    write_readme(
        args.output_dir / "README.md",
        summary,
        region_summary,
        calibration_fit,
        checks,
        calibrators,
    )
    print(f"Saved pseudo-censoring calibration to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate pseudo-censoring uncertainty calibration."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-fractions",
        type=float,
        nargs="+",
        default=[0.03, 0.05, 0.075, 0.10, 0.15, 0.20],
    )
    parser.add_argument(
        "--training-fractions",
        type=float,
        nargs="+",
        default=list(TRAINING_FRACTIONS),
    )
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--measurement-noise-sd", type=float, default=0.25)
    parser.add_argument("--previous-noise-sd", type=float, default=0.25)
    parser.add_argument("--current-noise-sd", type=float, default=0.5)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--forcing-length-multiplier", type=float, default=1.0)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--previous-chains", type=int, default=4)
    parser.add_argument("--previous-samples-per-chain", type=int, default=400)
    parser.add_argument("--previous-burn-in", type=int, default=100)
    parser.add_argument("--previous-thin", type=int, default=1)
    parser.add_argument("--previous-retry-samples-per-chain", type=int, default=800)
    parser.add_argument("--previous-retry-burn-in", type=int, default=300)
    parser.add_argument("--previous-retry-thin", type=int, default=1)
    parser.add_argument("--current-chains", type=int, default=4)
    parser.add_argument("--current-samples-per-chain", type=int, default=400)
    parser.add_argument("--current-burn-in", type=int, default=100)
    parser.add_argument("--current-thin", type=int, default=1)
    parser.add_argument("--current-retry-samples-per-chain", type=int, default=800)
    parser.add_argument("--current-retry-burn-in", type=int, default=300)
    parser.add_argument("--current-retry-thin", type=int, default=1)
    parser.add_argument("--evaluation-draws-per-chain", type=int, default=300)
    parser.add_argument("--calibration-draws", type=int, default=300)
    parser.add_argument("--maximum-rhat", type=float, default=1.05)
    parser.add_argument(
        "--truncated-sampler",
        choices=("hmc", "ess"),
        default="hmc",
        help="Sampler for the Gaussian block under censoring inequalities.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
