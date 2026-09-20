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
    sequential_advective_prior_blocks,
)
from src.censored_gp import RBFConfig, sample_censored_ess_fast
from src.dense_censored_gp import (
    pool_gaussian_predictions,
    sample_censored_gaussian_blocks,
)
from src.metrics import empirical_crps
from src.thermal_plotting import add_tail_contours, tail_temperature_norm
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    infer_previous_censored_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prepare_trajectory,
    previous_posterior_observations,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "by_experiment" / "35_lowered_ceiling_sweep"
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "32_uncertainty_oracle_v2"
    / "fixed_configuration.csv"
)
REFERENCE_RESULTS = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "30_final_architecture_comparison"
    / "sequential_rerun_results.csv"
)

SINGLE_FRAME = "single-frame censored RBF"
SEQUENTIAL = "advective posterior physics mean + sequential advective ST"
METHODS = (SINGLE_FRAME, SEQUENTIAL)
COLORS = {SINGLE_FRAME: "#D55E00", SEQUENTIAL: "#0072B2"}
SHORT_LABELS = {SINGLE_FRAME: "Single-frame RBF", SEQUENTIAL: "Sequential physics GP"}
PERCENTILE_EDGES = np.array(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0, 100.0]
)
REPRESENTATIVES = {
    "DiagonalScanPath_7",
    "HorizontalScanPath_10",
    "SpiralScanPath_12",
}


def pooled_fast_rbf_prediction(
    observations: dict[str, object],
    config: RBFConfig,
    *,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = [
        sample_censored_ess_fast(
            observations,
            config,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 100_000 * chain,
        )
        for chain in range(n_chains)
    ]
    return pool_gaussian_predictions(predictions)


def pooled_block_prediction(
    observations: dict[str, object],
    blocks: dict[str, np.ndarray],
    *,
    noise_sd: float,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = [
        sample_censored_gaussian_blocks(
            observations,
            **blocks,
            noise_sd=noise_sd,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 100_000 * chain,
        )
        for chain in range(n_chains)
    ]
    return pool_gaussian_predictions(predictions)


def percentile_masks(truth: np.ndarray) -> list[tuple[str, float, float, np.ndarray]]:
    values = np.asarray(truth, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) * 100.0 / len(values)
    masks = []
    for lower, upper in zip(PERCENTILE_EDGES[:-1], PERCENTILE_EDGES[1:]):
        mask = (ranks >= lower) & (ranks < upper)
        masks.append((f"{lower:g}-{upper:g}%", lower, upper, mask))
    return masks


def region_metrics(
    prediction: tuple[np.ndarray, ...],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    mean, sd, lower, upper, draws = prediction
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(selected):
        return {
            "n_pixels": 0,
            "rmse_K": np.nan,
            "mae_K": np.nan,
            "signed_error_K": np.nan,
            "crps_K": np.nan,
            "posterior_sd_K": np.nan,
            "coverage_95": np.nan,
            "interval_width_95_K": np.nan,
            "mean_abs_error_over_sd": np.nan,
        }
    truth = np.asarray(target, dtype=float).reshape(-1)
    error = mean[selected] - truth[selected]
    selected_sd = sd[selected]
    standardized = np.divide(
        np.abs(error),
        selected_sd,
        out=np.full_like(error, np.nan),
        where=selected_sd > 1e-12,
    )
    crps = empirical_crps(draws[:, selected], truth[selected])
    return {
        "n_pixels": int(np.sum(selected)),
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "signed_error_K": float(np.mean(error)),
        "crps_K": float(np.mean(crps)),
        "posterior_sd_K": float(np.mean(selected_sd)),
        "coverage_95": float(
            np.mean((lower[selected] <= truth[selected]) & (truth[selected] <= upper[selected]))
        ),
        "interval_width_95_K": float(np.mean((upper - lower)[selected])),
        "mean_abs_error_over_sd": float(np.nanmean(standardized)),
    }


def summarize_prediction(
    prepared,
    *,
    prediction: tuple[np.ndarray, ...],
    model: str,
    identity: dict[str, object],
    pseudo_camera: dict[str, object],
    reference_current: np.ndarray,
    reference_ceiling: float,
    inference_observations: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    truth = np.asarray(prepared.truth, dtype=float).ravel()
    mean = np.asarray(prediction[0], dtype=float).ravel()
    ceiling = float(pseudo_camera["current"]["threshold"])
    pseudo_values = np.asarray(pseudo_camera["current"]["clipped_full"], dtype=float).ravel()
    censored = np.asarray(pseudo_camera["current"]["saturated_full"], dtype=bool).ravel()
    hidden = (
        (np.asarray(reference_current, dtype=float).ravel() > ceiling)
        & (np.asarray(reference_current, dtype=float).ravel() < reference_ceiling)
    )
    top_one = percentile_masks(truth)[-1][3]
    near_ceiling = (~censored) & (pseudo_values >= ceiling - 1.0)
    region_definitions = {
        "overall": np.ones(len(truth), dtype=bool),
        "observed_censored_full_grid": censored,
        "unsaturated_full_grid": ~censored,
        "near_ceiling_unsaturated_1K": near_ceiling,
        "deliberately_hidden_observable": hidden,
        "top_1pct": top_one,
    }
    region_rows = [
        {
            **identity,
            "model": model,
            "region": region,
            **region_metrics(prediction, truth, mask),
        }
        for region, mask in region_definitions.items()
    ]
    percentile_rows = []
    for label, lower_pct, upper_pct, mask in percentile_masks(truth):
        percentile_rows.append(
            {
                **identity,
                "model": model,
                "percentile_bin": label,
                "percentile_lower": lower_pct,
                "percentile_upper": upper_pct,
                "percentile_midpoint": 0.5 * (lower_pct + upper_pct),
                "true_temperature_min_K": float(np.min(truth[mask])),
                "true_temperature_max_K": float(np.max(truth[mask])),
                **region_metrics(prediction, truth, mask),
            }
        )

    positive = censored & (truth > ceiling)
    if np.sum(positive) >= 2 and np.ptp(truth[positive] - ceiling) > 0.0:
        slope, intercept = np.polyfit(
            truth[positive] - ceiling, mean[positive] - ceiling, 1
        )
        correlation = float(
            pd.Series(truth[positive] - ceiling).corr(
                pd.Series(mean[positive] - ceiling), method="spearman"
            )
        )
    else:
        slope, intercept, correlation = np.nan, np.nan, np.nan
    hidden_observation_metrics = region_metrics(
        prediction,
        np.asarray(reference_current, dtype=float).ravel(),
        hidden,
    )
    overall = region_metrics(prediction, truth, np.ones(len(truth), dtype=bool))
    saturated_metrics = region_metrics(prediction, truth, censored)
    top_metrics = region_metrics(prediction, truth, top_one)
    excess_denominator = np.linalg.norm(truth - float(prepared.ambient))
    result = {
        **identity,
        "model": model,
        "n_prediction_pixels": len(truth),
        "n_current_inference_observations": len(
            np.asarray(inference_observations["y_obs"])
        ),
        "n_current_inference_censored": int(
            np.sum(np.asarray(inference_observations["sat_mask"], dtype=bool))
        ),
        "current_inference_censored_fraction": float(
            np.mean(np.asarray(inference_observations["sat_mask"], dtype=bool))
        ),
        "field_excess_rel_l2": float(np.linalg.norm(mean - truth) / excess_denominator),
        "overall_rmse_K": overall["rmse_K"],
        "overall_crps_K": overall["crps_K"],
        "censored_rmse_K": saturated_metrics["rmse_K"],
        "censored_mae_K": saturated_metrics["mae_K"],
        "censored_bias_K": saturated_metrics["signed_error_K"],
        "censored_crps_K": saturated_metrics["crps_K"],
        "censored_coverage_95": saturated_metrics["coverage_95"],
        "censored_interval_width_95_K": saturated_metrics["interval_width_95_K"],
        "top_1pct_rmse_K": top_metrics["rmse_K"],
        "top_1pct_crps_K": top_metrics["crps_K"],
        "top_1pct_bias_K": top_metrics["signed_error_K"],
        "top_1pct_coverage_95": top_metrics["coverage_95"],
        "top_1pct_interval_width_95_K": top_metrics["interval_width_95_K"],
        "peak_absolute_error_K": float(abs(np.max(mean) - np.max(truth))),
        "n_true_above_ceiling_within_censored": int(np.sum(positive)),
        "exceedance_calibration_slope": float(slope),
        "exceedance_calibration_intercept_K": float(intercept),
        "exceedance_spearman": correlation,
        "mean_true_exceedance_K": float(np.mean(truth[positive] - ceiling))
        if np.any(positive)
        else np.nan,
        "mean_predicted_exceedance_K": float(np.mean(mean[positive] - ceiling))
        if np.any(positive)
        else np.nan,
        "n_deliberately_hidden_observations": int(np.sum(hidden)),
        "hidden_observation_rmse_K": hidden_observation_metrics["rmse_K"],
        "hidden_observation_mae_K": hidden_observation_metrics["mae_K"],
        "hidden_observation_bias_K": hidden_observation_metrics["signed_error_K"],
        "hidden_observation_crps_K": hidden_observation_metrics["crps_K"],
    }
    return result, region_rows, percentile_rows


def worker_run_trajectory(payload: dict[str, object]) -> dict[str, object]:
    dataset_dir = Path(str(payload["dataset_dir"]))
    record_name = str(payload["record_name"])
    family = str(payload["family"])
    role = str(payload["role"])
    catalog_index = int(payload["catalog_index"])
    fixed = payload["fixed"]
    options = payload["options"]
    prepared, _ = prepare_trajectory(
        dataset_dir,
        record_name,
        nx=int(options["nx"]),
        ny=int(options["ny"]),
        heat_flux_cutoff=float(options["heat_flux_cutoff"]),
    )
    current_index = len(prepared.times) - 1
    previous_index = current_index - int(options["previous_frame_offset"])
    reference_ceiling = float(payload["reference_ceiling"])
    camera_seed = int(options["seed"]) + 10_000 * catalog_index
    reference_camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=reference_ceiling,
        observation_stride=int(options["observation_stride"]),
        noise_sd=float(options["measurement_noise_sd"]),
        seed=camera_seed,
    )
    reference_current = np.asarray(
        reference_camera["frames"][1]["clipped_full"], dtype=float
    )
    schedule = artificial_ceiling_schedule(
        reference_current,
        reference_ceiling,
        options["target_fractions"],
    )
    lengthscale = prepared.source_lengthscale * float(options["length_multiplier"])
    single_config = RBFConfig(
        mean_temp=prepared.ambient,
        signal_sd=float(fixed["signal_sd"]),
        lengthscale=lengthscale,
        noise_sd=float(options["current_noise_sd"]),
    )
    sequential_config = SequentialAdvectiveConfig(
        signal_sd=float(fixed["signal_sd"]),
        forcing_lengthscale=lengthscale * float(options["forcing_length_multiplier"]),
        diffusivity=float(fixed["diffusivity"]),
        cooling_rate=float(fixed["cooling_rate"]),
        current_noise_sd=float(options["current_noise_sd"]),
        quadrature_order=int(options["quadrature_order"]),
    )

    result_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    percentile_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    representative: dict[str, dict[str, np.ndarray | float | str]] = {}
    for level in schedule:
        level_index = int(level["level_index"])
        ceiling = float(level["ceiling_K"])
        pseudo_camera = lower_camera_ceiling(reference_camera, ceiling)
        if not np.array_equal(
            pseudo_camera["current"]["clipped_full"],
            np.minimum(reference_current, ceiling),
        ):
            raise AssertionError("Pseudo-camera was not constructed solely by reclipping")
        current_censored = np.asarray(
            pseudo_camera["frames"][1]["saturated_full"], dtype=bool
        )
        previous_censored = np.asarray(
            pseudo_camera["frames"][0]["saturated_full"], dtype=bool
        )
        identity = {
            "trajectory": record_name,
            "family": family,
            "role": role,
            "level_index": level_index,
            "level": level["level"],
            "target_censored_fraction": float(level["target_censored_fraction"]),
            "ceiling_K": ceiling,
            "reference_ceiling_K": reference_ceiling,
            "deliberately_hidden_range_K": reference_ceiling - ceiling,
            "true_peak_hidden_exceedance_K": float(np.max(prepared.truth) - ceiling),
            "relative_true_hidden_range": float(
                (np.max(prepared.truth) - ceiling)
                / (np.max(prepared.truth) - np.min(prepared.truth))
            ),
            "current_full_grid_censored_fraction": float(np.mean(current_censored)),
            "previous_full_grid_censored_fraction": float(np.mean(previous_censored)),
        }
        schedule_rows.append(
            {
                **identity,
                "ceiling_source": level["ceiling_source"],
                "camera_seed": camera_seed,
                "same_reference_noise_realization": True,
            }
        )

        single_observations, _, _ = previous_posterior_observations(
            prepared,
            frame=pseudo_camera["frames"][1],
            fixed_mask=pseudo_camera["fixed_observation_mask"],
            threshold=ceiling,
        )
        single_observations["x_pred"] = np.asarray(
            pseudo_camera["current"]["x_pred"], dtype=float
        )
        single_prediction = pooled_fast_rbf_prediction(
            single_observations,
            single_config,
            n_chains=int(options["current_chains"]),
            samples_per_chain=int(options["current_samples_per_chain"]),
            burn_in=int(options["current_burn_in"]),
            thin=int(options["thin"]),
            seed=int(options["seed"]) + 75_000 * catalog_index,
        )
        single_result, single_regions, single_percentiles = summarize_prediction(
            prepared,
            prediction=single_prediction,
            model=SINGLE_FRAME,
            identity=identity,
            pseudo_camera=pseudo_camera,
            reference_current=reference_current,
            reference_ceiling=reference_ceiling,
            inference_observations=single_observations,
        )
        result_rows.append(single_result)
        region_rows.extend(single_regions)
        percentile_rows.extend(single_percentiles)

        previous_mean, previous_draws, previous_diagnostics = infer_previous_censored_posterior(
            prepared,
            frame=pseudo_camera["frames"][0],
            fixed_mask=pseudo_camera["fixed_observation_mask"],
            threshold=ceiling,
            signal_sd=float(fixed["signal_sd"]),
            lengthscale=lengthscale,
            noise_sd=float(options["previous_noise_sd"]),
            n_chains=int(options["previous_chains"]),
            samples_per_chain=int(options["previous_samples_per_chain"]),
            burn_in=int(options["previous_burn_in"]),
            thin=int(options["thin"]),
            seed=int(options["seed"]) + 50_000 * catalog_index,
        )
        _, advective_mean, displacement = posterior_physics_means(
            prepared,
            previous_mean,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed["diffusivity"]),
            cooling_rate=float(fixed["cooling_rate"]),
            source_coupling=float(fixed["source_coupling"]),
            source_flux_threshold=float(options["source_flux_threshold"]),
        )
        blocks, covariance_diagnostics = sequential_advective_prior_blocks(
            prepared,
            camera=pseudo_camera,
            previous_mean=previous_mean,
            previous_draws=previous_draws,
            advective_mean=advective_mean,
            displacement=displacement,
            config=sequential_config,
        )
        sequential_prediction = pooled_block_prediction(
            pseudo_camera["current"],
            blocks,
            noise_sd=float(options["current_noise_sd"]),
            n_chains=int(options["current_chains"]),
            samples_per_chain=int(options["current_samples_per_chain"]),
            burn_in=int(options["current_burn_in"]),
            thin=int(options["thin"]),
            seed=int(options["seed"]) + 100_000 * catalog_index,
        )
        sequential_result, sequential_regions, sequential_percentiles = summarize_prediction(
            prepared,
            prediction=sequential_prediction,
            model=SEQUENTIAL,
            identity=identity,
            pseudo_camera=pseudo_camera,
            reference_current=reference_current,
            reference_ceiling=reference_ceiling,
            inference_observations=pseudo_camera["current"],
        )
        sequential_result.update(previous_diagnostics)
        sequential_result.update(covariance_diagnostics)
        result_rows.append(sequential_result)
        region_rows.extend(sequential_regions)
        percentile_rows.extend(sequential_percentiles)

        if record_name in REPRESENTATIVES and level_index in (0, 3, 5):
            representative[str(level_index)] = {
                "level": str(level["level"]),
                "ceiling_K": ceiling,
                "truth": np.asarray(prepared.truth, dtype=float),
                "camera": np.asarray(pseudo_camera["current"]["clipped_full"], dtype=float),
                "single": single_prediction[0].reshape(prepared.truth.shape),
                "sequential": sequential_prediction[0].reshape(prepared.truth.shape),
                "xs": np.asarray(prepared.xs, dtype=float),
                "ys": np.asarray(prepared.ys, dtype=float),
                "ambient": float(prepared.ambient),
                "reference_ceiling": reference_ceiling,
            }
    return {
        "results": result_rows,
        "regions": region_rows,
        "percentiles": percentile_rows,
        "schedule": schedule_rows,
        "representative": representative,
    }


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in set(groups)
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


def paired_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "field_excess_rel_l2",
        "overall_rmse_K",
        "overall_crps_K",
        "top_1pct_rmse_K",
        "top_1pct_crps_K",
        "top_1pct_coverage_95",
        "top_1pct_interval_width_95_K",
        "hidden_observation_rmse_K",
    )
    rows = []
    for (level_index, level), selected in frame.groupby(
        ["level_index", "level"], sort=False
    ):
        for metric in metrics:
            pivot = selected.pivot(index="trajectory", columns="model", values=metric).dropna()
            difference = pivot[SEQUENTIAL] - pivot[SINGLE_FRAME]
            if metric == "top_1pct_coverage_95":
                wins = (
                    np.abs(pivot[SEQUENTIAL] - 0.95)
                    < np.abs(pivot[SINGLE_FRAME] - 0.95)
                )
                win_definition = "closer to nominal 0.95"
            elif metric == "top_1pct_interval_width_95_K":
                wins = np.full(len(difference), False)
                win_definition = "not ranked; interpret jointly with coverage"
            else:
                wins = difference < 0.0
                win_definition = "lower is better"
            rows.append(
                {
                    "level_index": level_index,
                    "level": level,
                    "metric": metric,
                    "difference_definition": "sequential minus single-frame",
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "sequential_win_count": int(np.sum(wins))
                    if metric != "top_1pct_interval_width_95_K"
                    else np.nan,
                    "paired_trajectory_count": len(difference),
                    "win_definition": win_definition,
                }
            )
    return pd.DataFrame(rows)


def plot_metric_panel(
    axis,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for model in METHODS:
        selected = summary[summary["model"] == model].sort_values("level_index")
        x = 100.0 * selected["current_full_grid_censored_fraction"].to_numpy(float)
        y = selected[metric].to_numpy(float)
        error_column = f"{metric}_sem"
        yerr = selected[error_column].to_numpy(float) if error_column in selected else None
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=COLORS[model],
            label=SHORT_LABELS[model],
        )
    axis.set_title(title)
    axis.set_xlabel("Observed-censored fraction (%)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)


def plot_severity_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    panels = (
        ("field_excess_rel_l2", "Whole-field point error", "Relative excess-field L2"),
        ("overall_crps_K", "Whole-field predictive score", "CRPS (K)"),
        ("censored_rmse_K", "Pseudo-camera censored region", "RMSE (K)"),
        ("top_1pct_crps_K", "Fixed hottest 1%", "CRPS (K)"),
        ("censored_coverage_95", "Censored-region calibration", "95% coverage"),
        ("censored_interval_width_95_K", "Censored-region sharpness", "95% width (K)"),
    )
    for axis, (metric, title, ylabel) in zip(axes.ravel(), panels):
        plot_metric_panel(axis, summary, metric, title, ylabel)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "What breaks as the pseudo-camera ceiling is lowered?\n"
        "Frozen models; 30 held-out heat-equation trajectories",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_failure_mode_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    panels = (
        ("exceedance_calibration_slope", "Exceedance magnitude recovery", "Predicted-vs-true slope"),
        ("mean_predicted_exceedance_K", "Mean inferred exceedance", "Predicted exceedance (K)"),
        ("hidden_observation_rmse_K", "Deliberately hidden measurements", "RMSE against hidden Y (K)"),
        ("top_1pct_bias_K", "Fixed hottest 1%", "Signed error (K)"),
        ("top_1pct_coverage_95", "Hottest-tail calibration", "95% coverage"),
        ("top_1pct_interval_width_95_K", "Hottest-tail sharpness", "95% width (K)"),
    )
    for axis, (metric, title, ylabel) in zip(axes.ravel(), panels):
        plot_metric_panel(axis, summary, metric, title, ylabel)
    axes[0, 0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Magnitude compression and uncertainty response under lower ceilings",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_ceiling_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    panels = (
        ("overall_rmse_K", "Whole-field point error", "RMSE (K)"),
        ("overall_crps_K", "Whole-field predictive score", "CRPS (K)"),
        ("top_1pct_crps_K", "Fixed hottest 1%", "CRPS (K)"),
        ("top_1pct_coverage_95", "Fixed hottest 1%", "95% coverage"),
    )
    for axis, (metric, title, ylabel) in zip(axes.ravel(), panels):
        for model in METHODS:
            selected = summary[summary["model"] == model].sort_values(
                "ceiling_K", ascending=False
            )
            axis.errorbar(
                selected["ceiling_K"],
                selected[metric],
                yerr=selected[f"{metric}_sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=COLORS[model],
                label=SHORT_LABELS[model],
            )
        axis.invert_xaxis()
        axis.set_title(title)
        axis.set_xlabel("Mean pseudo-camera ceiling (K); lower is more severe")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Performance versus the actual artificial ceiling", fontsize=14)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_percentile_summary(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = (
        ("rmse_K", "RMSE (K)"),
        ("crps_K", "CRPS (K)"),
        ("coverage_95", "95% coverage"),
        ("interval_width_95_K", "95% width (K)"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), constrained_layout=True, sharex=True)
    levels = sorted(summary["level_index"].unique())
    cmap = plt.get_cmap("viridis")
    for column, model in enumerate(METHODS):
        selected_model = summary[summary["model"] == model]
        for row, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            for color_index, level_index in enumerate(levels):
                selected = selected_model[selected_model["level_index"] == level_index].sort_values(
                    "percentile_lower"
                )
                label = str(selected["level"].iloc[0]).replace("target_", "").replace("pct", "%")
                axis.plot(
                    selected["percentile_midpoint"],
                    selected[metric],
                    marker="o",
                    markersize=3,
                    color=cmap(color_index / max(len(levels) - 1, 1)),
                    label=label,
                )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(SHORT_LABELS[model])
            if row == len(metrics) - 1:
                axis.set_xlabel("True-temperature percentile")
    axes[0, 1].legend(title="Ceiling level", frameon=False, ncol=2, fontsize=8)
    figure.suptitle("Where lower sensor range changes error and uncertainty", fontsize=14)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_representative(
    trajectory: str,
    panels: dict[str, dict[str, np.ndarray | float | str]],
    output_path: Path,
) -> None:
    ordered = [panels[key] for key in sorted(panels, key=int)]
    figure, axes = plt.subplots(len(ordered), 4, figsize=(13.2, 3.6 * len(ordered)), constrained_layout=True)
    if len(ordered) == 1:
        axes = np.asarray(axes)[None, :]
    last_image = None
    for row, item in enumerate(ordered):
        ambient = float(item["ambient"])
        reference_ceiling = float(item["reference_ceiling"])
        norm = tail_temperature_norm(ambient, reference_ceiling)
        fields = (
            ("True current field", item["truth"]),
            (f"Pseudo-camera, c={float(item['ceiling_K']):.2f} K", item["camera"]),
            ("Single-frame censored RBF", item["single"]),
            ("Sequential physics GP", item["sequential"]),
        )
        for axis, (title, field) in zip(axes[row], fields):
            last_image = axis.imshow(
                field,
                origin="lower",
                extent=[item["xs"][0], item["xs"][-1], item["ys"][0], item["ys"][-1]],
                cmap="magma",
                norm=norm,
                aspect="auto",
            )
            add_tail_contours(
                axis,
                item["xs"],
                item["ys"],
                field,
                ambient=ambient,
                ceiling=reference_ceiling,
            )
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(str(item["level"]).replace("target_", "").replace("pct", "%"))
    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.018)
        colorbar.set_label("Temperature (K), tail-focused display scale")
    figure.suptitle(f"Lowered-ceiling reconstruction response: {trajectory}", fontsize=14)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_readme(
    output_path: Path,
    summary: pd.DataFrame,
    schedule_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    reference = summary[summary["level_index"] == 0].set_index("model")
    severe = summary[summary["level_index"] == summary["level_index"].max()].set_index("model")
    lines = [
        "# Lowered pseudo-camera ceiling experiment",
        "",
        "> **Status: provisional diagnostic.** The previous-frame sampler used here does not satisfy the convergence criterion established by Experiment 37, and the two models use different current observation sets. Preserve the qualitative ceiling-response result, but rerun under a matched, convergence-checked protocol before using absolute CRPS, coverage, or width values.",
        "",
        "This controlled experiment lowers the camera ceiling while leaving both model specifications and the noisy reference camera realization fixed. Artificial ceilings are computed only from the frozen reference observation; latent simulation truth is used only for evaluation.",
        "",
        "## Models",
        "",
        "- `single-frame censored RBF`: current-frame censored RBF GP.",
        "- `advective posterior physics mean + sequential advective ST`: the best-global sequential architecture from experiment 30, with previous posterior propagation, advective physics mean, and moment-matched advective stochastic space-time residual.",
        "",
        "The single-frame model follows the retained exceedance-diagnostic protocol: fixed-stride unsaturated observations plus every observed-saturated location. The sequential model remains frozen to experiment 30: its previous posterior uses that same saturation-mask rule, while its final current update uses the fixed stride-5 locations. This observation-set distinction is reported explicitly and is not changed across ceilings. No hyperparameter is retuned by ceiling.",
        "",
        "## Ceiling construction",
        "",
        "The trajectory-specific reference ceilings are read from experiment 30 rather than recomputed. Each lower ceiling is selected from that trajectory's reference observation to hit a common target censoring fraction, and `Y^(c) = min(Y^(c0), c)`. This is exactly equivalent to reclipping the same noisy pre-saturation measurement whenever `c < c0`, and it cannot use hidden truth. Reported Kelvin ceilings are held-out means across those trajectory-specific pseudo-cameras.",
        "",
        "The 20% level is a deliberate stress test: its ceiling approaches the ambient/noise floor for these trajectories. It should not be interpreted as a typical camera operating point.",
        "",
        "## Held-out response",
        "",
    ]
    for model in METHODS:
        start = reference.loc[model]
        end = severe.loc[model]
        lines.extend(
            [
                f"### {model}",
                "",
                f"- Whole-field RMSE: `{start['overall_rmse_K']:.3f}` to `{end['overall_rmse_K']:.3f}` K.",
                f"- Overall CRPS: `{start['overall_crps_K']:.3f}` to `{end['overall_crps_K']:.3f}` K.",
                f"- Censored-region RMSE: `{start['censored_rmse_K']:.3f}` to `{end['censored_rmse_K']:.3f}` K.",
                f"- Censored-region coverage/width: `{start['censored_coverage_95']:.3f}` / `{start['censored_interval_width_95_K']:.3f}` K to `{end['censored_coverage_95']:.3f}` / `{end['censored_interval_width_95_K']:.3f}` K.",
                f"- Exceedance recovery slope: `{start['exceedance_calibration_slope']:.3f}` to `{end['exceedance_calibration_slope']:.3f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- The sequential model remains better at every tested ceiling. At the 20% stress level its whole-field RMSE is `{severe.loc[SEQUENTIAL, 'overall_rmse_K']:.3f}` K versus `{severe.loc[SINGLE_FRAME, 'overall_rmse_K']:.3f}` K for the single-frame GP, and its top-1% CRPS is `{severe.loc[SEQUENTIAL, 'top_1pct_crps_K']:.3f}` versus `{severe.loc[SINGLE_FRAME, 'top_1pct_crps_K']:.3f}` K.",
            f"- The sequential top-1% CRPS nevertheless rises from `{reference.loc[SEQUENTIAL, 'top_1pct_crps_K']:.3f}` to `{severe.loc[SEQUENTIAL, 'top_1pct_crps_K']:.3f}` K, so physical/temporal structure reduces but does not remove the penalty from lost sensor range.",
            f"- Hottest-tail uncertainty responds in the wrong direction. Sequential top-1% width contracts from `{reference.loc[SEQUENTIAL, 'top_1pct_interval_width_95_K']:.3f}` to `{severe.loc[SEQUENTIAL, 'top_1pct_interval_width_95_K']:.3f}` K while coverage falls from `{reference.loc[SEQUENTIAL, 'top_1pct_coverage_95']:.3f}` to `{severe.loc[SEQUENTIAL, 'top_1pct_coverage_95']:.3f}`. The single-frame model reaches zero top-1% coverage by the 15% level.",
            f"- Single-frame exceedance slope falls from `{reference.loc[SINGLE_FRAME, 'exceedance_calibration_slope']:.3f}` to `{severe.loc[SINGLE_FRAME, 'exceedance_calibration_slope']:.3f}`; it detects the inequality but does not recover its magnitude. The sequential slope remains materially larger (`{reference.loc[SEQUENTIAL, 'exceedance_calibration_slope']:.3f}` to `{severe.loc[SEQUENTIAL, 'exceedance_calibration_slope']:.3f}`), although its mean exceedance is still compressed at severe censoring.",
            "- Whole-domain CRPS for the single-frame GP is not monotone because the domain is dominated by cool pixels and the increasingly large inequality set makes the posterior narrower. This must not be read as improved hot-field recovery: fixed top-1% CRPS worsens and coverage collapses.",
            "- Censored-region RMSE also changes its target population as the ceiling falls, progressively adding cooler pixels. Fixed top-1% and deliberately hidden-band metrics are the cleaner severity diagnostics.",
            "",
            "## Files",
            "",
            "- `results.csv`: trajectory-level model metrics.",
            "- `heldout30_summary.csv`: equal-trajectory held-out means and standard errors.",
            "- `family_summary.csv`: held-out results by trajectory family.",
            "- `paired_model_comparison.csv`: paired sequential-minus-single-frame changes and win counts.",
            "- `threshold_schedule.csv` and `threshold_schedule_summary.csv`: actual ceiling and censoring severity.",
            "- `region_results.csv`: overall, censored, unsaturated, near-ceiling, deliberately hidden, and fixed top-1% diagnostics.",
            "- `temperature_percentile_results.csv` and `temperature_percentile_summary.csv`: true-temperature-bin diagnostics with extra hot-tail resolution.",
            "- `severity_response.png`: point and distributional performance versus censoring severity.",
            "- `ceiling_response.png`: key metrics versus the actual mean pseudo-camera ceiling.",
            "- `failure_mode_response.png`: exceedance compression, hidden-observation recovery, and calibration response.",
            "- `temperature_percentile_response.png`: error and uncertainty by true-temperature percentile.",
            "- `representative_*.png`: shared, tail-focused reconstruction comparisons.",
            "",
            "## Fixed sampler",
            "",
            f"- Previous posterior: {args.previous_chains} chains, {args.previous_samples_per_chain} retained draws per chain, {args.previous_burn_in} burn-in, thin {args.thin}.",
            f"- Current posterior: {args.current_chains} chains, {args.current_samples_per_chain} retained draws per chain, {args.current_burn_in} burn-in, thin {args.thin}.",
            "- CRPS uses the unbiased `M(M-1)` estimator.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0].to_dict()
    reference_frame = pd.read_csv(REFERENCE_RESULTS)
    threshold_lookup = (
        reference_frame[["trajectory", "threshold_K"]]
        .drop_duplicates()
        .set_index("trajectory")["threshold_K"]
        .to_dict()
    )
    catalog = list(trajectory_catalog(args.dataset_dir))
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    if set(threshold_lookup) != {record.name for record in catalog}:
        raise AssertionError("Frozen reference ceilings do not match the 33-trajectory catalog")
    if len(DEVELOPMENT_TRAJECTORIES) != 3:
        raise AssertionError("Expected exactly three development trajectories")
    held_out = [record for record in catalog if record.name not in DEVELOPMENT_TRAJECTORIES]
    if len(held_out) != 30:
        raise AssertionError(f"Expected 30 held-out trajectories, found {len(held_out)}")

    selected_catalog = catalog
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        selected_catalog = [record for record in catalog if record.name in requested]
        missing = requested - {record.name for record in selected_catalog}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    options = {
        "nx": args.nx,
        "ny": args.ny,
        "target_fractions": args.target_fractions,
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
        "current_chains": args.current_chains,
        "current_samples_per_chain": args.current_samples_per_chain,
        "current_burn_in": args.current_burn_in,
        "thin": args.thin,
        "seed": args.seed,
    }
    payloads = [
        {
            "dataset_dir": str(args.dataset_dir),
            "record_name": record.name,
            "family": record.family,
            "role": "development" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation",
            "catalog_index": index,
            "reference_ceiling": threshold_lookup[record.name],
            "fixed": fixed,
            "options": options,
        }
        for index, record in enumerate(catalog)
        if record in selected_catalog
    ]
    results: list[dict[str, object]] = []
    regions: list[dict[str, object]] = []
    percentiles: list[dict[str, object]] = []
    schedules: list[dict[str, object]] = []
    representatives: dict[str, dict[str, dict[str, np.ndarray | float | str]]] = {}

    def collect(output: dict[str, object]) -> None:
        results.extend(output["results"])
        regions.extend(output["regions"])
        percentiles.extend(output["percentiles"])
        schedules.extend(output["schedule"])
        if output["representative"]:
            name = str(output["results"][0]["trajectory"])
            representatives[name] = output["representative"]
        pd.DataFrame(results).to_csv(args.output_dir / "checkpoint_results.csv", index=False)

    if args.workers == 1:
        for index, payload in enumerate(payloads):
            collect(worker_run_trajectory(payload))
            print(f"[{index + 1:02d}/33] {payload['record_name']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker_run_trajectory, payload): payload for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                payload = futures[future]
                collect(future.result())
                completed += 1
                print(f"[{completed:02d}/33] {payload['record_name']}", flush=True)

    result_frame = pd.DataFrame(results).sort_values(["trajectory", "level_index", "model"])
    region_frame = pd.DataFrame(regions).sort_values(["trajectory", "level_index", "model", "region"])
    percentile_frame = pd.DataFrame(percentiles).sort_values(
        ["trajectory", "level_index", "model", "percentile_lower"]
    )
    schedule_frame = pd.DataFrame(schedules).sort_values(["trajectory", "level_index"])
    evaluation = result_frame[result_frame["role"] == "evaluation"]
    summary = aggregate(evaluation, ["level_index", "level", "model"])
    family_summary = aggregate(evaluation, ["level_index", "level", "family", "model"])
    paired = paired_model_comparison(evaluation)
    evaluation_percentiles = percentile_frame[percentile_frame["role"] == "evaluation"]
    percentile_summary = aggregate(
        evaluation_percentiles,
        ["level_index", "level", "model", "percentile_bin", "percentile_lower", "percentile_upper", "percentile_midpoint"],
    )
    schedule_evaluation = schedule_frame[schedule_frame["role"] == "evaluation"]
    schedule_summary = aggregate(schedule_evaluation, ["level_index", "level"])
    expected_evaluation = sum(
        record.name not in DEVELOPMENT_TRAJECTORIES for record in selected_catalog
    )
    if expected_evaluation and not np.all(summary["n_trajectories"] == expected_evaluation):
        raise AssertionError(
            "Every held-out summary row must contain the selected evaluation trajectories"
        )

    result_frame.to_csv(args.output_dir / "results.csv", index=False)
    region_frame.to_csv(args.output_dir / "region_results.csv", index=False)
    percentile_frame.to_csv(args.output_dir / "temperature_percentile_results.csv", index=False)
    schedule_frame.to_csv(args.output_dir / "threshold_schedule.csv", index=False)
    summary.to_csv(args.output_dir / "heldout30_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_model_comparison.csv", index=False)
    percentile_summary.to_csv(args.output_dir / "temperature_percentile_summary.csv", index=False)
    schedule_summary.to_csv(args.output_dir / "threshold_schedule_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                **fixed,
                **options,
                "dataset_trajectories": 33,
                "development_trajectories": 3,
                "heldout_trajectories": 30,
                "artificial_ceiling_source": "frozen reference camera observation only",
                "truth_used_for_thresholds": False,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "top_1pct_rule": "exact rank-based hottest 25 of 2501 pixels",
                "full_posterior_representation": "not used in this experiment",
                "coherent_previous_posterior": False,
                "previous_posterior_representation": (
                    "censored RBF draws at saturated pixels; unsaturated pixels "
                    "fixed at noisy observed values"
                ),
            }
        ]
    ).to_csv(
        args.output_dir / "fixed_configuration.csv", index=False
    )
    plot_severity_summary(summary, args.output_dir / "severity_response.png")
    plot_failure_mode_summary(summary, args.output_dir / "failure_mode_response.png")
    plot_ceiling_summary(summary, args.output_dir / "ceiling_response.png")
    plot_percentile_summary(percentile_summary, args.output_dir / "temperature_percentile_response.png")
    expected_representatives = REPRESENTATIVES & {
        record.name for record in selected_catalog
    }
    if set(representatives) != expected_representatives:
        raise AssertionError(
            f"Missing representatives: {expected_representatives - set(representatives)}"
        )
    for trajectory, panels in representatives.items():
        plot_representative(
            trajectory,
            panels,
            args.output_dir / f"representative_{trajectory}.png",
        )
    write_readme(args.output_dir / "README.md", summary, schedule_summary, args)
    print(f"Saved lowered-ceiling experiment to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep artificial camera ceilings with frozen single-frame and sequential GPs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument(
        "--target-fractions",
        type=float,
        nargs="+",
        default=[0.03, 0.05, 0.075, 0.10, 0.15, 0.20],
    )
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
    parser.add_argument("--previous-samples-per-chain", type=int, default=700)
    parser.add_argument("--previous-burn-in", type=int, default=1200)
    parser.add_argument("--current-chains", type=int, default=4)
    parser.add_argument("--current-samples-per-chain", type=int, default=350)
    parser.add_argument("--current-burn-in", type=int, default=600)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
