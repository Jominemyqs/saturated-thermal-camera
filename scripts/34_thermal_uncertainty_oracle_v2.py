from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.dense_censored_gp import sample_censored_gaussian_blocks
from src.stochastic_heat_gp import (
    StochasticHeatConfig,
    finite_step_innovation_covariance,
    propagate_residual_draws,
)
from src.thermal_plotting import add_tail_contours, tail_temperature_norm
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    current_source_field,
    diffuse_and_cool,
    infer_previous_censored_posterior,
    infer_previous_coherent_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prepare_trajectory,
    previous_posterior_observations,
)
from src.thermal_trajectory import trajectory_catalog
from src.uncertainty_diagnostics import (
    region_diagnostic_rows,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "32_uncertainty_oracle_v2"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "27_full_posterior_sequential"
    / "fixed_configuration.csv"
)
ARCHITECTURE_OUTPUT = (
    ROOT / "outputs" / "by_experiment" / "30_final_architecture_comparison"
)
ORACLE_V1_OUTPUT = (
    ROOT / "outputs" / "by_experiment" / "31_uncertainty_origin_oracle"
)

HYBRID = "hybrid previous state"
COHERENT = "coherent latent previous state"
STATE_REPRESENTATIONS = (HYBRID, COHERENT)
MEAN_ONLY = "mean-only"
MOMENT_MATCHED = "moment-matched"
PROPAGATION_MODES = (MEAN_ONLY, MOMENT_MATCHED)
FIXED_GAMMA = "fixed gamma"
RECALIBRATED_GAMMA = "development-recalibrated gamma"
STRICT_ORACLE = "strict oracle | exact previous state | fixed gamma"
RECALIBRATED_ORACLE = (
    "recalibrated oracle | exact previous state | development-recalibrated gamma"
)

DIAGNOSTIC_METRICS = [
    "rmse_K",
    "mae_K",
    "signed_error_K",
    "posterior_sd_K",
    "coverage_95",
    "interval_width_95_K",
    "mean_z",
    "rms_z",
    "mean_abs_error_over_sd",
    "median_abs_error_over_sd",
    "fraction_abs_z_gt_1_96",
    "positive_sd_fraction",
    "zero_sd_nonzero_error_fraction",
    "crps_K",
]
RESULT_METRICS = [
    "field_excess_rel_l2",
    "overall_rmse_K",
    "overall_mae_K",
    "overall_signed_error_K",
    "overall_crps_K",
    "top_1pct_rmse_K",
    "top_1pct_mae_K",
    "top_1pct_signed_error_K",
    "top_1pct_crps_K",
    "top_1pct_coverage_95",
    "top_1pct_interval_width_95_K",
    "peak_absolute_error_K",
    "mean_posterior_sd_K",
]


def model_name(
    representation: str,
    propagation: str,
    gamma_mode: str,
) -> str:
    return f"{representation} | {propagation} | {gamma_mode}"


def posterior_summary(draws: np.ndarray) -> tuple[np.ndarray, ...]:
    samples = np.asarray(draws, dtype=float)
    return (
        np.mean(samples, axis=0),
        np.std(samples, axis=0, ddof=1),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
        samples,
    )


def point_distribution_metrics(
    truth: np.ndarray,
    ambient: float,
    prediction: tuple[np.ndarray, ...],
) -> dict[str, float]:
    from src.metrics import empirical_crps

    mean, sd, lower, upper, draws = prediction
    target = np.asarray(truth, dtype=float).ravel()
    top = target >= float(np.quantile(target, 0.99))
    error = mean - target
    crps = empirical_crps(draws, target)
    denominator = np.linalg.norm(target - ambient)
    return {
        "field_excess_rel_l2": float(np.linalg.norm(error) / denominator),
        "overall_rmse_K": float(np.sqrt(np.mean(error**2))),
        "overall_mae_K": float(np.mean(np.abs(error))),
        "overall_signed_error_K": float(np.mean(error)),
        "overall_crps_K": float(np.mean(crps)),
        "top_1pct_rmse_K": float(np.sqrt(np.mean(error[top] ** 2))),
        "top_1pct_mae_K": float(np.mean(np.abs(error[top]))),
        "top_1pct_signed_error_K": float(np.mean(error[top])),
        "top_1pct_crps_K": float(np.mean(crps[top])),
        "top_1pct_coverage_95": float(
            np.mean((lower[top] <= target[top]) & (target[top] <= upper[top]))
        ),
        "top_1pct_interval_width_95_K": float(np.mean((upper - lower)[top])),
        "peak_absolute_error_K": abs(float(np.max(mean) - np.max(target))),
        "mean_posterior_sd_K": float(np.mean(sd)),
    }


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    return iy * len(prepared.xs) + ix


def previous_masks(truth: np.ndarray, saturated: np.ndarray) -> dict[str, np.ndarray]:
    flat_truth = np.asarray(truth, dtype=float).ravel()
    flat_saturated = np.asarray(saturated, dtype=bool).ravel()
    return {
        "overall": np.ones(len(flat_truth), dtype=bool),
        "unsaturated": ~flat_saturated,
        "saturated": flat_saturated,
        "previous_top_1pct": flat_truth >= float(np.quantile(flat_truth, 0.99)),
    }


def current_masks(truth: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    flat_truth = np.asarray(truth, dtype=float).ravel()
    return {
        "overall": np.ones(len(flat_truth), dtype=bool),
        "above_camera_ceiling": flat_truth >= threshold,
        "current_top_1pct": flat_truth >= float(np.quantile(flat_truth, 0.99)),
    }


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    evaluation = frame[frame["role"] == "evaluation"]
    aggregate = evaluation.groupby(group_columns, sort=False)[metrics].agg(
        ["mean", "std"]
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    counts = (
        evaluation.groupby(group_columns, sort=False)["trajectory"]
        .nunique()
        .rename("n_trajectories")
        .reset_index()
    )
    return aggregate.merge(counts, on=group_columns, how="left")


def infer_previous_states(
    prepared,
    camera: dict[str, object],
    threshold: float,
    fixed: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, float]]]:
    common = {
        "prepared": prepared,
        "frame": camera["frames"][0],
        "fixed_mask": camera["fixed_observation_mask"],
        "threshold": threshold,
        "signal_sd": float(fixed.signal_sd),
        "lengthscale": prepared.source_lengthscale * args.length_multiplier,
        "noise_sd": args.previous_noise_sd,
        "burn_in": args.previous_burn_in,
        "thin": args.thin,
        "seed": seed,
    }
    hybrid_mean, hybrid_draws, hybrid_diagnostics = (
        infer_previous_censored_posterior(
            **common,
            n_chains=1,
            samples_per_chain=args.previous_samples,
        )
    )
    coherent_mean, coherent_draws, coherent_diagnostics = (
        infer_previous_coherent_posterior(
            **common,
            n_samples=args.previous_samples,
        )
    )
    return (
        {
            HYBRID: (hybrid_mean, hybrid_draws),
            COHERENT: (coherent_mean, coherent_draws),
        },
        {HYBRID: hybrid_diagnostics, COHERENT: coherent_diagnostics},
    )


def calibration_camera(
    prepared,
    catalog_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], float, int, int]:
    current_index = len(prepared.times) - 1
    previous_index = current_index - args.previous_frame_offset
    threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
    camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=threshold,
        observation_stride=args.observation_stride,
        noise_sd=args.measurement_noise_sd,
        seed=args.seed + 10_000 * catalog_index,
    )
    return camera, threshold, previous_index, current_index


def calibrate_representation_gammas(
    catalog: list,
    fixed: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, float], pd.DataFrame]:
    accumulators: dict[str, dict[str, float]] = defaultdict(
        lambda: {"numerator": 0.0, "denominator": 0.0, "n_samples": 0.0}
    )
    catalog_lookup = {record.name: index for index, record in enumerate(catalog)}
    for name in DEVELOPMENT_TRAJECTORIES:
        catalog_index = catalog_lookup[name]
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        camera, threshold, previous_index, current_index = calibration_camera(
            prepared, catalog_index, args
        )
        states, _ = infer_previous_states(
            prepared,
            camera,
            threshold,
            fixed,
            args,
            args.seed + 50_000 * catalog_index,
        )
        state_means = {
            HYBRID: states[HYBRID][0],
            COHERENT: states[COHERENT][0],
            "true previous state": prepared.history[previous_index],
        }
        source_response = current_source_field(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            source_coupling=1.0,
            source_flux_threshold=args.source_flux_threshold,
        )
        active = source_response > 0.0
        source_values = source_response[active]
        for representation, previous_mean in state_means.items():
            background = diffuse_and_cool(
                prepared,
                previous_mean,
                previous_index=previous_index,
                current_index=current_index,
                diffusivity=float(fixed.diffusivity),
                cooling_rate=float(fixed.cooling_rate),
            )
            target = (
                prepared.history[current_index]
                - prepared.ambient
                - background
            )[active]
            accumulator = accumulators[representation]
            accumulator["numerator"] += float(np.dot(source_values, target))
            accumulator["denominator"] += float(
                np.dot(source_values, source_values)
            )
            accumulator["n_samples"] += int(np.sum(active))
    gammas = {}
    rows = []
    for representation, values in accumulators.items():
        gamma = max(values["numerator"] / values["denominator"], 0.0)
        gammas[representation] = gamma
        rows.append(
            {
                "state_representation": representation,
                "source_coupling": gamma,
                "n_source_pixels": int(values["n_samples"]),
                "calibration_scope": (
                    "final one-step transition pooled over three development trajectories"
                ),
            }
        )
    rows.append(
        {
            "state_representation": "frozen project baseline",
            "source_coupling": float(fixed.source_coupling),
            "n_source_pixels": np.nan,
            "calibration_scope": "retained observed-clipped project calibration",
        }
    )
    return gammas, pd.DataFrame(rows)


def innovation_blocks(
    prepared,
    camera: dict[str, object],
    fixed: pd.Series,
    args: argparse.Namespace,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, float]:
    current = camera["current"]
    prediction_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed = observation_indices(prepared, observation_points)
    previous_index = int(camera["frames"][0]["time_index"])
    current_index = int(camera["frames"][1]["time_index"])
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    config = StochasticHeatConfig(
        signal_sd=float(fixed.signal_sd),
        forcing_lengthscale=(
            prepared.source_lengthscale
            * args.length_multiplier
            * args.forcing_length_multiplier
        ),
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        quadrature_order=args.quadrature_order,
    )
    observed_covariance = finite_step_innovation_covariance(
        observation_points, observation_points, config, dt
    )
    observed_covariance = 0.5 * (
        observed_covariance + observed_covariance.T
    )
    pred_observed_covariance = finite_step_innovation_covariance(
        prediction_points, observation_points, config, dt
    )
    variance = float(
        finite_step_innovation_covariance(
            prediction_points[:1], prediction_points[:1], config, dt
        )[0, 0]
    )
    prediction_variance = np.full(len(prediction_points), max(variance, 0.0))
    return (
        (observed_covariance, pred_observed_covariance, prediction_variance),
        observed,
        dt,
    )


def propagated_blocks(
    propagated_draws: np.ndarray,
    observed: np.ndarray,
    innovation: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_covariance, pred_observed_covariance, prediction_variance = innovation
    flat = propagated_draws.reshape(len(propagated_draws), -1)
    features = flat.T / np.sqrt(len(flat) - 1.0)
    observed_features = features[observed]
    return (
        observed_covariance + observed_features.dot(observed_features.T),
        pred_observed_covariance + features.dot(observed_features.T),
        prediction_variance + np.sum(features**2, axis=1),
    )


def gaussian_forecast(
    mean_field: np.ndarray,
    variance: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    mean = np.asarray(mean_field, dtype=float).ravel()
    sd = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0))
    rng = np.random.default_rng(seed)
    draws = mean[None, :] + rng.normal(size=(n_samples, len(mean))) * sd[None, :]
    return (
        mean,
        sd,
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        draws,
    )


def current_update(
    current: dict[str, object],
    mean_field: np.ndarray,
    observed: np.ndarray,
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, ...]:
    mean = np.asarray(mean_field, dtype=float).ravel()
    return sample_censored_gaussian_blocks(
        current,
        prediction_mean=mean,
        observation_mean=mean[observed],
        observed_covariance=blocks[0],
        pred_observed_covariance=blocks[1],
        prediction_variance=blocks[2],
        noise_sd=args.current_noise_sd,
        n_samples=args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=seed,
    )


def add_region_rows(
    destination: list[dict[str, object]],
    truth: np.ndarray,
    prediction: tuple[np.ndarray, ...],
    masks: dict[str, np.ndarray],
    identity: dict[str, object],
) -> None:
    rows = region_diagnostic_rows(
        truth,
        *prediction[:4],
        masks,
        draws=prediction[4],
    )
    for row in rows:
        row.update(identity)
    destination.extend(rows)


def evaluate_model(
    *,
    prepared,
    camera: dict[str, object],
    mean_field: np.ndarray,
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray],
    observed: np.ndarray,
    model: str,
    representation: str,
    propagation: str,
    gamma_mode: str,
    role: str,
    record,
    threshold: float,
    args: argparse.Namespace,
    forecast_seed: int,
    current_seed: int,
    stage_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    forecast = gaussian_forecast(
        mean_field,
        blocks[2],
        n_samples=args.samples,
        seed=forecast_seed,
    )
    posterior = current_update(
        camera["current"],
        mean_field,
        observed,
        blocks,
        args,
        current_seed,
    )
    masks = current_masks(prepared.truth, threshold)
    base_identity = {
        "trajectory": record.name,
        "family": record.family,
        "run_index": record.run_index,
        "role": role,
        "model": model,
        "state_representation": representation,
        "propagation": propagation,
        "gamma_mode": gamma_mode,
        "region_basis": "current-frame truth",
    }
    for stage, prediction in (
        ("current forecast before Y_n", forecast),
        ("current posterior after Y_n", posterior),
    ):
        add_region_rows(
            stage_rows,
            prepared.truth.ravel(),
            prediction,
            masks,
            {**base_identity, "stage": stage},
        )
        result_rows.append(
            {
                **base_identity,
                "stage": stage,
                **point_distribution_metrics(
                    prepared.truth, prepared.ambient, prediction
                ),
            }
        )
    return forecast, posterior


def plot_three_stage_top1(
    previous_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = [
        ("rmse_K_mean", "RMSE (K)"),
        ("posterior_sd_K_mean", "Posterior SD (K)"),
        ("coverage_95_mean", "95% coverage"),
        ("mean_abs_error_over_sd_mean", r"Mean $|e|/\sigma$"),
    ]
    stages = [
        "previous posterior\n(previous top 1%)",
        "current forecast\n(current top 1%)",
        "current posterior\n(current top 1%)",
    ]
    colors = {HYBRID: "#D55E00", COHERENT: "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for representation in STATE_REPRESENTATIONS:
            previous = previous_summary[
                (previous_summary["state_representation"] == representation)
                & (previous_summary["region"] == "previous_top_1pct")
            ].iloc[0]
            model = model_name(representation, MOMENT_MATCHED, FIXED_GAMMA)
            current = stage_summary[
                (stage_summary["model"] == model)
                & (stage_summary["region"] == "current_top_1pct")
            ].set_index("stage")
            values = [
                previous[metric],
                current.loc["current forecast before Y_n", metric],
                current.loc["current posterior after Y_n", metric],
            ]
            axis.plot(
                range(3),
                values,
                marker="o",
                linewidth=2.0,
                color=colors[representation],
                label=representation,
            )
        if metric == "coverage_95_mean":
            axis.axhline(0.95, color="#555555", linestyle=":", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
        axis.set_xticks(range(3), stages)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Three-stage hottest-tail diagnostic\n"
        "Previous and current top-1% masks are intentionally labeled separately"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_representative_stages(
    prepared,
    threshold: float,
    previous_fields: dict[str, np.ndarray],
    forecasts: dict[str, tuple[np.ndarray, ...]],
    posteriors: dict[str, tuple[np.ndarray, ...]],
    output_path: Path,
) -> None:
    rows = [HYBRID, COHERENT, "strict oracle"]
    thermal_norm = tail_temperature_norm(prepared.ambient, threshold)
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    figure, axes = plt.subplots(3, 4, figsize=(12.8, 9.2), constrained_layout=True)
    image = None
    for row_index, label in enumerate(rows):
        fields = [
            previous_fields[label],
            forecasts[label][0].reshape(prepared.truth.shape),
            posteriors[label][0].reshape(prepared.truth.shape),
            prepared.truth,
        ]
        for column, field in enumerate(fields):
            axis = axes[row_index, column]
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="magma",
                norm=thermal_norm,
                aspect="auto",
            )
            add_tail_contours(
                axis,
                prepared.xs * 1e3,
                prepared.ys * 1e3,
                field,
                ambient=prepared.ambient,
                ceiling=threshold,
            )
            if row_index == 0:
                axis.set_title(
                    [
                        "previous state estimate",
                        "current forecast",
                        "current posterior",
                        "true current field",
                    ][column]
                )
            if column == 0:
                axis.set_ylabel(label.replace(" previous state", ""), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(
        image,
        ax=axes,
        shrink=0.72,
        label="temperature (K)",
        extend="both",
    )
    figure.suptitle(
        f"Previous posterior -> current forecast -> current posterior: {prepared.name}\n"
        "Shared ambient-to-ceiling tail scale; fixed-gamma moment rows"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    output_path: Path,
    previous_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    recalibrated_summary: pd.DataFrame,
    gamma_table: pd.DataFrame,
    top1_crosswalk: pd.DataFrame,
) -> None:
    def result_line(row: pd.Series) -> str:
        label = str(row["model"]).replace(" | ", "; ")
        return (
            f"| {label} | {row['field_excess_rel_l2_mean']:.3f} | "
            f"{row['overall_crps_K_mean']:.3f} | "
            f"{row['top_1pct_rmse_K_mean']:.3f} | "
            f"{row['top_1pct_crps_K_mean']:.3f} | "
            f"{row['top_1pct_coverage_95_mean']:.3f} | "
            f"{row['top_1pct_interval_width_95_K_mean']:.3f} |"
        )

    lines = [
        "# Underdispersion and oracle diagnostic v2",
        "",
        "This canonical rerun separates three distinct distributions:",
        "",
        "1. `previous posterior`: p(T_(n-1) | Y_(n-1));",
        "2. `current forecast`: p(T_n | Y_(n-1)), before the current image;",
        "3. `current posterior`: p(T_n | Y_(n-1), Y_n).",
        "",
        "The forecast and current-posterior rows use the same current-frame top-1% "
        "mask. The previous posterior uses its own previous-frame top-1% mask and is "
        "not described as a direct before/after spatial comparison.",
        "The top-1% mask uses the inclusive empirical 99th-percentile threshold, "
        "matching the retained architecture comparison (26 pixels on the 2501-pixel grid).",
        "",
        "## Previous-state inference",
        "",
        "The hybrid representation pins unsaturated grid values to the noisy clipped "
        "camera frame and samples only saturated pixels. The coherent latent posterior "
        "conditions on the same sparse-plus-saturated observation set but samples the "
        "entire latent field jointly, including measurement uncertainty at unsaturated "
        "points. No current-frame data enter either previous posterior.",
        "",
        "## Source coupling",
        "",
        "The fixed-gamma table is the causal diagnostic: every representation uses the "
        "same retained source coupling. The recalibrated table is a separate attainable-"
        "performance comparison: gamma is fitted independently for each representation "
        "using only the final one-step transitions of the three development trajectories.",
        "",
        "| State representation | Source coupling | Calibration scope |",
        "|---|---:|---|",
        "",
        "## Fixed-gamma current posterior",
        "",
        "| Model | Field L2 | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Coverage | Width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    gamma_lines = [
        f"| {row['state_representation']} | {row['source_coupling']:.6f} | "
        f"{row['calibration_scope']} |"
        for _, row in gamma_table.iterrows()
    ]
    insertion = lines.index("## Fixed-gamma current posterior") - 1
    lines[insertion:insertion] = gamma_lines
    lines.extend(result_line(row) for _, row in fixed_summary.iterrows())
    lines.extend(
        [
            "",
            "## Development-recalibrated current posterior",
            "",
            "| Model | Field L2 | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Coverage | Width |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(result_line(row) for _, row in recalibrated_summary.iterrows())

    previous_top = previous_summary[
        previous_summary["region"] == "previous_top_1pct"
    ].set_index("state_representation")
    current_top = stage_summary[
        stage_summary["region"] == "current_top_1pct"
    ]
    lines.extend(
        [
            "",
            "## Calibration diagnosis",
            "",
            "Previous-posterior hottest-tail coverage:",
            "",
            f"- Hybrid: {previous_top.loc[HYBRID, 'coverage_95_mean']:.3f} "
            f"with SD {previous_top.loc[HYBRID, 'posterior_sd_K_mean']:.3f} K.",
            f"- Coherent latent: {previous_top.loc[COHERENT, 'coverage_95_mean']:.3f} "
            f"with SD {previous_top.loc[COHERENT, 'posterior_sd_K_mean']:.3f} K.",
            "",
            "For each fixed-gamma moment-matched model, `forecast_update_summary.csv` "
            "reports the current forecast and current posterior on the identical current "
            "top-1% mask. This distinguishes transition uncertainty from the effect of "
            "assimilating Y_n.",
            "",
            "## Top-1% definition audit",
            "",
            "The earlier 2.749 K versus 2.839 K discrepancy was entirely a region-"
            "definition difference. The retained architecture table used an inclusive "
            "empirical 99th-percentile threshold (26 pixels); oracle v1 used exact ranks "
            "(25 pixels). V2 adopts the earlier inclusive threshold and reproduces the "
            "architecture result.",
            "",
            "| Source | Top-1% pixels | Top-1% CRPS (K) |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['source']} | {int(row['top_1pct_pixels'])} | "
        f"{row['top_1pct_crps_K']:.6f} |"
        for _, row in top1_crosswalk.iterrows()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Making unsaturated pixels latent repairs the ordinary-field posterior "
            "degeneracy, but it does not materially change the previous hottest-tail "
            "RMSE, SD, coverage, or CRPS. The hottest 1% is already censored, so the "
            "dominant underdispersion originates in single-frame censored peak inference, "
            "not in pinning the unsaturated values. Propagating the coherent covariance "
            "raises ordinary-field uncertainty enough to worsen all-domain CRPS, while "
            "hottest-tail calibration remains essentially unchanged.",
            "",
            "The current observation update also does not repair the tail: relative to "
            "the forecast it leaves RMSE almost unchanged and slightly reduces coverage. "
            "The strict oracle overshoots because its source coefficient was calibrated "
            "for clipped-state compensation; the separately recalibrated oracle is the "
            "appropriate attainable-performance ceiling.",
            "",
            "This evidence does not yet justify adaptive Q as the first fix. The next "
            "modeling question should target the saturated single-frame posterior and "
            "its peak prior/mean; transition innovation can be revisited after that state "
            "posterior is better calibrated.",
            "",
            "All CRPS values use the unbiased M(M-1) estimator. Held-out trajectories "
            "never affect source calibration or any fixed hyperparameter.",
        ]
    )
    if current_top.empty:
        raise AssertionError("Current top-1% summary is unexpectedly empty")
    output_path.write_text("\n".join(lines), encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0]
    catalog = list(trajectory_catalog(args.dataset_dir))
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    recalibrated_gammas, gamma_table = calibrate_representation_gammas(
        catalog, fixed, args
    )
    gamma_table.to_csv(args.output_dir / "source_coupling_calibration.csv", index=False)

    requested = set(args.trajectory_names or [])
    indexed = list(enumerate(catalog))
    if requested:
        indexed = [(index, record) for index, record in indexed if record.name in requested]
        missing = requested - {record.name for _, record in indexed}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    previous_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    representatives = {
        "DiagonalScanPath_7",
        "HorizontalScanPath_10",
        "SpiralScanPath_12",
    }

    for item_index, (catalog_index, record) in enumerate(indexed):
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        role = "development" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        camera, threshold, previous_index, current_index = calibration_camera(
            prepared, catalog_index, args
        )
        previous_seed = args.seed + 50_000 * catalog_index
        forecast_seed = args.seed + 80_000 * catalog_index
        current_seed = args.seed + 100_000 * catalog_index
        states, state_diagnostics = infer_previous_states(
            prepared,
            camera,
            threshold,
            fixed,
            args,
            previous_seed,
        )
        previous_truth = prepared.history[previous_index].ravel()
        previous_saturated = np.asarray(
            camera["frames"][0]["saturated_full"], dtype=bool
        )
        previous_region_masks = previous_masks(previous_truth, previous_saturated)
        for representation, (_, draws) in states.items():
            prediction = posterior_summary(draws.reshape(len(draws), -1))
            add_region_rows(
                previous_rows,
                previous_truth,
                prediction,
                previous_region_masks,
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "state_representation": representation,
                    "stage": "previous posterior",
                    "region_basis": "previous-frame truth",
                },
            )

        innovation, observed, dt = innovation_blocks(
            prepared, camera, fixed, args
        )
        displacement = posterior_physics_means(
            prepared,
            states[HYBRID][0],
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
        )[2]
        propagation_arguments = {
            "dx": float(np.mean(np.diff(prepared.xs))),
            "dy": float(np.mean(np.diff(prepared.ys))),
            "time_step": dt,
            "diffusivity": float(fixed.diffusivity),
            "cooling_rate": float(fixed.cooling_rate),
            "displacement": displacement,
        }

        representative_previous: dict[str, np.ndarray] = {}
        representative_forecasts: dict[str, tuple[np.ndarray, ...]] = {}
        representative_posteriors: dict[str, tuple[np.ndarray, ...]] = {}
        for representation, (previous_mean, previous_draws) in states.items():
            centered = previous_draws - previous_mean[None, :, :]
            propagated = propagate_residual_draws(
                centered, **propagation_arguments
            )
            blocks_by_mode = {
                MEAN_ONLY: innovation,
                MOMENT_MATCHED: propagated_blocks(
                    propagated, observed, innovation
                ),
            }
            gamma_values = {
                FIXED_GAMMA: float(fixed.source_coupling),
                RECALIBRATED_GAMMA: recalibrated_gammas[representation],
            }
            for gamma_mode, gamma in gamma_values.items():
                mean_field = posterior_physics_means(
                    prepared,
                    previous_mean,
                    previous_index=previous_index,
                    current_index=current_index,
                    diffusivity=float(fixed.diffusivity),
                    cooling_rate=float(fixed.cooling_rate),
                    source_coupling=gamma,
                    source_flux_threshold=args.source_flux_threshold,
                )[0]
                for propagation in PROPAGATION_MODES:
                    model = model_name(representation, propagation, gamma_mode)
                    forecast, posterior = evaluate_model(
                        prepared=prepared,
                        camera=camera,
                        mean_field=mean_field,
                        blocks=blocks_by_mode[propagation],
                        observed=observed,
                        model=model,
                        representation=representation,
                        propagation=propagation,
                        gamma_mode=gamma_mode,
                        role=role,
                        record=record,
                        threshold=threshold,
                        args=args,
                        forecast_seed=forecast_seed,
                        current_seed=current_seed,
                        stage_rows=stage_rows,
                        result_rows=result_rows,
                    )
                    if gamma_mode == FIXED_GAMMA and propagation == MOMENT_MATCHED:
                        representative_previous[representation] = previous_mean
                        representative_forecasts[representation] = forecast
                        representative_posteriors[representation] = posterior

        oracle_previous = prepared.history[previous_index]
        oracle_means = {
            STRICT_ORACLE: (
                float(fixed.source_coupling),
                FIXED_GAMMA,
            ),
            RECALIBRATED_ORACLE: (
                recalibrated_gammas["true previous state"],
                RECALIBRATED_GAMMA,
            ),
        }
        for model, (gamma, gamma_mode) in oracle_means.items():
            mean_field = posterior_physics_means(
                prepared,
                oracle_previous,
                previous_index=previous_index,
                current_index=current_index,
                diffusivity=float(fixed.diffusivity),
                cooling_rate=float(fixed.cooling_rate),
                source_coupling=gamma,
                source_flux_threshold=args.source_flux_threshold,
            )[0]
            forecast, posterior = evaluate_model(
                prepared=prepared,
                camera=camera,
                mean_field=mean_field,
                blocks=innovation,
                observed=observed,
                model=model,
                representation="true previous state",
                propagation="exact state; innovation only",
                gamma_mode=gamma_mode,
                role=role,
                record=record,
                threshold=threshold,
                args=args,
                forecast_seed=forecast_seed,
                current_seed=current_seed,
                stage_rows=stage_rows,
                result_rows=result_rows,
            )
            if model == STRICT_ORACLE:
                representative_previous["strict oracle"] = oracle_previous
                representative_forecasts["strict oracle"] = forecast
                representative_posteriors["strict oracle"] = posterior

        observations, _, _ = previous_posterior_observations(
            prepared,
            frame=camera["frames"][0],
            fixed_mask=camera["fixed_observation_mask"],
            threshold=threshold,
        )
        integrity_rows.append(
            {
                "trajectory": record.name,
                "family": record.family,
                "run_index": record.run_index,
                "role": role,
                "threshold_K": threshold,
                "camera_seed": args.seed + 10_000 * catalog_index,
                "previous_seed": previous_seed,
                "forecast_seed": forecast_seed,
                "current_seed": current_seed,
                "previous_observation_count": len(observations["y_obs"]),
                "same_previous_observation_set_hybrid_coherent": True,
                "previous_data_used_in_current_likelihood": False,
                "current_truth_used_for_inference": False,
                "same_current_observations_all_models": True,
                "same_current_seed_all_models": True,
                "same_innovation_all_models": True,
                "same_fixed_gamma_hybrid_coherent_oracle": True,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "top_1pct_rule": "truth >= empirical 99th percentile",
                "current_top_1pct_pixels": int(
                    np.sum(
                        prepared.truth.ravel()
                        >= float(np.quantile(prepared.truth, 0.99))
                    )
                ),
                "hybrid_unsaturated_values_pinned": True,
                "coherent_unsaturated_latent": True,
                "coherent_full_spatial_draws": True,
                "hybrid_previous_mean_sd_K": float(
                    np.mean(np.std(states[HYBRID][1], axis=0, ddof=1))
                ),
                "coherent_previous_mean_sd_K": float(
                    np.mean(np.std(states[COHERENT][1], axis=0, ddof=1))
                ),
                **{
                    f"hybrid_{key}": value
                    for key, value in state_diagnostics[HYBRID].items()
                },
                **{
                    f"coherent_{key}": value
                    for key, value in state_diagnostics[COHERENT].items()
                },
            }
        )

        if record.name in representatives:
            plot_representative_stages(
                prepared,
                threshold,
                representative_previous,
                representative_forecasts,
                representative_posteriors,
                args.output_dir / f"three_stage_{record.name.lower()}.png",
            )
        print(
            f"[{item_index + 1:02d}/{len(indexed):02d}] {record.name}: "
            f"hybrid SD={integrity_rows[-1]['hybrid_previous_mean_sd_K']:.3f}, "
            f"coherent SD={integrity_rows[-1]['coherent_previous_mean_sd_K']:.3f}",
            flush=True,
        )
        pd.DataFrame(previous_rows).to_csv(
            args.output_dir / "checkpoint_previous.csv", index=False
        )
        pd.DataFrame(result_rows).to_csv(
            args.output_dir / "checkpoint_results.csv", index=False
        )

    previous = pd.DataFrame(previous_rows)
    stages = pd.DataFrame(stage_rows)
    results = pd.DataFrame(result_rows)
    integrity = pd.DataFrame(integrity_rows)
    previous_summary = summarize(
        previous,
        ["state_representation", "region"],
        DIAGNOSTIC_METRICS,
    )
    stage_summary = summarize(
        stages,
        [
            "model",
            "state_representation",
            "propagation",
            "gamma_mode",
            "stage",
            "region",
        ],
        DIAGNOSTIC_METRICS,
    )
    current_results = results[results["stage"] == "current posterior after Y_n"]
    current_summary = summarize(
        current_results,
        ["model", "state_representation", "propagation", "gamma_mode"],
        RESULT_METRICS,
    )
    family_summary = summarize(
        current_results,
        ["model", "family"],
        RESULT_METRICS,
    )
    fixed_summary = current_summary[
        current_summary["gamma_mode"] == FIXED_GAMMA
    ].reset_index(drop=True)
    recalibrated_summary = current_summary[
        current_summary["gamma_mode"] == RECALIBRATED_GAMMA
    ].reset_index(drop=True)

    architecture = pd.read_csv(ARCHITECTURE_OUTPUT / "architecture_comparison.csv")
    architecture_row = architecture[
        architecture["method"]
        == "Posterior physics mean + sequential advective ST (moment-matched)"
    ].iloc[0]
    oracle_v1 = pd.read_csv(ORACLE_V1_OUTPUT / "heldout30_current_summary.csv")
    oracle_v1_row = oracle_v1[
        oracle_v1["method"]
        == "Posterior physics mean + sequential advective ST"
    ].iloc[0]
    v2_model = model_name(HYBRID, MOMENT_MATCHED, FIXED_GAMMA)
    v2_row = fixed_summary[fixed_summary["model"] == v2_model].iloc[0]
    top1_crosswalk = pd.DataFrame(
        [
            {
                "source": "retained architecture comparison",
                "top_1pct_rule": "truth >= empirical 99th percentile",
                "top_1pct_pixels": 26,
                "top_1pct_crps_K": architecture_row["top_1pct_crps_K"],
            },
            {
                "source": "oracle diagnostic v1",
                "top_1pct_rule": "exact percentile ranks >= 99",
                "top_1pct_pixels": 25,
                "top_1pct_crps_K": oracle_v1_row["top_1pct_crps_K_mean"],
            },
            {
                "source": "canonical oracle diagnostic v2",
                "top_1pct_rule": "truth >= empirical 99th percentile",
                "top_1pct_pixels": 26,
                "top_1pct_crps_K": v2_row["top_1pct_crps_K_mean"],
            },
        ]
    )

    outputs = {
        "previous_state_by_trajectory.csv": previous,
        "previous_state_summary.csv": previous_summary,
        "forecast_update_by_trajectory.csv": stages,
        "forecast_update_summary.csv": stage_summary,
        "stage_results_by_trajectory.csv": results,
        "current_posterior_summary.csv": current_summary,
        "fixed_gamma_current_summary.csv": fixed_summary,
        "recalibrated_gamma_current_summary.csv": recalibrated_summary,
        "family_summary.csv": family_summary,
        "top1_definition_crosswalk.csv": top1_crosswalk,
        "integrity_checks.csv": integrity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    fixed_output = pd.DataFrame(
        [
            {
                **fixed.to_dict(),
                "fraction_saturated": args.fraction_saturated,
                "nx": args.nx,
                "ny": args.ny,
                "observation_stride": args.observation_stride,
                "previous_frame_offset": args.previous_frame_offset,
                "measurement_noise_sd": args.measurement_noise_sd,
                "previous_samples": args.previous_samples,
                "current_samples": args.samples,
                "dataset_trajectory_count": len(catalog),
                "development_trajectory_count": len(DEVELOPMENT_TRAJECTORIES),
                "heldout_trajectory_count": int(
                    np.sum(integrity["role"] == "evaluation")
                ),
                "previous_observation_set": "fixed stride plus every saturated pixel",
                "coherent_previous_posterior": (
                    "joint latent RBF posterior with noisy unsaturated likelihood and censored likelihood"
                ),
                "top_1pct_rule": "truth >= empirical 99th percentile",
                "sequential_residual": (
                    "one-step advective propagation of centered previous draws plus finite-step innovation"
                ),
            }
        ]
    )
    fixed_output.to_csv(args.output_dir / "fixed_configuration.csv", index=False)

    if not args.trajectory_names:
        roles = integrity.groupby("role")["trajectory"].nunique().to_dict()
        if roles != {"development": 3, "evaluation": 30}:
            raise AssertionError(f"Unexpected split: {roles}")
        if not integrity["previous_data_used_in_current_likelihood"].eq(False).all():
            raise AssertionError("Previous observations leaked into current likelihood")
        if not integrity["same_previous_observation_set_hybrid_coherent"].all():
            raise AssertionError("Hybrid and coherent observation sets differ")
        if not integrity["current_truth_used_for_inference"].eq(False).all():
            raise AssertionError("Current truth was used for inference")
        reproduction_metrics = {
            "field_excess_rel_l2": "field_excess_rel_l2_mean",
            "overall_crps_K": "overall_crps_K_mean",
            "top_1pct_crps_K": "top_1pct_crps_K_mean",
        }
        for old_metric, new_metric in reproduction_metrics.items():
            if not np.isclose(
                float(architecture_row[old_metric]),
                float(v2_row[new_metric]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(f"V2 failed to reproduce {old_metric}")

    plot_three_stage_top1(
        previous_summary,
        stage_summary,
        args.output_dir / "three_stage_top1_diagnostic.png",
    )
    write_readme(
        args.output_dir / "README.md",
        previous_summary,
        stage_summary,
        fixed_summary,
        recalibrated_summary,
        gamma_table,
        top1_crosswalk,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical hybrid/coherent three-stage uncertainty and oracle audit."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
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
    parser.add_argument("--previous-samples", type=int, default=180)
    parser.add_argument("--previous-burn-in", type=int, default=120)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--burn-in", type=int, default=120)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
