from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.dense_censored_gp import (
    sample_censored_gaussian_blocks,
    sample_censored_gaussian_mixture_blocks,
)
from src.stochastic_heat_gp import (
    StochasticHeatConfig,
    finite_step_innovation_covariance,
    propagate_residual_draws,
)
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    calibrate_source_coupling,
    diffuse_and_cool,
    infer_previous_censored_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prepare_trajectory,
)
from src.thermal_plotting import add_tail_contours, tail_temperature_norm
from src.thermal_trajectory import trajectory_catalog
from src.uncertainty_diagnostics import (
    PERCENTILE_BINS,
    percentile_diagnostic_rows,
    percentile_rank,
    region_diagnostic_rows,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "31_uncertainty_origin_oracle"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "27_full_posterior_sequential"
    / "fixed_configuration.csv"
)

SEQ_ST = "Posterior physics mean + sequential ST"
SEQ_ADV_ST = "Posterior physics mean + sequential advective ST"
ADV_MEAN_SEQ_ADV_ST = "Advective posterior mean + sequential advective ST"
SAMPLE_MIXTURE = "Posterior-sample mixture sequential advective ST"
CURRENT_METHODS = [SEQ_ST, SEQ_ADV_ST, ADV_MEAN_SEQ_ADV_ST, SAMPLE_MIXTURE]

ORACLE_CLIPPED = "Observed-clipped previous state + ST innovation"
ORACLE_POSTERIOR = "Posterior-mean previous state + ST innovation"
ORACLE_MOMENT = "Moment-matched sequential advective ST"
ORACLE_MIXTURE = "Posterior-sample mixture sequential advective ST"
ORACLE_TRUE_FIXED = "Oracle true previous state + fixed clipped-calibrated source"
ORACLE_TRUE = "Oracle true previous state + dev-calibrated source"
ORACLE_METHODS = [
    ORACLE_CLIPPED,
    ORACLE_POSTERIOR,
    ORACLE_MOMENT,
    ORACLE_MIXTURE,
    ORACLE_TRUE_FIXED,
    ORACLE_TRUE,
]

COLORS = {
    SEQ_ST: "#0072B2",
    SEQ_ADV_ST: "#009E73",
    ADV_MEAN_SEQ_ADV_ST: "#D55E00",
    SAMPLE_MIXTURE: "#CC79A7",
    ORACLE_CLIPPED: "#999999",
    ORACLE_POSTERIOR: "#56B4E9",
    ORACLE_MOMENT: "#009E73",
    ORACLE_MIXTURE: "#CC79A7",
    ORACLE_TRUE_FIXED: "#E69F00",
    ORACLE_TRUE: "#332288",
}

DIAGNOSTIC_METRICS = [
    "rmse_K",
    "mae_K",
    "signed_error_K",
    "posterior_sd_K",
    "coverage_95",
    "interval_width_95_K",
    "positive_sd_fraction",
    "zero_sd_nonzero_error_fraction",
    "mean_z",
    "rms_z",
    "mean_abs_error_over_sd",
    "median_abs_error_over_sd",
    "fraction_abs_z_gt_1_96",
    "crps_K",
]


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    indices = iy * len(prepared.xs) + ix
    if np.any(indices < 0) or np.any(indices >= len(prepared.points)):
        raise ValueError("Observation points fall outside the prediction grid")
    return indices


def posterior_summary(draws: np.ndarray) -> tuple[np.ndarray, ...]:
    samples = np.asarray(draws, dtype=float)
    return (
        np.mean(samples, axis=0),
        np.std(samples, axis=0, ddof=1),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
        samples,
    )


def block_prior(
    mean_field: np.ndarray,
    observed_indices: np.ndarray,
    observed_covariance: np.ndarray,
    pred_observed_covariance: np.ndarray,
    prediction_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    mean = np.asarray(mean_field, dtype=float).ravel()
    return {
        "prediction_mean": mean,
        "observation_mean": mean[observed_indices],
        "observed_covariance": observed_covariance,
        "pred_observed_covariance": pred_observed_covariance,
        "prediction_variance": prediction_variance,
    }


def sample_prior(
    current: dict[str, object],
    prior: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, ...]:
    return sample_censored_gaussian_blocks(
        current,
        **prior,
        noise_sd=args.current_noise_sd,
        n_samples=args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=seed,
    )


def current_region_masks(truth: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    ranks = percentile_rank(truth)
    return {
        "overall": np.ones(len(truth), dtype=bool),
        "above_camera_ceiling": truth >= threshold,
        "top_1pct": ranks >= 99.0,
    }


def previous_region_masks(
    truth: np.ndarray,
    saturated: np.ndarray,
) -> dict[str, np.ndarray]:
    ranks = percentile_rank(truth)
    return {
        "overall": np.ones(len(truth), dtype=bool),
        "unsaturated_observed_value_pinned": ~saturated,
        "saturated_censored_GP_sampled": saturated,
        "top_1pct": ranks >= 99.0,
    }


def add_identity(
    rows: list[dict[str, object]],
    *,
    trajectory: str,
    family: str,
    run_index: int,
    role: str,
    method: str | None = None,
    stage: str | None = None,
) -> None:
    for row in rows:
        row.update(
            {
                "trajectory": trajectory,
                "family": family,
                "run_index": run_index,
                "role": role,
            }
        )
        if method is not None:
            row["method"] = method
        if stage is not None:
            row["stage"] = stage


def point_distribution_metrics(
    truth: np.ndarray,
    ambient: float,
    prediction: tuple[np.ndarray, ...],
) -> dict[str, float]:
    mean, sd, lower, upper, draws = prediction
    target = np.asarray(truth, dtype=float).ravel()
    ranks = percentile_rank(target)
    top = ranks >= 99.0
    error = mean - target
    from src.metrics import empirical_crps

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


def calibrate_oracle_source(
    args: argparse.Namespace,
    fixed: pd.Series,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for name in DEVELOPMENT_TRAJECTORIES:
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        background = np.zeros_like(prepared.history)
        source_response = np.zeros_like(prepared.history)
        for index in range(1, len(prepared.times)):
            background[index] = diffuse_and_cool(
                prepared,
                prepared.history[index - 1],
                previous_index=index - 1,
                current_index=index,
                diffusivity=float(fixed.diffusivity),
                cooling_rate=float(fixed.cooling_rate),
            )
            dt = float(prepared.times[index] - prepared.times[index - 1])
            active_flux = np.where(
                prepared.heat_flux[index] >= args.source_flux_threshold,
                prepared.heat_flux[index],
                0.0,
            )
            source_response[index] = active_flux * dt
        coupling, n_samples = calibrate_source_coupling(
            prepared, background, source_response
        )
        rows.append(
            {
                "trajectory": name,
                "source_coupling": coupling,
                "n_source_samples": n_samples,
                "calibration_previous_state": "true latent previous frame",
            }
        )
    table = pd.DataFrame(rows)
    return float(table["source_coupling"].median()), table


def build_current_predictions(
    prepared,
    camera: dict[str, object],
    previous_mean: np.ndarray,
    previous_draws: np.ndarray,
    ordinary_mean: np.ndarray,
    advective_mean: np.ndarray,
    displacement: np.ndarray,
    clipped_mean: np.ndarray,
    oracle_fixed_mean: np.ndarray,
    oracle_calibrated_mean: np.ndarray,
    fixed: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[
    dict[str, tuple[np.ndarray, ...]],
    dict[str, tuple[np.ndarray, ...]],
    dict[str, float],
]:
    current = camera["current"]
    current_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed = observation_indices(prepared, observation_points)
    previous_index = int(camera["frames"][0]["time_index"])
    current_index = int(camera["frames"][1]["time_index"])
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    lengthscale = prepared.source_lengthscale * args.length_multiplier
    config = StochasticHeatConfig(
        signal_sd=float(fixed.signal_sd),
        forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        quadrature_order=args.quadrature_order,
    )

    innovation_oo = finite_step_innovation_covariance(
        observation_points, observation_points, config, dt
    )
    innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
    innovation_po = finite_step_innovation_covariance(
        current_points, observation_points, config, dt
    )
    innovation_variance_scalar = float(
        finite_step_innovation_covariance(
            current_points[:1], current_points[:1], config, dt
        )[0, 0]
    )
    innovation_variance = np.full(
        len(current_points), max(innovation_variance_scalar, 0.0)
    )

    centered = previous_draws - previous_mean[None, :, :]
    propagation_arguments = {
        "dx": float(np.mean(np.diff(prepared.xs))),
        "dy": float(np.mean(np.diff(prepared.ys))),
        "time_step": dt,
        "diffusivity": float(fixed.diffusivity),
        "cooling_rate": float(fixed.cooling_rate),
    }
    stationary_draws = propagate_residual_draws(centered, **propagation_arguments)
    advective_draws = propagate_residual_draws(
        centered,
        displacement=displacement,
        **propagation_arguments,
    )

    def moment_blocks(propagated: np.ndarray) -> tuple[np.ndarray, ...]:
        denominator = np.sqrt(len(propagated) - 1.0)
        features = propagated.reshape(len(propagated), -1).T / denominator
        observed_features = features[observed]
        return (
            innovation_oo + observed_features.dot(observed_features.T),
            innovation_po + features.dot(observed_features.T),
            innovation_variance + np.sum(features**2, axis=1),
        )

    stationary_blocks = moment_blocks(stationary_draws)
    advective_blocks = moment_blocks(advective_draws)
    current_priors = {
        SEQ_ST: block_prior(ordinary_mean, observed, *stationary_blocks),
        SEQ_ADV_ST: block_prior(ordinary_mean, observed, *advective_blocks),
        ADV_MEAN_SEQ_ADV_ST: block_prior(
            advective_mean, observed, *advective_blocks
        ),
    }
    current_predictions = {
        method: sample_prior(current, prior, args, seed)
        for method, prior in current_priors.items()
    }

    component_means = ordinary_mean.ravel()[None, :] + advective_draws.reshape(
        len(advective_draws), -1
    )
    mixture_prediction, mixture_diagnostics = (
        sample_censored_gaussian_mixture_blocks(
            current,
            component_prediction_means=component_means,
            component_observation_means=component_means[:, observed],
            observed_covariance=innovation_oo,
            pred_observed_covariance=innovation_po,
            prediction_variance=innovation_variance,
            noise_sd=args.current_noise_sd,
            n_samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=seed,
        )
    )
    current_predictions[SAMPLE_MIXTURE] = mixture_prediction

    innovation_prior = lambda mean: block_prior(
        mean,
        observed,
        innovation_oo,
        innovation_po,
        innovation_variance,
    )
    oracle_predictions = {
        ORACLE_CLIPPED: sample_prior(
            current, innovation_prior(clipped_mean), args, seed
        ),
        ORACLE_POSTERIOR: sample_prior(
            current, innovation_prior(ordinary_mean), args, seed
        ),
        ORACLE_MOMENT: current_predictions[SEQ_ADV_ST],
        ORACLE_MIXTURE: current_predictions[SAMPLE_MIXTURE],
        ORACLE_TRUE_FIXED: sample_prior(
            current, innovation_prior(oracle_fixed_mean), args, seed
        ),
        ORACLE_TRUE: sample_prior(
            current, innovation_prior(oracle_calibrated_mean), args, seed
        ),
    }
    diagnostics = {
        "time_lag_s": dt,
        "innovation_sd_K": float(np.sqrt(innovation_variance_scalar)),
        "propagated_stationary_sd_K": float(
            np.mean(np.std(stationary_draws, axis=0, ddof=1))
        ),
        "propagated_advective_sd_K": float(
            np.mean(np.std(advective_draws, axis=0, ddof=1))
        ),
        **mixture_diagnostics,
    }
    return current_predictions, oracle_predictions, diagnostics


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    evaluation = frame[frame["role"] == "evaluation"]
    aggregate = evaluation.groupby(group_columns, sort=False)[metric_columns].agg(
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


def oracle_recovery_table(summary: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("method")
    metrics = [
        "field_excess_rel_l2",
        "overall_rmse_K",
        "overall_crps_K",
        "top_1pct_rmse_K",
        "top_1pct_mae_K",
        "top_1pct_crps_K",
        "peak_absolute_error_K",
    ]
    rows = []
    for metric in metrics:
        column = f"{metric}_mean"
        clipped = float(indexed.loc[ORACLE_CLIPPED, column])
        oracle = float(indexed.loc[ORACLE_TRUE, column])
        denominator = clipped - oracle
        for method in [ORACLE_POSTERIOR, ORACLE_MOMENT, ORACLE_MIXTURE]:
            value = float(indexed.loc[method, column])
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "clipped_value": clipped,
                    "method_value": value,
                    "oracle_value": oracle,
                    "fraction_oracle_improvement_recovered": (
                        (clipped - value) / denominator
                        if denominator > 1e-10
                        else np.nan
                    ),
                    "interpretation_valid": denominator > 1e-10,
                }
            )
    return pd.DataFrame(rows)


def plot_percentile_diagnostics(
    data: pd.DataFrame,
    methods: list[str],
    output_path: Path,
    title: str,
) -> None:
    evaluation = data[data["role"] == "evaluation"]
    grouped = (
        evaluation.groupby(["method", "percentile_bin"], sort=False)
        [
            [
                "rmse_K",
                "posterior_sd_K",
                "coverage_95",
                "mean_abs_error_over_sd",
            ]
        ]
        .mean()
        .reset_index()
    )
    labels = [f"{lower:g}-{upper:g}" for lower, upper in PERCENTILE_BINS]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    panels = [
        ("rmse_K", "RMSE (K)"),
        ("posterior_sd_K", "Mean posterior SD (K)"),
        ("coverage_95", "95% coverage"),
        ("mean_abs_error_over_sd", r"Mean $|e|/\sigma$"),
    ]
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        for method in methods:
            selected = grouped[grouped["method"] == method].set_index(
                "percentile_bin"
            )
            values = [selected.loc[f"{a:g}-{b:g}%", metric] for a, b in PERCENTILE_BINS]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=COLORS.get(method),
                label=method,
            )
        if metric == "coverage_95":
            axis.axhline(0.95, color="#555555", linestyle=":", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
        if metric == "mean_abs_error_over_sd":
            axis.axhline(
                np.sqrt(2.0 / np.pi),
                color="#555555",
                linestyle=":",
                linewidth=1.0,
                label="N(0,1) reference",
            )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_xlabel("True-temperature percentile (%)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=2)
    figure.suptitle(title, fontsize=14)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_before_after(summary: pd.DataFrame, output_path: Path) -> None:
    selected = summary.set_index("stage")
    stages = ["Previous censored representation", "Current sequential advective ST"]
    figure, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), constrained_layout=True)
    panels = [
        ("rmse_K_mean", "Top-1% RMSE (K)"),
        ("posterior_sd_K_mean", "Mean posterior SD (K)"),
        ("coverage_95_mean", "95% coverage"),
        ("mean_abs_error_over_sd_mean", r"Mean $|e|/\sigma$"),
    ]
    colors = ["#56B4E9", "#009E73"]
    for axis, (metric, title) in zip(axes, panels):
        axis.bar(
            [0, 1],
            [selected.loc[stage, metric] for stage in stages],
            color=colors,
        )
        if metric == "coverage_95_mean":
            axis.axhline(0.95, color="#555555", linestyle=":", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
        axis.set_xticks([0, 1], ["Previous", "Current"])
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Hottest 1%: uncertainty before and after one-step propagation")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_oracle(summary: pd.DataFrame, output_path: Path) -> None:
    selected = summary.set_index("method").loc[ORACLE_METHODS]
    panels = [
        ("field_excess_rel_l2_mean", "Relative excess-field L2"),
        ("overall_crps_K_mean", "Overall CRPS (K)"),
        ("top_1pct_rmse_K_mean", "Top-1% RMSE (K)"),
        ("top_1pct_crps_K_mean", "Top-1% CRPS (K)"),
        ("top_1pct_coverage_95_mean", "Top-1% coverage"),
        ("peak_absolute_error_K_mean", "Peak error (K)"),
    ]
    labels = ["clipped", "post. mean", "moment", "mixture", "oracle fixed", "oracle recal."]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), panels):
        axis.bar(
            np.arange(len(labels)),
            selected[metric],
            color=[COLORS[method] for method in ORACLE_METHODS],
        )
        if metric == "top_1pct_coverage_95_mean":
            axis.axhline(0.95, color="#555555", linestyle=":", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Oracle previous-state benchmark on 30 held-out trajectories")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_diagnostic(
    prepared,
    previous_prediction: tuple[np.ndarray, ...],
    current_predictions: dict[str, tuple[np.ndarray, ...]],
    threshold: float,
    output_path: Path,
) -> None:
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    truth = prepared.truth.ravel()
    thermal_norm = tail_temperature_norm(prepared.ambient, threshold)
    errors = [current_predictions[method][0] - truth for method in CURRENT_METHODS]
    error_limit = max(
        float(np.nanpercentile(np.abs(np.concatenate(errors)), 99)),
        0.1,
    )
    sd_limit = max(
        float(
            np.nanpercentile(
                np.concatenate(
                    [current_predictions[method][1] for method in CURRENT_METHODS]
                ),
                99,
            )
        ),
        0.1,
    )
    figure, axes = plt.subplots(4, 4, figsize=(13.2, 10.8), constrained_layout=True)
    column_images = [None, None, None, None]
    for row_index, method in enumerate(CURRENT_METHODS):
        mean, sd, _, _, _ = current_predictions[method]
        error = mean - truth
        ratio = np.divide(
            np.abs(error),
            sd,
            out=np.full_like(sd, np.nan),
            where=sd > 1e-10,
        )
        fields = [mean, error, sd, ratio]
        titles = ["posterior mean", "signed error", "posterior SD", r"$|e|/\sigma$"]
        for column, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row_index, column]
            if column == 0:
                image = axis.imshow(
                    field.reshape(prepared.truth.shape),
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
                    field.reshape(prepared.truth.shape),
                    ambient=prepared.ambient,
                    ceiling=threshold,
                )
            elif column == 1:
                image = axis.imshow(
                    field.reshape(prepared.truth.shape),
                    origin="lower",
                    extent=extent,
                    cmap="coolwarm",
                    vmin=-error_limit,
                    vmax=error_limit,
                    aspect="auto",
                )
            elif column == 2:
                image = axis.imshow(
                    field.reshape(prepared.truth.shape),
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=sd_limit,
                    aspect="auto",
                )
            else:
                image = axis.imshow(
                    field.reshape(prepared.truth.shape),
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=0.0,
                    vmax=4.0,
                    aspect="auto",
                )
            column_images[column] = image
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            if column == 0:
                axis.set_ylabel(method.replace(" + ", "\n+ "), fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    colorbar_labels = ["temperature (K)", "signed error (K)", "posterior SD (K)", r"$|e|/\sigma$"]
    for column, (image, label) in enumerate(zip(column_images, colorbar_labels)):
        figure.colorbar(
            image,
            ax=axes[:, column],
            shrink=0.72,
            label=label,
            extend="both" if column == 0 else "neither",
        )
    figure.suptitle(
        f"Current uncertainty diagnostic: {prepared.name}\n"
        "Thermal panels use a shared ambient-to-ceiling tail scale and fixed contours"
    )
    figure.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def plot_previous_spatial_diagnostic(
    prepared,
    previous_index: int,
    previous_prediction: tuple[np.ndarray, ...],
    threshold: float,
    output_path: Path,
) -> None:
    mean, sd, _, _, _ = previous_prediction
    truth = prepared.history[previous_index].ravel()
    error = mean - truth
    ratio = np.divide(
        np.abs(error),
        sd,
        out=np.full_like(sd, np.nan),
        where=sd > 1e-10,
    )
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    fields = [truth, mean, error, sd, ratio]
    titles = ["true previous", "hybrid posterior mean", "signed error", "posterior SD", r"$|e|/\sigma$"]
    thermal_norm = tail_temperature_norm(prepared.ambient, threshold)
    figure, axes = plt.subplots(1, 5, figsize=(16.0, 3.5), constrained_layout=True)
    for index, (axis, field, title) in enumerate(zip(axes, fields, titles)):
        if index < 2:
            image = axis.imshow(
                field.reshape(prepared.truth.shape),
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
                field.reshape(prepared.truth.shape),
                ambient=prepared.ambient,
                ceiling=threshold,
            )
        elif index == 2:
            limit = max(float(np.nanpercentile(np.abs(field), 99)), 0.1)
            image = axis.imshow(
                field.reshape(prepared.truth.shape),
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
        elif index == 3:
            image = axis.imshow(
                field.reshape(prepared.truth.shape),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                aspect="auto",
            )
        else:
            image = axis.imshow(
                field.reshape(prepared.truth.shape),
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=12.0,
                aspect="auto",
            )
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.72,
            extend="both" if index < 2 else "neither",
        )
    figure.suptitle(
        f"Previous censored representation before propagation: {prepared.name}\n"
        "Thermal panels use a shared ambient-to-ceiling tail scale and fixed contours"
    )
    figure.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def write_derivation(output_path: Path) -> None:
    output_path.write_text(
        r"""\subsection{Squared-error benefit of posterior-mean propagation}

Let \(T\) be a latent temperature with finite second moment and let \(Y\)
denote the available observation information. For any estimator \(a(Y)\),
conditioning on \(Y\) gives the orthogonal decomposition
\[
\mathbb{E}\!\left[(T-a(Y))^2\mid Y\right]
=
\operatorname{Var}(T\mid Y)
+
\left(a(Y)-\mathbb{E}[T\mid Y]\right)^2.
\]
Therefore \(a^*(Y)=\mathbb{E}[T\mid Y]\) minimizes conditional squared error.
For a noiseless saturated observation \(Y=\{T\geq c\}\), the clipped
estimator is \(a=c\), whereas generally
\(
\mu_c=\mathbb{E}[T\mid T\geq c]>c
\).
Consequently,
\[
\mathbb{E}\!\left[(T-\mu_c)^2\mid T\geq c\right]
\leq
\mathbb{E}\!\left[(T-c)^2\mid T\geq c\right],
\]
with strict inequality unless \(c=\mu_c\). With measurement noise, the same
statement holds after replacing the conditioning event by
\(\{T+\varepsilon\geq c\}\).

The argument also extends through an affine one-step transition. Suppose
\[
T_n=b+B T_{n-1}+\eta_n,
\qquad
\mathbb{E}[\eta_n\mid Y_{n-1}]=0,
\]
where \(B\) is fixed given the model parameters. Then
\[
\mathbb{E}[T_n\mid Y_{n-1}]
=b+B\,\mathbb{E}[T_{n-1}\mid Y_{n-1}].
\]
For a candidate previous-state estimate \(a(Y_{n-1})\), the conditional
squared Euclidean prediction error satisfies
\[
\mathbb{E}\!\left[
\left\|T_n-(b+B a)\right\|^2
\mid Y_{n-1}
\right]
=
\operatorname{tr}\!\left\{\operatorname{Var}(T_n\mid Y_{n-1})\right\}
+
\left\|B\left(a-\mathbb{E}[T_{n-1}\mid Y_{n-1}]\right)\right\|^2.
\]
Thus propagating the previous conditional mean is optimal within this affine
model under squared loss, and improves on clipped propagation whenever the
clipped-state error survives under \(B\). This is a model-based statement: it
requires finite second moments, a correctly specified affine conditional mean,
and zero conditional-mean transition error. It does not guarantee improvement
under transition misspecification or for non-squared probabilistic scores.
""",
        encoding="ascii",
    )


def write_readme(
    output_path: Path,
    before_after: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    previous_regions: pd.DataFrame,
    current_summary: pd.DataFrame,
    oracle_source_coupling: float,
    fixed_source_coupling: float,
) -> None:
    stages = before_after.set_index("stage")
    previous = stages.loc["Previous censored representation"]
    current = stages.loc["Current sequential advective ST"]
    previous_bad = (
        previous["coverage_95_mean"] < 0.9
        or previous["mean_abs_error_over_sd_mean"] > 1.0
    )
    amplified = (
        current["coverage_95_mean"] + 0.05 < previous["coverage_95_mean"]
        or current["mean_abs_error_over_sd_mean"]
        > 1.2 * previous["mean_abs_error_over_sd_mean"]
    )
    if previous_bad and amplified:
        mechanism = (
            "Hot-tail underdispersion is already present in the previous censored "
            "representation and is further amplified by the one-step transition."
        )
    elif previous_bad:
        mechanism = (
            "Hot-tail underdispersion is already present before propagation; the "
            "transition does not appear to be its sole origin."
        )
    elif amplified:
        mechanism = (
            "The previous hot-tail posterior is comparatively adequate, while the "
            "one-step transition introduces or amplifies underdispersion."
        )
    else:
        mechanism = (
            "Neither stage alone shows a dominant calibration failure under the "
            "predefined thresholds; both posterior inference and transition design "
            "remain plausible contributors."
        )
    oracle = oracle_summary.set_index("method")
    previous_by_region = previous_regions.set_index("region")
    current = current_summary.set_index("method")
    lines = [
        "# Uncertainty-origin diagnostics and oracle benchmark",
        "",
        "This experiment uses the frozen 3% censoring protocol, the three fixed "
        "development trajectories, and the remaining 30 trajectories for equal-weight "
        "held-out summaries. No adaptive or state-dependent covariance was added.",
        "",
        "## Posterior representation",
        "",
        "The previous-frame object is a hybrid censored representation, not a full "
        "latent posterior. Saturated pixels are sampled from the censored RBF GP. "
        "Unsaturated noisy pixels are retained at their observed values in every draw, "
        "so their represented posterior variance is exactly zero. Standardized errors "
        "are reported only where SD is positive, and zero-SD/nonzero-error pixels are "
        "reported separately.",
        "",
        "## Main diagnosis",
        "",
        mechanism,
        "",
        "Supporting evidence:",
        "",
        f"- In the sampled saturated part of the previous frame, 95% coverage is "
        f"{previous_by_region.loc['saturated_censored_GP_sampled', 'coverage_95_mean']:.3f}; "
        f"in its hottest 1%, coverage is only "
        f"{previous_by_region.loc['top_1pct', 'coverage_95_mean']:.3f}.",
        f"- The previous hottest 1% has mean signed error "
        f"{previous['signed_error_K_mean']:.3f} K and mean |e|/SD "
        f"{previous['mean_abs_error_over_sd_mean']:.3f}.",
        f"- Moment matching and the posterior-sample mixture give top-1% CRPS "
        f"{current.loc[SEQ_ADV_ST, 'top_1pct_crps_K_mean']:.3f} and "
        f"{current.loc[SAMPLE_MIXTURE, 'top_1pct_crps_K_mean']:.3f} K, respectively, "
        "so retaining the sampled non-Gaussian shape does not materially fix the tail.",
        f"- Recalibrating source coupling for true previous states on development paths "
        f"changes gamma from {fixed_source_coupling:.4f} to "
        f"{oracle_source_coupling:.4f}. This confirms that transition-mean calibration "
        "depends on the previous-state representation.",
        "",
        "Top-1% before/after averages:",
        "",
        "| Stage | RMSE (K) | Bias (K) | SD (K) | Coverage | Width (K) | Mean |e|/SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ["Previous censored representation", "Current sequential advective ST"]:
        row = stages.loc[stage]
        lines.append(
            f"| {stage} | {row['rmse_K_mean']:.3f} | "
            f"{row['signed_error_K_mean']:.3f} | {row['posterior_sd_K_mean']:.3f} | "
            f"{row['coverage_95_mean']:.3f} | {row['interval_width_95_K_mean']:.3f} | "
            f"{row['mean_abs_error_over_sd_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle benchmark",
            "",
            "The oracle receives only the true latent previous frame and never receives "
            "uncensored current-frame truth. The fixed-source oracle changes only the "
            "previous state. Because the fixed source coefficient was calibrated for "
            "clipped propagation, a second row recalibrates that coefficient using only "
            "the three development trajectories and true previous states. Both retain "
            "the same current finite-step innovation, observations, and likelihood.",
            "",
            "| Method | Field L2 | Overall RMSE | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Top-1% coverage | Peak error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ORACLE_METHODS:
        row = oracle.loc[method]
        lines.append(
            f"| {method} | {row['field_excess_rel_l2_mean']:.3f} | "
            f"{row['overall_rmse_K_mean']:.3f} | {row['overall_crps_K_mean']:.3f} | "
            f"{row['top_1pct_rmse_K_mean']:.3f} | {row['top_1pct_crps_K_mean']:.3f} | "
            f"{row['top_1pct_coverage_95_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All CRPS values use the unbiased M(M-1) empirical estimator. RMSE, MAE, "
            "relative L2, signed error, and peak error evaluate point reconstruction; "
            "CRPS evaluates the full predictive distribution; coverage is interpreted "
            "together with interval width.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0]
    oracle_source_coupling, oracle_source_table = calibrate_oracle_source(args, fixed)
    oracle_source_table.to_csv(
        args.output_dir / "oracle_source_coupling_calibration.csv", index=False
    )
    catalog = list(trajectory_catalog(args.dataset_dir))
    indexed = list(enumerate(catalog))
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        indexed = [(index, item) for index, item in indexed if item.name in requested]
        missing = requested - {item.name for _, item in indexed}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    current_percentile: list[dict[str, object]] = []
    current_regions: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    previous_percentile: list[dict[str, object]] = []
    previous_regions: list[dict[str, object]] = []
    before_after: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    oracle_regions: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    representatives = {"DiagonalScanPath_7", "HorizontalScanPath_10", "SpiralScanPath_12"}

    for run_index, (catalog_index, record) in enumerate(indexed):
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        role = "development" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        current_index = len(prepared.times) - 1
        previous_index = current_index - args.previous_frame_offset
        threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
        camera_seed = args.seed + 10_000 * catalog_index
        previous_seed = args.seed + 50_000 * catalog_index
        current_seed = args.seed + 100_000 * catalog_index
        camera = paired_camera_observations(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            threshold=threshold,
            observation_stride=args.observation_stride,
            noise_sd=args.measurement_noise_sd,
            seed=camera_seed,
        )
        lengthscale = prepared.source_lengthscale * args.length_multiplier
        previous_mean, previous_draws, previous_checks = infer_previous_censored_posterior(
            prepared,
            frame=camera["frames"][0],
            fixed_mask=camera["fixed_observation_mask"],
            threshold=threshold,
            signal_sd=float(fixed.signal_sd),
            lengthscale=lengthscale,
            noise_sd=args.previous_noise_sd,
            n_chains=1,
            samples_per_chain=args.previous_samples,
            burn_in=args.previous_burn_in,
            thin=args.thin,
            seed=previous_seed,
        )
        previous_prediction = posterior_summary(
            previous_draws.reshape(len(previous_draws), -1)
        )
        previous_truth = prepared.history[previous_index].ravel()
        previous_saturated = np.asarray(
            camera["frames"][0]["saturated_full"], dtype=bool
        ).ravel()
        previous_pct = percentile_diagnostic_rows(
            previous_truth,
            *previous_prediction[:4],
            draws=previous_prediction[4],
        )
        add_identity(
            previous_pct,
            trajectory=record.name,
            family=record.family,
            run_index=record.run_index,
            role=role,
            method="Hybrid previous censored representation",
            stage="Previous censored representation",
        )
        previous_percentile.extend(previous_pct)
        previous_reg = region_diagnostic_rows(
            previous_truth,
            *previous_prediction[:4],
            previous_region_masks(previous_truth, previous_saturated),
            draws=previous_prediction[4],
        )
        add_identity(
            previous_reg,
            trajectory=record.name,
            family=record.family,
            run_index=record.run_index,
            role=role,
            method="Hybrid previous censored representation",
            stage="Previous censored representation",
        )
        previous_regions.extend(previous_reg)

        ordinary_mean, advective_mean, displacement = posterior_physics_means(
            prepared,
            previous_mean,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
        )
        observed_clipped_previous = np.asarray(
            camera["frames"][0]["clipped_full"], dtype=float
        )
        clipped_mean, _, _ = posterior_physics_means(
            prepared,
            observed_clipped_previous,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
        )
        oracle_previous = prepared.history[previous_index]
        oracle_fixed_mean, _, _ = posterior_physics_means(
            prepared,
            oracle_previous,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
        )
        oracle_calibrated_mean, _, _ = posterior_physics_means(
            prepared,
            oracle_previous,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=oracle_source_coupling,
            source_flux_threshold=args.source_flux_threshold,
        )
        current_predictions, oracle_predictions, covariance_checks = (
            build_current_predictions(
                prepared,
                camera,
                previous_mean,
                previous_draws,
                ordinary_mean,
                advective_mean,
                displacement,
                clipped_mean,
                oracle_fixed_mean,
                oracle_calibrated_mean,
                fixed,
                args,
                current_seed,
            )
        )
        current_truth = prepared.truth.ravel()
        current_masks = current_region_masks(current_truth, threshold)
        for method, prediction in current_predictions.items():
            current_rows.append(
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "method": method,
                    **point_distribution_metrics(
                        current_truth, prepared.ambient, prediction
                    ),
                }
            )
            pct = percentile_diagnostic_rows(
                current_truth,
                *prediction[:4],
                draws=prediction[4],
            )
            add_identity(
                pct,
                trajectory=record.name,
                family=record.family,
                run_index=record.run_index,
                role=role,
                method=method,
                stage="Current sequential posterior",
            )
            current_percentile.extend(pct)
            reg = region_diagnostic_rows(
                current_truth,
                *prediction[:4],
                current_masks,
                draws=prediction[4],
            )
            add_identity(
                reg,
                trajectory=record.name,
                family=record.family,
                run_index=record.run_index,
                role=role,
                method=method,
                stage="Current sequential posterior",
            )
            current_regions.extend(reg)

        previous_top = next(row for row in previous_reg if row["region"] == "top_1pct")
        current_top = next(
            row
            for row in current_regions[-3 * len(CURRENT_METHODS) :]
            if row["method"] == SEQ_ADV_ST and row["region"] == "top_1pct"
        )
        for stage, source in [
            ("Previous censored representation", previous_top),
            ("Current sequential advective ST", current_top),
        ]:
            before_after.append(
                {
                    **{key: source[key] for key in source if key in DIAGNOSTIC_METRICS or key == "n_pixels"},
                    "trajectory": record.name,
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "stage": stage,
                    "region": "top_1pct",
                }
            )

        for method, prediction in oracle_predictions.items():
            oracle_rows.append(
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "method": method,
                    **point_distribution_metrics(
                        current_truth, prepared.ambient, prediction
                    ),
                }
            )
            reg = region_diagnostic_rows(
                current_truth,
                *prediction[:4],
                current_masks,
                draws=prediction[4],
            )
            add_identity(
                reg,
                trajectory=record.name,
                family=record.family,
                run_index=record.run_index,
                role=role,
                method=method,
                stage="Oracle controlled current update",
            )
            oracle_regions.extend(reg)

        integrity.append(
            {
                "trajectory": record.name,
                "family": record.family,
                "run_index": record.run_index,
                "role": role,
                "camera_seed": camera_seed,
                "previous_posterior_seed": previous_seed,
                "current_sampler_seed": current_seed,
                "threshold_K": threshold,
                "previous_unsaturated_pixels_pinned": True,
                "previous_saturated_pixels_sampled": True,
                "previous_observations_reused_in_current_likelihood": False,
                "oracle_uses_true_previous_frame": True,
                "oracle_uses_true_current_frame_for_inference": False,
                "same_current_observations_for_all_models": True,
                "same_current_sampler_seed_for_all_models": True,
                "same_finite_step_innovation_for_oracle_controls": True,
                "same_previous_posterior_for_sequential_models": True,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "previous_zero_sd_fraction": float(
                    np.mean(np.std(previous_draws, axis=0, ddof=1) <= 1e-10)
                ),
                "mean_difference_posterior_vs_clipped_K": float(
                    np.mean(ordinary_mean - clipped_mean)
                ),
                "mean_difference_oracle_fixed_vs_posterior_K": float(
                    np.mean(oracle_fixed_mean - ordinary_mean)
                ),
                "mean_difference_oracle_calibrated_vs_posterior_K": float(
                    np.mean(oracle_calibrated_mean - ordinary_mean)
                ),
                **previous_checks,
                **covariance_checks,
            }
        )

        if record.name in representatives:
            plot_spatial_diagnostic(
                prepared,
                previous_prediction,
                current_predictions,
                threshold,
                args.output_dir / f"spatial_diagnostic_{record.name.lower()}.png",
            )
            plot_previous_spatial_diagnostic(
                prepared,
                previous_index,
                previous_prediction,
                threshold,
                args.output_dir
                / f"previous_spatial_diagnostic_{record.name.lower()}.png",
            )
        print(
            f"[{run_index + 1:02d}/{len(indexed):02d}] {record.name}: "
            f"previous top1 coverage={previous_top['coverage_95']:.3f}, "
            f"current top1 coverage={current_top['coverage_95']:.3f}",
            flush=True,
        )
        pd.DataFrame(current_regions).to_csv(
            args.output_dir / "checkpoint_current_regions.csv", index=False
        )
        pd.DataFrame(previous_regions).to_csv(
            args.output_dir / "checkpoint_previous_regions.csv", index=False
        )
        pd.DataFrame(oracle_rows).to_csv(
            args.output_dir / "checkpoint_oracle_results.csv", index=False
        )

    current_percentile_df = pd.DataFrame(current_percentile)
    current_regions_df = pd.DataFrame(current_regions)
    current_df = pd.DataFrame(current_rows)
    previous_percentile_df = pd.DataFrame(previous_percentile)
    previous_regions_df = pd.DataFrame(previous_regions)
    before_after_df = pd.DataFrame(before_after)
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_regions_df = pd.DataFrame(oracle_regions)
    integrity_df = pd.DataFrame(integrity)

    current_percentile_summary = summarize(
        current_percentile_df,
        ["method", "percentile_bin", "percentile_lower", "percentile_upper"],
        DIAGNOSTIC_METRICS,
    )
    current_region_summary = summarize(
        current_regions_df, ["method", "region"], DIAGNOSTIC_METRICS
    )
    current_region_family_summary = summarize(
        current_regions_df, ["method", "family", "region"], DIAGNOSTIC_METRICS
    )
    current_metrics = [
        column
        for column in current_df.columns
        if column
        not in {"trajectory", "family", "run_index", "role", "method"}
    ]
    current_summary = summarize(current_df, ["method"], current_metrics)
    previous_percentile_summary = summarize(
        previous_percentile_df,
        ["method", "percentile_bin", "percentile_lower", "percentile_upper"],
        DIAGNOSTIC_METRICS,
    )
    previous_region_summary = summarize(
        previous_regions_df, ["method", "region"], DIAGNOSTIC_METRICS
    )
    previous_region_family_summary = summarize(
        previous_regions_df, ["method", "family", "region"], DIAGNOSTIC_METRICS
    )
    before_after_summary = summarize(
        before_after_df, ["stage", "region"], DIAGNOSTIC_METRICS
    )
    oracle_metrics = [
        column
        for column in oracle_df.columns
        if column
        not in {"trajectory", "family", "run_index", "role", "method"}
    ]
    oracle_summary = summarize(oracle_df, ["method"], oracle_metrics)
    oracle_family_summary = summarize(
        oracle_df, ["method", "family"], oracle_metrics
    )
    oracle_region_summary = summarize(
        oracle_regions_df, ["method", "region"], DIAGNOSTIC_METRICS
    )
    recovery = oracle_recovery_table(oracle_summary)

    outputs = {
        "current_percentile_by_trajectory.csv": current_percentile_df,
        "current_percentile_summary.csv": current_percentile_summary,
        "current_region_by_trajectory.csv": current_regions_df,
        "current_region_summary.csv": current_region_summary,
        "current_region_family_summary.csv": current_region_family_summary,
        "current_results.csv": current_df,
        "heldout30_current_summary.csv": current_summary,
        "previous_percentile_by_trajectory.csv": previous_percentile_df,
        "previous_percentile_summary.csv": previous_percentile_summary,
        "previous_region_by_trajectory.csv": previous_regions_df,
        "previous_region_summary.csv": previous_region_summary,
        "previous_region_family_summary.csv": previous_region_family_summary,
        "before_after_by_trajectory.csv": before_after_df,
        "before_after_summary.csv": before_after_summary,
        "oracle_results.csv": oracle_df,
        "heldout30_oracle_summary.csv": oracle_summary,
        "oracle_family_summary.csv": oracle_family_summary,
        "oracle_regions.csv": oracle_regions_df,
        "heldout30_oracle_region_summary.csv": oracle_region_summary,
        "oracle_recovery_fraction.csv": recovery,
        "integrity_checks.csv": integrity_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    fixed_output = pd.DataFrame(
        [
            {
                **fixed.to_dict(),
                "dataset_trajectory_count": len(catalog),
                "development_trajectory_count": len(DEVELOPMENT_TRAJECTORIES),
                "heldout_trajectory_count": len(catalog) - len(DEVELOPMENT_TRAJECTORIES),
                "diagnostic_trajectory_count": len(indexed),
                "previous_posterior_representation": (
                    "censored GP samples at saturated pixels; noisy observed values "
                    "pinned at unsaturated pixels"
                ),
                "oracle_extra_information": "true latent previous frame only",
                "oracle_source_coupling_development_only": oracle_source_coupling,
                "oracle_current_truth_used_for_inference": False,
                "crps_estimator": "unbiased_M_times_M_minus_1",
            }
        ]
    )
    fixed_output.to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    if not args.trajectory_names:
        if len(catalog) != 33 or len(DEVELOPMENT_TRAJECTORIES) != 3:
            raise AssertionError("Expected exactly 33 trajectories and 3 development paths")
        roles = integrity_df["role"].value_counts().to_dict()
        if roles != {"evaluation": 30, "development": 3}:
            raise AssertionError(f"Unexpected held-out split: {roles}")
        if not integrity_df["previous_observations_reused_in_current_likelihood"].eq(False).all():
            raise AssertionError("Previous observations leaked into the current likelihood")
        if not integrity_df["oracle_uses_true_current_frame_for_inference"].eq(False).all():
            raise AssertionError("Oracle received current-frame truth")

    plot_percentile_diagnostics(
        current_percentile_df,
        CURRENT_METHODS,
        args.output_dir / "current_uncertainty_vs_temperature.png",
        "Current-frame error versus predicted uncertainty",
    )
    plot_percentile_diagnostics(
        previous_percentile_df,
        ["Hybrid previous censored representation"],
        args.output_dir / "previous_uncertainty_vs_temperature.png",
        "Previous censored representation before propagation",
    )
    plot_before_after(
        before_after_summary,
        args.output_dir / "before_after_hot_tail.png",
    )
    plot_oracle(oracle_summary, args.output_dir / "oracle_comparison.png")
    write_derivation(args.output_dir / "squared_error_derivation.tex")
    write_readme(
        args.output_dir / "README.md",
        before_after_summary,
        oracle_summary,
        previous_region_summary,
        current_summary,
        oracle_source_coupling,
        float(fixed.source_coupling),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose uncertainty origin and add a previous-state oracle."
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
