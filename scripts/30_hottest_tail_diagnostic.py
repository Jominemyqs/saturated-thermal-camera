from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from src.censored_gp import RBFConfig, rbf_covariance
from src.dense_censored_gp import (
    sample_censored_gaussian_blocks,
    sample_censored_gaussian_mixture_blocks,
)
from src.metrics import empirical_crps
from src.stochastic_heat_gp import (
    StochasticHeatConfig,
    finite_step_innovation_covariance,
    propagate_residual_draws,
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


EXPERIMENT_27 = (
    ROOT / "outputs" / "by_experiment" / "27_full_posterior_sequential"
)
DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "28_hottest_tail_diagnostic"
)

SEQUENTIAL_SCRIPT = ROOT / "scripts" / "29_thermal_full_posterior_sequential.py"
SPEC = importlib.util.spec_from_file_location("experiment_27_module", SEQUENTIAL_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SEQUENTIAL_SCRIPT}")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)

MOMENT_MATCHED = EXPERIMENT.MOMENT_MATCHED
FULL_MIXTURE = EXPERIMENT.FULL_MIXTURE
RBF_REFERENCE = EXPERIMENT.RBF_REFERENCE
MATCHED_RBF = "posterior physics mean + variance-matched RBF (diagnostic)"
MAIN_METHODS = [MOMENT_MATCHED, FULL_MIXTURE, RBF_REFERENCE]
PIXEL_METHODS = [FULL_MIXTURE, RBF_REFERENCE, MATCHED_RBF]
METHOD_LABELS = {
    MOMENT_MATCHED: "moment-matched ST",
    FULL_MIXTURE: "full-mixture ST",
    RBF_REFERENCE: "posterior mean + RBF",
    MATCHED_RBF: "variance-matched RBF",
}
METHOD_COLORS = {
    FULL_MIXTURE: "#009E73",
    RBF_REFERENCE: "#0072B2",
    MATCHED_RBF: "#CC79A7",
}
REGIONS = ["overall", "Q4", "above_camera_ceiling", "top_1pct"]
REGION_LABELS = {
    "overall": "overall",
    "Q4": "Q4",
    "above_camera_ceiling": "above camera ceiling",
    "top_1pct": "top 1%",
}
METRICS = [
    "rmse_K",
    "mae_K",
    "signed_error_K",
    "crps_K",
    "coverage_95",
    "interval_width_95_K",
    "posterior_sd_K",
]
PERCENTILE_BINS = [
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
]


def frozen_value(fixed: pd.Series, name: str, cast=float):
    return cast(fixed[name])


def run_frozen_trajectory(
    dataset_dir: Path,
    record,
    *,
    trajectory_index: int,
    fixed: pd.Series,
) -> dict[str, object]:
    """Reproduce experiment 27 for one trajectory with its original global seed index."""
    nx = frozen_value(fixed, "nx", int)
    ny = frozen_value(fixed, "ny", int)
    fraction_saturated = frozen_value(fixed, "fraction_saturated")
    observation_stride = frozen_value(fixed, "observation_stride", int)
    previous_frame_offset = frozen_value(fixed, "previous_frame_offset", int)
    measurement_noise_sd = frozen_value(fixed, "measurement_noise_sd")
    previous_noise_sd = frozen_value(fixed, "previous_noise_sd")
    current_noise_sd = frozen_value(fixed, "current_noise_sd")
    length_multiplier = frozen_value(fixed, "length_multiplier")
    forcing_length_multiplier = frozen_value(
        fixed, "forcing_length_multiplier"
    )
    quadrature_order = frozen_value(fixed, "quadrature_order", int)
    previous_samples = frozen_value(fixed, "previous_samples", int)
    current_samples = frozen_value(fixed, "current_samples", int)
    seed = frozen_value(fixed, "seed", int)

    prepared, _ = prepare_trajectory(
        dataset_dir,
        record.name,
        nx=nx,
        ny=ny,
        heat_flux_cutoff=300.0,
    )
    current_index = len(prepared.times) - 1
    previous_index = current_index - previous_frame_offset
    threshold = float(
        np.quantile(prepared.truth, 1.0 - fraction_saturated)
    )
    camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=threshold,
        observation_stride=observation_stride,
        noise_sd=measurement_noise_sd,
        seed=seed + 10_000 * trajectory_index,
    )
    current = camera["current"]
    current_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed_indices = EXPERIMENT.observation_indices(
        prepared, observation_points
    )
    lengthscale = prepared.source_lengthscale * length_multiplier
    stochastic_config = StochasticHeatConfig(
        signal_sd=frozen_value(fixed, "signal_sd"),
        forcing_lengthscale=lengthscale * forcing_length_multiplier,
        diffusivity=frozen_value(fixed, "diffusivity"),
        cooling_rate=frozen_value(fixed, "cooling_rate"),
        quadrature_order=quadrature_order,
    )
    previous_mean, previous_draws, _ = infer_previous_censored_posterior(
        prepared,
        frame=camera["frames"][0],
        fixed_mask=camera["fixed_observation_mask"],
        threshold=threshold,
        signal_sd=frozen_value(fixed, "signal_sd"),
        lengthscale=lengthscale,
        noise_sd=previous_noise_sd,
        n_chains=1,
        samples_per_chain=previous_samples,
        burn_in=120,
        thin=1,
        seed=seed + 50_000 * trajectory_index,
    )
    ordinary_mean, _, displacement = posterior_physics_means(
        prepared,
        previous_mean,
        previous_index=previous_index,
        current_index=current_index,
        diffusivity=frozen_value(fixed, "diffusivity"),
        cooling_rate=frozen_value(fixed, "cooling_rate"),
        source_coupling=frozen_value(fixed, "source_coupling"),
        source_flux_threshold=10_000.0,
    )

    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    centered_draws = previous_draws - previous_mean[None, :, :]
    propagated_components = propagate_residual_draws(
        centered_draws,
        dx=dx,
        dy=dy,
        time_step=dt,
        diffusivity=frozen_value(fixed, "diffusivity"),
        cooling_rate=frozen_value(fixed, "cooling_rate"),
        displacement=displacement,
    )
    component_prediction_means = ordinary_mean.ravel()[None, :] + (
        propagated_components.reshape(len(previous_draws), -1)
    )
    component_observation_means = component_prediction_means[
        :, observed_indices
    ]

    innovation_oo = finite_step_innovation_covariance(
        observation_points, observation_points, stochastic_config, dt
    )
    innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
    innovation_po = finite_step_innovation_covariance(
        current_points, observation_points, stochastic_config, dt
    )
    innovation_variance_scalar = float(
        finite_step_innovation_covariance(
            current_points[:1], current_points[:1], stochastic_config, dt
        )[0, 0]
    )
    innovation_variance = np.full(
        len(current_points), max(innovation_variance_scalar, 0.0)
    )

    rbf_config = RBFConfig(
        mean_temp=prepared.ambient,
        signal_sd=frozen_value(fixed, "signal_sd"),
        lengthscale=lengthscale,
        noise_sd=current_noise_sd,
    )
    rbf_oo = rbf_covariance(
        observation_points, observation_points, rbf_config
    )
    rbf_po = rbf_covariance(current_points, observation_points, rbf_config)
    rbf_variance = np.full(len(current_points), rbf_config.signal_sd**2)
    variance_scale = innovation_variance_scalar / rbf_config.signal_sd**2
    flat_mean = ordinary_mean.ravel()
    prediction_seed = seed + 100_000 * trajectory_index

    full_prediction, mixture_diagnostics = (
        sample_censored_gaussian_mixture_blocks(
            current,
            component_prediction_means=component_prediction_means,
            component_observation_means=component_observation_means,
            observed_covariance=innovation_oo,
            pred_observed_covariance=innovation_po,
            prediction_variance=innovation_variance,
            noise_sd=current_noise_sd,
            n_samples=current_samples,
            burn_in=120,
            thin=1,
            seed=prediction_seed,
        )
    )
    rbf_prediction = sample_censored_gaussian_blocks(
        current,
        prediction_mean=flat_mean,
        observation_mean=flat_mean[observed_indices],
        observed_covariance=rbf_oo,
        pred_observed_covariance=rbf_po,
        prediction_variance=rbf_variance,
        noise_sd=current_noise_sd,
        n_samples=current_samples,
        burn_in=120,
        thin=1,
        seed=prediction_seed,
    )
    matched_rbf_prediction = sample_censored_gaussian_blocks(
        current,
        prediction_mean=flat_mean,
        observation_mean=flat_mean[observed_indices],
        observed_covariance=variance_scale * rbf_oo,
        pred_observed_covariance=variance_scale * rbf_po,
        prediction_variance=variance_scale * rbf_variance,
        noise_sd=current_noise_sd,
        n_samples=current_samples,
        burn_in=120,
        thin=1,
        seed=prediction_seed,
    )
    return {
        "prepared": prepared,
        "current": current,
        "threshold": threshold,
        "predictions": {
            FULL_MIXTURE: full_prediction,
            RBF_REFERENCE: rbf_prediction,
            MATCHED_RBF: matched_rbf_prediction,
        },
        "innovation_variance_K2": innovation_variance_scalar,
        "innovation_sd_K": float(np.sqrt(innovation_variance_scalar)),
        "rbf_prior_variance_K2": float(rbf_config.signal_sd**2),
        "rbf_prior_sd_K": float(rbf_config.signal_sd),
        "variance_match_scale": variance_scale,
        "mixture_diagnostics": mixture_diagnostics,
    }


def select_heldout_trajectories(quartile_results: pd.DataFrame) -> pd.DataFrame:
    subset = quartile_results[
        (quartile_results["role"] == "evaluation")
        & (quartile_results["region"] == "top_1pct")
        & quartile_results["method"].isin([FULL_MIXTURE, RBF_REFERENCE])
    ]
    coverage = subset.pivot(
        index="trajectory", columns="method", values="coverage_95"
    )
    coverage["rbf_minus_st_coverage"] = (
        coverage[RBF_REFERENCE] - coverage[FULL_MIXTURE]
    )
    median_difference = float(coverage["rbf_minus_st_coverage"].median())
    choices = {
        "smallest difference": coverage["rbf_minus_st_coverage"].idxmin(),
        "median difference": (
            coverage["rbf_minus_st_coverage"] - median_difference
        ).abs().idxmin(),
        "largest difference": coverage["rbf_minus_st_coverage"].idxmax(),
    }
    rows = []
    for selection, trajectory in choices.items():
        row = coverage.loc[trajectory]
        rows.append(
            {
                "selection": selection,
                "trajectory": trajectory,
                "st_top_1pct_coverage": row[FULL_MIXTURE],
                "rbf_top_1pct_coverage": row[RBF_REFERENCE],
                "rbf_minus_st_coverage": row["rbf_minus_st_coverage"],
            }
        )
    return pd.DataFrame(rows)


def existing_diagnostic_table(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        summary["method"].isin(MAIN_METHODS) & summary["region"].isin(REGIONS)
    ][["method", "region"] + [f"{metric}_mean" for metric in METRICS]].copy()
    selected = selected.rename(
        columns={f"{metric}_mean": metric for metric in METRICS}
    )
    selected["method_label"] = selected["method"].map(METHOD_LABELS)
    rbf = selected[selected["method"] == RBF_REFERENCE].set_index("region")
    selected["rbf_to_method_interval_width_ratio"] = selected.apply(
        lambda row: rbf.loc[row["region"], "interval_width_95_K"]
        / row["interval_width_95_K"],
        axis=1,
    )
    selected["rbf_to_method_posterior_sd_ratio"] = selected.apply(
        lambda row: rbf.loc[row["region"], "posterior_sd_K"]
        / row["posterior_sd_K"],
        axis=1,
    )
    selected["region_label"] = selected["region"].map(REGION_LABELS)
    return selected[
        ["method", "method_label", "region", "region_label"]
        + METRICS
        + [
            "rbf_to_method_interval_width_ratio",
            "rbf_to_method_posterior_sd_ratio",
        ]
    ].sort_values(
        ["region", "method"],
        key=lambda column: column.map(
            {value: index for index, value in enumerate(REGIONS + MAIN_METHODS)}
        ).fillna(len(REGIONS + MAIN_METHODS)),
    )


def percentile_rank(truth: np.ndarray) -> np.ndarray:
    order = np.argsort(truth, kind="mergesort")
    percentiles = np.empty(len(truth), dtype=float)
    percentiles[order] = 100.0 * (np.arange(len(truth)) + 0.5) / len(truth)
    return percentiles


def percentile_rows(
    trajectory: str,
    truth: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, ...]],
) -> list[dict[str, object]]:
    percentiles = percentile_rank(truth)
    rows = []
    for method in PIXEL_METHODS:
        mean, sd, lower, upper, draws = predictions[method]
        error = mean - truth
        crps = empirical_crps(draws, truth)
        covered = (lower <= truth) & (truth <= upper)
        width = upper - lower
        standardized = np.abs(error) / np.maximum(sd, 1e-12)
        for lower_pct, upper_pct in PERCENTILE_BINS:
            mask = (percentiles >= lower_pct) & (percentiles < upper_pct)
            label = f"{lower_pct:g}-{upper_pct:g}%"
            rows.append(
                {
                    "trajectory": trajectory,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "percentile_bin": label,
                    "percentile_lower": lower_pct,
                    "percentile_upper": upper_pct,
                    "n_pixels": int(np.sum(mask)),
                    "rmse_K": float(np.sqrt(np.mean(error[mask] ** 2))),
                    "mae_K": float(np.mean(np.abs(error[mask]))),
                    "signed_error_K": float(np.mean(error[mask])),
                    "crps_K": float(np.mean(crps[mask])),
                    "coverage_95": float(np.mean(covered[mask])),
                    "interval_width_95_K": float(np.mean(width[mask])),
                    "posterior_sd_K": float(np.mean(sd[mask])),
                    "mean_abs_standardized_residual": float(
                        np.mean(standardized[mask])
                    ),
                }
            )
    return rows


def aggregate_percentile_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    groups = rows.groupby(
        [
            "method",
            "method_label",
            "percentile_bin",
            "percentile_lower",
            "percentile_upper",
        ],
        sort=False,
    )
    linear_metrics = [
        "mae_K",
        "signed_error_K",
        "crps_K",
        "coverage_95",
        "interval_width_95_K",
        "posterior_sd_K",
        "mean_abs_standardized_residual",
    ]
    for keys, group in groups:
        weights = group["n_pixels"].to_numpy(dtype=float)
        row = dict(
            zip(
                [
                    "method",
                    "method_label",
                    "percentile_bin",
                    "percentile_lower",
                    "percentile_upper",
                ],
                keys,
            )
        )
        row["n_pixels"] = int(np.sum(weights))
        row["n_trajectories"] = int(group["trajectory"].nunique())
        row["rmse_K"] = float(
            np.sqrt(np.average(group["rmse_K"] ** 2, weights=weights))
        )
        for metric in linear_metrics:
            row[metric] = float(np.average(group[metric], weights=weights))
        output.append(row)
    return pd.DataFrame(output).sort_values(
        ["percentile_lower", "method_label"]
    )


def mask_contours(axis, prepared, top_mask: np.ndarray, ceiling_mask: np.ndarray) -> None:
    axis.contour(
        prepared.xs,
        prepared.ys,
        ceiling_mask.reshape(prepared.truth.shape).astype(float),
        levels=[0.5],
        colors="white",
        linestyles="--",
        linewidths=1.0,
    )
    axis.contour(
        prepared.xs,
        prepared.ys,
        top_mask.reshape(prepared.truth.shape).astype(float),
        levels=[0.5],
        colors="#00E5FF",
        linestyles="-",
        linewidths=1.2,
    )


def plot_spatial_diagnostic(result: dict[str, object], output_path: Path) -> None:
    prepared = result["prepared"]
    predictions = result["predictions"]
    threshold = float(result["threshold"])
    truth = prepared.truth.ravel()
    top_mask = truth >= float(np.quantile(truth, 0.99))
    ceiling_mask = truth >= threshold
    shape = prepared.truth.shape
    extent = [prepared.xs[0], prepared.xs[-1], prepared.ys[0], prepared.ys[-1]]
    comparison_methods = [FULL_MIXTURE, RBF_REFERENCE]
    error_limit = max(
        float(np.max(np.abs(predictions[method][0] - truth)))
        for method in comparison_methods
    )
    sd_limit = max(
        float(np.max(predictions[method][1])) for method in comparison_methods
    )
    width_limit = max(
        float(np.max(predictions[method][3] - predictions[method][2]))
        for method in comparison_methods
    )
    temp_min = float(np.min(truth))
    temp_max = float(np.max(truth))
    coverage_cmap = ListedColormap(["#D73027", "#1A9850"])
    figure, axes = plt.subplots(
        2, 6, figsize=(20.0, 7.4), constrained_layout=True
    )
    temp_image = error_image = sd_image = width_image = coverage_image = None
    for row_index, method in enumerate(comparison_methods):
        mean, sd, lower, upper, _ = predictions[method]
        error = mean - truth
        covered = ((lower <= truth) & (truth <= upper)).astype(float)
        fields = [
            (truth.reshape(shape), "inferno", temp_min, temp_max),
            (mean.reshape(shape), "inferno", temp_min, temp_max),
            (error.reshape(shape), "coolwarm", -error_limit, error_limit),
            (sd.reshape(shape), "viridis", 0.0, sd_limit),
            ((upper - lower).reshape(shape), "viridis", 0.0, width_limit),
            (covered.reshape(shape), coverage_cmap, 0.0, 1.0),
        ]
        images = []
        for column_index, (field, cmap, vmin, vmax) in enumerate(fields):
            image = axes[row_index, column_index].imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            images.append(image)
            mask_contours(
                axes[row_index, column_index], prepared, top_mask, ceiling_mask
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
        temp_image = images[0]
        error_image = images[2]
        sd_image = images[3]
        width_image = images[4]
        coverage_image = images[5]
        axes[row_index, 0].set_ylabel(METHOD_LABELS[method], fontsize=11)
    titles = [
        "True field",
        "Posterior mean",
        "Mean error",
        "Posterior SD",
        "95% interval width",
        "95% coverage",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=10.5)
    figure.colorbar(
        temp_image,
        ax=axes[:, :2].ravel().tolist(),
        orientation="horizontal",
        shrink=0.55,
        pad=0.035,
        label="Temperature (K)",
    )
    for image, column, label in [
        (error_image, 2, "Error (K)"),
        (sd_image, 3, "SD (K)"),
        (width_image, 4, "Width (K)"),
    ]:
        figure.colorbar(
            image,
            ax=axes[:, column].tolist(),
            orientation="horizontal",
            shrink=0.86,
            pad=0.035,
            label=label,
        )
    coverage_bar = figure.colorbar(
        coverage_image,
        ax=axes[:, 5].tolist(),
        orientation="horizontal",
        shrink=0.86,
        pad=0.035,
        ticks=[0.25, 0.75],
    )
    coverage_bar.ax.set_xticklabels(["miss", "covered"])
    legend = [
        Line2D([0], [0], color="#00E5FF", lw=1.5, label="true top 1%"),
        Line2D(
            [0], [0], color="0.35", lw=1.2, ls="--", label="above camera ceiling"
        ),
    ]
    figure.legend(handles=legend, loc="upper right", frameon=True, ncol=2)
    figure.suptitle(
        f"Held-out hottest-tail diagnostic: {prepared.name}\n"
        "Identical method scales within each diagnostic column",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def plot_top_interval(result: dict[str, object], output_path: Path) -> None:
    prepared = result["prepared"]
    predictions = result["predictions"]
    truth = prepared.truth.ravel()
    top_mask = truth >= float(np.quantile(truth, 0.99))
    top_indices = np.flatnonzero(top_mask)
    order = np.argsort(truth[top_indices])
    selected = top_indices[order]
    x = np.arange(len(selected))
    methods = [FULL_MIXTURE, RBF_REFERENCE]
    y_values = [truth[selected]]
    for method in methods:
        mean, _, lower, upper, _ = predictions[method]
        y_values.extend([mean[selected], lower[selected], upper[selected]])
    y_min = min(float(np.min(values)) for values in y_values)
    y_max = max(float(np.max(values)) for values in y_values)
    margin = 0.04 * (y_max - y_min)
    figure, axes = plt.subplots(
        2, 1, figsize=(11.0, 7.0), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, method in zip(axes, methods):
        mean, _, lower, upper, _ = predictions[method]
        covered = (lower[selected] <= truth[selected]) & (
            truth[selected] <= upper[selected]
        )
        axis.fill_between(
            x,
            lower[selected],
            upper[selected],
            color=METHOD_COLORS[method],
            alpha=0.23,
            label="95% interval",
        )
        axis.plot(x, truth[selected], color="black", lw=1.8, label="truth")
        axis.plot(
            x,
            mean[selected],
            color=METHOD_COLORS[method],
            marker="o",
            ms=3.0,
            lw=1.2,
            label="posterior mean",
        )
        axis.scatter(
            x[~covered],
            truth[selected][~covered],
            color="#D73027",
            marker="x",
            s=35,
            linewidth=1.4,
            label="coverage miss",
            zorder=4,
        )
        axis.axhline(
            float(result["threshold"]), color="0.45", ls="--", lw=1.0
        )
        axis.set_ylim(y_min - margin, y_max + margin)
        axis.set_ylabel("Temperature (K)")
        axis.set_title(
            f"{METHOD_LABELS[method]}: coverage={np.mean(covered):.3f}, "
            f"mean width={np.mean(upper[selected] - lower[selected]):.2f} K, "
            f"RMSE={np.sqrt(np.mean((mean[selected] - truth[selected]) ** 2)):.2f} K"
        )
        axis.grid(alpha=0.18)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True)
    axes[-1].set_xlabel("True top-1% pixels, sorted by true temperature")
    figure.suptitle(
        f"Top-1% posterior intervals: {prepared.name}", fontsize=13
    )
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def plot_percentile_calibration(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(15.5, 8.0), constrained_layout=True
    )
    panels = [
        ("coverage_95", "Empirical 95% coverage"),
        ("posterior_sd_K", "Posterior SD (K)"),
        ("interval_width_95_K", "95% interval width (K)"),
        ("rmse_K", "RMSE (K)"),
        ("mean_abs_standardized_residual", "Mean |error| / posterior SD"),
        ("crps_K", "CRPS (K)"),
    ]
    labels = [f"{low:g}-{high:g}" for low, high in PERCENTILE_BINS]
    for axis, (metric, title) in zip(axes.ravel(), panels):
        for method in PIXEL_METHODS:
            method_rows = summary[summary["method"] == method].sort_values(
                "percentile_lower"
            )
            axis.plot(
                np.arange(len(method_rows)),
                method_rows[metric],
                color=METHOD_COLORS[method],
                marker="o",
                ms=4,
                lw=1.7,
                ls="--" if method == MATCHED_RBF else "-",
                label=METHOD_LABELS[method],
            )
        if metric == "coverage_95":
            axis.axhline(0.95, color="black", ls=":", lw=1.1)
            axis.set_ylim(-0.02, 1.03)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="lower left", fontsize=8.5, frameon=True)
    figure.suptitle(
        "Held-out calibration versus within-trajectory true-temperature percentile",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def aggregate_region_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "region"], sort=False)[METRICS]
        .mean()
        .reset_index()
    )


def reproduction_checks(
    rerun_rows: pd.DataFrame, saved_rows: pd.DataFrame
) -> pd.DataFrame:
    keys = ["trajectory", "method", "region"]
    saved = saved_rows[
        (saved_rows["role"] == "evaluation")
        & saved_rows["method"].isin([FULL_MIXTURE, RBF_REFERENCE])
    ][keys + METRICS]
    rerun = rerun_rows[rerun_rows["method"].isin([FULL_MIXTURE, RBF_REFERENCE])][
        keys + METRICS
    ]
    merged = rerun.merge(saved, on=keys, suffixes=("_rerun", "_saved"))
    for metric in METRICS:
        merged[f"{metric}_difference"] = (
            merged[f"{metric}_rerun"] - merged[f"{metric}_saved"]
        )
    return merged


def compact_markdown(table: pd.DataFrame) -> list[str]:
    lines = [
        "| Region | Method | RMSE | MAE | Bias | CRPS | Coverage | Width | SD | RBF/method width | RBF/method SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region in REGIONS:
        for method in MAIN_METHODS:
            row = table[(table["region"] == region) & (table["method"] == method)].iloc[0]
            lines.append(
                f"| {REGION_LABELS[region]} | {METHOD_LABELS[method]} | "
                f"{row['rmse_K']:.3f} | {row['mae_K']:.3f} | "
                f"{row['signed_error_K']:+.3f} | {row['crps_K']:.3f} | "
                f"{row['coverage_95']:.3f} | {row['interval_width_95_K']:.3f} | "
                f"{row['posterior_sd_K']:.3f} | "
                f"{row['rbf_to_method_interval_width_ratio']:.2f} |"
                f" {row['rbf_to_method_posterior_sd_ratio']:.2f} |"
            )
    return lines


def variance_control_markdown(table: pd.DataFrame) -> list[str]:
    lines = [
        "| Region | Method | RMSE | MAE | CRPS | Coverage | Width | SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region in REGIONS:
        for method in [FULL_MIXTURE, RBF_REFERENCE, MATCHED_RBF]:
            row = table[(table["region"] == region) & (table["method"] == method)].iloc[0]
            lines.append(
                f"| {REGION_LABELS[region]} | {METHOD_LABELS[method]} | "
                f"{row['rmse_K']:.3f} | {row['mae_K']:.3f} | "
                f"{row['crps_K']:.3f} | {row['coverage_95']:.3f} | "
                f"{row['interval_width_95_K']:.3f} | {row['posterior_sd_K']:.3f} |"
            )
    return lines


def write_summary(
    output_dir: Path,
    existing: pd.DataFrame,
    matched: pd.DataFrame,
    bins: pd.DataFrame,
    scales: pd.DataFrame,
    selected: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    full_top = existing[
        (existing["method"] == FULL_MIXTURE) & (existing["region"] == "top_1pct")
    ].iloc[0]
    rbf_top = existing[
        (existing["method"] == RBF_REFERENCE) & (existing["region"] == "top_1pct")
    ].iloc[0]
    full_hot = existing[
        (existing["method"] == FULL_MIXTURE)
        & (existing["region"] == "above_camera_ceiling")
    ].iloc[0]
    rbf_hot = existing[
        (existing["method"] == RBF_REFERENCE)
        & (existing["region"] == "above_camera_ceiling")
    ].iloc[0]
    matched_top = matched[
        (matched["method"] == MATCHED_RBF) & (matched["region"] == "top_1pct")
    ].iloc[0]
    coolest_st = bins[
        (bins["method"] == FULL_MIXTURE) & (bins["percentile_lower"] == 0.0)
    ].iloc[0]
    hottest_st = bins[
        (bins["method"] == FULL_MIXTURE) & (bins["percentile_lower"] == 99.0)
    ].iloc[0]
    st_scale = scales[scales["model"] == "ST innovation C"].iloc[0]
    rbf_scale = scales[scales["model"] == "spatial RBF"].iloc[0]
    max_check = max(
        float(checks[column].abs().max())
        for column in checks.columns
        if column.endswith("_difference")
    )
    lines = [
        "# Hottest-tail coverage diagnostic",
        "",
        "This diagnostic leaves the experiment-27 models unchanged. Pixel-level posteriors "
        "were not saved, so the held-out trajectories were rerun with the frozen configuration "
        "and their original global trajectory seed indices. Development trajectories were not rerun.",
        "",
        "## Existing held-out results",
        "",
        *compact_markdown(existing),
        "",
        "The width and SD ratios are uncertainty-scale comparisons; RMSE, MAE, and signed "
        "error describe posterior-mean reconstruction quality.",
        "",
        "## Selected held-out spatial cases",
        "",
        "| Selection | Trajectory | ST top-1% coverage | RBF top-1% coverage | Difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['selection']} | {row['trajectory']} | "
            f"{row['st_top_1pct_coverage']:.3f} | "
            f"{row['rbf_top_1pct_coverage']:.3f} | "
            f"{row['rbf_minus_st_coverage']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Direct covariance-scale check",
            "",
            f"The finite-step ST innovation has marginal SD {st_scale['sd_K']:.3f} K "
            f"(mean diag(C)={st_scale['variance_K2']:.3f} K^2), whereas the spatial RBF "
            f"has prior marginal SD {rbf_scale['sd_K']:.3f} K "
            f"(sigma_f^2={rbf_scale['variance_K2']:.3f} K^2). The RBF/ST prior-SD ratio "
            f"is {rbf_scale['sd_K'] / st_scale['sd_K']:.2f}.",
            "",
            "## Variance-matched RBF control",
            "",
            "This control keeps the posterior-physics prior mean, RBF correlation geometry, "
            "observations, likelihood, noise, and seeds fixed, while replacing the RBF prior "
            "marginal variance with mean diag(C) from the ST innovation.",
            "",
            *variance_control_markdown(matched),
            "",
            "## Explicit answers",
            "",
            "1. **RBF's top-1% coverage advantage is mainly interval width.** Its top-1% "
            f"RMSE improves by only {full_top['rmse_K'] - rbf_top['rmse_K']:.3f} K and MAE "
            f"by {full_top['mae_K'] - rbf_top['mae_K']:.3f} K, while its interval is "
            f"{rbf_top['interval_width_95_K'] / full_top['interval_width_95_K']:.2f} times "
            f"wider and coverage rises from {full_top['coverage_95']:.3f} to "
            f"{rbf_top['coverage_95']:.3f}.",
            "",
            "2. **The mean helps modestly, but the CRPS change is not additively decomposable.** "
            f"RBF reduces top-1% MAE by {full_top['mae_K'] - rbf_top['mae_K']:.3f} K and "
            f"CRPS by {full_top['crps_K'] - rbf_top['crps_K']:.3f} K. Holding the RBF prior "
            "physics mean and correlation geometry fixed while matching its prior variance "
            "to ST changes "
            f"top-1% CRPS from {rbf_top['crps_K']:.3f} to {matched_top['crps_K']:.3f} K and "
            f"coverage from {rbf_top['coverage_95']:.3f} to {matched_top['coverage_95']:.3f}. "
            f"Its matched RMSE and MAE are {matched_top['rmse_K']:.3f} and "
            f"{matched_top['mae_K']:.3f} K, so even the modest original posterior-mean "
            "advantage largely disappears. Covariance amplitude affects both posterior spread "
            "and how strongly censored observations update the posterior mean; an additive "
            "mean-versus-spread percentage would therefore be misleading.",
            "",
            "3. **Near-nominal above-ceiling RBF coverage does not imply a better forecast.** "
            f"RBF uses {rbf_hot['interval_width_95_K']:.3f} K intervals versus "
            f"{full_hot['interval_width_95_K']:.3f} K for ST. The diffuse RBF intervals cover "
            f"{rbf_hot['coverage_95']:.3f}, but its CRPS is worse "
            f"({rbf_hot['crps_K']:.3f} versus {full_hot['crps_K']:.3f} K) because CRPS "
            "penalizes unnecessary spread as well as misses.",
            "",
            "4. **ST uncertainty is strongly temperature-dependent in the wrong way.** In the "
            f"coolest decile its coverage is {coolest_st['coverage_95']:.3f}, posterior SD "
            f"{coolest_st['posterior_sd_K']:.3f} K, and mean |error|/SD "
            f"{coolest_st['mean_abs_standardized_residual']:.3f}; in the hottest 1% these "
            f"become {hottest_st['coverage_95']:.3f}, {hottest_st['posterior_sd_K']:.3f} K, "
            f"and {hottest_st['mean_abs_standardized_residual']:.3f}. Thus a roughly fixed "
            "uncertainty scale is excessive in the background and insufficient at the peak.",
            "",
            "5. **Variance matching is the clean scale control.** The matched-RBF top-1% "
            f"coverage is {matched_top['coverage_95']:.3f}, compared with "
            f"{rbf_top['coverage_95']:.3f} before matching and {full_top['coverage_95']:.3f} "
            f"for full-mixture ST; its CRPS is {matched_top['crps_K']:.3f} versus "
            f"{full_top['crps_K']:.3f} K for ST. Matching fully removes and slightly reverses "
            "the apparent RBF tail advantage. The remaining difference is the appropriate "
            "evidence for mean/correlation geometry rather than marginal amplitude.",
            "",
            "Overall, the remaining sequential-model problem is primarily a state-/temperature-"
            "dependent uncertainty problem, not evidence that RBF geometry is uniformly better. "
            "The full-mixture ST still has the better above-ceiling CRPS despite lower coverage.",
            "",
            "## Reproducibility",
            "",
            "The percentile plot uses stable within-trajectory ranks, so the 99-100% bin "
            "contains exactly 25 pixels per 61 x 41 field. The saved `top_1pct` region uses "
            "`truth >= quantile(truth, 0.99)` and contains 26 pixels; this explains the small "
            "difference between the two displayed hottest-bin aggregates.",
            "",
            f"The largest absolute difference between rerun and saved experiment-27 region "
            f"metrics is {max_check:.3e}. `reproduction_checks.csv` records every comparison.",
            "The variance-matched RBF changes only prior covariance amplitude; it retains the "
            "same posterior-physics prior mean, RBF correlation structure, observations, "
            "censored likelihood, measurement noise, and sampler seeds. Its conditioned "
            "posterior mean is allowed to change, as it must when the prior covariance changes.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose experiment-27 hottest-tail coverage without redesigning models."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Rerun only the three selected held-out cases (smoke-test mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(EXPERIMENT_27 / "fixed_configuration.csv").iloc[0]
    saved_regions = pd.read_csv(EXPERIMENT_27 / "quartile_results.csv")
    saved_summary = pd.read_csv(EXPERIMENT_27 / "quartile_summary.csv")
    existing = existing_diagnostic_table(saved_summary)
    selected = select_heldout_trajectories(saved_regions)
    existing.to_csv(args.output_dir / "compact_diagnostic.csv", index=False)
    selected.to_csv(args.output_dir / "selected_trajectories.csv", index=False)

    all_records = trajectory_catalog(args.dataset_dir)
    record_map = {
        record.name: (trajectory_index, record)
        for trajectory_index, record in enumerate(all_records)
    }
    selected_names = set(selected["trajectory"])
    if args.selected_only:
        run_records = [record_map[name] for name in selected["trajectory"]]
    else:
        run_records = [
            (trajectory_index, record)
            for trajectory_index, record in enumerate(all_records)
            if record.name not in DEVELOPMENT_TRAJECTORIES
        ]

    percentile_output = []
    region_output = []
    scale_output = []
    selected_results: dict[str, dict[str, object]] = {}
    for run_index, (trajectory_index, record) in enumerate(run_records):
        result = run_frozen_trajectory(
            args.dataset_dir,
            record,
            trajectory_index=trajectory_index,
            fixed=fixed,
        )
        prepared = result["prepared"]
        truth = prepared.truth.ravel()
        predictions = result["predictions"]
        percentile_output.extend(
            percentile_rows(record.name, truth, predictions)
        )
        for method in PIXEL_METHODS:
            region_output.extend(
                EXPERIMENT.prediction_region_rows(
                    prepared,
                    method=method,
                    role="evaluation",
                    threshold=float(result["threshold"]),
                    prediction=predictions[method],
                )
            )
        scale_output.append(
            {
                "trajectory": record.name,
                "st_innovation_variance_K2": result["innovation_variance_K2"],
                "st_innovation_sd_K": result["innovation_sd_K"],
                "rbf_prior_variance_K2": result["rbf_prior_variance_K2"],
                "rbf_prior_sd_K": result["rbf_prior_sd_K"],
                "rbf_to_st_prior_variance_ratio": (
                    result["rbf_prior_variance_K2"]
                    / result["innovation_variance_K2"]
                ),
                "rbf_to_st_prior_sd_ratio": (
                    result["rbf_prior_sd_K"] / result["innovation_sd_K"]
                ),
                "variance_match_scale": result["variance_match_scale"],
            }
        )
        if record.name in selected_names:
            selected_results[record.name] = result
            plot_spatial_diagnostic(
                result,
                args.output_dir
                / f"heldout_spatial_{record.name.lower()}.png",
            )
        print(
            f"[{run_index + 1:02d}/{len(run_records):02d}] {record.name}",
            flush=True,
        )
        pd.DataFrame(percentile_output).to_csv(
            args.output_dir / "percentile_checkpoint.csv", index=False
        )
        pd.DataFrame(region_output).to_csv(
            args.output_dir / "region_checkpoint.csv", index=False
        )

    percentile_trajectory = pd.DataFrame(percentile_output)
    percentile_summary = aggregate_percentile_rows(percentile_trajectory)
    region_rows = pd.DataFrame(region_output)
    matched_summary = aggregate_region_rows(region_rows)
    checks = reproduction_checks(region_rows, saved_regions)
    scale_rows = pd.DataFrame(scale_output)
    scales = pd.DataFrame(
        [
            {
                "model": "ST innovation C",
                "variance_K2": scale_rows["st_innovation_variance_K2"].mean(),
                "sd_K": scale_rows["st_innovation_sd_K"].mean(),
            },
            {
                "model": "spatial RBF",
                "variance_K2": scale_rows["rbf_prior_variance_K2"].mean(),
                "sd_K": scale_rows["rbf_prior_sd_K"].mean(),
            },
            {
                "model": "variance-matched RBF",
                "variance_K2": scale_rows["st_innovation_variance_K2"].mean(),
                "sd_K": scale_rows["st_innovation_sd_K"].mean(),
            },
        ]
    )
    percentile_trajectory.to_csv(
        args.output_dir / "percentile_bin_trajectory.csv", index=False
    )
    percentile_summary.to_csv(
        args.output_dir / "percentile_bin_summary.csv", index=False
    )
    region_rows.to_csv(args.output_dir / "rerun_region_results.csv", index=False)
    matched_summary.to_csv(
        args.output_dir / "variance_matched_region_summary.csv", index=False
    )
    checks.to_csv(args.output_dir / "reproduction_checks.csv", index=False)
    scale_rows.to_csv(args.output_dir / "covariance_scale_by_trajectory.csv", index=False)
    scales.to_csv(args.output_dir / "covariance_scale_summary.csv", index=False)
    plot_percentile_calibration(
        percentile_summary, args.output_dir / "calibration_by_temperature.png"
    )

    representative = selected.loc[
        selected["selection"] == "median difference", "trajectory"
    ].iloc[0]
    if representative not in selected_results:
        raise RuntimeError(f"Representative trajectory {representative} was not rerun")
    plot_top_interval(
        selected_results[representative],
        args.output_dir / "top_1pct_intervals_representative.png",
    )
    write_summary(
        args.output_dir,
        existing,
        matched_summary,
        percentile_summary,
        scales,
        selected,
        checks,
    )
    print(f"Saved hottest-tail diagnostic to {args.output_dir}")


if __name__ == "__main__":
    main()
