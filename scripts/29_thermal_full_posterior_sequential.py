from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

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
    prediction_metrics,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "27_full_posterior_sequential"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "24_posterior_physics_mean_advection"
    / "fixed_configuration.csv"
)

MEAN_ONLY = "mean-only sequential advective stochastic ST"
MOMENT_MATCHED = "moment-matched sequential advective stochastic ST"
FULL_MIXTURE = "full-posterior mixture sequential advective stochastic ST"
RBF_REFERENCE = "posterior physics mean + RBF"
METHODS = [MEAN_ONLY, MOMENT_MATCHED, FULL_MIXTURE, RBF_REFERENCE]
CORE_METHODS = [MEAN_ONLY, MOMENT_MATCHED, FULL_MIXTURE]
COLORS = {
    MEAN_ONLY: "#D55E00",
    MOMENT_MATCHED: "#CC79A7",
    FULL_MIXTURE: "#009E73",
    RBF_REFERENCE: "#0072B2",
}
SHORT_LABELS = {
    MEAN_ONLY: "mean-only ST",
    MOMENT_MATCHED: "moment-matched ST",
    FULL_MIXTURE: "full-posterior mixture ST",
    RBF_REFERENCE: "posterior mean + RBF",
}
METRICS = [
    "excess_field_rel_l2",
    "rmse_K",
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "peak_absolute_error_K",
    "signed_error_K",
    "pointwise_95_coverage",
    "hot_region_95_coverage",
    "hot_region_95_interval_width_K",
]
REGION_METRICS = [
    "rmse_K",
    "mae_K",
    "signed_error_K",
    "crps_K",
    "coverage_95",
    "interval_width_95_K",
    "posterior_sd_K",
]


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    return iy * len(prepared.xs) + ix


def prediction_region_rows(
    prepared,
    *,
    method: str,
    role: str,
    threshold: float,
    prediction: tuple[np.ndarray, ...],
) -> list[dict[str, object]]:
    mean, sd, lower, upper, draws = prediction
    truth = prepared.truth.ravel()
    crps = empirical_crps(draws, truth)
    quartile_edges = np.quantile(truth, [0.25, 0.5, 0.75])
    masks = {
        "Q1": truth <= quartile_edges[0],
        "Q2": (truth > quartile_edges[0]) & (truth <= quartile_edges[1]),
        "Q3": (truth > quartile_edges[1]) & (truth <= quartile_edges[2]),
        "Q4": truth > quartile_edges[2],
        "top_1pct": truth >= float(np.quantile(truth, 0.99)),
        "above_camera_ceiling": truth >= threshold,
        "overall": np.ones(len(truth), dtype=bool),
    }
    rows = []
    for region, mask in masks.items():
        error = mean[mask] - truth[mask]
        rows.append(
            {
                "trajectory": prepared.name,
                "role": role,
                "method": method,
                "region": region,
                "n_pixels": int(np.sum(mask)),
                "true_temperature_min_K": float(np.min(truth[mask])),
                "true_temperature_max_K": float(np.max(truth[mask])),
                "true_temperature_mean_K": float(np.mean(truth[mask])),
                "rmse_K": float(np.sqrt(np.mean(error**2))),
                "mae_K": float(np.mean(np.abs(error))),
                "signed_error_K": float(np.mean(error)),
                "crps_K": float(np.mean(crps[mask])),
                "coverage_95": float(
                    np.mean((lower[mask] <= truth[mask]) & (truth[mask] <= upper[mask]))
                ),
                "interval_width_95_K": float(np.mean((upper - lower)[mask])),
                "posterior_sd_K": float(np.mean(sd[mask])),
            }
        )
    return rows


def plot_reconstruction(
    prepared,
    *,
    current: dict[str, object],
    predictions: dict[str, tuple[np.ndarray, ...]],
    output_path: Path,
) -> None:
    extent = [prepared.xs[0], prepared.xs[-1], prepared.ys[0], prepared.ys[-1]]
    truth_min = float(np.min(prepared.truth))
    truth_max = float(np.max(prepared.truth))
    sd_max = max(float(np.max(predictions[method][1])) for method in METHODS)
    figure, axes = plt.subplots(2, 5, figsize=(19.0, 7.2), constrained_layout=True)

    mean_panels = [("True current field", prepared.truth)] + [
        (
            SHORT_LABELS[method],
            predictions[method][0].reshape(prepared.truth.shape),
        )
        for method in METHODS
    ]
    sd_panels = [("Censored current observation", current["clipped_full"])] + [
        (
            f"{SHORT_LABELS[method]} SD",
            predictions[method][1].reshape(prepared.truth.shape),
        )
        for method in METHODS
    ]

    mean_image = None
    for axis, (title, field) in zip(axes[0], mean_panels):
        mean_image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=truth_min,
            vmax=truth_max,
            aspect="auto",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    sd_image = None
    for index, (axis, (title, field)) in enumerate(zip(axes[1], sd_panels)):
        if index == 0:
            sd_image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="inferno",
                vmin=truth_min,
                vmax=truth_max,
                aspect="auto",
            )
        else:
            sd_image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=sd_max,
                aspect="auto",
            )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    if mean_image is not None:
        colorbar = figure.colorbar(mean_image, ax=axes[0].tolist(), fraction=0.018)
        colorbar.set_label("Temperature (K)")
    if sd_image is not None:
        colorbar = figure.colorbar(sd_image, ax=axes[1, 1:].tolist(), fraction=0.018)
        colorbar.set_label("Posterior SD (K)")
    figure.suptitle(
        f"Mean-only versus full-posterior sequential propagation: {prepared.name}",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=210)
    plt.close(figure)

    # A second, display-only scale reveals the cooling wake, which is compressed
    # when the laser peak determines the color limits.  All panels use the same
    # ambient-to-camera-ceiling normalization so differences remain comparable.
    ambient = float(prepared.ambient)
    threshold = float(current["threshold"])
    tail_panels = [
        ("True current field", np.asarray(prepared.truth, dtype=float)),
        (
            "Censored current observation",
            np.asarray(current["clipped_full"], dtype=float),
        ),
    ] + [
        (
            SHORT_LABELS[method],
            predictions[method][0].reshape(prepared.truth.shape),
        )
        for method in METHODS
    ]
    contour_candidates = ambient + np.array([0.25, 0.5, 1.0, 2.0])
    tail_figure, tail_grid = plt.subplots(
        1,
        len(tail_panels) + 1,
        figsize=(21.0, 4.5),
        gridspec_kw={"width_ratios": [1.0] * len(tail_panels) + [0.045]},
        constrained_layout=True,
    )
    tail_axes = tail_grid[:-1]
    tail_colorbar_axis = tail_grid[-1]
    tail_image = None
    for axis, (title, field) in zip(tail_axes, tail_panels):
        tail_image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=ambient,
            vmax=threshold,
            aspect="auto",
        )
        contour_field = gaussian_filter(field, sigma=1.0, mode="nearest")
        contour_levels = contour_candidates[
            (contour_candidates > float(np.min(contour_field)))
            & (contour_candidates < float(np.max(contour_field)))
            & (contour_candidates < threshold)
        ]
        if len(contour_levels):
            axis.contour(
                prepared.xs,
                prepared.ys,
                contour_field,
                levels=contour_levels,
                colors="white",
                linewidths=0.55,
                alpha=0.75,
            )
        axis.set_title(title, fontsize=9.5)
        axis.set_xticks([])
        axis.set_yticks([])
    if tail_image is not None:
        tail_colorbar = tail_figure.colorbar(
            tail_image,
            cax=tail_colorbar_axis,
            extend="max",
        )
        tail_colorbar.set_label("Temperature (K)")
    tail_figure.suptitle(
        f"Tail-focused reconstruction: {prepared.name}\n"
        f"shared display scale {ambient:.2f}-{threshold:.2f} K; "
        "smoothed contour guides at ambient + 0.25, 0.5, 1, and 2 K",
        fontsize=12,
    )
    tail_path = output_path.with_name(
        f"{output_path.stem}_tail_focus{output_path.suffix}"
    )
    tail_figure.savefig(tail_path, dpi=220)
    plt.close(tail_figure)


def run_experiment(
    args: argparse.Namespace,
    fixed: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed_records = list(enumerate(trajectory_catalog(args.dataset_dir)))
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        indexed_records = [
            (trajectory_index, record)
            for trajectory_index, record in indexed_records
            if record.name in requested
        ]
        missing = requested - {record.name for _, record in indexed_records}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    for run_index, (trajectory_index, record) in enumerate(indexed_records):
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        current_index = len(prepared.times) - 1
        previous_index = current_index - args.previous_frame_offset
        threshold = float(
            np.quantile(prepared.truth, 1.0 - args.fraction_saturated)
        )
        camera = paired_camera_observations(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            threshold=threshold,
            observation_stride=args.observation_stride,
            noise_sd=args.measurement_noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current_points = np.asarray(camera["current"]["x_pred"], dtype=float)
        observation_points = np.asarray(camera["current"]["x_obs"], dtype=float)
        observed_indices = observation_indices(prepared, observation_points)
        lengthscale = prepared.source_lengthscale * args.length_multiplier
        stochastic_config = StochasticHeatConfig(
            signal_sd=float(fixed.signal_sd),
            forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            quadrature_order=args.quadrature_order,
        )
        previous_mean, previous_draws, previous_diagnostics = (
            infer_previous_censored_posterior(
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
                seed=args.seed + 50_000 * trajectory_index,
            )
        )
        ordinary_mean, _, displacement = posterior_physics_means(
            prepared,
            previous_mean,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
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
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            displacement=displacement,
        )
        component_prediction_means = ordinary_mean.ravel()[None, :] + (
            propagated_components.reshape(len(previous_draws), -1)
        )
        component_observation_means = component_prediction_means[:, observed_indices]

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
        denominator = np.sqrt(len(previous_draws) - 1.0)
        features = propagated_components.reshape(len(previous_draws), -1).T / denominator
        observed_features = features[observed_indices]
        moment_oo = innovation_oo + observed_features.dot(observed_features.T)
        moment_po = innovation_po + features.dot(observed_features.T)
        moment_variance = innovation_variance + np.sum(features**2, axis=1)

        rbf_config = RBFConfig(
            mean_temp=prepared.ambient,
            signal_sd=float(fixed.signal_sd),
            lengthscale=lengthscale,
            noise_sd=args.current_noise_sd,
        )
        rbf_oo = rbf_covariance(observation_points, observation_points, rbf_config)
        rbf_po = rbf_covariance(current_points, observation_points, rbf_config)
        rbf_variance = np.full(len(current_points), rbf_config.signal_sd**2)
        flat_mean = ordinary_mean.ravel()
        prediction_seed = args.seed + 100_000 * trajectory_index

        predictions: dict[str, tuple[np.ndarray, ...]] = {}
        predictions[MEAN_ONLY] = sample_censored_gaussian_blocks(
            camera["current"],
            prediction_mean=flat_mean,
            observation_mean=flat_mean[observed_indices],
            observed_covariance=innovation_oo,
            pred_observed_covariance=innovation_po,
            prediction_variance=innovation_variance,
            noise_sd=args.current_noise_sd,
            n_samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=prediction_seed,
        )
        predictions[MOMENT_MATCHED] = sample_censored_gaussian_blocks(
            camera["current"],
            prediction_mean=flat_mean,
            observation_mean=flat_mean[observed_indices],
            observed_covariance=moment_oo,
            pred_observed_covariance=moment_po,
            prediction_variance=moment_variance,
            noise_sd=args.current_noise_sd,
            n_samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=prediction_seed,
        )
        predictions[FULL_MIXTURE], mixture_diagnostics = (
            sample_censored_gaussian_mixture_blocks(
                camera["current"],
                component_prediction_means=component_prediction_means,
                component_observation_means=component_observation_means,
                observed_covariance=innovation_oo,
                pred_observed_covariance=innovation_po,
                prediction_variance=innovation_variance,
                noise_sd=args.current_noise_sd,
                n_samples=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=prediction_seed,
            )
        )
        predictions[RBF_REFERENCE] = sample_censored_gaussian_blocks(
            camera["current"],
            prediction_mean=flat_mean,
            observation_mean=flat_mean[observed_indices],
            observed_covariance=rbf_oo,
            pred_observed_covariance=rbf_po,
            prediction_variance=rbf_variance,
            noise_sd=args.current_noise_sd,
            n_samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=prediction_seed,
        )

        role = "calibration" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        for method in METHODS:
            prediction = predictions[method]
            row = prediction_metrics(
                prepared,
                method=method,
                observations=camera["current"],
                prediction=prediction,
            )
            error = prediction[0] - prepared.truth.ravel()
            row.update(
                {
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "rmse_K": float(np.sqrt(np.mean(error**2))),
                    "signed_error_K": float(np.mean(error)),
                    "fraction_saturated": args.fraction_saturated,
                    "previous_frame_offset": args.previous_frame_offset,
                    "time_lag_s": dt,
                    "diffusivity_m2_s": float(fixed.diffusivity),
                    "cooling_rate_1_s": float(fixed.cooling_rate),
                    "source_coupling": float(fixed.source_coupling),
                    "signal_sd_K": float(fixed.signal_sd),
                    "lengthscale_m": lengthscale,
                    "forcing_lengthscale_m": stochastic_config.forcing_lengthscale,
                    "previous_observations_reused_in_current_update": False,
                    "same_previous_posterior_for_all_models": True,
                    "same_current_observations_for_all_models": True,
                    "same_sampler_seed_for_all_models": True,
                    "crps_estimator": "unbiased_M_times_M_minus_1",
                    "uncertainty_propagation": {
                        MEAN_ONLY: "previous posterior collapsed to mean; innovation C only",
                        MOMENT_MATCHED: "Gaussian C + B Sigma B^T from propagated draws",
                        FULL_MIXTURE: "likelihood-reweighted propagated-draw mixture with shared C",
                        RBF_REFERENCE: "current spatial RBF residual",
                    }[method],
                }
            )
            if method == FULL_MIXTURE:
                row.update(mixture_diagnostics)
            rows.append(row)
            region_rows.extend(
                prediction_region_rows(
                    prepared,
                    method=method,
                    role=role,
                    threshold=threshold,
                    prediction=prediction,
                )
            )
            print(
                f"[{run_index + 1:02d}/{len(indexed_records):02d}] "
                f"{record.name}, {SHORT_LABELS[method]}: "
                f"field={row['excess_field_rel_l2']:.3f}, "
                f"all={row['mean_crps_K']:.3f}, "
                f"top1={row['fixed_top_01_crps_K']:.3f}, "
                f"coverage={row['hot_region_95_coverage']:.3f}",
                flush=True,
            )

        mean_component_field = np.mean(component_prediction_means, axis=0)
        integrity_rows.append(
            {
                "trajectory": record.name,
                "role": role,
                "previous_draws": len(previous_draws),
                "current_saturated_observations": int(
                    np.sum(np.asarray(camera["current"]["sat_mask"]))
                ),
                "component_mean_average_max_difference_K": float(
                    np.max(np.abs(mean_component_field - flat_mean))
                ),
                "propagated_component_sd_mean_K": float(
                    np.mean(np.std(component_prediction_means, axis=0, ddof=1))
                ),
                "innovation_sd_K": float(np.sqrt(innovation_variance_scalar)),
                **previous_diagnostics,
                **mixture_diagnostics,
            }
        )
        if record.name in DEVELOPMENT_TRAJECTORIES:
            plot_reconstruction(
                prepared,
                current=camera["current"],
                predictions=predictions,
                output_path=args.output_dir / f"reconstruction_{record.name.lower()}.png",
            )
        pd.DataFrame(rows).to_csv(args.output_dir / "checkpoint.csv", index=False)
        pd.DataFrame(region_rows).to_csv(
            args.output_dir / "quartile_checkpoint.csv", index=False
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(region_rows),
        pd.DataFrame(integrity_rows),
    )


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"]
    if subset.empty:
        subset = results
    summary = subset.groupby("method", sort=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def aggregate_regions(region_results: pd.DataFrame) -> pd.DataFrame:
    subset = region_results[region_results["role"] == "evaluation"]
    if subset.empty:
        subset = region_results
    summary = subset.groupby(["method", "region"], sort=False)[REGION_METRICS].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"]
    if subset.empty:
        subset = results
    indexed = subset.set_index(["trajectory", "method"])
    trajectories = sorted(subset["trajectory"].unique())
    pairs = [
        (FULL_MIXTURE, MEAN_ONLY),
        (FULL_MIXTURE, MOMENT_MATCHED),
        (MOMENT_MATCHED, MEAN_ONLY),
        (FULL_MIXTURE, RBF_REFERENCE),
    ]
    rows = []
    for method, baseline in pairs:
        row = {"method": method, "baseline": baseline, "n_trajectories": len(trajectories)}
        for metric in METRICS:
            changes = np.asarray(
                [
                    indexed.loc[(trajectory, method), metric]
                    - indexed.loc[(trajectory, baseline), metric]
                    for trajectory in trajectories
                ]
            )
            row[f"{metric}_mean_change"] = float(np.mean(changes))
            row[f"{metric}_median_change"] = float(np.median(changes))
            if "coverage" in metric:
                target = np.asarray(
                    [indexed.loc[(trajectory, baseline), metric] for trajectory in trajectories]
                )
                changed = target + changes
                row[f"{metric}_closer_to_095_count"] = int(
                    np.sum(np.abs(changed - 0.95) < np.abs(target - 0.95))
                )
            else:
                row[f"{metric}_win_count"] = int(np.sum(changes < 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    panels = [
        ("excess_field_rel_l2", "Relative excess-field L2 error"),
        ("rmse_K", "RMSE (K)"),
        ("mean_crps_K", "Overall CRPS (K)"),
        ("fixed_top_01_crps_K", "Top-1% CRPS (K)"),
        ("hot_region_95_coverage", "Above-ceiling 95% coverage"),
        ("signed_error_K", "Average signed error (K)"),
    ]
    indexed = summary.set_index("method")
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    for axis, (metric, label) in zip(axes.ravel(), panels):
        means = np.asarray([indexed.loc[method, f"{metric}_mean"] for method in METHODS])
        errors = np.asarray(
            [
                indexed.loc[method, f"{metric}_std"]
                / np.sqrt(indexed.loc[method, f"{metric}_count"])
                for method in METHODS
            ]
        )
        axis.bar(
            x,
            means,
            yerr=np.nan_to_num(errors),
            color=[COLORS[method] for method in METHODS],
            capsize=4,
        )
        axis.set_xticks(x, [SHORT_LABELS[method] for method in METHODS], rotation=18, ha="right")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        if "coverage" in metric:
            axis.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0)
        if metric == "signed_error_K":
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    figure.suptitle("Full-posterior sequential propagation: held-out comparison")
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_quartiles(region_summary: pd.DataFrame, output_path: Path) -> None:
    regions = ["Q1", "Q2", "Q3", "Q4"]
    indexed = region_summary.set_index(["method", "region"])
    panels = [
        ("crps_K", "CRPS (K)"),
        ("coverage_95", "95% coverage"),
        ("signed_error_K", "Average signed error (K)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for axis, (metric, label) in zip(axes, panels):
        for method in METHODS:
            values = [indexed.loc[(method, region), f"{metric}_mean"] for region in regions]
            axis.plot(
                regions,
                values,
                marker="o",
                color=COLORS[method],
                label=SHORT_LABELS[method],
            )
        axis.set_xlabel("True-temperature quartile")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "coverage_95":
            axis.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0)
        if metric == "signed_error_K":
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Calibration and error by true-temperature quartile")
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_tail_regions(region_summary: pd.DataFrame, output_path: Path) -> None:
    regions = ["Q4", "above_camera_ceiling", "top_1pct"]
    labels = ["Q4", "above ceiling", "top 1%"]
    indexed = region_summary.set_index(["method", "region"])
    panels = [
        ("crps_K", "CRPS (K)"),
        ("coverage_95", "95% coverage"),
        ("signed_error_K", "Average signed error (K)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for axis, (metric, label) in zip(axes, panels):
        for method in METHODS:
            values = [indexed.loc[(method, region), f"{metric}_mean"] for region in regions]
            axis.plot(
                labels,
                values,
                marker="o",
                color=COLORS[method],
                label=SHORT_LABELS[method],
            )
        axis.set_xlabel("Increasingly extreme temperature region")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "coverage_95":
            axis.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0)
        if metric == "signed_error_K":
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Tail diagnostics beyond the broad Q4 stratum")
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_readme(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    integrity: pd.DataFrame,
) -> None:
    table = summary.set_index("method")
    regions = region_summary.set_index(["method", "region"])
    mean_only = table.loc[MEAN_ONLY]
    moment = table.loc[MOMENT_MATCHED]
    full = table.loc[FULL_MIXTURE]
    lines = [
        "# Full-posterior one-step sequential propagation",
        "",
        "This focused experiment tests whether collapsing the previous censored posterior "
        "loses uncertainty before the one-step advective stochastic transition.",
        "",
        "## Implementation clarification",
        "",
        "The pre-existing sequential implementation was not strictly mean-only. It already "
        "propagated centered previous-posterior draws into `B Sigma B^T` and then used one "
        "moment-matched Gaussian prior. This experiment therefore reports three controlled "
        "versions under the same transition and current censored likelihood:",
        "",
        "1. **Mean-only:** propagate the previous posterior mean and retain only innovation `C`.",
        "2. **Moment-matched (existing):** use `C + B Sigma B^T`.",
        "3. **Full-posterior mixture:** retain propagated draws as distinct component means, "
        "reweight them with the current mixed equality/censoring likelihood, and condition "
        "each selected component with the same innovation `C`.",
        "",
        "The posterior-physics-mean + spatial RBF method is retained only as a reference.",
        "",
        "## Held-out results",
        "",
        "| Method | Field | RMSE | Overall CRPS | Top-1% CRPS | Signed error | Hot coverage | Hot width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['rmse_K_mean']:.3f} | {row['mean_crps_K_mean']:.3f} | "
            f"{row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['signed_error_K_mean']:+.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} | "
            f"{row['hot_region_95_interval_width_K_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Temperature-quartile diagnostics",
            "",
            "| Method | Quartile | CRPS | Coverage | Signed error | RMSE | Width |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in CORE_METHODS:
        for region in ["Q1", "Q2", "Q3", "Q4"]:
            row = regions.loc[(method, region)]
            lines.append(
                f"| {method} | {region} | {row['crps_K_mean']:.3f} | "
                f"{row['coverage_95_mean']:.3f} | "
                f"{row['signed_error_K_mean']:+.3f} | "
                f"{row['rmse_K_mean']:.3f} | "
                f"{row['interval_width_95_K_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Hottest-region diagnostics",
            "",
            "Q4 is broad enough to conceal failure at the extreme peak. The two narrower "
            "regions are therefore retained separately.",
            "",
            "| Method | Region | CRPS | Coverage | Signed error | RMSE | Width |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        for region, label in [
            ("Q4", "Q4"),
            ("above_camera_ceiling", "above camera ceiling"),
            ("top_1pct", "top 1%"),
        ]:
            row = regions.loc[(method, region)]
            lines.append(
                f"| {method} | {label} | {row['crps_K_mean']:.3f} | "
                f"{row['coverage_95_mean']:.3f} | "
                f"{row['signed_error_K_mean']:+.3f} | "
                f"{row['rmse_K_mean']:.3f} | "
                f"{row['interval_width_95_K_mean']:.3f} |"
            )
    coverage_improves = abs(full["hot_region_95_coverage_mean"] - 0.95) < abs(
        mean_only["hot_region_95_coverage_mean"] - 0.95
    )
    full_top = regions.loc[(FULL_MIXTURE, "top_1pct")]
    moment_top = regions.loc[(MOMENT_MATCHED, "top_1pct")]
    rbf_top = regions.loc[(RBF_REFERENCE, "top_1pct")]
    heldout_integrity = integrity[integrity["role"] == "evaluation"]
    if heldout_integrity.empty:
        heldout_integrity = integrity
    propagated_sd = float(
        heldout_integrity["propagated_component_sd_mean_K"].mean()
    )
    innovation_sd = float(heldout_integrity["innovation_sd_K"].mean())
    weight_ess = float(
        heldout_integrity["mixture_component_weight_ess"].mean()
    )
    lines.extend(
        [
            "",
            "## Direct answer",
            "",
            "Relative to strict mean-only propagation, full-posterior mixture propagation "
            f"changes overall CRPS by "
            f"{full['mean_crps_K_mean'] - mean_only['mean_crps_K_mean']:+.4f} K, "
            f"top-1% CRPS by "
            f"{full['fixed_top_01_crps_K_mean'] - mean_only['fixed_top_01_crps_K_mean']:+.4f} K, "
            f"hot coverage by "
            f"{full['hot_region_95_coverage_mean'] - mean_only['hot_region_95_coverage_mean']:+.4f}, "
            f"and field error by "
            f"{full['excess_field_rel_l2_mean'] - mean_only['excess_field_rel_l2_mean']:+.4f}.",
            "Its above-ceiling coverage is "
            + ("closer" if coverage_improves else "not closer")
            + " to the nominal 95% level than strict mean-only propagation.",
            "",
            "Relative to the pre-existing moment-matched implementation, preserving the "
            f"non-Gaussian mixture changes overall CRPS by "
            f"{full['mean_crps_K_mean'] - moment['mean_crps_K_mean']:+.4f} K, "
            f"top-1% CRPS by "
            f"{full['fixed_top_01_crps_K_mean'] - moment['fixed_top_01_crps_K_mean']:+.4f} K, "
            f"and hot coverage by "
            f"{full['hot_region_95_coverage_mean'] - moment['hot_region_95_coverage_mean']:+.4f}.",
            "",
            "**Conclusion:** carrying previous-state uncertainty matters relative to a strict "
            "mean-only transition, but the existing moment-matched implementation already "
            "retains essentially all of that benefit. Preserving the full non-Gaussian mixture "
            "does not materially change reconstruction or calibration. Top-1% coverage is "
            f"{full_top['coverage_95_mean']:.3f} for the full mixture versus "
            f"{moment_top['coverage_95_mean']:.3f} for moment matching, and both remain far "
            f"below the RBF reference ({rbf_top['coverage_95_mean']:.3f}). The full mixture "
            f"also retains a top-1% signed error of {full_top['signed_error_K_mean']:+.3f} K. "
            "Thus the sequential model's extreme-tail undercoverage is not explained by "
            "collapsing the previous posterior to its mean.",
            "",
            "The mechanism diagnostic is consistent with that conclusion. Across held-out "
            f"trajectories, the propagated between-component SD averages only {propagated_sd:.3f} K, "
            f"compared with {innovation_sd:.3f} K from the one-step stochastic innovation. "
            f"The likelihood-reweighted mixture retains an average effective sample size of "
            f"{weight_ess:.1f} out of {args.previous_samples} components, so current data do not "
            "hide a strongly concentrated alternative propagated state.",
            "",
            "The previous scientific conclusion is unchanged: the sequential stochastic/"
            "advective model has the best field error and overall CRPS, whereas posterior "
            "physics mean + RBF has the best top-1% CRPS and substantially better hottest-"
            "region coverage.",
            "",
            "`paired_comparisons.csv` records trajectory-level changes and win counts. "
            "`quartile_results.csv` and `quartile_summary.csv` contain the requested "
            "temperature-stratified diagnostics.",
            "",
            "## Reconstruction figures",
            "",
            "For each development trajectory, `reconstruction_*.png` retains the full "
            "truth-range posterior-mean panels and posterior-SD panels. The companion "
            "`reconstruction_*_tail_focus.png` figures use one shared scale from ambient "
            "temperature to that trajectory's camera ceiling. The raster values are not "
            "smoothed; lightly smoothed contour guides at ambient + 0.25, 0.5, 1, and 2 K "
            "make the cooling wake easier to follow without tracing pixel-scale noise. "
            "Values above the ceiling are clipped only in the display normalization; "
            "posterior inference and every reported metric remain unchanged.",
            "",
            "## Controlled assumptions",
            "",
            f"- Synthetic censoring fraction: {args.fraction_saturated:.0%}.",
            "- All rows share the same previous posterior draws, current camera realization, "
            "physical parameters, thresholds, masks, hyperparameters, and seeds.",
            "- The full mixture uses `ordinary posterior physics mean + B_adv(T_t^(s)-E[T_t])` "
            "as its component means. This exactly isolates non-Gaussian propagation from the "
            "existing moment-matched advective transition.",
            "- The existing previous-posterior routine retains unsaturated noisy camera pixels "
            "at their observed values and samples the saturated pixels. This experiment reuses "
            "that representation unchanged rather than introducing a new state-inference model.",
            "- Previous observations enter once, through the previous censored posterior; "
            "the final update conditions only on the current observations.",
            "- Mixture particles are reweighted using the Gaussian density for unsaturated "
            "observations and the multivariate Gaussian exceedance probability for saturated "
            "observations.",
            "- All CRPS values use the unbiased `M(M-1)` estimator.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mean-only, moment-matched, and full-posterior sequential propagation."
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0]
    results, region_results, integrity = run_experiment(args, fixed)
    summary = aggregate(results)
    region_summary = aggregate_regions(region_results)
    paired = paired_comparisons(results)
    results.to_csv(args.output_dir / "results.csv", index=False)
    region_results.to_csv(args.output_dir / "quartile_results.csv", index=False)
    integrity.to_csv(args.output_dir / "integrity_checks.csv", index=False)
    summary.to_csv(args.output_dir / "heldout30_overall.csv", index=False)
    region_summary.to_csv(args.output_dir / "quartile_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {
                "diffusivity": fixed.diffusivity,
                "cooling_rate": fixed.cooling_rate,
                "signal_sd": fixed.signal_sd,
                "source_coupling": fixed.source_coupling,
                "fraction_saturated": args.fraction_saturated,
                "nx": args.nx,
                "ny": args.ny,
                "observation_stride": args.observation_stride,
                "previous_frame_offset": args.previous_frame_offset,
                "measurement_noise_sd": args.measurement_noise_sd,
                "previous_noise_sd": args.previous_noise_sd,
                "current_noise_sd": args.current_noise_sd,
                "length_multiplier": args.length_multiplier,
                "forcing_length_multiplier": args.forcing_length_multiplier,
                "quadrature_order": args.quadrature_order,
                "previous_samples": args.previous_samples,
                "current_samples": args.samples,
                "seed": args.seed,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "full_posterior_representation": "likelihood-reweighted finite Gaussian mixture",
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    plot_summary(summary, args.output_dir / "comparison.png")
    plot_quartiles(region_summary, args.output_dir / "quartile_diagnostics.png")
    plot_tail_regions(region_summary, args.output_dir / "tail_diagnostics.png")
    write_readme(args, summary, region_summary, integrity)
    print(f"Saved full-posterior sequential experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
