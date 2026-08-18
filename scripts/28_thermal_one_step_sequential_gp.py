from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.censored_gp import RBFConfig, rbf_covariance
from src.dense_censored_gp import sample_censored_gaussian_blocks
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
from matplotlib.lines import Line2D


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "26_one_step_sequential_gp"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "24_posterior_physics_mean_advection"
    / "fixed_configuration.csv"
)
LEGACY_KERNEL_SUMMARY = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "19_stochastic_spde_ablation"
    / "heldout30_overall.csv"
)

METHODS = [
    "posterior physics mean + RBF",
    "advective posterior physics mean + RBF",
    "posterior physics mean + sequential stochastic ST",
    "posterior physics mean + sequential advective stochastic ST",
    "advective posterior physics mean + sequential advective stochastic ST",
]
HISTORICAL_METHOD = (
    "legacy latent-clipped physics mean + joint advective stochastic ST"
)
COMPARISON_METHODS = [*METHODS[:2], HISTORICAL_METHOD, *METHODS[2:]]
METHOD_COLORS = {
    METHODS[0]: "#0072B2",
    METHODS[1]: "#56B4E9",
    HISTORICAL_METHOD: "#E69F00",
    METHODS[2]: "#009E73",
    METHODS[3]: "#CC79A7",
    METHODS[4]: "#D55E00",
}
SHORT_LABELS = {
    METHODS[0]: "posterior mean\n+ RBF",
    METHODS[1]: "advective posterior mean\n+ RBF",
    HISTORICAL_METHOD: "legacy clipped mean\n+ joint advective ST",
    METHODS[2]: "posterior mean\n+ sequential ST",
    METHODS[3]: "posterior mean\n+ sequential advective ST",
    METHODS[4]: "advective posterior mean\n+ sequential advective ST",
}
PLOT_CODES = {
    method: chr(ord("A") + index)
    for index, method in enumerate(COMPARISON_METHODS)
}
METRICS = [
    "excess_field_rel_l2",
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "peak_absolute_error_K",
    "hot_region_95_coverage",
    "hot_region_95_interval_width_K",
]


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    return iy * len(prepared.xs) + ix


def covariance_diagnostics(
    prediction_variance: np.ndarray,
    observed_covariance: np.ndarray,
    innovation_variance: np.ndarray | None,
) -> dict[str, float]:
    diagonal = np.maximum(np.asarray(prediction_variance, dtype=float), 0.0)
    diagnostics = {
        "predictive_variance_mean_K2": float(np.mean(diagonal)),
        "predictive_variance_max_K2": float(np.max(diagonal)),
        "covariance_symmetry_error": float(
            np.max(np.abs(observed_covariance - observed_covariance.T))
        ),
    }
    if innovation_variance is not None:
        innovation_diagonal = np.maximum(
            np.asarray(innovation_variance, dtype=float), 0.0
        )
        diagnostics.update(
            {
                "innovation_variance_mean_K2": float(
                    np.mean(innovation_diagonal)
                ),
                "propagated_variance_mean_K2": float(
                    np.mean(diagonal - innovation_diagonal)
                ),
            }
        )
    else:
        diagnostics.update(
            {
                "innovation_variance_mean_K2": np.nan,
                "propagated_variance_mean_K2": np.nan,
            }
        )
    return diagnostics


def plot_reconstruction(
    prepared,
    *,
    current: dict[str, object],
    previous_mean: np.ndarray,
    ordinary_mean: np.ndarray,
    advective_mean: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, ...]],
    output_path: Path,
) -> None:
    panels = [
        ("True current field", prepared.truth),
        ("Censored current observation", current["clipped_full"]),
        ("Previous censored posterior mean", previous_mean),
        ("Posterior physics mean", ordinary_mean),
        ("Advective posterior physics mean", advective_mean),
    ]
    panels.extend(
        (
            SHORT_LABELS[method].replace("\n", " "),
            predictions[method][0].reshape(prepared.truth.shape),
        )
        for method in METHODS
    )
    extent = [
        prepared.xs[0],
        prepared.xs[-1],
        prepared.ys[0],
        prepared.ys[-1],
    ]

    def render(
        path: Path,
        *,
        minimum: float,
        maximum: float,
        cmap: str,
        subtitle: str,
        tail_contours: bool,
    ) -> None:
        figure, axes = plt.subplots(
            2,
            5,
            figsize=(19.0, 7.2),
            constrained_layout=True,
        )
        image = None
        contour_candidates = prepared.ambient + np.asarray([0.5, 1.0, 2.0])
        for axis, (title, field) in zip(axes.ravel(), panels):
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=minimum,
                vmax=maximum,
                aspect="auto",
            )
            if tail_contours:
                levels = contour_candidates[
                    (contour_candidates > np.nanmin(field))
                    & (contour_candidates < np.nanmax(field))
                ]
                if len(levels):
                    axis.contour(
                        prepared.xs,
                        prepared.ys,
                        field,
                        levels=levels,
                        colors="white",
                        linewidths=0.45,
                        alpha=0.65,
                    )
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        if image is None:
            raise RuntimeError("No reconstruction panels were rendered")
        colorbar = figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.018,
            pad=0.012,
            extend="max" if tail_contours else "neither",
        )
        colorbar.set_label("Temperature (K)")
        figure.suptitle(
            f"One-step sequential GP: {prepared.name}\n{subtitle}",
            fontsize=13,
        )
        figure.savefig(path, dpi=210)
        plt.close(figure)

    render(
        output_path,
        minimum=float(np.min(prepared.truth)),
        maximum=float(np.max(prepared.truth)),
        cmap="inferno",
        subtitle="Shared full-field temperature scale",
        tail_contours=False,
    )
    threshold = float(current["threshold"])
    render(
        output_path.with_name(f"{output_path.stem}_tail_focus{output_path.suffix}"),
        minimum=float(prepared.ambient),
        maximum=threshold,
        cmap="magma",
        subtitle=(
            f"Tail-focused scale: ambient to camera ceiling ({threshold:.2f} K); "
            "white contours are 0.5, 1, and 2 K above ambient"
        ),
        tail_contours=True,
    )


def run_experiment(
    args: argparse.Namespace,
    fixed: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = trajectory_catalog(args.dataset_dir)
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        records = [record for record in records if record.name in requested]
        missing = requested - {record.name for record in records}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    rows = []
    integrity_rows = []
    for trajectory_index, record in enumerate(records):
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
        stationary_config = StochasticHeatConfig(
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

        rbf_config = RBFConfig(
            mean_temp=prepared.ambient,
            signal_sd=float(fixed.signal_sd),
            lengthscale=lengthscale,
            noise_sd=args.current_noise_sd,
        )
        rbf_oo = rbf_covariance(observation_points, observation_points, rbf_config)
        rbf_po = rbf_covariance(current_points, observation_points, rbf_config)
        rbf_variance = np.full(len(current_points), rbf_config.signal_sd**2)

        dt = float(prepared.times[current_index] - prepared.times[previous_index])
        centered_draws = previous_draws - previous_mean[None, :, :]
        dx = float(np.mean(np.diff(prepared.xs)))
        dy = float(np.mean(np.diff(prepared.ys)))
        propagated_st = propagate_residual_draws(
            centered_draws,
            dx=dx,
            dy=dy,
            time_step=dt,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
        )
        propagated_adv = propagate_residual_draws(
            centered_draws,
            dx=dx,
            dy=dy,
            time_step=dt,
            diffusivity=float(fixed.diffusivity),
            cooling_rate=float(fixed.cooling_rate),
            displacement=displacement,
        )
        denominator = np.sqrt(len(previous_draws) - 1.0)
        features_st = propagated_st.reshape(len(previous_draws), -1).T / denominator
        features_adv = propagated_adv.reshape(len(previous_draws), -1).T / denominator
        observed_features_st = features_st[observed_indices]
        observed_features_adv = features_adv[observed_indices]

        innovation_oo = finite_step_innovation_covariance(
            observation_points, observation_points, stationary_config, dt
        )
        innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
        innovation_po = finite_step_innovation_covariance(
            current_points, observation_points, stationary_config, dt
        )
        innovation_variance_scalar = float(
            finite_step_innovation_covariance(
                current_points[:1], current_points[:1], stationary_config, dt
            )[0, 0]
        )
        innovation_variance = np.full(
            len(current_points), max(innovation_variance_scalar, 0.0)
        )
        sequential_st_oo = innovation_oo + observed_features_st.dot(
            observed_features_st.T
        )
        sequential_st_po = innovation_po + features_st.dot(
            observed_features_st.T
        )
        sequential_st_variance = innovation_variance + np.sum(features_st**2, axis=1)
        sequential_adv_oo = innovation_oo + observed_features_adv.dot(
            observed_features_adv.T
        )
        sequential_adv_po = innovation_po + features_adv.dot(
            observed_features_adv.T
        )
        sequential_adv_variance = innovation_variance + np.sum(
            features_adv**2, axis=1
        )

        def prior_blocks(mean_field, observed_covariance, pred_observed, variance):
            flat_mean = np.asarray(mean_field, dtype=float).ravel()
            return {
                "prediction_mean": flat_mean,
                "observation_mean": flat_mean[observed_indices],
                "observed_covariance": observed_covariance,
                "pred_observed_covariance": pred_observed,
                "prediction_variance": variance,
            }

        priors = {
            METHODS[0]: prior_blocks(ordinary_mean, rbf_oo, rbf_po, rbf_variance),
            METHODS[1]: prior_blocks(advective_mean, rbf_oo, rbf_po, rbf_variance),
            METHODS[2]: prior_blocks(
                ordinary_mean,
                sequential_st_oo,
                sequential_st_po,
                sequential_st_variance,
            ),
            METHODS[3]: prior_blocks(
                ordinary_mean,
                sequential_adv_oo,
                sequential_adv_po,
                sequential_adv_variance,
            ),
            METHODS[4]: prior_blocks(
                advective_mean,
                sequential_adv_oo,
                sequential_adv_po,
                sequential_adv_variance,
            ),
        }
        covariance_details = {
            METHODS[0]: covariance_diagnostics(rbf_variance, rbf_oo, None),
            METHODS[1]: covariance_diagnostics(rbf_variance, rbf_oo, None),
            METHODS[2]: covariance_diagnostics(
                sequential_st_variance, sequential_st_oo, innovation_variance
            ),
            METHODS[3]: covariance_diagnostics(
                sequential_adv_variance, sequential_adv_oo, innovation_variance
            ),
            METHODS[4]: covariance_diagnostics(
                sequential_adv_variance, sequential_adv_oo, innovation_variance
            ),
        }
        mean_labels = {
            METHODS[0]: "posterior physics mean",
            METHODS[1]: "advective posterior physics mean",
            METHODS[2]: "posterior physics mean",
            METHODS[3]: "posterior physics mean",
            METHODS[4]: "advective posterior physics mean",
        }
        covariance_labels = {
            METHODS[0]: "current spatial RBF",
            METHODS[1]: "current spatial RBF",
            METHODS[2]: "moment-matched sequential stochastic ST",
            METHODS[3]: "moment-matched sequential advective stochastic ST",
            METHODS[4]: "moment-matched sequential advective stochastic ST",
        }
        prediction_seed = args.seed + 100_000 * trajectory_index
        predictions = {}
        role = (
            "calibration"
            if record.name in DEVELOPMENT_TRAJECTORIES
            else "evaluation"
        )
        for method in METHODS:
            prediction = sample_censored_gaussian_blocks(
                camera["current"],
                **priors[method],
                noise_sd=args.current_noise_sd,
                n_samples=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=prediction_seed,
            )
            predictions[method] = prediction
            row = prediction_metrics(
                prepared,
                method=method,
                observations=camera["current"],
                prediction=prediction,
            )
            row.update(
                {
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "fraction_saturated": args.fraction_saturated,
                    "previous_frame_offset": args.previous_frame_offset,
                    "time_lag_s": float(
                        prepared.times[current_index]
                        - prepared.times[previous_index]
                    ),
                    "diffusivity_m2_s": float(fixed.diffusivity),
                    "cooling_rate_1_s": float(fixed.cooling_rate),
                    "source_coupling": float(fixed.source_coupling),
                    "signal_sd_K": float(fixed.signal_sd),
                    "lengthscale_m": lengthscale,
                    "forcing_lengthscale_m": (
                        stationary_config.forcing_lengthscale
                    ),
                    "previous_posterior_prior": (
                        "ambient-mean spatial RBF; saturated-pixel correction"
                    ),
                    "mean_construction": mean_labels[method],
                    "residual_covariance": covariance_labels[method],
                    "update_structure": "one-step sequential; current likelihood uses Y_n only",
                    "previous_observations_reused_in_current_update": False,
                    "same_previous_posterior_for_all_models": True,
                    "same_current_observations_for_all_models": True,
                    "same_sampler_seed_for_all_models": True,
                    "displacement_x_m": displacement[0],
                    "displacement_y_m": displacement[1],
                    "crps_estimator": "unbiased_M_times_M_minus_1",
                    **covariance_details[method],
                }
            )
            rows.append(row)
            print(
                f"[{trajectory_index + 1:02d}/{len(records):02d}] "
                f"{record.name}, {method}: "
                f"field={row['excess_field_rel_l2']:.3f}, "
                f"all={row['mean_crps_K']:.3f}, "
                f"top1={row['fixed_top_01_crps_K']:.3f}",
                flush=True,
            )

        integrity_rows.append(
            {
                "trajectory": record.name,
                "role": role,
                "previous_draws": len(previous_draws),
                "previous_saturated_pixels": int(
                    np.sum(camera["frames"][0]["saturated_full"])
                ),
                "previous_posterior_sd_mean_K": float(
                    np.mean(np.std(previous_draws, axis=0, ddof=1))
                ),
                **previous_diagnostics,
                "same_previous_posterior_for_all_models": True,
                "same_current_observations_for_all_models": True,
                "same_sampler_seed_for_all_models": True,
                "previous_observations_reused_in_current_update": False,
                "stationary_propagated_feature_frobenius": float(
                    np.linalg.norm(features_st)
                ),
                "advective_propagated_feature_frobenius": float(
                    np.linalg.norm(features_adv)
                ),
            }
        )
        if record.name in DEVELOPMENT_TRAJECTORIES:
            plot_reconstruction(
                prepared,
                current=camera["current"],
                previous_mean=previous_mean,
                ordinary_mean=ordinary_mean,
                advective_mean=advective_mean,
                predictions=predictions,
                output_path=(
                    args.output_dir
                    / f"reconstruction_{record.name.lower()}.png"
                ),
            )
        pd.DataFrame(rows).to_csv(args.output_dir / "checkpoint.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(integrity_rows)


def aggregate(results: pd.DataFrame, *, by_family: bool) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"]
    if subset.empty:
        subset = results
    groups = ["method"]
    if by_family:
        groups.insert(0, "family")
    summary = subset.groupby(groups, sort=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def build_comparison_summary(controlled: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_KERNEL_SUMMARY)
    legacy = legacy[
        legacy["method"] == "physics mean + advective stochastic space-time"
    ].copy()
    if len(legacy) != 1:
        raise ValueError("Could not identify the legacy advective-ST reference row")
    legacy.loc[:, "method"] = HISTORICAL_METHOD
    combined = pd.concat([controlled, legacy], ignore_index=True, sort=False)
    order = {method: index for index, method in enumerate(COMPARISON_METHODS)}
    combined["_order"] = combined["method"].map(order)
    if combined["_order"].isna().any():
        raise ValueError("Comparison summary contains an unknown method")
    return combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def architecture_table() -> pd.DataFrame:
    rows = [
        {
            "method": METHODS[0],
            "comparison_role": "primary mean control",
            "previous_frame_input": "observed noisy clipped camera frame",
            "previous_posterior": "ambient-mean spatial RBF; saturated-pixel correction",
            "deterministic_mean": "posterior physics mean",
            "residual_covariance": "current spatial RBF",
            "update_structure": "current-only censored likelihood",
            "Y_previous_usage": "once, in previous censored posterior",
        },
        {
            "method": METHODS[1],
            "comparison_role": "mean-advection control",
            "previous_frame_input": "observed noisy clipped camera frame",
            "previous_posterior": "ambient-mean spatial RBF; saturated-pixel correction",
            "deterministic_mean": "advective posterior physics mean",
            "residual_covariance": "current spatial RBF",
            "update_structure": "current-only censored likelihood",
            "Y_previous_usage": "once, in previous censored posterior",
        },
        {
            "method": HISTORICAL_METHOD,
            "comparison_role": "historical kernel reference",
            "previous_frame_input": "latent noise-free simulation field clipped at c",
            "previous_posterior": "none",
            "deterministic_mean": "legacy latent-clipped physics mean",
            "residual_covariance": "joint advective stochastic ST",
            "update_structure": "legacy joint two-frame censored likelihood",
            "Y_previous_usage": "joint residual likelihood; not camera-realistic",
        },
        {
            "method": METHODS[2],
            "comparison_role": "sequential covariance candidate",
            "previous_frame_input": "observed noisy clipped camera frame",
            "previous_posterior": "ambient-mean spatial RBF; saturated-pixel correction",
            "deterministic_mean": "posterior physics mean",
            "residual_covariance": "moment-matched sequential stochastic ST",
            "update_structure": "one-step sequential; current likelihood uses Y_n only",
            "Y_previous_usage": "once, in previous censored posterior",
        },
        {
            "method": METHODS[3],
            "comparison_role": "main sequential candidate",
            "previous_frame_input": "observed noisy clipped camera frame",
            "previous_posterior": "ambient-mean spatial RBF; saturated-pixel correction",
            "deterministic_mean": "posterior physics mean",
            "residual_covariance": "moment-matched sequential advective stochastic ST",
            "update_structure": "one-step sequential; current likelihood uses Y_n only",
            "Y_previous_usage": "once, in previous censored posterior",
        },
        {
            "method": METHODS[4],
            "comparison_role": "optional combined-transport diagnostic",
            "previous_frame_input": "observed noisy clipped camera frame",
            "previous_posterior": "ambient-mean spatial RBF; saturated-pixel correction",
            "deterministic_mean": "advective posterior physics mean",
            "residual_covariance": "moment-matched sequential advective stochastic ST",
            "update_structure": "one-step sequential; current likelihood uses Y_n only",
            "Y_previous_usage": "once, in previous censored posterior",
        },
    ]
    return pd.DataFrame(rows)


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"]
    if subset.empty:
        subset = results
    indexed = subset.set_index(["trajectory", "method"])
    names = sorted(subset["trajectory"].unique())
    rows = []
    pairings = [
        (METHODS[1], METHODS[0]),
        (METHODS[2], METHODS[0]),
        (METHODS[3], METHODS[0]),
        (METHODS[3], METHODS[2]),
        (METHODS[4], METHODS[3]),
        (METHODS[4], METHODS[1]),
    ]
    for method, baseline in pairings:
        row = {
            "method": method,
            "baseline": baseline,
            "n_trajectories": len(names),
        }
        for metric in METRICS:
            changes = np.asarray(
                [
                    indexed.loc[(name, method), metric]
                    - indexed.loc[(name, baseline), metric]
                    for name in names
                ]
            )
            row[f"{metric}_mean_change"] = float(np.mean(changes))
            row[f"{metric}_median_change"] = float(np.median(changes))
            if metric == "hot_region_95_coverage":
                row[f"{metric}_win_count"] = int(np.sum(changes > 0.0))
            else:
                row[f"{metric}_win_count"] = int(np.sum(changes < 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("excess_field_rel_l2", "Relative excess-field L2 error"),
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_95_coverage", "Hot-region 95% coverage"),
        ("hot_region_95_interval_width_K", "Hot-region 95% width (K)"),
    ]
    indexed = summary.set_index("method")
    x = np.arange(len(COMPARISON_METHODS))
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        means = np.asarray(
            [indexed.loc[method, f"{metric}_mean"] for method in COMPARISON_METHODS]
        )
        errors = np.asarray(
            [
                indexed.loc[method, f"{metric}_std"]
                / np.sqrt(indexed.loc[method, f"{metric}_count"])
                for method in COMPARISON_METHODS
            ]
        )
        axis.errorbar(
            x,
            means,
            yerr=np.nan_to_num(errors),
            linestyle="none",
            color="#444444",
            capsize=4,
        )
        for index, method in enumerate(COMPARISON_METHODS):
            axis.scatter(index, means[index], color=METHOD_COLORS[method], s=58, zorder=3)
        axis.set_xticks(x, [PLOT_CODES[method] for method in COMPARISON_METHODS])
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "hot_region_95_coverage":
            axis.axhline(0.95, color="#777777", linestyle="--", linewidth=1.0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=METHOD_COLORS[method],
            label=f"{PLOT_CODES[method]}: {method}",
        )
        for method in COMPARISON_METHODS
    ]
    figure.suptitle("Architecture-aware sequential GP comparison")
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.91,
        bottom=0.16,
        wspace=0.32,
        hspace=0.34,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_readme(
    args: argparse.Namespace,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    table = comparison.set_index("method")
    baseline = table.loc[METHODS[0]]
    sequential = table.loc[METHODS[3]]
    stationary = table.loc[METHODS[2]]
    lines = [
        "# Architecture-aware one-step sequential GP comparison",
        "",
        "This comparison restores the canonical ambient-mean RBF censored posterior "
        "for the previous frame. Every newly fitted row uses the same posterior draws, "
        "camera realization, physical parameters, and current-only censored likelihood.",
        "",
        "For the sequential rows, the stochastic heat GP supplies the transition B and "
        "conditional innovation C. Previous RBF-posterior uncertainty Sigma_- is then "
        "propagated by moment matching:",
        "",
        "```text",
        "T_n | Y_(n-1) approximately N(m_n^physics, C + B Sigma_- B^T).",
        "```",
        "",
        "## Results",
        "",
        "| Method | Field | All CRPS | Top-1% CRPS | Peak error | Coverage | Hot width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in COMPARISON_METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['mean_crps_K_mean']:.3f} | "
            f"{row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} | "
            f"{row['hot_region_95_interval_width_K_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Controlled design",
            "",
            f"- Synthetic censoring: {args.fraction_saturated:.0%}.",
            "- One shared previous censored posterior and current camera realization.",
            "- Identical physical parameters, current likelihood, sampler settings, and seeds.",
            "- Previous observations are not supplied to the current likelihood.",
            "- All CRPS uses the unbiased M(M-1) estimator.",
            "- The forcing lengthscale is inherited from the frozen residual setup; "
            "this run does not add a new hyperparameter search.",
            "",
            "## Reconstruction scales",
            "",
            "- `reconstruction_*.png` uses one shared linear scale spanning the full "
            "true-field temperature range.",
            "- `reconstruction_*_tail_focus.png` uses one shared linear scale from "
            "ambient temperature to the synthetic camera ceiling. Values above the "
            "ceiling are clipped only in the visualization, not in inference or scoring.",
            "- White contours in the tail-focused figures mark 0.5, 1, and 2 K above "
            "ambient, making the cooling trail easier to compare across models.",
            "",
            "## Architecture notes following the result table",
            "",
            "- The first two rows are controlled current-only RBF models. They differ only "
            "in whether the posterior physics mean is translated before diffusion.",
            "- The legacy latent-clipped row is a historical kernel reference. It uses a "
            "noise-free simulation predecessor and a joint two-frame likelihood, so its raw "
            "scores are descriptive and not a paired comparison with the sequential rows.",
            "- The two main sequential rows use the ordinary posterior physics mean. They "
            "differ only by advection in the stochastic ST transition, and the final update "
            "conditions only on Y_n.",
            "- The final row is an optional diagnostic that advects both the deterministic "
            "mean and the residual transition. It is not the primary integrated candidate.",
            "",
            "## Numerical interpretation",
            "",
            "Relative to posterior physics mean + RBF, the main sequential advective-ST "
            f"candidate changes field error by "
            f"{sequential['excess_field_rel_l2_mean'] - baseline['excess_field_rel_l2_mean']:+.4f}, "
            f"all-domain CRPS by "
            f"{sequential['mean_crps_K_mean'] - baseline['mean_crps_K_mean']:+.4f} K, "
            f"top-1% CRPS by "
            f"{sequential['fixed_top_01_crps_K_mean'] - baseline['fixed_top_01_crps_K_mean']:+.4f} K, "
            f"and coverage by "
            f"{sequential['hot_region_95_coverage_mean'] - baseline['hot_region_95_coverage_mean']:+.4f}.",
            "Adding residual advection to the sequential stochastic-ST transition changes "
            f"all-domain CRPS by "
            f"{sequential['mean_crps_K_mean'] - stationary['mean_crps_K_mean']:+.4f} K "
            "under the same posterior physics mean.",
            "",
            "Exact architecture labels are recorded in architecture.csv, and controlled "
            "paired changes are recorded in paired_comparisons.csv.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one-step sequential censored GP experiment.")
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
    results, integrity = run_experiment(args, fixed)
    results.to_csv(args.output_dir / "results.csv", index=False)
    integrity.to_csv(args.output_dir / "integrity_checks.csv", index=False)
    overall = aggregate(results, by_family=False)
    family = aggregate(results, by_family=True)
    paired = paired_comparisons(results)
    comparison = build_comparison_summary(overall)
    architectures = architecture_table()
    overall.to_csv(args.output_dir / "heldout30_overall.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    family.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    architectures.to_csv(args.output_dir / "architecture.csv", index=False)
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
                "sequential_covariance_evaluation": (
                    "finite-step innovation plus propagated previous-posterior features"
                ),
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    plot_summary(comparison, args.output_dir / "comparison.png")
    write_readme(args, comparison, paired)
    print(f"Saved one-step sequential GP experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
