from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.censored_gp import RBFConfig, rbf_covariance
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
    infer_previous_censored_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prediction_metrics,
    prepare_trajectory,
)
from src.thermal_plotting import add_tail_contours, tail_temperature_norm
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "30_final_architecture_comparison"
)
SEQUENTIAL_OUTPUT = ROOT / "outputs" / "by_experiment" / "26_one_step_sequential_gp"
MIXTURE_OUTPUT = ROOT / "outputs" / "by_experiment" / "27_full_posterior_sequential"
SCALE_OUTPUT = ROOT / "outputs" / "by_experiment" / "28_hottest_tail_diagnostic"
LEGACY_OUTPUT = ROOT / "outputs" / "by_experiment" / "19_stochastic_spde_ablation"

POSTERIOR_RBF = "Posterior physics mean + RBF"
ADVECTIVE_POSTERIOR_RBF = "Advective posterior physics mean + RBF"
LEGACY_JOINT_ADV_ST = "Legacy latent-clipped mean + joint advective ST"
SEQUENTIAL_ST = "Posterior physics mean + sequential ST (moment-matched)"
SEQUENTIAL_ADV_ST = (
    "Posterior physics mean + sequential advective ST (moment-matched)"
)
ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST = (
    "Advective posterior mean + sequential advective ST (moment-matched)"
)
POSTERIOR_SAMPLE_MIXTURE = (
    "Posterior-sample mixture sequential advective ST"
)
STRICT_MEAN_ONLY = "Strict mean-only sequential advective ST"
MATCHED_RBF = "Variance-matched posterior physics mean + RBF (diagnostic)"

MAIN_METHODS = [
    POSTERIOR_RBF,
    ADVECTIVE_POSTERIOR_RBF,
    LEGACY_JOINT_ADV_ST,
    SEQUENTIAL_ST,
    SEQUENTIAL_ADV_ST,
    ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST,
    POSTERIOR_SAMPLE_MIXTURE,
]

SEQUENTIAL_SOURCE_NAMES = {
    "posterior physics mean + RBF": POSTERIOR_RBF,
    "advective posterior physics mean + RBF": ADVECTIVE_POSTERIOR_RBF,
    "posterior physics mean + sequential stochastic ST": SEQUENTIAL_ST,
    "posterior physics mean + sequential advective stochastic ST": SEQUENTIAL_ADV_ST,
    "advective posterior physics mean + sequential advective stochastic ST": (
        ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST
    ),
}

REGIONS = ["overall", "above_camera_ceiling", "top_1pct"]
REGION_METRICS = [
    "rmse_K",
    "mae_K",
    "signed_error_K",
    "crps_K",
    "coverage_95",
    "interval_width_95_K",
    "posterior_sd_K",
]


def load_script_module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_script_module(
    "final_comparison_stochastic_ablation",
    "21_thermal_stochastic_spde_ablation.py",
)
mixture = load_script_module(
    "final_comparison_posterior_mixture",
    "29_thermal_full_posterior_sequential.py",
)


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    return iy * len(prepared.xs) + ix


def region_rows(
    prepared,
    *,
    method: str,
    role: str,
    threshold: float,
    prediction: tuple[np.ndarray, ...],
) -> list[dict[str, object]]:
    mean, sd, lower, upper, draws = prediction
    truth = prepared.truth.ravel()
    crps = mixture.empirical_crps(draws, truth)
    masks = {
        "overall": np.ones(len(truth), dtype=bool),
        "above_camera_ceiling": truth >= threshold,
        "top_1pct": truth >= float(np.quantile(truth, 0.99)),
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


def overall_row(
    prepared,
    *,
    method: str,
    role: str,
    observations: dict[str, object],
    prediction: tuple[np.ndarray, ...],
) -> dict[str, object]:
    row = prediction_metrics(
        prepared,
        method=method,
        observations=observations,
        prediction=prediction,
    )
    error = prediction[0] - prepared.truth.ravel()
    row.update(
        {
            "role": role,
            "rmse_K": float(np.sqrt(np.mean(error**2))),
            "signed_error_K": float(np.mean(error)),
        }
    )
    return row


def sequential_predictions(
    prepared,
    *,
    camera: dict[str, object],
    previous_mean: np.ndarray,
    previous_draws: np.ndarray,
    ordinary_mean: np.ndarray,
    advective_mean: np.ndarray,
    displacement: np.ndarray,
    fixed: pd.Series,
    current_noise_sd: float,
    length_multiplier: float,
    forcing_length_multiplier: float,
    quadrature_order: int,
    samples: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, float]]:
    current = camera["current"]
    current_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed_indices = observation_indices(prepared, observation_points)
    lengthscale = prepared.source_lengthscale * length_multiplier
    config = StochasticHeatConfig(
        signal_sd=float(fixed.signal_sd),
        forcing_lengthscale=lengthscale * forcing_length_multiplier,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        quadrature_order=quadrature_order,
    )

    rbf_config = RBFConfig(
        mean_temp=prepared.ambient,
        signal_sd=float(fixed.signal_sd),
        lengthscale=lengthscale,
        noise_sd=current_noise_sd,
    )
    rbf_oo = rbf_covariance(observation_points, observation_points, rbf_config)
    rbf_po = rbf_covariance(current_points, observation_points, rbf_config)
    rbf_variance = np.full(len(current_points), rbf_config.signal_sd**2)

    previous_index = int(camera["frames"][0]["time_index"])
    current_index = int(camera["frames"][1]["time_index"])
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    centered = previous_draws - previous_mean[None, :, :]
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    propagated_st = propagate_residual_draws(
        centered,
        dx=dx,
        dy=dy,
        time_step=dt,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
    )
    propagated_adv = propagate_residual_draws(
        centered,
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
    sequential_st_oo = innovation_oo + observed_features_st.dot(
        observed_features_st.T
    )
    sequential_st_po = innovation_po + features_st.dot(observed_features_st.T)
    sequential_st_variance = innovation_variance + np.sum(features_st**2, axis=1)
    sequential_adv_oo = innovation_oo + observed_features_adv.dot(
        observed_features_adv.T
    )
    sequential_adv_po = innovation_po + features_adv.dot(observed_features_adv.T)
    sequential_adv_variance = innovation_variance + np.sum(features_adv**2, axis=1)

    def blocks(mean_field, covariance, cross_covariance, variance):
        flat = np.asarray(mean_field, dtype=float).ravel()
        return {
            "prediction_mean": flat,
            "observation_mean": flat[observed_indices],
            "observed_covariance": covariance,
            "pred_observed_covariance": cross_covariance,
            "prediction_variance": variance,
        }

    priors = {
        POSTERIOR_RBF: blocks(ordinary_mean, rbf_oo, rbf_po, rbf_variance),
        ADVECTIVE_POSTERIOR_RBF: blocks(
            advective_mean, rbf_oo, rbf_po, rbf_variance
        ),
        SEQUENTIAL_ST: blocks(
            ordinary_mean,
            sequential_st_oo,
            sequential_st_po,
            sequential_st_variance,
        ),
        SEQUENTIAL_ADV_ST: blocks(
            ordinary_mean,
            sequential_adv_oo,
            sequential_adv_po,
            sequential_adv_variance,
        ),
        ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST: blocks(
            advective_mean,
            sequential_adv_oo,
            sequential_adv_po,
            sequential_adv_variance,
        ),
    }
    predictions = {
        method: sample_censored_gaussian_blocks(
            current,
            **prior,
            noise_sd=current_noise_sd,
            n_samples=samples,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )
        for method, prior in priors.items()
    }
    diagnostics = {
        "rbf_marginal_sd_K": float(rbf_config.signal_sd),
        "st_innovation_sd_K": float(np.sqrt(innovation_variance_scalar)),
        "time_lag_s": dt,
    }
    return predictions, diagnostics


def run_sequential_architectures(
    args: argparse.Namespace,
    fixed: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    regions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    records = trajectory_catalog(args.dataset_dir)
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
        threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
        camera = paired_camera_observations(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            threshold=threshold,
            observation_stride=args.observation_stride,
            noise_sd=args.measurement_noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        lengthscale = prepared.source_lengthscale * args.length_multiplier
        previous_mean, previous_draws, _ = infer_previous_censored_posterior(
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
        predictions, covariance_diagnostics = sequential_predictions(
            prepared,
            camera=camera,
            previous_mean=previous_mean,
            previous_draws=previous_draws,
            ordinary_mean=ordinary_mean,
            advective_mean=advective_mean,
            displacement=displacement,
            fixed=fixed,
            current_noise_sd=args.current_noise_sd,
            length_multiplier=args.length_multiplier,
            forcing_length_multiplier=args.forcing_length_multiplier,
            quadrature_order=args.quadrature_order,
            samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=args.seed + 100_000 * trajectory_index,
        )
        role = "calibration" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        for method, prediction in predictions.items():
            row = overall_row(
                prepared,
                method=method,
                role=role,
                observations=camera["current"],
                prediction=prediction,
            )
            rows.append(row)
            regions.extend(
                region_rows(
                    prepared,
                    method=method,
                    role=role,
                    threshold=threshold,
                    prediction=prediction,
                )
            )
        diagnostics.append(
            {
                "trajectory": record.name,
                "role": role,
                **covariance_diagnostics,
            }
        )
        print(
            f"[sequential {trajectory_index + 1:02d}/{len(records):02d}] {record.name}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(
            args.output_dir / "sequential_rerun_results.csv", index=False
        )
        pd.DataFrame(regions).to_csv(
            args.output_dir / "sequential_rerun_regions.csv", index=False
        )
    return pd.DataFrame(rows), pd.DataFrame(regions), pd.DataFrame(diagnostics)


def run_legacy_joint_architecture(
    args: argparse.Namespace,
    fixed: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    regions: list[dict[str, object]] = []
    records = trajectory_catalog(args.dataset_dir)
    legacy_args = SimpleNamespace(
        noise_sd=args.measurement_noise_sd,
        noise_multiplier=2.0,
        signal_multiplier=1.0,
        beta_multiplier=1.0,
        length_multiplier=args.length_multiplier,
        forcing_length_multiplier=args.forcing_length_multiplier,
        forcing_quadrature_order=args.quadrature_order,
        source_flux_threshold=args.source_flux_threshold,
        source_coupling=float(fixed.source_coupling),
    )
    parameters = {
        "diffusivity": float(fixed.diffusivity),
        "cooling_rate": float(fixed.cooling_rate),
        "signal_sd": float(fixed.signal_sd),
    }
    source_method = "physics mean + advective stochastic space-time"
    for trajectory_index, record in enumerate(records):
        prepared, _ = legacy.thermal.prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
        history_indices = legacy.study.select_history_indices(
            prepared.times, [0.0, 0.01]
        )
        multitime = legacy.corrected.fixed_mask_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.measurement_noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current = legacy.study.current_frame_observations(multitime)
        _, physics_mean_function = legacy.ablation.physics_mean_for_trajectory(
            prepared,
            threshold=threshold,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=float(fixed.source_coupling),
            source_flux_threshold=args.source_flux_threshold,
        )
        models = legacy.make_model_configurations(
            legacy_args,
            prepared,
            current=current,
            multitime=multitime,
            physics_mean_function=physics_mean_function,
            parameters=parameters,
        )
        observations, config, _ = models[source_method]
        _, prediction = legacy.ablation.run_method(
            prepared,
            method=source_method,
            observations=observations,
            config=config,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=float(fixed.source_coupling),
            n_frames=len(history_indices),
            chains=1,
            samples=args.samples,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=args.seed + 100_000 * trajectory_index,
            return_prediction=True,
        )
        role = "calibration" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        rows.append(
            overall_row(
                prepared,
                method=LEGACY_JOINT_ADV_ST,
                role=role,
                observations=current,
                prediction=prediction,
            )
        )
        regions.extend(
            region_rows(
                prepared,
                method=LEGACY_JOINT_ADV_ST,
                role=role,
                threshold=threshold,
                prediction=prediction,
            )
        )
        print(
            f"[legacy {trajectory_index + 1:02d}/{len(records):02d}] {record.name}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(
            args.output_dir / "legacy_rerun_results.csv", index=False
        )
        pd.DataFrame(regions).to_csv(
            args.output_dir / "legacy_rerun_regions.csv", index=False
        )
    return pd.DataFrame(rows), pd.DataFrame(regions)


def reused_mixture_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(MIXTURE_OUTPUT / "results.csv")
    regions = pd.read_csv(MIXTURE_OUTPUT / "quartile_results.csv")
    source = "full-posterior mixture sequential advective stochastic ST"
    results = results[results["method"] == source].copy()
    regions = regions[
        (regions["method"] == source) & regions["region"].isin(REGIONS)
    ].copy()
    results["method"] = POSTERIOR_SAMPLE_MIXTURE
    regions["method"] = POSTERIOR_SAMPLE_MIXTURE
    return results, regions


def architecture_metadata(rbf_sd: float, innovation_sd: float) -> pd.DataFrame:
    rows = [
        {
            "method": POSTERIOR_RBF,
            "source_experiment_script": "scripts/28_thermal_one_step_sequential_gp.py",
            "source_output": "outputs/by_experiment/26_one_step_sequential_gp",
            "reuse_status": "rerun for common regional diagnostics; exact reproduction checked",
            "exact_architecture": "current-only censored GP",
            "prior_mean": "posterior physics mean",
            "covariance_type": "spatial RBF",
            "update_structure": "current-only",
            "previous_state_uncertainty": "previous censored posterior collapsed to its mean",
            "marginal_covariance_or_innovation_sd_K": rbf_sd,
        },
        {
            "method": ADVECTIVE_POSTERIOR_RBF,
            "source_experiment_script": "scripts/28_thermal_one_step_sequential_gp.py",
            "source_output": "outputs/by_experiment/26_one_step_sequential_gp",
            "reuse_status": "rerun for common regional diagnostics; exact reproduction checked",
            "exact_architecture": "current-only censored GP",
            "prior_mean": "advective posterior physics mean",
            "covariance_type": "spatial RBF",
            "update_structure": "current-only",
            "previous_state_uncertainty": "previous censored posterior collapsed to its mean",
            "marginal_covariance_or_innovation_sd_K": rbf_sd,
        },
        {
            "method": LEGACY_JOINT_ADV_ST,
            "source_experiment_script": "scripts/21_thermal_stochastic_spde_ablation.py",
            "source_output": "outputs/by_experiment/19_stochastic_spde_ablation",
            "reuse_status": "rerun for common regional diagnostics; exact reproduction checked",
            "exact_architecture": "joint two-frame censored GP",
            "prior_mean": "legacy noise-free latent-clipped physics mean",
            "covariance_type": "stationary advective stochastic space-time",
            "update_structure": "joint previous and current likelihood",
            "previous_state_uncertainty": "raw previous observations enter joint GP",
            "marginal_covariance_or_innovation_sd_K": rbf_sd,
        },
        {
            "method": SEQUENTIAL_ST,
            "source_experiment_script": "scripts/28_thermal_one_step_sequential_gp.py",
            "source_output": "outputs/by_experiment/26_one_step_sequential_gp",
            "reuse_status": "rerun for common regional diagnostics; exact reproduction checked",
            "exact_architecture": "one-step sequential censored GP",
            "prior_mean": "posterior physics mean",
            "covariance_type": "finite-step stochastic ST C + B Sigma B^T",
            "update_structure": "sequential; current likelihood uses Y_n only",
            "previous_state_uncertainty": "censored-region posterior moment matched",
            "marginal_covariance_or_innovation_sd_K": innovation_sd,
        },
        {
            "method": SEQUENTIAL_ADV_ST,
            "source_experiment_script": "scripts/28 and 29 (same implementation)",
            "source_output": "outputs/by_experiment/26 and 27",
            "reuse_status": "rerun; duplicate requested labels collapsed to one row",
            "exact_architecture": "one-step sequential censored GP",
            "prior_mean": "posterior physics mean",
            "covariance_type": "finite-step advective stochastic ST C + B_adv Sigma B_adv^T",
            "update_structure": "sequential; current likelihood uses Y_n only",
            "previous_state_uncertainty": "censored-region posterior moment matched",
            "marginal_covariance_or_innovation_sd_K": innovation_sd,
        },
        {
            "method": ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST,
            "source_experiment_script": "scripts/28_thermal_one_step_sequential_gp.py",
            "source_output": "outputs/by_experiment/26_one_step_sequential_gp",
            "reuse_status": "rerun for common regional diagnostics; exact reproduction checked",
            "exact_architecture": "one-step sequential censored GP",
            "prior_mean": "advective posterior physics mean",
            "covariance_type": "finite-step advective stochastic ST C + B_adv Sigma B_adv^T",
            "update_structure": "sequential; current likelihood uses Y_n only",
            "previous_state_uncertainty": "censored-region posterior moment matched",
            "marginal_covariance_or_innovation_sd_K": innovation_sd,
        },
        {
            "method": POSTERIOR_SAMPLE_MIXTURE,
            "source_experiment_script": "scripts/29_thermal_full_posterior_sequential.py",
            "source_output": "outputs/by_experiment/27_full_posterior_sequential",
            "reuse_status": "reused exact saved held-out posterior metrics",
            "exact_architecture": "one-step sequential finite Gaussian-mixture censored GP",
            "prior_mean": "posterior physics mean plus propagated centered sample component",
            "covariance_type": "shared finite-step advective innovation C",
            "update_structure": "sequential mixture; current likelihood uses Y_n only",
            "previous_state_uncertainty": "censored-region posterior samples retained and reweighted",
            "marginal_covariance_or_innovation_sd_K": innovation_sd,
        },
    ]
    return pd.DataFrame(rows)


def aggregate_architectures(
    overall_results: pd.DataFrame,
    region_results: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    overall = overall_results[overall_results["role"] == "evaluation"]
    regions = region_results[region_results["role"] == "evaluation"]
    summary = (
        overall.groupby("method", sort=False)[
            ["excess_field_rel_l2", "rmse_K", "mean_crps_K", "peak_absolute_error_K"]
        ]
        .mean()
        .rename(
            columns={
                "excess_field_rel_l2": "field_excess_rel_l2",
                "mean_crps_K": "overall_crps_K",
            }
        )
    )
    for region, prefix in [
        ("above_camera_ceiling", "above_ceiling"),
        ("top_1pct", "top_1pct"),
    ]:
        region_summary = (
            regions[regions["region"] == region]
            .groupby("method", sort=False)[REGION_METRICS]
            .mean()
            .rename(columns={metric: f"{prefix}_{metric}" for metric in REGION_METRICS})
        )
        summary = summary.join(region_summary)
    summary = summary.reindex(MAIN_METHODS).reset_index()
    summary = summary.merge(metadata, on="method", how="left")
    summary["heldout_trajectory_count"] = 30
    summary["development_trajectory_count"] = 3
    summary["trajectory_weighting"] = "equal weight per trajectory"
    summary["censoring_fraction"] = 0.03
    summary["observation_stride"] = 5
    summary["measurement_noise_sd_K"] = 0.25
    summary["likelihood_noise_sd_K"] = 0.5
    summary["crps_estimator"] = "unbiased M(M-1)"
    return summary


def reproduction_checks(
    sequential_results: pd.DataFrame,
    legacy_results: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    saved_sequential = pd.read_csv(SEQUENTIAL_OUTPUT / "results.csv")
    for source_name, final_name in SEQUENTIAL_SOURCE_NAMES.items():
        rerun = sequential_results[sequential_results["method"] == final_name].set_index(
            "trajectory"
        )
        saved = saved_sequential[saved_sequential["method"] == source_name].set_index(
            "trajectory"
        )
        for metric in [
            "excess_field_rel_l2",
            "mean_crps_K",
            "fixed_top_01_crps_K",
            "peak_absolute_error_K",
            "hot_region_95_coverage",
            "hot_region_95_interval_width_K",
        ]:
            difference = rerun.loc[saved.index, metric] - saved[metric]
            checks.append(
                {
                    "method": final_name,
                    "metric": metric,
                    "max_absolute_difference": float(np.max(np.abs(difference))),
                }
            )
    saved_legacy = pd.read_csv(LEGACY_OUTPUT / "results.csv")
    saved_legacy = saved_legacy[
        saved_legacy["method"] == "physics mean + advective stochastic space-time"
    ].set_index("trajectory")
    rerun_legacy = legacy_results.set_index("trajectory")
    for metric in [
        "excess_field_rel_l2",
        "mean_crps_K",
        "fixed_top_01_crps_K",
        "peak_absolute_error_K",
        "hot_region_95_coverage",
        "hot_region_95_interval_width_K",
    ]:
        difference = rerun_legacy.loc[saved_legacy.index, metric] - saved_legacy[metric]
        checks.append(
            {
                "method": LEGACY_JOINT_ADV_ST,
                "metric": metric,
                "max_absolute_difference": float(np.max(np.abs(difference))),
            }
        )
    return pd.DataFrame(checks)


def scale_control_table(innovation_sd: float, rbf_sd: float) -> pd.DataFrame:
    saved = pd.read_csv(SCALE_OUTPUT / "variance_matched_region_summary.csv")
    moment_regions = pd.read_csv(MIXTURE_OUTPUT / "quartile_summary.csv")
    source_names = {
        POSTERIOR_RBF: "posterior physics mean + RBF",
        MATCHED_RBF: "posterior physics mean + variance-matched RBF (diagnostic)",
        POSTERIOR_SAMPLE_MIXTURE: (
            "full-posterior mixture sequential advective stochastic ST"
        ),
    }
    rows = []
    for method, source in source_names.items():
        subset = saved[saved["method"] == source].set_index("region")
        rows.append(
            {
                "method": method,
                "overall_crps_K": subset.loc["overall", "crps_K"],
                "above_ceiling_crps_K": subset.loc[
                    "above_camera_ceiling", "crps_K"
                ],
                "top_1pct_crps_K": subset.loc["top_1pct", "crps_K"],
                "top_1pct_rmse_K": subset.loc["top_1pct", "rmse_K"],
                "top_1pct_coverage_95": subset.loc["top_1pct", "coverage_95"],
                "top_1pct_interval_width_95_K": subset.loc[
                    "top_1pct", "interval_width_95_K"
                ],
                "marginal_covariance_or_innovation_sd_K": (
                    rbf_sd if method == POSTERIOR_RBF else innovation_sd
                ),
            }
        )
    moment = moment_regions[
        moment_regions["method"]
        == "moment-matched sequential advective stochastic ST"
    ].set_index("region")
    rows.insert(
        2,
        {
            "method": SEQUENTIAL_ADV_ST,
            "overall_crps_K": moment.loc["overall", "crps_K_mean"],
            "above_ceiling_crps_K": moment.loc[
                "above_camera_ceiling", "crps_K_mean"
            ],
            "top_1pct_crps_K": moment.loc["top_1pct", "crps_K_mean"],
            "top_1pct_rmse_K": moment.loc["top_1pct", "rmse_K_mean"],
            "top_1pct_coverage_95": moment.loc[
                "top_1pct", "coverage_95_mean"
            ],
            "top_1pct_interval_width_95_K": moment.loc[
                "top_1pct", "interval_width_95_K_mean"
            ],
            "marginal_covariance_or_innovation_sd_K": innovation_sd,
        },
    )
    return pd.DataFrame(rows)


def propagation_ablation_table() -> pd.DataFrame:
    overall = pd.read_csv(MIXTURE_OUTPUT / "heldout30_overall.csv").set_index("method")
    regions = pd.read_csv(MIXTURE_OUTPUT / "quartile_summary.csv")
    source_names = {
        STRICT_MEAN_ONLY: "mean-only sequential advective stochastic ST",
        SEQUENTIAL_ADV_ST: "moment-matched sequential advective stochastic ST",
        POSTERIOR_SAMPLE_MIXTURE: (
            "full-posterior mixture sequential advective stochastic ST"
        ),
    }
    rows = []
    for method, source in source_names.items():
        selected = regions[regions["method"] == source].set_index("region")
        rows.append(
            {
                "method": method,
                "overall_crps_K": overall.loc[source, "mean_crps_K_mean"],
                "top_1pct_crps_K": overall.loc[
                    source, "fixed_top_01_crps_K_mean"
                ],
                "above_ceiling_coverage_95": selected.loc[
                    "above_camera_ceiling", "coverage_95_mean"
                ],
                "top_1pct_coverage_95": selected.loc[
                    "top_1pct", "coverage_95_mean"
                ],
                "top_1pct_signed_error_K": selected.loc[
                    "top_1pct", "signed_error_K_mean"
                ],
                "posterior_representation": {
                    STRICT_MEAN_ONLY: "previous posterior collapsed to mean; C only",
                    SEQUENTIAL_ADV_ST: "Gaussian moment match C + B Sigma B^T",
                    POSTERIOR_SAMPLE_MIXTURE: (
                        "censored-region posterior-sample mixture with shared C"
                    ),
                }[method],
            }
        )
    return pd.DataFrame(rows)


def plot_whole_field_vs_tail(table: pd.DataFrame, output_path: Path) -> None:
    colors = ["#0072B2", "#56B4E9", "#CC79A7", "#009E73", "#D55E00", "#E69F00", "#332288"]
    labels = [
        "PPM + RBF",
        "adv. PPM + RBF",
        "legacy joint adv.-ST",
        "seq. ST",
        "seq. adv.-ST",
        "adv. mean + seq. adv.-ST",
        "sample mixture",
    ]
    offsets = [(8, -10), (8, 8), (8, 7), (20, -40), (20, 8), (8, 8), (20, -18)]
    figure, axis = plt.subplots(figsize=(8.7, 6.3), constrained_layout=True)
    for index, row in table.iterrows():
        axis.scatter(
            row["overall_crps_K"],
            row["top_1pct_crps_K"],
            s=85,
            color=colors[index],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            labels[index],
            (row["overall_crps_K"], row["top_1pct_crps_K"]),
            xytext=offsets[index],
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Overall CRPS (K), lower is better")
    axis.set_ylabel("Top-1% CRPS (K), lower is better")
    axis.set_title("Whole-field versus hottest-tail probabilistic performance")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_tail_coverage_vs_width(
    table: pd.DataFrame,
    scale_control: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = ["#0072B2", "#56B4E9", "#CC79A7", "#009E73", "#D55E00", "#E69F00", "#332288"]
    labels = [
        "PPM + RBF",
        "adv. PPM + RBF",
        "legacy joint adv.-ST",
        "seq. ST",
        "seq. adv.-ST",
        "adv. mean + seq. adv.-ST",
        "sample mixture",
    ]
    offsets = [(8, 8), (8, -12), (8, 8), (20, 32), (20, -10), (20, -6), (20, 10)]
    figure, axis = plt.subplots(figsize=(8.7, 6.3), constrained_layout=True)
    for index, row in table.iterrows():
        axis.scatter(
            row["top_1pct_interval_width_95_K"],
            row["top_1pct_coverage_95"],
            s=85,
            color=colors[index],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            labels[index],
            (row["top_1pct_interval_width_95_K"], row["top_1pct_coverage_95"]),
            xytext=offsets[index],
            textcoords="offset points",
            fontsize=9,
        )
    matched = scale_control[scale_control["method"] == MATCHED_RBF].iloc[0]
    axis.scatter(
        matched["top_1pct_interval_width_95_K"],
        matched["top_1pct_coverage_95"],
        s=150,
        marker="*",
        facecolor="white",
        edgecolor="#0072B2",
        linewidth=1.5,
        zorder=4,
    )
    axis.annotate(
        "variance-matched RBF",
        (
            matched["top_1pct_interval_width_95_K"],
            matched["top_1pct_coverage_95"],
        ),
        xytext=(10, -2),
        textcoords="offset points",
        fontsize=9,
        color="#0072B2",
    )
    axis.axhline(0.95, color="#555555", linestyle=":", linewidth=1.1)
    axis.set_xlabel("Top-1% 95% interval width (K)")
    axis.set_ylabel("Top-1% empirical coverage")
    axis.set_title("Hottest-tail coverage must be read together with sharpness")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction_comparison(
    args: argparse.Namespace,
    fixed: pd.Series,
    *,
    trajectory_name: str,
    output_path: Path,
    linear_output_path: Path | None = None,
) -> None:
    records = trajectory_catalog(args.dataset_dir)
    indexed = {record.name: (index, record) for index, record in enumerate(records)}
    if trajectory_name not in indexed:
        raise ValueError(f"Unknown reconstruction trajectory: {trajectory_name}")
    trajectory_index, record = indexed[trajectory_name]
    prepared, _ = prepare_trajectory(
        args.dataset_dir,
        trajectory_name,
        nx=args.nx,
        ny=args.ny,
        heat_flux_cutoff=args.heat_flux_cutoff,
    )
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
        seed=args.seed + 10_000 * trajectory_index,
    )
    lengthscale = prepared.source_lengthscale * args.length_multiplier
    previous_mean, previous_draws, _ = infer_previous_censored_posterior(
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
    prediction_seed = args.seed + 100_000 * trajectory_index
    predictions, _ = sequential_predictions(
        prepared,
        camera=camera,
        previous_mean=previous_mean,
        previous_draws=previous_draws,
        ordinary_mean=ordinary_mean,
        advective_mean=advective_mean,
        displacement=displacement,
        fixed=fixed,
        current_noise_sd=args.current_noise_sd,
        length_multiplier=args.length_multiplier,
        forcing_length_multiplier=args.forcing_length_multiplier,
        quadrature_order=args.quadrature_order,
        samples=args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=prediction_seed,
    )

    current = camera["current"]
    current_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed = observation_indices(prepared, observation_points)
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    centered = previous_draws - previous_mean[None, :, :]
    propagated = propagate_residual_draws(
        centered,
        dx=dx,
        dy=dy,
        time_step=dt,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        displacement=displacement,
    )
    component_means = ordinary_mean.ravel()[None, :] + propagated.reshape(
        len(previous_draws), -1
    )
    stochastic_config = StochasticHeatConfig(
        signal_sd=float(fixed.signal_sd),
        forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        quadrature_order=args.quadrature_order,
    )
    innovation_oo = finite_step_innovation_covariance(
        observation_points, observation_points, stochastic_config, dt
    )
    innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
    innovation_po = finite_step_innovation_covariance(
        current_points, observation_points, stochastic_config, dt
    )
    innovation_variance = float(
        finite_step_innovation_covariance(
            current_points[:1], current_points[:1], stochastic_config, dt
        )[0, 0]
    )
    mixture_prediction, _ = sample_censored_gaussian_mixture_blocks(
        current,
        component_prediction_means=component_means,
        component_observation_means=component_means[:, observed],
        observed_covariance=innovation_oo,
        pred_observed_covariance=innovation_po,
        prediction_variance=np.full(len(current_points), innovation_variance),
        noise_sd=args.current_noise_sd,
        n_samples=args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=prediction_seed,
    )
    predictions[POSTERIOR_SAMPLE_MIXTURE] = mixture_prediction

    legacy_args = SimpleNamespace(
        noise_sd=args.measurement_noise_sd,
        noise_multiplier=2.0,
        signal_multiplier=1.0,
        beta_multiplier=1.0,
        length_multiplier=args.length_multiplier,
        forcing_length_multiplier=args.forcing_length_multiplier,
        forcing_quadrature_order=args.quadrature_order,
        source_flux_threshold=args.source_flux_threshold,
        source_coupling=float(fixed.source_coupling),
    )
    legacy_prepared, _ = legacy.thermal.prepare_trajectory(
        args.dataset_dir,
        trajectory_name,
        nx=args.nx,
        ny=args.ny,
        time_stride=1,
        heat_flux_cutoff=args.heat_flux_cutoff,
    )
    history_indices = legacy.study.select_history_indices(
        legacy_prepared.times, [0.0, 0.01]
    )
    multitime = legacy.corrected.fixed_mask_multitime_observations(
        legacy_prepared,
        threshold=threshold,
        history_indices=history_indices,
        observation_stride=args.observation_stride,
        noise_sd=args.measurement_noise_sd,
        seed=args.seed + 10_000 * trajectory_index,
    )
    legacy_current = legacy.study.current_frame_observations(multitime)
    _, legacy_mean_function = legacy.ablation.physics_mean_for_trajectory(
        legacy_prepared,
        threshold=threshold,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        source_coupling=float(fixed.source_coupling),
        source_flux_threshold=args.source_flux_threshold,
    )
    legacy_models = legacy.make_model_configurations(
        legacy_args,
        legacy_prepared,
        current=legacy_current,
        multitime=multitime,
        physics_mean_function=legacy_mean_function,
        parameters={
            "diffusivity": float(fixed.diffusivity),
            "cooling_rate": float(fixed.cooling_rate),
            "signal_sd": float(fixed.signal_sd),
        },
    )
    legacy_observations, legacy_config, _ = legacy_models[
        "physics mean + advective stochastic space-time"
    ]
    _, legacy_prediction = legacy.ablation.run_method(
        legacy_prepared,
        method="physics mean + advective stochastic space-time",
        observations=legacy_observations,
        config=legacy_config,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        source_coupling=float(fixed.source_coupling),
        n_frames=len(history_indices),
        chains=1,
        samples=args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=prediction_seed,
        return_prediction=True,
    )
    predictions[LEGACY_JOINT_ADV_ST] = legacy_prediction

    panels = [
        ("True current field", prepared.truth),
        ("Censored camera observation", np.asarray(current["clipped_full"])),
        ("Posterior physics mean + RBF", predictions[POSTERIOR_RBF][0]),
        (
            "Advective posterior physics mean + RBF",
            predictions[ADVECTIVE_POSTERIOR_RBF][0],
        ),
        ("Legacy clipped mean + joint advective ST", predictions[LEGACY_JOINT_ADV_ST][0]),
        ("Posterior physics mean + sequential ST", predictions[SEQUENTIAL_ST][0]),
        (
            "Posterior physics mean + sequential advective ST",
            predictions[SEQUENTIAL_ADV_ST][0],
        ),
        (
            "Advective posterior mean + sequential advective ST",
            predictions[ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST][0],
        ),
        ("Posterior-sample mixture + sequential advective ST", mixture_prediction[0]),
    ]
    extent = [prepared.xs[0], prepared.xs[-1], prepared.ys[0], prepared.ys[-1]]
    vmin = float(np.min(prepared.truth))
    vmax = float(np.max(prepared.truth))
    def save_panel_figure(
        path: Path,
        *,
        cmap: str,
        norm: matplotlib.colors.Normalize,
        scale_note: str,
    ) -> None:
        figure, axes = plt.subplots(
            3, 3, figsize=(12.6, 9.2), constrained_layout=True
        )
        image = None
        for axis, (title, field) in zip(axes.ravel(), panels):
            values = np.asarray(field, dtype=float).reshape(prepared.truth.shape)
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                aspect="auto",
            )
            if isinstance(norm, matplotlib.colors.PowerNorm):
                add_tail_contours(
                    axis,
                    prepared.xs,
                    prepared.ys,
                    values,
                    ambient=prepared.ambient,
                    ceiling=threshold,
                )
            axis.set_title(title, fontsize=9.5)
            axis.set_xticks([])
            axis.set_yticks([])
        if image is not None:
            colorbar = figure.colorbar(
                image,
                ax=axes.ravel().tolist(),
                fraction=0.022,
                extend=(
                    "both"
                    if isinstance(norm, matplotlib.colors.PowerNorm)
                    else "neither"
                ),
            )
            colorbar.set_label("Temperature (K)")
        figure.suptitle(
            f"Representative reconstruction comparison: {trajectory_name}\n"
            f"3% synthetic censoring; shared {norm.vmin:.1f}-{norm.vmax:.1f} K scale; "
            f"{scale_note}",
            fontsize=13,
        )
        figure.savefig(path, dpi=220)
        plt.close(figure)

    save_panel_figure(
        output_path,
        cmap="magma",
        norm=tail_temperature_norm(prepared.ambient, threshold),
        scale_note=(
            "tail-focused nonlinear colors; white contours at fixed excess "
            "temperatures"
        ),
    )
    if linear_output_path is not None:
        save_panel_figure(
            linear_output_path,
            cmap="inferno",
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            scale_note="linear colors",
        )


def markdown_table(table: pd.DataFrame) -> list[str]:
    lines = [
        "| # | Architecture | Field | RMSE | Overall CRPS | Top-1% CRPS | Peak error | Above RMSE | Above bias | Above cov. | Above width | Top-1% RMSE | Top-1% bias | Top-1% cov. | Top-1% width |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in table.iterrows():
        lines.append(
            f"| {index + 1} | {row['method']} | {row['field_excess_rel_l2']:.3f} | "
            f"{row['rmse_K']:.3f} | {row['overall_crps_K']:.3f} | "
            f"{row['top_1pct_crps_K']:.3f} | {row['peak_absolute_error_K']:.3f} | "
            f"{row['above_ceiling_rmse_K']:.3f} | {row['above_ceiling_signed_error_K']:+.3f} | "
            f"{row['above_ceiling_coverage_95']:.3f} | {row['above_ceiling_interval_width_95_K']:.3f} | "
            f"{row['top_1pct_rmse_K']:.3f} | {row['top_1pct_signed_error_K']:+.3f} | "
            f"{row['top_1pct_coverage_95']:.3f} | {row['top_1pct_interval_width_95_K']:.3f} |"
        )
    return lines


def write_architecture_markdown(
    output_path: Path,
    table: pd.DataFrame,
    scale_control: pd.DataFrame,
    propagation: pd.DataFrame,
) -> None:
    best_field = table.loc[table["field_excess_rel_l2"].idxmin()]
    best_overall = table.loc[table["overall_crps_K"].idxmin()]
    best_tail = table.loc[table["top_1pct_crps_K"].idxmin()]
    best_tail_rmse = table.loc[table["top_1pct_rmse_K"].idxmin()]
    original = scale_control[scale_control["method"] == POSTERIOR_RBF].iloc[0]
    matched = scale_control[scale_control["method"] == MATCHED_RBF].iloc[0]
    sample = propagation[propagation["method"] == POSTERIOR_SAMPLE_MIXTURE].iloc[0]
    mean_only = propagation[propagation["method"] == STRICT_MEAN_ONLY].iloc[0]
    moment = propagation[propagation["method"] == SEQUENTIAL_ADV_ST].iloc[0]
    lines = [
        "# Final architecture-aware comparison",
        "",
        "The eight requested labels reduce to seven distinct architectures. Requested item 5, `posterior physics mean + sequential advective ST`, is exactly the moment-matched `C + B Sigma B^T` implementation also named in item 7, so it appears once.",
        "",
        "## Main complete-model comparison",
        "",
        *markdown_table(table),
        "",
        "Coverage is shown beside interval width deliberately. The complete-model RBF rows retain their frozen, much larger covariance amplitude and are not covariance-geometry controls.",
        "",
        "## Direct answers",
        "",
        f"1. **Best whole-field reconstruction:** {best_field['method']} has the smallest excess-field error ({best_field['field_excess_rel_l2']:.3f}).",
        f"2. **Best overall CRPS:** {best_overall['method']} has overall CRPS {best_overall['overall_crps_K']:.3f} K.",
        f"3. **Best top-1% CRPS:** {best_tail['method']} has top-1% CRPS {best_tail['top_1pct_crps_K']:.3f} K under its frozen complete-model variance.",
        f"4. **Peak reconstruction is not substantially better for that method.** The smallest top-1% RMSE is {best_tail_rmse['top_1pct_rmse_K']:.3f} K for {best_tail_rmse['method']}; peak errors across the leading models remain close, so the unmatched RBF tail advantage is mainly probabilistic spread rather than a dramatically better peak mean.",
        f"5. **Covariance amplitude explains most of the original RBF tail advantage.** Original RBF top-1% width is {original['top_1pct_interval_width_95_K']:.3f} K versus {matched['top_1pct_interval_width_95_K']:.3f} K after variance matching; coverage changes from {original['top_1pct_coverage_95']:.3f} to {matched['top_1pct_coverage_95']:.3f}.",
        f"6. **Variance matching removes and reverses the apparent RBF geometry advantage.** Top-1% CRPS changes from {original['top_1pct_crps_K']:.3f} to {matched['top_1pct_crps_K']:.3f} K after matching, compared with {scale_control.loc[scale_control['method'] == POSTERIOR_SAMPLE_MIXTURE, 'top_1pct_crps_K'].iloc[0]:.3f} K for the posterior-sample mixture.",
        f"7. **Previous-state uncertainty matters modestly.** Mean-only to moment matching changes top-1% CRPS from {mean_only['top_1pct_crps_K']:.3f} to {moment['top_1pct_crps_K']:.3f} K and top-1% coverage from {mean_only['top_1pct_coverage_95']:.3f} to {moment['top_1pct_coverage_95']:.3f}. Retaining the censored-region sample mixture changes these only to {sample['top_1pct_crps_K']:.3f} K and {sample['top_1pct_coverage_95']:.3f}.",
        "8. **The remaining sequential weakness is temperature/state-dependent underdispersion and negative hottest-tail bias.** It is not primarily missing previous-state uncertainty, because moment matching and posterior-sample propagation are nearly identical. The variance-matched RBF also performs similarly to the sequential geometry, so inferior ST covariance geometry is not supported as the main explanation.",
        "",
        "## Scientific interpretation",
        "",
        "The sequential stochastic space-time architectures are the strongest for whole-field reconstruction and overall probabilistic sharpness. The originally configured snapshot RBF is a valid complete model and remains useful as a robust hot-tail reference, but its high hottest-tail coverage is purchased with intervals roughly four times wider. Once marginal amplitude is controlled, the RBF tail advantage disappears. There is therefore no universal winner: the current sequential model is sharp and strong globally, while its remaining error is concentrated at the hottest pixels, where it is negatively biased and underdispersed.",
    ]
    output_path.write_text("\n".join(lines), encoding="ascii")


def write_readme(
    output_path: Path,
    table: pd.DataFrame,
    scale_control: pd.DataFrame,
    checks: pd.DataFrame,
    rbf_sd: float,
    innovation_sd: float,
) -> None:
    max_check = float(checks["max_absolute_difference"].max())
    lines = [
        "# Final thermal-camera architecture comparison",
        "",
        "This directory consolidates the existing frozen models after the covariance-scale audit. No model or hyperparameter was invented or tuned here.",
        "",
        "## Protocol",
        "",
        "- 33 total trajectories: 3 development and 30 equally weighted held-out trajectories.",
        "- Synthetic 3% censoring, stride-5 camera mask, 0.25 K measurement noise, and 0.5 K likelihood noise.",
        "- Identical trajectory-index observation and sampler seed convention.",
        "- Unbiased empirical CRPS using `M(M-1)`.",
        "- Existing model-specific covariance amplitudes are preserved in the main table.",
        "",
        "## Architecture audit",
        "",
        "The requested posterior-physics sequential advective-ST row and the requested moment-matched sequential advective-ST row are the same implementation: `C + B Sigma B^T`. They are represented once.",
        "",
        f"The original RBF marginal SD is `{rbf_sd:.3f} K`; the one-step stochastic innovation SD is `{innovation_sd:.3f} K`. This difference is why the scale-control table is required before interpreting RBF-vs-ST coverage as a geometry result.",
        "",
        "## Files",
        "",
        "- `architecture_comparison.csv` and `.md`: seven distinct complete architectures and the direct scientific answers.",
        "- `rbf_st_scale_control.csv`: original RBF, variance-matched RBF, moment-matched sequential advective ST, and posterior-sample mixture.",
        "- `propagation_ablation.csv`: strict mean-only, moment-matched, and censored-region posterior-sample propagation.",
        "- `whole_field_vs_tail.png`: overall versus top-1% CRPS.",
        "- `tail_coverage_vs_width.png`: top-1% coverage versus interval width, including the matched-RBF diagnostic.",
        "- `reconstruction_comparison.png`: tail-enhanced shared-scale reconstruction panel for all seven architectures on `SpiralScanPath_13`.",
        "- `reconstruction_comparison_linear.png`: the same fields with the original shared linear color scale.",
        "- `reproduction_checks.csv`: rerun agreement with the frozen historical outputs.",
        "",
        "## Reuse and reruns",
        "",
        "The five current-only/sequential Gaussian rows and the legacy joint row were rerun only because their historical outputs did not save all requested regional posterior diagnostics. The posterior-sample mixture and variance-matched RBF diagnostics were reused. The largest absolute rerun difference in a saved metric is "
        f"`{max_check:.3e}`.",
        "",
        f"The main table contains `{len(table)}` distinct architectures and the scale control contains `{len(scale_control)}` rows.",
    ]
    output_path.write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final architecture-aware thermal-camera comparison."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument(
        "--reconstruction-trajectory", default="SpiralScanPath_13"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(MIXTURE_OUTPUT / "fixed_configuration.csv").iloc[0]
    sequential_results, sequential_regions, covariance = run_sequential_architectures(
        args, fixed
    )
    legacy_results, legacy_regions = run_legacy_joint_architecture(args, fixed)
    mixture_results, mixture_regions = reused_mixture_rows()

    all_results = pd.concat(
        [sequential_results, legacy_results, mixture_results],
        ignore_index=True,
        sort=False,
    )
    all_regions = pd.concat(
        [sequential_regions, legacy_regions, mixture_regions],
        ignore_index=True,
        sort=False,
    )
    evaluation_covariance = covariance[covariance["role"] == "evaluation"]
    rbf_sd = float(evaluation_covariance["rbf_marginal_sd_K"].mean())
    innovation_sd = float(evaluation_covariance["st_innovation_sd_K"].mean())
    metadata = architecture_metadata(rbf_sd, innovation_sd)
    table = aggregate_architectures(all_results, all_regions, metadata)
    checks = reproduction_checks(sequential_results, legacy_results)
    if float(checks["max_absolute_difference"].max()) > 1e-10:
        raise AssertionError("A rerun did not reproduce the frozen source experiment")
    scale_control = scale_control_table(innovation_sd, rbf_sd)
    propagation = propagation_ablation_table()

    table.to_csv(args.output_dir / "architecture_comparison.csv", index=False)
    scale_control.to_csv(args.output_dir / "rbf_st_scale_control.csv", index=False)
    propagation.to_csv(args.output_dir / "propagation_ablation.csv", index=False)
    checks.to_csv(args.output_dir / "reproduction_checks.csv", index=False)
    covariance.to_csv(args.output_dir / "covariance_scales_by_trajectory.csv", index=False)
    plot_whole_field_vs_tail(
        table, args.output_dir / "whole_field_vs_tail.png"
    )
    plot_tail_coverage_vs_width(
        table,
        scale_control,
        args.output_dir / "tail_coverage_vs_width.png",
    )
    plot_reconstruction_comparison(
        args,
        fixed,
        trajectory_name=args.reconstruction_trajectory,
        output_path=args.output_dir / "reconstruction_comparison.png",
        linear_output_path=args.output_dir / "reconstruction_comparison_linear.png",
    )
    write_architecture_markdown(
        args.output_dir / "architecture_comparison.md",
        table,
        scale_control,
        propagation,
    )
    write_readme(
        args.output_dir / "README.md",
        table,
        scale_control,
        checks,
        rbf_sd,
        innovation_sd,
    )
    print(f"Saved final architecture comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
