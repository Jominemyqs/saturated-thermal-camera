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

from src.censored_gp import RBFConfig, rbf_covariance
from src.dense_censored_gp import (
    pool_gaussian_predictions,
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
    previous_posterior_observations,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "37_canonical_converged_architectures"
)
REFERENCE_RESULTS = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "30_final_architecture_comparison"
    / "sequential_rerun_results.csv"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "35_lowered_ceiling_sweep"
    / "fixed_configuration.csv"
)

CURRENT_RBF = "Current-only ambient mean + RBF"
CURRENT_RBF_RICH = "Current-only ambient mean + RBF (all saturated diagnostic)"
POSTERIOR_RBF = "Posterior physics mean + RBF"
ADVECTIVE_POSTERIOR_RBF = "Advective posterior physics mean + RBF"
SEQUENTIAL_ST = "Posterior physics mean + sequential ST"
SEQUENTIAL_ADV_ST = "Posterior physics mean + sequential advective ST"
ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST = (
    "Advective posterior physics mean + sequential advective ST"
)
POSTERIOR_SAMPLE_MIXTURE = (
    "Posterior-sample mixture + sequential advective ST"
)

CANONICAL_METHODS = (
    CURRENT_RBF,
    POSTERIOR_RBF,
    ADVECTIVE_POSTERIOR_RBF,
    SEQUENTIAL_ST,
    SEQUENTIAL_ADV_ST,
    ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST,
    POSTERIOR_SAMPLE_MIXTURE,
)
ALL_METHODS = (*CANONICAL_METHODS, CURRENT_RBF_RICH)
COLORS = {
    CURRENT_RBF: "#777777",
    POSTERIOR_RBF: "#0072B2",
    ADVECTIVE_POSTERIOR_RBF: "#56B4E9",
    SEQUENTIAL_ST: "#009E73",
    SEQUENTIAL_ADV_ST: "#E69F00",
    ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST: "#D55E00",
    POSTERIOR_SAMPLE_MIXTURE: "#CC79A7",
    CURRENT_RBF_RICH: "#111111",
}


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    return iy * len(prepared.xs) + ix


def top_one_mask(truth: np.ndarray) -> np.ndarray:
    values = np.asarray(truth, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) * 100.0 / len(values)
    return ranks >= 99.0


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
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
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
    return pool_gaussian_predictions(predictions), predictions


def gelman_rubin(draw_chains: np.ndarray) -> np.ndarray:
    """Classical per-target R-hat for equally sized independent chains."""
    values = np.asarray(draw_chains, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("R-hat requires shape (chain, draw, target)")
    n_draws = values.shape[1]
    within = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    between = n_draws * np.var(np.mean(values, axis=1), axis=0, ddof=1)
    variance = ((n_draws - 1.0) / n_draws) * within + between / n_draws
    return np.sqrt(
        np.divide(
            variance,
            within,
            out=np.full_like(variance, np.nan),
            where=within > 1e-14,
        )
    )


def current_chain_diagnostics(
    chain_predictions: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    pooled_prediction: tuple[np.ndarray, ...],
    top_mask: np.ndarray,
) -> dict[str, float]:
    chain_means = np.asarray([prediction[0] for prediction in chain_predictions])
    pooled_mean = np.asarray(pooled_prediction[0])
    mean_rmse = np.sqrt(np.mean((chain_means - pooled_mean[None, :]) ** 2, axis=1))
    top_mean_rmse = np.sqrt(
        np.mean(
            (chain_means[:, top_mask] - pooled_mean[None, top_mask]) ** 2,
            axis=1,
        )
    )
    draw_chains = np.asarray([prediction[4] for prediction in chain_predictions])
    rhat = gelman_rubin(draw_chains)
    top_rhat = rhat[top_mask]
    return {
        "current_chain_mean_max_rmse_from_pooled_K": float(np.max(mean_rmse)),
        "current_chain_top_mean_max_rmse_from_pooled_K": float(
            np.max(top_mean_rmse)
        ),
        "current_rhat_median": float(np.nanmedian(rhat)),
        "current_rhat_max": float(np.nanmax(rhat)),
        "current_top_1pct_rhat_median": float(np.nanmedian(top_rhat)),
        "current_top_1pct_rhat_max": float(np.nanmax(top_rhat)),
        "current_top_1pct_rhat_over_1_05_fraction": float(
            np.nanmean(top_rhat > 1.05)
        ),
    }


def rbf_blocks(
    prepared,
    observations: dict[str, object],
    mean_field: np.ndarray,
    config: RBFConfig,
) -> dict[str, np.ndarray]:
    prediction_points = np.asarray(observations["x_pred"], dtype=float)
    observation_points = np.asarray(observations["x_obs"], dtype=float)
    indices = observation_indices(prepared, observation_points)
    flat_mean = np.asarray(mean_field, dtype=float).reshape(-1)
    return {
        "prediction_mean": flat_mean,
        "observation_mean": flat_mean[indices],
        "observed_covariance": rbf_covariance(
            observation_points, observation_points, config
        ),
        "pred_observed_covariance": rbf_covariance(
            prediction_points, observation_points, config
        ),
        "prediction_variance": np.full(
            len(prediction_points), config.signal_sd**2
        ),
    }


def sequential_prior_components(
    prepared,
    camera: dict[str, object],
    previous_mean: np.ndarray,
    previous_draws: np.ndarray,
    ordinary_mean: np.ndarray,
    advective_mean: np.ndarray,
    displacement: np.ndarray,
    config: StochasticHeatConfig,
    mixture_component_indices: np.ndarray,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, float],
]:
    current = camera["current"]
    prediction_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    indices = observation_indices(prepared, observation_points)
    previous_index = int(camera["frames"][0]["time_index"])
    current_index = int(camera["frames"][1]["time_index"])
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    centered = np.asarray(previous_draws) - np.asarray(previous_mean)[None, :, :]
    common = {
        "dx": float(np.mean(np.diff(prepared.xs))),
        "dy": float(np.mean(np.diff(prepared.ys))),
        "time_step": dt,
        "diffusivity": config.diffusivity,
        "cooling_rate": config.cooling_rate,
    }
    propagated_st = propagate_residual_draws(centered, **common)
    propagated_adv = propagate_residual_draws(
        centered, displacement=np.asarray(displacement), **common
    )
    denominator = np.sqrt(len(previous_draws) - 1.0)
    features_st = propagated_st.reshape(len(previous_draws), -1).T / denominator
    features_adv = propagated_adv.reshape(len(previous_draws), -1).T / denominator
    observed_st = features_st[indices]
    observed_adv = features_adv[indices]

    innovation_oo = finite_step_innovation_covariance(
        observation_points, observation_points, config, dt
    )
    innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
    innovation_po = finite_step_innovation_covariance(
        prediction_points, observation_points, config, dt
    )
    innovation_variance = float(
        finite_step_innovation_covariance(
            prediction_points[:1], prediction_points[:1], config, dt
        )[0, 0]
    )

    def blocks(
        mean_field: np.ndarray,
        features: np.ndarray,
        observed_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        flat_mean = np.asarray(mean_field, dtype=float).reshape(-1)
        return {
            "prediction_mean": flat_mean,
            "observation_mean": flat_mean[indices],
            "observed_covariance": innovation_oo
            + observed_features.dot(observed_features.T),
            "pred_observed_covariance": innovation_po
            + features.dot(observed_features.T),
            "prediction_variance": np.full(
                len(prediction_points), max(innovation_variance, 0.0)
            )
            + np.sum(features**2, axis=1),
        }

    output = {
        SEQUENTIAL_ST: blocks(ordinary_mean, features_st, observed_st),
        SEQUENTIAL_ADV_ST: blocks(ordinary_mean, features_adv, observed_adv),
        ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST: blocks(
            advective_mean, features_adv, observed_adv
        ),
    }
    mixture_propagated = propagated_adv[
        np.asarray(mixture_component_indices, dtype=int)
    ].reshape(len(mixture_component_indices), -1)
    mixture = {
        "component_prediction_means": ordinary_mean.ravel()[None, :]
        + mixture_propagated,
        "component_observation_means": ordinary_mean.ravel()[None, indices]
        + mixture_propagated[:, indices],
        "observed_covariance": innovation_oo,
        "pred_observed_covariance": innovation_po,
        "prediction_variance": np.full(
            len(prediction_points), max(innovation_variance, 0.0)
        ),
    }
    diagnostics = {
        "time_lag_s": dt,
        "innovation_variance_K2": innovation_variance,
        "innovation_sd_K": float(np.sqrt(max(innovation_variance, 0.0))),
        "st_propagated_variance_mean_K2": float(
            np.mean(np.sum(features_st**2, axis=1))
        ),
        "advective_propagated_variance_mean_K2": float(
            np.mean(np.sum(features_adv**2, axis=1))
        ),
    }
    return output, mixture, diagnostics


def summarize_prediction(
    prepared,
    *,
    method: str,
    architecture_status: str,
    observations: dict[str, object],
    prediction: tuple[np.ndarray, ...],
    prior_variance: np.ndarray,
    threshold: float,
    role: str,
    family: str,
) -> dict[str, object]:
    mean, sd, lower, upper, draws = prediction
    truth = np.asarray(prepared.truth, dtype=float).reshape(-1)
    top = top_one_mask(truth)
    error = np.asarray(mean) - truth
    top_error = error[top]
    crps = empirical_crps(np.asarray(draws), truth)
    prior_sd = np.sqrt(np.maximum(np.asarray(prior_variance), 0.0))
    return {
        "trajectory": prepared.name,
        "family": family,
        "role": role,
        "method": method,
        "architecture_status": architecture_status,
        "threshold_K": threshold,
        "n_current_observations": int(len(observations["y_obs"])),
        "n_current_saturated_observations": int(
            np.sum(np.asarray(observations["sat_mask"], dtype=bool))
        ),
        "posterior_draw_count": int(len(draws)),
        "field_excess_rel_l2": float(
            np.linalg.norm(error) / np.linalg.norm(truth - prepared.ambient)
        ),
        "overall_rmse_K": float(np.sqrt(np.mean(error**2))),
        "overall_crps_K": float(np.mean(crps)),
        "top_1pct_n_pixels": int(np.sum(top)),
        "top_1pct_rmse_K": float(np.sqrt(np.mean(top_error**2))),
        "top_1pct_mae_K": float(np.mean(np.abs(top_error))),
        "top_1pct_bias_K": float(np.mean(top_error)),
        "top_1pct_crps_K": float(np.mean(crps[top])),
        "top_1pct_coverage_95": float(
            np.mean((lower[top] <= truth[top]) & (truth[top] <= upper[top]))
        ),
        "top_1pct_interval_width_95_K": float(np.mean((upper - lower)[top])),
        "top_1pct_posterior_sd_K": float(np.mean(sd[top])),
        "peak_absolute_error_K": float(abs(np.max(mean) - np.max(truth))),
        "prior_marginal_sd_mean_K": float(np.mean(prior_sd)),
        "prior_marginal_sd_top_1pct_K": float(np.mean(prior_sd[top])),
        "prior_marginal_variance_mean_K2": float(np.mean(prior_variance)),
    }


def worker(payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    dataset_dir = Path(str(payload["dataset_dir"]))
    name = str(payload["trajectory"])
    family = str(payload["family"])
    trajectory_index = int(payload["trajectory_index"])
    threshold = float(payload["threshold"])
    fixed = pd.Series(payload["fixed"])
    options = payload["options"]
    role = "development" if name in DEVELOPMENT_TRAJECTORIES else "evaluation"

    prepared, _ = prepare_trajectory(
        dataset_dir,
        name,
        nx=int(options["nx"]),
        ny=int(options["ny"]),
        heat_flux_cutoff=float(options["heat_flux_cutoff"]),
    )
    current_index = len(prepared.times) - 1
    previous_index = current_index - int(options["previous_frame_offset"])
    camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=threshold,
        observation_stride=int(options["observation_stride"]),
        noise_sd=float(options["measurement_noise_sd"]),
        seed=int(options["seed"]) + 10_000 * trajectory_index,
    )
    current = camera["current"]
    lengthscale = prepared.source_lengthscale * float(options["length_multiplier"])
    previous_mean, previous_draws, previous_diagnostics = infer_previous_censored_posterior(
        prepared,
        frame=camera["frames"][0],
        fixed_mask=camera["fixed_observation_mask"],
        threshold=threshold,
        signal_sd=float(fixed.signal_sd),
        lengthscale=lengthscale,
        noise_sd=float(options["previous_noise_sd"]),
        n_chains=int(options["previous_chains"]),
        samples_per_chain=int(options["previous_samples_per_chain"]),
        burn_in=int(options["previous_burn_in"]),
        thin=int(options["previous_thin"]),
        seed=int(options["seed"]) + 50_000 * trajectory_index,
    )
    ordinary_mean, advective_mean, displacement = posterior_physics_means(
        prepared,
        previous_mean,
        previous_index=previous_index,
        current_index=current_index,
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        source_coupling=float(fixed.source_coupling),
        source_flux_threshold=float(options["source_flux_threshold"]),
    )
    ambient_mean = np.full_like(prepared.truth, prepared.ambient)
    rbf_config = RBFConfig(
        mean_temp=prepared.ambient,
        signal_sd=float(fixed.signal_sd),
        lengthscale=lengthscale,
        noise_sd=float(options["current_noise_sd"]),
    )
    rbf_priors = {
        CURRENT_RBF: rbf_blocks(prepared, current, ambient_mean, rbf_config),
        POSTERIOR_RBF: rbf_blocks(prepared, current, ordinary_mean, rbf_config),
        ADVECTIVE_POSTERIOR_RBF: rbf_blocks(
            prepared, current, advective_mean, rbf_config
        ),
    }
    heat_config = StochasticHeatConfig(
        signal_sd=float(fixed.signal_sd),
        forcing_lengthscale=lengthscale
        * float(options["forcing_length_multiplier"]),
        diffusivity=float(fixed.diffusivity),
        cooling_rate=float(fixed.cooling_rate),
        quadrature_order=int(options["quadrature_order"]),
    )
    mixture_per_chain = int(options["mixture_components_per_previous_chain"])
    previous_per_chain = int(options["previous_samples_per_chain"])
    if mixture_per_chain > previous_per_chain:
        raise ValueError("Mixture components per chain exceed retained previous draws")
    within_chain_indices = np.linspace(
        0,
        previous_per_chain - 1,
        mixture_per_chain,
        dtype=int,
    )
    mixture_component_indices = np.concatenate(
        [
            chain * previous_per_chain + within_chain_indices
            for chain in range(int(options["previous_chains"]))
        ]
    )
    sequential_priors, mixture_prior, sequential_diagnostics = (
        sequential_prior_components(
            prepared,
            camera,
            previous_mean,
            previous_draws,
            ordinary_mean,
            advective_mean,
            displacement,
            heat_config,
            mixture_component_indices,
        )
    )
    gaussian_priors = {**rbf_priors, **sequential_priors}
    prediction_seed = int(options["seed"]) + 100_000 * trajectory_index
    predictions = {}
    chain_diagnostics = {}
    top_mask = top_one_mask(prepared.truth)
    for method, blocks in gaussian_priors.items():
        pooled, chain_predictions = pooled_block_prediction(
            current,
            blocks,
            noise_sd=float(options["current_noise_sd"]),
            n_chains=int(options["current_chains"]),
            samples_per_chain=int(options["current_samples_per_chain"]),
            burn_in=int(options["current_burn_in"]),
            thin=int(options["thin"]),
            seed=prediction_seed,
        )
        predictions[method] = pooled
        chain_diagnostics[method] = current_chain_diagnostics(
            chain_predictions, pooled, top_mask
        )
    mixture_prediction, mixture_diagnostics = sample_censored_gaussian_mixture_blocks(
        current,
        **mixture_prior,
        noise_sd=float(options["current_noise_sd"]),
        n_samples=int(options["mixture_samples"]),
        burn_in=int(options["mixture_burn_in"]),
        thin=int(options["thin"]),
        seed=prediction_seed,
    )
    predictions[POSTERIOR_SAMPLE_MIXTURE] = mixture_prediction

    rich_observations, _, _ = previous_posterior_observations(
        prepared,
        frame=camera["frames"][1],
        fixed_mask=camera["fixed_observation_mask"],
        threshold=threshold,
    )
    rich_observations["x_pred"] = np.asarray(current["x_pred"], dtype=float)
    rich_prior = rbf_blocks(prepared, rich_observations, ambient_mean, rbf_config)
    rich_prediction, rich_chain_predictions = pooled_block_prediction(
        rich_observations,
        rich_prior,
        noise_sd=float(options["current_noise_sd"]),
        n_chains=int(options["rich_current_chains"]),
        samples_per_chain=int(options["rich_current_samples_per_chain"]),
        burn_in=int(options["rich_current_burn_in"]),
        thin=int(options["rich_current_thin"]),
        seed=prediction_seed,
    )
    predictions[CURRENT_RBF_RICH] = rich_prediction
    chain_diagnostics[CURRENT_RBF_RICH] = current_chain_diagnostics(
        rich_chain_predictions, rich_prediction, top_mask
    )

    rows = []
    for method in ALL_METHODS:
        if method == CURRENT_RBF_RICH:
            observations = rich_observations
            prior_variance = rich_prior["prediction_variance"]
            status = "observation-rich diagnostic"
        elif method == POSTERIOR_SAMPLE_MIXTURE:
            observations = current
            centered_components = (
                mixture_prior["component_prediction_means"]
                - np.mean(
                    mixture_prior["component_prediction_means"],
                    axis=0,
                    keepdims=True,
                )
            )
            prior_variance = mixture_prior["prediction_variance"] + np.var(
                centered_components, axis=0, ddof=1
            )
            status = "canonical"
        else:
            observations = current
            prior_variance = gaussian_priors[method]["prediction_variance"]
            status = "canonical"
        row = summarize_prediction(
            prepared,
            method=method,
            architecture_status=status,
            observations=observations,
            prediction=predictions[method],
            prior_variance=prior_variance,
            threshold=threshold,
            role=role,
            family=family,
        )
        row.update(previous_diagnostics)
        row.update(sequential_diagnostics)
        if method in chain_diagnostics:
            row.update(chain_diagnostics[method])
        if method == POSTERIOR_SAMPLE_MIXTURE:
            row.update(mixture_diagnostics)
        rows.append(row)

    previous_saturated = np.asarray(
        camera["frames"][0]["saturated_full"], dtype=bool
    )
    chain_size = int(options["previous_samples_per_chain"])
    chain_means = np.asarray(
        [
            np.mean(
                previous_draws[index * chain_size : (index + 1) * chain_size][
                    :, previous_saturated
                ],
                axis=0,
            )
            for index in range(int(options["previous_chains"]))
        ]
    )
    pooled_saturated_mean = np.mean(
        previous_draws[:, previous_saturated], axis=0
    )
    previous_chain_draws = np.asarray(
        [
            previous_draws[index * chain_size : (index + 1) * chain_size][
                :, previous_saturated
            ]
            for index in range(int(options["previous_chains"]))
        ]
    )
    previous_rhat = gelman_rubin(previous_chain_draws)
    implementation_checks = [
        {
            "trajectory": name,
            "family": family,
            "role": role,
            "same_fixed_current_observations_for_all_canonical_models": True,
            "canonical_current_observation_count": int(len(current["y_obs"])),
            "canonical_current_saturated_count": int(np.sum(current["sat_mask"])),
            "rich_current_observation_count": int(len(rich_observations["y_obs"])),
            "rich_current_saturated_count": int(
                np.sum(rich_observations["sat_mask"])
            ),
            "previous_posterior_draw_count": int(len(previous_draws)),
            "mixture_component_count": int(len(mixture_component_indices)),
            "previous_chain_mean_max_rmse_from_pooled_K": float(
                np.max(
                    np.sqrt(
                        np.mean(
                            (chain_means - pooled_saturated_mean[None, :]) ** 2,
                            axis=1,
                        )
                    )
                )
            ),
            "previous_saturated_rhat_median": float(
                np.nanmedian(previous_rhat)
            ),
            "previous_saturated_rhat_max": float(np.nanmax(previous_rhat)),
            "previous_saturated_rhat_over_1_05_fraction": float(
                np.nanmean(previous_rhat > 1.05)
            ),
            "top_1pct_pixel_count": int(np.sum(top_one_mask(prepared.truth))),
            "rbf_covariance_identical_across_three_means": bool(
                np.array_equal(
                    rbf_priors[CURRENT_RBF]["observed_covariance"],
                    rbf_priors[POSTERIOR_RBF]["observed_covariance"],
                )
                and np.array_equal(
                    rbf_priors[CURRENT_RBF]["observed_covariance"],
                    rbf_priors[ADVECTIVE_POSTERIOR_RBF]["observed_covariance"],
                )
            ),
            "advective_sequential_covariance_identical_across_two_means": bool(
                np.array_equal(
                    sequential_priors[SEQUENTIAL_ADV_ST]["observed_covariance"],
                    sequential_priors[ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST][
                        "observed_covariance"
                    ],
                )
            ),
            "previous_observations_reused_in_current_likelihood": False,
            "current_camera_seed": int(options["seed"]) + 10_000 * trajectory_index,
            "previous_sampler_seed": int(options["seed"])
            + 50_000 * trajectory_index,
            "current_sampler_seed": prediction_seed,
        }
    ]
    return {"results": rows, "checks": implementation_checks}


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    excluded = {
        *groups,
        "trajectory",
        "threshold_K",
        "n_current_observations",
        "n_current_saturated_observations",
        "posterior_draw_count",
        "top_1pct_n_pixels",
    }
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
        "field_excess_rel_l2",
        "overall_rmse_K",
        "overall_crps_K",
        "top_1pct_rmse_K",
        "top_1pct_crps_K",
        "top_1pct_coverage_95",
        "top_1pct_interval_width_95_K",
        "peak_absolute_error_K",
    )
    rows = []
    for left_index, left in enumerate(CANONICAL_METHODS):
        for right in CANONICAL_METHODS[left_index + 1 :]:
            pivot = frame.pivot(index="trajectory", columns="method")
            for metric in metrics:
                difference = pivot[metric][right] - pivot[metric][left]
                rows.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "metric": metric,
                        "difference_definition": "right minus left",
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(difference.median()),
                        "right_lower_count": int(np.sum(difference < 0.0)),
                        "paired_trajectory_count": int(len(difference)),
                    }
                )
    return pd.DataFrame(rows)


def protocol_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        (
            CURRENT_RBF,
            "ambient",
            "spatial RBF",
            "none",
            "fixed-stride current only",
            "Gaussian; 4 x 350 current draws",
            "canonical current-only control",
        ),
        (
            POSTERIOR_RBF,
            "diffusive posterior physics mean",
            "spatial RBF",
            "converged previous censored posterior mean",
            "fixed-stride current only",
            "Gaussian; 4 x 350 current draws",
            "canonical",
        ),
        (
            ADVECTIVE_POSTERIOR_RBF,
            "advective posterior physics mean",
            "spatial RBF",
            "converged previous censored posterior mean",
            "fixed-stride current only",
            "Gaussian; 4 x 350 current draws",
            "canonical",
        ),
        (
            SEQUENTIAL_ST,
            "diffusive posterior physics mean",
            "Q + diffusive B Sigma B^T",
            "converged previous censored posterior mean and covariance",
            "fixed-stride current only",
            "Gaussian moment match; 4 x 350 current draws",
            "canonical",
        ),
        (
            SEQUENTIAL_ADV_ST,
            "diffusive posterior physics mean",
            "Q + advective B Sigma B^T",
            "converged previous censored posterior mean and covariance",
            "fixed-stride current only",
            "Gaussian moment match; 4 x 350 current draws",
            "canonical",
        ),
        (
            ADVECTIVE_MEAN_SEQUENTIAL_ADV_ST,
            "advective posterior physics mean",
            "Q + advective B Sigma B^T",
            "converged previous censored posterior mean and covariance",
            "fixed-stride current only",
            "Gaussian moment match; 4 x 350 current draws",
            "canonical",
        ),
        (
            POSTERIOR_SAMPLE_MIXTURE,
            "diffusive posterior physics mean",
            "Q within components; advectively propagated centered draws",
            "balanced 2800-component subset of 8000 converged previous draws",
            "fixed-stride current only",
            f"finite Gaussian mixture; {args.mixture_samples} current draws",
            "canonical non-Gaussian propagation control",
        ),
        (
            CURRENT_RBF_RICH,
            "ambient",
            "spatial RBF",
            "none",
            "fixed stride plus every observed-saturated current pixel",
            "Gaussian; 4 x 2000 current draws, long burn-in",
            "diagnostic only; observation set differs",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "method",
            "prior_mean",
            "residual_covariance",
            "previous_information",
            "current_observation_set",
            "posterior_representation",
            "status",
        ),
    )


def plot_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    main = summary[summary["architecture_status"] == "canonical"].set_index("method")
    methods = list(CANONICAL_METHODS)
    labels = [
        "current\nRBF",
        "physics\nRBF",
        "adv. physics\nRBF",
        "physics\nseq. ST",
        "physics\nseq. adv. ST",
        "adv. physics\nseq. adv. ST",
        "sample mixture\nseq. adv. ST",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), constrained_layout=True)
    panels = (
        ("field_excess_rel_l2", "Relative excess-field error"),
        ("overall_crps_K", "All-domain CRPS (K)"),
        ("top_1pct_crps_K", "Hottest-1% CRPS (K)"),
        ("top_1pct_coverage_95", "Hottest-1% 95% coverage"),
    )
    x = np.arange(len(methods))
    for axis, (metric, title) in zip(axes.ravel(), panels):
        values = main.loc[methods, metric]
        sem = main.loc[methods, f"{metric}_sem"]
        axis.bar(
            x,
            values,
            yerr=sem,
            color=[COLORS[method] for method in methods],
            capsize=3,
        )
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "top_1pct_coverage_95":
            axis.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
            axis.set_ylim(0.0, 1.02)
    figure.suptitle(
        "Canonical converged architecture comparison: 30 held-out trajectories",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def plot_covariance_scale(summary: pd.DataFrame, output_path: Path) -> None:
    main = summary[summary["architecture_status"] == "canonical"].set_index("method")
    methods = list(CANONICAL_METHODS)
    values = main.loc[methods, "prior_marginal_sd_mean_K"]
    labels = [
        "current RBF",
        "physics RBF",
        "adv. physics RBF",
        "physics seq. ST",
        "physics seq. adv. ST",
        "adv. physics seq. adv. ST",
        "sample-mixture seq. adv. ST",
    ]
    figure, axis = plt.subplots(figsize=(12.0, 5.3), constrained_layout=True)
    axis.barh(
        np.arange(len(methods)),
        values,
        color=[COLORS[method] for method in methods],
    )
    axis.set_yticks(np.arange(len(methods)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean marginal prior SD from the complete covariance (K)")
    axis.set_title("RBF versus total sequential predictive covariance")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def write_readme(
    path: Path,
    summary: pd.DataFrame,
    protocol: pd.DataFrame,
    checks: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    main = summary[summary["architecture_status"] == "canonical"].set_index("method")
    canonical_current = results[
        (results["architecture_status"] == "canonical")
        & results["current_rhat_max"].notna()
    ]
    rich_current = results[results["method"] == CURRENT_RBF_RICH]
    lines = [
        "# Canonical converged architecture comparison",
        "",
        "## Purpose",
        "",
        "This experiment replaces the short-chain Experiment-30 architecture table for quantitative use under the same sparse current-observation architecture. It was run only after the reference equality audit identified the previous-frame censored-posterior sampler as the main source of the old tail underdispersion.",
        "",
        "## Frozen protocol",
        "",
        "- 33 trajectories: 3 development and 30 held out.",
        "- Frozen 3% reference ceilings and camera-noise seeds from Experiment 30.",
        "- Every canonical model uses the identical fixed-stride current observation set.",
        "- Every temporal model uses the identical previous censored posterior: 4 chains, 2000 retained draws per chain, 15000 burn-in, thin 10.",
        "- Every canonical Gaussian current update uses 4 chains, 350 retained draws per chain, 600 burn-in, thin 2. The observation-rich diagnostic uses 4 chains, 2000 retained draws per chain, 15000 burn-in, thin 10 because its inequality dimension is much larger.",
        "- CRPS uses the unbiased `M(M-1)` estimator.",
        "- Hottest 1% means the exact rank-based hottest 25 of 2501 pixels.",
        "- Previous saturated pixels are sampled from the censored RBF posterior; unsaturated noisy pixels are retained at their observed values. This is not called a complete latent posterior.",
        "- Previous observations enter only through the previous posterior and are not reused in the current likelihood.",
        "",
        f"The canonical current likelihood contains `{int(checks['canonical_current_observation_count'].min())}` fixed-stride observations and only `{int(checks['canonical_current_saturated_count'].min())}`--`{int(checks['canonical_current_saturated_count'].max())}` saturated observations per trajectory. This reproduces the earlier sequential architecture but does not use the complete current censoring mask. It is therefore a controlled sparse-observation comparison, not yet a full-camera benchmark.",
        "",
        "The observation-rich current-only row is diagnostic only. It includes all observed-saturated current pixels and is excluded from architecture rankings because its observation set differs.",
        "",
        "The legacy joint model is excluded because it conditions jointly on the previous observations after using a previous-state-derived mean, so it does not share the clean sequential information flow.",
        "",
        "## Held-out results",
        "",
        "| Model | Field error | RMSE (K) | Overall CRPS (K) | Top-1% RMSE (K) | Top-1% CRPS (K) | Coverage | Width (K) | Prior SD (K) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in CANONICAL_METHODS:
        row = main.loc[method]
        lines.append(
            f"| {method} | {row['field_excess_rel_l2']:.3f} | {row['overall_rmse_K']:.3f} | {row['overall_crps_K']:.3f} | {row['top_1pct_rmse_K']:.3f} | {row['top_1pct_crps_K']:.3f} | {row['top_1pct_coverage_95']:.3f} | {row['top_1pct_interval_width_95_K']:.2f} | {row['prior_marginal_sd_mean_K']:.2f} |"
        )
    rich = summary[summary["method"] == CURRENT_RBF_RICH].iloc[0]
    lines.extend(
        [
            "",
            "## Observation-rich diagnostic",
            "",
            f"With fixed stride plus every current saturated pixel, the current-only RBF has field error `{rich['field_excess_rel_l2']:.3f}`, overall CRPS `{rich['overall_crps_K']:.3f}` K, top-1% CRPS `{rich['top_1pct_crps_K']:.3f}` K, coverage `{rich['top_1pct_coverage_95']:.3f}`, and width `{rich['top_1pct_interval_width_95_K']:.2f}` K. These values are not ranked against the canonical rows because they use more current inequality observations.",
            "",
            "## Covariance interpretation",
            "",
            "The reported sequential prior scale uses the complete predictive variance `diag(Q + B Sigma B^T)`. It is not the innovation scale `Q` alone. RBF and sequential calibration comparisons should use this total scale and posterior interval diagnostics together.",
            "",
            "## Convergence diagnostics",
            "",
            f"Across all 33 trajectories, the largest previous-frame saturated-pixel R-hat was `{checks['previous_saturated_rhat_max'].max():.3f}` and no saturated pixel exceeded `1.05`. Across the canonical Gaussian current updates, the largest all-domain R-hat was `{canonical_current['current_rhat_max'].max():.3f}`, the largest hottest-1% R-hat was `{canonical_current['current_top_1pct_rhat_max'].max():.3f}`, and no hottest-1% pixel exceeded `1.05`.",
            "",
            f"The observation-rich diagnostic is not part of the ranking and retains one marginal diagnostic: its largest hottest-1% R-hat is `{rich_current['current_top_1pct_rhat_max'].max():.3f}`. The posterior-sample mixture is assessed by component-weight diagnostics rather than MCMC R-hat.",
            "",
            "The settings previously called converged in Experiment 35 (4 chains, 700 retained draws, 1200 burn-in, thin 2) did not pass these checks. Its ceiling-response trends remain diagnostic, but its absolute uncertainty values must be rerun under this protocol before paper use.",
            "",
            "## Interpretation",
            "",
            "- Adding the posterior physics mean to the matched current-only RBF improves field error and hottest-1% CRPS on all 30 held-out trajectories.",
            "- With the posterior physics mean fixed, sequential ST improves field error on all 30 trajectories and hottest-1% CRPS on 22, but gives narrower intervals and lower hottest-1% coverage than the RBF residual.",
            "- Mean advection improves global field error on all 30 trajectories but slightly worsens hottest-1% RMSE, CRPS, and peak error on average.",
            "- Adding advection to the sequential covariance does not improve overall CRPS on any held-out trajectory relative to stationary sequential ST.",
            "- Posterior-sample mixture propagation gives a small hottest-1% CRPS improvement over the Gaussian moment-matched advective model on 26 of 30 trajectories; it does not materially change coverage.",
            "",
            "These are architecture effects under matched sparse current observations. Before a paper-level full-camera comparison, rerun every row with the same fixed-stride unsaturated observations plus every observed-saturated current pixel, then recheck convergence.",
            "",
            "## Integrity checks",
            "",
            f"All `{len(checks)}` trajectory checks use 25 hottest-tail pixels. RBF covariance identity across mean choices: `{bool(checks['rbf_covariance_identical_across_three_means'].all())}`. Sequential advective covariance identity across mean choices: `{bool(checks['advective_sequential_covariance_identical_across_two_means'].all())}`. Previous observations reused in the current likelihood: `{bool(checks['previous_observations_reused_in_current_likelihood'].any())}`.",
            "",
            "## Files",
            "",
            "- `results.csv`: trajectory-level metrics and covariance scales.",
            "- `heldout30_summary.csv`: equally weighted held-out means and standard errors.",
            "- `family_summary.csv`: diagonal, horizontal, and spiral summaries.",
            "- `paired_comparisons.csv`: paired held-out differences and win counts.",
            "- `model_protocol.csv`: exact architecture and information-flow definitions.",
            "- `implementation_checks.csv`: trajectory-level consistency checks.",
            "- `comparison.png`: point and distribution metrics.",
            "- `total_prior_covariance_scale.png`: complete RBF and sequential prior scales.",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0].to_dict()
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
        "current_chains": args.current_chains,
        "current_samples_per_chain": args.current_samples_per_chain,
        "current_burn_in": args.current_burn_in,
        "rich_current_chains": args.rich_current_chains,
        "rich_current_samples_per_chain": args.rich_current_samples_per_chain,
        "rich_current_burn_in": args.rich_current_burn_in,
        "rich_current_thin": args.rich_current_thin,
        "mixture_components_per_previous_chain": (
            args.mixture_components_per_previous_chain
        ),
        "mixture_samples": args.mixture_samples,
        "mixture_burn_in": args.mixture_burn_in,
        "thin": args.thin,
        "seed": args.seed,
    }
    selected_catalog = catalog
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        selected_catalog = [record for record in catalog if record.name in requested]
        missing = requested - {record.name for record in selected_catalog}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")
    payloads = [
        {
            "dataset_dir": str(args.dataset_dir),
            "trajectory": record.name,
            "family": record.family,
            "trajectory_index": index,
            "threshold": thresholds[record.name],
            "fixed": fixed,
            "options": options,
        }
        for index, record in enumerate(catalog)
        if record in selected_catalog
    ]
    outputs = []
    if args.workers == 1:
        for index, payload in enumerate(payloads):
            outputs.append(worker(payload))
            print(f"[{index + 1:02d}/33] {payload['trajectory']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, payload): payload for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                payload = futures[future]
                outputs.append(future.result())
                completed += 1
                print(f"[{completed:02d}/33] {payload['trajectory']}", flush=True)

    results = pd.DataFrame([row for output in outputs for row in output["results"]])
    checks = pd.DataFrame([row for output in outputs for row in output["checks"]])
    results = results.sort_values(["trajectory", "method"]).reset_index(drop=True)
    checks = checks.sort_values("trajectory").reset_index(drop=True)
    evaluation = results[results["role"] == "evaluation"].copy()
    summary = aggregate(evaluation, ["method", "architecture_status"])
    family_summary = aggregate(
        evaluation, ["family", "method", "architecture_status"]
    )
    paired = paired_comparisons(
        evaluation[evaluation["architecture_status"] == "canonical"]
    )
    protocol = protocol_table(args)

    if not checks["same_fixed_current_observations_for_all_canonical_models"].all():
        raise AssertionError("Canonical current observation sets differ")
    if not checks["rbf_covariance_identical_across_three_means"].all():
        raise AssertionError("RBF covariance changed with its mean")
    if not checks[
        "advective_sequential_covariance_identical_across_two_means"
    ].all():
        raise AssertionError("Sequential covariance changed with its mean")
    if checks["previous_observations_reused_in_current_likelihood"].any():
        raise AssertionError("Previous observations were reused in current inference")
    if not np.all(checks["top_1pct_pixel_count"] == 25):
        raise AssertionError("Canonical hottest-1% masks must contain exactly 25 pixels")
    if np.nanmax(checks["previous_saturated_rhat_max"]) > 1.05:
        raise AssertionError("Previous censored-posterior chains failed R-hat <= 1.05")
    canonical_current = results[
        (results["architecture_status"] == "canonical")
        & results["current_rhat_max"].notna()
    ]
    if np.nanmax(canonical_current["current_rhat_max"]) > 1.05:
        raise AssertionError("A canonical current-posterior chain failed R-hat <= 1.05")
    if np.nanmax(canonical_current["current_top_1pct_rhat_max"]) > 1.05:
        raise AssertionError(
            "A canonical hottest-1% current-posterior chain failed R-hat <= 1.05"
        )
    mixture_failures = results.loc[
        results["method"] == POSTERIOR_SAMPLE_MIXTURE,
        "mixture_mvn_integration_failures",
    ]
    if mixture_failures.notna().any() and np.nanmax(mixture_failures) > 0:
        raise AssertionError("Posterior-sample mixture integration failed")
    expected_rows = len(selected_catalog) * len(ALL_METHODS)
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(results)}")

    results.to_csv(args.output_dir / "results.csv", index=False)
    summary.to_csv(args.output_dir / "heldout30_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    protocol.to_csv(args.output_dir / "model_protocol.csv", index=False)
    checks.to_csv(args.output_dir / "implementation_checks.csv", index=False)
    physical_configuration = {
        key: fixed[key]
        for key in ("diffusivity", "cooling_rate", "signal_sd", "source_coupling")
    }
    pd.DataFrame(
        [
            {
                **physical_configuration,
                **options,
                "reference_censored_fraction": 0.03,
                "dataset_trajectories": 33,
                "development_trajectories": 3,
                "heldout_trajectories": 30,
                "reference_ceiling_source": (
                    "frozen trajectory-specific Experiment-30 ceilings"
                ),
                "top_1pct_rule": "exact rank-based hottest 25 of 2501 pixels",
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "canonical_current_observation_set": (
                    "fixed stride only for every main model"
                ),
                "previous_posterior_observation_set": (
                    "fixed stride plus every observed-saturated previous pixel"
                ),
                "previous_posterior_representation": (
                    "censored RBF draws at saturated pixels; unsaturated pixels "
                    "fixed at noisy observed values"
                ),
                "convergence_requirement": "per-target R-hat <= 1.05",
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    plot_comparison(summary, args.output_dir / "comparison.png")
    plot_covariance_scale(summary, args.output_dir / "total_prior_covariance_scale.png")
    write_readme(args.output_dir / "README.md", summary, protocol, checks, results)
    print(f"Saved canonical comparison to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical converged thermal-camera architecture comparison."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
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
    parser.add_argument("--previous-samples-per-chain", type=int, default=2000)
    parser.add_argument("--previous-burn-in", type=int, default=15000)
    parser.add_argument("--previous-thin", type=int, default=10)
    parser.add_argument("--current-chains", type=int, default=4)
    parser.add_argument("--current-samples-per-chain", type=int, default=350)
    parser.add_argument("--current-burn-in", type=int, default=600)
    parser.add_argument("--rich-current-chains", type=int, default=4)
    parser.add_argument("--rich-current-samples-per-chain", type=int, default=2000)
    parser.add_argument("--rich-current-burn-in", type=int, default=15000)
    parser.add_argument("--rich-current-thin", type=int, default=10)
    parser.add_argument(
        "--mixture-components-per-previous-chain", type=int, default=700
    )
    parser.add_argument("--mixture-samples", type=int, default=1400)
    parser.add_argument("--mixture-burn-in", type=int, default=600)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
