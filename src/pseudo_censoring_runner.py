from __future__ import annotations

import numpy as np

from src.ceiling_sweep import (
    SequentialAdvectiveConfig,
    matched_censoring_observations,
    sequential_prior_blocks,
)
from src.dense_censored_gp import sample_censored_gaussian_blocks
from src.mcmc_diagnostics import chain_diagnostics
from src.pseudo_censoring_calibration import calibration_features
from src.thermal_posterior_physics import (
    infer_previous_censored_posterior,
    posterior_physics_means,
)


def prediction_from_draws(draws: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(draws, dtype=float)
    return (
        np.mean(values, axis=0),
        np.std(values, axis=0, ddof=1),
        np.quantile(values, 0.025, axis=0),
        np.quantile(values, 0.975, axis=0),
        values,
    )


def _balanced_draw_subset(
    chain_predictions: list[tuple[np.ndarray, ...]], per_chain: int
) -> np.ndarray:
    selected = []
    for prediction in chain_predictions:
        draws = np.asarray(prediction[4], dtype=float)
        if per_chain > len(draws):
            raise ValueError("Requested more draws than a chain retained")
        indices = np.linspace(0, len(draws) - 1, per_chain, dtype=int)
        selected.append(draws[indices])
    return np.vstack(selected)


def _current_prediction(
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
    last: dict[str, float | bool | int] | None = None
    for samples, burn_in, thin, retried in settings:
        predictions = [
            sample_censored_gaussian_blocks(
                observations,
                **blocks,
                noise_sd=float(options["current_noise_sd"]),
                n_samples=samples,
                burn_in=burn_in,
                thin=thin,
                seed=seed + 100_000 * chain,
                truncated_sampler=str(options["truncated_sampler"]),
            )
            for chain in range(int(options["current_chains"]))
        ]
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
        last = diagnostics
        if (
            diagnostics["rhat_max"] <= float(options["maximum_rhat"])
            and diagnostics["selected_rhat_max"]
            <= float(options["maximum_rhat"])
        ):
            draws = _balanced_draw_subset(
                predictions, int(options["evaluation_draws_per_chain"])
            )
            return prediction_from_draws(draws), diagnostics
    raise RuntimeError(
        "Current posterior failed convergence after retry: "
        f"all={last['rhat_max']:.4f}, censored={last['selected_rhat_max']:.4f}"
    )


def _previous_prediction(
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
    last: dict[str, float | bool | int] | None = None
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
        last = diagnostics
        if diagnostics["rhat_max"] <= float(options["maximum_rhat"]):
            return mean, draws, diagnostics
    raise RuntimeError(
        "Previous posterior failed convergence after retry: "
        f"R-hat={last['rhat_max']:.4f}"
    )


def run_base_prediction(
    prepared,
    *,
    pseudo_camera: dict[str, object],
    ceiling: float,
    fixed: dict[str, float],
    options: dict[str, object],
    trajectory_index: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, object]]:
    """Run the frozen Experiment 38 sequential posterior for one ceiling."""
    previous_index = int(pseudo_camera["frames"][0]["time_index"])
    current_index = int(pseudo_camera["frames"][1]["time_index"])
    lengthscale = prepared.source_lengthscale * float(options["length_multiplier"])
    previous_mean, previous_draws, previous_diagnostics = _previous_prediction(
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
    prediction, current_diagnostics = _current_prediction(
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
