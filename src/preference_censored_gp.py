from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr

from src.censored_gp import (
    RBFConfig,
    cholesky_with_jitter,
    gp_mean,
    kernel_diagonal,
    rbf_covariance,
)


def preference_log_likelihood(
    latent_saturated: np.ndarray,
    *,
    threshold: float,
    noise_sd: float,
    preference_pairs: np.ndarray,
    tau: float,
) -> float:
    """Censoring plus local probit-ordering log likelihood on latent values."""
    values = np.asarray(latent_saturated, dtype=float).reshape(-1)
    pairs = np.asarray(preference_pairs, dtype=int)
    if noise_sd <= 0.0 or tau <= 0.0:
        raise ValueError("Noise and preference softness must be positive")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Preference pairs must have shape (n_pairs, 2)")
    if pairs.size and (np.min(pairs) < 0 or np.max(pairs) >= len(values)):
        raise ValueError("Preference-pair index is outside the saturated vector")
    log_likelihood = float(
        np.sum(log_ndtr((values - threshold) / noise_sd))
    )
    if len(pairs):
        differences = values[pairs[:, 1]] - values[pairs[:, 0]]
        log_likelihood += float(np.sum(log_ndtr(differences / tau)))
    return log_likelihood


def _elliptical_slice_samples(
    mean: np.ndarray,
    covariance: np.ndarray,
    log_likelihood,
    *,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
    relative_jitter: float,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    center = np.asarray(mean, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    scale = max(float(np.max(np.diag(matrix))), 1.0)
    factor = cholesky_with_jitter(matrix, relative_jitter * scale)
    current = center.copy()
    centered = current - center
    current_log_likelihood = float(log_likelihood(current))
    samples: list[np.ndarray] = []
    bracket_evaluations = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        direction = factor.dot(rng.normal(size=len(center)))
        log_slice = current_log_likelihood + np.log(rng.uniform())
        angle = rng.uniform(0.0, 2.0 * np.pi)
        lower_angle = angle - 2.0 * np.pi
        upper_angle = angle
        for evaluation in range(1, 2501):
            proposal_centered = centered * np.cos(angle) + direction * np.sin(angle)
            proposal = center + proposal_centered
            proposal_log_likelihood = float(log_likelihood(proposal))
            if proposal_log_likelihood >= log_slice:
                current = proposal
                centered = proposal_centered
                current_log_likelihood = proposal_log_likelihood
                bracket_evaluations.append(evaluation)
                break
            if angle < 0.0:
                lower_angle = angle
            else:
                upper_angle = angle
            angle = rng.uniform(lower_angle, upper_angle)
        else:
            raise RuntimeError("Elliptical slice sampler exhausted its angle bracket")
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(current.copy())
    return np.asarray(samples), {
        "mean_bracket_evaluations": float(np.mean(bracket_evaluations)),
        "maximum_bracket_evaluations": int(np.max(bracket_evaluations)),
        "final_log_likelihood": current_log_likelihood,
    }


def sample_censored_gp_with_preferences(
    observations: dict[str, object],
    config: RBFConfig,
    *,
    preference_pairs: np.ndarray,
    tau: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Sample a latent RBF GP with censoring and optional local preferences.

    Preference indices refer to the saturated observations in their existing
    order. Unsaturated Gaussian observations are conditioned analytically;
    censoring and ordering are then applied to the latent saturated values.
    """
    x_obs = np.asarray(observations["x_obs"], dtype=float)
    y_obs = np.asarray(observations["y_obs"], dtype=float).reshape(-1)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool).reshape(-1)
    x_pred = np.asarray(observations["x_pred"], dtype=float)
    threshold = float(observations["threshold"])
    if len(x_obs) != len(y_obs) or len(sat_mask) != len(y_obs):
        raise ValueError("Observation arrays must have matching lengths")
    if not np.any(sat_mask):
        raise ValueError("Preference experiment requires saturated observations")

    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]
    mean_unsat = gp_mean(x_unsat, config)
    mean_sat = gp_mean(x_sat, config)
    mean_pred = gp_mean(x_pred, config)
    covariance_sat = rbf_covariance(x_sat, x_sat, config)
    covariance_pred_sat = rbf_covariance(x_pred, x_sat, config)
    pred_variance = kernel_diagonal(x_pred, config)

    if len(x_unsat):
        covariance_unsat = rbf_covariance(x_unsat, x_unsat, config)
        scale = max(config.signal_sd**2, 1.0)
        covariance_unsat[np.diag_indices_from(covariance_unsat)] += (
            config.noise_sd**2 + config.relative_jitter * scale
        )
        factor_unsat = cho_factor(
            covariance_unsat, lower=True, check_finite=False
        )
        covariance_sat_unsat = rbf_covariance(x_sat, x_unsat, config)
        covariance_pred_unsat = rbf_covariance(x_pred, x_unsat, config)
        residual = y_unsat - mean_unsat
        solved_residual = cho_solve(
            factor_unsat, residual, check_finite=False
        )
        mean_sat = mean_sat + covariance_sat_unsat.dot(solved_residual)
        mean_pred = mean_pred + covariance_pred_unsat.dot(solved_residual)
        solved_sat = cho_solve(
            factor_unsat, covariance_sat_unsat.T, check_finite=False
        )
        covariance_sat = covariance_sat - covariance_sat_unsat.dot(solved_sat)
        covariance_pred_sat = covariance_pred_sat - covariance_pred_unsat.dot(
            solved_sat
        )
        solved_pred = cho_solve(
            factor_unsat, covariance_pred_unsat.T, check_finite=False
        )
        pred_variance = pred_variance - np.sum(
            covariance_pred_unsat * solved_pred.T, axis=1
        )

    covariance_sat = 0.5 * (covariance_sat + covariance_sat.T)
    pairs = np.asarray(preference_pairs, dtype=int).reshape(-1, 2)

    def log_likelihood(values: np.ndarray) -> float:
        return preference_log_likelihood(
            values,
            threshold=threshold,
            noise_sd=config.noise_sd,
            preference_pairs=pairs,
            tau=tau,
        )

    latent_samples, diagnostics = _elliptical_slice_samples(
        mean_sat,
        covariance_sat,
        log_likelihood,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        relative_jitter=config.relative_jitter,
    )

    scale = max(float(np.max(np.diag(covariance_sat))), 1.0)
    factor_sat = cho_factor(
        covariance_sat + config.relative_jitter * scale * np.eye(len(x_sat)),
        lower=True,
        check_finite=False,
    )
    weights = cho_solve(
        factor_sat,
        (latent_samples - mean_sat[None, :]).T,
        check_finite=False,
    )
    conditional_means = mean_pred[None, :] + covariance_pred_sat.dot(weights).T
    solved = cho_solve(
        factor_sat, covariance_pred_sat.T, check_finite=False
    )
    conditional_variance = np.maximum(
        pred_variance - np.sum(covariance_pred_sat * solved.T, axis=1),
        0.0,
    )
    rng = np.random.default_rng(seed + 1_000_003)
    draws = conditional_means + rng.normal(size=conditional_means.shape) * np.sqrt(
        conditional_variance
    )
    mean = np.mean(conditional_means, axis=0)
    variance = conditional_variance + np.var(
        conditional_means, axis=0, ddof=1
    )
    if len(pairs):
        differences = latent_samples[:, pairs[:, 1]] - latent_samples[:, pairs[:, 0]]
        diagnostics["posterior_preference_satisfaction"] = float(
            np.mean(differences > 0.0)
        )
    else:
        diagnostics["posterior_preference_satisfaction"] = np.nan
    diagnostics.update(
        {
            "n_saturated": int(np.sum(sat_mask)),
            "n_unsaturated": int(np.sum(unsat_mask)),
            "n_preference_pairs": len(pairs),
            "tau_K": tau,
        }
    )
    prediction = (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        draws,
    )
    return prediction, diagnostics


def sample_preference_gp_multiple_chains(
    observations: dict[str, object],
    config: RBFConfig,
    *,
    preference_pairs: np.ndarray,
    tau: float,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict[str, float],
]:
    """Pool independent preference-posterior chains.

    The censored orthant can mix slowly when many neighboring pixels saturate.
    Pooling independently initialized chains makes that behavior visible and
    avoids treating one short trajectory through the orthant as the posterior.
    """
    if n_chains < 1:
        raise ValueError("At least one chain is required")
    if samples_per_chain < 2:
        raise ValueError("Each chain must retain at least two samples")

    predictions = []
    diagnostics = []
    for chain in range(n_chains):
        prediction, chain_diagnostics = sample_censored_gp_with_preferences(
            observations,
            config,
            preference_pairs=preference_pairs,
            tau=tau,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 100_000 * chain,
        )
        predictions.append(prediction)
        diagnostics.append(chain_diagnostics)

    chain_means = np.asarray([prediction[0] for prediction in predictions])
    chain_variances = np.asarray([prediction[1] ** 2 for prediction in predictions])
    mean = np.mean(chain_means, axis=0)
    variance = np.mean(chain_variances + chain_means**2, axis=0) - mean**2
    pooled_draws = np.vstack([prediction[4] for prediction in predictions])
    x_pred = np.asarray(observations["x_pred"], dtype=float)
    x_obs = np.asarray(observations["x_obs"], dtype=float)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool)
    prediction_lookup = {
        tuple(np.round(point, 13)): index for index, point in enumerate(x_pred)
    }
    saturated_prediction_indices = np.asarray(
        [
            prediction_lookup[tuple(np.round(point, 13))]
            for point in x_obs[sat_mask]
        ],
        dtype=int,
    )
    pooled_prediction = (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(pooled_draws, 0.025, axis=0),
        np.quantile(pooled_draws, 0.975, axis=0),
        pooled_draws,
    )
    pooled_diagnostics = {
        "n_chains": n_chains,
        "samples_per_chain": samples_per_chain,
        "pooled_samples": n_chains * samples_per_chain,
        "burn_in_per_chain": burn_in,
        "thin": thin,
        "mean_bracket_evaluations": float(
            np.mean([item["mean_bracket_evaluations"] for item in diagnostics])
        ),
        "maximum_bracket_evaluations": int(
            np.max([item["maximum_bracket_evaluations"] for item in diagnostics])
        ),
        "mean_final_log_likelihood": float(
            np.mean([item["final_log_likelihood"] for item in diagnostics])
        ),
        "between_chain_mean_absolute_sd_K": float(
            np.mean(np.std(chain_means, axis=0, ddof=1))
        )
        if n_chains > 1
        else 0.0,
        "between_chain_saturated_mean_absolute_sd_K": float(
            np.mean(
                np.std(
                    chain_means[:, saturated_prediction_indices],
                    axis=0,
                    ddof=1,
                )
            )
        )
        if n_chains > 1
        else 0.0,
        "posterior_preference_satisfaction": float(
            np.nanmean(
                [item["posterior_preference_satisfaction"] for item in diagnostics]
            )
        )
        if len(preference_pairs)
        else np.nan,
        "n_saturated": int(diagnostics[0]["n_saturated"]),
        "n_unsaturated": int(diagnostics[0]["n_unsaturated"]),
        "n_preference_pairs": int(diagnostics[0]["n_preference_pairs"]),
        "tau_K": float(tau),
    }
    return pooled_prediction, pooled_diagnostics
