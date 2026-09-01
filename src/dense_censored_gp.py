from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import _mvn, norm

from src.censored_gp import cholesky_with_jitter


@dataclass(frozen=True)
class DenseGaussianPrior:
    points: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    noise_sd: float
    relative_jitter: float = 1e-7


def _point_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    digits = 13
    lookup = {
        tuple(np.round(point, digits)): index
        for index, point in enumerate(reference)
    }
    try:
        return np.asarray(
            [lookup[tuple(np.round(point, digits))] for point in query],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError("Every observation point must belong to the prior grid") from error


def validate_prior(prior: DenseGaussianPrior) -> None:
    points = np.asarray(prior.points, dtype=float)
    mean = np.asarray(prior.mean, dtype=float).reshape(-1)
    covariance = np.asarray(prior.covariance, dtype=float)
    if points.ndim != 2:
        raise ValueError("Prior points must be a matrix")
    if mean.shape != (len(points),):
        raise ValueError("Prior mean must contain one value per point")
    if covariance.shape != (len(points), len(points)):
        raise ValueError("Prior covariance has the wrong shape")
    if prior.noise_sd <= 0.0:
        raise ValueError("Observation noise must be positive")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Prior contains non-finite values")
    tolerance = 1e-9 * max(float(np.max(np.diag(covariance))), 1.0)
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise ValueError("Prior covariance is not symmetric")


def sample_censored_dense_prior(
    observations: dict[str, object],
    prior: DenseGaussianPrior,
    *,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Censored GP sampling for a precomputed finite-grid Gaussian prior."""
    validate_prior(prior)
    points = np.asarray(prior.points, dtype=float)
    prior_mean = np.asarray(prior.mean, dtype=float).reshape(-1)
    prior_covariance = np.asarray(prior.covariance, dtype=float)
    x_obs = np.asarray(observations["x_obs"], dtype=float)
    y_obs = np.asarray(observations["y_obs"], dtype=float)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool)
    threshold = float(observations["threshold"])
    observed_indices = _point_indices(points, x_obs)
    unsat_mask = ~sat_mask
    rng = np.random.default_rng(seed)
    noise_variance = prior.noise_sd**2
    marginal_scale = max(float(np.max(np.diag(prior_covariance))), 1.0)
    jitter = prior.relative_jitter * marginal_scale

    observed_covariance = prior_covariance[np.ix_(observed_indices, observed_indices)]
    observed_covariance = 0.5 * (observed_covariance + observed_covariance.T)
    observed_covariance[np.diag_indices_from(observed_covariance)] += (
        noise_variance + jitter
    )
    observed_mean = prior_mean[observed_indices]

    if np.any(sat_mask):
        sat_indices = np.flatnonzero(sat_mask)
        unsat_indices = np.flatnonzero(unsat_mask)
        sat_covariance = observed_covariance[np.ix_(sat_indices, sat_indices)]
        sat_mean = observed_mean[sat_indices]
        if len(unsat_indices):
            unsat_covariance = observed_covariance[
                np.ix_(unsat_indices, unsat_indices)
            ]
            sat_unsat = observed_covariance[np.ix_(sat_indices, unsat_indices)]
            unsat_factor = cho_factor(
                unsat_covariance, lower=True, check_finite=False
            )
            residual = y_obs[unsat_mask] - observed_mean[unsat_indices]
            sat_mean = sat_mean + sat_unsat.dot(
                cho_solve(unsat_factor, residual, check_finite=False)
            )
            sat_covariance = sat_covariance - sat_unsat.dot(
                cho_solve(unsat_factor, sat_unsat.T, check_finite=False)
            )
        sat_factor = cholesky_with_jitter(sat_covariance, jitter)
        current = np.maximum(sat_mean, threshold + 0.5 * prior.noise_sd)
        centered = current - sat_mean
        saturated_samples = []
        for step in range(burn_in + n_samples * thin):
            direction = sat_factor.dot(rng.normal(size=len(sat_indices)))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            lower_angle = angle - 2.0 * np.pi
            upper_angle = angle
            for _ in range(2500):
                proposal_centered = (
                    centered * np.cos(angle) + direction * np.sin(angle)
                )
                proposal = sat_mean + proposal_centered
                if np.all(proposal >= threshold):
                    centered = proposal_centered
                    current = proposal
                    break
                if angle < 0.0:
                    lower_angle = angle
                else:
                    upper_angle = angle
                angle = rng.uniform(lower_angle, upper_angle)
            else:
                raise RuntimeError("No feasible censored elliptical-slice proposal")
            if step >= burn_in and (step - burn_in) % thin == 0:
                saturated_samples.append(current.copy())
        observation_samples = np.empty((n_samples, len(x_obs)))
        observation_samples[:, unsat_mask] = y_obs[unsat_mask]
        observation_samples[:, sat_mask] = np.asarray(saturated_samples)
    else:
        observation_samples = np.repeat(y_obs[None, :], n_samples, axis=0)

    observation_factor = cho_factor(
        observed_covariance, lower=True, check_finite=False
    )
    pred_obs = prior_covariance[:, observed_indices]
    centered_observations = observation_samples - observed_mean[None, :]
    weights = cho_solve(
        observation_factor,
        centered_observations.T,
        check_finite=False,
    )
    conditional_means = prior_mean[None, :] + pred_obs.dot(weights).T
    solved = cho_solve(observation_factor, pred_obs.T, check_finite=False)
    conditional_covariance = prior_covariance - pred_obs.dot(solved)
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    conditional_factor = cholesky_with_jitter(conditional_covariance, jitter)
    draws = conditional_means + rng.normal(
        size=(n_samples, len(points))
    ).dot(conditional_factor.T)
    mean = np.mean(conditional_means, axis=0)
    variance = np.diag(conditional_covariance) + np.var(
        conditional_means, axis=0, ddof=1
    )
    return (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        draws,
    )


def sample_censored_gaussian_blocks(
    observations: dict[str, object],
    *,
    prediction_mean: np.ndarray,
    observation_mean: np.ndarray,
    observed_covariance: np.ndarray,
    pred_observed_covariance: np.ndarray,
    prediction_variance: np.ndarray,
    noise_sd: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
    relative_jitter: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a censored Gaussian prior from observation/prediction blocks."""
    y_obs = np.asarray(observations["y_obs"], dtype=float).reshape(-1)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool).reshape(-1)
    threshold = float(observations["threshold"])
    pred_mean = np.asarray(prediction_mean, dtype=float).reshape(-1)
    obs_mean = np.asarray(observation_mean, dtype=float).reshape(-1)
    k_oo = np.asarray(observed_covariance, dtype=float)
    k_po = np.asarray(pred_observed_covariance, dtype=float)
    pred_variance = np.asarray(prediction_variance, dtype=float).reshape(-1)
    if obs_mean.shape != y_obs.shape or k_oo.shape != (len(y_obs), len(y_obs)):
        raise ValueError("Observation mean or covariance has the wrong shape")
    if k_po.shape != (len(pred_mean), len(y_obs)):
        raise ValueError("Prediction-observation covariance has the wrong shape")
    if pred_variance.shape != pred_mean.shape:
        raise ValueError("Prediction variance has the wrong shape")
    if noise_sd <= 0.0:
        raise ValueError("Observation noise must be positive")

    rng = np.random.default_rng(seed)
    noise_variance = noise_sd**2
    scale = max(float(np.max(pred_variance)), 1.0)
    jitter = relative_jitter * scale
    noisy_oo = 0.5 * (k_oo + k_oo.T)
    noisy_oo[np.diag_indices_from(noisy_oo)] += noise_variance + jitter
    unsat_mask = ~sat_mask

    if np.any(sat_mask):
        sat_indices = np.flatnonzero(sat_mask)
        unsat_indices = np.flatnonzero(unsat_mask)
        sat_covariance = noisy_oo[np.ix_(sat_indices, sat_indices)]
        sat_mean = obs_mean[sat_indices]
        if len(unsat_indices):
            unsat_covariance = noisy_oo[np.ix_(unsat_indices, unsat_indices)]
            sat_unsat = noisy_oo[np.ix_(sat_indices, unsat_indices)]
            unsat_factor = cho_factor(
                unsat_covariance, lower=True, check_finite=False
            )
            residual = y_obs[unsat_mask] - obs_mean[unsat_indices]
            sat_mean = sat_mean + sat_unsat.dot(
                cho_solve(unsat_factor, residual, check_finite=False)
            )
            sat_covariance = sat_covariance - sat_unsat.dot(
                cho_solve(unsat_factor, sat_unsat.T, check_finite=False)
            )
        sat_factor = cholesky_with_jitter(sat_covariance, jitter)
        current = np.maximum(sat_mean, threshold + 0.5 * noise_sd)
        centered = current - sat_mean
        saturated_samples = []
        for step in range(burn_in + n_samples * thin):
            direction = sat_factor.dot(rng.normal(size=len(sat_indices)))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            lower_angle = angle - 2.0 * np.pi
            upper_angle = angle
            for _ in range(2500):
                proposal_centered = (
                    centered * np.cos(angle) + direction * np.sin(angle)
                )
                proposal = sat_mean + proposal_centered
                if np.all(proposal >= threshold):
                    centered = proposal_centered
                    current = proposal
                    break
                if angle < 0.0:
                    lower_angle = angle
                else:
                    upper_angle = angle
                angle = rng.uniform(lower_angle, upper_angle)
            else:
                raise RuntimeError("No feasible censored elliptical-slice proposal")
            if step >= burn_in and (step - burn_in) % thin == 0:
                saturated_samples.append(current.copy())
        observation_samples = np.empty((n_samples, len(y_obs)))
        observation_samples[:, unsat_mask] = y_obs[unsat_mask]
        observation_samples[:, sat_mask] = np.asarray(saturated_samples)
    else:
        observation_samples = np.repeat(y_obs[None, :], n_samples, axis=0)

    observation_factor = cho_factor(noisy_oo, lower=True, check_finite=False)
    centered_observations = observation_samples - obs_mean[None, :]
    weights = cho_solve(
        observation_factor,
        centered_observations.T,
        check_finite=False,
    )
    conditional_means = pred_mean[None, :] + k_po.dot(weights).T
    solved = cho_solve(observation_factor, k_po.T, check_finite=False)
    conditional_variance = np.maximum(
        pred_variance - np.sum(k_po * solved.T, axis=1),
        0.0,
    )
    mean = np.mean(conditional_means, axis=0)
    variance = conditional_variance + np.var(
        conditional_means, axis=0, ddof=1
    )
    draws = conditional_means + rng.normal(size=conditional_means.shape) * np.sqrt(
        conditional_variance
    )
    return (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        draws,
    )


def _mixture_weight_state(
    observations: dict[str, object],
    *,
    component_observation_means: np.ndarray,
    observed_covariance: np.ndarray,
    noise_sd: float,
    relative_jitter: float,
) -> dict[str, object]:
    """Posterior component weights for a shared-covariance censored mixture."""
    y_obs = np.asarray(observations["y_obs"], dtype=float).reshape(-1)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool).reshape(-1)
    threshold = float(observations["threshold"])
    component_means = np.asarray(component_observation_means, dtype=float)
    k_oo = np.asarray(observed_covariance, dtype=float)
    if component_means.ndim != 2 or component_means.shape[1] != len(y_obs):
        raise ValueError(
            "Component observation means must have shape (component, observation)"
        )
    if k_oo.shape != (len(y_obs), len(y_obs)):
        raise ValueError("Observed covariance has the wrong shape")
    if noise_sd <= 0.0:
        raise ValueError("Observation noise must be positive")

    scale = max(float(np.max(np.diag(k_oo))), 1.0)
    jitter = relative_jitter * scale
    noisy_oo = 0.5 * (k_oo + k_oo.T)
    noisy_oo[np.diag_indices_from(noisy_oo)] += noise_sd**2 + jitter
    unsat_mask = ~sat_mask
    n_components = len(component_means)
    log_weights = np.zeros(n_components)
    conditional_sat_means = np.empty((n_components, int(np.sum(sat_mask))))
    conditional_sat_covariance = np.empty(
        (int(np.sum(sat_mask)), int(np.sum(sat_mask)))
    )

    if np.any(unsat_mask):
        unsat_covariance = noisy_oo[np.ix_(unsat_mask, unsat_mask)]
        unsat_factor = cho_factor(
            unsat_covariance, lower=True, check_finite=False
        )
        residuals = y_obs[unsat_mask][None, :] - component_means[:, unsat_mask]
        solved_residuals = cho_solve(
            unsat_factor, residuals.T, check_finite=False
        ).T
        log_determinant = 2.0 * np.sum(
            np.log(np.diag(unsat_factor[0]))
        )
        log_weights -= 0.5 * (
            np.sum(residuals * solved_residuals, axis=1)
            + log_determinant
            + np.sum(unsat_mask) * np.log(2.0 * np.pi)
        )
    else:
        unsat_factor = None
        solved_residuals = np.empty((n_components, 0))

    integration_failures = 0
    if np.any(sat_mask):
        sat_covariance = noisy_oo[np.ix_(sat_mask, sat_mask)]
        conditional_sat_means = component_means[:, sat_mask].copy()
        conditional_sat_covariance = sat_covariance.copy()
        if np.any(unsat_mask):
            sat_unsat = noisy_oo[np.ix_(sat_mask, unsat_mask)]
            conditional_sat_means += solved_residuals.dot(sat_unsat.T)
            conditional_sat_covariance -= sat_unsat.dot(
                cho_solve(unsat_factor, sat_unsat.T, check_finite=False)
            )
        conditional_sat_covariance = 0.5 * (
            conditional_sat_covariance + conditional_sat_covariance.T
        )
        n_saturated = int(np.sum(sat_mask))
        if n_saturated == 1:
            scale_sd = np.sqrt(max(float(conditional_sat_covariance[0, 0]), 0.0))
            standardized = (
                threshold - conditional_sat_means[:, 0]
            ) / max(scale_sd, np.finfo(float).tiny)
            log_weights += norm.logsf(standardized)
        else:
            lower = np.full(n_saturated, threshold)
            upper = np.full(n_saturated, np.inf)
            tiny = np.finfo(float).tiny
            for index, mean in enumerate(conditional_sat_means):
                probability, information = _mvn.mvnun(
                    lower,
                    upper,
                    mean,
                    conditional_sat_covariance,
                    100_000 * n_saturated,
                    1e-9,
                )
                integration_failures += int(information != 0)
                log_weights[index] += np.log(max(float(probability), tiny))

    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("Every propagated mixture component has zero likelihood")
    centered_log_weights = log_weights - np.max(log_weights[finite])
    unnormalized = np.exp(centered_log_weights)
    unnormalized[~finite] = 0.0
    weights = unnormalized / np.sum(unnormalized)
    return {
        "weights": weights,
        "log_weights": log_weights,
        "noisy_observed_covariance": noisy_oo,
        "conditional_sat_means": conditional_sat_means,
        "conditional_sat_covariance": conditional_sat_covariance,
        "integration_failures": integration_failures,
        "jitter": jitter,
    }


def censored_gaussian_mixture_component_weights(
    observations: dict[str, object],
    *,
    component_observation_means: np.ndarray,
    observed_covariance: np.ndarray,
    noise_sd: float,
    relative_jitter: float = 1e-7,
) -> np.ndarray:
    """Return likelihood-updated weights for equally weighted prior components."""
    state = _mixture_weight_state(
        observations,
        component_observation_means=component_observation_means,
        observed_covariance=observed_covariance,
        noise_sd=noise_sd,
        relative_jitter=relative_jitter,
    )
    return np.asarray(state["weights"], dtype=float)


def _sample_lower_truncated_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    lower: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    jitter: float,
) -> np.ndarray:
    """Elliptical-slice draws from N(mean, covariance) truncated below."""
    location = np.asarray(mean, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if n_samples <= 0:
        return np.empty((0, len(location)))
    factor = cholesky_with_jitter(matrix, jitter)
    marginal_sd = np.sqrt(max(float(np.max(np.diag(matrix))), jitter))
    current = np.maximum(location, lower + 0.5 * marginal_sd)
    centered = current - location
    draws = []
    for step in range(burn_in + n_samples * thin):
        direction = factor.dot(rng.normal(size=len(location)))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        lower_angle = angle - 2.0 * np.pi
        upper_angle = angle
        for _ in range(2500):
            proposal_centered = (
                centered * np.cos(angle) + direction * np.sin(angle)
            )
            proposal = location + proposal_centered
            if np.all(proposal >= lower):
                centered = proposal_centered
                break
            if angle < 0.0:
                lower_angle = angle
            else:
                upper_angle = angle
            angle = rng.uniform(lower_angle, upper_angle)
        else:
            raise RuntimeError("No feasible mixture elliptical-slice proposal")
        if step >= burn_in and (step - burn_in) % thin == 0:
            draws.append((location + centered).copy())
    return np.asarray(draws)


def sample_censored_gaussian_mixture_blocks(
    observations: dict[str, object],
    *,
    component_prediction_means: np.ndarray,
    component_observation_means: np.ndarray,
    observed_covariance: np.ndarray,
    pred_observed_covariance: np.ndarray,
    prediction_variance: np.ndarray,
    noise_sd: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
    relative_jitter: float = 1e-7,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Condition a finite equally weighted Gaussian mixture on censored data.

    Every component has its own prediction/observation mean and shares the same
    covariance blocks. Components are reweighted by the mixed Gaussian-density
    and Gaussian-tail likelihood before posterior draws are generated.
    """
    y_obs = np.asarray(observations["y_obs"], dtype=float).reshape(-1)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool).reshape(-1)
    threshold = float(observations["threshold"])
    prediction_means = np.asarray(component_prediction_means, dtype=float)
    observation_means = np.asarray(component_observation_means, dtype=float)
    k_oo = np.asarray(observed_covariance, dtype=float)
    k_po = np.asarray(pred_observed_covariance, dtype=float)
    pred_variance = np.asarray(prediction_variance, dtype=float).reshape(-1)
    if prediction_means.ndim != 2:
        raise ValueError("Component prediction means must be a matrix")
    if observation_means.shape != (len(prediction_means), len(y_obs)):
        raise ValueError("Component observation means have the wrong shape")
    if k_po.shape != (prediction_means.shape[1], len(y_obs)):
        raise ValueError("Prediction-observation covariance has the wrong shape")
    if pred_variance.shape != (prediction_means.shape[1],):
        raise ValueError("Prediction variance has the wrong shape")
    if n_samples < 2:
        raise ValueError("Mixture inference requires at least two output samples")

    state = _mixture_weight_state(
        observations,
        component_observation_means=observation_means,
        observed_covariance=k_oo,
        noise_sd=noise_sd,
        relative_jitter=relative_jitter,
    )
    weights = np.asarray(state["weights"], dtype=float)
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(n_samples)) / n_samples
    component_indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    rng.shuffle(component_indices)

    observation_samples = np.empty((n_samples, len(y_obs)))
    observation_samples[:, ~sat_mask] = y_obs[~sat_mask]
    if np.any(sat_mask):
        conditional_means = np.asarray(state["conditional_sat_means"])
        conditional_covariance = np.asarray(
            state["conditional_sat_covariance"]
        )
        for component in np.unique(component_indices):
            selected = np.flatnonzero(component_indices == component)
            observation_samples[selected[:, None], np.flatnonzero(sat_mask)] = (
                _sample_lower_truncated_gaussian(
                    conditional_means[component],
                    conditional_covariance,
                    lower=threshold,
                    n_samples=len(selected),
                    burn_in=burn_in,
                    thin=thin,
                    rng=rng,
                    jitter=float(state["jitter"]),
                )
            )

    noisy_oo = np.asarray(state["noisy_observed_covariance"])
    observation_factor = cho_factor(noisy_oo, lower=True, check_finite=False)
    selected_observation_means = observation_means[component_indices]
    centered_observations = observation_samples - selected_observation_means
    solved_weights = cho_solve(
        observation_factor, centered_observations.T, check_finite=False
    )
    conditional_means = prediction_means[component_indices] + k_po.dot(
        solved_weights
    ).T
    solved = cho_solve(observation_factor, k_po.T, check_finite=False)
    conditional_variance = np.maximum(
        pred_variance - np.sum(k_po * solved.T, axis=1), 0.0
    )
    draws = conditional_means + rng.normal(size=conditional_means.shape) * np.sqrt(
        conditional_variance
    )
    mean = np.mean(conditional_means, axis=0)
    variance = conditional_variance + np.var(conditional_means, axis=0, ddof=1)
    positive = weights > 0.0
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
    finite_log_weights = np.asarray(state["log_weights"])[
        np.isfinite(np.asarray(state["log_weights"]))
    ]
    diagnostics = {
        "mixture_component_count": float(len(weights)),
        "mixture_component_weight_ess": float(1.0 / np.sum(weights**2)),
        "mixture_max_component_weight": float(np.max(weights)),
        "mixture_normalized_weight_entropy": float(
            entropy / np.log(len(weights)) if len(weights) > 1 else 1.0
        ),
        "mixture_resampled_unique_components": float(
            len(np.unique(component_indices))
        ),
        "mixture_log_weight_range": float(
            np.max(finite_log_weights) - np.min(finite_log_weights)
        ),
        "mixture_mvn_integration_failures": float(state["integration_failures"]),
    }
    prediction = (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        draws,
    )
    return prediction, diagnostics
