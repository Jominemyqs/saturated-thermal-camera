from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class RBFConfig:
    mean_temp: float
    signal_sd: float
    lengthscale: float
    noise_sd: float
    mean_function: Callable[[np.ndarray], np.ndarray] | None = None
    relative_jitter: float = 1e-7


def gp_mean(points: np.ndarray, config: RBFConfig) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if config.mean_function is None:
        return np.full(len(points), config.mean_temp)
    values = np.asarray(config.mean_function(points), dtype=float)
    if values.ndim == 0:
        values = np.full(len(points), float(values))
    values = values.reshape(-1)
    if values.shape != (len(points),):
        raise ValueError("Mean function must return one value per point")
    if not np.all(np.isfinite(values)):
        raise ValueError("Mean function returned non-finite values")
    return values


def rbf_covariance(
    points1: np.ndarray,
    points2: np.ndarray,
    config: RBFConfig,
) -> np.ndarray:
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    delta = p1[:, None, :2] - p2[None, :, :2]
    squared_distance = np.sum(delta**2, axis=2) / config.lengthscale**2
    return config.signal_sd**2 * np.exp(-0.5 * squared_distance)


def kernel_diagonal(points: np.ndarray, config: RBFConfig) -> np.ndarray:
    return np.full(len(np.asarray(points)), config.signal_sd**2)


def cholesky_with_jitter(matrix: np.ndarray, base_jitter: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    identity = np.eye(len(symmetric))
    for multiplier in (0.0, 1.0, 10.0, 100.0, 1000.0):
        try:
            return np.linalg.cholesky(
                symmetric + multiplier * base_jitter * identity
            )
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(symmetric + 10000.0 * base_jitter * identity)


def predict_exact(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    config: RBFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = rbf_covariance(x_train, x_train, config)
    jitter = config.relative_jitter * config.signal_sd**2
    covariance[np.diag_indices_from(covariance)] += config.noise_sd**2 + jitter
    cross_covariance = rbf_covariance(x_pred, x_train, config)
    pred_covariance = rbf_covariance(x_pred, x_pred, config)
    factor = cho_factor(covariance, lower=True, check_finite=False)
    alpha = cho_solve(
        factor,
        np.asarray(y_train) - gp_mean(x_train, config),
        check_finite=False,
    )
    mean = gp_mean(x_pred, config) + cross_covariance.dot(alpha)
    solved = cho_solve(factor, cross_covariance.T, check_finite=False)
    variance = np.maximum(
        np.diag(pred_covariance - cross_covariance.dot(solved)),
        0.0,
    )
    return mean, np.sqrt(variance)


def sample_censored_ess_fast(
    observations: dict[str, object],
    config: RBFConfig,
    *,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample censored observations, then condition predictions analytically."""
    x_obs = np.asarray(observations["x_obs"], dtype=float)
    y_obs = np.asarray(observations["y_obs"], dtype=float)
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool)
    threshold = float(observations["threshold"])
    x_pred = np.asarray(observations["x_pred"], dtype=float)
    rng = np.random.default_rng(seed)
    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]
    noise_variance = config.noise_sd**2
    jitter = config.relative_jitter * config.signal_sd**2

    if len(x_sat) == 0:
        mean, sd = predict_exact(x_obs, y_obs, x_pred, config)
        draws = mean + rng.normal(size=(n_samples, len(x_pred))) * sd
        return (
            mean,
            sd,
            np.quantile(draws, 0.025, axis=0),
            np.quantile(draws, 0.975, axis=0),
            draws,
        )

    sat_covariance = rbf_covariance(x_sat, x_sat, config)
    sat_covariance[np.diag_indices_from(sat_covariance)] += (
        noise_variance + jitter
    )
    if len(x_unsat):
        unsat_covariance = rbf_covariance(x_unsat, x_unsat, config)
        unsat_covariance[np.diag_indices_from(unsat_covariance)] += (
            noise_variance + jitter
        )
        sat_unsat_covariance = rbf_covariance(x_sat, x_unsat, config)
        unsat_factor = cho_factor(
            unsat_covariance, lower=True, check_finite=False
        )
        alpha = cho_solve(
            unsat_factor,
            y_unsat - gp_mean(x_unsat, config),
            check_finite=False,
        )
        sat_mean = gp_mean(x_sat, config) + sat_unsat_covariance.dot(alpha)
        sat_covariance -= sat_unsat_covariance.dot(
            cho_solve(
                unsat_factor,
                sat_unsat_covariance.T,
                check_finite=False,
            )
        )
    else:
        sat_mean = gp_mean(x_sat, config)

    sat_factor = cholesky_with_jitter(sat_covariance, jitter)
    current = np.maximum(sat_mean, threshold + 0.5 * config.noise_sd)
    centered = current - sat_mean
    saturated_samples: list[np.ndarray] = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        direction = sat_factor.dot(rng.normal(size=len(x_sat)))
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
            raise RuntimeError("Elliptical slice sampler found no feasible proposal")
        if step >= burn_in and (step - burn_in) % thin == 0:
            saturated_samples.append(current.copy())

    observation_samples = np.empty((n_samples, len(x_obs)))
    observation_samples[:, unsat_mask] = y_unsat
    observation_samples[:, sat_mask] = np.asarray(saturated_samples)
    observation_covariance = rbf_covariance(x_obs, x_obs, config)
    observation_covariance[np.diag_indices_from(observation_covariance)] += (
        noise_variance + jitter
    )
    pred_obs_covariance = rbf_covariance(x_pred, x_obs, config)
    observation_factor = cho_factor(
        observation_covariance, lower=True, check_finite=False
    )
    alpha_samples = cho_solve(
        observation_factor,
        (observation_samples - gp_mean(x_obs, config)[None, :]).T,
        check_finite=False,
    )
    conditional_means = gp_mean(x_pred, config)[None, :] + (
        pred_obs_covariance.dot(alpha_samples)
    ).T
    solved = cho_solve(
        observation_factor,
        pred_obs_covariance.T,
        check_finite=False,
    )
    conditional_variance = np.maximum(
        kernel_diagonal(x_pred, config)
        - np.sum(pred_obs_covariance * solved.T, axis=1),
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


def sample_multiple_chains(
    observations: dict[str, object],
    config: RBFConfig,
    *,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = []
    variances = []
    draws = []
    for chain in range(n_chains):
        result = sample_censored_ess_fast(
            observations,
            config,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 10_000 * chain,
        )
        means.append(result[0])
        variances.append(result[1] ** 2)
        draws.append(result[4])
    mean_array = np.asarray(means)
    variance_array = np.asarray(variances)
    mean = np.mean(mean_array, axis=0)
    variance = np.mean(variance_array + mean_array**2, axis=0) - mean**2
    pooled = np.vstack(draws)
    return (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(pooled, 0.025, axis=0),
        np.quantile(pooled, 0.975, axis=0),
        pooled,
    )
