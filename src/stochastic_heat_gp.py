from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, shift as shift_image


@dataclass(frozen=True)
class StochasticHeatConfig:
    signal_sd: float
    forcing_lengthscale: float
    diffusivity: float
    cooling_rate: float
    quadrature_order: int = 24
    advection_path: Callable[[np.ndarray], np.ndarray] | None = None


def source_path_interpolator(times: np.ndarray, positions: np.ndarray):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)

    def path(query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times, dtype=float).reshape(-1)
        return np.column_stack(
            [
                np.interp(query, times, positions[:, 0]),
                np.interp(query, times, positions[:, 1]),
            ]
        )

    return path


def stochastic_heat_covariance(
    points1: np.ndarray,
    points2: np.ndarray,
    config: StochasticHeatConfig,
) -> np.ndarray:
    """Stationary covariance of a spatially forced heat process."""
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    if p1.ndim != 2 or p2.ndim != 2 or p1.shape[1] < 3 or p2.shape[1] < 3:
        raise ValueError("Space-time points must contain x, y, and time")
    if config.diffusivity < 0.0:
        raise ValueError("Diffusivity must be nonnegative")
    if config.cooling_rate <= 0.0:
        raise ValueError("A stationary heat process requires positive cooling")
    if config.forcing_lengthscale <= 0.0:
        raise ValueError("Forcing lengthscale must be positive")
    if config.quadrature_order <= 0:
        raise ValueError("Quadrature order must be positive")

    spatial_delta = p1[:, None, :2] - p2[None, :, :2]
    if config.advection_path is not None:
        path1 = np.asarray(config.advection_path(p1[:, 2]), dtype=float)
        path2 = np.asarray(config.advection_path(p2[:, 2]), dtype=float)
        if path1.shape != (len(p1), 2) or path2.shape != (len(p2), 2):
            raise ValueError("Advection path must return one 2D point per time")
        spatial_delta -= path1[:, None, :] - path2[None, :, :]

    distance_squared = np.sum(spatial_delta**2, axis=2)
    time_difference = np.abs(p1[:, None, 2] - p2[None, :, 2])
    nodes, weights = np.polynomial.laguerre.laggauss(config.quadrature_order)
    ell_squared = config.forcing_lengthscale**2
    covariance = np.zeros_like(distance_squared)
    for node, weight in zip(nodes, weights):
        propagated_time = time_difference + node / config.cooling_rate
        scale_squared = ell_squared + 2.0 * config.diffusivity * propagated_time
        covariance += (
            weight
            * ell_squared
            / scale_squared
            * np.exp(-0.5 * distance_squared / scale_squared)
        )
    covariance *= np.exp(-config.cooling_rate * time_difference)

    zero_lag_scales = (
        ell_squared
        + 2.0 * config.diffusivity * nodes / config.cooling_rate
    )
    zero_lag_integral = np.sum(weights * ell_squared / zero_lag_scales)
    return config.signal_sd**2 * covariance / zero_lag_integral


def sequential_moment_matched_covariance(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    previous_draws: np.ndarray,
    config: StochasticHeatConfig,
    *,
    relative_jitter: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C + B Sigma B^T, B, and C for one sequential prediction."""
    previous = np.asarray(previous_points, dtype=float)
    current = np.asarray(current_points, dtype=float)
    draws = np.asarray(previous_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(previous):
        raise ValueError("Previous draws must have shape (n_draws, n_points)")
    if len(draws) < 2:
        raise ValueError("At least two previous posterior draws are required")

    k_minus_minus = stochastic_heat_covariance(previous, previous, config)
    k_plus_minus = stochastic_heat_covariance(current, previous, config)
    k_plus_plus = stochastic_heat_covariance(current, current, config)
    jitter = relative_jitter * config.signal_sd**2
    stable_minus = 0.5 * (k_minus_minus + k_minus_minus.T)
    stable_minus[np.diag_indices_from(stable_minus)] += jitter
    factor = np.linalg.cholesky(stable_minus)
    solved = np.linalg.solve(factor.T, np.linalg.solve(factor, k_plus_minus.T))
    transition = solved.T
    innovation = k_plus_plus - transition.dot(k_plus_minus.T)
    innovation = 0.5 * (innovation + innovation.T)

    centered = draws - np.mean(draws, axis=0, keepdims=True)
    propagated_features = transition.dot(centered.T / np.sqrt(len(draws) - 1.0))
    covariance = innovation + propagated_features.dot(propagated_features.T)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, transition, innovation


def finite_step_innovation_covariance(
    points1: np.ndarray,
    points2: np.ndarray,
    config: StochasticHeatConfig,
    time_step: float,
) -> np.ndarray:
    """Covariance introduced by forcing over one finite heat-process step."""
    if time_step < 0.0:
        raise ValueError("Time step must be nonnegative")
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    stationary = StochasticHeatConfig(
        signal_sd=config.signal_sd,
        forcing_lengthscale=config.forcing_lengthscale,
        diffusivity=config.diffusivity,
        cooling_rate=config.cooling_rate,
        quadrature_order=config.quadrature_order,
    )
    same_time_1 = np.column_stack([p1[:, :2], np.zeros(len(p1))])
    same_time_2 = np.column_stack([p2[:, :2], np.zeros(len(p2))])
    twice_lagged_2 = np.column_stack(
        [p2[:, :2], np.full(len(p2), 2.0 * time_step)]
    )
    marginal = stochastic_heat_covariance(same_time_1, same_time_2, stationary)
    propagated = stochastic_heat_covariance(
        same_time_1, twice_lagged_2, stationary
    )
    return marginal - propagated


def propagate_residual_draws(
    draws: np.ndarray,
    *,
    dx: float,
    dy: float,
    time_step: float,
    diffusivity: float,
    cooling_rate: float,
    displacement: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the linear heat semigroup, optionally followed by translation."""
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3:
        raise ValueError("Residual draws must have shape (draw, y, x)")
    if time_step < 0.0 or diffusivity < 0.0:
        raise ValueError("Time step and diffusivity must be nonnegative")
    sigma_y = np.sqrt(2.0 * diffusivity * time_step) / dy
    sigma_x = np.sqrt(2.0 * diffusivity * time_step) / dx
    propagated = gaussian_filter(
        values,
        sigma=(0.0, sigma_y, sigma_x),
        mode="constant",
        cval=0.0,
    )
    propagated *= np.exp(-cooling_rate * time_step)
    if displacement is not None:
        shift_pixels = (
            0.0,
            float(displacement[1]) / dy,
            float(displacement[0]) / dx,
        )
        propagated = shift_image(
            propagated,
            shift=shift_pixels,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return propagated
