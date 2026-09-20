from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.stochastic_heat_gp import (
    StochasticHeatConfig,
    finite_step_innovation_covariance,
    propagate_residual_draws,
)


@dataclass(frozen=True)
class SequentialAdvectiveConfig:
    signal_sd: float
    forcing_lengthscale: float
    diffusivity: float
    cooling_rate: float
    current_noise_sd: float
    quadrature_order: int = 24
    relative_jitter: float = 1e-7


def artificial_ceiling_schedule(
    reference_observation: np.ndarray,
    reference_ceiling: float,
    target_fractions: list[float] | tuple[float, ...] | np.ndarray,
) -> list[dict[str, float | str]]:
    """Choose lower ceilings from an already observed reference camera field.

    No latent truth is accepted by this function. The first row is the frozen
    reference ceiling; later rows are observation quantiles below that ceiling.
    """
    observed = np.asarray(reference_observation, dtype=float).reshape(-1)
    targets = np.asarray(target_fractions, dtype=float).reshape(-1)
    if not np.all(np.isfinite(observed)):
        raise ValueError("Reference observations must be finite")
    if not np.isfinite(reference_ceiling):
        raise ValueError("Reference ceiling must be finite")
    if len(targets) < 1 or np.any((targets <= 0.0) | (targets >= 1.0)):
        raise ValueError("Target fractions must lie strictly between zero and one")
    if np.any(np.diff(targets) <= 0.0):
        raise ValueError("Target fractions must be strictly increasing")

    rows: list[dict[str, float | str]] = []
    previous_ceiling = float(reference_ceiling)
    for index, target in enumerate(targets):
        if index == 0:
            ceiling = float(reference_ceiling)
            source = "frozen reference ceiling"
        else:
            candidate = float(np.quantile(observed, 1.0 - target))
            ceiling = min(candidate, float(np.nextafter(previous_ceiling, -np.inf)))
            source = "reference-observation quantile"
        actual_fraction = float(np.mean(observed >= ceiling))
        rows.append(
            {
                "level_index": index,
                "level": "reference" if index == 0 else f"target_{100.0 * target:g}pct",
                "target_censored_fraction": float(target),
                "ceiling_K": ceiling,
                "reference_observed_fraction_at_or_above_ceiling": actual_fraction,
                "ceiling_source": source,
            }
        )
        previous_ceiling = ceiling
    return rows


def lower_camera_ceiling(
    reference_camera: dict[str, object],
    ceiling: float,
) -> dict[str, object]:
    """Apply a lower ceiling to an existing noisy clipped camera realization."""
    reference_ceiling = float(reference_camera["current"]["threshold"])
    if ceiling > reference_ceiling:
        raise ValueError("Artificial ceiling cannot exceed the reference ceiling")
    fixed_mask = np.asarray(reference_camera["fixed_observation_mask"], dtype=bool)
    lowered = deepcopy(reference_camera)
    lowered_frames = []
    for reference_frame in reference_camera["frames"]:
        frame = deepcopy(reference_frame)
        reference_values = np.asarray(reference_frame["clipped_full"], dtype=float)
        clipped = np.minimum(reference_values, ceiling)
        saturated = reference_values >= ceiling
        frame["clipped_full"] = clipped
        frame["saturated_full"] = saturated
        frame["values"] = clipped[fixed_mask]
        frame["sat_mask"] = saturated[fixed_mask]
        lowered_frames.append(frame)
    lowered["frames"] = lowered_frames
    current_frame = lowered_frames[1]
    current = deepcopy(reference_camera["current"])
    current["y_obs"] = np.asarray(current_frame["values"], dtype=float)
    current["sat_mask"] = np.asarray(current_frame["sat_mask"], dtype=bool)
    current["threshold"] = float(ceiling)
    current["clipped_full"] = np.asarray(current_frame["clipped_full"], dtype=float)
    current["saturated_full"] = np.asarray(current_frame["saturated_full"], dtype=bool)
    lowered["current"] = current
    return lowered


def matched_censoring_observations(
    prepared,
    *,
    frame: dict[str, object],
    fixed_mask: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Use sparse unsaturated values and every observed-censored location."""
    clipped = np.asarray(frame["clipped_full"], dtype=float)
    saturated = np.asarray(frame["saturated_full"], dtype=bool)
    included = np.asarray(fixed_mask, dtype=bool) | saturated
    points = np.column_stack(
        [prepared.points, np.full(len(prepared.points), float(frame["time"]))]
    )
    return {
        "x_obs": points[included.ravel()],
        "y_obs": clipped[included],
        "sat_mask": saturated[included],
        "x_pred": points,
        "threshold": float(threshold),
        "clipped_full": clipped,
        "saturated_full": saturated,
    }


def observation_indices(prepared, observation_points: np.ndarray) -> np.ndarray:
    points = np.asarray(observation_points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((points[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((points[:, 1] - prepared.ys[0]) / dy).astype(int)
    if np.any(ix < 0) or np.any(ix >= len(prepared.xs)):
        raise ValueError("Observation x coordinate is outside the prediction grid")
    if np.any(iy < 0) or np.any(iy >= len(prepared.ys)):
        raise ValueError("Observation y coordinate is outside the prediction grid")
    return iy * len(prepared.xs) + ix


def sequential_prior_blocks(
    prepared,
    *,
    camera: dict[str, object],
    previous_mean: np.ndarray,
    previous_draws: np.ndarray,
    mean_field: np.ndarray,
    displacement: np.ndarray | None,
    config: SequentialAdvectiveConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build moment-matched one-step prior blocks with optional transport."""
    current = camera["current"]
    current_points = np.asarray(current["x_pred"], dtype=float)
    observation_points = np.asarray(current["x_obs"], dtype=float)
    observed_indices = observation_indices(prepared, observation_points)
    heat_config = StochasticHeatConfig(
        signal_sd=config.signal_sd,
        forcing_lengthscale=config.forcing_lengthscale,
        diffusivity=config.diffusivity,
        cooling_rate=config.cooling_rate,
        quadrature_order=config.quadrature_order,
    )
    previous_index = int(camera["frames"][0]["time_index"])
    current_index = int(camera["frames"][1]["time_index"])
    time_step = float(prepared.times[current_index] - prepared.times[previous_index])
    centered = np.asarray(previous_draws, dtype=float) - np.asarray(
        previous_mean, dtype=float
    )[None, :, :]
    propagated = propagate_residual_draws(
        centered,
        dx=float(np.mean(np.diff(prepared.xs))),
        dy=float(np.mean(np.diff(prepared.ys))),
        time_step=time_step,
        diffusivity=config.diffusivity,
        cooling_rate=config.cooling_rate,
        displacement=(
            None if displacement is None else np.asarray(displacement, dtype=float)
        ),
    )
    denominator = np.sqrt(len(previous_draws) - 1.0)
    features = propagated.reshape(len(previous_draws), -1).T / denominator
    observed_features = features[observed_indices]

    innovation_oo = finite_step_innovation_covariance(
        observation_points, observation_points, heat_config, time_step
    )
    innovation_oo = 0.5 * (innovation_oo + innovation_oo.T)
    innovation_po = finite_step_innovation_covariance(
        current_points, observation_points, heat_config, time_step
    )
    innovation_variance_scalar = float(
        finite_step_innovation_covariance(
            current_points[:1], current_points[:1], heat_config, time_step
        )[0, 0]
    )
    observed_covariance = innovation_oo + observed_features.dot(
        observed_features.T
    )
    pred_observed_covariance = innovation_po + features.dot(
        observed_features.T
    )
    prediction_variance = np.full(
        len(current_points), max(innovation_variance_scalar, 0.0)
    ) + np.sum(features**2, axis=1)
    flat_mean = np.asarray(mean_field, dtype=float).ravel()
    blocks = {
        "prediction_mean": flat_mean,
        "observation_mean": flat_mean[observed_indices],
        "observed_covariance": observed_covariance,
        "pred_observed_covariance": pred_observed_covariance,
        "prediction_variance": prediction_variance,
    }
    diagnostics = {
        "time_lag_s": time_step,
        "innovation_sd_K": float(np.sqrt(max(innovation_variance_scalar, 0.0))),
        "mean_propagated_previous_variance_K2": float(np.mean(np.sum(features**2, axis=1))),
    }
    return blocks, diagnostics


def sequential_advective_prior_blocks(
    prepared,
    *,
    camera: dict[str, object],
    previous_mean: np.ndarray,
    previous_draws: np.ndarray,
    advective_mean: np.ndarray,
    displacement: np.ndarray,
    config: SequentialAdvectiveConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Backward-compatible wrapper for the advective one-step prior."""
    return sequential_prior_blocks(
        prepared,
        camera=camera,
        previous_mean=previous_mean,
        previous_draws=previous_draws,
        mean_field=advective_mean,
        displacement=displacement,
        config=config,
    )
