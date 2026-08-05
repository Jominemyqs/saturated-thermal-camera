from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusivityEstimate:
    diffusivity: float
    cooling_rate: float
    ambient_temperature: float
    r_squared: float
    rmse: float
    n_samples: int


def _interior_laplacian(fields: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (fields[:, 1:-1, 2:] - 2.0 * fields[:, 1:-1, 1:-1] + fields[:, 1:-1, :-2])
        / dx**2
        + (
            fields[:, 2:, 1:-1]
            - 2.0 * fields[:, 1:-1, 1:-1]
            + fields[:, :-2, 1:-1]
        )
        / dy**2
    )


def _nonnegative_two_parameter_fit(
    design: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    """Solve a two-parameter nonnegative least-squares problem."""
    candidates = [np.zeros(2)]
    unconstrained, *_ = np.linalg.lstsq(design, response, rcond=None)
    if np.all(unconstrained >= 0.0):
        candidates.append(unconstrained)
    for column in range(2):
        coefficient = max(
            float(np.dot(design[:, column], response))
            / float(np.dot(design[:, column], design[:, column])),
            0.0,
        )
        candidate = np.zeros(2)
        candidate[column] = coefficient
        candidates.append(candidate)
    return min(
        candidates,
        key=lambda value: float(np.linalg.norm(design.dot(value) - response)),
    )


def estimate_effective_diffusivity(
    history: np.ndarray,
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    ambient_temperature: float | None = None,
    minimum_excess: float = 1.0,
    valid_mask: np.ndarray | None = None,
    maximum_samples: int = 250_000,
    seed: int = 0,
) -> DiffusivityEstimate:
    """Fit dT/dt = alpha * Laplacian(T) - beta * (T - ambient).

    Only pixels that are cooling and remain above the ambient-temperature margin
    are used. An optional validity mask can exclude active heat-source regions.
    """
    fields = np.asarray(history, dtype=float)
    time_values = np.asarray(times, dtype=float)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    if fields.ndim != 3:
        raise ValueError(f"Expected history with shape (time, y, x), got {fields.shape}")
    if fields.shape[0] != len(time_values):
        raise ValueError("History and time arrays have inconsistent lengths")
    if fields.shape[2] != len(x_values) or fields.shape[1] != len(y_values):
        raise ValueError("History and spatial grids have inconsistent shapes")
    if len(time_values) < 2:
        raise ValueError("At least two time snapshots are required")

    dt = np.diff(time_values)
    if np.any(dt <= 0.0):
        raise ValueError("Time values must be strictly increasing")
    dx = float(np.mean(np.diff(x_values)))
    dy = float(np.mean(np.diff(y_values)))

    if ambient_temperature is None:
        ambient_temperature = float(np.quantile(fields, 0.02))

    dtemperature_dt = np.diff(fields, axis=0) / dt[:, None, None]
    laplacian = _interior_laplacian(fields[:-1], dx, dy)
    interior_temperature = fields[:-1, 1:-1, 1:-1]
    interior_rate = dtemperature_dt[:, 1:-1, 1:-1]
    excess = interior_temperature - ambient_temperature

    mask = (
        np.isfinite(interior_rate)
        & np.isfinite(laplacian)
        & (interior_rate < 0.0)
        & (excess >= minimum_excess)
    )
    if valid_mask is not None:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape == fields.shape:
            validity = validity[:-1, 1:-1, 1:-1]
        elif validity.shape == fields[:-1].shape:
            validity = validity[:, 1:-1, 1:-1]
        elif validity.shape != interior_rate.shape:
            raise ValueError(
                "Validity mask must match the full history, the derivative history, "
                f"or the interior derivative shape; got {validity.shape}"
            )
        mask &= validity
    response = interior_rate[mask]
    design = np.column_stack([laplacian[mask], -excess[mask]])
    if len(response) < 20:
        raise ValueError(f"Only {len(response)} cooling samples satisfy the fit criteria")

    if len(response) > maximum_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(response), size=maximum_samples, replace=False)
        response = response[selected]
        design = design[selected]

    column_scale = np.linalg.norm(design, axis=0)
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    response_scale = max(float(np.linalg.norm(response)), 1.0)
    scaled_design = design / column_scale
    scaled_response = response / response_scale
    scaled_coefficients = _nonnegative_two_parameter_fit(
        scaled_design,
        scaled_response,
    )
    coefficients = scaled_coefficients * response_scale / column_scale
    fitted = design.dot(coefficients)
    residual = response - fitted
    total = response - np.mean(response)
    denominator = float(np.dot(total, total))
    r_squared = 1.0 - float(np.dot(residual, residual)) / denominator if denominator > 0 else np.nan

    return DiffusivityEstimate(
        diffusivity=float(coefficients[0]),
        cooling_rate=float(coefficients[1]),
        ambient_temperature=float(ambient_temperature),
        r_squared=r_squared,
        rmse=float(np.sqrt(np.mean(residual**2))),
        n_samples=len(response),
    )


def cooling_age(
    history: np.ndarray,
    times: np.ndarray,
    *,
    target_index: int = -1,
) -> np.ndarray:
    """Return time elapsed since each pixel's maximum temperature."""
    fields = np.asarray(history, dtype=float)
    time_values = np.asarray(times, dtype=float)
    if target_index < 0:
        target_index += len(time_values)
    if target_index < 0 or target_index >= len(time_values):
        raise IndexError("Target index lies outside the history")
    partial = fields[: target_index + 1]
    peak_indices = np.argmax(partial, axis=0)
    return np.maximum(time_values[target_index] - time_values[peak_indices], 0.0)
