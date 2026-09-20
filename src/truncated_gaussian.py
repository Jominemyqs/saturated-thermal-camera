from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def _cholesky_with_jitter(matrix: np.ndarray, base_jitter: float) -> np.ndarray:
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


def _first_boundary_hit(
    constraint_matrix: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    lower_centered: np.ndarray,
    maximum_time: float,
) -> tuple[float, int | None]:
    cosine_term = constraint_matrix.dot(position)
    sine_term = constraint_matrix.dot(velocity)
    radius = np.hypot(cosine_term, sine_term)
    feasible = radius >= np.abs(lower_centered)
    ratio = np.divide(
        lower_centered,
        radius,
        out=np.zeros_like(lower_centered),
        where=radius > 0.0,
    )
    angle = np.arccos(np.clip(ratio, -1.0, 1.0))
    phase = np.arctan2(sine_term, cosine_term)
    roots = np.column_stack(
        [
            np.mod(phase - angle, 2.0 * np.pi),
            np.mod(phase + angle, 2.0 * np.pi),
        ]
    )
    derivatives = np.column_stack(
        [
            -cosine_term * np.sin(roots[:, 0])
            + sine_term * np.cos(roots[:, 0]),
            -cosine_term * np.sin(roots[:, 1])
            + sine_term * np.cos(roots[:, 1]),
        ]
    )
    valid = feasible[:, None] & (derivatives < -1e-12) & (roots > 1e-10)
    candidates = np.where(valid, roots, np.inf)
    flat_index = int(np.argmin(candidates))
    hit_time = float(candidates.ravel()[flat_index])
    if not np.isfinite(hit_time) or hit_time > maximum_time:
        return float(maximum_time), None
    return hit_time, flat_index // 2


def _advance_harmonic(
    position: np.ndarray,
    velocity: np.ndarray,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(time)
    sine = np.sin(time)
    return (
        position * cosine + velocity * sine,
        -position * sine + velocity * cosine,
    )


def sample_lower_truncated_gaussian_hmc(
    mean: np.ndarray,
    covariance: np.ndarray,
    lower: float | np.ndarray,
    *,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
    relative_jitter: float = 1e-10,
    minimum_travel_time: float = 0.25 * np.pi,
    maximum_travel_time: float = 0.75 * np.pi,
) -> np.ndarray:
    """Exact reflected-HMC draws from a Gaussian with componentwise lower bounds.

    The Gaussian is whitened, harmonic Hamiltonian dynamics are integrated
    analytically, and momentum is reflected at each linear constraint.
    """
    location = np.asarray(mean, dtype=float).reshape(-1)
    scale = np.asarray(covariance, dtype=float)
    if scale.shape != (len(location), len(location)):
        raise ValueError("Covariance shape does not match the mean")
    bounds = np.broadcast_to(np.asarray(lower, dtype=float), location.shape).copy()
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("Invalid sample, burn-in, or thinning count")
    if not 0.0 < minimum_travel_time <= maximum_travel_time:
        raise ValueError("Travel-time bounds must be positive and ordered")
    marginal_scale = max(float(np.max(np.diag(scale))), 1.0)
    factor = _cholesky_with_jitter(scale, relative_jitter * marginal_scale)
    constraint_matrix = factor
    lower_centered = bounds - location
    marginal_sd = np.sqrt(np.maximum(np.diag(scale), 1e-12))
    initial = np.maximum(location, bounds + 0.5 * marginal_sd)
    position = solve_triangular(
        factor,
        initial - location,
        lower=True,
        check_finite=False,
    )
    rng = np.random.default_rng(seed)
    retained: list[np.ndarray] = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        velocity = rng.normal(size=len(location))
        remaining = float(
            rng.uniform(minimum_travel_time, maximum_travel_time)
        )
        collisions = 0
        while remaining > 1e-10:
            hit_time, hit_index = _first_boundary_hit(
                constraint_matrix,
                position,
                velocity,
                lower_centered,
                remaining,
            )
            position, velocity = _advance_harmonic(
                position, velocity, hit_time
            )
            remaining -= hit_time
            if hit_index is None:
                break
            normal = constraint_matrix[hit_index]
            normal_norm = float(normal.dot(normal))
            velocity -= 2.0 * float(normal.dot(velocity)) / normal_norm * normal
            position += 1e-10 * normal / normal_norm
            collisions += 1
            if collisions > 10_000:
                raise RuntimeError("Reflected HMC exceeded the collision limit")
        if step >= burn_in and (step - burn_in) % thin == 0:
            draw = location + factor.dot(position)
            if np.any(draw < bounds - 1e-7):
                raise RuntimeError("Reflected HMC produced an infeasible draw")
            retained.append(np.maximum(draw, bounds))
    return np.asarray(retained)
