from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, differential_evolution

from src.losses import objective


def default_bounds(X: np.ndarray, Y: np.ndarray, T_obs: np.ndarray) -> list[tuple[float, float]]:
    """Reasonable parameter bounds for the toy problem."""
    xmin, xmax = float(np.min(X)), float(np.max(X))
    ymin, ymax = float(np.min(Y)), float(np.max(Y))
    xrange = xmax - xmin
    yrange = ymax - ymin
    tmin = float(np.min(T_obs))
    tmax = float(np.max(T_obs))
    return [
        (max(0.0, tmin - 300.0), tmax + 300.0),       # T0
        (1.0, max(200.0, 4.0 * (tmax - tmin + 1.0))), # A
        (xmin, xmax),                                  # xc
        (ymin, ymax),                                  # yc
        (0.05 * xrange, 0.8 * xrange),                 # sx
        (0.05 * yrange, 0.8 * yrange),                 # sy
    ]


def heuristic_initial_guess(X: np.ndarray, Y: np.ndarray, T_obs: np.ndarray) -> np.ndarray:
    """Simple initial guess from observed/clipped image."""
    idx = np.unravel_index(np.argmax(T_obs), T_obs.shape)
    T0 = float(np.percentile(T_obs, 5))
    A = float(max(np.max(T_obs) - T0, 50.0))
    xc = float(X[idx])
    yc = float(Y[idx])
    sx = 0.20 * (float(np.max(X)) - float(np.min(X)))
    sy = 0.20 * (float(np.max(Y)) - float(np.min(Y)))
    return np.array([T0, A, xc, yc, sx, sy], dtype=float)


def random_initial_guess(bounds: list[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)


def fit_model(
    method: str,
    X: np.ndarray,
    Y: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    Tmax: float,
    noise_sd: float = 20.0,
    hinge_lambda: float = 1.0,
    n_starts: int = 15,
    seed: int = 0,
    use_global: bool = False,
) -> dict:
    """Fit Gaussian parameters under one loss method.

    Returns a dictionary containing fitted parameters, final objective, and optimizer info.
    """
    rng = np.random.default_rng(seed)
    bounds = default_bounds(X, Y, T_obs)

    def fun(p: np.ndarray) -> float:
        return objective(
            p,
            method=method,
            X=X,
            Y=Y,
            T_obs=T_obs,
            sat_mask=sat_mask,
            Tmax=Tmax,
            noise_sd=noise_sd,
            hinge_lambda=hinge_lambda,
        )

    best = None

    # Optional coarse global pass. Useful if local starts fail, but slower.
    if use_global:
        de = differential_evolution(fun, bounds=bounds, seed=seed, polish=False, maxiter=80, popsize=8, tol=1e-5)
        starts = [de.x]
    else:
        starts = []

    starts.append(heuristic_initial_guess(X, Y, T_obs))
    starts.extend(random_initial_guess(bounds, rng) for _ in range(max(0, n_starts - len(starts))))

    for x0 in starts:
        res = minimize(fun, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res

    return {
        "method": method,
        "params_hat": best.x,
        "objective": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }
