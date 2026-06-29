from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

from src.gaussian_field import gaussian_temperature


METHODS = ("exact", "discard", "hinge", "censored")


def objective(
    params: np.ndarray,
    method: str,
    X: np.ndarray,
    Y: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    Tmax: float,
    noise_sd: float = 20.0,
    hinge_lambda: float = 1.0,
    temp_scale: float = 1000.0,
) -> float:
    """Objective for different treatments of saturated pixels.

    exact: treat clipped image as exact everywhere.
    discard: fit only unsaturated pixels.
    hinge: equality loss on unsaturated pixels + one-sided inequality penalty on saturated pixels.
    censored: Gaussian likelihood on unsaturated pixels + censored Gaussian survival likelihood on saturated pixels.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}.")

    pred = gaussian_temperature(X, Y, params)
    unsat = ~sat_mask

    if method == "exact":
        r = (pred - T_obs) / temp_scale
        return float(np.mean(r**2))

    if method == "discard":
        if np.sum(unsat) == 0:
            return np.inf
        r = (pred[unsat] - T_obs[unsat]) / temp_scale
        return float(np.mean(r**2))

    if method == "hinge":
        if np.sum(unsat) == 0:
            equality = 0.0
        else:
            r = (pred[unsat] - T_obs[unsat]) / temp_scale
            equality = float(np.mean(r**2))
        if np.sum(sat_mask) == 0:
            inequality = 0.0
        else:
            below_threshold = np.maximum(0.0, Tmax - pred[sat_mask]) / temp_scale
            inequality = float(np.mean(below_threshold**2))
        return equality + hinge_lambda * inequality

    # Censored Gaussian negative log-likelihood.
    # Unsaturated: y_i approx Normal(pred_i, noise_sd^2).
    # Saturated: observe y_i >= Tmax, so likelihood = P(Y >= Tmax | mean=pred_i)
    # = Phi((pred_i - Tmax)/noise_sd).
    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive for censored likelihood.")
    nll = 0.0
    if np.sum(unsat) > 0:
        z_unsat = (T_obs[unsat] - pred[unsat]) / noise_sd
        nll += 0.5 * float(np.mean(z_unsat**2))
    if np.sum(sat_mask) > 0:
        z_sat = (pred[sat_mask] - Tmax) / noise_sd
        # log_ndtr(z) = log Phi(z), stable even when z is very negative.
        nll += -float(np.mean(log_ndtr(z_sat)))
    return nll
