from __future__ import annotations

import numpy as np

from src.gaussian_field import gaussian_temperature

PARAM_NAMES = ["T0", "A", "xc", "yc", "sx", "sy"]


def relative_l2(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        truth = truth[mask]
    denom = np.linalg.norm(truth.ravel())
    if denom == 0:
        return float(np.linalg.norm((pred - truth).ravel()))
    return float(np.linalg.norm((pred - truth).ravel()) / denom)


def compute_metrics(
    params_hat: np.ndarray,
    params_true: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    T_true: np.ndarray,
    sat_mask: np.ndarray,
) -> dict:
    out = compute_field_metrics(params_hat, X, Y, T_true, sat_mask)
    for name, est, true in zip(PARAM_NAMES, params_hat, params_true):
        out[f"{name}_hat"] = float(est)
        out[f"{name}_true"] = float(true)
        out[f"{name}_rel_error"] = float(abs(est - true) / max(abs(true), 1e-12))
    return out


def compute_field_metrics(
    params_hat: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    T_true: np.ndarray,
    sat_mask: np.ndarray,
) -> dict:
    """Compute reconstruction metrics when the truth may not have Gaussian parameters."""
    pred = gaussian_temperature(X, Y, params_hat)
    unsat_mask = ~sat_mask
    return {
        "field_rel_l2": relative_l2(pred, T_true),
        "peak_abs_error": float(abs(np.max(pred) - np.max(T_true))),
        "peak_rel_error": float(abs(np.max(pred) - np.max(T_true)) / max(abs(np.max(T_true)), 1e-12)),
        "sat_region_rel_l2": relative_l2(pred, T_true, sat_mask) if np.any(sat_mask) else np.nan,
        "unsat_region_rel_l2": relative_l2(pred, T_true, unsat_mask) if np.any(unsat_mask) else np.nan,
    }
