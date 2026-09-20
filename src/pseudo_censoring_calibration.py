from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm


FEATURE_NAMES = (
    "observed_censored_fraction",
    "local_censored_fraction",
    "posterior_standardized_gap",
    "prior_standardized_gap",
    "posterior_exceedance_probability",
    "log_posterior_sd",
    "normalized_heat_flux",
)


def calibration_features(
    *,
    posterior_mean: np.ndarray,
    posterior_sd: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    observed_censored: np.ndarray,
    heat_flux: np.ndarray,
    ceiling: float,
    local_radius: int = 2,
) -> np.ndarray:
    """Construct deployment-available pixel features on a common grid."""
    mean = np.asarray(posterior_mean, dtype=float)
    sd = np.maximum(np.asarray(posterior_sd, dtype=float), 1e-8)
    prior = np.asarray(prior_mean, dtype=float)
    prior_scale = np.maximum(np.asarray(prior_sd, dtype=float), 1e-8)
    censored = np.asarray(observed_censored, dtype=bool)
    flux = np.maximum(np.asarray(heat_flux, dtype=float), 0.0)
    if not (
        mean.shape
        == sd.shape
        == prior.shape
        == prior_scale.shape
        == censored.shape
        == flux.shape
    ):
        raise ValueError("All calibration feature fields must share one shape")
    size = 2 * int(local_radius) + 1
    local_fraction = uniform_filter(censored.astype(float), size=size, mode="nearest")
    posterior_gap = (mean - ceiling) / sd
    prior_gap = (prior - ceiling) / prior_scale
    probability = norm.cdf(posterior_gap)
    peak_flux = float(np.max(flux))
    normalized_flux = flux / peak_flux if peak_flux > 0.0 else np.zeros_like(flux)
    severity = np.full(mean.size, float(np.mean(censored)))
    return np.column_stack(
        [
            severity,
            local_fraction.ravel(),
            posterior_gap.ravel(),
            prior_gap.ravel(),
            probability.ravel(),
            np.log(sd.ravel()),
            normalized_flux.ravel(),
        ]
    )


def equal_group_weights(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) == 0:
        raise ValueError("At least one calibration group is required")
    return 1.0 / (len(unique) * counts[inverse])


def _pairwise_dispersion(centered_draws: np.ndarray) -> np.ndarray:
    draws = np.asarray(centered_draws, dtype=float)
    n_draws = len(draws)
    if draws.ndim != 2 or n_draws < 2:
        raise ValueError("Centered draws must have shape (draw, example)")
    ordered = np.sort(draws, axis=0)
    weights = 2.0 * np.arange(1, n_draws + 1) - n_draws - 1.0
    return np.sum(weights[:, None] * ordered, axis=0) / (
        n_draws * (n_draws - 1)
    )


def scaled_crps(
    centered_draws: np.ndarray,
    target_offsets: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    draws = np.asarray(centered_draws, dtype=float)
    offsets = np.asarray(target_offsets, dtype=float).reshape(-1)
    scale = np.asarray(scales, dtype=float).reshape(-1)
    if draws.shape[1] != len(offsets) or len(scale) != len(offsets):
        raise ValueError("Draws, targets, and scales must share examples")
    first = np.mean(np.abs(draws * scale[None, :] - offsets[None, :]), axis=0)
    return first - scale * _pairwise_dispersion(draws)


@dataclass(frozen=True)
class ScaleCalibrator:
    name: str
    feature_indices: tuple[int, ...]
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    coefficients: np.ndarray
    global_scale: float | None = None
    maximum_scale: float = 20.0

    def scale(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if self.global_scale is not None:
            return np.full(len(values), self.global_scale)
        selected = values[:, self.feature_indices]
        standardized = (selected - self.feature_mean) / self.feature_sd
        linear = self.coefficients[0] + standardized.dot(self.coefficients[1:])
        return np.clip(1.0 + np.logaddexp(0.0, linear), 1.0, self.maximum_scale)


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None] * values, axis=0)
    variance = np.sum(weights[:, None] * (values - mean) ** 2, axis=0)
    return mean, np.maximum(np.sqrt(variance), 1e-8)


def fit_scale_calibrators(
    *,
    features: np.ndarray,
    centered_draws: np.ndarray,
    target_offsets: np.ndarray,
    group_ids: np.ndarray,
    regularization: float = 1e-3,
) -> dict[str, ScaleCalibrator]:
    """Fit global, severity-only, and compact feature-based inflation rules."""
    values = np.asarray(features, dtype=float)
    draws = np.asarray(centered_draws, dtype=float)
    offsets = np.asarray(target_offsets, dtype=float).reshape(-1)
    if values.shape != (len(offsets), len(FEATURE_NAMES)):
        raise ValueError("Calibration features have the wrong shape")
    if draws.shape[1] != len(offsets):
        raise ValueError("Calibration draws and targets do not align")
    weights = equal_group_weights(group_ids)

    def objective_for_scale(scale: np.ndarray) -> float:
        return float(np.sum(weights * scaled_crps(draws, offsets, scale)))

    global_result = minimize_scalar(
        lambda value: objective_for_scale(np.full(len(offsets), value)),
        bounds=(1.0, 20.0),
        method="bounded",
    )
    if not global_result.success:
        raise RuntimeError("Global inflation calibration failed")
    global_calibrator = ScaleCalibrator(
        name="global inflation",
        feature_indices=(),
        feature_mean=np.empty(0),
        feature_sd=np.empty(0),
        coefficients=np.empty(0),
        global_scale=float(global_result.x),
    )

    def fit_model(name: str, feature_indices: tuple[int, ...]) -> ScaleCalibrator:
        selected = values[:, feature_indices]
        mean, sd = _weighted_moments(selected, weights)
        standardized = (selected - mean) / sd

        def objective(coefficients: np.ndarray) -> float:
            linear = coefficients[0] + standardized.dot(coefficients[1:])
            scale = np.clip(1.0 + np.logaddexp(0.0, linear), 1.0, 20.0)
            penalty = regularization * float(np.sum(coefficients[1:] ** 2))
            return objective_for_scale(scale) + penalty

        initial_scale = max(float(global_result.x) - 1.0, 1e-5)
        initial = np.zeros(len(feature_indices) + 1)
        initial[0] = np.log(np.expm1(initial_scale))
        bounds = [(-12.0, 8.0)] + [(-5.0, 5.0)] * len(feature_indices)
        if feature_indices and feature_indices[0] == 0:
            bounds[1] = (0.0, 5.0)
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            raise RuntimeError(f"{name} calibration failed: {result.message}")
        return ScaleCalibrator(
            name=name,
            feature_indices=feature_indices,
            feature_mean=mean,
            feature_sd=sd,
            coefficients=np.asarray(result.x, dtype=float),
        )

    return {
        "global inflation": global_calibrator,
        "severity inflation": fit_model("severity inflation", (0,)),
        "feature inflation": fit_model(
            "feature inflation", tuple(range(len(FEATURE_NAMES)))
        ),
    }


def inflate_draws(
    draws: np.ndarray,
    posterior_mean: np.ndarray,
    scales: np.ndarray,
    apply_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(draws, dtype=float)
    mean = np.asarray(posterior_mean, dtype=float).reshape(-1)
    scale = np.asarray(scales, dtype=float).reshape(-1)
    mask = np.asarray(apply_mask, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(mean):
        raise ValueError("Draws and posterior mean do not align")
    if len(scale) != len(mean) or len(mask) != len(mean):
        raise ValueError("Scale and mask must contain one value per target")
    output = values.copy()
    output[:, mask] = mean[mask][None, :] + scale[mask][None, :] * (
        values[:, mask] - mean[mask][None, :]
    )
    return output
