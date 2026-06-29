from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GaussianParams:
    """Parameters for the synthetic thermal image.

    T(x,y) = T0 + A exp(-((x-xc)^2/(2 sx^2) + (y-yc)^2/(2 sy^2))).
    """

    T0: float
    A: float
    xc: float
    yc: float
    sx: float
    sy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.T0, self.A, self.xc, self.yc, self.sx, self.sy], dtype=float)


def create_grid(
    nx: int = 100,
    ny: int = 80,
    xlim: Tuple[float, float] = (-5.0, 5.0),
    ylim: Tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D Cartesian grid."""
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def gaussian_temperature(X: np.ndarray, Y: np.ndarray, params: np.ndarray | GaussianParams) -> np.ndarray:
    """Evaluate the synthetic Gaussian temperature field."""
    if isinstance(params, GaussianParams):
        p = params.as_array()
    else:
        p = np.asarray(params, dtype=float)
    T0, A, xc, yc, sx, sy = p
    sx = max(float(sx), 1e-8)
    sy = max(float(sy), 1e-8)
    exponent = -((X - xc) ** 2 / (2.0 * sx**2) + (Y - yc) ** 2 / (2.0 * sy**2))
    return T0 + A * np.exp(exponent)


def censor_by_fraction(T_true: np.ndarray, frac_saturated: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Clip the top frac_saturated fraction of pixels.

    Returns
    -------
    T_obs : observed/clipped temperature field
    sat_mask : boolean mask where T_true >= Tmax
    Tmax : saturation threshold
    """
    if not (0.0 <= frac_saturated < 1.0):
        raise ValueError("frac_saturated must be in [0, 1).")
    if frac_saturated == 0.0:
        Tmax = float(np.max(T_true) + 1.0)
    else:
        Tmax = float(np.quantile(T_true, 1.0 - frac_saturated))
    sat_mask = T_true >= Tmax
    T_obs = np.minimum(T_true, Tmax)
    return T_obs, sat_mask, Tmax


def add_noise_then_censor(
    T_true: np.ndarray,
    frac_saturated: float,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Optional noisy observation model: T_meas = T_true + noise, T_obs = min(T_meas, Tmax)."""
    T_noisy = T_true + rng.normal(0.0, noise_sd, size=T_true.shape)
    if frac_saturated == 0.0:
        Tmax = float(np.max(T_noisy) + 1.0)
    else:
        Tmax = float(np.quantile(T_noisy, 1.0 - frac_saturated))
    sat_mask = T_noisy >= Tmax
    T_obs = np.minimum(T_noisy, Tmax)
    return T_obs, sat_mask, Tmax, T_noisy
