from __future__ import annotations

import matplotlib.colors
import numpy as np
from scipy.ndimage import gaussian_filter


TAIL_EXCESS_MIN_K = 0.15
TAIL_CONTOUR_EXCESS_K = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


def tail_temperature_norm(
    ambient: float,
    ceiling: float,
    *,
    gamma: float = 0.60,
) -> matplotlib.colors.PowerNorm:
    """Shared display normalization that emphasizes cooling trails."""
    lower = float(ambient) + TAIL_EXCESS_MIN_K
    upper = max(float(ceiling), lower + 0.5)
    return matplotlib.colors.PowerNorm(gamma=gamma, vmin=lower, vmax=upper)


def add_tail_contours(
    axis,
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    *,
    ambient: float,
    ceiling: float,
) -> None:
    """Overlay fixed excess-temperature contours when they cross the field."""
    values = gaussian_filter(np.asarray(field, dtype=float), sigma=0.65, mode="nearest")
    candidates = float(ambient) + TAIL_CONTOUR_EXCESS_K
    levels = candidates[
        (candidates > float(np.min(values)))
        & (candidates < float(np.max(values)))
        & (candidates < float(ceiling))
    ]
    if len(levels):
        axis.contour(
            xs,
            ys,
            values,
            levels=levels,
            colors="white",
            linewidths=0.45,
            alpha=0.58,
        )
