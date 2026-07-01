from __future__ import annotations

import numpy as np

from src.gaussian_field import GaussianParams, gaussian_temperature


FIELD_NAMES = ("matched_gaussian", "rotated_gaussian", "two_component_wake", "skewed_wake")

FIELD_LABELS = {
    "matched_gaussian": "matched Gaussian",
    "rotated_gaussian": "rotated Gaussian",
    "two_component_wake": "two-component wake",
    "skewed_wake": "skewed wake",
}


def rotated_gaussian(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    T0: float,
    A: float,
    xc: float,
    yc: float,
    sx: float,
    sy: float,
    angle_degrees: float,
) -> np.ndarray:
    theta = np.deg2rad(angle_degrees)
    x = X - xc
    y = Y - yc
    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    exponent = -(xr**2 / (2.0 * sx**2) + yr**2 / (2.0 * sy**2))
    return T0 + A * np.exp(exponent)


def make_truth_field(name: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Generate truth fields for model-misspecification experiments.

    The fitted model is always the axis-aligned single Gaussian in gaussian_field.py.
    """
    if name == "matched_gaussian":
        params = GaussianParams(T0=473.15, A=1100.0, xc=0.4, yc=-0.2, sx=1.25, sy=0.45)
        return gaussian_temperature(X, Y, params)

    if name == "rotated_gaussian":
        return rotated_gaussian(
            X,
            Y,
            T0=473.15,
            A=1100.0,
            xc=0.4,
            yc=-0.2,
            sx=1.45,
            sy=0.38,
            angle_degrees=32.0,
        )

    if name == "two_component_wake":
        background = 473.15
        main = 850.0 * np.exp(-((X - 0.45) ** 2 / (2.0 * 1.05**2) + (Y + 0.2) ** 2 / (2.0 * 0.40**2)))
        wake = 430.0 * np.exp(-((X + 0.85) ** 2 / (2.0 * 2.20**2) + (Y + 0.12) ** 2 / (2.0 * 0.34**2)))
        return background + main + wake

    if name == "skewed_wake":
        background = 473.15
        xc = 0.35
        yc = -0.2
        hotspot = 800.0 * np.exp(-((X - xc) ** 2 / (2.0 * 0.85**2) + (Y - yc) ** 2 / (2.0 * 0.38**2)))
        left_distance = np.maximum(0.0, xc - X)
        right_distance = np.maximum(0.0, X - xc)
        wake = 360.0 * np.exp(
            -left_distance / 2.6
            -right_distance**2 / (2.0 * 0.30**2)
            -((Y - yc) ** 2) / (2.0 * 0.55**2)
        )
        return background + hotspot + wake

    raise ValueError(f"Unknown truth field {name!r}. Choose from {FIELD_NAMES}.")
