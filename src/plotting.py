from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thermal_censored_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def save_field_triplet(X, Y, T_true, T_obs, sat_mask, path: str | Path, title_suffix: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(T_true, origin="lower", extent=extent, aspect="auto")
    axes[0].set_title("True temperature")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(T_obs, origin="lower", extent=extent, aspect="auto")
    axes[1].set_title("Observed/clipped")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(sat_mask.astype(float), origin="lower", extent=extent, aspect="auto")
    axes[2].set_title("Saturated mask")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title_suffix)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reconstruction_grid(X, Y, T_true, T_obs, reconstructions: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    n = 2 + len(reconstructions)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5), constrained_layout=True)
    fields = {"true": T_true, "observed": T_obs, **reconstructions}
    vmin = min(float(np.min(v)) for v in fields.values())
    vmax = max(float(np.max(v)) for v in fields.values())

    for ax, (name, field) in zip(axes, fields.items()):
        im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reconstruction_diagnostics(
    X,
    Y,
    T_true,
    T_obs,
    sat_mask,
    reconstructions: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Save side-by-side fields and residuals for visual reconstruction checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    fields = {"true": T_true, "observed/clipped": T_obs, **reconstructions}
    names = list(fields)
    vmin = min(float(np.min(field)) for field in fields.values())
    vmax = max(float(np.max(field)) for field in fields.values())

    residuals = {name: field - T_true for name, field in fields.items() if name != "true"}
    resid_abs_max = max(float(np.max(np.abs(resid))) for resid in residuals.values())

    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.2), constrained_layout=True)

    for col, name in enumerate(names):
        ax = axes[0, col]
        im = ax.imshow(fields[name], origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        ax.contour(X, Y, sat_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes[0, :], shrink=0.85, label="temperature")

    for col, name in enumerate(names):
        ax = axes[1, col]
        if name == "true":
            resid = np.zeros_like(T_true)
            title = "true residual"
        else:
            resid = residuals[name]
            title = f"{name} - true"
        im_resid = ax.imshow(
            resid,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=-resid_abs_max,
            vmax=resid_abs_max,
            cmap="coolwarm",
        )
        ax.contour(X, Y, sat_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im_resid, ax=axes[1, :], shrink=0.85, label="prediction error")
    fig.savefig(path, dpi=220)
    plt.close(fig)
