from __future__ import annotations

from pathlib import Path

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
