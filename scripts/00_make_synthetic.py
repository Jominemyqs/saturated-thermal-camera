from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.gaussian_field import GaussianParams, censor_by_fraction, create_grid, gaussian_temperature
from src.plotting import save_field_triplet


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    X, Y = create_grid(nx=100, ny=80, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    true_params = GaussianParams(T0=473.15, A=1100.0, xc=0.4, yc=-0.2, sx=1.25, sy=0.45)
    T_true = gaussian_temperature(X, Y, true_params)

    frac_saturated = 0.10
    T_obs, sat_mask, Tmax = censor_by_fraction(T_true, frac_saturated)

    np.savez(
        out_dir / "synthetic_example.npz",
        X=X,
        Y=Y,
        T_true=T_true,
        T_obs=T_obs,
        sat_mask=sat_mask,
        Tmax=Tmax,
        true_params=true_params.as_array(),
        frac_saturated=frac_saturated,
    )

    save_field_triplet(
        X,
        Y,
        T_true,
        T_obs,
        sat_mask,
        out_dir / "synthetic_example.png",
        title_suffix=f"Synthetic Gaussian field, top {100*frac_saturated:.0f}% censored, Tmax={Tmax:.1f} K",
    )
    print(f"Saved {out_dir / 'synthetic_example.npz'}")
    print(f"Saved {out_dir / 'synthetic_example.png'}")
    print(f"Tmax = {Tmax:.3f}, saturated pixels = {sat_mask.mean():.3%}")


if __name__ == "__main__":
    main()
