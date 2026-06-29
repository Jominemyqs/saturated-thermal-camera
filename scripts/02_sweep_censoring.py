from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.fit import fit_model
from src.gaussian_field import GaussianParams, censor_by_fraction, create_grid, gaussian_temperature
from src.losses import METHODS
from src.metrics import compute_metrics


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    X, Y = create_grid(nx=100, ny=80, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    true_params_obj = GaussianParams(T0=473.15, A=1100.0, xc=0.4, yc=-0.2, sx=1.25, sy=0.45)
    true_params = true_params_obj.as_array()
    T_true = gaussian_temperature(X, Y, true_params)

    censor_fracs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.40]
    rows = []
    for frac in censor_fracs:
        T_obs, sat_mask, Tmax = censor_by_fraction(T_true, frac)
        print(f"\nCensoring fraction target={frac:.0%}, actual={sat_mask.mean():.2%}, Tmax={Tmax:.2f}")
        for method in METHODS:
            print(f"  fitting {method}")
            fit = fit_model(
                method,
                X,
                Y,
                T_obs,
                sat_mask,
                Tmax,
                noise_sd=20.0,
                hinge_lambda=1.0,
                n_starts=15,
                seed=123,
            )
            metrics = compute_metrics(fit["params_hat"], true_params, X, Y, T_true, sat_mask)
            rows.append(
                {
                    "target_frac_saturated": frac,
                    "actual_frac_saturated": float(sat_mask.mean()),
                    "Tmax": Tmax,
                    "method": method,
                    "objective": fit["objective"],
                    "success": fit["success"],
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df[["target_frac_saturated", "method", "field_rel_l2", "peak_rel_error", "A_rel_error", "sx_rel_error", "sy_rel_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
