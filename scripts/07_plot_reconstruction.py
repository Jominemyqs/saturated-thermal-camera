from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.fit import fit_model
from src.gaussian_field import gaussian_temperature
from src.losses import METHODS
from src.metrics import compute_metrics
from src.plotting import save_reconstruction_diagnostics, save_reconstruction_grid


def main() -> None:
    out_dir = ROOT / "outputs"
    data_path = out_dir / "synthetic_example.npz"
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/00_make_synthetic.py first.")

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    T_true = data["T_true"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    Tmax = float(data["Tmax"])
    true_params = data["true_params"]

    reconstructions = {}
    rows = []
    for method in METHODS:
        print(f"Fitting method: {method}")
        fit = fit_model(
            method,
            X,
            Y,
            T_obs,
            sat_mask,
            Tmax,
            noise_sd=20.0,
            hinge_lambda=1.0,
            n_starts=20,
            seed=42,
        )
        params_hat = fit["params_hat"]
        reconstructions[method] = gaussian_temperature(X, Y, params_hat)
        metrics = compute_metrics(params_hat, true_params, X, Y, T_true, sat_mask)
        rows.append({"method": method, "objective": fit["objective"], "success": fit["success"], **metrics})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "reconstruction_fit_results.csv"
    grid_path = out_dir / "reconstruction_predictions.png"
    diagnostic_path = out_dir / "reconstruction_diagnostics.png"

    df.to_csv(csv_path, index=False)
    save_reconstruction_grid(X, Y, T_true, T_obs, reconstructions, grid_path)
    save_reconstruction_diagnostics(X, Y, T_true, T_obs, sat_mask, reconstructions, diagnostic_path)

    print("\nReconstruction summary:")
    cols = ["method", "field_rel_l2", "peak_rel_error", "A_hat", "sx_hat", "sy_hat"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {grid_path}")
    print(f"Saved {diagnostic_path}")


if __name__ == "__main__":
    main()
