from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fit import fit_model
from src.gaussian_field import GaussianParams, censor_by_fraction, create_grid, gaussian_temperature
from src.losses import METHODS
from src.metrics import compute_metrics


def plot_noise_results(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    summary = (
        df.groupby(["noise_sd", "method"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    for method, group in summary.groupby("method"):
        group = group.sort_values("noise_sd")
        ax.plot(group["noise_sd"], group["mean"], marker="o", label=method)
        # Light uncertainty band over random noise replicates.
        y0 = group["mean"] - group["std"].fillna(0.0)
        y1 = group["mean"] + group["std"].fillna(0.0)
        ax.fill_between(group["noise_sd"], y0, y1, alpha=0.12)

    ax.set_xlabel("Observation noise standard deviation")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs observation noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    X, Y = create_grid(nx=80, ny=60, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    true_params_obj = GaussianParams(T0=473.15, A=1100.0, xc=0.4, yc=-0.2, sx=1.25, sy=0.45)
    true_params = true_params_obj.as_array()
    T_true = gaussian_temperature(X, Y, true_params)

    # Keep the same nominal camera threshold as the 10% case
    frac_saturated = 0.10
    _, _, Tmax = censor_by_fraction(T_true, frac_saturated)

    noise_levels = [0.0, 10.0, 20.0, 40.0, 80.0]
    n_reps = 3
    rows = []

    for noise_sd in noise_levels:
        for rep in range(n_reps):
            rng = np.random.default_rng(1000 + rep)
            T_meas = T_true + rng.normal(0.0, noise_sd, size=T_true.shape)
            sat_mask = T_meas >= Tmax
            T_obs = np.minimum(T_meas, Tmax)
            print(f"\nnoise_sd={noise_sd:g}, rep={rep}, actual saturated={sat_mask.mean():.2%}")

            # For the censored likelihood, use the known simulated noise level.
            # When noise_sd=0, use a small nominal value to avoid division by zero.
            likelihood_noise_sd = max(noise_sd, 1.0)

            for method in METHODS:
                print(f"  fitting {method}")
                fit = fit_model(
                    method,
                    X,
                    Y,
                    T_obs,
                    sat_mask,
                    Tmax,
                    noise_sd=likelihood_noise_sd,
                    hinge_lambda=1.0,
                    n_starts=5,
                    seed=200 + rep,
                )
                metrics = compute_metrics(fit["params_hat"], true_params, X, Y, T_true, sat_mask)
                rows.append(
                    {
                        "noise_sd": noise_sd,
                        "rep": rep,
                        "target_frac_saturated_clean": frac_saturated,
                        "actual_frac_saturated": float(sat_mask.mean()),
                        "Tmax": Tmax,
                        "method": method,
                        "objective": fit["objective"],
                        "success": fit["success"],
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "noise_sweep_results.csv"
    df.to_csv(csv_path, index=False)

    plot_noise_results(df, "field_rel_l2", "Relative L2 field error", out_dir / "noise_sweep_field_error.png")
    plot_noise_results(df, "peak_rel_error", "Relative peak temperature error", out_dir / "noise_sweep_peak_error.png")
    plot_noise_results(df, "A_rel_error", "Relative amplitude parameter error", out_dir / "noise_sweep_amplitude_error.png")

    print(f"\nSaved {csv_path}")
    print(f"Saved noise sweep plots to {out_dir}")
    summary = df.groupby(["noise_sd", "method"])[["field_rel_l2", "peak_rel_error", "A_rel_error"]].mean().reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
