from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.fit import fit_model
from src.gaussian_field import censor_by_fraction, create_grid, gaussian_temperature
from src.losses import METHODS
from src.metrics import compute_field_metrics
from src.plotting import save_field_triplet, save_reconstruction_diagnostics
from src.synthetic_fields import FIELD_LABELS, FIELD_NAMES, make_truth_field

import matplotlib.pyplot as plt


def plot_grouped_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    field_names = list(FIELD_NAMES)
    method_names = list(METHODS)
    x = np.arange(len(field_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for idx, method in enumerate(method_names):
        values = []
        for field in field_names:
            row = df[(df["field"] == field) & (df["method"] == method)]
            values.append(float(row.iloc[0][metric]))
        offset = (idx - (len(method_names) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_LABELS[name] for name in field_names], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " under model misspecification")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    X, Y = create_grid(nx=80, ny=60, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    frac_saturated = 0.10
    rows = []

    for field_index, field_name in enumerate(FIELD_NAMES):
        label = FIELD_LABELS[field_name]
        T_true = make_truth_field(field_name, X, Y)
        T_obs, sat_mask, Tmax = censor_by_fraction(T_true, frac_saturated)
        print(f"\nField: {label}, saturated={sat_mask.mean():.2%}, Tmax={Tmax:.2f}")

        save_field_triplet(
            X,
            Y,
            T_true,
            T_obs,
            sat_mask,
            out_dir / f"misspecification_truth_{field_name}.png",
            title_suffix=f"{label}, top {100 * frac_saturated:.0f}% censored",
        )

        reconstructions = {}
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
                n_starts=20,
                seed=300 + field_index,
            )
            params_hat = fit["params_hat"]
            reconstructions[method] = gaussian_temperature(X, Y, params_hat)
            metrics = compute_field_metrics(params_hat, X, Y, T_true, sat_mask)
            rows.append(
                {
                    "field": field_name,
                    "field_label": label,
                    "method": method,
                    "target_frac_saturated": frac_saturated,
                    "actual_frac_saturated": float(sat_mask.mean()),
                    "Tmax": Tmax,
                    "objective": fit["objective"],
                    "success": fit["success"],
                    "T0_hat": float(params_hat[0]),
                    "A_hat": float(params_hat[1]),
                    "xc_hat": float(params_hat[2]),
                    "yc_hat": float(params_hat[3]),
                    "sx_hat": float(params_hat[4]),
                    "sy_hat": float(params_hat[5]),
                    **metrics,
                }
            )

        save_reconstruction_diagnostics(
            X,
            Y,
            T_true,
            T_obs,
            sat_mask,
            reconstructions,
            out_dir / f"misspecification_reconstructions_{field_name}.png",
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "misspecification_results.csv"
    df.to_csv(csv_path, index=False)

    plot_grouped_metric(df, "field_rel_l2", "Relative L2 field error", out_dir / "misspecification_field_error.png")
    plot_grouped_metric(df, "peak_rel_error", "Relative peak temperature error", out_dir / "misspecification_peak_error.png")
    plot_grouped_metric(
        df,
        "sat_region_rel_l2",
        "Relative L2 error in saturated region",
        out_dir / "misspecification_saturated_region_error.png",
    )

    print(f"\nSaved {csv_path}")
    print(f"Saved misspecification plots to {out_dir}")
    summary_cols = ["field_label", "method", "field_rel_l2", "peak_rel_error", "sat_region_rel_l2"]
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
