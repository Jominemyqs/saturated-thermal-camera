from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fit import fit_model
from src.gaussian_field import GaussianParams, censor_by_fraction, create_grid, gaussian_temperature
from src.metrics import compute_metrics


def plot_lambda_results(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    hinge = df[df["method"] == "hinge"].sort_values("hinge_lambda")
    ax.semilogx(
        hinge["hinge_lambda"],
        hinge[metric],
        color="#0072B2",
        marker="o",
        linewidth=1.8,
        label="hinge",
    )

    # Plot non-hinge methods as horizontal reference lines.
    reference_styles = {
        "exact": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
        "discard": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
        "censored": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    }
    for method in ["exact", "discard", "censored"]:
        row = df[df["method"] == method]
        if not row.empty:
            value = float(row.iloc[0][metric])
            style = reference_styles[method]
            ax.semilogx(
                hinge["hinge_lambda"],
                np.full(len(hinge), value),
                linewidth=1.8,
                markersize=6,
                markevery=[0, len(hinge) - 1],
                label=method,
                **style,
            )

    ax.set_xlabel(r"Hinge weight $\lambda$")
    ax.set_ylabel(ylabel)
    frac = float(df["actual_frac_saturated"].iloc[0])
    ax.set_title(f"{ylabel} vs hinge weight " + r"$\lambda$" + f" ({frac:.0%} saturated)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hinge-weight sensitivity at a chosen censoring level.")
    parser.add_argument(
        "--frac-saturated",
        type=float,
        default=0.10,
        help="Target fraction of hottest pixels to censor, e.g. 0.40 for 40%%.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix inserted into output filenames, e.g. 40pct.",
    )
    parser.add_argument("--n-starts", type=int, default=15, help="Number of optimizer starts per fit.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for optimizer starts.")
    return parser.parse_args()


def output_path(out_dir: Path, stem: str, suffix: str, extension: str) -> Path:
    if suffix:
        return out_dir / f"{stem}_{suffix}{extension}"
    return out_dir / f"{stem}{extension}"


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    X, Y = create_grid(nx=100, ny=80, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    true_params_obj = GaussianParams(T0=473.15, A=1100.0, xc=0.4, yc=-0.2, sx=1.25, sy=0.45)
    true_params = true_params_obj.as_array()
    T_true = gaussian_temperature(X, Y, true_params)

    frac_saturated = args.frac_saturated
    T_obs, sat_mask, Tmax = censor_by_fraction(T_true, frac_saturated)

    rows = []

    # Reference methods.
    for method in ["exact", "discard", "censored"]:
        print(f"Fitting reference method: {method}")
        fit = fit_model(
            method,
            X,
            Y,
            T_obs,
            sat_mask,
            Tmax,
            noise_sd=20.0,
            hinge_lambda=1.0,
            n_starts=args.n_starts,
            seed=args.seed,
        )
        metrics = compute_metrics(fit["params_hat"], true_params, X, Y, T_true, sat_mask)
        rows.append(
            {
                "method": method,
                "hinge_lambda": np.nan,
                "target_frac_saturated": frac_saturated,
                "actual_frac_saturated": float(sat_mask.mean()),
                "Tmax": Tmax,
                "objective": fit["objective"],
                "success": fit["success"],
                **metrics,
            }
        )

    # Hinge sensitivity.
    lambda_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    for lam in lambda_values:
        print(f"Fitting hinge loss with lambda={lam:g}")
        fit = fit_model(
            "hinge",
            X,
            Y,
            T_obs,
            sat_mask,
            Tmax,
            noise_sd=20.0,
            hinge_lambda=lam,
            n_starts=args.n_starts,
            seed=args.seed,
        )
        metrics = compute_metrics(fit["params_hat"], true_params, X, Y, T_true, sat_mask)
        rows.append(
            {
                "method": "hinge",
                "hinge_lambda": lam,
                "target_frac_saturated": frac_saturated,
                "actual_frac_saturated": float(sat_mask.mean()),
                "Tmax": Tmax,
                "objective": fit["objective"],
                "success": fit["success"],
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_path(out_dir, "lambda_sensitivity_results", args.output_suffix, ".csv")
    df.to_csv(csv_path, index=False)

    plot_lambda_results(
        df,
        "field_rel_l2",
        "Relative L2 field error",
        output_path(out_dir, "lambda_sensitivity_field_error", args.output_suffix, ".png"),
    )
    plot_lambda_results(
        df,
        "peak_rel_error",
        "Relative peak temperature error",
        output_path(out_dir, "lambda_sensitivity_peak_error", args.output_suffix, ".png"),
    )
    plot_lambda_results(
        df,
        "A_rel_error",
        "Relative amplitude parameter error",
        output_path(out_dir, "lambda_sensitivity_amplitude_error", args.output_suffix, ".png"),
    )

    print(f"Saved {csv_path}")
    print(f"Saved lambda sensitivity plots to {out_dir}")
    print(df[["method", "hinge_lambda", "field_rel_l2", "peak_rel_error", "A_rel_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
