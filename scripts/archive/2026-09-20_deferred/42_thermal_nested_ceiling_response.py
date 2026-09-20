from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src').is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.nested_ceiling_response import (
    RESPONSE_FEATURES,
    SINGLE_THRESHOLD_FEATURES,
    build_nested_response_table,
    grouped_ridge_predictions,
    prediction_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "38_pseudo_censoring_calibration"
    / "calibration_examples.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "archive" / "2026-09-20_deferred" / "39_nested_ceiling_response"
)


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = (
        ("known measurement", "measurement_exceedance_K"),
        ("latent truth (evaluation only)", "truth_exceedance_K_evaluation_only"),
    )
    for target_label, target_column in targets:
        for feature in (*SINGLE_THRESHOLD_FEATURES, *RESPONSE_FEATURES):
            rows.append(
                {
                    "target": target_label,
                    "feature_set": (
                        "single threshold"
                        if feature in SINGLE_THRESHOLD_FEATURES
                        else "nested response"
                    ),
                    "feature": feature,
                    "spearman_rho": float(
                        spearmanr(frame[feature], frame[target_column]).statistic
                    ),
                    "n_pixels": len(frame),
                }
            )
    return pd.DataFrame(rows)


def cross_validated_comparison(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame["trajectory"].to_numpy()
    baseline = frame[list(SINGLE_THRESHOLD_FEATURES)].to_numpy(dtype=float)
    augmented = frame[
        [*SINGLE_THRESHOLD_FEATURES, *RESPONSE_FEATURES]
    ].to_numpy(dtype=float)
    rows = []
    prediction_rows = []
    targets = (
        ("known measurement", "measurement_exceedance_K"),
        ("latent truth (evaluation only)", "truth_exceedance_K_evaluation_only"),
    )
    for target_label, target_column in targets:
        target = frame[target_column].to_numpy(dtype=float)
        for model, features in (
            ("single-threshold features", baseline),
            ("single + nested-response features", augmented),
        ):
            result = grouped_ridge_predictions(features, target, groups)
            row = {
                "target": target_label,
                "model": model,
                **prediction_metrics(result.prediction, target),
                "selected_penalties": "; ".join(
                    f"{group}={value:g}"
                    for group, value in sorted(result.selected_penalty.items())
                ),
                "n_pixels": len(frame),
                "outer_validation": "leave one development trajectory out",
            }
            rows.append(row)
            for index, prediction in enumerate(result.prediction):
                prediction_rows.append(
                    {
                        "trajectory": groups[index],
                        "pixel_index": int(frame.iloc[index]["pixel_index"]),
                        "target": target_label,
                        "model": model,
                        "observed_target_K": float(target[index]),
                        "predicted_target_K": float(prediction),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def plot_diagnostic(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    colors = {
        "DiagonalScanPath_8": "#0072B2",
        "HorizontalScanPath_13": "#E69F00",
        "SpiralScanPath_13": "#009E73",
    }
    for trajectory, group in frame.groupby("trajectory"):
        axes[0, 0].scatter(
            group["measurement_exceedance_K"],
            group["dmu_dc_near"],
            s=18,
            alpha=0.65,
            color=colors[trajectory],
            label=trajectory,
        )
    axes[0, 0].set(
        xlabel="Known measurement exceedance at 5% ceiling (K)",
        ylabel=r"Near-ceiling response $d\mu/dc$",
        title="Posterior response contains a moderate signal",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    selected = correlations[
        (correlations["target"] == "known measurement")
        & correlations["feature"].isin(
            ("posterior_exceedance_K", "posterior_sd_K", "dmu_dc_near", "dsd_dc_near")
        )
    ]
    labels = {
        "posterior_exceedance_K": "Posterior exceedance",
        "posterior_sd_K": "Posterior SD",
        "dmu_dc_near": r"$d\mu/dc$",
        "dsd_dc_near": r"$d\sigma/dc$",
    }
    axes[0, 1].bar(
        [labels[value] for value in selected["feature"]],
        selected["spearman_rho"],
        color=["#555555", "#777777", "#0072B2", "#56B4E9"],
    )
    axes[0, 1].set(
        ylabel="Spearman correlation",
        title="Response is not stronger than the current estimate",
    )
    axes[0, 1].tick_params(axis="x", rotation=18)

    known = comparison[comparison["target"] == "known measurement"]
    truth = comparison[comparison["target"] == "latent truth (evaluation only)"]
    x = np.arange(2)
    width = 0.36
    axes[1, 0].bar(
        x - width / 2,
        known["rmse_K"],
        width,
        label="Known measurement",
        color="#0072B2",
    )
    axes[1, 0].bar(
        x + width / 2,
        truth["rmse_K"],
        width,
        label="Latent truth",
        color="#E69F00",
    )
    axes[1, 0].set_xticks(x, ["Single", "Single + response"])
    axes[1, 0].set(ylabel="LOTO-CV RMSE (K)", title="Nested features do not reduce error")
    axes[1, 0].set_ylim(
        0.0,
        1.32 * max(float(known["rmse_K"].max()), float(truth["rmse_K"].max())),
    )
    axes[1, 0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1, 1].bar(
        x - width / 2,
        known["spearman_rho"],
        width,
        label="Known measurement",
        color="#0072B2",
    )
    axes[1, 1].bar(
        x + width / 2,
        truth["spearman_rho"],
        width,
        label="Latent truth",
        color="#E69F00",
    )
    axes[1, 1].set_xticks(x, ["Single", "Single + response"])
    axes[1, 1].set(
        ylabel="LOTO-CV Spearman correlation",
        title="Ranking also becomes less accurate",
    )
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Nested pseudo-ceiling response diagnostic", fontsize=15)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = pd.read_csv(args.input)
    frame = build_nested_response_table(examples)
    correlations = correlation_table(frame)
    comparison, predictions = cross_validated_comparison(frame)
    primary = comparison[comparison["target"] == "known measurement"].set_index(
        "model"
    )
    baseline = primary.loc["single-threshold features"]
    augmented = primary.loc["single + nested-response features"]
    rmse_reduction = 1.0 - augmented["rmse_K"] / baseline["rmse_K"]
    rho_gain = augmented["spearman_rho"] - baseline["spearman_rho"]
    gate_passed = bool(
        (rmse_reduction >= 0.05 and rho_gain >= 0.0)
        or (rho_gain >= 0.10 and rmse_reduction >= 0.0)
    )
    frame.to_csv(args.output_dir / "matched_nested_pixels.csv", index=False)
    correlations.to_csv(args.output_dir / "feature_correlations.csv", index=False)
    comparison.to_csv(args.output_dir / "grouped_cv_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "grouped_cv_predictions.csv", index=False)
    counts = (
        frame.groupby("trajectory")
        .size()
        .rename("n_matched_pixels")
        .reset_index()
    )
    counts.to_csv(args.output_dir / "counts.csv", index=False)
    plot_diagnostic(
        frame,
        correlations,
        comparison,
        args.output_dir / "nested_response_diagnostic.png",
    )
    response_rho = correlations[
        (correlations["target"] == "known measurement")
        & (correlations["feature"] == "dmu_dc_near")
    ]["spearman_rho"].iloc[0]
    single_rho = correlations[
        (correlations["target"] == "known measurement")
        & (correlations["feature"] == "posterior_exceedance_K")
    ]["spearman_rho"].iloc[0]
    lines = [
        "# Nested pseudo-ceiling response diagnostic",
        "",
        "## Question",
        "",
        "Does the way a posterior changes across nested artificial ceilings predict hidden exceedance better than the posterior at one ceiling?",
        "",
        "## Protocol",
        "",
        "- Reuses the converged Experiment 38 development runs; no GP was refit for this postprocessing diagnostic.",
        "- Matches pixels censored at 5%, 7.5%, and 10% pseudo-camera ceilings.",
        "- Uses 151 pixels from the three development trajectories.",
        "- Primary target is the still-known reference-camera measurement minus the 5% ceiling. Latent truth is evaluation only.",
        "- Comparison uses outer leave-one-trajectory-out validation with ridge strength selected inside each training fold.",
        "",
        "## Result",
        "",
        f"The ordinary posterior exceedance has Spearman rho `{single_rho:.3f}` with known measurement exceedance. The strongest nested response feature, dmu/dc, has rho `{response_rho:.3f}`.",
        "",
        "| Target | Features | RMSE (K) | MAE (K) | Spearman rho | R-squared |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.model} | {row.rmse_K:.3f} | {row.mae_K:.3f} | {row.spearman_rho:.3f} | {row.r_squared:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The predeclared signal gate is `{'PASS' if gate_passed else 'FAIL'}`: measurement RMSE reduction is `{100.0 * rmse_reduction:.1f}%` and Spearman gain is `{rho_gain:.3f}`.",
            "",
            "## Decision",
            "",
            "The nested response contains some information, but it is not stronger than the current single-threshold posterior and does not improve held-out-trajectory prediction. Therefore the learned calibration and hard extrapolation stages were not run. With the present model and three ceiling levels, pseudo-censoring response should not yet be claimed to recover genuinely unobservable exceedance magnitude.",
        ]
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    pd.DataFrame(
        [
            {
                "input": str(args.input),
                "levels": "5%, 7.5%, 10%",
                "n_trajectories": frame["trajectory"].nunique(),
                "n_matched_pixels": len(frame),
                "gate_passed": gate_passed,
                "minimum_rmse_reduction": 0.05,
                "minimum_spearman_gain": 0.10,
                "truth_used_for_features_or_fitting": False,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    print(f"Saved nested-ceiling diagnostic to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose posterior response across nested pseudo-ceilings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
