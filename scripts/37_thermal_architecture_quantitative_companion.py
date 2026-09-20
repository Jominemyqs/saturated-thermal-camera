from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    paired_camera_observations,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "30_final_architecture_comparison"
)
ARCHITECTURE_INPUT = DEFAULT_OUTPUT_DIR / "architecture_comparison.csv"

CAMERA = "Censored camera observation (reference)"
CAMERA_LABEL = "Censored camera\n(reference)"

DISPLAY_LABELS = {
    CAMERA: CAMERA_LABEL,
    "Posterior physics mean + RBF": "Posterior mean\n+ RBF",
    "Advective posterior physics mean + RBF": "Advective posterior mean\n+ RBF",
    "Legacy latent-clipped mean + joint advective ST": (
        "Legacy clipped mean\n+ joint advective ST"
    ),
    "Posterior physics mean + sequential ST (moment-matched)": (
        "Posterior mean\n+ sequential ST"
    ),
    "Posterior physics mean + sequential advective ST (moment-matched)": (
        "Posterior mean\n+ sequential advective ST"
    ),
    "Advective posterior mean + sequential advective ST (moment-matched)": (
        "Advective posterior mean\n+ sequential advective ST"
    ),
    "Posterior-sample mixture sequential advective ST": (
        "Posterior-sample mixture\n+ sequential advective ST"
    ),
}

SHORT_LABELS = {
    method: label.replace("\n", " ") for method, label in DISPLAY_LABELS.items()
}

METHOD_COLORS = {
    CAMERA: "#8A8A8A",
    "Posterior physics mean + RBF": "#E69F00",
    "Advective posterior physics mean + RBF": "#B9770E",
    "Legacy latent-clipped mean + joint advective ST": "#8E5EA2",
    "Posterior physics mean + sequential ST (moment-matched)": "#56B4E9",
    "Posterior physics mean + sequential advective ST (moment-matched)": "#0072B2",
    "Advective posterior mean + sequential advective ST (moment-matched)": (
        "#009E73"
    ),
    "Posterior-sample mixture sequential advective ST": "#CC79A7",
}

POINT_METRICS = [
    "field_excess_rel_l2",
    "rmse_K",
    "top_1pct_rmse_K",
    "peak_absolute_error_K",
]
PROBABILISTIC_METRICS = ["overall_crps_K", "top_1pct_crps_K"]
ERROR_METRICS = POINT_METRICS + PROBABILISTIC_METRICS


def camera_reference_row(
    dataset_dir: Path,
    *,
    nx: int,
    ny: int,
    fraction_saturated: float,
    observation_stride: int,
    measurement_noise_sd: float,
    previous_frame_offset: int,
    heat_flux_cutoff: float,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    catalog = list(trajectory_catalog(dataset_dir))
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    catalog_lookup = {record.name: index for index, record in enumerate(catalog)}
    held_out = [
        record for record in catalog if record.name not in DEVELOPMENT_TRAJECTORIES
    ]
    if len(held_out) != 30:
        raise AssertionError(f"Expected 30 held-out trajectories, found {len(held_out)}")

    rows = []
    for record in held_out:
        catalog_index = catalog_lookup[record.name]
        prepared, _ = prepare_trajectory(
            dataset_dir,
            record.name,
            nx=nx,
            ny=ny,
            heat_flux_cutoff=heat_flux_cutoff,
        )
        current_index = len(prepared.times) - 1
        previous_index = current_index - previous_frame_offset
        threshold = float(
            np.quantile(prepared.truth, 1.0 - fraction_saturated)
        )
        camera = paired_camera_observations(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            threshold=threshold,
            observation_stride=observation_stride,
            noise_sd=measurement_noise_sd,
            seed=seed + 10_000 * catalog_index,
        )
        observed = np.asarray(camera["current"]["clipped_full"], dtype=float)
        truth = np.asarray(prepared.truth, dtype=float)
        error = observed - truth
        top = truth >= float(np.quantile(truth, 0.99))
        rows.append(
            {
                "trajectory": record.name,
                "family": record.family,
                "field_excess_rel_l2": float(
                    np.linalg.norm(error.ravel())
                    / np.linalg.norm((truth - prepared.ambient).ravel())
                ),
                "rmse_K": float(np.sqrt(np.mean(error**2))),
                "top_1pct_rmse_K": float(np.sqrt(np.mean(error[top] ** 2))),
                "peak_absolute_error_K": abs(
                    float(np.max(observed) - np.max(truth))
                ),
                "top_1pct_signed_error_K": float(np.mean(error[top])),
            }
        )
    trajectory_rows = pd.DataFrame(rows)
    camera_row = {
        "method": CAMERA,
        **{
            metric: float(trajectory_rows[metric].mean())
            for metric in POINT_METRICS
        },
        "top_1pct_signed_error_K": float(
            trajectory_rows["top_1pct_signed_error_K"].mean()
        ),
        "overall_crps_K": np.nan,
        "top_1pct_crps_K": np.nan,
        "top_1pct_coverage_95": np.nan,
        "top_1pct_interval_width_95_K": np.nan,
        "row_type": "observed-field reference; not a predictive distribution",
        "heldout_trajectory_count": len(trajectory_rows),
    }
    return camera_row, trajectory_rows


def build_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    architectures = pd.read_csv(args.architecture_input)
    if len(architectures) != 7:
        raise AssertionError(
            f"Expected seven distinct GP architectures, found {len(architectures)}"
        )
    camera_row, camera_trajectories = camera_reference_row(
        args.dataset_dir,
        nx=args.nx,
        ny=args.ny,
        fraction_saturated=args.fraction_saturated,
        observation_stride=args.observation_stride,
        measurement_noise_sd=args.measurement_noise_sd,
        previous_frame_offset=args.previous_frame_offset,
        heat_flux_cutoff=args.heat_flux_cutoff,
        seed=args.seed,
    )
    architectures["row_type"] = "GP reconstruction"
    selected_columns = [
        "method",
        *POINT_METRICS,
        "top_1pct_signed_error_K",
        *PROBABILISTIC_METRICS,
        "top_1pct_coverage_95",
        "top_1pct_interval_width_95_K",
        "row_type",
        "heldout_trajectory_count",
    ]
    comparison = pd.concat(
        [pd.DataFrame([camera_row]), architectures[selected_columns]],
        ignore_index=True,
    )
    comparison["display_label"] = comparison["method"].map(DISPLAY_LABELS)
    if comparison["display_label"].isna().any():
        missing = comparison.loc[comparison["display_label"].isna(), "method"]
        raise ValueError(f"Missing display labels for {missing.tolist()}")
    return comparison, camera_trajectories


def format_value(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def format_kelvin(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f} K"


def write_markdown(output_path: Path, comparison: pd.DataFrame) -> None:
    lines = [
        "# Quantitative companion to the reconstruction comparison",
        "",
        "Values are equal-weight means over the 30 frozen held-out trajectories. "
        "The censored camera row is the observed-field reference shown in the "
        "reconstruction panel; it is not a probabilistic reconstruction, so CRPS, "
        "coverage, and interval width are not assigned.",
        "",
        "| Output | Field L2 | RMSE | Top-1% RMSE | Peak error | Overall CRPS | "
        "Top-1% CRPS | Top-1% coverage / width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        interval = "--"
        if np.isfinite(row["top_1pct_coverage_95"]):
            interval = (
                f"{row['top_1pct_coverage_95']:.3f} / "
                f"{row['top_1pct_interval_width_95_K']:.2f} K"
            )
        lines.append(
            f"| {SHORT_LABELS[row['method']]} | "
            f"{format_value(row['field_excess_rel_l2'])} | "
            f"{format_kelvin(row['rmse_K'])} | "
            f"{format_kelvin(row['top_1pct_rmse_K'])} | "
            f"{format_kelvin(row['peak_absolute_error_K'])} | "
            f"{format_kelvin(row['overall_crps_K'])} | "
            f"{format_kelvin(row['top_1pct_crps_K'])} | {interval} |"
        )
    lines.extend(
        [
            "",
            "Lower is better for all error and CRPS columns. Coverage is not ranked "
            "alone and must be read together with interval width. The RBF rows use "
            "their frozen larger covariance amplitude, as in the original comparison.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def plot_table(comparison: pd.DataFrame, output_path: Path) -> None:
    headers = [
        "Compared output",
        "Field rel.\nL2",
        "Overall\nRMSE (K)",
        "Top-1%\nRMSE (K)",
        "Peak\nerror (K)",
        "Overall\nCRPS (K)",
        "Top-1%\nCRPS (K)",
        "Top-1% 95%\ncoverage / width",
    ]
    body = []
    for _, row in comparison.iterrows():
        interval = "--"
        if np.isfinite(row["top_1pct_coverage_95"]):
            interval = (
                f"{row['top_1pct_coverage_95']:.3f} / "
                f"{row['top_1pct_interval_width_95_K']:.2f} K"
            )
        body.append(
            [
                row["display_label"],
                format_value(row["field_excess_rel_l2"]),
                format_value(row["rmse_K"]),
                format_value(row["top_1pct_rmse_K"]),
                format_value(row["peak_absolute_error_K"]),
                format_value(row["overall_crps_K"]),
                format_value(row["top_1pct_crps_K"]),
                interval,
            ]
        )

    figure, axis = plt.subplots(figsize=(16.0, 6.8), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=body,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.29, 0.085, 0.09, 0.09, 0.09, 0.09, 0.09, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.25)

    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor("#243447")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")
    for row_index, (_, row) in enumerate(comparison.iterrows(), start=1):
        base_color = "#F0F0F0" if row["method"] == CAMERA else "white"
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor(base_color)
            cell.set_edgecolor("#D7D7D7")
        label_cell = table[(row_index, 0)]
        label_cell.get_text().set_ha("left")
        label_cell.get_text().set_color(METHOD_COLORS[row["method"]])
        label_cell.get_text().set_weight("bold")

    gp_rows = comparison[comparison["row_type"] == "GP reconstruction"]
    column_lookup = {
        "field_excess_rel_l2": 1,
        "rmse_K": 2,
        "top_1pct_rmse_K": 3,
        "peak_absolute_error_K": 4,
        "overall_crps_K": 5,
        "top_1pct_crps_K": 6,
    }
    for metric, column in column_lookup.items():
        ranked = gp_rows[metric].sort_values()
        for rank, (source_index, _) in enumerate(ranked.iloc[:2].items()):
            table_row = int(source_index) + 1
            cell = table[(table_row, column)]
            cell.set_facecolor("#D9EEDB" if rank == 0 else "#E7F2F8")
            if rank == 0:
                cell.get_text().set_weight("bold")

    figure.suptitle(
        "Quantitative companion to the representative reconstruction panel",
        fontsize=17,
        fontweight="bold",
    )
    axis.text(
        0.0,
        0.02,
        "30 held-out trajectory mean; lower is better for errors and CRPS. "
        "Green marks the best GP architecture, blue the second best. "
        "Coverage must be read with width.",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#444444",
        va="bottom",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_bars(comparison: pd.DataFrame, output_path: Path) -> None:
    panels = [
        ("field_excess_rel_l2", "Relative excess-field L2", True),
        ("top_1pct_rmse_K", "Top-1% RMSE (K)", True),
        ("overall_crps_K", "Overall CRPS (K)", False),
        ("top_1pct_crps_K", "Top-1% CRPS (K)", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)
    for axis, (metric, title, include_camera) in zip(axes.ravel(), panels):
        selected = comparison.copy()
        if not include_camera:
            selected = selected[selected[metric].notna()]
        selected = selected.sort_values(metric, ascending=True)
        y = np.arange(len(selected))
        colors = [METHOD_COLORS[method] for method in selected["method"]]
        axis.barh(y, selected[metric], color=colors, alpha=0.9, height=0.7)
        axis.set_yticks(y, [SHORT_LABELS[method] for method in selected["method"]])
        axis.invert_yaxis()
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        maximum = float(selected[metric].max())
        axis.set_xlim(0.0, maximum * 1.18)
        for position, value in zip(y, selected[metric]):
            axis.text(
                value + 0.015 * maximum,
                position,
                f"{value:.3f}",
                va="center",
                fontsize=8.5,
            )
    figure.suptitle(
        "Eight displayed outputs: point reconstruction and predictive quality",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison, camera_trajectories = build_comparison(args)
    comparison.to_csv(args.output_dir / "quantitative_comparison.csv", index=False)
    camera_trajectories.to_csv(
        args.output_dir / "censored_camera_reference_by_trajectory.csv", index=False
    )
    write_markdown(
        args.output_dir / "quantitative_comparison.md",
        comparison,
    )
    plot_table(
        comparison,
        args.output_dir / "quantitative_comparison_table.png",
    )
    plot_bars(
        comparison,
        args.output_dir / "quantitative_comparison_bars.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a quantitative companion to experiment 30 reconstructions."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--architecture-input", type=Path, default=ARCHITECTURE_INPUT)
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--measurement-noise-sd", type=float, default=0.25)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
