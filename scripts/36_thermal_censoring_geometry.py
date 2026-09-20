from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.censored_gp import RBFConfig
from src.censoring_geometry import (
    global_to_local_edges,
    local_ordering_edges,
    observed_mask_geometry,
    randomize_edge_orientations,
)
from src.gaussian_field import create_grid
from src.metrics import empirical_crps
from src.preference_censored_gp import sample_preference_gp_multiple_chains
from src.synthetic_fields import FIELD_LABELS, FIELD_NAMES, make_truth_field
from src.thermal_plotting import add_tail_contours
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    paired_camera_observations,
    prepare_trajectory,
    previous_posterior_observations,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "by_experiment" / "34_censoring_geometry"
STAGE1_INPUT = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "33_single_frame_exceedance"
    / "censored_pixel_diagnostics.csv"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "33_single_frame_exceedance"
    / "fixed_configuration.csv"
)

CURRENT_POSTERIOR = "coherent latent single-frame posterior"
STANDARD = "standard censored GP"
RANDOM = "matched random local preferences"
GEOMETRY = "observed-mask geometry preferences"
METHODS = (STANDARD, RANDOM, GEOMETRY)
COLORS = {
    STANDARD: "#777777",
    RANDOM: "#CC79A7",
    GEOMETRY: "#0072B2",
}
REPRESENTATIVES = (
    "DiagonalScanPath_7",
    "HorizontalScanPath_10",
    "SpiralScanPath_12",
)
LOWER_IS_BETTER = {
    "field_excess_rel_l2",
    "overall_rmse_K",
    "overall_mae_K",
    "overall_crps_K",
    "saturated_rmse_K",
    "saturated_mae_K",
    "saturated_crps_K",
    "top_1pct_rmse_K",
    "top_1pct_crps_K",
    "peak_absolute_error_K",
}


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).reshape(-1)
    y = np.asarray(second, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def mask_from_pixel_indices(indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    return mask.reshape(shape)


def geometry_rows_for_trajectory(
    trajectory_frame: pd.DataFrame,
    *,
    shape: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = mask_from_pixel_indices(trajectory_frame["pixel_index"], shape)
    labels, depth, components = observed_mask_geometry(mask)
    geometry_edges = local_ordering_edges(mask, labels, depth)
    edge_component = labels.ravel()[geometry_edges[:, 0]] if len(geometry_edges) else np.array([], dtype=int)
    frame = trajectory_frame.copy()
    indices = frame["pixel_index"].to_numpy(int)
    frame["observed_component_id"] = labels.ravel()[indices]
    frame["boundary_depth_pixels"] = depth.ravel()[indices]
    size_lookup = {row["component_id"]: row["component_size"] for row in components}
    frame["component_size"] = frame["observed_component_id"].map(size_lookup)

    component_rows = []
    for component in components:
        component_id = component["component_id"]
        selected = frame[frame["observed_component_id"] == component_id]
        informative = (
            len(selected) >= 3
            and selected["boundary_depth_pixels"].nunique() >= 2
        )
        component_rows.append(
            {
                "trajectory": selected["trajectory"].iloc[0],
                "family": selected["family"].iloc[0],
                **component,
                "n_boundary_pixels": int(
                    np.sum(selected["boundary_depth_pixels"] == 0.0)
                ),
                "n_local_ordering_edges": int(np.sum(edge_component == component_id)),
                "informative_for_rank": informative,
                "rho_depth_true_temperature": safe_spearman(
                    selected["boundary_depth_pixels"], selected["truth_K"]
                )
                if informative
                else np.nan,
                "rho_depth_current_gp_mean": safe_spearman(
                    selected["boundary_depth_pixels"],
                    selected["posterior_mean_K"],
                )
                if informative
                else np.nan,
            }
        )
    edge_rows = pd.DataFrame(
        {
            "trajectory": frame["trajectory"].iloc[0],
            "shallow_pixel_index": geometry_edges[:, 0]
            if len(geometry_edges)
            else np.array([], dtype=int),
            "deep_pixel_index": geometry_edges[:, 1]
            if len(geometry_edges)
            else np.array([], dtype=int),
        }
    )
    return frame, pd.DataFrame(component_rows), edge_rows


def run_stage1(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not STAGE1_INPUT.is_file():
        raise FileNotFoundError("Run scripts/35_thermal_single_frame_exceedance.py first")
    source = pd.read_csv(STAGE1_INPUT)
    source = source[source["representation"] == CURRENT_POSTERIOR].copy()
    if source["trajectory"].nunique() != 30:
        raise AssertionError("Stage 1 requires the frozen 30 held-out trajectories")
    pixel_frames = []
    component_frames = []
    edge_frames = []
    for _, trajectory_frame in source.groupby("trajectory", sort=False):
        pixels, components, edges = geometry_rows_for_trajectory(
            trajectory_frame, shape=(args.ny, args.nx)
        )
        pixel_frames.append(pixels)
        component_frames.append(components)
        edge_frames.append(edges)
    pixels = pd.concat(pixel_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    pooled = summarize_stage1_signal(pixels, components, edges, args)
    return pixels, components, edges, pooled


def summarize_stage1_signal(
    pixels: pd.DataFrame,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    informative = components[components["informative_for_rank"]]
    rho_true = safe_spearman(pixels["boundary_depth_pixels"], pixels["delta_true_K"])
    rho_hat = safe_spearman(pixels["boundary_depth_pixels"], pixels["delta_hat_K"])
    rho_gap = rho_true - rho_hat
    median_component_rho = float(informative["rho_depth_true_temperature"].median())
    latent_above = pixels[pixels["delta_true_K"] > 0.0]
    exceedance_slope = float(
        np.polyfit(
            latent_above["delta_true_K"],
            latent_above["delta_hat_K"],
            deg=1,
        )[0]
    )
    stage2_warranted = bool(
        rho_gap >= args.minimum_rho_gap
        and median_component_rho >= args.minimum_component_rho
        and len(informative) >= args.minimum_informative_components
    )
    return pd.DataFrame(
        [
            {
                "n_trajectories": pixels["trajectory"].nunique(),
                "n_observed_censored_pixels": len(pixels),
                "n_connected_components": len(components),
                "n_informative_components": len(informative),
                "median_component_size": components["component_size"].median(),
                "q75_component_size": components["component_size"].quantile(0.75),
                "maximum_component_size": components["component_size"].max(),
                "n_local_ordering_edges": len(edges),
                "rho_depth_true_exceedance": rho_true,
                "rho_depth_current_gp_exceedance": rho_hat,
                "rho_gap": rho_gap,
                "rho_true_vs_gp_exceedance_latent_above": safe_spearman(
                    latent_above["delta_true_K"], latent_above["delta_hat_K"]
                ),
                "gp_vs_true_exceedance_slope_latent_above": exceedance_slope,
                "mean_true_exceedance_latent_above_K": latent_above[
                    "delta_true_K"
                ].mean(),
                "mean_gp_exceedance_latent_above_K": latent_above[
                    "delta_hat_K"
                ].mean(),
                "median_within_component_rho_depth_true_temperature": median_component_rho,
                "median_within_component_rho_depth_current_gp": informative[
                    "rho_depth_current_gp_mean"
                ].median(),
                "q25_within_component_rho": informative[
                    "rho_depth_true_temperature"
                ].quantile(0.25),
                "q75_within_component_rho": informative[
                    "rho_depth_true_temperature"
                ].quantile(0.75),
                "fraction_informative_component_rho_positive": np.mean(
                    informative["rho_depth_true_temperature"] > 0.0
                ),
                "fraction_current_gp_component_rho_positive": np.mean(
                    informative["rho_depth_current_gp_mean"] > 0.0
                ),
                "stage2_warranted": stage2_warranted,
                "minimum_rho_gap": args.minimum_rho_gap,
                "minimum_component_rho": args.minimum_component_rho,
                "minimum_informative_components": args.minimum_informative_components,
                "geometry_source": "observed noisy censoring mask only",
                "gp_posterior_source": pixels["gp_posterior_source"].iloc[0]
                if "gp_posterior_source" in pixels
                else "retained short-chain experiment 33 posterior",
            }
        ]
    )


def replace_stage1_posterior(
    pixels: pd.DataFrame,
    fresh_posterior: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace the legacy short-chain GP columns with converged Stage-2 values."""
    updated = pixels.merge(
        fresh_posterior,
        on=["trajectory", "pixel_index"],
        how="left",
        validate="one_to_one",
        suffixes=("_legacy", ""),
    )
    if updated["posterior_mean_K"].isna().any():
        raise AssertionError("Fresh standard posterior is missing Stage-1 pixels")
    updated["delta_hat_K"] = updated["posterior_mean_K"] - updated["threshold_K"]
    updated["signed_error_K"] = updated["posterior_mean_K"] - updated["truth_K"]
    updated["absolute_error_K"] = np.abs(updated["signed_error_K"])
    updated["abs_error_over_sd"] = updated["absolute_error_K"] / np.maximum(
        updated["posterior_sd_K"], 1e-12
    )
    updated["covered_95"] = (
        (updated["lower_95_K"] <= updated["truth_K"])
        & (updated["truth_K"] <= updated["upper_95_K"])
    )
    updated["gp_posterior_source"] = "pooled converged latent-ESS standard posterior"

    pixel_frames = []
    component_frames = []
    edge_frames = []
    for _, trajectory_frame in updated.groupby("trajectory", sort=False):
        new_pixels, new_components, new_edges = geometry_rows_for_trajectory(
            trajectory_frame, shape=(args.ny, args.nx)
        )
        pixel_frames.append(new_pixels)
        component_frames.append(new_components)
        edge_frames.append(new_edges)
    final_pixels = pd.concat(pixel_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    summary = summarize_stage1_signal(final_pixels, components, edges, args)
    return final_pixels, components, edges, summary


def plot_stage1_depth_signal(
    pixels: pd.DataFrame,
    pooled: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = [
        ("delta_true_K", r"true exceedance $\Delta^{true}$ (K)", "#D55E00"),
        ("delta_hat_K", r"current GP exceedance $\hat\Delta$ (K)", "#0072B2"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    for axis, (metric, ylabel, color) in zip(axes, metrics):
        grouped = pixels.groupby("boundary_depth_pixels")[metric]
        summary = grouped.agg(
            mean="mean",
            median="median",
            q10=lambda values: values.quantile(0.10),
            q90=lambda values: values.quantile(0.90),
            count="size",
        ).reset_index()
        x = summary["boundary_depth_pixels"].to_numpy(float)
        axis.fill_between(
            x,
            summary["q10"],
            summary["q90"],
            color=color,
            alpha=0.17,
            linewidth=0.0,
            label="10th-90th percentile",
        )
        axis.plot(
            x,
            summary["mean"],
            color=color,
            marker="o",
            linewidth=2.5,
            label="pooled mean",
        )
        axis.plot(
            x,
            summary["median"],
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="pooled median",
        )
        rho_column = (
            "rho_depth_true_exceedance"
            if metric == "delta_true_K"
            else "rho_depth_current_gp_exceedance"
        )
        axis.text(
            0.96,
            0.92,
            rf"pooled $\rho={pooled.iloc[0][rho_column]:.3f}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
        )
        for _, row in summary.iterrows():
            axis.annotate(
                f"n={int(row['count'])}",
                (row["boundary_depth_pixels"], row["q90"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#555555",
            )
        axis.set_xlabel("distance from observed component boundary (pixels)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Observed-mask depth vs hidden magnitude", fontweight="bold")
    axes[1].set_title("Observed-mask depth vs current GP", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Observed-mask depth contains ordering information already reflected by the GP"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_component_correlations(components: pd.DataFrame, output_path: Path) -> None:
    selected = components[components["informative_for_rank"]]
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 17)
    axis.hist(
        selected["rho_depth_true_temperature"],
        bins=bins,
        color="#D55E00",
        alpha=0.65,
        label="true temperature",
    )
    axis.hist(
        selected["rho_depth_current_gp_mean"],
        bins=bins,
        histtype="step",
        linewidth=2.2,
        color="#0072B2",
        label="current GP mean",
    )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xlabel("within-component Spearman correlation with boundary depth")
    axis.set_ylabel("component count")
    axis.set_title(
        f"Informative connected components (n={len(selected)})",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def summarize_mask_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["fraction_saturated", "noise_sd_K"], sort=True)
        .agg(
            n_trajectories=("trajectory", "nunique"),
            mean_rho_depth_true_exceedance=(
                "rho_depth_true_exceedance",
                "mean",
            ),
            median_rho_depth_true_exceedance=(
                "rho_depth_true_exceedance",
                "median",
            ),
            mean_observed_censored_pixels=("n_observed_censored", "mean"),
            mean_connected_components=("n_components", "mean"),
            mean_informative_components=("n_informative_components", "mean"),
        )
        .reset_index()
    )


def plot_mask_robustness(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    for noise_sd, selected in summary.groupby("noise_sd_K", sort=True):
        axis.plot(
            100.0 * selected["fraction_saturated"],
            selected["mean_rho_depth_true_exceedance"],
            marker="o",
            linewidth=2.0,
            label=f"noise SD = {noise_sd:g} K",
        )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel("synthetic censoring fraction (%)")
    axis.set_ylabel(r"mean trajectory $\rho(d,\Delta^{true})$")
    axis.set_title("Observed-mask depth signal across censoring and noise")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def camera_for_trajectory(prepared, catalog_index: int, args: argparse.Namespace):
    current_index = len(prepared.times) - 1
    previous_index = current_index - args.previous_frame_offset
    threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
    camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=threshold,
        observation_stride=args.observation_stride,
        noise_sd=args.measurement_noise_sd,
        seed=args.seed + 10_000 * catalog_index,
    )
    return camera, threshold, previous_index


def full_grid_observations(prepared, camera, threshold: float):
    observations, _, saturated = previous_posterior_observations(
        prepared,
        frame=camera["frames"][0],
        fixed_mask=camera["fixed_observation_mask"],
        threshold=threshold,
    )
    time = float(camera["frames"][0]["time"])
    observations["x_pred"] = np.column_stack(
        [prepared.points, np.full(len(prepared.points), time)]
    )
    return observations, saturated


def posterior_metrics(
    *,
    truth: np.ndarray,
    ambient: float,
    observed_saturated: np.ndarray,
    prediction: tuple[np.ndarray, ...],
) -> dict[str, float]:
    mean, sd, lower, upper, draws = prediction
    target = np.asarray(truth, dtype=float).ravel()
    saturated = np.asarray(observed_saturated, dtype=bool).ravel()
    top = target >= float(np.quantile(target, 0.99))
    error = mean - target
    crps = empirical_crps(draws, target)
    denominator = np.linalg.norm(target - ambient)
    return {
        "field_excess_rel_l2": float(np.linalg.norm(error) / denominator),
        "overall_rmse_K": float(np.sqrt(np.mean(error**2))),
        "overall_mae_K": float(np.mean(np.abs(error))),
        "overall_crps_K": float(np.mean(crps)),
        "saturated_rmse_K": float(np.sqrt(np.mean(error[saturated] ** 2))),
        "saturated_mae_K": float(np.mean(np.abs(error[saturated]))),
        "saturated_bias_K": float(np.mean(error[saturated])),
        "saturated_crps_K": float(np.mean(crps[saturated])),
        "saturated_rank_spearman": safe_spearman(mean[saturated], target[saturated]),
        "top_1pct_rmse_K": float(np.sqrt(np.mean(error[top] ** 2))),
        "top_1pct_mae_K": float(np.mean(np.abs(error[top]))),
        "top_1pct_crps_K": float(np.mean(crps[top])),
        "top_1pct_coverage_95": float(
            np.mean((lower[top] <= target[top]) & (target[top] <= upper[top]))
        ),
        "top_1pct_interval_width_95_K": float(np.mean((upper - lower)[top])),
        "peak_bias_K": float(np.max(mean) - np.max(target)),
        "peak_absolute_error_K": abs(float(np.max(mean) - np.max(target))),
    }


def summarize_results(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = [column for column in frame.columns if column in LOWER_IS_BETTER or column in {
        "saturated_bias_K",
        "saturated_rank_spearman",
        "top_1pct_mae_K",
        "top_1pct_coverage_95",
        "top_1pct_interval_width_95_K",
        "peak_bias_K",
    }]
    result = frame.groupby(groups, sort=False)[metrics].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result = result.reset_index()
    counts = frame.groupby(groups, sort=False)["trajectory"].nunique().rename("n_trajectories")
    return result.merge(counts.reset_index(), on=groups, how="left")


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = ((GEOMETRY, STANDARD), (GEOMETRY, RANDOM), (RANDOM, STANDARD))
    metrics = [
        "field_excess_rel_l2",
        "overall_crps_K",
        "saturated_rmse_K",
        "saturated_mae_K",
        "top_1pct_crps_K",
        "peak_absolute_error_K",
        "saturated_rank_spearman",
    ]
    indexed = results.set_index(["trajectory", "method"])
    trajectories = results["trajectory"].unique()
    for candidate, reference in comparisons:
        for metric in metrics:
            candidate_values = np.array(
                [indexed.loc[(name, candidate), metric] for name in trajectories]
            )
            reference_values = np.array(
                [indexed.loc[(name, reference), metric] for name in trajectories]
            )
            change = candidate_values - reference_values
            lower_is_better = metric in LOWER_IS_BETTER
            wins = change < 0.0 if lower_is_better else change > 0.0
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "mean_change_candidate_minus_reference": np.mean(change),
                    "median_change_candidate_minus_reference": np.median(change),
                    "candidate_win_count": int(np.sum(wins)),
                    "n_trajectories": len(trajectories),
                    "better_direction": "lower" if lower_is_better else "higher",
                }
            )
    return pd.DataFrame(rows)


def run_mask_robustness(
    prepared,
    record,
    previous_index: int,
    catalog_index: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    truth = prepared.history[previous_index]
    for fraction in args.robustness_fractions:
        threshold = float(np.quantile(prepared.truth, 1.0 - fraction))
        for noise_sd in args.robustness_noise_sd:
            rng = np.random.default_rng(
                args.seed
                + 700_000
                + 10_000 * catalog_index
                + int(round(10_000 * fraction))
                + int(round(100 * noise_sd))
            )
            measured = truth + rng.normal(0.0, noise_sd, size=truth.shape)
            observed = measured >= threshold
            labels, depth, components = observed_mask_geometry(observed)
            if np.any(observed):
                rows.append(
                    {
                        "trajectory": record.name,
                        "family": record.family,
                        "fraction_saturated": fraction,
                        "noise_sd_K": noise_sd,
                        "n_observed_censored": int(np.sum(observed)),
                        "n_components": len(components),
                        "n_informative_components": int(
                            np.sum(
                                [row["maximum_depth_pixels"] > 0 for row in components]
                            )
                        ),
                        "rho_depth_true_exceedance": safe_spearman(
                            depth[observed], truth[observed] - threshold
                        ),
                    }
                )
    return rows


def run_main_stage2(
    args: argparse.Namespace,
    fixed: pd.Series,
    catalog: list,
    *,
    methods: tuple[str, ...] = METHODS,
    collect_mask_robustness: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    requested = set(args.trajectory_names or [])
    indexed = [
        (index, record)
        for index, record in enumerate(catalog)
        if record.name not in DEVELOPMENT_TRAJECTORIES
        and (not requested or record.name in requested)
    ]
    if requested - {record.name for _, record in indexed}:
        raise ValueError(f"Unknown held-out trajectories: {sorted(requested - {record.name for _, record in indexed})}")
    results = []
    sampler_rows = []
    robustness_rows = []
    representatives: dict[str, dict[str, object]] = {}
    stage1 = pd.read_csv(STAGE1_INPUT)
    stage1 = stage1[stage1["representation"] == CURRENT_POSTERIOR].set_index(
        ["trajectory", "pixel_index"]
    )
    baseline_rows = []
    standard_pixel_rows = []

    for run_index, (catalog_index, record) in enumerate(indexed):
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        camera, threshold, previous_index = camera_for_trajectory(
            prepared, catalog_index, args
        )
        observations, saturated = full_grid_observations(prepared, camera, threshold)
        labels, depth, components = observed_mask_geometry(saturated)
        geometry_edges_global = local_ordering_edges(saturated, labels, depth)
        random_edges_global = randomize_edge_orientations(
            geometry_edges_global,
            seed=args.seed + 300_000 + catalog_index,
        )
        saturated_global = np.flatnonzero(saturated.ravel())
        geometry_edges = global_to_local_edges(
            geometry_edges_global, saturated_global
        )
        random_edges = global_to_local_edges(random_edges_global, saturated_global)
        all_edge_sets = {
            STANDARD: np.empty((0, 2), dtype=int),
            RANDOM: random_edges,
            GEOMETRY: geometry_edges,
        }
        edge_sets = {method: all_edge_sets[method] for method in methods}
        config = RBFConfig(
            mean_temp=prepared.ambient,
            signal_sd=float(fixed.signal_sd),
            lengthscale=prepared.source_lengthscale * args.length_multiplier,
            noise_sd=args.previous_noise_sd,
        )
        tau = args.tau_noise_multiplier * config.noise_sd
        predictions = {}
        for method, pairs in edge_sets.items():
            prediction, diagnostics = sample_preference_gp_multiple_chains(
                observations,
                config,
                preference_pairs=pairs,
                tau=tau,
                n_chains=args.n_chains,
                samples_per_chain=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=args.seed + 50_000 * catalog_index,
            )
            predictions[method] = prediction
            truth = prepared.history[previous_index]
            results.append(
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "run_index": record.run_index,
                    "method": method,
                    "threshold_K": threshold,
                    "n_observed_censored": int(np.sum(saturated)),
                    "n_components": len(components),
                    "n_local_edges": len(geometry_edges),
                    "tau_K": tau,
                    **posterior_metrics(
                        truth=truth,
                        ambient=prepared.ambient,
                        observed_saturated=saturated,
                        prediction=prediction,
                    ),
                }
            )
            sampler_rows.append(
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "method": method,
                    **diagnostics,
                }
            )
        if STANDARD in predictions:
            saturated_indices = np.flatnonzero(saturated.ravel())
            standard_prediction = predictions[STANDARD]
            for pixel_index in saturated_indices:
                standard_pixel_rows.append(
                    {
                        "trajectory": record.name,
                        "pixel_index": int(pixel_index),
                        "posterior_mean_K": float(standard_prediction[0][pixel_index]),
                        "posterior_sd_K": float(standard_prediction[1][pixel_index]),
                        "lower_95_K": float(standard_prediction[2][pixel_index]),
                        "upper_95_K": float(standard_prediction[3][pixel_index]),
                    }
                )
            current_saved = stage1.loc[record.name]
            standard_mean = standard_prediction[0]
            baseline_rows.append(
                {
                    "trajectory": record.name,
                    "mean_absolute_difference_K": float(
                        np.mean(
                            np.abs(
                                standard_mean[current_saved.index.to_numpy(int)]
                                - current_saved["posterior_mean_K"].to_numpy(float)
                            )
                        )
                    ),
                    "rank_correlation": safe_spearman(
                        standard_mean[current_saved.index.to_numpy(int)],
                        current_saved["posterior_mean_K"],
                    ),
                    "comparison": (
                        "pooled converged latent-ESS standard versus retained "
                        "120-burn-in data-augmentation standard"
                    ),
                }
            )
        if collect_mask_robustness and not args.skip_mask_robustness:
            robustness_rows.extend(
                run_mask_robustness(
                    prepared, record, previous_index, catalog_index, args
                )
            )
        if record.name in REPRESENTATIVES and set(METHODS).issubset(predictions):
            representatives[record.name] = {
                "prepared": prepared,
                "truth": prepared.history[previous_index],
                "clipped": camera["frames"][0]["clipped_full"],
                "threshold": threshold,
                "predictions": predictions,
            }
        trajectory_results = results[-len(methods) :]
        metric_text = ", ".join(
            f"{row['method'].split()[0]} sat RMSE={row['saturated_rmse_K']:.3f}"
            for row in trajectory_results
        )
        print(
            f"[{run_index + 1:02d}/{len(indexed):02d}] {record.name}: "
            f"components={len(components)}, pairs={len(geometry_edges)}, {metric_text}",
            flush=True,
        )
        pd.DataFrame(results).to_csv(
            args.output_dir / "checkpoint_results.csv", index=False
        )
    return (
        pd.DataFrame(results),
        pd.DataFrame(sampler_rows),
        representatives,
        pd.DataFrame(robustness_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(standard_pixel_rows),
    )


def synthetic_observations(
    truth: np.ndarray,
    points: np.ndarray,
    *,
    fraction_saturated: float,
    noise_sd: float,
    observation_stride: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, float]:
    threshold = float(np.quantile(truth, 1.0 - fraction_saturated))
    rng = np.random.default_rng(seed)
    measured = truth + rng.normal(0.0, noise_sd, size=truth.shape)
    saturated = measured >= threshold
    fixed = np.zeros(truth.shape, dtype=bool)
    fixed[::observation_stride, ::observation_stride] = True
    included = fixed | saturated
    observations = {
        "x_obs": points[included.ravel()],
        "y_obs": np.minimum(measured[included], threshold),
        "sat_mask": saturated[included],
        "x_pred": points,
        "threshold": threshold,
    }
    return observations, saturated, threshold


def run_synthetic_robustness(args: argparse.Namespace) -> pd.DataFrame:
    x_grid, y_grid = create_grid(
        nx=args.synthetic_nx,
        ny=args.synthetic_ny,
        xlim=(-5.0, 5.0),
        ylim=(-3.0, 3.0),
    )
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    rows = []
    for field_index, field_name in enumerate(FIELD_NAMES):
        truth = make_truth_field(field_name, x_grid, y_grid)
        observations, saturated, threshold = synthetic_observations(
            truth,
            points,
            fraction_saturated=args.synthetic_fraction_saturated,
            noise_sd=args.synthetic_noise_sd,
            observation_stride=args.synthetic_observation_stride,
            seed=args.seed + 900_000 + field_index,
        )
        labels, depth, components = observed_mask_geometry(saturated)
        geometry_global = local_ordering_edges(saturated, labels, depth)
        random_global = randomize_edge_orientations(
            geometry_global, seed=args.seed + 910_000 + field_index
        )
        selected = np.flatnonzero(saturated.ravel())
        edge_sets = {
            STANDARD: np.empty((0, 2), dtype=int),
            RANDOM: global_to_local_edges(random_global, selected),
            GEOMETRY: global_to_local_edges(geometry_global, selected),
        }
        config = RBFConfig(
            mean_temp=473.15,
            signal_sd=args.synthetic_signal_sd,
            lengthscale=args.synthetic_lengthscale,
            noise_sd=args.synthetic_noise_sd,
        )
        tau = args.tau_noise_multiplier * args.synthetic_noise_sd
        for method, pairs in edge_sets.items():
            prediction, diagnostics = sample_preference_gp_multiple_chains(
                observations,
                config,
                preference_pairs=pairs,
                tau=tau,
                n_chains=args.synthetic_chains,
                samples_per_chain=args.synthetic_samples,
                burn_in=args.synthetic_burn_in,
                thin=args.synthetic_thin,
                seed=args.seed + 920_000 + field_index,
            )
            rows.append(
                {
                    "trajectory": field_name,
                    "field": field_name,
                    "field_label": FIELD_LABELS[field_name],
                    "method": method,
                    "threshold_K": threshold,
                    "n_observed_censored": int(np.sum(saturated)),
                    "n_components": len(components),
                    "n_local_edges": len(geometry_global),
                    "rho_depth_true_exceedance": safe_spearman(
                        depth[saturated], truth[saturated] - threshold
                    ),
                    "tau_K": tau,
                    **posterior_metrics(
                        truth=truth,
                        ambient=473.15,
                        observed_saturated=saturated,
                        prediction=prediction,
                    ),
                    **{
                        f"sampler_{key}": value
                        for key, value in diagnostics.items()
                    },
                }
            )
        print(
            f"[synthetic] {field_name}: components={len(components)}, "
            f"pairs={len(geometry_global)}",
            flush=True,
        )
    return pd.DataFrame(rows)


def plot_method_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("field_excess_rel_l2_mean", "Relative field error"),
        ("saturated_rmse_K_mean", "Observed-censored RMSE (K)"),
        ("top_1pct_crps_K_mean", "Top-1% CRPS (K)"),
        ("saturated_rank_spearman_mean", "Censored-region rank correlation"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    labels = ["standard", "random local", "geometry local"]
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        values = []
        errors = []
        for method in METHODS:
            row = summary[summary["method"] == method].iloc[0]
            values.append(row[metric])
            errors.append(row[metric.replace("_mean", "_std")] / np.sqrt(row["n_trajectories"]))
        axis.errorbar(
            x,
            values,
            yerr=errors,
            fmt="o-",
            linewidth=2.0,
            capsize=4,
            color="#333333",
            markerfacecolor="white",
            markersize=7,
        )
        for index, method in enumerate(METHODS):
            axis.plot(index, values[index], "o", color=COLORS[method], markersize=7)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Observed-mask local ordering: 30 held-out heat-equation trajectories"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_representative_reconstructions(
    representatives: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    if set(representatives) != set(REPRESENTATIVES):
        return
    figure, axes = plt.subplots(3, 5, figsize=(14.2, 8.6), constrained_layout=True)
    image = None
    for row_index, name in enumerate(REPRESENTATIVES):
        item = representatives[name]
        prepared = item["prepared"]
        truth = np.asarray(item["truth"])
        threshold = float(item["threshold"])
        norm = matplotlib.colors.PowerNorm(
            gamma=0.52,
            vmin=prepared.ambient + 0.15,
            vmax=max(float(np.quantile(truth, 0.999)), threshold + 1.0),
        )
        extent = [
            prepared.xs[0] * 1e3,
            prepared.xs[-1] * 1e3,
            prepared.ys[0] * 1e3,
            prepared.ys[-1] * 1e3,
        ]
        fields = [
            truth,
            np.asarray(item["clipped"]),
            item["predictions"][STANDARD][0].reshape(truth.shape),
            item["predictions"][RANDOM][0].reshape(truth.shape),
            item["predictions"][GEOMETRY][0].reshape(truth.shape),
        ]
        for column_index, field in enumerate(fields):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="magma",
                norm=norm,
                aspect="auto",
            )
            add_tail_contours(
                axis,
                prepared.xs * 1e3,
                prepared.ys * 1e3,
                field,
                ambient=prepared.ambient,
                ceiling=float(np.max(truth)),
            )
            if row_index == 0:
                axis.set_title(
                    ["true", "observed clipped", "standard", "random local", "geometry local"][column_index]
                )
            if column_index == 0:
                axis.set_ylabel(name.replace("ScanPath", " "), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.72, label="temperature (K)", extend="max")
    figure.suptitle(
        "Single-frame reconstruction with local ordering from the observed mask"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_synthetic_robustness(results: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), constrained_layout=True)
    metrics = [
        ("saturated_rmse_K", "Observed-censored RMSE (K)"),
        ("saturated_rank_spearman", "Censored-region rank correlation"),
    ]
    x = np.arange(len(FIELD_NAMES))
    offsets = np.linspace(-0.22, 0.22, len(METHODS))
    for axis, (metric, label) in zip(axes, metrics):
        for offset, method in zip(offsets, METHODS):
            selected = results[results["method"] == method].set_index("field")
            axis.plot(
                x + offset,
                [selected.loc[name, metric] for name in FIELD_NAMES],
                "o",
                color=COLORS[method],
                label=method,
                markersize=7,
            )
        axis.set_xticks(x, [FIELD_LABELS[name] for name in FIELD_NAMES], rotation=16)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Compact robustness check on existing synthetic field families")
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    output_path: Path,
    stage1: pd.DataFrame,
    overall: pd.DataFrame | None,
    paired: pd.DataFrame | None,
    synthetic: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    diagnostic = stage1.iloc[0]
    lines = [
        "# Geometry of observed censoring",
        "",
        "All connected components, boundary depths, and local pairs are constructed "
        "only from the noisy observed censoring mask `S_obs = {Y == c}`. Hidden truth "
        "is used only after construction for evaluation.",
        "",
        "Boundary depth is the four-neighbor erosion layer: boundary pixels have depth "
        "zero, and each inward erosion layer adds one camera-grid pixel.",
        "",
        "## Stage 1: diagnostic gate",
        "",
        f"- Observed-censored pixels: {int(diagnostic['n_observed_censored_pixels'])}.",
        f"- Connected components: {int(diagnostic['n_connected_components'])}; "
        f"informative multi-depth components: {int(diagnostic['n_informative_components'])}.",
        f"- Pooled rho(depth, true exceedance): {diagnostic['rho_depth_true_exceedance']:.3f}.",
        f"- Pooled rho(depth, current-GP exceedance): {diagnostic['rho_depth_current_gp_exceedance']:.3f}.",
        f"- Correlation gap: {diagnostic['rho_gap']:.3f}.",
        f"- Direct predicted-vs-true exceedance slope among truly above-ceiling "
        f"pixels: {diagnostic['gp_vs_true_exceedance_slope_latent_above']:.3f}.",
        f"- Median within-component rho(depth, true temperature): "
        f"{diagnostic['median_within_component_rho_depth_true_temperature']:.3f}.",
        f"- Median within-component rho(depth, current GP mean): "
        f"{diagnostic['median_within_component_rho_depth_current_gp']:.3f}.",
        f"- Positive within-component correlations: "
        f"{diagnostic['fraction_informative_component_rho_positive']:.1%}.",
        "",
        f"The predeclared gate was {'passed' if diagnostic['stage2_warranted'] else 'not passed'}. "
        f"It required a rho gap of at least {args.minimum_rho_gap:g}, median component "
        f"rho of at least {args.minimum_component_rho:g}, and at least "
        f"{args.minimum_informative_components} informative components.",
    ]
    if overall is None:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "A positive morphology signal exists in the latent truth, but the "
                "converged standard RBF censored posterior reflects that ordering at "
                "least as strongly. The defining Stage-2 condition, namely that "
                "rho(depth, truth) materially exceed rho(depth, GP), is therefore not "
                "satisfied.",
                "",
                "This does not mean that the GP recovers exceedance magnitude well. "
                "The direct predicted-versus-true exceedance slope remains small. "
                "The diagnostic distinguishes a captured spatial ordering from the "
                "still-unresolved magnitude-compression problem.",
                "",
                "Stage 2 was not run in the canonical experiment. Consequently there "
                "is no claim that geometry-derived preferences improve reconstruction "
                "over standard censoring or the matched random-order control.",
                "",
                "The retained experiment-33 posterior used 120 burn-in steps and gave "
                "a smaller depth correlation. Longer independent chains changed its "
                "censored-pixel posterior means materially; "
                "`legacy_short_chain_convergence_audit.csv` records that discrepancy. "
                "This geometry diagnostic uses four independent chains with 1,200 "
                "burn-in steps and 700 retained draws per chain.",
                "",
                "`mask_signal_robustness.csv` and "
                "`mask_signal_robustness_summary.csv` report the truth-side morphology "
                "diagnostic across four censoring fractions and three noise levels. "
                "These checks use truth only for post-construction evaluation, never "
                "to form the observed-mask geometry.",
            ]
        )
        output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return
    lines.extend(
        [
            "",
            "## Stage 2: local preference likelihood",
            "",
            "The geometry model uses only neighboring observed-censored pixels in the "
            "same component and orients an edge from lower to higher observed-mask "
            "depth. The random control uses the exact same undirected edges with random "
            "orientation. The softness was fixed before held-out evaluation as "
            f"`tau = {args.tau_noise_multiplier:g} * measurement_noise_sd`; no truth was "
            "used for pair construction or tuning.",
            "",
            "| Method | Field L2 | Saturated RMSE | Saturated MAE | Top-1% CRPS | Peak bias | Censored rank rho |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | {row['field_excess_rel_l2_mean']:.3f} | "
            f"{row['saturated_rmse_K_mean']:.3f} K | {row['saturated_mae_K_mean']:.3f} K | "
            f"{row['top_1pct_crps_K_mean']:.3f} K | {row['peak_bias_K_mean']:.3f} K | "
            f"{row['saturated_rank_spearman_mean']:.3f} |"
        )
    geometry_vs_standard = paired[
        (paired["candidate"] == GEOMETRY) & (paired["reference"] == STANDARD)
    ].set_index("metric")
    geometry_vs_random = paired[
        (paired["candidate"] == GEOMETRY) & (paired["reference"] == RANDOM)
    ].set_index("metric")
    sat_standard = geometry_vs_standard.loc["saturated_rmse_K"]
    sat_random = geometry_vs_random.loc["saturated_rmse_K"]
    rank_standard = geometry_vs_standard.loc["saturated_rank_spearman"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Geometry changes saturated-region RMSE by {sat_standard['mean_change_candidate_minus_reference']:+.3f} K "
            f"relative to standard censoring ({int(sat_standard['candidate_win_count'])}/"
            f"{int(sat_standard['n_trajectories'])} trajectory wins) and by "
            f"{sat_random['mean_change_candidate_minus_reference']:+.3f} K relative to "
            f"the matched random control ({int(sat_random['candidate_win_count'])}/"
            f"{int(sat_random['n_trajectories'])} wins).",
            "",
            f"Its censored-region rank correlation changes by "
            f"{rank_standard['mean_change_candidate_minus_reference']:+.3f} relative to "
            "standard censoring. Improvements that appear for geometry but not random "
            "ordering support a genuine morphology effect; similar changes for both "
            "controls would instead indicate generic regularization.",
        ]
    )
    if synthetic is not None and not synthetic.empty:
        geometry_wins = 0
        for field in FIELD_NAMES:
            subset = synthetic[synthetic["field"] == field].set_index("method")
            geometry_wins += int(
                subset.loc[GEOMETRY, "saturated_rmse_K"]
                < subset.loc[STANDARD, "saturated_rmse_K"]
            )
        lines.extend(
            [
                "",
                "## Robustness",
                "",
                "`mask_signal_robustness.csv` varies censoring fraction and measurement "
                "noise on the held-out heat trajectories. `synthetic_robustness.csv` "
                "uses the existing matched Gaussian, rotated Gaussian, two-component "
                "wake, and skewed-wake generators.",
                "",
                f"Geometry improves synthetic saturated-region RMSE over standard "
                f"censoring on {geometry_wins}/{len(FIELD_NAMES)} field families. This "
                "check is descriptive and was not used to tune tau.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_pixels, stage1_components, stage1_edges, stage1_summary = run_stage1(args)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0]
    catalog = list(trajectory_catalog(args.dataset_dir))
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    (
        standard_results,
        standard_sampler,
        _,
        mask_robustness,
        baseline,
        standard_pixels,
    ) = run_main_stage2(
        args,
        fixed,
        catalog,
        methods=(STANDARD,),
        collect_mask_robustness=True,
    )
    if not args.trajectory_names:
        (
            stage1_pixels,
            stage1_components,
            stage1_edges,
            stage1_summary,
        ) = replace_stage1_posterior(stage1_pixels, standard_pixels, args)
    stage1_pixels.to_csv(args.output_dir / "geometry_pixel_diagnostic.csv", index=False)
    stage1_components.to_csv(args.output_dir / "component_summary.csv", index=False)
    stage1_edges.to_csv(args.output_dir / "local_geometry_edges.csv", index=False)
    stage1_summary.to_csv(args.output_dir / "geometry_signal_summary.csv", index=False)
    plot_stage1_depth_signal(
        stage1_pixels,
        stage1_summary,
        args.output_dir / "geometry_depth_signal.png",
    )
    plot_component_correlations(
        stage1_components,
        args.output_dir / "within_component_correlations.png",
    )
    standard_results.to_csv(
        args.output_dir / "standard_posterior_results.csv", index=False
    )
    standard_sampler.to_csv(
        args.output_dir / "standard_sampler_diagnostics.csv", index=False
    )
    baseline.to_csv(
        args.output_dir / "legacy_short_chain_convergence_audit.csv", index=False
    )
    mask_robustness.to_csv(
        args.output_dir / "mask_signal_robustness.csv", index=False
    )
    mask_robustness_summary = summarize_mask_robustness(mask_robustness)
    mask_robustness_summary.to_csv(
        args.output_dir / "mask_signal_robustness_summary.csv", index=False
    )
    plot_mask_robustness(
        mask_robustness_summary,
        args.output_dir / "mask_signal_robustness.png",
    )
    pd.DataFrame(
        [
            {
                **{
                    key: value
                    for key, value in fixed.to_dict().items()
                    if key not in {"posterior_samples", "burn_in", "thin"}
                },
                "geometry_source": "observed censoring mask only",
                "connectivity": "four-neighbor",
                "depth_definition": (
                    "iterated four-neighbor erosion with boundary depth zero"
                ),
                "preference_edges": (
                    "same-component neighboring censored pixels with increasing depth"
                ),
                "random_control": (
                    "same undirected edges with randomized orientation"
                ),
                "tau_noise_multiplier": args.tau_noise_multiplier,
                "tau_K": args.tau_noise_multiplier * args.previous_noise_sd,
                "posterior_chains": args.n_chains,
                "samples_per_chain": args.samples,
                "pooled_posterior_samples": args.n_chains * args.samples,
                "burn_in_per_chain": args.burn_in,
                "thin": args.thin,
                "stage1_gp_posterior": (
                    "pooled converged latent-ESS standard posterior"
                ),
                "stage2_warranted": bool(
                    stage1_summary.iloc[0]["stage2_warranted"]
                ),
                "hidden_truth_used_for_features_pairs_or_tuning": False,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    if not bool(stage1_summary.iloc[0]["stage2_warranted"]):
        write_readme(
            args.output_dir / "README.md",
            stage1_summary,
            None,
            None,
            None,
            args,
        )
        return

    (
        results,
        sampler,
        representatives,
        _,
        _,
        _,
    ) = run_main_stage2(
        args,
        fixed,
        catalog,
        methods=METHODS,
        collect_mask_robustness=False,
    )
    overall = summarize_results(results, ["method"])
    family = summarize_results(results, ["method", "family"])
    paired = paired_comparisons(results)
    synthetic = (
        pd.DataFrame()
        if args.skip_synthetic_robustness
        else run_synthetic_robustness(args)
    )
    outputs = {
        "results.csv": results,
        "heldout30_overall.csv": overall,
        "family_summary.csv": family,
        "paired_comparisons.csv": paired,
        "sampler_diagnostics.csv": sampler,
        "synthetic_robustness.csv": synthetic,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    plot_method_comparison(overall, args.output_dir / "method_comparison.png")
    plot_representative_reconstructions(
        representatives,
        args.output_dir / "representative_reconstructions.png",
    )
    if not synthetic.empty:
        plot_synthetic_robustness(
            synthetic, args.output_dir / "synthetic_robustness.png"
        )
    write_readme(
        args.output_dir / "README.md",
        stage1_summary,
        overall,
        paired,
        synthetic,
        args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observed-mask geometry diagnostic and local preference experiment."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--measurement-noise-sd", type=float, default=0.25)
    parser.add_argument("--previous-noise-sd", type=float, default=0.25)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=700)
    parser.add_argument("--burn-in", type=int, default=1200)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--tau-noise-multiplier", type=float, default=4.0)
    parser.add_argument("--minimum-rho-gap", type=float, default=0.10)
    parser.add_argument("--minimum-component-rho", type=float, default=0.20)
    parser.add_argument("--minimum-informative-components", type=int, default=20)
    parser.add_argument(
        "--robustness-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.10],
    )
    parser.add_argument(
        "--robustness-noise-sd",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5],
    )
    parser.add_argument("--skip-mask-robustness", action="store_true")
    parser.add_argument("--skip-synthetic-robustness", action="store_true")
    parser.add_argument("--synthetic-nx", type=int, default=31)
    parser.add_argument("--synthetic-ny", type=int, default=25)
    parser.add_argument("--synthetic-fraction-saturated", type=float, default=0.10)
    parser.add_argument("--synthetic-noise-sd", type=float, default=20.0)
    parser.add_argument("--synthetic-observation-stride", type=int, default=3)
    parser.add_argument("--synthetic-signal-sd", type=float, default=500.0)
    parser.add_argument("--synthetic-lengthscale", type=float, default=0.70)
    parser.add_argument("--synthetic-chains", type=int, default=3)
    parser.add_argument("--synthetic-samples", type=int, default=180)
    parser.add_argument("--synthetic-burn-in", type=int, default=500)
    parser.add_argument("--synthetic-thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
