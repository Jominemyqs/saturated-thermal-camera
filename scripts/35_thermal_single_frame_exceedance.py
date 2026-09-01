from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr

from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    infer_previous_censored_posterior,
    infer_previous_coherent_posterior,
    paired_camera_observations,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "33_single_frame_exceedance"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "32_uncertainty_oracle_v2"
    / "fixed_configuration.csv"
)

HYBRID = "hybrid saturated-only sampling"
COHERENT = "coherent latent single-frame posterior"
REPRESENTATIONS = (HYBRID, COHERENT)
EXCEEDANCE_EDGES_K = np.array(
    [-np.inf, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf]
)
REPRESENTATIVES = {
    "DiagonalScanPath_7",
    "HorizontalScanPath_10",
    "SpiralScanPath_12",
}


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


def infer_single_frame_posteriors(
    prepared,
    camera: dict[str, object],
    threshold: float,
    fixed: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    common = {
        "prepared": prepared,
        "frame": camera["frames"][0],
        "fixed_mask": camera["fixed_observation_mask"],
        "threshold": threshold,
        "signal_sd": float(fixed.signal_sd),
        "lengthscale": prepared.source_lengthscale * args.length_multiplier,
        "noise_sd": args.previous_noise_sd,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "seed": seed,
    }
    hybrid_mean, hybrid_draws, _ = infer_previous_censored_posterior(
        **common,
        n_chains=1,
        samples_per_chain=args.samples,
    )
    coherent_mean, coherent_draws, _ = infer_previous_coherent_posterior(
        **common,
        n_samples=args.samples,
    )
    return {
        HYBRID: (hybrid_mean, hybrid_draws),
        COHERENT: (coherent_mean, coherent_draws),
    }


def censored_pixel_frame(
    *,
    prepared,
    record,
    role: str,
    threshold: float,
    previous_index: int,
    saturated: np.ndarray,
    representation: str,
    posterior_mean: np.ndarray,
    posterior_draws: np.ndarray,
) -> pd.DataFrame:
    mask = np.asarray(saturated, dtype=bool).ravel()
    truth = np.asarray(prepared.history[previous_index], dtype=float).ravel()[mask]
    mean = np.asarray(posterior_mean, dtype=float).ravel()[mask]
    draws = np.asarray(posterior_draws, dtype=float).reshape(
        len(posterior_draws), -1
    )[:, mask]
    sd = np.std(draws, axis=0, ddof=1)
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    error = mean - truth
    delta_true = truth - threshold
    delta_hat = mean - threshold
    ratio = np.divide(
        np.abs(error),
        sd,
        out=np.full_like(error, np.nan),
        where=sd > 1e-10,
    )
    flat_indices = np.flatnonzero(mask)
    iy, ix = np.unravel_index(flat_indices, prepared.truth.shape)
    return pd.DataFrame(
        {
            "trajectory": record.name,
            "family": record.family,
            "run_index": record.run_index,
            "role": role,
            "representation": representation,
            "pixel_index": flat_indices,
            "x_m": prepared.xs[ix],
            "y_m": prepared.ys[iy],
            "threshold_K": threshold,
            "truth_K": truth,
            "posterior_mean_K": mean,
            "posterior_sd_K": sd,
            "lower_95_K": lower,
            "upper_95_K": upper,
            "delta_true_K": delta_true,
            "delta_hat_K": delta_hat,
            "signed_error_K": error,
            "absolute_error_K": np.abs(error),
            "abs_error_over_sd": ratio,
            "covered_95": (lower <= truth) & (truth <= upper),
            "latent_truth_above_ceiling": delta_true > 0.0,
            "posterior_mean_above_ceiling": delta_hat > 0.0,
            "posterior_probability_above_ceiling": np.mean(
                draws > threshold, axis=0
            ),
        }
    )


def finite_pair_statistic(x: np.ndarray, y: np.ndarray, statistic) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2 or np.ptp(x[valid]) <= 0.0 or np.ptp(y[valid]) <= 0.0:
        return np.nan
    return float(statistic(x[valid], y[valid]).statistic)


def summarize_pixels(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(group_key, sort=False, dropna=False):
        keys = (key,) if len(group_columns) == 1 else key
        identity = dict(zip(group_columns, keys))
        delta_true = group["delta_true_K"].to_numpy(float)
        delta_hat = group["delta_hat_K"].to_numpy(float)
        true_above = group["latent_truth_above_ceiling"].to_numpy(bool)
        positive_truth = delta_true[true_above]
        positive_hat = delta_hat[true_above]
        positive_sd = group.loc[true_above, "posterior_sd_K"].to_numpy(float)
        if np.sum(true_above) >= 2 and np.ptp(positive_truth) > 0.0:
            slope, intercept = np.polyfit(positive_truth, positive_hat, 1)
            sd_slope, sd_intercept = np.polyfit(positive_truth, positive_sd, 1)
        else:
            slope, intercept = np.nan, np.nan
            sd_slope, sd_intercept = np.nan, np.nan
        denominator = float(np.sum(positive_truth))
        rows.append(
            {
                **identity,
                "n_observed_censored_pixels": len(group),
                "n_latent_truth_above_ceiling": int(np.sum(true_above)),
                "fraction_latent_truth_above_ceiling": float(np.mean(true_above)),
                "false_saturation_fraction": float(np.mean(~true_above)),
                "fraction_posterior_mean_above_ceiling": float(
                    np.mean(group["posterior_mean_above_ceiling"])
                ),
                "mean_posterior_probability_above_ceiling": float(
                    group["posterior_probability_above_ceiling"].mean()
                ),
                "mean_true_exceedance_K": float(np.mean(positive_truth))
                if len(positive_truth)
                else np.nan,
                "mean_predicted_exceedance_K": float(np.mean(positive_hat))
                if len(positive_hat)
                else np.nan,
                "median_true_exceedance_K": float(np.median(positive_truth))
                if len(positive_truth)
                else np.nan,
                "median_predicted_exceedance_K": float(np.median(positive_hat))
                if len(positive_hat)
                else np.nan,
                "exceedance_recovery_fraction": float(np.sum(positive_hat) / denominator)
                if denominator > 0.0
                else np.nan,
                "calibration_slope": float(slope),
                "calibration_intercept_K": float(intercept),
                "posterior_sd_slope_K_per_K": float(sd_slope),
                "posterior_sd_intercept_K": float(sd_intercept),
                "pearson_correlation": finite_pair_statistic(
                    positive_truth, positive_hat, pearsonr
                ),
                "spearman_correlation": finite_pair_statistic(
                    positive_truth, positive_hat, spearmanr
                ),
                "posterior_sd_spearman_correlation": finite_pair_statistic(
                    positive_truth, positive_sd, spearmanr
                ),
                "rmse_K": float(np.sqrt(np.mean(group["signed_error_K"] ** 2))),
                "mae_K": float(group["absolute_error_K"].mean()),
                "bias_K": float(group["signed_error_K"].mean()),
                "posterior_sd_K": float(group["posterior_sd_K"].mean()),
                "coverage_95": float(group["covered_95"].mean()),
                "interval_width_95_K": float(
                    (group["upper_95_K"] - group["lower_95_K"]).mean()
                ),
                "mean_abs_error_over_sd": float(
                    group["abs_error_over_sd"].mean()
                ),
                "median_abs_error_over_sd": float(
                    group["abs_error_over_sd"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def add_exceedance_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels = [
        "latent <= ceiling",
        "0-0.5 K",
        "0.5-1 K",
        "1-2 K",
        "2-5 K",
        "5-10 K",
        "10-20 K",
        ">20 K",
    ]
    result["true_exceedance_bin"] = pd.cut(
        result["delta_true_K"],
        bins=EXCEEDANCE_EDGES_K,
        labels=labels,
        right=True,
        include_lowest=True,
        ordered=True,
    )
    return result


def binned_plot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for representation in REPRESENTATIONS:
        selected = frame[frame["representation"] == representation]
        for label, group in selected.groupby(
            "true_exceedance_bin", observed=True, sort=False
        ):
            rows.append(
                {
                    "representation": representation,
                    "true_exceedance_bin": str(label),
                    "n_pixels": len(group),
                    "mean_delta_true_K": group["delta_true_K"].mean(),
                    "mean_delta_hat_K": group["delta_hat_K"].mean(),
                    "median_delta_hat_K": group["delta_hat_K"].median(),
                    "q25_delta_hat_K": group["delta_hat_K"].quantile(0.25),
                    "q75_delta_hat_K": group["delta_hat_K"].quantile(0.75),
                    "mean_posterior_sd_K": group["posterior_sd_K"].mean(),
                    "q25_posterior_sd_K": group["posterior_sd_K"].quantile(0.25),
                    "q75_posterior_sd_K": group["posterior_sd_K"].quantile(0.75),
                    "mean_abs_error_over_sd": group["abs_error_over_sd"].mean(),
                    "median_abs_error_over_sd": group["abs_error_over_sd"].median(),
                    "q25_abs_error_over_sd": group["abs_error_over_sd"].quantile(0.25),
                    "q75_abs_error_over_sd": group["abs_error_over_sd"].quantile(0.75),
                    "coverage_95": group["covered_95"].mean(),
                    "mean_posterior_probability_above_ceiling": group[
                        "posterior_probability_above_ceiling"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def add_binned_trend(axis, bins: pd.DataFrame, value: str, *, color: str) -> None:
    ordered = bins.sort_values("mean_delta_true_K")
    axis.plot(
        ordered["mean_delta_true_K"],
        ordered[value],
        color=color,
        marker="o",
        linewidth=2.0,
        markersize=4,
        label="binned mean",
        zorder=4,
    )


def plot_exceedance_diagnostic(
    pixels: pd.DataFrame,
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = pixels[
        (pixels["role"] == "evaluation")
        & (pixels["representation"] == COHERENT)
    ]
    selected_bins = bins[bins["representation"] == COHERENT]
    x = selected["delta_true_K"].to_numpy(float)
    maximum = float(max(np.quantile(x, 0.995), 1.0))
    minimum = float(min(np.quantile(x, 0.005), -0.25))
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)

    scatter = axes[0].hexbin(
        x,
        selected["delta_hat_K"],
        gridsize=48,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(),
    )
    axes[0].plot([minimum, maximum], [minimum, maximum], "--", color="#444444")
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].axvline(0.0, color="#777777", linewidth=0.8)
    add_binned_trend(axes[0], selected_bins, "mean_delta_hat_K", color="#D55E00")
    axes[0].set_ylabel(r"predicted exceedance $\hat\Delta_i$ (K)")
    figure.colorbar(scatter, ax=axes[0], label="censored-pixel count")

    axes[1].scatter(
        x,
        selected["posterior_sd_K"],
        s=7,
        alpha=0.16,
        color="#0072B2",
        edgecolors="none",
    )
    add_binned_trend(
        axes[1], selected_bins, "mean_posterior_sd_K", color="#D55E00"
    )
    axes[1].axvline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel(r"posterior SD $\sigma_i$ (K)")

    positive_ratio = np.maximum(
        selected["abs_error_over_sd"].to_numpy(float), 1e-3
    )
    axes[2].scatter(
        x,
        positive_ratio,
        s=7,
        alpha=0.16,
        color="#009E73",
        edgecolors="none",
    )
    add_binned_trend(
        axes[2], selected_bins, "mean_abs_error_over_sd", color="#D55E00"
    )
    axes[2].axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[2].axhline(1.96, color="#555555", linestyle=":", linewidth=0.9)
    axes[2].axvline(0.0, color="#777777", linewidth=0.8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"standardized error $|e_i|/\sigma_i$")

    for axis in axes:
        axis.set_xlim(minimum, maximum)
        axis.set_xlabel(r"true exceedance $\Delta_i^{true}$ (K)")
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Single-frame censored posterior: exceedance magnitude and uncertainty\n"
        "30 held-out trajectories; coherent latent RBF posterior"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_binned_calibration(bins: pd.DataFrame, output_path: Path) -> None:
    selected = bins[bins["representation"] == COHERENT].sort_values(
        "mean_delta_true_K"
    )
    metrics = [
        ("mean_delta_hat_K", "Predicted exceedance (K)"),
        ("mean_posterior_sd_K", "Posterior SD (K)"),
        ("coverage_95", "95% coverage"),
        ("mean_abs_error_over_sd", r"Mean $|e|/\sigma$"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.3), constrained_layout=True)
    x = selected["mean_delta_true_K"]
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        axis.plot(x, selected[metric], marker="o", linewidth=2.0, color="#0072B2")
        if metric == "mean_delta_hat_K":
            limits = [min(float(x.min()), 0.0), max(float(x.max()), 1.0)]
            axis.plot(limits, limits, linestyle="--", color="#555555")
        elif metric == "coverage_95":
            axis.axhline(0.95, linestyle="--", color="#555555")
            axis.set_ylim(0.0, 1.02)
        elif metric == "mean_abs_error_over_sd":
            axis.axhline(1.0, linestyle="--", color="#555555")
            axis.axhline(1.96, linestyle=":", color="#555555")
        axis.set_xlabel(r"mean true exceedance in bin (K)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Calibration by true exceedance bin\nObserved-censored pixels on 30 held-out trajectories"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_slide_exceedance_summary(
    bins: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = bins[
        (bins["representation"] == COHERENT)
        & (bins["mean_delta_true_K"] > 0.0)
    ].sort_values("mean_delta_true_K")
    x = selected["mean_delta_true_K"].to_numpy(float)
    delta_hat = selected["mean_delta_hat_K"].to_numpy(float)
    delta_lower = selected["q25_delta_hat_K"].to_numpy(float)
    delta_upper = selected["q75_delta_hat_K"].to_numpy(float)
    ratio = selected["mean_abs_error_over_sd"].to_numpy(float)
    ratio_lower = selected["q25_abs_error_over_sd"].to_numpy(float)
    ratio_upper = selected["q75_abs_error_over_sd"].to_numpy(float)
    maximum = float(np.ceil(max(np.max(x), 1.0) / 5.0) * 5.0)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    left, right = axes
    left.fill_between(
        x,
        delta_lower,
        delta_upper,
        color="#D55E00",
        alpha=0.18,
        linewidth=0.0,
        label="interquartile range",
    )
    left.plot(
        x,
        delta_hat,
        color="#D55E00",
        marker="o",
        linewidth=3.0,
        markersize=6,
        label="GP binned mean",
    )
    left.plot(
        [0.0, maximum],
        [0.0, maximum],
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.8,
        label="perfect magnitude recovery",
    )
    left.set_xlim(0.0, maximum)
    left.set_ylim(0.0, maximum)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel(r"true exceedance $\Delta_i^{true}=T_i^{true}-c$ (K)")
    left.set_ylabel(r"predicted exceedance $\hat\Delta_i=E[T_i\mid Y]-c$ (K)")
    left.set_title("GP detects exceedance but not its magnitude", fontweight="bold")
    left.legend(frameon=False, loc="upper right", fontsize=9)

    right.fill_between(
        x,
        ratio_lower,
        ratio_upper,
        color="#0072B2",
        alpha=0.18,
        linewidth=0.0,
        label="interquartile range",
    )
    right.plot(
        x,
        ratio,
        color="#0072B2",
        marker="o",
        linewidth=3.0,
        markersize=6,
        label=r"binned mean $|e|/\sigma$",
    )
    right.set_xlim(0.0, maximum)
    right.set_ylim(0.0, max(80.0, float(np.max(ratio_upper)) * 1.06))
    right.set_xlabel(r"true exceedance $\Delta_i^{true}=T_i^{true}-c$ (K)")
    right.set_ylabel(r"standardized absolute error $|e_i|/\sigma_i$")
    right.set_title(
        "Posterior uncertainty does not scale with hidden error",
        fontweight="bold",
    )
    right.legend(frameon=False, loc="upper left", fontsize=9)

    for axis in axes:
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Single-frame censored GP: correct inequality, compressed magnitude",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_representative_maps(
    representative_rows: dict[str, tuple[object, pd.DataFrame]],
    output_path: Path,
) -> None:
    ordered = [
        "DiagonalScanPath_7",
        "HorizontalScanPath_10",
        "SpiralScanPath_12",
    ]
    all_rows = pd.concat([representative_rows[name][1] for name in ordered])
    delta_max = max(float(all_rows["delta_true_K"].quantile(0.99)), 1.0)
    sd_max = max(float(all_rows["posterior_sd_K"].quantile(0.99)), 0.25)
    ratio_max = max(float(all_rows["abs_error_over_sd"].quantile(0.95)), 2.0)
    figure, axes = plt.subplots(3, 4, figsize=(13.4, 8.4), constrained_layout=True)
    for row_index, name in enumerate(ordered):
        prepared, data = representative_rows[name]
        extent = [
            prepared.xs[0] * 1e3,
            prepared.xs[-1] * 1e3,
            prepared.ys[0] * 1e3,
            prepared.ys[-1] * 1e3,
        ]
        fields = []
        for column in (
            "delta_true_K",
            "delta_hat_K",
            "posterior_sd_K",
            "abs_error_over_sd",
        ):
            field = np.full(prepared.truth.size, np.nan)
            field[data["pixel_index"].to_numpy(int)] = data[column].to_numpy(float)
            fields.append(field.reshape(prepared.truth.shape))
        images = []
        for column_index, field in enumerate(fields):
            if column_index < 2:
                image = axes[row_index, column_index].imshow(
                    field,
                    origin="lower",
                    extent=extent,
                    cmap="coolwarm",
                    vmin=-0.5,
                    vmax=delta_max,
                    aspect="auto",
                )
            elif column_index == 2:
                image = axes[row_index, column_index].imshow(
                    field,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=sd_max,
                    aspect="auto",
                )
            else:
                image = axes[row_index, column_index].imshow(
                    field,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=0.0,
                    vmax=ratio_max,
                    aspect="auto",
                )
            images.append(image)
            axis = axes[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    [
                        r"true $\Delta$",
                        r"predicted $\hat\Delta$",
                        r"posterior $\sigma$",
                        r"$|e|/\sigma$",
                    ][column_index]
                )
            if column_index == 0:
                axis.set_ylabel(name.replace("ScanPath", " "), fontsize=9)
        if row_index == 2:
            figure.colorbar(images[0], ax=axes[:, :2], shrink=0.70, label="K")
            figure.colorbar(images[2], ax=axes[:, 2], shrink=0.70, label="K")
            figure.colorbar(images[3], ax=axes[:, 3], shrink=0.70, label="ratio")
    figure.suptitle(
        "Where the single-frame censored posterior loses exceedance magnitude\n"
        "Only observed-censored pixels are colored"
    )
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    output_path: Path,
    pooled: pd.DataFrame,
    bin_summary: pd.DataFrame,
) -> None:
    coherent = pooled[
        (pooled["representation"] == COHERENT)
        & (pooled["subset"] == "latent truth above ceiling")
    ].iloc[0]
    highest = bin_summary[
        (bin_summary["representation"] == COHERENT)
        & (bin_summary["true_exceedance_bin"] == ">20 K")
    ]
    if highest.empty:
        highest = bin_summary[
            bin_summary["representation"] == COHERENT
        ].sort_values("mean_delta_true_K").tail(1)
    hottest = highest.iloc[0]
    lines = [
        "# Single-frame censored-posterior exceedance diagnostic",
        "",
        "This experiment asks whether the censored RBF GP recognizes that a pixel is "
        "above the camera ceiling but fails to infer how far above it lies. No model, "
        "hyperparameter, camera realization, held-out split, or sampler setting was "
        "changed from the canonical uncertainty audit.",
        "",
        "For each observed-censored previous-frame pixel,",
        "",
        "- `delta_true_K = T_true - c`;",
        "- `delta_hat_K = E[T | Y] - c`;",
        "- `posterior_sd_K` is the sample posterior standard deviation;",
        "- `abs_error_over_sd = |E[T | Y] - T_true| / posterior_sd`.",
        "",
        "Observed censoring means the noisy pre-clipped measurement exceeded `c`; it "
        "does not guarantee that the noise-free latent truth exceeds `c`. The CSV files "
        "therefore label false saturations explicitly, and the primary magnitude "
        "calibration is computed on pixels with latent truth above the ceiling.",
        "",
        "`single_frame_exceedance_diagnostic.png` is the presentation-facing two-panel "
        "summary. `single_frame_exceedance_detailed.png` retains the raw-pixel diagnostic "
        "for audit.",
        "",
        "## Held-out result",
        "",
        "| Posterior representation | Observed censored | Latent above | True-above fraction | True exceedance | Predicted exceedance | Recovery | Slope | SD | Coverage | Mean abs. error/SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        all_row = pooled[
            (pooled["representation"] == representation)
            & (pooled["subset"] == "all observed-censored pixels")
        ].iloc[0]
        row = pooled[
            (pooled["representation"] == representation)
            & (pooled["subset"] == "latent truth above ceiling")
        ].iloc[0]
        lines.append(
            f"| {row['representation']} | "
            f"{int(all_row['n_observed_censored_pixels'])} | "
            f"{int(row['n_observed_censored_pixels'])} | "
            f"{all_row['fraction_latent_truth_above_ceiling']:.3f} | "
            f"{row['mean_true_exceedance_K']:.3f} K | "
            f"{row['mean_predicted_exceedance_K']:.3f} K | "
            f"{row['exceedance_recovery_fraction']:.3f} | "
            f"{row['calibration_slope']:.3f} | "
            f"{row['posterior_sd_K']:.3f} K | "
            f"{row['coverage_95']:.3f} | "
            f"{row['mean_abs_error_over_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"For the coherent latent posterior, the regression slope of predicted on "
            f"true exceedance is {coherent['calibration_slope']:.3f}, and it recovers "
            f"{coherent['exceedance_recovery_fraction']:.1%} of the aggregate true "
            "exceedance. A slope or recovery fraction well below one is direct evidence "
            "of magnitude compression rather than merely noisy ranking.",
            "",
            f"The posterior-SD slope is "
            f"{coherent['posterior_sd_slope_K_per_K']:.4f} K per K of true exceedance. "
            "Thus uncertainty does not expand enough as the hidden magnitude grows.",
            "",
            f"In the hottest available exceedance bin, the mean true exceedance is "
            f"{hottest['mean_delta_true_K']:.3f} K while the inferred exceedance is "
            f"{hottest['mean_delta_hat_K']:.3f} K. Posterior SD remains "
            f"{hottest['mean_posterior_sd_K']:.3f} K, coverage is "
            f"{hottest['coverage_95']:.3f}, and mean standardized absolute error is "
            f"{hottest['mean_abs_error_over_sd']:.3f}.",
            "",
            "The hybrid and coherent posterior rows are included to check whether "
            "making unsaturated locations latent changes the censored-pixel result. "
            "The coherent posterior is the primary scientific representation.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(FIXED_CONFIG).iloc[0]
    catalog = list(trajectory_catalog(args.dataset_dir))
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    held_out = [record for record in catalog if record.name not in DEVELOPMENT_TRAJECTORIES]
    if len(held_out) != 30:
        raise AssertionError(f"Expected 30 held-out trajectories, found {len(held_out)}")

    rows: list[pd.DataFrame] = []
    representative_rows: dict[str, tuple[object, pd.DataFrame]] = {}
    catalog_lookup = {record.name: index for index, record in enumerate(catalog)}
    for run_index, record in enumerate(held_out):
        catalog_index = catalog_lookup[record.name]
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
        posteriors = infer_single_frame_posteriors(
            prepared,
            camera,
            threshold,
            fixed,
            args,
            args.seed + 50_000 * catalog_index,
        )
        saturated = np.asarray(camera["frames"][0]["saturated_full"], dtype=bool)
        for representation, (mean, draws) in posteriors.items():
            pixel_frame = censored_pixel_frame(
                prepared=prepared,
                record=record,
                role="evaluation",
                threshold=threshold,
                previous_index=previous_index,
                saturated=saturated,
                representation=representation,
                posterior_mean=mean,
                posterior_draws=draws,
            )
            rows.append(pixel_frame)
            if record.name in REPRESENTATIVES and representation == COHERENT:
                representative_rows[record.name] = (prepared, pixel_frame)
        checkpoint = pd.concat(rows, ignore_index=True)
        checkpoint.to_csv(args.output_dir / "checkpoint_censored_pixels.csv", index=False)
        print(
            f"[{run_index + 1:02d}/30] {record.name}: "
            f"observed-censored={int(np.sum(saturated))}, "
            f"latent-above={float(np.mean(prepared.history[previous_index][saturated] > threshold)):.3f}",
            flush=True,
        )

    pixels = add_exceedance_bins(pd.concat(rows, ignore_index=True))
    all_summary_rows = []
    for representation in REPRESENTATIONS:
        selected = pixels[pixels["representation"] == representation]
        for subset, subset_frame in (
            ("all observed-censored pixels", selected),
            (
                "latent truth above ceiling",
                selected[selected["latent_truth_above_ceiling"]],
            ),
        ):
            summary = summarize_pixels(subset_frame, ["representation"])
            summary["subset"] = subset
            all_summary_rows.append(summary)
    pooled_summary = pd.concat(all_summary_rows, ignore_index=True)
    trajectory_summary = summarize_pixels(
        pixels,
        ["representation", "trajectory", "family", "run_index"],
    )
    family_summary = trajectory_summary.groupby(
        ["representation", "family"], sort=False
    ).mean(numeric_only=True).reset_index()
    family_summary["n_trajectories"] = (
        trajectory_summary.groupby(["representation", "family"], sort=False)[
            "trajectory"
        ]
        .nunique()
        .to_numpy()
    )
    bin_summary = binned_plot_rows(pixels)

    coherent = pixels[pixels["representation"] == COHERENT].set_index(
        ["trajectory", "pixel_index"]
    )
    hybrid = pixels[pixels["representation"] == HYBRID].set_index(
        ["trajectory", "pixel_index"]
    )
    agreement = pd.DataFrame(
        {
            "quantity": ["posterior_mean_K", "posterior_sd_K"],
            "mean_absolute_difference": [
                float(
                    np.mean(
                        np.abs(
                            coherent["posterior_mean_K"]
                            - hybrid["posterior_mean_K"]
                        )
                    )
                ),
                float(
                    np.mean(
                        np.abs(
                            coherent["posterior_sd_K"]
                            - hybrid["posterior_sd_K"]
                        )
                    )
                ),
            ],
        }
    )

    outputs = {
        "censored_pixel_diagnostics.csv": pixels,
        "overall_summary.csv": pooled_summary,
        "trajectory_summary.csv": trajectory_summary,
        "family_summary.csv": family_summary,
        "exceedance_bin_summary.csv": bin_summary,
        "representation_agreement.csv": agreement,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    pd.DataFrame(
        [
            {
                **fixed.to_dict(),
                "nx": args.nx,
                "ny": args.ny,
                "fraction_saturated": args.fraction_saturated,
                "observation_stride": args.observation_stride,
                "previous_frame_offset": args.previous_frame_offset,
                "measurement_noise_sd": args.measurement_noise_sd,
                "previous_noise_sd": args.previous_noise_sd,
                "length_multiplier": args.length_multiplier,
                "posterior_samples": args.samples,
                "burn_in": args.burn_in,
                "thin": args.thin,
                "seed": args.seed,
                "dataset_trajectories": 33,
                "development_trajectories": 3,
                "heldout_trajectories": 30,
                "diagnostic_stage": "single previous frame",
                "observation_interpretation": (
                    "observed censoring means latent temperature plus measurement noise exceeds ceiling"
                ),
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)

    plot_exceedance_diagnostic(
        pixels,
        bin_summary,
        args.output_dir / "single_frame_exceedance_detailed.png",
    )
    plot_slide_exceedance_summary(
        bin_summary,
        args.output_dir / "single_frame_exceedance_diagnostic.png",
    )
    plot_binned_calibration(
        bin_summary,
        args.output_dir / "exceedance_bin_calibration.png",
    )
    if set(representative_rows) != REPRESENTATIVES:
        raise AssertionError(
            f"Missing representative trajectories: {REPRESENTATIVES - set(representative_rows)}"
        )
    plot_representative_maps(
        representative_rows,
        args.output_dir / "representative_exceedance_maps.png",
    )
    write_readme(
        args.output_dir / "README.md",
        pooled_summary,
        bin_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose exceedance-magnitude compression in the single-frame censored posterior."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--measurement-noise-sd", type=float, default=0.25)
    parser.add_argument("--previous-noise-sd", type=float, default=0.25)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--burn-in", type=int, default=120)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
