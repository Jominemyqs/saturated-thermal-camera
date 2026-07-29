from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_shape_density_module():
    module_path = ROOT / "scripts" / "11_gp_2d_shape_density.py"
    spec = importlib.util.spec_from_file_location("gp2d_shape_density", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp2d_shape_density"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


shape_density = load_shape_density_module()
gp2d = shape_density.gp2d


METHOD_ORDER = shape_density.METHOD_ORDER
COLORS = shape_density.COLORS
SHAPE_LABELS = shape_density.SHAPE_LABELS

SCENARIO_ORDER = ["fixed_ell_0.70", "unsat_mll_tuned"]
SCENARIO_LABELS = {
    "fixed_ell_0.70": r"fixed $\ell=0.70$",
    "unsat_mll_tuned": "unsat. MLL tuned",
}
SCENARIO_COLORS = {
    "fixed_ell_0.70": "#999999",
    "unsat_mll_tuned": "#0072B2",
}


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peak_interval_width"] = df["peak_ci_upper"] - df["peak_ci_lower"]
    grouped = df.groupby(["scenario", "shape", "shape_label", "method"], sort=False)
    summary = grouped.agg(
        n_seeds=("observation_seed", "nunique"),
        field_rel_l2_mean=("field_rel_l2", "mean"),
        field_rel_l2_std=("field_rel_l2", "std"),
        hot_region_rel_l2_mean=("hot_region_rel_l2", "mean"),
        peak_abs_error_mean=("peak_abs_error", "mean"),
        peak_abs_error_std=("peak_abs_error", "std"),
        posterior_sd_at_true_peak_mean=("posterior_sd_at_true_peak", "mean"),
        peak_interval_width_mean=("peak_interval_width", "mean"),
        true_peak_coverage=("true_peak_in_95", "mean"),
        selected_lengthscale_mean=("selected_lengthscale", "mean"),
        selected_lengthscale_std=("selected_lengthscale", "std"),
        actual_frac_saturated_mean=("actual_frac_saturated", "mean"),
    ).reset_index()
    return summary


def plot_scenario_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    shapes = list(SHAPE_LABELS)
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.5 * len(shapes), 4.4), sharey=True, constrained_layout=True)
    if len(shapes) == 1:
        axes = [axes]

    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    offsets = {
        "fixed_ell_0.70": -width / 2,
        "unsat_mll_tuned": width / 2,
    }

    for ax, shape_name in zip(axes, shapes):
        sub = summary[summary["shape"] == shape_name]
        for scenario in SCENARIO_ORDER:
            values = []
            errors = []
            for method in METHOD_ORDER:
                row = sub[(sub["scenario"] == scenario) & (sub["method"] == method)]
                values.append(float(row.iloc[0][metric]) if len(row) else np.nan)
                std_metric = metric.replace("_mean", "_std")
                errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns and len(row) else 0.0)
            show_errors = metric.endswith("_mean") and any(np.isfinite(errors))
            ax.bar(
                x + offsets[scenario],
                values,
                width=width,
                yerr=errors if show_errors else None,
                capsize=2 if show_errors else 0,
                color=SCENARIO_COLORS[scenario],
                alpha=0.78,
                label=SCENARIO_LABELS[scenario],
            )
        ax.set_title(SHAPE_LABELS[shape_name])
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=24, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_selected_lengthscales(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    short_labels = {
        "axis_gaussian": "axis",
        "rotated_wake": "rotated",
        "laser_path": "laser path",
    }
    positions = []
    labels = []
    data = []
    for i, shape_name in enumerate(SHAPE_LABELS):
        fixed = results[(results["shape"] == shape_name) & (results["scenario"] == "fixed_ell_0.70")]
        tuned = results[(results["shape"] == shape_name) & (results["scenario"] == "unsat_mll_tuned")]
        positions.extend([3 * i, 3 * i + 1])
        labels.extend([f"{short_labels[shape_name]}\nfixed", f"{short_labels[shape_name]}\ntuned"])
        data.extend(
            [
                fixed.drop_duplicates("observation_seed")["selected_lengthscale"].to_numpy(),
                tuned.drop_duplicates("observation_seed")["selected_lengthscale"].to_numpy(),
            ]
        )
    box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
    colors = ["#999999", "#0072B2"] * len(SHAPE_LABELS)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"lengthscale $\ell$")
    ax.set_title("Fixed versus unsaturated-MLL selected lengthscale")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    base_config = gp2d.GP2DConfig(lengthscale=0.70)
    obs_n = 17
    pred_n = 30
    seeds = list(range(20))
    frames = []

    for shape_index, shape_name in enumerate(SHAPE_LABELS):
        for seed in seeds:
            fixed_config = gp2d.GP2DConfig(
                mean_temp=base_config.mean_temp,
                signal_sd=base_config.signal_sd,
                lengthscale=0.70,
                noise_sd=base_config.noise_sd,
                relative_jitter=base_config.relative_jitter,
            )
            fixed_data = shape_density.make_observations(
                fixed_config,
                shape_name,
                obs_n=obs_n,
                pred_n=pred_n,
                seed=seed,
            )
            print(f"{SHAPE_LABELS[shape_name]}, seed={seed:02d}: fixed ell=0.700")
            fixed_predictions = shape_density.run_methods(
                fixed_config,
                fixed_data,
                sampler_seed=50_000 + 1_000 * shape_index + seed,
            )
            fixed_metrics = shape_density.metrics_for_run(
                fixed_predictions,
                fixed_data,
                fixed_config,
                shape_name,
                obs_n,
                selected_lengthscale=0.70,
            )
            fixed_metrics["scenario"] = "fixed_ell_0.70"
            fixed_metrics["observation_seed"] = seed
            frames.append(fixed_metrics)

            probe_data = shape_density.make_observations(
                base_config,
                shape_name,
                obs_n=obs_n,
                pred_n=pred_n,
                seed=seed,
            )
            ell_hat = shape_density.select_lengthscale_by_unsat_mll(probe_data, base_config)
            tuned_config = gp2d.GP2DConfig(
                mean_temp=base_config.mean_temp,
                signal_sd=base_config.signal_sd,
                lengthscale=ell_hat,
                noise_sd=base_config.noise_sd,
                relative_jitter=base_config.relative_jitter,
            )
            tuned_data = shape_density.make_observations(
                tuned_config,
                shape_name,
                obs_n=obs_n,
                pred_n=pred_n,
                seed=seed,
            )
            print(f"{SHAPE_LABELS[shape_name]}, seed={seed:02d}: selected ell={ell_hat:.3f}")
            tuned_predictions = shape_density.run_methods(
                tuned_config,
                tuned_data,
                sampler_seed=60_000 + 1_000 * shape_index + seed,
            )
            tuned_metrics = shape_density.metrics_for_run(
                tuned_predictions,
                tuned_data,
                tuned_config,
                shape_name,
                obs_n,
                selected_lengthscale=ell_hat,
            )
            tuned_metrics["scenario"] = "unsat_mll_tuned"
            tuned_metrics["observation_seed"] = seed
            frames.append(tuned_metrics)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results)

    results_path = out_dir / "gp2d_multiseed_fixed_vs_tuned_results.csv"
    summary_path = out_dir / "gp2d_multiseed_fixed_vs_tuned_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_scenario_metric(
        summary,
        "field_rel_l2_mean",
        "Relative L2 field error",
        out_dir / "gp2d_multiseed_fixed_vs_tuned_field_error.png",
    )
    plot_scenario_metric(
        summary,
        "hot_region_rel_l2_mean",
        "Relative L2 error in hot region",
        out_dir / "gp2d_multiseed_fixed_vs_tuned_hot_region_error.png",
    )
    plot_scenario_metric(
        summary,
        "peak_abs_error_mean",
        "Peak absolute error",
        out_dir / "gp2d_multiseed_fixed_vs_tuned_peak_error.png",
    )
    plot_scenario_metric(
        summary,
        "true_peak_coverage",
        "True peak inside 95% interval",
        out_dir / "gp2d_multiseed_fixed_vs_tuned_peak_coverage.png",
        ylim=(0.0, 1.05),
    )
    plot_scenario_metric(
        summary,
        "posterior_sd_at_true_peak_mean",
        "Posterior SD at true peak",
        out_dir / "gp2d_multiseed_fixed_vs_tuned_peak_sd.png",
    )
    plot_selected_lengthscales(results, out_dir / "gp2d_multiseed_fixed_vs_tuned_lengthscales.png")

    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(
        summary[
            [
                "scenario",
                "shape_label",
                "method",
                "field_rel_l2_mean",
                "hot_region_rel_l2_mean",
                "peak_abs_error_mean",
                "posterior_sd_at_true_peak_mean",
                "true_peak_coverage",
                "selected_lengthscale_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
