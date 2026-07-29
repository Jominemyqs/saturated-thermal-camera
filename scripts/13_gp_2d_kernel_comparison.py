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
SHAPE_LABELS = shape_density.SHAPE_LABELS

KERNEL_ORDER = ["isotropic_fixed", "isotropic_tuned", "anisotropic_tuned", "anisotropic_informed"]
KERNEL_LABELS = {
    "isotropic_fixed": r"isotropic fixed",
    "isotropic_tuned": r"isotropic tuned",
    "anisotropic_tuned": r"anisotropic tuned",
    "anisotropic_informed": r"anisotropic informed",
}
KERNEL_COLORS = {
    "isotropic_fixed": "#999999",
    "isotropic_tuned": "#0072B2",
    "anisotropic_tuned": "#009E73",
    "anisotropic_informed": "#D55E00",
}

INFORMED_KERNELS = {
    "axis_gaussian": {"lengthscale": 0.75, "lengthscale_y": 0.48, "angle_degrees": 0.0},
    "rotated_wake": {"lengthscale": 1.25, "lengthscale_y": 0.28, "angle_degrees": 34.0},
    "laser_path": {"lengthscale": 1.10, "lengthscale_y": 0.20, "angle_degrees": 0.0},
}


def make_config(
    base_config: gp2d.GP2DConfig,
    *,
    lengthscale: float,
    lengthscale_y: float | None = None,
    angle_degrees: float = 0.0,
) -> gp2d.GP2DConfig:
    return gp2d.GP2DConfig(
        mean_temp=base_config.mean_temp,
        signal_sd=base_config.signal_sd,
        lengthscale=lengthscale,
        lengthscale_y=lengthscale_y,
        angle_degrees=angle_degrees,
        noise_sd=base_config.noise_sd,
        relative_jitter=base_config.relative_jitter,
    )


def select_anisotropic_by_unsat_mll(
    data: dict[str, object],
    base_config: gp2d.GP2DConfig,
    *,
    observation_seed: int,
    shape_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    x_obs = data["x_obs"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)

    x_unsat = x_obs[~sat_mask]
    y_unsat = T_obs[~sat_mask]

    ell_x_values = [0.55, 0.75, 1.00, 1.35, 1.75]
    ell_y_values = [0.18, 0.28, 0.42, 0.65, 0.90]
    angle_values = [-55.0, -35.0, -15.0, 0.0, 15.0, 35.0, 55.0]

    rows = []
    best = None
    for ell_x in ell_x_values:
        for ell_y in ell_y_values:
            for angle in angle_values:
                config = make_config(
                    base_config,
                    lengthscale=ell_x,
                    lengthscale_y=ell_y,
                    angle_degrees=angle,
                )
                nll = shape_density.unsat_negative_log_marginal_likelihood(x_unsat, y_unsat, config)
                row = {
                    "shape": shape_name,
                    "shape_label": SHAPE_LABELS[shape_name],
                    "observation_seed": observation_seed,
                    "lengthscale": ell_x,
                    "lengthscale_y": ell_y,
                    "angle_degrees": angle,
                    "unsat_neg_log_marginal_likelihood": nll,
                }
                rows.append(row)
                if best is None or nll < best["unsat_neg_log_marginal_likelihood"]:
                    best = row

    assert best is not None
    params = {
        "lengthscale": float(best["lengthscale"]),
        "lengthscale_y": float(best["lengthscale_y"]),
        "angle_degrees": float(best["angle_degrees"]),
    }
    return params, pd.DataFrame(rows)


def config_for_kernel(
    kernel_name: str,
    shape_name: str,
    seed: int,
    base_config: gp2d.GP2DConfig,
    obs_n: int,
    pred_n: int,
) -> tuple[gp2d.GP2DConfig, dict[str, float], pd.DataFrame | None]:
    probe_data = shape_density.make_observations(
        base_config,
        shape_name,
        obs_n=obs_n,
        pred_n=pred_n,
        seed=seed,
    )

    if kernel_name == "isotropic_fixed":
        params = {"lengthscale": 0.70, "lengthscale_y": np.nan, "angle_degrees": 0.0}
        return make_config(base_config, lengthscale=0.70), params, None

    if kernel_name == "isotropic_tuned":
        ell_hat = shape_density.select_lengthscale_by_unsat_mll(probe_data, base_config)
        params = {"lengthscale": ell_hat, "lengthscale_y": np.nan, "angle_degrees": 0.0}
        return make_config(base_config, lengthscale=ell_hat), params, None

    if kernel_name == "anisotropic_tuned":
        params, grid = select_anisotropic_by_unsat_mll(
            probe_data,
            base_config,
            observation_seed=seed,
            shape_name=shape_name,
        )
        return make_config(base_config, **params), params, grid

    if kernel_name == "anisotropic_informed":
        params = INFORMED_KERNELS[shape_name]
        return make_config(base_config, **params), dict(params), None

    raise ValueError(f"Unknown kernel {kernel_name!r}")


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peak_interval_width"] = df["peak_ci_upper"] - df["peak_ci_lower"]
    grouped = df.groupby(["kernel", "shape", "shape_label", "method"], sort=False)
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
        lengthscale_mean=("kernel_lengthscale", "mean"),
        lengthscale_y_mean=("kernel_lengthscale_y", "mean"),
        angle_degrees_mean=("kernel_angle_degrees", "mean"),
    ).reset_index()
    summary["kernel_label"] = summary["kernel"].map(KERNEL_LABELS)
    return summary


def plot_kernel_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    shapes = list(SHAPE_LABELS)
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.7 * len(shapes), 4.7), sharey=True, constrained_layout=True)
    if len(shapes) == 1:
        axes = [axes]

    x = np.arange(len(METHOD_ORDER))
    width = 0.18
    offsets = {
        kernel: (idx - (len(KERNEL_ORDER) - 1) / 2.0) * width
        for idx, kernel in enumerate(KERNEL_ORDER)
    }

    for ax, shape_name in zip(axes, shapes):
        sub = summary[summary["shape"] == shape_name]
        for kernel in KERNEL_ORDER:
            values = []
            errors = []
            for method in METHOD_ORDER:
                row = sub[(sub["kernel"] == kernel) & (sub["method"] == method)]
                values.append(float(row.iloc[0][metric]) if len(row) else np.nan)
                std_metric = metric.replace("_mean", "_std")
                errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns and len(row) else 0.0)
            show_errors = metric.endswith("_mean") and any(np.isfinite(errors))
            ax.bar(
                x + offsets[kernel],
                values,
                width=width,
                yerr=errors if show_errors else None,
                capsize=1.8 if show_errors else 0,
                color=KERNEL_COLORS[kernel],
                alpha=0.78,
                label=KERNEL_LABELS[kernel],
            )
        ax.set_title(SHAPE_LABELS[shape_name])
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sampled_kernel_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    method = "censored sampled"
    sub = summary[summary["method"] == method]
    shapes = list(SHAPE_LABELS)
    x = np.arange(len(shapes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for idx, kernel in enumerate(KERNEL_ORDER):
        values = []
        errors = []
        for shape_name in shapes:
            row = sub[(sub["kernel"] == kernel) & (sub["shape"] == shape_name)]
            values.append(float(row.iloc[0][metric]) if len(row) else np.nan)
            std_metric = metric.replace("_mean", "_std")
            errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns and len(row) else 0.0)
        show_errors = metric.endswith("_mean") and any(np.isfinite(errors))
        ax.bar(
            x + (idx - (len(KERNEL_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            yerr=errors if show_errors else None,
            capsize=2 if show_errors else 0,
            color=KERNEL_COLORS[kernel],
            alpha=0.78,
            label=KERNEL_LABELS[kernel],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHAPE_LABELS[name] for name in shapes])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: censored sampled GP")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_anisotropic_parameters(results: pd.DataFrame, out_path: Path) -> None:
    tuned = results[results["kernel"] == "anisotropic_tuned"].drop_duplicates(["shape", "observation_seed"])
    shapes = list(SHAPE_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, column, ylabel in zip(
        axes,
        ["kernel_lengthscale", "kernel_lengthscale_y", "kernel_angle_degrees"],
        [r"$\ell_x$", r"$\ell_y$", r"angle"],
    ):
        data = [tuned[tuned["shape"] == shape][column].to_numpy() for shape in shapes]
        box = ax.boxplot(data, tick_labels=[SHAPE_LABELS[s] for s in shapes], patch_artist=True, showmeans=True)
        for patch in box["boxes"]:
            patch.set_facecolor(KERNEL_COLORS["anisotropic_tuned"])
            patch.set_alpha(0.45)
            patch.set_edgecolor(KERNEL_COLORS["anisotropic_tuned"])
        for median in box["medians"]:
            median.set_color("black")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Anisotropic kernel selected from unsaturated observations")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def grid(values: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(values).reshape(n, n)


def plot_reconstruction_comparison(
    shape_name: str,
    representative: dict[str, tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]],
    out_path: Path,
) -> None:
    first_data = next(iter(representative.values()))[1]
    pred_xs = first_data["pred_xs"]
    pred_ys = first_data["pred_ys"]
    obs_xs = first_data["obs_xs"]
    obs_ys = first_data["obs_ys"]
    x_obs = first_data["x_obs"]
    T_obs = first_data["T_obs"]
    sat_mask = first_data["sat_mask"]
    T_true_pred = first_data["T_true_pred"]
    threshold = first_data["threshold"]
    assert isinstance(pred_xs, np.ndarray)
    assert isinstance(pred_ys, np.ndarray)
    assert isinstance(obs_xs, np.ndarray)
    assert isinstance(obs_ys, np.ndarray)
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(T_true_pred, np.ndarray)
    assert isinstance(threshold, float)

    pred_n = len(pred_xs)
    obs_n = int(np.sqrt(len(T_obs)))
    extent = [float(pred_xs[0]), float(pred_xs[-1]), float(pred_ys[0]), float(pred_ys[-1])]
    obs_extent = [float(obs_xs[0]), float(obs_xs[-1]), float(obs_ys[0]), float(obs_ys[-1])]
    vmin = float(np.min(T_true_pred))
    vmax = float(np.max(T_true_pred))

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), constrained_layout=True)
    panels = [
        ("true field", T_true_pred),
        ("clipped obs.", T_obs),
        *[(KERNEL_LABELS[k], representative[k][0]["censored sampled"][0]) for k in KERNEL_ORDER],
    ]
    for ax, (title, values) in zip(axes.ravel(), panels):
        if title == "clipped obs.":
            im = ax.imshow(grid(values, obs_n), origin="lower", extent=obs_extent, cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(grid(values, pred_n), origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(
            grid(T_true_pred, pred_n),
            levels=[threshold],
            origin="lower",
            extent=extent,
            colors="white",
            linewidths=1.0,
            linestyles="--",
        )
        if title != "true field":
            ax.scatter(x_obs[sat_mask, 0], x_obs[sat_mask, 1], facecolor="none", edgecolor="red", s=14, linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.82, label="temperature")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs" / "gp2d_kernel_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = gp2d.GP2DConfig(lengthscale=0.70)
    obs_n = 17
    pred_n = 30
    seeds = list(range(10))
    frames = []
    grid_frames = []
    representative: dict[str, dict[str, tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]]] = {
        shape_name: {} for shape_name in SHAPE_LABELS
    }

    for shape_index, shape_name in enumerate(SHAPE_LABELS):
        for seed in seeds:
            for kernel_index, kernel_name in enumerate(KERNEL_ORDER):
                config, params, grid = config_for_kernel(
                    kernel_name,
                    shape_name,
                    seed,
                    base_config,
                    obs_n,
                    pred_n,
                )
                data = shape_density.make_observations(
                    config,
                    shape_name,
                    obs_n=obs_n,
                    pred_n=pred_n,
                    seed=seed,
                )
                print(
                    f"{SHAPE_LABELS[shape_name]}, seed={seed:02d}, {KERNEL_LABELS[kernel_name]}: "
                    f"ell_x={params['lengthscale']:.3f}, "
                    f"ell_y={params['lengthscale_y'] if np.isfinite(params['lengthscale_y']) else np.nan:.3f}, "
                    f"angle={params['angle_degrees']:.1f}"
                )
                predictions = shape_density.run_methods(
                    config,
                    data,
                    sampler_seed=70_000 + 2_000 * shape_index + 100 * kernel_index + seed,
                )
                metrics = shape_density.metrics_for_run(
                    predictions,
                    data,
                    config,
                    shape_name,
                    obs_n,
                    selected_lengthscale=float(params["lengthscale"]),
                )
                metrics["kernel"] = kernel_name
                metrics["kernel_label"] = KERNEL_LABELS[kernel_name]
                metrics["observation_seed"] = seed
                metrics["kernel_lengthscale"] = float(params["lengthscale"])
                metrics["kernel_lengthscale_y"] = float(params["lengthscale_y"])
                metrics["kernel_angle_degrees"] = float(params["angle_degrees"])
                frames.append(metrics)
                if grid is not None:
                    grid["kernel"] = kernel_name
                    grid_frames.append(grid)
                if seed == 0:
                    representative[shape_name][kernel_name] = (predictions, data)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results)

    results_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    if grid_frames:
        pd.concat(grid_frames, ignore_index=True).to_csv(out_dir / "anisotropic_mll_grid.csv", index=False)

    plot_kernel_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "field_error.png")
    plot_kernel_metric(summary, "hot_region_rel_l2_mean", "Relative L2 error in hot region", out_dir / "hot_region_error.png")
    plot_kernel_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "peak_error.png")
    plot_kernel_metric(summary, "true_peak_coverage", "True peak inside 95% interval", out_dir / "peak_coverage.png", ylim=(0.0, 1.05))
    plot_sampled_kernel_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "sampled_field_error.png")
    plot_sampled_kernel_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "sampled_peak_error.png")
    plot_anisotropic_parameters(results, out_dir / "anisotropic_selected_parameters.png")
    for shape_name, shape_representative in representative.items():
        plot_reconstruction_comparison(
            shape_name,
            shape_representative,
            out_dir / f"sampled_reconstructions_{shape_name}_seed00.png",
        )

    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(
        summary[
            [
                "kernel_label",
                "shape_label",
                "method",
                "field_rel_l2_mean",
                "hot_region_rel_l2_mean",
                "peak_abs_error_mean",
                "true_peak_coverage",
                "lengthscale_mean",
                "lengthscale_y_mean",
                "angle_degrees_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
