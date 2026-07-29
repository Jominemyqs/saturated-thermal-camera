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
SHAPE_NAME = "laser_path"
EXTRA_RECONSTRUCTION_SHAPES = ["axis_gaussian", "rotated_wake"]

KERNEL_ORDER = ["isotropic_fixed", "anisotropic_informed", "path_aligned_fixed", "path_aligned_tuned"]
KERNEL_LABELS = {
    "isotropic_fixed": "isotropic fixed",
    "anisotropic_informed": "global anisotropic",
    "path_aligned_fixed": "path-aligned fixed",
    "path_aligned_tuned": "path-aligned tuned",
}
KERNEL_COLORS = {
    "isotropic_fixed": "#999999",
    "anisotropic_informed": "#D55E00",
    "path_aligned_fixed": "#009E73",
    "path_aligned_tuned": "#56B4E9",
}
GLOBAL_ANISOTROPIC_KERNELS = {
    "axis_gaussian": {"lengthscale": 0.75, "lengthscale_y": 0.48, "angle_degrees": 0.0},
    "rotated_wake": {"lengthscale": 1.25, "lengthscale_y": 0.28, "angle_degrees": 34.0},
    "laser_path": {"lengthscale": 1.10, "lengthscale_y": 0.20, "angle_degrees": 0.0},
}


def make_config(
    base_config: gp2d.GP2DConfig,
    *,
    kernel: str = "rbf",
    lengthscale: float,
    lengthscale_y: float | None = None,
    angle_degrees: float = 0.0,
) -> gp2d.GP2DConfig:
    return gp2d.GP2DConfig(
        mean_temp=base_config.mean_temp,
        signal_sd=base_config.signal_sd,
        kernel=kernel,
        lengthscale=lengthscale,
        lengthscale_y=lengthscale_y,
        angle_degrees=angle_degrees,
        noise_sd=base_config.noise_sd,
        relative_jitter=base_config.relative_jitter,
    )


def select_path_kernel_by_unsat_mll(
    data: dict[str, object],
    base_config: gp2d.GP2DConfig,
    observation_seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    x_obs = data["x_obs"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)

    x_unsat = x_obs[~sat_mask]
    y_unsat = T_obs[~sat_mask]
    ell_s_values = [0.45, 0.70, 0.95, 1.25, 1.60, 2.10]
    ell_r_values = [0.10, 0.14, 0.18, 0.24, 0.32, 0.45]

    rows = []
    best = None
    for ell_s in ell_s_values:
        for ell_r in ell_r_values:
            config = make_config(
                base_config,
                kernel="path_aligned",
                lengthscale=ell_s,
                lengthscale_y=ell_r,
            )
            nll = shape_density.unsat_negative_log_marginal_likelihood(x_unsat, y_unsat, config)
            row = {
                "observation_seed": observation_seed,
                "kernel": "path_aligned_tuned",
                "lengthscale_s": ell_s,
                "lengthscale_r": ell_r,
                "unsat_neg_log_marginal_likelihood": nll,
            }
            rows.append(row)
            if best is None or nll < best["unsat_neg_log_marginal_likelihood"]:
                best = row

    assert best is not None
    params = {
        "lengthscale": float(best["lengthscale_s"]),
        "lengthscale_y": float(best["lengthscale_r"]),
    }
    return params, pd.DataFrame(rows)


def config_for_kernel(
    kernel_name: str,
    seed: int,
    base_config: gp2d.GP2DConfig,
    obs_n: int,
    pred_n: int,
    shape_name: str = SHAPE_NAME,
) -> tuple[gp2d.GP2DConfig, dict[str, float], pd.DataFrame | None]:
    if kernel_name == "isotropic_fixed":
        params = {"lengthscale": 0.70, "lengthscale_y": np.nan}
        return make_config(base_config, lengthscale=0.70), params, None

    if kernel_name == "anisotropic_informed":
        params = GLOBAL_ANISOTROPIC_KERNELS[shape_name]
        return make_config(base_config, **params), dict(params), None

    if kernel_name == "path_aligned_fixed":
        params = {"lengthscale": 1.10, "lengthscale_y": 0.20}
        return make_config(base_config, kernel="path_aligned", lengthscale=1.10, lengthscale_y=0.20), params, None

    if kernel_name == "path_aligned_tuned":
        probe_data = shape_density.make_observations(
            base_config,
            shape_name,
            obs_n=obs_n,
            pred_n=pred_n,
            seed=seed,
        )
        params, grid = select_path_kernel_by_unsat_mll(probe_data, base_config, seed)
        return make_config(base_config, kernel="path_aligned", **params), params, grid

    raise ValueError(f"Unknown kernel {kernel_name!r}")


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peak_interval_width"] = df["peak_ci_upper"] - df["peak_ci_lower"]
    grouped = df.groupby(["kernel", "kernel_label", "method"], sort=False)
    return grouped.agg(
        n_seeds=("observation_seed", "nunique"),
        field_rel_l2_mean=("field_rel_l2", "mean"),
        field_rel_l2_std=("field_rel_l2", "std"),
        hot_region_rel_l2_mean=("hot_region_rel_l2", "mean"),
        peak_abs_error_mean=("peak_abs_error", "mean"),
        peak_abs_error_std=("peak_abs_error", "std"),
        posterior_sd_at_true_peak_mean=("posterior_sd_at_true_peak", "mean"),
        peak_interval_width_mean=("peak_interval_width", "mean"),
        true_peak_coverage=("true_peak_in_95", "mean"),
        lengthscale_s_mean=("kernel_lengthscale", "mean"),
        lengthscale_r_mean=("kernel_lengthscale_y", "mean"),
    ).reset_index()


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(METHOD_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    for idx, kernel in enumerate(KERNEL_ORDER):
        sub = summary[summary["kernel"] == kernel]
        values = []
        errors = []
        for method in METHOD_ORDER:
            row = sub[sub["method"] == method]
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
    ax.set_xticklabels(METHOD_ORDER, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Moving-laser path: {ylabel}")
    ax.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sampled_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    sub = summary[summary["method"] == "censored sampled"]
    x = np.arange(len(KERNEL_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    values = []
    errors = []
    for kernel in KERNEL_ORDER:
        row = sub[sub["kernel"] == kernel]
        values.append(float(row.iloc[0][metric]))
        std_metric = metric.replace("_mean", "_std")
        errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns else 0.0)
    ax.bar(
        x,
        values,
        yerr=errors if metric.endswith("_mean") else None,
        color=[KERNEL_COLORS[k] for k in KERNEL_ORDER],
        alpha=0.80,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_LABELS[k] for k in KERNEL_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Censored sampled GP: {ylabel}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def grid(values: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(values).reshape(n, n)


def plot_reconstruction_comparison(
    representative: dict[str, tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]],
    out_path: Path,
) -> None:
    first_data = next(iter(representative.values()))[1]
    pred_xs = first_data["pred_xs"]
    pred_ys = first_data["pred_ys"]
    x_obs = first_data["x_obs"]
    T_obs = first_data["T_obs"]
    sat_mask = first_data["sat_mask"]
    T_true_pred = first_data["T_true_pred"]
    threshold = first_data["threshold"]
    assert isinstance(pred_xs, np.ndarray)
    assert isinstance(pred_ys, np.ndarray)
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(T_true_pred, np.ndarray)
    assert isinstance(threshold, float)

    pred_n = len(pred_xs)
    obs_n = int(np.sqrt(len(T_obs)))
    extent = [float(pred_xs[0]), float(pred_xs[-1]), float(pred_ys[0]), float(pred_ys[-1])]
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
            obs_xs = first_data["obs_xs"]
            obs_ys = first_data["obs_ys"]
            assert isinstance(obs_xs, np.ndarray)
            assert isinstance(obs_ys, np.ndarray)
            obs_extent = [float(obs_xs[0]), float(obs_xs[-1]), float(obs_ys[0]), float(obs_ys[-1])]
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


def make_representative_reconstructions(
    shape_name: str,
    base_config: gp2d.GP2DConfig,
    obs_n: int,
    pred_n: int,
    out_dir: Path,
    *,
    seed: int = 0,
) -> None:
    representative = {}
    for kernel_index, kernel_name in enumerate(KERNEL_ORDER):
        config, params, _ = config_for_kernel(kernel_name, seed, base_config, obs_n, pred_n, shape_name=shape_name)
        data = shape_density.make_observations(
            config,
            shape_name,
            obs_n=obs_n,
            pred_n=pred_n,
            seed=seed,
        )
        print(
            f"{shape_density.SHAPE_LABELS[shape_name]} reconstruction, {KERNEL_LABELS[kernel_name]}: "
            f"ell_s={params['lengthscale']:.3f}, "
            f"ell_r={params['lengthscale_y'] if np.isfinite(params['lengthscale_y']) else np.nan:.3f}"
        )
        predictions = shape_density.run_methods(
            config,
            data,
            sampler_seed=90_000 + 2_000 * list(shape_density.SHAPE_LABELS).index(shape_name) + 100 * kernel_index + seed,
        )
        representative[kernel_name] = (predictions, data)
    plot_reconstruction_comparison(
        representative,
        out_dir / f"sampled_reconstructions_{shape_name}_seed00.png",
    )


def plot_path_mll_grid(grid_df: pd.DataFrame, out_path: Path) -> None:
    best_by_seed = grid_df.loc[grid_df.groupby("observation_seed")["unsat_neg_log_marginal_likelihood"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].hist(best_by_seed["lengthscale_s"], bins=np.arange(0.35, 2.31, 0.25), color=KERNEL_COLORS["path_aligned_tuned"], alpha=0.75)
    axes[0].set_xlabel(r"selected along-path $\ell_s$")
    axes[0].set_ylabel("number of seeds")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].hist(best_by_seed["lengthscale_r"], bins=np.arange(0.08, 0.50, 0.05), color=KERNEL_COLORS["path_aligned_tuned"], alpha=0.75)
    axes[1].set_xlabel(r"selected across-path $\ell_r$")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Path-aligned kernel selected from unsaturated observations")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs" / "gp2d_path_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = gp2d.GP2DConfig(lengthscale=0.70)
    obs_n = 17
    pred_n = 30
    seeds = list(range(10))
    frames = []
    grid_frames = []
    representative = {}

    for seed in seeds:
        for kernel_index, kernel_name in enumerate(KERNEL_ORDER):
            config, params, grid_df = config_for_kernel(kernel_name, seed, base_config, obs_n, pred_n, shape_name=SHAPE_NAME)
            data = shape_density.make_observations(
                config,
                SHAPE_NAME,
                obs_n=obs_n,
                pred_n=pred_n,
                seed=seed,
            )
            print(
                f"seed={seed:02d}, {KERNEL_LABELS[kernel_name]}: "
                f"ell_s={params['lengthscale']:.3f}, ell_r={params['lengthscale_y'] if np.isfinite(params['lengthscale_y']) else np.nan:.3f}"
            )
            predictions = shape_density.run_methods(
                config,
                data,
                sampler_seed=80_000 + 100 * kernel_index + seed,
            )
            metrics = shape_density.metrics_for_run(
                predictions,
                data,
                config,
                SHAPE_NAME,
                obs_n,
                selected_lengthscale=float(params["lengthscale"]),
            )
            metrics["kernel"] = kernel_name
            metrics["kernel_label"] = KERNEL_LABELS[kernel_name]
            metrics["observation_seed"] = seed
            metrics["kernel_type"] = config.kernel
            metrics["kernel_lengthscale"] = float(params["lengthscale"])
            metrics["kernel_lengthscale_y"] = float(params["lengthscale_y"])
            frames.append(metrics)
            if grid_df is not None:
                grid_frames.append(grid_df)
            if seed == 0:
                representative[kernel_name] = (predictions, data)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results)
    results_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    if grid_frames:
        grid_df = pd.concat(grid_frames, ignore_index=True)
        grid_df.to_csv(out_dir / "path_mll_grid.csv", index=False)
        plot_path_mll_grid(grid_df, out_dir / "path_selected_parameters.png")

    plot_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "field_error.png")
    plot_metric(summary, "hot_region_rel_l2_mean", "Relative L2 error in hot region", out_dir / "hot_region_error.png")
    plot_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "peak_error.png")
    plot_metric(summary, "true_peak_coverage", "True peak inside 95% interval", out_dir / "peak_coverage.png", ylim=(0.0, 1.05))
    plot_sampled_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "sampled_field_error.png")
    plot_sampled_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "sampled_peak_error.png")
    plot_reconstruction_comparison(representative, out_dir / "sampled_reconstructions_seed00.png")
    plot_reconstruction_comparison(representative, out_dir / f"sampled_reconstructions_{SHAPE_NAME}_seed00.png")
    for shape_name in EXTRA_RECONSTRUCTION_SHAPES:
        make_representative_reconstructions(shape_name, base_config, obs_n, pred_n, out_dir)

    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(
        summary[
            [
                "kernel_label",
                "method",
                "field_rel_l2_mean",
                "hot_region_rel_l2_mean",
                "peak_abs_error_mean",
                "true_peak_coverage",
                "lengthscale_s_mean",
                "lengthscale_r_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
