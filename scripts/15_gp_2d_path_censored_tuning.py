from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal


def load_path_kernel_module():
    module_path = ROOT / "scripts" / "14_gp_2d_path_kernel.py"
    spec = importlib.util.spec_from_file_location("gp2d_path_kernel", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp2d_path_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


path_kernel = load_path_kernel_module()
shape_density = path_kernel.shape_density
gp2d = path_kernel.gp2d


METHOD_ORDER = shape_density.METHOD_ORDER
SHAPE_NAME = "laser_path"
TUNING_ORDER = ["path_fixed", "path_unsat_mll", "path_censored_mll"]
TUNING_LABELS = {
    "path_fixed": "fixed geometry",
    "path_unsat_mll": "unsat MLL tuned",
    "path_censored_mll": "censored MLL tuned",
}
TUNING_COLORS = {
    "path_fixed": "#009E73",
    "path_unsat_mll": "#56B4E9",
    "path_censored_mll": "#D55E00",
}
METHOD_COLORS = gp2d.COLORS
ELL_S_VALUES = [0.45, 0.70, 0.95, 1.25, 1.60, 2.10]
ELL_R_VALUES = [0.10, 0.14, 0.18, 0.24, 0.32, 0.45]


def make_path_config(base_config: gp2d.GP2DConfig, ell_s: float, ell_r: float) -> gp2d.GP2DConfig:
    return path_kernel.make_config(
        base_config,
        kernel="path_aligned",
        lengthscale=ell_s,
        lengthscale_y=ell_r,
    )


def log_mvn_pdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    cov = 0.5 * (cov + cov.T)
    cf = cho_factor(cov, lower=True, check_finite=False)
    centered = y - mean
    alpha = cho_solve(cf, centered, check_finite=False)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    return -0.5 * float(np.dot(centered, alpha)) - 0.5 * log_det - 0.5 * len(y) * np.log(2.0 * np.pi)


def event_logprob_above(mean: np.ndarray, cov: np.ndarray, threshold: float) -> float:
    if len(mean) == 0:
        return 0.0
    cov = 0.5 * (cov + cov.T)
    lower = np.full(len(mean), threshold)
    upper = np.full(len(mean), np.inf)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            logprob = float(
                multivariate_normal.logcdf(
                    upper,
                    mean=mean,
                    cov=cov,
                    allow_singular=False,
                    lower_limit=lower,
                    maxpts=250_000,
                    abseps=1e-8,
                    releps=1e-8,
                )
            )
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            logprob = float(
                multivariate_normal.logcdf(
                    -lower,
                    mean=-mean,
                    cov=cov,
                    allow_singular=True,
                    maxpts=250_000,
                    abseps=1e-8,
                    releps=1e-8,
                )
            )
    if np.isfinite(logprob):
        return logprob
    prob = float(
        multivariate_normal.cdf(
            upper,
            mean=mean,
            cov=cov,
            allow_singular=True,
            lower_limit=lower,
            maxpts=250_000,
            abseps=1e-8,
            releps=1e-8,
        )
    )
    return float(np.log(max(prob, np.finfo(float).tiny)))


def censored_negative_log_marginal_likelihood(
    data: dict[str, object],
    config: gp2d.GP2DConfig,
) -> float:
    x_obs = data["x_obs"]
    y_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    threshold = data["threshold"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(y_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(threshold, float)

    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]

    noise_var = config.noise_sd**2
    jitter = config.relative_jitter * config.signal_sd**2

    if len(x_unsat) == 0:
        K_sat = gp2d.rbf_kernel(x_sat, x_sat, config)
        K_sat[np.diag_indices_from(K_sat)] += noise_var + jitter
        mean_sat = np.full(len(x_sat), config.mean_temp)
        return -event_logprob_above(mean_sat, K_sat, threshold)

    K_uu = gp2d.rbf_kernel(x_unsat, x_unsat, config)
    K_uu[np.diag_indices_from(K_uu)] += noise_var + jitter
    mean_unsat = np.full(len(x_unsat), config.mean_temp)
    try:
        logp_unsat = log_mvn_pdf(y_unsat, mean_unsat, K_uu)
    except np.linalg.LinAlgError:
        return np.inf

    if len(x_sat) == 0:
        return -logp_unsat

    K_su = gp2d.rbf_kernel(x_sat, x_unsat, config)
    K_ss = gp2d.rbf_kernel(x_sat, x_sat, config)
    K_ss[np.diag_indices_from(K_ss)] += noise_var + jitter
    cf = cho_factor(K_uu, lower=True, check_finite=False)
    alpha = cho_solve(cf, y_unsat - config.mean_temp, check_finite=False)
    cond_mean = config.mean_temp + K_su.dot(alpha)
    solved = cho_solve(cf, K_su.T, check_finite=False)
    cond_cov = K_ss - K_su.dot(solved)
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    cond_cov[np.diag_indices_from(cond_cov)] += jitter

    logp_sat_event = event_logprob_above(cond_mean, cond_cov, threshold)
    if not np.isfinite(logp_sat_event):
        return np.inf
    return -(logp_unsat + logp_sat_event)


def unsat_negative_log_marginal_likelihood(data: dict[str, object], config: gp2d.GP2DConfig) -> float:
    x_obs = data["x_obs"]
    y_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(y_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    return shape_density.unsat_negative_log_marginal_likelihood(x_obs[~sat_mask], y_obs[~sat_mask], config)


def grid_search_path_hyperparameters(
    data: dict[str, object],
    base_config: gp2d.GP2DConfig,
    observation_seed: int,
    objective_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    best = None
    for ell_s in ELL_S_VALUES:
        for ell_r in ELL_R_VALUES:
            config = make_path_config(base_config, ell_s, ell_r)
            if objective_name == "unsat_mll":
                nll = unsat_negative_log_marginal_likelihood(data, config)
            elif objective_name == "censored_mll":
                nll = censored_negative_log_marginal_likelihood(data, config)
            else:
                raise ValueError(f"Unknown objective {objective_name!r}")
            row = {
                "observation_seed": observation_seed,
                "objective": objective_name,
                "lengthscale_s": ell_s,
                "lengthscale_r": ell_r,
                "negative_log_marginal_likelihood": nll,
            }
            rows.append(row)
            if np.isfinite(nll) and (best is None or nll < best["negative_log_marginal_likelihood"]):
                best = row
    if best is None:
        best = {"lengthscale_s": 1.10, "lengthscale_r": 0.20}
    params = {
        "lengthscale": float(best["lengthscale_s"]),
        "lengthscale_y": float(best["lengthscale_r"]),
    }
    return params, pd.DataFrame(rows)


def config_for_tuning_strategy(
    strategy: str,
    data: dict[str, object],
    seed: int,
    base_config: gp2d.GP2DConfig,
) -> tuple[gp2d.GP2DConfig, dict[str, float], pd.DataFrame | None]:
    if strategy == "path_fixed":
        params = {"lengthscale": 1.10, "lengthscale_y": 0.20}
        return make_path_config(base_config, params["lengthscale"], params["lengthscale_y"]), params, None

    if strategy == "path_unsat_mll":
        params, grid = grid_search_path_hyperparameters(data, base_config, seed, "unsat_mll")
        return make_path_config(base_config, params["lengthscale"], params["lengthscale_y"]), params, grid

    if strategy == "path_censored_mll":
        params, grid = grid_search_path_hyperparameters(data, base_config, seed, "censored_mll")
        return make_path_config(base_config, params["lengthscale"], params["lengthscale_y"]), params, grid

    raise ValueError(f"Unknown tuning strategy {strategy!r}")


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peak_interval_width"] = df["peak_ci_upper"] - df["peak_ci_lower"]
    grouped = df.groupby(["tuning_strategy", "tuning_label", "method"], sort=False)
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


def plot_method_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(METHOD_ORDER))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for idx, strategy in enumerate(TUNING_ORDER):
        sub = summary[summary["tuning_strategy"] == strategy]
        values = []
        errors = []
        for method in METHOD_ORDER:
            row = sub[sub["method"] == method]
            values.append(float(row.iloc[0][metric]) if len(row) else np.nan)
            std_metric = metric.replace("_mean", "_std")
            errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns and len(row) else 0.0)
        show_errors = metric.endswith("_mean") and any(np.isfinite(errors))
        ax.bar(
            x + (idx - (len(TUNING_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            yerr=errors if show_errors else None,
            capsize=2 if show_errors else 0,
            color=TUNING_COLORS[strategy],
            alpha=0.78,
            label=TUNING_LABELS[strategy],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Moving-laser path-aligned kernel: {ylabel}")
    ax.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(ncol=3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sampled_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    sub = summary[summary["method"] == "censored sampled"]
    x = np.arange(len(TUNING_ORDER))
    values = []
    errors = []
    for strategy in TUNING_ORDER:
        row = sub[sub["tuning_strategy"] == strategy]
        values.append(float(row.iloc[0][metric]))
        std_metric = metric.replace("_mean", "_std")
        errors.append(float(row.iloc[0][std_metric]) if std_metric in row.columns else 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.bar(
        x,
        values,
        yerr=errors if metric.endswith("_mean") else None,
        color=[TUNING_COLORS[s] for s in TUNING_ORDER],
        alpha=0.82,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([TUNING_LABELS[s] for s in TUNING_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Censored sampled GP: {ylabel}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_selected_parameters(selection: pd.DataFrame, out_path: Path) -> None:
    tuned = selection[selection["tuning_strategy"].isin(["path_unsat_mll", "path_censored_mll"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    for ax, column, ylabel in zip(axes, ["lengthscale_s", "lengthscale_r"], [r"along-path $\ell_s$", r"across-path $\ell_r$"]):
        positions = np.arange(2)
        data = [
            tuned[tuned["tuning_strategy"] == strategy][column].to_numpy()
            for strategy in ["path_unsat_mll", "path_censored_mll"]
        ]
        box = ax.boxplot(
            data,
            tick_labels=[TUNING_LABELS["path_unsat_mll"], TUNING_LABELS["path_censored_mll"]],
            patch_artist=True,
            showmeans=True,
        )
        for patch, strategy in zip(box["boxes"], ["path_unsat_mll", "path_censored_mll"]):
            patch.set_facecolor(TUNING_COLORS[strategy])
            patch.set_alpha(0.45)
            patch.set_edgecolor(TUNING_COLORS[strategy])
        ax.scatter(
            np.repeat(0, len(data[0])) + 0.06,
            data[0],
            s=18,
            color=TUNING_COLORS["path_unsat_mll"],
            alpha=0.7,
        )
        ax.scatter(
            np.repeat(1, len(data[1])) + 0.06,
            data[1],
            s=18,
            color=TUNING_COLORS["path_censored_mll"],
            alpha=0.7,
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(True, axis="y", alpha=0.25)
        _ = positions
    fig.suptitle("Path-aligned hyperparameters selected from different likelihoods")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_objective_grids(grid: pd.DataFrame, seed: int, out_path: Path) -> None:
    seed_grid = grid[grid["observation_seed"] == seed]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for ax, objective, title in zip(axes, ["unsat_mll", "censored_mll"], ["unsaturated MLL", "full censored MLL"]):
        sub = seed_grid[seed_grid["objective"] == objective].copy()
        pivot = sub.pivot(index="lengthscale_r", columns="lengthscale_s", values="negative_log_marginal_likelihood")
        values = pivot.to_numpy()
        values = values - np.nanmin(values)
        im = ax.imshow(values, origin="lower", cmap="magma_r", aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel(r"$\ell_s$")
        ax.set_ylabel(r"$\ell_r$")
        ax.set_title(title)
        best_idx = np.unravel_index(np.nanargmin(values), values.shape)
        ax.scatter(best_idx[1], best_idx[0], marker="*", s=160, color="#0072B2", edgecolor="white", linewidth=0.8)
        fig.colorbar(im, ax=ax, shrink=0.82, label="relative NLL")
    fig.suptitle(f"Path-kernel hyperparameter objectives, seed {seed}")
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

    panels = [
        ("true field", T_true_pred),
        ("clipped obs.", T_obs),
        *[(TUNING_LABELS[s], representative[s][0]["censored sampled"][0]) for s in TUNING_ORDER],
        ("oracle true", representative["path_fixed"][0]["oracle true"][0]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), constrained_layout=True)
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
    out_dir = ROOT / "outputs" / "gp2d_path_censored_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = gp2d.GP2DConfig(lengthscale=0.70)
    obs_n = 17
    pred_n = 30
    seeds = list(range(10))
    frames = []
    grid_frames = []
    selection_rows = []
    representative = {}

    for seed in seeds:
        base_data = shape_density.make_observations(
            base_config,
            SHAPE_NAME,
            obs_n=obs_n,
            pred_n=pred_n,
            seed=seed,
        )
        for strategy_index, strategy in enumerate(TUNING_ORDER):
            config, params, grid_df = config_for_tuning_strategy(strategy, base_data, seed, base_config)
            selection_rows.append(
                {
                    "observation_seed": seed,
                    "tuning_strategy": strategy,
                    "tuning_label": TUNING_LABELS[strategy],
                    "lengthscale_s": float(params["lengthscale"]),
                    "lengthscale_r": float(params["lengthscale_y"]),
                }
            )
            data = shape_density.make_observations(
                config,
                SHAPE_NAME,
                obs_n=obs_n,
                pred_n=pred_n,
                seed=seed,
            )
            print(
                f"seed={seed:02d}, {TUNING_LABELS[strategy]}: "
                f"ell_s={params['lengthscale']:.3f}, ell_r={params['lengthscale_y']:.3f}"
            )
            predictions = shape_density.run_methods(
                config,
                data,
                sampler_seed=100_000 + 100 * strategy_index + seed,
            )
            metrics = shape_density.metrics_for_run(
                predictions,
                data,
                config,
                SHAPE_NAME,
                obs_n,
                selected_lengthscale=float(params["lengthscale"]),
            )
            metrics["tuning_strategy"] = strategy
            metrics["tuning_label"] = TUNING_LABELS[strategy]
            metrics["observation_seed"] = seed
            metrics["kernel_type"] = config.kernel
            metrics["kernel_lengthscale"] = float(params["lengthscale"])
            metrics["kernel_lengthscale_y"] = float(params["lengthscale_y"])
            frames.append(metrics)
            if grid_df is not None:
                grid_df = grid_df.copy()
                grid_df["tuning_strategy"] = strategy
                grid_df["tuning_label"] = TUNING_LABELS[strategy]
                grid_frames.append(grid_df)
            if seed == 0:
                representative[strategy] = (predictions, data)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results)
    selection = pd.DataFrame(selection_rows)
    grid_df = pd.concat(grid_frames, ignore_index=True)

    results.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    selection.to_csv(out_dir / "selected_parameters.csv", index=False)
    grid_df.to_csv(out_dir / "tuning_grid.csv", index=False)

    plot_method_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "field_error.png")
    plot_method_metric(summary, "hot_region_rel_l2_mean", "Relative L2 error in hot region", out_dir / "hot_region_error.png")
    plot_method_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "peak_error.png")
    plot_method_metric(summary, "true_peak_coverage", "True peak inside 95% interval", out_dir / "peak_coverage.png", ylim=(0.0, 1.05))
    plot_sampled_metric(summary, "field_rel_l2_mean", "Relative L2 field error", out_dir / "sampled_field_error.png")
    plot_sampled_metric(summary, "peak_abs_error_mean", "Peak absolute error", out_dir / "sampled_peak_error.png")
    plot_selected_parameters(selection, out_dir / "selected_parameters.png")
    plot_objective_grids(grid_df, seed=0, out_path=out_dir / "objective_grids_seed00.png")
    plot_reconstruction_comparison(representative, out_dir / "sampled_reconstructions_seed00.png")

    print(f"Saved outputs to {out_dir}")
    print(
        summary[
            [
                "tuning_label",
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
