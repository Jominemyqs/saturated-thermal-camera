from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar


def load_gp1d_module():
    module_path = ROOT / "scripts" / "08_gp_1d_censored.py"
    spec = importlib.util.spec_from_file_location("gp1d_censored", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp1d_censored"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gp1d = load_gp1d_module()


METHOD_ORDER = ["exact clipped", "discard saturated", "censored Laplace", "censored sampled", "oracle true"]
COLORS = {
    "exact clipped": "#D55E00",
    "discard saturated": "#CC79A7",
    "censored Laplace": "#009E73",
    "censored sampled": "#56B4E9",
    "oracle true": "#0072B2",
}


def unsat_negative_log_marginal_likelihood(
    x_unsat: np.ndarray,
    y_unsat: np.ndarray,
    config: gp1d.GPConfig,
) -> float:
    if len(x_unsat) == 0:
        return np.inf
    K = gp1d.rbf_kernel(x_unsat, x_unsat, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += config.noise_sd**2 + jitter
    cf = cho_factor(K, lower=True, check_finite=False)
    centered = y_unsat - config.mean_temp
    alpha = cho_solve(cf, centered, check_finite=False)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    return 0.5 * float(np.dot(centered, alpha)) + 0.5 * log_det + 0.5 * len(x_unsat) * np.log(2.0 * np.pi)


def select_lengthscale_by_unsat_mll(
    seed: int,
    candidate_lengthscales: np.ndarray,
    base_config: gp1d.GPConfig,
) -> tuple[float, pd.DataFrame]:
    probe_config = gp1d.GPConfig(
        mean_temp=base_config.mean_temp,
        signal_sd=base_config.signal_sd,
        lengthscale=base_config.lengthscale,
        noise_sd=base_config.noise_sd,
        relative_jitter=base_config.relative_jitter,
    )
    x_obs, _, _, _, T_obs, sat_mask, _, _ = gp1d.make_observations(probe_config, seed=seed)
    x_unsat = x_obs[~sat_mask]
    y_unsat = T_obs[~sat_mask]

    rows = []
    for ell in candidate_lengthscales:
        config = gp1d.GPConfig(
            mean_temp=base_config.mean_temp,
            signal_sd=base_config.signal_sd,
            lengthscale=float(ell),
            noise_sd=base_config.noise_sd,
            relative_jitter=base_config.relative_jitter,
        )
        rows.append(
            {
                "observation_seed": seed,
                "lengthscale": float(ell),
                "unsat_neg_log_marginal_likelihood": unsat_negative_log_marginal_likelihood(x_unsat, y_unsat, config),
            }
        )

    grid = pd.DataFrame(rows)

    def objective(log_ell: float) -> float:
        config = gp1d.GPConfig(
            mean_temp=base_config.mean_temp,
            signal_sd=base_config.signal_sd,
            lengthscale=float(np.exp(log_ell)),
            noise_sd=base_config.noise_sd,
            relative_jitter=base_config.relative_jitter,
        )
        return unsat_negative_log_marginal_likelihood(x_unsat, y_unsat, config)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(np.min(candidate_lengthscales))), float(np.log(np.max(candidate_lengthscales)))),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if result.success:
        selected_lengthscale = float(np.exp(result.x))
    else:
        best_idx = int(grid["unsat_neg_log_marginal_likelihood"].idxmin())
        selected_lengthscale = float(grid.loc[best_idx, "lengthscale"])
    return selected_lengthscale, grid


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    scenarios = df["scenario"].unique() if "scenario" in df.columns else ["default"]
    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario] if "scenario" in df.columns else df
        for method in METHOD_ORDER:
            group = scenario_df[scenario_df["method"] == method]
            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "n_seeds": int(group["observation_seed"].nunique()),
                    "field_rel_l2_mean": float(group["field_rel_l2"].mean()),
                    "field_rel_l2_std": float(group["field_rel_l2"].std(ddof=1)),
                    "hot_region_rel_l2_mean": float(group["hot_region_rel_l2"].mean()),
                    "peak_abs_error_mean": float(group["peak_abs_error"].mean()),
                    "posterior_sd_at_true_peak_mean": float(group["posterior_sd_at_true_peak"].mean()),
                    "peak_interval_width_mean": float((group["peak_ci_upper"] - group["peak_ci_lower"]).mean()),
                    "true_peak_coverage": float(group["true_peak_in_95"].mean()),
                    "selected_lengthscale_mean": float(group["lengthscale"].mean()),
                }
            )
    return pd.DataFrame(summary_rows)


def plot_metric_box(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, title_suffix: str = "") -> None:
    data = [df[df["method"] == method][metric].to_numpy() for method in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    box = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showmeans=True)
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[method])
    for median in box["medians"]:
        median.set_color("black")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across random seeds" + title_suffix)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_coverage(summary: pd.DataFrame, out_path: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    methods = summary["method"].tolist()
    values = summary["true_peak_coverage"].to_numpy()
    ax.bar(methods, values, color=[COLORS[m] for m in methods], alpha=0.75)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.2, label="nominal 95%")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction of seeds")
    ax.set_title("True peak inside 95% interval" + title_suffix)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_selected_lengthscales(selection: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.hist(selection["selected_lengthscale"], bins=np.geomspace(0.25, 2.2, 13), color="#0072B2", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"selected lengthscale $\ell$")
    ax.set_ylabel("number of seeds")
    ax.set_title("Lengthscale selected from unsaturated marginal likelihood")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mll_curves(grid: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for seed, group in grid.groupby("observation_seed"):
        group = group.sort_values("lengthscale")
        y = group["unsat_neg_log_marginal_likelihood"].to_numpy()
        ax.semilogx(group["lengthscale"], y - np.min(y), color="#0072B2", alpha=0.25, linewidth=1.2)
    mean_curve = (
        grid.groupby("lengthscale")["unsat_neg_log_marginal_likelihood"]
        .mean()
        .reset_index()
        .sort_values("lengthscale")
    )
    mean_y = mean_curve["unsat_neg_log_marginal_likelihood"].to_numpy()
    ax.semilogx(mean_curve["lengthscale"], mean_y - np.min(mean_y), color="black", linewidth=2.2, label="mean")
    ax.set_xlabel(r"candidate lengthscale $\ell$")
    ax.set_ylabel("relative negative log marginal likelihood")
    ax.set_title("Unsaturated-data lengthscale selection")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    base_config = gp1d.GPConfig()
    seeds = list(range(30))
    candidate_lengthscales = np.geomspace(0.25, 2.2, 28)

    all_results = []
    all_grids = []
    selected_rows = []

    for seed in seeds:
        fixed_config = gp1d.GPConfig(
            mean_temp=base_config.mean_temp,
            signal_sd=base_config.signal_sd,
            lengthscale=0.55,
            noise_sd=base_config.noise_sd,
            relative_jitter=base_config.relative_jitter,
        )
        print(f"Seed {seed:02d}: running fixed ell=0.550")
        _, _, fixed_results, _ = gp1d.run_methods(
            fixed_config,
            include_sampled=True,
            observation_seed=seed,
            sampler_seed=20_000 + seed,
            n_samples=700,
            burn_in=400,
            thin=2,
        )
        fixed_results["scenario"] = "fixed_ell_0.55"
        all_results.append(fixed_results)

        ell_hat, mll_grid = select_lengthscale_by_unsat_mll(seed, candidate_lengthscales, base_config)
        all_grids.append(mll_grid)
        selected_rows.append({"observation_seed": seed, "selected_lengthscale": ell_hat})
        config = gp1d.GPConfig(
            mean_temp=base_config.mean_temp,
            signal_sd=base_config.signal_sd,
            lengthscale=ell_hat,
            noise_sd=base_config.noise_sd,
            relative_jitter=base_config.relative_jitter,
        )
        print(f"Seed {seed:02d}: selected ell={ell_hat:.3f}")
        _, _, results, _ = gp1d.run_methods(
            config,
            include_sampled=True,
            observation_seed=seed,
            sampler_seed=10_000 + seed,
            n_samples=700,
            burn_in=400,
            thin=2,
        )
        results["scenario"] = "unsat_mll_tuned"
        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)
    selection_df = pd.DataFrame(selected_rows)
    mll_grid_df = pd.concat(all_grids, ignore_index=True)
    summary_df = summarize_results(results_df)

    results_path = out_dir / "gp1d_multiseed_results.csv"
    summary_path = out_dir / "gp1d_multiseed_summary.csv"
    selection_path = out_dir / "gp1d_hyperparameter_selection.csv"
    grid_path = out_dir / "gp1d_unsat_mll_grid.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    mll_grid_df.to_csv(grid_path, index=False)

    for scenario, scenario_df in results_df.groupby("scenario", sort=False):
        label = f" ({scenario.replace('_', ' ')})"
        plot_metric_box(
            scenario_df,
            "field_rel_l2",
            "Relative L2 field error",
            out_dir / f"gp1d_multiseed_field_error_{scenario}.png",
            title_suffix=label,
        )
        plot_metric_box(
            scenario_df,
            "peak_abs_error",
            "Peak absolute error",
            out_dir / f"gp1d_multiseed_peak_error_{scenario}.png",
            title_suffix=label,
        )
        plot_metric_box(
            scenario_df,
            "posterior_sd_at_true_peak",
            "Posterior SD at true peak",
            out_dir / f"gp1d_multiseed_peak_sd_{scenario}.png",
            title_suffix=label,
        )
        plot_coverage(
            summary_df[summary_df["scenario"] == scenario],
            out_dir / f"gp1d_multiseed_peak_coverage_{scenario}.png",
            title_suffix=label,
        )
    plot_selected_lengthscales(selection_df, out_dir / "gp1d_selected_lengthscales.png")
    plot_mll_curves(mll_grid_df, out_dir / "gp1d_unsat_mll_curves.png")

    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {selection_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
