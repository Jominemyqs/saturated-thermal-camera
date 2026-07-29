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


def load_gp2d_module():
    module_path = ROOT / "scripts" / "10_gp_2d_censored.py"
    spec = importlib.util.spec_from_file_location("gp2d_censored", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp2d_censored"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gp2d = load_gp2d_module()


METHOD_ORDER = gp2d.METHOD_ORDER
COLORS = gp2d.COLORS


SHAPE_LABELS = {
    "axis_gaussian": "axis Gaussian",
    "rotated_wake": "rotated wake",
    "laser_path": "moving-laser path",
}


def axis_gaussian(points: np.ndarray) -> np.ndarray:
    return gp2d.true_temperature_2d(points)


def rotated_wake(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    T0 = 473.15
    theta = np.deg2rad(34.0)
    xc = 0.25
    yc = -0.10
    xr = np.cos(theta) * (x - xc) + np.sin(theta) * (y - yc)
    yr = -np.sin(theta) * (x - xc) + np.cos(theta) * (y - yc)
    hotspot = 760.0 * np.exp(-0.5 * ((xr / 0.62) ** 2 + (yr / 0.34) ** 2))

    wxc = -0.70
    wyc = -0.58
    wxr = np.cos(theta) * (x - wxc) + np.sin(theta) * (y - wyc)
    wyr = -np.sin(theta) * (x - wxc) + np.cos(theta) * (y - wyc)
    wake = 430.0 * np.exp(-0.5 * ((wxr / 1.70) ** 2 + (wyr / 0.25) ** 2))
    return T0 + hotspot + wake


def laser_path(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    T0 = 473.15
    centers_x = np.linspace(-2.0, 2.0, 70)
    centers_y = 0.42 * np.sin(1.25 * centers_x) - 0.12
    heat_profile = 0.45 + 0.55 * np.exp(-0.5 * ((centers_x - 0.45) / 0.62) ** 2)
    dx = x[:, None] - centers_x[None, :]
    dy = y[:, None] - centers_y[None, :]
    tube = np.max(heat_profile[None, :] * np.exp(-0.5 * (dx**2 + dy**2) / 0.18**2), axis=1)
    endpoint = 0.42 * np.exp(-0.5 * (((x - 0.75) / 0.36) ** 2 + ((y - 0.18) / 0.30) ** 2))
    return T0 + 790.0 * (tube + endpoint)


SHAPE_FUNCTIONS = {
    "axis_gaussian": axis_gaussian,
    "rotated_wake": rotated_wake,
    "laser_path": laser_path,
}


def make_observations(
    config: gp2d.GP2DConfig,
    shape_name: str,
    obs_n: int,
    pred_n: int = 30,
    frac_saturated: float = 0.03,
    seed: int = 12,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_obs, obs_xs, obs_ys = gp2d.make_grid(obs_n)
    x_pred, pred_xs, pred_ys = gp2d.make_grid(pred_n)
    truth_fn = SHAPE_FUNCTIONS[shape_name]
    T_true_obs = truth_fn(x_obs)
    T_true_pred = truth_fn(x_pred)
    threshold = float(np.quantile(T_true_obs, 1.0 - frac_saturated))
    T_meas = T_true_obs + rng.normal(0.0, config.noise_sd, size=T_true_obs.shape)
    sat_mask = T_meas >= threshold
    T_obs = np.minimum(T_meas, threshold)
    return {
        "shape": shape_name,
        "x_obs": x_obs,
        "x_pred": x_pred,
        "obs_xs": obs_xs,
        "obs_ys": obs_ys,
        "pred_xs": pred_xs,
        "pred_ys": pred_ys,
        "T_true_obs": T_true_obs,
        "T_true_pred": T_true_pred,
        "T_obs": T_obs,
        "sat_mask": sat_mask,
        "threshold": threshold,
    }


def unsat_negative_log_marginal_likelihood(
    x_unsat: np.ndarray,
    y_unsat: np.ndarray,
    config: gp2d.GP2DConfig,
) -> float:
    if len(x_unsat) == 0:
        return np.inf
    K = gp2d.rbf_kernel(x_unsat, x_unsat, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += config.noise_sd**2 + jitter
    cf = cho_factor(K, lower=True, check_finite=False)
    centered = y_unsat - config.mean_temp
    alpha = cho_solve(cf, centered, check_finite=False)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    return 0.5 * float(np.dot(centered, alpha)) + 0.5 * log_det + 0.5 * len(x_unsat) * np.log(2.0 * np.pi)


def select_lengthscale_by_unsat_mll(
    data: dict[str, object],
    base_config: gp2d.GP2DConfig,
    lower: float = 0.25,
    upper: float = 2.2,
) -> float:
    x_obs = data["x_obs"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)

    x_unsat = x_obs[~sat_mask]
    y_unsat = T_obs[~sat_mask]

    def objective(log_ell: float) -> float:
        config = gp2d.GP2DConfig(
            mean_temp=base_config.mean_temp,
            signal_sd=base_config.signal_sd,
            lengthscale=float(np.exp(log_ell)),
            noise_sd=base_config.noise_sd,
            relative_jitter=base_config.relative_jitter,
        )
        return unsat_negative_log_marginal_likelihood(x_unsat, y_unsat, config)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        return base_config.lengthscale
    return float(np.exp(result.x))


def run_methods(
    config: gp2d.GP2DConfig,
    data: dict[str, object],
    sampler_seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x_obs = data["x_obs"]
    x_pred = data["x_pred"]
    T_true_obs = data["T_true_obs"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    threshold = data["threshold"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(x_pred, np.ndarray)
    assert isinstance(T_true_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(threshold, float)

    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    gp2d.add_normal_interval(predictions, "exact clipped", *gp2d.gp_predict_exact(x_obs, T_obs, x_pred, config))
    gp2d.add_normal_interval(
        predictions,
        "discard saturated",
        *gp2d.gp_predict_exact(x_obs[~sat_mask], T_obs[~sat_mask], x_pred, config),
    )
    lap_mean, lap_sd, _ = gp2d.fit_censored_gp_laplace(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    gp2d.add_normal_interval(predictions, "censored Laplace", lap_mean, lap_sd)
    predictions["censored sampled"] = gp2d.sample_censored_gp_ess(
        x_obs,
        T_obs,
        sat_mask,
        threshold,
        x_pred,
        config,
        n_samples=320,
        burn_in=220,
        thin=2,
        seed=sampler_seed,
    )
    gp2d.add_normal_interval(predictions, "oracle true", *gp2d.gp_predict_exact(x_obs, T_true_obs, x_pred, config))
    return predictions


def metrics_for_run(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    data: dict[str, object],
    config: gp2d.GP2DConfig,
    shape_name: str,
    obs_n: int,
    selected_lengthscale: float,
) -> pd.DataFrame:
    x_pred = data["x_pred"]
    T_true_pred = data["T_true_pred"]
    threshold = data["threshold"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_pred, np.ndarray)
    assert isinstance(T_true_pred, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(sat_mask, np.ndarray)

    rows = []
    for method in METHOD_ORDER:
        mean, sd, lower, upper = predictions[method]
        row = gp2d.compute_metrics(method, x_pred, T_true_pred, mean, threshold, sd, lower, upper)
        row["shape"] = shape_name
        row["shape_label"] = SHAPE_LABELS[shape_name]
        row["obs_n"] = obs_n
        row["n_observations"] = obs_n * obs_n
        row["n_saturated"] = int(np.sum(sat_mask))
        row["actual_frac_saturated"] = float(np.mean(sat_mask))
        row["selected_lengthscale"] = selected_lengthscale
        row["lengthscale"] = config.lengthscale
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_by_density(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    shapes = list(SHAPE_LABELS)
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.2 * len(shapes), 4.2), sharey=True, constrained_layout=True)
    if len(shapes) == 1:
        axes = [axes]
    for ax, shape_name in zip(axes, shapes):
        sub = df[df["shape"] == shape_name]
        for method in METHOD_ORDER:
            group = sub[sub["method"] == method].sort_values("n_observations")
            ax.plot(
                group["n_observations"],
                group[metric],
                marker="o",
                linewidth=1.8,
                color=COLORS[method],
                label=method,
            )
        ax.set_title(SHAPE_LABELS[shape_name])
        ax.set_xlabel("number of observations")
        ax.grid(True, alpha=0.25)
        if metric == "true_peak_in_95":
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 1.0])
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_selected_lengthscales(df: pd.DataFrame, out_path: Path) -> None:
    selection = df[["shape", "shape_label", "obs_n", "n_observations", "selected_lengthscale"]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for shape_name, group in selection.groupby("shape", sort=False):
        group = group.sort_values("n_observations")
        ax.plot(
            group["n_observations"],
            group["selected_lengthscale"],
            marker="o",
            linewidth=2.0,
            label=SHAPE_LABELS[shape_name],
        )
    ax.set_xlabel("number of observations")
    ax.set_ylabel(r"selected lengthscale $\ell$")
    ax.set_title("Lengthscale selected from unsaturated observations")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    base_config = gp2d.GP2DConfig(lengthscale=0.70)
    obs_sizes = [13, 17, 21]
    pred_n = 30
    frames = []
    representative_predictions = {}
    representative_data = {}

    for shape_index, shape_name in enumerate(SHAPE_LABELS):
        for obs_n in obs_sizes:
            probe_data = make_observations(base_config, shape_name, obs_n=obs_n, pred_n=pred_n)
            ell_hat = select_lengthscale_by_unsat_mll(probe_data, base_config)
            config = gp2d.GP2DConfig(
                mean_temp=base_config.mean_temp,
                signal_sd=base_config.signal_sd,
                lengthscale=ell_hat,
                noise_sd=base_config.noise_sd,
                relative_jitter=base_config.relative_jitter,
            )
            data = make_observations(config, shape_name, obs_n=obs_n, pred_n=pred_n)
            print(f"{SHAPE_LABELS[shape_name]}, obs_n={obs_n}: selected ell={ell_hat:.3f}")
            predictions = run_methods(config, data, sampler_seed=40_000 + 100 * shape_index + obs_n)
            frames.append(metrics_for_run(predictions, data, config, shape_name, obs_n, ell_hat))
            if obs_n == 17:
                representative_predictions[shape_name] = predictions
                representative_data[shape_name] = data

    results = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "gp2d_shape_density_results.csv"
    results.to_csv(csv_path, index=False)

    plot_metric_by_density(
        results,
        "field_rel_l2",
        "Relative L2 field error",
        out_dir / "gp2d_shape_density_field_error.png",
    )
    plot_metric_by_density(
        results,
        "hot_region_rel_l2",
        "Relative L2 error in hot region",
        out_dir / "gp2d_shape_density_hot_region_error.png",
    )
    plot_metric_by_density(
        results,
        "peak_abs_error",
        "Peak absolute error",
        out_dir / "gp2d_shape_density_peak_error.png",
    )
    plot_metric_by_density(
        results,
        "posterior_sd_at_true_peak",
        "Posterior SD at true peak",
        out_dir / "gp2d_shape_density_peak_sd.png",
    )
    plot_metric_by_density(
        results,
        "true_peak_in_95",
        "True peak inside 95% interval",
        out_dir / "gp2d_shape_density_peak_coverage.png",
    )
    plot_selected_lengthscales(results, out_dir / "gp2d_shape_density_selected_lengthscales.png")

    for shape_name, predictions in representative_predictions.items():
        gp2d.plot_reconstruction_panels(
            predictions,
            representative_data[shape_name],
            out_dir / f"gp2d_shape_density_reconstructions_{shape_name}.png",
        )
        gp2d.plot_sd_panels(
            predictions,
            representative_data[shape_name],
            out_dir / f"gp2d_shape_density_posterior_sd_{shape_name}.png",
        )

    print(f"Saved {csv_path}")
    print(f"Saved plots to {out_dir}")
    summary = results[
        [
            "shape_label",
            "obs_n",
            "method",
            "field_rel_l2",
            "hot_region_rel_l2",
            "peak_abs_error",
            "posterior_sd_at_true_peak",
            "true_peak_in_95",
            "selected_lengthscale",
        ]
    ]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
