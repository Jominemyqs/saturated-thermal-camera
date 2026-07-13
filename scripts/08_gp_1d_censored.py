from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr


@dataclass(frozen=True)
class GPConfig:
    mean_temp: float = 473.15
    signal_sd: float = 500.0
    lengthscale: float = 0.55
    noise_sd: float = 20.0
    relative_jitter: float = 1e-6


def true_temperature(x: np.ndarray) -> np.ndarray:
    T0 = 473.15
    A = 850.0
    xc = 0.25
    sigma = 0.70
    return T0 + A * np.exp(-0.5 * ((x - xc) / sigma) ** 2)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, config: GPConfig) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float)[:, None]
    x2 = np.asarray(x2, dtype=float)[None, :]
    sqdist = (x1 - x2) ** 2
    return config.signal_sd**2 * np.exp(-0.5 * sqdist / config.lengthscale**2)


def log_mills_ratio(z: np.ndarray) -> np.ndarray:
    log_phi = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
    return log_phi - log_ndtr(z)


def gp_predict_exact(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    config: GPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(x_train, x_train, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += config.noise_sd**2 + jitter
    K_s = rbf_kernel(x_pred, x_train, config)
    K_ss = rbf_kernel(x_pred, x_pred, config)

    cf = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(cf, y_train - config.mean_temp, check_finite=False)
    mean = config.mean_temp + K_s.dot(alpha)

    v = cho_solve(cf, K_s.T, check_finite=False)
    cov = K_ss - K_s.dot(v)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, sd


def fit_censored_gp_laplace(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sat_mask: np.ndarray,
    threshold: float,
    x_pred: np.ndarray,
    config: GPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x_obs)
    mean_vec = np.full(n, config.mean_temp)
    K = rbf_kernel(x_obs, x_obs, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += jitter
    K_cf = cho_factor(K, lower=True, check_finite=False)

    unsat_mask = ~sat_mask
    noise_var = config.noise_sd**2

    def K_inv(v: np.ndarray) -> np.ndarray:
        return cho_solve(K_cf, v, check_finite=False)

    def objective_and_grad(f: np.ndarray) -> tuple[float, np.ndarray]:
        centered = f - mean_vec
        Kinv_centered = K_inv(centered)
        obj = 0.5 * float(np.dot(centered, Kinv_centered))
        grad = Kinv_centered.copy()

        if np.any(unsat_mask):
            residual = f[unsat_mask] - y_obs[unsat_mask]
            obj += 0.5 * float(np.sum(residual**2) / noise_var)
            grad[unsat_mask] += residual / noise_var

        if np.any(sat_mask):
            z = (f[sat_mask] - threshold) / config.noise_sd
            obj += -float(np.sum(log_ndtr(z)))
            ratio = np.exp(log_mills_ratio(z))
            grad[sat_mask] += -ratio / config.noise_sd

        return obj, grad

    def fun(f: np.ndarray) -> float:
        return objective_and_grad(f)[0]

    def jac(f: np.ndarray) -> np.ndarray:
        return objective_and_grad(f)[1]

    x0 = y_obs.copy()
    x0[sat_mask] = threshold + 0.5 * config.noise_sd
    result = minimize(
        fun,
        x0=x0,
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": 8000, "maxls": 50, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        print(f"Warning: censored GP mode optimization did not fully converge: {result.message}")
    f_hat = np.asarray(result.x)

    W = np.zeros(n)
    if np.any(unsat_mask):
        W[unsat_mask] = 1.0 / noise_var
    if np.any(sat_mask):
        z = (f_hat[sat_mask] - threshold) / config.noise_sd
        ratio = np.exp(log_mills_ratio(z))
        W[sat_mask] = ratio * (z + ratio) / noise_var

    K_inv_mat = cho_solve(K_cf, np.eye(n), check_finite=False)
    post_precision = K_inv_mat + np.diag(W)
    post_cf = cho_factor(post_precision, lower=True, check_finite=False)
    Sigma = cho_solve(post_cf, np.eye(n), check_finite=False)

    K_s = rbf_kernel(x_pred, x_obs, config)
    K_ss = rbf_kernel(x_pred, x_pred, config)
    A = K_s.dot(K_inv_mat)
    mean = config.mean_temp + A.dot(f_hat - mean_vec)
    conditional_cov = K_ss - A.dot(K_s.T)
    cov = conditional_cov + A.dot(Sigma).dot(A.T)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, sd, f_hat


def compute_metrics(name: str, x_pred: np.ndarray, truth: np.ndarray, mean: np.ndarray, threshold: float) -> dict:
    rel_l2 = np.linalg.norm(mean - truth) / np.linalg.norm(truth)
    peak_abs_error = abs(float(np.max(mean) - np.max(truth)))
    peak_rel_error = peak_abs_error / float(np.max(truth))
    hot_mask = truth >= threshold
    hot_rel_l2 = np.linalg.norm(mean[hot_mask] - truth[hot_mask]) / np.linalg.norm(truth[hot_mask])
    return {
        "method": name,
        "field_rel_l2": rel_l2,
        "hot_region_rel_l2": hot_rel_l2,
        "peak_true": float(np.max(truth)),
        "peak_pred": float(np.max(mean)),
        "peak_abs_error": peak_abs_error,
        "peak_rel_error": peak_rel_error,
    }


def plot_gp_panels(
    x_obs: np.ndarray,
    T_true_obs: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    x_pred: np.ndarray,
    T_true_pred: np.ndarray,
    threshold: float,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    colors = {
        "exact clipped": "#D55E00",
        "discard saturated": "#CC79A7",
        "censored likelihood": "#009E73",
        "oracle true": "#0072B2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, (mean, sd)) in zip(axes.ravel(), predictions.items()):
        color = colors[name]
        ax.fill_between(x_pred, mean - 1.96 * sd, mean + 1.96 * sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(x_pred, mean, color=color, linewidth=2.2, label="GP mean")
        ax.plot(x_pred, T_true_pred, color="black", linewidth=1.8, label="true field")
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.2, label=r"threshold $c$")
        ax.scatter(x_obs[~sat_mask], T_obs[~sat_mask], color="black", s=22, zorder=3, label="unsaturated obs.")
        ax.scatter(x_obs[sat_mask], T_obs[sat_mask], facecolor="white", edgecolor="red", s=35, zorder=4, label="saturated obs.")
        if name == "oracle true":
            ax.scatter(x_obs, T_true_obs, color="#0072B2", s=9, alpha=0.45, zorder=2, label="oracle labels")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].set_ylabel("temperature")
    axes[1, 0].set_ylabel("temperature")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_overlay(
    x_obs: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    x_pred: np.ndarray,
    T_true_pred: np.ndarray,
    threshold: float,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    colors = {
        "exact clipped": "#D55E00",
        "discard saturated": "#CC79A7",
        "censored likelihood": "#009E73",
        "oracle true": "#0072B2",
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(x_pred, T_true_pred, color="black", linewidth=2.2, label="true field")
    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.2, label=r"threshold $c$")
    for name, (mean, _) in predictions.items():
        ax.plot(x_pred, mean, color=colors[name], linewidth=2.0, label=name)
    ax.scatter(x_obs[~sat_mask], T_obs[~sat_mask], color="black", s=22, zorder=3, label="unsaturated obs.")
    ax.scatter(x_obs[sat_mask], T_obs[sat_mask], facecolor="white", edgecolor="red", s=35, zorder=4, label="saturated obs.")
    ax.set_xlabel("x")
    ax.set_ylabel("temperature")
    ax.set_title("1D GP reconstruction from clipped observations")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    config = GPConfig()
    rng = np.random.default_rng(12)
    x_obs = np.linspace(-3.5, 3.5, 55)
    x_pred = np.linspace(-3.8, 3.8, 350)
    T_true_obs = true_temperature(x_obs)
    T_true_pred = true_temperature(x_pred)

    frac_saturated = 0.25
    threshold = float(np.quantile(T_true_obs, 1.0 - frac_saturated))
    T_meas = T_true_obs + rng.normal(0.0, config.noise_sd, size=T_true_obs.shape)
    sat_mask = T_meas >= threshold
    T_obs = np.minimum(T_meas, threshold)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    predictions["exact clipped"] = gp_predict_exact(x_obs, T_obs, x_pred, config)
    predictions["discard saturated"] = gp_predict_exact(x_obs[~sat_mask], T_obs[~sat_mask], x_pred, config)
    cens_mean, cens_sd, cens_mode = fit_censored_gp_laplace(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    predictions["censored likelihood"] = (cens_mean, cens_sd)
    predictions["oracle true"] = gp_predict_exact(x_obs, T_true_obs, x_pred, config)

    rows = [
        compute_metrics(name, x_pred, T_true_pred, mean, threshold)
        for name, (mean, _) in predictions.items()
    ]
    for row in rows:
        row["target_frac_saturated"] = frac_saturated
        row["actual_frac_saturated"] = float(np.mean(sat_mask))
        row["threshold"] = threshold
        row["noise_sd"] = config.noise_sd
        row["lengthscale"] = config.lengthscale
        row["signal_sd"] = config.signal_sd

    results = pd.DataFrame(rows)
    csv_path = out_dir / "gp1d_censored_results.csv"
    results.to_csv(csv_path, index=False)

    mode_path = out_dir / "gp1d_censored_mode.csv"
    pd.DataFrame({"x": x_obs, "posterior_mode": cens_mode, "T_obs": T_obs, "T_true": T_true_obs, "saturated": sat_mask}).to_csv(
        mode_path, index=False
    )

    plot_gp_panels(
        x_obs,
        T_true_obs,
        T_obs,
        sat_mask,
        x_pred,
        T_true_pred,
        threshold,
        predictions,
        out_dir / "gp1d_censored_reconstruction.png",
    )
    plot_overlay(
        x_obs,
        T_obs,
        sat_mask,
        x_pred,
        T_true_pred,
        threshold,
        predictions,
        out_dir / "gp1d_censored_overlay.png",
    )

    print(f"Saved {csv_path}")
    print(f"Saved {mode_path}")
    print(f"Saved plots to {out_dir}")
    print(results[["method", "field_rel_l2", "hot_region_rel_l2", "peak_pred", "peak_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
