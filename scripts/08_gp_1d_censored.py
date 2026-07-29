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


METHOD_ORDER = ["exact clipped", "discard saturated", "censored Laplace", "censored sampled", "oracle true"]
COLORS = {
    "exact clipped": "#D55E00",
    "discard saturated": "#CC79A7",
    "censored Laplace": "#009E73",
    "censored sampled": "#56B4E9",
    "oracle true": "#0072B2",
}


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


def compute_metrics(
    name: str,
    x_pred: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    threshold: float,
    sd: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    rel_l2 = np.linalg.norm(mean - truth) / np.linalg.norm(truth)
    peak_abs_error = abs(float(np.max(mean) - np.max(truth)))
    peak_rel_error = peak_abs_error / float(np.max(truth))
    hot_mask = truth >= threshold
    hot_rel_l2 = np.linalg.norm(mean[hot_mask] - truth[hot_mask]) / np.linalg.norm(truth[hot_mask])
    peak_idx = int(np.argmax(truth))
    if lower is None or upper is None:
        if sd is None:
            lower_peak = np.nan
            upper_peak = np.nan
        else:
            lower_peak = float(mean[peak_idx] - 1.96 * sd[peak_idx])
            upper_peak = float(mean[peak_idx] + 1.96 * sd[peak_idx])
    else:
        lower_peak = float(lower[peak_idx])
        upper_peak = float(upper[peak_idx])
    return {
        "method": name,
        "field_rel_l2": rel_l2,
        "hot_region_rel_l2": hot_rel_l2,
        "peak_true": float(np.max(truth)),
        "peak_pred": float(np.max(mean)),
        "peak_abs_error": peak_abs_error,
        "peak_rel_error": peak_rel_error,
        "true_peak_location": float(x_pred[peak_idx]),
        "posterior_mean_at_true_peak": float(mean[peak_idx]),
        "posterior_sd_at_true_peak": np.nan if sd is None else float(sd[peak_idx]),
        "peak_ci_lower": lower_peak,
        "peak_ci_upper": upper_peak,
        "true_peak_in_95": bool(lower_peak <= float(truth[peak_idx]) <= upper_peak)
        if np.isfinite(lower_peak) and np.isfinite(upper_peak)
        else False,
    }


def cholesky_with_jitter(matrix: np.ndarray, base_jitter: float) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eye = np.eye(sym.shape[0])
    for multiplier in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        try:
            return np.linalg.cholesky(sym + multiplier * base_jitter * eye)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(sym + 10000.0 * base_jitter * eye)


def make_observations(
    config: GPConfig,
    frac_saturated: float = 0.25,
    seed: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_obs = np.linspace(-3.5, 3.5, 55)
    x_pred = np.linspace(-3.8, 3.8, 350)
    T_true_obs = true_temperature(x_obs)
    T_true_pred = true_temperature(x_pred)
    threshold = float(np.quantile(T_true_obs, 1.0 - frac_saturated))
    T_meas = T_true_obs + rng.normal(0.0, config.noise_sd, size=T_true_obs.shape)
    sat_mask = T_meas >= threshold
    T_obs = np.minimum(T_meas, threshold)
    return x_obs, x_pred, T_true_obs, T_true_pred, T_obs, sat_mask, threshold, T_meas


def sample_censored_gp_ess(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sat_mask: np.ndarray,
    threshold: float,
    x_pred: np.ndarray,
    config: GPConfig,
    n_samples: int = 900,
    burn_in: int = 500,
    thin: int = 2,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample p(f_pred | unsaturated y, saturated threshold events).

    The saturated event is modeled as f(x_i) + eps_i >= threshold,
    matching the censored Gaussian likelihood used in the Laplace fit.
    """
    rng = np.random.default_rng(seed)
    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]

    if len(x_sat) == 0:
        mean, sd = gp_predict_exact(x_obs, y_obs, x_pred, config)
        lower = mean - 1.96 * sd
        upper = mean + 1.96 * sd
        return mean, sd, lower, upper

    noise_var = config.noise_sd**2
    base_jitter = config.relative_jitter * config.signal_sd**2

    # Variables to sample are a = [f_pred, z_sat], where z_sat = f_sat + eps_sat.
    n_pred = len(x_pred)
    mean_a = np.full(n_pred + len(x_sat), config.mean_temp)

    K_pp = rbf_kernel(x_pred, x_pred, config)
    K_pz = rbf_kernel(x_pred, x_sat, config)
    K_zz = rbf_kernel(x_sat, x_sat, config)
    K_zz[np.diag_indices_from(K_zz)] += noise_var
    K_aa = np.block([[K_pp, K_pz], [K_pz.T, K_zz]])
    K_aa[np.diag_indices_from(K_aa)] += base_jitter

    if len(x_unsat) > 0:
        K_yy = rbf_kernel(x_unsat, x_unsat, config)
        K_yy[np.diag_indices_from(K_yy)] += noise_var + base_jitter
        K_ay = np.vstack([rbf_kernel(x_pred, x_unsat, config), rbf_kernel(x_sat, x_unsat, config)])
        cf = cho_factor(K_yy, lower=True, check_finite=False)
        alpha = cho_solve(cf, y_unsat - config.mean_temp, check_finite=False)
        mean_cond = mean_a + K_ay.dot(alpha)
        solved = cho_solve(cf, K_ay.T, check_finite=False)
        cov_cond = K_aa - K_ay.dot(solved)
    else:
        mean_cond = mean_a
        cov_cond = K_aa

    L = cholesky_with_jitter(cov_cond, base_jitter)
    current = mean_cond.copy()
    current[n_pred:] = np.maximum(current[n_pred:], threshold + 0.5 * config.noise_sd)
    centered = current - mean_cond

    def feasible(sample: np.ndarray) -> bool:
        return bool(np.all(sample[n_pred:] >= threshold))

    samples = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        nu = L.dot(rng.normal(size=len(mean_cond)))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        for _ in range(2000):
            proposal_centered = centered * np.cos(theta) + nu * np.sin(theta)
            proposal = mean_cond + proposal_centered
            if feasible(proposal):
                centered = proposal_centered
                current = proposal
                break
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = rng.uniform(theta_min, theta_max)
        else:
            raise RuntimeError("Elliptical slice sampler could not find a feasible proposal.")

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(current[:n_pred].copy())

    sample_array = np.asarray(samples)
    mean = np.mean(sample_array, axis=0)
    sd = np.std(sample_array, axis=0, ddof=1)
    lower = np.quantile(sample_array, 0.025, axis=0)
    upper = np.quantile(sample_array, 0.975, axis=0)
    return mean, sd, lower, upper


def run_methods(
    config: GPConfig,
    include_sampled: bool = False,
    observation_seed: int = 12,
    sampler_seed: int | None = None,
    frac_saturated: float = 0.25,
    n_samples: int = 900,
    burn_in: int = 500,
    thin: int = 2,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pd.DataFrame,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray],
]:
    data = make_observations(config, frac_saturated=frac_saturated, seed=observation_seed)
    x_obs, x_pred, T_true_obs, T_true_pred, T_obs, sat_mask, threshold, _ = data

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    interval_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def add_normal(name: str, mean: np.ndarray, sd: np.ndarray) -> None:
        predictions[name] = (mean, sd)
        interval_predictions[name] = (mean, sd, mean - 1.96 * sd, mean + 1.96 * sd)

    add_normal("exact clipped", *gp_predict_exact(x_obs, T_obs, x_pred, config))
    add_normal("discard saturated", *gp_predict_exact(x_obs[~sat_mask], T_obs[~sat_mask], x_pred, config))
    cens_mean, cens_sd, _ = fit_censored_gp_laplace(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    add_normal("censored Laplace", cens_mean, cens_sd)

    if include_sampled:
        if sampler_seed is None:
            sampler_seed = 321 + int(round(100 * config.lengthscale)) + 1000 * observation_seed
        samp_mean, samp_sd, samp_lower, samp_upper = sample_censored_gp_ess(
            x_obs,
            T_obs,
            sat_mask,
            threshold,
            x_pred,
            config,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            seed=sampler_seed,
        )
        predictions["censored sampled"] = (samp_mean, samp_sd)
        interval_predictions["censored sampled"] = (samp_mean, samp_sd, samp_lower, samp_upper)

    add_normal("oracle true", *gp_predict_exact(x_obs, T_true_obs, x_pred, config))

    rows = []
    for name, (mean, sd, lower, upper) in interval_predictions.items():
        row = compute_metrics(name, x_pred, T_true_pred, mean, threshold, sd=sd, lower=lower, upper=upper)
        row["target_frac_saturated"] = frac_saturated
        row["observation_seed"] = observation_seed
        row["actual_frac_saturated"] = float(np.mean(sat_mask))
        row["threshold"] = threshold
        row["noise_sd"] = config.noise_sd
        row["lengthscale"] = config.lengthscale
        row["signal_sd"] = config.signal_sd
        rows.append(row)
    return predictions, interval_predictions, pd.DataFrame(rows), data


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
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, (mean, sd)) in zip(axes.ravel(), predictions.items()):
        color = COLORS[name]
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
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(x_pred, T_true_pred, color="black", linewidth=2.2, label="true field")
    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.2, label=r"threshold $c$")
    for name, (mean, _) in predictions.items():
        ax.plot(x_pred, mean, color=COLORS[name], linewidth=2.0, label=name)
    ax.scatter(x_obs[~sat_mask], T_obs[~sat_mask], color="black", s=22, zorder=3, label="unsaturated obs.")
    ax.scatter(x_obs[sat_mask], T_obs[sat_mask], facecolor="white", edgecolor="red", s=35, zorder=4, label="saturated obs.")
    ax.set_xlabel("x")
    ax.set_ylabel("temperature")
    ax.set_title("1D GP reconstruction from clipped observations")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_uncertainty_comparison(
    x_obs: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    x_pred: np.ndarray,
    T_true_pred: np.ndarray,
    threshold: float,
    interval_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(x_pred, T_true_pred, color="black", linewidth=2.2, label="true field")
    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.2, label=r"threshold $c$")
    for name in ["discard saturated", "censored Laplace", "censored sampled"]:
        mean, _, lower, upper = interval_predictions[name]
        ax.fill_between(x_pred, lower, upper, color=COLORS[name], alpha=0.13, linewidth=0)
        ax.plot(x_pred, mean, color=COLORS[name], linewidth=2.0, label=name)
    ax.scatter(x_obs[~sat_mask], T_obs[~sat_mask], color="black", s=22, zorder=3, label="unsaturated obs.")
    ax.scatter(x_obs[sat_mask], T_obs[sat_mask], facecolor="white", edgecolor="red", s=35, zorder=4, label="saturated obs.")
    ax.set_xlabel("x")
    ax.set_ylabel("temperature")
    ax.set_title("Censored GP uncertainty comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_one_sd_panels(
    x_obs: np.ndarray,
    T_obs: np.ndarray,
    sat_mask: np.ndarray,
    x_pred: np.ndarray,
    T_true_pred: np.ndarray,
    threshold: float,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    methods = [method for method in METHOD_ORDER if method in predictions]
    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, 3.0 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).ravel()
    for ax, name in zip(axes_flat, methods):
        mean, sd = predictions[name]
        color = COLORS[name]
        ax.fill_between(x_pred, mean - sd, mean + sd, color=color, alpha=0.23, linewidth=0, label="mean +/- 1 SD")
        ax.plot(x_pred, mean, color=color, linewidth=2.1, label="GP mean")
        ax.plot(x_pred, T_true_pred, color="black", linewidth=1.7, label="true field")
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.1, label=r"threshold $c$")
        ax.scatter(x_obs[~sat_mask], T_obs[~sat_mask], color="black", s=16, zorder=3, label="unsaturated obs.")
        ax.scatter(x_obs[sat_mask], T_obs[sat_mask], facecolor="white", edgecolor="red", s=28, zorder=4, label="saturated obs.")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel("x")
    for ax in axes_flat[::ncols]:
        ax.set_ylabel("temperature")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_posterior_sd_overlay(
    x_obs: np.ndarray,
    sat_mask: np.ndarray,
    x_pred: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    for name in METHOD_ORDER:
        if name not in predictions:
            continue
        _, sd = predictions[name]
        ax.plot(x_pred, sd, color=COLORS[name], linewidth=2.0, label=name)
    ymin, ymax = ax.get_ylim()
    for x in x_obs[sat_mask]:
        ax.vlines(x, ymin, ymin + 0.055 * (ymax - ymin), color="red", alpha=0.45, linewidth=1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("posterior SD")
    ax.set_title("Posterior standard deviation across the 1D slice")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sweep_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method, group in df.groupby("method", sort=False):
        group = group.sort_values("lengthscale")
        ax.semilogx(
            group["lengthscale"],
            group[metric],
            marker="o",
            linewidth=1.8,
            color=COLORS.get(method),
            label=method,
        )
    ax.set_xlabel(r"RBF lengthscale $\ell$")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + r" vs GP lengthscale $\ell$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_peak_intervals(df: pd.DataFrame, out_path: Path) -> None:
    methods = ["exact clipped", "discard saturated", "censored Laplace", "censored sampled"]
    lengthscales = sorted(df["lengthscale"].unique())
    fig, axes = plt.subplots(
        len(lengthscales),
        1,
        figsize=(8.0, 1.75 * len(lengthscales)),
        sharex=True,
        constrained_layout=True,
    )
    if len(lengthscales) == 1:
        axes = [axes]

    true_peak = float(df["peak_true"].iloc[0])
    for ax, ell in zip(axes, lengthscales):
        sub = df[df["lengthscale"] == ell].set_index("method")
        ax.axvline(true_peak, color="black", linewidth=1.6, label="true peak")
        for j, method in enumerate(methods):
            row = sub.loc[method]
            mean = float(row["posterior_mean_at_true_peak"])
            lo = float(row["peak_ci_lower"])
            hi = float(row["peak_ci_upper"])
            ax.errorbar(
                mean,
                j,
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=COLORS[method],
                capsize=3,
                linewidth=1.8,
            )
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title(rf"$\ell={ell:g}$")
        ax.grid(True, axis="x", alpha=0.25)
    axes[-1].set_xlabel("posterior interval at true peak location")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_lengthscale_sweep(out_dir: Path) -> pd.DataFrame:
    lengthscales = [0.3, 0.5, 0.8, 1.2, 1.8]
    frames = []
    for ell in lengthscales:
        print(f"Running lengthscale sweep ell={ell:g}")
        _, _, results, _ = run_methods(GPConfig(lengthscale=ell), include_sampled=True)
        frames.append(results)
    sweep = pd.concat(frames, ignore_index=True)
    sweep_path = out_dir / "gp1d_lengthscale_sweep_results.csv"
    sweep.to_csv(sweep_path, index=False)
    plot_sweep_metric(
        sweep,
        "field_rel_l2",
        "Relative L2 field error",
        out_dir / "gp1d_lengthscale_sweep_field_error.png",
    )
    plot_sweep_metric(
        sweep,
        "posterior_sd_at_true_peak",
        "Posterior SD at true peak",
        out_dir / "gp1d_lengthscale_sweep_peak_sd.png",
    )
    plot_peak_intervals(sweep, out_dir / "gp1d_lengthscale_sweep_peak_intervals.png")
    print(f"Saved {sweep_path}")
    return sweep


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    config = GPConfig()
    predictions, interval_predictions, results, data = run_methods(config, include_sampled=True)
    x_obs, x_pred, T_true_obs, T_true_pred, T_obs, sat_mask, threshold, _ = data
    csv_path = out_dir / "gp1d_censored_results.csv"
    results.to_csv(csv_path, index=False)

    _, _, cens_mode = fit_censored_gp_laplace(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    mode_path = out_dir / "gp1d_censored_mode.csv"
    pd.DataFrame({"x": x_obs, "posterior_mode": cens_mode, "T_obs": T_obs, "T_true": T_true_obs, "saturated": sat_mask}).to_csv(
        mode_path, index=False
    )

    panel_predictions = {
        name: predictions[name]
        for name in ["exact clipped", "discard saturated", "censored Laplace", "oracle true"]
    }
    plot_gp_panels(
        x_obs,
        T_true_obs,
        T_obs,
        sat_mask,
        x_pred,
        T_true_pred,
        threshold,
        panel_predictions,
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
    plot_uncertainty_comparison(
        x_obs,
        T_obs,
        sat_mask,
        x_pred,
        T_true_pred,
        threshold,
        interval_predictions,
        out_dir / "gp1d_censored_uncertainty_comparison.png",
    )
    plot_one_sd_panels(
        x_obs,
        T_obs,
        sat_mask,
        x_pred,
        T_true_pred,
        threshold,
        predictions,
        out_dir / "gp1d_censored_one_sd_bands.png",
    )
    plot_posterior_sd_overlay(
        x_obs,
        sat_mask,
        x_pred,
        predictions,
        out_dir / "gp1d_censored_posterior_sd_overlay.png",
    )
    sweep = run_lengthscale_sweep(out_dir)

    print(f"Saved {csv_path}")
    print(f"Saved {mode_path}")
    print(f"Saved plots to {out_dir}")
    print(
        results[
            [
                "method",
                "field_rel_l2",
                "hot_region_rel_l2",
                "peak_pred",
                "posterior_sd_at_true_peak",
                "true_peak_in_95",
            ]
        ].to_string(index=False)
    )
    print(
        sweep[
            [
                "lengthscale",
                "method",
                "field_rel_l2",
                "posterior_sd_at_true_peak",
                "true_peak_in_95",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
