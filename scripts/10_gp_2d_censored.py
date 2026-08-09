from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

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
class GP2DConfig:
    mean_temp: float = 473.15
    mean_function: Callable[[np.ndarray], np.ndarray] | None = None
    signal_sd: float = 500.0
    kernel: str = "rbf"
    lengthscale: float = 0.70
    lengthscale_y: float | None = None
    angle_degrees: float = 0.0
    diffusivity: float | None = None
    cooling_rate: float = 0.0
    advection_path: Callable[[np.ndarray], np.ndarray] | None = None
    forcing_lengthscale: float | None = None
    forcing_quadrature_order: int = 20
    source_amplitude_basis: Callable[[np.ndarray], np.ndarray] | None = None
    source_amplitude_fraction_sd: float = 0.0
    source_amplitude_timescale: float | None = None
    noise_sd: float = 20.0
    relative_jitter: float = 1e-6


def gp_mean(points: np.ndarray, config: GP2DConfig) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if config.mean_function is None:
        return np.full(len(p), config.mean_temp)
    values = np.asarray(config.mean_function(p), dtype=float)
    if values.ndim == 0:
        values = np.full(len(p), float(values))
    values = values.reshape(-1)
    if len(values) != len(p):
        raise ValueError(
            f"Mean function returned {len(values)} values for {len(p)} points"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Mean function returned non-finite values")
    return values


def true_temperature_2d(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    T0 = 473.15
    A = 850.0
    xc = 0.25
    yc = -0.20
    sigma_x = 0.75
    sigma_y = 0.48
    exponent = -0.5 * (((x - xc) / sigma_x) ** 2 + ((y - yc) / sigma_y) ** 2)
    return T0 + A * np.exp(exponent)


def make_grid(n: int, low: float = -3.5, high: float = 3.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(low, high, n)
    ys = np.linspace(low, high, n)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    return points, xs, ys


def laser_path_reference(n_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_x = np.linspace(-2.0, 2.0, n_points)
    path_y = 0.42 * np.sin(1.25 * path_x) - 0.12
    segment_lengths = np.sqrt(np.diff(path_x) ** 2 + np.diff(path_y) ** 2)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return path_x, path_y, arclength


def path_aligned_coordinates(points: np.ndarray) -> np.ndarray:
    """Map (x, y) to approximate coordinates along/across the synthetic laser path."""
    p = np.asarray(points, dtype=float)
    path_x, path_y, arclength = laser_path_reference()
    dx = p[:, None, 0] - path_x[None, :]
    dy = p[:, None, 1] - path_y[None, :]
    nearest = np.argmin(dx**2 + dy**2, axis=1)

    nearest_x = path_x[nearest]
    nearest_y = path_y[nearest]
    slope = 0.42 * 1.25 * np.cos(1.25 * nearest_x)
    normal_x = -slope / np.sqrt(1.0 + slope**2)
    normal_y = 1.0 / np.sqrt(1.0 + slope**2)
    signed_distance = (p[:, 0] - nearest_x) * normal_x + (p[:, 1] - nearest_y) * normal_y
    return np.column_stack([arclength[nearest], signed_distance])


def rbf_kernel(points1: np.ndarray, points2: np.ndarray, config: GP2DConfig) -> np.ndarray:
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    if config.kernel == "path_aligned":
        z1 = path_aligned_coordinates(p1[:, :2])
        z2 = path_aligned_coordinates(p2[:, :2])
        delta = z1[:, None, :] - z2[None, :, :]
        ell_s = config.lengthscale
        ell_r = 0.20 if config.lengthscale_y is None else config.lengthscale_y
        sqdist = (delta[:, :, 0] / ell_s) ** 2 + (delta[:, :, 1] / ell_r) ** 2
        return config.signal_sd**2 * np.exp(-0.5 * sqdist)

    if config.kernel == "diffusion_gibbs":
        if p1.shape[1] < 3 or p2.shape[1] < 3:
            raise ValueError("Diffusion kernel points must include x, y, and cooling age")
        if config.diffusivity is None or config.diffusivity < 0.0:
            raise ValueError("Diffusion kernel requires a nonnegative diffusivity")
        spatial_delta = p1[:, None, :2] - p2[None, :, :2]
        distance_squared = np.sum(spatial_delta**2, axis=2)
        ell1_squared = config.lengthscale**2 + 2.0 * config.diffusivity * np.maximum(p1[:, 2], 0.0)
        ell2_squared = config.lengthscale**2 + 2.0 * config.diffusivity * np.maximum(p2[:, 2], 0.0)
        denominator = ell1_squared[:, None] + ell2_squared[None, :]
        prefactor = 2.0 * np.sqrt(ell1_squared[:, None] * ell2_squared[None, :]) / denominator
        return config.signal_sd**2 * prefactor * np.exp(-distance_squared / denominator)

    if config.kernel in {
        "spatiotemporal_heat",
        "spatiotemporal_advection",
        "spatiotemporal_forced_heat",
        "spatiotemporal_advective_forced_heat",
        "spatiotemporal_advective_forced_heat_source_amplitude",
    }:
        if p1.shape[1] < 3 or p2.shape[1] < 3:
            raise ValueError("Space-time heat-kernel points must include x, y, and time")
        if config.diffusivity is None or config.diffusivity < 0.0:
            raise ValueError("Space-time heat kernel requires a nonnegative diffusivity")
        if config.cooling_rate < 0.0:
            raise ValueError("Space-time heat kernel requires a nonnegative cooling rate")
        spatial_delta = p1[:, None, :2] - p2[None, :, :2]
        if config.kernel in {
            "spatiotemporal_advection",
            "spatiotemporal_advective_forced_heat",
            "spatiotemporal_advective_forced_heat_source_amplitude",
        }:
            if config.advection_path is None:
                raise ValueError("Advection heat kernel requires an advection path")
            path1 = np.asarray(config.advection_path(p1[:, 2]), dtype=float)
            path2 = np.asarray(config.advection_path(p2[:, 2]), dtype=float)
            if path1.shape != (len(p1), 2) or path2.shape != (len(p2), 2):
                raise ValueError("Advection path must return one two-dimensional point per time")
            spatial_delta = spatial_delta - (
                path1[:, None, :] - path2[None, :, :]
            )
        distance_squared = np.sum(spatial_delta**2, axis=2)
        time_difference = np.abs(p1[:, None, 2] - p2[None, :, 2])

        if config.kernel in {
            "spatiotemporal_forced_heat",
            "spatiotemporal_advective_forced_heat",
            "spatiotemporal_advective_forced_heat_source_amplitude",
        }:
            if config.cooling_rate <= 0.0:
                raise ValueError(
                    "A stationary forced heat kernel requires a positive cooling rate"
                )
            forcing_lengthscale = (
                config.lengthscale
                if config.forcing_lengthscale is None
                else config.forcing_lengthscale
            )
            if forcing_lengthscale <= 0.0:
                raise ValueError("Stochastic forcing requires a positive lengthscale")
            if config.forcing_quadrature_order <= 0:
                raise ValueError("Forcing quadrature order must be positive")

            # W is white in time and has an RBF spatial covariance. Integrating the
            # heat semigroup from the infinite past gives the stationary covariance.
            nodes, weights = np.polynomial.laguerre.laggauss(
                config.forcing_quadrature_order
            )
            ell_squared = forcing_lengthscale**2
            covariance = np.zeros_like(distance_squared)
            for node, weight in zip(nodes, weights):
                propagated_time = time_difference + node / config.cooling_rate
                scale_squared = (
                    ell_squared + 2.0 * config.diffusivity * propagated_time
                )
                covariance += (
                    weight
                    * ell_squared
                    / scale_squared
                    * np.exp(-0.5 * distance_squared / scale_squared)
                )
            covariance *= np.exp(-config.cooling_rate * time_difference)

            zero_lag_scale = (
                ell_squared
                + 2.0
                * config.diffusivity
                * nodes
                / config.cooling_rate
            )
            zero_lag_integral = np.sum(weights * ell_squared / zero_lag_scale)
            covariance = config.signal_sd**2 * covariance / zero_lag_integral
            if (
                config.kernel
                == "spatiotemporal_advective_forced_heat_source_amplitude"
            ):
                if config.source_amplitude_basis is None:
                    raise ValueError("Source-amplitude kernel requires a source basis")
                if config.source_amplitude_fraction_sd < 0.0:
                    raise ValueError(
                        "Source-amplitude fractional scale must be nonnegative"
                    )
                timescale = config.source_amplitude_timescale
                if timescale is None or timescale <= 0.0:
                    raise ValueError(
                        "Source-amplitude kernel requires a positive timescale"
                    )
                basis1 = np.asarray(
                    config.source_amplitude_basis(p1),
                    dtype=float,
                ).reshape(-1)
                basis2 = np.asarray(
                    config.source_amplitude_basis(p2),
                    dtype=float,
                ).reshape(-1)
                if basis1.shape != (len(p1),) or basis2.shape != (len(p2),):
                    raise ValueError("Source basis must return one value per point")
                covariance += (
                    config.source_amplitude_fraction_sd**2
                    * basis1[:, None]
                    * basis2[None, :]
                    * np.exp(-time_difference / timescale)
                )
            return covariance

        scale_squared = config.lengthscale**2 + 2.0 * config.diffusivity * time_difference
        spatial_prefactor = config.lengthscale**2 / scale_squared
        temporal_decay = np.exp(-config.cooling_rate * time_difference)
        return (
            config.signal_sd**2
            * spatial_prefactor
            * temporal_decay
            * np.exp(-0.5 * distance_squared / scale_squared)
        )

    if config.kernel != "rbf":
        raise ValueError(f"Unknown GP kernel {config.kernel!r}")

    delta = p1[:, None, :2] - p2[None, :, :2]
    if config.lengthscale_y is None and config.angle_degrees == 0.0:
        sqdist = np.sum(delta**2, axis=2) / config.lengthscale**2
    else:
        ell_x = config.lengthscale
        ell_y = config.lengthscale if config.lengthscale_y is None else config.lengthscale_y
        theta = np.deg2rad(config.angle_degrees)
        dx = delta[:, :, 0]
        dy = delta[:, :, 1]
        x_rot = np.cos(theta) * dx + np.sin(theta) * dy
        y_rot = -np.sin(theta) * dx + np.cos(theta) * dy
        sqdist = (x_rot / ell_x) ** 2 + (y_rot / ell_y) ** 2
    return config.signal_sd**2 * np.exp(-0.5 * sqdist)


def kernel_diagonal(
    points: np.ndarray,
    config: GP2DConfig,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Evaluate the exact kernel diagonal without building the full matrix."""
    p = np.asarray(points, dtype=float)
    diagonal = np.empty(len(p), dtype=float)
    for start in range(0, len(p), chunk_size):
        stop = min(start + chunk_size, len(p))
        block = rbf_kernel(p[start:stop], p[start:stop], config)
        diagonal[start:stop] = np.diag(block)
    return diagonal


def log_mills_ratio(z: np.ndarray) -> np.ndarray:
    log_phi = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
    return log_phi - log_ndtr(z)


def cholesky_with_jitter(matrix: np.ndarray, base_jitter: float) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eye = np.eye(sym.shape[0])
    for multiplier in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        try:
            return np.linalg.cholesky(sym + multiplier * base_jitter * eye)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(sym + 10000.0 * base_jitter * eye)


def gp_predict_exact(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    config: GP2DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(x_train, x_train, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += config.noise_sd**2 + jitter
    K_s = rbf_kernel(x_pred, x_train, config)
    K_ss = rbf_kernel(x_pred, x_pred, config)

    cf = cho_factor(K, lower=True, check_finite=False)
    train_mean = gp_mean(x_train, config)
    pred_mean = gp_mean(x_pred, config)
    alpha = cho_solve(cf, y_train - train_mean, check_finite=False)
    mean = pred_mean + K_s.dot(alpha)

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
    config: GP2DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x_obs)
    mean_vec = gp_mean(x_obs, config)
    pred_mean = gp_mean(x_pred, config)
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

    result = minimize(
        lambda f: objective_and_grad(f)[0],
        x0=np.where(sat_mask, threshold + 0.5 * config.noise_sd, y_obs),
        jac=lambda f: objective_and_grad(f)[1],
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
    mean = pred_mean + A.dot(f_hat - mean_vec)
    conditional_cov = K_ss - A.dot(K_s.T)
    cov = conditional_cov + A.dot(Sigma).dot(A.T)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, sd, f_hat


def sample_censored_gp_ess(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sat_mask: np.ndarray,
    threshold: float,
    x_pred: np.ndarray,
    config: GP2DConfig,
    n_samples: int = 450,
    burn_in: int = 250,
    thin: int = 2,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]

    if len(x_sat) == 0:
        mean, sd = gp_predict_exact(x_obs, y_obs, x_pred, config)
        return mean, sd, mean - 1.96 * sd, mean + 1.96 * sd

    noise_var = config.noise_sd**2
    base_jitter = config.relative_jitter * config.signal_sd**2
    n_pred = len(x_pred)
    mean_a = np.concatenate([gp_mean(x_pred, config), gp_mean(x_sat, config)])

    K_pp = rbf_kernel(x_pred, x_pred, config)
    K_pz = rbf_kernel(x_pred, x_sat, config)
    K_zz = rbf_kernel(x_sat, x_sat, config)
    K_zz[np.diag_indices_from(K_zz)] += noise_var
    K_aa = np.block([[K_pp, K_pz], [K_pz.T, K_zz]])
    K_aa[np.diag_indices_from(K_aa)] += base_jitter

    K_yy = rbf_kernel(x_unsat, x_unsat, config)
    K_yy[np.diag_indices_from(K_yy)] += noise_var + base_jitter
    K_ay = np.vstack([rbf_kernel(x_pred, x_unsat, config), rbf_kernel(x_sat, x_unsat, config)])
    cf = cho_factor(K_yy, lower=True, check_finite=False)
    alpha = cho_solve(
        cf,
        y_unsat - gp_mean(x_unsat, config),
        check_finite=False,
    )
    mean_cond = mean_a + K_ay.dot(alpha)
    solved = cho_solve(cf, K_ay.T, check_finite=False)
    cov_cond = K_aa - K_ay.dot(solved)

    L = cholesky_with_jitter(cov_cond, base_jitter)
    current = mean_cond.copy()
    current[n_pred:] = np.maximum(current[n_pred:], threshold + 0.5 * config.noise_sd)
    centered = current - mean_cond

    samples = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        nu = L.dot(rng.normal(size=len(mean_cond)))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        for _ in range(2500):
            proposal_centered = centered * np.cos(theta) + nu * np.sin(theta)
            proposal = mean_cond + proposal_centered
            if np.all(proposal[n_pred:] >= threshold):
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


def sample_censored_gp_ess_fast(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sat_mask: np.ndarray,
    threshold: float,
    x_pred: np.ndarray,
    config: GP2DConfig,
    n_samples: int = 450,
    burn_in: int = 250,
    thin: int = 2,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample censored observations, then condition predictions analytically."""
    rng = np.random.default_rng(seed)
    unsat_mask = ~sat_mask
    x_unsat = x_obs[unsat_mask]
    y_unsat = y_obs[unsat_mask]
    x_sat = x_obs[sat_mask]
    noise_var = config.noise_sd**2
    jitter = config.relative_jitter * config.signal_sd**2

    if len(x_sat) == 0:
        mean, sd = gp_predict_exact(x_obs, y_obs, x_pred, config)
        draws = mean + rng.normal(size=(n_samples, len(x_pred))) * sd
        return (
            mean,
            sd,
            np.quantile(draws, 0.025, axis=0),
            np.quantile(draws, 0.975, axis=0),
            draws,
        )

    K_ss = rbf_kernel(x_sat, x_sat, config)
    K_ss[np.diag_indices_from(K_ss)] += noise_var + jitter
    if len(x_unsat):
        K_uu = rbf_kernel(x_unsat, x_unsat, config)
        K_uu[np.diag_indices_from(K_uu)] += noise_var + jitter
        K_su = rbf_kernel(x_sat, x_unsat, config)
        cf_uu = cho_factor(K_uu, lower=True, check_finite=False)
        alpha = cho_solve(
            cf_uu,
            y_unsat - gp_mean(x_unsat, config),
            check_finite=False,
        )
        sat_mean = gp_mean(x_sat, config) + K_su.dot(alpha)
        sat_cov = K_ss - K_su.dot(
            cho_solve(cf_uu, K_su.T, check_finite=False)
        )
    else:
        sat_mean = gp_mean(x_sat, config)
        sat_cov = K_ss

    L_sat = cholesky_with_jitter(sat_cov, jitter)
    current = np.maximum(sat_mean, threshold + 0.5 * config.noise_sd)
    centered = current - sat_mean
    saturated_samples: list[np.ndarray] = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        nu = L_sat.dot(rng.normal(size=len(x_sat)))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        for _ in range(2500):
            proposal_centered = centered * np.cos(theta) + nu * np.sin(theta)
            proposal = sat_mean + proposal_centered
            if np.all(proposal >= threshold):
                centered = proposal_centered
                current = proposal
                break
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = rng.uniform(theta_min, theta_max)
        else:
            raise RuntimeError("Elliptical slice sampler could not find a feasible proposal")
        if step >= burn_in and (step - burn_in) % thin == 0:
            saturated_samples.append(current.copy())

    observation_samples = np.empty((n_samples, len(x_obs)))
    observation_samples[:, unsat_mask] = y_unsat
    observation_samples[:, sat_mask] = np.asarray(saturated_samples)

    K_oo = rbf_kernel(x_obs, x_obs, config)
    K_oo[np.diag_indices_from(K_oo)] += noise_var + jitter
    K_po = rbf_kernel(x_pred, x_obs, config)
    cf_oo = cho_factor(K_oo, lower=True, check_finite=False)
    alpha_samples = cho_solve(
        cf_oo,
        (observation_samples - gp_mean(x_obs, config)[None, :]).T,
        check_finite=False,
    )
    conditional_means = gp_mean(x_pred, config)[None, :] + (K_po.dot(alpha_samples)).T
    solved = cho_solve(cf_oo, K_po.T, check_finite=False)
    prior_variance = kernel_diagonal(x_pred, config)
    conditional_variance = np.maximum(
        prior_variance - np.sum(K_po * solved.T, axis=1),
        0.0,
    )
    mean = np.mean(conditional_means, axis=0)
    variance = conditional_variance + np.var(conditional_means, axis=0, ddof=1)
    sd = np.sqrt(np.maximum(variance, 0.0))
    draws = conditional_means + rng.normal(size=conditional_means.shape) * np.sqrt(
        conditional_variance
    )
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    return mean, sd, lower, upper, draws


def make_observations(
    config: GP2DConfig,
    obs_n: int = 17,
    pred_n: int = 32,
    frac_saturated: float = 0.03,
    seed: int = 12,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_obs, obs_xs, obs_ys = make_grid(obs_n)
    x_pred, pred_xs, pred_ys = make_grid(pred_n)
    T_true_obs = true_temperature_2d(x_obs)
    T_true_pred = true_temperature_2d(x_pred)
    threshold = float(np.quantile(T_true_obs, 1.0 - frac_saturated))
    T_meas = T_true_obs + rng.normal(0.0, config.noise_sd, size=T_true_obs.shape)
    sat_mask = T_meas >= threshold
    T_obs = np.minimum(T_meas, threshold)
    return {
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


def compute_metrics(
    name: str,
    points: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    threshold: float,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    peak_idx = int(np.argmax(truth))
    hot_mask = truth >= threshold
    true_peak = float(truth[peak_idx])
    lower_peak = float(lower[peak_idx])
    upper_peak = float(upper[peak_idx])
    return {
        "method": name,
        "field_rel_l2": float(np.linalg.norm(mean - truth) / np.linalg.norm(truth)),
        "hot_region_rel_l2": float(np.linalg.norm(mean[hot_mask] - truth[hot_mask]) / np.linalg.norm(truth[hot_mask])),
        "peak_true": true_peak,
        "peak_pred": float(np.max(mean)),
        "peak_abs_error": abs(float(np.max(mean) - np.max(truth))),
        "true_peak_x": float(points[peak_idx, 0]),
        "true_peak_y": float(points[peak_idx, 1]),
        "posterior_mean_at_true_peak": float(mean[peak_idx]),
        "posterior_sd_at_true_peak": float(sd[peak_idx]),
        "peak_ci_lower": lower_peak,
        "peak_ci_upper": upper_peak,
        "true_peak_in_95": bool(lower_peak <= true_peak <= upper_peak),
    }


def add_normal_interval(
    store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    name: str,
    mean: np.ndarray,
    sd: np.ndarray,
) -> None:
    store[name] = (mean, sd, mean - 1.96 * sd, mean + 1.96 * sd)


def run_methods(config: GP2DConfig) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
    data = make_observations(config)
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
    add_normal_interval(predictions, "exact clipped", *gp_predict_exact(x_obs, T_obs, x_pred, config))
    add_normal_interval(predictions, "discard saturated", *gp_predict_exact(x_obs[~sat_mask], T_obs[~sat_mask], x_pred, config))
    lap_mean, lap_sd, _ = fit_censored_gp_laplace(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    add_normal_interval(predictions, "censored Laplace", lap_mean, lap_sd)
    sampled = sample_censored_gp_ess(x_obs, T_obs, sat_mask, threshold, x_pred, config)
    predictions["censored sampled"] = sampled
    add_normal_interval(predictions, "oracle true", *gp_predict_exact(x_obs, T_true_obs, x_pred, config))
    return predictions, data


def grid(values: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(values).reshape(n, n)


def plot_reconstruction_panels(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    data: dict[str, object],
    out_path: Path,
) -> None:
    x_obs = data["x_obs"]
    T_obs = data["T_obs"]
    sat_mask = data["sat_mask"]
    T_true_pred = data["T_true_pred"]
    pred_xs = data["pred_xs"]
    pred_ys = data["pred_ys"]
    threshold = data["threshold"]

    assert isinstance(x_obs, np.ndarray)
    assert isinstance(T_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(T_true_pred, np.ndarray)
    assert isinstance(pred_xs, np.ndarray)
    assert isinstance(pred_ys, np.ndarray)
    assert isinstance(threshold, float)

    pred_n = len(pred_xs)
    extent = [float(pred_xs[0]), float(pred_xs[-1]), float(pred_ys[0]), float(pred_ys[-1])]
    panels = [
        ("true field", T_true_pred, "viridis"),
        ("clipped obs.", None, "viridis"),
        ("exact clipped", predictions["exact clipped"][0], "viridis"),
        ("discard saturated", predictions["discard saturated"][0], "viridis"),
        ("censored Laplace", predictions["censored Laplace"][0], "viridis"),
        ("censored sampled", predictions["censored sampled"][0], "viridis"),
        ("oracle true", predictions["oracle true"][0], "viridis"),
        ("saturated mask", None, "gray_r"),
    ]
    vmin = float(np.min(T_true_pred))
    vmax = float(np.max(T_true_pred))

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), constrained_layout=True)
    image_handles = []
    for ax, (title, values, cmap) in zip(axes.ravel(), panels):
        if title == "clipped obs.":
            obs_n = int(np.sqrt(len(T_obs)))
            obs_xs = data["obs_xs"]
            obs_ys = data["obs_ys"]
            assert isinstance(obs_xs, np.ndarray)
            assert isinstance(obs_ys, np.ndarray)
            obs_extent = [float(obs_xs[0]), float(obs_xs[-1]), float(obs_ys[0]), float(obs_ys[-1])]
            im = ax.imshow(grid(T_obs, obs_n), origin="lower", extent=obs_extent, cmap=cmap, vmin=vmin, vmax=vmax)
        elif title == "saturated mask":
            obs_n = int(np.sqrt(len(T_obs)))
            obs_xs = data["obs_xs"]
            obs_ys = data["obs_ys"]
            assert isinstance(obs_xs, np.ndarray)
            assert isinstance(obs_ys, np.ndarray)
            obs_extent = [float(obs_xs[0]), float(obs_xs[-1]), float(obs_ys[0]), float(obs_ys[-1])]
            im = ax.imshow(grid(sat_mask.astype(float), obs_n), origin="lower", extent=obs_extent, cmap=cmap, vmin=0, vmax=1)
        else:
            assert values is not None
            im = ax.imshow(grid(values, pred_n), origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        image_handles.append(im)
        ax.contour(
            grid(T_true_pred, pred_n),
            levels=[threshold],
            origin="lower",
            extent=extent,
            colors="white",
            linewidths=1.0,
            linestyles="--",
        )
        if title not in {"true field", "saturated mask"}:
            ax.scatter(x_obs[sat_mask, 0], x_obs[sat_mask, 1], facecolor="none", edgecolor="red", s=14, linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image_handles[0], ax=axes[:, :3], shrink=0.82, label="temperature")
    fig.colorbar(image_handles[-1], ax=axes[:, 3], shrink=0.82, label="saturated")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sd_panels(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    data: dict[str, object],
    out_path: Path,
) -> None:
    x_obs = data["x_obs"]
    sat_mask = data["sat_mask"]
    pred_xs = data["pred_xs"]
    pred_ys = data["pred_ys"]
    assert isinstance(x_obs, np.ndarray)
    assert isinstance(sat_mask, np.ndarray)
    assert isinstance(pred_xs, np.ndarray)
    assert isinstance(pred_ys, np.ndarray)

    pred_n = len(pred_xs)
    extent = [float(pred_xs[0]), float(pred_xs[-1]), float(pred_ys[0]), float(pred_ys[-1])]
    methods = ["discard saturated", "censored Laplace", "censored sampled", "oracle true"]
    vmax = max(float(np.max(predictions[name][1])) for name in methods)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), constrained_layout=True)
    for ax, name in zip(axes.ravel(), methods):
        sd = predictions[name][1]
        im = ax.imshow(grid(sd, pred_n), origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax)
        ax.scatter(x_obs[sat_mask, 0], x_obs[sat_mask, 1], facecolor="none", edgecolor="cyan", s=14, linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.82, label="posterior SD")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    config = GP2DConfig()
    predictions, data = run_methods(config)

    x_pred = data["x_pred"]
    T_true_pred = data["T_true_pred"]
    threshold = data["threshold"]
    sat_mask = data["sat_mask"]
    assert isinstance(x_pred, np.ndarray)
    assert isinstance(T_true_pred, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(sat_mask, np.ndarray)

    rows = []
    for name in METHOD_ORDER:
        mean, sd, lower, upper = predictions[name]
        row = compute_metrics(name, x_pred, T_true_pred, mean, threshold, sd, lower, upper)
        row["lengthscale"] = config.lengthscale
        row["noise_sd"] = config.noise_sd
        row["signal_sd"] = config.signal_sd
        row["actual_frac_saturated"] = float(np.mean(sat_mask))
        row["threshold"] = threshold
        rows.append(row)
    results = pd.DataFrame(rows)

    csv_path = out_dir / "gp2d_censored_results.csv"
    results.to_csv(csv_path, index=False)
    plot_reconstruction_panels(predictions, data, out_dir / "gp2d_censored_reconstructions.png")
    plot_sd_panels(predictions, data, out_dir / "gp2d_censored_posterior_sd.png")

    print(f"Saved {csv_path}")
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


if __name__ == "__main__":
    main()
