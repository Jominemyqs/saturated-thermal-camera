from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.diffusion import cooling_age, estimate_effective_diffusivity
from src.metrics import empirical_crps as unbiased_empirical_crps
from src.thermal_trajectory import SurfaceGridProjector, ThermalTrajectory

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_gp2d_module():
    module_path = ROOT / "scripts" / "10_gp_2d_censored.py"
    spec = importlib.util.spec_from_file_location("gp2d_thermal", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gp2d_thermal"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gp2d = load_gp2d_module()

DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thermal_diffusion_kernel"
DEFAULT_TRAJECTORIES = [
    "DiagonalScanPath_8",
    "HorizontalScanPath_13",
    "SpiralScanPath_13",
]
LENGTHSCALE_CANDIDATES = np.asarray(
    [
        0.00020,
        0.00030,
        0.00045,
        0.00065,
        0.00090,
        0.00130,
        0.00180,
        0.00250,
        0.00350,
        0.00500,
        0.00700,
    ]
)
METHOD_ORDER = [
    "RBF source-scale",
    "diffusion source-scale",
    "RBF unsat-tuned",
    "diffusion unsat-tuned",
]
METHOD_COLORS = {
    "RBF source-scale": "#E69F00",
    "diffusion source-scale": "#0072B2",
    "RBF unsat-tuned": "#CC79A7",
    "diffusion unsat-tuned": "#009E73",
}


@dataclass(frozen=True)
class PreparedTrajectory:
    name: str
    xs: np.ndarray
    ys: np.ndarray
    points: np.ndarray
    times: np.ndarray
    history: np.ndarray
    heat_flux: np.ndarray
    truth: np.ndarray
    age: np.ndarray
    ambient: float
    source_lengthscale: float
    signal_scale: float


def estimate_source_lengthscale(
    heat_flux: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> float:
    """Estimate a Gaussian source width from active heat-flux footprints."""
    grid_x, grid_y = np.meshgrid(xs, ys)
    widths: list[float] = []
    for field in np.asarray(heat_flux):
        peak = float(np.max(field))
        if peak < 10_000.0:
            continue
        weights = np.maximum(field, 0.0)
        weights = np.where(weights >= 0.01 * peak, weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        center_x = float(np.sum(weights * grid_x) / total)
        center_y = float(np.sum(weights * grid_y) / total)
        radial_variance = float(
            np.sum(weights * ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2))
            / total
        )
        widths.append(np.sqrt(radial_variance / 2.0))
    if not widths:
        raise ValueError("No active heat-flux footprints were found")
    return float(np.median(widths))


def trajectory_path(dataset_dir: Path, name: str) -> Path:
    candidates = [
        dataset_dir / f"{name}.xdmf",
        dataset_dir / f"Copy of {name}.xdmf",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {name!r} in {dataset_dir}")


def prepare_trajectory(
    dataset_dir: Path,
    name: str,
    *,
    nx: int,
    ny: int,
    time_stride: int,
    heat_flux_cutoff: float,
) -> tuple[PreparedTrajectory, dict[str, float | int | str]]:
    trajectory = ThermalTrajectory(trajectory_path(dataset_dir, name))
    indices = list(range(0, trajectory.n_times, time_stride))
    if indices[-1] != trajectory.n_times - 1:
        indices.append(trajectory.n_times - 1)
    times, native_history = trajectory.read_history(
        time_indices=indices,
        surface_only=True,
    )
    _, native_heat_flux = trajectory.read_history(
        "HeatFluxZ",
        time_indices=indices,
        surface_only=True,
    )
    projector = SurfaceGridProjector(
        trajectory.surface_coordinates(),
        nx=nx,
        ny=ny,
    )
    history = projector.project(native_history)
    heat_flux = projector.project(native_heat_flux)
    source_width = estimate_source_lengthscale(
        heat_flux,
        projector.xs,
        projector.ys,
    )
    grid_spacing = max(
        float(np.mean(np.diff(projector.xs))),
        float(np.mean(np.diff(projector.ys))),
    )
    source_lengthscale = max(source_width, grid_spacing)
    estimate = estimate_effective_diffusivity(
        history,
        times,
        projector.xs,
        projector.ys,
        minimum_excess=0.5,
        valid_mask=np.abs(heat_flux) < heat_flux_cutoff,
    )
    cooling_excess = history[:-1] - estimate.ambient_temperature
    cooling_mask = (
        (np.diff(history, axis=0) < 0.0)
        & (cooling_excess >= 0.5)
        & (np.abs(heat_flux[:-1]) < heat_flux_cutoff)
    )
    signal_scale = float(np.quantile(cooling_excess[cooling_mask], 0.95))
    age = cooling_age(history, times)
    heated = np.max(history, axis=0) >= estimate.ambient_temperature + 0.5
    age = np.where(heated, age, 0.0)
    prepared = PreparedTrajectory(
        name=name,
        xs=projector.xs,
        ys=projector.ys,
        points=projector.grid_points,
        times=times,
        history=history,
        heat_flux=heat_flux,
        truth=history[-1],
        age=age,
        ambient=estimate.ambient_temperature,
        source_lengthscale=source_lengthscale,
        signal_scale=signal_scale,
    )
    estimate_row: dict[str, float | int | str] = {
        "trajectory": name,
        "diffusivity_m2_s": estimate.diffusivity,
        "cooling_rate_1_s": estimate.cooling_rate,
        "ambient_temperature_K": estimate.ambient_temperature,
        "r_squared": estimate.r_squared,
        "rmse_K_s": estimate.rmse,
        "n_cooling_samples": estimate.n_samples,
        "n_time_snapshots": len(times),
        "end_time_s": float(times[-1]),
        "heat_flux_cutoff": heat_flux_cutoff,
        "source_width_m": source_width,
        "source_lengthscale_m": source_lengthscale,
        "cooling_excess_q95_K": signal_scale,
    }
    return prepared, estimate_row


def make_censored_observations(
    prepared: PreparedTrajectory,
    *,
    fraction_saturated: float,
    noise_sd: float,
    observation_stride: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    threshold = float(np.quantile(prepared.truth, 1.0 - fraction_saturated))
    measured = prepared.truth + rng.normal(0.0, noise_sd, size=prepared.truth.shape)
    saturated_full = measured >= threshold

    coarse_mask = np.zeros(prepared.truth.shape, dtype=bool)
    coarse_mask[::observation_stride, ::observation_stride] = True
    observation_mask = coarse_mask | saturated_full

    age_flat = prepared.age.ravel()
    augmented_points = np.column_stack([prepared.points, age_flat])
    measured_flat = measured.ravel()
    saturated_flat = saturated_full.ravel()
    observation_flat = observation_mask.ravel()
    observed_values = np.minimum(measured_flat[observation_flat], threshold)
    return {
        "x_obs": augmented_points[observation_flat],
        "y_obs": observed_values,
        "sat_mask": saturated_flat[observation_flat],
        "x_pred": augmented_points,
        "threshold": threshold,
        "clipped_full": np.minimum(measured, threshold),
        "observation_mask": observation_mask,
        "saturated_full": saturated_full,
    }


def make_config(
    prepared: PreparedTrajectory,
    *,
    method: str,
    lengthscale: float,
    diffusivity: float,
    signal_sd: float,
    noise_sd: float,
):
    if method.startswith("RBF"):
        kernel = "rbf"
        alpha = None
    elif method.startswith("diffusion"):
        kernel = "diffusion_gibbs"
        alpha = diffusivity
    else:
        raise ValueError(f"Unknown method {method!r}")
    return gp2d.GP2DConfig(
        mean_temp=prepared.ambient,
        signal_sd=signal_sd,
        kernel=kernel,
        lengthscale=lengthscale,
        diffusivity=alpha,
        noise_sd=noise_sd,
        relative_jitter=1e-7,
    )


def unsaturated_nll(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sat_mask: np.ndarray,
    config,
) -> float:
    x_train = x_obs[~sat_mask]
    y_train = y_obs[~sat_mask]
    K = gp2d.rbf_kernel(x_train, x_train, config)
    jitter = config.relative_jitter * config.signal_sd**2
    K[np.diag_indices_from(K)] += config.noise_sd**2 + jitter
    try:
        cf = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf
    centered = y_train - gp2d.gp_mean(x_train, config)
    alpha = cho_solve(cf, centered, check_finite=False)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    return (
        0.5 * float(np.dot(centered, alpha))
        + 0.5 * log_det
        + 0.5 * len(y_train) * np.log(2.0 * np.pi)
    )


def select_lengthscale(
    prepared: PreparedTrajectory,
    observations: dict[str, np.ndarray | float],
    *,
    method: str,
    diffusivity: float,
    signal_sd: float,
    noise_sd: float,
) -> tuple[float, list[dict[str, float | str]]]:
    x_obs = np.asarray(observations["x_obs"])
    y_obs = np.asarray(observations["y_obs"])
    sat_mask = np.asarray(observations["sat_mask"], dtype=bool)
    rows: list[dict[str, float | str]] = []
    for lengthscale in LENGTHSCALE_CANDIDATES:
        config = make_config(
            prepared,
            method=method,
            lengthscale=float(lengthscale),
            diffusivity=diffusivity,
            signal_sd=signal_sd,
            noise_sd=noise_sd,
        )
        rows.append(
            {
                "trajectory": prepared.name,
                "method": method,
                "lengthscale_m": float(lengthscale),
                "unsaturated_nll": unsaturated_nll(
                    x_obs,
                    y_obs,
                    sat_mask,
                    config,
                ),
            }
        )
    finite_rows = [row for row in rows if np.isfinite(row["unsaturated_nll"])]
    if not finite_rows:
        raise RuntimeError(f"No finite lengthscale score for {prepared.name}, {method}")
    selected = min(finite_rows, key=lambda row: row["unsaturated_nll"])
    return float(selected["lengthscale_m"]), rows


def empirical_crps(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return unbiased_empirical_crps(draws, truth)


def sample_censored_multiple_chains(
    observations: dict[str, np.ndarray | float],
    config,
    *,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = []
    variances = []
    draws = []
    for chain in range(n_chains):
        result = gp2d.sample_censored_gp_ess_fast(
            np.asarray(observations["x_obs"]),
            np.asarray(observations["y_obs"]),
            np.asarray(observations["sat_mask"], dtype=bool),
            float(observations["threshold"]),
            np.asarray(observations["x_pred"]),
            config,
            n_samples=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 10_000 * chain,
        )
        means.append(result[0])
        variances.append(result[1] ** 2)
        draws.append(result[4])
    mean_array = np.asarray(means)
    variance_array = np.asarray(variances)
    mean = np.mean(mean_array, axis=0)
    variance = np.mean(variance_array + mean_array**2, axis=0) - mean**2
    pooled = np.vstack(draws)
    return (
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.quantile(pooled, 0.025, axis=0),
        np.quantile(pooled, 0.975, axis=0),
        pooled,
    )


def compute_metrics(
    prepared: PreparedTrajectory,
    observations: dict[str, np.ndarray | float],
    *,
    method: str,
    lengthscale: float,
    learned_diffusivity: float,
    signal_sd: float,
    mean: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    draws: np.ndarray,
) -> dict[str, float | int | str | bool]:
    truth = prepared.truth.ravel()
    excess_truth = truth - prepared.ambient
    threshold = float(observations["threshold"])
    hot_mask = truth >= threshold
    peak_index = int(np.argmax(truth))
    crps = empirical_crps(draws, truth)
    return {
        "trajectory": prepared.name,
        "method": method,
        "lengthscale_m": lengthscale,
        "learned_diffusivity_m2_s": learned_diffusivity,
        "signal_sd_K": signal_sd,
        "n_observations": len(np.asarray(observations["y_obs"])),
        "n_saturated": int(np.sum(np.asarray(observations["sat_mask"]))),
        "threshold_K": threshold,
        "excess_field_rel_l2": float(
            np.linalg.norm(mean - truth) / np.linalg.norm(excess_truth)
        ),
        "hot_region_rel_l2": float(
            np.linalg.norm(mean[hot_mask] - truth[hot_mask])
            / np.linalg.norm(excess_truth[hot_mask])
        ),
        "peak_true_K": float(np.max(truth)),
        "peak_predicted_K": float(np.max(mean)),
        "peak_absolute_error_K": abs(float(np.max(mean) - np.max(truth))),
        "posterior_sd_at_true_peak_K": float(sd[peak_index]),
        "true_peak_in_95": bool(
            lower[peak_index] <= truth[peak_index] <= upper[peak_index]
        ),
        "pointwise_95_coverage": float(np.mean((lower <= truth) & (truth <= upper))),
        "hot_region_95_coverage": float(
            np.mean((lower[hot_mask] <= truth[hot_mask]) & (truth[hot_mask] <= upper[hot_mask]))
        ),
        "mean_crps_K": float(np.mean(crps)),
        "hot_region_crps_K": float(np.mean(crps[hot_mask])),
    }


def plot_diffusivity_estimates(estimates: pd.DataFrame, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    labels = estimates["trajectory"].str.replace("ScanPath_", " ")
    values = estimates["diffusivity_m2_s"] * 1e6
    bars = axis.bar(labels, values, color=["#009E73", "#E69F00", "#56B4E9"])
    axis.axhline(values.median(), color="#333333", linestyle="--", linewidth=1.4)
    for bar, r_squared in zip(bars, estimates["r_squared"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            rf"$R^2={r_squared:.2f}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel(r"Effective diffusivity ($10^{-6}\,\mathrm{m}^2/\mathrm{s}$)")
    axis.set_title("Diffusivity learned from cooling regions")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_reconstruction(
    prepared: PreparedTrajectory,
    observations: dict[str, np.ndarray | float],
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    selected_lengths: dict[str, float],
    learned_diffusivity: float,
    out_path: Path,
) -> None:
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    shape = prepared.truth.shape
    vmin = prepared.ambient
    vmax = float(np.max(prepared.truth))
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.2), constrained_layout=True)
    truth_image = axes[0, 0].imshow(
        prepared.truth,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[0, 0].set_title("True final field")
    axes[0, 1].imshow(
        np.asarray(observations["clipped_full"]),
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[0, 1].set_title(f"Clipped at {float(observations['threshold']):.2f} K")
    age_image = axes[0, 2].imshow(
        prepared.age * 1e3,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
    )
    axes[0, 2].set_title("Cooling age (ms)")
    axes[0, 3].imshow(
        np.asarray(observations["saturated_full"]),
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[0, 3].set_title("Saturated pixels")

    for column, method in enumerate(METHOD_ORDER):
        mean, _, _, _, _ = predictions[method]
        axes[1, column].imshow(
            mean.reshape(shape),
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axes[1, column].set_title(
            f"{method} mean\n"
            rf"$\ell_0={selected_lengths[method] * 1e3:.2f}$ mm"
        )

    for axis in axes.ravel():
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.colorbar(
        truth_image,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.03,
        label="Temperature (K)",
    )
    figure.colorbar(
        age_image,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.03,
        label="Time since local peak (ms)",
    )
    figure.suptitle(
        prepared.name
        + rf": leave-one-trajectory-out $\alpha={learned_diffusivity * 1e6:.2f}"
        + r"\times10^{-6}\,\mathrm{m}^2/\mathrm{s}$",
        fontsize=14,
    )
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_metric_summary(results: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("excess_field_rel_l2", "Relative error\n(temperature excess)"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_crps_K", "Hot-region CRPS (K)"),
    ]
    trajectories = list(dict.fromkeys(results["trajectory"]))
    x = np.arange(len(trajectories))
    width = 0.19
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, (column, label) in zip(axes, metrics):
        for method_index, method in enumerate(METHOD_ORDER):
            subset = results.set_index(["trajectory", "method"])
            values = [float(subset.loc[(name, method), column]) for name in trajectories]
            offset = (method_index - 1.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                label=method,
                color=METHOD_COLORS[method],
            )
        axis.set_xticks(x, [name.replace("ScanPath_", "\n") for name in trajectories])
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Censored GP on simulated laser trajectories (lower is better)")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_lengthscale_selection(selection: pd.DataFrame, out_path: Path) -> None:
    trajectories = list(dict.fromkeys(selection["trajectory"]))
    figure, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(4.6 * len(trajectories), 4.1),
        sharey=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, trajectory in zip(axes, trajectories):
        subset = selection[
            (selection["trajectory"] == trajectory)
            & selection["method"].str.endswith("unsat-tuned")
        ]
        for method in ["RBF unsat-tuned", "diffusion unsat-tuned"]:
            method_rows = subset[subset["method"] == method].sort_values("lengthscale_m")
            axis.plot(
                method_rows["lengthscale_m"] * 1e3,
                method_rows["unsaturated_nll"],
                marker="o",
                label=method.replace(" unsat-tuned", ""),
                color=METHOD_COLORS[method],
            )
        source_scale = float(
            selection[
                (selection["trajectory"] == trajectory)
                & (selection["method"] == "RBF source-scale")
            ]["lengthscale_m"].iloc[0]
        )
        axis.axvline(
            source_scale * 1e3,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="resolved source scale",
        )
        axis.set_xscale("log")
        axis.set_xlabel(r"Base lengthscale $\ell_0$ (mm)")
        axis.set_title(trajectory.replace("ScanPath_", " "))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Unsaturated negative log marginal likelihood")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Unsaturated-only tuning favors a much smoother prior")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare isotropic and cooling-aware diffusion GP kernels."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=DEFAULT_TRAJECTORIES,
    )
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--observation-stride", type=int, default=3)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument(
        "--signal-sd",
        type=float,
        default=None,
        help="Override the leave-one-trajectory-out cooling-excess signal scale.",
    )
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared_items: dict[str, PreparedTrajectory] = {}
    estimate_rows: list[dict[str, float | int | str]] = []
    for name in args.trajectories:
        prepared, estimate_row = prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            time_stride=args.time_stride,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        prepared_items[name] = prepared
        estimate_rows.append(estimate_row)
        print(
            f"{name}: alpha={estimate_row['diffusivity_m2_s']:.4e} m^2/s, "
            f"R2={estimate_row['r_squared']:.3f}, "
            f"source width={estimate_row['source_width_m'] * 1e3:.3f} mm, "
            f"resolved ell0={estimate_row['source_lengthscale_m'] * 1e3:.3f} mm, "
            f"signal scale={estimate_row['cooling_excess_q95_K']:.2f} K"
        )

    estimate_frame = pd.DataFrame(estimate_rows)
    estimate_frame.to_csv(args.output_dir / "effective_diffusivity_estimates.csv", index=False)
    plot_diffusivity_estimates(
        estimate_frame,
        args.output_dir / "effective_diffusivity_estimates.png",
    )

    selection_rows: list[dict[str, float | str]] = []
    result_rows: list[dict[str, float | int | str | bool]] = []
    all_estimates = {
        row["trajectory"]: float(row["diffusivity_m2_s"])
        for row in estimate_rows
    }
    all_signal_scales = {
        row["trajectory"]: float(row["cooling_excess_q95_K"])
        for row in estimate_rows
    }
    for trajectory_index, (name, prepared) in enumerate(prepared_items.items()):
        other_estimates = [
            value for other_name, value in all_estimates.items() if other_name != name
        ]
        learned_diffusivity = float(
            np.median(other_estimates or list(all_estimates.values()))
        )
        other_signal_scales = [
            value
            for other_name, value in all_signal_scales.items()
            if other_name != name
        ]
        signal_sd = (
            float(args.signal_sd)
            if args.signal_sd is not None
            else float(np.median(other_signal_scales or list(all_signal_scales.values())))
        )
        observations = make_censored_observations(
            prepared,
            fraction_saturated=args.fraction_saturated,
            noise_sd=args.noise_sd,
            observation_stride=args.observation_stride,
            seed=args.seed + trajectory_index,
        )
        print(
            f"{name}: {len(np.asarray(observations['y_obs']))} observations, "
            f"{np.sum(np.asarray(observations['sat_mask']))} saturated, "
            f"signal_sd={signal_sd:.2f} K"
        )

        selected_lengths: dict[str, float] = {}
        predictions: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            if method.endswith("source-scale"):
                selected_lengthscale = prepared.source_lengthscale
                source_config = make_config(
                    prepared,
                    method=method,
                    lengthscale=selected_lengthscale,
                    diffusivity=learned_diffusivity,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                )
                method_selection_rows = [
                    {
                        "trajectory": prepared.name,
                        "method": method,
                        "lengthscale_m": selected_lengthscale,
                        "unsaturated_nll": unsaturated_nll(
                            np.asarray(observations["x_obs"]),
                            np.asarray(observations["y_obs"]),
                            np.asarray(observations["sat_mask"], dtype=bool),
                            source_config,
                        ),
                    }
                ]
            else:
                selected_lengthscale, method_selection_rows = select_lengthscale(
                    prepared,
                    observations,
                    method=method,
                    diffusivity=learned_diffusivity,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                )
            selection_rows.extend(method_selection_rows)
            selected_lengths[method] = selected_lengthscale
            config = make_config(
                prepared,
                method=method,
                lengthscale=selected_lengthscale,
                diffusivity=learned_diffusivity,
                signal_sd=signal_sd,
                noise_sd=args.noise_sd,
            )
            prediction = sample_censored_multiple_chains(
                observations,
                config,
                n_chains=args.chains,
                samples_per_chain=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=args.seed + 100 * trajectory_index + method_index,
            )
            predictions[method] = prediction
            result_rows.append(
                compute_metrics(
                    prepared,
                    observations,
                    method=method,
                    lengthscale=selected_lengthscale,
                    learned_diffusivity=learned_diffusivity,
                    signal_sd=signal_sd,
                    mean=prediction[0],
                    sd=prediction[1],
                    lower=prediction[2],
                    upper=prediction[3],
                    draws=prediction[4],
                )
            )
            print(
                f"  {method}: ell0={selected_lengthscale * 1e3:.2f} mm, "
                f"peak={prediction[0].max():.2f} K"
            )

        plot_reconstruction(
            prepared,
            observations,
            predictions,
            selected_lengths,
            learned_diffusivity,
            args.output_dir / f"{name}_reconstruction.png",
        )

    selection_frame = pd.DataFrame(selection_rows)
    selection_frame["selected"] = False
    for (trajectory, method), group in selection_frame.groupby(["trajectory", "method"]):
        selected_index = group["unsaturated_nll"].idxmin()
        selection_frame.loc[selected_index, "selected"] = True
    selection_frame.to_csv(
        args.output_dir / "lengthscale_selection.csv",
        index=False,
    )
    plot_lengthscale_selection(
        selection_frame,
        args.output_dir / "lengthscale_selection.png",
    )

    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(args.output_dir / "diffusion_kernel_results.csv", index=False)
    plot_metric_summary(
        result_frame,
        args.output_dir / "diffusion_kernel_metric_comparison.png",
    )
    print(result_frame.to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
