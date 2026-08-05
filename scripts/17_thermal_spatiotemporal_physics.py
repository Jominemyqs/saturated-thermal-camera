from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_thermal_module():
    module_path = ROOT / "scripts" / "16_thermal_diffusion_kernel.py"
    spec = importlib.util.spec_from_file_location("thermal_diffusion", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_diffusion"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


thermal = load_thermal_module()
gp2d = thermal.gp2d

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thermal_spatiotemporal_physics_short"
METHOD_ORDER = [
    "snapshot RBF",
    "snapshot age diffusion",
    "space-time heat",
    "physics mean + RBF",
    "physics mean + space-time",
]
METHOD_COLORS = {
    "snapshot RBF": "#E69F00",
    "snapshot age diffusion": "#CC79A7",
    "space-time heat": "#0072B2",
    "physics mean + RBF": "#D55E00",
    "physics mean + space-time": "#009E73",
}
RECONSTRUCTION_METHODS = [
    "snapshot RBF",
    "space-time heat",
    "physics mean + RBF",
    "physics mean + space-time",
]


def build_one_step_physics_components(
    prepared,
    *,
    diffusivity: float,
    cooling_rate: float,
    threshold: float,
    source_flux_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a diffused clipped-frame background and current-source response."""
    background = np.zeros_like(prepared.history)
    source_response = np.zeros_like(prepared.history)
    time_steps = np.diff(prepared.times)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    for index in range(1, len(prepared.times)):
        dt = float(time_steps[index - 1])
        sigma_y = np.sqrt(2.0 * diffusivity * dt) / dy
        sigma_x = np.sqrt(2.0 * diffusivity * dt) / dx
        previous_clipped_excess = np.maximum(
            np.minimum(prepared.history[index - 1], threshold) - prepared.ambient,
            0.0,
        )
        background[index] = gaussian_filter(
            previous_clipped_excess,
            sigma=(sigma_y, sigma_x),
            mode="constant",
            cval=0.0,
        )
        background[index] *= np.exp(-cooling_rate * dt)
        active_source = np.where(
            prepared.heat_flux[index] >= source_flux_threshold,
            prepared.heat_flux[index],
            0.0,
        )
        source_response[index] = active_source * dt
    return background, source_response


def calibrate_source_coupling(
    prepared,
    background: np.ndarray,
    source_response: np.ndarray,
) -> tuple[float, int]:
    """Calibrate source coupling from an uncensored simulation trajectory."""
    target = prepared.history - prepared.ambient - background
    active = source_response > 0.0
    source_values = source_response[active]
    target_values = target[active]
    denominator = float(np.dot(source_values, source_values))
    if denominator <= 0.0:
        raise ValueError(f"No active source samples found for {prepared.name}")
    coupling = max(float(np.dot(source_values, target_values)) / denominator, 0.0)
    return coupling, len(source_values)


def make_physics_mean_function(prepared, mean_history: np.ndarray):
    interpolator = RegularGridInterpolator(
        (prepared.times, prepared.ys, prepared.xs),
        mean_history,
        method="linear",
        bounds_error=False,
        fill_value=prepared.ambient,
    )

    def mean_function(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        if p.shape[1] < 3:
            raise ValueError("Physics mean points must include x, y, and time")
        return np.asarray(interpolator(np.column_stack([p[:, 2], p[:, 1], p[:, 0]])))

    return mean_function


def select_history_indices(times: np.ndarray, lags: list[float]) -> np.ndarray:
    target_time = float(times[-1])
    indices = [int(np.argmin(np.abs(times - (target_time - lag)))) for lag in lags]
    return np.asarray(sorted(set(indices)), dtype=int)


def make_multitime_observations(
    prepared,
    *,
    threshold: float,
    history_indices: np.ndarray,
    observation_stride: int,
    noise_sd: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    coarse_mask = np.zeros(prepared.truth.shape, dtype=bool)
    coarse_mask[::observation_stride, ::observation_stride] = True
    point_blocks = []
    value_blocks = []
    saturation_blocks = []
    frame_records: list[dict[str, object]] = []
    saturation_count = np.zeros(prepared.truth.shape, dtype=int)

    for time_index in history_indices:
        measured = prepared.history[time_index] + rng.normal(
            0.0,
            noise_sd,
            size=prepared.truth.shape,
        )
        saturated = measured >= threshold
        included = coarse_mask | saturated
        points = np.column_stack(
            [
                prepared.points,
                np.full(len(prepared.points), prepared.times[time_index]),
            ]
        )
        point_blocks.append(points[included.ravel()])
        value_blocks.append(np.minimum(measured[included], threshold))
        saturation_blocks.append(saturated[included])
        saturation_count += saturated.astype(int)
        frame_records.append(
            {
                "time_index": int(time_index),
                "time": float(prepared.times[time_index]),
                "points": points[included.ravel()],
                "values": np.minimum(measured[included], threshold),
                "sat_mask": saturated[included],
                "included": included,
                "clipped_full": np.minimum(measured, threshold),
            }
        )

    prediction_points = np.column_stack(
        [prepared.points, np.full(len(prepared.points), prepared.times[-1])]
    )
    return {
        "x_obs": np.vstack(point_blocks),
        "y_obs": np.concatenate(value_blocks),
        "sat_mask": np.concatenate(saturation_blocks),
        "x_pred": prediction_points,
        "threshold": threshold,
        "frames": frame_records,
        "saturation_count": saturation_count,
    }


def current_frame_observations(multitime: dict[str, object]) -> dict[str, object]:
    frame = multitime["frames"][-1]
    assert isinstance(frame, dict)
    return {
        "x_obs": np.asarray(frame["points"]),
        "y_obs": np.asarray(frame["values"]),
        "sat_mask": np.asarray(frame["sat_mask"], dtype=bool),
        "x_pred": np.asarray(multitime["x_pred"]),
        "threshold": float(multitime["threshold"]),
        "clipped_full": np.asarray(frame["clipped_full"]),
    }


def age_observations(prepared, current: dict[str, object]) -> dict[str, object]:
    time_points = np.asarray(current["x_obs"])
    age_flat = prepared.age.ravel()
    grid_lookup = {
        (float(point[0]), float(point[1])): age
        for point, age in zip(prepared.points, age_flat)
    }
    observation_ages = np.asarray(
        [grid_lookup[(float(point[0]), float(point[1]))] for point in time_points]
    )
    return {
        **current,
        "x_obs": np.column_stack([time_points[:, :2], observation_ages]),
        "x_pred": np.column_stack([prepared.points, age_flat]),
    }


def make_config(
    prepared,
    *,
    kernel: str,
    diffusivity: float,
    cooling_rate: float,
    signal_sd: float,
    noise_sd: float,
    mean_function=None,
):
    return gp2d.GP2DConfig(
        mean_temp=prepared.ambient,
        mean_function=mean_function,
        signal_sd=signal_sd,
        kernel=kernel,
        lengthscale=prepared.source_lengthscale,
        diffusivity=diffusivity if kernel in {"diffusion_gibbs", "spatiotemporal_heat"} else None,
        cooling_rate=cooling_rate if kernel == "spatiotemporal_heat" else 0.0,
        noise_sd=noise_sd,
        relative_jitter=1e-7,
    )


def compute_metrics(
    prepared,
    *,
    method: str,
    observations: dict[str, object],
    prediction,
    diffusivity: float,
    cooling_rate: float,
    signal_sd: float,
    source_coupling: float,
    n_frames: int,
) -> dict[str, object]:
    mean, sd, lower, upper, draws = prediction
    truth = prepared.truth.ravel()
    excess = truth - prepared.ambient
    threshold = float(observations["threshold"])
    hot = truth >= threshold
    peak_index = int(np.argmax(truth))
    crps = thermal.empirical_crps(draws, truth)
    return {
        "trajectory": prepared.name,
        "method": method,
        "diffusivity_m2_s": diffusivity,
        "cooling_rate_1_s": cooling_rate,
        "signal_sd_K": signal_sd,
        "source_coupling": source_coupling,
        "n_frames": n_frames,
        "n_observations": len(np.asarray(observations["y_obs"])),
        "n_saturated": int(np.sum(np.asarray(observations["sat_mask"]))),
        "threshold_K": threshold,
        "excess_field_rel_l2": float(np.linalg.norm(mean - truth) / np.linalg.norm(excess)),
        "hot_region_rel_l2": float(
            np.linalg.norm(mean[hot] - truth[hot]) / np.linalg.norm(excess[hot])
        ),
        "peak_true_K": float(np.max(truth)),
        "peak_predicted_K": float(np.max(mean)),
        "peak_absolute_error_K": abs(float(np.max(mean) - np.max(truth))),
        "posterior_sd_at_true_peak_K": float(sd[peak_index]),
        "true_peak_in_95": bool(lower[peak_index] <= truth[peak_index] <= upper[peak_index]),
        "pointwise_95_coverage": float(np.mean((lower <= truth) & (truth <= upper))),
        "hot_region_95_coverage": float(
            np.mean((lower[hot] <= truth[hot]) & (truth[hot] <= upper[hot]))
        ),
        "mean_crps_K": float(np.mean(crps)),
        "hot_region_crps_K": float(np.mean(crps[hot])),
    }


def plot_reconstruction(
    prepared,
    current: dict[str, object],
    multitime: dict[str, object],
    physics_mean: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, ...]],
    out_path: Path,
) -> None:
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    vmin = prepared.ambient
    vmax = float(np.max(prepared.truth))
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), constrained_layout=True)
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
        np.asarray(current["clipped_full"]),
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[0, 1].set_title(f"Final clipped field ({float(current['threshold']):.2f} K)")
    axes[0, 2].imshow(
        physics_mean,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[0, 2].set_title("One-step physics mean")
    count_image = axes[0, 3].imshow(
        np.asarray(multitime["saturation_count"]),
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=0,
        aspect="auto",
    )
    axes[0, 3].set_title("Times pixel was saturated")

    for axis, method in zip(axes[1], RECONSTRUCTION_METHODS):
        mean = predictions[method][0].reshape(prepared.truth.shape)
        axis.imshow(
            mean,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_title(method)
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
        count_image,
        ax=axes[0, 3],
        fraction=0.046,
        pad=0.03,
        label="Frame count",
    )
    figure.suptitle(prepared.name + ": space-time and physics-mean censored GP")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_metrics(results: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("excess_field_rel_l2", "Relative field error\n(temperature excess)"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_crps_K", "Hot-region CRPS (K)"),
        ("hot_region_95_coverage", "Hot-region 95% coverage"),
    ]
    trajectories = list(dict.fromkeys(results["trajectory"]))
    x = np.arange(len(trajectories))
    width = 0.16
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.4), constrained_layout=True)
    indexed = results.set_index(["trajectory", "method"])
    for axis, (metric, label) in zip(axes, metrics):
        for method_index, method in enumerate(METHOD_ORDER):
            values = [float(indexed.loc[(name, method), metric]) for name in trajectories]
            axis.bar(
                x + (method_index - 2.0) * width,
                values,
                width,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.set_xticks(x, [name.replace("ScanPath_", "\n") for name in trajectories])
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Space-time heat kernel and source-driven mean")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test spatiotemporal heat covariance and a source-driven GP mean."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectories", nargs="+", default=thermal.DEFAULT_TRAJECTORIES)
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--history-lags", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--burn-in", type=int, default=250)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared_items = {}
    estimate_rows = {}
    for name in args.trajectories:
        prepared, row = thermal.prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        prepared_items[name] = prepared
        estimate_rows[name] = row

    coupling_calibrations = {}
    coupling_rows = []
    for name, prepared in prepared_items.items():
        row = estimate_rows[name]
        background, source_response = build_one_step_physics_components(
            prepared,
            diffusivity=float(row["diffusivity_m2_s"]),
            cooling_rate=float(row["cooling_rate_1_s"]),
            threshold=float(
                np.quantile(prepared.truth, 1.0 - args.fraction_saturated)
            ),
            source_flux_threshold=args.source_flux_threshold,
        )
        coupling, n_source_samples = calibrate_source_coupling(
            prepared,
            background,
            source_response,
        )
        coupling_calibrations[name] = coupling
        coupling_rows.append(
            {
                "trajectory": name,
                "source_coupling": coupling,
                "n_source_samples": n_source_samples,
            }
        )

    result_rows = []
    mean_rows = []
    for trajectory_index, (name, prepared) in enumerate(prepared_items.items()):
        other_rows = [row for other_name, row in estimate_rows.items() if other_name != name]
        training_rows = other_rows or list(estimate_rows.values())
        diffusivity = float(np.median([float(row["diffusivity_m2_s"]) for row in training_rows]))
        cooling_rate = float(np.median([float(row["cooling_rate_1_s"]) for row in training_rows]))
        signal_sd = float(np.median([float(row["cooling_excess_q95_K"]) for row in training_rows]))
        threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))

        training_couplings = [
            value
            for other_name, value in coupling_calibrations.items()
            if other_name != name
        ]
        coupling = float(
            np.median(training_couplings or list(coupling_calibrations.values()))
        )
        background, source_response = build_one_step_physics_components(
            prepared,
            diffusivity=diffusivity,
            cooling_rate=cooling_rate,
            threshold=threshold,
            source_flux_threshold=args.source_flux_threshold,
        )
        mean_history = prepared.ambient + background + coupling * source_response
        physics_mean_function = make_physics_mean_function(prepared, mean_history)

        history_indices = select_history_indices(prepared.times, args.history_lags)
        multitime = make_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + trajectory_index,
        )
        current = current_frame_observations(multitime)
        age_current = age_observations(prepared, current)

        configurations = {
            "snapshot RBF": (
                current,
                make_config(
                    prepared,
                    kernel="rbf",
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                ),
            ),
            "snapshot age diffusion": (
                age_current,
                make_config(
                    prepared,
                    kernel="diffusion_gibbs",
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                ),
            ),
            "space-time heat": (
                multitime,
                make_config(
                    prepared,
                    kernel="spatiotemporal_heat",
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                ),
            ),
            "physics mean + RBF": (
                current,
                make_config(
                    prepared,
                    kernel="rbf",
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                    mean_function=physics_mean_function,
                ),
            ),
            "physics mean + space-time": (
                multitime,
                make_config(
                    prepared,
                    kernel="spatiotemporal_heat",
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    noise_sd=args.noise_sd,
                    mean_function=physics_mean_function,
                ),
            ),
        }

        predictions = {}
        for method_index, method in enumerate(METHOD_ORDER):
            observations, config = configurations[method]
            prediction = thermal.sample_censored_multiple_chains(
                observations,
                config,
                n_chains=args.chains,
                samples_per_chain=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=args.seed + 100 * trajectory_index + method_index,
            )
            predictions[method] = prediction
            frames_used = len(history_indices) if "space-time" in method else 1
            result_rows.append(
                compute_metrics(
                    prepared,
                    method=method,
                    observations=observations,
                    prediction=prediction,
                    diffusivity=diffusivity,
                    cooling_rate=cooling_rate,
                    signal_sd=signal_sd,
                    source_coupling=coupling,
                    n_frames=frames_used,
                )
            )
            print(
                f"{name}, {method}: peak={prediction[0].max():.2f} K, "
                f"observations={len(np.asarray(observations['y_obs']))}"
            )

        mean_truth = prepared.truth
        mean_prediction = mean_history[-1]
        hot = mean_truth >= threshold
        mean_rows.append(
            {
                "trajectory": name,
                "diffusivity_m2_s": diffusivity,
                "cooling_rate_1_s": cooling_rate,
                "signal_sd_K": signal_sd,
                "source_coupling": coupling,
                "source_coupling_calibration": "leave-one-trajectory-out",
                "physics_mean_peak_K": float(np.max(mean_prediction)),
                "true_peak_K": float(np.max(mean_truth)),
                "physics_mean_excess_rel_l2": float(
                    np.linalg.norm(mean_prediction - mean_truth)
                    / np.linalg.norm(mean_truth - prepared.ambient)
                ),
                "physics_mean_hot_rel_l2": float(
                    np.linalg.norm(mean_prediction[hot] - mean_truth[hot])
                    / np.linalg.norm((mean_truth - prepared.ambient)[hot])
                ),
                "history_indices": ";".join(str(int(value)) for value in history_indices),
                "history_times_s": ";".join(
                    f"{prepared.times[value]:.6f}" for value in history_indices
                ),
            }
        )
        plot_reconstruction(
            prepared,
            current,
            multitime,
            mean_history[-1],
            predictions,
            args.output_dir / f"{name}_reconstruction.png",
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "spatiotemporal_physics_results.csv", index=False)
    mean_frame = pd.DataFrame(mean_rows)
    mean_frame.to_csv(args.output_dir / "physics_mean_fits.csv", index=False)
    pd.DataFrame(coupling_rows).to_csv(
        args.output_dir / "source_coupling_calibration.csv",
        index=False,
    )
    plot_metrics(results, args.output_dir / "spatiotemporal_physics_metric_comparison.png")
    print(results.to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
