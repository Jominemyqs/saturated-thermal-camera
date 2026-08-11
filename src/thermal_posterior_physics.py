from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as shift_image

from src.censored_gp import RBFConfig, sample_multiple_chains
from src.diffusion import cooling_age, estimate_effective_diffusivity
from src.metrics import empirical_crps
from src.thermal_trajectory import SurfaceGridProjector, ThermalTrajectory


DEVELOPMENT_TRAJECTORIES = (
    "DiagonalScanPath_8",
    "HorizontalScanPath_13",
    "SpiralScanPath_13",
)


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
    grid_x, grid_y = np.meshgrid(xs, ys)
    widths = []
    for field in np.asarray(heat_flux):
        peak = float(np.max(field))
        if peak < 10_000.0:
            continue
        weights = np.where(field >= 0.01 * peak, np.maximum(field, 0.0), 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        center_x = float(np.sum(weights * grid_x) / total)
        center_y = float(np.sum(weights * grid_y) / total)
        radial_variance = float(
            np.sum(
                weights
                * ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
            )
            / total
        )
        widths.append(np.sqrt(radial_variance / 2.0))
    if not widths:
        raise ValueError("No active heat-flux footprints were found")
    return float(np.median(widths))


def trajectory_path(dataset_dir: Path, name: str) -> Path:
    for path in (
        dataset_dir / f"{name}.xdmf",
        dataset_dir / f"Copy of {name}.xdmf",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {name!r} in {dataset_dir}")


def prepare_trajectory(
    dataset_dir: Path,
    name: str,
    *,
    nx: int,
    ny: int,
    heat_flux_cutoff: float,
) -> tuple[PreparedTrajectory, dict[str, float | int | str]]:
    trajectory = ThermalTrajectory(trajectory_path(dataset_dir, name))
    indices = list(range(trajectory.n_times))
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
        heat_flux, projector.xs, projector.ys
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
    row: dict[str, float | int | str] = {
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
    return prepared, row


def paired_camera_observations(
    prepared: PreparedTrajectory,
    *,
    previous_index: int,
    current_index: int,
    threshold: float,
    observation_stride: int,
    noise_sd: float,
    seed: int,
) -> dict[str, object]:
    """Create one shared noisy previous/current camera realization."""
    rng = np.random.default_rng(seed)
    fixed_mask = np.zeros(prepared.truth.shape, dtype=bool)
    fixed_mask[::observation_stride, ::observation_stride] = True
    frames = []
    for time_index in (previous_index, current_index):
        measured = prepared.history[time_index] + rng.normal(
            0.0, noise_sd, size=prepared.truth.shape
        )
        saturated = measured >= threshold
        points = np.column_stack(
            [
                prepared.points,
                np.full(len(prepared.points), prepared.times[time_index]),
            ]
        )
        frames.append(
            {
                "time_index": time_index,
                "time": float(prepared.times[time_index]),
                "points": points[fixed_mask.ravel()],
                "values": np.minimum(measured[fixed_mask], threshold),
                "sat_mask": saturated[fixed_mask],
                "clipped_full": np.minimum(measured, threshold),
                "saturated_full": saturated,
            }
        )
    current_points = np.column_stack(
        [
            prepared.points,
            np.full(len(prepared.points), prepared.times[current_index]),
        ]
    )
    current = {
        "x_obs": np.asarray(frames[1]["points"]),
        "y_obs": np.asarray(frames[1]["values"]),
        "sat_mask": np.asarray(frames[1]["sat_mask"], dtype=bool),
        "x_pred": current_points,
        "threshold": threshold,
        "clipped_full": np.asarray(frames[1]["clipped_full"]),
    }
    return {
        "frames": frames,
        "fixed_observation_mask": fixed_mask,
        "current": current,
    }


def previous_posterior_observations(
    prepared: PreparedTrajectory,
    *,
    frame: dict[str, object],
    fixed_mask: np.ndarray,
    threshold: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    clipped = np.asarray(frame["clipped_full"], dtype=float)
    saturated = np.asarray(frame["saturated_full"], dtype=bool)
    included = np.asarray(fixed_mask, dtype=bool) | saturated
    points = np.column_stack(
        [prepared.points, np.full(len(prepared.points), float(frame["time"]))]
    )
    return (
        {
            "x_obs": points[included.ravel()],
            "y_obs": clipped[included],
            "sat_mask": saturated[included],
            "x_pred": points[saturated.ravel()],
            "threshold": threshold,
        },
        clipped,
        saturated,
    )


def infer_previous_censored_posterior(
    prepared: PreparedTrajectory,
    *,
    frame: dict[str, object],
    fixed_mask: np.ndarray,
    threshold: float,
    signal_sd: float,
    lengthscale: float,
    noise_sd: float,
    n_chains: int,
    samples_per_chain: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    observations, clipped, saturated = previous_posterior_observations(
        prepared,
        frame=frame,
        fixed_mask=fixed_mask,
        threshold=threshold,
    )
    n_draws = n_chains * samples_per_chain
    draws = np.repeat(clipped[None, :, :], n_draws, axis=0)
    if np.any(saturated):
        config = RBFConfig(
            mean_temp=prepared.ambient,
            signal_sd=signal_sd,
            lengthscale=lengthscale,
            noise_sd=noise_sd,
        )
        prediction = sample_multiple_chains(
            observations,
            config,
            n_chains=n_chains,
            samples_per_chain=samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )
        draws[:, saturated] = prediction[4]
    posterior_mean = np.mean(draws, axis=0)
    previous_truth = prepared.history[int(frame["time_index"])]
    diagnostics = {
        "previous_n_saturated": int(np.sum(saturated)),
        "previous_saturated_fraction": float(np.mean(saturated)),
        "previous_clipped_hot_mae_K": float(
            np.mean(np.abs(clipped[saturated] - previous_truth[saturated]))
        ),
        "previous_posterior_hot_mae_K": float(
            np.mean(np.abs(posterior_mean[saturated] - previous_truth[saturated]))
        ),
    }
    return posterior_mean, draws, diagnostics


def diffuse_and_cool(
    prepared: PreparedTrajectory,
    fields: np.ndarray,
    *,
    previous_index: int,
    current_index: int,
    diffusivity: float,
    cooling_rate: float,
) -> np.ndarray:
    values = np.asarray(fields, dtype=float)
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    sigma_y = np.sqrt(2.0 * diffusivity * dt) / dy
    sigma_x = np.sqrt(2.0 * diffusivity * dt) / dx
    sigma = (sigma_y, sigma_x) if values.ndim == 2 else (0.0, sigma_y, sigma_x)
    excess = np.maximum(values - prepared.ambient, 0.0)
    propagated = gaussian_filter(excess, sigma=sigma, mode="constant", cval=0.0)
    return propagated * np.exp(-cooling_rate * dt)


def translate_field(
    field: np.ndarray,
    *,
    displacement: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return f(x-d): positive d translates the field toward positive axes."""
    values = np.asarray(field, dtype=float)
    shift_pixels = (displacement[1] / dy, displacement[0] / dx)
    if values.ndim == 3:
        shift_pixels = (0.0, *shift_pixels)
    return shift_image(
        values,
        shift=shift_pixels,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def advect_diffuse_and_cool(
    prepared: PreparedTrajectory,
    fields: np.ndarray,
    *,
    previous_index: int,
    current_index: int,
    displacement: np.ndarray,
    diffusivity: float,
    cooling_rate: float,
) -> np.ndarray:
    propagated = diffuse_and_cool(
        prepared,
        fields,
        previous_index=previous_index,
        current_index=current_index,
        diffusivity=diffusivity,
        cooling_rate=cooling_rate,
    )
    return translate_field(
        propagated,
        displacement=np.asarray(displacement, dtype=float),
        dx=float(np.mean(np.diff(prepared.xs))),
        dy=float(np.mean(np.diff(prepared.ys))),
    )


def source_centroid_path(
    prepared: PreparedTrajectory,
    *,
    source_flux_threshold: float,
) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(prepared.xs, prepared.ys)
    positions = np.full((len(prepared.times), 2), np.nan)
    for index, flux in enumerate(prepared.heat_flux):
        weights = np.where(flux >= source_flux_threshold, flux, 0.0)
        total = float(np.sum(weights))
        if total > 0.0:
            positions[index] = (
                float(np.sum(weights * grid_x) / total),
                float(np.sum(weights * grid_y) / total),
            )
    valid = np.isfinite(positions[:, 0])
    if not np.any(valid):
        raise ValueError(f"No active source found for {prepared.name}")
    indices = np.arange(len(positions))
    for coordinate in range(2):
        positions[:, coordinate] = np.interp(
            indices, indices[valid], positions[valid, coordinate]
        )
    return positions


def current_source_field(
    prepared: PreparedTrajectory,
    *,
    previous_index: int,
    current_index: int,
    source_coupling: float,
    source_flux_threshold: float,
) -> np.ndarray:
    dt = float(prepared.times[current_index] - prepared.times[previous_index])
    active_flux = np.where(
        prepared.heat_flux[current_index] >= source_flux_threshold,
        prepared.heat_flux[current_index],
        0.0,
    )
    return source_coupling * active_flux * dt


def posterior_physics_means(
    prepared: PreparedTrajectory,
    previous_posterior_mean: np.ndarray,
    *,
    previous_index: int,
    current_index: int,
    diffusivity: float,
    cooling_rate: float,
    source_coupling: float,
    source_flux_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = current_source_field(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        source_coupling=source_coupling,
        source_flux_threshold=source_flux_threshold,
    )
    diffusive_background = diffuse_and_cool(
        prepared,
        previous_posterior_mean,
        previous_index=previous_index,
        current_index=current_index,
        diffusivity=diffusivity,
        cooling_rate=cooling_rate,
    )
    path = source_centroid_path(
        prepared, source_flux_threshold=source_flux_threshold
    )
    displacement = path[current_index] - path[previous_index]
    advective_background = translate_field(
        diffusive_background,
        displacement=displacement,
        dx=float(np.mean(np.diff(prepared.xs))),
        dy=float(np.mean(np.diff(prepared.ys))),
    )
    ordinary = prepared.ambient + diffusive_background + source
    advective = prepared.ambient + advective_background + source
    return ordinary, advective, displacement


def grid_mean_function(prepared: PreparedTrajectory, field: np.ndarray):
    flat = np.asarray(field, dtype=float).ravel()
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))

    def mean_function(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        ix = np.rint((values[:, 0] - prepared.xs[0]) / dx).astype(int)
        iy = np.rint((values[:, 1] - prepared.ys[0]) / dy).astype(int)
        if np.any(ix < 0) or np.any(ix >= len(prepared.xs)):
            raise ValueError("Mean-function x coordinates fall outside the grid")
        if np.any(iy < 0) or np.any(iy >= len(prepared.ys)):
            raise ValueError("Mean-function y coordinates fall outside the grid")
        return flat[iy * len(prepared.xs) + ix]

    return mean_function


def build_calibration_components(
    prepared: PreparedTrajectory,
    *,
    diffusivity: float,
    cooling_rate: float,
    threshold: float,
    source_flux_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    background = np.zeros_like(prepared.history)
    source_response = np.zeros_like(prepared.history)
    for index in range(1, len(prepared.times)):
        background[index] = diffuse_and_cool(
            prepared,
            np.minimum(prepared.history[index - 1], threshold),
            previous_index=index - 1,
            current_index=index,
            diffusivity=diffusivity,
            cooling_rate=cooling_rate,
        )
        dt = float(prepared.times[index] - prepared.times[index - 1])
        active_flux = np.where(
            prepared.heat_flux[index] >= source_flux_threshold,
            prepared.heat_flux[index],
            0.0,
        )
        source_response[index] = active_flux * dt
    return background, source_response


def calibrate_source_coupling(
    prepared: PreparedTrajectory,
    background: np.ndarray,
    source_response: np.ndarray,
) -> tuple[float, int]:
    target = prepared.history - prepared.ambient - background
    active = source_response > 0.0
    source_values = source_response[active]
    target_values = target[active]
    denominator = float(np.dot(source_values, source_values))
    if denominator <= 0.0:
        raise ValueError(f"No active source samples found for {prepared.name}")
    coupling = max(float(np.dot(source_values, target_values)) / denominator, 0.0)
    return coupling, len(source_values)


def prediction_metrics(
    prepared: PreparedTrajectory,
    *,
    method: str,
    observations: dict[str, object],
    prediction: tuple[np.ndarray, ...],
) -> dict[str, float | int | str | bool]:
    mean, sd, lower, upper, draws = prediction
    truth = prepared.truth.ravel()
    excess = truth - prepared.ambient
    threshold = float(observations["threshold"])
    hot = truth >= threshold
    top_one = truth >= float(np.quantile(truth, 0.99))
    peak_index = int(np.argmax(truth))
    crps = empirical_crps(draws, truth)
    return {
        "trajectory": prepared.name,
        "method": method,
        "n_observations": len(np.asarray(observations["y_obs"])),
        "n_saturated": int(np.sum(np.asarray(observations["sat_mask"]))),
        "threshold_K": threshold,
        "excess_field_rel_l2": float(
            np.linalg.norm(mean - truth) / np.linalg.norm(excess)
        ),
        "peak_true_K": float(np.max(truth)),
        "peak_predicted_K": float(np.max(mean)),
        "peak_absolute_error_K": abs(float(np.max(mean) - np.max(truth))),
        "posterior_sd_at_true_peak_K": float(sd[peak_index]),
        "true_peak_in_95": bool(
            lower[peak_index] <= truth[peak_index] <= upper[peak_index]
        ),
        "pointwise_95_coverage": float(
            np.mean((lower <= truth) & (truth <= upper))
        ),
        "hot_region_95_coverage": float(
            np.mean((lower[hot] <= truth[hot]) & (truth[hot] <= upper[hot]))
        ),
        "hot_region_95_interval_width_K": float(
            np.mean((upper - lower)[hot])
        ),
        "mean_95_interval_width_K": float(np.mean(upper - lower)),
        "mean_crps_K": float(np.mean(crps)),
        "hot_region_crps_K": float(np.mean(crps[hot])),
        "fixed_top_01_crps_K": float(np.mean(crps[top_one])),
    }
