from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, shift as shift_image

from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_corrected_module():
    module_path = ROOT / "scripts" / "19_thermal_corrected_crps.py"
    spec = importlib.util.spec_from_file_location("thermal_corrected_crps", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_corrected_crps"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


corrected = load_corrected_module()
ablation = corrected.ablation
study = ablation.study
thermal = ablation.thermal
gp2d = study.gp2d

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "by_experiment"
SETUP_STAGE = "13_thermal_advection_setup"
REFERENCE_FRACTION = 0.03
METHODS = [
    "physics mean + RBF",
    "physics mean + space-time",
    "physics mean + advective space-time",
    "advective physics mean + space-time",
    "physics mean + forced space-time",
    "physics mean + advective forced space-time",
    "physics mean + advective forced + source amplitude",
]
STAGES = {
    "14_residual_advection": [
        "physics mean + RBF",
        "physics mean + space-time",
        "physics mean + advective space-time",
    ],
    "15_mean_advection": [
        "physics mean + space-time",
        "advective physics mean + space-time",
    ],
    "16_stochastic_forcing": [
        "physics mean + space-time",
        "physics mean + forced space-time",
    ],
    "17_advective_stochastic_forcing": [
        "physics mean + RBF",
        "physics mean + space-time",
        "physics mean + advective space-time",
        "physics mean + forced space-time",
        "physics mean + advective forced space-time",
    ],
    "18_source_amplitude_correction": [
        "physics mean + RBF",
        "physics mean + space-time",
        "physics mean + advective space-time",
        "physics mean + forced space-time",
        "physics mean + advective forced space-time",
        "physics mean + advective forced + source amplitude",
    ],
}
BASELINES = {
    "14_residual_advection": "physics mean + space-time",
    "15_mean_advection": "physics mean + space-time",
    "16_stochastic_forcing": "physics mean + space-time",
    "17_advective_stochastic_forcing": "physics mean + space-time",
    "18_source_amplitude_correction": "physics mean + advective forced space-time",
}
METHOD_COLORS = {
    "physics mean + RBF": "#D55E00",
    "physics mean + space-time": "#009E73",
    "physics mean + advective space-time": "#0072B2",
    "advective physics mean + space-time": "#CC79A7",
    "physics mean + forced space-time": "#E69F00",
    "physics mean + advective forced space-time": "#56B4E9",
    "physics mean + advective forced + source amplitude": "#9467BD",
}
SUMMARY_METRICS = [
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "excess_field_rel_l2",
    "peak_absolute_error_K",
    "hot_region_95_coverage",
    "pointwise_95_coverage",
    "mean_95_interval_width_K",
    "hot_region_95_interval_width_K",
]


def setup_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / SETUP_STAGE


def source_centroid_path(prepared, *, source_flux_threshold: float) -> np.ndarray:
    xx, yy = np.meshgrid(prepared.xs, prepared.ys)
    positions = np.full((len(prepared.times), 2), np.nan, dtype=float)
    for index, flux in enumerate(prepared.heat_flux):
        weights = np.where(flux >= source_flux_threshold, flux, 0.0)
        total = float(np.sum(weights))
        if total > 0.0:
            positions[index, 0] = float(np.sum(weights * xx) / total)
            positions[index, 1] = float(np.sum(weights * yy) / total)
    valid = np.isfinite(positions[:, 0])
    if not np.any(valid):
        raise ValueError(f"No active heat source found for {prepared.name}")
    indices = np.arange(len(positions))
    for coordinate in range(2):
        positions[:, coordinate] = np.interp(
            indices,
            indices[valid],
            positions[valid, coordinate],
        )
    return positions


def make_path_function(times: np.ndarray, positions: np.ndarray):
    def path_function(query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times, dtype=float).reshape(-1)
        return np.column_stack(
            [
                np.interp(query, times, positions[:, 0]),
                np.interp(query, times, positions[:, 1]),
            ]
        )

    return path_function


def build_advective_mean_components(
    prepared,
    *,
    diffusivity: float,
    cooling_rate: float,
    threshold: float,
    source_flux_threshold: float,
    source_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a moving-frame transport correction to the previous clipped field."""
    background = np.zeros_like(prepared.history)
    source_response = np.zeros_like(prepared.history)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    for index in range(1, len(prepared.times)):
        dt = float(prepared.times[index] - prepared.times[index - 1])
        displacement = source_positions[index] - source_positions[index - 1]
        previous_excess = np.maximum(
            np.minimum(prepared.history[index - 1], threshold) - prepared.ambient,
            0.0,
        )
        transported = shift_image(
            previous_excess,
            shift=(displacement[1] / dy, displacement[0] / dx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        sigma_y = np.sqrt(2.0 * diffusivity * dt) / dy
        sigma_x = np.sqrt(2.0 * diffusivity * dt) / dx
        background[index] = gaussian_filter(
            transported,
            sigma=(sigma_y, sigma_x),
            mode="constant",
            cval=0.0,
        )
        background[index] *= np.exp(-cooling_rate * dt)
        source_response[index] = np.where(
            prepared.heat_flux[index] >= source_flux_threshold,
            prepared.heat_flux[index] * dt,
            0.0,
        )
    return background, source_response


def make_mean_function(prepared, mean_history: np.ndarray):
    interpolator = RegularGridInterpolator(
        (prepared.times, prepared.ys, prepared.xs),
        mean_history,
        method="linear",
        bounds_error=False,
        fill_value=prepared.ambient,
    )

    def mean_function(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        query = np.column_stack([values[:, 2], values[:, 1], values[:, 0]])
        return np.asarray(interpolator(query), dtype=float)

    return mean_function


def make_zero_fill_function(prepared, history: np.ndarray):
    interpolator = RegularGridInterpolator(
        (prepared.times, prepared.ys, prepared.xs),
        history,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    def field_function(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        query = np.column_stack([values[:, 2], values[:, 1], values[:, 0]])
        return np.asarray(interpolator(query), dtype=float)

    return field_function


def calibrate_advective_source_coupling(
    prepared_items,
    *,
    fraction: float,
    diffusivity: float,
    cooling_rate: float,
    source_flux_threshold: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    values = []
    for name, prepared in prepared_items.items():
        threshold = float(np.quantile(prepared.truth, 1.0 - fraction))
        positions = source_centroid_path(
            prepared,
            source_flux_threshold=source_flux_threshold,
        )
        background, source_response = build_advective_mean_components(
            prepared,
            diffusivity=diffusivity,
            cooling_rate=cooling_rate,
            threshold=threshold,
            source_flux_threshold=source_flux_threshold,
            source_positions=positions,
        )
        coupling, n_samples = study.calibrate_source_coupling(
            prepared,
            background,
            source_response,
        )
        values.append(coupling)
        rows.append(
            {
                "trajectory": name,
                "fraction_saturated": fraction,
                "source_coupling": coupling,
                "n_source_samples": n_samples,
            }
        )
    return float(np.median(values)), pd.DataFrame(rows)


def residual_multipliers(args: argparse.Namespace) -> dict[str, float]:
    return {
        "signal_multiplier": args.signal_multiplier,
        "noise_multiplier": args.noise_multiplier,
        "beta_multiplier": args.beta_multiplier,
        "length_multiplier": args.length_multiplier,
    }


def build_model_context(
    args: argparse.Namespace,
    prepared,
    *,
    threshold: float,
    multitime: dict[str, object],
    current: dict[str, object],
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
    source_amplitude_fraction_sd: float,
    source_amplitude_timescale: float,
) -> tuple[dict[str, tuple[dict[str, object], object, float]], np.ndarray]:
    regular_history, regular_mean = ablation.physics_mean_for_trajectory(
        prepared,
        threshold=threshold,
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_coupling=regular_source_coupling,
        source_flux_threshold=args.source_flux_threshold,
    )
    positions = source_centroid_path(
        prepared,
        source_flux_threshold=args.source_flux_threshold,
    )
    advective_background, source_response = build_advective_mean_components(
        prepared,
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        threshold=threshold,
        source_flux_threshold=args.source_flux_threshold,
        source_positions=positions,
    )
    advective_history = (
        prepared.ambient
        + advective_background
        + advective_source_coupling * source_response
    )
    advective_mean = make_mean_function(prepared, advective_history)
    path_function = make_path_function(prepared.times, positions)
    _, regular_source_response = study.build_one_step_physics_components(
        prepared,
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        threshold=threshold,
        source_flux_threshold=args.source_flux_threshold,
    )
    source_basis = make_zero_fill_function(
        prepared,
        regular_source_coupling * regular_source_response,
    )

    multipliers = residual_multipliers(args)
    signal_sd = parameters["signal_sd"] * multipliers["signal_multiplier"]
    noise_sd = args.noise_sd * multipliers["noise_multiplier"]
    cooling_rate = parameters["cooling_rate"] * multipliers["beta_multiplier"]
    lengthscale = prepared.source_lengthscale * multipliers["length_multiplier"]

    def config(kernel: str, mean_function):
        base = study.make_config(
            prepared,
            kernel="rbf" if kernel == "rbf" else "spatiotemporal_heat",
            diffusivity=parameters["diffusivity"],
            cooling_rate=cooling_rate,
            signal_sd=signal_sd,
            noise_sd=noise_sd,
            mean_function=mean_function,
        )
        return replace(base, kernel=kernel, lengthscale=lengthscale)

    regular_rbf = config("rbf", regular_mean)
    regular_space_time = config("spatiotemporal_heat", regular_mean)
    advective_space_time = replace(
        config("spatiotemporal_advection", regular_mean),
        advection_path=path_function,
    )
    advective_mean_space_time = config("spatiotemporal_heat", advective_mean)
    forced_space_time = replace(
        config("spatiotemporal_forced_heat", regular_mean),
        forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
        forcing_quadrature_order=args.forcing_quadrature_order,
    )
    advective_forced_space_time = replace(
        config("spatiotemporal_advective_forced_heat", regular_mean),
        advection_path=path_function,
        forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
        forcing_quadrature_order=args.forcing_quadrature_order,
    )
    source_amplitude_space_time = replace(
        config(
            "spatiotemporal_advective_forced_heat_source_amplitude",
            regular_mean,
        ),
        advection_path=path_function,
        forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
        forcing_quadrature_order=args.forcing_quadrature_order,
        source_amplitude_basis=source_basis,
        source_amplitude_fraction_sd=source_amplitude_fraction_sd,
        source_amplitude_timescale=source_amplitude_timescale,
    )
    models = {
        "physics mean + RBF": (current, regular_rbf, regular_source_coupling),
        "physics mean + space-time": (
            multitime,
            regular_space_time,
            regular_source_coupling,
        ),
        "physics mean + advective space-time": (
            multitime,
            advective_space_time,
            regular_source_coupling,
        ),
        "advective physics mean + space-time": (
            multitime,
            advective_mean_space_time,
            advective_source_coupling,
        ),
        "physics mean + forced space-time": (
            multitime,
            forced_space_time,
            regular_source_coupling,
        ),
        "physics mean + advective forced space-time": (
            multitime,
            advective_forced_space_time,
            regular_source_coupling,
        ),
        "physics mean + advective forced + source amplitude": (
            multitime,
            source_amplitude_space_time,
            regular_source_coupling,
        ),
    }
    return models, regular_history


def implied_forcing_intensity(config) -> float:
    ell = float(config.forcing_lengthscale or config.lengthscale)
    beta = float(config.cooling_rate)
    nodes, weights = np.polynomial.laguerre.laggauss(
        config.forcing_quadrature_order
    )
    scales = ell**2 + 2.0 * config.diffusivity * nodes / beta
    zero_lag_integral = float(np.sum(weights * ell**2 / scales))
    return 2.0 * beta * config.signal_sd**2 / zero_lag_integral


def validate_kernels(models, points: np.ndarray) -> list[dict[str, float | str]]:
    probe_indices = np.linspace(0, len(points) - 1, min(70, len(points))).astype(int)
    probe = points[probe_indices]
    rows = []
    for method in (
        "physics mean + space-time",
        "physics mean + advective space-time",
        "physics mean + forced space-time",
        "physics mean + advective forced space-time",
        "physics mean + advective forced + source amplitude",
    ):
        config = models[method][1]
        matrix = gp2d.rbf_kernel(probe, probe, config)
        symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
        expected_diagonal = np.full(len(probe), config.signal_sd**2)
        if "source amplitude" in method:
            basis = np.asarray(config.source_amplitude_basis(probe), dtype=float)
            expected_diagonal += config.source_amplitude_fraction_sd**2 * basis**2
        diagonal_error = float(np.max(np.abs(np.diag(matrix) - expected_diagonal)))
        quadrature_reference_order = np.nan
        quadrature_relative_error = np.nan
        if "forced" in method:
            quadrature_reference_order = max(
                2 * config.forcing_quadrature_order,
                24,
            )
            reference = gp2d.rbf_kernel(
                probe,
                probe,
                replace(
                    config,
                    forcing_quadrature_order=quadrature_reference_order,
                ),
            )
            quadrature_relative_error = float(
                np.max(np.abs(matrix - reference)) / config.signal_sd**2
            )
        tolerance = 1e-8 * max(config.signal_sd**2, 1.0)
        if symmetry_error > tolerance or minimum_eigenvalue < -tolerance:
            raise AssertionError(
                f"Invalid covariance for {method}: symmetry={symmetry_error}, "
                f"minimum eigenvalue={minimum_eigenvalue}"
            )
        rows.append(
            {
                "method": method,
                "symmetry_error": symmetry_error,
                "minimum_eigenvalue": minimum_eigenvalue,
                "maximum_diagonal_error": diagonal_error,
                "quadrature_reference_order": quadrature_reference_order,
                "quadrature_relative_error": quadrature_relative_error,
            }
        )
    return rows


def select_source_amplitude_setting(
    args: argparse.Namespace,
    *,
    prepared_items,
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    setup_dir = setup_output_dir(args)
    selected_path = setup_dir / "selected_source_amplitude.csv"
    sensitivity_path = setup_dir / "source_amplitude_sensitivity.csv"
    summary_path = setup_dir / "source_amplitude_sensitivity_summary.csv"
    if (
        args.resume
        and selected_path.is_file()
        and sensitivity_path.is_file()
        and summary_path.is_file()
    ):
        selected = pd.read_csv(selected_path).iloc[0]
        return (
            float(selected["source_amplitude_fraction_sd"]),
            float(selected["source_amplitude_timescale_s"]),
            pd.read_csv(sensitivity_path),
            pd.read_csv(summary_path),
        )

    catalog = trajectory_catalog(args.dataset_dir)
    trajectory_indices = {record.name: index for index, record in enumerate(catalog)}
    rows = []
    method = "physics mean + advective forced + source amplitude"
    for name, prepared in prepared_items.items():
        trajectory_index = trajectory_indices[name]
        threshold = float(np.quantile(prepared.truth, 1.0 - REFERENCE_FRACTION))
        history_indices = study.select_history_indices(prepared.times, args.main_lags)
        multitime = corrected.fixed_mask_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current = study.current_frame_observations(multitime)
        base_dt = float(np.median(np.diff(prepared.times)))
        for fraction_sd in args.source_amplitude_fraction_sds:
            timescale_multipliers = args.source_amplitude_timescale_multipliers
            if np.isclose(fraction_sd, 0.0):
                timescale_multipliers = timescale_multipliers[:1]
            for timescale_multiplier in timescale_multipliers:
                timescale = base_dt * timescale_multiplier
                models, _ = build_model_context(
                    args,
                    prepared,
                    threshold=threshold,
                    multitime=multitime,
                    current=current,
                    parameters=parameters,
                    regular_source_coupling=regular_source_coupling,
                    advective_source_coupling=advective_source_coupling,
                    source_amplitude_fraction_sd=fraction_sd,
                    source_amplitude_timescale=timescale,
                )
                observations, config, source_coupling = models[method]
                row, prediction = ablation.run_method(
                    prepared,
                    method=method,
                    observations=observations,
                    config=config,
                    diffusivity=parameters["diffusivity"],
                    cooling_rate=config.cooling_rate,
                    source_coupling=source_coupling,
                    n_frames=len(history_indices),
                    chains=1,
                    samples=args.source_amplitude_tuning_samples,
                    burn_in=args.source_amplitude_tuning_burn_in,
                    thin=args.thin,
                    seed=args.seed + 900_000 + 10_000 * trajectory_index,
                    return_prediction=True,
                )
                row.update(corrected.fixed_region_metrics(prepared, prediction))
                row.update(
                    {
                        "trajectory": name,
                        "source_amplitude_fraction_sd": fraction_sd,
                        "source_amplitude_timescale_multiplier": timescale_multiplier,
                        "source_amplitude_timescale_s": timescale,
                    }
                )
                rows.append(row)
                print(
                    f"Source-amplitude sensitivity {name}: fraction SD="
                    f"{fraction_sd:g}, timescale x{timescale_multiplier:g}, "
                    f"top1 CRPS={row['fixed_top_01_crps_K']:.3f}",
                    flush=True,
                )
    sensitivity = pd.DataFrame(rows)
    summary = (
        sensitivity.groupby(
            [
                "source_amplitude_fraction_sd",
                "source_amplitude_timescale_multiplier",
                "source_amplitude_timescale_s",
            ],
            sort=False,
        )[
            [
                "fixed_top_01_crps_K",
                "mean_crps_K",
                "excess_field_rel_l2",
                "peak_absolute_error_K",
                "hot_region_95_coverage",
            ]
        ]
        .mean()
        .reset_index()
    )
    selected = summary.sort_values(
        ["fixed_top_01_crps_K", "mean_crps_K", "excess_field_rel_l2"]
    ).iloc[0]
    selected_frame = pd.DataFrame([selected])
    sensitivity.to_csv(sensitivity_path, index=False)
    summary.to_csv(summary_path, index=False)
    selected_frame.to_csv(selected_path, index=False)
    return (
        float(selected["source_amplitude_fraction_sd"]),
        float(selected["source_amplitude_timescale_s"]),
        sensitivity,
        summary,
    )


def run_all_models(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
    source_amplitude_fraction_sd: float,
    source_amplitude_timescale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = trajectory_catalog(args.dataset_dir)
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        records = [record for record in records if record.name in requested]
        missing = requested - {record.name for record in records}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    checkpoint_path = setup_output_dir(args) / "all_models_checkpoint.csv"
    if args.resume and checkpoint_path.is_file():
        checkpoint = pd.read_csv(checkpoint_path)
        rows = checkpoint.to_dict("records")
        completed = {
            (str(row["trajectory"]), float(row["fraction_saturated"]), str(row["method"]))
            for row in rows
        }
        print(f"Resuming after {len(completed)} completed fits", flush=True)
    else:
        rows = []
        completed = set()
    validation_rows = []

    for trajectory_index, record in enumerate(records):
        prepared, _ = thermal.prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        role = (
            "calibration"
            if record.name in ablation.DEVELOPMENT_TRAJECTORIES
            else "evaluation"
        )
        history_indices = study.select_history_indices(prepared.times, args.main_lags)
        observation_seed = args.seed + 10_000 * trajectory_index
        reference_points: np.ndarray | None = None

        for fraction_index, fraction in enumerate(args.fractions_saturated):
            threshold = float(np.quantile(prepared.truth, 1.0 - fraction))
            multitime = corrected.fixed_mask_multitime_observations(
                prepared,
                threshold=threshold,
                history_indices=history_indices,
                observation_stride=args.observation_stride,
                noise_sd=args.noise_sd,
                seed=observation_seed,
            )
            current = study.current_frame_observations(multitime)
            models, _ = build_model_context(
                args,
                prepared,
                threshold=threshold,
                multitime=multitime,
                current=current,
                parameters=parameters,
                regular_source_coupling=regular_source_coupling,
                advective_source_coupling=advective_source_coupling,
                source_amplitude_fraction_sd=source_amplitude_fraction_sd,
                source_amplitude_timescale=source_amplitude_timescale,
            )
            space_time_points = np.asarray(multitime["x_obs"])
            if reference_points is None:
                reference_points = space_time_points.copy()
            elif not np.array_equal(reference_points, space_time_points):
                raise AssertionError(
                    f"Observation coordinates changed across ceilings for {record.name}"
                )
            if fraction_index == 0:
                for item in validate_kernels(models, space_time_points):
                    item.update({"trajectory": record.name, "family": record.family})
                    validation_rows.append(item)

            prediction_seed = args.seed + 100_000 * trajectory_index + 1_000 * fraction_index
            for method in METHODS:
                key = (record.name, float(fraction), method)
                if key in completed:
                    continue
                observations, config, source_coupling = models[method]
                row, prediction = ablation.run_method(
                    prepared,
                    method=method,
                    observations=observations,
                    config=config,
                    diffusivity=parameters["diffusivity"],
                    cooling_rate=config.cooling_rate,
                    source_coupling=source_coupling,
                    n_frames=1 if method == "physics mean + RBF" else len(history_indices),
                    chains=args.chains,
                    samples=args.samples,
                    burn_in=args.burn_in,
                    thin=args.thin,
                    seed=prediction_seed,
                    return_prediction=True,
                )
                row.update(corrected.fixed_region_metrics(prepared, prediction))
                row.update(
                    {
                        "family": record.family,
                        "run_index": record.run_index,
                        "role": role,
                        "fraction_saturated": fraction,
                        "observed_saturated_fraction": float(
                            np.mean(np.asarray(observations["sat_mask"], dtype=bool))
                        ),
                        "observation_design": f"fixed_stride_{args.observation_stride}",
                        "residual_lengthscale_m": config.lengthscale,
                        "forcing_lengthscale_m": (
                            config.forcing_lengthscale
                            if "forced" in method
                            else np.nan
                        ),
                        "forcing_intensity_K2_per_s": (
                            implied_forcing_intensity(config)
                            if "forced" in method
                            else np.nan
                        ),
                        "source_amplitude_fraction_sd": (
                            source_amplitude_fraction_sd
                            if "source amplitude" in method
                            else np.nan
                        ),
                        "source_amplitude_timescale_s": (
                            source_amplitude_timescale
                            if "source amplitude" in method
                            else np.nan
                        ),
                    }
                )
                rows.append(row)
                completed.add(key)
                print(
                    f"[{trajectory_index + 1:02d}/{len(records)}] {record.name}, "
                    f"{fraction:.0%}, {method}: field={row['excess_field_rel_l2']:.3f}, "
                    f"all CRPS={row['mean_crps_K']:.3f}, top1 CRPS="
                    f"{row['fixed_top_01_crps_K']:.3f}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def aggregate(results: pd.DataFrame, *, heldout_only: bool, by_family: bool) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"] if heldout_only else results
    groups = ["fraction_saturated", "method"]
    if by_family:
        groups.insert(0, "family")
    summary = subset.groupby(groups, sort=False)[SUMMARY_METRICS].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_summary(results: pd.DataFrame, methods: list[str], baseline: str) -> pd.DataFrame:
    evaluation = results[results["role"] == "evaluation"]
    rows = []
    metrics = [
        "mean_crps_K",
        "fixed_top_01_crps_K",
        "excess_field_rel_l2",
        "peak_absolute_error_K",
        "hot_region_95_coverage",
    ]
    for fraction in sorted(evaluation["fraction_saturated"].unique()):
        subset = evaluation[np.isclose(evaluation["fraction_saturated"], fraction)]
        indexed = subset.set_index(["trajectory", "method"])
        trajectories = sorted(subset["trajectory"].unique())
        for method in methods:
            if method == baseline:
                continue
            row: dict[str, object] = {
                "fraction_saturated": fraction,
                "method": method,
                "baseline": baseline,
                "n_held_out": len(trajectories),
            }
            for metric in metrics:
                differences = np.asarray(
                    [
                        indexed.loc[(name, method), metric]
                        - indexed.loc[(name, baseline), metric]
                        for name in trajectories
                    ],
                    dtype=float,
                )
                row[f"{metric}_mean_change"] = float(np.mean(differences))
                row[f"{metric}_median_change"] = float(np.median(differences))
                if metric == "hot_region_95_coverage":
                    row[f"{metric}_win_count"] = int(np.sum(differences > 0.0))
                else:
                    row[f"{metric}_win_count"] = int(np.sum(differences < 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_stage(summary: pd.DataFrame, methods: list[str], out_path: Path, title: str) -> None:
    metrics = [
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("excess_field_rel_l2", "Relative field error"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_95_coverage", "Hot 95% coverage"),
        ("mean_95_interval_width_K", "Mean 95% interval width (K)"),
    ]
    fractions = sorted(summary["fraction_saturated"].unique())
    x = np.arange(len(fractions))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        for method_index, method in enumerate(methods):
            rows = summary[summary["method"] == method].set_index(
                "fraction_saturated"
            )
            means = np.asarray([rows.loc[value, f"{metric}_mean"] for value in fractions])
            errors = np.asarray(
                [
                    rows.loc[value, f"{metric}_std"]
                    / np.sqrt(rows.loc[value, f"{metric}_count"])
                    for value in fractions
                ]
            )
            positions = x
            if len(fractions) == 1:
                positions = x + 0.12 * (method_index - 0.5 * (len(methods) - 1))
            axis.errorbar(
                positions,
                means,
                yerr=np.nan_to_num(errors),
                marker="o",
                linestyle="none" if len(fractions) == 1 else "-",
                linewidth=1.8,
                capsize=3,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.set_xticks(x, [f"{value:.0%}" for value in fractions])
        if len(fractions) == 1:
            axis.set_xlim(-0.35, 0.35)
        axis.set_xlabel("Censored fraction")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 1].axhline(0.95, color="#555555", linestyle="--", linewidth=1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(title)
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_stage_readme(
    stage_dir: Path,
    *,
    stage: str,
    methods: list[str],
    baseline: str,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    combined_pairs: pd.DataFrame | None,
    args: argparse.Namespace,
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
    source_amplitude_fraction_sd: float,
    source_amplitude_timescale: float,
) -> None:
    fraction = args.fractions_saturated[0]
    table = overall[np.isclose(overall["fraction_saturated"], fraction)].set_index(
        "method"
    )
    lines = [
        f"# {stage.replace('_', ' ').title()}",
        "",
        "All models use the same fixed stride-only observations, noise realization, "
        "two-frame lag, residual hyperparameters, and censored posterior sampler. "
        "The table reports the 30 held-out trajectories; trajectory-level files include "
        "all 33 trajectories.",
        "",
        f"Reference censoring fraction: `{fraction:.0%}`. Regular source coupling: "
        f"`{regular_source_coupling:.6f}`; advective-mean source coupling: "
        f"`{advective_source_coupling:.6f}`.",
        "",
    ]
    if stage == "14_residual_advection":
        lines.extend(
            [
                "The advective residual kernel is",
                "",
                "```text",
                "k_adv = sigma_f^2 exp(-beta |dt|) ell^2/L^2",
                "        * exp(-||x-x'-(s(t)-s(t'))||^2/(2 L^2)),",
                "L^2 = ell^2 + 2 alpha |dt|.",
                "```",
                "",
                "The path s(t) is the HeatFluxZ-weighted source centroid. The physics "
                "mean is unchanged, so this stage isolates transport in the residual "
                "covariance.",
                "",
            ]
        )
    elif stage == "15_mean_advection":
        lines.extend(
            [
                "The advective one-step mean is",
                "",
                "```text",
                "m_n(x)-T_amb = exp(-beta dt) [G_(alpha dt) * u_(n-1)](x-d_n)",
                "                 + gamma_adv q_n(x) dt.",
                "```",
                "",
                "Both models use the regular space-time heat kernel. The advective "
                "source coupling is recalibrated on the same three development paths. "
                "This is a moving-frame correction, not literal material advection in "
                "the stationary solid.",
                "",
            ]
        )
    elif stage == "16_stochastic_forcing":
        lines.extend(
            [
                "The forced residual satisfies dr = (alpha Laplacian r - beta r)dt + dW, "
                "where W is white in time and has an RBF spatial covariance. Its "
                "stationary covariance is evaluated with positive Gauss-Laguerre "
                "quadrature and normalized to the same marginal variance sigma_f^2 as "
                "the regular space-time kernel. No extra variance parameter is tuned. "
                "Dimensionally, sigma_f^2 is a field variance in K^2; the implied RBF "
                "forcing intensity q has units K^2/s and is saved in `results.csv` as "
                "`forcing_intensity_K2_per_s`.",
                "",
                "For the two-dimensional kernel, q is chosen from",
                "",
                "```text",
                "sigma_f^2 = q * integral_0^infinity exp(-2 beta u)",
                "            * ell_W^2/(ell_W^2 + 4 alpha u) du.",
                "```",
                "",
            ]
        )
    elif stage == "17_advective_stochastic_forcing":
        lines.extend(
            [
                "The combined residual satisfies",
                "",
                "```text",
                "dr = (alpha Laplacian r - v(t) dot grad r - beta r) dt + dW,",
                "E[dW(x,t)dW(x',t)] = q exp(-||x-x'||^2/(2 ell_W^2)) dt.",
                "```",
                "",
                "Its stationary covariance is evaluated in the moving coordinates "
                "x-s(t), so each forcing contribution is both diffused and shifted by "
                "s(t)-s(t'). The regular physics mean is unchanged. This isolates the "
                "interaction between residual transport and continuous stochastic "
                "forcing.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The source-amplitude model augments the advective-forced covariance by",
                "",
                "```text",
                "k_amp(z,z') = sigma_eta^2 b(z)b(z') exp(-|t-t'|/tau_eta),",
                "b(x,t) = gamma_0 q(x,t) dt.",
                "```",
                "",
                f"The selected fractional source SD is "
                f"`{source_amplitude_fraction_sd:.3f}`. The three tested persistence "
                "values produced identical development metrics because no coarse "
                "observation landed inside the narrow source basis on the development "
                f"paths. The reported `{source_amplitude_timescale:.4f} s` is therefore "
                "a tie-breaking convention, not an identified timescale. The 30 held-out "
                "trajectories test whether the uncertainty correction transfers.",
                "",
            ]
        )
    lines.extend(
        [
            "## Held-out results",
            "",
            "| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['mean_crps_K_mean']:.3f} | {row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Paired changes use `{baseline}` as the baseline. Negative changes are "
            "better for errors and CRPS; positive changes are better for coverage.",
            "",
        ]
    )
    for _, row in paired.iterrows():
        lines.append(
            f"- `{row['method']}`: field `{row['excess_field_rel_l2_mean_change']:+.4f}`, "
            f"all CRPS `{row['mean_crps_K_mean_change']:+.4f} K`, top-1% CRPS "
            f"`{row['fixed_top_01_crps_K_mean_change']:+.4f} K`, hot coverage "
            f"`{row['hot_region_95_coverage_mean_change']:+.4f}`."
        )
    if combined_pairs is not None and not combined_pairs.empty:
        comparison_heading = "Combined model compared with each component:"
        if stage == "18_source_amplitude_correction":
            comparison_heading = "Source-amplitude model compared with each baseline:"
        lines.extend(["", comparison_heading, ""])
        for _, row in combined_pairs.iterrows():
            lines.append(
                f"- versus `{row['baseline']}`: field "
                f"`{row['excess_field_rel_l2_mean_change']:+.4f}`, all CRPS "
                f"`{row['mean_crps_K_mean_change']:+.4f} K`, top-1% CRPS "
                f"`{row['fixed_top_01_crps_K_mean_change']:+.4f} K`, hot coverage "
                f"`{row['hot_region_95_coverage_mean_change']:+.4f}`."
            )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `results.csv`: all trajectory-level results for this comparison.",
            "- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.",
            "- `paired_vs_baseline.csv`.",
            "- `combined_vs_each_baseline.csv` or "
            "`source_amplitude_vs_each_baseline.csv` for the added model.",
            "- `comparison.png`.",
            "",
            f"Diffusivity: `{parameters['diffusivity']:.6e} m^2/s`; cooling rate: "
            f"`{parameters['cooling_rate']:.6f} 1/s`; residual setting: signal "
            f"`x{args.signal_multiplier:g}`, noise `x{args.noise_multiplier:g}`, beta "
            f"`x{args.beta_multiplier:g}`, length `x{args.length_multiplier:g}`.",
            "",
        ]
    )
    stage_dir.joinpath("README.md").write_text("\n".join(lines), encoding="ascii")


def write_outputs(
    args: argparse.Namespace,
    *,
    results: pd.DataFrame,
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
    source_amplitude_fraction_sd: float,
    source_amplitude_timescale: float,
) -> None:
    for stage, methods in STAGES.items():
        stage_dir = args.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        subset = results[results["method"].isin(methods)].copy()
        subset.to_csv(stage_dir / "results.csv", index=False)
        heldout = aggregate(subset, heldout_only=True, by_family=False)
        all33 = aggregate(subset, heldout_only=False, by_family=False)
        family = aggregate(subset, heldout_only=True, by_family=True)
        paired = paired_summary(subset, methods, BASELINES[stage])
        combined_pairs = None
        pair_filename = None
        if stage == "17_advective_stochastic_forcing":
            combined_method = "physics mean + advective forced space-time"
            pair_filename = "combined_vs_each_baseline.csv"
            combined_pairs = pd.concat(
                [
                    paired_summary(
                        subset,
                        [baseline, combined_method],
                        baseline,
                    )
                    for baseline in methods
                    if baseline != combined_method
                ],
                ignore_index=True,
            )
        elif stage == "18_source_amplitude_correction":
            combined_method = "physics mean + advective forced + source amplitude"
            pair_filename = "source_amplitude_vs_each_baseline.csv"
            combined_pairs = pd.concat(
                [
                    paired_summary(
                        subset,
                        [baseline, combined_method],
                        baseline,
                    )
                    for baseline in methods
                    if baseline != combined_method
                ],
                ignore_index=True,
            )
        heldout.to_csv(stage_dir / "heldout30_overall.csv", index=False)
        all33.to_csv(stage_dir / "all33_overall.csv", index=False)
        family.to_csv(stage_dir / "family_summary.csv", index=False)
        paired.to_csv(stage_dir / "paired_vs_baseline.csv", index=False)
        if combined_pairs is not None:
            combined_pairs.to_csv(
                stage_dir / pair_filename,
                index=False,
            )
        plot_stage(
            heldout,
            methods,
            stage_dir / "comparison.png",
            stage.replace("_", " ").title(),
        )
        write_stage_readme(
            stage_dir,
            stage=stage,
            methods=methods,
            baseline=BASELINES[stage],
            overall=heldout,
            paired=paired,
            combined_pairs=combined_pairs,
            args=args,
            parameters=parameters,
            regular_source_coupling=regular_source_coupling,
            advective_source_coupling=advective_source_coupling,
            source_amplitude_fraction_sd=source_amplitude_fraction_sd,
            source_amplitude_timescale=source_amplitude_timescale,
        )


def generate_reconstruction_comparisons(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    regular_source_coupling: float,
    advective_source_coupling: float,
    source_amplitude_fraction_sd: float,
    source_amplitude_timescale: float,
    results: pd.DataFrame,
) -> None:
    stage_dir = args.output_dir / "17_advective_stochastic_forcing"
    reconstruction_dir = stage_dir / "reconstructions"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    names = ablation.DEVELOPMENT_TRAJECTORIES
    expected = [reconstruction_dir / f"{name}.png" for name in names]
    if args.resume and all(path.is_file() for path in expected):
        return

    catalog = trajectory_catalog(args.dataset_dir)
    trajectory_indices = {record.name: index for index, record in enumerate(catalog)}
    methods = [
        "physics mean + RBF",
        "physics mean + space-time",
        "physics mean + advective space-time",
        "physics mean + forced space-time",
        "physics mean + advective forced space-time",
    ]
    short_labels = {
        "physics mean + RBF": "Physics mean + RBF",
        "physics mean + space-time": "Physics mean + space-time",
        "physics mean + advective space-time": "Physics mean + advective space-time",
        "physics mean + forced space-time": "Physics mean + forced space-time",
        "physics mean + advective forced space-time": "Physics mean + advective-forced",
    }
    for name in names:
        trajectory_index = trajectory_indices[name]
        prepared, _ = thermal.prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        threshold = float(np.quantile(prepared.truth, 1.0 - REFERENCE_FRACTION))
        history_indices = study.select_history_indices(prepared.times, args.main_lags)
        multitime = corrected.fixed_mask_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current = study.current_frame_observations(multitime)
        models, _ = build_model_context(
            args,
            prepared,
            threshold=threshold,
            multitime=multitime,
            current=current,
            parameters=parameters,
            regular_source_coupling=regular_source_coupling,
            advective_source_coupling=advective_source_coupling,
            source_amplitude_fraction_sd=source_amplitude_fraction_sd,
            source_amplitude_timescale=source_amplitude_timescale,
        )
        means = {}
        prediction_seed = args.seed + 100_000 * trajectory_index
        for method in methods:
            observations, config, source_coupling = models[method]
            _, prediction = ablation.run_method(
                prepared,
                method=method,
                observations=observations,
                config=config,
                diffusivity=parameters["diffusivity"],
                cooling_rate=config.cooling_rate,
                source_coupling=source_coupling,
                n_frames=1 if method == "physics mean + RBF" else len(history_indices),
                chains=args.chains,
                samples=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=prediction_seed,
                return_prediction=True,
            )
            means[method] = np.asarray(prediction[0]).reshape(prepared.truth.shape)

        clipped = np.asarray(multitime["frames"][-1]["clipped_full"])
        panels = [("True field", prepared.truth), ("Clipped current", clipped)]
        panels.extend((short_labels[method], means[method]) for method in methods)
        figure, axes_grid = plt.subplots(
            2,
            4,
            figsize=(15.5, 7.4),
            constrained_layout=True,
        )
        axes = axes_grid.ravel()
        vmin = float(np.min(prepared.truth))
        vmax = float(np.max(prepared.truth))
        image = None
        result_rows = results[
            (results["trajectory"] == name)
            & np.isclose(results["fraction_saturated"], REFERENCE_FRACTION)
        ].set_index("method")
        for axis, (label, field) in zip(axes, panels):
            image = axis.imshow(
                field,
                origin="lower",
                extent=[prepared.xs[0], prepared.xs[-1], prepared.ys[0], prepared.ys[-1]],
                vmin=vmin,
                vmax=vmax,
                cmap="inferno",
                aspect="auto",
            )
            title = label
            matching = [method for method, short in short_labels.items() if short == label]
            if matching:
                row = result_rows.loc[matching[0]]
                title += (
                    f"\nfield {row['excess_field_rel_l2']:.3f}, "
                    f"peak {row['peak_absolute_error_K']:.2f} K"
                )
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes[len(panels) :]:
            axis.set_visible(False)
        if image is not None:
            figure.colorbar(
                image,
                ax=[axis for axis in axes[: len(panels)]],
                shrink=0.88,
                label="Temperature (K)",
            )
        figure.suptitle(f"{name}: matched 3% censored reconstructions")
        figure.savefig(reconstruction_dir / f"{name}.png", dpi=220)
        plt.close(figure)
    results[
        results["trajectory"].isin(names)
        & results["method"].isin(methods)
        & np.isclose(results["fraction_saturated"], REFERENCE_FRACTION)
    ].to_csv(reconstruction_dir / "reconstruction_metrics.csv", index=False)


def write_parent_readme(
    args: argparse.Namespace,
    *,
    results: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    evaluation = results[results["role"] == "evaluation"]
    means = evaluation.groupby("method", sort=False)[SUMMARY_METRICS].mean()
    combined_method = "physics mean + advective forced space-time"
    forced_method = "physics mean + forced space-time"
    combined_all_crps_change = (
        means.loc[combined_method, "mean_crps_K"]
        - means.loc[forced_method, "mean_crps_K"]
    )
    combined_field_change = (
        means.loc[combined_method, "excess_field_rel_l2"]
        - means.loc[forced_method, "excess_field_rel_l2"]
    )
    paired = evaluation.pivot(
        index="trajectory",
        columns="method",
        values="mean_crps_K",
    )
    combined_all_crps_wins = int(
        np.sum(paired[combined_method] < paired[forced_method])
    )
    source_method = "physics mean + advective forced + source amplitude"
    source_all_crps_change = (
        means.loc[source_method, "mean_crps_K"]
        - means.loc[combined_method, "mean_crps_K"]
    )
    source_field_change = (
        means.loc[source_method, "excess_field_rel_l2"]
        - means.loc[combined_method, "excess_field_rel_l2"]
    )
    source_all_crps_wins = int(
        np.sum(paired[source_method] < paired[combined_method])
    )
    forced_checks = validation[
        validation["method"].str.contains("forced", regex=False)
    ]
    lines = [
        "# Controlled advection and forcing restart",
        "",
        "This restart separates Adrienne's three suggestions into three comparisons. "
        "All seven unique models use the same 3% synthetic ceiling, fixed stride-only "
        "observation mask, noise realization, two-frame lag, residual hyperparameters, "
        "and censored posterior sampler. Results contain all 33 trajectories; the main "
        "table below uses the 30 trajectories not used for physical-parameter and source-"
        "coupling calibration.",
        "",
        "No source-local covariance is included. The final stage adds only the "
        "source-amplitude correction described below.",
        "",
        "## Overall held-out results",
        "",
        "| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = means.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2']:.3f} | "
            f"{row['mean_crps_K']:.3f} | {row['fixed_top_01_crps_K']:.3f} | "
            f"{row['peak_absolute_error_K']:.3f} | "
            f"{row['hot_region_95_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. `physics mean + RBF` retains the smallest mean field error. The regular "
            "space-time kernel improves all-domain CRPS but not mean field error.",
            "2. Adding the trajectory displacement to the space-time kernel improves "
            "all-domain CRPS on all 30 held-out trajectories. Its mean field and hottest-"
            "region changes are small, so this supports a covariance improvement rather "
            "than a uniformly better reconstruction mean.",
            "3. Adding the same moving-frame correction to the one-step mean is neutral "
            "to slightly worse. Transporting the whole previously deposited field is "
            "therefore not supported by this experiment.",
            "4. The exact white-in-time, spatial-RBF forcing covariance substantially "
            "improves CRPS and slightly improves peak error, but worsens field error and "
            "reduces hot-region coverage. It is promising as a probabilistic prior, not "
            "yet as the best posterior-mean reconstruction.",
            f"5. The advective and forced covariance changes are complementary: relative "
            f"to forcing alone, the combined model changes all-domain CRPS by "
            f"`{combined_all_crps_change:+.4f} K` and wins on "
            f"`{combined_all_crps_wins}/30` held-out trajectories, while changing field "
            f"error by `{combined_field_change:+.4f}`. It is the strongest probabilistic "
            "model in this focused comparison, although RBF still has the lowest mean "
            "field error and the combined model has lower hot-region coverage.",
            f"6. Adding source-amplitude uncertainty to the combined model changes "
            f"all-domain CRPS by `{source_all_crps_change:+.4f} K` and wins on "
            f"`{source_all_crps_wins}/30` held-out trajectories, while changing field "
            f"error by `{source_field_change:+.4f}`. It primarily widens uncertainty at "
            "the active source: the sparse mask did not observe the narrow source basis, "
            "so its temporal persistence was not identifiable and its posterior mean was "
            "essentially unchanged.",
            "",
            "## Mathematical checks",
            "",
            f"All `{len(validation)}` kernel checks are symmetric and positive "
            f"semidefinite. The minimum tested eigenvalue is "
            f"`{validation['minimum_eigenvalue'].min():.3e}`, and the maximum marginal-"
            f"variance error is `{validation['maximum_diagonal_error'].max():.3e}`. "
            f"The 24-node forced-kernel quadrature differs from a 48-node reference by "
            f"at most `{forced_checks['quadrature_relative_error'].max():.3e}` of the "
            "marginal variance.",
            "",
            "Folders:",
            "",
            "- `13_thermal_advection_setup`: shared configuration, calibration, "
            "checkpoints, complete results, and kernel validation.",
            "- `14_residual_advection`: RBF, regular space-time, and advective space-time.",
            "- `15_mean_advection`: regular and advective one-step means with the same "
            "regular space-time kernel.",
            "- `16_stochastic_forcing`: regular space-time and the stationary forced-SPDE "
            "covariance.",
            "- `17_advective_stochastic_forcing`: RBF, regular space-time, advective, "
            "forced, and combined advective-forced residual covariances.",
            "- `18_source_amplitude_correction`: the selected source-amplitude term added "
            "to the combined advective-forced covariance.",
            "",
            "Shared files include `all_models_results.csv`, `fixed_configuration.csv`, "
            "source-coupling calibrations, and `kernel_validation.csv`.",
            "",
        ]
    )
    setup_output_dir(args).joinpath("README.md").write_text(
        "\n".join(lines), encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild controlled advection and stochastic-forcing comparisons."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--main-lags", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument(
        "--fractions-saturated",
        nargs="+",
        type=float,
        default=[REFERENCE_FRACTION],
    )
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--signal-multiplier", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--beta-multiplier", type=float, default=1.0)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--forcing-length-multiplier", type=float, default=1.0)
    parser.add_argument("--forcing-quadrature-order", type=int, default=24)
    parser.add_argument(
        "--source-amplitude-fraction-sds",
        nargs="+",
        type=float,
        default=[0.0, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument(
        "--source-amplitude-timescale-multipliers",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 10.0],
    )
    parser.add_argument("--source-amplitude-tuning-samples", type=int, default=120)
    parser.add_argument("--source-amplitude-tuning-burn-in", type=int, default=80)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--burn-in", type=int, default=120)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_dir = setup_output_dir(args)
    setup_dir.mkdir(parents=True, exist_ok=True)
    prepared_items, estimate_rows, parameters = ablation.prepare_development_set(args)
    regular_couplings, regular_rows = ablation.calibrate_source_couplings(
        prepared_items,
        fractions=[REFERENCE_FRACTION],
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_flux_threshold=args.source_flux_threshold,
    )
    regular_source_coupling = regular_couplings[REFERENCE_FRACTION]
    advective_source_coupling, advective_rows = calibrate_advective_source_coupling(
        prepared_items,
        fraction=REFERENCE_FRACTION,
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_flux_threshold=args.source_flux_threshold,
    )
    (
        source_amplitude_fraction_sd,
        source_amplitude_timescale,
        _,
        _,
    ) = select_source_amplitude_setting(
        args,
        prepared_items=prepared_items,
        parameters=parameters,
        regular_source_coupling=regular_source_coupling,
        advective_source_coupling=advective_source_coupling,
    )
    pd.DataFrame(estimate_rows.values()).to_csv(
        setup_dir / "development_physics_parameters.csv", index=False
    )
    regular_rows.assign(mean_type="regular").to_csv(
        setup_dir / "regular_source_coupling.csv", index=False
    )
    advective_rows.assign(mean_type="advective").to_csv(
        setup_dir / "advective_source_coupling.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **parameters,
                **residual_multipliers(args),
                "regular_source_coupling": regular_source_coupling,
                "advective_source_coupling": advective_source_coupling,
                "forcing_length_multiplier": args.forcing_length_multiplier,
                "forcing_quadrature_order": args.forcing_quadrature_order,
                "source_amplitude_fraction_sd": source_amplitude_fraction_sd,
                "source_amplitude_timescale_s": source_amplitude_timescale,
                "observation_stride": args.observation_stride,
                "seed": args.seed,
            }
        ]
    ).to_csv(setup_dir / "fixed_configuration.csv", index=False)

    results, validation = run_all_models(
        args,
        parameters=parameters,
        regular_source_coupling=regular_source_coupling,
        advective_source_coupling=advective_source_coupling,
        source_amplitude_fraction_sd=source_amplitude_fraction_sd,
        source_amplitude_timescale=source_amplitude_timescale,
    )
    results.to_csv(setup_dir / "all_models_results.csv", index=False)
    validation.to_csv(setup_dir / "kernel_validation.csv", index=False)
    write_outputs(
        args,
        results=results,
        parameters=parameters,
        regular_source_coupling=regular_source_coupling,
        advective_source_coupling=advective_source_coupling,
        source_amplitude_fraction_sd=source_amplitude_fraction_sd,
        source_amplitude_timescale=source_amplitude_timescale,
    )
    generate_reconstruction_comparisons(
        args,
        parameters=parameters,
        regular_source_coupling=regular_source_coupling,
        advective_source_coupling=advective_source_coupling,
        source_amplitude_fraction_sd=source_amplitude_fraction_sd,
        source_amplitude_timescale=source_amplitude_timescale,
        results=results,
    )
    write_parent_readme(args, results=results, validation=validation)
    print(f"Saved rebuilt comparisons to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
