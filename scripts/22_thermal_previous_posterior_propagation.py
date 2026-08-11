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
from scipy.ndimage import gaussian_filter

from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_spde_module():
    module_path = ROOT / "scripts" / "21_thermal_stochastic_spde_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "thermal_stochastic_spde_for_propagation", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_stochastic_spde_for_propagation"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


spde = load_spde_module()
corrected = spde.corrected
ablation = spde.ablation
study = spde.study
thermal = spde.thermal
gp2d = spde.gp2d

DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "20_physics_mean_propagation"
)
REFERENCE_FRACTION = 0.03
METHODS = [
    "clipped propagation",
    "posterior-mean propagation",
    "full-posterior propagation",
]
METHOD_COLORS = {
    "clipped propagation": "#D55E00",
    "posterior-mean propagation": "#0072B2",
    "full-posterior propagation": "#009E73",
}
SUMMARY_METRICS = [
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "hot_region_crps_K",
    "excess_field_rel_l2",
    "peak_absolute_error_K",
    "hot_region_95_coverage",
    "pointwise_95_coverage",
    "mean_95_interval_width_K",
    "hot_region_95_interval_width_K",
    "prior_mean_field_rel_l2",
    "prior_mean_peak_absolute_error_K",
    "propagated_prior_sd_at_true_peak_K",
]
PAIRINGS = [
    ("posterior-mean propagation", "clipped propagation"),
    ("full-posterior propagation", "posterior-mean propagation"),
    ("full-posterior propagation", "clipped propagation"),
]


def grid_indices(prepared, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    dx = float(np.mean(np.diff(prepared.xs)))
    dy = float(np.mean(np.diff(prepared.ys)))
    ix = np.rint((values[:, 0] - prepared.xs[0]) / dx).astype(int)
    iy = np.rint((values[:, 1] - prepared.ys[0]) / dy).astype(int)
    if np.any(ix < 0) or np.any(ix >= len(prepared.xs)):
        raise ValueError("Points fall outside the thermal grid in x")
    if np.any(iy < 0) or np.any(iy >= len(prepared.ys)):
        raise ValueError("Points fall outside the thermal grid in y")
    if not np.allclose(values[:, 0], prepared.xs[ix], atol=1e-10, rtol=0.0):
        raise ValueError("Points do not align with the thermal x grid")
    if not np.allclose(values[:, 1], prepared.ys[iy], atol=1e-10, rtol=0.0):
        raise ValueError("Points do not align with the thermal y grid")
    return iy * len(prepared.xs) + ix


def make_grid_mean_function(prepared, field: np.ndarray):
    flat = np.asarray(field, dtype=float).ravel()

    def mean_function(points: np.ndarray) -> np.ndarray:
        return flat[grid_indices(prepared, points)]

    return mean_function


def make_low_rank_adjustment(prepared, propagated_draws: np.ndarray):
    draws = np.asarray(propagated_draws, dtype=float)
    if draws.ndim != 3 or draws.shape[1:] != prepared.truth.shape:
        raise ValueError("Propagated draws must have shape (samples, ny, nx)")
    if len(draws) < 2:
        raise ValueError("At least two propagated draws are required")
    flat = draws.reshape(len(draws), -1)
    mean = np.mean(flat, axis=0)
    features = (flat - mean[None, :]).T / np.sqrt(len(draws) - 1)

    def covariance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        feature1 = features[grid_indices(prepared, points1)]
        feature2 = features[grid_indices(prepared, points2)]
        return feature1.dot(feature2.T)

    def variance(points: np.ndarray) -> np.ndarray:
        selected = features[grid_indices(prepared, points)]
        return np.sum(selected**2, axis=1)

    return mean.reshape(prepared.truth.shape), covariance, variance, features


def propagate_previous_fields(
    prepared,
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
    excess = np.maximum(values - prepared.ambient, 0.0)
    sigma = (sigma_y, sigma_x) if values.ndim == 2 else (0.0, sigma_y, sigma_x)
    propagated = gaussian_filter(
        excess,
        sigma=sigma,
        mode="constant",
        cval=0.0,
    )
    return propagated * np.exp(-cooling_rate * dt)


def current_source_field(
    prepared,
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


def make_previous_posterior_observations(
    prepared,
    frame: dict[str, object],
    *,
    threshold: float,
    fixed_mask: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    clipped = np.asarray(frame["clipped_full"], dtype=float)
    saturated = clipped >= threshold - 1e-12
    included = np.asarray(fixed_mask, dtype=bool) | saturated
    time = float(frame["time"])
    points = np.column_stack(
        [prepared.points, np.full(len(prepared.points), time)]
    )
    observations = {
        "x_obs": points[included.ravel()],
        "y_obs": clipped[included],
        "sat_mask": saturated[included],
        "x_pred": points[saturated.ravel()],
        "threshold": threshold,
    }
    return observations, clipped, saturated


def infer_previous_posterior(
    args: argparse.Namespace,
    prepared,
    *,
    frame: dict[str, object],
    fixed_mask: np.ndarray,
    threshold: float,
    parameters: dict[str, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    observations, clipped, saturated = make_previous_posterior_observations(
        prepared,
        frame,
        threshold=threshold,
        fixed_mask=fixed_mask,
    )
    n_draws = args.previous_samples * args.previous_chains
    previous_draws = np.repeat(clipped[None, :, :], n_draws, axis=0)
    if np.any(saturated):
        config = study.make_config(
            prepared,
            kernel="rbf",
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            signal_sd=parameters["signal_sd"] * args.signal_multiplier,
            noise_sd=args.noise_sd * args.previous_noise_multiplier,
        )
        config = replace(
            config,
            lengthscale=prepared.source_lengthscale * args.length_multiplier,
        )
        prediction = thermal.sample_censored_multiple_chains(
            observations,
            config,
            n_chains=args.previous_chains,
            samples_per_chain=args.previous_samples,
            burn_in=args.previous_burn_in,
            thin=args.thin,
            seed=seed,
        )
        previous_draws[:, saturated] = prediction[4]

    posterior_mean = np.mean(previous_draws, axis=0)
    previous_truth = prepared.history[int(frame["time_index"])]
    if np.any(saturated):
        clipped_hot_mae = float(
            np.mean(np.abs(clipped[saturated] - previous_truth[saturated]))
        )
        posterior_hot_mae = float(
            np.mean(np.abs(posterior_mean[saturated] - previous_truth[saturated]))
        )
        posterior_hot_coverage = float(
            np.mean(
                (np.quantile(previous_draws[:, saturated], 0.025, axis=0)
                 <= previous_truth[saturated])
                & (previous_truth[saturated]
                   <= np.quantile(previous_draws[:, saturated], 0.975, axis=0))
            )
        )
    else:
        clipped_hot_mae = 0.0
        posterior_hot_mae = 0.0
        posterior_hot_coverage = 1.0
    diagnostics = {
        "previous_n_saturated": int(np.sum(saturated)),
        "previous_saturated_fraction": float(np.mean(saturated)),
        "previous_clipped_hot_mae_K": clipped_hot_mae,
        "previous_posterior_hot_mae_K": posterior_hot_mae,
        "previous_posterior_hot_95_coverage": posterior_hot_coverage,
    }
    return posterior_mean, previous_draws, saturated, diagnostics


def make_current_config(
    args: argparse.Namespace,
    prepared,
    *,
    parameters: dict[str, float],
    mean_field: np.ndarray,
    covariance_adjustment=None,
    variance_adjustment=None,
):
    config = study.make_config(
        prepared,
        kernel="rbf",
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        signal_sd=parameters["signal_sd"] * args.signal_multiplier,
        noise_sd=args.noise_sd * args.noise_multiplier,
        mean_function=make_grid_mean_function(prepared, mean_field),
    )
    return replace(
        config,
        lengthscale=prepared.source_lengthscale * args.length_multiplier,
        covariance_adjustment=covariance_adjustment,
        variance_adjustment=variance_adjustment,
    )


def validate_full_prior(config, points: np.ndarray) -> dict[str, float]:
    probe_indices = np.linspace(0, len(points) - 1, min(70, len(points))).astype(int)
    probe = np.asarray(points)[probe_indices]
    matrix = gp2d.rbf_kernel(probe, probe, config)
    diagonal = gp2d.kernel_diagonal(probe, config)
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - diagonal)))
    tolerance = 1e-8 * max(float(np.max(diagonal)), 1.0)
    if symmetry_error > tolerance or minimum_eigenvalue < -tolerance:
        raise AssertionError(
            "Invalid full-posterior covariance: "
            f"symmetry={symmetry_error}, minimum eigenvalue={minimum_eigenvalue}"
        )
    return {
        "symmetry_error": symmetry_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_diagonal_error": diagonal_error,
    }


def run_experiment(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = trajectory_catalog(args.dataset_dir)
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        records = [record for record in records if record.name in requested]
        missing = requested - {record.name for record in records}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    checkpoint_path = args.output_dir / "checkpoint.csv"
    if args.resume and checkpoint_path.is_file():
        checkpoint = pd.read_csv(checkpoint_path)
        rows = checkpoint.to_dict("records")
        completed = {
            (str(row["trajectory"]), str(row["method"])) for row in rows
        }
        print(f"Resuming after {len(completed)} completed fits", flush=True)
    else:
        rows = []
        completed = set()
    validation_rows: list[dict[str, object]] = []

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
        threshold = float(
            np.quantile(prepared.truth, 1.0 - args.fraction_saturated)
        )
        current_index = len(prepared.times) - 1
        previous_index = current_index - args.previous_frame_offset
        if previous_index < 0:
            raise ValueError("Previous-frame offset exceeds the available trajectory")
        history_indices = np.asarray([previous_index, current_index], dtype=int)
        multitime = corrected.fixed_mask_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current = study.current_frame_observations(multitime)
        previous_frame = multitime["frames"][0]
        fixed_mask = np.asarray(multitime["fixed_observation_mask"], dtype=bool)
        posterior_previous_mean, previous_draws, previous_saturated, diagnostics = (
            infer_previous_posterior(
                args,
                prepared,
                frame=previous_frame,
                fixed_mask=fixed_mask,
                threshold=threshold,
                parameters=parameters,
                seed=args.seed + 50_000 * trajectory_index,
            )
        )
        clipped_previous = np.asarray(previous_frame["clipped_full"], dtype=float)
        clipped_background = propagate_previous_fields(
            prepared,
            clipped_previous,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
        )
        posterior_background_draws = propagate_previous_fields(
            prepared,
            previous_draws,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
        )
        source = current_source_field(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            source_coupling=args.source_coupling,
            source_flux_threshold=args.source_flux_threshold,
        )
        clipped_mean = prepared.ambient + clipped_background + source
        propagated_draws = prepared.ambient + posterior_background_draws + source
        posterior_mean, covariance, variance, features = make_low_rank_adjustment(
            prepared, propagated_draws
        )
        posterior_mean_direct = (
            prepared.ambient
            + propagate_previous_fields(
                prepared,
                posterior_previous_mean,
                previous_index=previous_index,
                current_index=current_index,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
            )
            + source
        )
        if not np.allclose(posterior_mean, posterior_mean_direct, atol=1e-10):
            raise AssertionError("Linear heat propagation did not preserve sample means")

        configs = {
            "clipped propagation": make_current_config(
                args,
                prepared,
                parameters=parameters,
                mean_field=clipped_mean,
            ),
            "posterior-mean propagation": make_current_config(
                args,
                prepared,
                parameters=parameters,
                mean_field=posterior_mean,
            ),
            "full-posterior propagation": make_current_config(
                args,
                prepared,
                parameters=parameters,
                mean_field=posterior_mean,
                covariance_adjustment=covariance,
                variance_adjustment=variance,
            ),
        }
        validation = validate_full_prior(
            configs["full-posterior propagation"], np.asarray(current["x_pred"])
        )
        validation.update(
            {
                "trajectory": record.name,
                "family": record.family,
                "role": role,
                "propagated_covariance_rank": int(np.linalg.matrix_rank(features)),
                "propagated_variance_mean_K2": float(np.mean(np.sum(features**2, axis=1))),
                "propagated_variance_max_K2": float(np.max(np.sum(features**2, axis=1))),
                **diagnostics,
            }
        )
        validation_rows.append(validation)

        prediction_seed = args.seed + 100_000 * trajectory_index
        predictions = {}
        mean_fields = {
            "clipped propagation": clipped_mean,
            "posterior-mean propagation": posterior_mean,
            "full-posterior propagation": posterior_mean,
        }
        propagated_variance = np.sum(features**2, axis=1)
        truth = prepared.truth.ravel()
        excess = truth - prepared.ambient
        peak_index = int(np.argmax(truth))
        hot = truth >= threshold
        for method in METHODS:
            key = (record.name, method)
            if key in completed:
                continue
            config = configs[method]
            row, prediction = ablation.run_method(
                prepared,
                method=method,
                observations=current,
                config=config,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                source_coupling=args.source_coupling,
                n_frames=2,
                chains=args.chains,
                samples=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=prediction_seed,
                return_prediction=True,
            )
            predictions[method] = prediction
            row.update(corrected.fixed_region_metrics(prepared, prediction))
            prior_mean = mean_fields[method].ravel()
            has_propagated_covariance = method == "full-posterior propagation"
            row.update(
                {
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "fraction_saturated": args.fraction_saturated,
                    "observation_design": f"fixed_stride_{args.observation_stride}",
                    "previous_posterior_design": "saturation_only_correction",
                    "prior_mean_field_rel_l2": float(
                        np.linalg.norm(prior_mean - truth) / np.linalg.norm(excess)
                    ),
                    "prior_mean_peak_absolute_error_K": abs(
                        float(np.max(prior_mean) - np.max(truth))
                    ),
                    "propagated_prior_sd_at_true_peak_K": (
                        float(np.sqrt(propagated_variance[peak_index]))
                        if has_propagated_covariance
                        else 0.0
                    ),
                    "propagated_prior_sd_hot_mean_K": (
                        float(np.mean(np.sqrt(propagated_variance[hot])))
                        if has_propagated_covariance
                        else 0.0
                    ),
                    "uses_propagated_covariance": has_propagated_covariance,
                    "crps_estimator": "unbiased_M_times_M_minus_1",
                    **diagnostics,
                }
            )
            rows.append(row)
            completed.add(key)
            print(
                f"[{trajectory_index + 1:02d}/{len(records)}] {record.name}, {method}: "
                f"field={row['excess_field_rel_l2']:.3f}, "
                f"all CRPS={row['mean_crps_K']:.3f}, "
                f"top1 CRPS={row['fixed_top_01_crps_K']:.3f}, "
                f"coverage={row['hot_region_95_coverage']:.3f}",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        if record.name in args.reconstruction_trajectories and len(predictions) == len(METHODS):
            plot_reconstruction(
                prepared,
                previous_index=previous_index,
                clipped_previous=clipped_previous,
                posterior_previous_mean=posterior_previous_mean,
                previous_draws=previous_draws,
                predictions=predictions,
                out_path=args.output_dir / f"reconstruction_{record.family.lower()}.png",
            )

    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def aggregate(results: pd.DataFrame, *, heldout_only: bool, by_family: bool) -> pd.DataFrame:
    subset = results[results["role"] == "evaluation"] if heldout_only else results
    groups = ["method"]
    if by_family:
        groups.insert(0, "family")
    summary = subset.groupby(groups, sort=False)[SUMMARY_METRICS].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    evaluation = results[results["role"] == "evaluation"]
    indexed = evaluation.set_index(["trajectory", "method"])
    trajectories = sorted(evaluation["trajectory"].unique())
    metrics = [
        "mean_crps_K",
        "fixed_top_01_crps_K",
        "excess_field_rel_l2",
        "peak_absolute_error_K",
        "hot_region_95_coverage",
        "hot_region_95_interval_width_K",
    ]
    rows = []
    for method, baseline in PAIRINGS:
        row: dict[str, object] = {
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
                ]
            )
            row[f"{metric}_mean_change"] = float(np.mean(differences))
            row[f"{metric}_median_change"] = float(np.median(differences))
            higher_is_better = metric == "hot_region_95_coverage"
            row[f"{metric}_win_count"] = int(
                np.sum(differences > 0.0) if higher_is_better else np.sum(differences < 0.0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("excess_field_rel_l2", "Relative field error"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_95_coverage", "Hot 95% coverage"),
        ("hot_region_95_interval_width_K", "Hot 95% interval width (K)"),
    ]
    indexed = summary.set_index("method")
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        means = np.asarray([indexed.loc[method, f"{metric}_mean"] for method in METHODS])
        errors = np.asarray(
            [
                indexed.loc[method, f"{metric}_std"]
                / np.sqrt(indexed.loc[method, f"{metric}_count"])
                for method in METHODS
            ]
        )
        axis.errorbar(
            x,
            means,
            yerr=np.nan_to_num(errors),
            marker="o",
            linestyle="none",
            capsize=4,
            color="#333333",
        )
        for index, method in enumerate(METHODS):
            axis.scatter(index, means[index], s=60, color=METHOD_COLORS[method], zorder=3)
        axis.set_xticks(x, ["clipped", "posterior\nmean", "full\nposterior"])
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 1].axhline(0.95, color="#666666", linestyle="--", linewidth=1.0)
    figure.suptitle(
        "Previous-frame censored-posterior propagation: 30 held-out trajectories"
    )
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_by_family(summary: pd.DataFrame, out_path: Path) -> None:
    families = ["Diagonal", "Horizontal", "Spiral"]
    metrics = [
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("excess_field_rel_l2", "Relative field error"),
        ("hot_region_95_coverage", "Hot 95% coverage"),
    ]
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(3, 3, figsize=(14.5, 10.8), constrained_layout=True)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, family in enumerate(families):
            axis = axes[row_index, column_index]
            indexed = summary[summary["family"] == family].set_index("method")
            means = np.asarray(
                [indexed.loc[method, f"{metric}_mean"] for method in METHODS]
            )
            errors = np.asarray(
                [
                    indexed.loc[method, f"{metric}_std"]
                    / np.sqrt(indexed.loc[method, f"{metric}_count"])
                    for method in METHODS
                ]
            )
            axis.errorbar(
                x,
                means,
                yerr=np.nan_to_num(errors),
                marker="o",
                linestyle="none",
                capsize=3,
                color="#333333",
            )
            for index, method in enumerate(METHODS):
                axis.scatter(index, means[index], s=50, color=METHOD_COLORS[method], zorder=3)
            axis.set_xticks(x, ["clip", "mean", "full"])
            if row_index == 0:
                axis.set_title(family)
            if column_index == 0:
                axis.set_ylabel(ylabel)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Propagation ablation by trajectory family")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_reconstruction(
    prepared,
    *,
    previous_index: int,
    clipped_previous: np.ndarray,
    posterior_previous_mean: np.ndarray,
    previous_draws: np.ndarray,
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
    vmax = float(max(np.max(prepared.history[previous_index]), np.max(prepared.truth)))
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.3))
    figure.subplots_adjust(
        left=0.055,
        right=0.91,
        bottom=0.09,
        top=0.89,
        wspace=0.30,
        hspace=0.34,
    )
    temperature_panels = [
        (prepared.history[previous_index], "True previous frame"),
        (clipped_previous, "Clipped previous frame"),
        (posterior_previous_mean, "Previous posterior mean"),
    ]
    image = None
    for axis, (field, title) in zip(axes[0, :3], temperature_panels):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_title(title)
    previous_sd = np.std(previous_draws, axis=0, ddof=1)
    sd_image = axes[0, 3].imshow(
        previous_sd,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
    )
    axes[0, 3].set_title(f"Previous posterior SD (max {np.max(previous_sd):.2f} K)")
    axes[1, 0].imshow(
        prepared.truth,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[1, 0].set_title("True current field")
    for axis, method in zip(axes[1, 1:], METHODS):
        axis.imshow(
            predictions[method][0].reshape(prepared.truth.shape),
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_title(method.replace(" propagation", ""))
    for axis in axes.ravel():
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    assert image is not None
    color_axis = figure.add_axes([0.925, 0.16, 0.012, 0.66])
    figure.colorbar(image, cax=color_axis, label="Temperature (K)")
    figure.suptitle(f"{prepared.name}: previous-frame posterior propagation")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_readme(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    table = overall.set_index("method")
    pair_table = paired.set_index(["method", "baseline"])
    mean_effect = pair_table.loc[
        ("posterior-mean propagation", "clipped propagation")
    ]
    covariance_effect = pair_table.loc[
        ("full-posterior propagation", "posterior-mean propagation")
    ]
    heldout_validation = validation[validation["role"] == "evaluation"]
    previous_clipped_mae = float(
        heldout_validation["previous_clipped_hot_mae_K"].mean()
    )
    previous_posterior_mae = float(
        heldout_validation["previous_posterior_hot_mae_K"].mean()
    )
    previous_posterior_coverage = float(
        heldout_validation["previous_posterior_hot_95_coverage"].mean()
    )
    lines = [
        "# Previous-frame censored-posterior propagation",
        "",
        "This experiment tests Adrienne's suggestion that a saturated previous camera "
        "frame should not be propagated as if every clipped value were exactly equal to "
        "the ceiling.",
        "",
        "The three methods use the same current-frame RBF residual and censored "
        "likelihood:",
        "",
        "```text",
        "A. clipped:        m_n = A y_(n-1)^clip + source_n",
        "B. posterior mean: m_n = A E[T_(n-1) | y_(n-1)] + source_n",
        "C. full posterior: same mean, with K_n = K_RBF + A Sigma_(n-1) A^T.",
        "```",
        "",
        "Here `A` is the linear diffusion-and-cooling operator. The full covariance is "
        "represented by propagated posterior draws as a positive low-rank kernel and is "
        "included before conditioning on the current censored frame.",
        "",
        "All unsaturated previous-frame pixels retain their noisy measured values. The "
        "censored GP only corrects pixels identified as saturated, conditioned on all "
        "saturated inequalities and a stride-based unsaturated support set. This avoids "
        "discarding available camera pixels while keeping the censored inference "
        "tractable.",
        "",
        "A current-frame RBF residual is used for all three methods. The previous frame "
        "is not also supplied to a space-time residual kernel, which would count the "
        "same observation twice.",
        "The propagated frame is the immediately adjacent simulation frame by default, "
        "rather than the 0.01 s residual-covariance lag used in earlier space-time "
        "experiments.",
        "",
        "## Held-out results",
        "",
        "| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage | Hot width (K) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['mean_crps_K_mean']:.3f} | "
            f"{row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} | "
            f"{row['hot_region_95_interval_width_K_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "Replacing clipped propagation by the previous posterior mean changes "
            f"field error by `{mean_effect['excess_field_rel_l2_mean_change']:+.4f}`, "
            f"all-domain CRPS by `{mean_effect['mean_crps_K_mean_change']:+.4f} K`, "
            f"and top-1% CRPS by "
            f"`{mean_effect['fixed_top_01_crps_K_mean_change']:+.4f} K`. It lowers "
            f"field error on `{int(mean_effect['excess_field_rel_l2_win_count'])}/30` "
            f"and top-1% CRPS on "
            f"`{int(mean_effect['fixed_top_01_crps_K_win_count'])}/30` held-out paths.",
            "",
            "Adding propagated covariance on top of the posterior mean changes field "
            f"error by `{covariance_effect['excess_field_rel_l2_mean_change']:+.4f}`, "
            f"all-domain CRPS by "
            f"`{covariance_effect['mean_crps_K_mean_change']:+.4f} K`, hot coverage by "
            f"`{covariance_effect['hot_region_95_coverage_mean_change']:+.4f}`, and hot "
            f"interval width by "
            f"`{covariance_effect['hot_region_95_interval_width_K_mean_change']:+.4f} K`.",
            "",
            "The previous-frame censored posterior itself reduces saturated-region MAE "
            f"from `{previous_clipped_mae:.3f} K` for clipped values to "
            f"`{previous_posterior_mae:.3f} K`, but its raw 95% coverage is only "
            f"`{previous_posterior_coverage:.3f}`. Its propagated covariance is therefore "
            "too concentrated to produce a substantial full-posterior gain. The next "
            "uncertainty experiment should calibrate this previous-frame posterior, not "
            "merely inflate the current intervals after propagation.",
            "",
            "## Numerical checks",
            "",
            f"All {len(validation)} full-prior checks are symmetric and positive "
            f"semidefinite. The minimum tested eigenvalue is "
            f"`{validation['minimum_eigenvalue'].min():.3e}` and the largest diagonal "
            f"consistency error is `{validation['maximum_diagonal_error'].max():.3e}`.",
            "",
            "Files:",
            "",
            "- `results.csv`: all 99 trajectory-model fits.",
            "- `heldout30_overall.csv`: main aggregate results.",
            "- `family_summary.csv`: family-specific results.",
            "- `paired_comparisons.csv`: within-trajectory changes and win counts.",
            "- `prior_validation.csv`: low-rank covariance checks and previous-frame diagnostics.",
            "- `comparison.png` and `comparison_by_family.png`: aggregate plots.",
            "- `reconstruction_*.png`: one representative reconstruction per family.",
            "",
            f"alpha = {parameters['diffusivity']:.6e} m^2/s; beta = "
            f"{parameters['cooling_rate']:.6f} 1/s; gamma = {args.source_coupling:.6f}.",
            "",
        ]
    )
    args.output_dir.joinpath("README.md").write_text(
        "\n".join(lines), encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate clipped, posterior-mean, and full-posterior propagation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument(
        "--reconstruction-trajectories",
        nargs="+",
        default=ablation.DEVELOPMENT_TRAJECTORIES,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--fraction-saturated", type=float, default=REFERENCE_FRACTION)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--source-coupling", type=float)
    parser.add_argument("--signal-multiplier", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--previous-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--previous-chains", type=int, default=1)
    parser.add_argument("--previous-samples", type=int, default=180)
    parser.add_argument("--previous-burn-in", type=int, default=120)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--burn-in", type=int, default=120)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_items, estimate_rows, parameters = ablation.prepare_development_set(args)
    calibrated_couplings, coupling_rows = ablation.calibrate_source_couplings(
        prepared_items,
        fractions=[args.fraction_saturated],
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_flux_threshold=args.source_flux_threshold,
    )
    if args.source_coupling is None:
        args.source_coupling = calibrated_couplings[args.fraction_saturated]
    pd.DataFrame(estimate_rows.values()).to_csv(
        args.output_dir / "development_physics_parameters.csv", index=False
    )
    coupling_rows.to_csv(
        args.output_dir / "source_coupling_calibration.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **parameters,
                "source_coupling": args.source_coupling,
                "signal_multiplier": args.signal_multiplier,
                "noise_multiplier": args.noise_multiplier,
                "previous_noise_multiplier": args.previous_noise_multiplier,
                "length_multiplier": args.length_multiplier,
                "fraction_saturated": args.fraction_saturated,
                "observation_stride": args.observation_stride,
                "previous_frame_offset": args.previous_frame_offset,
                "previous_samples": args.previous_samples * args.previous_chains,
                "current_samples": args.samples * args.chains,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "seed": args.seed,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)

    results, validation = run_experiment(args, parameters=parameters)
    results.to_csv(args.output_dir / "results.csv", index=False)
    validation.to_csv(args.output_dir / "prior_validation.csv", index=False)
    heldout = aggregate(results, heldout_only=True, by_family=False)
    all33 = aggregate(results, heldout_only=False, by_family=False)
    family = aggregate(results, heldout_only=True, by_family=True)
    paired = paired_comparisons(results)
    heldout.to_csv(args.output_dir / "heldout30_overall.csv", index=False)
    all33.to_csv(args.output_dir / "all33_overall.csv", index=False)
    family.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    plot_summary(heldout, args.output_dir / "comparison.png")
    plot_by_family(family, args.output_dir / "comparison_by_family.png")
    write_readme(
        args,
        parameters=parameters,
        overall=heldout,
        paired=paired,
        validation=validation,
    )
    print(f"Saved propagation ablation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
