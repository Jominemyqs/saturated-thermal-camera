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

from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_spatiotemporal_module():
    module_path = ROOT / "scripts" / "17_thermal_spatiotemporal_physics.py"
    spec = importlib.util.spec_from_file_location("thermal_spatiotemporal", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_spatiotemporal"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


study = load_spatiotemporal_module()
thermal = study.thermal

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thermal_two_frame_ablation"
DEVELOPMENT_TRAJECTORIES = [
    "DiagonalScanPath_8",
    "HorizontalScanPath_13",
    "SpiralScanPath_13",
]
METHOD_ORDER = [
    "snapshot RBF",
    "space-time heat",
    "physics mean + RBF",
    "physics mean + space-time",
]
METHOD_COLORS = {
    "snapshot RBF": "#E69F00",
    "space-time heat": "#0072B2",
    "physics mean + RBF": "#D55E00",
    "physics mean + space-time": "#009E73",
}
SENSITIVITY_SETTINGS = {
    "base": {},
    "signal x0.75": {"signal_multiplier": 0.75},
    "signal x1.5": {"signal_multiplier": 1.5},
    "signal x2": {"signal_multiplier": 2.0},
    "noise x0.5": {"noise_multiplier": 0.5},
    "noise x2": {"noise_multiplier": 2.0},
    "noise x3": {"noise_multiplier": 3.0},
    "noise x4": {"noise_multiplier": 4.0},
    "beta x0.5": {"beta_multiplier": 0.5},
    "beta x2": {"beta_multiplier": 2.0},
    "length x0.75": {"length_multiplier": 0.75},
    "length x1.25": {"length_multiplier": 1.25},
    "length x1.5": {"length_multiplier": 1.5},
    "noise x2 + length x1.25": {
        "noise_multiplier": 2.0,
        "length_multiplier": 1.25,
    },
    "noise x2 + length x1.5": {
        "noise_multiplier": 2.0,
        "length_multiplier": 1.5,
    },
    "noise x3 + length x1.25": {
        "noise_multiplier": 3.0,
        "length_multiplier": 1.25,
    },
}


def parameter_values(overrides: dict[str, float] | None = None) -> dict[str, float]:
    values = {
        "signal_multiplier": 1.0,
        "noise_multiplier": 1.0,
        "beta_multiplier": 1.0,
        "length_multiplier": 1.0,
    }
    values.update(overrides or {})
    return values


def prepare_development_set(args: argparse.Namespace):
    prepared_items = {}
    estimate_rows = {}
    for name in DEVELOPMENT_TRAJECTORIES:
        prepared, estimate = thermal.prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        prepared_items[name] = prepared
        estimate_rows[name] = estimate
    parameters = {
        "diffusivity": float(
            np.median([row["diffusivity_m2_s"] for row in estimate_rows.values()])
        ),
        "cooling_rate": float(
            np.median([row["cooling_rate_1_s"] for row in estimate_rows.values()])
        ),
        "signal_sd": float(
            np.median([row["cooling_excess_q95_K"] for row in estimate_rows.values()])
        ),
    }
    return prepared_items, estimate_rows, parameters


def calibrate_source_couplings(
    prepared_items,
    *,
    fractions: list[float],
    diffusivity: float,
    cooling_rate: float,
    source_flux_threshold: float,
) -> tuple[dict[float, float], pd.DataFrame]:
    rows = []
    calibrated = {}
    for fraction in fractions:
        values = []
        for name, prepared in prepared_items.items():
            threshold = float(np.quantile(prepared.truth, 1.0 - fraction))
            background, source_response = study.build_one_step_physics_components(
                prepared,
                diffusivity=diffusivity,
                cooling_rate=cooling_rate,
                threshold=threshold,
                source_flux_threshold=source_flux_threshold,
            )
            coupling, n_samples = study.calibrate_source_coupling(
                prepared,
                background,
                source_response,
            )
            values.append(coupling)
            rows.append(
                {
                    "fraction_saturated": fraction,
                    "trajectory": name,
                    "source_coupling": coupling,
                    "n_source_samples": n_samples,
                }
            )
        calibrated[fraction] = float(np.median(values))
    return calibrated, pd.DataFrame(rows)


def physics_mean_for_trajectory(
    prepared,
    *,
    threshold: float,
    diffusivity: float,
    cooling_rate: float,
    source_coupling: float,
    source_flux_threshold: float,
):
    background, source_response = study.build_one_step_physics_components(
        prepared,
        diffusivity=diffusivity,
        cooling_rate=cooling_rate,
        threshold=threshold,
        source_flux_threshold=source_flux_threshold,
    )
    mean_history = prepared.ambient + background + source_coupling * source_response
    return mean_history, study.make_physics_mean_function(prepared, mean_history)


def make_observations(
    prepared,
    *,
    fraction_saturated: float,
    lags: list[float],
    observation_stride: int,
    noise_sd: float,
    seed: int,
):
    threshold = float(np.quantile(prepared.truth, 1.0 - fraction_saturated))
    history_indices = study.select_history_indices(prepared.times, lags)
    multitime = study.make_multitime_observations(
        prepared,
        threshold=threshold,
        history_indices=history_indices,
        observation_stride=observation_stride,
        noise_sd=noise_sd,
        seed=seed,
    )
    return threshold, history_indices, multitime, study.current_frame_observations(multitime)


def make_method_configurations(
    prepared,
    *,
    current,
    multitime,
    physics_mean_function,
    diffusivity: float,
    cooling_rate: float,
    signal_sd: float,
    noise_sd: float,
    multipliers: dict[str, float],
):
    residual_signal = signal_sd * multipliers["signal_multiplier"]
    residual_noise = noise_sd * multipliers["noise_multiplier"]
    residual_beta = cooling_rate * multipliers["beta_multiplier"]
    residual_lengthscale = prepared.source_lengthscale * multipliers["length_multiplier"]

    def config(kernel: str, mean_function=None):
        base = study.make_config(
            prepared,
            kernel=kernel,
            diffusivity=diffusivity,
            cooling_rate=residual_beta,
            signal_sd=residual_signal,
            noise_sd=residual_noise,
            mean_function=mean_function,
        )
        return replace(base, lengthscale=residual_lengthscale)

    return {
        "snapshot RBF": (current, config("rbf")),
        "space-time heat": (multitime, config("spatiotemporal_heat")),
        "physics mean + RBF": (current, config("rbf", physics_mean_function)),
        "physics mean + space-time": (
            multitime,
            config("spatiotemporal_heat", physics_mean_function),
        ),
    }


def run_method(
    prepared,
    *,
    method: str,
    observations,
    config,
    diffusivity: float,
    cooling_rate: float,
    source_coupling: float,
    n_frames: int,
    chains: int,
    samples: int,
    burn_in: int,
    thin: int,
    seed: int,
    interval_calibration_scale: float = 1.0,
    return_prediction: bool = False,
):
    prediction = thermal.sample_censored_multiple_chains(
        observations,
        config,
        n_chains=chains,
        samples_per_chain=samples,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )
    row = study.compute_metrics(
        prepared,
        method=method,
        observations=observations,
        prediction=prediction,
        diffusivity=diffusivity,
        cooling_rate=cooling_rate,
        signal_sd=config.signal_sd,
        source_coupling=source_coupling,
        n_frames=n_frames,
    )
    truth = prepared.truth.ravel()
    hot = truth >= float(observations["threshold"])
    lower = prediction[2]
    upper = prediction[3]
    interval_score = upper - lower
    interval_score += 40.0 * (lower - truth) * (truth < lower)
    interval_score += 40.0 * (truth - upper) * (truth > upper)
    row["mean_95_interval_width_K"] = float(np.mean(upper - lower))
    row["hot_region_95_interval_width_K"] = float(np.mean((upper - lower)[hot]))
    row["hot_region_interval_score_K"] = float(np.mean(interval_score[hot]))
    mean = prediction[0]
    calibrated_lower = mean - interval_calibration_scale * (mean - lower)
    calibrated_upper = mean + interval_calibration_scale * (upper - mean)
    calibrated_interval_score = calibrated_upper - calibrated_lower
    calibrated_interval_score += (
        40.0 * (calibrated_lower - truth) * (truth < calibrated_lower)
    )
    calibrated_interval_score += (
        40.0 * (truth - calibrated_upper) * (truth > calibrated_upper)
    )
    peak_index = int(np.argmax(truth))
    row["interval_calibration_scale"] = interval_calibration_scale
    row["calibrated_true_peak_in_95"] = bool(
        calibrated_lower[peak_index] <= truth[peak_index] <= calibrated_upper[peak_index]
    )
    row["calibrated_hot_region_95_coverage"] = float(
        np.mean(
            (calibrated_lower[hot] <= truth[hot])
            & (truth[hot] <= calibrated_upper[hot])
        )
    )
    row["calibrated_hot_region_95_interval_width_K"] = float(
        np.mean((calibrated_upper - calibrated_lower)[hot])
    )
    row["calibrated_hot_region_interval_score_K"] = float(
        np.mean(calibrated_interval_score[hot])
    )
    if return_prediction:
        return row, prediction
    return row


def fit_interval_calibration(
    args: argparse.Namespace,
    prepared_items,
    parameters: dict[str, float],
    source_coupling: float,
    selected: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    required_scales = {method: [] for method in METHOD_ORDER}
    for trajectory_index, (name, prepared) in enumerate(prepared_items.items()):
        threshold, indices, multitime, current = make_observations(
            prepared,
            fraction_saturated=args.calibration_fraction,
            lags=args.main_lags,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + trajectory_index,
        )
        _, mean_function = physics_mean_for_trajectory(
            prepared,
            threshold=threshold,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=source_coupling,
            source_flux_threshold=args.source_flux_threshold,
        )
        configurations = make_method_configurations(
            prepared,
            current=current,
            multitime=multitime,
            physics_mean_function=mean_function,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            signal_sd=parameters["signal_sd"],
            noise_sd=args.noise_sd,
            multipliers=selected,
        )
        truth = prepared.truth.ravel()
        hot = truth >= threshold
        for method_index, method in enumerate(METHOD_ORDER):
            observations, config = configurations[method]
            row, prediction = run_method(
                prepared,
                method=method,
                observations=observations,
                config=config,
                diffusivity=parameters["diffusivity"],
                cooling_rate=config.cooling_rate,
                source_coupling=source_coupling,
                n_frames=len(indices) if "space-time" in method else 1,
                chains=args.sensitivity_chains,
                samples=args.sensitivity_samples,
                burn_in=args.sensitivity_burn_in,
                thin=args.thin,
                seed=args.seed + 2_000 + 100 * trajectory_index + method_index,
                return_prediction=True,
            )
            mean, _, lower, upper, _ = prediction
            lower_radius = np.maximum(mean - lower, 1e-8)
            upper_radius = np.maximum(upper - mean, 1e-8)
            scores = np.where(
                truth < mean,
                (mean - truth) / lower_radius,
                (truth - mean) / upper_radius,
            )
            required_scales[method].extend(scores[hot].tolist())
            row.update(
                {
                    "family": name.split("ScanPath_", 1)[0],
                    "fraction_saturated": args.calibration_fraction,
                }
            )
            rows.append(row)

    scales = {}
    for method, values in required_scales.items():
        scores = np.asarray(values, dtype=float)
        quantile_level = min(
            np.ceil((len(scores) + 1) * 0.95) / len(scores),
            1.0,
        )
        scales[method] = max(
            1.0,
            float(np.quantile(scores, quantile_level, method="higher")),
        )
    return scales, pd.DataFrame(rows)


def run_sensitivity(
    args: argparse.Namespace,
    prepared_items,
    parameters: dict[str, float],
    source_coupling: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows = []
    fraction = args.calibration_fraction
    for trajectory_index, (name, prepared) in enumerate(prepared_items.items()):
        threshold, history_indices, multitime, current = make_observations(
            prepared,
            fraction_saturated=fraction,
            lags=args.main_lags,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + trajectory_index,
        )
        _, mean_function = physics_mean_for_trajectory(
            prepared,
            threshold=threshold,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=source_coupling,
            source_flux_threshold=args.source_flux_threshold,
        )
        for setting_index, (setting, overrides) in enumerate(SENSITIVITY_SETTINGS.items()):
            multipliers = parameter_values(overrides)
            configurations = make_method_configurations(
                prepared,
                current=current,
                multitime=multitime,
                physics_mean_function=mean_function,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                signal_sd=parameters["signal_sd"],
                noise_sd=args.noise_sd,
                multipliers=multipliers,
            )
            observations, config = configurations["physics mean + space-time"]
            row = run_method(
                prepared,
                method="physics mean + space-time",
                observations=observations,
                config=config,
                diffusivity=parameters["diffusivity"],
                cooling_rate=config.cooling_rate,
                source_coupling=source_coupling,
                n_frames=len(history_indices),
                chains=args.sensitivity_chains,
                samples=args.sensitivity_samples,
                burn_in=args.sensitivity_burn_in,
                thin=args.thin,
                seed=args.seed + 100 * trajectory_index + setting_index,
            )
            row.update({"setting": setting, "fraction_saturated": fraction, **multipliers})
            rows.append(row)
            print(
                f"sensitivity {name}, {setting}: hot CRPS="
                f"{row['hot_region_crps_K']:.3f}, coverage="
                f"{row['hot_region_95_coverage']:.3f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("setting", sort=False)
        .agg(
            hot_region_crps_K=("hot_region_crps_K", "mean"),
            hot_region_95_coverage=("hot_region_95_coverage", "mean"),
            hot_region_interval_score_K=("hot_region_interval_score_K", "mean"),
            excess_field_rel_l2=("excess_field_rel_l2", "mean"),
            peak_absolute_error_K=("peak_absolute_error_K", "mean"),
        )
        .reset_index()
    )
    summary["coverage_error"] = np.abs(summary["hot_region_95_coverage"] - 0.95)
    eligible = summary[summary["hot_region_95_coverage"] >= args.minimum_calibration_coverage]
    selection_pool = eligible if len(eligible) else summary
    selected_setting = str(
        selection_pool.sort_values(
            ["hot_region_crps_K", "hot_region_interval_score_K"]
        ).iloc[0]["setting"]
    )
    selected = parameter_values(SENSITIVITY_SETTINGS[selected_setting])
    summary["selected"] = summary["setting"] == selected_setting
    return results, summary, selected


def run_history_sensitivity(
    args: argparse.Namespace,
    prepared_items,
    parameters: dict[str, float],
    source_coupling: float,
    selected: dict[str, float],
) -> pd.DataFrame:
    rows = []
    lag_sets = {
        "two frames": args.main_lags,
        "four frames": args.four_frame_lags,
    }
    for trajectory_index, (name, prepared) in enumerate(prepared_items.items()):
        for history_index, (label, lags) in enumerate(lag_sets.items()):
            threshold, indices, multitime, current = make_observations(
                prepared,
                fraction_saturated=args.calibration_fraction,
                lags=lags,
                observation_stride=args.observation_stride,
                noise_sd=args.noise_sd,
                seed=args.seed + trajectory_index,
            )
            _, mean_function = physics_mean_for_trajectory(
                prepared,
                threshold=threshold,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                source_coupling=source_coupling,
                source_flux_threshold=args.source_flux_threshold,
            )
            configurations = make_method_configurations(
                prepared,
                current=current,
                multitime=multitime,
                physics_mean_function=mean_function,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                signal_sd=parameters["signal_sd"],
                noise_sd=args.noise_sd,
                multipliers=selected,
            )
            observations, config = configurations["physics mean + space-time"]
            row = run_method(
                prepared,
                method="physics mean + space-time",
                observations=observations,
                config=config,
                diffusivity=parameters["diffusivity"],
                cooling_rate=config.cooling_rate,
                source_coupling=source_coupling,
                n_frames=len(indices),
                chains=args.sensitivity_chains,
                samples=args.sensitivity_samples,
                burn_in=args.sensitivity_burn_in,
                thin=args.thin,
                seed=args.seed + 500 + 100 * trajectory_index + history_index,
            )
            row.update(
                {
                    "history_setup": label,
                    "history_lags_s": ";".join(str(value) for value in lags),
                    "fraction_saturated": args.calibration_fraction,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_full_ablation(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    source_couplings: dict[float, float],
    selected: dict[str, float],
    interval_scales: dict[str, float],
) -> pd.DataFrame:
    records = trajectory_catalog(args.dataset_dir)
    checkpoint_path = args.output_dir / "trajectory_ablation_checkpoint.csv"
    if args.resume and checkpoint_path.is_file():
        rows = pd.read_csv(checkpoint_path).to_dict("records")
        completed = {str(row["trajectory"]) for row in rows}
        print(f"Resuming after {len(completed)} completed trajectories", flush=True)
    else:
        rows = []
        completed = set()
    for trajectory_index, record in enumerate(records):
        if record.name in completed:
            continue
        prepared, _ = thermal.prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            time_stride=1,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        role = "calibration" if record.name in DEVELOPMENT_TRAJECTORIES else "evaluation"
        for fraction_index, fraction in enumerate(args.fractions_saturated):
            threshold, indices, multitime, current = make_observations(
                prepared,
                fraction_saturated=fraction,
                lags=args.main_lags,
                observation_stride=args.observation_stride,
                noise_sd=args.noise_sd,
                seed=args.seed + 10_000 * trajectory_index + fraction_index,
            )
            _, mean_function = physics_mean_for_trajectory(
                prepared,
                threshold=threshold,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                source_coupling=source_couplings[fraction],
                source_flux_threshold=args.source_flux_threshold,
            )
            configurations = make_method_configurations(
                prepared,
                current=current,
                multitime=multitime,
                physics_mean_function=mean_function,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                signal_sd=parameters["signal_sd"],
                noise_sd=args.noise_sd,
                multipliers=selected,
            )
            for method_index, method in enumerate(METHOD_ORDER):
                observations, config = configurations[method]
                row = run_method(
                    prepared,
                    method=method,
                    observations=observations,
                    config=config,
                    diffusivity=parameters["diffusivity"],
                    cooling_rate=config.cooling_rate,
                    source_coupling=source_couplings[fraction],
                    n_frames=len(indices) if "space-time" in method else 1,
                    chains=args.chains,
                    samples=args.samples,
                    burn_in=args.burn_in,
                    thin=args.thin,
                    seed=(
                        args.seed
                        + 100_000 * trajectory_index
                        + 1_000 * fraction_index
                        + 10 * method_index
                    ),
                    interval_calibration_scale=interval_scales[method],
                )
                row.update(
                    {
                        "family": record.family,
                        "run_index": record.run_index,
                        "role": role,
                        "fraction_saturated": fraction,
                        **selected,
                    }
                )
                rows.append(row)
                print(
                    f"[{trajectory_index + 1:02d}/{len(records)}] {record.name}, "
                    f"{fraction:.0%}, {method}: field={row['excess_field_rel_l2']:.3f}, "
                    f"coverage={row['hot_region_95_coverage']:.3f}",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def summarize_ablation(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "excess_field_rel_l2",
        "hot_region_rel_l2",
        "peak_absolute_error_K",
        "hot_region_crps_K",
        "hot_region_95_coverage",
        "true_peak_in_95",
        "hot_region_95_interval_width_K",
        "hot_region_interval_score_K",
        "calibrated_hot_region_95_coverage",
        "calibrated_true_peak_in_95",
        "calibrated_hot_region_95_interval_width_K",
        "calibrated_hot_region_interval_score_K",
    ]
    evaluation = results[results["role"] == "evaluation"]
    family = (
        evaluation.groupby(["family", "fraction_saturated", "method"], sort=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    family.columns = ["_".join(column) for column in family.columns]
    family = family.reset_index()
    by_ceiling = (
        evaluation.groupby(["fraction_saturated", "method"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    main = by_ceiling[
        np.isclose(by_ceiling["fraction_saturated"], 0.03)
    ].copy()
    main["method"] = pd.Categorical(main["method"], METHOD_ORDER, ordered=True)
    main = main.sort_values("method")
    return family, by_ceiling, main


def plot_sensitivity(summary: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(summary))
    colors = ["#009E73" if value else "#A9A9A9" for value in summary["selected"]]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    axes[0].bar(x, summary["hot_region_crps_K"], color=colors)
    axes[0].set_ylabel("Mean hot-region CRPS (K)")
    axes[1].bar(x, summary["hot_region_95_coverage"], color=colors)
    axes[1].axhline(0.95, color="#333333", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Mean hot-region 95% coverage")
    axes[2].bar(x, summary["excess_field_rel_l2"], color=colors)
    axes[2].set_ylabel("Mean relative field error")
    for axis in axes:
        axis.set_xticks(x, summary["setting"], rotation=55, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Two-frame residual-hyperparameter sensitivity")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_family_results(family: pd.DataFrame, out_path: Path) -> None:
    fractions = sorted(family["fraction_saturated"].unique())
    families = ["Diagonal", "Horizontal", "Spiral"]
    x = np.arange(len(families))
    width = 0.19
    figure, axes = plt.subplots(2, len(fractions), figsize=(16.5, 7.8), constrained_layout=True)
    for column, fraction in enumerate(fractions):
        subset = family[np.isclose(family["fraction_saturated"], fraction)]
        indexed = subset.set_index(["family", "method"])
        for method_index, method in enumerate(METHOD_ORDER):
            positions = x + (method_index - 1.5) * width
            field_values = [indexed.loc[(name, method), "excess_field_rel_l2_mean"] for name in families]
            coverage_values = [indexed.loc[(name, method), "hot_region_95_coverage_mean"] for name in families]
            axes[0, column].bar(positions, field_values, width, color=METHOD_COLORS[method], label=method)
            axes[1, column].bar(positions, coverage_values, width, color=METHOD_COLORS[method])
        axes[0, column].set_title(f"{fraction:.0%} censored")
        axes[0, column].set_xticks(x, families)
        axes[1, column].set_xticks(x, families)
        axes[1, column].axhline(0.95, color="#333333", linestyle="--", linewidth=1.0)
        for axis in axes[:, column]:
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean relative field error")
    axes[1, 0].set_ylabel("Mean hot-region 95% coverage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out trajectories: ablation by family and censoring level")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_uncertainty_calibration(by_ceiling: pd.DataFrame, out_path: Path) -> None:
    fractions = sorted(by_ceiling["fraction_saturated"].unique())
    x = np.arange(len(fractions))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for method in METHOD_ORDER:
        subset = by_ceiling[by_ceiling["method"] == method].set_index(
            "fraction_saturated"
        )
        axes[0].plot(
            x,
            [subset.loc[value, "hot_region_95_coverage"] for value in fractions],
            marker="o",
            color=METHOD_COLORS[method],
            label=method,
        )
        axes[1].plot(
            x,
            [
                subset.loc[value, "calibrated_hot_region_95_coverage"]
                for value in fractions
            ],
            marker="o",
            color=METHOD_COLORS[method],
            label=method,
        )
    for axis, title in zip(axes, ["Raw posterior intervals", "Development-calibrated intervals"]):
        axis.axhline(0.95, color="#333333", linestyle="--", linewidth=1.1)
        axis.set_xticks(x, [f"{value:.0%}" for value in fractions])
        axis.set_xlabel("Censored fraction")
        axis.set_ylabel("Held-out hot-region 95% coverage")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_readme(
    output_dir: Path,
    *,
    parameters: dict[str, float],
    selected_setting: str,
    selected: dict[str, float],
    source_couplings: dict[float, float],
    interval_scales: dict[str, float],
) -> None:
    coupling_text = ", ".join(
        f"{fraction:.0%}: {value:.5f}" for fraction, value in source_couplings.items()
    )
    interval_text = ", ".join(
        f"{method}: {value:.3f}" for method, value in interval_scales.items()
    )
    findings = ""
    history_path = output_dir / "history_length_sensitivity.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history_summary = history.groupby("history_setup")[
            ["excess_field_rel_l2", "hot_region_crps_K"]
        ].mean()
        findings += (
            "## Findings\n\n"
            "Two frames outperformed four on the development trajectories: mean "
            f"field error {history_summary.loc['two frames', 'excess_field_rel_l2']:.3f} "
            f"versus {history_summary.loc['four frames', 'excess_field_rel_l2']:.3f}, "
            "and hot-region CRPS "
            f"{history_summary.loc['two frames', 'hot_region_crps_K']:.3f} K versus "
            f"{history_summary.loc['four frames', 'hot_region_crps_K']:.3f} K.\n\n"
        )
    results_path = output_dir / "trajectory_ablation_results.csv"
    if results_path.is_file():
        results = pd.read_csv(results_path)
        evaluation = results[results["role"] == "evaluation"]
        pivot = evaluation.pivot(
            index=["trajectory", "fraction_saturated"],
            columns="method",
            values="excess_field_rel_l2",
        )
        physics_wins = int(
            np.sum(pivot["physics mean + RBF"] < pivot["snapshot RBF"])
        )
        combined_wins = int(
            np.sum(pivot["physics mean + space-time"] < pivot["snapshot RBF"])
        )
        three_percent = evaluation[np.isclose(evaluation["fraction_saturated"], 0.03)]
        table = three_percent.groupby("method")[
            [
                "excess_field_rel_l2",
                "hot_region_crps_K",
                "hot_region_95_coverage",
                "calibrated_hot_region_95_coverage",
            ]
        ].mean()
        findings += (
            "Across the 120 held-out trajectory/ceiling cases, the physics mean plus "
            f"snapshot RBF beat the snapshot-only field error in {physics_wins}/120 "
            "cases, and the combined model did so in "
            f"{combined_wins}/120 cases. At 3% censoring, the physics-mean snapshot "
            f"and combined field errors were {table.loc['physics mean + RBF', 'excess_field_rel_l2']:.3f} "
            f"and {table.loc['physics mean + space-time', 'excess_field_rel_l2']:.3f}; "
            "their hot-region CRPS values were "
            f"{table.loc['physics mean + RBF', 'hot_region_crps_K']:.3f} K and "
            f"{table.loc['physics mean + space-time', 'hot_region_crps_K']:.3f} K.\n\n"
            "The snapshot residual remained better calibrated before correction. At "
            "3% censoring, raw hot-region coverage was "
            f"{table.loc['physics mean + RBF', 'hot_region_95_coverage']:.3f} for the "
            "snapshot residual and "
            f"{table.loc['physics mean + space-time', 'hot_region_95_coverage']:.3f} "
            "for the space-time residual. Development-derived interval scaling raised "
            "the corresponding held-out coverages to "
            f"{table.loc['physics mean + RBF', 'calibrated_hot_region_95_coverage']:.3f} "
            f"and {table.loc['physics mean + space-time', 'calibrated_hot_region_95_coverage']:.3f}. "
            "Peak coverage remains substantially harder than pointwise hot-region "
            "coverage.\n\n"
        )

    output_dir.joinpath("README.md").write_text(
        "# Fixed two-frame thermal GP ablation\n\n"
        "The main model uses the immediately previous frame, the current censored "
        "frame, a one-step physics forecast mean, and a short-lag space-time heat "
        "kernel for the residual. The three original trajectories are used only to "
        "select one global residual configuration; the other 30 trajectories are "
        "marked as evaluation cases.\n\n"
        "The four-frame version is retained only as a sensitivity comparison. The "
        "ablation isolates the snapshot residual, temporal covariance, physics mean, "
        "and their combination. Synthetic ceilings are trajectory-specific truth "
        "quantiles at the requested censoring fractions.\n\n"
        f"Selected sensitivity setting: `{selected_setting}` with multipliers "
        f"`{selected}`.\n\n"
        f"Development diffusivity: {parameters['diffusivity']:.6e} m^2/s; "
        f"cooling rate: {parameters['cooling_rate']:.4f} 1/s; residual signal "
        f"scale: {parameters['signal_sd']:.4f} K.\n\n"
        f"Transferred source couplings by ceiling: {coupling_text}.\n\n"
        f"Development interval-inflation factors: {interval_text}. Raw and "
        "calibrated interval metrics are both retained.\n\n"
        + findings
        + "Files:\n\n"
        "- `residual_sensitivity_results.csv` and `residual_sensitivity_summary.csv`\n"
        "- `history_length_sensitivity.csv`\n"
        "- `trajectory_ablation_results.csv`\n"
        "- `trajectory_family_summary.csv`\n"
        "- `ablation_by_ceiling.csv` and `ablation_table_3pct.csv`\n"
        "- `interval_calibration_scales.csv`\n"
        "- `residual_sensitivity.png`, `family_ablation.png`, and "
        "`uncertainty_calibration.png`\n",
        encoding="ascii",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate the fixed two-frame thermal GP ablation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["sensitivity", "full", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--main-lags", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument(
        "--four-frame-lags", nargs="+", type=float, default=[0.0, 0.01, 0.025, 0.05]
    )
    parser.add_argument(
        "--fractions-saturated", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.10]
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.03)
    parser.add_argument("--minimum-calibration-coverage", type=float, default=0.85)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--sensitivity-chains", type=int, default=1)
    parser.add_argument("--sensitivity-samples", type=int, default=160)
    parser.add_argument("--sensitivity-burn-in", type=int, default=100)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--burn-in", type=int, default=120)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_items, estimate_rows, parameters = prepare_development_set(args)
    fractions = sorted(set(args.fractions_saturated + [args.calibration_fraction]))
    source_couplings, coupling_rows = calibrate_source_couplings(
        prepared_items,
        fractions=fractions,
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_flux_threshold=args.source_flux_threshold,
    )
    pd.DataFrame(estimate_rows.values()).to_csv(
        args.output_dir / "development_physics_parameters.csv", index=False
    )
    coupling_rows.to_csv(args.output_dir / "development_source_couplings.csv", index=False)

    sensitivity_path = args.output_dir / "residual_sensitivity_results.csv"
    sensitivity_summary_path = args.output_dir / "residual_sensitivity_summary.csv"
    if args.stage in {"sensitivity", "all"}:
        sensitivity, sensitivity_summary, selected = run_sensitivity(
            args,
            prepared_items,
            parameters,
            source_couplings[args.calibration_fraction],
        )
        sensitivity.to_csv(sensitivity_path, index=False)
        sensitivity_summary.to_csv(sensitivity_summary_path, index=False)
        plot_sensitivity(sensitivity_summary, args.output_dir / "residual_sensitivity.png")
        history = run_history_sensitivity(
            args,
            prepared_items,
            parameters,
            source_couplings[args.calibration_fraction],
            selected,
        )
        history.to_csv(args.output_dir / "history_length_sensitivity.csv", index=False)
    else:
        if not sensitivity_summary_path.is_file():
            raise FileNotFoundError(
                f"Run --stage sensitivity first; missing {sensitivity_summary_path}"
            )
        sensitivity_summary = pd.read_csv(sensitivity_summary_path)
        selected_setting = str(sensitivity_summary.loc[sensitivity_summary["selected"], "setting"].iloc[0])
        selected = parameter_values(SENSITIVITY_SETTINGS[selected_setting])

    selected_setting = str(
        sensitivity_summary.loc[sensitivity_summary["selected"], "setting"].iloc[0]
    )
    pd.DataFrame([{"setting": selected_setting, **selected}]).to_csv(
        args.output_dir / "selected_residual_configuration.csv", index=False
    )

    calibration_path = args.output_dir / "interval_calibration_scales.csv"
    if args.stage in {"sensitivity", "all"}:
        interval_scales, calibration_rows = fit_interval_calibration(
            args,
            prepared_items,
            parameters,
            source_couplings[args.calibration_fraction],
            selected,
        )
        calibration_rows.to_csv(
            args.output_dir / "development_interval_calibration.csv", index=False
        )
        pd.DataFrame(
            [
                {"method": method, "interval_calibration_scale": value}
                for method, value in interval_scales.items()
            ]
        ).to_csv(calibration_path, index=False)
    else:
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Run --stage sensitivity first; missing {calibration_path}"
            )
        scale_frame = pd.read_csv(calibration_path)
        interval_scales = dict(
            zip(scale_frame["method"], scale_frame["interval_calibration_scale"])
        )

    if args.stage in {"full", "all"}:
        results = run_full_ablation(
            args,
            parameters=parameters,
            source_couplings=source_couplings,
            selected=selected,
            interval_scales=interval_scales,
        )
        results.to_csv(args.output_dir / "trajectory_ablation_results.csv", index=False)
        family, by_ceiling, main_table = summarize_ablation(results)
        family.to_csv(args.output_dir / "trajectory_family_summary.csv", index=False)
        by_ceiling.to_csv(args.output_dir / "ablation_by_ceiling.csv", index=False)
        main_table.to_csv(args.output_dir / "ablation_table_3pct.csv", index=False)
        plot_family_results(family, args.output_dir / "family_ablation.png")
        plot_uncertainty_calibration(
            by_ceiling, args.output_dir / "uncertainty_calibration.png"
        )

    write_readme(
        args.output_dir,
        parameters=parameters,
        selected_setting=selected_setting,
        selected=selected,
        source_couplings={fraction: source_couplings[fraction] for fraction in args.fractions_saturated},
        interval_scales=interval_scales,
    )
    print(f"Saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
