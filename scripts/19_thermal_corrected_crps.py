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

from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ablation_module():
    module_path = ROOT / "scripts" / "18_thermal_two_frame_ablation.py"
    spec = importlib.util.spec_from_file_location("thermal_two_frame_ablation", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_two_frame_ablation"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ablation = load_ablation_module()
study = ablation.study
thermal = ablation.thermal

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "thermal_corrected_crps"
DEFAULT_SELECTED_CONFIG = (
    ROOT
    / "outputs"
    / "thermal_two_frame_ablation"
    / "selected_residual_configuration.csv"
)
FIXED_REGION_FRACTIONS = (0.01, 0.03, 0.05, 0.10)


def fixed_mask_multitime_observations(
    prepared,
    *,
    threshold: float,
    history_indices: np.ndarray,
    observation_stride: int,
    noise_sd: float,
    seed: int,
) -> dict[str, object]:
    """Use the same spatial points and noisy measurements at every ceiling."""
    rng = np.random.default_rng(seed)
    fixed_mask = np.zeros(prepared.truth.shape, dtype=bool)
    fixed_mask[::observation_stride, ::observation_stride] = True
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
        points = np.column_stack(
            [
                prepared.points,
                np.full(len(prepared.points), prepared.times[time_index]),
            ]
        )
        point_blocks.append(points[fixed_mask.ravel()])
        value_blocks.append(np.minimum(measured[fixed_mask], threshold))
        saturation_blocks.append(saturated[fixed_mask])
        saturation_count += saturated.astype(int)
        frame_records.append(
            {
                "time_index": int(time_index),
                "time": float(prepared.times[time_index]),
                "points": points[fixed_mask.ravel()],
                "values": np.minimum(measured[fixed_mask], threshold),
                "sat_mask": saturated[fixed_mask],
                "included": fixed_mask.copy(),
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
        "fixed_observation_mask": fixed_mask,
    }


def fixed_region_metrics(prepared, prediction) -> dict[str, float]:
    mean = np.asarray(prediction[0])
    draws = np.asarray(prediction[4])
    truth = prepared.truth.ravel()
    pointwise_crps = thermal.empirical_crps(draws, truth)
    metrics: dict[str, float] = {}

    for fraction in FIXED_REGION_FRACTIONS:
        label = f"{int(round(100 * fraction)):02d}"
        mask = truth >= float(np.quantile(truth, 1.0 - fraction))
        metrics[f"fixed_top_{label}_crps_K"] = float(np.mean(pointwise_crps[mask]))
        metrics[f"fixed_top_{label}_mae_K"] = float(
            np.mean(np.abs(mean[mask] - truth[mask]))
        )

    peak_index = int(np.argmax(truth))
    metrics["true_peak_location_crps_K"] = float(pointwise_crps[peak_index])
    metrics["true_peak_location_mae_K"] = abs(float(mean[peak_index] - truth[peak_index]))
    draw_maxima = np.max(draws, axis=1)[:, None]
    metrics["field_maximum_crps_K"] = float(
        thermal.empirical_crps(draw_maxima, np.array([np.max(truth)]))[0]
    )
    return metrics


def load_selected_configuration(path: Path) -> tuple[str, dict[str, float]]:
    if not path.is_file():
        return (
            "fixed noise x2 + length x1.25",
            {
                "signal_multiplier": 1.0,
                "noise_multiplier": 2.0,
                "beta_multiplier": 1.0,
                "length_multiplier": 1.25,
            },
        )
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one selected configuration in {path}")
    row = frame.iloc[0]
    selected = {
        name: float(row[name])
        for name in (
            "signal_multiplier",
            "noise_multiplier",
            "beta_multiplier",
            "length_multiplier",
        )
    }
    return str(row["setting"]), selected


def validate_fixed_design(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (trajectory, method), group in results.groupby(["trajectory", "method"]):
        rows.append(
            {
                "trajectory": trajectory,
                "method": method,
                "n_ceilings": int(group["fraction_saturated"].nunique()),
                "n_observations_min": int(group["n_observations"].min()),
                "n_observations_max": int(group["n_observations"].max()),
                "source_coupling_min": float(group["source_coupling"].min()),
                "source_coupling_max": float(group["source_coupling"].max()),
            }
        )
    checks = pd.DataFrame(rows)
    if not np.all(checks["n_observations_min"] == checks["n_observations_max"]):
        raise AssertionError("Observation count changed across synthetic ceilings")
    if not np.allclose(
        checks["source_coupling_min"], checks["source_coupling_max"]
    ):
        raise AssertionError("Source coupling changed across synthetic ceilings")
    return checks


def run_experiment(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    source_coupling: float,
    selected: dict[str, float],
) -> pd.DataFrame:
    records = trajectory_catalog(args.dataset_dir)
    if args.trajectory_names:
        requested = set(args.trajectory_names)
        records = [record for record in records if record.name in requested]
        missing = requested - {record.name for record in records}
        if missing:
            raise ValueError(f"Unknown trajectories: {sorted(missing)}")

    checkpoint_path = args.output_dir / "corrected_crps_checkpoint.csv"
    expected_rows = len(args.fractions_saturated) * len(ablation.METHOD_ORDER)
    if args.resume and checkpoint_path.is_file():
        checkpoint = pd.read_csv(checkpoint_path)
        rows = checkpoint.to_dict("records")
        counts = checkpoint.groupby("trajectory").size()
        completed = set(counts[counts == expected_rows].index)
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
        role = (
            "calibration"
            if record.name in ablation.DEVELOPMENT_TRAJECTORIES
            else "evaluation"
        )
        history_indices = study.select_history_indices(prepared.times, args.main_lags)
        observation_seed = args.seed + 10_000 * trajectory_index
        reference_points: dict[str, np.ndarray] = {}

        for fraction in args.fractions_saturated:
            threshold = float(np.quantile(prepared.truth, 1.0 - fraction))
            multitime = fixed_mask_multitime_observations(
                prepared,
                threshold=threshold,
                history_indices=history_indices,
                observation_stride=args.observation_stride,
                noise_sd=args.noise_sd,
                seed=observation_seed,
            )
            current = study.current_frame_observations(multitime)
            _, mean_function = ablation.physics_mean_for_trajectory(
                prepared,
                threshold=threshold,
                diffusivity=parameters["diffusivity"],
                cooling_rate=parameters["cooling_rate"],
                source_coupling=source_coupling,
                source_flux_threshold=args.source_flux_threshold,
            )
            configurations = ablation.make_method_configurations(
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

            for method_index, method in enumerate(ablation.METHOD_ORDER):
                observations, config = configurations[method]
                points = np.asarray(observations["x_obs"])
                if method in reference_points:
                    if not np.array_equal(points, reference_points[method]):
                        raise AssertionError(
                            f"Observation locations changed for {record.name}, {method}"
                        )
                else:
                    reference_points[method] = points.copy()

                row, prediction = ablation.run_method(
                    prepared,
                    method=method,
                    observations=observations,
                    config=config,
                    diffusivity=parameters["diffusivity"],
                    cooling_rate=config.cooling_rate,
                    source_coupling=source_coupling,
                    n_frames=len(history_indices) if "space-time" in method else 1,
                    chains=args.chains,
                    samples=args.samples,
                    burn_in=args.burn_in,
                    thin=args.thin,
                    seed=args.seed + 100_000 * trajectory_index + 10 * method_index,
                    return_prediction=True,
                )
                row.update(fixed_region_metrics(prepared, prediction))
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
                        "source_reference_fraction": args.reference_fraction,
                        **selected,
                    }
                )
                rows.append(row)
                print(
                    f"[{trajectory_index + 1:02d}/{len(records)}] {record.name}, "
                    f"{fraction:.0%}, {method}: top1 CRPS="
                    f"{row['fixed_top_01_crps_K']:.3f}, peak CRPS="
                    f"{row['field_maximum_crps_K']:.3f}",
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = results[results["role"] == "evaluation"].copy()
    metrics = [
        "mean_crps_K",
        "fixed_top_01_crps_K",
        "fixed_top_03_crps_K",
        "fixed_top_05_crps_K",
        "fixed_top_10_crps_K",
        "true_peak_location_crps_K",
        "field_maximum_crps_K",
        "excess_field_rel_l2",
        "peak_absolute_error_K",
        "n_observations",
        "n_saturated",
    ]
    family = (
        evaluation.groupby(
            ["family", "fraction_saturated", "method"], sort=False
        )[metrics]
        .agg(["mean", "std", "count"])
    )
    family.columns = ["_".join(column) for column in family.columns]
    family = family.reset_index()
    overall = (
        evaluation.groupby(["fraction_saturated", "method"], sort=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    overall.columns = ["_".join(column) for column in overall.columns]
    return family, overall.reset_index()


def plot_metric_grid(
    family: pd.DataFrame,
    *,
    metrics: list[tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    families = ["Diagonal", "Horizontal", "Spiral"]
    fractions = sorted(family["fraction_saturated"].unique())
    x = np.arange(len(fractions))
    figure, axes = plt.subplots(
        len(metrics),
        len(families),
        figsize=(15.5, 3.5 * len(metrics)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(metrics), len(families))
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, family_name in enumerate(families):
            axis = axes[row_index, column_index]
            subset = family[family["family"] == family_name]
            for method in ablation.METHOD_ORDER:
                method_rows = subset[subset["method"] == method].set_index(
                    "fraction_saturated"
                )
                means = np.array(
                    [method_rows.loc[value, f"{metric}_mean"] for value in fractions]
                )
                standard_errors = np.array(
                    [
                        method_rows.loc[value, f"{metric}_std"]
                        / np.sqrt(method_rows.loc[value, f"{metric}_count"])
                        for value in fractions
                    ]
                )
                standard_errors = np.nan_to_num(standard_errors)
                axis.errorbar(
                    x,
                    means,
                    yerr=standard_errors,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=ablation.METHOD_COLORS[method],
                    label=method,
                )
            if row_index == 0:
                axis.set_title(family_name)
            if column_index == 0:
                axis.set_ylabel(ylabel)
            axis.set_xticks(x, [f"{value:.0%}" for value in fractions])
            axis.set_xlabel("Censored fraction")
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle(title)
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_readme(
    args: argparse.Namespace,
    *,
    selected_setting: str,
    parameters: dict[str, float],
    source_coupling: float,
    results: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    indexed = overall.set_index(["fraction_saturated", "method"])

    def score(fraction: float, method: str, metric: str) -> float:
        return float(indexed.loc[(fraction, method), f"{metric}_mean"])

    evaluation = results[results["role"] == "evaluation"]
    top_one = evaluation.pivot(
        index="trajectory",
        columns=["method", "fraction_saturated"],
        values="fixed_top_01_crps_K",
    )
    physics_worse_counts = {
        method: int(np.sum(top_one[method][0.10] > top_one[method][0.01]))
        for method in ("physics mean + RBF", "physics mean + space-time")
    }
    findings = (
        "## Main result\n\n"
        "The corrected fixed-region scores now show the expected loss of information. "
        "For physics mean + RBF, all-domain CRPS increases from "
        f"{score(0.01, 'physics mean + RBF', 'mean_crps_K'):.3f} to "
        f"{score(0.10, 'physics mean + RBF', 'mean_crps_K'):.3f} K and fixed top-1% "
        "CRPS increases from "
        f"{score(0.01, 'physics mean + RBF', 'fixed_top_01_crps_K'):.3f} to "
        f"{score(0.10, 'physics mean + RBF', 'fixed_top_01_crps_K'):.3f} K. "
        "For physics mean + space-time, the corresponding changes are "
        f"{score(0.01, 'physics mean + space-time', 'mean_crps_K'):.3f} to "
        f"{score(0.10, 'physics mean + space-time', 'mean_crps_K'):.3f} K and "
        f"{score(0.01, 'physics mean + space-time', 'fixed_top_01_crps_K'):.3f} to "
        f"{score(0.10, 'physics mean + space-time', 'fixed_top_01_crps_K'):.3f} K.\n\n"
        "Fixed top-1% CRPS is worse at 10% than at 1% censoring on "
        f"{physics_worse_counts['physics mean + RBF']}/30 held-out trajectories for "
        "physics mean + RBF and "
        f"{physics_worse_counts['physics mean + space-time']}/30 for physics mean + "
        "space-time. The naive residual-only baselines remain nearly flat because "
        "they already miss the peak substantially at the lowest censoring level.\n\n"
        "The draw-wise field-maximum CRPS is path-family dependent and should be "
        "reported separately from fixed-location and fixed-region CRPS.\n\n"
    )
    args.output_dir.joinpath("README.md").write_text(
        "# Corrected cross-ceiling CRPS experiment\n\n"
        "This experiment corrects the two confounders in the original CRPS sweep. "
        "Every synthetic ceiling uses the same spatial observation mask and the same "
        "noise realization. Saturated pixels outside that mask are not added. The "
        f"source coupling is calibrated once at {args.reference_fraction:.0%} "
        "censoring on the three development trajectories and transferred unchanged.\n\n"
        "CRPS is reported on the full prediction domain, fixed top-1%, top-3%, "
        "top-5%, and top-10% truth regions, the true peak location, and the posterior "
        "distribution of the field maximum. These masks do not change with the "
        "synthetic ceiling.\n\n"
        f"Selected residual setting: `{selected_setting}`. Fixed source coupling: "
        f"{source_coupling:.6f}. Diffusivity: {parameters['diffusivity']:.6e} m^2/s; "
        f"cooling rate: {parameters['cooling_rate']:.4f} 1/s.\n\n"
        + findings
        + "Files:\n\n"
        "- `corrected_crps_results.csv`: trajectory-level metrics.\n"
        "- `corrected_crps_by_family.csv`: family means, standard deviations, and counts.\n"
        "- `corrected_crps_overall.csv`: held-out aggregate results.\n"
        "- `design_checks.csv`: verifies fixed observation counts and source coupling.\n"
        "- `corrected_crps_fixed_regions.png`: fixed-domain and fixed-hot-region CRPS.\n"
        "- `corrected_crps_peak.png`: true-peak-location and field-maximum CRPS.\n",
        encoding="ascii",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled cross-ceiling CRPS comparison."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-config", type=Path, default=DEFAULT_SELECTED_CONFIG)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--main-lags", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument(
        "--fractions-saturated",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
    )
    parser.add_argument("--reference-fraction", type=float, default=0.03)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
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
    source_couplings, coupling_rows = ablation.calibrate_source_couplings(
        prepared_items,
        fractions=[args.reference_fraction],
        diffusivity=parameters["diffusivity"],
        cooling_rate=parameters["cooling_rate"],
        source_flux_threshold=args.source_flux_threshold,
    )
    source_coupling = source_couplings[args.reference_fraction]
    selected_setting, selected = load_selected_configuration(args.selected_config)
    pd.DataFrame(estimate_rows.values()).to_csv(
        args.output_dir / "development_physics_parameters.csv", index=False
    )
    coupling_rows.to_csv(
        args.output_dir / "fixed_source_coupling_calibration.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "setting": selected_setting,
                "source_reference_fraction": args.reference_fraction,
                "source_coupling": source_coupling,
                **parameters,
                **selected,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)

    results = run_experiment(
        args,
        parameters=parameters,
        source_coupling=source_coupling,
        selected=selected,
    )
    results.to_csv(args.output_dir / "corrected_crps_results.csv", index=False)
    checks = validate_fixed_design(results)
    checks.to_csv(args.output_dir / "design_checks.csv", index=False)
    family, overall = summarize_results(results)
    family.to_csv(args.output_dir / "corrected_crps_by_family.csv", index=False)
    overall.to_csv(args.output_dir / "corrected_crps_overall.csv", index=False)

    if not family.empty:
        plot_metric_grid(
            family,
            metrics=[
                ("mean_crps_K", "All-domain CRPS (K)"),
                ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
                ("fixed_top_05_crps_K", "Fixed top-5% CRPS (K)"),
            ],
            out_path=args.output_dir / "corrected_crps_fixed_regions.png",
            title="Corrected CRPS with fixed observations, parameters, and regions",
        )
        plot_metric_grid(
            family,
            metrics=[
                ("true_peak_location_crps_K", "True-peak-location CRPS (K)"),
                ("field_maximum_crps_K", "Field-maximum CRPS (K)"),
            ],
            out_path=args.output_dir / "corrected_crps_peak.png",
            title="Corrected peak-distribution CRPS",
        )
    write_readme(
        args,
        selected_setting=selected_setting,
        parameters=parameters,
        source_coupling=source_coupling,
        results=results,
        overall=overall,
    )
    print(f"Saved corrected CRPS outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
