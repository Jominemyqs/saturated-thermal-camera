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
from src.thermal_posterior_physics import source_centroid_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_corrected_module():
    module_path = ROOT / "scripts" / "19_thermal_corrected_crps.py"
    spec = importlib.util.spec_from_file_location(
        "thermal_corrected_crps_spde", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["thermal_corrected_crps_spde"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


corrected = load_corrected_module()
ablation = corrected.ablation
study = ablation.study
thermal = ablation.thermal
gp2d = study.gp2d

DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "by_experiment" / "19_stochastic_spde_ablation"
)
REFERENCE_FRACTION = 0.03
METHODS = [
    "regular mean + RBF",
    "regular mean + stochastic space-time",
    "physics mean + RBF",
    "physics mean + stochastic space-time",
    "physics mean + advective stochastic space-time",
]
METHOD_COLORS = {
    "regular mean + RBF": "#E69F00",
    "regular mean + stochastic space-time": "#0072B2",
    "physics mean + RBF": "#D55E00",
    "physics mean + stochastic space-time": "#009E73",
    "physics mean + advective stochastic space-time": "#CC79A7",
}
METHOD_LABELS = {
    "regular mean + RBF": "regular\nRBF",
    "regular mean + stochastic space-time": "regular\nstochastic ST",
    "physics mean + RBF": "physics\nRBF",
    "physics mean + stochastic space-time": "physics\nstochastic ST",
    "physics mean + advective stochastic space-time": "physics\nadvective ST",
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
]
PAIRINGS = [
    (
        "regular mean + stochastic space-time",
        "regular mean + RBF",
        "stochastic covariance effect with regular mean",
    ),
    (
        "physics mean + RBF",
        "regular mean + RBF",
        "physics-mean effect with RBF covariance",
    ),
    (
        "physics mean + stochastic space-time",
        "regular mean + stochastic space-time",
        "physics-mean effect with stochastic covariance",
    ),
    (
        "physics mean + stochastic space-time",
        "physics mean + RBF",
        "stochastic covariance effect with physics mean",
    ),
    (
        "physics mean + advective stochastic space-time",
        "physics mean + stochastic space-time",
        "advection effect with stochastic covariance",
    ),
]


def make_path_function(times: np.ndarray, positions: np.ndarray):
    """Interpolate the HeatFluxZ source centroid at arbitrary query times."""

    def path_function(query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times, dtype=float).reshape(-1)
        return np.column_stack(
            [
                np.interp(query, times, positions[:, 0]),
                np.interp(query, times, positions[:, 1]),
            ]
        )

    return path_function


def residual_parameters(args: argparse.Namespace) -> dict[str, float]:
    return {
        "signal_multiplier": args.signal_multiplier,
        "noise_multiplier": args.noise_multiplier,
        "beta_multiplier": args.beta_multiplier,
        "length_multiplier": args.length_multiplier,
    }


def make_model_configurations(
    args: argparse.Namespace,
    prepared,
    *,
    current: dict[str, object],
    multitime: dict[str, object],
    physics_mean_function,
    parameters: dict[str, float],
) -> dict[str, tuple[dict[str, object], object, float]]:
    signal_sd = parameters["signal_sd"] * args.signal_multiplier
    noise_sd = args.noise_sd * args.noise_multiplier
    cooling_rate = parameters["cooling_rate"] * args.beta_multiplier
    lengthscale = prepared.source_lengthscale * args.length_multiplier

    def rbf_config(mean_function=None):
        base = study.make_config(
            prepared,
            kernel="rbf",
            diffusivity=parameters["diffusivity"],
            cooling_rate=cooling_rate,
            signal_sd=signal_sd,
            noise_sd=noise_sd,
            mean_function=mean_function,
        )
        return replace(base, lengthscale=lengthscale)

    def stochastic_config(mean_function=None):
        # study.make_config predates the SPDE kernel, so initialize its shared
        # space-time fields first and then select the stationary forced kernel.
        base = study.make_config(
            prepared,
            kernel="spatiotemporal_heat",
            diffusivity=parameters["diffusivity"],
            cooling_rate=cooling_rate,
            signal_sd=signal_sd,
            noise_sd=noise_sd,
            mean_function=mean_function,
        )
        return replace(
            base,
            kernel="spatiotemporal_forced_heat",
            lengthscale=lengthscale,
            forcing_lengthscale=lengthscale * args.forcing_length_multiplier,
            forcing_quadrature_order=args.forcing_quadrature_order,
        )

    positions = source_centroid_path(
        prepared,
        source_flux_threshold=args.source_flux_threshold,
    )
    path_function = make_path_function(prepared.times, positions)
    advective_stochastic = replace(
        stochastic_config(physics_mean_function),
        kernel="spatiotemporal_advective_forced_heat",
        advection_path=path_function,
    )

    return {
        "regular mean + RBF": (current, rbf_config(), np.nan),
        "regular mean + stochastic space-time": (
            multitime,
            stochastic_config(),
            np.nan,
        ),
        "physics mean + RBF": (
            current,
            rbf_config(physics_mean_function),
            args.source_coupling,
        ),
        "physics mean + stochastic space-time": (
            multitime,
            stochastic_config(physics_mean_function),
            args.source_coupling,
        ),
        "physics mean + advective stochastic space-time": (
            multitime,
            advective_stochastic,
            args.source_coupling,
        ),
    }


def implied_forcing_intensity(config) -> float:
    ell = float(config.forcing_lengthscale or config.lengthscale)
    beta = float(config.cooling_rate)
    nodes, weights = np.polynomial.laguerre.laggauss(
        config.forcing_quadrature_order
    )
    scales = ell**2 + 2.0 * config.diffusivity * nodes / beta
    zero_lag_integral = float(np.sum(weights * ell**2 / scales))
    return 2.0 * beta * config.signal_sd**2 / zero_lag_integral


def validate_stochastic_kernel(config, points: np.ndarray) -> dict[str, float]:
    probe_indices = np.linspace(0, len(points) - 1, min(70, len(points))).astype(int)
    probe = points[probe_indices]
    matrix = gp2d.rbf_kernel(probe, probe, config)
    reference = gp2d.rbf_kernel(
        probe,
        probe,
        replace(
            config,
            forcing_quadrature_order=max(2 * config.forcing_quadrature_order, 48),
        ),
    )
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    diagonal_error = float(
        np.max(np.abs(np.diag(matrix) - config.signal_sd**2))
    )
    quadrature_relative_error = float(
        np.max(np.abs(matrix - reference)) / config.signal_sd**2
    )
    tolerance = 1e-8 * max(config.signal_sd**2, 1.0)
    if symmetry_error > tolerance or minimum_eigenvalue < -tolerance:
        raise AssertionError(
            "Invalid stochastic space-time covariance: "
            f"symmetry={symmetry_error}, minimum eigenvalue={minimum_eigenvalue}"
        )
    return {
        "symmetry_error": symmetry_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_diagonal_error": diagonal_error,
        "quadrature_order": config.forcing_quadrature_order,
        "quadrature_reference_order": max(
            2 * config.forcing_quadrature_order, 48
        ),
        "quadrature_relative_error": quadrature_relative_error,
        "forcing_intensity_K2_per_s": implied_forcing_intensity(config),
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
        history_indices = study.select_history_indices(
            prepared.times, args.main_lags
        )
        multitime = corrected.fixed_mask_multitime_observations(
            prepared,
            threshold=threshold,
            history_indices=history_indices,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        current = study.current_frame_observations(multitime)
        _, physics_mean_function = ablation.physics_mean_for_trajectory(
            prepared,
            threshold=threshold,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=args.source_coupling,
            source_flux_threshold=args.source_flux_threshold,
        )
        models = make_model_configurations(
            args,
            prepared,
            current=current,
            multitime=multitime,
            physics_mean_function=physics_mean_function,
            parameters=parameters,
        )

        for validation_method in (
            "regular mean + stochastic space-time",
            "physics mean + advective stochastic space-time",
        ):
            stochastic_config = models[validation_method][1]
            validation = validate_stochastic_kernel(
                stochastic_config, np.asarray(multitime["x_obs"])
            )
            validation.update(
                {
                    "trajectory": record.name,
                    "family": record.family,
                    "role": role,
                    "method": validation_method,
                }
            )
            validation_rows.append(validation)

        prediction_seed = args.seed + 100_000 * trajectory_index
        for method in METHODS:
            key = (record.name, method)
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
                n_frames=(
                    len(history_indices)
                    if "stochastic space-time" in method
                    else 1
                ),
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
                    "fraction_saturated": args.fraction_saturated,
                    "observed_saturated_fraction": float(
                        np.mean(np.asarray(observations["sat_mask"], dtype=bool))
                    ),
                    "observation_design": (
                        f"fixed_stride_{args.observation_stride}"
                    ),
                    "mean_type": (
                        "physics one-step clipped"
                        if method.startswith("physics")
                        else "constant ambient"
                    ),
                    "residual_lengthscale_m": config.lengthscale,
                    "forcing_lengthscale_m": (
                        config.forcing_lengthscale
                        if "stochastic space-time" in method
                        else np.nan
                    ),
                    "forcing_intensity_K2_per_s": (
                        implied_forcing_intensity(config)
                        if "stochastic space-time" in method
                        else np.nan
                    ),
                    "crps_estimator": "unbiased_M_times_M_minus_1",
                }
            )
            rows.append(row)
            completed.add(key)
            print(
                f"[{trajectory_index + 1:02d}/{len(records)}] {record.name}, "
                f"{method}: field={row['excess_field_rel_l2']:.3f}, "
                f"all CRPS={row['mean_crps_K']:.3f}, "
                f"top1 CRPS={row['fixed_top_01_crps_K']:.3f}",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def aggregate(
    results: pd.DataFrame, *, heldout_only: bool, by_family: bool
) -> pd.DataFrame:
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
    ]
    rows = []
    for method, baseline, comparison in PAIRINGS:
        row: dict[str, object] = {
            "comparison": comparison,
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
            if metric == "hot_region_95_coverage":
                row[f"{metric}_win_count"] = int(np.sum(differences > 0.0))
            else:
                row[f"{metric}_win_count"] = int(np.sum(differences < 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_overall(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("excess_field_rel_l2", "Relative field error"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_95_coverage", "Hot 95% coverage"),
        ("mean_95_interval_width_K", "Mean 95% interval width (K)"),
    ]
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    indexed = summary.set_index("method")
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
            markersize=7,
            color="#333333",
        )
        for index, method in enumerate(METHODS):
            axis.scatter(index, means[index], s=55, color=METHOD_COLORS[method], zorder=3)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 1].axhline(0.95, color="#666666", linestyle="--", linewidth=1.0)
    figure.suptitle(
        "Stationary stochastic heat prior: 30 held-out trajectories at 3% censoring"
    )
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_by_family(summary: pd.DataFrame, out_path: Path) -> None:
    families = ["Diagonal", "Horizontal", "Spiral"]
    metrics = [
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("excess_field_rel_l2", "Relative field error"),
    ]
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), constrained_layout=True)
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
                axis.scatter(
                    index, means[index], s=45, color=METHOD_COLORS[method], zorder=3
                )
            axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
            if row_index == 0:
                axis.set_title(family)
            if column_index == 0:
                axis.set_ylabel(ylabel)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Stochastic heat-prior results by trajectory family")
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
    pair_table = paired.set_index("comparison")
    covariance_effect = pair_table.loc[
        "stochastic covariance effect with physics mean"
    ]
    physics_effect = pair_table.loc[
        "physics-mean effect with stochastic covariance"
    ]
    advection_effect = pair_table.loc[
        "advection effect with stochastic covariance"
    ]
    lines = [
        "# Stationary stochastic heat-SPDE ablation",
        "",
        "This experiment replaces the earlier deterministic propagation covariance with "
        "the stationary covariance of",
        "",
        "```text",
        "dr(t) = (alpha Laplacian r(t) - beta r(t)) dt + dW_Q(t),",
        "Q(x,x') = sigma_W^2 exp(-||x-x'||^2/(2 ell_W^2)).",
        "```",
        "",
        "For the stationary process,",
        "",
        "```text",
        "K(t,t') = integral_0^infinity exp(L(|t-t'|+u)) Q exp(L* u) du.",
        "```",
        "",
        "The positive covariance integral is evaluated by 24-node Gauss-Laguerre "
        "quadrature and normalized to the same marginal variance as the RBF residual. "
        "The absolute time-lag dependence is therefore justified by stationarity, not "
        "by the deterministic forward equation alone.",
        "The advective variant evaluates the same stationary covariance in "
        "source-centroid coordinates, using the HeatFluxZ trajectory without "
        "changing the clipped physics mean.",
        "",
        "All CRPS values use the unbiased empirical estimator with denominator "
        "`M(M-1)` in the pairwise-dispersion term.",
        "",
        "## Controlled setup",
        "",
        f"- All 33 trajectories are fit; the table below reports the 30 held-out paths.",
        f"- Synthetic censoring fraction: {args.fraction_saturated:.0%}.",
        f"- Current and immediately previous frames: lags {args.main_lags} s.",
        f"- Observation stride: {args.observation_stride}; measurement noise SD: "
        f"{args.noise_sd:.2f} K.",
        "- Regular mean means a constant ambient-temperature prior mean.",
        "- The physics mean is the existing clipped one-step forecast. Posterior "
        "propagation of the previous censored frame is intentionally deferred so this "
        "experiment isolates the new stochastic covariance and CRPS correction.",
        "",
        "## Held-out results",
        "",
        "| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['mean_crps_K_mean']:.3f} | "
            f"{row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Direct comparisons",
            "",
            "With the physics mean fixed, replacing RBF by the stochastic space-time "
            f"covariance changes field error by "
            f"`{covariance_effect['excess_field_rel_l2_mean_change']:+.4f}`, all-domain "
            f"CRPS by `{covariance_effect['mean_crps_K_mean_change']:+.4f} K`, and "
            f"top-1% CRPS by "
            f"`{covariance_effect['fixed_top_01_crps_K_mean_change']:+.4f} K`. It wins "
            f"on all-domain CRPS for "
            f"`{int(covariance_effect['mean_crps_K_win_count'])}/30` and on fixed "
            f"top-1% CRPS for "
            f"`{int(covariance_effect['fixed_top_01_crps_K_win_count'])}/30` held-out "
            "paths. Its field error is lower on only "
            f"`{int(covariance_effect['excess_field_rel_l2_win_count'])}/30` paths, "
            "and mean hot-region coverage changes by "
            f"`{covariance_effect['hot_region_95_coverage_mean_change']:+.4f}`.",
            "",
            "With the stochastic covariance fixed, adding the one-step physics mean "
            f"changes field error by "
            f"`{physics_effect['excess_field_rel_l2_mean_change']:+.4f}`, all-domain "
            f"CRPS by `{physics_effect['mean_crps_K_mean_change']:+.4f} K`, and top-1% "
            f"CRPS by `{physics_effect['fixed_top_01_crps_K_mean_change']:+.4f} K`.",
            "",
            "With the clipped physics mean fixed, adding source-centroid advection "
            f"to the stochastic covariance changes field error by "
            f"`{advection_effect['excess_field_rel_l2_mean_change']:+.4f}`, "
            f"all-domain CRPS by "
            f"`{advection_effect['mean_crps_K_mean_change']:+.4f} K`, and top-1% "
            f"CRPS by "
            f"`{advection_effect['fixed_top_01_crps_K_mean_change']:+.4f} K`.",
            "",
            "The stationary SPDE covariance is therefore the strongest distributional "
            "model family in this five-way comparison, whereas physics mean + RBF retains the "
            "best posterior-mean field error and slightly better hot-region coverage. "
            "This separates two claims that should not be conflated: temporal stochastic "
            "covariance improves predictive scoring, while the one-step physics mean "
            "provides most of the reconstruction accuracy.",
            "",
            "## Numerical checks",
            "",
            f"All {len(validation)} trajectory-specific stochastic covariance checks "
            "are symmetric and positive semidefinite within numerical tolerance. The "
            f"minimum tested eigenvalue is "
            f"`{validation['minimum_eigenvalue'].min():.3e}`. The largest diagonal "
            f"normalization error is "
            f"`{validation['maximum_diagonal_error'].max():.3e}`, and the largest "
            "24-node versus 48-node quadrature difference is "
            f"`{validation['quadrature_relative_error'].max():.3e}` of the marginal "
            "variance.",
            "",
            "Files:",
            "",
            "- `results.csv`: all trajectory-level fits and metrics.",
            "- `heldout30_overall.csv`: aggregate held-out results.",
            "- `all33_overall.csv`: aggregate results including development paths.",
            "- `family_summary.csv`: results separated by trajectory family.",
            "- `paired_comparisons.csv`: within-trajectory model differences and wins.",
            "- `kernel_validation.csv`: covariance and quadrature checks.",
            "- `comparison.png` and `comparison_by_family.png`: result plots.",
            "",
            "Fixed physical parameters:",
            "",
            f"- alpha = {parameters['diffusivity']:.6e} m^2/s",
            f"- beta = {parameters['cooling_rate']:.6f} 1/s",
            f"- source coupling gamma = {args.source_coupling:.6f}",
            "",
        ]
    )
    args.output_dir.joinpath("README.md").write_text(
        "\n".join(lines), encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RBF and stationary stochastic heat-SPDE residual priors."
    )
    parser.add_argument("--dataset-dir", type=Path, default=thermal.DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--main-lags", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument("--fraction-saturated", type=float, default=REFERENCE_FRACTION)
    parser.add_argument("--observation-stride", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--heat-flux-cutoff", type=float, default=300.0)
    parser.add_argument("--source-flux-threshold", type=float, default=10_000.0)
    parser.add_argument("--source-coupling", type=float)
    parser.add_argument("--signal-multiplier", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--beta-multiplier", type=float, default=1.0)
    parser.add_argument("--length-multiplier", type=float, default=1.25)
    parser.add_argument("--forcing-length-multiplier", type=float, default=1.0)
    parser.add_argument("--forcing-quadrature-order", type=int, default=24)
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
                **residual_parameters(args),
                "source_coupling": args.source_coupling,
                "forcing_length_multiplier": args.forcing_length_multiplier,
                "forcing_quadrature_order": args.forcing_quadrature_order,
                "fraction_saturated": args.fraction_saturated,
                "observation_stride": args.observation_stride,
                "crps_estimator": "unbiased_M_times_M_minus_1",
                "seed": args.seed,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)

    results, validation = run_experiment(args, parameters=parameters)
    results.to_csv(args.output_dir / "results.csv", index=False)
    validation.to_csv(args.output_dir / "kernel_validation.csv", index=False)
    heldout = aggregate(results, heldout_only=True, by_family=False)
    all33 = aggregate(results, heldout_only=False, by_family=False)
    family = aggregate(results, heldout_only=True, by_family=True)
    paired = paired_comparisons(results)
    heldout.to_csv(args.output_dir / "heldout30_overall.csv", index=False)
    all33.to_csv(args.output_dir / "all33_overall.csv", index=False)
    family.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    plot_overall(heldout, args.output_dir / "comparison.png")
    plot_by_family(family, args.output_dir / "comparison_by_family.png")
    write_readme(
        args,
        parameters=parameters,
        overall=heldout,
        paired=paired,
        validation=validation,
    )
    print(f"Saved stochastic SPDE ablation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
