from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

from src.censored_gp import RBFConfig, rbf_covariance, sample_multiple_chains
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    build_calibration_components,
    calibrate_source_coupling,
    grid_mean_function,
    infer_previous_censored_posterior,
    paired_camera_observations,
    posterior_physics_means,
    prediction_metrics,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "24_posterior_physics_mean_advection"
)
PROPAGATION_OUTPUT = (
    ROOT / "outputs" / "by_experiment" / "20_physics_mean_propagation"
)
KERNEL_OUTPUT = ROOT / "outputs" / "by_experiment" / "19_stochastic_spde_ablation"
METHODS = [
    "posterior physics mean + RBF",
    "advective posterior physics mean + RBF",
]
METHOD_COLORS = {
    "posterior physics mean + RBF": "#0072B2",
    "advective posterior physics mean + RBF": "#009E73",
}
METRICS = [
    "excess_field_rel_l2",
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "peak_absolute_error_K",
    "hot_region_95_coverage",
    "hot_region_95_interval_width_K",
]


def prepare_development(args: argparse.Namespace):
    prepared = {}
    estimates = {}
    for name in DEVELOPMENT_TRAJECTORIES:
        item, estimate = prepare_trajectory(
            args.dataset_dir,
            name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        prepared[name] = item
        estimates[name] = estimate
    parameters = {
        "diffusivity": float(
            np.median([row["diffusivity_m2_s"] for row in estimates.values()])
        ),
        "cooling_rate": float(
            np.median([row["cooling_rate_1_s"] for row in estimates.values()])
        ),
        "signal_sd": float(
            np.median([row["cooling_excess_q95_K"] for row in estimates.values()])
        ),
    }
    coupling_rows = []
    couplings = []
    for name, item in prepared.items():
        threshold = float(np.quantile(item.truth, 1.0 - args.fraction_saturated))
        background, source_response = build_calibration_components(
            item,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            threshold=threshold,
            source_flux_threshold=args.source_flux_threshold,
        )
        coupling, n_samples = calibrate_source_coupling(
            item, background, source_response
        )
        couplings.append(coupling)
        coupling_rows.append(
            {
                "trajectory": name,
                "fraction_saturated": args.fraction_saturated,
                "source_coupling": coupling,
                "n_source_samples": n_samples,
            }
        )
    parameters["source_coupling"] = (
        float(np.median(couplings))
        if args.source_coupling is None
        else args.source_coupling
    )
    return prepared, estimates, parameters, pd.DataFrame(coupling_rows)


def make_current_config(
    args: argparse.Namespace,
    prepared,
    *,
    parameters: dict[str, float],
    mean_field: np.ndarray,
) -> RBFConfig:
    return RBFConfig(
        mean_temp=prepared.ambient,
        mean_function=grid_mean_function(prepared, mean_field),
        signal_sd=parameters["signal_sd"] * args.signal_multiplier,
        lengthscale=prepared.source_lengthscale * args.length_multiplier,
        noise_sd=args.noise_sd * args.noise_multiplier,
        relative_jitter=1e-7,
    )


def assert_paired_design(
    observations: dict[str, object],
    ordinary_config: RBFConfig,
    advective_config: RBFConfig,
) -> None:
    observation_times = np.unique(
        np.asarray(observations["x_obs"], dtype=float)[:, 2]
    )
    prediction_times = np.unique(
        np.asarray(observations["x_pred"], dtype=float)[:, 2]
    )
    if len(observation_times) != 1 or len(prediction_times) != 1:
        raise AssertionError("Current update contains more than one time frame")
    if not np.isclose(observation_times[0], prediction_times[0]):
        raise AssertionError("Current observations and predictions use different times")
    for attribute in (
        "signal_sd",
        "lengthscale",
        "noise_sd",
        "relative_jitter",
    ):
        if getattr(ordinary_config, attribute) != getattr(advective_config, attribute):
            raise AssertionError(f"RBF configuration differs in {attribute}")
    probe = np.asarray(observations["x_pred"])[:: max(len(observations["x_pred"]) // 20, 1)]
    if not np.array_equal(
        rbf_covariance(probe, probe, ordinary_config),
        rbf_covariance(probe, probe, advective_config),
    ):
        raise AssertionError("Paired models do not use the same RBF covariance")


def prior_mean_metrics(prepared, field: np.ndarray) -> dict[str, float]:
    truth = prepared.truth
    excess = truth - prepared.ambient
    return {
        "prior_mean_field_rel_l2": float(
            np.linalg.norm(field - truth) / np.linalg.norm(excess)
        ),
        "prior_mean_peak_absolute_error_K": abs(
            float(np.max(field) - np.max(truth))
        ),
    }


def plot_reconstruction(
    prepared,
    *,
    current: dict[str, object],
    previous_posterior_mean: np.ndarray,
    ordinary_mean: np.ndarray,
    advective_mean: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, ...]],
    displacement: np.ndarray,
    out_path: Path,
) -> None:
    extent = [
        prepared.xs[0] * 1e3,
        prepared.xs[-1] * 1e3,
        prepared.ys[0] * 1e3,
        prepared.ys[-1] * 1e3,
    ]
    fields = [
        ("True current field", prepared.truth, "inferno", None, None),
        ("Censored current observation", current["clipped_full"], "inferno", None, None),
        ("Previous censored posterior mean", previous_posterior_mean, "inferno", None, None),
        ("Posterior physics mean", ordinary_mean, "inferno", None, None),
        ("Advective posterior physics mean", advective_mean, "inferno", None, None),
        (
            "Posterior physics mean + RBF",
            predictions["posterior physics mean + RBF"][0].reshape(prepared.truth.shape),
            "inferno",
            None,
            None,
        ),
        (
            "Advective posterior physics mean + RBF",
            predictions["advective posterior physics mean + RBF"][0].reshape(prepared.truth.shape),
            "inferno",
            None,
            None,
        ),
        (
            "Mean difference: advective - ordinary",
            advective_mean - ordinary_mean,
            "coolwarm",
            None,
            None,
        ),
    ]
    vmin = prepared.ambient
    vmax = float(np.max(prepared.truth))
    difference_limit = float(np.max(np.abs(advective_mean - ordinary_mean)))
    figure, axes = plt.subplots(2, 4, figsize=(17.0, 7.6), constrained_layout=True)
    thermal_image = None
    difference_image = None
    for axis, (title, field, cmap, _, _) in zip(axes.ravel(), fields):
        if cmap == "inferno":
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            thermal_image = image
        else:
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=-difference_limit,
                vmax=difference_limit,
                aspect="auto",
            )
            difference_image = image
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    if thermal_image is not None:
        figure.colorbar(
            thermal_image,
            ax=axes[:, :3],
            shrink=0.82,
            label="Temperature (K)",
        )
    if difference_image is not None:
        figure.colorbar(
            difference_image,
            ax=axes[:, 3],
            shrink=0.82,
            label="Temperature difference (K)",
        )
    figure.suptitle(
        f"{prepared.name}: mean-only advection, displacement "
        f"({displacement[0] * 1e3:.3f}, {displacement[1] * 1e3:.3f}) mm"
    )
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


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
    rows = []
    checks = []
    for trajectory_index, record in enumerate(records):
        prepared, _ = prepare_trajectory(
            args.dataset_dir,
            record.name,
            nx=args.nx,
            ny=args.ny,
            heat_flux_cutoff=args.heat_flux_cutoff,
        )
        current_index = len(prepared.times) - 1
        previous_index = current_index - args.previous_frame_offset
        threshold = float(np.quantile(prepared.truth, 1.0 - args.fraction_saturated))
        camera = paired_camera_observations(
            prepared,
            previous_index=previous_index,
            current_index=current_index,
            threshold=threshold,
            observation_stride=args.observation_stride,
            noise_sd=args.noise_sd,
            seed=args.seed + 10_000 * trajectory_index,
        )
        previous_mean, previous_draws, previous_diagnostics = (
            infer_previous_censored_posterior(
                prepared,
                frame=camera["frames"][0],
                fixed_mask=camera["fixed_observation_mask"],
                threshold=threshold,
                signal_sd=parameters["signal_sd"] * args.signal_multiplier,
                lengthscale=prepared.source_lengthscale * args.length_multiplier,
                noise_sd=args.noise_sd * args.previous_noise_multiplier,
                n_chains=args.previous_chains,
                samples_per_chain=args.previous_samples,
                burn_in=args.previous_burn_in,
                thin=args.thin,
                seed=args.seed + 50_000 * trajectory_index,
            )
        )
        ordinary_mean, advective_mean, displacement = posterior_physics_means(
            prepared,
            previous_mean,
            previous_index=previous_index,
            current_index=current_index,
            diffusivity=parameters["diffusivity"],
            cooling_rate=parameters["cooling_rate"],
            source_coupling=parameters["source_coupling"],
            source_flux_threshold=args.source_flux_threshold,
        )
        configs = {
            "posterior physics mean + RBF": make_current_config(
                args, prepared, parameters=parameters, mean_field=ordinary_mean
            ),
            "advective posterior physics mean + RBF": make_current_config(
                args, prepared, parameters=parameters, mean_field=advective_mean
            ),
        }
        assert_paired_design(camera["current"], *configs.values())
        prediction_seed = args.seed + 100_000 * trajectory_index
        predictions = {}
        role = (
            "calibration"
            if record.name in DEVELOPMENT_TRAJECTORIES
            else "evaluation"
        )
        mean_fields = {
            "posterior physics mean + RBF": ordinary_mean,
            "advective posterior physics mean + RBF": advective_mean,
        }
        for method in METHODS:
            prediction = sample_multiple_chains(
                camera["current"],
                configs[method],
                n_chains=args.chains,
                samples_per_chain=args.samples,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=prediction_seed,
            )
            predictions[method] = prediction
            row = prediction_metrics(
                prepared,
                method=method,
                observations=camera["current"],
                prediction=prediction,
            )
            row.update(prior_mean_metrics(prepared, mean_fields[method]))
            row.update(
                {
                    "family": record.family,
                    "run_index": record.run_index,
                    "role": role,
                    "fraction_saturated": args.fraction_saturated,
                    "diffusivity_m2_s": parameters["diffusivity"],
                    "cooling_rate_1_s": parameters["cooling_rate"],
                    "source_coupling": parameters["source_coupling"],
                    "signal_sd_K": configs[method].signal_sd,
                    "lengthscale_m": configs[method].lengthscale,
                    "noise_sd_K": configs[method].noise_sd,
                    "previous_posterior_seed": args.seed + 50_000 * trajectory_index,
                    "current_observation_seed": args.seed + 10_000 * trajectory_index,
                    "current_sampler_seed": prediction_seed,
                    "previous_observations_reused_in_current_update": False,
                    "same_previous_posterior_for_pair": True,
                    "same_current_observations_for_pair": True,
                    "same_rbf_covariance_for_pair": True,
                    "displacement_x_m": displacement[0],
                    "displacement_y_m": displacement[1],
                    "displacement_m": float(np.linalg.norm(displacement)),
                    "crps_estimator": "unbiased_M_times_M_minus_1",
                    **previous_diagnostics,
                }
            )
            rows.append(row)
            print(
                f"[{trajectory_index + 1:02d}/{len(records):02d}] "
                f"{record.name}, {method}: field={row['excess_field_rel_l2']:.3f}, "
                f"all={row['mean_crps_K']:.3f}, "
                f"top1={row['fixed_top_01_crps_K']:.3f}",
                flush=True,
            )
        checks.append(
            {
                "trajectory": record.name,
                "role": role,
                "previous_draws": len(previous_draws),
                "current_observation_count": len(camera["current"]["y_obs"]),
                "current_observation_time_count": len(
                    np.unique(np.asarray(camera["current"]["x_obs"])[:, 2])
                ),
                "same_previous_posterior_for_pair": True,
                "same_current_observations_for_pair": True,
                "same_sampler_seed_for_pair": True,
                "same_rbf_covariance_for_pair": True,
                "previous_observations_reused_in_current_update": False,
            }
        )
        if record.name in DEVELOPMENT_TRAJECTORIES:
            plot_reconstruction(
                prepared,
                current=camera["current"],
                previous_posterior_mean=previous_mean,
                ordinary_mean=ordinary_mean,
                advective_mean=advective_mean,
                predictions=predictions,
                displacement=displacement,
                out_path=args.output_dir
                / f"reconstruction_{record.name.lower()}.png",
            )
        pd.DataFrame(rows).to_csv(args.output_dir / "checkpoint.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def aggregate(results: pd.DataFrame, *, by_family: bool) -> pd.DataFrame:
    heldout = results[results["role"] == "evaluation"]
    groups = ["method"]
    if by_family:
        groups.insert(0, "family")
    summary = heldout.groupby(groups, sort=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_comparison(results: pd.DataFrame) -> pd.DataFrame:
    heldout = results[results["role"] == "evaluation"]
    indexed = heldout.set_index(["trajectory", "method"])
    rows = []
    for family, group in [("all", heldout), *heldout.groupby("family")]:
        names = sorted(group["trajectory"].unique())
        row: dict[str, object] = {
            "family": family,
            "method": "advective posterior physics mean + RBF",
            "baseline": "posterior physics mean + RBF",
            "n_trajectories": len(names),
        }
        for metric in METRICS:
            changes = np.asarray(
                [
                    indexed.loc[
                        (name, "advective posterior physics mean + RBF"), metric
                    ]
                    - indexed.loc[(name, "posterior physics mean + RBF"), metric]
                    for name in names
                ]
            )
            higher_is_better = metric == "hot_region_95_coverage"
            row[f"{metric}_mean_change"] = float(np.mean(changes))
            row[f"{metric}_median_change"] = float(np.median(changes))
            row[f"{metric}_win_count"] = int(
                np.sum(changes > 0.0) if higher_is_better else np.sum(changes < 0.0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    indexed = summary.set_index("method")
    panels = [
        ("excess_field_rel_l2", "Relative excess-field L2 error"),
        ("mean_crps_K", "All-domain CRPS (K)"),
        ("fixed_top_01_crps_K", "Fixed top-1% CRPS (K)"),
        ("peak_absolute_error_K", "Peak absolute error (K)"),
        ("hot_region_95_coverage", "Hot-region 95% coverage"),
        ("hot_region_95_interval_width_K", "Hot-region interval width (K)"),
    ]
    labels = ["posterior physics\nmean + RBF", "advective posterior\nphysics mean + RBF"]
    x = np.arange(2)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        means = np.asarray([indexed.loc[method, f"{metric}_mean"] for method in METHODS])
        standard_errors = np.asarray(
            [
                indexed.loc[method, f"{metric}_std"]
                / np.sqrt(indexed.loc[method, f"{metric}_count"])
                for method in METHODS
            ]
        )
        axis.errorbar(
            x,
            means,
            yerr=np.nan_to_num(standard_errors),
            color="#333333",
            capsize=4,
            linestyle="none",
        )
        for index, method in enumerate(METHODS):
            axis.scatter(index, means[index], s=65, color=METHOD_COLORS[method], zorder=3)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 1].axhline(0.95, color="#666666", linestyle="--", linewidth=1)
    figure.suptitle("Mean-only advection ablation with an identical RBF residual")
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def retained_result_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_readme(
    args: argparse.Namespace,
    *,
    parameters: dict[str, float],
    overall: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    inequality = retained_result_rows(PROPAGATION_OUTPUT / "heldout30_overall.csv")
    kernel = retained_result_rows(KERNEL_OUTPUT / "heldout30_overall.csv")
    table = overall.set_index("method")
    pair = paired[paired["family"] == "all"].iloc[0]
    lines = [
        "# Posterior-physics-mean advection experiment",
        "",
        "This is a controlled mean-only ablation. Both models use the same previous-frame censored posterior, current observations, RBF covariance, censored likelihood, hyperparameters, sampler settings, and random seeds. The only change is whether the diffused previous posterior mean is translated by the source-centroid displacement before adding the current heat source.",
        "",
        "```text",
        "ordinary:  m_n - T_amb = exp(-beta dt) G_(alpha dt) * (mu_(n-1)-T_amb) + gamma q_n dt",
        "advective: m_n - T_amb = exp(-beta dt) [G_(alpha dt) * (mu_(n-1)-T_amb)](x-d_n) + gamma q_n dt",
        "```",
        "",
        "Positive displacement translates the previous thermal field in the positive coordinate direction: the implementation returns `f(x-d_n)`. The current source term is not translated.",
        "",
        "## 1. Inequality / mean result",
        "",
    ]
    if len(inequality):
        indexed = inequality.set_index("method")
        clipped = indexed.loc["clipped propagation"]
        posterior = indexed.loc["posterior-mean propagation"]
        full = indexed.loc["full-posterior propagation"]
        lines.extend(
            [
                "The retained RBF-residual ablation compares observed-clipped physics mean, posterior physics mean, and full-posterior propagation. Posterior physics propagation improves the held-out field error and hidden-tail CRPS without reusing the previous frame in the current likelihood.",
                "",
                "| Retained model | Field error | All CRPS | Top-1% CRPS | Hot coverage |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| observed-clipped physics mean + RBF | {clipped.excess_field_rel_l2_mean:.3f} | {clipped.mean_crps_K_mean:.3f} | {clipped.fixed_top_01_crps_K_mean:.3f} | {clipped.hot_region_95_coverage_mean:.3f} |",
                f"| posterior physics mean + RBF | {posterior.excess_field_rel_l2_mean:.3f} | {posterior.mean_crps_K_mean:.3f} | {posterior.fixed_top_01_crps_K_mean:.3f} | {posterior.hot_region_95_coverage_mean:.3f} |",
                f"| full-posterior propagation + RBF | {full.excess_field_rel_l2_mean:.3f} | {full.mean_crps_K_mean:.3f} | {full.fixed_top_01_crps_K_mean:.3f} | {full.hot_region_95_coverage_mean:.3f} |",
                "",
            ]
        )
    lines.extend(["## 2. Kernel result", ""])
    if len(kernel):
        indexed = kernel.set_index("method")
        rbf = indexed.loc["physics mean + RBF"]
        stochastic = indexed.loc["physics mean + stochastic space-time"]
        lines.extend(
            [
                "The retained stochastic heat-process covariance is a separate supporting kernel ablation. With its clipped physics mean fixed, it improves probabilistic scoring but is not required by the main posterior-physics-mean architecture.",
                "",
                f"At 3% censoring, RBF gives all-domain CRPS `{rbf.mean_crps_K_mean:.3f} K` and stochastic space-time gives `{stochastic.mean_crps_K_mean:.3f} K`; their field errors are `{rbf.excess_field_rel_l2_mean:.3f}` and `{stochastic.excess_field_rel_l2_mean:.3f}` respectively.",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Advection result",
            "",
            "| Model | Field error | All CRPS | Top-1% CRPS | Peak error | Hot coverage | Hot width |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row.excess_field_rel_l2_mean:.3f} | "
            f"{row.mean_crps_K_mean:.3f} | {row.fixed_top_01_crps_K_mean:.3f} | "
            f"{row.peak_absolute_error_K_mean:.3f} | "
            f"{row.hot_region_95_coverage_mean:.3f} | "
            f"{row.hot_region_95_interval_width_K_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Across 30 held-out trajectories, advection changes field error by `{pair.excess_field_rel_l2_mean_change:+.4f}` and wins on `{int(pair.excess_field_rel_l2_win_count)}/30`; changes all-domain CRPS by `{pair.mean_crps_K_mean_change:+.4f} K` and wins on `{int(pair.mean_crps_K_win_count)}/30`; changes top-1% CRPS by `{pair.fixed_top_01_crps_K_mean_change:+.4f} K` and wins on `{int(pair.fixed_top_01_crps_K_win_count)}/30`.",
            "",
            "Family-specific and median paired changes are in `paired_comparisons.csv`. The three reconstruction figures expose the deterministic mean difference directly.",
            "",
            "## Controlled design",
            "",
            "- `Y_(n-1)` is used only to infer the previous censored posterior.",
            "- The final likelihood contains only the shared current observation set `Y_n`.",
            "- Both rows use the same ambient-mean previous RBF posterior and the same current RBF covariance.",
            "- HeatFluxZ supplies the simulated source centroid and current source term. Deployment assumes commanded laser path and power are available.",
            "- All CRPS values use the unbiased `M(M-1)` estimator.",
            "",
            f"alpha = `{parameters['diffusivity']:.6e} m^2/s`; beta = `{parameters['cooling_rate']:.6f} 1/s`; gamma = `{parameters['source_coupling']:.6f}`.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare posterior and advective posterior physics means with a fixed RBF residual."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-names", nargs="+")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--previous-frame-offset", type=int, default=1)
    parser.add_argument("--fraction-saturated", type=float, default=0.03)
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
    prepared, estimates, parameters, coupling_rows = prepare_development(args)
    results, checks = run_experiment(args, parameters=parameters)
    results.to_csv(args.output_dir / "results.csv", index=False)
    checks.to_csv(args.output_dir / "integrity_checks.csv", index=False)
    overall = aggregate(results, by_family=False)
    family = aggregate(results, by_family=True)
    paired = paired_comparison(results)
    overall.to_csv(args.output_dir / "heldout30_overall.csv", index=False)
    family.to_csv(args.output_dir / "family_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(estimates.values()).to_csv(
        args.output_dir / "development_physics_parameters.csv", index=False
    )
    coupling_rows.to_csv(
        args.output_dir / "source_coupling_calibration.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **parameters,
                "fraction_saturated": args.fraction_saturated,
                "observation_stride": args.observation_stride,
                "measurement_noise_sd_K": args.noise_sd,
                "previous_noise_sd_K": args.noise_sd * args.previous_noise_multiplier,
                "current_rbf_noise_sd_K": args.noise_sd * args.noise_multiplier,
                "length_multiplier": args.length_multiplier,
                "previous_samples": args.previous_samples * args.previous_chains,
                "current_samples": args.samples * args.chains,
                "seed": args.seed,
                "crps_estimator": "unbiased_M_times_M_minus_1",
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    plot_comparison(overall, args.output_dir / "comparison.png")
    write_readme(
        args,
        parameters=parameters,
        overall=overall,
        paired=paired,
    )
    print(f"Saved posterior-physics advection experiment to {args.output_dir}")


if __name__ == "__main__":
    main()
