from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src').is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.ceiling_sweep import artificial_ceiling_schedule, lower_camera_ceiling
from src.nested_ceiling_curves import (
    SINGLE_CURVE_BASELINE_FEATURES,
    SMOOTH_RESPONSE_FEATURES,
    build_curve_examples,
)
from src.nested_ceiling_response import (
    fit_grouped_ridge_model,
    grouped_ridge_transfer_predictions,
    prediction_metrics,
)
from src.pseudo_censoring_calibration import FEATURE_NAMES
from src.pseudo_censoring_runner import run_base_prediction
from src.thermal_posterior_physics import (
    DEVELOPMENT_TRAJECTORIES,
    paired_camera_observations,
    prepare_trajectory,
)
from src.thermal_trajectory import trajectory_catalog

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_DIR = ROOT.parent / "heat_eq_laser_trajectories"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "archive" / "2026-09-20_deferred" / "39b_nested_ceiling_response"
)
FIXED_CONFIG = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "38_pseudo_censoring_calibration"
    / "fixed_configuration.csv"
)
REFERENCE_RESULTS = (
    ROOT
    / "outputs"
    / "by_experiment"
    / "30_final_architecture_comparison"
    / "sequential_rerun_results.csv"
)

RAW = "uncorrected single-threshold posterior"
SINGLE = "single-threshold residual correction"
RESPONSE = "single + smooth-response residual correction"
MODELS = (RAW, SINGLE, RESPONSE)
DEFAULT_FRACTIONS = (0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40)
DEFAULT_ANCHORS = (0.05, 0.075, 0.10, 0.125, 0.15)


def trajectory_schedule(
    prepared,
    *,
    reference_ceiling: float,
    options: dict[str, object],
    trajectory_index: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    current_index = len(prepared.times) - 1
    previous_index = current_index - int(options["previous_frame_offset"])
    reference_camera = paired_camera_observations(
        prepared,
        previous_index=previous_index,
        current_index=current_index,
        threshold=reference_ceiling,
        observation_stride=int(options["observation_stride"]),
        noise_sd=float(options["measurement_noise_sd"]),
        seed=int(options["seed"]) + 10_000 * trajectory_index,
    )
    schedule = artificial_ceiling_schedule(
        reference_camera["current"]["clipped_full"],
        reference_ceiling,
        options["target_fractions"],
    )
    return reference_camera, schedule


def response_worker(payload: dict[str, object]) -> dict[str, object]:
    options = payload["options"]
    fixed = payload["fixed"]
    prepared, _ = prepare_trajectory(
        Path(str(payload["dataset_dir"])),
        str(payload["trajectory"]),
        nx=int(options["nx"]),
        ny=int(options["ny"]),
        heat_flux_cutoff=float(options["heat_flux_cutoff"]),
    )
    trajectory_index = int(payload["trajectory_index"])
    reference_ceiling = float(payload["reference_ceiling"])
    reference_camera, schedule = trajectory_schedule(
        prepared,
        reference_ceiling=reference_ceiling,
        options=options,
        trajectory_index=trajectory_index,
    )
    reference_current = np.asarray(
        reference_camera["current"]["clipped_full"], dtype=float
    ).ravel()
    truth = np.asarray(prepared.truth, dtype=float).ravel()
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    schedules: list[dict[str, object]] = []
    for level in schedule:
        ceiling = float(level["ceiling_K"])
        pseudo_camera = lower_camera_ceiling(reference_camera, ceiling)
        prediction, features, diagnostics = run_base_prediction(
            prepared,
            pseudo_camera=pseudo_camera,
            ceiling=ceiling,
            fixed=fixed,
            options=options,
            trajectory_index=trajectory_index,
        )
        mean = np.asarray(prediction[0], dtype=float).ravel()
        sd = np.asarray(prediction[1], dtype=float).ravel()
        censored = np.asarray(
            pseudo_camera["current"]["saturated_full"], dtype=bool
        ).ravel()
        identity = {
            "trajectory": prepared.name,
            "family": str(payload["family"]),
            "role": str(payload["role"]),
            "level_index": int(level["level_index"]),
            "target_censored_fraction": float(level["target_censored_fraction"]),
            "ceiling_K": ceiling,
            "reference_ceiling_K": reference_ceiling,
            "actual_censored_fraction": float(np.mean(censored)),
        }
        for pixel_index in np.flatnonzero(censored):
            row = {
                **identity,
                "pixel_index": int(pixel_index),
                "reference_measurement_K": float(reference_current[pixel_index]),
                "latent_truth_K_evaluation_only": float(truth[pixel_index]),
                "posterior_mean_K": float(mean[pixel_index]),
                "posterior_sd_K": float(sd[pixel_index]),
            }
            row.update(
                {
                    name: float(features[pixel_index, feature_index])
                    for feature_index, name in enumerate(FEATURE_NAMES)
                }
            )
            rows.append(row)
        checks.append(
            {
                **identity,
                "same_reference_noise_across_ceilings": True,
                "truth_used_for_features_or_inference": False,
                "current_observation_protocol": (
                    "fixed-stride unsaturated plus every observed-censored pixel"
                ),
                **diagnostics,
            }
        )
        schedules.append(
            {
                **identity,
                "ceiling_source": str(level["ceiling_source"]),
                "reference_observed_fraction_at_or_above_ceiling": float(
                    level["reference_observed_fraction_at_or_above_ceiling"]
                ),
            }
        )
    return {"rows": rows, "checks": checks, "schedule": schedules}


def run_workers(
    payloads: list[dict[str, object]],
    *,
    workers: int,
    label: str,
    checkpoint: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    collected: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
        futures = {
            executor.submit(response_worker, payload): payload for payload in payloads
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            payload = futures[future]
            collected.append(future.result())
            pd.DataFrame(
                [row for output in collected for row in output["rows"]]
            ).to_csv(checkpoint, index=False)
            print(
                f"[{label} {completed:02d}/{len(payloads):02d}] "
                f"{payload['trajectory']}",
                flush=True,
            )
    rows = pd.DataFrame([row for output in collected for row in output["rows"]])
    checks = pd.DataFrame(
        [row for output in collected for row in output["checks"]]
    )
    schedule = pd.DataFrame(
        [row for output in collected for row in output["schedule"]]
    )
    return rows, checks, schedule


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if np.ptp(first_values) <= 1e-12 or np.ptp(second_values) <= 1e-12:
        return np.nan
    result = spearmanr(first_values, second_values)
    return float(result.statistic)


def feature_correlations(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    rows = []
    for feature in (*SINGLE_CURVE_BASELINE_FEATURES, *SMOOTH_RESPONSE_FEATURES):
        rows.append(
            {
                "feature_set": (
                    "single threshold"
                    if feature in SINGLE_CURVE_BASELINE_FEATURES
                    else "smooth nested response"
                ),
                "feature": feature,
                "target": "known measurement residual",
                "split": split,
                "spearman_rho": safe_spearman(
                    frame[feature], frame["measurement_residual_K"]
                ),
                "n_examples": len(frame),
            }
        )
    return pd.DataFrame(rows)


def corrected_metrics(
    frame: pd.DataFrame,
    residual_prediction: np.ndarray,
) -> dict[str, float]:
    estimate = (
        frame["posterior_exceedance_K"].to_numpy(dtype=float)
        + np.asarray(residual_prediction, dtype=float)
    )
    target = frame["measurement_exceedance_K"].to_numpy(dtype=float)
    metrics = prediction_metrics(estimate, target)
    metrics["residual_prediction_rho"] = safe_spearman(
        residual_prediction, frame["measurement_residual_K"]
    )
    return metrics


def development_comparison(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_groups = training["trajectory"].to_numpy()
    evaluation_groups = evaluation["trajectory"].to_numpy()
    target_residual = training["measurement_residual_K"].to_numpy(dtype=float)
    feature_sets = {
        SINGLE: list(SINGLE_CURVE_BASELINE_FEATURES),
        RESPONSE: [*SINGLE_CURVE_BASELINE_FEATURES, *SMOOTH_RESPONSE_FEATURES],
    }
    residual_predictions = {RAW: np.zeros(len(evaluation))}
    selected_penalties = {RAW: "not applicable"}
    prediction_rows = []
    for model, names in feature_sets.items():
        result = grouped_ridge_transfer_predictions(
            training[names].to_numpy(dtype=float),
            target_residual,
            train_groups,
            evaluation[names].to_numpy(dtype=float),
            evaluation_groups,
        )
        residual_predictions[model] = result.prediction
        selected_penalties[model] = "; ".join(
            f"{key}={value:g}"
            for key, value in sorted(result.selected_penalty.items())
        )
    rows = []
    split_masks = {
        "moderate pseudo-exceedance": (
            evaluation["measurement_exceedance_K"].to_numpy(dtype=float) <= cutoff
        ),
        "larger unseen pseudo-exceedance": (
            evaluation["measurement_exceedance_K"].to_numpy(dtype=float) > cutoff
        ),
    }
    for split, mask in split_masks.items():
        frame = evaluation.loc[mask].reset_index(drop=True)
        for model in MODELS:
            rows.append(
                {
                    "split": split,
                    "model": model,
                    **corrected_metrics(frame, residual_predictions[model][mask]),
                    "selected_penalties": selected_penalties[model],
                    "outer_validation": "leave one development trajectory out",
                    "training_regime": "moderate pseudo-exceedance only",
                    "n_examples": len(frame),
                }
            )
    for model in (SINGLE, RESPONSE):
        for index, prediction in enumerate(residual_predictions[model]):
            prediction_rows.append(
                {
                    "trajectory": evaluation_groups[index],
                    "pixel_index": int(evaluation.iloc[index]["pixel_index"]),
                    "anchor_fraction": float(
                        evaluation.iloc[index]["anchor_fraction"]
                    ),
                    "split": (
                        "moderate pseudo-exceedance"
                        if float(
                            evaluation.iloc[index]["measurement_exceedance_K"]
                        )
                        <= cutoff
                        else "larger unseen pseudo-exceedance"
                    ),
                    "model": model,
                    "known_measurement_exceedance_K": float(
                        evaluation.iloc[index]["measurement_exceedance_K"]
                    ),
                    "uncorrected_posterior_exceedance_K": float(
                        evaluation.iloc[index]["posterior_exceedance_K"]
                    ),
                    "true_residual_K": float(
                        evaluation.iloc[index]["measurement_residual_K"]
                    ),
                    "predicted_residual_K": float(prediction),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def trajectory_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, trajectory, model), group in predictions.groupby(
        ["split", "trajectory", "model"], sort=True
    ):
        estimate = (
            group["uncorrected_posterior_exceedance_K"].to_numpy(dtype=float)
            + group["predicted_residual_K"].to_numpy(dtype=float)
        )
        target = group["known_measurement_exceedance_K"].to_numpy(dtype=float)
        rows.append(
            {
                "split": split,
                "trajectory": trajectory,
                "model": model,
                **prediction_metrics(estimate, target),
                "n_examples": len(group),
            }
        )
    return pd.DataFrame(rows)


def evaluation_comparison(
    frame: pd.DataFrame,
    *,
    single_model,
    response_model,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = {
        RAW: np.zeros(len(frame)),
        SINGLE: single_model.predict(
            frame[list(SINGLE_CURVE_BASELINE_FEATURES)].to_numpy(dtype=float)
        ),
        RESPONSE: response_model.predict(
            frame[
                [*SINGLE_CURVE_BASELINE_FEATURES, *SMOOTH_RESPONSE_FEATURES]
            ].to_numpy(dtype=float)
        ),
    }
    rows = []
    predictions = []
    for model, correction in residuals.items():
        rows.append(
            {
                "split": split,
                "model": model,
                **corrected_metrics(frame, correction),
                "n_examples": len(frame),
                "n_trajectories": frame["trajectory"].nunique(),
            }
        )
        estimate = frame["posterior_mean_K"].to_numpy(dtype=float) + correction
        for index, value in enumerate(estimate):
            predictions.append(
                {
                    "split": split,
                    "trajectory": frame.iloc[index]["trajectory"],
                    "family": frame.iloc[index]["family"],
                    "pixel_index": int(frame.iloc[index]["pixel_index"]),
                    "anchor_fraction": float(frame.iloc[index]["anchor_fraction"]),
                    "model": model,
                    "known_reference_measurement_K": float(
                        frame.iloc[index]["reference_measurement_K"]
                    ),
                    "predicted_measurement_K": float(value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def gate(comparison: pd.DataFrame) -> tuple[bool, float, float]:
    indexed = comparison.set_index("model")
    baseline = indexed.loc[SINGLE]
    augmented = indexed.loc[RESPONSE]
    rmse_reduction = 1.0 - augmented["rmse_K"] / baseline["rmse_K"]
    rho_gain = augmented["spearman_rho"] - baseline["spearman_rho"]
    passed = bool(
        (rmse_reduction >= 0.05 and rho_gain >= 0.0)
        or (rho_gain >= 0.10 and rmse_reduction >= 0.0)
    )
    return passed, float(rmse_reduction), float(rho_gain)


def plot_development(
    response_rows: pd.DataFrame,
    examples: pd.DataFrame,
    correlations: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    colors = ("#0072B2", "#E69F00", "#009E73")
    for color, (trajectory, trajectory_examples) in zip(
        colors, examples.groupby("trajectory", sort=True)
    ):
        selected = trajectory_examples.sort_values("measurement_exceedance_K").iloc[-1]
        curve = response_rows[
            (response_rows["trajectory"] == trajectory)
            & (response_rows["pixel_index"] == int(selected["pixel_index"]))
            & (response_rows["level_index"] >= int(selected["anchor_level_index"]))
        ].sort_values("ceiling_K")
        x = curve["reference_ceiling_K"] - curve["ceiling_K"]
        axes[0, 0].scatter(
            x,
            curve["posterior_mean_K"],
            s=26,
            color=color,
            alpha=0.55,
            label=trajectory,
        )
        coefficients = np.polyfit(x, curve["posterior_mean_K"], deg=2)
        smooth_x = np.linspace(float(x.min()), float(x.max()), 100)
        axes[0, 0].plot(
            smooth_x,
            np.polyval(coefficients, smooth_x),
            color=color,
            linewidth=2.0,
        )
        axes[0, 0].axhline(
            float(selected["reference_measurement_K"]),
            color=color,
            alpha=0.35,
            linestyle=":",
        )
    axes[0, 0].set(
        xlabel="Artificial ceiling drop from reference (K)",
        ylabel="Posterior mean (K)",
        title="Representative smooth nested responses",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    all_correlations = correlations[correlations["split"] == "all development"]
    selected_correlations = all_correlations.reindex(
        all_correlations["spearman_rho"].abs().sort_values(ascending=False).index
    ).head(8)
    axes[0, 1].barh(
        selected_correlations["feature"],
        selected_correlations["spearman_rho"],
        color=[
            "#0072B2" if value == "smooth nested response" else "#777777"
            for value in selected_correlations["feature_set"]
        ],
    )
    axes[0, 1].set(
        xlabel="Spearman correlation with posterior residual",
        title="Strongest development features",
    )
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)

    shown = comparison[
        comparison["split"] == "larger unseen pseudo-exceedance"
    ]
    x = np.arange(len(shown))
    axes[1, 0].bar(
        x,
        shown["rmse_K"],
        color=("#777777", "#E69F00", "#0072B2"),
    )
    axes[1, 0].set_xticks(x, ("Raw", "Single", "Single + response"))
    axes[1, 0].set(
        ylabel="LOTO-CV exceedance RMSE (K)",
        title="Larger unseen pseudo-exceedances",
    )
    axes[1, 1].bar(
        x,
        shown["spearman_rho"],
        color=("#777777", "#E69F00", "#0072B2"),
    )
    axes[1, 1].set_xticks(x, ("Raw", "Single", "Single + response"))
    axes[1, 1].set(
        ylabel="LOTO-CV Spearman correlation",
        title="Larger unseen pseudo-exceedances",
    )
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Experiment 39b: smooth nested-ceiling response", fontsize=15)
    figure.savefig(path, dpi=210)
    plt.close(figure)


def write_readme(
    path: Path,
    *,
    development: pd.DataFrame,
    gate_passed: bool,
    rmse_reduction: float,
    rho_gain: float,
    cutoff: float,
    heldout: pd.DataFrame | None,
    hard_gate: tuple[bool, float, float] | None,
    n_curve_examples: int,
    exceedance_range: tuple[float, float],
    larger_rmse_wins: int,
    larger_rho_wins: int,
) -> None:
    lines = [
        "# Experiment 39b: smooth nested pseudo-ceiling response",
        "",
        "## Question",
        "",
        "Does a smooth posterior response across many nested artificial ceilings predict the part of a known pseudo-hidden measurement that the single-threshold posterior misses?",
        "",
        "## Controlled protocol",
        "",
        "- Reuses the audited Experiment 38 posterior-physics-mean plus sequential stationary stochastic-heat GP, observation protocol, HMC convergence checks, and common reference noise realization.",
        "- Uses ten nested ceilings from the 3% reference through 40% censoring and five anchor ceilings from 5% through 15%.",
        "- Fits quadratic response curves using at least five ceiling levels. Features summarize anchor slope, curvature, total response, and integrated response for posterior mean and SD.",
        "- The learned target is the known residual `reference measurement - single-threshold posterior mean`. Latent truth is not used for fitting or feature construction.",
        "- Development comparison uses leave-one-trajectory-out validation. Multiple anchors from a pixel always stay in the same trajectory fold.",
        f"- The five anchors produce `{n_curve_examples}` known pseudo-hidden examples spanning `{exceedance_range[0]:.3f}` to `{exceedance_range[1]:.3f} K`.",
        f"- Moderate training examples are defined using the development-only 70th percentile, `{cutoff:.3f} K`; larger exceedances are reserved for extrapolation.",
        "- A real-case equality audit against the original Experiment 38 driver gave maximum absolute difference `0.0` for posterior draws, summaries, intervals, and features at the 10% ceiling; see `implementation_consistency_audit.csv`.",
        "",
        "## Development result",
        "",
        "| Evaluation split | Model | RMSE (K) | MAE (K) | Spearman rho | Residual rho |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in development.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.model} | {row.rmse_K:.3f} | "
            f"{row.mae_K:.3f} | {row.spearman_rho:.3f} | "
            f"{row.residual_prediction_rho:.3f} |"
        )
    lines.extend(
        [
            "",
            f"On the larger unseen pseudo-exceedances, the unchanged signal gate is `{'PASS' if gate_passed else 'FAIL'}`: smooth-response versus single-feature RMSE change is `{100.0 * rmse_reduction:.1f}%` and Spearman change is `{rho_gain:.3f}`.",
            f"The response model has lower RMSE on `{larger_rmse_wins}/3` development trajectories, but higher Spearman correlation on only `{larger_rho_wins}/3`; the per-trajectory effect sizes are in `development_trajectory_comparison.csv`.",
            "",
        ]
    )
    if not gate_passed:
        lines.extend(
            [
                "## Decision",
                "",
                "The stronger development diagnostic still finds no material incremental out-of-trajectory signal from the nested response. The held-out trajectories and genuine hardware-saturated tail were therefore not evaluated. This closes the nested-response branch under the predeclared one-modification rule; lowering the ceiling remains useful as a diagnostic and calibration-data generator, but its response curve should not be claimed to reveal hidden magnitude beyond the single-threshold posterior.",
            ]
        )
    else:
        assert heldout is not None and hard_gate is not None
        lines.extend(
            [
                "## Held-out transfer",
                "",
                "| Split | Model | RMSE (K) | MAE (K) | Spearman rho |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in heldout.itertuples(index=False):
            lines.append(
                f"| {row.split} | {row.model} | {row.rmse_K:.3f} | "
                f"{row.mae_K:.3f} | {row.spearman_rho:.3f} |"
            )
        lines.extend(
            [
                "",
                f"The larger-exceedance transfer gate is `{'PASS' if hard_gate[0] else 'FAIL'}`: RMSE change `{100.0 * hard_gate[1]:.1f}%`, Spearman change `{hard_gate[2]:.3f}`.",
                "",
                "The genuine hardware-saturated evaluation is included only when this hard extrapolation gate passes.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_row = pd.read_csv(FIXED_CONFIG).iloc[0]
    fixed = {
        key: float(config_row[key])
        for key in ("diffusivity", "cooling_rate", "signal_sd", "source_coupling")
    }
    options = {
        "nx": int(config_row["nx"]),
        "ny": int(config_row["ny"]),
        "target_fractions": tuple(args.target_fractions),
        "observation_stride": int(config_row["observation_stride"]),
        "previous_frame_offset": int(config_row["previous_frame_offset"]),
        "measurement_noise_sd": float(config_row["measurement_noise_sd"]),
        "previous_noise_sd": float(config_row["previous_noise_sd"]),
        "current_noise_sd": float(config_row["current_noise_sd"]),
        "length_multiplier": float(config_row["length_multiplier"]),
        "forcing_length_multiplier": float(config_row["forcing_length_multiplier"]),
        "quadrature_order": int(config_row["quadrature_order"]),
        "heat_flux_cutoff": float(config_row["heat_flux_cutoff"]),
        "source_flux_threshold": float(config_row["source_flux_threshold"]),
        "previous_chains": int(config_row["previous_chains"]),
        "previous_samples_per_chain": int(config_row["previous_samples_per_chain"]),
        "previous_burn_in": int(config_row["previous_burn_in"]),
        "previous_thin": int(config_row["previous_thin"]),
        "previous_retry_samples_per_chain": int(
            config_row["previous_retry_samples_per_chain"]
        ),
        "previous_retry_burn_in": int(config_row["previous_retry_burn_in"]),
        "previous_retry_thin": int(config_row["previous_retry_thin"]),
        "current_chains": int(config_row["current_chains"]),
        "current_samples_per_chain": int(config_row["current_samples_per_chain"]),
        "current_burn_in": int(config_row["current_burn_in"]),
        "current_thin": int(config_row["current_thin"]),
        "current_retry_samples_per_chain": int(
            config_row["current_retry_samples_per_chain"]
        ),
        "current_retry_burn_in": int(config_row["current_retry_burn_in"]),
        "current_retry_thin": int(config_row["current_retry_thin"]),
        "evaluation_draws_per_chain": int(config_row["evaluation_draws_per_chain"]),
        "maximum_rhat": float(config_row["maximum_rhat"]),
        "truncated_sampler": str(config_row["truncated_sampler"]),
        "seed": int(config_row["seed"]),
    }
    catalog = trajectory_catalog(args.dataset_dir)
    if len(catalog) != 33:
        raise AssertionError(f"Expected 33 trajectories, found {len(catalog)}")
    if len(DEVELOPMENT_TRAJECTORIES) != 3:
        raise AssertionError("Expected exactly three development trajectories")
    if sum(item.name not in DEVELOPMENT_TRAJECTORIES for item in catalog) != 30:
        raise AssertionError("Expected exactly 30 held-out trajectories")
    reference = pd.read_csv(REFERENCE_RESULTS)
    thresholds = reference.groupby("trajectory")["threshold_K"].first().to_dict()
    if set(thresholds) != {item.name for item in catalog}:
        raise AssertionError("Reference ceilings do not match all 33 trajectories")

    common_payloads = [
        {
            "dataset_dir": str(args.dataset_dir),
            "trajectory": item.name,
            "family": item.family,
            "trajectory_index": index,
            "reference_ceiling": thresholds[item.name],
            "fixed": fixed,
            "options": options,
            "role": (
                "development"
                if item.name in DEVELOPMENT_TRAJECTORIES
                else "heldout"
            ),
        }
        for index, item in enumerate(catalog)
    ]
    development_payloads = [
        payload for payload in common_payloads if payload["role"] == "development"
    ]
    if args.reuse_development:
        development_rows = pd.read_csv(
            args.output_dir / "development_response_rows.csv"
        )
        development_checks = pd.read_csv(
            args.output_dir / "development_implementation_checks.csv"
        )
        development_schedule = pd.read_csv(
            args.output_dir / "development_threshold_schedule.csv"
        )
    else:
        development_rows, development_checks, development_schedule = run_workers(
            development_payloads,
            workers=min(args.workers, 3),
            label="development",
            checkpoint=args.output_dir / "development_response_checkpoint.csv",
        )
    development_rows = development_rows.sort_values(
        ["trajectory", "pixel_index", "level_index"]
    )
    development_checks = development_checks.sort_values(
        ["trajectory", "level_index"]
    )
    development_schedule = development_schedule.sort_values(
        ["trajectory", "level_index"]
    )
    if len(development_checks) != 3 * len(args.target_fractions):
        raise AssertionError("Development ceiling grid is incomplete")
    if development_checks["previous_rhat_max"].max() > options["maximum_rhat"]:
        raise AssertionError("A previous posterior failed convergence")
    if development_checks["current_rhat_max"].max() > options["maximum_rhat"]:
        raise AssertionError("A current posterior failed convergence")
    if development_checks["current_selected_rhat_max"].max() > options["maximum_rhat"]:
        raise AssertionError("A censored current posterior failed convergence")

    development_examples = build_curve_examples(
        development_rows,
        anchor_fractions=tuple(args.anchor_fractions),
    )
    if development_examples.empty:
        raise AssertionError("No development response-curve examples were constructed")
    cutoff = float(
        np.quantile(development_examples["measurement_exceedance_K"], 0.70)
    )
    moderate = development_examples[
        development_examples["measurement_exceedance_K"] <= cutoff
    ].copy()
    if moderate["trajectory"].nunique() != 3:
        raise AssertionError("Moderate development split lost a trajectory")
    larger = development_examples[
        development_examples["measurement_exceedance_K"] > cutoff
    ].copy()
    correlations = pd.concat(
        [
            feature_correlations(
                development_examples, split="all development"
            ),
            feature_correlations(moderate, split="moderate pseudo-exceedance"),
            feature_correlations(
                larger, split="larger unseen pseudo-exceedance"
            ),
        ],
        ignore_index=True,
    )
    comparison, cv_predictions = development_comparison(
        moderate, development_examples, cutoff=cutoff
    )
    trajectory_comparison = trajectory_prediction_summary(cv_predictions)
    gate_rows = comparison[
        comparison["split"] == "larger unseen pseudo-exceedance"
    ]
    gate_passed, rmse_reduction, rho_gain = gate(gate_rows)
    larger_by_trajectory = trajectory_comparison[
        trajectory_comparison["split"] == "larger unseen pseudo-exceedance"
    ].pivot(index="trajectory", columns="model", values=["rmse_K", "spearman_rho"])
    larger_rmse_wins = int(
        np.sum(
            larger_by_trajectory[("rmse_K", RESPONSE)]
            < larger_by_trajectory[("rmse_K", SINGLE)]
        )
    )
    larger_rho_wins = int(
        np.sum(
            larger_by_trajectory[("spearman_rho", RESPONSE)]
            > larger_by_trajectory[("spearman_rho", SINGLE)]
        )
    )

    development_rows.to_csv(
        args.output_dir / "development_response_rows.csv", index=False
    )
    development_examples.to_csv(
        args.output_dir / "development_curve_examples.csv", index=False
    )
    counts = (
        development_examples.groupby(["trajectory", "anchor_fraction"], sort=True)[
            "measurement_exceedance_K"
        ]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_examples",
                "min": "minimum_exceedance_K",
                "median": "median_exceedance_K",
                "max": "maximum_exceedance_K",
            }
        )
    )
    counts.to_csv(args.output_dir / "development_example_counts.csv", index=False)
    development_checks.to_csv(
        args.output_dir / "development_implementation_checks.csv", index=False
    )
    development_schedule.to_csv(
        args.output_dir / "development_threshold_schedule.csv", index=False
    )
    correlations.to_csv(args.output_dir / "feature_correlations.csv", index=False)
    comparison.to_csv(
        args.output_dir / "development_grouped_cv_comparison.csv", index=False
    )
    cv_predictions.to_csv(
        args.output_dir / "development_grouped_cv_predictions.csv", index=False
    )
    trajectory_comparison.to_csv(
        args.output_dir / "development_trajectory_comparison.csv", index=False
    )
    plot_development(
        development_rows,
        moderate,
        correlations,
        comparison,
        args.output_dir / "smooth_response_diagnostic.png",
    )

    heldout_comparison = None
    hard_gate = None
    if gate_passed:
        groups = moderate["trajectory"].to_numpy()
        target = moderate["measurement_residual_K"].to_numpy(dtype=float)
        single_model = fit_grouped_ridge_model(
            moderate[list(SINGLE_CURVE_BASELINE_FEATURES)].to_numpy(dtype=float),
            target,
            groups,
        )
        response_model = fit_grouped_ridge_model(
            moderate[
                [*SINGLE_CURVE_BASELINE_FEATURES, *SMOOTH_RESPONSE_FEATURES]
            ].to_numpy(dtype=float),
            target,
            groups,
        )
        heldout_payloads = [
            payload for payload in common_payloads if payload["role"] == "heldout"
        ]
        heldout_rows, heldout_checks, heldout_schedule = run_workers(
            heldout_payloads,
            workers=args.workers,
            label="heldout",
            checkpoint=args.output_dir / "heldout_response_checkpoint.csv",
        )
        heldout_rows = heldout_rows.sort_values(
            ["trajectory", "pixel_index", "level_index"]
        )
        heldout_examples = build_curve_examples(
            heldout_rows,
            anchor_fractions=tuple(args.anchor_fractions),
        )
        in_range = heldout_examples[
            heldout_examples["measurement_exceedance_K"] <= cutoff
        ].copy()
        extrapolation = heldout_examples[
            heldout_examples["measurement_exceedance_K"] > cutoff
        ].copy()
        comparison_parts = []
        prediction_parts = []
        for split, frame in (
            ("heldout in-range pseudo-exceedance", in_range),
            ("heldout larger pseudo-exceedance", extrapolation),
        ):
            result, predictions = evaluation_comparison(
                frame,
                single_model=single_model,
                response_model=response_model,
                split=split,
            )
            comparison_parts.append(result)
            prediction_parts.append(predictions)
        heldout_comparison = pd.concat(comparison_parts, ignore_index=True)
        heldout_predictions = pd.concat(prediction_parts, ignore_index=True)
        hard_gate = gate(
            heldout_comparison[
                heldout_comparison["split"] == "heldout larger pseudo-exceedance"
            ]
        )
        heldout_rows.to_csv(
            args.output_dir / "heldout_response_rows.csv", index=False
        )
        heldout_examples.to_csv(
            args.output_dir / "heldout_curve_examples.csv", index=False
        )
        heldout_checks.to_csv(
            args.output_dir / "heldout_implementation_checks.csv", index=False
        )
        heldout_schedule.to_csv(
            args.output_dir / "heldout_threshold_schedule.csv", index=False
        )
        heldout_comparison.to_csv(
            args.output_dir / "heldout_transfer_comparison.csv", index=False
        )
        heldout_predictions.to_csv(
            args.output_dir / "heldout_transfer_predictions.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "model": SINGLE,
                    "selected_penalty": single_model.penalty,
                    "n_features": len(SINGLE_CURVE_BASELINE_FEATURES),
                },
                {
                    "model": RESPONSE,
                    "selected_penalty": response_model.penalty,
                    "n_features": len(SINGLE_CURVE_BASELINE_FEATURES)
                    + len(SMOOTH_RESPONSE_FEATURES),
                },
            ]
        ).to_csv(args.output_dir / "frozen_model_parameters.csv", index=False)

        if hard_gate[0]:
            hardware_examples = build_curve_examples(
                heldout_rows,
                anchor_fractions=(float(args.target_fractions[0]),),
                include_reference_saturated=True,
            )
            hardware_examples = hardware_examples[
                hardware_examples["is_reference_saturated"]
            ].copy()
            hardware_target = hardware_examples.copy()
            hardware_target["measurement_exceedance_K"] = hardware_target[
                "truth_exceedance_K_evaluation_only"
            ]
            hardware_target["measurement_residual_K"] = hardware_target[
                "truth_residual_K_evaluation_only"
            ]
            hardware_result, hardware_predictions = evaluation_comparison(
                hardware_target,
                single_model=single_model,
                response_model=response_model,
                split="genuine hardware-saturated latent truth",
            )
            heldout_comparison = pd.concat(
                [heldout_comparison, hardware_result], ignore_index=True
            )
            heldout_comparison.to_csv(
                args.output_dir / "heldout_transfer_comparison.csv", index=False
            )
            hardware_examples.to_csv(
                args.output_dir / "hardware_saturated_curve_examples.csv",
                index=False,
            )
            hardware_predictions.to_csv(
                args.output_dir / "hardware_saturated_predictions.csv", index=False
            )

    pd.DataFrame(
        [
            {
                **fixed,
                **options,
                "anchor_fractions": tuple(args.anchor_fractions),
                "minimum_curve_levels": 5,
                "curve_model": "quadratic in normalized ceiling displacement",
                "training_target": "known reference measurement minus posterior mean",
                "moderate_cutoff_rule": "development-only 70th percentile",
                "moderate_cutoff_K": cutoff,
                "n_development_curve_examples": len(development_examples),
                "n_moderate_development_examples": len(moderate),
                "minimum_known_exceedance_K": float(
                    development_examples["measurement_exceedance_K"].min()
                ),
                "maximum_known_exceedance_K": float(
                    development_examples["measurement_exceedance_K"].max()
                ),
                "development_gate_passed": gate_passed,
                "minimum_rmse_reduction": 0.05,
                "minimum_spearman_gain": 0.10,
                "truth_used_for_features_or_fitting": False,
                "heldout_run_only_after_development_gate": True,
                "hardware_tail_run_only_after_hard_transfer_gate": True,
            }
        ]
    ).to_csv(args.output_dir / "fixed_configuration.csv", index=False)
    write_readme(
        args.output_dir / "README.md",
        development=comparison,
        gate_passed=gate_passed,
        rmse_reduction=rmse_reduction,
        rho_gain=rho_gain,
        cutoff=cutoff,
        heldout=heldout_comparison,
        hard_gate=hard_gate,
        n_curve_examples=len(development_examples),
        exceedance_range=(
            float(development_examples["measurement_exceedance_K"].min()),
            float(development_examples["measurement_exceedance_K"].max()),
        ),
        larger_rmse_wins=larger_rmse_wins,
        larger_rho_wins=larger_rho_wins,
    )
    print(f"Saved Experiment 39b to {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded smooth nested-ceiling response experiment."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
    )
    parser.add_argument(
        "--anchor-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_ANCHORS),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--reuse-development",
        action="store_true",
        help="Reuse saved development response rows and rerun only analysis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
