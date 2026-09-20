from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src').is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ceiling_sweep import (SequentialAdvectiveConfig, lower_camera_ceiling,
                               matched_censoring_observations, sequential_prior_blocks)
from src.metrics import empirical_crps
from src.pseudo_censoring_runner import (_current_prediction, _previous_prediction,
                                         run_base_prediction)
from src.source_history import source_history_mean
from src.thermal_posterior_physics import (DEVELOPMENT_TRAJECTORIES,
    paired_camera_observations, posterior_physics_means, prepare_trajectory)
from src.thermal_trajectory import trajectory_catalog
from src.thermal_plotting import tail_temperature_norm, add_tail_contours

BASE = ROOT / "outputs/by_experiment/38_pseudo_censoring_calibration"
OUT = ROOT / "outputs/archive/2026-09-20_deferred/40_source_history_mean"
METRICS = ["field_excess_rel_l2", "overall_rmse_K", "overall_crps_K",
           "top_1pct_rmse_K", "top_1pct_mae_K", "top_1pct_bias_K",
           "top_1pct_crps_K", "top_1pct_coverage_95", "top_1pct_width_95_K",
           "peak_absolute_error_K"]


def digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.ascontiguousarray(array).tobytes())
    return value.hexdigest()


def score(prediction, truth, ambient, censored, ceiling):
    mean, sd, lower, upper, draws = prediction
    truth = truth.ravel()
    error = mean - truth
    top = np.zeros(len(truth), dtype=bool)
    top[np.argsort(truth, kind="mergesort")[-max(1, int(.01 * len(truth))):]] = True
    crps = empirical_crps(draws, truth)
    result = {
        "field_excess_rel_l2": np.linalg.norm(error) / np.linalg.norm(truth - ambient),
        "peak_absolute_error_K": abs(mean.max() - truth.max()),
        "peak_signed_error_K": mean.max() - truth.max(),
    }
    for name, mask in [("overall", np.ones(len(truth), dtype=bool)),
                       ("top_1pct", top), ("censored", censored.ravel())]:
        result.update({
            f"{name}_rmse_K": np.sqrt(np.mean(error[mask] ** 2)),
            f"{name}_mae_K": np.mean(abs(error[mask])),
            f"{name}_bias_K": np.mean(error[mask]),
            f"{name}_crps_K": np.mean(crps[mask]),
            f"{name}_coverage_95": np.mean((lower[mask] <= truth[mask]) & (truth[mask] <= upper[mask])),
            f"{name}_width_95_K": np.mean(upper[mask] - lower[mask]),
            f"{name}_sd_K": np.mean(sd[mask]),
        })
    selected = censored.ravel()
    result["exceedance_slope"] = np.polyfit(truth[selected] - ceiling, mean[selected] - ceiling, 1)[0]
    return result


def worker(payload):
    name, family, index, ceiling, options, output = payload
    output = Path(output)
    prepared, _ = prepare_trajectory(ROOT.parent / "heat_eq_laser_trajectories", name,
        nx=int(options["nx"]), ny=int(options["ny"]), heat_flux_cutoff=options["heat_flux_cutoff"])
    n = len(prepared.times) - 1
    if n < 8:
        raise ValueError(f"{name}: insufficient history for eight steps")
    fixed = {k: float(options[k]) for k in ("diffusivity", "cooling_rate", "signal_sd", "source_coupling")}
    physical = {k: fixed[k] for k in ("diffusivity", "cooling_rate", "source_coupling")}
    physical["source_flux_threshold"] = options["source_flux_threshold"]
    camera_seed = int(options["seed"]) + 10_000 * index
    prior_seed = int(options["seed"]) + 50_000 * index
    current_seed = int(options["seed"]) + 100_000 * index
    camera = paired_camera_observations(prepared, previous_index=n-1, current_index=n,
        threshold=ceiling, observation_stride=int(options["observation_stride"]),
        noise_sd=options["measurement_noise_sd"], seed=camera_seed)
    camera = lower_camera_ceiling(camera, ceiling)
    observations = matched_censoring_observations(prepared, frame=camera["frames"][1],
        fixed_mask=camera["fixed_observation_mask"], threshold=ceiling)
    lengthscale = prepared.source_lengthscale * options["length_multiplier"]

    def previous(frame, seed):
        if not np.any(frame["saturated_full"]):
            # Preserve the inherited hybrid representation when nothing is censored.
            mean = np.asarray(frame["clipped_full"]).copy()
            return mean, None, {"rhat_max": np.nan, "no_censored_pixels": True}
        return _previous_prediction(prepared, frame=frame,
            fixed_mask=camera["fixed_observation_mask"], threshold=ceiling,
            signal_sd=fixed["signal_sd"], lengthscale=lengthscale,
            noise_sd=options["previous_noise_sd"], options=options, seed=seed)

    previous_mean, previous_draws, previous_checks = previous(camera["frames"][0], prior_seed)
    if previous_draws is None:
        raise ValueError("Reference predecessor has no censored pixels; audit separately")
    ordinary, _, _ = posterior_physics_means(prepared, previous_mean,
        previous_index=n-1, current_index=n, **physical)
    config = SequentialAdvectiveConfig(signal_sd=fixed["signal_sd"],
        forcing_lengthscale=lengthscale * options["forcing_length_multiplier"],
        diffusivity=fixed["diffusivity"], cooling_rate=fixed["cooling_rate"],
        current_noise_sd=options["current_noise_sd"], quadrature_order=int(options["quadrature_order"]))
    blocks, covariance_checks = sequential_prior_blocks(prepared,
        camera={**camera, "current": observations}, previous_mean=previous_mean,
        previous_draws=previous_draws, mean_field=ordinary, displacement=None, config=config)
    covariance_keys = ("observed_covariance", "pred_observed_covariance", "prediction_variance")
    covariance_hash = digest(*(blocks[k] for k in covariance_keys))
    observed_hash = digest(observations["y_obs"], observations["sat_mask"])
    observed_indices = np.flatnonzero((camera["fixed_observation_mask"] | observations["saturated_full"]).ravel())
    rows, checks, audits = [], [], []
    images = {"truth": prepared.truth, "observed": observations["clipped_full"]}
    role = "development" if name in DEVELOPMENT_TRAJECTORIES else "heldout"
    saved = pd.read_csv(BASE / "results.csv")
    reference = saved[(saved.trajectory == name) & (saved.level_index == 0) & (saved.method == "uncalibrated")]
    for steps in (1, 2, 4, 8):
        if steps == 1:
            initial, prior_check, frame = previous_mean, previous_checks, camera["frames"][0]
            initial_seed = prior_seed
        else:
            initial_seed = prior_seed + 1_000_000 + 1000 * steps
            earlier = paired_camera_observations(prepared, previous_index=n-steps, current_index=n,
                threshold=ceiling, observation_stride=int(options["observation_stride"]),
                noise_sd=options["measurement_noise_sd"], seed=camera_seed + 1_000_000 + 1000 * steps)
            frame = earlier["frames"][0]
            initial, _, prior_check = previous(frame, initial_seed)
        mean = source_history_mean(prepared, initial, start_index=n-steps, current_index=n, **physical)
        if steps == 1:
            np.testing.assert_array_equal(mean, ordinary)
        trial_blocks = {**blocks, "prediction_mean": mean.ravel(), "observation_mean": mean.ravel()[observed_indices]}
        assert digest(*(trial_blocks[k] for k in covariance_keys)) == covariance_hash
        prediction, current_checks = _current_prediction(observations, trial_blocks,
            options=options, seed=current_seed)
        identity = {"trajectory": name, "family": family, "role": role, "steps": steps}
        metrics = score(prediction, prepared.truth, prepared.ambient, observations["saturated_full"], ceiling)
        rows.append({**identity, "ceiling_K": ceiling, **metrics})
        top_indices = np.argsort(prepared.truth.ravel(), kind="mergesort")[-25:]
        checks.append({**identity, "current_observation_sha256": observed_hash,
            "covariance_sha256": covariance_hash, "current_sampler_seed": current_seed,
            "initial_sampler_seed": initial_seed, "initial_frame": n-steps,
            "target_frame": n, "history_duration_s": prepared.times[n]-prepared.times[n-steps],
            "initial_censored_fraction": np.mean(frame["saturated_full"]),
            "initial_rhat_max": prior_check["rhat_max"],
            "forecast_top_1pct_rmse_K": np.sqrt(np.mean((mean.ravel()[top_indices]-prepared.truth.ravel()[top_indices])**2)),
            "forecast_top_1pct_bias_K": np.mean(mean.ravel()[top_indices]-prepared.truth.ravel()[top_indices]),
            **covariance_checks, **{f"current_{k}": v for k,v in current_checks.items()}})
        if steps == 1 and len(reference):
            for metric in METRICS:
                old = float(reference.iloc[0][metric])
                delta = float(metrics[metric])-old
                audits.append({"trajectory": name, "metric": metric, "saved_experiment38": old,
                               "reproduced": metrics[metric], "absolute_difference": abs(delta)})
                if abs(delta) > 1e-8:
                    raise AssertionError(f"Baseline mismatch {name} {metric}: {delta}")
        if steps == 1 and name == "DiagonalScanPath_8":
            reproduction, _, _ = run_base_prediction(prepared, pseudo_camera=camera, ceiling=ceiling,
                fixed=fixed, options=options, trajectory_index=index)
            for label, a, b in zip(("mean", "sd", "lower", "upper", "draws"), prediction, reproduction):
                np.testing.assert_array_equal(a, b)
                audits.append({"trajectory": name, "metric": f"runner_{label}_exact", "absolute_difference": 0.0})
        images[f"mean_{steps}"] = mean
        images[f"prediction_{steps}"] = prediction[0].reshape(prepared.truth.shape)
        print(f"{name}: L={steps} complete", flush=True)
    pd.DataFrame(rows).to_csv(output / f"{name}_results.csv", index=False)
    pd.DataFrame(checks).to_csv(output / f"{name}_checks.csv", index=False)
    pd.DataFrame(audits, columns=["trajectory", "metric", "saved_experiment38", "reproduced", "absolute_difference"]).to_csv(output / f"{name}_audit.csv", index=False)
    if role == "development":
        fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
        fields = [images["truth"], *(images[f"mean_{s}"] for s in (1,2,4,8)),
                  images["observed"], *(images[f"prediction_{s}"] for s in (1,2,4,8))]
        labels = ["Truth", *(f"{s}-step forecast" for s in (1,2,4,8)),
                  "Camera observation", *(f"{s}-step reconstruction" for s in (1,2,4,8))]
        for axis, field, label in zip(axes.ravel(), fields, labels):
            im = axis.imshow(field, origin="lower", cmap="inferno",
                norm=tail_temperature_norm(prepared.ambient, ceiling),
                extent=[prepared.xs[0], prepared.xs[-1], prepared.ys[0], prepared.ys[-1]])
            add_tail_contours(axis, prepared.xs, prepared.ys, field, ambient=prepared.ambient, ceiling=ceiling)
            axis.set_title(label, fontsize=10)
            axis.set_axis_off()
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Temperature (K)", shrink=.7, extend="both")
        fig.suptitle(f"{name}: source-history mean; identical one-step residual covariance")
        fig.savefig(output.parent / f"reconstruction_{name}.png", dpi=170)
        plt.close(fig)
    return name


def summarize(output):
    cache = output / "by_trajectory"
    results = pd.concat([pd.read_csv(p) for p in sorted(cache.glob("*_results.csv"))], ignore_index=True)
    checks = pd.concat([pd.read_csv(p) for p in sorted(cache.glob("*_checks.csv"))], ignore_index=True)
    audits = pd.concat([pd.read_csv(p) for p in sorted(cache.glob("*_audit.csv")) if p.stat().st_size > 2], ignore_index=True)
    results.to_csv(output / "results.csv", index=False)
    checks.to_csv(output / "implementation_checks.csv", index=False)
    audits.to_csv(output / "baseline_equality_audit.csv", index=False)
    heldout = results[results.role == "heldout"]
    if heldout.empty:
        return
    metrics = list(results.select_dtypes(include=np.number).columns.difference(["steps", "ceiling_K"]))
    summary = heldout.groupby("steps")[metrics].mean()
    summary.to_csv(output / "heldout30_summary.csv")
    heldout.groupby(["family", "steps"])[metrics].mean().to_csv(output / "family_summary.csv")
    paired = []
    for family, frame in [("All", heldout), *heldout.groupby("family")]:
        baseline = frame[frame.steps == 1].set_index("trajectory")
        for steps in (2,4,8):
            candidate = frame[frame.steps == steps].set_index("trajectory")
            for metric in metrics:
                delta = candidate[metric]-baseline[metric]
                paired.append({"family": family, "steps": steps, "metric": metric,
                    "mean_change": delta.mean(), "median_change": delta.median(),
                    "n_decreased": int((delta < 0).sum()), "n_increased": int((delta > 0).sum()),
                    "n_trajectories": len(delta)})
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    fig, axes = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for axis, metric in zip(axes.ravel(), ["field_excess_rel_l2", "top_1pct_rmse_K", "top_1pct_bias_K",
            "top_1pct_crps_K", "top_1pct_coverage_95", "top_1pct_width_95_K"]):
        axis.plot(summary.index, summary[metric], "o-", color="#0072B2")
        axis.set(title=metric.replace("_", " "), xlabel="History steps", xticks=[1,2,4,8])
        axis.grid(alpha=.2)
    fig.suptitle("30 held-out trajectories: source-history mean diagnostic")
    fig.savefig(output / "comparison.png",dpi=180)
    plt.close(fig)
    assert len(heldout) == 120 and len(results) == 132
    assert checks.groupby("trajectory").covariance_sha256.nunique().max() == 1
    assert checks.groupby("trajectory").current_observation_sha256.nunique().max() == 1
    lines = ["# Source-history mean diagnostic", "",
        "The 1-, 2-, 4-, and 8-step means start at their respective censored camera posteriors and replay each intervening source snapshot using the actual time steps.", "",
        "The residual covariance is the same Experiment 38 one-step Q + propagated previous covariance in every row. This is a mean-only sensitivity experiment, not an L-step Bayesian filter. The n-1 frame still supplies covariance for all variants; only the deterministic mean uses the earlier frame.", "",
        "The initial posterior is the inherited hybrid: sampled censored pixels, observed noisy values at unsaturated pixels. Current likelihood uses only the current frame. Source flux is thresholded at 10000; below-ambient excess is floored before diffusion; the grid boundary is zero excess. These inherited approximations are preserved for equality.", "",
        "All physical coefficients, source threshold, noise scales, observation stride, HMC settings and seeds are frozen from Experiment 38. Its reference ceilings and ambient/source-width preparation are also inherited (simulation-defined reference protocol, not a new deployment calibration). Earlier frames have explicit distinct noise seeds; current and n-1 observations reproduce Experiment 38.", "",
        "## Held-out results", "", "| Steps | Field L2 | Top1 RMSE | Top1 bias | All CRPS | Top1 CRPS | Coverage | Width |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for steps,r in summary.iterrows():
        lines.append(f"| {steps} | {r.field_excess_rel_l2:.3f} | {r.top_1pct_rmse_K:.3f} | {r.top_1pct_bias_K:.3f} | {r.overall_crps_K:.3f} | {r.top_1pct_crps_K:.3f} | {r.top_1pct_coverage_95:.3f} | {r.top_1pct_width_95_K:.3f} |")
    lines += ["", "## Interpretation", "",
        "The one-step baseline remains preferred. Longer rollouts amplify positive hot-tail bias. The paired table reports decreases and increases separately: signed bias, coverage, width and exceedance slope do not have a universal lower-is-better interpretation. In particular, an exceedance slope exceeding one indicates excessive amplitude response, not improved recovery.", "",
        "The source coefficient was inherited from a development fit that propagated a clipped predecessor. It can compensate for the omitted predecessor heat as well as represent actual heat input. Reusing it at every rollout step can accumulate that compensation. This is a plausible explanation supported by calibration provenance, not an isolated causal test of gamma. The simplified diffusion/cooling approximation and initial-state errors can also accumulate.", "",
        "The longest window spans only 0.004 seconds, and the development starting frames remain similarly censored. Thus the experiment supplies neither a clearly unsaturated initial anchor nor a new later cooling observation. It rejects repeated use of the existing one-step effective forecast as an improvement; it does not establish that physically calibrated source-history reconstruction is unhelpful. See BASELINE_CONTRACT.md for the calibration code paths."]
    lines += ["", f"Maximum saved-baseline metric difference: {audits.absolute_difference.max():.3g}. Every held-out reference row is checked against saved Experiment 38 values, with a 1e-8 tolerance. The development representative additionally reproduces the original reusable runner's full draws exactly.", "",
        "Top1 uses exactly the 25 hottest truth pixels, fixed across models. Truth is evaluation only after preparation; CRPS retains the M(M-1) estimator. Coverage and width are descriptive under the frozen one-step covariance. These scores must not be interchanged with Experiment 30 or the sparse-observation Experiment 37."]
    (output / "README.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    cache = args.output_dir / "by_trajectory"
    cache.mkdir(parents=True, exist_ok=True)
    options = pd.read_csv(BASE / "fixed_configuration.csv").iloc[0].to_dict()
    pd.DataFrame([{**options, "history_steps": "1,2,4,8", "covariance": "frozen one-step Experiment38", "mean_only_diagnostic": True}]).to_csv(args.output_dir / "fixed_configuration.csv",index=False)
    provenance = {}
    for p in [Path(__file__), *sorted((ROOT / "src").glob("*.py")),
              BASE / "fixed_configuration.csv", BASE / "results.csv"]:
        provenance[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output_dir / "source_fingerprints.json").write_text(json.dumps(provenance,indent=2)+"\n")
    catalog = trajectory_catalog(ROOT.parent / "heat_eq_laser_trajectories")
    assert len(catalog) == 33 and len(DEVELOPMENT_TRAJECTORIES) == 3
    reference = pd.read_csv(ROOT / "outputs/by_experiment/30_final_architecture_comparison/sequential_rerun_results.csv")
    thresholds = reference.groupby("trajectory").threshold_K.first().to_dict()
    for dev in (True, False):
        if not dev and args.development_only:
            break
        payloads = [(r.name, r.family, i, thresholds[r.name], options, str(cache))
            for i,r in enumerate(catalog) if (r.name in DEVELOPMENT_TRAJECTORIES) == dev
            and not (cache / f"{r.name}_checks.csv").exists()]
        if payloads:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(worker,p) for p in payloads]
                for future in as_completed(futures):
                    print("Finished", future.result(), flush=True)
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
