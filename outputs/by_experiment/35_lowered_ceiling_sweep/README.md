# Lowered pseudo-camera ceiling experiment

> **Status: provisional diagnostic.** The previous-frame sampler used here does not satisfy the convergence criterion established by Experiment 37, and the two models use different current observation sets. Preserve the qualitative ceiling-response result, but rerun under a matched, convergence-checked protocol before using absolute CRPS, coverage, or width values.

This controlled experiment lowers the camera ceiling while leaving both model specifications and the noisy reference camera realization fixed. Artificial ceilings are computed only from the frozen reference observation; latent simulation truth is used only for evaluation.

## Models

- `single-frame censored RBF`: current-frame censored RBF GP.
- `advective posterior physics mean + sequential advective ST`: the best-global sequential architecture from experiment 30, with previous posterior propagation, advective physics mean, and moment-matched advective stochastic space-time residual.

The single-frame model follows the retained exceedance-diagnostic protocol: fixed-stride unsaturated observations plus every observed-saturated location. The sequential model remains frozen to experiment 30: its previous posterior uses that same saturation-mask rule, while its final current update uses the fixed stride-5 locations. This observation-set distinction is reported explicitly and is not changed across ceilings. No hyperparameter is retuned by ceiling.

## Ceiling construction

The trajectory-specific reference ceilings are read from experiment 30 rather than recomputed. Each lower ceiling is selected from that trajectory's reference observation to hit a common target censoring fraction, and `Y^(c) = min(Y^(c0), c)`. This is exactly equivalent to reclipping the same noisy pre-saturation measurement whenever `c < c0`, and it cannot use hidden truth. Reported Kelvin ceilings are held-out means across those trajectory-specific pseudo-cameras.

The 20% level is a deliberate stress test: its ceiling approaches the ambient/noise floor for these trajectories. It should not be interpreted as a typical camera operating point.

## Held-out response

### single-frame censored RBF

- Whole-field RMSE: `0.918` to `1.142` K.
- Overall CRPS: `0.764` to `0.690` K.
- Censored-region RMSE: `4.561` to `2.465` K.
- Censored-region coverage/width: `0.733` / `7.099` K to `0.675` / `1.799` K.
- Exceedance recovery slope: `0.039` to `0.001`.

### advective posterior physics mean + sequential advective ST

- Whole-field RMSE: `0.541` to `0.781` K.
- Overall CRPS: `0.222` to `0.289` K.
- Censored-region RMSE: `2.655` to `1.706` K.
- Censored-region coverage/width: `0.888` / `6.358` K to `0.801` / `2.720` K.
- Exceedance recovery slope: `0.714` to `0.504`.

## Interpretation

- The sequential model remains better at every tested ceiling. At the 20% stress level its whole-field RMSE is `0.781` K versus `1.142` K for the single-frame GP, and its top-1% CRPS is `5.239` versus `7.204` K.
- The sequential top-1% CRPS nevertheless rises from `1.764` to `5.239` K, so physical/temporal structure reduces but does not remove the penalty from lost sensor range.
- Hottest-tail uncertainty responds in the wrong direction. Sequential top-1% width contracts from `7.202` to `2.822` K while coverage falls from `0.801` to `0.019`. The single-frame model reaches zero top-1% coverage by the 15% level.
- Single-frame exceedance slope falls from `0.039` to `0.001`; it detects the inequality but does not recover its magnitude. The sequential slope remains materially larger (`0.714` to `0.504`), although its mean exceedance is still compressed at severe censoring.
- Whole-domain CRPS for the single-frame GP is not monotone because the domain is dominated by cool pixels and the increasingly large inequality set makes the posterior narrower. This must not be read as improved hot-field recovery: fixed top-1% CRPS worsens and coverage collapses.
- Censored-region RMSE also changes its target population as the ceiling falls, progressively adding cooler pixels. Fixed top-1% and deliberately hidden-band metrics are the cleaner severity diagnostics.

## Files

- `results.csv`: trajectory-level model metrics.
- `heldout30_summary.csv`: equal-trajectory held-out means and standard errors.
- `family_summary.csv`: held-out results by trajectory family.
- `paired_model_comparison.csv`: paired sequential-minus-single-frame changes and win counts.
- `threshold_schedule.csv` and `threshold_schedule_summary.csv`: actual ceiling and censoring severity.
- `region_results.csv`: overall, censored, unsaturated, near-ceiling, deliberately hidden, and fixed top-1% diagnostics.
- `temperature_percentile_results.csv` and `temperature_percentile_summary.csv`: true-temperature-bin diagnostics with extra hot-tail resolution.
- `severity_response.png`: point and distributional performance versus censoring severity.
- `ceiling_response.png`: key metrics versus the actual mean pseudo-camera ceiling.
- `failure_mode_response.png`: exceedance compression, hidden-observation recovery, and calibration response.
- `temperature_percentile_response.png`: error and uncertainty by true-temperature percentile.
- `representative_*.png`: shared, tail-focused reconstruction comparisons.

## Fixed sampler

- Previous posterior: 4 chains, 700 retained draws per chain, 1200 burn-in, thin 2.
- Current posterior: 4 chains, 350 retained draws per chain, 600 burn-in, thin 2.
- CRPS uses the unbiased `M(M-1)` estimator.
