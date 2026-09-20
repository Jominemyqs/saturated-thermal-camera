# Pseudo-censoring calibration

## Question

Can measurements deliberately hidden below the hardware ceiling teach a frozen uncertainty correction that transfers to unseen trajectories, unseen censoring severities, and the true hardware ceiling?

## Protocol

- Base reconstruction: posterior physics mean plus sequential stationary stochastic heat covariance.
- Every run uses fixed-stride unsaturated observations plus every observed-censored current pixel.
- Calibration targets are reference-camera measurements in the deliberately hidden band; latent truth is never used for fitting.
- Training uses only the three development trajectories at 5%, 7.5%, and 10% pseudo-censoring.
- The fitted rule is frozen before evaluating 30 held-out trajectories at 3%, 5%, 7.5%, 10%, 15%, and 20% censoring.
- Inflation is applied only to observed-censored pixels. Posterior means and all point metrics are held fixed.
- CRPS uses the unbiased M(M-1) estimator and the hottest 1% is the exact hottest 25 of 2501 pixels.

## Calibration rules

- Global inflation: one development-fitted scalar.
- Severity inflation: a monotone function of the observed censored fraction.
- Feature inflation: a small softplus regression using only posterior, prior, mask, and heat-flux features available at deployment.

## Held-out reference ceiling

| Method | Overall CRPS | Top-1% CRPS | Coverage | Width (K) |
|---|---:|---:|---:|---:|
| uncalibrated | 0.241 | 1.738 | 0.845 | 7.37 |
| global inflation | 0.235 | 1.792 | 0.969 | 16.68 |
| severity inflation | 0.235 | 1.792 | 0.969 | 16.68 |
| feature inflation | 0.235 | 1.845 | 0.915 | 14.61 |

## Held-out 20% stress ceiling

| Method | Overall CRPS | Top-1% CRPS | Coverage | Width (K) |
|---|---:|---:|---:|---:|
| uncalibrated | 0.661 | 1.896 | 0.812 | 8.00 |
| global inflation | 0.612 | 2.020 | 0.969 | 18.12 |
| severity inflation | 0.612 | 2.020 | 0.969 | 18.12 |
| feature inflation | 0.613 | 2.085 | 0.887 | 14.85 |

## Interpretation

The simplest fitted rule is a global posterior-SD multiplier of `2.264` on observed-censored pixels. The severity-only coefficient is `0.000`, so this experiment did not learn an additional monotone severity response beyond the constant inflation.

At the reference ceiling, global inflation improves all-domain CRPS from `0.241` to `0.235` K and observed-censored-region CRPS from `1.846` to `1.649` K. Hottest-1% CRPS instead changes from `1.738` to `1.792` K, while coverage rises from `0.845` to `0.969` and width from `7.37` to `16.68` K.

At the unseen 20% level, global inflation improves all-domain CRPS from `0.661` to `0.612` K and observed-censored-region CRPS from `1.791` to `1.547` K. Hottest-1% CRPS worsens from `1.896` to `2.020` K despite coverage increasing from `0.812` to `0.969`.

Thus pseudo-censoring learns a transferable broad uncertainty correction, but the current scalar and feature rules do not improve hottest-tail distribution quality. The richer feature rule does not outperform global inflation, so it should not be promoted as the preferred calibrator.

Point metrics are identical by construction. Any improvement is therefore a distribution-calibration effect rather than a reconstruction-mean change.

## Convergence and leakage checks

Maximum previous-posterior R-hat: `1.011`. Maximum current all-domain R-hat: `1.009`. Maximum current censored-pixel R-hat: `1.009`.

Development calibration contains `1017` deliberately hidden pixel examples before equal group weighting. Held-out truth was not used by any fitted calibrator.

## Files

- `results.csv`, `region_results.csv`: trajectory-level metrics.
- `heldout30_summary.csv`, `region_summary.csv`, `family_summary.csv`: held-out summaries.
- `calibration_examples.csv`: development pseudo-labels and deployment-available features.
- `calibrator_parameters.csv`: frozen calibration rules.
- `development_calibration_fit.csv`: in-sample hidden-measurement scores.
- `calibration_scale_summary.csv`: learned inflation versus ceiling.
- `paired_comparisons.csv`, `region_paired_comparisons.csv`: paired held-out changes.
- `implementation_checks.csv`: observation, convergence, and leakage checks.
- `calibration_response.png`: held-out severity response.
