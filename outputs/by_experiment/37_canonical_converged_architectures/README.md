# Canonical converged architecture comparison

## Purpose

This experiment replaces the short-chain Experiment-30 architecture table for quantitative use under the same sparse current-observation architecture. It was run only after the reference equality audit identified the previous-frame censored-posterior sampler as the main source of the old tail underdispersion.

## Frozen protocol

- 33 trajectories: 3 development and 30 held out.
- Frozen 3% reference ceilings and camera-noise seeds from Experiment 30.
- Every canonical model uses the identical fixed-stride current observation set.
- Every temporal model uses the identical previous censored posterior: 4 chains, 2000 retained draws per chain, 15000 burn-in, thin 10.
- Every canonical Gaussian current update uses 4 chains, 350 retained draws per chain, 600 burn-in, thin 2. The observation-rich diagnostic uses 4 chains, 2000 retained draws per chain, 15000 burn-in, thin 10 because its inequality dimension is much larger.
- CRPS uses the unbiased `M(M-1)` estimator.
- Hottest 1% means the exact rank-based hottest 25 of 2501 pixels.
- Previous saturated pixels are sampled from the censored RBF posterior; unsaturated noisy pixels are retained at their observed values. This is not called a complete latent posterior.
- Previous observations enter only through the previous posterior and are not reused in the current likelihood.

The canonical current likelihood contains `117` fixed-stride observations and only `0`--`4` saturated observations per trajectory. This reproduces the earlier sequential architecture but does not use the complete current censoring mask. It is therefore a controlled sparse-observation comparison, not yet a full-camera benchmark.

The observation-rich current-only row is diagnostic only. It includes all observed-saturated current pixels and is excluded from architecture rankings because its observation set differs.

The legacy joint model is excluded because it conditions jointly on the previous observations after using a previous-state-derived mean, so it does not share the clean sequential information flow.

## Held-out results

| Model | Field error | RMSE (K) | Overall CRPS (K) | Top-1% RMSE (K) | Top-1% CRPS (K) | Coverage | Width (K) | Prior SD (K) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current-only ambient mean + RBF | 0.907 | 1.131 | 0.779 | 10.260 | 5.885 | 0.693 | 12.72 | 3.43 |
| Posterior physics mean + RBF | 0.484 | 0.589 | 0.740 | 3.193 | 1.706 | 0.923 | 12.97 | 3.43 |
| Advective posterior physics mean + RBF | 0.467 | 0.569 | 0.737 | 3.288 | 1.742 | 0.924 | 12.95 | 3.43 |
| Posterior physics mean + sequential ST | 0.457 | 0.556 | 0.227 | 3.173 | 1.676 | 0.839 | 8.30 | 0.76 |
| Posterior physics mean + sequential advective ST | 0.457 | 0.556 | 0.230 | 3.181 | 1.685 | 0.835 | 7.42 | 0.75 |
| Advective posterior physics mean + sequential advective ST | 0.440 | 0.536 | 0.221 | 3.288 | 1.736 | 0.816 | 7.40 | 0.75 |
| Posterior-sample mixture + sequential advective ST | 0.456 | 0.554 | 0.229 | 3.184 | 1.664 | 0.835 | 7.33 | 0.75 |

## Observation-rich diagnostic

With fixed stride plus every current saturated pixel, the current-only RBF has field error `0.735`, overall CRPS `0.762` K, top-1% CRPS `2.976` K, coverage `0.836`, and width `7.87` K. These values are not ranked against the canonical rows because they use more current inequality observations.

## Covariance interpretation

The reported sequential prior scale uses the complete predictive variance `diag(Q + B Sigma B^T)`. It is not the innovation scale `Q` alone. RBF and sequential calibration comparisons should use this total scale and posterior interval diagnostics together.

## Convergence diagnostics

Across all 33 trajectories, the largest previous-frame saturated-pixel R-hat was `1.044` and no saturated pixel exceeded `1.05`. Across the canonical Gaussian current updates, the largest all-domain R-hat was `1.028`, the largest hottest-1% R-hat was `1.009`, and no hottest-1% pixel exceeded `1.05`.

The observation-rich diagnostic is not part of the ranking and retains one marginal diagnostic: its largest hottest-1% R-hat is `1.051`. The posterior-sample mixture is assessed by component-weight diagnostics rather than MCMC R-hat.

The settings previously called converged in Experiment 35 (4 chains, 700 retained draws, 1200 burn-in, thin 2) did not pass these checks. Its ceiling-response trends remain diagnostic, but its absolute uncertainty values must be rerun under this protocol before paper use.

## Interpretation

- Adding the posterior physics mean to the matched current-only RBF improves field error and hottest-1% CRPS on all 30 held-out trajectories.
- With the posterior physics mean fixed, sequential ST improves field error on all 30 trajectories and hottest-1% CRPS on 22, but gives narrower intervals and lower hottest-1% coverage than the RBF residual.
- Mean advection improves global field error on all 30 trajectories but slightly worsens hottest-1% RMSE, CRPS, and peak error on average.
- Adding advection to the sequential covariance does not improve overall CRPS on any held-out trajectory relative to stationary sequential ST.
- Posterior-sample mixture propagation gives a small hottest-1% CRPS improvement over the Gaussian moment-matched advective model on 26 of 30 trajectories; it does not materially change coverage.

These are architecture effects under matched sparse current observations. Before a paper-level full-camera comparison, rerun every row with the same fixed-stride unsaturated observations plus every observed-saturated current pixel, then recheck convergence.

## Integrity checks

All `33` trajectory checks use 25 hottest-tail pixels. RBF covariance identity across mean choices: `True`. Sequential advective covariance identity across mean choices: `True`. Previous observations reused in the current likelihood: `False`.

## Files

- `results.csv`: trajectory-level metrics and covariance scales.
- `heldout30_summary.csv`: equally weighted held-out means and standard errors.
- `family_summary.csv`: diagonal, horizontal, and spiral summaries.
- `paired_comparisons.csv`: paired held-out differences and win counts.
- `model_protocol.csv`: exact architecture and information-flow definitions.
- `implementation_checks.csv`: trajectory-level consistency checks.
- `comparison.png`: point and distribution metrics.
- `total_prior_covariance_scale.png`: complete RBF and sequential prior scales.