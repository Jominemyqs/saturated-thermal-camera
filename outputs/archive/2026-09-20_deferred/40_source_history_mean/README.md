# Source-history mean diagnostic

The 1-, 2-, 4-, and 8-step means start at their respective censored camera posteriors and replay each intervening source snapshot using the actual time steps.

The residual covariance is the same Experiment 38 one-step Q + propagated previous covariance in every row. This is a mean-only sensitivity experiment, not an L-step Bayesian filter. The n-1 frame still supplies covariance for all variants; only the deterministic mean uses the earlier frame.

The initial posterior is the inherited hybrid: sampled censored pixels, observed noisy values at unsaturated pixels. Current likelihood uses only the current frame. Source flux is thresholded at 10000; below-ambient excess is floored before diffusion; the grid boundary is zero excess. These inherited approximations are preserved for equality.

All physical coefficients, source threshold, noise scales, observation stride, HMC settings and seeds are frozen from Experiment 38. Its reference ceilings and ambient/source-width preparation are also inherited (simulation-defined reference protocol, not a new deployment calibration). Earlier frames have explicit distinct noise seeds; current and n-1 observations reproduce Experiment 38.

## Held-out results

| Steps | Field L2 | Top1 RMSE | Top1 bias | All CRPS | Top1 CRPS | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.516 | 3.195 | 0.432 | 0.241 | 1.738 | 0.845 | 7.368 |
| 2 | 0.811 | 8.226 | 3.199 | 0.262 | 3.218 | 0.848 | 7.381 |
| 4 | 1.710 | 20.038 | 8.812 | 0.311 | 8.209 | 0.717 | 7.466 |
| 8 | 2.986 | 35.723 | 19.681 | 0.420 | 18.718 | 0.507 | 7.650 |

## Interpretation

The one-step baseline remains preferred. Longer rollouts amplify positive hot-tail bias. The paired table reports decreases and increases separately: signed bias, coverage, width and exceedance slope do not have a universal lower-is-better interpretation. In particular, an exceedance slope exceeding one indicates excessive amplitude response, not improved recovery.

The source coefficient was inherited from a development fit that propagated a clipped predecessor. It can compensate for the omitted predecessor heat as well as represent actual heat input. Reusing it at every rollout step can accumulate that compensation. This is a plausible explanation supported by calibration provenance, not an isolated causal test of gamma. The simplified diffusion/cooling approximation and initial-state errors can also accumulate.

The longest window spans only 0.004 seconds, and the development starting frames remain similarly censored. Thus the experiment supplies neither a clearly unsaturated initial anchor nor a new later cooling observation. It rejects repeated use of the existing one-step effective forecast as an improvement; it does not establish that physically calibrated source-history reconstruction is unhelpful. See BASELINE_CONTRACT.md for the calibration code paths.

Maximum saved-baseline metric difference: 3.11e-14. Every held-out reference row is checked against saved Experiment 38 values, with a 1e-8 tolerance. The development representative additionally reproduces the original reusable runner's full draws exactly.

Top1 uses exactly the 25 hottest truth pixels, fixed across models. Truth is evaluation only after preparation; CRPS retains the M(M-1) estimator. Coverage and width are descriptive under the frozen one-step covariance. These scores must not be interchanged with Experiment 30 or the sparse-observation Experiment 37.
