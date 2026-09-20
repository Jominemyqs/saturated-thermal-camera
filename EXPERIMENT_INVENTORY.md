# Experiment inventory

**September 20 focus:** see ACTIVE_RESEARCH.md. The active next experiment is
script 49 / output 45 (censoring-severity failure decomposition). Nested-response,
source-history and smoothing branches below are historical: their outputs are
now under outputs/archive/2026-09-20_deferred and drivers under
scripts/archive/2026-09-20_deferred. Original numerical files are unchanged.

This inventory identifies the current presentation path without deleting
historical work. The current forward reference is Experiment 38's uncalibrated
posterior physics mean plus stationary sequential covariance, with stride-5
unsaturated and all observed-censored current observations. Earlier RBF and
sparse-observation comparisons remain separately labeled retained evidence.

## Canonical retained

| Topic | Driver | Output | Status |
| --- | --- | --- | --- |
| Corrected CRPS/scoring | `scripts/19_thermal_corrected_crps.py` and `src/metrics.py` | Retained scoring used by later experiments | All CRPS uses the unbiased `M(M-1)` estimator |
| Previous-frame inequality propagation | `scripts/22_thermal_previous_posterior_propagation.py` | `outputs/by_experiment/20_physics_mean_propagation/` | Central observed-clipped vs posterior physics mean vs full-posterior result |
| Mean-only advection | `scripts/26_thermal_posterior_physics_advection.py` | `outputs/by_experiment/24_posterior_physics_mean_advection/` | Current controlled advection experiment with an identical RBF residual |
| Broad retained-model summary | `scripts/27_thermal_broad_model_comparison.py` | `outputs/by_experiment/25_broad_model_comparison/` | Descriptive cross-experiment table with architecture labels and matched paired comparisons |
| Historical architecture consolidation | `scripts/32_thermal_final_architecture_comparison.py` | `outputs/by_experiment/30_final_architecture_comparison/` | Retained for provenance, but its short previous-posterior chains make its probabilistic values noncanonical |
| Canonical converged sparse-observation comparison | `scripts/40_thermal_canonical_converged_architectures.py` | `outputs/by_experiment/37_canonical_converged_architectures/` | Current quantitative reference for the inherited stride-5 current-observation architecture: matched observations, exact hottest-1% mask, total sequential covariance, and checked previous/current posterior convergence |
| Pseudo-censoring calibration | `scripts/41_thermal_pseudo_censoring_calibration.py` | `outputs/by_experiment/38_pseudo_censoring_calibration/` | Canonical full-censor-mask severity experiment: fits uncertainty-only corrections on 3 development paths at 5--10% pseudo-censoring, freezes them, and evaluates 30 held-out paths through unseen 15--20% severity; reflected HMC resolves the former severe-censoring convergence failure |

## Supporting ablations

| Topic | Driver/output | Status |
| --- | --- | --- |
| L=1 stochastic smoothing pilot | `scripts/46_thermal_one_step_smoothing_pilot.py`; `outputs/by_experiment/42_one_step_smoothing_pilot/` | Development-only midpoint pilot with opt-in joint current draws and stochastic future likelihood. All three weight-ESS/repeat-stability gates failed (pooled ESS 2--4 of 1200); no lag/held-out expansion and no valid smoothing winner. |
| Observation bridge and smoothing readiness | `scripts/45_thermal_observation_bridge.py`; `outputs/by_experiment/41_observation_bridge_and_smoothing_readiness/` | Frozen 33-trajectory sparse-vs-all-censored current likelihood audit; development-only observed-mask cooling visibility and joint-sample audit. No smoother implemented. |
| Source-history mean | `scripts/44_thermal_source_history_mean.py`; `outputs/by_experiment/40_source_history_mean/` | Controlled L=1,2,4,8 deterministic means with identical Experiment 38 one-step covariance/current likelihood. All 30 baseline rows reproduce saved metrics within 3.2e-14. Longer histories worsen field and hot-tail point/distribution errors on every held-out trajectory. The inherited clipping-compensating source coefficient is not validated for repeated open-loop use; retain L=1. |
| Stochastic heat-process covariance | `scripts/21_thermal_stochastic_spde_ablation.py`; `outputs/by_experiment/19_stochastic_spde_ablation/` | Retained supporting RBF vs stochastic ST vs advective stochastic ST ablation; not mandatory in the main model |
| Architecture-aware one-step sequential GP | `scripts/28_thermal_one_step_sequential_gp.py`; `outputs/by_experiment/26_one_step_sequential_gp/` | Uses the canonical RBF previous posterior, separates controlled sequential rows from the legacy joint reference, and shows improved field/global scores but substantial hot-tail undercoverage |
| Posterior-sample sequential propagation | `scripts/29_thermal_full_posterior_sequential.py`; `outputs/by_experiment/27_full_posterior_sequential/` | Focused mean-only vs existing moment-matched vs likelihood-reweighted censored-region posterior-sample mixture comparison, with quartile and extreme-tail calibration diagnostics |
| Hottest-tail uncertainty-scale diagnostic | `scripts/30_hottest_tail_diagnostic.py`; `outputs/by_experiment/28_hottest_tail_diagnostic/` | Frozen held-out rerun separating posterior-mean quality from interval scale, with pixel maps, temperature-percentile calibration, and a variance-matched RBF control |
| Uncertainty-origin and oracle diagnostic | `scripts/33_thermal_uncertainty_origin_oracle.py`; `outputs/by_experiment/31_uncertainty_origin_oracle/` | Frozen 33-trajectory before/after calibration audit, explicit hybrid previous-posterior diagnosis, controlled true-previous-state oracle, development-only oracle source recalibration, and point/distribution metrics; no adaptive covariance added |
| Canonical uncertainty/oracle diagnostic v2 | `scripts/34_thermal_uncertainty_oracle_v2.py`; `outputs/by_experiment/32_uncertainty_oracle_v2/` | Retained final diagnostic separating previous posterior, current forecast, and current posterior; compares hybrid with coherent full-latent previous inference, keeps fixed- and development-recalibrated source questions separate, and resolves the former top-1% definition discrepancy |
| Lowered pseudo-camera ceiling sweep | `scripts/38_thermal_lowered_ceiling_sweep.py`; `outputs/by_experiment/35_lowered_ceiling_sweep/` | Provisional diagnostic: controlled reclipping signal is useful, but absolute UQ values require a matched-observation, convergence-checked rerun |
| Nested pseudo-ceiling response | `scripts/42_thermal_nested_ceiling_response.py`; `outputs/by_experiment/39_nested_ceiling_response/` | Development-only gated diagnostic using Experiment 38's converged nested runs; response slopes/curvature did not improve leave-one-trajectory-out exceedance prediction, so no learned hard-extrapolation model was run |
| Smooth nested pseudo-ceiling response (39b) | `scripts/43_thermal_nested_ceiling_response_v2.py`; `outputs/by_experiment/39b_nested_ceiling_response/` | Final bounded test of the nested-response idea: 10 ceilings, five anchors, quadratic response summaries, a 0--2.63 K known pseudo-exceedance range, and explicit residual prediction. Trained on moderate examples, smooth-response features improved larger-exceedance development LOTO RMSE by only 1.3% while reducing Spearman rho by 0.007; the predeclared gate stopped before the 30 held-out trajectories and the branch is closed. |
| Friday three-figure meeting package | `scripts/31_prepare_friday_meeting.py`; `outputs/by_experiment/29_friday_meeting/` | Presentation-only export of percentile calibration, representative top-1% intervals, compact tail table, and meeting narrative |
| Toy parametric and GP studies | `scripts/00` through `15`; `outputs/by_experiment/01` through `12` | Retained background and prior-sensitivity evidence |
| Thermal diffusion and two-frame studies | `scripts/16` through `18` and their named output folders | Retained implementation and supporting sensitivity studies |

## Reusable implementation

### Development-only smoothing diagnostic

`scripts/48_thermal_smoothing_collapse_diagnostic.py` and
`outputs/by_experiment/44_smoothing_collapse_diagnostic` retain the three-path
L=1 weight-collapse decomposition. Original Experiment 42 weights reproduce;
tempering, dimension subsets, Q sensitivity and exact Gaussian-only integration
identify both proposal concentration and finite-path likelihood noise.
The full-likelihood reliability gate fails. No SMC, longer lags or held-out
smoothing are implemented. `src/smoothing_collapse.py` contains diagnostic helpers.

### Matched ceiling comparison

`scripts/47_thermal_matched_ceiling_model_gap.py` and
`outputs/by_experiment/43_matched_ceiling_model_gap` are the canonical retained
three-model full-camera-mask ceiling comparison: ambient RBF, posterior physics
RBF, and posterior physics sequential stationary ST. All 33 trajectories and
six ceilings completed; sequential held-out results reproduce Experiment 38.
`src/ceiling_model_comparison.py` provides shared RBF blocks and fixed-region
scoring. Existing historical results remain intact; smoothing stays paused.

| Module | Responsibility |
| --- | --- |
| `src/source_history.py` | Repeated inherited diffusion/cooling and source forecast over actual timestamps; tested for exact L=1 equality and nonuniform source accumulation |
| `src/metrics.py` | Unbiased empirical CRPS and field metrics |
| `src/uncertainty_diagnostics.py` | Temperature-percentile and named-region error-versus-uncertainty diagnostics, including explicit zero-SD failure reporting |
| `src/censored_gp.py` | Current-frame RBF covariance and censored GP sampling |
| `src/dense_censored_gp.py` | Censored inference from either a dense Gaussian prior or observation/prediction covariance blocks |
| `src/truncated_gaussian.py` | Reflected exact-HMC sampling for high-dimensional Gaussian inequality blocks |
| `src/pseudo_censoring_calibration.py` | Deployment-available calibration features, CRPS-fitted uncertainty scaling, and posterior-draw inflation |
| `src/nested_ceiling_response.py` | Matching across nested ceilings and leakage-safe grouped ridge diagnostics for posterior response features |
| `src/nested_ceiling_curves.py` | Smooth multi-ceiling response summaries and multiple-anchor pseudo-hidden examples |
| `src/pseudo_censoring_runner.py` | Reusable, equality-audited Experiment 38 sequential posterior inference for a specified pseudo-camera ceiling |
| `src/stochastic_heat_gp.py` | Stationary stochastic heat covariance, finite-step innovation, and one-step residual propagation |
| `src/thermal_posterior_physics.py` | Trajectory preparation, hybrid and coherent full-latent previous censored posteriors, diffusion/cooling, source displacement, and posterior physics means |
| `src/thermal_plotting.py` | Shared ambient-to-ceiling nonlinear temperature scale and fixed excess-temperature contours for tail-visible reconstruction figures |
| `src/thermal_trajectory.py` | XDMF/HDF5 loading and surface-grid projection |
| `src/diffusion.py` | Effective diffusivity and cooling-rate estimation |
| `scripts/10_gp_2d_censored.py` | Validated general kernel implementation, including the supporting stochastic space-time covariance |

Future experiments must follow `EXPERIMENT_PROTOCOL.md` and state any intentional
departure from the chosen historical or Experiment 38 baseline's observation,
inference, and evaluation protocol before comparing numerical results.

## Smoke/debug

Smoke outputs are written under `/private/tmp` and are not part of the
repository. Superseded sequential, joint, and broad-restart driver scripts
were removed after the August 11 controlled audit. Their historical outputs
and duplicate top-level output folders remain available pending a separately
confirmed cleanup pass.
