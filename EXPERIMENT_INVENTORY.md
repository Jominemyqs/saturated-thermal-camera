# Experiment inventory

This inventory identifies the current presentation path without deleting
historical work. The canonical thermal-camera architecture ends at a
posterior physics mean followed by a current-frame spatial RBF residual and a
censored likelihood.

## Canonical retained

| Topic | Driver | Output | Status |
| --- | --- | --- | --- |
| Corrected CRPS/scoring | `scripts/19_thermal_corrected_crps.py` and `src/metrics.py` | Retained scoring used by later experiments | All CRPS uses the unbiased `M(M-1)` estimator |
| Previous-frame inequality propagation | `scripts/22_thermal_previous_posterior_propagation.py` | `outputs/by_experiment/20_physics_mean_propagation/` | Central observed-clipped vs posterior physics mean vs full-posterior result |
| Mean-only advection | `scripts/26_thermal_posterior_physics_advection.py` | `outputs/by_experiment/24_posterior_physics_mean_advection/` | Current controlled advection experiment with an identical RBF residual |
| Broad retained-model summary | `scripts/27_thermal_broad_model_comparison.py` | `outputs/by_experiment/25_broad_model_comparison/` | Descriptive cross-experiment table with architecture labels and matched paired comparisons |
| Final architecture consolidation | `scripts/32_thermal_final_architecture_comparison.py` | `outputs/by_experiment/30_final_architecture_comparison/` | Final common-protocol table, RBF/ST scale control, propagation ablation, and whole-field versus tail interpretation |

## Supporting ablations

| Topic | Driver/output | Status |
| --- | --- | --- |
| Stochastic heat-process covariance | `scripts/21_thermal_stochastic_spde_ablation.py`; `outputs/by_experiment/19_stochastic_spde_ablation/` | Retained supporting RBF vs stochastic ST vs advective stochastic ST ablation; not mandatory in the main model |
| Architecture-aware one-step sequential GP | `scripts/28_thermal_one_step_sequential_gp.py`; `outputs/by_experiment/26_one_step_sequential_gp/` | Uses the canonical RBF previous posterior, separates controlled sequential rows from the legacy joint reference, and shows improved field/global scores but substantial hot-tail undercoverage |
| Posterior-sample sequential propagation | `scripts/29_thermal_full_posterior_sequential.py`; `outputs/by_experiment/27_full_posterior_sequential/` | Focused mean-only vs existing moment-matched vs likelihood-reweighted censored-region posterior-sample mixture comparison, with quartile and extreme-tail calibration diagnostics |
| Hottest-tail uncertainty-scale diagnostic | `scripts/30_hottest_tail_diagnostic.py`; `outputs/by_experiment/28_hottest_tail_diagnostic/` | Frozen held-out rerun separating posterior-mean quality from interval scale, with pixel maps, temperature-percentile calibration, and a variance-matched RBF control |
| Uncertainty-origin and oracle diagnostic | `scripts/33_thermal_uncertainty_origin_oracle.py`; `outputs/by_experiment/31_uncertainty_origin_oracle/` | Frozen 33-trajectory before/after calibration audit, explicit hybrid previous-posterior diagnosis, controlled true-previous-state oracle, development-only oracle source recalibration, and point/distribution metrics; no adaptive covariance added |
| Canonical uncertainty/oracle diagnostic v2 | `scripts/34_thermal_uncertainty_oracle_v2.py`; `outputs/by_experiment/32_uncertainty_oracle_v2/` | Retained final diagnostic separating previous posterior, current forecast, and current posterior; compares hybrid with coherent full-latent previous inference, keeps fixed- and development-recalibrated source questions separate, and resolves the former top-1% definition discrepancy |
| Friday three-figure meeting package | `scripts/31_prepare_friday_meeting.py`; `outputs/by_experiment/29_friday_meeting/` | Presentation-only export of percentile calibration, representative top-1% intervals, compact tail table, and meeting narrative |
| Toy parametric and GP studies | `scripts/00` through `15`; `outputs/by_experiment/01` through `12` | Retained background and prior-sensitivity evidence |
| Thermal diffusion and two-frame studies | `scripts/16` through `18` and their named output folders | Retained implementation and supporting sensitivity studies |

## Reusable implementation

| Module | Responsibility |
| --- | --- |
| `src/metrics.py` | Unbiased empirical CRPS and field metrics |
| `src/uncertainty_diagnostics.py` | Temperature-percentile and named-region error-versus-uncertainty diagnostics, including explicit zero-SD failure reporting |
| `src/censored_gp.py` | Current-frame RBF covariance and censored GP sampling |
| `src/dense_censored_gp.py` | Censored inference from either a dense Gaussian prior or observation/prediction covariance blocks |
| `src/stochastic_heat_gp.py` | Stationary stochastic heat covariance, finite-step innovation, and one-step residual propagation |
| `src/thermal_posterior_physics.py` | Trajectory preparation, hybrid and coherent full-latent previous censored posteriors, diffusion/cooling, source displacement, and posterior physics means |
| `src/thermal_plotting.py` | Shared ambient-to-ceiling nonlinear temperature scale and fixed excess-temperature contours for tail-visible reconstruction figures |
| `src/thermal_trajectory.py` | XDMF/HDF5 loading and surface-grid projection |
| `src/diffusion.py` | Effective diffusivity and cooling-rate estimation |
| `scripts/10_gp_2d_censored.py` | Validated general kernel implementation, including the supporting stochastic space-time covariance |

## Smoke/debug

Smoke outputs are written under `/private/tmp` and are not part of the
repository. Superseded sequential, joint, and broad-restart driver scripts
were removed after the August 11 controlled audit. Their historical outputs
and duplicate top-level output folders remain available pending a separately
confirmed cleanup pass.
