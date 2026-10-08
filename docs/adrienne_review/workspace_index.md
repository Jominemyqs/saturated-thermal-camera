# Workspace navigation for Adrienne

Start with [review README](README.md), then [code map](code_map.md).
This small index supplements, rather than replaces, the older
[experiment inventory](../../EXPERIMENT_INVENTORY.md),
[protocol](../../EXPERIMENT_PROTOCOL.md) and
[active research notes](../../ACTIVE_RESEARCH.md).
Their dated headings and historical uses of "current" should not override
the executed native scope documented here.

## Current native baseline and controlled branches

| Output directory/report | Driver filename | Purpose |
| --- | --- | --- |
| [49 matched native/grid B/C](../../outputs/by_experiment/49_matched_native_grid_gp/README.md) | [53_thermal_matched_projection_gp.py](../../scripts/53_thermal_matched_projection_gp.py) | Two development paths, matched native observations/evaluation; projection-on/off prior inputs. Native B/C baseline. |
| [49 native diagnostic plots](../../outputs/by_experiment/49_matched_native_grid_gp/native_diagnostics/) | [54_thermal_native_advection.py](../../scripts/54_thermal_native_advection.py), plot helpers | Signed-error maps and native-node sections; not another inference result. |
| [50 advection pilot](../../outputs/by_experiment/50_native_advection/README.md) | [54_thermal_native_advection.py](../../scripts/54_thermal_native_advection.py) | Separate development-only transport diagnostic; no advection in baseline B/C. |
| [51 forecast attribution](../../outputs/by_experiment/51_native_forecast_attribution/README.md) | [55_thermal_native_forecast_attribution.py](../../scripts/55_thermal_native_forecast_attribution.py) | Cached inferred-state vs oracle propagation and unchanged numerical gradient addition; no GP fitting. |
| [52 reference/320/325 K](../../outputs/by_experiment/52_native_higher_ceiling/README.md) | [56_thermal_native_higher_ceiling.py](../../scripts/56_thermal_native_higher_ceiling.py) | Same two native cases, frozen B/C; previous inference recomputed per ceiling. Non-nested current observation sets. |
| [53 observation retention](../../outputs/by_experiment/53_native_observation_retention/README.md) | [57_thermal_native_observation_retention.py](../../scripts/57_thermal_native_observation_retention.py) | C-only controlled branch at 320/325, cached same-ceiling prior; changes current measurement retention only. |

The output numbers and driver prefixes differ. Configs live beside reports;
cache arrays are in `by_trajectory/`. Do not run these scripts to browse them:
they write in-place to their experiment directories. No new run is needed
to review this package. Source inputs are in the sibling
`../heat_eq_laser_trajectories/` directory, not copied here.

## Retained grid history and provenance

- [24 propagation/advection baseline](../../outputs/by_experiment/24_posterior_physics_mean_advection/README.md): surviving development calibration tables used by the later coefficient lineage.
- [19 stochastic covariance ablation](../../outputs/by_experiment/19_stochastic_spde_ablation/README.md): supporting kernel comparison, not the present finite-step native covariance experiment.
- [37 converged sparse architectures](../../outputs/by_experiment/37_canonical_converged_architectures/README.md), [38 pseudo-censoring](../../outputs/by_experiment/38_pseudo_censoring_calibration/README.md): earlier convergence/settings and calibration provenance. Native runs do not apply learned inflation.
- [43 matched ceiling model gaps](../../outputs/by_experiment/43_matched_ceiling_model_gap/README.md): retained 33-path, six-ceiling grid A/B/C result and immediate config parent of native49.
- [45 failure decomposition](../../outputs/by_experiment/45_censoring_severity_decomposition/README.md), [46 oracle attribution](../../outputs/by_experiment/46_oracle_forecast_decomposition/README.md): historical grid diagnostics.
- [48 projection audit](../../outputs/by_experiment/48_controlled_projection_audit/README.md): deterministic representation audit preceding native49. Output47's native FEM pilot uses a different propagation operator; do not substitute it for native49's matched Gaussian transfer.
- [30 architecture comparison](../../outputs/by_experiment/30_final_architecture_comparison/README.md): historical short-chain results, not the current converged native inference protocol.

## Archived or deferred

See the [dated archive guide](../../outputs/archive/2026-09-20_deferred/README.md)
and [relocation manifest](../../outputs/archive/2026-09-20_deferred/relocation_manifest.json)
for nested ceilings (39/39b), source-history (40), smoothing (42), and
collapse diagnostics (44). Their drivers are under
[scripts/archive/2026-09-20_deferred](../../scripts/archive/2026-09-20_deferred/).
Older sequential/joint branches have a separate
[superseded diagnostic guide](../../outputs/archive/superseded_diagnostics/README.md).
Reusable source modules may remain in `src/` without implying an active branch.

Future non-saturation-mask and within-step integration ideas are **not
implemented by this review**. No scientific files or archives were reorganized.
