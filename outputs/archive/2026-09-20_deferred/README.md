# Deferred branches: September 20, 2026

Removed from the active experiment list, not erased:

| Former output | Archived driver | Status |
| --- | --- | --- |
| 39_nested_ceiling_response | 42 | Narrow initial development gate failed |
| 39b_nested_ceiling_response | 43 | Stronger nested-response gate failed |
| 40_source_history_mean | 44 | Repeated effective forecast accumulated bias |
| 42_one_step_smoothing_pilot | 46 | Importance-reweighting reliability gate failed |
| 44_smoothing_collapse_diagnostic | 48 | Diagnostic only; no validated smoother |

Drivers are in scripts/archive/2026-09-20_deferred. They received only repository-
root discovery and archived-output-path changes. Existing numerical results,
original source manifests and all unique code remain. Historical manifests
describe the original run, not the relocated driver bytes. The relocation
manifest records original hashes and mappings; all 188 moved files were verified
byte-for-byte immediately after the move. No archived experiment was rerun.

Keep active:
- Script 45 / output 41: observation-set bridge audit, valid infrastructure.
- Script 47 / output 43: central matched A/B/C ceiling comparison.
- Script 41 / output 38: canonical configuration, baseline lineage and supporting
  calibration results. Shared calibration-feature code is imported by the runner.
- Script 49 / output 45: new failure decomposition, no changed model.

Shared src modules and their tests remain because moving them risks inference
dependencies and loses reusable verified code. Presence in src is not an active
research recommendation. Calibration variants remain supporting evidence only.

Older PDFs and manifests may cite the former outputs/by_experiment locations;
resolve those paths using relocation_manifest.json. Do not rewrite historical
results to make a new narrative appear retrospectively canonical.
