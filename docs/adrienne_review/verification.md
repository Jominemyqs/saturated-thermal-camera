# Inspection and verification record

Date: 2026-10-08. Scope: parameter/noise provenance and workspace navigation,
not a new scientific result. Source HEAD:
`29786d196c4b5689d678584c42d5283939fafb41`.
The native drivers/modules/results are untracked in this working tree, so HEAD
alone cannot reproduce them. The file fingerprints below identify inspected bytes.

## Scope and preservation

Inspected git status first, then the existing protocol, experiment inventory,
active notes, dated archive README/relocation manifest and superseded archive
guide. Traced drivers 53/55/56/57, shared preparation/propagation/covariance/
likelihood/sampler functions, historical calibration and configuration parents.
Read saved configs, checks, source manifests and selected cached arrays.

Pre-existing tracked modifications were `.DS_Store`, `outputs/.DS_Store` and
`ACTIVE_RESEARCH.md`. Pre-existing untracked work comprised output folders
49-53 (including 50), drivers 53-57, four native helper modules
(`forecast_attribution`, `matched_projection`, `native_ceiling_diagnostics`,
`native_diagnostics`) and their four tests. None was cleaned, staged or reverted.

Only the five files in this review directory and a small root README link were
authored for this task. No scientific code, config, result, cache or archive was
moved, renamed, deleted or edited. No inference driver, cooling calibration,
GP refit, sweep, non-saturation likelihood or within-step integration was run.

Preservation check: SHA-256 over sorted repository-relative paths and file
content digests for **2084 files** under `src`, `scripts`, `tests`, `outputs`,
excluding `__pycache__` and `.DS_Store`:

`3326fd31a442b91386f9773a9618264168d25d3ca0da81a57e95f67b38bf5eb1`

The post-documentation digest and file count match this pre-edit baseline exactly.
Pre-existing `ACTIVE_RESEARCH.md` digest:
`be8e1c1be171a448c1692a252c9ec27b70bf780861b31b5ec3e1788dd1de74b4`.

## Checks performed

- CSV configs for 49/52/53 agree exactly as parsed; 35 inventory rows have
  complete columns, existing source/artifact paths and valid source line spans.
- Alpha, beta, signal SD and gamma agree through output24 -> 27 -> 32 -> 35 ->
  37 -> 38 -> 43 -> 49. Aggregating the saved development tables by the coded
  medians reproduces those four values within 1e-14 relative tolerance,
  **without refitting**. Other protocol settings were not assumed invariant.
- Read-only array equality: 49 native B/C vs 52 level0 means/SDs/limits/draws,
  previous means/cameras, truth and observation IDs agree exactly. For all four
  high-ceiling cases, 53 `existing_*` arrays equal 52 C arrays exactly, and
  physics means/prior diagonals/cameras/hot-area weights are identical.
- Cached dimensions confirm 6275 native nodes; previous draws 1600x6275;
  final current draws 1200x6275; total area 0.00015 m2 for both paths.
  These are two development cases, not 30 held-out cases.
- Saved checks show no previous/current retries. Largest previous R-hat in52
  is 1.009186; largest current R-hat across49/52 is 1.009348. Output53 maximum
  is 1.007662. These are recorded diagnostics, not new convergence runs.
- Output51 saved equality checks: cached forecast maximum difference 0;
  component-budget roundoff <=5.69e-14 K. Output53 saved overlap difference 0.
  These numerical checks do not certify physical correctness.
- Existing lightweight suite: **81 tests passed**, including small toy sampling
  tests, in 1.678 s of unittest-reported test time. A known precision-loss
  warning arose in the near-constant skewness test; it passed. No trajectory GP
  reconstruction was launched.
- `compileall` passed for all 112 Python files in `src`, `scripts`, `tests` (including archived scripts);
  bytecode cache redirected to `/tmp/thermal_pycache`.
- Relative Markdown links, source anchors/spans and inventory paths checked;
  `git diff --check` run. Scientific-file preservation and documentation-only
  scope checked against initial status.

Lightweight commands, from repository root:

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPYCACHEPREFIX=/tmp/thermal_pycache .venv/bin/python -m unittest discover -s tests -v
env PYTHONPYCACHEPREFIX=/tmp/thermal_pycache .venv/bin/python -m compileall -q src scripts tests
git diff --check
```

No new validation claim is made about calibration optimality, initialization
sensitivity, effective sample size, or the entire historical result inventory.
Checks used the repository `.venv`: Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1,
pandas 2.3.3. This records the inspection environment, not a recovered
environment lock for every historical run. Original HDF5 dataset files were
not copied or newly fingerprinted by this documentation audit.

## Discrepancies and unresolved questions

| Evidence | Finding | Appropriate follow-up, not performed here |
| --- | --- | --- |
| 49/52/53 copied `frozen_configuration.csv` | Still labels 33 paths, A/B/C, six fractions, exact hottest25/2501, pseudo-calibration target and `truth_used_for_calibration=false`. Native drivers actually use two paths, area-weighted hot1%, no calibration; 53 only C. `calibration_draws`, training fractions and several historical labels are inactive. | In a separately authorized change, add execution-specific metadata without overwriting historical files. Do not infer camera-only historical calibration from the copied boolean. |
| Root README/inventory September20 wording; protocol | Historical "current" labels and relocated paths are stale. Protocol asks for hot1% R-hat; actual wrappers gate on all-domain and observed-censored targets, not separately hottest1%. | Use this package for native review; decide whether protocol or future gate should change explicitly. |
| Historical driver18 search and driver19 fallback | Intended finite search exists, but `outputs/thermal_two_frame_ablation/selected_residual_configuration.csv` and `residual_sensitivity_summary.csv` were not found anywhere under local outputs. Later values are saved; original winning search and then-used CRPS revision are unresolved. | Recover original artifacts/command logs if available. Do not certify a selection objective was actually attained. |
| Gamma calibration tables/helper | Gamma uses clipped **noise-free truth**, not observed-clipped noisy values or reconstructed states; later report wording can obscure this. | Describe historical calibration literally. No argument for retuning follows solely from this documentation finding. |
| `prepare_trajectory` called by53/56 | Ambient and source width use full projected simulator histories; diagnostic coefficient fits are recomputed but ignored. Reference ceiling originally uses simulator truth, and current gradient is solution-derived. | Separate synthetic preparation/auxiliary information from deployment-accessible inputs before making deployment claims. |
| Supplied export, output52 physical provenance | Gradient K/m and gamma m/s are conditional on metre/second coordinate conventions. Exact generating notebook/version, export projection space, relation to prescribed heating and deployment availability remain unresolved. Stored dt0.0005 s differs from thesis0.005 s. | Obtain original notebook/version, not a guessed unit/sign conversion. |
| C construction and marginal draws | Inherited covariance is empirical from a hybrid, non-coherent full-field representation; Q is free-space, A uses finite-grid zero extension; the mean floor is absent from A. | Retain these approximation labels. Do not call current native C an exact 3D stochastic discretization. |
| Sampler implementation and records | Classical R-hat only; no saved ESS or initialization sweep; numerical jitter escalation is not logged. Function defaults say ESS and HMC jitter1e-10, but executed callers select HMC and pass1e-7. | Do not confuse passing gates with calibrated uncertainty or optimizer convergence. |

## Existing fingerprint audit

Across the source/plot/input manifests in these four outputs, **29 entries
match and 3 differ** from the inspected working tree. No referenced path is
missing. The differences are:

| Manifest | Differing source |
| --- | --- |
| [49 source fingerprints](../../outputs/by_experiment/49_matched_native_grid_gp/source_fingerprints.json) | `src/matched_projection.py` |
| [52 source fingerprints](../../outputs/by_experiment/52_native_higher_ceiling/source_fingerprints.json) | `scripts/56_thermal_native_higher_ceiling.py`; `src/native_ceiling_diagnostics.py` |

Current `matched_projection.py` includes optional transport; scoped drivers do
not pass it. Current native ceiling helpers include retention used by output53.
The [52 plot fingerprint](../../outputs/by_experiment/52_native_higher_ceiling/plot_source_fingerprints.json)
matches today's driver56. These facts and saved replay evidence help explain
the chronology, but without old source snapshots they do **not** prove every
historical source change was harmless. The original manifests were preserved.

All [51 source/input fingerprints](../../outputs/by_experiment/51_native_forecast_attribution/source_fingerprints.json),
[53 source fingerprints](../../outputs/by_experiment/53_native_observation_retention/source_fingerprints.json)
and all six [53 input-cache fingerprints](../../outputs/by_experiment/53_native_observation_retention/input_cache_fingerprints.json)
match. Cache equality is narrower evidence than byte-identical historical code.

## Inspected source fingerprints

Paths below are relative to repository root. Full SHA-256 values supplement
the existing partial manifests without modifying them.

| File | SHA-256 |
| --- | --- |
| scripts/53_thermal_matched_projection_gp.py | `11754570b9f188c890d700fdd39887f9ee27c9f5bbe5461327f4e31ea5d4af7e` |
| scripts/55_thermal_native_forecast_attribution.py | `c12af11fabccdd00856818c02aa287292938f195b165dd6b62557cd4554a870e` |
| scripts/56_thermal_native_higher_ceiling.py | `87c5fa409988e782f287d1d406704935eb4f0c9ea94d247d3b545690aa200398` |
| scripts/57_thermal_native_observation_retention.py | `68314586837e89336b32d17389290bf272f1b362a8f4ddd0788d48bd06409902` |
| src/matched_projection.py | `eb804ff4422256e753ffbc08a618bc7547c22191af296ab6311f3e7f2939319b` |
| src/native_ceiling_diagnostics.py | `cb4672b12db7991f0083a04825677882208e445efe93484defb6bbec22b5f682` |
| src/pseudo_censoring_runner.py | `887e1b247936257a6d04cbe181bad8545e0aef2229f04b12960ad4d460d7ee6f` |
| src/thermal_posterior_physics.py | `c99fe3a5ffbaeedabb16e8b9fdee0ba3fcc8f19f791600bbb62ab10459ab4836` |
| src/censored_gp.py | `6fb7e71634abf7810ca43b21b7764dc6b76453be02b5c61940d3290136e8bd04` |
| src/dense_censored_gp.py | `e075a45362425a013651444097561c43f72fc65e368e12e334fd054a1a055034` |
| src/stochastic_heat_gp.py | `de5d51c2fd3d1243a815c69fadd6b826e47dbf9dc43f909f6d2dbc761c9a9087` |
| src/truncated_gaussian.py | `a2bb4b5fc9c0d8aeab34af3a031f96c02aa0ad21570feccd6b722f11fba66a6b` |
| src/diffusion.py | `f5d73b8c2d02ba0784585b9bda36bcbcde5403e97815e680182a6856ff737463` |
| outputs/by_experiment/49_matched_native_grid_gp/frozen_configuration.csv | `c7f15cf77dff969a9604f8a8186e856e120d2170e010280495e4f10d890c333e` |
