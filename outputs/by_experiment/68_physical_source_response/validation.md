# Validation record

- Output 68 is new; driver 72 and two new helpers/tests are isolated additions.
  No historical scientific code, configuration, cache or result was edited.
- `fingerprints.json` records the working-tree revision/status plus hashes of
  executed pilot code, source dependencies, notebooks, XDMF/HDF inputs and
  inherited caches. All were rechecked after GP execution.
- `reporting_verification.json` records the evaluation-only reporting helper,
  its test, and independent response caches. Updating reporting did not alter
  inference code or frozen scientific controls.
- Geometry-only FEM inputs reject unexpected fields. Unit tests verify the
  assembly receives zero initial values and selects only the source-response
  solve with homogeneous bottom increments. Temperature/gradient values are
  absent from the input payload. The worker never opens thermal-state caches.
- Native surface coordinates, ordering, triangles and area weights match the
  cached current-update geometry exactly. The response is constant across
  ceilings for each trajectory. The original source/physics parameters are
  not recalibrated.
- Independent response vs Output67 on-minus-off: maximum 1.52e-12 K.
  New oracle error vs prior propagation discrepancy: maximum 5.29e-12 K.
  Independent linear-system relative residuals: 1.42e-15 and 8.25e-16.
- All six primary baseline replay posterior summaries, draws, monitored chains
  and noisy constrained chains are array-exact against Output61.
- All observed covariance, cross-covariance and prediction-variance blocks
  remain array-identical within pairs. Both prediction and observation means
  use the same replacement. No mask or duplicate source was added.
- All eight new fits completed and passed the original gates, with no retries.
  Two paired reference seeds agree on the adverse hot-area RMSE/CRPS direction.

## Tests and checks

GP environment:

```
.venv/bin/python -m unittest \
  tests.test_physical_source_response tests.test_physical_response_reporting \
  tests.test_fem_replay tests.test_source_off_audit \
  tests.test_prescribed_laser_audit tests.test_unpinning_current_update
```

23 tests: 22 passed, one FEM-environment-only test skipped. In `tmp/fem_env`,
`tests.test_fem_replay` and `tests.test_physical_source_response` ran nine tests,
all passed, including the previously skipped constant-state/linear-response
FEM test. The new reporting pass also checks the remaining-error decomposition
on every pixel/case and confirms the separate Gaussian CRPS formula.

Compilation passed with `PYTHONPYCACHEPREFIX` inside the repository's `tmp/`.
`git diff --check` passed; no-index whitespace checks of new scientific files
emitted no errors (exit 1 reflects new-file differences). Relative report links
were checked. Reference and high-ceiling cross-sections and a shared-scale
signed-error map were visually inspected; axes, labels and legends are readable.

The initial ArviZ import probe needed the existing helper's SciPy compatibility
alias before import; this was corrected before preparation and inference.
No library or sampler implementation was changed. During source-integral
reporting, NumPy/BLAS emitted dot-product floating-point warnings. All four
reported integrals were finite and agreed with explicit elementwise reductions
to relative tolerance 1e-14; these reductions are saved in the reporting check.
The original GP numerical path was untouched and its baseline replay is exact.

## Limits

Two development trajectories only; no claim of held-out generalization.
One primary current seed at high ceilings; two at reference. Diagnostics assess
sampling, not calibration or physical correctness. Final field draws retain
the historical pointwise-marginal approximation. The deterministic oracle uses
true surface state only; full-volume truth is confined to the earlier replay
and subsequent evaluation, never the candidate predictor.
