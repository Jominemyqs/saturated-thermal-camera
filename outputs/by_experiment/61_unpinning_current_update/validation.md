# Execution and validation record

Completed on 2026-11-03. Output 61 is an isolated current-update follow-up;
driver number 65 is not the output number. Working-tree base revision:
`b0d80e51f5772e1feaa5dc5c43f7613a97b1d3b7`. This revision alone does not identify
the uncommitted audit implementation: use the source/input fingerprint manifest.

## Reproduction and preservation

- [Frozen configuration](frozen_configuration.json) records the actual inherited
  coefficients, current sampling schedule, seeds, thresholds and gates.
- [Source/input fingerprints](source_and_input_fingerprints.json) include the
  driver, new helper/tests, shared inference code and previous-run inputs.
  All entries were checked before and after execution.
- [Prior verification](prior_verification.csv) has all 18 prior constructions.
  Legacy variances and means reproduce historical caches exactly; new inherited
  diagonals agree with output 60 within 1.56e-14 K^2. The full off-diagonal blocks
  are constructed independently and checked for symmetry and diagonal agreement.
- All six original-budget C replays reproduce saved mean, SD, lower/upper bounds
  and marginal draws exactly: reference output 52, retained 320/325 K output 53.
- All four longer-chain reference legacy fits (two trajectories, two seeds)
  additionally reproduce output 55's C0 arrays exactly, including marginal draws.
- Camera, retained observation IDs, gradient increment and Q hashes agree across
  the three arms within every case. Previous-state inputs were reused, not fit again.
- Existing output-59/60 files and their uncommitted code were preserved. No
  historical scientific driver, cache, coefficient or result was overwritten.
  No commit or push was made.

## Tests

Run from the repository root using its existing Python environment:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MPLCONFIGDIR=$PWD/tmp/matplotlib PYTHONPATH=$PWD/tmp/thermal_diagnostics \
  .venv/bin/python -m unittest discover -s tests -q
```

122 tests passed. One pre-existing constant-data skewness fixture emits a SciPy
precision-loss warning. The four new tests check explicit joint covariance
equivalence for pinned/full branches, sparse/dense agreement, a correlated case
that differs from a diagonal replacement, and exact reproduction of the old
lower-only current sampler. An initial pytest invocation could not run because
pytest is not installed; the repository's unittest suite above was used instead.

Compilation of the new helper, driver, tests and recovery script passed.
All 18 saved prior blocks were independently rehashed against the executed
verification table. Visual inspection covered Spiral reference cross-sections,
Diagonal 325 K cross-sections and Diagonal shared-scale signed-error/SD maps.
All 20 relative report links resolve. The reference summary table was checked
against the saved CSV at its displayed precision; all 24 fit gates and peak
coverage flags were checked independently. `git diff --check` passed. Large
posterior/prior NPZ caches and the temporary diagnostic packages remain ignored
by Git and are preserved locally.

## Environment interruption

The first execution stopped after seven saved fits. A temporary ArviZ directory
lost its package initializer and imported as an empty namespace, so diagnostic
evaluation failed after the unfinished Spiral pinned fit's chains had run.
This was not a statistical convergence failure. The cause of removal is unknown.

The same diagnostic versions (ArviZ 0.17.1, xarray 2024.3.0, xarray-einstats 0.7.0)
were restored without dependencies in ignored `tmp/thermal_diagnostics`.
No production virtual-environment packages or requirements were changed.
Diagnostics replayed on two completed fits agree at rtol 1e-12 / atol 1e-11.
The unfinished fit was rerun with its original seed and unchanged settings;
all saved fits were skipped and their hashes preserved. There were no statistical
retries or relaxed gates. See [recovery record](environment_recovery.json) and
[recovery orchestration](resume_after_environment_repair.py).

Reported successful-fit timings exclude compressed-cache writing after the
fit timer, the failed unsaved attempt and environment repair. Preparation,
historical replay, successful fitting and elapsed wall times are recorded
separately in [verification](verification.json) and [preparation](preparation.json).
Elapsed wall time includes interruption and is not a compute-cost benchmark.

## Scope limits

Two development trajectories, not held-out validation. Two paired seeds at
reference, one at each higher ceiling; high-ceiling small differences do not have
a repeat-based stability estimate. Convergence diagnostics assess numerical
sampling, not model calibration. Full covariance is used for current conditioning,
but final field draws are marginal-only and must not be propagated as joint draws.
Mean flooring, linear covariance propagation, Gaussian moment matching, empirical
gradient input and its deployment-availability limitation are unchanged.
