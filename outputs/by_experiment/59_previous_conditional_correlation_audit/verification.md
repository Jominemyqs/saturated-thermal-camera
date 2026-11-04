# Verification: previous conditional correlations

Executed October 29, 2026. Source revision, package versions and the exact
inherited configuration are in `frozen_configuration.json`; code and six
output-52 cache hashes are in `source_fingerprints.json`.

## Scope and preservation

Only an isolated driver, helper, tests, output 59 and an inventory link were
added. Existing scientific source, coefficients, previous caches, current
reconstructions, mask results and archived smoothing branches were unchanged.
The source-fingerprint assertion checks every inspected numerical input after
the completed execution. The repository was clean at task start. No commit or
push was requested or performed.

The old protocol's grid top-25 definitions and the inherited configuration's
old model labels were not silently adopted or edited: this native audit uses
the cached current hot-1%-physical-area weights. The current likelihood is not
run, so neither historical observation policy nor retained-current policy is
changed. Previous observation IDs, sparse numerical IDs and censored IDs are
saved separately. Unsaturated pinned values use exactly the original readings.

## Numerical checks

- First original HMC chain replay in each of six cases: zero maximum draw
  difference. All cached previous-draw hashes match output-52 metadata.
- Independently constructed propagation matrix versus production action:
  maximum absolute difference below 1.78e-15 K.
- Recomputed original sample inherited variance versus cache: maximum absolute
  difference below 1.12e-14 K^2. Frozen Q diagonal checked against its source
  construction; physical forecast and predecessor means are held fixed.
- Analytic pre-propagation marginal variances are identical by construction;
  only conditional off-diagonal residual covariance changes.
- PSD factors add no variance/jitter. Roundoff-sized negative eigenvalues are
  clipped, with covariance perturbations below 1.12e-14 K^2. Substantive negative
  eigenvalues cause a hard failure.
- Actual joint empirical previous means differ by up to 0.01750 K from the
  cached hybrid mean. These are recorded, not propagated as a mean modification.
- The residual-covariance change is computed without residual Monte Carlo
  error. The common conditional-mean covariance still depends on saved MCMC;
  three residual repeats are explicitly NOT three independent posterior runs.
- Diagnostics monitor geometry-selected nodes, not all 6,275 locations.
  No convergence claim is made for a new full-latent or current posterior.

## Tests and environment

`python -m unittest discover -s tests`: 112 tests pass (7.590 s), including seven
new tests for exact conditioning/RNG recovery, rejecting the wrong sampler,
identity marginal preservation, positive/negative covariance effects, analytic
propagation, Monte Carlo joint covariance/pinning and non-PSD rejection.
The existing near-constant skewness test retains its precision warning.

`PYTHONPYCACHEPREFIX=/tmp/thermal_audit_pycache python -m compileall -q src scripts tests`
passes. An earlier attempt without a writable bytecode prefix failed solely
because the system cache directory was sandbox-protected; no source syntax
error was reported. `git diff --check` passes. The summary PNG was inspected
for labels, legends, axes and nonblank content.

The first scratch run under `/tmp/thermal_previous_correlation_audit_check`
stopped before producing scientific results because ArviZ was absent. It also
exposed NumPy Accelerate matmul warnings with finite outputs. Final isolated
code follows the existing SciPy BLAS pattern, tests finite products, and runs
without those warnings. Diagnostics dependencies were installed into `/tmp`,
with no change to the scientific virtual environment or requirements file.
Final timing excludes this environment setup and the stopped scratch run.

## Boundaries of the result

No current GP inference, parameter fit or coherent full-field previous
posterior was performed. The result is about inherited and total **prior**
marginal variance, not posterior coverage, interval width, CRPS or reconstruction
accuracy. The information-set audit required for unpinning unsaturated values
is documented as the next separate task, not treated as completed here.
