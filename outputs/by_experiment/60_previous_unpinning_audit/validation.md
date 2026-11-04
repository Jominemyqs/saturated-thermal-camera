# Validation

Run completed October 29, 2026. Working-tree base revision:
`b0d80e51f5772e1feaa5dc5c43f7613a97b1d3b7`. The previously uncommitted output-59
work was preserved. Exact source and input contents, rather than revision alone,
are identified by [source_fingerprints.json](source_fingerprints.json).

## Scope and invariants

- Six native cases completed: Diagonal 8 and Spiral 13 at reference, 320 and 325 K.
- Both reference gates passed before either higher-ceiling case was sampled.
- All saved previous-draw hashes match output 52's executed verification records.
- Historical deterministic forecasts reproduce exactly, element by element.
- Recomputed historical inherited variances differ by at most 1.155e-14 K^2.
- The constructed linear propagation map agrees with the existing operator to 8.882e-16 on independent probe fields. Sparse storage removes exact zeros only.
- Full noisy precision inversion check: maximum error 8.979e-13.
- All-readings pinned and latent arms have exactly identical censored marginal means and variances, using the same HMC draws.
- Propagated censored, unsaturated and signed cross contributions sum to the total inherited variance within asserted floating-point tolerances.
- No source, Q, camera, noise, coefficient, native geometry or area-weight changes.
- All fingerprinted inputs and executed source files remained unchanged during the run.
- No current posterior inference, refit, non-saturation flags, smoothing or held-out runs.

## Sampling and resources

All six cases passed without retries: four chains, 800 retained per chain,
300 burn-in, thinning 1. Worst rank-normalized split R-hat 1.007799; lowest
bulk ESS 674.25 and lowest tail ESS 768.94. Existing classical R-hat is also
recorded. Conditional joint residual monitor factorizations required no added
jitter or eigenvalue clipping. These checks do not establish calibration.

| Case | Ceiling | Case time (s) |
| --- | --- | ---: |
| Diagonal 8 | Reference | 91.94 |
| Spiral 13 | Reference | 229.53 |
| Diagonal 8 | 320 K | 6.23 |
| Spiral 13 | 320 K | 7.22 |
| Diagonal 8 | 325 K | 3.11 |
| Spiral 13 | 325 K | 5.73 |

Shared setup took 30.21 s, including one 6,275-dimensional dense precision
factorization/inversion reused across cases. Total driver runtime: 376.44 s
(6.27 min), including setup and reporting. Peak process RSS: 1623.48 MiB
(1.59 GiB). Per-case times include inference, moment calculation and cached
outputs, but exclude the subsequently rendered cross-section image. The
900 s / 4096 MiB gate was met with substantial margin.

One dense matrix occupies 300.41 MiB. The exact-zero sparse propagation map
has 577,515 stored entries (6.63 MiB). No expansion to larger native meshes
or thousands more constrained variables is implied by these timings.

## Tests and inspection

Executed:

```sh
.venv/bin/python -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/thermal_audit_pycache .venv/bin/python -m py_compile \
  scripts/64_thermal_previous_unpinning_audit.py src/previous_unpinning_audit.py
git diff --check
```

All 118 tests passed, including six new tests for precision-vs-Schur
conditioning, exclusion of saturated numerical values, conditional latent
covariance, dense/sparse propagation with cross terms, matched censored
marginals, and the all-numerical/small-noise limit. An existing constant-draw
skewness test emits a precision-loss warning; there were no failures.

The reference Spiral cross-section was visually inspected for legibility and
correct previous/current labels. Report links and numerical tables were checked
against saved artifacts. Large posterior caches are local ignored NPZ files;
no historical files were moved or overwritten.

The optional ArviZ environment was reused from
`/tmp/thermal_correlation_diagnostics` (ArviZ 0.17.1). Runtime versions and exact
sampler seeds are recorded in the frozen configuration. A clean environment
must supply compatible diagnostic dependencies; no package installation or
production dependency edit was made in this task.
