# Verification record

Inspected revision before the new pilot:
`d28f8d86f944728bb57fa0926a4b66b10f055aba` (2026-10-08 working tree).
Initial uncommitted change was `.DS_Store`; it was preserved. An unrelated
untracked `src.zip` appeared during the task and was also left untouched.

## Scope and preservation

- Added only driver 58, `src/native_mask_pilot.py`, its unit tests, output 54,
  and the requested pre-change inspection ZIP. No historical scientific code,
  configuration, result, cache, or archive was moved, renamed, or deleted.
- ZIP was made before experimental code edits; its internal manifest records
  the original working-tree status and SHA-256 of 13 copied files. Copies are
  byte-identical to their unchanged source files. The archive is code-reading
  material, not a standalone execution package.
- Saved [source hashes](source_fingerprints.json) identify the exact new driver,
  helper, propagation/covariance/sampler code and frozen configuration used.
  [Input hashes](input_fingerprints.json) cover the reused output-52/53 caches.
- No email, commit, push, held-out expansion, previous-state inference, model
  fitting, or high-ceiling update was performed.

## Mathematical and input checks

- Both primary reference C0 predictions reproduce output 52 C exactly, including
  posterior mean, SD, interval limits, and all 1,200 evaluation marginal draws.
  The output-52 reference is the historical native output-49 reference path.
  Higher-ceiling output-53 replay remains unexecuted because the gate failed.
- Propagated inherited variances and full prior diagonals reproduce output 52
  bit-for-bit; previous-draw hashes agree with its verification records.
- Additional covariance blocks use exactly the same `Q_dt + H H^T` construction.
  Overlapping observation/prediction blocks pass `rtol=atol=1e-13` checks.
- All arms within a case have identical predictive mean/diagonal hashes and
  paired current seeds. H, Q, cached cameras, and physical coefficients are
  common; only likelihood inputs change.
- Selection contains 256 unique physical-geometry-selected nodes outside the
  retained set; the common cached pre-clipping signal is identical across
  reference/320/325 K. Selected nodes are observed-unsaturated at all three.
- Saved observation IDs are unique and retain every baseline node. Numerical
  values equal the cached readings only where likelihood kind is numerical;
  every upper/lower entry has NaN in that field. Cmask does not receive the
  numerical readings at U; Cnumeric does not receive their flag factors.
- Evaluation-chain shape is `(4, 300, 6275)` for every fit. Monitored latent and
  noisy-constraint chains retain all 400 post-burn-in states and chain IDs.
- During preflight, reconstructing dt from a text CSV produced a 3.1e-15-level
  inherited-variance mismatch before any sampling. The driver now reads the
  original stored trajectory timestamps, exactly as the historical driver does.
  Equality checks were not relaxed, and no coefficients were changed.

Exact-replay array SHA-256 (C-contiguous array bytes; identical old/new):

| Case | Array | SHA-256 |
|---|---|---|
| Diagonal 8 | posterior mean | `6d0b34abf1aca692482c247a7fcbc2cb19de16c3b829aa6ea2ad79e436f47f42` |
| Diagonal 8 | evaluation draws | `158753bd8429d854adca136368656e16c5894f480bed7868040f27be5bfd8ea4` |
| Spiral 13 | posterior mean | `1717e1eb43bb575bdcec7a4d8a2e73914af53b788ffb7d006e2e4dfd4bf438d7` |
| Spiral 13 | evaluation draws | `253b9a8f2ba121ebae81ab4e1a1798c20b960eeb719830c5747463c21c16dc12` |

Prior-mean/diagonal hashes are in every row of
`reference_convergence_runtime.csv`; previous-draw/mean and propagated-feature
hashes are in each `prior_level_0.json`. End-of-task verification rechecked all
source/input fingerprints and the 13 snapshot members against current files.

## Tests and visual checks

`python -m unittest discover -s tests -q`: **88 tests passed** (last run 5.416 s).
Seven new tests cover old-path exact equality, numerical quadrature for noisy
one-dimensional upper/lower cases, a correlated mixed-bound independent
rejection check, sign-transformed cross-covariances, remote-upper-bound
negligibility, observation disjointness/value exclusion, and deterministic
physical-coordinate selection. Stochastic checks use finite Monte Carlo
tolerances and do not require truth-based reconstruction improvement.

The existing near-constant-array skewness test emits its known SciPy precision
warning; it passes. New Python files compile successfully. All four generated
figures were visually inspected: shared scales and labels are readable, and
native section nodes/95% marginal bands are preserved. Background panels mask
any triangle touching an excluded node; their checkerboard gaps are display
masking, not zero uncertainty or missing inference. Black dots show U.

`git diff --check` passed. Because new files are untracked, their whitespace was
also checked explicitly with `git diff --no-index --check /dev/null <file>`.
All 24 Markdown links were checked: local targets resolve; external documentation
URLs are references rather than locally validated files. New compressed NPZ
caches occupy approximately 1,463.6 MiB locally and were not put in the ZIP.

## Reliability decision

[The preset gate failed](reference_gate.json); no thresholds were weakened.
All classical screens passed without retry. Two rank-normalized split R-hat
maxima narrowly exceed 1.01, and one numerical-control bulk ESS is below 400.
No tail ESS or runtime/RSS threshold failed. Repeats support the unused-region
point/CRPS gains, but do not establish convergence or calibrated uncertainty.
Spiral hot-area coverage changes are smaller than repeat variation and have
different signs across seeds; report them as inconclusive.

The pilot therefore stops at the reference ceiling. Secondary high-ceiling
effects and full-mask scalability have not been measured. Existing hybrid
previous fields and marginal-only field draws remain approximation limitations;
this task changes neither representation.

## Environment

Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1; single-thread BLAS environment as
recorded in the README command. ArviZ 0.17.1 and its diagnostic dependencies were
installed only in a temporary target directory, not the project virtualenv.
The diagnostic import compatibility alias changes no inference or ESS/R-hat
algorithm. Numerical masks, priors, and seeds were not adjusted for performance.
