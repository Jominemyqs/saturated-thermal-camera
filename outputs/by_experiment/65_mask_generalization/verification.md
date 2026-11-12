# Verification and scope

## Preserved protocol

Source revision at freeze: `61438c90332e82c458809dfc148290a1a56f2662` plus the
existing uncommitted output-63/64 work. This task added driver 69, helper
`src/mask_generalization.py`, its tests, and output 65. Historical source,
coefficients, datasets, masks, cached outputs and uncommitted work were retained.
No git commit/push, refit, source replacement or larger flag budget was performed.

Trajectory selection was recorded before held-out inference in
`prospective_selection.csv`. All six reference U sets and the sole planned
320 K U set were generated before inspecting new predictions. The milder case
has exactly the same U, baseline IDs, reference-unused pool, physical weights,
underlying camera signals and evaluation mask as its reference case.

## Equality and information checks

- Development cameras and 256 IDs reproduce existing caches exactly.
- Independently regenerated development previous constrained chains are exact.
- Development baseline mean, covariance blocks and diagonal have zero maximum
  absolute difference from output 61/64.
- After the documented ordering correction, every extended prior block has zero
  difference from output 64, and actual development Cmask posterior summaries,
  marginal draws and noisy constraint draws reproduce output 64 exactly.
- The three valid held-out cases have identical prior mean/variance hashes across
  all arms and seeds. Baseline covariance sub-blocks are exactly preserved in
  aligned extended priors, including cross-covariances.
- All extended likelihood IDs match stored prior IDs. Sets are disjoint. The
  mask arm contains NaN numerical values at U; the numerical control contains
  the cached readings and no upper-flag factors at those locations.
- Historical source/data fingerprints were rechecked. Only the new driver,
  helper and regression test changed during the documented implementation fix.
  Their initial copies/hashes and corrected hashes are both retained.

No failed or interrupted mask draws are used as predictions. The first two
mask watchdog events and the interrupted Spiral 1 attempt preceded the ordering
fix and are invalid implementation attempts, not inference-performance evidence.
The independent predecessor stops remain as described in `case_status.csv`.

## Tests and reporting

- `python -m unittest discover -s tests`: **133 tests passed**. An existing
  nearly-constant-draw skewness test emits SciPy's precision-loss warning; no
  test failed.
- Added a regression test that rejects concatenated-prior/sorted-likelihood
  mismatch and verifies the corrected blocks against a direct dense covariance.
- Compiled new driver/helper/tests/reporting script; checked whitespace and links.
- Inspected `validated_comparisons.png`: labels and axes fit; only completed
  cases are plotted and the 3-of-6 limitation is explicit. Whiskers are paired
  seed ranges, not confidence intervals.
- `summarize.py` is reporting-only and rechecks covariance overlaps, likelihood
  types and numerical-reading exclusion at U before writing summary artifacts.

## Remaining limits

Only three of six selected cases completed. No 320 K current posterior was run;
prepared cameras alone do not establish flag redundancy. No trajectory was
replaced after a failed gate. Family summaries have only one completed case per
family and are descriptive, not population estimates.

Diagonal 1 did not finish its predeclared longer-chain check within budget.
Horizontal 1's longer chains passed, but its combined predecessor/preparation
stage timed out afterward; its valid previous cache is retained. Spiral 12
failed the inherited HMC feasibility check. None of these is evidence that the
mask improves or worsens that trajectory. No additional sampler fix or budget
extension was attempted.

The current source remains FEM-derived; native inputs retain interpolation in
propagation. Current prediction residuals remain marginal rather than coherent
joint-field draws. Two current seeds measure only limited Monte Carlo stability,
not sensitivity to independent previous posteriors or new camera realizations.
