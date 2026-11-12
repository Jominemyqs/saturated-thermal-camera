# Verification record

## Scope and preservation

Output 64, driver 68, is an isolated source-on/full-latent mask pilot. Existing
uncommitted output-63 source-off code, tests and results were present at task
start and remain untouched. No scientific baseline, coefficient, original
driver or historical output is overwritten. Git revision and file-level
fingerprints are in the frozen configuration and source/input manifest.

The output-61 manifest is checked before preparation. New fingerprints cover
the old selection, previous constrained chains, current cases and priors,
baseline posterior caches, shared code, configuration and plotting sections.
Final verification rechecks the manifest after sampling.

## Covariance and information checks

- Selection NPZ and CSV files are copied byte-for-byte from output 54.
- Reference baseline observation IDs, native coordinates and area weights match
  the old selection. All extra nodes belong to the fixed unused pool.
- Reference and higher-ceiling cached masks confirm these nodes remain
  unsaturated. No omitted reading magnitude chooses U.
- Added blocks are evaluated using output 61's full-latent analytic covariance,
  including unsaturated-censored cross terms; previous inference is not rerun.
- Recomputed predictive diagonals match cached diagonals exactly for both
  trajectories. Maximum overlap reconciliation is 1.1102230246251565e-16 K^2.
  Cached baseline observation/cross blocks are preserved exactly.
- Source-on means and the cached gradient increment match output 60/61.
- Cmask cannot accept extra numerical readings through its layout API.
  Likelihood sets are disjoint; numerical controls never receive upper flags.
- Current likelihood noise stays 0.50 K; camera noise remains 0.25 K.

Preparation took 77.45 seconds, with peak RSS 1,402.94 MiB and previous noisy
precision inverse residual 8.98e-13. These costs are covariance construction,
not previous-state posterior sampling.

## Tests

`python -m unittest discover -s tests`: 130 tests passed in 4.928 seconds.
The existing constant-array skewness test emits a precision-loss warning.
Compilation of the new driver/helper/tests passed with a workspace-local
`PYTHONPYCACHEPREFIX`.

Four new tests cover exact baseline-block preservation/full joint-block
agreement, rejection of pinned covariance substitution, no-addition identity,
and disjoint mask/numerical layouts that reject mask numerical leakage.
Existing noisy mixed-bound, analytic conditional, sign-transform and sampler
tests remain unchanged and pass.

## Completed checks

All ten fits passed every prospective gate with no retries. Four C0 paired
controls reproduced output 61 posterior summaries, evaluation draws, noisy
constrained chains and monitored latent chains exactly. The final manifest
check confirmed unchanged source/input files. Independent cache inspection
also confirmed exact baseline mean/diagonal/old covariance blocks and disjoint
likelihood layouts, including NaN readings at mask-only locations.
All README relative links resolve, copied selection files match byte-for-byte,
and `git diff --check` plus explicit new-code whitespace checks pass. Only the
new output-64 driver/helper/tests and outputs were added in this task; the
pre-existing uncommitted output-63 work remains unchanged.

Visual QA inspected the Spiral unused-region signed-error/SD map and Diagonal
cross-sections. Row-shared map scales, labels, legends and uncertainty bands
render correctly. Native-node cross-section connecting lines do not establish
a continuous off-mesh posterior or simultaneous credible band.

The primary and repeat metric tables agree on unused-region RMSE/CRPS gains
beyond U, on small Diagonal hot-area deterioration, and on Spiral hot-area
point/CRPS improvement. All ten true-peak coverage flags are false. Reported
runtime separates fit workers from full-run wall time and preparation.

## Interpretation limits

This is a two-development-case reference-ceiling comparison. New-baseline
320/325 K flag compatibility is computed from cached latent marginal draws
only; no high-ceiling posterior comparison is claimed. No generalization,
larger flag budget, source replacement or automatic follow-on is run.
Sampling gates assess numerical reliability, not physical calibration.
Prediction draws are marginal-only and spatial nodes are not independent
replicates. See the README for the inherited CRPS convention and limitations.
