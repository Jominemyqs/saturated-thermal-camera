# Native C-only observation-retention control

## Question and scope

Does the exaggerated peak persist when newly revealed current measurements
are assimilated rather than dropped? This is four paired comparisons on two
development trajectories, at 320 and 325 K. No previous inference, coefficient
fit, physics change, new covariance or held-out sweep is performed.

For each case the previous hybrid draws, previous estimate, deterministic
physics mean and total prior covariance are taken from frozen output 52. The
covariance blocks needed at additional observation nodes are evaluated using
the same finite-step innovation and the same propagated cached deviations.
Only the set of current likelihood inputs changes. The resulting posterior
covariance is allowed to change; the PREDICTIVE PRIOR is held fixed.

## Observation policies

Existing: fixed numerical candidates that are currently unsaturated, plus all
currently observed-censored nodes.

Retained: O_star = numerical candidates UNION observed reference-camera
saturation. At every node of O_star use the high-ceiling numerical measurement
if unsaturated, or its lower-threshold censored likelihood otherwise. No node
contributes both likelihood types. Neither construction uses hidden truth.

All high-ceiling saturated nodes belong to O_star. Cameras use the same noisy
pre-clipping signal. Reclipping each high-ceiling camera reproduces the observed
reference camera exactly. Thus the current observations on this fixed set are
nested across the reference/320/325 ceilings. The two policies are compared
WITHIN each ceiling: previous inference and the prior still differ between
ceilings, as in output 52. This control is not an across-ceiling fixed-prior test.

| Case | Ceiling K | Existing numerical | Retained numerical | Censored, both |
|---|---:|---:|---:|---:|
| Diagonal | 320 | 117 | 1281 | 25 |
| Diagonal | 325 | 117 | 1291 | 15 |
| Spiral | 320 | 117 | 1337 | 20 |
| Spiral | 325 | 117 | 1346 | 11 |

The retained total count is 1306 diagonal and 1357 spiral at both ceilings,
exactly the corresponding reference observation set. This is retention of a
predefined mask, NOT use of every unsaturated camera value. No independent
upper-inequality factor is added to any numerical observation.

## Results: existing -> retained

Hot metrics use the identical fixed hottest 1% of physical area, with fractional
cutoff weight, not the hottest 25 adaptive-mesh nodes. Units are K except
coverage. Bias = estimate minus truth. CRPS uses the existing M(M-1) estimator;
its formal unbiasedness assumes independent draws, while these are MCMC draws.

| Case | Ceiling | Hot RMSE | Hot CRPS | Hot bias | Hot coverage % | Hot width |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal | 320 | 5.564 -> 2.457 | 2.295 -> 0.751 | 2.455 -> 0.450 | 72.6 -> 58.9 | 2.675 -> 0.431 |
| Diagonal | 325 | 5.981 -> 2.478 | 2.490 -> 0.767 | 2.645 -> 0.400 | 72.4 -> 56.3 | 2.644 -> 0.401 |
| Spiral | 320 | 5.784 -> 2.523 | 2.423 -> 0.747 | 2.582 -> 0.441 | 72.6 -> 69.1 | 2.642 -> 0.428 |
| Spiral | 325 | 6.095 -> 2.204 | 2.545 -> 0.660 | 2.703 -> 0.244 | 72.6 -> 70.5 | 2.643 -> 0.406 |

Hot-area RMSE improves in all four pairs (roughly 56-64%). Newly retained-node
RMSE falls from 1.697-2.485 K to 0.517-0.686 K. Assimilating the revealed values
therefore materially changes the current correction; the earlier sparse
high-ceiling posterior was not a sufficient test of how much the update could
repair the forecast.

Whole-field RMSE improves in all four pairs, but global CRPS is mixed:

| Case | Ceiling | Field RMSE | Whole-field CRPS |
|---|---:|---:|---:|
| Diagonal | 320 | 0.581 -> 0.340 | 0.1903 -> 0.1838 |
| Diagonal | 325 | 0.621 -> 0.370 | 0.1922 -> 0.1884 |
| Spiral | 320 | 0.598 -> 0.381 | 0.1893 -> 0.1911 |
| Spiral | 325 | 0.628 -> 0.439 | 0.1905 -> 0.2018 |

## The extreme peak still fails

At the true peak location (different from comparing maxima anywhere):

| Case | Ceiling | True K | Existing prediction K | Retained prediction K | Retained 95% interval K |
|---|---:|---:|---:|---:|---|
| Diagonal | 320 | 340.430 | 366.146 | 352.645 | [352.081, 353.227] |
| Diagonal | 325 | 340.430 | 367.647 | 353.720 | [353.416, 354.023] |
| Spiral | 320 | 341.182 | 376.237 | 363.900 | [363.437, 364.360] |
| Spiral | 325 | 341.182 | 376.809 | 362.123 | [361.827, 362.405] |

All four intervals still miss the true peak. Maximum-temperature absolute error
falls from 30.25-35.63 K to 18.14-22.72 K but is still substantial. Over the
remaining observed-censored region, retained coverage is 2.6% for diagonal at
320 K and 0% in the other three cases. Censored-region RMSE is reduced to
8.996-11.424 K from 18.457-25.126 K.

The current update can correct much of the revealed shoulder and reduce the
peak, but cannot remove the remaining peak mismatch under this prior. More
numerical conditioning narrows the intervals, so lower RMSE and CRPS do not
imply better coverage. This is not evidence that an inflation rule alone is
the appropriate remedy.

## Interpretation

Keep two conclusions separate:

1. Dropped current information was a substantial, correctable part of the
   high-ceiling reconstruction error. Retention is a cleaner observation policy
   for future higher-ceiling comparisons, but these two development cases do
   not establish held-out performance or warrant replacing every old result.
2. Forecast/covariance adequacy remains unresolved. The identical exaggerated
   prior peak persists as a smaller but severe posterior peak error even after
   the newly revealed measurements are assimilated. Return to the empirical
   gradient-based mean contribution and generating-model provenance before
   fitting a remedy. This control does not identify missing depth conduction,
   an optimal gamma, or a physically correct replacement source.

The inherited approximation remains: unsaturated previous noisy values are
pinned and censored marginal draws are sampled; these are not coherent draws
from a complete latent spatial posterior. The final likelihood conditions only
on current observations. Neither limitation was silently changed here.

## Verification and artifacts

- All four existing-policy replays are bit-for-bit equal to output 52 for
  posterior mean, SD, lower/upper limits and all 1200 evaluation draws.
- Rebuilt inherited and total prior variances equal the caches exactly.
  Prior means, previous draws, covariance features, cameras, seeds and fixed
  hot-area masks have identical hashes between each paired policy.
- Covariance blocks on overlapping observation locations are identical in the
  executed runs (maximum absolute difference 0). Additional blocks evaluate
  the same covariance, not a fitted replacement.
- Same current likelihood, noise scale, four-chain HMC settings and seeds as
  output 52. All R-hat maxima are below 1.008; no retry was needed. These checks
  establish replay/convergence under the frozen protocol, not calibration.
- 81 unit tests pass; source/scripts/tests compile; git diff --check passes.
- Each case, including its replay, takes about 30 s, approximately 120 s total
  before final plotting. Per-update covariance/inference timings are saved.

Files: results.csv; paired_retained_minus_existing.csv; verification.csv;
peak_summary.csv; newly_retained_region.csv; frozen_configuration.csv; input
cache/source fingerprints; per-case NPZ reconstructions; and along/across
cross-sections using the exact native nodes from output 49. Connecting bands
are visual guides between nodewise intervals, not off-mesh credible intervals.
Output 52 and its results are preserved, not overwritten.

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/57_thermal_native_observation_retention.py
```
