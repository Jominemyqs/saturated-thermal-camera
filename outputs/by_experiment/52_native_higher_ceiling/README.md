# Native-input higher-ceiling comparison

## Scope and main result

Adrienne requested the reference/320/325 K cross-sections and signed-error maps.
This is a frozen-model pilot on DiagonalScanPath_8 and SpiralScanPath_13, both
development trajectories, not a held-out model ranking. No physical coefficient,
covariance, source threshold, inference method or calibration is changed.

Higher ceilings reveal the shoulder, but the unchanged predictor generates a
larger exaggerated peak. The physics hot-area bias changes from -0.798 to
+2.621 K (diagonal) and +0.332 to +2.707 K (spiral). The better-observed previous
state removes some of the negative input error that previously compensated for
the positive oracle-forecast error. This is consistent with output 51's
attribution, not proof of which physical term should replace the empirical
gradient-based increment.

## Architecture and protocol

- B: posterior physics mean plus a spatial RBF discrepancy.
- C: the same posterior physics mean plus finite-step heat innovation Q_dt and
  covariance of propagated previous deviations. No advection in either model.
- One previous hybrid posterior is recomputed at each ceiling and shared B/C:
  censored marginal draws are sampled; all unsaturated noisy predecessor values
  are pinned. These are NOT coherent full latent spatial posterior samples.
- Final likelihood uses only the current frame. No repeated previous likelihood.
- Both cameras are reclipped from the same original truth-plus-noise signals,
  regenerated with the original seeds. A higher-ceiling camera cannot be
  obtained from the already-clipped reference camera.
- Same 6275 native surface nodes, geometry-selected numerical candidates,
  area weights, target time, lengthscales, coefficients, source field, seeds,
  samplers and 601 x 401 computational propagation grid as output 49.
- Selected numerical observations are stride-5 grid candidates mapped to
  nearest native nodes, restricted to currently unsaturated nodes; all
  observed-censored native nodes contribute inequalities. Truth is not used to
  choose observations. Native truth defines evaluation regions/sections only.
- Four chains with 400 retained / 100 burn-in, the existing retry gate at
  R-hat 1.05, and 300 evaluation draws per chain. No retry was required.

IMPORTANT: this fixed sampling rule does not yield nested current likelihoods
as the ceiling rises. A newly unsaturated node outside the numerical candidates
loses its lower inequality and its exact numerical value is not used. The
experiment therefore tests the whole frozen camera/subsampling/forecast
pipeline. It is NOT a demonstration that adding exact measurements worsens an
otherwise fixed Bayesian posterior. No upper-inequality likelihood or denser
numerical sampling has been added here. The previous hybrid representation also
changes which locations are pinned versus sampled when the ceiling changes.

## Censoring severity

These are fractions of physical surface area, not adaptive-mesh node counts.
Hidden peak rise is (true maximum - ceiling)/(true maximum - ambient), evaluation
only. Every case uses the same underlying physical field.

| Case | Ceiling K | Censored area % | Hidden peak rise % | Numerical inputs | Censored inputs |
|---|---:|---:|---:|---:|---:|
| Diagonal | 296.121 | 3.121 | 93.7 | 115 | 1191 |
| Diagonal | 320 | 0.066 | 43.2 | 117 | 25 |
| Diagonal | 325 | 0.040 | 32.6 | 117 | 15 |
| Spiral | 296.490 | 3.244 | 93.0 | 114 | 1243 |
| Spiral | 320 | 0.052 | 44.1 | 117 | 20 |
| Spiral | 325 | 0.028 | 33.7 | 117 | 11 |

## Fixed hot-area results

The evaluation region is the same hottest 1% of physical area at every ceiling,
with a fractional cutoff weight. It is not the top 25 nodes, and at high ceilings
it is much larger than the remaining observed-censored region. Bias is estimate
minus truth. CRPS uses the existing distinct-pair M(M-1) estimator (its formal
unbiasedness assumes independent posterior draws; the inference remains MCMC).
Coverage always accompanies interval width. Units are K except coverage.

| Case | Ceiling | Model | RMSE | CRPS | Bias | Coverage % | Width |
|---|---:|---|---:|---:|---:|---:|---:|
| Diagonal | 296.121 | B | 4.772 | 2.481 | 1.721 | 84.5 | 10.090 |
| Diagonal | 296.121 | C | 4.409 | 2.248 | -0.098 | 77.3 | 6.369 |
| Diagonal | 320 | B | 5.559 | 2.344 | 2.474 | 86.6 | 12.028 |
| Diagonal | 320 | C | 5.564 | 2.295 | 2.455 | 72.6 | 2.675 |
| Diagonal | 325 | B | 5.960 | 2.524 | 2.654 | 86.3 | 12.137 |
| Diagonal | 325 | C | 5.981 | 2.490 | 2.645 | 72.4 | 2.644 |
| Spiral | 296.490 | B | 5.127 | 2.882 | 3.320 | 70.1 | 10.117 |
| Spiral | 296.490 | C | 4.074 | 2.155 | 1.317 | 76.4 | 6.424 |
| Spiral | 320 | B | 5.820 | 2.470 | 2.601 | 86.6 | 12.132 |
| Spiral | 320 | C | 5.784 | 2.423 | 2.582 | 72.6 | 2.642 |
| Spiral | 325 | B | 6.096 | 2.571 | 2.702 | 86.1 | 12.189 |
| Spiral | 325 | C | 6.095 | 2.545 | 2.703 | 72.6 | 2.643 |

All four model/path pairs worsen hot-area RMSE at 320 and 325 K versus reference.
B's broader intervals and better recovered shoulder can improve aggregate
hot-area CRPS/coverage without repairing the much narrower extreme peak.
The remaining observed-censored pixels have ZERO 95% coverage for both models
at both high ceilings in these two cases. That changing region must not be
confused with the fixed hottest-1%-area region above.

At the true peak node, C's mean increases from 350.885 to 367.647 K (diagonal,
truth 340.430 K), and 364.074 to 376.809 K (spiral, truth 341.182 K), between
reference and 325 K. The corresponding previous estimates increase toward the
known previous truth. The numerical gradient increments at those locations
remain exactly 50.249 and 62.312 K. These values explain why revealing the
predecessor can improve the shoulder yet expose a larger peak overshoot.

## Spatial uncertainty question

`uncertainty_distance_*.png` and `distance_summary.csv` compare SD, width,
actual error and coverage in observed-censored pixels against physical distance
to the nearest RETAINED uncensored numerical input. Omitted unsaturated pixels
are not treated as numerical observations for this diagnostic.

At the reference ceiling, binned C SD rises from about 0.396 to 1.601 K across
the diagonal distance bins, and 0.327 to 1.666 K for spiral. B also rises. Thus
these summaries do not support a universal decrease of uncertainty with
distance from the actual numerical inputs. They do not establish adequate
calibration: the actual RMSE often substantially exceeds posterior SD.

At high ceilings there are only 11-25 censored nodes, occupying one or two
distance bins. There is not enough spatial support to infer a detailed distance
law. Their positive peak errors overwhelm the intervals. Counts and physical
area per bin are saved; no IID spatial hypothesis tests or fitted correction
are used. Changing-region summaries are descriptive, not matched-population
causal effects. This spatial question is separate from width versus ceiling.

## Figures and saved objects

- `cross_sections_<case>_along.png` / `_across.png`: same native section nodes
  as output 49; truth, unchanged-form physics forecast, posterior mean, 95%
  marginal interval and ceiling. B/C share physical locations and temperature
  scales. Nearest-node offsets are retained in the section CSV; connecting
  bands do not constitute off-mesh or simultaneous credible intervals.
- `signed_errors_<case>.png` and `signed_errors_zoom_<case>.png`: forecast and
  B/C error, one shared symmetric colour scale across both paths and all levels.
  Zooms use a 3 mm window around the truth peak for evaluation only.
- `results.csv`, `hot_area_summary.csv`, `physics_forecast_summary.csv`,
  `true_peak_node_summary.csv`: complementary regional and peak metrics.
- `by_trajectory/<case>/level_<j>.npz`: both pre-clipping noisy signals,
  cameras, masks/IDs, previous mean and hybrid draws, deterministic components,
  B/C marginal draws and summaries, total/inherited prior variance, distances,
  truth and fixed area weights. Cached for analysis without another GP run.
- `PHYSICS_PROVENANCE.md`: HeatFluxZ reinterpretation, thesis equations and
  boundary conditions, and unresolved timestamp/discretization provenance.

## Verification and reproduction

All four native reference B/C rows reproduce output 49 bit-for-bit: previous
means/draw hashes, camera arrays, observation IDs, physics means, final means,
SDs, interval limits and all posterior evaluation draws. Across ceilings the
pre-clipping signals, candidate mask, hot-area weights, gradient term, seeds
and lengthscale hashes are unchanged; B/C share previous draws and current
likelihood data at each ceiling. The previous posterior is not reused across
different ceilings. All 33 trajectories load and their stored timestamps are
audited; no held-out trajectory participates in fitting or this two-case pilot.

Maximum previous/current R-hat values are 1.00919/1.00935. These diagnostics do
not prove posterior calibration or validate the hybrid approximation. Total
recorded previous-plus-model computation across six cases is about 150 s,
excluding setup, cache writing and plotting; see per-fit verification timings.
80 unit tests pass, including the new camera/distance tests. Compilation and
git diff whitespace checks pass.

Use the original single-thread BLAS settings for exact numerical replay. An
initial unpinned invocation stopped at the equality guard (maximum predecessor
mean difference 2.84e-13 K); rerunning under the documented environment restored
bit-for-bit equality. The driver now refuses an unpinned inference launch.

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/56_thermal_native_higher_ceiling.py
```

Use `--plots-only` to rebuild figures/tables from the saved arrays. Nothing here
refits gamma, converts the gradient to watts, modifies Q, adds advection, or
establishes missing vertical conduction as the cause of the residual mismatch.
