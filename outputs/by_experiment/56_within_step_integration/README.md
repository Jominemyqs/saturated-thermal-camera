# Experiment 56: Within-step gradient integration

## Decision

**Stop after deterministic Stage 1. No GP inference was performed.** Within-step
evolution reduces excessive peaks and improves hot-area RMSE, but no candidate
passes the frozen peak/shoulder/whole-field gate across both development cases.
This is a useful numerical sensitivity result, not a validated replacement mean.

The driver is [script 60](../../../scripts/60_thermal_within_step_integration.py),
not script 56 (which generated historical output 52). The isolated helper is
[within_step_integration.py](../../../src/within_step_integration.py). Historical
drivers, caches, coefficients and inference functions are unchanged.

## Frozen comparison

- DiagonalScanPath_8 and SpiralScanPath_13; 6,275 native surface nodes each.
- Cached reference ceilings 296.121498/296.490128 K, then 320 and 325 K.
- Exactly the previous means, cameras, geometry and hot-area weights from
  [output 52](../52_native_higher_ceiling/README.md). No camera regeneration or
  previous-state inference. Oracle controls use the true previous temperature.
- Stored previous/current times are 0.1770/0.1775 s and 0.2715/0.2720 s;
  dt = 0.0005000000000000004 s. Both gradient snapshots use the same native
  surface-node index array as temperature.
- alpha = 3.6109129986233175e-6 m^2/s; beta = 12.422901570529978 /s;
  gamma = 0.1702227206133784; gradient threshold = 10,000. No fitting.
- HeatFluxZ is the projected vertical temperature derivative according to the
  supplied notebook export. K/m is its supplied unit interpretation, not an
  independently verified HDF unit attribute. Gamma is empirical (m/s under this
  convention). Nothing here reinterprets the derivative as prescribed heating.
- In the unchanged predecessor operator, P1 lifting to the 601x401 fine grid
  precedes positive-excess flooring, Gaussian diffusion with zero extension,
  cooling and bilinear evaluation at native nodes. All variants share exactly
  the same propagated predecessor. Native inputs do not imply a 3D solver or
  interpolation-free propagation.
- The source uses that same linear positive-age operator **without flooring**.
  Thresholds are applied separately at the two recorded snapshots, never again
  after interpolation. No earlier-frame rollout is used.
- Noise, covariance and likelihood settings are untouched; no likelihood is
  evaluated. A conditional GP test would require output 53's retained policy,
  not output 52's high-ceiling observation subset. The failed gate prevents that
  step, so no mask or numerical-retention intervention is mixed into this run.

Inherited CSV options include stale grid/top-25 and calibration labels. They
are preserved as provenance, not adopted as this experiment's evaluation rule.
Here hot1% is **one percent of physical area**, with a fractional final node.

## Variants

Let q_previous and q_current be separately thresholded gradients and A(t) the
frozen diffusion/cooling/interpolation map. All means are ambient + propagated
previous excess + the following increment:

| Label | Increment |
|---|---|
| S0 | gamma * dt * q_current (historical endpoint) |
| S_time | gamma * dt/2 * (q_previous + q_current) |
| S_evolve | gamma * dt/2 * (A(dt)q_current + q_current) |
| S_both | gamma * dt/2 * (A(dt)q_previous + q_current) |
| S_GL2, S_GL4 | 2-/4-point Gauss-Legendre integration of linearly interpolated thresholded endpoints, propagated for the remaining time |

Endpoint zero-age contributions remain direct native values, as in S0. These
are numerical surrogates, not exact physical source or boundary-condition models.

## Results

**Inferred predecessor, deterministic forecast only.** Entries below are
area-weighted hot1% RMSE (K), not posterior RMSE or CRPS.

| Case | Ceiling (K) | S0 | Time only | Evolution only | Both, trapezoid | Both, GL4 |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | 296.12 | 4.535 | 3.032 | 4.009 | 2.558 | 2.627 |
| Diagonal 8 | 320 | 5.574 | 6.326 | 4.941 | 5.670 | 5.733 |
| Diagonal 8 | 325 | 5.981 | 6.866 | 5.377 | 6.208 | 6.272 |
| Spiral 13 | 296.49 | 3.907 | 2.857 | 3.318 | 2.297 | 2.414 |
| Spiral 13 | 320 | 5.779 | 6.628 | 5.166 | 5.969 | 6.034 |
| Spiral 13 | 325 | 6.095 | 7.049 | 5.504 | 6.381 | 6.450 |

At reference, true-peak signed errors change:

| Case | S0 | Time only | Evolution only | Both, trapezoid | Both, GL4 |
|---|---:|---:|---:|---:|---:|
| Diagonal 8 | +9.964 | +2.392 | +5.292 | +0.387 | -0.823 |
| Spiral 13 | +21.691 | +8.691 | +10.140 | +3.795 | -2.641 |

### A. Temporal variation

Temporal averaging shifts the increment centroid toward the earlier gradient
snapshot by about 0.071 mm (Diagonal) and 0.074 mm (Spiral). Its area integral
changes by only +0.070% and -0.029%. This is principally redistribution, not
reducing total amplitude. It helps the reference inferred forecast but worsens
hot-area RMSE at both higher ceilings and with an oracle predecessor.

### B. Within-step evolution

Evolution-only reduces hot-area RMSE and true-peak absolute error in every
inferred case, as well as oracle hot-area and whole-field RMSE. But shoulder
improvement is not uniform. At reference, Spiral's across-section shoulder
RMSE rises **0.444 -> 0.664 K**; at 320/325 K it rises **0.787 -> 0.998** and
**0.797 -> 1.008 K**. The oracle across-section RMSE rises **0.783 -> 1.007 K**.
These exceed the frozen 0.10 K or 5% shoulder tolerance. The small positive
Diagonal along-section change (2.794 -> 2.887 K at reference) stays within it.

Shoulders are saved cross-section nodes with **0.25 <= |along_mm| <= 1.0**,
fixed before computing variants, weighted using their native lumped areas.
They are section-node diagnostics, not continuous line integrals or an exhaustive
definition of the thermal wake. Both along- and cross-direction errors matter.
The scope/tolerance choice is explicit; the gate is not a significance test.

### C. Combined effect and error compensation

Combined integration gives the largest reference peak correction, but it is not
uniformly better than evolution-only. GL4 increases hot-area RMSE at 320/325 K
and on both oracle controls. Oracle hot-area RMSE is **6.854 -> 7.392 K** for
Diagonal and **6.650 -> 7.138 K** for Spiral, despite lower peak errors. In
contrast, evolution-only reduces oracle RMSE to **6.307/6.091 K**.

Thus the strong reference improvement from temporal averaging does not transfer
unchanged to better-known predecessors. Reduced pointwise peak error is not
equivalent to improved spatial shape. The oracle forecast is exactly identical
across ceilings; its repetitions are controls, not independent replicates.

### D. Redistribution rather than reduced total increment

For GL4 versus S0, integrated temperature contribution changes by only
**-0.185% / -0.341%**, while RMS source radius increases
**0.182 -> 0.206 mm / 0.184 -> 0.211 mm** (Diagonal / Spiral).
The integral is in K*m^2, **not physical energy**. Source maxima, centroids,
areas above a fixed 1 K increment and radii are in
[source_component_comparison.csv](source_component_comparison.csv).

### E. Quadrature and spatial accuracy

- GL2 versus GL4: relative source L2 differences **0.225% / 0.221%**;
  maximum node differences **0.114 / 0.128 K**; hot-area RMSE differences
  at most **0.00285 K**. This supports quadrature stability at the frozen grid,
  conditional on linear interpolation of thresholded snapshots.
- Combined trapezoid versus GL4: relative source differences **8.38% / 8.49%**,
  maximum node differences **5.335 / 6.436 K**. The trapezoid is not definitive
  for peak amplitude; its apparently excellent reference point error should
  not be substituted for the better-resolved integral.
- Source-only refinement to 1201x801 and 1801x1201 leaves the predecessor fixed.
  Hot-area RMSE shifts are at most **0.032 K**. GL source L2 changes reach
  **1.20%**, narrowly exceeding the predeclared 1% spatial tolerance; trapezoidal
  evolution changes are below 0.41%. No new grid was selected or substituted.
- **Known operator caveat:** the existing P1-lift/bilinear-return map is not an
  exact identity on arbitrary native fields even at zero elapsed time. Applied
  to q_current and scaled by gamma*dt, pure remapping alone changes individual
  increments by up to **4.725 / 5.758 K**. Therefore part of the numerical
  spreading cannot be attributed solely to physical diffusion. We did not fix
  this operator or redefine the historical baseline. The no-diffusion identity
  tests pass for an identity/matching-node map, not arbitrary off-grid fields;
  the latter limitation is explicitly tested and recorded.

## Gate and stopping rule

[frozen_configuration.json](frozen_configuration.json) was saved before thermal
variants were computed. Every inferred case must improve hot-area RMSE by at
least 5% and absolute true-peak error by at least 1 K; shoulders may worsen by
at most max(5%, 0.10 K), whole-field RMSE by max(2%, 0.02 K). Oracle hot-area
RMSE must not worsen by more than max(2%, 0.10 K), nor absolute peak error by
more than 0.25 K. Quadrature/refinement criteria are recorded there too.

No candidate passes across both paths and all ceilings. Evolution-only passes
the individual-case gate for Diagonal but not Spiral; combined variants also
fail oracle/high-ceiling shape checks. **No current GP inference, new mean
selection, gamma refit or covariance change follows this result.** This does
not establish that integration is useless; it establishes a peak/shoulder
tradeoff under the frozen empirical predictor and numerical map.

## Files and reproduction

- [deterministic_metrics.csv](deterministic_metrics.csv): all regions, states,
  variants and ceilings; signed/absolute true-peak errors and forecast maxima.
- [oracle_vs_inferred.csv](oracle_vs_inferred.csv),
  [quadrature_checks.csv](quadrature_checks.csv),
  [spatial_refinement.csv](spatial_refinement.csv), [gate_checks.csv](gate_checks.csv).
- [equality_checks.csv](equality_checks.csv), [verification.json](verification.json),
  [gradient_provenance.json](gradient_provenance.json),
  [source_fingerprints.json](source_fingerprints.json), [validation.json](validation.json).
- `by_trajectory/*/level_*.npz`: native inputs, previous means, every increment,
  propagation-only and complete forecasts, truth and fixed evaluation weights.
- `cross_sections_*`, `source_sections_*`, `error_maps_*`: consistent scales
  across variants, ceilings and predecessor types within each trajectory.
  There are no posterior bands because there was no new posterior inference.

Example: [Diagonal reference sections](cross_sections_DiagonalScanPath_8_level0_inferred.png),
[Spiral reference sections](cross_sections_SpiralScanPath_13_level0_inferred.png),
[Spiral 325 K sections](cross_sections_SpiralScanPath_13_level2_inferred.png).

From repository root, use the existing `.venv` with single-threaded BLAS:

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/60_thermal_within_step_integration.py
```

The driver refuses to overwrite an existing output directory; reproduce in a
separate checkout/output copy, preserving this run. Runtime was **25.83 s**
including plots/caches, with **1,265 MiB** peak process RSS. Numerical work plus
cache handling took 4.48/4.94 s per trajectory; the rest was mostly plotting.
This is not an inference-runtime estimate. All baseline/source arrays match
exactly, and all oracle forecasts are ceiling-invariant. No remedy or further
model branch was launched.
