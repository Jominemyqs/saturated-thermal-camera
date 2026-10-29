# Experiment 58: Source amplitude versus spatial shape

## Bottom line

Reducing the endpoint contribution helps the **oracle-state predictor** on all
30 held-out trajectories. Adding combined GL4 redistribution does not improve
whole-field, hot-area or shoulder-annulus RMSE beyond amplitude reduction alone
in this bounded comparison. However, the same reduced amplitudes do not safely
transfer to heavily censored inferred predecessors. There is evidence of both
amplitude sensitivity and incompatible regional/state errors, not a universally
validated replacement gamma. **No GP inference or production parameter update.**

## Exact scope and split

- [Driver 62](../../../scripts/62_thermal_source_amplitude_shape.py), output 58;
  [small fitting helper](../../../src/source_amplitude_diagnostic.py).
- Sweep kappa = 0.6, 0.7, ..., 1.3 for two fixed source shapes: endpoint and
  **Experiment-56 combined GL4**, which interpolates the two thresholded gradient
  snapshots and evolves each contribution. This is NOT Experiment 57's constant
  current-gradient integral. Lambda is only 0 or 1; no continuous lambda search.
- Both shapes already contain historical gamma = 0.1702227206133784. Thresholding,
  source shape, predecessor propagation, native 601x401 map, boundary convention,
  alpha and beta are unchanged. Multiplication happens after thresholding.
- The cached inferred/oracle sweep uses Diagonal 8 and Spiral 13 at their saved
  reference ceilings, 320 K and 325 K. All kappa=1 means reproduce Experiment 56
  exactly. No camera or previous posterior is regenerated.
- Shared scale selection uses exactly the historical three development paths:
  Diagonal 8, Horizontal 13 and Spiral 13. For each shape separately, minimize
  **the mean of the three whole-field native-area MSEs with oracle predecessors**.
  Trajectories receive equal weight; adaptive-mesh node counts do not determine
  weight. The target is current simulator truth. This is supervised simulation
  calibration, not camera-only/self-supervised calibration.
- Select only from the stated grid, save/fingerprint the choice, THEN load the
  30 held-out native field pairs for oracle evaluation. There is no ceiling-specific
  fit. Oracle calculations are ceiling-independent and counted once per path.
- The held-out test is **oracle-state validation only**, not inferred-state or GP
  reconstruction validation. No native inferred-state caches exist for this
  complete 30-path/three-ceiling test. Only the two cached development paths are
  used to examine transfer to inferred states.

The objective, split and finite grid were fixed before results were computed in
[frozen_configuration.json](frozen_configuration.json). Regional analytic optima
are diagnostic and were NOT used to choose the shared scale.

## What was selected?

| Shape | Selected grid kappa | Effective gamma in this diagnostic | Unconstrained development optimum |
|---|---:|---:|---:|
| Endpoint | 0.6 | 0.1021336324 | 0.3451 |
| Combined GL4 | 0.6 | 0.1021336324 | 0.1410 |

Both selections hit the lower boundary. Therefore **0.6 is the best tested grid
value, not the estimated global optimum**. The grid was not expanded after
seeing this. The shared values coincide because of the bound; the two objectives
were independently evaluated, not fitted to endpoint then copied to GL4.

The continuous values are closed-form least-squares diagnostics, not adopted
parameters. Original gamma, configurations and historical outputs were not edited.

## Frozen 2 x 2 oracle validation

Mean of 30 trajectory-specific metrics (K). Hot region = hottest 1% of physical
area, including the fractional final node. Peak error = absolute error at the
true peak location, NOT difference between independently located maxima.

| Arm | Shape | kappa | Whole RMSE | Hot1% RMSE | Shoulder-annulus RMSE | Peak absolute error |
|---|---|---:|---:|---:|---:|---:|
| A | Endpoint | 1.0 | 0.599 | 5.971 | 1.561 | 26.689 |
| B | Endpoint | 0.6 | 0.378 | 3.761 | 1.302 | 11.952 |
| C | Combined GL4 | 1.0 | 0.750 | 7.470 | 2.352 | 23.048 |
| D | Combined GL4 | 0.6 | 0.530 | 5.272 | 1.793 | 11.738 |

Shoulder annulus for this common 33-path evaluation is 0.25--1.0 mm radial distance
from the fixed true current peak, with physical-area weights. This is distinct
from the saved along/across section-node shoulders used on the two cached paths.
Both are labelled separately. Truth defines evaluation regions only, except for
the explicitly supervised development fit and oracle predecessor.

- **B versus A:** whole, hot and annulus RMSE improve on **30/30** paths; true-peak
  absolute error improves on 30/30. Mean hot-RMSE change is -2.210 K (median -3.041 K).
- **D versus C:** whole, hot and annulus RMSE improve on 30/30; peak error on 29/30.
- **D versus B:** whole, hot and annulus RMSE worsen on **30/30**. Peak error is
  slightly better on average (11.952 -> 11.738 K) and improves on 18/30 paths.
  Therefore peak error alone would give a misleadingly favorable view of D.
- **C versus A:** hot, whole and annulus RMSE worsen on 30/30. This is combined
  temporal redistribution, not the more consistent constant-source evolution
  tested in Experiment 57.

Held-out hot-area RMSE by family:

| Family (paths) | A | B | C | D |
|---|---:|---:|---:|---:|
| Diagonal (7) | 6.718 | 3.677 | 7.224 | 4.942 |
| Horizontal (11) | 4.751 | 3.983 | 7.988 | 5.921 |
| Spiral (12) | 6.652 | 3.605 | 7.138 | 4.870 |

These are historically held-out trajectories reused from earlier research, not
a newly collected independent validation dataset. No target from these 30 paths
entered the new kappa selection.

## Does that scale transfer to inferred states?

Not uniformly. Applying the frozen development-oracle kappa=0.6 to all ceilings:

| Case / ceiling | Endpoint RMSE: 1.0 -> 0.6 | Combined GL4 RMSE: 1.0 -> 0.6 |
|---|---:|---:|
| Diagonal / reference | 4.535 -> 4.576 | 2.627 -> 4.391 |
| Diagonal / 320 K | 5.574 -> 2.286 | 5.733 -> 3.585 |
| Diagonal / 325 K | 5.981 -> 2.769 | 6.272 -> 4.045 |
| Spiral / reference | 3.907 -> 3.426 | 2.414 -> 3.573 |
| Spiral / 320 K | 5.779 -> 2.594 | 6.034 -> 3.903 |
| Spiral / 325 K | 6.095 -> 2.968 | 6.450 -> 4.263 |

Entries are hot-area RMSE, K, before any current GP update. Reference GL4 true-peak
signed errors become **-16.608 / -17.833 K**, from -0.823 / -2.641 K. A source
reduction favorable with known predecessors can reveal missing magnitude in the
censored predecessor. There is no claim of improved CRPS or coverage here.

## Regional optima and identifiability

For a fixed base B, source S and normalized region weights w, the unconstrained
conditional least-squares solution is

`kappa* = sum(w*S*(truth-B)) / sum(w*S*S)`.

Saved tables include unconstrained, interval-bounded and discrete-grid solutions,
source RMS and quadratic curvature. If source is identically zero in a region,
kappa is unidentifiable there and is reported as NaN, not an arbitrary optimum.
Very weak source support also limits interpretation of large optimal scales.
Negative values are diagnostic only and are not proposed negative heating.

Examples:

- Diagonal reference, endpoint: peak wants kappa approximately **0.802**, while
  the along-section shoulder wants **3.144**. Its across-section shoulder has
  zero endpoint source support, so no amplitude adjustment can affect it.
- Spiral reference, GL4: peak wants **1.070**, along-section shoulder **1.394**,
  across-section shoulder **0.561**. A single scalar cannot match all three.
- Diagonal GL4 whole-field optimum changes from **1.024** at reference to
  **0.431 / 0.376** at 320/325 K, while the source field itself is unchanged.
  This is evidence of amplitude adjustment compensating for predecessor error,
  not evidence that physical source strength depends on the camera ceiling.
- Horizontal 13 oracle: the propagation-only forecast is already **10.686 K too
  hot at the true peak**, before adding the nonnegative increment. The endpoint
  pointwise optimum is -1.139. Scaling a positive source alone cannot correct
  that location under the frozen propagation model.

## Interpretation of H1--H4

**H1 has conditional support:** the old empirical amplitude is excessive relative
to this frozen oracle-state predictor's residual target, and reducing it improves
the tested metrics across trajectories. That does not establish excess physical
laser energy, nor prove a uniform scalar repairs every region.

**H2 is not supported as a general fix** for the tested combined GL4 shape.
Redistribution alone worsens the held-out oracle field metrics.

**H3 is not demonstrated within this grid/objective:** amplitude plus combined
integration does not beat endpoint amplitude adjustment on field/hot/annular RMSE.
This does not rule out every untested scale or intermediate lambda, and no such
search was launched.

**H4 remains open:** incompatible regional optima, the oracle propagation-only
error and the dependence on predecessor quality indicate that source scaling is
not a complete physical explanation. These results motivate discussion of the
propagation/source formulation, not automatically another covariance change.

## Provenance and limitations

The source is still the contemporaneous simulator-derived projected vertical
temperature gradient. Independent deployment availability remains unresolved.
Neither current-gradient access nor oracle predecessors are available from a
censored camera alone. Temperature-increment area integrals have units K*m^2,
not energy. Multiplying gamma commutes with the fixed linear source evolution
only because the gradient threshold and all other operations remain fixed.

For newly evaluated paths, ambient uses the **existing** preparation rule: 2nd
percentile of all-time temperature fields projected onto the legacy 61x41 grid.
Development ambient values were reused from the saved coefficient CSV. This
simulation-derived preprocessing is explicitly retained; it is not a camera-only
ambient estimate or a new held-out calibration of kappa. Native forecasts and
metrics remain on native nodes. The new input-provenance CSV lists 31 newly
prepared paths; the other two use fingerprinted Experiment-56 caches.

Experiment 56's interpolation/refinement qualifications remain: GL4 quadrature
consistency is not physical validation; the native input map includes P1 lifting,
fine-grid diffusion and bilinear return. No new grid or quadrature was selected.

## Files and checks

- [development_amplitude_sweep.csv](development_amplitude_sweep.csv): two cached
  paths, two predecessor types, all three ceilings, two shapes and eight scales.
- [regional_optima.csv](regional_optima.csv) and
  [development_oracle_regional_optima.csv](development_oracle_regional_optima.csv).
- [frozen_selection.json](frozen_selection.json) and
  [development_selection_objective.csv](development_selection_objective.csv).
- [oracle_33_results.csv](oracle_33_results.csv),
  [heldout30_oracle_summary.csv](heldout30_oracle_summary.csv),
  [heldout30_oracle_paired.csv](heldout30_oracle_paired.csv),
  [heldout30_oracle_family_summary.csv](heldout30_oracle_family_summary.csv).
- [frozen_scale_cached_cases.csv](frozen_scale_cached_cases.csv),
  [native_input_provenance.csv](native_input_provenance.csv),
  [verification.json](verification.json), [source_fingerprints.json](source_fingerprints.json).
- [Diagonal curves](amplitude_curves_DiagonalScanPath_8.png) and
  [Spiral curves](amplitude_curves_SpiralScanPath_13.png); native oracle inputs in
  `oracle_inputs/` allow all four arm forecasts to be reconstructed without GP runs.

All 33 paths were loaded or their exact cached native inputs reused. The 3/30 split
was asserted and selections were saved before preparing held-out fields. Cached
kappa=1 equality and source/input fingerprints passed. **105 tests passed**;
compilation and `git diff --check` passed. The pre-existing near-constant-draw
skewness test warning remains. Runtime: **8.27 s**, peak RSS **763 MiB**.

Run script 62 with the existing `.venv`, single-threaded BLAS and
`MPLCONFIGDIR=/tmp/thermal_mpl`. It refuses to overwrite this output directory.
Only new isolated code/results were added. No original gamma, historical result,
previous/current sampler, Q, covariance, mask policy or observation noise changed.
