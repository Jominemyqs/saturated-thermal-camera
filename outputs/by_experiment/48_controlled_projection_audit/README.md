# Controlled projection audit: representation before GP inference

## Answer

Projection materially changes local peaks and one-step forecasts, but **it does
not create the positive oracle-forecast bias on its own**. With the frozen heat
formula and native input fields, the hottest-area oracle bias is still positive,
between +2.04 and +2.98 K on the three development trajectories.

Projecting the previous temperature and current source lowers that bias by
0.47-0.79 K. Projecting the current reference field raises the measured bias,
and the remaining coarse-pipeline numerical effect raises it further. These
effects partly cancel. A small change in the final aggregate oracle bias would
therefore not mean that projection is harmless locally.

No GP inference, observations, censoring sweep, refitting, new kernel or native
FEM propagation is used. The existing native pilot and canonical regular-grid
results are preserved unchanged. This is script 52 / output 48.

## Controlled construction

Use the same last two frames of DiagonalScanPath_8, HorizontalScanPath_13 and
SpiralScanPath_13. Read native temperature and HeatFluxZ directly from the stored
surface nodes and actual tetrahedron-derived surface triangles. Apply the
existing `SurfaceGridProjector` to obtain the exact legacy 61 x 41 fields.

Let N denote the piecewise-linear native field, P the existing coarse projection,
and R bilinear reconstruction of a coarse-grid array. Both input representations
are evaluated on the same fine computational grid and passed through **the same
frozen formula**, using the existing `diffuse_and_cool` and `current_source_field`:

    F_h(T, q) = T_amb
                + exp(-beta dt) Gaussian_h(max(T - T_amb, 0))
                + gamma q 1[q >= 10000] dt.

Diffusivity, cooling, ambient, source coupling, time step, source cutoff, floor,
Gaussian truncation and zero-extension boundary convention stay fixed. Only the
input field representation differs in the principal comparison:

    native-input oracle    = F_h(N T_previous, N q_current)
    projected-input oracle = F_h(R P T_previous, R P q_current).

Both are evaluated against the **same native current reference**. Additional
counterfactuals project only the previous temperature or only the current source.
Because this predictor adds source and propagated temperature separately, their
effects add exactly, even though the floor and source cutoff are nonlinear.

Important distinction: a fine computational grid is still used to evaluate the
Gaussian operator. This is not a native-mesh PDE solve. It avoids confounding the
projection experiment with the different FEM operator introduced in output 47.
The fine operator is not declared identical to the old coarse operator; their
difference is measured separately.

## Exact bridge to the old pipeline

All quantities below are evaluated at the same physical locations:

    legacy_error = native_oracle_error
                   + input_projection
                   + reference_projection
                   + coarse_pipeline_bridge

where

    native_oracle_error = F_h(N T_previous, N q) - N T_current
    input_projection = F_h(R P T_previous, R P q) - F_h(N T_previous, N q)
    reference_projection = N T_current - R P T_current
    coarse_pipeline_bridge = R F_coarse(P T_previous, P q)
                             - F_h(R P T_previous, R P q).

The last term is named `coarse_discretization` in the CSV/cache. It includes
Gaussian discretization, evaluating before/after interpolation, boundary
discretization and the noncommuting floor/source-threshold operations. It must
**not** be interpreted as diffusion discretization error alone.

Bias adds; RMSE does not. The complete bridge is exact to floating-point precision
for these represented arrays, not an attribution to the unknown original 3D
physical model.

## Forecast results

Primary region: fixed hottest **1% of native physical area**, held identical
across all counterfactuals. Values below are area weighted, in K.

| Development trajectory | Native-input oracle bias | Projected-input oracle bias | Input-projection change | Native-input RMSE | Projected-input RMSE |
|---|---:|---:|---:|---:|---:|
| DiagonalScanPath_8 | +2.980 | +2.511 | -0.469 | 6.809 | 4.675 |
| HorizontalScanPath_13 | +2.043 | +1.520 | -0.522 | 4.801 | 3.494 |
| SpiralScanPath_13 | +2.918 | +2.125 | -0.794 | 6.599 | 4.520 |

The reduction in error after projection is not evidence that coarsening is
physically preferable: it partly offsets the positive error of this frozen
predictor. Its coefficients were calibrated in the old projected setting and
were deliberately not recalibrated here.

### Signed-bias bridge

| Trajectory | Native oracle | Previous-input projection | Source projection | Reference projection | Coarse-pipeline bridge | Legacy error on common region |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | +2.980348 | -0.290974 | -0.178254 | +0.216242 | +0.117676 | +2.845038 |
| Horizontal 13 | +2.042664 | -0.531141 | +0.008804 | +0.343547 | +0.196091 | +2.059964 |
| Spiral 13 | +2.918206 | -0.402583 | -0.390979 | +0.531031 | +0.207822 | +2.863497 |

This is not a claim about the held-out 30-trajectory average. It is a three-path
development diagnostic on new, common physical evaluation regions.

## Peak and location changes

Native/projected current field peaks are unchanged from the initial projection
audit: temperatures 340.430 -> 336.215 K (diagonal), 319.130 -> 319.130 K
(horizontal), and 341.182 -> 335.708 K (spiral). The previous spiral temperature
peak drops by 15.055 K. Current HeatFluxZ peaks decrease by 17.39%, 10.26% and
39.09%, respectively; integrated flux changes by -1.49%, +12.93% and -1.62%.
Stored HeatFluxZ units are retained; interpreting its integral as watts still
requires confirmation that the field is W/m2.

| Current temperature field | Peak-location shift (mm) | Hottest-area centroid shift (mm) | Hottest-area overlap (IoU) |
|---|---:|---:|---:|
| Diagonal 8 | 0.0625 | 0.0176 | 0.953 |
| Horizontal 13 | 0.0000 | 0.0977 | 0.897 |
| Spiral 13 | 0.0884 | 0.1296 | 0.879 |

An unchanged peak coordinate does not imply unchanged hottest-region shape.
Peak locations use native/coarse nodal argmax, with first-index tie breaking.
Area centroids and overlaps use each representation's hottest 1% on common
locations; unlike the forecast scoring region, these masks intentionally differ
to measure the change in shape.

| Forecast peak (K) | Native inputs, fine F | Projected inputs, same fine F | Legacy coarse F |
|---|---:|---:|---:|
| Diagonal 8 | 374.109 | 353.519 | 356.278 |
| Horizontal 13 | 341.650 | 334.309 | 344.888 |
| Spiral 13 | 378.065 | 358.010 | 366.746 |

Thus representation affects forecast maxima strongly, even while aggregate
oracle biases partly cancel. Forecast peaks use the full finest computational
grid; reference peaks use original native or coarse nodal maxima. The forecasts
shown here are **oracle-state deterministic means**, not GP reconstructions.

## Equality, weights and numerical checks

- The three coarse oracle arrays reproduce the cached output-46 arrays **exactly**.
  Native-to-grid temperature and current source arrays reproduce the canonical
  preparation exactly. Existing output-46 source fingerprints must match.
- The old, unweighted coarse top-25 oracle biases are reproduced separately:
  2.756566 K, 2.158606 K and 2.898470 K. They differ from the table's common-region
  legacy values because the evaluation region and weighting differ, not because
  the old oracle changed.
- Gaussian calculations are checked at 25, 12.5 and 6.25 micrometre spacings
  (601x401, 1201x801, 2401x1601). Primary forecasts use the finest result.
- All area-weighted errors use fixed 601x401 evaluation locations and trapezoidal
  area weights; the hottest 1% uses a fractional final weight. It is a numerical
  area approximation, not an exact continuous level-set region. It is not the
  old 25-pixel mask and not the native pilot's lumped nodal mask.
- Refinement from 12.5 to 6.25 micrometres changes hottest-area forecast bias by
  at most 0.0011 K, forecast fields by under 0.004 K RMSE there, and the four
  counterfactual forecast maxima by under 0.027 K. Projection effects are much
  larger than these tested numerical differences.
- Exact bias-bridge residuals are zero in the saved run; splitting input effects
  into temperature/source contributions closes within 1e-11 K.
- Original nodal P1 fields and coarse bilinear fields have exact area integrals
  under their respective mass/trapezoidal weights. Error and hot-region metrics
  between representations use common physical quadrature locations.
- No held-out trajectory is used to select parameters. No inferred previous
  posterior is needed: known previous truth is an explicitly oracle-only input.
- The provided mesh fields may have upstream modeling limitations. This audit
  attributes differences within the available dataset, not physical ground truth.

## Figures and files

- `field_comparison.csv`: previous/current temperature and flux peaks, peak
  locations, hot-area centroids/overlap, area integrals and representation RMSE.
- `forecast_metrics.csv`: four input counterfactuals and the bridges to projected
  reference/coarse prediction, with whole-field and fixed native-hot-area metrics.
- `bias_decomposition.csv`: signed-bias contributions, RMSE and MAE separately.
- `forecast_error_maps_*.png`: forecasts and **shared-scale absolute errors against
  the same native current field**. First two columns isolate input representation;
  the third also changes the numerical operator.
- `forecast_error_zoom_*.png`: the same quantities/color scales around the native
  peak, so narrow-source differences can be inspected.
- `projection_contributions_*.png`: native reference and signed contribution maps.
- `resolution_check.csv`, `baseline_checks.csv`, `verification.json`,
  `frozen_configuration.csv`, `source_fingerprints.json`: reproducibility checks.
- `by_trajectory/*/projection_audit.npz`: local, gitignored common-grid arrays.

The full three-path deterministic audit takes about 10 seconds on this machine,
excluding Python startup; exact runtime is in `verification.json`.

```sh
env MPLCONFIGDIR=/private/tmp/thermal_mpl PYTHONPYCACHEPREFIX=/private/tmp/thermal_pycache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/52_thermal_controlled_projection_audit.py
```

## Decision

Representation is not a negligible peak effect, so a **small matched native/grid
B/C sensitivity comparison** is defensible. It should harmonize observation
locations, noise and physical evaluation regions, and either hold propagation
fixed or separately factor its discretization. The unmatched native FEM pilot
cannot answer that question by itself.

However, these data do not justify switching the whole project to native inference
or expecting it to fix the frozen mean automatically. The positive oracle error
persists with native inputs and can be larger. Understanding the approximate
source/heat predictor remains relevant. No matched GP follow-up was launched in
this audit, and no physical correction has been fitted.
