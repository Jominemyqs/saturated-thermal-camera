# Output 66: prescribed laser input and oracle reconciliation

2026-11-06. Driver: [70_thermal_prescribed_laser_audit.py](../../../scripts/70_thermal_prescribed_laser_audit.py).
This completes the numerical comparison that was blocked in
[Output 62](../62_physics_input_provenance/README.md), using the newly supplied
notebooks. Output 62 remains an accurate record of what was unavailable then.
No FEM solve, GP inference, parameter fitting, source replacement or baseline change.

## What is now confirmed

- In the supplied Diagonal/Spiral code, `g` is an independently specified moving
  Gaussian boundary input. `HeatFluxZ` is computed **after** the solve as
  `project(T.dx(2), FunctionSpace(mesh, "CG", 1))`. They are different constructions.
- The original path functions and laser `eval` methods were extracted and used
  directly. No temperature-derived centroid, fitted alignment, amplitude scaling,
  surrogate cutoff or gamma enters the flux comparison.
- Notebook `dt=0.0005 s` is both the solver step and export interval. All 355
  Diagonal-8 and 544 Spiral-13 saved timestamps agree with the reconstructed loop
  to at most `5.56e-17 s`; final times are 0.1775 and 0.2720 s. Spiral `dt_path=0.005`
  is an angular path-discretization setting, **not** a thermal timestep.
- The weak form has positive inward `g` on the top boundary, whose outward normal
  is +z. Its continuum natural condition is `k*dT/dz = g` there. The discrete
  projected gradient need not satisfy that relation pointwise.
- The source is evaluated at the endpoint inside an **implicit 3D conduction
  solve**. This does not justify adding an endpoint temperature increment after
  a separate surface propagation step.

See [parameter provenance](provenance_parameters.csv), [source/availability notes](provenance.md),
[model correspondence](model_correspondence.md), and [clock checks](clock_checks.json).
Adrienne's Nov 5 dataset attribution plus matching clocks/bounds support the
notebook-to-data link; this is not a bitwise reproduction of the original FEM run.

## Raw prescribed input versus exported gradient

At the same 6,275 native top nodes (12,448 triangles), compare `g` against
`15.11 * HeatFluxZ`. Signed error is exported minus prescribed. No values are clipped.
Relative L2 uses the prescribed-field norm and physical-area weights, not node counts.

| Case/frame time (s) | Relative L2 | Exported/prescribed native peak | Native peak separation (um) | Integrated k*gradient (W) |
| --- | ---: | ---: | ---: | ---: |
| Diagonal, 0.1770 | 33.33% | 0.706 | 88.39 | 1.00920 |
| Diagonal, 0.1775 | 33.77% | 0.619 | 88.39 | 1.00635 |
| Spiral, 0.2715 | 33.35% | 0.821 | 62.50 | 1.01487 |
| Spiral, 0.2720 | 33.57% | 0.718 | 88.39 | 1.01355 |

The independently integrated prescribed Gaussian is approximately **1.00000 W**
at all four frames. Its integral over the infinite plane is exactly the notebook
`power=1.0`; finite-domain integration was checked separately, not normalized to 1.
Triangle subdivision levels 2 and 3 change this integral by at most `1.39e-7 W`.
The exported P1 field's integral is exactly represented by triangle-lumped nodal
weights. Applying those weights to the sampled Gaussian instead gives 1.00070--1.00257 W;
the CSV explicitly retains that nodal approximation rather than conflating it with
the refined analytic-field quadrature.

**Total input is close, spatial equivalence is not.** Within a fixed radius
`3*sigma=0.3 mm` around the independently prescribed laser focus, relative L2 remains
32.50--32.92%; exported/prescribed regional integrals are 0.824--0.850. Whole-field
averages alone hide this local discrepancy. The exported field has small negative
tails (total negative integral about -0.003 W), retained rather than thresholded.

Figures: [Diagonal flux](flux_comparison_DiagonalScanPath_8.png),
[Spiral flux](flux_comparison_SpiralScanPath_13.png).
Maps linearly display native nodal fields on the original triangles; these are
not reconstructions of the boundary quadrature used by the FEM assembler.
Tables: [region metrics](flux_metrics.csv), [peaks/locations](flux_peaks.csv),
[integral refinement](flux_integral_quadrature.csv).

Derivative projection, weak boundary enforcement, mesh resolution and load
quadrature are possible contributors. This audit does not separate them or prove
that the generator's physical model is wrong. Matching total flux does not establish
the validity of `gamma*HeatFluxZ*dt` as a surface-temperature increment.

## Adrienne's oracle diagnostic, reconciled with the latest implementation

Recomputed with the latest 61/63 frozen operator and coefficients:

```text
m_off = ambient + propagate(previous - ambient, floor=True)
s     = gamma * dt * gradient * 1[gradient >= 10000]
m_on  = m_off + s
r_needed = current_truth - m_off
m_on - current_truth = s - r_needed
```

The oracle off/on fields and required residual are **array-exactly equal to
Output 51**, at all three cached ceiling cases. Latest inferred-state source-on
means reproduce Output 63 exactly; subtraction-based source-off caches agree
within `5.69e-14 K`. The internal propagation grid remains 601 x 401; the inherited
61 x 41 settings refer to the coarse preparation grid, not that internal grid.

All numbers below are evaluated at the **current true peak location**, before GP:

| Case | True T (K) | Oracle off (K) | Required residual (K) | Added increment (K) | Oracle on (K) | Oracle on error (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Diagonal 8 | 340.43 | 319.79 | 20.64 | 50.25 | 370.04 | +29.60 |
| Spiral 13 | 341.18 | 315.62 | 25.56 | 62.31 | 377.93 | +36.75 |

With oracle input, hot-1%-area forecast RMSE increases from 4.785 to 6.854 K
(Diagonal) and 4.790 to 6.650 K (Spiral) when the increment is added. In contrast,
with the latest inferred predecessor at the reference ceiling it decreases from
8.241 to 3.691 K and 8.247 to 3.581 K. Thus the inferred predecessor partly
compensates for the complete forecast mismatch. This is a deterministic result;
it neither contradicts nor re-evaluates the current-GP results of Output 63.

The **required residual is not true heating**: it includes whatever is wrong
with the approximate surface propagation, and with inferred predecessors when
those are used. Oracle-input mismatch rules out previous-state estimation as
the sole explanation; it does not isolate a uniquely incorrect physical term.

Figures: [Diagonal oracle sections](oracle_sections_DiagonalScanPath_8.png),
[Spiral oracle sections](oracle_sections_SpiralScanPath_13.png).
These retain the historical section nodes/directions for comparability; they
are not used to derive the prescribed laser trajectory.
Tables: [all deterministic metrics](oracle_forecast_metrics.csv),
[equality checks](oracle_equality_checks.csv).

## Next decision, not a model change

We can now specify `g` independently for these simulations. We still need a
justified mapping from prescribed boundary flux to the surface-temperature
response, with heat capacity, depth evolution and boundary effects accounted
for. Do not replace `HeatFluxZ` with `g/k` under the unchanged gamma and call
that a physically derived predictor. Discuss this distinction with Adrienne;
no new solver, fitted coupling or baseline replacement has been implemented.

Audit runtime was 13.02 seconds, including hashes, quadrature checks, propagation
and figures. Ten lightweight tests pass. See [verification](verification.md)
and [input/code fingerprints](fingerprints.json).
