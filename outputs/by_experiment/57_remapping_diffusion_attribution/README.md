# Experiment 57: Remapping versus within-step evolution

**Conclusion: close the quadrature branch here.** Remapping explains part, but
not all, of the peak reduction. Additional diffusion/cooling under the frozen
discrete operator improves hot-area and whole-field RMSE, yet still worsens
particular shoulders. No GP inference, fitting, new source model or change to
Experiment 56 was made.

## Controlled calculation

[Driver 61](../../../scripts/61_thermal_remapping_attribution.py) uses only the
cached Experiment-56 inputs for Diagonal 8 and Spiral 13 at reference, 320 and
325 K. The helper is [remapping_attribution.py](../../../src/remapping_attribution.py).
Geometry, area weights, hot1% weights, shoulder nodes, predecessor contribution,
gamma, alpha, beta, threshold and stored dt remain fixed. Oracle and inferred
predecessors are both evaluated. These are deterministic forecasts, not new
posterior reconstructions.

For the constant, thresholded **current** gradient q_n:

- S0 = gamma * dt * q_n.
- S_map = gamma * dt * A(0) q_n, including the existing zero-time interpolation.
- S_evolved = gamma * integral_0^dt A(s) q_n ds, evaluated with GL4; GL2 checks
  quadrature consistency. There is no temporal interpolation of gradient snapshots.

Every evaluation starts from the original native q_n, not an already remapped
field. If A(s) = E D(s) L, with lift L and evaluation E, this compares E L q_n
with the integral of E D(s) L q_n. It does **not** apply a second E/L round trip.
The source is never floored or rethresholded during propagation.

Pointwise attribution is additive:

`S_evolved - S0 = (S_map - S0) + (S_evolved - S_map)`.

The second term is numerical diffusion/cooling **conditional on the frozen
interpolant, grid and boundary convention**, not a measurement of physical 3D
diffusion. Boundary and discretization effects remain within this operator.

## Peak attribution

Values at the **true current peak node**, not at each variant's separate maximum:

| Case | S0 (K) | S_map (K) | S_evolved, GL4 (K) | Remapping reduction | Additional evolution reduction |
|---|---:|---:|---:|---:|---:|
| Diagonal 8 | 50.249 | 48.272 | 45.019 | 1.977 K | 3.253 K |
| Spiral 13 | 62.312 | 56.555 | 45.387 | 5.758 K | 11.168 K |

Remapping accounts for approximately **37.8% / 34.0%** of these pointwise total
reductions. These are not universal fractions of forecast error, global RMSE
or physical dissipation. The source contributions are identical across ceilings
and predecessor types; only the common propagated predecessor changes.

The Diagonal remapping reduction here (1.977 K) is smaller than Experiment 56's
4.725 K maximum remapping change, because those refer to different nodes.

## Shape remains a tradeoff

Inferred-predecessor physical-area hot1% RMSE:

| Case | Ceiling (K) | S0 | S_map | S_evolved, GL4 |
|---|---:|---:|---:|---:|
| Diagonal 8 | 296.12 | 4.535 | 4.205 | 3.916 |
| Diagonal 8 | 320 | 5.574 | 5.259 | 4.825 |
| Diagonal 8 | 325 | 5.981 | 5.685 | 5.267 |
| Spiral 13 | 296.49 | 3.907 | 3.548 | 3.219 |
| Spiral 13 | 320 | 5.779 | 5.480 | 5.055 |
| Spiral 13 | 325 | 6.095 | 5.810 | 5.398 |

Whole-field RMSE also improves in each case. Oracle hot-area RMSE changes
6.854 -> 6.593 -> 6.210 K for Diagonal and 6.650 -> 6.384 -> 5.992 K for Spiral.
Oracle repetitions across ceilings are identical, not independent replicates.

However, the additional evolution still worsens shoulder shape:

- Spiral reference across-section shoulder RMSE: **0.444 -> 0.484 -> 0.738 K**.
- Spiral oracle across-section shoulder: **0.783 -> 0.837 -> 1.076 K**.
- Diagonal reference along-section shoulder: **2.794 -> 2.782 -> 2.944 K**.
- Diagonal 320 K across-section shoulder: **0.216 -> 0.246 -> 0.350 K**.

Arrows denote S0, S_map, S_evolved. The fixed shoulder definition is the same
0.25 <= |along_mm| <= 1.0 section-node subset as Experiment 56, with native
area weights. This is not an exhaustive measure of the wake.

Thus the peak/shoulder tradeoff survives removing the remapping contribution
from the attribution. We stop rather than optimize quadrature or launch GP
inference. Partial improvement is real within the numerical surrogate; it is
not a robust correction of the complete spatial profile.

**Do not confuse the variants:** Experiment 56's S_evolve was a trapezoidal
constant-gradient approximation. This experiment's S_evolved_GL4 is its
higher-order integral. Experiment 56's S_GL4 interpolated between previous and
current gradients; this experiment holds the current gradient constant.

## Numerical checks

- S0 and complete baseline forecasts reproduce Experiment 56 exactly.
- The previous S_evolve trapezoid also reproduces exactly as an equality check.
- Oracle forecasts and source fields are exactly invariant across ceilings.
- GL2/GL4 relative source L2 differences: 0.170% / 0.165%; maximum node
  differences: 0.090 / 0.109 K. Hot-RMSE differences are below 0.003 K.
- No new spatial refinement or tuning was run. Experiment 56's grid sensitivity,
  nonidentity zero-time map and slight spatial-tolerance miss remain relevant.
- Runtime including six figures and caches: **3.10 s**; peak RSS **298 MiB**.
- **101 tests passed**, including three new attribution tests; compilation of
  src/scripts/tests and `git diff --check` passed. The existing skewness test
  emitted its known near-constant-draw precision warning; no test failed.

## Current-gradient availability: unresolved deployment input

The executed higher-ceiling driver calls `read_heat_flux_z(-1, surface_only=True)`
at [script 56, line 117](../../../scripts/56_thermal_native_higher_ceiling.py),
then constructs the increment at line 134. The loader's `read_heat_flux_z`
function at [thermal_trajectory.py, line 224](../../../src/thermal_trajectory.py)
reads the saved HeatFluxZ field at that frame. Experiment 56's
[gradient provenance](../56_within_step_integration/gradient_provenance.json)
identifies the actual current-frame HDF datasets.

According to Adrienne's supplied export, that quantity is a projected derivative
of simulated T, not an independently supplied laser command. The code therefore
uses a **contemporaneous simulator-derived gradient input**. Its availability
from the thermal camera or other independent deployment input has not been
established. These frozen comparisons remain valid conditional on that input;
they should not be described as camera-only reconstruction or as using only
known commanded heating.

Questions for Adrienne: Does the export use the same current solution T? Can the
required boundary input instead be computed from known laser path/power and
the original boundary specification, without evaluating the hidden current T?
If not, a future separately controlled source replacement is needed for a
deployable claim. No substitution, sign/unit conversion or gamma recalibration
was attempted here, and no physical source formulation is claimed validated.

## Outputs and reproduction

- [metrics.csv](metrics.csv): per-region RMSE, MAE, bias and true-peak errors.
- [paired_changes.csv](paired_changes.csv): remapping, subsequent evolution and
  total changes. MSE changes telescope; RMSE contributions are not asserted additive.
- [true_peak_attribution.csv](true_peak_attribution.csv),
  [source_components.csv](source_components.csv), [quadrature_checks.csv](quadrature_checks.csv).
- [verification.json](verification.json), [equality_checks.csv](equality_checks.csv),
  [frozen_configuration.json](frozen_configuration.json),
  [source_fingerprints.json](source_fingerprints.json).
- Six `sections_*.png` figures and native forecast arrays in `by_trajectory/`.
  [Spiral reference sections](sections_SpiralScanPath_13_level0.png) illustrate
  the persistent shoulder problem with both inferred and oracle predecessors.

Use the existing `.venv` and single-threaded BLAS to run script 61. The driver
refuses to overwrite this output directory. Historical outputs, the mask branch,
all GP code and parameter settings were preserved.
