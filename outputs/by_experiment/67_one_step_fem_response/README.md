# Output 67: one-step original-FEM replay and response decomposition

2026-11-06. Driver [71](../../../scripts/71_thermal_one_step_fem_replay.py),
helper [fem_replay.py](../../../src/fem_replay.py).

**Both source-on replays pass at floating-point precision. The frozen surface
oracle's peak overshoot has two positive contributions: propagation mismatch
and an excessive increment relative to the original FEM boundary response.**
This is an oracle attribution diagnostic, not a replacement reconstruction model.

## Frozen scope and gate

Only the last stored transition of DiagonalScanPath_8 and SpiralScanPath_13.
Each solve uses the saved full-volume mesh and true volume predecessor, constant
notebook material coefficients, bottom temperature, top flux expression and
endpoint time. No mesh regeneration, source scaling, fitted timing, GP, camera
likelihood, ceiling sweep or full-trajectory regeneration.

The [configuration](frozen_configuration.json) was saved before thermal solves:
both volume and surface weighted RMSE <= 1e-4 K and max absolute error <= 1e-3 K;
linear relative residual <= 1e-10; <=900 seconds and <=4096 MiB worker peak RSS
per stage. **Both cases had to pass source-on before either source-off ran.**
These tolerances were not changed after inspecting temperature results.

| Replay | Volume RMSE (K) | Surface RMSE (K) | Maximum volume error (K) | Gate |
| --- | ---: | ---: | ---: | --- |
| Diagonal 8 | 5.56e-13 | 5.01e-13 | 3.87e-12 | Pass |
| Spiral 13 | 5.30e-13 | 4.74e-13 | 5.34e-12 | Pass |

Weights are physical lumped-volume/triangle-area weights, not uniform node
weights. Maxima cover all nodes, not just the peak. This is a strong numerical
confirmation of the matching one-step generator/data relationship, beyond the
clock consistency established by [Output 66](../66_prescribed_laser_input_audit/README.md).
It is not a full-trajectory rerun or physical validation against real measurements.

## What was solved

The supplied P1 weak form is reproduced with legacy DOLFIN 2019.1.0. Equivalently,
after shifting by the constant bottom temperature and imposing homogeneous
bottom values:

```text
(M + dt*K) u_on  = M*u_previous_true + dt*f(g_current)
(M + dt*K) u_off = M*u_previous_true
s_FEM = T_on - T_off
```

M includes rho*Cp. The implementation solves absolute temperatures with the
original 293.15 K bottom condition. The same predecessor is used in both arms:
source-off removes only this step's boundary input, not historical deposited heat.

An additional zero-bottom solve of `(M+dt*K)s = dt*f(g_current)` independently
checks that the difference really is the linear boundary response. It agrees
with on-off within **1.69e-12 K across the full volume**. The three quantities are
evaluated on the same 6,275 native surface nodes for the reduced-model comparison.

The independently prescribed g comes from the supplied notebook expression,
not the exported gradient. UserExpression degree=2 and automatic FFC quadrature
are retained. Its assembled boundary integrals are 1.0000154 W and 1.0000064 W;
no amplitude normalization was applied. See [implementation notes](verification.md).

## Peak attribution

At the **current true temperature maximum**, in K:

| Quantity | Diagonal 8 | Spiral 13 |
| --- | ---: | ---: |
| Current truth / FEM source-on | 340.43 | 341.18 |
| Original FEM source-off | 310.25 | 306.83 |
| Frozen surface oracle, propagation only | 319.79 | 315.62 |
| Actual FEM one-step boundary response | **30.18** | **34.35** |
| Frozen gradient-based increment | **50.25** | **62.31** |
| Propagation discrepancy | **+9.54** | **+8.79** |
| Increment-response discrepancy | **+20.07** | **+27.96** |
| Replay discrepancy | +2.10e-12 | -3.24e-12 |
| Complete frozen forecast error | **+29.60** | **+36.75** |

At every surface node the implemented identity is:

```text
old_oracle_on - truth
  = (old_oracle_off - FEM_off)
  + (old_increment - FEM_response)
  + (FEM_on - truth).
```

The sum is checked to 2e-12 K. The first term includes surface-state reduction,
interpolation, floor, ambient, diffusion/cooling and boundary differences; it is
**not** an isolated measurement of omitted vertical conduction. The second term
compares the complete empirical increment with the original discrete boundary
response; it does not identify a uniquely correct new gamma or source shape.

An important correction to interpretation: Output 66's needed residuals of
20.64 / 25.56 K are **smaller than the FEM boundary responses** of 30.18 / 34.35 K,
because the approximate no-increment propagation is already too warm at those
locations. This concretely demonstrates why `truth - old_off` is not true heating.

## Beyond one peak

Area-weighted bias over the fixed hottest 1% of physical surface area:

| Contribution (K) | Diagonal 8 | Spiral 13 |
| --- | ---: | ---: |
| Propagation discrepancy | +1.457 | +1.413 |
| Increment-response discrepancy | +1.487 | +1.468 |
| Complete forecast bias | +2.944 | +2.880 |

Thus, while the increment dominates the error at the extreme peak, the two
terms make almost equal contributions to **hot-area mean bias**. Do not extend
the pointwise peak attribution to the entire hot region. RMSE values are saved
separately and are not additive. No posterior CRPS or coverage is measured here.

Figures: [Diagonal sections](sections_DiagonalScanPath_8.png),
[Spiral sections](sections_SpiralScanPath_13.png),
[Diagonal spatial decomposition](discrepancies_DiagonalScanPath_8.png),
[Spiral spatial decomposition](discrepancies_SpiralScanPath_13.png).
Truth and FEM-on overlap at plotting precision. Historical section nodes and
directions are retained only for display; they do not define the laser input.

Tables: [true-peak decomposition](true_peak_decomposition.csv),
[surface metrics](surface_metrics.csv), [runtime/solver checks](runtime_and_solver.csv).
`old_*` means the frozen surface oracle reconciled in Output 66, not a newly
selected baseline. `FEM_off` is a controlled counterfactual, not a candidate
prediction expected to recover the actual heated current field.

## Cost and limits

Each mesh has **26,414 P1 DOFs, 130,664 tetrahedra, 6,275 top nodes**.
Worker times were 16.67 / 2.73 seconds for source-on and 13.35 / 3.26 seconds
for source-off plus the independent response solve (Diagonal / Spiral).
First-use JIT compilation accounts for part of the difference. Worker peak RSS
was 343--348 MiB; this is not an aggregate memory measurement including compiler
children. Total driver wall time, including hashes and figures, was **47.08 s**,
excluding environment installation and separate unit tests.

The isolated environment is `tmp/fem_env` (about 1.7 GB), with an approximately
495 MB package cache. Existing `.venv` and other Conda environments were not
modified. [Exact package URLs](environment_explicit.txt) and a
[portable dependency request](environment.yml) are saved.

**No model remedy was implemented.** The full previous volume is unobservable
from the censored camera, so the exact FEM response is a mechanistic oracle, not
a deployable reconstruction or a fair competing GP baseline. The next discussion
should address both the reduced propagation and the boundary-flux response;
reducing source amplitude alone would leave a measured propagation discrepancy.
Neither contribution has yet been separated into individual physical/numerical
causes. Historical source, gamma, Q, noise, mask and GP outputs remain unchanged.

Fourteen tests passed. [Verification](verification.md) records the setup issue,
unchanged gate, mappings, checks and reproducibility commands.
