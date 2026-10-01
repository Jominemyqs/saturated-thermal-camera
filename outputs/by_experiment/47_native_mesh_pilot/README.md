# Native-mesh inference pilot and projection audit

## Scope and conclusion

Script 51 adds an isolated, regular-grid-free **surface-node inference** path.
All 33 existing trajectories load on their stored mesh (3 development / 30
held-out). Field comparisons use the last two frames of the three development
trajectories. End-to-end native B/C inference is run only on DiagonalScanPath_8.
No coefficients are refitted and no existing canonical implementation is changed.

The supplied files already contain tetrahedral connectivity and nonuniform
surface nodes. We do not need new data to perform this pilot. "Native" here
means the nodal fields in those files, not a claim about their provenance before
they were supplied or a recreation of the original simulation solver.

**Projection materially changes narrow peaks despite nearly unchanged average
temperature.** This is evidence of a representation effect, not proof that
projection caused the earlier positive oracle-forecast bias.

## Representation comparison

Every tested mesh has 6,275 upper-surface nodes and 12,448 actual surface
triangles extracted from the volume tetrahedra. The surface area is 150 mm2.
The existing projection has 61 x 41 = 2,501 grid locations. Actual connectivity
is used for the native field; no Delaunay replacement is made on this path.

Current-frame results (projected minus native):

| Development trajectory | Native T peak (K) | Grid T peak (K) | Peak change (K) | Area-mean T change (K) | Field projection RMSE (K) |
|---|---:|---:|---:|---:|---:|
| DiagonalScanPath_8 | 340.430 | 336.215 | -4.216 | -0.000276 | 0.146856 |
| HorizontalScanPath_13 | 319.130 | 319.130 | 0.000 | -0.000340 | 0.088977 |
| SpiralScanPath_13 | 341.182 | 335.708 | -5.473 | -0.001267 | 0.238735 |

The previous spiral frame has a larger peak change: **344.087 -> 329.032 K**,
a loss of 15.055 K. Thus checking just the current frame would miss an important
effect on the field supplied to the one-step forecast.

| Current source field | Native HeatFluxZ peak | Grid peak | Peak change | Surface-integral change |
|---|---:|---:|---:|---:|
| DiagonalScanPath_8 | 651742.3 | 538400.9 | -17.39% | -1.49% |
| HorizontalScanPath_13 | 122869.1 | 110266.8 | -10.26% | +12.93% |
| SpiralScanPath_13 | 732128.4 | 445931.1 | -39.09% | -1.62% |

Source values are in **stored HeatFluxZ units**. Their integrals have those units
times m2 under the existing coordinate convention. They are power in watts only
if HeatFluxZ is confirmed to be W/m2; file metadata has not established that here.
These comparisons use raw HeatFluxZ, including its negative values, not the
thresholded effective heating term or accumulated source energy.

### Area weighting and interpolation definitions

- Native nodal weights are lumped P1 triangle masses, `w_i = sum(area_K / 3)`.
  This integrates a piecewise-linear native temperature/source field exactly.
- The projected field is the existing `SurfaceGridProjector` output. Its spatial
  interpretation for this audit is bilinear reconstruction of the grid samples;
  tensor trapezoidal weights integrate that bilinear field exactly.
- Projection RMSE/MAE/bias compare both representations at the **same physical
  quadrature locations**, not arrays of different sizes. The triangle rule is
  applied after two subdivisions: 597,504 points. Increasing from one to two
  subdivisions changes reported whole-field RMSE by at most 0.0031%.
- Projection hottest-1%-area metrics use native truth on these common quadrature
  points and a fractional final weight. GP hottest-area metrics use the same
  area-ranking rule on lumped nodal weights. The latter is a nodal quadrature
  approximation to a physical region, not exact integration of a level set.
- Field error includes sampling/interpolation and the difference between native
  triangulated and grid-bilinear representations. It is not an isolated estimate
  of one interpolation algorithm's error.
- Area-mean temperature is background dominated. Small average changes do not
  establish preservation of peak shape, local gradients, or heating magnitude.

## Native inference implementation

The previous camera, current camera, source, truth, likelihood locations and GP
predictions all live on the original 6,275 surface nodes. No projected temperature
or projected source values enter these native forecasts or likelihoods.

The existing calibrated coefficients, ambient value, lengthscale convention,
reference ceiling, source threshold and sampling protocol are inherited. Grid
preparation is still called to recover frozen baseline metadata and make the
comparison figures, not to interpolate the native inference fields.

For the diffusion step, assemble the triangular P1 stiffness matrix S and lumped
mass matrix M. On interior nodes use

    A_h = exp(dt * (-alpha M^-1 S - beta I))
    m = T_amb + A_h max(mu_previous - T_amb, 0)
        + gamma * q * 1[q >= source_threshold] * dt.

This is a **2D surface heat approximation**. Zero excess temperature is imposed
on the 100 outer boundary nodes during propagation; the source is added afterward.
It is not the original 3D FEM solver, and this boundary treatment is not identical
to the old zero-extended Gaussian-filter approximation. The stored connectivity
is essential, but using it alone does not reproduce the original physics.

- Native B: the above physics mean plus the unchanged coordinate RBF residual.
- Native C: the same physics mean, with covariance
  `Q_free_space(dt) + Cov(A_h (previous_draw - previous_mean))`.
  The inherited free-space heat innovation Q is evaluated at native coordinates.
  **It is not a boundary-consistent FEM-discretized SPDE innovation.**
- Previous representation is retained honestly: unsaturated noisy measurements
  are pinned, while censored locations receive marginal predictive draws. It is
  not a coherent full latent spatial posterior. Propagating these draws retains
  the canonical approximation; this pilot does not repair it silently.
- The final likelihood receives current observations only. No raw previous
  observations are reused there. Truth is used for synthetic sensor generation,
  evaluation and the separately saved oracle forecast, not inference or tuning.
- The 117 old numerical-sensor candidate positions are mapped by geometry alone
  to unique nearest native nodes (maximum displacement 0.354 mm). All native
  observed-censored locations are also included. Native noise is independently
  generated on these native locations using the inherited seed convention.

These observation and discretization changes mean this is **not a controlled
native-versus-grid GP ranking**. Do not compare its numbers directly with old
top-25 metrics, or use it to replace the canonical architecture table.

## Pilot results and cost

DiagonalScanPath_8, last frame, frozen reference ceiling 296.1215 K:

- 1,213 previous censored nodes; 1,191 current censored nodes.
- Current likelihood: 115 numerical observations + 1,191 inequalities = 1,306.
- Current censoring is **18.98% of nodes but only 3.121% of surface area**.
- 1,600 previous retained draws; current evaluation uses 1,200 marginal draws.
- Previous maximum Rhat 1.00491; current B/C maximum Rhat about 1.00648.
  All pass the inherited 1.05 gate without retry. This is the existing convergence
  screen, not a proof of joint-posterior accuracy.

| Native pilot | Area-weighted field RMSE (K) | Overall CRPS (K) | Hot-1%-area RMSE (K) | Hot CRPS (K) | Hot 95% coverage | Hot width (K) |
|---|---:|---:|---:|---:|---:|---:|
| B: physics + RBF | 1.058 | 0.803 | 4.812 | 2.501 | 0.849 | 10.068 |
| C: physics + sequential covariance | 0.720 | 0.256 | 4.433 | 2.253 | 0.773 | 6.438 |

C has better point error/CRPS in this single pilot but narrower intervals and
lower hottest-area coverage. Peak absolute error is 18.24 K for B and 15.95 K
for C. Peak magnitude remains unresolved; no general winner is selected.
CRPS uses the retained M(M-1) pairwise correction.

Runtime is recorded separately for shared previous inference, physics propagation,
covariance construction and each current update in `native_pilot_runtime.csv`.
On this machine, the shared previous inference is roughly 11-13 s, the two mean
forecasts together under 0.01 s, and each current inference roughly 10-15 s.
C covariance construction/propagation is roughly 5 s. Both models plus local cache
and plot writing take about one minute, excluding catalog loading and the three
field-comparison preparations. Peak process RSS is about 2 GiB. These are pilot
measurements, not an all-trajectory scaling guarantee.

## Checks and caveats

- Existing Experiment-45 source fingerprints are asserted unchanged at startup.
- All 33 native datasets and final two temperature/source frames load and contain
  finite values; native geometry hashes and split are saved in `native_inventory.csv`.
- Native tests check area, linear integrals, orientation invariance, stiffness
  symmetry/constant nullspace, a closed-form diffusion-decay case, linear residual
  propagation, energy decrease and exact fractional hot-area weighting.
- Both pilot runs reproduced identical reported inference metrics with the same
  seeds. All covariance blocks, posterior moments and draws are checked finite.
- This host's NumPy/BLAS emitted floating-point status warnings in dense products.
  The high-variance cross-covariance subblock is independently recomputed with
  non-BLAS einsum and checked to 1e-12 tolerance; no discrepancy is accepted.
  Warnings are not suppressed. They are not evidence of NaNs in the saved results.
- No native parameter fitting, held-out inference sweep, calibration model or
  modification to the old canonical drivers has been performed.

## Outputs and reproduction

Run from the repository using its existing virtual environment:

```sh
env MPLCONFIGDIR=/private/tmp/thermal_mpl PYTHONPYCACHEPREFIX=/private/tmp/thermal_pycache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/51_thermal_native_mesh_pilot.py
```

- `native_projected_fields.csv`: both frames, all three development paths,
  temperature/source peaks, means, integrals, common-location area-weighted errors.
- `field_comparison_*.png`: native/projected/difference figures, with a shared
  nonlinear temperature scale that reveals the thermal tail.
- `native_reconstruction_DiagonalScanPath_8.png`: native reconstructions and
  shared-scale absolute-error maps.
- `native_pilot_metrics.csv`, `native_pilot_runtime.csv`,
  `native_pilot_configuration.json`: pilot scores, costs and assumptions.
- `frozen_configuration.csv`, `source_fingerprints.json`, `mesh_checks.csv`,
  `native_inventory.csv`: reproducibility and compatibility records.
- `by_trajectory/DiagonalScanPath_8/native_pilot.npz`: local, gitignored arrays
  and marginal posterior draws. Existing local results remain intact.

The next controlled question is whether representation and operator discretization
separately explain the oracle residual. This pilot demonstrates feasibility and
measures field differences; it does not yet isolate those two causes.
