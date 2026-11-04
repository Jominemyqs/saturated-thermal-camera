# Physics-input provenance / equivalence audit

2026-11-04. Output 62, [inspection driver 66](../../../scripts/66_thermal_physics_input_provenance.py).
**Status: provenance audit completed; numerical flux-equivalence comparison blocked.**
No GP, propagation, fitting, source replacement or baseline change was performed.

## Confirmed versus unresolved

The reviewed local project contains exported FEM fields, but no matching FEM
generator notebook, source-expression configuration, or generator revision.
The available thesis and Adrienne's supplied export snippet are separate evidence,
not a substitute for the executed generator. See the [provenance table](provenance.md)
first, then the [FEM/surrogate correspondence](model_correspondence.md).

| Directly checked fact | Result |
| --- | --- |
| Stored fields | Nodal scalar `Temperature` and `HeatFluxZ` only; no independently exported laser-input field |
| Mesh | Both cases: 26,414 volume nodes, 130,664 tetrahedra; 6,275 top nodes, 12,448 surface triangles |
| Geometry | x in [-0.0075,0.0075], y in [-0.005,0.005], z in [-0.001,0.001]; surface area 0.00015 in stored coordinate units squared |
| Native cache identity | Coordinates, connectivity, area weights and last-two temperature fields exactly match output 52; all four gradient hashes match output 56 |
| Metadata | No HDF root attributes or XDMF Information records; the only object attribute is `partition=[0]` |
| Missing generator evidence | PDE assembly, facet markers, material settings, laser expression/parameters, trajectory functions, integration clock and export loop |

Metres, seconds and kelvin are supplied-context/downstream conventions, not
explicit unit attributes in these files. The geometry is consistent with the
thesis domain dimensions; that does not authenticate its other parameters.

| Trajectory | Stored frames | Previous index / time | Current index / time | Saved interval |
| --- | ---: | --- | --- | ---: |
| DiagonalScanPath_8 | 355 | 353 / 0.1770 | 354 / 0.1775 | 0.0005 |
| SpiralScanPath_13 | 544 | 542 / 0.2715 | 543 / 0.2720 | 0.0005 |

Indices are zero-based; times are interpreted as seconds by the pipeline.
Temperature and gradient have the same XDMF timestamp, but the export loop is
needed to establish their relationship to the actual source-evaluation time.
**Saved frame interval is not evidence of solver timestep.** The thesis states
0.005 s, while these files are saved at 0.0005 intervals. Whether this reflects
a different dataset version, a documentation error or another time convention
is unresolved. No time shift or rescaling was estimated.

## Why numerical comparison stops here

The supplied snippet `project(T.dx(2), W_grad_z)` identifies a solution-derived
vertical derivative, conditional on that export being the one used here. The
projection space and exact generator/export correspondence remain unverified.

The thesis supplies a Gaussian boundary formula, but not the executable
Diagonal-8/Spiral-13 trajectory rules. Its power, width, material values,
absorption interpretation and switch time are not authenticated for these files.
Thus neither an independent prescribed flux nor its conversion with the actual
conductivity can be constructed defensibly.

**No prescribed-flux maps, relative-L2 errors, source-region scores, amplitude
ratios or equivalence claims are reported.** They are unavailable, not zero.
No gradient centroid, temperature-derived fit, gamma or cutoff was used to
manufacture an independently specified input. The [data metadata](data_metadata.json)
contain identity/timestamp checks only, not source-equivalence results.

The geometric upper surface has outward normal +e_z. If the generator uses
Fourier's law q=-k grad(T) and inward flux q_in=-n dot q, then q_in=k*dT/dz
on that surface. This is a **conditional boundary identity**, not verification
of the generator's sign convention or of pointwise equality after projection.
Weak boundary enforcement and derivative projection can cause numerical
differences even for a correct FEM solution.

## Next bounded step

Ask Adrienne for the small matching generator/configuration subset listed in
[open questions](open_questions.md), ideally with a mapping to the two local
HDF/XDMF file hashes. No new simulation is required merely to supply provenance.
If confirmed, compare raw prescribed inward flux with k times the exported raw
gradient on the four recorded frames. Include whole-area and independently
source-centred local diagnostics; do not fit alignment or normalization.

Even close agreement would not validate `gamma*gradient*dt` as a surface
temperature increment. Boundary flux-to-temperature conversion depends on the
heat capacity, depth dynamics, boundary conditions and time response of the
thermal system. It would also not make a solution-derived derivative available
in deployment. Numerical equivalence and independent availability remain two
distinct questions.

## Reproducibility

- [Provenance / parameter table](provenance.md)
- [Original-model / surface-predictor correspondence](model_correspondence.md)
- [Search scope, checks and limitations](verification.md)
- [File fingerprints and initial git status](fingerprints.json)
- [Executed audit configuration](audit_configuration.json) and [machine checks](verification.json)

Existing outputs 59-61 and all uncommitted work are preserved. This task adds
only the isolated audit driver and report directory. No existing code, protocol,
inventory, configuration, scientific result or cache is modified.
