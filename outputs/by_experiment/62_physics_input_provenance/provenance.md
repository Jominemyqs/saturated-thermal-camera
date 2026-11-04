# Provenance and parameter inventory

Evidence classes: **D** = directly checked local data; **S** = executed surrogate
code/configuration; **T** = thesis description only; **E** = supplied email snippet.
T/E evidence does not establish the exact generator version for D.

Source pointers below use function names and inspected line ranges. The PDF is
`/Users/qiushiyu/Downloads/MasterThesis_Trentini.pdf`; printed/PDF pages coincide.
Both available PDF copies are fingerprinted; neither is generation code.

## Generator and independent inputs

| Quantity / definition | Evidence and value | Units | Available without current FEM temperature? | Source / location | Confirmed for these files? |
| --- | --- | --- | --- | --- | --- |
| Generator revision and run configuration | Not found in reviewed local scope | n/a | Yes, if supplied | [Search record](verification.md); no matching code position exists | **unresolved** |
| Thermal PDE | T: q=-k grad(T), rho*Cp*dT/dt+div(q)=Q | SI heat balance | Yes, as a model specification | Thesis p10, eqs2.1-2.5 | **unresolved** executed assembly |
| Weak thermal loading | T: volume Q neglected; boundary integral of prescribed inward flux | W after integration against dimensionless test function | Yes | Thesis p11, eqs2.6-2.7 | **unresolved** generator |
| k | T: 15.11 | W/(m K) | In principle material input; current value unverified | Thesis p13 | **unresolved**; not used to convert gradients here |
| rho | T: 7860.51 | kg/m^3 | In principle yes | Thesis p13 | **unresolved** |
| Cp | T: 472.84 | J/(kg K) | In principle yes | Thesis p13 | **unresolved** |
| Material dependence | T: constant properties at293.15 K for linear equations; introductory equations allow temperature dependence | n/a | Constant table yes; state-dependent evaluation may require T | Thesis pp10,12-13 | **unresolved** executed choice |
| Laser power P | T:90 | W | Yes if prescribed; whether incident or absorbed is unresolved | Thesis p13, eq2.15 | **unresolved** |
| Width sigma | T:1.25e-5, appearing as (2*sigma)^2 in formula | m | Yes if prescribed | Thesis p13, eq2.15 | **unresolved**; not inferred from gradient footprint |
| Gaussian normalization | T:2P/[pi(2*sigma)^2] * exp[-2*r_focus^2/(2*sigma)^2] | W/m^2 | Yes given actual laser parameters | Thesis p13, eq2.15 | **unresolved**; formula not executed |
| Absorption / efficiency | No separate factor shown in thesis eq2.15; not evidence that actual eta=1 | dimensionless | Yes if independently specified | Missing generator and parameter dictionary | **unresolved**, including whether incorporated into P |
| x_focus(t), y_focus(t) | T names path families and turn counts; no exact local run functions | m; time convention unresolved | Yes if prescribed trajectory is supplied | Thesis p15; local filenames identify family/index only | **unresolved** for Diagonal8 and Spiral13 |
| Speed, initial position, turn spacing, phase and time origin | No executable rule/configuration available | m/s, m, s as applicable | In principle yes | Missing generator trajectory definitions | **unresolved**; no centroid or fitted substitute |
| Laser switch | T: nonzero if t<=t_laserOn; zero otherwise | s | Yes if actual schedule supplied | Thesis eq2.15 and p15 | **unresolved** actual switch times and endpoint semantics |
| Inward/outward flux convention | T:q_in=-n dot q; q=-k grad(T) | W/m^2 | Yes from boundary formulation | Thesis p10 eqs2.1,2.5; no assembly/facet markers | **unresolved** for generator |
| Geometric upper normal | D:max-z plane of a volume lying below it; +e_z | unit vector | Yes from mesh | HDF `/Mesh/0/mesh/geometry` and topology; [reader](../../../src/thermal_trajectory.py) `surface_node_indices`,186-194 | Confirmed geometry; generator facet label not confirmed |
| Bottom and side boundary conditions | T:fixed bottom temperature, adiabatic sides; initial293.15 K | K; W/m^2 | Yes as specification | Thesis p13 | **unresolved** actual expressions/values/markers |
| Solver timestep | T:0.005; local saved spacing0.0005 does not determine it | s by supplied context | Yes if solver loop supplied | Thesis p15; XDMF Time; no time loop | **unresolved** |
| Time integrator | T says forward Euler but eq2.7 evaluates diffusion/load at n+1 | n/a | Yes from code | Thesis p11 eq2.7 | **unresolved**; formula/label conflict retained |

## Exported fields and saved-time identity

| Quantity / definition | Evidence and value | Units | Available without current FEM temperature? | Source / location | Confirmed for these files? |
| --- | --- | --- | --- | --- | --- |
| Temperature export | D:nodal scalar; current and previous datasets mapped in metadata | K by supplied context; no unit attribute | No for latent truth | [Reader](../../../src/thermal_trajectory.py) `read_field`,200-225; [dataset paths](data_metadata.json) | Arrays and cache identity confirmed |
| HeatFluxZ meaning | E:`grad_T_z=project(T.dx(2),W_grad_z)` then rename to HeatFluxZ | K/m if coordinates are m; export metadata unlabelled | **No under supplied export**; depends on simulated T | [Recorded supplied snippet](../52_native_higher_ceiling/PHYSICS_PROVENANCE.md),8-20 | Meaning supported by communication; exact generator correspondence **unresolved** |
| Derivative projection | E:project into W_grad_z; family, degree, solver and boundary treatment missing | K/m under above convention | No under supplied export | Missing definition of W_grad_z / export cell | **unresolved**; nodal storage does not determine FE space |
| Field time alignment | D:Temperature and HeatFluxZ share each Time grid | s convention | Saved metadata yes; actual load phase requires loop | XDMF final two grids; [reader](../../../src/thermal_trajectory.py),129-159 | Saved labels confirmed; solve/project/write order **unresolved** |
| Saved interval | D:0.0005 up to floating-point representation | s convention, not encoded unit | Yes | [Last-two frame metadata](data_metadata.json) | Confirmed label interval, not solver step |
| Mesh size and bounds | D:26414 nodes,130664 tetrahedra;15x10x2e-3 extent | m by context; no unit attribute | Yes | HDF mesh datasets; [metadata](data_metadata.json) | Geometry confirmed; SI scale context-supported |
| Surface weights | D:6275 nodes,12448 triangles; sum0.00015 | m^2 if metre coordinates | Yes, geometry only | [native_surface](../../../src/native_surface.py) `surface_mesh`,22-46, `extract_surface`,49-57 | Exactly matches current native cache |

## Current surrogate settings (not FEM generator parameters)

Values below come from [output52 frozen configuration](../52_native_higher_ceiling/frozen_configuration.csv)
and match the inherited output61 settings. No values were estimated in this audit.

| Quantity | Value / rule and units | Provenance / availability | Code / artifact | Status |
| --- | --- | --- | --- | --- |
| Effective alpha | 3.6109129986233175e-6 m^2/s | Historical simulator-history fit; fixed now, not verified k/(rho*Cp) | [Development physics CSV](../24_posterior_physics_mean_advection/development_physics_parameters.csv); [diffusion](../../../src/diffusion.py) `estimate_effective_diffusivity`,55-152 | S confirmed; not original material value |
| Effective beta | 12.422901570529978 s^-1 | Same historical calibration; not an identified physical boundary loss coefficient | Same CSV/function | S confirmed |
| Gamma | 0.1702227206133784; dimensionally m/s under supplied gradient/time units | Historical nonnegative least-squares fit using clipped simulated predecessors and current truth; not laser absorption or measured material property | [Coupling CSV](../24_posterior_physics_mean_advection/source_coupling_calibration.csv); [physics](../../../src/thermal_posterior_physics.py) `build_calibration_components`549-575 / `calibrate_source_coupling`578-591 | S confirmed; physical interpretation unresolved |
| Current increment cutoff | g>=10000; retain positive gradients only; K/m convention | Prescribed inherited rule, not part of verified FEM input | [MatchedHeatOperator.mean](../../../src/matched_projection.py),93-97 | S confirmed; no cutoff applied in audit |
| Preparation gradient cutoff | abs(g)<300, K/m convention | Simulator-derived historical cooling fit eligibility; separate from increment cutoff | [prepare_trajectory](../../../src/thermal_posterior_physics.py),113-127 | S confirmed |
| Ambient reference | Diagonal293.1497049086644; Spiral293.1499365025225 K | Cached full-history projected-temperature 2nd percentile; **not independently camera-only** | [diffusion](../../../src/diffusion.py),91-92; output52 cache | S confirmed; distinct from thesis T0=293.15 |
| Prepared source/kernel length | Gradient-footprint estimate floored at coarse-grid spacing, then inherited multiplier | Depends on simulator gradients; **not independently specified laser sigma** | [prepare_trajectory](../../../src/thermal_posterior_physics.py),103-112 | S confirmed; not used to reconstruct laser |
| One-step interval | Last two saved timestamps;0.0005000000000000004 | Independent of current field values, but unit convention inherited | [driver56](../../../scripts/56_thermal_native_higher_ceiling.py) `main`,113-134 | S/D confirmed; does not identify solver substeps |
| Propagation/interpolation | Native triangle interpolation ->601x401 grid->Gaussian filter/cooling->native bilinear evaluation | Existing numerical approximation; no depth state | [MatchedHeatOperator](../../../src/matched_projection.py),47-97 | S confirmed; not native 3D FEM |

The gradient export, prepared lengths/ambient and historical coefficient fitting
have different dependencies. Calling the whole method camera-only would obscure
these dependencies even if gamma is frozen at prediction time. See the existing
[review inventory](../../../docs/adrienne_review/README.md) for the fuller
statistical-parameter audit; it is not copied or changed here.
