# Thermal model correspondence

**The original generator is unavailable in the reviewed scope.** The FEM column
below is explicitly the thesis formulation, not a verified reconstruction of
the code that generated these files. The surface-predictor column is executed
code traced through [driver56](../../../scripts/56_thermal_native_higher_ceiling.py)
`main`113-159 and reused in output61. Native P1 adapters and integration variants
from other diagnostics are not silently substituted for this baseline.

| Component | Thesis FEM formulation | Current-data generator confirmation | Executed surface predictor |
| --- | --- | --- | --- |
| State and domain | 3D temperature;15x10x2 mm block | 3D tetrahedral export and dimensions consistent; PDE assembly unresolved | Native top-node temperatures only; propagation via fine 2D Cartesian grid |
| Thermal storage | rho*Cp*dT/dt | Material values/implementation unresolved | No explicit rho, Cp, mass matrix or effective depth; fitted alpha,beta,gamma |
| Lateral conduction | div(k grad(T)) includes x,y | Unresolved assembly/discretization | Gaussian filter with widths sqrt(2*alpha*dt)/grid_spacing |
| Vertical conduction | Same 3D operator includes z | Volume mesh confirmed, vertical assembly unresolved | No z state, second derivative or explicit transfer to lower layers; effective cooling may absorb some effects, not demonstrated |
| Heating input | Prescribed moving Gaussian inward boundary flux; volume Q omitted in stated weak form | Expression/run parameters and facet load unresolved | Current simulated projected dT/dz, thresholded, multiplied by gamma*dt and added at endpoint |
| Bottom boundary | Fixed temperature | Actual values/markers unresolved | No represented bottom surface |
| Side boundary | Adiabatic | Actual facet condition unresolved | Zero-extended excess outside the 2D rectangle (`mode='constant',cval=0`); not the same as insulated sides or a 3D bottom boundary |
| Cooling | Conduction and stated boundary conditions; no separate beta term shown | Extra implemented loss terms unresolved | Uniform exp(-beta*dt) decay of propagated excess |
| Initial/reference temperature | T0=293.15 K | Actual initial-condition code unresolved; first stored frame is not time zero | Cached, simulation-derived ambient quantile; noisy previous latent estimate supplies state |
| Mean nonlinearity | No positive-excess floor stated | Generator unresolved | Interpolate excess, then max(excess,0) on fine grid before diffusion; covariance propagation omits floor |
| Time stepping | Eq2.7 has n+1 diffusion/input; caption says forward Euler | Time loop, source phase and substeps unresolved | One saved-frame interval; predecessor propagated, then entire thresholded current gradient increment added |
| Input interpolation | Gaussian evaluated on FEM boundary (description) | FE spaces/quadrature unresolved | Triangle interpolation of predecessor to grid, bilinear return; current native gradient increment added directly at native nodes |
| Stochastic uncertainty | Deterministic thermal formulation described | Solver/model uncertainty not specified | Frozen Q plus propagated previous covariance; not changed in this audit |

## Exact implemented surrogate structure

Let R map native values to the fine grid, E return them to native nodes, and
G_dt be the actual zero-extended Gaussian filter. With cached ambient a,

```text
base = a + E [ exp(-beta*dt) G_dt max(R(previous_mean - a), 0) ]
increment = gamma * dt * g_current * 1[g_current >= 10000]
forecast = base + increment
```

This is [MatchedHeatOperator.propagate/mean](../../../src/matched_projection.py),
66-97. The floor occurs after interpolation; this is not asserted to equal
propagating a differently floored native field. Source thresholding and the
endpoint increment do not undergo that same within-step propagation. Output56
tested alternatives but did not replace this baseline.

## Conditional boundary relation, not a temperature update

With verified Fourier sign convention and top normal +e_z:

```text
q = -k grad(T)
q_in = -n dot q = k * dT/dz
```

This does not imply pointwise equality of a projected derivative export with
the prescribed boundary expression: weak enforcement, derivative-space projection,
mesh resolution, quadrature and solve/export timing can all affect the comparison.
Those are possible explanations to test, not identified causes in this audit.

Even exact boundary-flux agreement does not establish the surrogate increment:
`gamma*g*dt` has K units under the supplied conventions, but dimensional consistency
is not a derivation. A flux-driven surface response also depends on volumetric
heat capacity, depthwise temperature evolution, geometry, boundaries and time.
No effective thickness, absorption coefficient or new gamma is inferred here.
