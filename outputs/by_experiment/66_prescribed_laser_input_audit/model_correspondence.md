# Supplied FEM versus frozen surface predictor

This supersedes unresolved entries in the historical Output62 correspondence
only where the newly supplied notebooks provide evidence. It does not change
any predictor. Sources: [Diagonal cell6](Diagonal_cell6_source.txt),
[Spiral cell6](Spiral_cell6_source.txt), and
[MatchedHeatOperator.propagate/mean](../../../src/matched_projection.py), L66-97.

## Original deterministic formulation

The implemented weak step, for test functions vanishing at the bottom, is

```text
(rho*Cp/dt) (T_new - T_old, v)_volume
  + (k*grad(T_new), grad(v))_volume = (g(t_new), v)_top.
```

Material coefficients are constant. This corresponds to 3D conduction with
thermal storage, a prescribed inward top flux, a fixed-temperature bottom,
and homogeneous natural side flux. It is backward/implicit Euler for diffusion
and uses endpoint g. There is no separate volumetric heat-source or beta term
in the supplied weak form. The saved derivative is a CG1 volume projection
restricted to the top nodes by our reader, not the analytic boundary expression.

## Implemented surrogate

Let R map native top nodes to the 601 x 401 internal Cartesian grid, E return
values to native nodes, and G be the zero-extended Gaussian filter. The exact
operational structure (including floor placement) is

```text
off = a + E { exp(-beta*dt) G_dt [max(R(previous - a), 0)] }
s   = gamma*dt*gradient_current*1[gradient_current >= 10000]
on  = off+s.
```

| Component | Supplied FEM | Frozen surface predictor |
| --- | --- | --- |
| State | P1 3D volume temperature | Native top surface; no subsurface state |
| Storage | Explicit rho*Cp mass term | No explicit capacity or effective thickness; empirical coupling |
| Horizontal conduction | k times full gradient in implicit stiffness solve | Gaussian diffusion with fitted alpha |
| Vertical conduction | Included in 3D gradient and coupling to bottom | No explicit z derivative or depth evolution |
| Material diffusivity | k/(rho*Cp) = 4.06536488e-6 m2/s | Frozen alpha = 3.610912999e-6 m2/s; not replaced by material value |
| Input | Independent moving Gaussian g, W/m2, weak boundary load | Solution-derived projected vertical gradient, thresholded, empirical gamma*dt increment |
| Bottom | 293.15 K Dirichlet | No bottom state/condition represented explicitly |
| Sides | Natural zero flux | Zero-extension of 2D excess; not the same boundary condition |
| Cooling | Through conduction and boundaries; no beta term | exp(-12.42290157*dt) effective decay |
| Ambient | Initial and bottom value 293.15 K | Cached simulation-preparation ambient, retained exactly |
| Spatial representation | Refined tetrahedral mesh and FE projection | Native input, triangle-to-grid interpolation, bilinear return; not interpolation-free |
| Positivity/floor | No such excess floor in weak form | Floor after native-to-grid map before diffusion; covariance uses linear propagation without this floor |
| Time/input handling | Endpoint input inside implicit heat balance | Predecessor propagated for one saved interval, then full current increment added |
| Uncertainty | Deterministic FEM solve | Q plus inherited full-latent covariance in latest C; not exercised or modified here |

## Why flux equivalence would not be enough

At the top, the continuum condition supports comparing g and k*dT/dz with
the stated sign. But a surface-temperature increment is not obtained merely
by dividing that flux by k and multiplying by an empirical gamma. The response
also depends on storage, depth profile, geometry, boundary conditions and time.
A true previous **surface** does not supply the entire previous 3D state.

This audit finds a local discrepancy even before that temperature-response
question: the projected gradient has lower sampled peaks and spatial shifts
relative to the prescribed g. The responsible numerical mechanisms were not
isolated. Neither exact total-flux agreement nor the oracle forecast mismatch
identifies a unique corrective term, thickness, absorption or new gamma.
