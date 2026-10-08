# Physical interpretation and provenance

This experiment does not change the frozen predictor. The new information
changes its interpretation and identifies checks for the generating notebook.

## HeatFluxZ

Adrienne's supplied notebook expression is:

```python
grad_T_z = project(T.dx(2), W_grad_z)
grad_T_z.rename("HeatFluxZ", "HeatFluxZField")
```

Assuming this is the export used in the local data, HeatFluxZ is a projected
vertical temperature derivative, in K/m for metre coordinates. It is not by
definition the prescribed boundary heat flux in W/m^2. The existing increment
`gamma * HeatFluxZ * (HeatFluxZ >= threshold) * dt` is therefore an empirically
calibrated gradient-based contribution. Gamma has units m/s under this
interpretation; the existing threshold is a gradient threshold in K/m.
Legacy code/configuration identifiers retain their names for reproducibility.
No conductivity multiplication, sign change, threshold change or recalibration
has been performed.

The derivative is computed from the simulated solution. Its availability from
laser commands in deployment is not established by this experiment. In
particular, using the current simulated gradient is external information,
not a quantity measured by the clipped thermal camera alone.

## Thesis reference

Source: MasterThesis_Trentini.pdf supplied by the user, printed pages 10-15.

- Pages 10-11 give Fourier's law q = -k grad(T), the heat balance, and inward
  boundary-flux convention -n dot q = q_bar. The thermal weak form used there
  neglects the volumetric source Q and applies boundary heating instead.
- Pages 12-13 describe a 15 x 10 x 2 mm domain, refinement near its top surface,
  fixed bottom temperatures, insulated sides and a moving Gaussian top flux.
  Initial temperature is 293.15 K. Constant thermal properties listed are
  rho = 7860.51 kg/m^3, Cp = 472.84 J/(kg K), k = 15.11 W/(m K).
- On a top boundary with outward normal +e_z, the stated convention implies
  k * dT/dz = q_bar. This is a boundary relation, not proof that the projected
  exported derivative equals the prescribed boundary function pointwise.
- Page 11 calls the discretization forward Euler, but its written diffusion
  and boundary terms are evaluated at n+1. The formula and label disagree;
  the notebook is needed to confirm the actual time discretization.
- Page 15 states dt = 0.005 s. `dataset_timestamps.csv` checks all 33 local
  trajectories: the stored steps, including the two target frame pairs, are
  0.0005 s (up to floating-point representation). The experiment uses those
  stored times, not the thesis value. This is an unresolved version/provenance
  difference, not grounds for silently changing the data or model.

## Limits of physical attribution

The current model is an effective 2D surface forecast with Gaussian diffusion,
cooling, a positive-excess floor and a thresholded empirical gradient increment.
Native inputs are evaluated through the same fine computational grid as
Experiment 49; this is not the original 3D solver or a new native FEM operator.
Depth conduction and bottom-boundary effects may be partly absorbed by fitted
coefficients. The thesis does not establish that missing depth transfer alone
causes the observed positive oracle forecast error.

The component attribution in output 51 remains valid numerically. Its
`source` component should now be read as the gradient-based increment, and its
`needed` residual is still not a measurement of the true physical heat input.

Next provenance request: the exact generating notebook/version, exported
coordinate/time units, top-surface normal, prescribed boundary-flux function,
and the space/projection used for HeatFluxZ. Keep any physical correction
separate from the present ceiling-only experiment.
