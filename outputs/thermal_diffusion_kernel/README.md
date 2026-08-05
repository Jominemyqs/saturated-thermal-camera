# Thermal diffusion-kernel experiment

This folder contains the default run of `scripts/16_thermal_diffusion_kernel.py`
with a 61-by-41 prediction grid and every third unsaturated grid point retained.
All saturated pixels are retained.

The experiment:

- learns effective diffusivity from cooling pixels with
  `abs(HeatFluxZ) < 300`;
- estimates the laser-source width from active heat-flux footprints;
- transfers diffusivity and signal scale leave-one-trajectory-out;
- compares source-scale and unsaturated-tuned isotropic/diffusion kernels;
- pools three censored-GP sampling chains; and
- reports field error, peak error, coverage, and empirical CRPS.

Main files:

- `effective_diffusivity_estimates.csv/.png`: learned physical parameters.
- `lengthscale_selection.csv/.png`: unsaturated marginal-likelihood sweep.
- `diffusion_kernel_results.csv`: all reconstruction and calibration metrics.
- `diffusion_kernel_metric_comparison.png`: key aggregate metrics.
- `*reconstruction.png`: truth, clipped field, cooling age, saturation mask,
  and posterior means.

The sibling folder `thermal_diffusion_kernel_dense` repeats the same experiment
while retaining every second unsaturated grid point.

Preliminary conclusion: the source-scale priors reconstruct more of the hot
field than unsaturated-only tuning, which selects an overly smooth prior.
The diffusion-shaped kernel helps on some trajectories but is not consistently
better, especially for the spiral, and the sharp true peak remains uncovered.
