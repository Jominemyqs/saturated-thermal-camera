# Thermal camera censored Gaussian experiment

## Current thermal-camera workflow

The current primary architecture is:

```text
previous censored camera frame
-> previous-frame censored RBF posterior
-> posterior physics mean
-> current-frame spatial RBF residual
-> current censored likelihood
```

The ten experiments to use in current notes and presentations are:

1. corrected unbiased CRPS in `src/metrics.py` and script 19;
2. the supporting stochastic space-time kernel ablation in script 21;
3. the posterior-physics-mean propagation ablation in script 22;
4. the controlled posterior-physics-mean advection experiment in script 26.
5. the broad retained-model comparison and architecture audit in script 27.
6. the architecture-aware one-step sequential comparison in script 28.
7. the focused mean-only versus moment-matched versus posterior-sample mixture
   propagation experiment in script 29;
8. the frozen hottest-tail uncertainty-scale diagnostic and variance-matched
   RBF control in script 30;
9. the final seven-architecture consolidation and scale-aware interpretation in
   script 32;
10. the frozen uncertainty-origin and true-previous-state oracle diagnostic in
    script 33.

Script 28 is a deliberately limited one-step sequential GP exploration. It
uses the canonical RBF previous censored posterior, compares stationary and
advective stochastic residual propagation without reusing the previous
observations, and labels the older joint two-frame result only as a historical
reference. Its global scores are strong, but its hot-region intervals are too
narrow, so it remains a supporting diagnostic.

Run the current advection comparison with:

```bash
python scripts/26_thermal_posterior_physics_advection.py
python scripts/27_thermal_broad_model_comparison.py
python scripts/28_thermal_one_step_sequential_gp.py
python scripts/29_thermal_full_posterior_sequential.py
python scripts/30_hottest_tail_diagnostic.py
python scripts/32_thermal_final_architecture_comparison.py
python scripts/33_thermal_uncertainty_origin_oracle.py
```

See `EXPERIMENT_INVENTORY.md` for canonical and supporting experiments.
Superseded sequential SPDE/filtering, joint/double-use, and broad-restart
drivers were removed after the controlled audit and are not current model
results. Historical outputs remain available until the cleanup list is
explicitly confirmed.


The synthetic field is

```math
T_{true}(x,y) = T_0 + A\exp\left[-\frac{(x-x_c)^2}{2\sigma_x^2} - \frac{(y-y_c)^2}{2\sigma_y^2}\right].
```

The observed camera image is clipped by

```math
T_{obs}(x,y) = \min\{T_{true}(x,y), T_{max}\}.
```

The code compares four treatments of saturated pixels:

1. `exact`: treat clipped values as exact observations.
2. `discard`: remove saturated pixels from the loss.
3. `hinge`: fit unsaturated pixels and penalize saturated predictions below `Tmax`.
4. `censored`: use a censored Gaussian likelihood for saturated pixels.

## Setup

```bash
cd thermal_censored_gaussian
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/00_make_synthetic.py
python scripts/01_fit_single.py
python scripts/02_sweep_censoring.py
python scripts/03_plot_sweep.py
python scripts/06_model_misspecification.py
python scripts/07_plot_reconstruction.py
python scripts/08_gp_1d_censored.py
python scripts/09_gp_1d_robustness.py
python scripts/10_gp_2d_censored.py
python scripts/11_gp_2d_shape_density.py
python scripts/12_gp_2d_multiseed.py
python scripts/13_gp_2d_kernel_comparison.py
python scripts/14_gp_2d_path_kernel.py
python scripts/15_gp_2d_path_censored_tuning.py
python scripts/16_thermal_diffusion_kernel.py
python scripts/17_thermal_spatiotemporal_physics.py
```

The fixed two-frame calibration and 33-trajectory ablation can then be run with:

```bash
python scripts/18_thermal_two_frame_ablation.py
```

Use `--stage sensitivity` to run only the compact residual-hyperparameter and
two-versus-four-frame checks. After reviewing the selected setting, use
`--stage full` to run the four-model ablation at all synthetic ceilings.

Outputs are written to `outputs/`.

The thermal-trajectory experiment expects the downloaded data in the sibling
directory `../heat_eq_laser_trajectories`. A different location can be supplied
with `--dataset-dir`.

## Thermal trajectory experiment

The raw-data loader reads the XDMF metadata and accesses the linked HDF5 fields
lazily. It extracts the top surface, interpolates it to a regular grid, and
provides temperature and vertical heat-flux histories.

The first physics-shaped prior estimates an effective surface diffusivity from
cooling pixels:

```math
\frac{\partial T}{\partial t}
=
\alpha_{\mathrm{eff}}\nabla^2T
-\beta(T-T_{\mathrm{amb}}).
```

Pixels with substantial applied `HeatFluxZ` are excluded from this fit. The
learned diffusivity enters a nonstationary Gibbs kernel through

```math
\ell_i^2 = \ell_0^2 + 2\alpha_{\mathrm{eff}}a_i,
```

where \(a_i\) is the time since the local temperature maximum. Script 16
compares this diffusion-shaped kernel with an isotropic RBF using both a
heat-source-scale base lengthscale and unsaturated-likelihood tuning. It uses
leave-one-trajectory-out physical parameters and reports field error, peak
error, coverage, and CRPS.

Script 17 tests two refinements. First, its space-time heat covariance is

```math
k(r,\Delta t)
=\sigma_f^2 e^{-\beta|\Delta t|}
\frac{\ell_0^2}{\ell_0^2+2\alpha_{\mathrm{eff}}|\Delta t|}
\exp\!\left[-\frac{r^2}{2(\ell_0^2+2\alpha_{\mathrm{eff}}|\Delta t|)}\right].
```

Second, its physics-informed mean diffuses the previous clipped camera frame
and adds the current known laser-flux footprint. The source coupling is
calibrated on other uncensored simulation trajectories and transferred to the
held-out trajectory. The default comparison uses the final frame and the frame
10 ms earlier; a longer four-frame history is retained as a sensitivity study.

## Main files

- `src/gaussian_field.py`: grid, true field, and censoring model.
- `src/losses.py`: exact, discard, hinge, and censored objectives.
- `src/fit.py`: parameter fitting with multi-start L-BFGS-B.
- `src/metrics.py`: field, peak, and parameter errors.
- `src/uncertainty_diagnostics.py`: percentile/region calibration summaries and explicit zero-posterior-SD diagnostics.
- `src/thermal_trajectory.py`: lazy XDMF/HDF5 trajectory loader and top-surface grid interpolation.
- `src/diffusion.py`: effective-diffusivity fitting and local cooling-age calculation.
- `src/censored_gp.py`: reusable spatial RBF censored-GP inference.
- `src/dense_censored_gp.py`: censored inference for a precomputed dense Gaussian prior.
- `src/stochastic_heat_gp.py`: stationary stochastic heat covariance and one-step moment-matched propagation.
- `src/thermal_posterior_physics.py`: hybrid and coherent full-latent previous-frame censored posteriors and ordinary/advective posterior physics means.
- `src/thermal_plotting.py`: shared tail-focused thermal normalization and fixed excess-temperature contours for comparable reconstruction figures.
- `scripts/00_make_synthetic.py`: generate one synthetic image.
- `scripts/01_fit_single.py`: fit all methods on one image.
- `scripts/02_sweep_censoring.py`: compare methods over censoring levels.
- `scripts/03_plot_sweep.py`: plot summary metrics.
- `scripts/06_model_misspecification.py`: compare methods when the true field is not the fitted Gaussian model.
- `scripts/07_plot_reconstruction.py`: plot true, clipped, and fitted reconstruction fields.
- `scripts/08_gp_1d_censored.py`: 1D GP proof-of-concept comparing clipped, discard, censored, and oracle observations, including a lengthscale sweep, sampled censored-GP check, and one-standard-deviation uncertainty plots.
- `scripts/09_gp_1d_robustness.py`: repeat the 1D GP experiment over random seeds, tune lengthscale by optimizing the unsaturated marginal likelihood, and summarize peak coverage.
- `scripts/10_gp_2d_censored.py`: first 2D GP proof-of-concept on a clipped Gaussian field, including reconstruction and posterior-SD maps.
- `scripts/11_gp_2d_shape_density.py`: test 2D GP reconstruction on axis-aligned, rotated-wake, and moving-laser-path fields while varying observation density and selecting lengthscale from unsaturated observations.
- `scripts/12_gp_2d_multiseed.py`: repeat the 2D shape experiment over random seeds, comparing fixed lengthscale with lengthscale selected from unsaturated marginal likelihood.
- `scripts/13_gp_2d_kernel_comparison.py`: compare isotropic and anisotropic 2D GP kernels, including unsaturated-MLL tuned and physics-informed anisotropic variants, with seed-0 reconstruction panels.
- `scripts/14_gp_2d_path_kernel.py`: compare ordinary 2D kernels with a path-aligned kernel for the moving-laser field, with additional axis-Gaussian and rotated-wake reconstruction checks.
- `scripts/15_gp_2d_path_censored_tuning.py`: compare fixed path-aligned hyperparameters with hyperparameters selected by unsaturated marginal likelihood and full censored marginal likelihood.
- `scripts/16_thermal_diffusion_kernel.py`: load simulated laser trajectories, learn cooling diffusivity and source scale, and compare isotropic with diffusion-shaped censored GPs.
- `scripts/17_thermal_spatiotemporal_physics.py`: compare snapshot, space-time heat-kernel, calibrated physics-mean, and combined censored GPs on held-out laser trajectories.
- `scripts/18_thermal_two_frame_ablation.py`: calibrate the fixed two-frame residual, compare two and four frames, and run the four-model ablation across every thermal trajectory and synthetic ceiling.
- `scripts/19_thermal_corrected_crps.py`: controlled scoring with the unbiased empirical CRPS estimator.
- `scripts/21_thermal_stochastic_spde_ablation.py`: supporting stochastic heat-process covariance ablation.
- `scripts/22_thermal_previous_posterior_propagation.py`: observed-clipped, posterior physics mean, and full-posterior propagation ablation with an RBF residual.
- `scripts/26_thermal_posterior_physics_advection.py`: controlled ordinary-versus-advective posterior physics mean comparison with an identical RBF residual.
- `scripts/28_thermal_one_step_sequential_gp.py`: exploratory one-step sequential combination of the posterior physics mean and stochastic space-time residual covariance.
- `scripts/29_thermal_full_posterior_sequential.py`: controlled mean-only, moment-matched, and full-posterior sequential uncertainty propagation comparison.
- `scripts/30_hottest_tail_diagnostic.py`: frozen held-out hottest-tail coverage diagnostic with pixel maps, percentile calibration, covariance-scale checks, and a variance-matched RBF control.
- `scripts/31_prepare_friday_meeting.py`: regenerate the three-figure Friday meeting package and concise post-sequential narrative from saved experiment outputs.
- `scripts/32_thermal_final_architecture_comparison.py`: consolidate the retained architectures under a common frozen protocol and separate covariance-scale effects from architecture effects.
- `scripts/33_thermal_uncertainty_origin_oracle.py`: diagnose previous-versus-current hot-tail underdispersion, compare error with posterior SD by temperature percentile, and run the controlled true-previous-state oracle benchmark.
- `scripts/34_thermal_uncertainty_oracle_v2.py`: canonical three-stage previous-posterior/current-forecast/current-posterior audit with hybrid versus coherent latent-state inference and strict versus development-recalibrated oracle controls.
