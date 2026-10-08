# Native B/C implementation: reviewer guide

Inspection date: 2026-10-08. This package documents executed outputs **49, 51,
52 and 53**, not a new experiment. Read [verification](verification.md) for
provenance gaps; [parameter inventory](parameter_inventory.csv) contains the
detailed values and fitting records. Paths and line numbers refer to this
inspected working tree, which includes uncommitted native work.

## What is learned and what is fixed?

**No forecast coefficient or GP hyperparameter is optimized against the current
native reconstruction.** These runs inherit the following settings:

| Quantity | Executed value | Origin |
| --- | ---: | --- |
| Diffusivity alpha | 3.6109129986233175e-6 m2/s | Median of three development cooling fits |
| Cooling beta | 12.422901570529978 /s | Median of the same fits |
| Gradient coupling gamma | 0.1702227206133784 | Median of three nonnegative least-squares calibrations |
| `signal_sd` | 3.4305133895116966 K | Median development 95th-percentile cooling excess; not GP marginal-likelihood fitting |
| RBF length | 0.3125 mm in both native cases | 1.25 times prepared source length, whose 0.25 mm grid-spacing floor binds |
| Forcing length | Same 0.3125 mm | Prescribed multiplier 1 of the RBF length |
| Camera / previous likelihood / current likelihood SD | 0.25 / 0.25 / 0.50 K | Prescribed camera noise and inherited likelihood settings |

The development paths were DiagonalScanPath_8, HorizontalScanPath_13 and
SpiralScanPath_13. Alpha/beta fit cooling rates from projected simulator histories.
Gamma fits current simulator excess after propagating **noise-free clipped
simulator predecessors**, at a ceiling set from the final grid truth's 97th
percentile. It was not calibrated using posterior-reconstructed predecessors,
native inputs, or camera-only self-supervision. See the surviving
[development estimates](../../outputs/by_experiment/24_posterior_physics_mean_advection/development_physics_parameters.csv)
and [gamma fits](../../outputs/by_experiment/24_posterior_physics_mean_advection/source_coupling_calibration.csv).

The historical noise/length sensitivity search and its fallback survive in code,
but its original selected-row/search-result files are absent locally. Later
saved configurations establish the values used, not proof of their optimality.
There is no independent current fit of forcing amplitude or length.

Preparation still computes per-trajectory cooling fits, but native drivers
discard those fitted alpha/beta/signal estimates. They **do** use ambient
temperature and source width derived from full projected simulation histories.
The inherited reference ceiling is also simulation-derived. Thus the whole
pipeline is not deployment-only, even though observations enter the final
likelihood without hidden temperature values. [Code map](code_map.md) separates
preparation, historical calibration and current inference.

## How information and uncertainty flow

1. Generate previous/current noisy cameras at native nodes. Infer the previous
   censored values with an ambient-mean spatial RBF. The representation is
   **hybrid**: all unsaturated noisy predecessor values are pinned, even outside
   the sparse numerical likelihood set. Censored latent marginal draws replace
   the saturated entries. Their empirical mean is the previous-state estimate.
2. Construct the common deterministic physics mean using native-triangle
   interpolation to a 601 x 401 computational grid, positive-excess flooring,
   Gaussian diffusion, cooling, zero extension, and bilinear return to native
   nodes. Add the unchanged thresholded gradient increment. This is not an
   interpolation-free native 3D FEM forecast.
3. **B:** use spatial RBF residual covariance with marginal variance
   `signal_sd**2` (about 11.7684 K2). No predecessor covariance is added.
4. **C:** use `K_prior = Q_dt + H H^T`, where column s of H is
   `A(draw_s - previous_mean) / sqrt(S-1)`. A is the same native interpolation,
   diffusion/cooling and evaluation map, **without the positive floor or source**.
   This empirical inherited covariance is an approximation based on hybrid
   marginal draws, not coherent full-field posterior samples. It is not the
   historical GP block-conditional construction.
5. Apply **only the current-frame** numerical/censoring likelihood. Previous
   observations are not supplied again. Return latent-temperature marginal
   means, empirical SDs and pointwise quantile intervals.

For C, the normalized stationary heat covariance `K_stat` has zero-lag marginal
variance `signal_sd**2`; the innovation is
`Q_dt(x,x') = K_stat(x,x';0) - K_stat(x,x';2*dt)`.
It represents uncertainty introduced over one step, not measurement noise.
Its scale depends jointly on the inherited signal scale, alpha, beta, forcing
length, timestep and quadrature. **`signal_sd` is not the SD of the forcing or
of Q_dt.** Output 49's native area-averaged total C prior variance is about
0.608/0.603 K2 (diagonal/spiral), including inherited variance 0.095/0.091 K2.
The innovation is free-space heat-based; the finite-grid propagation boundary
is not used to rederive it.

Camera SD controls synthetic noise before clipping. Likelihood SD enters as
noise variance in the observation covariance; censoring concerns the noisy
signal, not a hard bound on latent temperature. Numerical jitter is a separate
stabilizer, not estimated noise. Posterior SD is an output, not a tuned parameter.
The current 0.50 K likelihood SD is larger than the 0.25 K generated camera SD.

`HeatFluxZ` means the supplied projected vertical temperature gradient
`project(T.dx(2), W_grad_z)`, not prescribed heat flux. Gamma would have units
m/s under metre/second/K conventions. Exact dataset/export provenance and
deployment availability remain unresolved; see [physical provenance](../../outputs/by_experiment/52_native_higher_ceiling/PHYSICS_PROVENANCE.md).

## What does initialization mean here?

**Historical fitting:** alpha/beta use a constrained two-variable least-squares
candidate enumeration (interior, axes, zero), not an iterative optimizer.
Gamma has a closed-form nonnegative least-squares solution. Neither needs a
starting guess or restarts. Signal scale/source width are quantile/width rules;
noise and length multipliers are inherited choices, not current optimization.

**Current latent sampling:** four reflected-HMC chains, 100 burn-in, 400 retained
per chain, thinning 1. The noisy censored block starts at
`max(conditional_mean, ceiling + 0.5*conditional_SD)` and is whitened; seeds
vary momenta, not a bank of optimized starting points. A failed classical
R-hat <= 1.05 gate triggers one fresh 300-burn-in/800-retained retry. None of
the inspected runs retried. Previous propagation uses all 1600 draws; current
summaries use 300 balanced draws per chain (1200 total).

The gate checks previous censored targets, and current all-domain plus censored
targets. It is not a rank-normalized/split R-hat or ESS guarantee. Initialization
sensitivity was not established by these native runs; convergence is not proof
of uncertainty calibration, unbiased MCMC scoring, or optimal hyperparameters.

## Where to start reading

- [Driver 53, `main`, lines 65-229](../../scripts/53_thermal_matched_projection_gp.py#L65): output 49's native/projected B/C construction.
- [Previous hybrid estimate, lines 247-301](../../src/thermal_posterior_physics.py#L247), then [matched operator, lines 47-97](../../src/matched_projection.py#L47).
- [Finite-step innovation, lines 123-150](../../src/stochastic_heat_gp.py#L123), then [current wrapper, lines 43-103](../../src/pseudo_censoring_runner.py#L43) and [censored blocks, lines 200-349](../../src/dense_censored_gp.py#L200).
- [Workspace index](workspace_index.md): output 51 is attribution only; 52 changes ceilings; **53 retains extra current observations for C only**. It does not replace historical B/C observation policies.

No non-saturation-mask likelihood, within-step integration, refit or new model
was implemented during this review.
