# Controlled advection and forcing restart

This restart separates Adrienne's three suggestions into three comparisons. All seven unique models use the same 3% synthetic ceiling, fixed stride-only observation mask, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. Results contain all 33 trajectories; the main table below uses the 30 trajectories not used for physical-parameter and source-coupling calibration.

No source-local covariance is included. The final stage adds only the source-amplitude correction described below.

## Overall held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + RBF | 0.435 | 0.736 | 2.710 | 7.835 | 0.936 |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| physics mean + advective space-time | 0.441 | 0.718 | 2.688 | 7.799 | 0.933 |
| advective physics mean + space-time | 0.446 | 0.732 | 2.714 | 7.828 | 0.932 |
| physics mean + forced space-time | 0.461 | 0.598 | 2.568 | 7.743 | 0.924 |
| physics mean + advective forced space-time | 0.446 | 0.573 | 2.553 | 7.695 | 0.922 |
| physics mean + advective forced + source amplitude | 0.446 | 0.572 | 2.493 | 7.702 | 0.935 |

## Interpretation

1. `physics mean + RBF` retains the smallest mean field error. The regular space-time kernel improves all-domain CRPS but not mean field error.
2. Adding the trajectory displacement to the space-time kernel improves all-domain CRPS on all 30 held-out trajectories. Its mean field and hottest-region changes are small, so this supports a covariance improvement rather than a uniformly better reconstruction mean.
3. Adding the same moving-frame correction to the one-step mean is neutral to slightly worse. Transporting the whole previously deposited field is therefore not supported by this experiment.
4. The exact white-in-time, spatial-RBF forcing covariance substantially improves CRPS and slightly improves peak error, but worsens field error and reduces hot-region coverage. It is promising as a probabilistic prior, not yet as the best posterior-mean reconstruction.
5. The advective and forced covariance changes are complementary: relative to forcing alone, the combined model changes all-domain CRPS by `-0.0254 K` and wins on `30/30` held-out trajectories, while changing field error by `-0.0145`. It is the strongest probabilistic model in this focused comparison, although RBF still has the lowest mean field error and the combined model has lower hot-region coverage.
6. Adding source-amplitude uncertainty to the combined model changes all-domain CRPS by `-0.0007 K` and wins on `29/30` held-out trajectories, while changing field error by `+0.0000`. It primarily widens uncertainty at the active source: the sparse mask did not observe the narrow source basis, so its temporal persistence was not identifiable and its posterior mean was essentially unchanged.

## Mathematical checks

All `165` kernel checks are symmetric and positive semidefinite. The minimum tested eigenvalue is `4.581e+00`, and the maximum marginal-variance error is `1.776e-15`. The 24-node forced-kernel quadrature differs from a 48-node reference by at most `7.309e-04` of the marginal variance.

Folders:

- `13_thermal_advection_setup`: shared configuration, calibration, checkpoints, complete results, and kernel validation.
- `14_residual_advection`: RBF, regular space-time, and advective space-time.
- `15_mean_advection`: regular and advective one-step means with the same regular space-time kernel.
- `16_stochastic_forcing`: regular space-time and the stationary forced-SPDE covariance.
- `17_advective_stochastic_forcing`: RBF, regular space-time, advective, forced, and combined advective-forced residual covariances.
- `18_source_amplitude_correction`: the selected source-amplitude term added to the combined advective-forced covariance.

Shared files include `all_models_results.csv`, `fixed_configuration.csv`, source-coupling calibrations, and `kernel_validation.csv`.
