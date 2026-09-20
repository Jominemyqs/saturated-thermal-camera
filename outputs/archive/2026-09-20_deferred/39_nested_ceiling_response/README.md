# Nested pseudo-ceiling response diagnostic

## Question

Does the way a posterior changes across nested artificial ceilings predict hidden exceedance better than the posterior at one ceiling?

## Protocol

- Reuses the converged Experiment 38 development runs; no GP was refit for this postprocessing diagnostic.
- Matches pixels censored at 5%, 7.5%, and 10% pseudo-camera ceilings.
- Uses 151 pixels from the three development trajectories.
- Primary target is the still-known reference-camera measurement minus the 5% ceiling. Latent truth is evaluation only.
- Comparison uses outer leave-one-trajectory-out validation with ridge strength selected inside each training fold.

## Result

The ordinary posterior exceedance has Spearman rho `0.460` with known measurement exceedance. The strongest nested response feature, dmu/dc, has rho `0.409`.

| Target | Features | RMSE (K) | MAE (K) | Spearman rho | R-squared |
|---|---|---:|---:|---:|---:|
| known measurement | single-threshold features | 0.308 | 0.202 | 0.343 | -0.467 |
| known measurement | single + nested-response features | 0.309 | 0.204 | 0.328 | -0.476 |
| latent truth (evaluation only) | single-threshold features | 0.330 | 0.183 | 0.796 | 0.096 |
| latent truth (evaluation only) | single + nested-response features | 0.330 | 0.185 | 0.789 | 0.094 |

The predeclared signal gate is `FAIL`: measurement RMSE reduction is `-0.3%` and Spearman gain is `-0.014`.

## Decision

The nested response contains some information, but it is not stronger than the current single-threshold posterior and does not improve held-out-trajectory prediction. Therefore the learned calibration and hard extrapolation stages were not run. With the present model and three ceiling levels, pseudo-censoring response should not yet be claimed to recover genuinely unobservable exceedance magnitude.
