# Experiment 39b: smooth nested pseudo-ceiling response

## Question

Does a smooth posterior response across many nested artificial ceilings predict the part of a known pseudo-hidden measurement that the single-threshold posterior misses?

## Controlled protocol

- Reuses the audited Experiment 38 posterior-physics-mean plus sequential stationary stochastic-heat GP, observation protocol, HMC convergence checks, and common reference noise realization.
- Uses ten nested ceilings from the 3% reference through 40% censoring and five anchor ceilings from 5% through 15%.
- Fits quadratic response curves using at least five ceiling levels. Features summarize anchor slope, curvature, total response, and integrated response for posterior mean and SD.
- The learned target is the known residual `reference measurement - single-threshold posterior mean`. Latent truth is not used for fitting or feature construction.
- Development comparison uses leave-one-trajectory-out validation. Multiple anchors from a pixel always stay in the same trajectory fold.
- The five anchors produce `2633` known pseudo-hidden examples spanning `0.000` to `2.628 K`.
- Moderate training examples are defined using the development-only 70th percentile, `1.115 K`; larger exceedances are reserved for extrapolation.
- A real-case equality audit against the original Experiment 38 driver gave maximum absolute difference `0.0` for posterior draws, summaries, intervals, and features at the 10% ceiling; see `implementation_consistency_audit.csv`.

## Development result

| Evaluation split | Model | RMSE (K) | MAE (K) | Spearman rho | Residual rho |
|---|---|---:|---:|---:|---:|
| moderate pseudo-exceedance | uncorrected single-threshold posterior | 3.474 | 3.136 | 0.409 | nan |
| moderate pseudo-exceedance | single-threshold residual correction | 0.248 | 0.203 | 0.646 | 0.918 |
| moderate pseudo-exceedance | single + smooth-response residual correction | 0.253 | 0.206 | 0.630 | 0.916 |
| larger unseen pseudo-exceedance | uncorrected single-threshold posterior | 3.000 | 2.967 | 0.099 | nan |
| larger unseen pseudo-exceedance | single-threshold residual correction | 0.979 | 0.915 | 0.181 | 0.445 |
| larger unseen pseudo-exceedance | single + smooth-response residual correction | 0.966 | 0.902 | 0.174 | 0.446 |

On the larger unseen pseudo-exceedances, the unchanged signal gate is `FAIL`: smooth-response versus single-feature RMSE change is `1.3%` and Spearman change is `-0.007`.
The response model has lower RMSE on `3/3` development trajectories, but higher Spearman correlation on only `1/3`; the per-trajectory effect sizes are in `development_trajectory_comparison.csv`.

## Decision

The stronger development diagnostic still finds no material incremental out-of-trajectory signal from the nested response. The held-out trajectories and genuine hardware-saturated tail were therefore not evaluated. This closes the nested-response branch under the predeclared one-modification rule; lowering the ceiling remains useful as a diagnostic and calibration-data generator, but its response curve should not be claimed to reveal hidden magnitude beyond the single-threshold posterior.
