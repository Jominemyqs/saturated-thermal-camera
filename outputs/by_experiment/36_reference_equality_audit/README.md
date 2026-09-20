# Reference-ceiling equality audit

## Verdict

The lowered-ceiling reference row does **not** reproduce the previously quoted Experiment-30 row because the labels refer to different mean architectures and the samplers are materially different. The pseudo-camera itself is identical at the reference ceiling, and the finite-step innovation covariance `Q` is unchanged.

- Camera arrays and masks are bit-for-bit identical across the two reference constructions: `True`.
- The direct Experiment-30 driver and the audit's reconstructed short-chain blocks produce identical posterior means, SDs, intervals, and draws on the direct replay trajectory: `True`.
- The largest trajectory-level metric discrepancy when replaying the stored Experiment-30 and Experiment-35 rows is `1.139e-13`.
- Experiment 35 uses the **advective** posterior physics mean. The commonly quoted Experiment-30 sequential row uses the **ordinary diffusive** posterior physics mean.
- Experiment 35 uses four long chains for both previous and current inference; Experiment 30 used one short chain for each.
- The Experiment-35 single-frame RBF is not posterior physics mean + RBF. It is a current-only ambient-mean RBF with a different current observation set.

## Held-out 30 decomposition

| Stage | Field error | Overall CRPS (K) | Top-1% CRPS (K) | Top-1% coverage | Top-1% width (K) |
|---|---:|---:|---:|---:|---:|
| E30 ordinary mean; short previous/current chains | 0.407 | 0.210 | 2.749 | 0.388 | 3.33 |
| E30 advective mean; short previous/current chains | 0.404 | 0.206 | 2.843 | 0.358 | 3.32 |
| advective mean; converged previous, short current | 0.445 | 0.222 | 1.734 | 0.806 | 7.07 |
| advective mean; converged previous/current, E30 top-1% mask | 0.444 | 0.222 | 1.736 | 0.809 | 7.19 |
| E35 reference; converged previous/current, rank top-1% mask | 0.444 | 0.222 | 1.764 | 0.801 | 7.20 |

The four transitions isolate different causes:

1. Ordinary to advective mean changes only the deterministic mean architecture.
2. Short to converged previous inference changes the previous censored posterior, its propagated covariance, and the physics mean derived from it; the current sampler remains short.
3. Short to converged current inference changes only current posterior sampling.
4. Quantile to rank top-1% changes only one evaluation pixel per 2501-pixel trajectory; whole-field metrics are identical.

Numerically, changing only the deterministic mean from ordinary to advective has a small effect: field error changes from `0.407` to `0.404`, while top-1% CRPS changes from `2.749` to `2.843` K. Replacing the short previous posterior with the converged one is the dominant change: top-1% CRPS changes from `2.843` to `1.734` K, coverage from `0.358` to `0.806`, and width from `3.32` to `7.07` K. The same change worsens field error from `0.404` to `0.445` and overall CRPS from `0.206` to `0.222` K. Converging only the current sampler then changes top-1% CRPS by `0.002` K and width by `0.13` K.

## Intermediate objects

Across held-out trajectories, the converged and short previous posterior means differ by `2.519` K RMSE on previous saturated pixels. Mean saturated-pixel variance changes from `0.417` to `3.811` K^2.

After one-step advective propagation, the mean diagonal contribution from `B Sigma B^T` changes from `0.011` to `0.094` K^2. In contrast, the finite-step innovation SD is `0.715932` versus `0.715932` K and is numerically identical because its inputs and implementation are frozen.

The observation realization, threshold, censoring masks, source displacement, physical parameters, current likelihood, and covariance formula are not sources of the discrepancy.

## Consequence for the ceiling sweep

The ceiling-response curve remains a valid internally controlled experiment because every level within Experiment 35 uses one frozen architecture and one converged sampling protocol. Its reference row should be described as an updated reference for that architecture, not as a reproduction of the older Experiment-30 table.

The older Experiment-30 probabilistic values should not be mixed numerically with the ceiling-sweep values in one table. Before paper use, the architecture-wide probabilistic comparison should be rerun with the converged protocol. Existing within-experiment mean/kernel ablations remain useful as controlled historical ablations, but their short-chain CRPS, coverage, and width should not be treated as final estimates.

## Files

- `trajectory_stage_metrics.csv`: every held-out trajectory at every decomposition stage.
- `heldout30_stage_summary.csv`: table above with standard errors.
- `stage_change_decomposition.csv`: paired metric changes attributed to one factor at a time.
- `intermediate_object_audit.csv`: previous posterior, physics mean, `B Sigma B^T`, `Q`, prior, and posterior comparisons.
- `camera_identity_checks.csv`: array-level identity checks.
- `stored_row_reproduction_checks.csv`: exact comparisons against stored Experiment-30 and Experiment-35 metrics.
- `direct_driver_replay_checks.csv`: direct posterior-array equality against the Experiment-30 driver.
- `architecture_identity.csv`: definitions explaining which similarly named rows are and are not comparable.
