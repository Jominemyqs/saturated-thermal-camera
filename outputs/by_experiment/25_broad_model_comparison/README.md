# Broad model comparison

This summary places the retained mean and covariance experiments in one table. All rows use 3% censoring, the same 30 held-out trajectories, development-only physical calibration, seed 41, and unbiased M(M-1) CRPS.

The table is descriptive rather than a single-factor ablation. Current-only RBF models update on Y_n after constructing their mean. Stochastic space-time rows use a joint two-frame residual GP and the legacy noise-free simulation field clipped at c. They therefore use a different information architecture and should not be declared universally superior from raw cross-row scores alone.

## Held-out results

| Method | Field | All CRPS | Top-1% CRPS | Peak error | Hot coverage | Hot width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ambient mean + RBF | 0.908 | 0.780 | 5.747 | 28.394 | 0.896 | 12.059 |
| ambient mean + stochastic ST | 0.819 | 0.619 | 5.138 | 27.826 | 0.880 | 9.941 |
| legacy latent-clipped physics mean + RBF | 0.435 | 0.727 | 2.700 | 7.835 | 0.936 | 12.144 |
| legacy latent-clipped physics mean + stochastic ST | 0.461 | 0.591 | 2.560 | 7.743 | 0.924 | 10.039 |
| legacy latent-clipped physics mean + advective stochastic ST | 0.446 | 0.566 | 2.545 | 7.695 | 0.922 | 9.483 |
| observed-clipped physics mean + RBF | 0.461 | 0.731 | 2.701 | 7.833 | 0.937 | 12.167 |
| posterior physics mean + RBF | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 | 12.231 |
| advective posterior physics mean + RBF | 0.419 | 0.727 | 2.281 | 7.678 | 0.946 | 12.231 |
| full-posterior propagation + RBF | 0.424 | 0.728 | 2.234 | 7.611 | 0.949 | 12.444 |

## Main comparisons

- Replacing the observed-clipped physics mean by the posterior physics mean changes field error by -0.0363 and top-1% CRPS by -0.4580 K.
- Adding advection to the posterior physics mean changes field error by -0.0055 and top-1% CRPS by +0.0379 K.
- In the matched legacy mean ablation, stochastic ST versus RBF changes all-domain CRPS by -0.1359 K and field error by +0.0253.
- In the matched stochastic-ST ablation, residual advection changes all-domain CRPS by -0.0251 K and field error by -0.0145.

## Interpretation

The posterior physics mean remains the clean primary architecture for the camera problem. It improves the hidden hot tail while preserving a strictly causal current-only update. Mean advection offers a small global-field gain but slightly worsens top-1% CRPS. The stochastic space-time covariance gives the lowest probabilistic scores in its legacy two-frame setup, and residual advection improves it further, but that result should motivate a later clean camera-realistic temporal model rather than replace the current primary model.

Files:

- results.csv: all trajectory-level rows and architecture labels.
- heldout30_overall.csv: overall held-out summaries.
- primary_comparison.csv: the five rows requested for the broad comparison.
- family_summary.csv and paired_comparisons.csv: family and paired results.
- configuration_audit.csv: cross-experiment compatibility checks.
- comparison.png: complete nine-row overview.
- primary_comparison.png: zoomed view of the five requested models.
