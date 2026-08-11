# Stationary stochastic heat-SPDE ablation

This experiment replaces the earlier deterministic propagation covariance with the stationary covariance of

```text
dr(t) = (alpha Laplacian r(t) - beta r(t)) dt + dW_Q(t),
Q(x,x') = sigma_W^2 exp(-||x-x'||^2/(2 ell_W^2)).
```

For the stationary process,

```text
K(t,t') = integral_0^infinity exp(L(|t-t'|+u)) Q exp(L* u) du.
```

The positive covariance integral is evaluated by 24-node Gauss-Laguerre quadrature and normalized to the same marginal variance as the RBF residual. The absolute time-lag dependence is therefore justified by stationarity, not by the deterministic forward equation alone.
The advective variant evaluates the same stationary covariance in source-centroid coordinates, using the HeatFluxZ trajectory without changing the clipped physics mean.

All CRPS values use the unbiased empirical estimator with denominator `M(M-1)` in the pairwise-dispersion term.

## Controlled setup

- All 33 trajectories are fit; the table below reports the 30 held-out paths.
- Synthetic censoring fraction: 3%.
- Current and immediately previous frames: lags [0.0, 0.01] s.
- Observation stride: 5; measurement noise SD: 0.25 K.
- Regular mean means a constant ambient-temperature prior mean.
- The physics mean is the existing clipped one-step forecast. Posterior propagation of the previous censored frame is intentionally deferred so this experiment isolates the new stochastic covariance and CRPS correction.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| regular mean + RBF | 0.908 | 0.780 | 5.747 | 28.394 | 0.896 |
| regular mean + stochastic space-time | 0.819 | 0.619 | 5.138 | 27.826 | 0.880 |
| physics mean + RBF | 0.435 | 0.727 | 2.700 | 7.835 | 0.936 |
| physics mean + stochastic space-time | 0.461 | 0.591 | 2.560 | 7.743 | 0.924 |
| physics mean + advective stochastic space-time | 0.446 | 0.566 | 2.545 | 7.695 | 0.922 |

## Direct comparisons

With the physics mean fixed, replacing RBF by the stochastic space-time covariance changes field error by `+0.0253`, all-domain CRPS by `-0.1359 K`, and top-1% CRPS by `-0.1405 K`. It wins on all-domain CRPS for `30/30` and on fixed top-1% CRPS for `22/30` held-out paths. Its field error is lower on only `1/30` paths, and mean hot-region coverage changes by `-0.0123`.

With the stochastic covariance fixed, adding the one-step physics mean changes field error by `-0.3585`, all-domain CRPS by `-0.0280 K`, and top-1% CRPS by `-2.5784 K`.

With the clipped physics mean fixed, adding source-centroid advection to the stochastic covariance changes field error by `-0.0145`, all-domain CRPS by `-0.0251 K`, and top-1% CRPS by `-0.0149 K`.

The stationary SPDE covariance is therefore the strongest distributional model family in this five-way comparison, whereas physics mean + RBF retains the best posterior-mean field error and slightly better hot-region coverage. This separates two claims that should not be conflated: temporal stochastic covariance improves predictive scoring, while the one-step physics mean provides most of the reconstruction accuracy.

## Numerical checks

All 66 trajectory-specific stochastic covariance checks are symmetric and positive semidefinite within numerical tolerance. The minimum tested eigenvalue is `4.581e+00`. The largest diagonal normalization error is `1.776e-15`, and the largest 24-node versus 48-node quadrature difference is `7.309e-04` of the marginal variance.

Files:

- `results.csv`: all trajectory-level fits and metrics.
- `heldout30_overall.csv`: aggregate held-out results.
- `all33_overall.csv`: aggregate results including development paths.
- `family_summary.csv`: results separated by trajectory family.
- `paired_comparisons.csv`: within-trajectory model differences and wins.
- `kernel_validation.csv`: covariance and quadrature checks.
- `comparison.png` and `comparison_by_family.png`: result plots.

Fixed physical parameters:

- alpha = 3.610913e-06 m^2/s
- beta = 12.422902 1/s
- source coupling gamma = 0.170223
