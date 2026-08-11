# Joint stationary versus sequential GP audit

This experiment rechecks the previously strong advective, stochastically forced space-time GP under corrected CRPS and a shared current observation realization.

The stationary covariance is the covariance of

```text
dr = (alpha Laplacian r - v(t) dot grad r - beta r) dt + dW_Q(t),
```

where W_Q is white in time and has an RBF spatial covariance. Its Gauss-Laguerre integral is normalized so the equal-time marginal variance is sigma_f^2.

Legacy physics mean uses the latent synthetic field clipped at the camera ceiling. Observed-clipped mean uses a noisy clipped predecessor frame, and posterior mean propagates the censored posterior expectation of that predecessor. Joint ST models condition directly on current and 0.01 s earlier observations. The sequential model uses only current observations after propagating the immediately previous posterior.

All CRPS values use the unbiased M(M-1) empirical estimator.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy physics mean + RBF | 0.435 | 0.727 | 2.700 | 7.835 | 0.936 |
| legacy physics mean + stochastic ST | 0.461 | 0.591 | 2.560 | 7.743 | 0.924 |
| legacy physics mean + advective stochastic ST | 0.446 | 0.566 | 2.545 | 7.695 | 0.922 |
| observed-clipped mean + advective stochastic ST | 0.474 | 0.572 | 2.567 | 7.724 | 0.921 |
| posterior mean + RBF | 0.430 | 0.729 | 2.253 | 7.555 | 0.949 |
| posterior mean + stochastic ST | 0.473 | 0.596 | 2.201 | 7.493 | 0.932 |
| posterior mean + advective stochastic ST | 0.448 | 0.570 | 2.210 | 7.496 | 0.930 |
| posterior mean + sequential SPDE | 0.438 | 0.851 | 2.237 | 7.556 | 0.965 |

## Interpretation

The old probabilistic result is reproduced with corrected CRPS. With the legacy mean, the advective stationary covariance changes all-domain CRPS from `0.727` to `0.566` K. Its covariance, normalization, transport sign, and numerical quadrature all pass the checks below.

The legacy mean has a mild oracle feature: it propagates the latent noiseless synthetic predecessor after clipping. Replacing that input by the actually observed noisy-clipped frame changes all-domain CRPS by only `+0.006` K, so this does not explain the large stationary-covariance gain.

Posterior-mean propagation and the stationary covariance are complementary. Relative to posterior mean + RBF, the advective stationary model changes all-domain CRPS from `0.729` to `0.570` K and fixed top-1% CRPS from `2.253` to `2.210` K. The RBF retains the smaller mean field error and nearly nominal hot coverage.

The finite-step sequential SPDE should not replace the stationary joint model in its current form: its all-domain CRPS is `0.851` K because its process-noise scale was selected for hot-tail coverage and is too diffuse over the full domain.

Paired trajectory-level changes are recorded in `paired_comparisons.csv`.

All 99 covariance checks passed. Minimum tested eigenvalue: `4.581e+00`; maximum stationary marginal-normalization error: `1.776e-15`; maximum 24-versus-48 node quadrature difference: `7.309e-04` of marginal variance.

alpha = 3.610913e-06 m^2/s; beta = 12.422902 1/s; gamma = 0.170223.
