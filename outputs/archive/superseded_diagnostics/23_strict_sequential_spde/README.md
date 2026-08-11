# Strict sequential stochastic SPDE experiment

This replaces the earlier finite-step process-noise experiment. Every model uses the strict filtering sequence:

```text
Y_(n-1) -> p(T_(n-1) | Y_(n-1)) -> p(T_n | Y_(n-1)) -> update with Y_n only.
```

The previous frame is never supplied again in the final GP update. The previous posterior is ambient-mean + isotropic RBF, using only the noisy observed previous frame. The baseline uses the propagated posterior mean plus an independent current-frame RBF residual. The two SPDE models use `A Sigma A^T + Q_dt`; they do not add another current RBF covariance.

## Development-only Q selection

The forcing lengthscale was selected from `[0.5, 1.0, 2.0, 4.0]` times the source width, followed by amplitude selection from `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]`. The frozen choice is `ell_W = 0.5 ell_source` and amplitude multiplier `2`.

Both SPDE variants share this Q. Selection minimizes `0.5 * all-domain CRPS + 0.5 * fixed top-1% CRPS`, averaged over the three development trajectories and both transport variants. Coverage is reported but is not used as a hard selection constraint.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage | Hot width (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior mean + RBF | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 | 12.231 |
| sequential diffusive SPDE | 0.405 | 0.426 | 2.471 | 7.618 | 0.898 | 6.936 |
| sequential advective SPDE | 0.403 | 0.425 | 2.535 | 7.690 | 0.891 | 6.861 |

## Why the previous finite-step result differed

The earlier experiment selected a 32x forcing intensity by first requiring hot coverage >= 0.90 and then minimizing top-1% CRPS. That produced about 16.5 K^2 finite-step forcing variance and raised held-out all-domain CRPS to about 0.850 K. The present experiment changes the prior selection, not the sequential filtering algebra.

The trajectory-level comparisons are in `paired_comparisons.csv`. All covariance checks in `prior_validation.csv` are symmetric and positive semidefinite.

alpha = 3.610913e-06 m^2/s; beta = 12.422902 1/s; gamma = 0.170223.
