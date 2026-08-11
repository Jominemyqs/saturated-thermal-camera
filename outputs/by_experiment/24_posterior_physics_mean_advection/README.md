# Posterior-physics-mean advection experiment

This is a controlled mean-only ablation. Both models use the same previous-frame censored posterior, current observations, RBF covariance, censored likelihood, hyperparameters, sampler settings, and random seeds. The only change is whether the diffused previous posterior mean is translated by the source-centroid displacement before adding the current heat source.

```text
ordinary:  m_n - T_amb = exp(-beta dt) G_(alpha dt) * (mu_(n-1)-T_amb) + gamma q_n dt
advective: m_n - T_amb = exp(-beta dt) [G_(alpha dt) * (mu_(n-1)-T_amb)](x-d_n) + gamma q_n dt
```

Positive displacement translates the previous thermal field in the positive coordinate direction: the implementation returns `f(x-d_n)`. The current source term is not translated.

## 1. Inequality / mean result

The retained RBF-residual ablation compares observed-clipped physics mean, posterior physics mean, and full-posterior propagation. Posterior physics propagation improves the held-out field error and hidden-tail CRPS without reusing the previous frame in the current likelihood.

| Retained model | Field error | All CRPS | Top-1% CRPS | Hot coverage |
| --- | ---: | ---: | ---: | ---: |
| observed-clipped physics mean + RBF | 0.461 | 0.731 | 2.701 | 0.937 |
| posterior physics mean + RBF | 0.424 | 0.728 | 2.243 | 0.950 |
| full-posterior propagation + RBF | 0.424 | 0.728 | 2.234 | 0.949 |

## 2. Kernel result

The retained stochastic heat-process covariance is a separate supporting kernel ablation. With its clipped physics mean fixed, it improves probabilistic scoring but is not required by the main posterior-physics-mean architecture.

At 3% censoring, RBF gives all-domain CRPS `0.727 K` and stochastic space-time gives `0.591 K`; their field errors are `0.435` and `0.461` respectively.

## 3. Advection result

| Model | Field error | All CRPS | Top-1% CRPS | Peak error | Hot coverage | Hot width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior physics mean + RBF | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 | 12.231 |
| advective posterior physics mean + RBF | 0.419 | 0.727 | 2.281 | 7.678 | 0.946 | 12.231 |

Across 30 held-out trajectories, advection changes field error by `-0.0055` and wins on `24/30`; changes all-domain CRPS by `-0.0009 K` and wins on `25/30`; changes top-1% CRPS by `+0.0379 K` and wins on `6/30`.

Family-specific and median paired changes are in `paired_comparisons.csv`. The three reconstruction figures expose the deterministic mean difference directly.

## Controlled design

- `Y_(n-1)` is used only to infer the previous censored posterior.
- The final likelihood contains only the shared current observation set `Y_n`.
- Both rows use the same ambient-mean previous RBF posterior and the same current RBF covariance.
- HeatFluxZ supplies the simulated source centroid and current source term. Deployment assumes commanded laser path and power are available.
- All CRPS values use the unbiased `M(M-1)` estimator.

alpha = `3.610913e-06 m^2/s`; beta = `12.422902 1/s`; gamma = `0.170223`.
