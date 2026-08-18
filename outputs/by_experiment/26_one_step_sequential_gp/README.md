# Architecture-aware one-step sequential GP comparison

This comparison restores the canonical ambient-mean RBF censored posterior for the previous frame. Every newly fitted row uses the same posterior draws, camera realization, physical parameters, and current-only censored likelihood.

For the sequential rows, the stochastic heat GP supplies the transition B and conditional innovation C. Previous RBF-posterior uncertainty Sigma_- is then propagated by moment matching:

```text
T_n | Y_(n-1) approximately N(m_n^physics, C + B Sigma_- B^T).
```

## Results

| Method | Field | All CRPS | Top-1% CRPS | Peak error | Coverage | Hot width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| posterior physics mean + RBF | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 | 12.231 |
| advective posterior physics mean + RBF | 0.419 | 0.727 | 2.281 | 7.678 | 0.946 | 12.231 |
| legacy latent-clipped physics mean + joint advective stochastic ST | 0.446 | 0.566 | 2.545 | 7.695 | 0.922 | 9.483 |
| posterior physics mean + sequential stochastic ST | 0.407 | 0.210 | 2.738 | 7.598 | 0.790 | 3.354 |
| posterior physics mean + sequential advective stochastic ST | 0.407 | 0.210 | 2.749 | 7.639 | 0.771 | 3.174 |
| advective posterior physics mean + sequential advective stochastic ST | 0.404 | 0.206 | 2.843 | 7.663 | 0.754 | 3.178 |

## Controlled design

- Synthetic censoring: 3%.
- One shared previous censored posterior and current camera realization.
- Identical physical parameters, current likelihood, sampler settings, and seeds.
- Previous observations are not supplied to the current likelihood.
- All CRPS uses the unbiased M(M-1) estimator.
- The forcing lengthscale is inherited from the frozen residual setup; this run does not add a new hyperparameter search.

## Reconstruction scales

- `reconstruction_*.png` uses one shared linear scale spanning the full true-field temperature range.
- `reconstruction_*_tail_focus.png` uses one shared linear scale from ambient temperature to the synthetic camera ceiling. Values above the ceiling are clipped only in the visualization, not in inference or scoring.
- White contours in the tail-focused figures mark 0.5, 1, and 2 K above ambient, making the cooling trail easier to compare across models.

## Architecture notes following the result table

- The first two rows are controlled current-only RBF models. They differ only in whether the posterior physics mean is translated before diffusion.
- The legacy latent-clipped row is a historical kernel reference. It uses a noise-free simulation predecessor and a joint two-frame likelihood, so its raw scores are descriptive and not a paired comparison with the sequential rows.
- The two main sequential rows use the ordinary posterior physics mean. They differ only by advection in the stochastic ST transition, and the final update conditions only on Y_n.
- The final row is an optional diagnostic that advects both the deterministic mean and the residual transition. It is not the primary integrated candidate.

## Numerical interpretation

Relative to posterior physics mean + RBF, the main sequential advective-ST candidate changes field error by -0.0168, all-domain CRPS by -0.5181 K, top-1% CRPS by +0.5063 K, and coverage by -0.1789.
Adding residual advection to the sequential stochastic-ST transition changes all-domain CRPS by +0.0003 K under the same posterior physics mean.

Exact architecture labels are recorded in architecture.csv, and controlled paired changes are recorded in paired_comparisons.csv.
