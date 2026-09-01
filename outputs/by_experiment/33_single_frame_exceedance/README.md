# Single-frame censored-posterior exceedance diagnostic

This experiment asks whether the censored RBF GP recognizes that a pixel is above the camera ceiling but fails to infer how far above it lies. No model, hyperparameter, camera realization, held-out split, or sampler setting was changed from the canonical uncertainty audit.

For each observed-censored previous-frame pixel,

- `delta_true_K = T_true - c`;
- `delta_hat_K = E[T | Y] - c`;
- `posterior_sd_K` is the sample posterior standard deviation;
- `abs_error_over_sd = |E[T | Y] - T_true| / posterior_sd`.

Observed censoring means the noisy pre-clipped measurement exceeded `c`; it does not guarantee that the noise-free latent truth exceeds `c`. The CSV files therefore label false saturations explicitly, and the primary magnitude calibration is computed on pixels with latent truth above the ceiling.

`single_frame_exceedance_diagnostic.png` is the presentation-facing two-panel summary. `single_frame_exceedance_detailed.png` retains the raw-pixel diagnostic for audit.

## Held-out result

| Posterior representation | Observed censored | Latent above | True-above fraction | True exceedance | Predicted exceedance | Recovery | Slope | SD | Coverage | Mean abs. error/SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid saturated-only sampling | 2414 | 2108 | 0.873 | 2.779 K | 0.994 K | 0.358 | 0.009 | 0.584 K | 0.616 | 4.651 |
| coherent latent single-frame posterior | 2414 | 2108 | 0.873 | 2.779 K | 0.994 K | 0.358 | 0.009 | 0.584 K | 0.614 | 4.636 |

## Interpretation

For the coherent latent posterior, the regression slope of predicted on true exceedance is 0.009, and it recovers 35.8% of the aggregate true exceedance. A slope or recovery fraction well below one is direct evidence of magnitude compression rather than merely noisy ranking.

The posterior-SD slope is 0.0025 K per K of true exceedance. Thus uncertainty does not expand enough as the hidden magnitude grows.

In the hottest available exceedance bin, the mean true exceedance is 33.326 K while the inferred exceedance is 1.156 K. Posterior SD remains 0.598 K, coverage is 0.000, and mean standardized absolute error is 62.888.

The hybrid and coherent posterior rows are included to check whether making unsaturated locations latent changes the censored-pixel result. The coherent posterior is the primary scientific representation.