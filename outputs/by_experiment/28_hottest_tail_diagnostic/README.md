# Hottest-tail coverage diagnostic

This diagnostic leaves the experiment-27 models unchanged. Pixel-level posteriors were not saved, so the held-out trajectories were rerun with the frozen configuration and their original global trajectory seed indices. Development trajectories were not rerun.

## Existing held-out results

| Region | Method | RMSE | MAE | Bias | CRPS | Coverage | Width | SD | RBF/method width | RBF/method SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | moment-matched ST | 0.495 | 0.183 | +0.037 | 0.210 | 0.992 | 2.537 | 0.663 | 4.52 | 4.52 |
| overall | full-mixture ST | 0.495 | 0.183 | +0.037 | 0.210 | 0.993 | 2.537 | 0.664 | 4.52 | 4.52 |
| overall | posterior mean + RBF | 0.515 | 0.207 | +0.045 | 0.728 | 0.998 | 11.458 | 2.996 | 1.00 | 1.00 |
| Q4 | moment-matched ST | 0.937 | 0.350 | -0.067 | 0.318 | 0.970 | 2.620 | 0.685 | 4.44 | 4.45 |
| Q4 | full-mixture ST | 0.937 | 0.351 | -0.066 | 0.318 | 0.970 | 2.621 | 0.686 | 4.44 | 4.44 |
| Q4 | posterior mean + RBF | 0.969 | 0.397 | -0.003 | 0.801 | 0.994 | 11.646 | 3.046 | 1.00 | 1.00 |
| above camera ceiling | moment-matched ST | 2.583 | 1.441 | -0.736 | 1.197 | 0.771 | 3.174 | 0.832 | 3.85 | 3.85 |
| above camera ceiling | full-mixture ST | 2.581 | 1.442 | -0.728 | 1.197 | 0.773 | 3.191 | 0.839 | 3.83 | 3.82 |
| above camera ceiling | posterior mean + RBF | 2.611 | 1.574 | -0.381 | 1.359 | 0.950 | 12.231 | 3.201 | 1.00 | 1.00 |
| top 1% | moment-matched ST | 4.302 | 3.154 | -2.622 | 2.749 | 0.388 | 3.330 | 0.873 | 3.77 | 3.76 |
| top 1% | full-mixture ST | 4.297 | 3.149 | -2.620 | 2.745 | 0.399 | 3.352 | 0.879 | 3.75 | 3.74 |
| top 1% | posterior mean + RBF | 4.173 | 2.927 | -2.367 | 2.243 | 0.855 | 12.557 | 3.286 | 1.00 | 1.00 |

The width and SD ratios are uncertainty-scale comparisons; RMSE, MAE, and signed error describe posterior-mean reconstruction quality.

## Selected held-out spatial cases

| Selection | Trajectory | ST top-1% coverage | RBF top-1% coverage | Difference |
| --- | --- | ---: | ---: | ---: |
| smallest difference | HorizontalScanPath_10 | 0.731 | 0.885 | +0.154 |
| median difference | HorizontalScanPath_5 | 0.385 | 0.846 | +0.462 |
| largest difference | DiagonalScanPath_7 | 0.077 | 0.808 | +0.731 |

## Direct covariance-scale check

The finite-step ST innovation has marginal SD 0.716 K (mean diag(C)=0.513 K^2), whereas the spatial RBF has prior marginal SD 3.431 K (sigma_f^2=11.768 K^2). The RBF/ST prior-SD ratio is 4.79.

## Variance-matched RBF control

This control keeps the posterior-physics prior mean, RBF correlation geometry, observations, likelihood, noise, and seeds fixed, while replacing the RBF prior marginal variance with mean diag(C) from the ST innovation.

| Region | Method | RMSE | MAE | CRPS | Coverage | Width | SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | full-mixture ST | 0.495 | 0.183 | 0.210 | 0.993 | 2.537 | 0.664 |
| overall | posterior mean + RBF | 0.515 | 0.207 | 0.728 | 0.998 | 11.458 | 2.996 |
| overall | variance-matched RBF | 0.495 | 0.182 | 0.211 | 0.991 | 2.528 | 0.661 |
| Q4 | full-mixture ST | 0.937 | 0.351 | 0.318 | 0.970 | 2.621 | 0.686 |
| Q4 | posterior mean + RBF | 0.969 | 0.397 | 0.801 | 0.994 | 11.646 | 3.046 |
| Q4 | variance-matched RBF | 0.937 | 0.349 | 0.321 | 0.962 | 2.552 | 0.667 |
| above camera ceiling | full-mixture ST | 2.581 | 1.442 | 1.197 | 0.773 | 3.191 | 0.839 |
| above camera ceiling | posterior mean + RBF | 2.611 | 1.574 | 1.359 | 0.950 | 12.231 | 3.201 |
| above camera ceiling | variance-matched RBF | 2.581 | 1.433 | 1.213 | 0.718 | 2.638 | 0.689 |
| top 1% | full-mixture ST | 4.297 | 3.149 | 2.745 | 0.399 | 3.352 | 0.879 |
| top 1% | posterior mean + RBF | 4.173 | 2.927 | 2.243 | 0.855 | 12.557 | 3.286 |
| top 1% | variance-matched RBF | 4.304 | 3.160 | 2.808 | 0.303 | 2.674 | 0.700 |

## Explicit answers

1. **RBF's top-1% coverage advantage is mainly interval width.** Its top-1% RMSE improves by only 0.123 K and MAE by 0.222 K, while its interval is 3.75 times wider and coverage rises from 0.399 to 0.855.

2. **The mean helps modestly, but the CRPS change is not additively decomposable.** RBF reduces top-1% MAE by 0.222 K and CRPS by 0.502 K. Holding the RBF prior physics mean and correlation geometry fixed while matching its prior variance to ST changes top-1% CRPS from 2.243 to 2.808 K and coverage from 0.855 to 0.303. Its matched RMSE and MAE are 4.304 and 3.160 K, so even the modest original posterior-mean advantage largely disappears. Covariance amplitude affects both posterior spread and how strongly censored observations update the posterior mean; an additive mean-versus-spread percentage would therefore be misleading.

3. **Near-nominal above-ceiling RBF coverage does not imply a better forecast.** RBF uses 12.231 K intervals versus 3.191 K for ST. The diffuse RBF intervals cover 0.950, but its CRPS is worse (1.359 versus 1.197 K) because CRPS penalizes unnecessary spread as well as misses.

4. **ST uncertainty is strongly temperature-dependent in the wrong way.** In the coolest decile its coverage is 1.000, posterior SD 0.660 K, and mean |error|/SD 0.198; in the hottest 1% these become 0.376, 0.880 K, and 3.832. Thus a roughly fixed uncertainty scale is excessive in the background and insufficient at the peak.

5. **Variance matching is the clean scale control.** The matched-RBF top-1% coverage is 0.303, compared with 0.855 before matching and 0.399 for full-mixture ST; its CRPS is 2.808 versus 2.745 K for ST. Matching fully removes and slightly reverses the apparent RBF tail advantage. The remaining difference is the appropriate evidence for mean/correlation geometry rather than marginal amplitude.

Overall, the remaining sequential-model problem is primarily a state-/temperature-dependent uncertainty problem, not evidence that RBF geometry is uniformly better. The full-mixture ST still has the better above-ceiling CRPS despite lower coverage.

## Reproducibility

The percentile plot uses stable within-trajectory ranks, so the 99-100% bin contains exactly 25 pixels per 61 x 41 field. The saved `top_1pct` region uses `truth >= quantile(truth, 0.99)` and contains 26 pixels; this explains the small difference between the two displayed hottest-bin aggregates.

The largest absolute difference between rerun and saved experiment-27 region metrics is 1.776e-15. `reproduction_checks.csv` records every comparison.
The variance-matched RBF changes only prior covariance amplitude; it retains the same posterior-physics prior mean, RBF correlation structure, observations, censored likelihood, measurement noise, and sampler seeds. Its conditioned posterior mean is allowed to change, as it must when the prior covariance changes.
