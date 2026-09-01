# Final architecture-aware comparison

The eight requested labels reduce to seven distinct architectures. Requested item 5, `posterior physics mean + sequential advective ST`, is exactly the moment-matched `C + B Sigma B^T` implementation also named in item 7, so it appears once.

## Main complete-model comparison

| # | Architecture | Field | RMSE | Overall CRPS | Top-1% CRPS | Peak error | Above RMSE | Above bias | Above cov. | Above width | Top-1% RMSE | Top-1% bias | Top-1% cov. | Top-1% width |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Posterior physics mean + RBF | 0.424 | 0.515 | 0.728 | 2.243 | 7.618 | 2.611 | -0.381 | 0.950 | 12.231 | 4.173 | -2.367 | 0.855 | 12.557 |
| 2 | Advective posterior physics mean + RBF | 0.419 | 0.509 | 0.727 | 2.281 | 7.678 | 2.635 | -0.483 | 0.946 | 12.231 | 4.223 | -2.451 | 0.844 | 12.554 |
| 3 | Legacy latent-clipped mean + joint advective ST | 0.446 | 0.542 | 0.566 | 2.545 | 7.695 | 2.793 | -0.815 | 0.922 | 9.483 | 4.553 | -2.938 | 0.776 | 9.897 |
| 4 | Posterior physics mean + sequential ST (moment-matched) | 0.407 | 0.495 | 0.210 | 2.738 | 7.598 | 2.584 | -0.740 | 0.790 | 3.354 | 4.302 | -2.634 | 0.415 | 3.513 |
| 5 | Posterior physics mean + sequential advective ST (moment-matched) | 0.407 | 0.495 | 0.210 | 2.749 | 7.639 | 2.583 | -0.736 | 0.771 | 3.174 | 4.302 | -2.622 | 0.388 | 3.330 |
| 6 | Advective posterior mean + sequential advective ST (moment-matched) | 0.404 | 0.492 | 0.206 | 2.843 | 7.663 | 2.623 | -0.867 | 0.754 | 3.178 | 4.367 | -2.738 | 0.358 | 3.324 |
| 7 | Posterior-sample mixture sequential advective ST | 0.407 | 0.495 | 0.210 | 2.745 | 7.635 | 2.581 | -0.728 | 0.773 | 3.191 | 4.297 | -2.620 | 0.399 | 3.352 |

Coverage is shown beside interval width deliberately. The complete-model RBF rows retain their frozen, much larger covariance amplitude and are not covariance-geometry controls.

## Direct answers

1. **Best whole-field reconstruction:** Advective posterior mean + sequential advective ST (moment-matched) has the smallest excess-field error (0.404).
2. **Best overall CRPS:** Advective posterior mean + sequential advective ST (moment-matched) has overall CRPS 0.206 K.
3. **Best top-1% CRPS:** Posterior physics mean + RBF has top-1% CRPS 2.243 K under its frozen complete-model variance.
4. **Peak reconstruction is not substantially better for that method.** The smallest top-1% RMSE is 4.173 K for Posterior physics mean + RBF; peak errors across the leading models remain close, so the unmatched RBF tail advantage is mainly probabilistic spread rather than a dramatically better peak mean.
5. **Covariance amplitude explains most of the original RBF tail advantage.** Original RBF top-1% width is 12.557 K versus 2.674 K after variance matching; coverage changes from 0.855 to 0.303.
6. **Variance matching removes and reverses the apparent RBF geometry advantage.** Top-1% CRPS changes from 2.243 to 2.808 K after matching, compared with 2.745 K for the posterior-sample mixture.
7. **Previous-state uncertainty matters modestly.** Mean-only to moment matching changes top-1% CRPS from 2.808 to 2.749 K and top-1% coverage from 0.300 to 0.388. Retaining the censored-region sample mixture changes these only to 2.745 K and 0.399.
8. **The remaining sequential weakness is temperature/state-dependent underdispersion and negative hottest-tail bias.** It is not primarily missing previous-state uncertainty, because moment matching and posterior-sample propagation are nearly identical. The variance-matched RBF also performs similarly to the sequential geometry, so inferior ST covariance geometry is not supported as the main explanation.

## Scientific interpretation

The sequential stochastic space-time architectures are the strongest for whole-field reconstruction and overall probabilistic sharpness. The originally configured snapshot RBF is a valid complete model and remains useful as a robust hot-tail reference, but its high hottest-tail coverage is purchased with intervals roughly four times wider. Once marginal amplitude is controlled, the RBF tail advantage disappears. There is therefore no universal winner: the current sequential model is sharp and strong globally, while its remaining error is concentrated at the hottest pixels, where it is negatively biased and underdispersed.