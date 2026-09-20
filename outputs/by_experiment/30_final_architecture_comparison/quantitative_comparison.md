# Quantitative companion to the reconstruction comparison

Values are equal-weight means over the 30 frozen held-out trajectories. The censored camera row is the observed-field reference shown in the reconstruction panel; it is not a probabilistic reconstruction, so CRPS, coverage, and interval width are not assigned.

| Output | Field L2 | RMSE | Top-1% RMSE | Peak error | Overall CRPS | Top-1% CRPS | Top-1% coverage / width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Censored camera (reference) | 0.752 | 0.936 K | 8.771 K | 30.315 K | -- | -- | -- |
| Posterior mean + RBF | 0.424 | 0.515 K | 4.173 K | 7.618 K | 0.728 K | 2.243 K | 0.855 / 12.56 K |
| Advective posterior mean + RBF | 0.419 | 0.509 K | 4.223 K | 7.678 K | 0.727 K | 2.281 K | 0.844 / 12.55 K |
| Legacy clipped mean + joint advective ST | 0.446 | 0.542 K | 4.553 K | 7.695 K | 0.566 K | 2.545 K | 0.776 / 9.90 K |
| Posterior mean + sequential ST | 0.407 | 0.495 K | 4.302 K | 7.598 K | 0.210 K | 2.738 K | 0.415 / 3.51 K |
| Posterior mean + sequential advective ST | 0.407 | 0.495 K | 4.302 K | 7.639 K | 0.210 K | 2.749 K | 0.388 / 3.33 K |
| Advective posterior mean + sequential advective ST | 0.404 | 0.492 K | 4.367 K | 7.663 K | 0.206 K | 2.843 K | 0.358 / 3.32 K |
| Posterior-sample mixture + sequential advective ST | 0.407 | 0.495 K | 4.297 K | 7.635 K | 0.210 K | 2.745 K | 0.399 / 3.35 K |

Lower is better for all error and CRPS columns. Coverage is not ranked alone and must be read together with interval width. The RBF rows use their frozen larger covariance amplitude, as in the original comparison.
