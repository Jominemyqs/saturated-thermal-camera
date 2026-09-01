# Uncertainty-origin diagnostics and oracle benchmark

This experiment uses the frozen 3% censoring protocol, the three fixed development trajectories, and the remaining 30 trajectories for equal-weight held-out summaries. No adaptive or state-dependent covariance was added.

## Posterior representation

The previous-frame object is a hybrid censored representation, not a full latent posterior. Saturated pixels are sampled from the censored RBF GP. Unsaturated noisy pixels are retained at their observed values in every draw, so their represented posterior variance is exactly zero. Standardized errors are reported only where SD is positive, and zero-SD/nonzero-error pixels are reported separately.

## Main diagnosis

Hot-tail underdispersion is already present before propagation; the transition does not appear to be its sole origin.

Supporting evidence:

- In the sampled saturated part of the previous frame, 95% coverage is 0.594; in its hottest 1%, coverage is only 0.223.
- The previous hottest 1% has mean signed error -5.382 K and mean |e|/SD 11.114.
- Moment matching and the posterior-sample mixture give top-1% CRPS 2.839 and 2.835 K, respectively, so retaining the sampled non-Gaussian shape does not materially fix the tail.
- Recalibrating source coupling for true previous states on development paths changes gamma from 0.1702 to 0.0588. This confirms that transition-mean calibration depends on the previous-state representation.

Top-1% before/after averages:

| Stage | RMSE (K) | Bias (K) | SD (K) | Coverage | Width (K) | Mean |e|/SD |
|---|---:|---:|---:|---:|---:|---:|
| Previous censored representation | 9.666 | -5.382 | 0.614 | 0.223 | 2.192 | 11.114 |
| Current sequential advective ST | 4.384 | -2.698 | 0.874 | 0.365 | 3.332 | 3.849 |

## Oracle benchmark

The oracle receives only the true latent previous frame and never receives uncensored current-frame truth. The fixed-source oracle changes only the previous state. Because the fixed source coefficient was calibrated for clipped propagation, a second row recalibrates that coefficient using only the three development trajectories and true previous states. Both retain the same current finite-step innovation, observations, and likelihood.

| Method | Field L2 | Overall RMSE | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Top-1% coverage | Peak error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Observed-clipped previous state + ST innovation | 0.457 | 0.556 | 0.218 | 5.081 | 3.756 | 0.027 | 7.836 |
| Posterior-mean previous state + ST innovation | 0.407 | 0.495 | 0.211 | 4.386 | 2.901 | 0.273 | 7.620 |
| Moment-matched sequential advective ST | 0.407 | 0.495 | 0.210 | 4.384 | 2.839 | 0.365 | 7.639 |
| Posterior-sample mixture sequential advective ST | 0.407 | 0.495 | 0.210 | 4.379 | 2.835 | 0.376 | 7.635 |
| Oracle true previous state + fixed clipped-calibrated source | 0.550 | 0.678 | 0.182 | 6.723 | 2.463 | 0.756 | 26.430 |
| Oracle true previous state + dev-calibrated source | 0.366 | 0.445 | 0.174 | 4.374 | 1.695 | 0.793 | 14.064 |

All CRPS values use the unbiased M(M-1) empirical estimator. RMSE, MAE, relative L2, signed error, and peak error evaluate point reconstruction; CRPS evaluates the full predictive distribution; coverage is interpreted together with interval width.