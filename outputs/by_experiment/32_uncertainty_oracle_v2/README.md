# Underdispersion and oracle diagnostic v2

This canonical rerun separates three distinct distributions:

1. `previous posterior`: p(T_(n-1) | Y_(n-1));
2. `current forecast`: p(T_n | Y_(n-1)), before the current image;
3. `current posterior`: p(T_n | Y_(n-1), Y_n).

The forecast and current-posterior rows use the same current-frame top-1% mask. The previous posterior uses its own previous-frame top-1% mask and is not described as a direct before/after spatial comparison.
The top-1% mask uses the inclusive empirical 99th-percentile threshold, matching the retained architecture comparison (26 pixels on the 2501-pixel grid).

## Previous-state inference

The hybrid representation pins unsaturated grid values to the noisy clipped camera frame and samples only saturated pixels. The coherent latent posterior conditions on the same sparse-plus-saturated observation set but samples the entire latent field jointly, including measurement uncertainty at unsaturated points. No current-frame data enter either previous posterior.

## Source coupling

The fixed-gamma table is the causal diagnostic: every representation uses the same retained source coupling. The recalibrated table is a separate attainable-performance comparison: gamma is fitted independently for each representation using only the final one-step transitions of the three development trajectories.

| State representation | Source coupling | Calibration scope |
|---|---:|---|
| hybrid previous state | 0.167411 | final one-step transition pooled over three development trajectories |
| coherent latent previous state | 0.166314 | final one-step transition pooled over three development trajectories |
| true previous state | 0.052082 | final one-step transition pooled over three development trajectories |
| frozen project baseline | 0.170223 | retained observed-clipped project calibration |

## Fixed-gamma current posterior

| Model | Field L2 | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|
| hybrid previous state; mean-only; fixed gamma | 0.407 | 0.211 | 4.304 | 2.808 | 0.300 | 2.671 |
| hybrid previous state; moment-matched; fixed gamma | 0.407 | 0.210 | 4.302 | 2.749 | 0.388 | 3.330 |
| coherent latent previous state; mean-only; fixed gamma | 0.440 | 0.228 | 4.311 | 2.817 | 0.304 | 2.663 |
| coherent latent previous state; moment-matched; fixed gamma | 0.501 | 0.595 | 4.304 | 2.759 | 0.390 | 3.292 |
| strict oracle; exact previous state; fixed gamma | 0.550 | 0.182 | 6.592 | 2.375 | 0.765 | 2.696 |

## Development-recalibrated current posterior

| Model | Field L2 | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|
| hybrid previous state; mean-only; development-recalibrated gamma | 0.407 | 0.211 | 4.304 | 2.809 | 0.299 | 2.671 |
| hybrid previous state; moment-matched; development-recalibrated gamma | 0.408 | 0.210 | 4.302 | 2.750 | 0.390 | 3.330 |
| coherent latent previous state; mean-only; development-recalibrated gamma | 0.440 | 0.228 | 4.312 | 2.818 | 0.304 | 2.663 |
| coherent latent previous state; moment-matched; development-recalibrated gamma | 0.501 | 0.595 | 4.304 | 2.761 | 0.388 | 3.292 |
| recalibrated oracle; exact previous state; development-recalibrated gamma | 0.364 | 0.174 | 4.268 | 1.632 | 0.805 | 2.696 |

## Calibration diagnosis

Previous-posterior hottest-tail coverage:

- Hybrid: 0.237 with SD 0.614 K.
- Coherent latent: 0.238 with SD 0.614 K.

For each fixed-gamma moment-matched model, `forecast_update_summary.csv` reports the current forecast and current posterior on the identical current top-1% mask. This distinguishes transition uncertainty from the effect of assimilating Y_n.

## Top-1% definition audit

The earlier 2.749 K versus 2.839 K discrepancy was entirely a region-definition difference. The retained architecture table used an inclusive empirical 99th-percentile threshold (26 pixels); oracle v1 used exact ranks (25 pixels). V2 adopts the earlier inclusive threshold and reproduces the architecture result.

| Source | Top-1% pixels | Top-1% CRPS (K) |
|---|---:|---:|
| retained architecture comparison | 26 | 2.749229 |
| oracle diagnostic v1 | 25 | 2.838705 |
| canonical oracle diagnostic v2 | 26 | 2.749229 |

## Interpretation

Making unsaturated pixels latent repairs the ordinary-field posterior degeneracy, but it does not materially change the previous hottest-tail RMSE, SD, coverage, or CRPS. The hottest 1% is already censored, so the dominant underdispersion originates in single-frame censored peak inference, not in pinning the unsaturated values. Propagating the coherent covariance raises ordinary-field uncertainty enough to worsen all-domain CRPS, while hottest-tail calibration remains essentially unchanged.

The current observation update also does not repair the tail: relative to the forecast it leaves RMSE almost unchanged and slightly reduces coverage. The strict oracle overshoots because its source coefficient was calibrated for clipped-state compensation; the separately recalibrated oracle is the appropriate attainable-performance ceiling.

This evidence does not yet justify adaptive Q as the first fix. The next modeling question should target the saturated single-frame posterior and its peak prior/mean; transition innovation can be revisited after that state posterior is better calibrated.

All CRPS values use the unbiased M(M-1) estimator. Held-out trajectories never affect source calibration or any fixed hyperparameter.