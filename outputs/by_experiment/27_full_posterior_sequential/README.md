# Full-posterior one-step sequential propagation

This focused experiment tests whether collapsing the previous censored posterior loses uncertainty before the one-step advective stochastic transition.

## Implementation clarification

The pre-existing sequential implementation was not strictly mean-only. It already propagated centered previous-posterior draws into `B Sigma B^T` and then used one moment-matched Gaussian prior. This experiment therefore reports three controlled versions under the same transition and current censored likelihood:

1. **Mean-only:** propagate the previous posterior mean and retain only innovation `C`.
2. **Moment-matched (existing):** use `C + B Sigma B^T`.
3. **Full-posterior mixture:** retain propagated draws as distinct component means, reweight them with the current mixed equality/censoring likelihood, and condition each selected component with the same innovation `C`.

The posterior-physics-mean + spatial RBF method is retained only as a reference.

## Held-out results

| Method | Field | RMSE | Overall CRPS | Top-1% CRPS | Signed error | Hot coverage | Hot width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean-only sequential advective stochastic ST | 0.407 | 0.495 | 0.211 | 2.808 | +0.036 | 0.717 | 2.631 |
| moment-matched sequential advective stochastic ST | 0.407 | 0.495 | 0.210 | 2.749 | +0.037 | 0.771 | 3.174 |
| full-posterior mixture sequential advective stochastic ST | 0.407 | 0.495 | 0.210 | 2.745 | +0.037 | 0.773 | 3.191 |
| posterior physics mean + RBF | 0.424 | 0.515 | 0.728 | 2.243 | +0.045 | 0.950 | 12.231 |

## Temperature-quartile diagnostics

| Method | Quartile | CRPS | Coverage | Signed error | RMSE | Width |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean-only sequential advective stochastic ST | Q1 | 0.174 | 1.000 | +0.070 | 0.182 | 2.517 |
| mean-only sequential advective stochastic ST | Q2 | 0.173 | 1.000 | +0.071 | 0.183 | 2.500 |
| mean-only sequential advective stochastic ST | Q3 | 0.174 | 1.000 | +0.072 | 0.186 | 2.511 |
| mean-only sequential advective stochastic ST | Q4 | 0.321 | 0.962 | -0.068 | 0.937 | 2.542 |
| moment-matched sequential advective stochastic ST | Q1 | 0.174 | 1.000 | +0.070 | 0.182 | 2.514 |
| moment-matched sequential advective stochastic ST | Q2 | 0.173 | 1.000 | +0.071 | 0.183 | 2.500 |
| moment-matched sequential advective stochastic ST | Q3 | 0.174 | 1.000 | +0.072 | 0.186 | 2.512 |
| moment-matched sequential advective stochastic ST | Q4 | 0.318 | 0.970 | -0.067 | 0.937 | 2.620 |
| full-posterior mixture sequential advective stochastic ST | Q1 | 0.174 | 1.000 | +0.070 | 0.182 | 2.513 |
| full-posterior mixture sequential advective stochastic ST | Q2 | 0.173 | 1.000 | +0.071 | 0.183 | 2.501 |
| full-posterior mixture sequential advective stochastic ST | Q3 | 0.175 | 1.000 | +0.072 | 0.186 | 2.514 |
| full-posterior mixture sequential advective stochastic ST | Q4 | 0.318 | 0.970 | -0.066 | 0.937 | 2.621 |

## Hottest-region diagnostics

Q4 is broad enough to conceal failure at the extreme peak. The two narrower regions are therefore retained separately.

| Method | Region | CRPS | Coverage | Signed error | RMSE | Width |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean-only sequential advective stochastic ST | Q4 | 0.321 | 0.962 | -0.068 | 0.937 | 2.542 |
| mean-only sequential advective stochastic ST | above camera ceiling | 1.213 | 0.717 | -0.741 | 2.581 | 2.631 |
| mean-only sequential advective stochastic ST | top 1% | 2.808 | 0.300 | -2.634 | 4.304 | 2.671 |
| moment-matched sequential advective stochastic ST | Q4 | 0.318 | 0.970 | -0.067 | 0.937 | 2.620 |
| moment-matched sequential advective stochastic ST | above camera ceiling | 1.197 | 0.771 | -0.736 | 2.583 | 3.174 |
| moment-matched sequential advective stochastic ST | top 1% | 2.749 | 0.388 | -2.622 | 4.302 | 3.330 |
| full-posterior mixture sequential advective stochastic ST | Q4 | 0.318 | 0.970 | -0.066 | 0.937 | 2.621 |
| full-posterior mixture sequential advective stochastic ST | above camera ceiling | 1.197 | 0.773 | -0.728 | 2.581 | 3.191 |
| full-posterior mixture sequential advective stochastic ST | top 1% | 2.745 | 0.399 | -2.620 | 4.297 | 3.352 |
| posterior physics mean + RBF | Q4 | 0.801 | 0.994 | -0.003 | 0.969 | 11.646 |
| posterior physics mean + RBF | above camera ceiling | 1.359 | 0.950 | -0.381 | 2.611 | 12.231 |
| posterior physics mean + RBF | top 1% | 2.243 | 0.855 | -2.367 | 4.173 | 12.557 |

## Direct answer

Relative to strict mean-only propagation, full-posterior mixture propagation changes overall CRPS by -0.0006 K, top-1% CRPS by -0.0633 K, hot coverage by +0.0561, and field error by -0.0001.
Its above-ceiling coverage is closer to the nominal 95% level than strict mean-only propagation.

Relative to the pre-existing moment-matched implementation, preserving the non-Gaussian mixture changes overall CRPS by +0.0001 K, top-1% CRPS by -0.0045 K, and hot coverage by +0.0022.

**Conclusion:** carrying previous-state uncertainty matters relative to a strict mean-only transition, but the existing moment-matched implementation already retains essentially all of that benefit. Preserving the full non-Gaussian mixture does not materially change reconstruction or calibration. Top-1% coverage is 0.399 for the full mixture versus 0.388 for moment matching, and both remain far below the RBF reference (0.855). The full mixture also retains a top-1% signed error of -2.620 K. Thus the sequential model's extreme-tail undercoverage is not explained by collapsing the previous posterior to its mean.

The mechanism diagnostic is consistent with that conclusion. Across held-out trajectories, the propagated between-component SD averages only 0.017 K, compared with 0.716 K from the one-step stochastic innovation. The likelihood-reweighted mixture retains an average effective sample size of 156.0 out of 180 components, so current data do not hide a strongly concentrated alternative propagated state.

The previous scientific conclusion is unchanged: the sequential stochastic/advective model has the best field error and overall CRPS, whereas posterior physics mean + RBF has the best top-1% CRPS and substantially better hottest-region coverage.

`paired_comparisons.csv` records trajectory-level changes and win counts. `quartile_results.csv` and `quartile_summary.csv` contain the requested temperature-stratified diagnostics.

## Reconstruction figures

For each development trajectory, `reconstruction_*.png` retains the full truth-range posterior-mean panels and posterior-SD panels. The companion `reconstruction_*_tail_focus.png` figures use one shared scale from ambient temperature to that trajectory's camera ceiling. The raster values are not smoothed; lightly smoothed contour guides at ambient + 0.25, 0.5, 1, and 2 K make the cooling wake easier to follow without tracing pixel-scale noise. Values above the ceiling are clipped only in the display normalization; posterior inference and every reported metric remain unchanged.

## Controlled assumptions

- Synthetic censoring fraction: 3%.
- All rows share the same previous posterior draws, current camera realization, physical parameters, thresholds, masks, hyperparameters, and seeds.
- The full mixture uses `ordinary posterior physics mean + B_adv(T_t^(s)-E[T_t])` as its component means. This exactly isolates non-Gaussian propagation from the existing moment-matched advective transition.
- The existing previous-posterior routine retains unsaturated noisy camera pixels at their observed values and samples the saturated pixels. This experiment reuses that representation unchanged rather than introducing a new state-inference model.
- Previous observations enter once, through the previous censored posterior; the final update conditions only on the current observations.
- Mixture particles are reweighted using the Gaussian density for unsaturated observations and the multivariate Gaussian exceedance probability for saturated observations.
- All CRPS values use the unbiased `M(M-1)` estimator.
