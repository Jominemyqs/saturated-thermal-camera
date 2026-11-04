# Matched current updates after previous state unpinning

Output 61, [driver 65](../../../scripts/65_thermal_unpinning_current_update.py).
This closes the current-update layer of [output 60](../60_previous_unpinning_audit/README.md)
on the same two native development trajectories. It does not replace historical
baselines or introduce a source/noise/covariance calibration.

## Results: reference ceiling

Primary paired seed; errors, CRPS and widths in K. Coverage is physical-area
weighted within the fixed hottest 1%, not coverage at the true peak.

| Trajectory | Previous representation | Hot RMSE | Hot CRPS | 95% coverage | 95% width | True-peak signed error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal 8 | Legacy hybrid | 4.418 | 2.257 | 77.26% | 6.350 | +10.487 |
| Diagonal 8 | All-readings pinned | 3.611 | 1.777 | 79.69% | 5.508 | +11.270 |
| Diagonal 8 | All-readings latent | 3.616 | 1.782 | 79.69% | 5.516 | +11.249 |
| Spiral 13 | Legacy hybrid | 4.077 | 2.151 | 75.52% | 6.402 | +22.899 |
| Spiral 13 | All-readings pinned | 3.441 | 1.670 | 68.23% | 3.142 | +21.515 |
| Spiral 13 | All-readings latent | 3.443 | 1.670 | 68.23% | 3.153 | +21.490 |

All reference true-peak intervals miss truth, including the second paired seed.
The conditioning bridge improves aggregate hot RMSE/CRPS, but calibration does
not improve uniformly: Spiral coverage falls while its intervals nearly halve
in width. The Diagonal true peak is slightly worse despite better hot-area RMSE.

Unpinning alone has little reference hot-tail effect. Across the two seeds,
its paired hot RMSE change is +0.0040 to +0.0050 K (Diagonal) and +0.0017 to
+0.0027 K (Spiral). These changes are no larger than the corresponding
same-arm repeat variation; they are not evidence of a useful hot-tail gain.
The largest same-arm reference repeat changes are 0.0115 K in hot RMSE,
0.0159 K in hot CRPS, 1.05 percentage points in coverage and 0.0813 K in width.
Two seeds are a numerical sensitivity check, not a formal Monte Carlo error bound.

| Trajectory | Previous representation | Current-unused RMSE | Current-unused CRPS | Current-unused 95% width |
| --- | --- | ---: | ---: | ---: |
| Diagonal 8 | Legacy hybrid | 0.2882 | 0.1935 | 2.6937 |
| Diagonal 8 | All-readings pinned | 0.2294 | 0.1904 | 2.6768 |
| Diagonal 8 | All-readings latent | 0.2256 | 0.1945 | 2.7728 |
| Spiral 13 | Legacy hybrid | 0.3549 | 0.2101 | 2.6829 |
| Spiral 13 | All-readings pinned | 0.2842 | 0.2061 | 2.6537 |
| Spiral 13 | All-readings latent | 0.2832 | 0.2109 | 2.7499 |

In the previously unused current region, unpinning modestly lowers point RMSE
but increases width and CRPS relative to the pinned bridge. More uncertainty
is not automatically better calibrated predictive performance. These locations
are omitted current observations, not necessarily ambient background.

## Results: higher ceilings

Same retained current observations within each case, one paired seed. Each
entry lists legacy hybrid -> all-readings pinned -> all-readings latent.

| Case | Hot RMSE (K) | Hot CRPS (K) | Hot 95% coverage |
| --- | --- | --- | --- |
| Diagonal, 320 K | 2.458 -> 2.515 -> 2.462 | 0.751 -> 0.797 -> 0.798 | 58.85% -> 59.20% -> 55.03% |
| Diagonal, 325 K | 2.478 -> 2.491 -> 2.304 | 0.767 -> 0.819 -> 0.778 | 56.60% -> 53.47% -> 52.08% |
| Spiral, 320 K | 2.521 -> 2.522 -> 2.427 | 0.746 -> 0.767 -> 0.725 | 69.10% -> 68.75% -> 73.96% |
| Spiral, 325 K | 2.203 -> 2.228 -> 2.067 | 0.660 -> 0.694 -> 0.628 | 70.49% -> 69.79% -> 70.66% |

Unpinning lowers hot RMSE relative to the pinned bridge in all four high-ceiling
cases, by 0.053-0.188 K. This is larger than its reference effect; it would be
incorrect to conclude that unpinning never affects the final posterior.
However, Diagonal coverage falls and its CRPS remains worse than legacy;
Spiral hot scores improve but whole-field and omitted-region errors worsen.
In all four cases, unpinning worsens the fixed along-path shoulder RMSE relative
to pinned (Diagonal 320: 1.623 -> 1.872 K; Diagonal 325: 2.165 -> 2.360 K;
Spiral 320: 1.527 -> 1.931 K; Spiral 325: 2.291 -> 2.730 K).

All 24 completed fits miss the true peak with their pointwise 95% interval.
The full-latent true-peak error remains +11.907 / +10.991 K for Diagonal at
320/325 K, and +21.435 / +19.207 K for Spiral. Better hot-area RMSE must not
be confused with resolving extreme-peak undercoverage.

**Decision:** close this bounded statistical audit without replacing the
historical baseline or expanding trajectories. Preserve the full-latent
implementation as the more complete representation, but do not claim it is a
uniform performance improvement. Return the next scientific discussion to
independently available source inputs and the frozen physics-mean assumptions;
this experiment does not itself establish the correct physical replacement.

## Reliability and cost

All 24 fits passed unchanged gates. Maximum rank-normalized split R-hat was
1.00752; minimum bulk/tail ESS were 921/1,408; maximum classical R-hat was
1.00481. Reference repeats support the large legacy-to-bridge differences,
not a useful reference hot-tail benefit from unpinning alone. High ceilings
have only one paired seed, so small differences remain qualified.

Successful fit timers total 636.90 s, 10.11-61.73 s per four-chain fit. Prior
preparation took 247.72 s, and six historical replays took 51.95 s. Peak RSS
was 1,612 MiB during preparation and at most 720 MiB in fit workers. These are
measured single-process peaks, not a whole-machine memory bound. Elapsed time
from configuration creation to completion was 1,596.98 s, including cache I/O,
environment recovery and interruptions; it is not the sum of pure fit timers.

An interrupted diagnostic import required restoring identical temporary
packages and repeating one unsaved fit with the same seed. No completed fit
was overwritten, and no statistical retry or gate relaxation was used.
See [validation](validation.md), [execution verification](verification.json)
and [environment recovery](environment_recovery.json).

## Figures and numerical results

- [Spiral reference cross-sections](cross_sections_SpiralScanPath_13_level_0.png)
  show improved shoulder reconstruction but persistent peak error.
- [Diagonal 325 K cross-sections](cross_sections_DiagonalScanPath_8_level_2.png)
  show the full current-update comparison with retained measurements.
- [Diagonal signed-error/SD maps](signed_errors_DiagonalScanPath_8.png) and
  [Spiral signed-error/SD maps](signed_errors_SpiralScanPath_13.png) use shared scales.
- [All region metrics](results.csv), [paired changes](paired_changes.csv), and
  [sampling diagnostics/runtime](convergence_runtime.csv) retain every case/seed.

## Controlled question

Do the changes to previous-state conditioning and representation survive the
same current likelihood, and how do they affect reconstruction and calibration?

| Arm | Previous representation | Attribution |
| --- | --- | --- |
| Legacy hybrid | Original sparse censored inference, all unsaturated noisy values pinned | Historical baseline |
| All-readings pinned | All unsaturated readings condition censored inference; unsaturated latent values still pinned | Changes how already-used readings enter inference |
| All-readings latent | Exactly the same all-readings conditioning; latent unsaturated mean/covariance and cross terms retained | Unpinning relative to the matched pinned bridge |

All previous posterior chains and forecasts are reused from outputs 52/60.
No previous inference is rerun. The same current noisy camera, retained IDs,
likelihood SD 0.50 K, source contribution, physical coefficients and Q apply
to all arms within each trajectory/ceiling. The mean and inherited covariance
are allowed to differ according to the saved previous representation.

The legacy-to-pinned contrast also inherits output 60's analytic joint censored
residual covariance, whereas legacy uses its saved empirical marginal-draw
covariance. It is therefore a conditioning bridge, not a mathematically pure
one-factor current-posterior ablation. Output 59 found a small correlation-only
prior-diagonal effect, but did not prove a negligible current-posterior effect.
The pinned-to-latent contrast is matched on conditioning and censored marginals.

Current retained observations are the fixed union of numerical candidates and
observed reference saturation. At every retained node, use a numerical reading
if unsaturated and a noisy lower-bound likelihood if saturated, exactly once.
No upper/non-saturation flags or additional measurements are added. Reference
IDs match output 52; 320/325 K IDs match output 53's retained branch.

## Full covariance construction

Let P=(K_previous+vI)^(-1), with v equal to previous observation variance plus
the inherited numerical jitter. For completed noisy previous signals Z,

```text
C = Cov(T_previous | Z) = v I - v^2 P
W = C[:, S_previous] / v
Sigma_previous = W Cov(Z_S | previous camera) W^T + C
```

The new [block helper](../../../src/unpinning_current_update.py) computes both
observation-observation and prediction-observation blocks of

```text
K_current_prior = Q_dt + A Sigma_previous A^T
```

For the pinned bridge, mask previous unsaturated coordinates on BOTH sides
of Sigma. For the full latent arm, retain all coordinates and cross terms.
The residual contribution is integrated analytically, not replaced by its
diagonal or independently sampled pixelwise before propagation. Each resulting
prior diagonal is verified against output 60; this is an equality check, not
the method used to construct off-diagonal entries.

Only current observations enter the final likelihood. Censoring applies to
the noisy camera signal, not a hard latent temperature bound. The existing
current block sampler is reused via its tested lower-only interface. Final
residual prediction draws preserve the historical **marginal-only** convention
for pointwise scores and intervals; they are not coherent spatial field draws
and are not propagated further. Full cross-covariance is used in conditioning.

## Frozen protocol and reliability

- DiagonalScanPath_8 and SpiralScanPath_13; reference, 320 K and 325 K.
- Historical short-chain legacy runs are replayed exactly before comparisons.
- Four current chains, 800 retained per chain after 300 burn-in, thinning 1.
- Same 300 balanced evaluation draws per chain (1,200 pooled) in every arm.
- Reference has two paired current seeds; higher ceilings have one paired seed.
- No statistical retries. All reference arm/repeat fits must pass before high-ceiling runs.
- Rank-normalized split R-hat <= 1.01; bulk/tail ESS >= 400; classical R-hat <= 1.05.
- Monitor every constrained noisy signal and latent temperatures at every retained observation plus 64 geometry-selected locations; classical diagnostics also cover the full latent field.
- New task budget: 900 seconds per fit/worker and 4,096 MiB peak RSS. This does not alter output 55's historical 300-second gate or its stop decision.

Initialization remains the reflected-HMC conditional-mean/threshold rule.
Hyperparameters, camera noise, previous likelihood noise and current likelihood
noise are not fitted. The inherited relative numerical-jitter rule is unchanged;
its absolute value can differ with each prior's scale.

## Evaluation and limitations

Scores use the cached native physical-area weights and fixed current hottest
1% of area. Other regions are current censored, current unsaturated, omitted
current locations, whole field and the frozen along/across shoulder sections.
The true-peak interval/error is reported separately. Area weights do not enter
the likelihood. No pixels are treated as independent experimental replicates.

CRPS retains the historical M(M-1) convention; its formal unbiasedness assumes
independent samples, whereas these are MCMC draws. Paired seeds measure numerical
sensitivity, not held-out generalization or formal confidence intervals.

The historical mean-floor versus linear covariance-propagation approximation
is unchanged. Gaussian moment matching of the previous posterior is unchanged.
The gradient input remains simulator-derived HeatFluxZ, a projected vertical
temperature gradient, not a newly available independent laser source. Native
inputs still use the existing fine-grid propagation/interpolation operator.

The nested inherited configuration includes historical grid/calibration labels;
effective masks are native area weights from caches, not a top-25 native-node
rule. Top-level sampling, likelihood and execution flags govern this audit.

## Files

- `results.csv`: per-arm, per-region metrics and true-peak intervals.
- `paired_changes.csv`: conditioning, unpinning and combined differences within each case/seed.
- `convergence_runtime.csv`: every fit's diagnostics, runtime and gate.
- `prior_verification.csv`: covariance diagonal equality and matched observation/source/Q hashes.
- `reference_gate.json` and `verification.json`: executed scope and gate decisions.
- `source_and_input_fingerprints.json`: working-tree source and frozen inputs.
- Local ignored `by_trajectory/` caches contain full prior covariance blocks, retained observation layouts, evaluation chains, noisy constrained chains and monitored latent chains.

Cross-sections show the unchanged truth, arm-specific physics forecast, current
posterior mean and pointwise 95% intervals on the same physical sections.
Reference error/SD maps share scales across arms.
