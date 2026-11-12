# Experiment 63: remove the current gradient increment

## Result

**Removing the frozen increment is not an effective replacement.** All five
fully gate-passing primary case comparisons worsen hot-area posterior RMSE,
CRPS and whole-field RMSE. The sixth (Spiral 320 K) shows the same direction but
is provisional because one rank-R-hat narrowly fails. Neither arm covers the
true peak in any completed fit, including the reference repeats.

Primary paired-seed results below use K for RMSE, CRPS and signed peak error;
each cell is source-on -> source-off. Hot area is fixed hottest 1% physical area,
not 1% of mesh nodes. Peak error is evaluated at the fixed true peak node.

| Case / ceiling | Hot-area RMSE | Hot-area CRPS | True-peak signed error | Whole-field RMSE |
| --- | --- | --- | --- | --- |
| Diagonal / reference 296.121 K | 3.616 -> 7.528 | 1.782 -> 3.374 | +11.25 -> -37.43 | 0.518 -> 0.842 |
| Diagonal / 320 K | 2.462 -> 2.868 | 0.798 -> 0.906 | +11.91 -> -18.22 | 0.355 -> 0.401 |
| Diagonal / 325 K | 2.304 -> 2.572 | 0.778 -> 0.876 | +10.99 -> -16.14 | 0.381 -> 0.396 |
| Spiral / reference 296.490 K | 3.443 -> 7.540 | 1.670 -> 3.317 | +21.49 -> -39.99 | 0.486 -> 0.829 |
| Spiral / 320 K (provisional) | 2.427 -> 2.812 | 0.725 -> 0.854 | +21.43 -> -20.78 | 0.406 -> 0.479 |
| Spiral / 325 K | 2.067 -> 2.496 | 0.628 -> 0.831 | +19.21 -> -18.59 | 0.473 -> 0.518 |

At reference, source-off current observations add only 2.10 K (Diagonal) and
1.86 K (Spiral) at the peak, leaving about 37--40 K underprediction. At higher
ceilings they add 8.12--9.58 K, but substantial underprediction remains. The
source-on gradient increments removed at these nodes are 50.249 and 62.312 K,
respectively, identical across ceilings within a trajectory.

The deterministic and posterior conclusions differ. At reference, removing the
increment raises forecast hot-area RMSE from 3.691 to 8.241 K (Diagonal) and
3.581 to 8.247 K (Spiral). At 320/325 K, it lowers forecast hot-area RMSE from
5.60--5.98 to 4.83--4.96 K, yet the final posterior still has worse hot-area
RMSE. A smaller deterministic error is therefore not sufficient evidence for
an improved reconstruction after current conditioning.

Shoulder results are mixed, not a uniform improvement:

| Case / ceiling | Along-shoulder posterior RMSE | Across-shoulder posterior RMSE |
| --- | --- | --- |
| Diagonal / reference | 1.928 -> 3.041 | 0.609 -> 0.785 |
| Diagonal / 320 K | 1.872 -> 1.587 | 1.977 -> 2.303 |
| Diagonal / 325 K | 2.360 -> 1.754 | 2.722 -> 2.579 |
| Spiral / reference | 3.564 -> 5.145 | 0.476 -> 0.765 |
| Spiral / 320 K (provisional) | 1.931 -> 3.275 | 2.375 -> 2.121 |
| Spiral / 325 K | 2.730 -> 4.033 | 3.351 -> 2.313 |

Reference hot-area coverage falls from 79.7% to 75.7% (Diagonal) and 68.2% to
64.8% (Spiral), while interval widths barely change (5.516 -> 5.455 K and
3.153 -> 3.110 K). Higher-ceiling hot-area coverage moves in mixed directions,
but the true peak remains uncovered in both arms. The slightly smaller absolute
Spiral high-ceiling peak error does not offset worse area-wide scores.

## Reliability and runtime

All four reference source-off fits pass unchanged gates. Across reference
repeats, the maximum absolute change in hot-area RMSE is 0.0095 K across both
arms, much smaller than the approximately 3.9--4.1 K on/off differences.
No peak-error sign or coverage conclusion changes with the second seed.

Seven of eight new source-off fits pass every gate. Spiral 320 K has maximum
rank-normalized split R-hat 1.01070085, exceeding 1.01 at noisy constrained node
1906; its bulk/tail ESS there are 781.5/1080.6. This row is explicitly
provisional, not rounded into passing. Other fits have maximum rank-R-hat <=
1.007431. No gates were relaxed and no statistical retries were performed.
No further inference expansion is made from the provisional row.

New source-off fit time totals 138.18 s (8.68--31.33 s per fit); six exact
source-on replays take 55.10 s. The successful complete run, including loading,
cache writes, metrics, plotting and verification, takes 277.43 s (4.62 min).
Peak source-off RSS is 736.09 MiB. Cached source-on fit times in the CSV are
historical and are not included as new computation.

**Interpretation:** in this frozen model and these two development cases, the
explicit increment supplies temperature magnitude that current censored
observations cannot replace, particularly at the low reference ceiling. Its
removal exchanges overprediction for underprediction; this does not validate
its physical interpretation, shape or amplitude. Keep the historical baseline
unchanged and return to matching-generator/source-provenance clarification,
not an automatic gamma fit, covariance change or new source model.

## Figures

- [Spiral reference cross-sections](cross_sections_SpiralScanPath_13_level_0.png)
- [Diagonal 325 K cross-sections](cross_sections_DiagonalScanPath_8_level_2.png)
- [Spiral 325 K signed errors](signed_errors_SpiralScanPath_13_level_2.png)
- Every completed ceiling also has matching cross-section and shared-scale
  forecast/posterior signed-error maps in this directory. Spiral 320 K figures
  must be presented with the provisional convergence qualification above.

## Question and control

Does removing the explicit current gradient-based increment improve the final
native C reconstruction when the latest previous-frame representation is held
fixed? This is a controlled ablation, not a replacement baseline or a physical
source model.

The source-on control is Experiment 61's `all_readings_latent` C branch, not its
historical hybrid or pinned-conditioning bridge. We compare DiagonalScanPath_8
and SpiralScanPath_13 at reference, 320 K and 325 K. The reference source-off
reliability gate must pass before higher-ceiling source-off fits run.

With the cached propagated previous-temperature excess p and current numerical
increment s, the two means are `m_on = ambient + p + s` and
`m_off = ambient + p`. The latter is independently checked against the frozen
positive-excess-floor, diffusion/cooling, boundary and interpolation operator.
Both prediction and observation-location means change consistently.

The full predictive covariance `Q_dt + H H^T`, previous posterior, current
retained observation IDs and readings, lower-censoring factors, physical
coefficients and numerical environment are identical within each comparison.
Current likelihood noise SD is 0.50 K. There are no extra non-saturation factors,
new camera realizations, coefficient fits or previous-state inference runs.

`HeatFluxZ` remains a projected vertical temperature gradient; its thresholded
empirical increment is not relabeled as prescribed laser heating. Historical
heat already present in the previous state remains. This ablation does not
remove all physics or make the complete pipeline gradient-independent.

## Inference and evaluation

Each current fit uses four chains, 300 burn-in steps and 800 retained draws per
chain, thinning one. Pointwise summaries use 300 balanced evaluation draws per
chain (1,200 total). Reference fits have two paired current seeds; higher ceilings
have one. All six primary source-on controls are replayed and compared exactly
with saved Experiment 61 predictions and sampled chains. The second reference
source-on seeds are reused from fingerprinted caches.

Unchanged gates: rank-normalized split R-hat <= 1.01; minimum bulk/tail ESS >=
400; classical R-hat <= 1.05; fit runtime <= 900 seconds; peak RSS <= 4,096 MiB.
Diagnostics monitor all constrained noisy signals and latent temperatures at
current observations plus 64 spatially spread nodes; classical R-hat also
screens the field. No statistical retries or relaxed gates are planned.

Regions and native area weights are inherited from Experiment 61, including
fixed hot 1% physical area and frozen along/across shoulder section nodes.
The latter are section-node diagnostics, not independently sampled observations.
Forecast and posterior errors are reported separately. Coverage and intervals
are pointwise. Final prediction draws preserve marginals for scoring, not joint
field coherence; conditioning still uses full covariance blocks. The inherited
CRPS convention is retained; MCMC draws are not independent replicates.

## Files

- [Frozen configuration](frozen_configuration.json)
- [Input/covariance equality checks](equality_checks.csv)
- [Deterministic forecast metrics](forecast_metrics.csv)
- [Posterior metrics](posterior_metrics.csv)
- [Paired off-minus-on changes](paired_changes.csv)
- [Peak recovery by the current update](peak_recovery.csv)
- [Convergence and runtime](convergence_runtime.csv)
- [Verification](verification.json)
- [Verification notes and tests](verification.md)
- [Source/input fingerprints](source_and_input_fingerprints.json)
- Driver: [67_thermal_native_source_on_off.py](../../../scripts/67_thermal_native_source_on_off.py)
- Mean-only helper: [source_off_audit.py](../../../src/source_off_audit.py)

The output number is 63; the driver number is 67. No existing baseline or
historical result was replaced.
