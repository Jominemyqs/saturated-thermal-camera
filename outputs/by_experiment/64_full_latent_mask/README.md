# Experiment 64: does the mask benefit survive full-latent previous inference?

## Completed result

**Yes, on these two development cases the mask benefit survives, but is smaller
than under the older hybrid predecessor.** It improves unused-region RMSE and
CRPS beyond the added locations, with slightly better coverage rather than
merely narrower intervals. It does not resolve the extreme peak, and hot-area
effects remain trajectory-dependent. No universal or held-out benefit is claimed.

Primary paired-seed results on the fixed unused pool; errors, CRPS and widths
are in K, and coverage is a fraction:

| Case | Arm | RMSE | Bias | CRPS | Coverage95 | Width95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal | C0 | 0.2256 | 0.0871 | 0.1945 | 0.9873 | 2.7728 |
| Diagonal | Cmask | 0.1911 | 0.0686 | 0.1870 | 0.9902 | 2.7456 |
| Diagonal | Cnumeric | 0.2002 | 0.0580 | 0.1610 | 0.9896 | 2.1907 |
| Spiral | C0 | 0.2832 | 0.1260 | 0.2109 | 0.9711 | 2.7499 |
| Spiral | Cmask | 0.2196 | 0.0912 | 0.1949 | 0.9792 | 2.7120 |
| Spiral | Cnumeric | 0.2240 | 0.0633 | 0.1690 | 0.9797 | 2.1742 |

| Case | Old hybrid mask RMSE reduction (output 55) | Full-latent mask RMSE reduction |
| --- | ---: | ---: |
| Diagonal | 27.5% | 15.3% |
| Spiral | 29.7% | 22.5% |

These are within-baseline paired effects, not a claim that all differences
between outputs 55 and 64 are caused by unpinning alone. The newer previous
representation also conditions on all previous-frame readings. Better previous
inference reduces some, but not all, of the error that the mask can address.

The effect extends beyond U. On the SAME remaining-unused nodes, C0 -> Cmask:

| Case | RMSE | CRPS | Coverage95 | Width95 |
| --- | --- | --- | --- | --- |
| Diagonal | 0.2522 -> 0.2077 | 0.2011 -> 0.1905 | 0.9818 -> 0.9858 | 2.7557 -> 2.7221 |
| Spiral | 0.3259 -> 0.2451 | 0.2247 -> 0.2016 | 0.9580 -> 0.9698 | 2.7213 -> 2.6677 |

Numerical readings have better unused-pool CRPS and lower signed bias in both
primary comparisons, although mask RMSE is slightly lower. Outside U, numerical
CRPS is also better (0.1775 and 0.1897 K). Numerical readings are not an oracle
or a guaranteed realized-error bound; the numerical arm has only one seed here.
Neither these cases nor the mask's high coverage establishes exact calibration.

## Tail tradeoff and repeat stability

| Case | Hot-area RMSE, C0 -> Cmask | Hot-area CRPS, C0 -> Cmask | Coverage95, C0 -> Cmask |
| --- | --- | --- | --- |
| Diagonal | 3.6157 -> 3.6310 | 1.7820 -> 1.8085 | 0.7969 -> 0.7899 |
| Spiral | 3.4428 -> 3.4195 | 1.6702 -> 1.6278 | 0.6823 -> 0.6823 |

The small Diagonal hot-area deterioration and Spiral point/CRPS improvement
repeat at the second seed. No true-peak 95% interval covers truth in any of the
ten fits. Primary mask peak signed errors are +11.181 K and +21.545 K versus
+11.249 K and +21.490 K for C0. These are not a peak-physics solution.

Unused-pool mask-minus-baseline RMSE changes at the two seeds are
-0.03449/-0.03527 K (Diagonal) and -0.06362/-0.06420 K (Spiral). The paired
effect spread is below 0.0008 K. CRPS changes are -0.00748/-0.00740 K and
-0.01605/-0.01639 K. Remaining-unused improvements also repeat, well beyond
the observed seed variation. This supports numerical stability, not proof of
exact convergence or generalization from two trajectories.

## Reliability and cost

All ten fits pass the prospectively frozen gates, and all four C0 controls
reproduce output 61 arrays and monitored chains exactly. Maximum rank-normalized
split R-hat is 1.00615; minimum bulk/tail ESS are 1,060.51/1,537.20; maximum
classical R-hat is 1.00539. No retries or threshold changes were used.

| Case | C0 fit seconds (two seeds) | Cmask fit seconds (two seeds) | Cnumeric fit seconds |
| --- | --- | --- | ---: |
| Diagonal | 35.2 / 33.9 | 128.1 / 120.3 | 60.3 |
| Spiral | 57.1 / 60.2 | 266.9 / 279.7 | 84.2 |

Mask-only information is NOT computationally cheaper in this implementation.
The ten fits total 1,125.78 s (18.76 min); full run wall time is 1,389.02 s
(23.15 min), including preparation, compression, plots and verification.
Preparation took 77.45 s with peak RSS 1,402.94 MiB; maximum fit-worker RSS
was 776.55 MiB. These are process high-water marks, not isolated matrix sizes.
The historical output-55 300-second stop remains unchanged.

**Decision:** retain this as evidence of complementary current-mask information
under improved previous inference. The next substantive validation should use
a small prospectively selected held-out set and a bounded less-severe ceiling,
with the same 256-location budget. No such expansion was launched here, and no
larger-budget experiment or global baseline replacement is implied.

Figures: [Diagonal unused-region maps](background_DiagonalScanPath_8.png),
[Spiral unused-region maps](background_SpiralScanPath_13.png),
[Diagonal cross-sections](cross_sections_DiagonalScanPath_8.png),
[Spiral cross-sections](cross_sections_SpiralScanPath_13.png).

## Frozen question

This reference-ceiling-only experiment follows output 55 without treating its
hybrid-predecessor mask gain as universal. The control is source-on native C
with output 61's `all_readings_latent` previous representation. Output 63's
source-off ablation is NOT used. No coefficient, source, Q, noise or flag-budget
change is combined with this test.

Two development cases: DiagonalScanPath_8 (296.121498 K) and SpiralScanPath_13
(296.490128 K). Every arm retains output 53/61's current observation policy:
fixed numerical candidates UNION observed reference-saturated locations, each
contributing once. All 256 extra IDs and their selection files are copied
unchanged from output 54; no truth, error or posterior feature selects them.

| Arm | Additional information at the same U |
| --- | --- |
| C0 | None |
| Cmask | Noisy non-saturation probability `Phi((c-T)/0.50)` only |
| Cnumeric | Actual noisy readings; no additional flag factors |

The mask layout cannot accept omitted numerical values. Upper bounds apply to
the latent NOISY camera signal, not a hard bound on temperature. Correlated
mixed-bound conditioning and the verified sign transform use the unchanged
`native_mask_pilot` sampler. There is no importance reweighting, clipping,
fabricated temperature at c, or area weighting in the likelihood.

## Full-latent covariance, not old hybrid features

Write `P=(K_previous+v I)^-1`, `v=0.25^2`,
`C_cond=v I-v^2 P`, `W=C_cond[:,S]/v`. The implemented inherited covariance is

`A [W Cov(Z_S) W^T + C_cond] A^T`.

It includes the full correlated conditional residual analytically, including
unsaturated-censored cross terms. The current prior adds the SAME frozen
`Q_dt`. New columns at U use `propagated_blocks(..., pinned=False)` and the
cached previous noisy constrained chains, never the old mask driver's hybrid
`H H^T` construction. No previous-state inference is rerun.

Prediction means and diagonals are reused exactly from output 61. Original
observation and cross-covariance blocks are retained exactly; independently
computed overlaps and added-node diagonals must agree within 1e-10 absolute
and relative tolerance before cached-value roundoff reconciliation. The actual
differences are saved in [prior verification](prior_verification.csv). A unit
test rejects substituting pinned covariance in these added blocks.

The mean still uses the inherited positive-excess floor and approximate
native/grid propagation. Its source-on gradient contribution remains unchanged.
This is not interpolation-free 3D heat propagation or a new physical model.

## Sampling, gate and scope

Four chains, 300 burn-in, 800 retained samples per chain, thinning one; pointwise
summaries use 300 balanced draws per chain (1,200 total). C0/Cmask use two paired
current seeds; Cnumeric uses the primary seed. All four C0 fits replay saved
output 61 posterior arrays and sampled chains exactly before added-information
arms run. All arms monitor every constrained noisy signal, all baseline/U
latent locations, and 64 spatially spread latent locations; classical R-hat
also screens the field.

Prospective gates: maximum rank-normalized split R-hat <= 1.01, minimum bulk
and tail ESS >= 400, classical R-hat <= 1.05, fit time <= 900 s, peak worker
RSS <= 4,096 MiB. No statistical retry, chain shortening or relaxed threshold.
The 900-second budget does NOT revise output 55's historical 300-second stop.
Fit timers include sampling, diagnostics and metrics, but exclude final cache
compression. The parent process also enforces a 900-second worker timeout.

No held-out or high-ceiling posterior inference, extra flags, or automatic
generalization phase is included. Those require a separately frozen plan after
interpreting this comparison, even if the reference gate passes.

## Evaluation

The fixed unused pool, U and pool-minus-U are identical across arms, as are
native physical-area weights. Scores cover RMSE, MAE, bias, CRPS, coverage,
width and SD; hot 1% physical area, current censored locations, whole field and
the fixed true peak are separate checks. These are not all ambient-temperature
nodes. The 256 nodes account for about 27.3% of lumped surface area; the full
unused pool accounts for about 84.2%. Node counts alone are misleading on this
adaptive mesh. Weights are used for evaluation, not the likelihood.

In the CSV, inherited `peak_absolute_error_K` is the absolute difference between
the predicted and true field maxima, which can occur at different nodes.
`true_peak_signed_error_K`, `true_peak_covered`, and the `true_peak_location`
region explicitly evaluate the fixed true peak node instead.

Final field samples retain the inherited marginal-only representation; they
are appropriate for pointwise scoring, not downstream joint-field propagation.
CRPS retains the existing M(M-1) convention for comparability; its formal
unbiasedness assumes independent draws, whereas these are MCMC samples.
Two seeds test stability but do not establish physical calibration or provide
independent spatial replicates.

## Cached compatibility diagnostic

Using output 61 baseline marginal draws at the SAME U, reference mean flag
compatibilities are 0.94633 (Diagonal) and 0.90741 (Spiral); minima are 0.00177
and 0.00202. Thus some flags still conflict strongly with the improved baseline.
At 320/325 K all estimates round to 1 in double precision. This is a finite-draw
redundancy diagnostic, not proof of zero effect or a high-ceiling posterior
comparison. These probabilities never choose U or tune the model.

## Artifacts

- [Frozen configuration](frozen_configuration.json)
- [Preflight dimensions and core-array estimates](preflight.csv)
- [Full-prior equality checks](prior_verification.csv)
- [Metrics](results.csv), [paired changes](paired_changes.csv),
  [repeat-seed changes](repeat_seed_comparisons.csv)
- [Old/new baseline effect comparison](historical_effect_comparison.csv)
- [Convergence and runtime](convergence_runtime.csv), [gate](reference_gate.json)
- [Cached flag compatibility](cached_flag_compatibility.csv)
- [Source/input fingerprints](source_and_input_fingerprints.json)
- [Verification](verification.json)
- [Verification notes and tests](verification.md)
- [Driver 68](../../../scripts/68_thermal_full_latent_mask.py),
  [isolated helper](../../../src/latent_mask_audit.py)

All historical inputs remain unchanged; source fingerprints were rechecked at
completion. No previous inference or physical calibration was repeated.
