# Bounded mask-reliability and ceiling follow-up

This is output **55**, run by [driver 59](../../../scripts/59_thermal_mask_reliability_check.py).
It follows [the 256-location pilot](../54_native_non_saturation_mask/README.md)
without overwriting its results, gate, source fingerprints, or code-reading ZIP.
It is a numerical reliability check followed, only if the gate passes, by the
same-location 320/325 K comparison. No flag-budget sweep, held-out expansion,
time integration, parameter fitting, or physics modification is authorized here.

## Completed outcome

**All ten longer-chain reference fits pass the statistical checks. The full
expansion gate fails only the runtime criterion, so no 320/325 K posterior
updates were performed.** One spiral mask fit took 318.904 seconds, above the
unchanged 300-second limit; its paired repeat took 278.175 seconds. No threshold
was relaxed, no chain shortened, and no extra seed was tried to obtain a pass.

Across all ten fits, maximum rank-normalized split R-hat is **1.00616**, minimum
bulk ESS **1,027.21**, minimum tail ESS **1,380.73**, and maximum classical R-hat
**1.00481**. The three former statistical misses are resolved:

| Previously failing case | Short budget | Longer budget |
|---|---:|---:|
| Diagonal mask, primary seed: maximum rank R-hat | 1.01132 | 1.00501 |
| Spiral baseline, primary seed: maximum rank R-hat | 1.01119 | 1.00421 |
| Spiral numerical control: minimum bulk ESS | 345.76 | 1,027.21 |

The ten fits total **1,146.84 seconds (19.11 minutes)** of sampling plus
diagnostics. This excludes worker startup, input hashing, covariance preparation,
cache compression, and report generation. Mask fits took 149.3-318.9 seconds;
baseline fits 24.5-38.7 seconds; numerical controls 44.4-68.9 seconds. Maximum
worker high-water RSS was 2,141.94 MiB (2.09 GiB). Fresh-worker RSS is not directly
comparable to the previous persistent process's accumulated high-water mark.

## Longer-chain results

Primary-seed, area-weighted results on the fixed previously unused pool; all
temperature errors, CRPS and widths are in K, coverage is a fraction.

| Case | Arm | RMSE | Bias | CRPS | Coverage | 95% width |
|---|---|---:|---:|---:|---:|---:|
| Diagonal 8 | C0 | 0.2882 | 0.0905 | 0.1935 | 0.9900 | 2.6937 |
| Diagonal 8 | Cmask | 0.2088 | 0.0665 | 0.1832 | 0.9917 | 2.6600 |
| Diagonal 8 | Cnumeric | 0.2156 | 0.0559 | 0.1582 | 0.9914 | 2.1271 |
| Spiral 13 | C0 | 0.3549 | 0.1282 | 0.2101 | 0.9773 | 2.6829 |
| Spiral 13 | Cmask | 0.2495 | 0.0884 | 0.1910 | 0.9836 | 2.6364 |
| Spiral 13 | Cnumeric | 0.2578 | 0.0650 | 0.1679 | 0.9833 | 2.1191 |

C0 retains the numerical/reference-saturation union with no added information.
Cmask adds only 256 noisy non-saturation flags; Cnumeric supplies readings at
exactly those locations. All share the same within-case predictive prior.
Numerical readings remain an information-content control, not an oracle bound.

Both paired seeds retain the mask's unused-region gains. Diagonal RMSE changes
are -0.07933/-0.08042 K and CRPS changes -0.01033/-0.01014 K; spiral changes are
-0.10541/-0.10566 K and -0.01908/-0.01943 K. Effects also extend beyond U: primary
remaining-unused RMSE/CRPS change 0.3279/0.2017 -> 0.2303/0.1874 (diagonal), and
0.4120/0.2256 -> 0.2842/0.1987 (spiral).

Fixed hottest-1%-area checks remain a tradeoff, not a uniform improvement:

| Case | Arm | RMSE | CRPS | Coverage | 95% width |
|---|---|---:|---:|---:|---:|
| Diagonal 8 | C0 | 4.4175 | 2.2568 | 0.7726 | 6.3496 |
| Diagonal 8 | Cmask | 4.5047 | 2.3420 | 0.7448 | 5.8589 |
| Diagonal 8 | Cnumeric | 4.5166 | 2.3612 | 0.7309 | 5.8897 |
| Spiral 13 | C0 | 4.0770 | 2.1507 | 0.7552 | 6.4024 |
| Spiral 13 | Cmask | 3.9451 | 1.9190 | 0.7830 | 5.7717 |
| Spiral 13 | Cnumeric | 3.9430 | 1.9227 | 0.7656 | 5.7422 |

The diagonal hot-area deterioration and spiral improvement persist across the
two longer-chain seeds. Spiral hot-area coverage now improves in both repeats,
by +0.02778/+0.02257; two repeats still do not establish a formal uncertainty
bound. Neither method fixes the extreme peak: all primary-seed true-peak intervals
miss truth. Mask true-peak signed error is 10.588 K (diagonal) and 23.079 K
(spiral), versus 10.487 and 22.899 K for C0. Small peak interval-width differences
are not stable relative to repeat variation.

Compared with the short-chain pilot, the largest absolute unused-pool change
over matched arms/repeats is only **0.00200 K RMSE** and **0.000434 K CRPS**.
Hot-area changes are at most 0.01124 K RMSE and 0.01320 K CRPS; coverage changes
can reach 0.02431 and should not be hidden by rounding. The main point/CRPS
findings survive stronger sampling. These are stability observations, not proof
of exact convergence or physical calibration.

See [all metrics](results.csv), [paired changes](paired_comparisons.csv),
[paired repeats](repeat_seed_comparisons.csv),
[short-versus-long metrics](short_vs_long_metrics.csv),
[diagnostics/runtime](reference_convergence_runtime.csv), and the
[unchanged-criteria gate decision](reference_gate.json).

Plots: [diagonal background error/SD](background_DiagonalScanPath_8_level_0.png),
[spiral background error/SD](background_SpiralScanPath_13_level_0.png),
[diagonal native cross-sections](cross_sections_DiagonalScanPath_8_level_0.png),
[spiral native cross-sections](cross_sections_SpiralScanPath_13_level_0.png).
See [verification](verification.md) for preservation and equality checks.

## Frozen design

The only initial change is the current sampler budget: four chains with **800
retained states after 300 burn-in**, thinning 1, using the original initialization
and paired seeds. This is the existing historical retry-sized budget, applied
once to every arm instead of repeatedly extending chains until a gate passes.
There is no additional retry. Evaluation still uses 300 balanced states per
chain, 1,200 total, and the same CRPS convention and native area weights.

All ten reference fits are rerun: C0 and Cmask at both paired seeds, Cnumeric at
the primary seed, for Diagonal 8 and Spiral 13. The original 256 IDs are copied
byte-for-byte, not reselected. The previous hybrid posterior, numerical gradient
contribution, native propagation, Q, total prior, camera noise (0.25 K), current
likelihood noise (0.50 K), numerical jitter, and retained observations are fixed.
The posterior itself may change because inference changes.

The unchanged expansion criteria require **every** reference arm/repeat to have:

- maximum rank-normalized split R-hat <= 1.01;
- minimum bulk ESS and tail ESS >= 400;
- classical latent/noisy-signal R-hat <= 1.05;
- fit-plus-diagnostics time <= 300 seconds;
- process peak RSS <= 4,096 MiB.

The longer-chain check is bounded even if these criteria fail. The driver does
not silently remove the runtime or memory criteria when increasing chain length.
Each fit runs in a fresh process; RSS is that worker's high-water mark, including
input/preparation arrays, not an isolated covariance allocation. Inference is
serial with the same single-thread BLAS settings as the pilot.

If reference reliability passes, higher ceilings use all three arms at the same
256 IDs and primary paired seed. Within each ceiling the cached previous draws,
physics mean and total prior remain identical across arms; **the prior is not
claimed identical across ceilings**. Retained baseline IDs are checked against
the actual output-53 cache. Longer-chain posterior draws are not expected to
hash-match short-chain posterior draws; the exact short-chain replay is already
documented in output 54, and prior/input hashes remain the relevant equality test.

## Cached diagnostic

[Cached flag probabilities](cached_baseline_flag_compatibility.csv) use only the
historical baseline latent marginal draws to average
`Phi((c - T_draw) / 0.50)` at the fixed U. This calculation neither chooses nodes
nor modifies inference. At reference, the mean probabilities are 0.9458
(diagonal) and 0.9120 (spiral), with a small subset of strongly conflicting flags.
At 320/325 K all 256 estimated probabilities round to 1.0 in double precision
for both cases. This supports a redundancy hypothesis but is **not** proof of
zero effect or a new high-ceiling comparison: the estimate uses finitely many
historical MCMC marginal draws and cannot resolve unvisited posterior tails.

## Reproduction and preservation

[Frozen settings](frozen_configuration.json) record the longer budget and the
unchanged gates. [Source fingerprints](source_fingerprints.json) include the
unchanged driver-58 dependency and the pilot's input/source manifests. Driver 59
is explicitly a bounded audit harness: it loads driver 58's existing preparation,
covariance, metric and plotting routines rather than editing or replacing that
historical driver. The actual mixed-bound likelihood and reflected-HMC helper
are unchanged.

```sh
env PYTHONPATH=/tmp/thermal-mask-diagnostics MPLCONFIGDIR=/tmp/thermal_mpl \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPYCACHEPREFIX=/tmp/thermal_pycache \
  .venv/bin/python scripts/59_thermal_mask_reliability_check.py --stage all
```

The diagnostic dependency versions and their temporary installation are those
documented in output 54. `--stage report` reads caches without inference;
`--stage secondary` refuses to run after a failed reference gate. Completed fits
are not overwritten. Stage `prepare` only checks inputs, freezes settings,
copies selection IDs, and computes cached-baseline compatibility.

All existing limitations remain: hybrid pinned unsaturated previous values,
marginal rather than fully coherent field draws, approximate native-input
propagation, and MCMC rather than independent draws in the `M(M-1)` CRPS
estimator. Repeat seeds do not prove convergence or physical calibration.
