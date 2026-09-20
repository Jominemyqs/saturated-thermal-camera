# Censoring-severity error decomposition

## What fails first?

The clearest severity-dependent change is already present in the deterministic
posterior physics mean: its fixed hottest-25 bias becomes increasingly negative.
It decreases from reference to 20% censoring on all 30 held-out trajectories.
The C update offsets that bias by approximately +0.56 to +0.60 K at every level;
it does not introduce the downward trend. The posterior bias crosses zero between
5% and 7.5%, but that is a descriptive sign change, not an established failure
threshold.

There is also a hottest-tail calibration mismatch at the reference ceiling.
It does not newly appear at high censoring: C's within-trajectory top-25 SD(z)
decreases slightly, and its intervals widen rather than collapse. Thus the
evidence supports **a growing negative forecast bias superimposed on an existing
tail calibration mismatch**, not a new variance collapse or demonstrated loss
of posterior skewness.

No model or parameter was changed. B = posterior physics mean + spatial RBF;
C = the same mean + sequential stationary ST covariance. The final update uses
current observations only. These are diagnostics of frozen Experiment 43,
not another ranking or tuning experiment.

## Main result: fixed hottest 25 pixels

Equal-weight averages over 30 held-out trajectories; errors are prediction minus
truth in K. The labels below are target censoring fractions; the figure uses
actual observed fractions (reference approximately 3.2%).

| Censoring | Physics-prior bias | B posterior bias | C posterior bias | C update to prior bias | C RMSE | C 95% coverage | C 95% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | -0.129 | +1.881 | +0.432 | +0.561 | 3.195 | 84.5% | 7.368 |
| 5% | -0.459 | +1.542 | +0.138 | +0.597 | 3.243 | 84.0% | 7.548 |
| 7.5% | -0.743 | +1.266 | -0.144 | +0.599 | 3.308 | 83.7% | 7.724 |
| 10% | -0.987 | +0.897 | -0.416 | +0.571 | 3.397 | 82.8% | 7.810 |
| 15% | -1.412 | +0.359 | -0.845 | +0.567 | 3.538 | 80.8% | 8.002 |
| 20% | -1.549 | +0.233 | -0.986 | +0.563 | 3.597 | 81.2% | 8.005 |

- The physics-prior top-25 RMSE rises from 3.169 to 3.774 K. All three families
  show a downward prior-bias shift; this is not only a pooled-family effect.
- B's upward update is larger: +2.009 K at reference and +1.782 K at 20%.
  Its positive bias diminishes as the prior becomes colder. This explains why
  B's hot-tail performance need not worsen monotonically with censoring.
- C's top-25 SD(z) is 1.657 at reference and 1.585 at 20%, while mean(z) moves
  from -0.266 to +0.479. The dominant new trend is centering, not growing SD(z).
- The fraction of truths above C's posterior 95th percentile rises from 12.3%
  to 18.9%; the below-5th-percentile fraction falls from 6.0% to 4.3%.
- C's average marginal posterior skewness stays near 0.22-0.23; mean-minus-median
  stays near 0.07 K. These summaries do not show a major new shape transition.
  They do not prove that the tails are correctly specified.

The whole field tells a different story from the fixed hot tail: C's field bias
increases from +0.143 to +0.862 K, while top-25 bias becomes negative. A single
global offset would not address both. Whole-field coverage decreases from 98.7%
to 91.3%; raw coverage should always be read with width and location-specific bias.

## Directional supplement

Coordinates use only the source centroids from HeatFluxZ in the last two frames.
Positive along-coordinate is ahead of source motion. Every plotted bin contains
all 30 held-out trajectories; per-bin pixel counts are saved.

Mean physics-prior bias contrast, ahead minus behind (K):

| Family | Held-out paths | Reference | 10% | 20% |
|---|---:|---:|---:|---:|
| Diagonal | 7 | +0.484 | +0.713 | +1.136 |
| Horizontal | 11 | -0.329 | -0.857 | -1.235 |
| Spiral | 12 | +0.979 | +1.383 | +2.609 |

At 20%, the contrast is positive on all 7 diagonal and 12 spiral paths, but
negative on all 11 horizontal paths. Therefore the pooled directional plot
must not be interpreted as a universal advection sign error or a universal
ahead/behind correction. Domain geometry, source discretization and differing
thermal histories remain possible contributors. These are observational
contrasts, not causal identification of a missing physical term.

## Interpretation limits and next discussion

This experiment localizes the growing hot-tail bias to the forecast *before*
the current update. It does not isolate whether that forecast bias originates
in the previous censored estimate, approximate diffusion/source physics, or
their interaction. That is the focused physics question to bring to Adrienne.

Neither SD(z)>1 nor extreme posterior quantiles uniquely identifies an incorrect
covariance or a missing tail. The hottest-25 subset is truth-selected, censored
posteriors are non-Gaussian, and pixels are correlated. All summaries are
descriptive, not IID normality/uniformity tests. No new model is justified solely
by a sign crossing or a pooled directional curve.

## Reproducibility and verification

Experiment 43 did not retain posterior draws, so only B and C were replayed.
The replay checked exact hashes for observations, previous posterior samples,
physics means, and covariance blocks, and all original metrics.

- 33 trajectories, unchanged 3 development / 30 held-out split; 198 ceiling cases.
- 396 model cases; 10,296 metric comparisons; maximum discrepancy **0.0**.
- Same paired observations, previous posterior, physics mean and sampler seed.
- All exact hottest-25 masks checked; censoring masks checked against camera Y=c.
- No invalid posterior SDs; saved diagnostic and source-coordinate arrays finite.
- Recorded maximum previous/current R-hat: 1.01050 / 1.01123. These reproduce the
  frozen protocol, not a new blanket guarantee of Monte Carlo convergence.
- Source fingerprints unchanged throughout replay. All 64 unit tests pass;
  source/scripts/tests compile and git diff --check passes.

The historical simulator-derived reference ceiling and preparation provenance
are inherited unchanged. Artificial ceiling masks use the observed camera;
truth is used only in this diagnostic evaluation, not to modify inference.
See PROTOCOL.md for the complete contract and limitations.

## Files

- severity_decomposition.png: main six-panel meeting figure.
- directional_physics_residual.png: source-relative supplemental figure.
- means.csv, posterior.csv: trajectory/ceiling/model/region diagnostics.
- *_heldout30_summary.csv, *_family_summary.csv: equal-trajectory summaries.
- *_paired_changes.csv: within-trajectory changes from the reference ceiling.
- paired_mean_update.csv, mean_update_summary.csv: GP update versus prior bias.
- directional_paired_contrasts.csv, directional_family_summary.csv: family caveat.
- audits.csv, checks.csv, verification.json: equality and integrity checks.
- by_trajectory/*/level_*.npz: float64 marginal draws and all diagnostic fields.
  These are not coherent joint spatial draws suitable for smoothing or maxima.

Reproduce: .venv/bin/python scripts/49_thermal_severity_error_decomposition.py
--workers 3. Completed cases resume only with matching source fingerprints.

## Active-workspace cleanup

Nested-ceiling response, source-history, and particle-smoothing branches were
relocated to scripts/archive/2026-09-20_deferred and
outputs/archive/2026-09-20_deferred. No unique results were deleted: 183 result
files remain byte-identical; five drivers have only archive-path adjustments.
The archive relocation manifest preserves original paths and hashes.

Scripts 45 (observation bridge) and 47 (matched ceiling comparison) remain
active because they establish the canonical protocol and central result.
Scripts 46 and 48 are archived. Shared implementation and tests remain wherever
needed by retained work. No new kernel, calibration, source-history or smoothing
branch was introduced.
