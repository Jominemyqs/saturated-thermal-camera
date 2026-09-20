# Matched ceiling-response architecture comparison

## Scope and architectures

Completed 33 trajectories x 6 ceilings x 3 architectures (594 predictions).
All headline tables average the 30 held-out trajectories equally; the three
development trajectories are excluded. No parameters were retuned.

- A: current-only ambient mean + censored RBF GP.
- B: posterior physics mean + censored RBF GP.
- C: posterior physics mean + sequential stationary stochastic ST covariance.

A and B use identical RBF covariance arrays. B and C use the exact same previous
posterior draws and deterministic physics mean. All three use the same current
observations (stride-5 unsaturated + every observed-censored pixel), likelihood
noise, seeds, and converged HMC protocol. C updates only with current observations.
No advection, smoothing, calibration, or new kernel was added.

See PROTOCOL.md for numerical settings, inherited approximations and provenance.

## Results

All dimensional errors and CRPS are in K. Field error is relative excess-field L2.
Coverage and width refer to the same fixed true hottest 1% (25 of 2501 pixels).

| Censoring | Model | Field error | Overall CRPS | Top-1% RMSE | Top-1% CRPS | Top-1% bias | Coverage | Width | Peak error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference, 3.20% | A | 0.738 | 0.762 | 6.924 | 2.970 | -2.113 | 83.2% | 7.854 | 25.912 |
| Reference, 3.20% | B | 0.745 | 0.789 | 3.698 | 2.097 | 1.881 | 92.0% | 10.410 | 7.068 |
| Reference, 3.20% | C | 0.516 | 0.241 | 3.195 | 1.738 | 0.432 | 84.5% | 7.368 | 6.633 |
| Severe, 20.03% | A | 1.691 | 1.156 | 7.578 | 3.448 | -3.588 | 79.7% | 8.563 | 27.671 |
| Severe, 20.03% | B | 2.125 | 1.425 | 3.530 | 1.871 | 0.233 | 89.1% | 10.733 | 7.058 |
| Severe, 20.03% | C | 1.343 | 0.661 | 3.597 | 1.896 | -0.986 | 81.2% | 8.005 | 6.903 |

| Approximate censoring | A top-1% CRPS | B top-1% CRPS | C top-1% CRPS |
| --- | ---: | ---: | ---: |
| Reference (3.20%) | 2.970 | 2.097 | 1.738 |
| 5% | 3.028 | 2.009 | 1.722 |
| 7.5% | 3.103 | 1.955 | 1.735 |
| 10% | 3.181 | 1.892 | 1.774 |
| 15% | 3.375 | 1.855 | 1.857 |
| 20% | 3.448 | 1.871 | 1.896 |

## Does the advantage grow with censoring?

Define physics gap A-B, sequential covariance gap B-C, and total gap A-C.
Positive gaps favor the second model for lower-is-better metrics. Growth below
is the paired 20%-minus-reference gap, not a claim of strict monotonicity.

| Comparison and metric | Reference gap | 20% gap | Mean growth | Trajectories with increasing gap |
| --- | ---: | ---: | ---: | ---: |
| A-B, top-1% CRPS | 0.873 | 1.576 | 0.703 | 30/30 |
| A-B, field error | -0.008 | -0.434 | -0.426 | 0/30 |
| B-C, overall CRPS | 0.548 | 0.764 | 0.216 | 30/30 |
| B-C, field error | 0.229 | 0.782 | 0.553 | 30/30 |
| B-C, top-1% CRPS | 0.358 | -0.025 | -0.383 | 0/30 |
| A-C, top-1% CRPS | 1.232 | 1.552 | 0.320 | 30/30 |
| A-C, field error | 0.221 | 0.349 | 0.127 | 27/30 |
| A-C, overall CRPS | 0.521 | 0.495 | -0.026 | 8/30 |

The increasing total hot-tail CRPS advantage holds in all families: diagonal
7/7, horizontal 11/11, spiral 12/12. Total field-error gap increases in 6/7,
11/11, and 10/12 respectively. The physics hot-tail gap grows in every family;
the sequential covariance hot-tail gap shrinks in every trajectory.

At 20%, C beats A on field error, overall CRPS and top-1% CRPS in all 30 cases.
C beats B on field error and overall CRPS in all 30 cases, but only 15/30 on
top-1% CRPS. Its mean hot-tail CRPS is slightly worse than B, despite a slightly
positive median paired gap. There is no universal hot-tail winner at this ceiling.

Paired trajectory bootstrap 95% intervals for gap growth are [0.587, 0.820] K
for A-B top-1% CRPS, [-0.464, -0.306] K for B-C top-1% CRPS, and
[0.277, 0.361] K for A-C top-1% CRPS. These are empirical trajectory-resampling
intervals, not guarantees for unseen physics or camera regimes.

## Interpretation

The posterior physics mean substantially recovers hot magnitude relative to
the ambient model. Its hot-tail benefit grows as censoring increases. However,
B's whole-field error and CRPS worsen relative to A. Its positive top-1% bias
also decreases as the ceiling falls; its improving hot-tail metrics should not
be interpreted as evidence that losing information is intrinsically beneficial.

C remains the strongest whole-field model among these three. Its covariance
contribution increasingly improves global performance over B, but the extra
hot-tail benefit disappears at severe censoring. B-C changes covariance scale
as well as structure; it does not isolate temporal correlation alone.

C's top-1% width expands from 7.37 to 8.00 K and coverage declines modestly
from 84.5% to 81.2%. These canonical results do NOT support the earlier claim
of collapsing widths with increased censoring. Coverage remains below 95%.

The observed-censored region changes with ceiling: its CRPS is not a fixed-region
severity comparison. Furthermore, although camera images are nested by reclipping,
the selected likelihood sets are not strictly nested: off-stride pixels enter
when they become censored. This is a controlled response of a fixed selection
rule, not a pure information-deletion experiment with identical locations.

## Existing calibration and smoothing findings

PSEUDO_CENSORING_RECAP.md records the earlier calibration and nested-response
tests. Global inflation improved some broad/censored-region scores but worsened
top-1% CRPS in 170/180 cases; feature inflation worsened it in 179/180. The
stronger nested-response development gate failed. Neither is adopted here.
The L=1 smoothing pilot remains paused after weight degeneracy; no longer-lag
or held-out smoothing runs were started.

## Audit and files

- All 33 trajectories loaded: 3 development and 30 held-out.
- All 180 held-out C predictions reproduce Experiment 38 across ten metrics;
  maximum absolute discrepancy across 1800 comparisons: 2.815e-12.
- All paired observation/seed hashes agree; A/B covariance and B/C previous
  draws and mean hashes agree for every trajectory and ceiling.
- Maximum previous R-hat: 1.010504; maximum current R-hat: 1.012807. No retries.
  These diagnostics passed the frozen criterion, not a proof of exact sampling.
- CRPS uses the distinct-pair M(M-1) estimator; its formal unbiasedness assumes
  independent draws, with MCMC dependence remaining a numerical consideration.

results.csv contains every prediction; heldout30_summary.csv and family_summary.csv
contain averages. paired_gaps.csv, gap_summary.csv and gap_growth.csv contain
paired changes, win counts and bootstrap intervals. implementation_checks.csv,
experiment38_equality_audit.csv, fixed_configuration.csv and source_fingerprints.json
record reproducibility. Per-trajectory JSON checkpoints retain completed cases.

Figures: performance_vs_censoring.png, model_gaps_vs_censoring.png,
coverage_and_width.png. These are quantitative summaries, not reconstructions.
