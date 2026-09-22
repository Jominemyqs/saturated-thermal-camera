# Where does the severity-dependent physics-forecast bias enter?

## Answer

The severity trend enters through the inferred previous state, but the frozen
physics forecast is not intrinsically unbiased. Its oracle-state hot-tail bias
is **+2.600 K**. A negative propagated previous-state error partially cancels
that positive error. With more censoring, the negative contribution grows,
and the baseline forecast becomes increasingly cold.

This is error cancellation, not evidence that perfect previous-state information
necessarily improves this fixed approximate predictor. No parameters were refit
to accommodate the oracle input. The oracle is an attribution control, not a
guaranteed performance upper bound or a newly optimized physical model.

## Controlled experiment

Driver: scripts/50_thermal_oracle_forecast_decomposition.py.
Inputs: frozen Experiment 45 caches and Experiment 43 physical configuration.
33 trajectories, unchanged 3 development / 30 held-out split, six ceilings.
No GP inference, covariance change, parameter learning, advection experiment,
calibration or remedy was performed.

For the exact existing operator F, including ambient floor, diffusion filter,
constant-zero excess boundary and thresholded source:

    e_previous = previous_estimate - previous_truth
    e_state    = F(previous_estimate, q) - F(previous_truth, q)
    e_physics  = F(previous_truth, q) - current_truth
    e_total    = e_state + e_physics

Because the ambient floor is nonlinear, e_state is evaluated by two full
forecasts, not by assuming F is linear on the raw previous error.

## Fixed current hottest 25 pixels

Equal trajectory-weighted averages over 30 held-out paths. Bias in K means
estimate minus truth. Reference censoring is approximately 3.2%.

| Censoring | Raw previous bias | Propagated state bias | Oracle physics bias | Total forecast bias | State/previous RMSE ratio |
|---|---:|---:|---:|---:|---:|
| Reference | -2.746 | -2.728 | +2.600 | -0.129 | 0.9933 |
| 5% | -3.078 | -3.058 | +2.600 | -0.459 | 0.9933 |
| 7.5% | -3.364 | -3.343 | +2.600 | -0.743 | 0.9933 |
| 10% | -3.610 | -3.587 | +2.600 | -0.987 | 0.9934 |
| 15% | -4.037 | -4.012 | +2.600 | -1.412 | 0.9934 |
| 20% | -4.175 | -4.148 | +2.600 | -1.549 | 0.9934 |

- Previous bias decreases on all 30 paths between reference and 20%.
- Oracle physics bias is positive on all 30 paths, and its entire field is
  bitwise identical across ceilings for each trajectory.
- The total bias shift of -1.420 K equals the propagated-state bias shift.
  This attribution is an algebraic consequence of holding F, truth and q fixed,
  not an independently estimated causal percentage.
- The local RMSE ratio is below one in all 180 held-out cases, approximately
  0.9933-0.9934: very slight attenuation, not error amplification. Whole-field
  average ratios range from 0.9794 to 0.9886. These matched-subset ratios are
  descriptive, not operator norms; diffusion can move error across a subset.

RMSEs are not additive:

| Censoring | Raw previous RMSE | Propagated state RMSE | Oracle physics RMSE | Baseline RMSE |
|---|---:|---:|---:|---:|
| Reference | 8.383 | 8.327 | 6.721 | 3.169 |
| 20% | 9.045 | 8.985 | 6.721 | 3.774 |

The saved MSE decomposition explicitly includes

    MSE(total) = MSE(state) + MSE(physics) + 2 mean(e_state * e_physics).

The cross term is negative in all 180 held-out current-top25 cases, confirming
spatial error cancellation as well as cancellation of average bias. Do not
square the table's average RMSEs to reconstruct mean MSE: averaging and square
root do not commute. Per-trajectory MSE components are in amplification.csv.

## Family check

| Family | Oracle bias | State bias, reference | State bias, 20% |
|---|---:|---:|---:|
| Diagonal (7 held-out) | +2.862 | -3.058 | -4.289 |
| Horizontal (11) | +2.157 | -2.452 | -3.848 |
| Spiral (12) | +2.851 | -2.789 | -4.341 |

The opposite-sign components occur in every family. This is not just a
family-mixture artifact.

## Previous-state status diagnostic

At the current hottest 25 locations, **99.1% were already observed-censored in
the previous reference frame**; the fraction reaches 100% by 10% censoring.
The mean contribution from previous unsaturated pixels to raw previous bias
is only -0.0009 K at reference and zero once all these locations are censored.

Therefore the main top-tail mechanism is not simply that many formerly pinned
hot pixels switch status. Most are already censored. Their inferred magnitude
continues to decline as the ceiling and surrounding available information
change. This experiment does not separate the weaker inequality from the loss
of informative surrounding observations.

The cached representation is hybrid: saturated locations use sampled censored
inference; unsaturated locations are pinned to noisy observations, up to about
1.3e-11 K roundoff from averaging repeated identical draws. It is not a full
latent posterior over every pixel.

previous_status.csv records conditional bias/RMSE/MAE, counts, fractions and
each status's contribution to the parent-region bias. Empty strata have NaN
conditional metrics and zero bias contribution, not fabricated zero error.
Conditional summaries omit empty strata; use counts and bias contributions
when comparing statuses across ceilings.

## Interpretation for the next meeting

Both the previous-state estimate and the frozen physics operator have important
errors, with opposite signs. Improving only previous-state reconstruction does
not guarantee that the existing forecast improves, because it can remove a
compensating error. The results do not identify whether the positive oracle
error comes from source conversion, timing, cooling, spatial discretization,
boundaries or inherited calibration. Those remain hypotheses, not conclusions.

The clean conclusion is: increasing censoring makes previous-state inference
colder; the frozen operator transmits almost all of that change; its own positive
oracle-state error remains fixed. No remedy was implemented.

## Verification and provenance

- All 198 cached physics-prior fields reproduced bitwise and matched their
  saved Experiment 45 hashes. Recomputed baseline metric discrepancy at most
  4.44e-16 K (CSV roundoff).
- Pixelwise e_total - e_state - e_physics maximum absolute residual: 0.0 K.
- Oracle forecast re-evaluated and asserted identical at every ceiling.
- Exact current top25 masks and current truth matched Experiment 45.
- Source fingerprints from Experiment 45 checked before execution.
- Physical coefficients loaded unchanged; trajectory preparation's diagnostic
  fits are not used to select coefficients. Historical preparation/ambient
  provenance remains inherited and is not relabeled as camera-only calibration.
- Truth enters only the requested oracle and evaluation; no new inference is run.

The oracle field and fixed-mask metrics must be ceiling-independent. Metrics on
the changing current-observed-censored region need not be horizontal, because
the evaluation population changes. No such metric change is attributed to F.

## Outputs

- oracle_decomposition.png: main four-panel figure.
- results.csv: all four errors, four regions, bias/RMSE/MAE and counts.
- amplification.csv: per-case ratios and exact MSE/cross-term decomposition.
- previous_status.csv: observed-status split of raw previous-state error.
- *_heldout30_summary.csv, *_family_summary.csv: equal-trajectory summaries.
- paired_changes.csv: within-trajectory reference-to-ceiling changes.
- by_trajectory/: baseline/oracle fields, errors, truth and masks for attribution.
- audits.csv, verification.json, provenance.json, source_fingerprints.json.

Run with .venv/bin/python scripts/50_thermal_oracle_forecast_decomposition.py.
This only replays inexpensive deterministic forecasts, not MCMC.
