# Canonical experiment protocol

This checklist prevents architecture, observation, inference, and metric changes
from being mistaken for scientific model effects. Experiment 38's uncalibrated
stationary sequential model is the current forward baseline: stride-5
unsaturated plus all observed-censored pixels, with convergence-checked HMC.
Experiment 37 remains the historical sparse-observation reference, not an
interchangeable numerical baseline.

## Observation bridge and archived smoothing checks

Smoothing is deferred as of September 20. These checks remain provenance,
not instructions to start a new smoother. The active next step is script 49's
frozen censoring-severity decomposition; see ACTIVE_RESEARCH.md.

- Experiment 41 isolates the current observation set with one cached previous
  posterior/prior per trajectory. Sparse field error is 0.456412 versus 0.516370
  with all observed-censored pixels. Historical sparse error is 0.457218; the
  remaining protocol difference is separate from the mask intervention.
- Reproduce Experiment 38 with its exact recorded configuration; do not replace
  HMC settings with the older ESS counts merely because both passed diagnostics.
- Fast prediction draws preserve marginal uncertainty but omit conditional
  spatial cross-covariance. They are not complete joint field draws for a
  smoother. The hybrid previous posterior also pins unsaturated noisy pixels.
- Before an interior-frame smoother, audit joint sampling and time-available
  preparation. Explicitly specify whether future off-stride pixels that become
  unsaturated remain excluded or are retained under a new matched protocol.
- See `outputs/by_experiment/41_observation_bridge_and_smoothing_readiness/`
  for the frozen bridge, observed-mask cooling diagnostic, and sample audit.

## Before implementation

- Inspect the source drivers and reusable modules for every comparison row. Do
  not infer an architecture from a plot label or README alone.
- Write down the information flow for each model, including which observations
  construct the previous posterior and which observations enter the current
  likelihood.
- Classify every difference as one of: mean, covariance, observation set,
  posterior representation, sampler, or metric. A controlled comparison changes
  only the intended category.
- Freeze the 3 development and 30 held-out trajectories. Held-out trajectories
  must not affect physical parameters, hyperparameters, thresholds, or
  calibration choices.

## Frozen data and evaluation

- Reuse identical pre-clipping noise, camera observations, censoring masks, and
  random seeds within paired comparisons.
- Reuse an identical current observation set unless a row is explicitly marked
  as an observation-rich diagnostic and excluded from ranking.
- State whether the current likelihood receives only a sparse fixed-stride mask
  or also every observed-saturated pixel. Never compare these protocols as if
  they differed only by model architecture.
- Define hottest 1% by exact rank: 25 of 2501 pixels. Store the mask rule and
  count with every result.
- Compute empirical CRPS with the unbiased `M(M-1)` pairwise denominator.
- Report point and distribution metrics separately. Coverage must be accompanied
  by interval width.

## Previous-frame posterior

- Previous observations may enter a sequential model only through
  `p(T_(n-1) | Y_(n-1))`; do not reuse `Y_(n-1)` in the current likelihood.
- State the posterior representation exactly. The current implementation samples
  saturated pixels but pins unsaturated noisy pixels to their observations, so it
  is a hybrid censored posterior, not a complete latent posterior.
- Use at least the canonical sampler settings from Experiment 37: 4 chains, 2000
  retained draws per chain, 15000 burn-in, and thinning 10, unless a documented
  convergence study supports a cheaper setting.
- For large censoring blocks, Experiment 38's reflected exact-HMC sampler is the
  documented alternative: 4 chains, 400 retained draws, 100 burn-in, thinning
  1, and per-target R-hat checks. Do not transfer these shorter settings back to
  the older elliptical sampler.
- Require saturated-pixel R-hat <= 1.05 across every trajectory used in a final
  comparison. Record per-trajectory maxima and chain-mean disagreement.

## Current posterior and covariance

- Use matched chain counts, burn-in, retained draws, thinning, and seeds across
  directly compared Gaussian current posteriors.
- Require all-domain and hottest-1% R-hat <= 1.05 for canonical rows.
- For sequential models, report the total predictive covariance
  `Q + B Sigma B^T`, not `Q` alone.
- Verify covariance matrices are numerically identical when a comparison is
  intended to change only the mean.
- For posterior-sample mixtures, record component count, weight ESS, maximum
  weight, resampled unique components, and integration failures.

## Required output audit

Every canonical experiment should save:

- trajectory-level results;
- held-out and family summaries;
- paired mean/median differences and win counts;
- a fixed-configuration table;
- explicit model/information-flow definitions;
- implementation and convergence checks;
- a README distinguishing canonical rows from diagnostics.

Before reporting results, compile retained Python files, run the test suite, run
`git diff --check`, and compare any claimed reproduction against intermediate
objects rather than accepting merely similar final metrics.

## Current status of older results

- The source-history diagnostic in `outputs/by_experiment/40_source_history_mean`
  compares means under Experiment 38's full observed-censor-mask architecture at
  its reference ceiling. Its `BASELINE_CONTRACT.md`, saved metric equality audit,
  configuration table, and source fingerprints specify the reproduction target.
  The 1-step baseline is checked against all 30 saved Experiment 38 rows. The
  longer-history rows hold the complete one-step covariance fixed and must be
  described as mean-only sensitivities, not L-step Bayesian filters.

- Experiment 30 is historical: its short previous-frame chains underestimated
  both the previous posterior mean and variance in censored regions.
- Experiment 35 is internally controlled across ceilings, but its former
  4-chain, 700-draw, 1200-burn-in previous posterior does not satisfy the current
  convergence criterion. Its qualitative severity trends are diagnostic; its
  absolute CRPS, coverage, and width values require a canonical rerun.
- Experiment 37 is the current reference for architecture comparisons at the 3%
  ceiling under the inherited sparse stride-5 current-observation protocol. A
  matched full-camera-mask A/B/C ceiling benchmark is now Experiment 43, in
  `outputs/by_experiment/43_matched_ceiling_model_gap`. Its sequential row
  reproduces all 180 held-out Experiment 38 cases across ten metrics. Use its
  matched A/B/C rows for these ceiling comparisons, not mixed historical tables.
- Experiment 39b exactly reproduces Experiment 38 intermediate outputs through
  the reusable pseudo-ceiling runner in a real 10% case. Its development-only
  gate found no incremental transfer signal from smooth nested-ceiling response
  features, so no held-out or hardware-tail claims are made from that branch.
