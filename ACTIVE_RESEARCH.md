# Active research: censoring-severity failure decomposition

## Scientific focus

As sensor range decreases, what changes first: physical forecast bias, the
current GP correction, standardized errors, or posterior-tail behavior?
No new kernel, calibration fit, source rollout or smoother is active.

## Retained reference

1. Converged inference and observation-set audit: script 45 / output 41.
2. Matched A/B/C six-ceiling sweep: script 47 / output 43.
3. Pseudo-censoring as supporting motivation only: output 38. Coverage can
   improve without fixing the peak or improving hot-tail CRPS.

## Completed severity diagnosis

Script 49 / output 45 replays only B and C when draws are unavailable, preserving
Experiment 43 settings and validating hashes and every saved metric. It adds
physics-prior versus posterior errors, standardized errors, posterior quantiles,
skewness and source-relative residual summaries. Existing marginal sampling is
preserved; these are not newly coherent joint spatial samples.

Read outputs/by_experiment/45_censoring_severity_decomposition/README.md for
status, limitations and results when complete. Never pool intermediate dev
results into the final held-out interpretation.

## Current attribution experiment

Script 50 / output 46 uses cached previous estimates and the unchanged physics
operator to separate propagated state error from oracle-state physics error.
No GP inference or fitting is run. The oracle has positive hot-tail bias while
the state contribution is negative: the small reference forecast bias partly
reflects cancellation. Increasing censoring makes the latter more negative;
the local propagation ratio stays near 0.993, not amplification.

Read outputs/by_experiment/46_oracle_forecast_decomposition/README.md. This
attribution oracle is not a guaranteed performance upper bound. No remedy is
active; source/physics and previous-inference changes require a separate decision.

## Deferred, not deleted

Nested response, source-history and smoothing outputs/drivers were moved to
dated archive locations. See outputs/archive/2026-09-20_deferred/README.md.
Scripts 45 and 47 were deliberately retained: they are valid infrastructure and
the central comparison, not failed side branches. Script and output numbering
are different; use names as well as numbers when requesting cleanup.
