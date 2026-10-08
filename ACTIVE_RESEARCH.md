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

## Native representation pilot

Script 51 / output 47 adds an isolated native-surface adapter and compares native
versus projected temperature/source fields on the three development trajectories.
It runs B/C inference on DiagonalScanPath_8 without interpolating those inference
fields. Canonical code and coefficients are unchanged. Projection changes narrow
peaks substantially, even when area means are nearly preserved; this does not yet
explain the oracle residual causally. Native propagation uses an explicit 2D P1
surface FEM approximation, not the original 3D solver. Read output 47's README
before comparing its metrics with regular-grid results: observations, boundaries
and area-weighted hot-region definitions differ. No held-out native sweep is run.

## Controlled projection audit

Script 52 / output 48 isolates representation before further GP work. On the
three development trajectories it reproduces the saved coarse-grid oracle
exactly, then passes native and projected inputs through the SAME fine-grid
implementation of the frozen Gaussian diffusion/cooling/source formula. No
native FEM operator, GP, refitting or new observations are introduced.

Projection lowers hot-area input-forecast bias by 0.47-0.79 K, but native-input
oracle bias remains +2.04 to +2.98 K. Projecting the reference and the remaining
coarse-pipeline numerical effect partly cancel that reduction. Narrow peaks
change strongly; removing projection alone does not repair the frozen mean.
Read output 48's README: its fixed native hottest-1%-area scores are not the old
top-25 scores. The native pilot remains supporting feasibility evidence, not a
replacement baseline. A small observation/evaluation-matched GP follow-up can
be considered separately; none was run in this audit.

## Matched projection-on/off GP pilot

Script 53 / output 49 runs B and C on DiagonalScanPath_8 and SpiralScanPath_13.
All four fits per path use the same native camera observations, previous hybrid
posterior, current likelihood, seeds, Gaussian heat operator and area-weighted
evaluation. Only the prior inputs are native or passed through the existing
61 x 41 representation; C also propagates the corresponding previous uncertainty.
No native FEM operator, coefficient refit or held-out expansion is introduced.

Projected inputs reduce hot-area RMSE and CRPS in all four model/path pairs.
This is not a universal UQ improvement: spiral C coverage falls from 76.4% to
67.2% as intervals narrow. Native peaks remain more overpredicted under the
frozen coefficients. Projection may compensate for predictor mismatch; this
does not establish that coarsening is physically preferable. Read output 49's
README before comparing with historical grid/top-25 metrics. The repeat run
reproduces metrics within 4e-15; the canonical baseline remains unchanged.

## Meeting follow-up: native diagnostics and transport

Forward development now uses explicitly labeled native-input cases while
preserving the regular-grid baseline. This does not mean native scores were
better or that the approximate surface operator is the original 3D solver.

Script 54 adds cached shared-scale signed errors and native-node uncertainty
sections in output 49/native_diagnostics. Positive average bias is not uniform
overprediction: B/C underpredict on 32-37% of surface area in these examples.

Output 50 isolates mean transport and previous-uncertainty transport on the
same two native development cases. B/C controls reproduce output 49 exactly.
Mean transport worsens hot RMSE/CRPS on both paths; uncertainty transport is
mixed. Do not promote it or expand the sweep. HeatFluxZ centroid displacement
is a proxy, with its export definition still unresolved. The next physics
decision awaits the original PDE/boundary/source setup; missing depth transfer
remains a hypothesis, not an established cause.

## Native forecast component attribution

Script 55 / output 51 uses cached native previous means and the unchanged
operator, with no GP inference or fitting. Exact replay of the output-49
forecast separates propagated previous excess from the numerical source.
The exaggerated current maximum appears on source addition; the predecessor
already lacks much of the shoulder. With true previous input, shoulder
structure improves but the source-plus-propagation combination overshoots the
peak more. This diagnoses numerical imbalance, not the physical source's true
magnitude. A scalar source reduction is not justified by this diagnostic alone.
Read output 51's README and fixed-section component plots; await the export
definition and PDE/boundary information before selecting a remedy.

## Higher-ceiling native comparison and export clarification

Script 56 / output 52 compares the reference ceiling with 320 and 325 K on the
same two native development cases. Both previous/current cameras are generated
from the same original noisy signals; reference B/C predictions reproduce output
49 exactly under its documented single-thread runtime. Parameters, numerical
candidate nodes, sections and fixed hottest-1%-area evaluation are unchanged.
Higher ceilings reveal the shoulder but expose a larger positive peak forecast
error under the frozen gradient contribution. They do not uniformly improve
hot-area RMSE/CRPS. Current likelihoods are not nested: newly unsaturated nodes
outside the numerical candidates lose their inequality without contributing
their values. Do not claim these curves establish that more observed information
intrinsically harms inference. No remedy or full-camera likelihood was tested.

Adrienne's notebook expression identifies HeatFluxZ as projected dT/dz, K/m,
conditional on that export matching these files. This supersedes the earlier
unresolved-source description: the frozen increment is an empirical
gradient-based contribution, not automatically known laser heating. The thesis
provides 3D boundaries but does not prove depth effects cause the mismatch.
All 33 stored trajectories have dt=0.0005 s; the thesis states 0.005 s. Preserve
stored time and resolve that provenance difference with the notebook. Read
output 52/PHYSICS_PROVENANCE.md before changing units, signs or coefficients.

## Current-observation retention control

Script 57 / output 53 isolates the high-ceiling current observation policy for C
on the same two native development cases, at 320/325 K. Cached previous draws,
forecast and prior covariance remain fixed within each paired comparison.
Keeping numerical candidates plus the observed reference saturation set lets
newly revealed nodes contribute their values. The four existing-policy replays
match output 52 exactly, including posterior draws.

Retention improves hot-area RMSE/CRPS in all four pairs and reduces peak error,
but does not fix the extreme peak. Intervals narrow substantially, hot-area
coverage falls, and whole-field CRPS is mixed. Thus discarded current information
and remaining predictor/UQ mismatch are separate findings. This pilot does not
justify rewriting historical results or adding an inflation rule. Read output
53's README and paired tables; no physics coefficient or covariance was refit.

## Deferred, not deleted

Nested response, source-history and smoothing outputs/drivers were moved to
dated archive locations. See outputs/archive/2026-09-20_deferred/README.md.
Scripts 45 and 47 were deliberately retained: they are valid infrastructure and
the central comparison, not failed side branches. Script and output numbering
are different; use names as well as numbers when requesting cleanup.
