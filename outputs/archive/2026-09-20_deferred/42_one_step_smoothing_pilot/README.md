# L=1 stochastic reweighting pilot: reliability gate failed

Only the three development trajectories were evaluated. No longer lags,
held-out trajectories, new kernels, likelihood tempering, or resampling were run.

## Setup and baseline identity

The target is the predetermined midpoint frame on each trajectory, with one
immediately subsequent observation at +0.5 ms. These are new interior targets,
not the historical final-frame targets, so their numbers must not be compared
as reproductions of Experiment38/41. Both L0 and L1 reconstruct the same T_n.

Physical coefficients, noise assumptions, RBF scales, stationary innovation,
stride-5 unsaturated + all observed-censored observation policy, and HMC
convergence settings are frozen from Experiment38. Source width uses commanded
source history only through n; ambient is the median initial camera frame.
This replaces the old full-history nuisance preparation for this causal pilot.
The per-trajectory synthetic reference ceiling is inherited and fixed in time.
Truth is used to generate simulated camera data and evaluate predictions, not
to construct weights or choose the midpoint. Future source flux is assumed
available as commanded input.

The target reconstruction still uses the inherited hybrid n-1 posterior
(sampled censored pixels, pinned noisy unsaturated pixels). It is a two-frame
approximation, not a filter that has assimilated every earlier frame.

## Joint spatial sampling prerequisite

`sample_censored_gaussian_blocks` now accepts an optional full prediction
covariance. With it, the conditional Gaussian remainder is sampled jointly
through its Cholesky factor. The default fast marginal behavior is unchanged.
The censored HMC and the analytic marginal mean/variance formulas are unchanged.

The full prior's extracted blocks agree exactly with the old construction on
these three cases. The joint target draws passed the inherited R-hat criterion
(maximum 1.0066). Compared with the marginal-draw control, empirical mean RMSE
is about 0.023 K; average top1 RMSE is 2.9789 versus 2.9787 K. This is a sampling
representation comparison, not a future-data benefit. A unit test verifies the
analytic marginal parameters are identical and the missing cross-covariance
is restored. Numerical Cholesky stabilization is retained.

This does NOT repair or relabel the hybrid previous posterior as a full latent
posterior. It produces joint draws from the declared approximate current prior
and censored likelihood, which is sufficient to define this bounded pilot.

## Stochastic transition and weights

For each target draw X_s, the frozen transition is

    T_(n+1) = F(X_s, q_(n+1)) + eta,     eta ~ N(0, Q_dt).

F diffuses/cools the positive temperature excess, using the inherited
below-ambient floor and zero-excess boundaries, then adds the source term.
Q_dt is the existing stationary heat-process finite-step innovation, restricted
to future observation locations. Sampling only those coordinates is sufficient
for the one-step likelihood. This is a valid discrete stochastic transition;
the numerical floor/boundary approximations are not an exact linear PDE.

For eight independent stochastic futures per target, compute

    Lhat_s = (1/8) sum_b p(Y_(n+1) | T_(n+1,s,b)),
    w_s = Lhat_s / sum_r Lhat_r.

Unsaturated data use the Gaussian density with the frozen 0.5 K inference
noise SD; censored data use Phi((T-c)/0.5). Camera-generation noise remains
0.25 K, as in the baseline. The measurement realization is not resampled.
There is no future truth input to this likelihood.

Two independent groups of four futures also yield repeat A/B estimates to
diagnose process-Monte-Carlo sensitivity. Original target draws are retained;
weighted metrics describe T_n, not the forecast field.

## Predeclared reliability gate

For each repeat and the pooled run: target ESS >= 120 of 1200; maximum weight
<= 0.02. Repeat smoothed-mean disagreement must be <= 0.25 K RMSE in top1 and
<= 0.1 K globally. ESS is the nominal weight ESS, not corrected for residual
MCMC dependence, so it can be optimistic. Target-chain mass and path-level ESS
are also saved. This is a practical pilot gate, not a convergence theorem.

| Development trajectory | Pooled ESS / 1200 | Maximum weight | Particles holding 95% mass | Gate |
|---|---:|---:|---:|---|
| DiagonalScanPath_8 | 2.23 | 59.2% | 3 | Fail |
| HorizontalScanPath_13 | 1.98 | 67.2% | 4 | Fail |
| SpiralScanPath_13 | 3.54 | 39.1% | 4 | Fail |

Repeat A/B target-weight total variation is effectively 1. Their top1
smoothed means differ by 1.50--1.92 K RMSE. Thus eight paths per particle do
not give a stable marginal future likelihood in this setup.

## Scores: diagnostic only, not model rankings

Weighted CRPS is the exact score of the weighted empirical CDF:

    sum_s w_s |X_s-y| - (1/2) sum_sr w_s w_r |X_s-X_r|.

The same empirical-CDF convention is used for the uniform L0 control. These
columns are explicitly named empirical_crps, NOT the unbiased M(M-1) posterior
Monte-Carlo estimator from historical tables. Self-normalized importance
weights do not acquire an unbiased CRPS estimator by substituting an ESS.
Historical scoring code was not changed. Intervals are weighted-CDF quantiles.

The pooled L1 diagnostic has top1 RMSE 3.165 K versus 2.979 K for L0, and
coverage 48% versus 84%. These are outcomes of a degenerate importance
approximation, not evidence that correctly inferred smoothing is worse.
Metrics.csv flags every case with the pilot gate outcome; L0 itself passed
the target HMC checks. Independent explicit-sum score verification is saved
in saved_score_audit.csv.

## Interpretation and stopping decision

The simple prior-proposal reweighting pilot is not reliable. Weight collapse
and instability to stochastic-future repeats are the dominant diagnostic.
This does not distinguish all contributions from transition bias, proposal
mismatch, or insufficient Monte Carlo, and does not reject smoothing itself.

The future frame also supplies little new cooling evidence: 7, 5, and 12
initially censored pixels become unsaturated, but only 1, 0, and 1 respectively
are retained by the frozen stride-5 policy. No observation policy was changed.

Stop here. Before extending lag or dataset size, a future implementation would
need better future-likelihood integration or an observation-informed sampling
proposal and a new reliability check. Merely resampling these weights would
duplicate a few particles, not fix the lack of support.

## Reproduction

Driver: scripts/46_thermal_one_step_smoothing_pilot.py.
Per-trajectory NPZ files retain joint target particles, all eight future
log-likelihood arrays, weights, and evaluation data for independent rescoring.
The driver resumes completed trajectories. Fixed configuration, source hashes,
checks.csv, ess.csv, development_summary.csv and gate.json record the protocol.
