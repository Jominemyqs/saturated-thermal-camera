# Matched model-gap versus censoring protocol

## Question and models

Does the advantage of a fixed physics-based reconstruction grow as camera
information is removed?

- A: current-only ambient mean + RBF.
- B: posterior physics mean + RBF.
- C: posterior physics mean + sequential stationary stochastic-heat covariance.

A/B share exactly the RBF covariance blocks; only their means change. B/C
share exactly the previous-posterior draws and deterministic physics mean.
C uses the existing innovation plus propagated hybrid previous covariance.
It is not an advective model, joint raw-two-frame likelihood, or smoother.

All three current likelihoods receive the identical stride-5 unsaturated plus
all observed-censored pixels. A has no previous observations in inference.
B/C use previous observations only through the previous inferred state, never
again in the current likelihood. A shares externally fixed/calibrated physical
nuisance preparation, but not the posterior physics forecast.

The current observation noise parameter is 0.5 K for ALL three methods, while
simulated camera noise and previous posterior noise are 0.25 K, following
Experiment38. RBF signal SD and lengthscale, physics coefficients, and C's
forcing configuration are frozen. There is no independent threshold tuning,
calibration, amplitude matching, or model selection on held-out truth.

## Cameras and severity

One noisy previous/current reference camera pair is generated per trajectory.
The reference ceiling is the inherited Experiment30/38 simulation-defined
threshold, not a new threshold learned from held-out truth. Lower ceilings
are quantiles of the observed reference current frame, targeting 5%, 7.5%,
10%, 15%, and 20%, plus the original roughly 3% reference. Both frames are
reclipped at the same ceiling. No latent truth constructs artificial masks.
Actual observed censored fractions and Kelvin ceilings are saved per case.

The full camera images are nested by reclipping. The selected likelihoods are
not strictly nested information sets: an off-stride formerly unsaturated pixel
enters the likelihood once it becomes censored, while its earlier unsaturated
value (and its implied upper inequality) were not used. All architectures are
matched at each ceiling, and the selection rule stays fixed, but this is not
a theorem-level pure information-loss intervention on the likelihood itself.
Do not infer monotonic Bayes-risk claims from these curves.

The inherited preparation uses simulator-based ambient estimation and
source-scale metadata. This study preserves that provenance for numerical
consistency; it is not a new fully camera-only deployment demonstration.
Alpha, beta, gamma, noise settings and scale multipliers are not refit.

## Inference and equality audit

Use the existing four-chain reflected-HMC protocol, 400 retained draws per
chain after 100 burn-in, thin 1; prescribed retry 800/300/1; maximum R-hat 1.05.
Each prediction uses a balanced 300 draws per chain for scoring. These settings
are not interchangeable with older ESS chain lengths. Retry counts are saved.

Seeds and current camera realizations are paired across all models/ceilings.
Within each trajectory/ceiling, B/C reuse one previous posterior inference.
All 180 held-out C rows are checked against saved Experiment38 uncalibrated
results, including observed-censored-region CRPS, with tolerance 1e-8.
SHA256 records identify observation arrays, shared means/covariances and
previous posterior objects. Per-level checkpoints are written only after all
three methods and the reference audit pass. Resume requires matching source
fingerprints.

No opt-in joint spatial sampler is used: this reproduces the historical fast
marginal representation. Pointwise scores remain meaningful for that declared
approximation. Previous spatial uncertainty remains hybrid/approximate; the
sampling audit from Experiment41 is not silently treated as resolved here.

## Evaluation and interpretation

Thirty held-out trajectories are equally weighted; three development paths
are also run and labeled separately. Hottest 1% is exactly 25 of 2501 pixels,
fixed by truth ranks at the common target frame across models and ceilings.
Truth enters metric evaluation only after the inherited preparation.

CRPS uses the existing distinct-pair M(M-1) estimator. Its exact unbiasedness
requires independent draws; convergence-checked MCMC remains the numerical
approximation. Historical scoring is unchanged.

For error/CRPS M, compute paired A-B (physics), B-C (sequential covariance),
and A-C (total) gaps. Positive means the right-hand model performs better.
Coverage, width and signed-bias differences do not have that universal
direction; interpret them separately, not as improvement win counts.

Report reference-to-20% gap growth, mean/median paired changes, and trajectory
counts. Bootstrap intervals resample whole trajectories (4000 replicates),
not dependent pixels or thresholds. Endpoint growth is not a proof that the
gap is monotone or a derivative is positive everywhere.

Observed-censored-region populations change with ceiling. The fixed top1
and all-domain regions provide the cleaner across-severity comparisons.

B-C includes BOTH covariance shape and the existing total marginal scale.
It describes the frozen architectures, not an isolated variance-matched test
of temporal correlation or a comparison after separately optimal tuning.
