# L=1 weight-collapse decomposition: development only

## Scope and baseline audit

This is a diagnostic of Experiment 42, not a replacement for its inference or
the canonical Experiment 43 ceiling-response comparison. We reuse its exact
1200 coherent target-field draws on the three development trajectories and
midpoint target frames. No target posterior was refitted. The future remains
one time step (0.0005 s). No held-out or longer-lag runs were performed.

The canonical observation rule remains stride-5 unsaturated plus all
observed-censored pixels. At these interior targets the future likelihood
contains 172, 179 and 221 observations. The frozen physical mean, positive-excess
floor, grid boundary approximation, measurement likelihood (0.5 K), and Q are
inherited from Experiment 42. Reconstructed original log likelihoods agree
within 1.5e-12 and pooled weights within 1e-14; see equality_audit.csv.

## Diagnostics

- Three independent innovation repeats, eight forward paths per target particle.
- Likelihood powers 0, .01, .05, .1, .25, .5, 1.
- Random nested subsets at approximately 10%, 25%, 50% and 100% of future
  observations, plus separate real-camera censored/unsaturated subsets.
- Q covariance multipliers .5, 1, 2, 4 (SD multipliers are their square roots).
- A fixed-design matched-model future generated from a target particle,
  original Q and 0.5 K measurement noise, then clipped. Its generating particle
  is excluded from both real and synthetic reweighting in that repeat.
- Exact Gaussian marginalization on unsaturated observations, compared with
  the eight-path estimator on the identical subset.

For a particle s with J innovation paths, powered weights use

    w_s(a) proportional to (1/J) sum_j exp(a * log L_sj).

This integrates the powered path likelihood; it is NOT [mean_j L_sj]^a.
Power less than one is a diagnostic intermediate distribution, not the desired
full-likelihood posterior. Resampling alone would only duplicate particles;
it would not restore lost support or validate the endpoint.

## Main results

The following are averages over three repeats, Q multiplier 1. ESS is out of
1199 eligible particles (1200 before excluding the synthetic generator).

| Development trajectory | Full likelihood ESS | Mean maximum weight | Censored-only ESS | Unsaturated-only ESS, 8 paths | Unsaturated-only ESS, exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal 8 | 1.75 | 76.1% | 616 | 1.06 | 153.29 |
| Horizontal 13 | 2.19 | 62.0% | 719 | 2.47 | 129.55 |
| Spiral 13 | 2.04 | 72.4% | 445 | 1.36 | 25.55 |

Exact unsaturated maximum weights are still 2.8%, 4.4% and 13.7% respectively.
Thus even the exact Gaussian-only check does not pass the full three-path
reliability criterion. It also omits future censoring information, so it is
not a finished smoother.

At likelihood power .1, real full-set ESS rises to 420, 440 and 189, and
maximum pairwise repeat disagreement in top-1% posterior means is only
0.074, 0.113 and 0.164 K. But at power .25, ESS is already 22, 22 and 10;
repeat disagreement rises to 1.223, 0.577 and 0.876 K. At power 1 it is
2.383, 1.847 and 1.802 K. Small-power stability does not certify a stable
full-likelihood endpoint.

With only approximately 10% of the real future observations, full-power ESS
is 119, 119 and 7; by 50% it is 1.3, 1.7 and 1.6. This supports strong
likelihood concentration, especially from unsaturated Gaussian terms, but
subset composition and noisy marginalization also contribute.

## What Q sensitivity does and does not show

| Q covariance multiplier | Diagonal full ESS | Horizontal full ESS | Spiral full ESS |
| --- | ---: | ---: | ---: |
| .5 | 2.09 | 2.59 | 1.29 |
| 1 | 1.75 | 2.19 | 2.04 |
| 2 | 1.41 | 1.12 | 1.87 |
| 4 | 1.15 | 1.03 | 1.00 |

The eight-path estimator does NOT become reliable with larger Q. This is not
evidence that larger process covariance is scientifically worse: in the exact
unsaturated check, Q=4 increases real ESS to 884, 887 and 316. Broader forward
noise can instead make a prior-path Monte Carlo likelihood estimate noisier.
Do not tune Q using this degenerating estimator.

## Interpretation and limits

There are at least two numerical mechanisms, not a clean mismatch-versus-
dimension dichotomy:

1. The product of many future likelihood terms concentrates weights. Even
   matched-model synthetic futures collapse (full-set ESS 1.20, 1.39, 1.02).
   On their unsaturated subsets, exact integration still yields only
   6.28, 10.67 and 4.15 ESS at Q=1. Model/FEM mismatch is therefore not
   necessary for a finite-particle importance proposal to fail.
2. Eight-path innovation marginalization materially amplifies collapse. The
   exact same real Gaussian subset improves greatly when integrated exactly.
   Consequently the original tiny ESS is not a pure diagnostic of thermal
   model mismatch or even of the exact marginal likelihood's concentration.

These checks do not quantify the fraction of collapse caused by each mechanism,
nor do they exonerate the approximate heat model. Synthetic futures are drawn
from the empirical target ensemble, use the inference noise scale, and retain
real-camera-selected coordinates. They are a conditional algorithmic control,
not independent thermal truth or a new camera-policy simulation. Region labels
in the main synthetic dimension table refer to real-camera regions; the exact
Gaussian check instead selects actually unsaturated synthetic observations.
Synthetic top-1% scores against FEM truth are not performance evidence.

Forecast residual summaries on unsaturated observations are descriptive and
selection-truncated; do not interpret them as unconditional standardized tests.
Repeats share the target particle ensemble, varying innovations (and synthetic
future in that control), so they test conditional Monte Carlo stability, not
all sources of posterior sampling uncertainty.

## Decision

The inherited full-likelihood reliability gate fails: ESS >=120, maximum weight
<=2%, pairwise top-1% mean RMSE <=0.25 K and whole-field mean RMSE <=0.1 K.
Several weighted intervals collapse and repeat summaries are unstable. Stop
here before Friday: no tempered/resample-move SMC implementation, no held-out
smoothing and no longer lags. Stable heavily tempered weights alone are not
enough to justify expansion.

Ask Adrienne whether to pursue a conditionally integrated future likelihood
and guided/rejuvenated proposal before investing in smoothing. This would be
an inference improvement, not evidence for a new thermal covariance model.

## Outputs

diagnostics.csv records individual runs; summary.csv averages repeats;
repeat_stability.csv compares target summaries; exact_unsaturated_check.csv
isolates marginalization error; forecast_mismatch.csv records descriptive
forecast errors. equality_audit.csv, fixed_configuration.csv, source_fingerprints.json
and gate.json document the inherited setup. collapse_decomposition.png compares
tempering and likelihood dimension for real and matched-model futures.

No CRPS ranking is made from degenerate reweighting. Existing paper tables
retain their M(M-1) CRPS convention; importance-weighted empirical CDF scores
would require separate labeling and are not interchangeable with those tables.
