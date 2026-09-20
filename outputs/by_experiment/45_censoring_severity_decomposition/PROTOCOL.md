# Frozen censoring-severity error decomposition

## Question

How do errors already present in the deterministic physics mean differ from
errors after the current censored GP update as the camera range is lowered?
This is a descriptive diagnostic, not another model ranking or fitting sweep.

## Replay contract

Experiment 43 saved scores and implementation hashes, but not posterior draws.
Therefore replay only B and C with exactly its parameters, camera generation,
ceilings, trajectory indices, seeds, HMC settings, previous posterior, and
current-only likelihood. A is not rerun. Three development and 30 held-out
trajectories retain the original split. For every model/ceiling, verify all
saved metrics to 1e-8 and exact hashes for observations, previous posterior,
physics mean and covariance blocks. A mismatch aborts rather than being
described as an approximately reproduced reference.

No parameter is selected with held-out truth. The existing simulator-derived
reference threshold and preparation provenance are inherited, not relabeled
as camera-only deployment calibration. Artificial thresholds and masks come
from the same reference camera, not truth. Selected likelihood locations are
stride-5 unsaturated plus every observed-censored location. These selected
sets are not strictly nested across ceilings, although the full images are.

## Cached quantities

Per trajectory/ceiling, level_*.npz stores float64 marginal posterior draws,
posterior mean, SD, quantiles, physics prior, previous posterior mean, truth
(evaluation only), exact hottest-25 mask, observed-censored mask, camera frames,
selected likelihood locations/values, and source-relative coordinates when valid.
Two models share the same prior field and previous inference. The source hash
manifest and level JSON records protect resume consistency. This is about 10 GB
of uncompressed draw arrays, compressed on disk; inference need not be repeated
for future marginal diagnostics.

The sampler is unchanged: marginal predictions do not constitute a fully
coherent joint spatial draw. Do not use these caches for a maximum distribution,
joint field propagation, or a new smoother without a separate inference audit.

## Diagnostic definitions

- Prior error: m_physics - truth; posterior error: mu - truth.
- Report whole-field, exact hottest 25, and observed-censored bias/RMSE.
- Peak signed error = max(predicted mean) - max(truth). Also retain the error
  at the true peak location; these are different quantities.
- Standardized error z=(truth-mu)/sd. Record mean, population SD, RMS, empirical
  95% coverage, width, mean posterior SD and invalid-SD counts.
- Posterior quantile u = mean(draw <= truth), using the actual marginal draws.
  Report mean/median u, fractions below .05 and above .95, posterior skewness,
  and posterior mean-minus-median. No Gaussianization is introduced.
- A zero or nonfinite SD is reported as invalid, never repaired with an arbitrary
  denominator. Regions exclude such values only from undefined standardized/shape
  summaries and retain their counts.
- Changes from reference are paired by trajectory. Held-out and family summaries
  give trajectories equal weight, rather than pooling differently sized regions.

## Interpretation limits

Mean(z), SD(z), and RMS(z) are descriptive. They do not uniquely identify bias
or variance failure: heterogeneity, spatial correlation, truth-selected hot-tail
regions and skewed censored posteriors also affect their distributions. SD(z)=1
is a heuristic Gaussian reference, not a formal null for the top-25 subset.
Likewise, empirical posterior quantiles are not required to be IID uniform
within a truth-selected hot tail. Extreme upper quantiles can result from bias,
underdispersion, or shape, not uniquely a missing tail. No IID tests are used.

Do not predeclare monotone worsening or a pass/fail threshold. If reference
coverage is already low, describe baseline mismatch instead of inventing a new
onset ceiling. Diagnose the mean increment (posterior minus physics-prior bias)
alongside the prior sign change before attributing a problem to the GP update.
Stronger causal scale-versus-shape claims would require a separate controlled
test, not just these descriptive statistics.

## Optional direction analysis

Use HeatFluxZ weighted centroids in the final two frames only, with the inherited
source cutoff. Do not derive direction from truth or interpolate a future path.
Mark an inactive/stationary source as unavailable. Positive along-coordinate is
ahead of motion; negative is behind. Normalize distances by the fixed source
lengthscale and use bins from -8 to +8, with transverse strip +/-2 lengthscales.
At reference, 10%, and 20%, summarize physics-prior errors along/cross the path
and ahead/behind. Save pixel and contributing-trajectory counts because strip
composition and domain boundaries can confound directional contrasts.

## Scope

One main severity figure, one source-relative supplemental figure, raw and
family tables, and a concise interpretation. No calibration, source model,
kernel, parameter optimization, SMC, smoothing, or long history is introduced.
