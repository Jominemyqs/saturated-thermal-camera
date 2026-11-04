# Previous state unpinning audit

Output 60, driver [64](../../../scripts/64_thermal_previous_unpinning_audit.py).
This is a previous-posterior and deterministic-forecast audit, not a current-GP
experiment or a replacement for the saved native B/C baselines.

## Result

All six cases completed and passed the fixed reliability/runtime gate on the
first schedule. **Unpinning restores missing uncertainty at unsaturated nodes,
but does not by itself resolve the exaggerated peak.** Letting all readings
condition the censored posterior has a substantially larger effect than
unpinning alone. Neither comparison establishes current-posterior calibration.

### Previous state

At reference, the unsaturated-node area-weighted mean latent SD increases from
zero to 0.224 K in both cases. Its RMSE changes from 0.2432 to 0.2221 K for
Diagonal and 0.2448 to 0.2261 K for Spiral. At 320 K the denoising gain is smaller;
at 325 K unsaturated RMSE instead worsens (0.2437 to 0.2493 K and 0.2458 to
0.2519 K). A coherent latent representation is not a guarantee of lower realized
error under the frozen spatial prior.

The reference hot-region previous RMSE changes primarily during all-readings
conditioning: Diagonal 9.104 to 7.971 K, Spiral 7.721 to 7.717 K. Spiral's
hot-region previous bias becomes more negative, -2.690 to -3.834 K. Thus even
the previous-state response is not a uniform improvement in magnitude.

### Forecast means

Area-weighted RMSE on the fixed current hottest 1% of physical area, in K:

| Case | Ceiling | Legacy hybrid | All-readings pinned | All-readings latent |
| --- | --- | ---: | ---: | ---: |
| Diagonal 8 | Reference | 4.535 | 3.688 | 3.691 |
| Spiral 13 | Reference | 3.907 | 3.580 | 3.581 |
| Diagonal 8 | 320 K | 5.574 | 5.528 | 5.649 |
| Spiral 13 | 320 K | 5.779 | 5.464 | 5.601 |
| Diagonal 8 | 325 K | 5.981 | 5.981 | 5.972 |
| Spiral 13 | 325 K | 6.095 | 5.877 | 5.975 |

At reference, whole-field forecast RMSE improves from 0.631 to 0.439 K and
0.588 to 0.386 K (legacy to full latent). Almost all that improvement already
appears in the pinned all-readings bridge. True-peak forecast signed errors
remain +10.720 K and +20.464 K; the legacy values were +9.964 K and +21.691 K.
Unpinning does not remove the forecast's excessive peak contribution.

At higher ceilings unpinning has competing effects: along-section shoulder
RMSE improves relative to the matched pinned bridge in all four cases, but
cross-section shoulder RMSE worsens in all four. For example, Diagonal at
325 K changes from 1.962 to 1.856 K along the section, but 0.216 to 0.421 K
across it. This is a mean/shape tradeoff, not a general reconstruction fix.

The paired four-chain hot-region forecast RMSE differences (latent minus pinned)
are stable: reference +0.002981 to +0.003095 K for Diagonal and +0.001387 to
+0.001413 K for Spiral; 320 K +0.12046 to +0.12064 K and +0.13717 to +0.13726 K.
These are changes in deterministic forecasts based on independent chain means,
not confidence intervals or independent trajectory replications.

### Propagated covariance

The table isolates **unpinning only**, relative to the all-readings pinned arm.
Entries are percentage changes in area-averaged total predictive-prior variance
`Q + A Sigma A^T`, not percentage changes in SD or current-posterior intervals.

| Case | Ceiling | Whole field | Fixed hot 1% |
| --- | --- | ---: | ---: |
| Diagonal 8 | Reference | +6.041% | +0.0122% |
| Spiral 13 | Reference | +6.468% | +0.0428% |
| Diagonal 8 | 320 K | +6.558% | +0.5259% |
| Spiral 13 | 320 K | +6.558% | +0.5111% |
| Diagonal 8 | 325 K | +6.558% | +0.5286% |
| Spiral 13 | 325 K | +6.558% | +0.5159% |

At reference, in the previously unsaturated region, inherited marginal variance
increases from about 0.000014/0.000018 to 0.03466/0.03471 K^2. Q remains
0.512558 K^2. The signed cross terms are not zero: in the hot region they
contribute 0.000262 K^2 (Diagonal) and 0.000332 K^2 (Spiral), alongside the
unsaturated-block contributions of 0.000084 and 0.000078 K^2. These small
terms were retained rather than discarded.

Separately, assimilating all readings substantially **reduces** reference
hot-region total prior variance relative to the historical hybrid: 3.690 to
2.836 K^2 for Diagonal and 3.812 to 0.959 K^2 for Spiral (full latent). This
reduction is not the unpinning effect. Better conditioning under the frozen
prior can make it sharper without eliminating forecast mismatch.

### Decision

This audit supports representing previously pinned readings as uncertain latent
temperatures, but not claiming that this solves hot-tail undercoverage. It also
shows that a future current-GP comparison must preserve the distinction between
all-readings conditioning and unpinning. No current GP was run, so current
CRPS, coverage, and likelihood-conditioning effects remain untested. Preserve
this as a controlled audit; do not silently replace the historical baseline.

## Question and controls

What changes when noisy unsaturated predecessor readings are represented as
uncertain latent temperatures? A direct comparison against a sparse coherent
posterior would change the actual information used: the historical hybrid pins
every unsaturated camera value, including thousands outside its GP likelihood.
The audit therefore separates conditioning on those readings from unpinning.

| Arm | Previous numerical likelihood | Unsaturated latent representation | Censored latent representation |
| --- | --- | --- | --- |
| Legacy hybrid | Historical sparse candidates | All readings pinned | Cached marginal draws |
| Correlation control | Same historical candidates | All readings pinned | Output 59 joint residual covariance |
| All-readings pinned | Every unsaturated reading | Same readings pinned | Joint all-readings-conditioned posterior |
| All-readings latent | Exactly the same likelihood as pinned bridge | Latent posterior mean and covariance | Exactly the same censored marginal posterior as pinned bridge |

The first-to-third comparison changes **how already-used information enters
inference**, not just uncertainty. The third-to-fourth comparison isolates
unpinning with identical conditioning and censored-signal draws. Unpinning also
allows denoising of unsaturated posterior means; it is not a variance-only edit.
The fourth arm retains unsaturated-censored cross-covariance.

Two development trajectories are used: DiagonalScanPath_8 and SpiralScanPath_13.
Both reference cases must pass the predeclared reliability/runtime gate before
320/325 K cases run. Each ceiling uses its own cached camera and predecessor;
comparisons are paired within a case, not a claim of identical priors across ceilings.

## Computation

The isolated [helper](../../../src/previous_unpinning_audit.py) conditions on
all native camera observations without sampling thousands of independent latent
residuals. Let U be the observed-unsaturated nodes and S the observed-censored
nodes. With latent RBF covariance K, previous likelihood SD 0.25 K and inherited
numerical jitter j, define

```text
v = 0.25^2 + j
P = (K + v I)^(-1)
Cov(Z_S | Z_U=y_U) = P_SS^(-1)
E[Z_S | Z_U=y_U] = m_S - P_SS^(-1) P_SU (y_U-m_U)
```

The existing reflected HMC samples this correlated noisy-signal distribution
subject to Z_S >= ceiling. Censoring is not a hard bound on latent temperature.
Saturated numerical magnitudes never enter numerical conditioning.

Given a completed noisy signal Z, the latent conditional distribution is

```text
E[T | Z] = Z - v P (Z-m)
C = Cov(T | Z) = v I - v^2 P
Sigma_previous = Cov(E[T | Z]) + C
```

The empirical covariance of conditional means uses all retained HMC samples;
the correlated Gaussian residual C is integrated analytically. This preserves
the joint covariance without pretending that independent marginal residuals
are spatial field draws. Small geometry-selected monitor blocks do receive
joint residual draws for diagnostics.

For the unchanged linear covariance propagation A:

```text
Sigma_forecast = Q_dt + A Sigma_previous A^T
inherited diagonal = censored contribution + unsaturated contribution + signed cross term
cross term = 2 diag(A_S Sigma_SU A_U^T)
```

The cross term is retained, not set to zero. The pinned bridge contains only
the censored contribution of this same all-readings posterior. Q's marginal
variance is read from each frozen cache, not recalculated or tuned.

The deterministic forecast still uses the existing positive-excess floor,
native-to-fine-grid interpolation, Gaussian diffusion/cooling, zero extension,
and interpolation back to native nodes. Covariance propagation retains the
historical floor-free linear approximation; this is **not** exact propagation
through the nonlinear positive floor. Native input is not interpolation-free
3D propagation. The identical cached gradient-based contribution is added to
every arm. HeatFluxZ remains a projected vertical temperature gradient, not a
newly interpreted independent heating input.

## Effective settings and provenance

Inputs: [output 52](../52_native_higher_ceiling/README.md),
[output 59](../59_previous_conditional_correlation_audit/README.md), and
output 56's [fixed sections](../56_within_step_integration/frozen_sections.csv)
and [timestamp provenance](../56_within_step_integration/gradient_provenance.json).
No cameras or previous input caches are regenerated.

| Quantity | Effective setting |
| --- | --- |
| Native evaluation nodes | 6,275 |
| Previous RBF signal SD / lengthscale | 3.4305133895 K / 0.0003125 m |
| Synthetic camera SD / previous likelihood SD | 0.25 K / 0.25 K |
| Inherited numerical jitter | 1.1768422116e-6 K^2 |
| Diffusivity / cooling | 3.6109129986e-6 m^2/s / 12.4229015705 s^-1 |
| Gradient coupling / threshold | 0.1702227206 / 10,000 stored gradient units |
| Timestep | 0.0005000000000000004 s |
| Propagation computation grid | 601 by 401; same operator as baseline |
| Current likelihood | Not executed; its inherited SD remains 0.50 K |
| Primary hot region | Same cached hottest 1% of physical area, not 25 native nodes |

`frozen_configuration.json` preserves inherited metadata verbatim. Its nested
`inherited_options` contains historical grid labels (including top-25 of 2501),
old sampler defaults, model lists, and calibration fields. Those are provenance,
not this audit's effective masks, sampling schedule, or a new calibration.
Effective geometry/masks come from native caches; the explicit top-level
`hmc_schedule`, `gate`, and execution flags govern this audit. Historical
ambient/preparation/coefficient calibration remains simulation-derived; this
audit does not relabel it as camera-only fitting.

## Reliability and scope

Initial schedule: four chains, 800 retained per chain, 300 burn-in, thinning 1.
One bounded retry is permitted: four chains, 1600 retained, 600 burn-in. Both
use inherited previous seeds plus documented offsets. Initialization is the
existing HMC rule max(conditional mean, bound + 0.5 conditional SD), transformed
to whitened coordinates. No hyperparameters are optimized.

The gate requires rank-normalized split R-hat <= 1.01, bulk and tail ESS >= 400
at all monitored variables, <= 900 seconds per case, and <= 4096 MiB peak RSS.
Monitors include every noisy censored variable and up to 64 geometry-selected
latent nodes from each status. Classical R-hat is also saved. These are finite
MCMC diagnostics, not a proof of convergence, calibration or correctness of
the physics. The new case budget does not alter output 55's historical gate.

Independent chain means provide sensitivity checks for previous and forecast
errors. Truth is used only for evaluation and the inherited fixed evaluation
regions. No new source, mask likelihood, noise, Q, calibration, current GP,
held-out expansion or smoothing is introduced.

## Outputs

- [metrics.csv](metrics.csv): previous/forecast error and variance by arm and region.
- [propagated_covariance_components.csv](propagated_covariance_components.csv): censored, unsaturated and signed cross contributions.
- [information_sets.csv](information_sets.csv): actual likelihood and pinned-reading counts.
- [sampling_runtime.csv](sampling_runtime.csv) and [sampling_diagnostics.csv](sampling_diagnostics.csv): all attempts and diagnostics.
- [chain_mean_sensitivity.csv](chain_mean_sensitivity.csv): four independent chain-mean summaries.
- [verification_cases.csv](verification_cases.csv): baseline reproduction and gates.
- [verification.json](verification.json) and [validation.md](validation.md): completed scope, equality, runtime and tests.
- [source_fingerprints.json](source_fingerprints.json): inspected working-tree code and immutable input hashes.
- Local `by_trajectory/` caches preserve chain identities, conditional camera blocks, monitor covariance, posterior moments and propagated components. Large NPZ files stay local under the existing ignore policy.

Cross-section figures show previous latent means and deterministic current
forecasts. They are not final current-posterior means or credible intervals.
