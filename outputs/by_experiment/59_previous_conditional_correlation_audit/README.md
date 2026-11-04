# Experiment 59: Previous conditional spatial-correlation audit

## Answer

**Restoring the missing conditional residual correlations increases inherited
variance, but only slightly changes the total forecast marginal variance in
these six native cases.** Current hot-area mean prior variance increases by
0.015--0.063%; whole-area mean prior variance by 0.00015--0.0131%.
This narrow omission is not a large source of missing marginal forecast
variance under the frozen hybrid representation. It is not evidence that
full latent previous-state inference is unnecessary, nor a posterior-coverage
result. No current GP update was run.

This is the first, correlation-only part of the October 28 meeting follow-up.
Unsaturated predecessor uncertainty remains a separate untested intervention.

## Exact control

[Driver 63](../../../scripts/63_thermal_previous_correlation_audit.py) and
[isolated helper](../../../src/previous_correlation_audit.py) reuse output 52's
DiagonalScanPath_8 and SpiralScanPath_13 caches at reference, 320 K and 325 K.
All 6,275 native nodes and their area weights are retained. The previous
likelihood remains fixed numerical candidates plus all previous observed
censored locations. All other previous unsaturated noisy readings remain
pinned, just as in the cached hybrid. There are no new numerical observations,
non-saturation flags or noise changes.

The compared residuals live only at previous observed-censored nodes. This is
**joint sampling on that block embedded into the existing pinned hybrid**, not
a coherent full latent field. The same empirical conditional-mean draws, camera,
source input, physics mean, Q, coefficients, grid and boundary convention are
shared. The cached predecessor mean and deterministic physics forecast are
explicitly fixed: finite-sample mean shifts from new residual draws do not feed
back into the forecast.

Let M_s be the latent conditional mean given one completed noisy-observation
sample. On the censored predecessor block, the correct residual covariance is

`C = K_PP - K_PO (K_OO + (noise_sd^2 + jitter) I)^(-1) K_OP`.

The old fast samples use `M_s + diag(C)^(1/2) z_s`. The paired joint samples
use `M_s + L z_s`, with `L L^T = C` to floating-point tolerance. The old samples
already contain the spatial covariance `Cov(M_s)`; they do not omit all spatial
correlations. Only the off-diagonal part of C is restored here.

For the linear native heat map A, restricted to the same censored block:

```
Inherited_independent = A Cov(M) A^T + A diag(diag(C)) A^T
Inherited_joint       = A Cov(M) A^T + A C A^T
Total_prior           = Q_dt + Inherited
```

The principal comparison integrates Gaussian residuals analytically rather
than relying on empirical sample-covariance differences. `Cov(M)` still uses
the fixed cached MCMC sample, not an exact posterior integral. The absolute
change `A (C - diag(diag(C))) A^T` is independent of those sampled M values.
Previous marginal variances agree analytically; propagated marginal variances
need not agree. The same floor-free/source-free propagation map as C is used.

## Results

Physical-area weighted quantities over the fixed current hottest 1% area.
The percent change is the ratio of area-mean variances; SD columns are the
area means of nodewise SD, not square roots of area-mean variance.

| Case | Ceiling K | Total prior variance change | Mean prior SD, independent K | Mean prior SD, joint K |
|---|---:|---:|---:|---:|
| Diagonal 8 | 296.121 | +0.06333% | 1.907539 | 1.908148 |
| Diagonal 8 | 320 | +0.03161% | 0.729721 | 0.729817 |
| Diagonal 8 | 325 | +0.02047% | 0.721133 | 0.721199 |
| Spiral 13 | 296.490 | +0.06334% | 1.937351 | 1.937972 |
| Spiral 13 | 320 | +0.02599% | 0.724643 | 0.724724 |
| Spiral 13 | 325 | +0.01478% | 0.718535 | 0.718585 |

Why so small? At the Diagonal reference ceiling, hot-area propagated residual
variance rises from 0.000363 to 0.002700 K^2, but the shared conditional-mean
term is already 3.177156 K^2 and Q adds 0.512558 K^2. The residual's relative
increase looks large only when separated from those larger terms. Spiral has
the same pattern. See [variance_summary.csv](variance_summary.csv) and
[covariance_summary.png](covariance_summary.png).

Three paired residual RNG realizations are saved in
[residual_repeat_comparison.csv](residual_repeat_comparison.csv). These repeat
Gaussian residual noise only, not the underlying HMC chains. Small empirical
differences can be comparable to the covariance effect: for example, Diagonal
reference joint hot-area inherited variance differs from its analytic value by
0.000058--0.001581 K^2 across residual repeats, versus a 0.002337 K^2 analytic
joint-minus-independent change. The analytic comparison is the primary result.

No RMSE, CRPS or coverage improvement is claimed: the forecast mean is fixed,
and there is no new current posterior. Small prior-diagonal changes alone do
not prove that every posterior conditional quantity is unaffected. In
particular, this audit does not characterize arbitrary full-field covariance
functionals or the effect of restoring correlations in a different observation
regime.

## Reliability, replay and computational size

- Four inherited chains, 400 retained samples per chain, 100 burn-in, thin 1.
  All original 1,600 previous draws are reused. The first full legacy HMC chain
  is independently replayed in every case and matches cached draws exactly.
- Conditional means are recovered by subtracting the original independent
  residuals using the audited RNG contract. HMC uses a separate seed; this
  recovery is explicitly forbidden for the old elliptical-slice path. Unit
  tests also compare recovered means against direct Gaussian conditioning.
- Existing inherited variance is reproduced within 1.12e-14 K^2, and the
  basis-matrix propagation agrees with the production operator within
  1.78e-15 K. Every input fingerprint remains unchanged.
- Up to 128 geometry-selected censored locations per case are monitored for
  cached independent draws, joint draws and conditional means. Across these
  monitors, maximum rank-normalized split R-hat is 1.00711, minimum bulk ESS
  522.25 and minimum tail ESS 711.72. These pass the numerical thresholds used
  in the mask pilot; they are not an exhaustive all-node convergence proof.
- Conditional covariance has roundoff-scale negative eigenvalues at reference
  (minimum about -4.4e-14 K^2). Only these are clipped for a PSD factor; maximum
  covariance reconstruction error is 1.12e-14 K^2. No residual jitter or
  uncertainty inflation is added.
- Full audit: **65.45 seconds**, **865.52 MiB** peak process RSS. Reference
  cases take 31.87/25.34 s; higher ceilings 1.24--1.79 s per case, reusing
  each trajectory's propagation basis. This is not the runtime of a full
  latent predecessor or current GP fit.

| Case | Previous censored nodes: reference / 320 / 325 | Pinned unsaturated at reference |
|---|---|---:|
| Diagonal 8 | 1213 / 25 / 17 | 5062 |
| Spiral 13 | 1259 / 20 / 13 | 5016 |

Reference residual covariance matrices occupy 11.23/12.09 MiB, and their
native-output propagation matrices 58.07/60.27 MiB. A single dense 6275-square
full-native matrix would occupy about 300.41 MiB before factors, observations,
draws and temporary arrays. No full-native expansion was launched. See
[runtime.csv](runtime.csv) and [computational_size.csv](computational_size.csv).

## Interpretation and next boundary

The shared conditional-mean draws already carry most inherited uncertainty.
Restoring residual correlations alone is not a convincing explanation for the
large peak errors or undercoverage in these cases. This result does not address
the **zero represented uncertainty at pinned unsaturated predecessor nodes**.
That should be the next separate comparison, with all actually used previous
camera values aligned. Do not compare a hybrid using all unsaturated values
through pinning with a latent posterior conditioned only on sparse readings
and attribute the difference solely to uncertainty representation.

The old `infer_previous_coherent_posterior` also uses an older sampler and
grid-shaped interface, so its name alone is not a matched current-native
baseline. No source reformulation, mask experiment, parameter fit, posterior
mean correction, smoothing, held-out expansion or canonical replacement was
performed here. Contemporary simulator-derived gradient availability remains
an independent open issue.

## Reproduction and files

The production environment lacked ArviZ. ArviZ 0.17.1, xarray 2024.3.0 and
xarray-einstats 0.7.0 were installed in `/tmp/thermal_correlation_diagnostics`
only, not in the project's virtual environment. The historical SciPy Gaussian
import alias is applied for ArviZ compatibility. Explicit SciPy BLAS products
follow the native drivers and avoid NumPy/Accelerate matmul warning flags.

```sh
env PYTHONPATH=/tmp/thermal_correlation_diagnostics OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/thermal_mpl .venv/bin/python scripts/63_thermal_previous_correlation_audit.py --output /tmp/thermal_correlation_replay
```

The driver refuses to overwrite any destination. Reinstall the pinned optional
diagnostics packages into an isolated location if the temporary directory is
absent. Seven new tests and **112 total tests pass**. Compilation passes with
a writable temporary bytecode cache; `git diff --check` passes. See
[verification.md](verification.md), [verification.json](verification.json),
[frozen_configuration.json](frozen_configuration.json),
[equality_checks.csv](equality_checks.csv),
[sampling_diagnostics.csv](sampling_diagnostics.csv), and
[source_fingerprints.json](source_fingerprints.json).

`by_trajectory/*/level_*.npz` stores chain-indexed conditional means and joint
censored-block draws, residual covariance, observation IDs, fixed means,
propagated variance components and monitored covariance blocks. Original
independent draws remain in output 52; they are not duplicated here.
