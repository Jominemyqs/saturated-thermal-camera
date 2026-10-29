# Verification: bounded longer-chain check

Date: 2026-10-08. Base revision:
`d28f8d86f944728bb57fa0926a4b66b10f055aba`.

## Preserved inputs and implementation

- Historical driver 58, mixed-bound helper, reflected-HMC sampler, physics and
  covariance source files remain unchanged. All output-54 source/input hashes
  and output-55 source hashes were rechecked after inference.
- All 13 original code-snapshot source files remain byte-identical to their
  manifest hashes. The ZIP was neither replaced nor resent.
- The inherited output-54 parameter dictionary equals output 49's executed
  configuration exactly. Only current burn-in and retained-chain length change
  here; previous inference is not rerun, and evaluation draws remain 300/chain.
- Selected-location CSV and NPZ files are byte-identical to output 54. All ten
  fits have exactly the same observation IDs, likelihood kinds, numerical
  payloads (including NaN at flags), monitor IDs, constraint IDs, and paired
  seeds as their corresponding short-chain run.
- Prior-mean and total-prior-diagonal hashes match each original fit. Worker
  preparation asserts identical previous-draw/mean, physics-mean, propagated-H,
  prior-diagonal, and camera hashes. Overlapping enlarged covariance blocks
  again pass `rtol=atol=1e-13` checks.
- Longer-chain posterior draws are intentionally different; they are not
  falsely described as an exact posterior replay. The original exact short-chain
  baseline replay remains recorded in output 54.
- Every evaluation cache has shape `(4, 300, 6275)`; all monitored latent and
  constrained noisy-signal caches preserve `(4, 800, ...)` chain identity.
- No high-ceiling fit artifacts exist. The high-ceiling compatibility CSV reads
  existing output-53 baseline draws and is not a new posterior fit.

## Gate and scope

The ten reference fits all pass the unchanged statistical limits: maximum rank
R-hat 1.00616, minimum bulk ESS 1,027.21, minimum tail ESS 1,380.73, maximum
classical R-hat 1.00481. The primary spiral mask fit takes 318.904 s, above the
300 s cap; the second paired seed takes 278.175 s. The first run was not replaced
by its faster repeat. The full gate therefore remains false. No extra sampling
budget, changed seed, threshold relaxation, or high-ceiling exception was used.

No flag-budget expansion, held-out study, within-step integration, advection,
source reinterpretation, coefficient fit, Q change, noise change, or posterior
inflation was performed. Existing dirty `.DS_Store` files and the unrelated
untracked `src.zip` were preserved; nothing was committed or pushed.

## Tests and reports

- Full existing suite: **88 tests pass**, including the seven noisy mixed-bound
  and old-path-equality tests from the pilot (4.972 s reported test time).
- The existing nearly constant skewness test retains its SciPy precision warning.
  The test process also used a temporary Matplotlib cache because its default
  directory was not writable; inference itself used the prescribed writable
  `/tmp/thermal_mpl` directory.
- Driver 59 compiles. The underlying inference helper is unmodified; this new
  script is a bounded audit harness around the saved protocol, not a refactor.
- `short_vs_long_metrics.csv` is a one-to-one merge of long and short result
  rows keyed by trajectory/level/arm/repeat/region. Each difference is simply
  long minus short, with no model fitting or changed evaluation population.
- The unchanged plotting functions regenerate shared-scale unused-region
  signed-error/SD maps and fixed native cross-sections. Representative spiral
  maps and diagonal sections were visually inspected. White checkerboard gaps
  reflect masking of triangles touching retained nodes, not zero variance.

Outputs: [reference gate](reference_gate.json), [diagnostics](reference_convergence_runtime.csv),
[metrics](results.csv), [repeat stability](repeat_seed_comparisons.csv),
[source fingerprints](source_fingerprints.json), [README](README.md).
