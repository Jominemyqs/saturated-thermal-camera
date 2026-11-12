# Verification record

## Scope

- Source revision: `61438c90332e82c458809dfc148290a1a56f2662`.
- New isolated driver 67, helper and four unit tests; historical scientific
  implementations and outputs are read-only inputs.
- Source-on uses the `all_readings_latent` branch of output 61. This is a
  specific comparison branch, not a retrospective replacement of old baselines.
- Complete source/input hashes are recorded in
  [source_and_input_fingerprints.json](source_and_input_fingerprints.json).
- All six primary source-on runs reproduce cached posterior arrays, constrained
  noisy chains and monitored latent chains exactly. Reference repeat 1 source-on
  results are reused without resampling.
- Source-off covariance arrays are the same objects as source-on arrays in each
  worker. Hashes cover observation covariance, prediction-observation covariance
  and the predictive diagonal. No diagonal-only covariance approximation was
  introduced by the ablation.
- Source-off means independently reproduce the cached-predecessor propagation
  using the frozen native/grid mapping, positive-excess floor, diffusion/cooling
  and timestep to absolute tolerance 1e-12 K.

## Tests

`python -m unittest discover -s tests`: 126 tests passed (3.044 seconds) after
repairing the new output-directory creation. Existing constant-data skewness
test emits a precision-loss warning; it is not a sampling failure.

The new tests check unchanged covariance and original inputs, no repeated floor,
zero-increment exact sampler reproduction, an analytic all-numerical Gaussian
posterior mean change, and invalid inputs. Compilation passed with
`PYTHONPYCACHEPREFIX=$PWD/tmp/pycache` (the default macOS cache location was not
writable in the sandbox). `git diff --check` passed.

## Completed run

Eight new source-off fits and six exact source-on primary replays completed.
The reference gate passed. Seven new fits pass all checks; Spiral 320 K is
provisional because rank-R-hat is 1.01070085 at noisy constrained node 1906,
above the frozen 1.01 threshold. The JSON correctly records `all_fits_pass:
false`. No retry or changed threshold was used to hide this failure.

Independent review of the saved tables confirms worse hot-area RMSE/CRPS and
whole-field RMSE in the five passing primary pairs (and the same observed
direction in the provisional pair). All true-peak coverage flags are false.
Reference repeat differences are much smaller than on/off changes. Final
fingerprints confirm unchanged historical inputs and frozen source code.

Visual inspection covered Spiral reference and Diagonal 325 K cross-sections,
and the Spiral 325 K signed-error map. Axes, shared scales, labels and intervals
render correctly. The cross-sections show marginal intervals only; connecting
lines do not establish a continuous off-mesh posterior.

## Startup correction

The first preparation attempt stopped before sampling because a missing
`by_trajectory` parent directory was not created. The isolated driver now uses
`mkdir(parents=True)`. Its initial configuration/hash files were preserved at
`tmp/source_off_prepare_failed_20261104`; they are not scientific results.
The successful preparation records fresh fingerprints of the corrected driver.
No historical files or inference settings were changed for this repair.

## Limitations

The experiment is a frozen-mean ablation on two development trajectories, not
an independent test set or an optimized source replacement. Removing the
explicit increment does not remove solution-derived information elsewhere in
the inherited preparation/calibration. Generator provenance remains unresolved
as recorded in output 62. No new interpretation of gamma or HeatFluxZ is made.

Sampler convergence is checked but does not imply physical calibration.
Pointwise marginal draws are used for scoring, not spatial joint-field draws.
The mean's positive-excess floor and covariance's floor-free propagation remain
the same inherited approximation. Runtime gates apply to each fit, and fit
timers exclude final compressed-cache writes. Reused source-on fit timings are
historical; fresh replay and source-off runtimes are reported separately.
