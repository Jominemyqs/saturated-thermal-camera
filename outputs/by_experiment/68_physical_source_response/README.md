# Output 68: independent boundary-response replacement

## Question and scope

Does replacing the empirical current gradient increment with the original FEM
response to independently prescribed laser input improve the actual censored
reconstruction? This is a mean-component replacement, not an oracle predictor
or a new calibration. Driver number **72** produces output number **68**.

Two development trajectories (Diagonal 8, Spiral 13), reference/320/325 K,
non-advective C, all-readings latent predecessor. No held-out expansion, mask,
parameter fit, source scaling, noise/Q change, or previous-inference rerun.
Historical baselines are not replaced.

## Results

**Independent physical response improves the four 320/325 K reconstructions,
but worsens hot-area reconstruction at both severe reference ceilings.** All
eight candidate fits passed the unchanged diagnostic/budget criteria. Six
primary baseline replays were array-exact. The reference result has the same
direction under both paired seeds; it is not inferred from one attractive plot.

Primary-seed posterior results, baseline -> physical response (K except coverage):

| Case | Ceiling (K) | Hot-area RMSE | Hot-area CRPS | True-peak signed error | Hot-area coverage |
|---|---:|---:|---:|---:|---:|
| Diagonal 8 | 296.121 | 3.616 -> 3.953 | 1.782 -> 1.907 | +11.249 -> -8.781 | 79.7% -> 81.8% |
| Diagonal 8 | 320 | 2.462 -> 0.462 | 0.798 -> 0.216 | +11.907 -> -0.712 | 55.0% -> 63.4% |
| Diagonal 8 | 325 | 2.304 -> 0.439 | 0.778 -> 0.224 | +10.991 -> -0.261 | 52.1% -> 60.8% |
| Spiral 13 | 296.490 | 3.443 -> 3.627 | 1.670 -> 1.770 | +21.490 -> -6.545 | 68.2% -> 68.8% |
| Spiral 13 | 320 | 2.427 -> 0.427 | 0.725 -> 0.182 | +21.435 -> +0.703 | 74.0% -> 77.6% |
| Spiral 13 | 325 | 2.067 -> 0.379 | 0.628 -> 0.177 | +19.207 -> +0.942 | 70.7% -> 74.0% |

The four milder-ceiling hot-area RMSE reductions are approximately 81-82%.
At reference, paired hot-RMSE increases are 0.338/0.346 K (Diagonal) and
0.184/0.194 K (Spiral); paired CRPS increases are 0.125/0.130 and 0.099/0.102 K.
The adverse direction persists across seeds. Very small unused-region changes
are not promoted as an additional benefit.

Spatial reconstruction, primary-seed posterior RMSE (K):

| Case / ceiling | Along shoulder | Across shoulder | Whole field |
|---|---:|---:|---:|
| Diagonal / reference | 1.928 -> 2.934 | 0.609 -> 0.599 | 0.518 -> 0.541 |
| Diagonal / 320 | 1.872 -> 0.942 | 1.977 -> 0.484 | 0.355 -> 0.181 |
| Diagonal / 325 | 2.360 -> 1.096 | 2.722 -> 0.687 | 0.381 -> 0.188 |
| Spiral / reference | 3.564 -> 5.064 | 0.476 -> 0.397 | 0.486 -> 0.498 |
| Spiral / 320 | 1.931 -> 0.835 | 2.375 -> 0.756 | 0.406 -> 0.201 |
| Spiral / 325 | 2.730 -> 0.908 | 3.351 -> 1.047 | 0.473 -> 0.217 |

Thus a smaller absolute peak error is not equivalent to a better field: at
reference, the along-path shoulder and whole-field RMSE worsen even though the
absolute peak error shrinks. At higher ceilings, both shoulder directions and
whole-field RMSE improve, not just the peak.

**Uncertainty remains inadequate.** Only Diagonal 325 K covers the true peak
among the six candidate primary cases (and 1/8 including reference repeats).
All eight baseline fits miss the peak. Candidate hot-area mean interval widths
are 5.516/0.404/0.397 K for Diagonal and 3.143/0.409/0.403 K for Spiral, almost
unchanged from baseline. Better point predictions do not establish calibrated
intervals; even the improved hot-area coverage remains below 95%.

## Forecast versus posterior

Deterministic hot-area RMSE (baseline -> candidate):

| Predecessor / ceiling | Diagonal (K) | Spiral (K) |
|---|---:|---:|
| Inferred / reference | 3.691 -> 4.236 | 3.581 -> 4.110 |
| Inferred / 320 | 5.649 -> 1.795 | 5.601 -> 1.822 |
| Inferred / 325 | 5.972 -> 2.114 | 5.975 -> 2.196 |
| True surface / all ceilings | 6.854 -> 3.014 | 6.650 -> 2.862 |

Oracle peak errors improve from +29.605/+36.752 K to +9.539/+8.791 K.
Those remaining positive errors are the Output-67 propagation discrepancy,
not an error in the independently reconstructed boundary response. With an
inferred reference predecessor, the propagated state-error terms at the true
peak are -18.884/-16.287 K; adding the positive propagation discrepancy leaves
candidate forecast errors -9.346/-7.496 K. Current observations correct these
only partly, to posterior errors -8.781/-6.545 K.

[Remaining-error decomposition](remaining_error_decomposition.csv) verifies
this identity at every native node, including replay and independent-solve
roundoff terms. These oracle terms are diagnostic only. Bias terms add; RMSE
terms do not.

This points toward the **combination of censored predecessor error and the old
surface propagation/state approximation**. It does not uniquely prove missing
depth physics, nor justify choosing a particular depth model. Even a true
surface predecessor leaves a nonzero discrepancy from full-volume evolution.
The result supports investigating that remaining approximation rather than
assuming a physically specified current response alone solves reconstruction.

The increment at the true peak changes from 50.249 to 30.183 K (Diagonal) and
62.312 to 34.352 K (Spiral). Surface increment integrals change from
5.086e-6 to 2.815e-6 K m^2 and 5.028e-6 to 2.822e-6 K m^2 respectively.
These are temperature-area integrals, **not energies**: prescribed laser power
remains 1 W. The candidate changes the boundary-input-to-temperature mapping,
not the prescribed power through a fitted multiplier.

## Reliability and runtime

- Candidate maximum rank-normalized split R-hat: 1.0070.
- Minimum bulk ESS: 830; minimum tail ESS: 939. All eight fits pass.
- Candidate fit/diagnostic times: 12.38-228.06 s; measured child-process wall
  times including cache output: 30.53-250.29 s. Peak recorded RSS: 726.25 MiB.
- Eight candidate processes: 868.24 s; six exact baseline replays: 389.59 s.
  Total GP orchestration: 1257.86 s (20.96 min), excluding later collection/plots.
- Two independent FEM assemblies/solves: 15.15 and 4.54 s; peak RSS 294.39 MiB.
  These fit timers exclude environment startup. No full trajectory was solved.
- Original mesh: 26,414 volume P1 DOFs, 130,664 tetrahedra, 6,275 surface nodes.
  Reference current constrained dimensions: 1,191 (Diagonal), 1,243 (Spiral).

No threshold relaxation, resampling retry, or parameter tuning was performed.
See [verification](verification.json), [runtime/diagnostics](convergence_runtime.csv),
[execution wall times](execution_status.json), and [validation notes](validation.md).

## Independent input and exact controls

The new response solves `(M/dt + K)b = f(g_current)` with zero initial increment
and zero bottom increment boundary, then restricts `b` to the native surface.
The response worker loads **only** geometry, surface IDs and timestamps from a
dedicated NPZ, plus the inspected notebook's prescribed path, laser expression,
material properties and boundary conventions. It never loads a temperature or
HeatFluxZ dataset. The old FEM assembly interface receives a newly allocated
zero vector; the response solve does not use the predecessor load at all.

This linear superposition is valid for the supplied constant-material linear
heat equation and fixed boundary conditions. It is not asserted for nonlinear
temperature-dependent properties or arbitrary thermal boundary models.

`m_old = old_propagated_predecessor + ambient + old_gradient_increment`

`m_new = old_propagated_predecessor + ambient + physical_boundary_response`

No historical gamma or gradient cutoff enters the new response. The old source
is removed, not retained alongside it. One response per trajectory/time is
reused unchanged across all camera ceilings. Prediction and observation means
are replaced consistently. Complete observed covariance, prediction-observation
cross-covariance and prediction variance are byte-identical to Output 61.
These include the full-latent inherited covariance, not a diagonal substitute.

Output 67 thermal states are read **after** response generation, for validation
only. Independent response agrees with its on-minus-off response within
1.52e-12 K. The new true-surface-predecessor oracle error agrees with Output 67's
surface-propagation discrepancy within 5.29e-12 K (including negligible replay
error). No true previous volume enters the candidate predictor.

## Inference and evaluation

Frozen Output 61 current retained observations and 0.50 K likelihood SD; camera
generation SD remains 0.25 K. Four chains, 800 retained samples per chain,
300 burn-in, thinning 1, 300 evaluation draws per chain. Reference has two paired
seeds; 320/325 K use the existing primary seed. All six primary baseline fits
are replayed and checked array-exactly; the existing verified reference repeat
baseline caches supply the second paired baseline.

Unchanged reliability thresholds: rank-normalized split R-hat <= 1.01,
bulk/tail ESS >= 400, classical R-hat <= 1.05, per-process wall budget 900 s,
peak RSS <= 4096 MiB. No retries or weakened thresholds. Numerical validation,
not improvement of every deterministic metric, gates this bounded six-case
pilot. Individual sampling failures are flagged rather than hidden.

Fixed physical-area weighting and hot-1%-area masks; along/across shoulder nodes
are inherited from Output 56. Truth is evaluation-only, apart from explicitly
labeled surface-oracle controls. Each field is correlated; nodes are not
independent experimental replicates.

- `forecast_metrics.csv`: deterministic inferred/oracle means; no fabricated
  uncertainty assigned to these deterministic forecasts.
- `gaussian_predictive_prior_metrics.csv`: separate pre-current-update Gaussian
  prior CRPS, bias, coverage/width and peak intervals, using the frozen prior
  variance. These are **not** posterior metrics or intervals for an oracle.
- `posterior_metrics.csv`: empirical posterior CRPS, point errors, coverage/width,
  peak intervals, and shoulder/whole-field metrics. The inherited empirical
  CRPS convention is preserved; its M(M-1) unbiasedness argument assumes IID
  draws, whereas the retained draws here originate from MCMC.
- `paired_changes.csv`: physical minus baseline; negative RMSE/CRPS is better.
- `reference_repeat_comparison.csv`: reference seed stability, not a formal
  confidence interval over trajectories.
- Cross-sections show pointwise marginal intervals at the existing section nodes;
  connecting lines do not constitute off-mesh or joint-field uncertainty.

## Files and reproduction

- [Driver](../../../scripts/72_thermal_physical_source_response.py)
- [Independent response and mean replacement](../../../src/physical_source_response.py)
- [Evaluation-only prior summaries](../../../src/physical_response_reporting.py)
- [Frozen protocol](frozen_configuration.json), [input/source fingerprints](fingerprints.json)
- [Equality checks](equality_checks.csv), [numerical gate](numerical_gate.json)
- [Original FEM audit](../67_one_step_fem_response/README.md)
- [Original current-update baseline](../61_unpinning_current_update/README.md)

Use the existing `.venv` for preparation, validation, current inference and
reporting; use `tmp/fem_env` only for the independent response. The exact FEM
environment is frozen in Output 67's `environment_explicit.txt`.

1. Driver `--phase prepare` freezes inputs and creates geometry-only caches.
2. In the FEM environment, run `python -m src.physical_source_response` with
   `geometry.npz`, the original notebook, family, interval and a fresh response
   output stem, once per trajectory.
3. Driver `--phase validate`, then `--phase run-gp`.
4. `python -m src.physical_response_reporting` produces the separate Gaussian
   predictive-prior summaries and checks the source-integral reductions.

Keep `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`. GP diagnostics use `tmp/thermal_diagnostics` on
`PYTHONPATH`, with the existing ArviZ/SciPy import compatibility helper.
Set cache directories inside `tmp/` as in the existing drivers. The serial
legacy FEM runtime requires local MPI initialization permission.

## Interpretation limits

Only the current increment is replaced. Old surface propagation (including its
floor/interpolation/boundary conventions) and old stochastic covariance remain
approximations. The complete inherited pipeline still includes simulator-derived
preparation and is not claimed to be camera-only. Oracle improvement is not
evidence by itself of improvement with inferred predecessors. A reduced peak
alone is not success if the shoulder or field reconstruction worsens.

Useful figures: [Spiral reference cross-sections](cross_sections_SpiralScanPath_13_level_0.png),
[Diagonal 325 K cross-sections](cross_sections_DiagonalScanPath_8_level_2.png),
[Spiral 320 K error maps](signed_errors_SpiralScanPath_13_level_1.png).
All six cases have corresponding cross-sections and shared-scale signed-error maps.
