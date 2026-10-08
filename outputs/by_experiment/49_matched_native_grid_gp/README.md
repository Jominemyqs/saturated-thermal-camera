# Matched native/projected-input GP pilot

## Question and scope

Does passing the previous state and current source through the existing
61 x 41 grid change B/C reconstruction when observations and evaluation are
matched? Two development trajectories were selected: DiagonalScanPath_8 and
SpiralScanPath_13. This is eight fits, not a held-out architecture ranking.

Every fit observes and predicts at the same native surface coordinates. This
is a projection-on/off prior-input comparison, NOT an end-to-end comparison of
native and regular-grid camera layouts. It follows the controlled deterministic
projection audit (output 48); it does not reuse the FEM operator in output 47.

## Exactly what is shared

- Native camera noise, reference ceiling, observation IDs and censor masks.
- Geometry-only nearest-native-node mapping of stride-5 numerical candidates,
  plus all observed-censored native nodes. No truth-based observation selection.
- One ambient-mean censored RBF previous-posterior inference per trajectory.
- Frozen alpha, beta, gamma, source cutoff, RBF scale and length, innovation Q,
  sampler settings, seeds, target times and native area weights.
- Current likelihood receives only current observations. Previous observations
  are not added again. No fitted parameters or correction are learned here.
- The same Gaussian diffusion/cooling operator with zero extension, evaluated
  through a common 601 x 401 computational grid, rather than a native FEM solver.

The previous representation is hybrid: unsaturated noisy measurements are
pinned and censored marginal draws are sampled. These are not coherent samples
from a full joint latent spatial posterior. This inherited approximation is
held fixed; the pilot does not validate it.

Let P be the existing native-to-coarse projector, R the coarse bilinear lift,
N native-triangle interpolation, E evaluation at native nodes and G the shared
Gaussian diffusion plus cooling. The uncertainty maps are A_N = E G N and
A_P = E G R P. Means use the inherited positive-excess floor and thresholded
source contribution. Native source values are unchanged; projected source
values make a coarse-grid round trip to native nodes before thresholding.

- B: posterior physics mean + identical spatial RBF residual. Only the mean
  changes between the two representations.
- C: posterior physics mean + Q_dt + Cov_s[A(T_previous^(s) - mu_previous)].
  Both the mean and propagated previous covariance change; Q is identical.

The finite-step innovation remains the inherited free-space heat covariance.
It is not a re-derived covariance for the original 3D FEM boundary conditions.

## Results

Scores use lumped native physical-area weights. The hottest region is exactly
1% of weighted area, with fractional weight at the cutoff node. It is fixed
between the paired models. It is NOT the old grid's hottest 25 pixels.
All CRPS uses the M(M-1) sample estimator. Coverage is shown with interval width.

| Path | Model | Input | Field RMSE K | All CRPS K | Hot RMSE K | Hot CRPS K | Hot coverage | Hot width K | Peak abs. K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | B | Native | 1.044 | 0.8003 | 4.772 | 2.481 | 84.5% | 10.09 | 18.12 |
| Diagonal 8 | B | Projected | 1.007 | 0.7980 | 3.616 | 2.038 | 88.0% | 10.18 | 6.72 |
| Diagonal 8 | C | Native | 0.707 | 0.2515 | 4.409 | 2.248 | 77.3% | 6.37 | 15.89 |
| Diagonal 8 | C | Projected | 0.650 | 0.2511 | 3.412 | 1.884 | 79.2% | 5.79 | 4.61 |
| Spiral 13 | B | Native | 1.263 | 0.8553 | 5.127 | 2.882 | 70.1% | 10.12 | 25.22 |
| Spiral 13 | B | Projected | 1.231 | 0.8536 | 3.999 | 2.423 | 71.7% | 10.13 | 0.05 |
| Spiral 13 | C | Native | 0.741 | 0.2715 | 4.074 | 2.155 | 76.4% | 6.42 | 22.89 |
| Spiral 13 | C | Projected | 0.691 | 0.2724 | 2.993 | 1.786 | 67.2% | 5.65 | 2.15 |

Projected inputs improve hot RMSE, hot CRPS, peak absolute error and whole-field
RMSE in all four pairs. All-domain CRPS improves in three of four. Spiral C
coverage worsens as its intervals narrow, despite better point and hot CRPS
scores. Thus this is not uniform improvement in uncertainty calibration.

Native peaks are overpredicted in all four native fits. Projecting the input
can compensate for the frozen predictor's source/mean mismatch. These results
do not prove that coarsening is more physically correct: parameters were frozen
from the regular-grid setting, and only two selected development trajectories
and one noise/seed realization were tested. No migration or retuning follows.

## Verification and computational size

- 6,275 native prediction nodes per trajectory; 12,448 surface triangles.
- Diagonal: 115 numerical + 1,191 censored likelihood points; 3.12% censored area.
- Spiral: 114 numerical + 1,243 censored likelihood points; 3.24% censored area.
- Four chains, 400 retained per chain after 100 burn-in steps, inherited retry
  rule; no retry triggered. Current evaluation uses 1,200 marginal draws.
- Maximum previous/current Rhat below 1.010. Rhat is a screen, not proof of
  convergence or of correctness of the hybrid representation.
- Shared-object hashes checked again after each inference call, including
  cameras, draws, coordinates, observation values, masks, RBF and Q matrices.
- Diagonal camera arrays and previous posterior mean exactly reproduce output 47.
- Second complete run reproduces all metrics within 3.56e-15.
- Refining the propagation grid to 1201 x 801 changes hot-area mean by at most
  0.0064 K RMSE. A shared subset of eight centered draws is also checked.
- Replay wall time, including preparation, resolution checks and saving:
  95.2 s Diagonal and 120.0 s Spiral. This includes all four models per path.
  Peak process RSS about 2.49 GiB. Stage timings are in runtime.csv.

The area-averaged RBF prior variance is 11.768 K^2 in both arms. Total C prior
variance is about 0.604-0.608 K^2 for native and 0.584-0.592 K^2 for projected
inputs, including inherited uncertainty; Q contributes about 0.513 K^2.
This is a controlled within-B/within-C representation test, not variance-matched
evidence for choosing B over C.

## Records and reproduction

Driver: `scripts/53_thermal_matched_projection_gp.py`.
Reusable interpolation/propagation: `src/matched_projection.py`.

Run from the repository with the dataset sibling folder available:

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/53_thermal_matched_projection_gp.py
```

`frozen_configuration.csv` is an unchanged COPY of output 43's source
configuration, not a description of every pilot override: its old top-25 rule,
six target fractions and calibration metadata are historical. Actual pilot
scope is two development paths at the reference ceiling, native area weighting,
no calibration, and common native observations as specified above and recorded
in `matching_checks.csv` and `verification.json`.

`results.csv` includes whole-area, hot-area and observed-censored-region scores.
`paired_projected_minus_native.csv` stores paired signed metric changes.
`resolution_checks.csv`, `runtime.csv` and `source_fingerprints.json` preserve
the audit. Local ignored `by_trajectory/*/matched.npz` stores all four marginal
draw arrays, means, SDs, intervals and shared camera/truth fields.
Reconstruction and absolute-error figures use shared scales across the fits.
No earlier results, canonical model code or local caches were deleted.
