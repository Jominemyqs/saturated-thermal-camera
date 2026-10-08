# Native-input advection pilot following the meeting

## Scope

Two development trajectories, one reference ceiling and the frozen output-49
native-input protocol. No refit, kernel search, depth correction, source-unit
assumption change, or held-out expansion. Previous hybrid draws, camera values,
observation IDs, seeds and physical/GP parameters are shared across six rows.
Both ordinary B/C controls exactly reproduce output 49's saved means, SDs,
intervals AND all marginal predictive samples (maximum difference zero).

## What "advection" means here

The displacement d is the difference of native-area-weighted centroids of
HeatFluxZ above the existing source threshold, using previous/current times.
It is a source-motion proxy, not a confirmed material velocity or commanded
laser path. No latent temperature truth enters its construction.

- Diagonal 8: d = (0.142180, 0.002640) mm.
- Spiral 13: d = (0.021746, 0.147446) mm.

Define A as the frozen Gaussian diffusion/cooling map and S_d f(x)=f(x-d).
Positive d moves a feature in the positive d direction. A deterministic pulse
test verifies both coordinate signs and that the current source stays unshifted.
For the mean, translate only the propagated previous temperature excess; retain
the same current source gamma*q_n*dt. The positive-excess floor is unchanged.
For uncertainty, use A_d=S_d A on centered previous marginal samples.

| Row | Deterministic mean | Current residual prior covariance |
|---|---|---|
| B | Ordinary posterior physics mean | Fixed RBF |
| B_mean_advection | Advective posterior physics mean | SAME RBF |
| C | Ordinary posterior physics mean | Q_dt + Cov[A delta_previous] |
| C_mean_advection | Advective posterior physics mean | SAME covariance as C |
| C_uncertainty_advection | Ordinary posterior physics mean | Q_dt + Cov[A_d delta_previous] |
| C_both_advection | Advective posterior physics mean | Q_dt + Cov[A_d delta_previous] |

Mean-only and uncertainty-only C rows are component ablations; they do not
claim one common advection PDE governs both components. There is no advective
snapshot RBF, and no reintroduced joint previous/current likelihood. The final
likelihood receives current observations only.

For the inherited spatially homogeneous, free-space forcing, the equal-time
innovation is unchanged by spatially uniform advection: both covariance
arguments receive the same displacement for any common forcing time. Hence Q
is held fixed, not silently enlarged. This is the existing one-step sequential
construction, not the older joint stationary-ST regression or K+- K--^-1 model.
The free-space Q and zero-extended propagation remain an approximate boundary
treatment, not the original 3D simulation operator.

Native means the original surface nodal inputs/observations/evaluation. Heat
propagation still uses the shared fine computational grid and interpolation
from output 49. This is not an interpolation-free native 3D PDE solver.

## Results

Hot = the same fixed native hottest 1% of physical area, not the old top-25
grid mask. CRPS uses M(M-1). All quantities below are K except coverage.

| Path | Row | Field RMSE | All CRPS | Hot RMSE | Hot CRPS | Hot bias | Hot coverage | Hot width | Peak abs. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diagonal | B | 1.0444 | .8003 | 4.7721 | 2.4810 | 1.7206 | 84.55% | 10.090 | 18.122 |
| Diagonal | B mean adv. | 1.0466 | .8009 | 4.8901 | 2.5220 | 1.6297 | 85.24% | 10.161 | 17.424 |
| Diagonal | C | .7071 | .2515 | 4.4088 | 2.2476 | -.0981 | 77.26% | 6.369 | 15.889 |
| Diagonal | C mean adv. | .7289 | .2558 | 4.6384 | 2.4030 | -.0472 | 76.91% | 6.371 | 15.838 |
| Diagonal | C uncertainty adv. | .7070 | .2517 | 4.4764 | 2.2626 | -.0216 | 78.13% | 6.477 | 16.224 |
| Diagonal | C both adv. | .7182 | .2526 | 4.6670 | 2.3992 | -.0316 | 76.91% | 6.506 | 15.873 |
| Spiral | B | 1.2630 | .8553 | 5.1274 | 2.8821 | 3.3200 | 70.14% | 10.117 | 25.224 |
| Spiral | B mean adv. | 1.2655 | .8567 | 5.3002 | 3.0653 | 3.5108 | 67.19% | 10.099 | 24.624 |
| Spiral | C | .7410 | .2715 | 4.0741 | 2.1545 | 1.3173 | 76.39% | 6.424 | 22.892 |
| Spiral | C mean adv. | .7848 | .2809 | 4.3766 | 2.4875 | 1.7813 | 60.94% | 6.311 | 22.545 |
| Spiral | C uncertainty adv. | .7529 | .2738 | 4.1457 | 2.1513 | 1.3008 | 76.91% | 6.415 | 23.348 |
| Spiral | C both adv. | .7660 | .2752 | 4.3104 | 2.3442 | 1.6546 | 70.14% | 6.490 | 22.708 |

## Interpretation

Mean advection modestly reduces maximum-temperature error, but worsens hot-area
RMSE and CRPS for both B and C on both paths. C mean advection also markedly
reduces spiral hot-area coverage. Combining mean and uncertainty transport is
not a systematic improvement over C.

Uncertainty-only transport is mixed: spiral hot CRPS falls by only about 0.003 K
while hot RMSE and peak error worsen. Diagonal hot CRPS/RMSE worsen slightly.
Small differences from one seed and two development trajectories should not be
interpreted as statistically established effects. Do not expand or select a
new default model from this pilot.

The baseline signed-error maps and cross-sections in output 49/native_diagnostics
show an exaggerated sharp peak and an underpredicted shoulder, not uniformly
positive errors. Those diagnostics justify checking the original PDE, source
definition and depth heat balance; advection does not resolve that question.
No evidence here establishes missing vertical diffusion as the cause.

## Verification and reproduction

- All four native B/C controls exactly reproduce output 49 arrays.
- Recomputed previous means/draws and camera/current likelihood hashes match 49.
- Baseline RBF and innovation Q hashes match 49; hashes checked after updates.
- Same current sampler seed/settings in every row, no refit or new information.
- Rhat screens pass (maximum below 1.010); no current retries.
- Refining the advective mean computation to 1201 x 801 changes hot-area mean
  by 0.0071 K RMSE or less; full maximum point change is below 0.078 K.
- Translation-sign, unchanged-source and zero-displacement unit tests pass.
- Six fits plus preparation/checks/cache saving take about 139 s (Diagonal)
  and 182 s (Spiral) on this machine; this excludes the separate cached-plot pass.
- Hybrid previous representation remains pinned unsaturated measurements plus
  censored marginal draws, NOT a full coherent latent spatial posterior.

Run from the repository:

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/54_thermal_native_advection.py
```

The source configuration is output 49/frozen_configuration.csv, with the same
native overrides documented in output 49's README. Actual trajectory-specific
seeds, displacement and hashes are in matching_checks.csv. See results.csv for
all regions, paired_changes.csv for contrasts, runtime.csv for actual times,
and resolution_checks.csv for numerical sensitivity. Local ignored per-path
caches preserve all new marginal samples and forecasts. Historical results and
the existing report were not overwritten.
