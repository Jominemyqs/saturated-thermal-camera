# Native forecast component attribution

## Question

Where does the exaggerated spike enter the existing forecast, and where is the
shoulder lost? This is numerical attribution on the SAME two native-input
development cases as output 49, not a new GP/model comparison or physics fit.

## Frozen calculation

With the existing implemented map P, including native-triangle interpolation,
positive-excess floor, Gaussian diffusion/cooling, zero extension and bilinear
evaluation back at native nodes:

```
p_base   = P(previous_inferred - ambient)
p_oracle = P(previous_truth - ambient)
s        = existing thresholded source contribution
m_base   = ambient + p_base + s
m_oracle = ambient + p_oracle + s
needed   = current_truth - (ambient + p)
error    = s - needed = m - current_truth
```

P is not asserted to be linear: it includes the actual floor. The source is
isolated by calling the existing mean implementation with zero excess and zero
ambient. This preserves the exact source-threshold rule and multiplication
order without replacing the operator with a simplified formula.

The source is identical for inferred-state and oracle-state inputs. No
advection, GP inference, hyperparameter fitting, calibration, source correction,
or new measurement noise is used. Current truth is evaluation only; previous
truth is supplied only to the attribution oracle. Stored native camera arrays,
previous means and physical-area weights are unchanged.

**Needed residual is not true heating.** It absorbs predecessor error,
propagation error and omitted physical effects. Its sign can be negative.
These deterministic forecasts are before the current GP update; they are not
new B/C posterior reconstructions and have no new credible intervals or CRPS.

## 1. Source addition creates the exaggerated current spike

Values at the actual current true-peak node, in K (source is a temperature
increment, not a flux measurement):

| Case | Current truth | Inferred-state propagated-only | Added source | Complete inferred forecast | Oracle propagated-only | Complete oracle forecast |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | 340.430 | 300.145 | 50.249 | 350.394 | 319.786 | 370.035 |
| Spiral 13 | 341.182 | 300.560 | 62.312 | 362.873 | 315.621 | 377.933 |

At those locations, the source exceeds the needed residual by 9.964 K and
21.691 K for inferred inputs, or 29.605 K and 36.752 K for oracle inputs.
Before source addition the maxima anywhere on the surface are only
300.53/300.96 K (inferred state) and 334.98/330.68 K (oracle state), below the
current true maxima of 340.43/341.18 K.

Thus the exaggerated maximum enters on source addition in this numerical
predictor. That does NOT prove that the imposed physical heating is too large:
the exported flux definition and the adequacy of P remain unresolved.

The Diagonal forecast maximum is at a different node from the truth maximum.
`point_components.csv` records the true peak, forecast peak, oracle peak and
source peak separately; do not confuse pointwise peak-location error with
maximum-temperature error.

## 2. Much of the shoulder is already missing in the predecessor

For illustration, at the saved along-direction section node nearest -0.5 mm
behind the current peak (diagnostic location only):

| Case | Section position mm | Previous truth | Previous inferred | Propagated inferred | Current truth | Complete inferred | Complete oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | -0.500 | 309.713 | 300.320 | 300.202 | 306.711 | 301.779 | 310.416 |
| Spiral 13 | -0.504 | 313.726 | 300.819 | 300.740 | 309.587 | 302.546 | 313.891 |

The predecessor is already compressed toward about 300 K there. Propagation
changes that inferred value little, and the localized source adds only about
1.58/1.81 K, insufficient to recover the current shoulder. Oracle propagation
retains much more of the shoulder, but overshoots the current truth at these
locations. Replacing the predecessor does not remove all propagation mismatch.

These examples are descriptive samples from the fixed sections, not training
points, newly selected fitting regions, or a quantified causal fraction for the
entire shoulder. Physical temporal evolution is not itself a prediction defect.

## 3. Regional means reveal cancellation, not a uniformly positive error

Fixed current hottest 1% of native physical area:

| Case | Source mean K | Needed / inferred K | Needed / oracle K | Total inferred bias K | Propagated-state contribution K | Oracle bias K |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal 8 | 3.244 | 4.042 | 0.300 | -0.798 | -3.742 | +2.944 |
| Spiral 13 | 3.255 | 2.923 | 0.374 | +0.332 | -2.549 | +2.880 |

The exact error identity is `total = propagated_state + oracle_physics`.
Whole-region negative inherited error partly cancels the positive oracle error,
even though the sharp current peak remains too high. RMSE does not add:
inferred/oracle hot-area forecast RMSE is 4.535/6.854 K (Diagonal) and
3.907/6.650 K (Spiral). The oracle is an attribution control, not a guaranteed
performance upper bound with this frozen approximate predictor.

These are lumped native-node area scores from output 49, not output 48's fine
quadrature scores or the historical grid top-25 scores. Small numerical
differences between those tables are not new changes to the physical model.

## Interpretation and limits

The predictor combines a broadly underestimated inferred predecessor with a
highly localized source increment that overshoots the peak and does not replace
the lost shoulder. True-previous-state input restores more of the shoulder but
makes the peak overshoot larger. This locates the numerical imbalance without
identifying whether the remedy is the source interpretation, coupling,
time integration, depth heat transfer, or another aspect of propagation.

A uniform reduction of gamma is not established as the solution: it would also
reduce already-insufficient source contributions at underpredicted shoulder
locations. Source definition/PDE information and a separately controlled test
are needed before changing the mean. No remedy is implemented here.

## Figures and saved arrays

`component_sections_*.png` reuses the exact native node IDs and along/normal
coordinates from output 49/native_diagnostics. Four rows show predecessor,
propagated-only temperatures, complete forecasts, and added versus needed source.
Section directions are the frozen HeatFluxZ-centroid motion proxy; truth selects
only the diagnostic current-peak origin. Lines connect nearest native nodes,
not continuous straight-cut predictions. No new uncertainty bands are invented.

- `region_metrics.csv`: bias/RMSE/MAE for previous-state, propagated-state,
  total forecast and oracle errors over whole area, current hot area, previous
  hot area, observed-censored area and source-active area.
- `region_budgets.csv`: corresponding mean source, needed residual and
  propagation contributions; source-active regions use the existing threshold.
- `section_components.csv`, `point_components.csv`: all diagnostic values.
- `by_trajectory/*/components.npz`: local ignored full component arrays.
- `equality_checks.csv`: cached forecast max difference exactly zero; additive
  identities close within 5.7e-14 K, with frozen coefficients and hashes.
- `source_fingerprints.json`: driver/helper/operator/configuration and input
  cache/section hashes. `verification.json`: scope and no-inference/no-fit flags.

Reproduce from the repository with existing caches available:

```sh
env MPLCONFIGDIR=/tmp/thermal_mpl OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/55_thermal_native_forecast_attribution.py
```

Existing outputs and model implementations are preserved unchanged.
