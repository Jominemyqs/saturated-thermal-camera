# Native B/C signed errors and uncertainty sections

Requested after the October meeting: distinguish the spatial sign of error and
inspect the profile together with its uncertainty. These plots use saved output
49 native-input reconstructions; no inference, calibration or model change is
needed to produce them.

## Reading the figures

- `signed_errors_*`: physics forecast, B posterior mean, C posterior mean;
  error = estimate minus native truth. Red is positive, blue is negative.
  All six panels across both trajectories use the same symmetric, zero-centered
  linear scale. Black contours show the observed censoring mask, not truth-based
  saturation. Dashed lines locate the cross-sections.
- `cross_sections_*`: along/across the displacement of the area-weighted,
  thresholded HeatFluxZ centroid. This is a source-motion PROXY, not a confirmed
  commanded laser velocity. The exact HeatFluxZ definition is still unresolved.
- Sections pass through the true current peak for diagnostic visualization only.
  Truth-selected sections never affect inference, hyperparameters or transport.
- Each curve uses the nearest actual native nodes to the geometric section.
  B/C use exactly the same node IDs. Truth, deterministic forecast, posterior
  mean, marginal 95% intervals and camera ceiling are shown together.
- Connecting lines and shaded bands are visual guides between those nodes.
  They are NOT newly inferred off-mesh credible intervals or simultaneous bands.
  No interpolation of independent marginal samples is used to manufacture a
  spatial posterior. CSV files preserve node IDs, coordinates and normal offsets
  from the ideal section so coarse/background sampling is transparent.
  Maximum normal offsets are 0.235 mm (Diagonal) and 0.154 mm (Spiral), so these
  should be read as nearest-node transects, not exact continuous straight cuts.

## What the maps establish

Positive spatial-average bias does not mean positive error everywhere. Native B/C
underpredict on about 32-37% of physical area in these two cases. The physics
forecasts themselves have negative errors on about 36-37% of area. Very small
background signed errors are included in these sign counts; they are descriptive,
not counts of materially wrong locations.

The peak-section plots show an exaggerated sharp peak and an underpredicted
shoulder. C often has narrower intervals than B and misses parts of these
profiles. These are inferred-state physics forecasts, NOT the separate
true-previous-state oracle. They do not by themselves identify missing vertical
heat transfer as the cause.

Reproduce without new inference:

```sh
.venv/bin/python scripts/54_thermal_native_advection.py --diagnostics-only
```

All baseline arrays and prior output-49 results are retained unchanged.
