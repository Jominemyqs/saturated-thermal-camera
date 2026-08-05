# Two-frame space-time and physics-mean experiment

This is the primary run of `scripts/17_thermal_spatiotemporal_physics.py`. It
uses the final thermal frame and the frame 10 ms earlier.

The source-driven one-step mean diffuses the preceding clipped frame and adds
the current `HeatFluxZ` footprint. Its source coupling is calibrated on the
other two uncensored simulation trajectories, not on the held-out true field.

Main conclusion: the calibrated physics mean provides the large improvement.
Combining it with the two-frame heat kernel gives excess-field errors of 0.362,
0.399, and 0.294 on the diagonal, horizontal, and spiral trajectories. The
spiral true peak is covered; diagonal and horizontal peak intervals remain too
narrow.

Files:

- `spatiotemporal_physics_results.csv`: reconstruction and calibration metrics.
- `physics_mean_fits.csv`: held-out physics-mean results and transferred values.
- `source_coupling_calibration.csv`: per-trajectory source calibration.
- `spatiotemporal_physics_metric_comparison.png`: aggregate method comparison.
- `*reconstruction.png`: truth, clipped data, physics mean, temporal saturation,
  and posterior reconstructions.
