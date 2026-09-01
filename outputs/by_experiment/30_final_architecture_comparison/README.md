# Final thermal-camera architecture comparison

This directory consolidates the existing frozen models after the covariance-scale audit. No model or hyperparameter was invented or tuned here.

## Protocol

- 33 total trajectories: 3 development and 30 equally weighted held-out trajectories.
- Synthetic 3% censoring, stride-5 camera mask, 0.25 K measurement noise, and 0.5 K likelihood noise.
- Identical trajectory-index observation and sampler seed convention.
- Unbiased empirical CRPS using `M(M-1)`.
- Existing model-specific covariance amplitudes are preserved in the main table.

## Architecture audit

The requested posterior-physics sequential advective-ST row and the requested moment-matched sequential advective-ST row are the same implementation: `C + B Sigma B^T`. They are represented once.

The original RBF marginal SD is `3.431 K`; the one-step stochastic innovation SD is `0.716 K`. This difference is why the scale-control table is required before interpreting RBF-vs-ST coverage as a geometry result.

## Files

- `architecture_comparison.csv` and `.md`: seven distinct complete architectures and the direct scientific answers.
- `rbf_st_scale_control.csv`: original RBF, variance-matched RBF, moment-matched sequential advective ST, and posterior-sample mixture.
- `propagation_ablation.csv`: strict mean-only, moment-matched, and censored-region posterior-sample propagation.
- `whole_field_vs_tail.png`: overall versus top-1% CRPS.
- `tail_coverage_vs_width.png`: top-1% coverage versus interval width, including the matched-RBF diagnostic.
- `reconstruction_comparison.png`: tail-enhanced shared-scale reconstruction panel for all seven architectures on `SpiralScanPath_13`.
- `reconstruction_comparison_linear.png`: the same fields with the original shared linear color scale.
- `reproduction_checks.csv`: rerun agreement with the frozen historical outputs.

## Reuse and reruns

The five current-only/sequential Gaussian rows and the legacy joint row were rerun only because their historical outputs did not save all requested regional posterior diagnostics. The posterior-sample mixture and variance-matched RBF diagnostics were reused. The largest absolute rerun difference in a saved metric is `1.137e-13`.

The main table contains `7` distinct architectures and the scale control contains `4` rows.