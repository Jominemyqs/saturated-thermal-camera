# Fixed two-frame thermal GP ablation

The main model uses the immediately previous frame, the current censored frame, a one-step physics forecast mean, and a short-lag space-time heat kernel for the residual. The three original trajectories are used only to select one global residual configuration; the other 30 trajectories are marked as evaluation cases.

The four-frame version is retained only as a sensitivity comparison. The ablation isolates the snapshot residual, temporal covariance, physics mean, and their combination. Synthetic ceilings are trajectory-specific truth quantiles at the requested censoring fractions.

Selected sensitivity setting: `noise x2 + length x1.25` with multipliers `{'signal_multiplier': 1.0, 'noise_multiplier': 2.0, 'beta_multiplier': 1.0, 'length_multiplier': 1.25}`.

Development diffusivity: 3.610913e-06 m^2/s; cooling rate: 12.4229 1/s; residual signal scale: 3.4305 K.

Transferred source couplings by ceiling: 1%: 0.15766, 3%: 0.17022, 5%: 0.17478, 10%: 0.18245.

Development interval-inflation factors: snapshot RBF: 4.251, space-time heat: 10.937, physics mean + RBF: 2.544, physics mean + space-time: 5.948. Raw and calibrated interval metrics are both retained.

## Findings

Two frames outperformed four on the development trajectories: mean field error 0.337 versus 0.370, and hot-region CRPS 1.220 K versus 1.537 K.

Across the 120 held-out trajectory/ceiling cases, the physics mean plus snapshot RBF beat the snapshot-only field error in 119/120 cases, and the combined model did so in 120/120 cases. At 3% censoring, the physics-mean snapshot and combined field errors were 0.467 and 0.421; their hot-region CRPS values were 1.449 K and 1.336 K.

The snapshot residual remained better calibrated before correction. At 3% censoring, raw hot-region coverage was 0.724 for the snapshot residual and 0.649 for the space-time residual. Development-derived interval scaling raised the corresponding held-out coverages to 0.940 and 0.939. Peak coverage remains substantially harder than pointwise hot-region coverage.

Files:

- `residual_sensitivity_results.csv` and `residual_sensitivity_summary.csv`
- `history_length_sensitivity.csv`
- `trajectory_ablation_results.csv`
- `trajectory_family_summary.csv`
- `ablation_by_ceiling.csv` and `ablation_table_3pct.csv`
- `interval_calibration_scales.csv`
- `residual_sensitivity.png`, `family_ablation.png`, and `uncertainty_calibration.png`
