# 15 Mean Advection

All models use the same fixed stride-only observations, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. The table reports the 30 held-out trajectories; trajectory-level files include all 33 trajectories.

Reference censoring fraction: `3%`. Regular source coupling: `0.170223`; advective-mean source coupling: `0.170342`.

The advective one-step mean is

```text
m_n(x)-T_amb = exp(-beta dt) [G_(alpha dt) * u_(n-1)](x-d_n)
                 + gamma_adv q_n(x) dt.
```

Both models use the regular space-time heat kernel. The advective source coupling is recalibrated on the same three development paths. This is a moving-frame correction, not literal material advection in the stationary solid.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| advective physics mean + space-time | 0.446 | 0.732 | 2.714 | 7.828 | 0.932 |

Paired changes use `physics mean + space-time` as the baseline. Negative changes are better for errors and CRPS; positive changes are better for coverage.

- `advective physics mean + space-time`: field `+0.0034`, all CRPS `+0.0006 K`, top-1% CRPS `+0.0011 K`, hot coverage `-0.0022`.

Files:

- `results.csv`: all trajectory-level results for this comparison.
- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.
- `paired_vs_baseline.csv`.
- `combined_vs_each_baseline.csv` or `source_amplitude_vs_each_baseline.csv` for the added model.
- `comparison.png`.

Diffusivity: `3.610913e-06 m^2/s`; cooling rate: `12.422902 1/s`; residual setting: signal `x1`, noise `x2`, beta `x1`, length `x1.25`.
