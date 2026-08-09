# 14 Residual Advection

All models use the same fixed stride-only observations, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. The table reports the 30 held-out trajectories; trajectory-level files include all 33 trajectories.

Reference censoring fraction: `3%`. Regular source coupling: `0.170223`; advective-mean source coupling: `0.170342`.

The advective residual kernel is

```text
k_adv = sigma_f^2 exp(-beta |dt|) ell^2/L^2
        * exp(-||x-x'-(s(t)-s(t'))||^2/(2 L^2)),
L^2 = ell^2 + 2 alpha |dt|.
```

The path s(t) is the HeatFluxZ-weighted source centroid. The physics mean is unchanged, so this stage isolates transport in the residual covariance.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + RBF | 0.435 | 0.736 | 2.710 | 7.835 | 0.936 |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| physics mean + advective space-time | 0.441 | 0.718 | 2.688 | 7.799 | 0.933 |

Paired changes use `physics mean + space-time` as the baseline. Negative changes are better for errors and CRPS; positive changes are better for coverage.

- `physics mean + RBF`: field `-0.0070`, all CRPS `+0.0048 K`, top-1% CRPS `-0.0020 K`, hot coverage `+0.0022`.
- `physics mean + advective space-time`: field `-0.0015`, all CRPS `-0.0135 K`, top-1% CRPS `-0.0242 K`, hot coverage `-0.0009`.

Files:

- `results.csv`: all trajectory-level results for this comparison.
- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.
- `paired_vs_baseline.csv`.
- `combined_vs_each_baseline.csv` or `source_amplitude_vs_each_baseline.csv` for the added model.
- `comparison.png`.

Diffusivity: `3.610913e-06 m^2/s`; cooling rate: `12.422902 1/s`; residual setting: signal `x1`, noise `x2`, beta `x1`, length `x1.25`.
