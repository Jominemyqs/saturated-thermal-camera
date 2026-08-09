# 17 Advective Stochastic Forcing

All models use the same fixed stride-only observations, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. The table reports the 30 held-out trajectories; trajectory-level files include all 33 trajectories.

Reference censoring fraction: `3%`. Regular source coupling: `0.170223`; advective-mean source coupling: `0.170342`.

The combined residual satisfies

```text
dr = (alpha Laplacian r - v(t) dot grad r - beta r) dt + dW,
E[dW(x,t)dW(x',t)] = q exp(-||x-x'||^2/(2 ell_W^2)) dt.
```

Its stationary covariance is evaluated in the moving coordinates x-s(t), so each forcing contribution is both diffused and shifted by s(t)-s(t'). The regular physics mean is unchanged. This isolates the interaction between residual transport and continuous stochastic forcing.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + RBF | 0.435 | 0.736 | 2.710 | 7.835 | 0.936 |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| physics mean + advective space-time | 0.441 | 0.718 | 2.688 | 7.799 | 0.933 |
| physics mean + forced space-time | 0.461 | 0.598 | 2.568 | 7.743 | 0.924 |
| physics mean + advective forced space-time | 0.446 | 0.573 | 2.553 | 7.695 | 0.922 |

Paired changes use `physics mean + space-time` as the baseline. Negative changes are better for errors and CRPS; positive changes are better for coverage.

- `physics mean + RBF`: field `-0.0070`, all CRPS `+0.0048 K`, top-1% CRPS `-0.0020 K`, hot coverage `+0.0022`.
- `physics mean + advective space-time`: field `-0.0015`, all CRPS `-0.0135 K`, top-1% CRPS `-0.0242 K`, hot coverage `-0.0009`.
- `physics mean + forced space-time`: field `+0.0184`, all CRPS `-0.1331 K`, top-1% CRPS `-0.1443 K`, hot coverage `-0.0101`.
- `physics mean + advective forced space-time`: field `+0.0039`, all CRPS `-0.1585 K`, top-1% CRPS `-0.1596 K`, hot coverage `-0.0118`.

Combined model compared with each component:

- versus `physics mean + RBF`: field `+0.0108`, all CRPS `-0.1633 K`, top-1% CRPS `-0.1576 K`, hot coverage `-0.0140`.
- versus `physics mean + space-time`: field `+0.0039`, all CRPS `-0.1585 K`, top-1% CRPS `-0.1596 K`, hot coverage `-0.0118`.
- versus `physics mean + advective space-time`: field `+0.0054`, all CRPS `-0.1450 K`, top-1% CRPS `-0.1354 K`, hot coverage `-0.0110`.
- versus `physics mean + forced space-time`: field `-0.0145`, all CRPS `-0.0254 K`, top-1% CRPS `-0.0153 K`, hot coverage `-0.0018`.

Files:

- `results.csv`: all trajectory-level results for this comparison.
- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.
- `paired_vs_baseline.csv`.
- `combined_vs_each_baseline.csv` or `source_amplitude_vs_each_baseline.csv` for the added model.
- `comparison.png`.

Diffusivity: `3.610913e-06 m^2/s`; cooling rate: `12.422902 1/s`; residual setting: signal `x1`, noise `x2`, beta `x1`, length `x1.25`.
