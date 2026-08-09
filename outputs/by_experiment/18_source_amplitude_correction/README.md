# 18 Source Amplitude Correction

All models use the same fixed stride-only observations, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. The table reports the 30 held-out trajectories; trajectory-level files include all 33 trajectories.

Reference censoring fraction: `3%`. Regular source coupling: `0.170223`; advective-mean source coupling: `0.170342`.

The source-amplitude model augments the advective-forced covariance by

```text
k_amp(z,z') = sigma_eta^2 b(z)b(z') exp(-|t-t'|/tau_eta),
b(x,t) = gamma_0 q(x,t) dt.
```

The selected fractional source SD is `0.250`. The three tested persistence values produced identical development metrics because no coarse observation landed inside the narrow source basis on the development paths. The reported `0.0005 s` is therefore a tie-breaking convention, not an identified timescale. The 30 held-out trajectories test whether the uncertainty correction transfers.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + RBF | 0.435 | 0.736 | 2.710 | 7.835 | 0.936 |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| physics mean + advective space-time | 0.441 | 0.718 | 2.688 | 7.799 | 0.933 |
| physics mean + forced space-time | 0.461 | 0.598 | 2.568 | 7.743 | 0.924 |
| physics mean + advective forced space-time | 0.446 | 0.573 | 2.553 | 7.695 | 0.922 |
| physics mean + advective forced + source amplitude | 0.446 | 0.572 | 2.493 | 7.702 | 0.935 |

Paired changes use `physics mean + advective forced space-time` as the baseline. Negative changes are better for errors and CRPS; positive changes are better for coverage.

- `physics mean + RBF`: field `-0.0108`, all CRPS `+0.1633 K`, top-1% CRPS `+0.1576 K`, hot coverage `+0.0140`.
- `physics mean + space-time`: field `-0.0039`, all CRPS `+0.1585 K`, top-1% CRPS `+0.1596 K`, hot coverage `+0.0118`.
- `physics mean + advective space-time`: field `-0.0054`, all CRPS `+0.1450 K`, top-1% CRPS `+0.1354 K`, hot coverage `+0.0110`.
- `physics mean + forced space-time`: field `+0.0145`, all CRPS `+0.0254 K`, top-1% CRPS `+0.0153 K`, hot coverage `+0.0018`.
- `physics mean + advective forced + source amplitude`: field `+0.0000`, all CRPS `-0.0007 K`, top-1% CRPS `-0.0603 K`, hot coverage `+0.0123`.

Source-amplitude model compared with each baseline:

- versus `physics mean + RBF`: field `+0.0109`, all CRPS `-0.1640 K`, top-1% CRPS `-0.2179 K`, hot coverage `-0.0018`.
- versus `physics mean + space-time`: field `+0.0039`, all CRPS `-0.1592 K`, top-1% CRPS `-0.2199 K`, hot coverage `+0.0004`.
- versus `physics mean + advective space-time`: field `+0.0054`, all CRPS `-0.1457 K`, top-1% CRPS `-0.1957 K`, hot coverage `+0.0013`.
- versus `physics mean + forced space-time`: field `-0.0144`, all CRPS `-0.0261 K`, top-1% CRPS `-0.0756 K`, hot coverage `+0.0105`.
- versus `physics mean + advective forced space-time`: field `+0.0000`, all CRPS `-0.0007 K`, top-1% CRPS `-0.0603 K`, hot coverage `+0.0123`.

Files:

- `results.csv`: all trajectory-level results for this comparison.
- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.
- `paired_vs_baseline.csv`.
- `combined_vs_each_baseline.csv` or `source_amplitude_vs_each_baseline.csv` for the added model.
- `comparison.png`.

Diffusivity: `3.610913e-06 m^2/s`; cooling rate: `12.422902 1/s`; residual setting: signal `x1`, noise `x2`, beta `x1`, length `x1.25`.
