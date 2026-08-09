# 16 Stochastic Forcing

All models use the same fixed stride-only observations, noise realization, two-frame lag, residual hyperparameters, and censored posterior sampler. The table reports the 30 held-out trajectories; trajectory-level files include all 33 trajectories.

Reference censoring fraction: `3%`. Regular source coupling: `0.170223`; advective-mean source coupling: `0.170342`.

The forced residual satisfies dr = (alpha Laplacian r - beta r)dt + dW, where W is white in time and has an RBF spatial covariance. Its stationary covariance is evaluated with positive Gauss-Laguerre quadrature and normalized to the same marginal variance sigma_f^2 as the regular space-time kernel. No extra variance parameter is tuned. Dimensionally, sigma_f^2 is a field variance in K^2; the implied RBF forcing intensity q has units K^2/s and is saved in `results.csv` as `forcing_intensity_K2_per_s`.

For the two-dimensional kernel, q is chosen from

```text
sigma_f^2 = q * integral_0^infinity exp(-2 beta u)
            * ell_W^2/(ell_W^2 + 4 alpha u) du.
```

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics mean + space-time | 0.442 | 0.731 | 2.712 | 7.819 | 0.934 |
| physics mean + forced space-time | 0.461 | 0.598 | 2.568 | 7.743 | 0.924 |

Paired changes use `physics mean + space-time` as the baseline. Negative changes are better for errors and CRPS; positive changes are better for coverage.

- `physics mean + forced space-time`: field `+0.0184`, all CRPS `-0.1331 K`, top-1% CRPS `-0.1443 K`, hot coverage `-0.0101`.

Files:

- `results.csv`: all trajectory-level results for this comparison.
- `heldout30_overall.csv`, `all33_overall.csv`, and `family_summary.csv`.
- `paired_vs_baseline.csv`.
- `combined_vs_each_baseline.csv` or `source_amplitude_vs_each_baseline.csv` for the added model.
- `comparison.png`.

Diffusivity: `3.610913e-06 m^2/s`; cooling rate: `12.422902 1/s`; residual setting: signal `x1`, noise `x2`, beta `x1`, length `x1.25`.
