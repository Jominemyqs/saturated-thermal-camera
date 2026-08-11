# Sequential advection and stochastic forcing

> **Superseded calibration.** This experiment used a hot-coverage constraint
> followed by top-1% CRPS to select a 32x forcing intensity. It is retained only
> as an audit record. The replacement independently tunes the forcing
> lengthscale and amplitude on the three development trajectories, then
> evaluates a frozen strict sequential filter in
> `../23_strict_sequential_spde/` using
> `scripts/25_thermal_strict_sequential_spde.py`.

Posterior-mean propagation is fixed as the deterministic baseline. The forecast is

```text
mu_n^- = A_v mu_(n-1) + gamma q_n dt,
Sigma_n^- = A_v Sigma_(n-1) A_v^T + Q_dt,
Q_dt = integral_0^dt S_v(s) Q S_v(s)^* ds.
```

The current censored frame is used once in the update. Previous observations enter only through the previous censored posterior, avoiding a two-frame likelihood that would reuse them.

For homogeneous transport and stationary RBF Q, equal-time Q_dt is invariant to translation. Advection changes the transported previous mean and nonstationary covariance, while W_Q supplies new uncertainty independently.

The selected forcing-intensity multiplier is `32`, chosen on the three development trajectories from [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] with minimum required hot coverage 0.90.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion mean + RBF | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 |
| advective mean + RBF | 0.419 | 0.727 | 2.281 | 7.678 | 0.946 |
| diffusive stochastic forecast | 0.432 | 0.850 | 2.206 | 7.610 | 0.963 |
| advective stochastic forecast | 0.428 | 0.850 | 2.242 | 7.674 | 0.963 |

Paired trajectory-level effects are in `paired_comparisons.csv`; forcing selection is in `forcing_sensitivity_summary.csv`.

All 66 stochastic-prior checks are symmetric and positive semidefinite. Minimum tested eigenvalue: `1.565e+01`; maximum diagonal error: `7.105e-15`.

Files include `results.csv`, `heldout30_overall.csv`, `family_summary.csv`, `paired_comparisons.csv`, `forcing_sensitivity*.csv`, `prior_validation.csv`, `comparison.png`, `forcing_sensitivity.png`, and representative `reconstruction_*.png` files.

alpha = 3.610913e-06 m^2/s; beta = 12.422902 1/s; gamma = 0.170223.
