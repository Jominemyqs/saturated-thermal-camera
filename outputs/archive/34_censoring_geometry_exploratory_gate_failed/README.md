# Geometry of observed censoring

> **Archived noncanonical diagnostic.** This preference run was launched from
> the preliminary experiment-33 posterior before its short-chain convergence
> issue was identified. The converged Stage-1 diagnostic fails the predeclared
> gate, so these Stage-2 numbers must not be presented as a project result.
> The canonical output is `outputs/by_experiment/34_censoring_geometry`.

All connected components, boundary depths, and local pairs are constructed only from the noisy observed censoring mask `S_obs = {Y == c}`. Hidden truth is used only after construction for evaluation.

Boundary depth is the four-neighbor erosion layer: boundary pixels have depth zero, and each inward erosion layer adds one camera-grid pixel.

## Stage 1: diagnostic gate

- Observed-censored pixels: 2414.
- Connected components: 143; informative multi-depth components: 31.
- Pooled rho(depth, true exceedance): 0.478.
- Pooled rho(depth, current-GP exceedance): 0.546.
- Correlation gap: -0.068.
- Median within-component rho(depth, true temperature): 0.433.
- Positive within-component correlations: 96.8%.

The predeclared gate was not passed. It required a rho gap of at least 0.1, median component rho of at least 0.2, and at least 20 informative components.

## Stage 2: local preference likelihood

The geometry model uses only neighboring observed-censored pixels in the same component and orients an edge from lower to higher observed-mask depth. The random control uses the exact same undirected edges with random orientation. The softness was fixed before held-out evaluation as `tau = 4 * measurement_noise_sd`; no truth was used for pair construction or tuning.

| Method | Field L2 | Saturated RMSE | Saturated MAE | Top-1% CRPS | Peak bias | Censored rank rho |
|---|---:|---:|---:|---:|---:|---:|
| standard censored GP | 0.764 | 5.215 K | 3.047 K | 3.384 K | -33.100 K | 0.446 |
| matched random local preferences | 0.808 | 5.426 K | 2.960 K | 3.837 K | -32.091 K | 0.059 |
| observed-mask geometry preferences | 0.771 | 5.430 K | 3.448 K | 3.708 K | -29.811 K | 0.521 |

## Interpretation

Geometry changes saturated-region RMSE by +0.215 K relative to standard censoring (6/30 trajectory wins) and by +0.004 K relative to the matched random control (16/30 wins).

Its censored-region rank correlation changes by +0.075 relative to standard censoring. Improvements that appear for geometry but not random ordering support a genuine morphology effect; similar changes for both controls would instead indicate generic regularization.

## Robustness

`mask_signal_robustness.csv` varies censoring fraction and measurement noise on the held-out heat trajectories. `synthetic_robustness.csv` uses the existing matched Gaussian, rotated Gaussian, two-component wake, and skewed-wake generators.

Geometry improves synthetic saturated-region RMSE over standard censoring on 1/4 field families. This check is descriptive and was not used to tune tau.
