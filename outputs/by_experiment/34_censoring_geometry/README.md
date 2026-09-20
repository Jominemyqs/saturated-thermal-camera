# Geometry of observed censoring

All connected components, boundary depths, and local pairs are constructed only from the noisy observed censoring mask `S_obs = {Y == c}`. Hidden truth is used only after construction for evaluation.

Boundary depth is the four-neighbor erosion layer: boundary pixels have depth zero, and each inward erosion layer adds one camera-grid pixel.

## Stage 1: diagnostic gate

- Observed-censored pixels: 2414.
- Connected components: 143; informative multi-depth components: 31.
- Pooled rho(depth, true exceedance): 0.478.
- Pooled rho(depth, current-GP exceedance): 0.646.
- Correlation gap: -0.168.
- Direct predicted-vs-true exceedance slope among truly above-ceiling pixels: 0.028.
- Median within-component rho(depth, true temperature): 0.433.
- Median within-component rho(depth, current GP mean): 0.663.
- Positive within-component correlations: 96.8%.

The predeclared gate was not passed. It required a rho gap of at least 0.1, median component rho of at least 0.2, and at least 20 informative components.

## Decision

A positive morphology signal exists in the latent truth, but the converged standard RBF censored posterior reflects that ordering at least as strongly. The defining Stage-2 condition, namely that rho(depth, truth) materially exceed rho(depth, GP), is therefore not satisfied.

This does not mean that the GP recovers exceedance magnitude well. The direct predicted-versus-true exceedance slope remains small. The diagnostic distinguishes a captured spatial ordering from the still-unresolved magnitude-compression problem.

Stage 2 was not run in the canonical experiment. Consequently there is no claim that geometry-derived preferences improve reconstruction over standard censoring or the matched random-order control.

The retained experiment-33 posterior used 120 burn-in steps and gave a smaller depth correlation. Longer independent chains changed its censored-pixel posterior means materially; `legacy_short_chain_convergence_audit.csv` records that discrepancy. This geometry diagnostic uses four independent chains with 1,200 burn-in steps and 700 retained draws per chain.

`mask_signal_robustness.csv` and `mask_signal_robustness_summary.csv` report the truth-side morphology diagnostic across four censoring fractions and three noise levels. These checks use truth only for post-construction evaluation, never to form the observed-mask geometry.
