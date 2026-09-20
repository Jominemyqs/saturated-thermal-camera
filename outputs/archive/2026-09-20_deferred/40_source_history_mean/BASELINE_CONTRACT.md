# Baseline reproduction contract

Reference: Experiment 38, `uncalibrated`, `level_index == 0` in its saved results.
Architecture: posterior physics mean plus sequential stationary stochastic-heat
covariance, using the full observed-censor mask with stride-5 unsaturated data.
This is not the sparse current-observation architecture of Experiment 37 or the
historical short-chain architecture of Experiment 30.

## Information and numerics

- Target: last recorded frame on the same 61 x 41 surface grid.
- Previous posterior: ambient-mean censored RBF; sampled censored pixels and
  pinned noisy unsaturated pixels (hybrid, not a complete latent posterior).
- Mean: inherited nonnegative temperature excess, zero-excess boundary,
  Gaussian-filter diffusion, cooling, and thresholded current source input.
- Covariance: one-step innovation plus propagated n-1 posterior covariance.
- Current likelihood: current noisy/censored observations only.
- Camera noise: 0.25 K; previous likelihood noise: 0.25 K; current likelihood
  noise: 0.5 K. These distinct inherited values are intentional.
- Reflected HMC: four chains, 400 retained draws, 100 burn-in, thinning one;
  inherited retry 800/300 if R-hat exceeds 1.05. Evaluation balances 300 draws
  from each current chain.
- Measurement, previous-inference and current-inference seeds retain Experiment
  38's trajectory-index offsets. Earlier starting frames use explicit additional
  seed offsets without changing current or n-1 observations.
- Exact top 1%: the same 25 truth-ranked pixels across all history lengths.
- CRPS: the inherited M(M-1) estimator, with its independent-draw unbiasedness
  interpretation; retained MCMC draws are not assumed mathematically independent.

Physical parameters, ceilings, source width and ambient preparation are
inherited. In particular, the reference camera ceilings originated in the
simulation experiment; this comparison does not establish deployment estimation
of those quantities from censored measurements. No held-out fitting is added.

## Controlled change

For L=1,2,4,8, infer the starting state at n-L and replay the source at each
intervening timestamp. The current likelihood and complete one-step covariance
arrays remain identical. L>1 is therefore a deterministic-mean sensitivity with
a frozen one-step discrepancy, not an L-step uncertainty propagation scheme.
The n-1 observations continue to inform covariance in these rows, and the n-L
observations inform the mean. There are no intermediate likelihood updates.

The one-step mean must be exactly equal to the inherited mean. The representative
development case additionally checks full posterior draws and summaries against
the existing reusable runner. All 30 held-out one-step rows must match ten saved
Experiment 38 metrics to absolute tolerance 1e-8 before the run can complete.

`baseline_equality_audit.csv` records the comparisons;
`implementation_checks.csv` records observation and covariance hashes;
`fixed_configuration.csv` and `source_fingerprints.json` preserve settings and
code/data fingerprints. Source-history windows are reported in seconds as well
as frame steps.

## Source-coupling provenance

The inherited gamma is approximately 0.170223. Its calibration runs through
`scripts/18_thermal_two_frame_ablation.py:calibrate_source_couplings` and
`scripts/17_thermal_spatiotemporal_physics.py:build_one_step_physics_components`.
That calibration propagates a clipped predecessor, then fits the source term
against development simulation truth. Gamma can therefore absorb missing heat
from predecessor clipping and other forecast discrepancies as well as physical
source conversion. It is an effective one-step coefficient, not an independently
measured material property. Its repeated use in an open-loop rollout is precisely
the transfer assumption tested here.
