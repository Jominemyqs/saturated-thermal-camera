# Observation bridge and smoothing readiness

Only the stationary sequential model is tested. One previous posterior and one prior are built per trajectory; sparse observations are exact subblocks of the richer set. No smoother or covariance change is implemented.

| Protocol | Field error | Overall CRPS | Top1 RMSE | Top1 CRPS | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|
| sparse_stride5 | 0.456412 | 0.226645 | 3.170250 | 1.669339 | 0.842667 | 8.276749 |
| stride5_unsaturated_all_censored | 0.516370 | 0.241155 | 3.194982 | 1.738061 | 0.845333 | 7.368025 |

Largest Experiment38 reproduction difference: 1.22e-12.

Canonical observations: stride-5 unsaturated plus every observed-censored location (not all unsaturated camera pixels). Previous representation, physical coefficients, seeds, HMC convergence/retry settings and exact top-25 metric mask are frozen from Experiment38. Current likelihood sees current observations only.

Historical Experiment37 uses ESS instead of HMC for its current block sampler, a different draw pooling/mean estimator, and different retained sample settings. The controlled mask effect must be separated from its residual mismatch to the saved sparse row; see historical_sparse_lineage.csv.

Cooling readiness uses only three development paths. Targets are fixed at 25%, 50%, 75% of frame count, not selected using truth or performance. Noise is generated once per trajectory, and the inherited final-frame reference ceiling stays constant over time. Observed masks alone define censoring and visibility. Persistent visibility requires three consecutive observations more than two measurement SD below the ceiling. This guards against, but does not eliminate, noise flicker. All endpoint and ever-visible counts are also retained.

The stride5 visibility columns count only newly unsaturated locations that would actually enter the existing likelihood. Off-stride pixels can become observable in the camera but then drop out of the canonical inference subset. Camera-visible and assimilated information must not be conflated in a future smoothing test.

These preparation routines inherit simulator-based ambient and reference-ceiling provenance. Before a causal interior-frame experiment, ambient/source-width preparation needs an availability audit; no interior-frame GP was fitted here.

See SAMPLE_AUDIT.md for the joint-draw limitation. Do not feed current marginal draws into a spatial smoother as complete field samples.
