# Spatial sample readiness audit

## Finding

The existing fast prediction arrays are marginally valid conditional samples but
are **not complete joint latent-field samples**. Do not use them unchanged for
spatial sample propagation or importance-reweighted smoothing. No sampler or
historical result has been changed in this audit.

## Code paths

- `src/pseudo_censoring_runner.py:_current_prediction` calls
  `src/dense_censored_gp.py:sample_censored_gaussian_blocks`.
- The latter samples censored noisy observation values jointly, computes
  conditional means, then adds independent Gaussian noise at prediction pixels:
  `draws = conditional_means + rng.normal(size=conditional_means.shape) * sqrt(conditional_variance)`.
- The API receives only the prior prediction diagonal, not the full K_pp.
  Thus it omits off-diagonal entries of
  `C = K_pp - K_po (K_oo + noise + jitter)^(-1) K_op`.
- Correlation induced by the sampled conditional means is retained. It is not
  correct to describe all output pixels as independent; the missing part is the
  within-component conditional spatial covariance C.
- Previous inference goes through `infer_previous_censored_posterior`,
  `sample_multiple_chains`, and `sample_censored_ess_fast`. This fast RBF path
  has the same independent conditional-noise construction. Furthermore, the
  previous field keeps unsaturated pixels fixed at their noisy observations.
  It is a hybrid representation, not a full latent posterior.
- `sample_censored_dense_prior` already contains a full conditional Cholesky
  draw and is a useful implementation reference, but uses a different sampling
  path. It was not substituted here, since that would invalidate the controlled
  bridge and requires its own convergence/equality checks.

## Mathematical consequence

The intended covariance of a conditional mixture is

    Cov(conditional means) + C.

The fast draw representation has instead

    Cov(conditional means) + diag(diag(C)).

This distinction does not by itself invalidate pointwise RMSE, pointwise CRPS,
or pointwise intervals. It does matter for spatial averages, maxima of draws,
and propagation through an operator A: the desired A C A^T is generally not
equal to A diag(diag(C)) A^T. Consequently, convergence alone does not certify
the spatial uncertainty propagated from the hybrid previous representation.
The size and direction of that practical effect have not been tested here.

## Verification and persistence

`tests/test_observation_bridge.py:test_block_sampler_is_marginal_not_joint`
uses a valid Gaussian example with two targets of unit variance and covariance
0.8, independent of an observed third coordinate. The legacy block sampler
reproduces unit marginal variances but approximately zero target cross-covariance.
This is an audit test documenting current behavior, not a claim it is the
desired spatial behavior.

The inspected Experiment37/38 drivers retain score/diagnostic tables, not a
complete saved full-grid joint posterior ensemble. Experiment40 likewise does
not save a joint current-state ensemble. Rerunning their fast samplers would
not repair the missing covariance.

Before smoothing: generate coherent spatial current draws with an explicit
full conditional covariance, verify their marginal agreement against the
frozen baseline, and distinguish that change from any future-data benefit.
Previous hybrid uncertainty is a separate inherited approximation and must
remain explicitly labeled. This audit implements neither change.
