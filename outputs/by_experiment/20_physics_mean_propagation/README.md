# Previous-frame censored-posterior propagation

This experiment tests Adrienne's suggestion that a saturated previous camera frame should not be propagated as if every clipped value were exactly equal to the ceiling.

The three methods use the same current-frame RBF residual and censored likelihood:

```text
A. clipped:        m_n = A y_(n-1)^clip + source_n
B. posterior mean: m_n = A E[T_(n-1) | y_(n-1)] + source_n
C. full posterior: same mean, with K_n = K_RBF + A Sigma_(n-1) A^T.
```

Here `A` is the linear diffusion-and-cooling operator. The full covariance is represented by propagated posterior draws as a positive low-rank kernel and is included before conditioning on the current censored frame.

All unsaturated previous-frame pixels retain their noisy measured values. The censored GP only corrects pixels identified as saturated, conditioned on all saturated inequalities and a stride-based unsaturated support set. This avoids discarding available camera pixels while keeping the censored inference tractable.

A current-frame RBF residual is used for all three methods. The previous frame is not also supplied to a space-time residual kernel, which would count the same observation twice.
The propagated frame is the immediately adjacent simulation frame by default, rather than the 0.01 s residual-covariance lag used in earlier space-time experiments.

## Held-out results

| Method | Field error | All CRPS (K) | Top-1% CRPS (K) | Peak error (K) | Hot coverage | Hot width (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clipped propagation | 0.461 | 0.731 | 2.701 | 7.833 | 0.937 | 12.167 |
| posterior-mean propagation | 0.424 | 0.728 | 2.243 | 7.618 | 0.950 | 12.231 |
| full-posterior propagation | 0.424 | 0.728 | 2.234 | 7.611 | 0.949 | 12.444 |

## Paired effects

Replacing clipped propagation by the previous posterior mean changes field error by `-0.0363`, all-domain CRPS by `-0.0031 K`, and top-1% CRPS by `-0.4580 K`. It lowers field error on `30/30` and top-1% CRPS on `30/30` held-out paths.

Adding propagated covariance on top of the posterior mean changes field error by `-0.0002`, all-domain CRPS by `+0.0003 K`, hot coverage by `-0.0009`, and hot interval width by `+0.2133 K`.

The previous-frame censored posterior itself reduces saturated-region MAE from `2.455 K` for clipped values to `2.116 K`, but its raw 95% coverage is only `0.594`. Its propagated covariance is therefore too concentrated to produce a substantial full-posterior gain. The next uncertainty experiment should calibrate this previous-frame posterior, not merely inflate the current intervals after propagation.

## Numerical checks

All 33 full-prior checks are symmetric and positive semidefinite. The minimum tested eigenvalue is `1.122e+01` and the largest diagonal consistency error is `1.776e-15`.

Files:

- `results.csv`: all 99 trajectory-model fits.
- `heldout30_overall.csv`: main aggregate results.
- `family_summary.csv`: family-specific results.
- `paired_comparisons.csv`: within-trajectory changes and win counts.
- `prior_validation.csv`: low-rank covariance checks and previous-frame diagnostics.
- `comparison.png` and `comparison_by_family.png`: aggregate plots.
- `reconstruction_*.png`: one representative reconstruction per family.

alpha = 3.610913e-06 m^2/s; beta = 12.422902 1/s; gamma = 0.170223.
