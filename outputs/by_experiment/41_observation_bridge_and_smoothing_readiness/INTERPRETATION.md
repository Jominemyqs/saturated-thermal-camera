# Bridge and readiness conclusions

## Observation-set bridge

All 33 trajectories load and run; the summary uses the same 30 held-out paths.
The stationary sequential mean/covariance, previous posterior, camera noise,
ceiling, exact hottest-25 mask and HMC settings/seeds are fixed. Sparse blocks
are indexed from the richer prior, not independently fitted. All-censored
baseline scores reproduce Experiment38 to within 1.3e-12.

Relative excess-field error:

- Historical Experiment37 sparse: 0.457218.
- Controlled HMC sparse: 0.456412.
- Controlled HMC stride-5 unsaturated + all censored: 0.516370.

The observation-set effect is +0.059958; the remaining historical protocol
effect is -0.000806. The mask explains the dominant change, not every digit.
Historical differences include ESS versus HMC, retained counts, previous
sampling, and analytic versus empirical mean/pooling. The residual has not
been causally partitioned among those differences.

Adding observed-censored locations raises field error in 30/30 trajectories
(7 diagonal, 11 horizontal, 12 spiral). This is not evidence that information
is intrinsically harmful: the prior/mean and inference are approximate, and
the new inequalities shift the fitted field. Top1 bias changes from -0.106 K
to +0.432 K. Keep the richer protocol as the explicit common baseline, rather
than selecting observations for the smaller error.

## Cooling visibility

Nine fixed interior targets were inspected: three calendar positions on each
of three development trajectories. Selection does not use truth or model
performance. The original per-trajectory reference ceiling is constant across
all frames. Synthetic noise is generated once; masks use observed Y == c.

Persistent visibility means three consecutive future observations below
c - 2 sigma_measurement. At 4 frames (2 ms), only 0--2.7% of initial censored
pixels meet this criterion across the nine targets, and none of those lie
on the stride-5 observation grid.

At 64 frames (32 ms), mean persistent fractions across each family's three
targets are 47.3% diagonal, 46.8% horizontal, 28.3% spiral. However, only 2.2%,
0.5%, and 2.0% of initial censored pixels respectively are both persistently
visible AND retained by stride-5 unsaturated sampling.

L=1,2,4 consecutive frames is therefore a weak test of the proposed cooling
mechanism. Future lag selection should be in elapsed time, not just frame
count. A pilot can keep an interior calendar target such as the midpoint and
study 16--64 ms visibility without selecting held-out examples by success.
This is not yet a recommendation to roll the existing approximate dynamics
forward 128 times; transition reliability at those lags is unresolved.

Full-camera visibility and information admitted to the likelihood differ.
A future experiment that retains newly unsaturated values at formerly
censored locations changes observation policy and must be explicit and
matched across its controls. No such policy change was implemented here.

## Sample readiness

The fast current sampler omits within-component spatial cross-covariance;
the previous sampler has the same approximation plus pinned unsaturated
values. See SAMPLE_AUDIT.md and the Gaussian counterexample unit test.
Thus the archived arrays are not ready for direct spatial particle smoothing.
This does not automatically invalidate their pointwise scores, but covariance
propagation can be affected and the effect has not been isolated here.

Before fitting an interior-frame pilot, also replace or explicitly declare
the inherited full-history ambient/source-scale preparation. It must not be
silently presented as causal filtering based only on past observations.

No smoothing, reweighting, new kernel, calibration, or held-out tuning was done.
