# Formulation and next control

Inspection as of 2026-11-08. Equations below distinguish executed code from
unimplemented state-space designs. All temperatures/temperature excesses are
in K. Boundary flux is in W/m2; neither flux nor its area integral is a
temperature increment.

## 1. Executed surface formulation

Let x be the native surface temperature vector, a the inherited ambient,
and D_j the admitted observations from frame j. Output 60 computes an
approximation to q(x_{n-1} | D_{n-1}) under a constant-mean RBF prior, not a
history-informed p(x_{n-1} | D_{0:n-1}, g_{1:n-1}).

For the completed noisy camera vector Z, define

\[
P=(K_{\rm RBF}+vI)^{-1},\qquad
C=vI-v^2P,\qquad W=C_{:,S}/v,
\]

where S contains censored locations and v includes previous likelihood noise
variance and the inherited numerical jitter. Conditional on Z,

\[
E[x\mid Z]=Z-vP(Z-a\mathbf1),\qquad
\Sigma_{n-1}=W\widehat{\operatorname{Cov}}(Z_S\mid D_{n-1})W^T+C.
\]

Unsaturated readings condition this distribution; they are not pinned in the
all-readings-latent arm. C includes spatial and unsaturated-censored cross terms.
The censored-camera covariance is estimated from MCMC; this is not an assertion
of exact sampling. See [previous_unpinning_audit.py](../../src/previous_unpinning_audit.py#L14),
`noisy_precision`, `conditional_camera`, `latent_from_camera`, `mixture_moments`
(lines 14-94), and [driver 64](../../scripts/64_thermal_previous_unpinning_audit.py#L196),
`main` (case conditioning and sampling, lines 196-220).

Write N for native-to-fine-grid interpolation, E for evaluation back to native
nodes, and G for the zero-extended Gaussian diffusion/cooling map. The executed
mean uses

\[
\mathcal P^+(r)=E G_{\Delta t}\max(Nr,0),\qquad A=E G_{\Delta t}N,
\]
\[
m_n^{68}=a\mathbf1+\mathcal P^+(\mu_{n-1}-a\mathbf1)+H b_n,
\qquad K_n^-=Q_{\Delta t}+A\Sigma_{n-1}A^T.
\]

This is a Gaussian moment-based predictive prior with a nonlinear floored mean
and a linear covariance map; it is not exact propagation of the full posterior
through the floored operator. `MatchedHeatOperator.propagate/mean`,
[lines 47-97](../../src/matched_projection.py#L47), implements N/G/E and the floor.
Native inputs therefore do not imply interpolation-free propagation.

`propagated_blocks`, [lines 9-47](../../src/unpinning_current_update.py#L9),
retains the full inherited covariance in required blocks. `prepare`,
[driver 65 lines 111-172](../../scripts/65_thermal_unpinning_current_update.py#L111),
adds Q from `finite_step_innovation_covariance`,
[lines 123-150](../../src/stochastic_heat_gp.py#L123). This Q is the old surface
stochastic heat innovation, not derived from the original volume FEM or from
uncertain laser power. Output 68 reuses it unchanged.

Only the current retained observations enter the final likelihood:
numerical Normal(y_i | x_i, 0.50^2), or saturated
Phi((x_i-c)/0.50), once per selected location. The frozen selection is numerical
candidates union reference observed saturation. Camera generation SD is 0.25 K;
previous likelihood SD is 0.25 K. These are distinct from posterior SD and jitter.
See [driver 65 lines 140-153](../../scripts/65_thermal_unpinning_current_update.py#L140),
[driver 72 `sample`, lines 176-215](../../scripts/72_thermal_physical_source_response.py#L176),
and [Output 68 configuration](../../outputs/by_experiment/68_physical_source_response/frozen_configuration.json).
No extra unused-node upper bounds are included. Final prediction draws are
pointwise marginals, not a joint-state posterior for recursive filtering.

## 2. Original volume state and the replaced component

Set u=T-T_0 with notebook T_0=293.15 K. After eliminating fixed bottom DOFs,
M is the heat-capacity mass matrix and K the conduction matrix. On the remaining
DOFs the constant-material backward-Euler model is

\[
u_n=F u_{n-1}+b_n,\qquad
F=(M+\Delta t K)^{-1}M,\qquad
b_n=(M+\Delta t K)^{-1}\Delta t f(g_n),\qquad x_n=T_0\mathbf1+H u_n.
\]

H selects native top-surface DOFs. Zero bottom excess is essential to this form.
These symbols describe solves; no dense F was assembled in this audit.
The code's equivalent form is M/dt+K. `build_problem` and `solve_problem`,
[fem_replay.py lines 55-137](../../src/fem_replay.py#L55), implement the weak form,
top load and bottom constraints. The notebook starts with uniform T_0, has
adiabatic natural side conditions, and no separate volumetric beta cooling term.
The step/export interval is 0.0005 s in these notebooks.

Output 68 computes only b_n with zero initial increment. `solve_independent_response`,
[physical_source_response.py lines 19-55](../../src/physical_source_response.py#L19),
accepts geometry/times and prescribed notebook parameters, not temperatures or
HeatFluxZ. Its surface H b_n replaces the old increment at prediction AND
observation nodes; old covariance blocks are untouched. Linear superposition
applies to this fixed-material, fixed-boundary discretization, not automatically
to a nonlinear or temperature-dependent physical model.

The inherited ambient a is not exactly notebook T_0. Output 68 deliberately did
not replace it. Full projected-history preparation derives ambient and source
length information ([prepare_trajectory](../../src/thermal_posterior_physics.py#L79),
lines 79-160; [estimate_effective_diffusivity](../../src/diffusion.py#L91), lines 91-92).
A future independent formulation needs an explicit temperature reference and
parameter provenance; this audit does not silently change either.

## 3. Why a surface state may require memory or an internal state

Partition the free volume excess into surface h and unobserved subsurface d:

\[
\begin{pmatrix}h_n\\d_n\end{pmatrix}
=\begin{pmatrix}F_{ss}&F_{sd}\\F_{ds}&F_{dd}\end{pmatrix}
\begin{pmatrix}h_{n-1}\\d_{n-1}\end{pmatrix}
+\begin{pmatrix}b_{s,n}\\b_{d,n}\end{pmatrix}.
\]

An exact surface-only linear Markov operator A_s valid for all volume states
would require HF=A_s H, equivalently ker(H) contained in ker(HF).
That condition has NOT been tested numerically here. In general, states with
the same surface and different subsurface heat can yield different next
surfaces. Eliminating d introduces history dependence unless a closure is
justified. This does not imply that depth is wholly unobservable from a time
sequence of surface measurements; observability depends on the dynamics,
sampling, censoring and allowed uncertainty.

The true-surface oracle does not supply d_{n-1}. Output 67 used the true full
volume for attribution; Output 68's candidate did not. Their comparison does
not alone select a low-dimensional state, establish an observability rank,
or quantify the fraction of remaining error attributable only to depth.

## Bounded follow-up design

**Proposal only; initialization and uncertainty choices must precede execution.**

| Arm | State/information | Role |
|---|---|---|
| Historical comparator | Frozen Output 68, frame-pair surface inference and old propagation | Preserve its established scores; not a matched test of history alone |
| History-only control | One specified state model, admissible initialization, known laser history; no camera assimilation | Establish how much the forward/input information already predicts |
| Matched camera update | Exactly the history-only model, initialization, parameter and discrepancy priors, plus the declared camera window | Isolate additional value of censored camera assimilation |

For a linear model, an informative bookkeeping option is u_n=u_n^forced+delta_n,
where u^forced follows known input history from a specified reference initial
state, and delta represents uncertain initial/internal heat and model error.
This is an exact decomposition for the full fixed linear model, not an endorsed
low-rank approximation or a newly fitted discrepancy. Exact known cold start
and exact model can make delta identically zero. Do not add arbitrary uncertainty
or use target truth to manufacture an inverse problem.

Before these arms can be implemented, specify cold-start versus warm-start,
history window, permitted observations, initial joint state distribution and
which physical quantities are uncertain. Never initialize a late-time candidate
from the stored true volume. Do not learn a depth basis from held-out or target
truth and present it as input-only. A reduced basis/closure needs its own declared
construction and validation; no choice is made here.

For a proposed linear augmented/reduced state z with dynamics F_z, a consistent
Gaussian prediction would use P_n^-=F_z P_{n-1} F_z^T+Q_z and surface blocks
H_z P_n^- H_z^T. Retain surface-internal cross covariance. Q_z and initial P_0
must have an explicit scientific justification; the old Q is not automatically
the correct new-state discrepancy. Nonlinear states require an explicitly
documented approximation. Parameter uncertainty, if introduced later, must
also be propagated rather than assumed captured by Q. Frozen old covariance
is permissible only as a labeled mean-only screening control, not proof of
consistent new-state UQ.

Keep the two development cases and reference/320/325 K, identical observations
between matched arms, fixed physical-area/shoulder regions and paired seeds.
Separate forecast from posterior, inferred from oracle, point error from
coverage/width, and history-only from observation-conditioned results. Do not
add mask flags, retune gamma/cooling/noise, or expand to held-out data here.
Reliability/runtime budgets must be prospectively defined for the chosen state
implementation; the existing per-fit budget is not an estimate for a rollout.

Existing response metadata records 26,414 volume nodes, 130,664 tetrahedra and
6,275 surface nodes. A full dense double-precision covariance at 26,414 nodes
alone is about 5.20 GiB, before factorization or work arrays. This is a storage
calculation, not a runtime benchmark; boundary elimination reduces dimension.
Neither a cheap deterministic FEM response nor the present marginal-GP timing
establishes that full-volume probabilistic filtering is affordable.

**Stop condition for design:** if actual initial/input knowledge and a defensible
uncertainty model cannot be specified, report that ambiguity to Adrienne before
implementing a depth model. Output 68 remains a candidate component, not a new
universal baseline.
