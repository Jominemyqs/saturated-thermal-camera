# State formulation audit after Output 68

2026-11-08. Inspection and mathematical design only. No new experiment number,
GP inference, FEM solve, parameter fitting, or baseline replacement.

## Main finding

**Output 68 supplies a physically computed current boundary response, but it
does not yet infer or evolve an internal thermal state using input history.**
Its predecessor is a surface GP conditioned on the previous frame. Its
propagation and stochastic covariance remain the older surface approximations.
This identifies a formulation gap worth investigating, not proof that a
particular depth model will fix reconstruction or uncertainty.

The [Output 68 report](../../outputs/by_experiment/68_physical_source_response/README.md)
already separates the evidence: the replacement improves the four 320/325 K
development cases, but worsens reference hot-area reconstruction. Even a true
surface predecessor leaves a propagation discrepancy. That discrepancy does
not uniquely identify missing depth: numerical mapping, boundary treatment,
effective coefficients and the surface-state approximation remain entangled.

## What is the current state?

| Layer | Executed formulation | What it does not establish |
|---|---|---|
| Previous inference, Output 60 | All previous-frame unsaturated readings and saturated flags condition a constant-mean spatial RBF GP; full latent covariance, including cross terms, is retained analytically | Not a filter assimilating all historical frames or laser inputs |
| Predictive mean, Output 68 | Old floored surface propagation of the previous mean, plus an independent one-step FEM boundary response | No subsurface state is estimated or carried between steps |
| Predictive covariance, Output 61 reused in 68 | Old finite-step innovation plus full propagated previous covariance | Not the covariance of a new 3D state model; no laser/material uncertainty added |
| Current update | Retained current numerical/censored likelihood, SD 0.50 K; no extra non-saturation flags | Current marginal draws are not a ready-made coherent volume state for another time step |
| Original FEM | Full-volume P1 temperature, implicit conduction, prescribed top flux, fixed bottom temperature | A known true surface alone is not the same initial state as the true volume |

See [the equations and exact code pointers](formulation_and_next_control.md).
Output numbers are not filenames: 60/61/68 use drivers 64/65/72 respectively.

## Information and initialization audit

| Quantity | Available independently of current temperature? | Status for a future predictor |
|---|---|---|
| Prescribed path, power, width, clock and endpoint flux `g` | Yes in the supplied notebooks | Can construct the entire input history without gradient centroids; experimental knowledge/absorbed-power interpretation still needs confirmation |
| Initial volume temperature | Notebook specifies uniform 293.15 K at t=0 | Legitimate independently specified synthetic cold start; not evidence that a real recording starts in this state |
| Material, geometry and boundary conditions | Specified in notebook; saved geometry used in replay | Independent of temperature in this generator; real uncertainty and boundary knowledge remain unspecified |
| Late-time subsurface temperature | Not measured by the surface camera | Must be predicted/inferred or assigned a justified prior, never initialized from saved true volume in a candidate |
| Previous and current camera observations | Available with their clipping/noise model | Current experiment uses a frame pair; a longer observation window would be an explicit new information budget |
| `HeatFluxZ` | No: projected derivative of the solved FEM temperature | Excluded from the new source worker; not an independent input-history measurement |
| Ambient/GP scales/effective propagation coefficients | Some inherited preparation/calibration uses simulation histories | Output 68 has not removed these dependencies; do not call the full pipeline camera-only |
| Truth, hot-area masks and oracle volume | Evaluation/attribution only | Cannot determine a candidate's state, basis, uncertainty or input alignment from target truth |

Evidence: original [mesh cell](../../outputs/by_experiment/66_prescribed_laser_input_audit/Diagonal_cell3_source.txt),
[initialization and weak-form cell](../../outputs/by_experiment/66_prescribed_laser_input_audit/Diagonal_cell6_source.txt),
[provenance table](../../outputs/by_experiment/66_prescribed_laser_input_audit/provenance.md),
and `prepare_trajectory` in [thermal_posterior_physics.py, lines 79-160](../../src/thermal_posterior_physics.py#L79).
The six saved Diagonal/Spiral cell extracts were checked against the original
notebooks in this audit. Output 67 subsequently verified the last-step replay;
Output 66's older statement that no FEM replay was attempted is scoped to 66.

## Smallest useful next comparison, not executed

Keep Output 68 as the historical comparator. Before another inference run,
define **one** input-history/state model and compare it without versus with
camera assimilation. Those two arms must share initialization, dynamics,
input history, covariance and model-discrepancy assumptions. Only their use of
camera information differs. This isolates assimilation; comparison against
Output 68 also changes dynamics/information and cannot isolate history alone.

**Important control:** exact known cold start + exact matching FEM + exact
laser history can in principle determine the synthetic trajectory without any
camera. Such a run is a forward-consistency control, not evidence that hidden
temperature reconstruction or uncertainty calibration has been solved.
Only the final transition has been replay-verified so far, not a complete rollout.

Before implementation, resolve three choices with Adrienne:

1. Start at a genuinely known cold state, or at an unknown warm state? Specify
   available past laser commands and camera frames, not a truth-initialized shortcut.
2. What uncertainty/discrepancy is scientifically justified under that scenario?
   Do not invent uncertainty merely to make the synthetic inverse problem nontrivial.
3. Which internal-state representation and matching covariance propagation are
   feasible? No depth dimension, basis or fitted closure is selected by this audit.

The [proposed controls](formulation_and_next_control.md#bounded-follow-up-design)
retain the two development paths and reference/320/325 K, with no mask, source
retuning or held-out expansion. Nothing is launched by this document.

## Verification and scope

[verification.json](verification.json) records inspected revision, fingerprints,
notebook-extract checks, existing geometry metadata and documentation checks.
Existing caches/code and pre-existing uncommitted work are preserved. Historical
protocol/index documents contain dated grid/hybrid descriptions; executed
Output 60/61/68 configurations, not those labels, define this audit's branch.
