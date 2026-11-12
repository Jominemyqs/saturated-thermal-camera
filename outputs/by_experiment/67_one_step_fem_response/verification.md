# Execution, verification and preservation

## Actual formulation and mapping

- [Helper](../../../src/fem_replay.py) `build_problem` reconstructs the saved
  tetrahedra with DOLFIN MeshEditor, checks canonical cell connectivity and a
  bijection from original coordinates to mesh vertices and P1 DOF coordinates.
  It does not assign HDF nodal values under an assumed DOF ordering.
- The reconstructed predecessor's vertex values equal the original saved values
  exactly. Domain bounds, top/bottom areas and volume are checked. The true
  predecessor has the notebook bottom temperature before solving.
- The actual previous/current times are 0.1770/0.1775 s and 0.2715/0.2720 s.
  Solver dt is the notebook's literal 0.0005, not a fitted interval. Full clock
  agreement from Output66 is rechecked. Path position is evaluated at the saved
  current time using the original notebook class; no centroid or phase shift.
- rho=7860.51, Cp=472.84, k=15.11 and bottom/initial T0=293.15 are extracted from
  the supplied notebook. There is no beta or surrogate gradient in the FEM load.
- `solve(a == load, result, bc)` preserves notebook defaults. PETSc is the
  recorded backend; no claim is made about matching the historical binary
  build or whichever default LU package that old kernel selected. The measured
  replay validates the numerical result despite that provenance limit.
- FFC parameters are saved per worker, including automatic quadrature, degree-2
  UserExpression, `-O2` compiler optimization and cache paths. No quadrature or
  solver parameter was chosen to minimize discrepancy with the saved truth.
- Source-off retains the full-volume predecessor and original bottom condition.
  The independent response solve has zero bottom data, as required for the
  difference between two solutions with the same fixed-temperature boundary.
- The mean decomposition uses the frozen Output66 oracle fields on the same
  surface nodes, with the original source increment, not inferred predecessors
  or a retuned mean. Native physical-area hottest-1% weights are retained.

## Gate and results

[Frozen configuration](frozen_configuration.json) preceded the first thermal
solve. The volume/surface RMSE and maximum-error limits, relative linear-residual
screen, per-stage time budget and memory screen remain unchanged. Both source-on
fits passed before any source-off was launched. Execution order is recorded in
[execution_status.json](execution_status.json).

The maximum full-volume replay error is 5.35e-12 K. Linear residuals are below
1.42e-15, including the independent boundary-response solves. Superposition is
verified over the full volume to at most 1.69e-12 K. Surface decomposition checks
include the replay term and use 2e-12 K absolute tolerance.

Weighted RMSE uses lumped physical volume/area weights; it is not the exact
consistent-mass FE L2 norm. Maximum nodal error supplies a separate check and is
not diluted by the large cool part of the domain. No temperature error criterion
was used to select parameters, source location, shape, time or observation sets.

## Setup issue, not a failed temperature replay

The initial attempt stopped before any thermal solve because an overstrict
`1e-12` relative geometry-volume check rejected serial accumulation roundoff:
assembled volume differed from 3e-7 m3 by 1.618e-18 m3 (5.393e-12 relative).
The helper now uses the explicit `2*n_cells*eps` accumulation bound (5.803e-11
for this mesh). This changes only a geometry sanity check, **not** the
predeclared temperature replay gate or PDE. The complete initial setup attempt
is retained locally at `tmp/67_geometry_preflight_attempt`.

No subsequent thermal attempt failed. No current or historical scientific
output was overwritten to hide a failed temperature comparison.

## Environment and reproducibility

The legacy release and installation route were checked against the
[FEniCS legacy installation page](https://fenicsproject.org/download/archive/)
and the [conda-forge package](https://anaconda.org/conda-forge/fenics-dolfin).
This experiment uses DOLFIN 2019.1.0, not a port to DOLFINx. The isolated prefix
and package cache are inside the repository's ignored `tmp/` directory.
The exact osx-arm64 package set is in [environment_explicit.txt](environment_explicit.txt).

Example initial environment preparation from the repository root:

```sh
CONDA_PKGS_DIRS=$PWD/tmp/conda_pkgs mamba create -p "$PWD/tmp/fem_env" \
  --override-channels -c conda-forge python=3.11 fenics-dolfin=2019.1.0 \
  h5py scipy matplotlib pandas
```

Run the experiment with the original project Python (it launches activated FEM
workers). The driver intentionally refuses to overwrite an existing Output67:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MPLCONFIGDIR=$PWD/tmp/matplotlib PYTHONPYCACHEPREFIX=$PWD/tmp/pycache \
  .venv/bin/python scripts/71_thermal_one_step_fem_replay.py
```

On this machine, serial MPI initialization required execution outside the app's
filesystem/network sandbox. This was approved; no distributed job or external
data upload was used. XDG, FFC, Dijitso and Python caches were redirected into
`tmp/`. Existing Python environments were not upgraded or replaced.

## Tests and preservation

Fourteen tests passed in the FEM environment, including the real small-cube
operator test: constant temperature under zero input, positive response with
homogeneous bottom, and linear on/off superposition. Additional tests cover
coordinate permutations/rejection, tetrahedral weights, gate logic, exact error
bookkeeping, notebook expressions/clocks, existing attribution and native maps.

```sh
XDG_CACHE_HOME=$PWD/tmp/fem_cache DIJITSO_CACHE_DIR=$PWD/tmp/fem_dijitso \
  PYTHONPYCACHEPREFIX=$PWD/tmp/fem_pycache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$PWD/tmp/matplotlib \
  mamba run -p "$PWD/tmp/fem_env" python -m unittest \
  tests.test_fem_replay tests.test_prescribed_laser_audit \
  tests.test_forecast_attribution tests.test_native_surface tests.test_matched_projection -v
```

The three new Python files compile; `git diff --check` passes. Figures were
visually inspected. [Fingerprints](fingerprints.json) record source revision,
pre-existing work and all consumed data/code; their bytes were rechecked after
execution. All original notebooks, datasets, scientific modules, parameters,
caches and historical outputs are preserved. This task adds driver71, one helper,
one test module, Output67 and the isolated runtime only. No GP inference,
parameter fitting, source replacement or source/noise/Q/mask adjustment occurred.
