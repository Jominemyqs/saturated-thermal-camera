# Verification and scope

- Working-tree revision and pre-existing untracked work: [fingerprints.json](fingerprints.json).
  Source files, four supplied notebooks, both XDMF/HDF datasets and all consumed
  caches were hashed. All 174 recorded files remained unchanged during the run.
- New code only: driver70, `src/prescribed_laser_audit.py`, its unit tests, and
  this isolated output directory. No existing scientific source or baseline changed.
- The exact inspected notebook path functions and laser classes were extracted
  through AST selection. Only those definitions were executed, with a minimal
  UserExpression base for point evaluation. No installer, Google Drive code,
  mesh refinement, variational assembly, solve, GP sampling or optimization ran.
- Tests check both path clocks, Gaussian normalization/radial values at prescribed
  focus, point-expression behavior and weighted metric definitions; existing
  forecast attribution, native area/integration and matched propagation tests
  were also run. Ten tests passed using:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MPLCONFIGDIR=$PWD/tmp/matplotlib PYTHONPYCACHEPREFIX=$PWD/tmp/pycache \
  .venv/bin/python -m unittest tests.test_prescribed_laser_audit \
  tests.test_forecast_attribution tests.test_native_surface tests.test_matched_projection -v
```

- Pytest is not installed in the existing virtual environment; the new tests
  use the repository's unittest convention. No dependency installation needed.
- Initial scalar matrix-product reductions emitted floating-point warnings on
  this runtime despite finite results. The audit's scalar reductions were changed
  to explicit `sum(weights*values)` and the audit rerun without those warnings.
  The initial locally generated output is retained at
  `tmp/66_initial_reduction_warning_audit`; this is not a historical experiment.
  All common numeric CSV values agree within 8.89e-16 absolute difference.
- Native-node primary metrics and analytic-Gaussian surface quadrature are
  reported separately. Refinement 2 to 3 changes g's integral by at most
  1.3884e-7 W; the P1 exported-field integral is subdivision-invariant to roundoff.
- All 899 stored timestamps checked, maximum error 5.56e-17 s. Final frames:
  Diagonal indices 353/354, Spiral 542/543 (zero based).
- Six oracle checks (two cases x three cached ceilings) reproduce Output51
  arrays exactly. Latest full-latent source-on forecast equals Output63 exactly;
  source-off differs only by floating-point subtraction (max 5.69e-14 K).
- Source-relative regions use the prescribed focus and fixed 3*sigma radius,
  not temperature/gradient errors. Truth enters only oracle evaluation. The
  historical section directions remain display-only, never laser inputs.
- Recorded runtime 13.023 s includes fingerprinting, quadrature, deterministic
  propagation and plotting. No FEM/GP runtime is implied. No peak-memory study
  or production inference feasibility claim is made.
- Figures visually inspected for color scales, legends and readable sections.
  Flux panels use the same scales at both times, with signed differences; oracle
  sections use matched vertical ranges. Only last-two-frame input audit, not a
  trajectory-wide physical validation or fitted temporal alignment.
- Compilation of the three new Python files and `git diff --check` pass.
  All 33 relative Markdown links resolve; new files have no trailing whitespace;
  JSON parses and every numeric CSV entry is finite. Existing tracked files have
  no diff; earlier untracked experiments remain in place.

Machine checks: [verification.json](verification.json),
[oracle checks](oracle_equality_checks.csv), [clock checks](clock_checks.json).
Open provenance and physical-interpretation limits are in [provenance.md](provenance.md).
