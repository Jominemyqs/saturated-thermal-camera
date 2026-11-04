# Minimal evidence needed to reopen flux equivalence

Request the **matching generation code and run configuration**, not a newly
fitted Gaussian. Send these file hashes from [fingerprints](fingerprints.json)
to identify DiagonalScanPath_8 and SpiralScanPath_13 unambiguously.

1. Generator notebook/script revision, dependency versions and the configuration
   selecting these two runs. Confirm whether local HDF/XDMF files came from it;
   a filename family or matching domain dimensions alone is not enough.
2. Thermal variational form, material k/rho/Cp definitions and facet marking;
   top/bottom/side conditions, normal/flux sign convention and any other losses.
3. Exact laser boundary expression, normalization and width convention; incident
   versus absorbed power, absorption/efficiency handling, and laser on/off rule.
4. Executable `x_focus(t), y_focus(t)` definitions plus speed, origin, offsets,
   turn/spiral parameters and each run's switch time. No gradient-centroid substitute.
5. Integration loop and export loop: simulation step, adaptive/substeps if any,
   saved-frame schedule, source time evaluation, and solve/project/write ordering.
   Explain the thesis0.005 versus saved0.0005 difference without assuming either
   value is wrong. Confirm coordinate/time units.
6. `W_grad_z` definition (element family/degree/domain), `project` call and settings,
   exported field association and confirmation of the supplied `T.dx(2)` snippet.

## Bounded comparison once evidence is confirmed

Use only the existing last two frames of these two paths. Reconstruct the raw
prescribed input with the verified expression and parameters, then compare it
with the conductivity-scaled raw exported gradient at the identical native nodes.
No gamma, cutoff, sign adjustment chosen from data, fitted shift or fitted scale.

Predefine the local source window from the independently prescribed centre/width
and use the same native area weights. Report whole-area and source-local bias,
MAE/RMSE, relative L2, maxima, area integrals, peak-coordinate differences and
explicit input/field timestamps. If the laser is off or the reference norm is
zero, relative errors are undefined, not silently assigned zero. Compare each
frame separately; do not tune a time offset to improve agreement. Interpret
local nodal sampling with the mesh resolution and projected-derivative space.

Keep two decisions separate: (a) numerical boundary equivalence, and (b) whether
the input is independently known in deployment. A close match could support a
later controlled prescribed-input branch, but would not authorize keeping the
same empirical gamma or treating it as physically justified without further work.
