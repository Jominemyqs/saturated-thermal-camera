# Scope, evidence checks and limitations

Audit date: 2026-11-04, America/New_York. Revision and initial dirty state are
recorded in [fingerprints.json](fingerprints.json). The pre-existing uncommitted
output59-61 code/results and inventory edits remain untouched.

## Bounded search

| Scope | Inspection | Finding / limit |
| --- | --- | --- |
| Project data directory `../heat_eq_laser_trajectories` | File inventory and the two selected HDF/XDMF pairs | 33 exported trajectory pairs; no notebook/configuration in this directory. Only two pairs' field data inspected |
| Repository source, scripts, docs and archived branches | File discovery; searches for FEniCS/dolfin, W_grad_z, laser power, absorption, prescribed flux and provenance | Existing surrogate/calibration code and supplied snippet, not a matching FEM generator |
| Outer project `workspace/` | Script/document names and relevant provenance text | Previous reports and preprocessing audit scripts; no matching generator |
| Local Git history | `git log --all --oneline -- '*.ipynb' '*generate*' '*fenics*' '*simulation*'` | No matches under these filename patterns; does not prove no differently named or un-fetched generator exists |
| Downloads | Filename discovery for notebooks, scripts, relevant thermal/laser files and archives | Two byte-identical thesis PDFs; remaining notebook/archive names unrelated to this project. Unrelated contents were not inspected |
| Original notebook export | User-supplied email excerpt, already recorded in output52 | Not an authenticated full notebook; W_grad_z, surrounding time loop and version missing |
| Remote/email storage | Not searched or modified | No claim that a generator is absent from Adrienne's machine, mail, cloud storage or another repository |

No new source was fetched to substitute another dataset's formulation. The
absence claim is limited to the reviewed local scope. A differently stored
matching notebook can reopen the audit without rerunning any GP.

## Direct data checks

[Inspection driver](../../../scripts/66_thermal_physics_input_provenance.py)
opens HDF read-only and parses XDMF. Only final-two-frame temperature/gradient
arrays are read; all saved time labels and HDF attribute metadata are inspected.
Full HDF byte hashing reads file bytes for provenance but performs no additional
thermal analysis. The script does not import or invoke a GP inference driver,
physics propagator, optimizer or source reconstruction routine.

- Four last-frame gradient hashes reproduce output56 exactly.
- Two cases' coordinates, triangles, area weights, previous truth and current
  truth reproduce output52 cache arrays exactly.
- Top-surface identification uses geometric max-z nodes, not a temperature or
  gradient threshold. Extracted nodal area weights are geometry-only.
- T and HeatFluxZ both have scalar Node association and the same Time attribute
  at each selected frame. No temporal shift, interpolation or independent-input
  equivalence is inferred from that shared label.
- All HDF groups/datasets were visited for attribute metadata. Only partition
  attributes were found; no material values, source parameters, units or generator
  identifier. No XDMF Information tags were found.
- Output52 and output61 inherited alpha, beta, gamma, increment cutoff and
  preparation cutoff agree exactly. No calibration was rerun.

The local geometry establishes a geometric upper normal, not the original
variational load sign or boundary marker. Exported nodal association likewise
does not identify the finite-element family/order of T or W_grad_z.

## Thesis check

Read complete PDF pages9-17; visually inspected pages11,13,15 after rendering.
Equations2.1-2.7 and2.15, thermal material values and timestep statement were
checked directly rather than copied solely from previous report prose.
The two Downloads copies are byte-identical. All thesis settings remain labeled
reference-only, not authenticated values for the local generator.

Page17 describes a downstream graph-dataset sampling procedure; it is not
evidence for the export schedule of these HDF/XDMF files. No solver timestep
or sampling schedule was deduced from that paragraph.

## Preservation and validation

The inspection run hashed **136 source/test files and 21 input artifacts** before
and after reading; every hash agreed. The set includes both native data pairs,
the inspected caches/configurations, calibration records, thesis copies and
pre-existing protocol/inventory files. It does not claim to fingerprint every
historical cache in the workspace. No existing files were edited.

The new script refuses to overwrite an existing audit directory. Reproduction
should use a separate designated output path if a future run is needed, rather
than deleting this record. The dependency environment is the existing `.venv`;
no package was installed or changed for this audit.

Python compilation passed. Existing unittest discovery passed 122 tests; no
expensive native GP inference was launched by this task. The suite contains
small toy inference tests. It emits the existing constant-data skewness warning;
Matplotlib also used a temporary cache because the default home cache was not
writable. Neither warning changed scientific inputs or results.

Final checks resolved all 37 local report links and revalidated all 157 captured
file fingerprints. `git diff --check` passed. Two additional `.DS_Store` changes
appeared in the working tree during the audit; they were not edited or reverted
by this task. Existing inventory edits also remain as found at task start.

The gate remains closed: prescribed flux and k*g comparison, local/whole-area
source error metrics and comparison figures were **not generated**. There is no
claim of agreement or disagreement, physical model error, or required retuning.
