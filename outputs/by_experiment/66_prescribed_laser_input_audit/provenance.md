# Provenance and independent availability

Cells below use **zero-based notebook cell indices**, with one-based lines within
the cell. Exact source extracts are included for cells 3, 5 and 6; no notebook
installation, Drive-mounting, mesh construction or FEM solve was executed.
The original four notebooks live one directory above the repository and are
fingerprinted in [fingerprints.json](fingerprints.json). Only Diagonal 8 and
Spiral 13 were numerically audited. Horizontal/Vertical common PDE/export code
was inspected, but their paths and data were not validated by this comparison.

| Quantity | Actual source | Available without current FEM temperature? | Confirmation and limitations |
| --- | --- | --- | --- |
| Prescribed g | [Diagonal cell6](Diagonal_cell6_source.txt), `SmoothZigZagLaser.eval`, L58-81; [Spiral cell6](Spiral_cell6_source.txt), `SmoothLaserPath.eval`, L41-58 | Yes, given specified laser settings | Original eval methods used directly, not fitted to the gradient |
| Normalization | Same expressions: `2*power/(pi*(2*sigma)^2)` times `exp(-2*r^2/(2*sigma)^2)` | Yes | Equivalent to `P/(2*pi*sigma^2)*exp(-r^2/(2*sigma^2))`; full-plane integral P |
| Power and width | Diagonal cell6 L118-119; Spiral L102-103 | Yes | 1.0 and 100e-6, interpreted as W and m; 2*sigma is the 1/e2 radius in this expression |
| Absorption / switching | Entire path class and timestep loop in cell6 | No additional variable required by this code | No explicit absorptivity factor or within-run on/off switch. The physical meaning of `power` as incident versus already-absorbed power is unresolved; do not infer a material absorptivity |
| Diagonal path | Cell6 L9-44; `generate_zigzag_path_with_linespace`; num_interval=8 is in cell5 batch list | Yes | 81 path points; length 0.053254834 m; velocity 0.3 m/s; modulo total-distance rule retained |
| Spiral path | Cell6 L7-28, L41-55, L63-68; `generate_archimedean_spiral_path_and_length`; num_interval=13 in cell5 | Yes | 16,336 path points; length 0.081749834 m; velocity 0.3 m/s. dt_path=.005 determines angular sampling, not saved-frame timing |
| Actual batch selection | [Diagonal cell5](Diagonal_cell5_source.txt), [Spiral cell5](Spiral_cell5_source.txt) | Yes | The isolated `num_interval=4` in cell4 is overridden by cell6's loop over the batch; it is not the executed setting for these two files |
| Mesh/domain | [Cell3](Diagonal_cell3_source.txt), L4-36 (same in Spiral) | Yes | 15 x 10 x 2 mm. Three refinements of a fixed central top rectangle, not a mesh following the instantaneous laser. Bounds/top area match stored geometry; mesh construction not reproduced |
| Material rho, Cp, k | Diagonal cell6 L92-99; Spiral L76-83 | Yes | Constant-return functions: 7860.51, 472.84, 15.11; SI context implies kg/m3, J/(kg K), W/(m K). No temperature dependence despite function arguments |
| Thermal initial/bottom T | Diagonal L86-108; Spiral L70-92 | Yes | 293.15 K in notebook; constant Dirichlet bottom condition |
| Weak form / top sign | Diagonal L101-106, L130; Spiral L85-90, L123 | Yes | Top marker 2, +g on RHS. Outward top normal +z gives positive inward g = k*dT/dz under the continuum natural condition |
| Side conditions | Same weak form and bottom-only Dirichlet constraint | Yes | No side loads or side Dirichlet constraints: homogeneous natural flux in this formulation |
| Solver time and source phase | Diagonal L5, L122-135; Spiral L4, L106-128 | Yes | dt=.0005; increment t, evaluate g(t), implicit conduction solve, assign state; no hidden substep in this loop |
| Save times | Diagonal L144-145; Spiral L137-138 | Yes | One temperature/gradient pair per solve, no t=0 export; all stored clocks checked |
| HeatFluxZ | Diagonal L137-145; Spiral L130-138 | **No** | `project(T.dx(2), CG1)` after solve; units K/m under SI coordinates. Output of the current solution, not prescribed input |
| Derivative conductivity conversion | Notebook constant k and verified top sign | Depends on solution for the gradient | `k*HeatFluxZ` interpreted as W/m2; this is what was compared, without any surrogate threshold |
| Current surrogate alpha/beta/gamma | [Output61 configuration](../61_unpinning_current_update/frozen_configuration.json), `inherited_settings` | Historically calibrated, not derived here | 3.6109129986233175e-6 m2/s, 12.422901570529978 1/s, 0.1702227206133784 (dimensional m/s for a K/m input). Gamma remains empirical |
| Surrogate ambient and predecessor | Output52/60 caches, fingerprinted; [review inventory](../../../docs/adrienne_review/parameter_inventory.csv) | Not automatically independent camera-only information | Ambient is inherited simulation-based preparation, not silently replaced by FEM T0; current full-latent predecessor reused unchanged |
| Required correction | Oracle current truth minus propagation-only forecast | **No, diagnostic only** | Absorbs propagation/model mismatch; not an independently measured source |

## What remains unresolved

The email explicitly associates these notebook path/time settings with the
two datasets. Matching all clocks and domain bounds corroborates that claim.
However, the HDF files contain no generator commit or full run configuration;
the notebook kernel/library version, exact historical assembly quadrature,
solver tolerance and original mesh ordering are not independently authenticated.
No FEM temperature reproduction was attempted. The thesis is context, not a
substitute for this supplied code; its differently stated timestep is not used.

Actual deployment availability remains separate: power/path settings are
independent *within these simulations*. Whether all those quantities, effective
absorption and material/boundary properties are known sufficiently accurately
in a real experiment still needs specification. No temperature-derived fitting
has been relabeled as independent laser information.
