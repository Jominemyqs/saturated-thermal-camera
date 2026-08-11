# Superseded diagnostic outputs

These outputs are preserved for audit and are not part of the current main
model comparison:

- `21_sequential_advection_forcing`: finite-step process-covariance diagnostic;
- `22_joint_sequential_audit`: implementation audit containing joint/double-use rows;
- `23_strict_sequential_spde`: deferred strict state-space/SPDE experiment.

The canonical current result is
`../../by_experiment/24_posterior_physics_mean_advection/`. The retained
posterior-physics-mean ablation is
`../../by_experiment/20_physics_mean_propagation/`, and the stochastic
space-time kernel remains a supporting result in
`../../by_experiment/19_stochastic_spde_ablation/`.
