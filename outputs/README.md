# Output Organization

The original flat output files are kept for backward compatibility with existing scripts and notes. For easier browsing, grouped copies are available under `outputs/by_experiment/`.

## Grouped Outputs

- `01_parametric_baselines/`: initial synthetic Gaussian image, single fit, and censoring sweep.
- `02_prior_sensitivity/`: hinge/censored prior-sensitivity and higher-censoring studies.
- `03_noise_sweep/`: additive-noise robustness study.
- `04_reconstruction_plots/`: reconstruction diagnostics and prediction plots.
- `05_model_misspecification/`: matched Gaussian, rotated Gaussian, skewed wake, and two-component wake parametric fits.
- `06_gp1d/`: 1D censored GP proof-of-concept, lengthscale sweeps, multi-seed robustness, and uncertainty plots.
- `07_gp2d_single/`: first 2D censored GP proof-of-concept.
- `08_gp2d_shape_density/`: 2D axis-Gaussian, rotated-wake, and moving-laser-path experiments across observation densities.
- `09_gp2d_multiseed_fixed_vs_tuned/`: 2D multi-seed comparison of fixed versus unsaturated-MLL tuned isotropic lengthscales.
- `10_gp2d_kernel_comparison/`: 2D comparison of isotropic, tuned isotropic, tuned anisotropic, and informed anisotropic kernels, including seed-0 reconstruction panels.
- `11_gp2d_path_kernel/`: moving-laser-path comparison of isotropic, global anisotropic, path-aligned fixed, and path-aligned tuned kernels, including extra axis-Gaussian and rotated-wake reconstruction panels.
- `12_gp2d_path_censored_tuning/`: path-aligned moving-laser comparison of fixed geometry-informed hyperparameters, unsaturated-MLL tuning, and full censored-MLL tuning.
- `pdf/`: meeting-summary PDF artifacts, including the current censored-GP experiment summary.
