# Dense-observation repeat

This folder repeats the thermal diffusion-kernel experiment with every second
unsaturated grid point retained (about 710 observations per trajectory).
All other settings match `../thermal_diffusion_kernel`.

Compared with the source-scale isotropic RBF, the source-scale diffusion kernel
slightly lowers temperature-excess field error for the diagonal and horizontal
paths, but slightly increases it for the spiral. No method covers the sharp
true peak.
