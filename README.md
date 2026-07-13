# Thermal camera censored Gaussian experiment


The synthetic field is

```math
T_{true}(x,y) = T_0 + A\exp\left[-\frac{(x-x_c)^2}{2\sigma_x^2} - \frac{(y-y_c)^2}{2\sigma_y^2}\right].
```

The observed camera image is clipped by

```math
T_{obs}(x,y) = \min\{T_{true}(x,y), T_{max}\}.
```

The code compares four treatments of saturated pixels:

1. `exact`: treat clipped values as exact observations.
2. `discard`: remove saturated pixels from the loss.
3. `hinge`: fit unsaturated pixels and penalize saturated predictions below `Tmax`.
4. `censored`: use a censored Gaussian likelihood for saturated pixels.

## Setup

```bash
cd thermal_censored_gaussian
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/00_make_synthetic.py
python scripts/01_fit_single.py
python scripts/02_sweep_censoring.py
python scripts/03_plot_sweep.py
python scripts/06_model_misspecification.py
python scripts/07_plot_reconstruction.py
python scripts/08_gp_1d_censored.py
```

Outputs are written to `outputs/`.

## Main files

- `src/gaussian_field.py`: grid, true field, and censoring model.
- `src/losses.py`: exact, discard, hinge, and censored objectives.
- `src/fit.py`: parameter fitting with multi-start L-BFGS-B.
- `src/metrics.py`: field, peak, and parameter errors.
- `scripts/00_make_synthetic.py`: generate one synthetic image.
- `scripts/01_fit_single.py`: fit all methods on one image.
- `scripts/02_sweep_censoring.py`: compare methods over censoring levels.
- `scripts/03_plot_sweep.py`: plot summary metrics.
- `scripts/06_model_misspecification.py`: compare methods when the true field is not the fitted Gaussian model.
- `scripts/07_plot_reconstruction.py`: plot true, clipped, and fitted reconstruction fields.
- `scripts/08_gp_1d_censored.py`: 1D GP proof-of-concept comparing clipped, discard, censored, and oracle observations.
