# Short literature pass: censored thermal observations

## Big picture

The basic censored-data likelihood is not the novelty by itself. Tobit models, censored regression, and censored likelihoods are already well established. The interesting opening for this project is to bring that observation model into a thermal/AM setting where the saturation mask is produced by a physical process, and then show that using the inequality information improves either physical inference or surrogate training.

The papers suggest that two directions are strongest for us:

1. **Data assimilation / inverse modeling with censored observations.**
2. **Neural-network surrogate modeling trained directly with a censored or inequality-aware loss.**

Gaussian processes are still useful, but probably as a baseline or small-scale diagnostic rather than the main contribution, because censored/Tobit GP methods are already quite developed and scaling them to full thermal images is nontrivial.

## Paper-by-paper takeaways

- **Amemiya, Tobit Models: A Survey.** This establishes the classical framework for limited, censored, and truncated dependent variables. Useful for terminology and likelihood framing, but the basic Tobit idea is already done.

- **Miller and Halpern, Regression with Censored Data.** Reviews older censored-regression methods such as Cox and Buckley-James-type approaches. Useful background, but mostly survival/regression-oriented rather than image or physics-model-oriented.

- **Lai and Ying, Estimating a Distribution Function with Truncated and Censored Data.** A theoretical paper on product-limit-type estimators and asymptotics under truncation/censoring. Good for statistical depth, but not the most direct route for our numerical project.

- **Wani et al., Parameter Estimation of Hydrologic Models Using a Likelihood Function for Censored and Binary Observations.** Very relevant analogy. They show that censored/binary observations can still inform physical model parameters when used through a formal likelihood. This supports our idea that saturated pixels can inform heat-source or state estimation.

- **Thacker, Data Assimilation with Inequality Constraints.** Shows how inequality constraints can be incorporated into variational data assimilation, with active constraints treated carefully rather than corrected afterward. This is one of the strongest references for a physics-based direction.

- **Lauvernet et al., A Truncated Gaussian Filter for Data Assimilation with Inequality Constraints.** Extends the inequality-constraint idea to a truncated Gaussian filtering framework and demonstrates it on ocean models. This strengthens the case that constrained data assimilation is a real, established area, but not yet specialized to thermal-camera saturation.

- **Maatouk and Bay, Gaussian Process Emulators for Computer Experiments with Inequality Constraints.** Builds GP emulators that satisfy inequality constraints such as boundedness, monotonicity, or convexity. Useful as a GP reference, but their focus is constraints on emulator behavior over the domain, not necessarily censored sensor observations.

- **Basson et al., Variational Tobit Gaussian Process Regression.** Gives a sparse variational GP method for censored observations. This is close to a direct GP version of our problem, which means a pure censored-GP project may be less novel unless we add thermal physics or AM-specific structure.

- **Chakraborty and Chakraborty, Efficient and Scalable Tobit Gaussian Process Regression.** Pushes censored GP regression toward large-scale settings using sparse ordered conditional / Vecchia-type ideas. This reinforces that scalable Tobit GP is an active and already-developed area.

- **Spiller et al., The Zero Problem.** Uses a zero-censored GP idea for range-constrained simulator outputs, especially geophysical flows. The range-constraint viewpoint is relevant, but their problem is more about simulator output constraints than camera saturation as an observation process.

- **Guo et al., Machine Learning for Metal Additive Manufacturing.** Supports the broader shift from black-box ML toward physics-informed data-driven modeling in metal AM. Helpful motivation for training surrogates with physically/statistically meaningful losses.

- **Faegh et al., Review on Physics-Informed ML for Process-Structure-Property Modeling in AM.** Organizes PIML into physics-based features, architectures, and loss functions. Our censored/inequality-aware loss fits naturally into the loss-function category and gives a concrete sensor-driven PIML example.

## What seems mostly done

- The censored likelihood/Tobit model itself is classical.
- Censored regression and survival-style methods are mature.
- Censored Gaussian process regression exists, including variational and scalable versions.
- Generic inequality-constrained GP emulators also exist.
- Physics-informed ML for AM is already a large review-level topic.

So the project should avoid presenting "we have a censored likelihood" or "we can do a Tobit GP" as the main contribution. Those are better used as foundations or baselines.

## Most promising directions for this project

### 1. Data assimilation or inverse modeling with censored thermal observations

This is the strongest physics-facing direction. The heat equation or a thermal simulator gives the state/model, while the camera gives censored observations:

\[
    Y_i = \min\{H_i(u) + \varepsilon_i, T_{\max}\},
    \qquad \varepsilon_i \sim \mathcal N(0,\sigma^2),
\]

where \(u\) may be the thermal state, source parameters, material parameters, or boundary/input parameters. Unsaturated pixels contribute ordinary Gaussian mismatch terms, while saturated pixels contribute threshold-exceedance probabilities:

\[
    \mathcal L_{\mathrm{obs}}(u)
    =
    \sum_{i\in\mathcal U}
    \frac{(Y_i-H_i(u))^2}{2\sigma^2}
    -
    \sum_{i\in\mathcal S}
    \log \Phi\!\left(\frac{H_i(u)-T_{\max}}{\sigma}\right).
\]

This direction is promising because it connects censored observations to physically meaningful inference, not just image reconstruction. A clean first study could infer heat-source parameters from clipped synthetic thermal data and compare exact, discard, hinge, and censored likelihood treatments.

### 2. Neural-network surrogate modeling with censored loss

This is the strongest ML-facing direction. Instead of reconstructing the missing peak first, train a surrogate directly from censored thermal images. For a network \(T_\eta(x;\theta)\), an inequality-aware training loss could be

\[
    \mathcal L_{\mathrm{hinge}}(\eta)
    =
    \sum_{i\in\mathcal U}
    \bigl(T_\eta(x_i;\theta)-Y_i\bigr)^2
    +
    \lambda
    \sum_{i\in\mathcal S}
    \bigl[T_{\max}-T_\eta(x_i;\theta)\bigr]_+^2.
\]

A probabilistic version would use the censored Gaussian likelihood above. The key experiment would compare three strategies: train on clipped fields as exact, reconstruct first and then train, or train directly with a censored/hinge loss. Synthetic data makes this test clean because the true uncensored fields are known.

This direction fits naturally with the AM/PIML literature because it is a physics/statistics-informed loss modification rather than a generic black-box network. It is also practically useful if the long-term goal is surrogate modeling from imperfect thermal-camera data.

## Recommendation

Use GP methods as a small baseline or explanatory figure if needed, but center the project around the two directions above. Together they give a balanced story: data assimilation/inverse modeling for physical interpretability and uncertainty, and censored-loss surrogate modeling for practical ML use in additive manufacturing.
