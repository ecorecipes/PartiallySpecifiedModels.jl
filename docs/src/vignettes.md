# Vignettes

The `vignettes/` directory contains 26 worked examples covering every solver, approximator, and likelihood type. Each vignette is a self-contained [Quarto](https://quarto.org/) document with rendered markdown available on GitHub.

## Getting Started

| # | Vignette | Description |
|:--|:---------|:------------|
| 01 | [Getting Started](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/01_getting_started/01_getting_started.md) | Basic PSM workflow with exponential and logistic growth |
| 02 | [Likelihoods](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/02_likelihoods/02_likelihoods.md) | Gaussian, Poisson, Negative Binomial, and custom likelihoods |

## Ecological Models

| # | Vignette | Description |
|:--|:---------|:------------|
| 03 | [Lotka–Volterra](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/03_lotka_volterra/03_lotka_volterra.md) | Hare–lynx predator-prey with LAML and collocation |
| 04 | [Copepod](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/04_copepod/04_copepod.md) | 11-stage structured population model with multiple unknown functions |
| 08 | [Rosenzweig–MacArthur](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/08_rosenzweig_macarthur/08_rosenzweig_macarthur.md) | Recovering functional responses in consumer-resource dynamics |
| 10 | [Chemostat](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/10_chemostat/10_chemostat.md) | Microbial dynamics recovering unknown Monod growth kinetics |
| 11 | [Count Data SIR](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/11_count_data_sir/11_count_data_sir.md) | SIR model with Poisson and Negative Binomial likelihoods |
| 12 | [Discrete Time](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/12_discrete_time/12_discrete_time.md) | Ricker, Beverton–Holt, and discrete competition models |
| 20 | [DDE](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/20_dde/20_dde.md) | Delay differential equations with unknown functions |

## Approximators

| # | Vignette | Description |
|:--|:---------|:------------|
| 05 | [Neural Networks](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/05_neural_networks/05_neural_networks.md) | Comparing B-spline, GP, and neural network approximators |
| 13 | [Shape Constraints](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/13_shape_constraints/13_shape_constraints.md) | Monotonicity, convexity, and zero-at-endpoint constraints |
| 16 | [COMONet](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/16_comonet/16_comonet.md) | Shape-constrained neural network approximators |
| 26 | [SPDE](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/26_spde/26_spde.md) | Matérn SPDE approximator with shape constraints and profile range optimization |

## Solvers

| # | Vignette | Description |
|:--|:---------|:------------|
| 06 | [Solver Comparison](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/06_solver_comparison/06_solver_comparison.md) | Side-by-side comparison of seven solvers |
| 07 | [Probabilistic Fitting](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/07_probabilistic_fitting/07_probabilistic_fitting.md) | Probabilistic ODE fitting with uncertainty quantification |
| 09 | [Gradient Matching](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/09_gradient_matching/09_gradient_matching.md) | Integration-free inference with adaptive gradient matching |
| 14 | [MCMC](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/14_mcmc/14_mcmc.md) | Full Bayesian inference with HMC/NUTS posterior sampling |
| 15 | [MAGI](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/15_magi/15_magi.md) | Manifold-constrained Gaussian process inference |
| 17 | [BNG](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/17_bng/17_bng.md) | Bayesian neural gradient matching |
| 18 | [Dalton](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/18_dalton/18_dalton.md) | Data-adaptive Kalman likelihood fitting |
| 19 | [Pseudo-Marginal](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/19_pseudo_marginal/19_pseudo_marginal.md) | Probabilistic ODE + Bayesian MCMC |
| 21 | [GCV](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/21_gcv/21_gcv.md) | Generalized Cross-Validation vs LAML smoothing |
| 22 | [Two-Stage](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/22_two_stage/22_two_stage.md) | Smooth-then-differentiate baseline approach |
| 23 | [Derivative-Free](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/23_derivative_free/23_derivative_free.md) | Nelder-Mead and Particle Swarm optimization |
| 24 | [Variational](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/24_variational/24_variational.md) | Fast approximate Bayesian inference via variational methods |
| 25 | [ABC](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/vignettes/25_abc/25_abc.md) | Likelihood-free inference with ABC-SMC |

## Running Vignettes Locally

Each vignette is a Quarto `.qmd` file that can be rendered locally:

```bash
cd vignettes/01_getting_started
quarto render 01_getting_started.qmd --to html
```

Requires [Quarto](https://quarto.org/) and a Julia installation with the package dependencies.
