# PartiallySpecifiedModels.jl

[![CI](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/PartiallySpecifiedModels.jl/dev/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A Julia package for fitting **partially specified dynamical models** with nonparametric functional responses.

## Overview

**Partially specified models (PSMs)** are dynamical systems (ODEs, DDEs, or discrete-time maps) in which one or more functional responses are left unspecified and estimated directly from data. Instead of assuming a fixed parametric form for processes like density dependence or predation rates, PSMs replace these unknown functions with flexible nonparametric approximators — penalized B-splines, Gaussian processes, or neural networks — and fit them jointly with the model dynamics.

This approach is particularly valuable in **ecology**, where the form of key biological processes (e.g., Holling-type functional responses, density-dependent growth, transmission rates) is often uncertain or debated. PSMs allow researchers to let the data inform the shape of these relationships, combining mechanistic understanding of system structure with statistical flexibility for unknown components.

PartiallySpecifiedModels.jl provides a unified interface for specifying and fitting PSMs using two complementary approximation strategies:

- **Basis function approximators** (B-splines, shape-constrained splines, Gaussian processes): fewer parameters, automatic smoothing via LAML/GCV, interpretable, and easy to constrain (monotonicity, convexity, positivity).
- **Neural network approximators** (Lux.jl networks, COMONet): more flexible for high-dimensional or complex functional forms, compatible with gradient-based UDE-style training.

The package builds on the [SciML ecosystem](https://sciml.ai/) and supports 17 fitting algorithms, 5 approximator types, 4 likelihood families, and 14 shape constraint types.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/PartiallySpecifiedModels.jl")
```

Requires Julia ≥ 1.10.

## Quick Start

A simple example: recover the unknown per-capita growth rate $r(N)$ from logistic growth data.

```julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using Random
Random.seed!(42)

# Generate synthetic logistic growth data
r0, K, N0 = 0.5, 10.0, 0.5
tspan = (0.0, 15.0)
data_times = collect(0.0:0.5:15.0)

function logistic!(du, u, p, t)
    du[1] = r0 * (1 - u[1] / K) * u[1]
end
true_prob = ODEProblem(logistic!, [N0], tspan)
true_sol = OrdinaryDiffEq.solve(true_prob, Tsit5(); saveat=data_times)
true_N = [true_sol(t)[1] for t in data_times]
observed_N = true_N .+ 0.3 .* randn(length(data_times))
observed_N = max.(observed_N, 0.01)

# Define PSM: dN/dt = r(N) * N, where r(N) is unknown
function growth!(du, u, p, t)
    N = u[1]
    du[1] = p.r(N) * N
end

# Approximate r(N) with a penalized cubic B-spline (8 knots)
approx_r = BSplineApproximator(:r, (0.0, 12.0), 8; initial = x -> 0.3)

# Build problem
prob = PSMProblem(
    growth!, [N0], tspan, [approx_r];
    data_times = data_times,
    data_values = reshape(observed_N, :, 1),
    obs_to_state = [1],
    likelihood = Gaussian(),
    solver = Tsit5()
)

# Fit with LAML (≡ REML for Gaussian data)
sol = solve(prob, LAML())

# Evaluate the recovered unknown function
r_fitted = sol.unknown_functions[:r]
r_estimated = [r_fitted(N) for N in range(0.1, 11.0, length=100)]
```

## Solvers

PartiallySpecifiedModels.jl provides 17 solvers spanning penalized likelihood, gradient matching, probabilistic numerics, and Bayesian inference:

| Solver | Method | ODE-free? | Bayesian? | Reference |
|--------|--------|:---------:|:---------:|-----------|
| `LAML` | Penalized IRLS + LAML smoothing | No | No | Wood et al. (2016) |
| `GCVSolver` | Penalized IRLS + GCV smoothing | No | No | Wood (2001) |
| `CollocationLAML` | Generalized profiling | No | No | Ramsay et al. (2007) |
| `GradientMatching` | Smooth then match derivatives | Yes | No | Calderhead et al. (2009) |
| `TwoStageSolver` | Smooth then match (simple) | Yes | No | Wood (2001) |
| `BNGSolver` | Bayesian neural gradient matching | Yes | No | Bonnaffé et al. (2023) |
| `AdaptiveGradientMatching` | GP product-of-experts | Yes | No | Macdonald & Husmeier (2015) |
| `AdamSolver` | Adam through ODE (UDE-style) | No | No | Rackauckas et al. (2020) |
| `MultipleShootingSolver` | Multiple shooting + Adam | No | No | Turan & Jäschke (2021) |
| `DerivativeFreeSolver` | NelderMead / Particle Swarm | No | No | — |
| `RodeoSolver` | Probabilistic ODE (Kalman) | No | No | Tronarp et al. (2022) |
| `DaltonSolver` | Data-adaptive Kalman likelihood | No | No | Wu & Lysy (2024) |
| `MCMCSolver` | HMC/NUTS posterior sampling | No | Yes | — |
| `MagiSolver` | Manifold-constrained GP inference | No | Yes | Yang et al. (2021) |
| `PseudoMarginalSolver` | Probabilistic ODE + NUTS | No | Yes | Chkrebtii et al. (2016) |
| `VariationalSolver` | Mean-field variational inference | No | Yes | — |
| `ABCSolver` | ABC-SMC (likelihood-free) | No | Yes | — |

## Approximators

| Approximator | Description | Parameters |
|-------------|-------------|------------|
| `BSplineApproximator` | Cubic B-spline basis | Spline coefficients |
| `ShapeConstrainedBSplineApproximator` | SCOP-spline (Pya & Wood 2015) | Constrained coefficients |
| `NeuralApproximator` | Lux.jl neural network | Network weights |
| `GPApproximator` | Gaussian process | GP hyperparameters |
| `COMONetApproximator` | Constrained monotone network | exp(W) weights |

## Features

### Likelihoods

- **`Gaussian()`** — Gaussian errors with identity link (default)
- **`Poisson()`** — Count data with log link
- **`NegativeBinomial()`** — Overdispersed counts with estimated dispersion
- **`CustomLikelihood(loglik, dloglik, d2loglik)`** — User-defined likelihood

### Dynamical System Support

- **Continuous-time**: ODEs via `OrdinaryDiffEq.jl`, DDEs via `DelayDiffEq.jl`
- **Discrete-time**: Maps via `DiscreteProblem`
- Construct from SciML problem types directly: `PSMProblem(ODEProblem(...), approximators; ...)`

### Shape Constraints (14 types)

For `ShapeConstrainedBSplineApproximator` (SCOP-splines):

| Constraint | Description |
|------------|-------------|
| `:increasing` / `:decreasing` | Monotonicity |
| `:convex` / `:concave` | Curvature |
| `:inc_convex` / `:inc_concave` | Increasing + curvature |
| `:dec_convex` / `:dec_concave` | Decreasing + curvature |
| `:positive` / `:dec_positive` | Positivity (with optional monotonicity) |
| `:inc_zero_left` / `:inc_zero_right` | Increasing, zero at endpoint |
| `:dec_zero_left` / `:dec_zero_right` | Decreasing, zero at endpoint |

## Vignettes

The `vignettes/` directory contains 25 worked examples:

| # | Vignette | Description |
|---|----------|-------------|
| 01 | Getting Started | Basic PSM workflow with exponential/logistic growth |
| 02 | Likelihoods | Gaussian, Poisson, Negative Binomial, and custom likelihoods |
| 03 | Lotka–Volterra | Hare–lynx predator-prey with LAML and collocation |
| 04 | Copepod | 11-stage structured population model with multiple unknown functions |
| 05 | Neural Networks | Comparing B-spline, GP, and neural network approximators on SIR |
| 06 | Solver Comparison | Side-by-side comparison of seven solvers |
| 07 | Probabilistic Fitting | Probabilistic ODE fitting with uncertainty quantification |
| 08 | Rosenzweig–MacArthur | Recovering functional responses in consumer-resource dynamics |
| 09 | Gradient Matching | Integration-free inference with adaptive gradient matching |
| 10 | Chemostat | Microbial dynamics recovering unknown Monod growth kinetics |
| 11 | Count Data SIR | SIR model with Poisson and Negative Binomial likelihoods |
| 12 | Discrete Time | Ricker, Beverton–Holt, and discrete competition models |
| 13 | Shape Constraints | Monotonicity, convexity, and zero-at-endpoint constraints |
| 14 | MCMC | Full Bayesian inference with HMC/NUTS posterior sampling |
| 15 | MAGI | Manifold-constrained Gaussian process inference |
| 16 | COMONet | Shape-constrained neural network approximators |
| 17 | BNG | Bayesian neural gradient matching |
| 18 | Dalton | Data-adaptive Kalman likelihood fitting |
| 19 | Pseudo-Marginal | Probabilistic ODE + Bayesian MCMC |
| 20 | DDE | Delay differential equations with unknown functions |
| 21 | GCV | Generalized Cross-Validation vs LAML smoothing |
| 22 | Two-Stage | Smooth-then-differentiate baseline approach |
| 23 | Derivative-Free | Nelder-Mead and Particle Swarm optimization |
| 24 | Variational | Fast approximate Bayesian inference via variational methods |
| 25 | ABC | Likelihood-free inference with ABC-SMC |

## References

- Wood, S.N. (2001). "Partially specified ecological models." *Ecological Monographs*, 71(1), 1–25.
- Wood, S.N., Pya, N. & Säfken, B. (2016). "Smoothing parameter and model selection for general smooth models." *JASA*, 111(516), 1548–1575.
- Ramsay, J.O., Hooker, G., Campbell, D. & Cao, J. (2007). "Parameter estimation for differential equations: a generalized smoothing approach." *JRSS-B*, 69(5), 741–796.
- Pya, N. & Wood, S.N. (2015). "Shape constrained additive models." *Statistics and Computing*, 25(3), 543–559.
- Rackauckas, C. et al. (2020). "Universal differential equations for scientific machine learning." *arXiv:2001.04385*.
- Yang, S., Wong, S.W.K. & Kou, S.C. (2021). "Inference of dynamic systems from noisy and sparse data via manifold-constrained Gaussian processes." *PNAS*, 118(15).
- Bonnaffé, W., Sheldon, B.C. & Coulson, T. (2023). "Neural ordinary differential equations for ecological and evolutionary time-series analysis." *Methods in Ecology and Evolution*, 14, 1301–1315.

## License

GPL-3.0 — see [LICENSE](LICENSE).
