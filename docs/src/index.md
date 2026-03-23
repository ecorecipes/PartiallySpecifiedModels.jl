# PartiallySpecifiedModels.jl

A Julia package for fitting **partially specified dynamical models** with nonparametric functional responses.

## Overview

**Partially specified models (PSMs)** are dynamical systems (ODEs, DDEs, or discrete-time maps) in which one or more functional responses are left unspecified and estimated directly from data. Instead of assuming a fixed parametric form for processes like density dependence or predation rates, PSMs replace these unknown functions with flexible nonparametric approximators — penalized B-splines, Gaussian processes, or neural networks — and fit them jointly with the model dynamics.

This approach is particularly valuable in **ecology**, where the form of key biological processes (e.g., Holling-type functional responses, density-dependent growth, transmission rates) is often uncertain or debated. PSMs allow researchers to let the data inform the shape of these relationships, combining mechanistic understanding of system structure with statistical flexibility for unknown components.

PartiallySpecifiedModels.jl provides a unified interface for specifying and fitting PSMs using two complementary approximation strategies:

- **Basis function approximators** (B-splines, shape-constrained splines, Gaussian processes): fewer parameters, automatic smoothing via LAML/GCV, interpretable, and easy to constrain (monotonicity, convexity, positivity).
- **Neural network approximators** (Lux.jl networks, COMONet): more flexible for high-dimensional or complex functional forms, compatible with gradient-based UDE-style training.

The package supports 17 fitting algorithms, 5 approximator types, 4 likelihood families, and 14 shape constraint types.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/PartiallySpecifiedModels.jl")
```

Requires Julia ≥ 1.10.

## Quick Start

A simple example: recover the unknown per-capita growth rate ``r(N)`` from logistic growth data.

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

## Package Features

### Solvers

| Solver | Method | ODE-free? | Bayesian? | Reference |
|--------|--------|:---------:|:---------:|-----------|
| [`LAML`](@ref) | Penalized IRLS + LAML smoothing | No | No | Wood et al. (2016) |
| [`GCVSolver`](@ref) | Penalized IRLS + GCV smoothing | No | No | Wood (2001) |
| [`CollocationLAML`](@ref) | Generalized profiling | No | No | Ramsay et al. (2007) |
| [`GradientMatching`](@ref) | Smooth then match derivatives | Yes | No | Calderhead et al. (2009) |
| [`TwoStageSolver`](@ref) | Smooth then match (simple) | Yes | No | Wood (2001) |
| [`BNGSolver`](@ref) | Bayesian neural gradient matching | Yes | No | Bonnaffé et al. (2023) |
| [`AdaptiveGradientMatching`](@ref) | GP product-of-experts | Yes | No | Macdonald & Husmeier (2015) |
| [`AdamSolver`](@ref) | Adam through ODE (UDE-style) | No | No | Rackauckas et al. (2020) |
| [`MultipleShootingSolver`](@ref) | Multiple shooting + Adam | No | No | Turan & Jäschke (2021) |
| [`DerivativeFreeSolver`](@ref) | NelderMead / Particle Swarm | No | No | — |
| [`RodeoSolver`](@ref) | Probabilistic ODE (Kalman) | No | No | Tronarp et al. (2022) |
| [`DaltonSolver`](@ref) | Data-adaptive Kalman likelihood | No | No | Wu & Lysy (2024) |
| [`MCMCSolver`](@ref) | HMC/NUTS posterior sampling | No | Yes | — |
| [`MagiSolver`](@ref) | Manifold-constrained GP inference | No | Yes | Yang et al. (2021) |
| [`PseudoMarginalSolver`](@ref) | Probabilistic ODE + NUTS | No | Yes | Chkrebtii et al. (2016) |
| [`VariationalSolver`](@ref) | Mean-field variational inference | No | Yes | — |
| [`ABCSolver`](@ref) | ABC-SMC (likelihood-free) | No | Yes | — |

### Approximators

| Approximator | Description | Parameters |
|-------------|-------------|------------|
| [`BSplineApproximator`](@ref) | Cubic B-spline basis | Spline coefficients |
| [`ShapeConstrainedBSplineApproximator`](@ref) | SCOP-spline (Pya & Wood 2015) | Constrained coefficients |
| [`NeuralApproximator`](@ref) | Lux.jl neural network | Network weights |
| [`GPApproximator`](@ref) | Gaussian process | GP hyperparameters |
| [`COMONetApproximator`](@ref) | Constrained monotone network | exp(W) weights |

### Likelihoods

- [`Gaussian`](@ref) — Gaussian errors with identity link (default)
- [`Poisson`](@ref) — Count data with log link
- [`NegativeBinomial`](@ref) — Overdispersed counts with estimated dispersion
- [`CustomLikelihood`](@ref) — User-defined likelihood

See the [Getting Started](@ref) guide for a detailed tutorial, or browse the [Solvers](@ref) and [Approximators](@ref) pages for full documentation.
