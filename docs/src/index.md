# PartiallySpecifiedModels.jl

A Julia package for fitting **partially specified dynamical models** with nonparametric functional responses.

## Overview

**Partially specified models (PSMs)** are dynamical systems (ODEs, DDEs, or discrete-time maps) in which one or more functional responses are left unspecified and estimated directly from data. Instead of assuming a fixed parametric form for processes like density dependence or predation rates, PSMs replace these unknown functions with flexible nonparametric approximators — penalized B-splines, Gaussian processes, or neural networks — and fit them jointly with the model dynamics.

This approach is particularly valuable in **ecology**, where the form of key biological processes (e.g., Holling-type functional responses, density-dependent growth, transmission rates) is often uncertain or debated. PSMs allow researchers to let the data inform the shape of these relationships, combining mechanistic understanding of system structure with statistical flexibility for unknown components.

PartiallySpecifiedModels.jl provides a unified interface for specifying and fitting PSMs using two complementary approximation strategies:

- **Basis function approximators** (B-splines, shape-constrained splines, Gaussian processes): fewer parameters, automatic smoothing via LAML/GCV, interpretable, and easy to constrain (monotonicity, convexity, positivity).
- **Neural network approximators** (Lux.jl networks, COMONet): more flexible for high-dimensional or complex functional forms, compatible with gradient-based UDE-style training.

The package supports 22 fitting algorithms, 7 approximator types, 5 likelihood families, and 14 shape constraint types.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/PartiallySpecifiedModels.jl")
```

Requires Julia ≥ 1.10.

## Quick Start

A simple example: recover the unknown per-capita growth rate ``r(N)`` from logistic growth data.

```@example quickstart
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

# Build problem and fit with LAML
prob = PSMProblem(
    growth!, [N0], tspan, [approx_r];
    data_times = data_times,
    data_values = reshape(observed_N, :, 1),
    obs_to_state = [1],
    likelihood = Gaussian(),
    solver = Tsit5()
)
sol = solve(prob, LAML())

# Evaluate the recovered unknown function
r_fitted = sol.unknown_functions[:r]
for N in [1.0, 5.0, 9.0]
    println("r($N): estimated=$(round(r_fitted(N), digits=3)), true=$(round(r0*(1-N/K), digits=3))")
end
```

## Package Features

### Solvers

| Solver | Method | ODE-free? | Bayesian? | Reference |
|--------|--------|:---------:|:---------:|-----------|
| [`LAML`](@ref) | Penalized IRLS + LAML smoothing | No | No | Wood et al. (2016) |
| [`GCVSolver`](@ref) | Penalized IRLS + GCV smoothing | No | No | Wood (2001) |
| [`CollocationLAML`](@ref) | Generalized profiling | No | No | Ramsay et al. (2007) |
| [`GradientMatching`](@ref) | Smooth then match derivatives | Yes | No | — |
| [`TwoStageSolver`](@ref) | Smooth then match (simple) | Yes | No | Varah (1982) |
| [`BNGSolver`](@ref) | Ensemble Bayesian gradient matching | Yes | Yes | Bonnaffé & Coulson (2023) |
| [`AdaptiveGradientMatching`](@ref) | GP product-of-experts | Yes | No | Macdonald & Husmeier (2015) |
| [`AdamSolver`](@ref) | Adam through ODE (UDE-style) | No | No | Rackauckas et al. (2020) |
| [`MultipleShootingSolver`](@ref) | Multiple shooting + Adam | No | No | Turan & Jäschke (2021) |
| [`DerivativeFreeSolver`](@ref) | NelderMead / Particle Swarm | No | No | — |
| [`RodeoSolver`](@ref) | Probabilistic ODE (Kalman) | No | No | Wu & Lysy (2024); Tronarp et al. (2022) for `:fenrir` |
| [`DaltonSolver`](@ref) | Data-adaptive Kalman likelihood | No | No | Wu & Lysy (2024) |
| [`MCMCSolver`](@ref) | HMC/NUTS posterior sampling | No | Yes | — |
| [`MagiSolver`](@ref) | Manifold-constrained GP inference | No | Yes | Yang et al. (2021) |
| [`PseudoMarginalSolver`](@ref) | Pseudo-marginal MCMC (probabilistic ODE likelihood) | No | Yes | Andrieu & Roberts (2009) |
| [`VariationalSolver`](@ref) | Mean-field variational inference | No | Yes | — |
| [`ABCSolver`](@ref) | ABC-SMC (likelihood-free) | No | Yes | — |
| [`IntegralMatchingSolver`](@ref) | Integral matching (smooth, then match integrals) | Yes | No | Dattner & Klaassen (2015) |
| [`ProfileLikelihoodSolver`](@ref) | Profile likelihood identifiability/CIs | No | No | Simpson & Maclaren (2023) |
| [`EnsembleKalmanSolver`](@ref) | Ensemble Kalman inversion | No | No | Iglesias et al. (2013) |
| [`ODINSolver`](@ref) | ODIN-style Mahalanobis gradient matching | Yes | No | Wenk et al. (2020) |
| [`RKHSSolver`](@ref) | Trajectory-in-RKHS gradient matching | Yes | No | González et al. (2014) |

### Approximators

| Approximator | Description | Parameters |
|-------------|-------------|------------|
| [`BSplineApproximator`](@ref) | Cubic B-spline basis | Spline coefficients |
| [`ShapeConstrainedBSplineApproximator`](@ref) | SCOP-spline (Pya & Wood 2015) | Constrained coefficients |
| [`SPDEApproximator`](@ref) | Matérn SPDE penalty (Lindgren et al. 2011) | Mesh node values |
| [`ShapeConstrainedSPDEApproximator`](@ref) | SPDE + shape constraints | Constrained mesh values |
| [`NeuralApproximator`](@ref) | Lux.jl neural network | Network weights |
| [`GPApproximator`](@ref) | Gaussian process | GP hyperparameters |
| [`COMONetApproximator`](@ref) | Constrained monotone network | exp(W) weights |

### Likelihoods

- [`Gaussian`](@ref) — Gaussian errors with identity link (default)
- [`Poisson`](@ref) — Count data, fitted on the response scale (identity link)
- [`NegativeBinomial`](@ref) — Overdispersed counts with fixed dispersion θ, identity link
- [`TruncatedNormal`](@ref) — Continuous data bounded below (e.g. non-negative densities)
- [`CustomLikelihood`](@ref) — User-defined likelihood

See the [Getting Started](getting_started.md) guide for a detailed tutorial, or browse the [Solvers](solvers.md) and [Approximators](approximators.md) pages for full documentation. The [Vignettes](vignettes.md) page has 35 worked examples covering every solver and approximator.
