# Getting Started

This tutorial introduces the basic workflow of PartiallySpecifiedModels.jl using a simple example: recovering an unknown per-capita growth rate from population data.

## Overview

**Partially specified models (PSMs)** are dynamical systems where one or more functional responses are left unspecified and estimated from data using flexible approximators such as penalized B-splines. This approach is particularly useful in ecology, where the form of density-dependent processes is often unknown.

The basic workflow is:

1. Define an ODE model with unknown functions
2. Choose approximators for the unknowns (B-splines, neural networks, etc.)
3. Build a [`PSMProblem`](@ref) with data
4. Solve with a fitting algorithm (e.g., [`LAML`](@ref))
5. Inspect the fitted solution

## Setup

```julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using Random
Random.seed!(42)
```

## A Simple Example: Exponential Growth

Consider a population ``N(t)`` growing according to

```math
\frac{dN}{dt} = r(N) \cdot N
```

where ``r(N)`` is the per-capita growth rate as a function of population size. In standard exponential growth, ``r`` is constant. In logistic growth, ``r(N) = r_0(1 - N/K)``. In a PSM, we leave ``r(N)`` unspecified and estimate it from data.

### Generate synthetic data

We generate data from a logistic model with ``r_0 = 0.5`` and ``K = 10``, observed with Gaussian noise:

```julia
r0, K, N0 = 0.5, 10.0, 0.5
tspan = (0.0, 15.0)
data_times = collect(0.0:0.5:15.0)

function logistic!(du, u, p, t)
    du[1] = r0 * (1 - u[1] / K) * u[1]
end
true_prob = ODEProblem(logistic!, [N0], tspan)
true_sol = OrdinaryDiffEq.solve(true_prob, Tsit5(); saveat=data_times)
true_N = [true_sol(t)[1] for t in data_times]

σ_noise = 0.3
observed_N = true_N .+ σ_noise .* randn(length(data_times))
observed_N = max.(observed_N, 0.01)  # ensure positive
```

### Define the PSM

The ODE function receives a parameter struct `p` containing callable unknown functions and any known parameters. Here, `p.r` is the unknown growth rate function:

```julia
function growth!(du, u, p, t)
    N = u[1]
    du[1] = p.r(N) * N
end
```

### Choose an approximator

We model ``r(N)`` with a **penalized cubic B-spline** having 8 evenly-spaced knots over the range of population sizes:

```julia
approx_r = BSplineApproximator(:r, (0.0, 12.0), 8;
                                initial = x -> 0.3)
```

This creates a spline approximator with:

- **Name**: `:r` — how it appears in the parameter struct
- **Domain**: ``[0, 12]`` — the range of ``N`` values
- **8 knots** — degrees of freedom for the shape of ``r(N)``
- **Initial value**: constant ``r = 0.3`` everywhere

### Build the problem

```julia
prob = PSMProblem(
    growth!,                        # ODE dynamics
    [N0],                           # initial conditions
    tspan,                          # time span
    [approx_r];                     # unknown function approximators
    data_times = data_times,
    data_values = reshape(observed_N, :, 1),  # n_times × 1 matrix
    obs_to_state = [1],             # observe state variable 1
    likelihood = Gaussian(),        # Gaussian errors
    solver = Tsit5()                # ODE solver
)
```

### Solve with LAML

The [`LAML`](@ref) algorithm estimates the spline coefficients and the smoothing parameter ``\lambda`` simultaneously. For Gaussian data, LAML is equivalent to **Restricted Maximum Likelihood (REML)**.

```julia
sol = solve(prob, LAML())
```

### Inspect the solution

The solution contains fitted values, estimated unknown functions, and diagnostics:

```julia
# Fitted trajectory
sol.fitted_values[:, 1]

# Estimated unknown function
r_fitted = sol.unknown_functions[:r]

# Evaluate at new points
N_grid = range(0.1, 11.0, length=100)
r_estimated = [r_fitted(N) for N in N_grid]
```

The fitted ``r(N)`` should resemble the true logistic form ``r(N) = r_0(1 - N/K)`` — an approximately linear decline — without assuming any parametric form.

## Key Concepts

### The `PSMProblem`

A [`PSMProblem`](@ref) combines:

| Component       | Description                                             |
|-----------------|---------------------------------------------------------|
| `dynamics!`     | ODE right-hand side `f!(du, u, p, t)`                   |
| `u0`            | Initial conditions (vector or function of `p`)          |
| `tspan`         | Time interval `(t₀, t₁)`                                |
| `approximators` | Vector of [`AbstractApproximator`] (splines, neural nets) |
| `data_times`    | Observation times                                       |
| `data_values`   | Data matrix (n\_times × n\_obs)                         |
| `likelihood`    | Error distribution ([`Gaussian`](@ref), [`Poisson`](@ref), etc.) |
| `solver`        | ODE solver from OrdinaryDiffEq.jl                       |

You can also construct a `PSMProblem` from SciML problem types:

```julia
# Continuous-time (ODEProblem)
ode = ODEProblem(dynamics!, u0, tspan)
prob = PSMProblem(ode, approximators; data_times=..., data_values=...)

# Discrete-time (DiscreteProblem)
disc = DiscreteProblem(map!, u0, tspan)
prob = PSMProblem(disc, approximators; data_times=..., data_values=...)
```

### Smoothing and EDF

The **effective degrees of freedom (EDF)** measures model complexity. With 8 knots, the maximum EDF is 8 (unpenalized). LAML estimates ``\lambda`` to balance fit and smoothness:

- **Small ``\lambda``**: less smoothing, higher EDF, more flexible
- **Large ``\lambda``**: more smoothing, lower EDF, smoother curves

For the logistic growth example, the EDF should be close to 2 (since the true ``r(N)`` is linear).

## Summary

The basic workflow is:

1. Write `dynamics!(du, u, p, t)` — access unknown functions via `p.name(x)`
2. Create [`BSplineApproximator`](@ref)`(:name, domain, nknots)` for each unknown
3. Build [`PSMProblem`](@ref)`(dynamics!, u0, tspan, approximators; data=..., likelihood=...)`
4. Call `solve(prob, LAML())`
5. Access `sol.unknown_functions[:name]` for the fitted functions

See the [Approximators](@ref) and [Solvers](@ref) pages for detailed documentation of all available options.
