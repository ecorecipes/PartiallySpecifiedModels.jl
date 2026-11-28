# PartiallySpecifiedModels.jl

A Julia package for fitting **partially specified dynamical models** with nonparametric functional responses.

## Overview

**Partially specified models (PSMs)** are dynamical systems (ODEs, DDEs, or discrete-time maps) in which one or more functional responses are left unspecified and estimated directly from data. Instead of assuming a fixed parametric form for processes like density dependence or predation rates, PSMs replace these unknown functions with flexible nonparametric approximators — penalized B-splines, Gaussian processes, or neural networks — and fit them jointly with the model dynamics.

This approach is particularly valuable in **ecology**, where the form of key biological processes (e.g., Holling-type functional responses, density-dependent growth, transmission rates) is often uncertain or debated. PSMs allow researchers to let the data inform the shape of these relationships, combining mechanistic understanding of system structure with statistical flexibility for unknown components.

PartiallySpecifiedModels.jl provides a unified interface for specifying and fitting PSMs using two complementary approximation strategies:

- **Basis function approximators** (B-splines, shape-constrained splines, Gaussian processes): fewer parameters, automatic smoothing via LAML/GCV, interpretable, and easy to constrain (monotonicity, convexity, positivity).
- **Neural network approximators** (Lux.jl networks, fixed-grid KANs, COMONet): more flexible for high-dimensional or complex functional forms, compatible with gradient-based UDE-style training.

The package supports 24 fitting algorithms (including continuous-discrete EKF for SDEs), 15 approximator types (including optional `FluxKAN.jl` and `GAM.jl` backends), 9 likelihood families (plus direct `Distributions.jl` coercion), and 14 shape constraint types.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/PartiallySpecifiedModels.jl")
```

Requires Julia ≥ 1.13. Upgrading from v0.2? See [Migration to v0.3](migration_v03.md).

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
    likelihood = GaussianLikelihood(),
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
| [`LAML`](@ref) | Penalized IRLS + LAML smoothing (exact Wood 2011 Newton) | No | No | Wood et al. (2016) |
| [`EKFSolver`](@ref) | Continuous-discrete EKF + RTS smoother (`SDEPSMProblem`) | No | No | Särkkä & Solin (2019) |
| [`MultiStart`](@ref) | Best of several `LAML` fits from re-seeded random initializations | No | No | — |
| [`GCVSolver`](@ref) | Penalized IRLS + GCV/NCV smoothing | No | No | Wood (2001, 2024) |
| [`CollocationLAML`](@ref) | Generalized profiling | No | No | Ramsay et al. (2007) |
| [`GradientMatching`](@ref) | Smooth then match derivatives | Yes | No | — |
| [`TwoStageSolver`](@ref) | Smooth then match (simple) | Yes | No | Varah (1982); Brunel (2008) |
| [`BNGSolver`](@ref) | Ensemble Bayesian gradient matching | Yes | Yes | Bonnaffé & Coulson (2023) |
| [`AdaptiveGradientMatching`](@ref) | GP product-of-experts | Yes | No | Macdonald & Husmeier (2015) |
| [`AdamSolver`](@ref) | Adam through ODE (UDE-style) | No | No | Rackauckas et al. (2020) |
| [`MultipleShootingSolver`](@ref) | Multiple shooting (Augmented Lagrangian + L-BFGS) | No | No | Turan & Jäschke (2021) |
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
| [`FGPGMSolver`](@ref) | Fast GP-based gradient matching (posterior samples) | Yes | Yes | Wenk et al. (2019) |

### Approximators

| Approximator | Description | Parameters |
|-------------|-------------|------------|
| [`BSplineApproximator`](@ref) | Cubic B-spline basis | Spline coefficients |
| [`CyclicBSplineApproximator`](@ref) | $C^2$ periodic cubic B-spline for seasonal forcing | Periodic spline coefficients |
| [`ShapeConstrainedBSplineApproximator`](@ref) | SCOP-spline (Pya & Wood 2015) | Constrained coefficients |
| [`SPDEApproximator`](@ref) | Matérn SPDE penalty (Lindgren et al. 2011) | Mesh node values |
| [`ShapeConstrainedSPDEApproximator`](@ref) | SPDE + shape constraints | Constrained mesh values |
| [`NeuralApproximator`](@ref) | Lux.jl neural network | Network weights |
| [`KANApproximator`](@ref) | Fixed-grid cubic KAN (load `FluxKAN.jl`) | Edge spline and residual weights |
| [`GAMSmoothApproximator`](@ref) | Any `GAM.jl` smooth basis (load `GAM.jl`) | Smooth basis coefficients |
| [`RandomEffectApproximator`](@ref) | Group-level random effects (`S = I`) | Group random effects |
| [`GPApproximator`](@ref) | Gaussian process kernel interpolant | Inducing-point function values |
| [`ShapeConstrainedGPApproximator`](@ref) | GP + shape constraints | Constrained inducing values |
| [`COMONetApproximator`](@ref) | Constrained monotone network | exp(W) weights |
| [`TensorBSplineApproximator`](@ref) | Bivariate tensor-product spline (isotropic or `t2` anisotropic) | Tensor-product coefficients |
| [`SingleIndexApproximator`](@ref) | Univariate smooth of one learned direction (Fasiolo et al. 2025) | Index loadings + outer-smooth coefficients |
| [`TransformedCovariateApproximator`](@ref) | Smooth of a learned transform of an exogenous covariate | Transform + smooth coefficients |

### Likelihoods

- [`GaussianLikelihood`](@ref) — Gaussian errors with identity link (default)
- [`PoissonLikelihood`](@ref) — Count data, fitted on the response scale (identity link)
- [`NegativeBinomialLikelihood`](@ref) — Overdispersed counts with fixed dispersion θ, identity link
- [`TruncatedNormalLikelihood`](@ref) — Continuous data bounded below (e.g. non-negative densities)
- [`GammaLikelihood`](@ref) — Positive skewed continuous data (biomass, CPUE)
- [`LogNormalLikelihood`](@ref) — Multiplicative log-scale observation error
- [`BinomialLikelihood`](@ref) — Binomial counts / proportions
- [`StudentTLikelihood`](@ref) — Heavy-tailed robust observations
- [`CustomLikelihood`](@ref) — User-defined likelihood
- Direct `Distributions.jl` coercion (`Normal`, `Poisson`, `NegativeBinomial`, `truncated(Normal(...), lo, Inf)`, `Gamma`, `LogNormal`, `Binomial`, `TDist`) when `Distributions` is loaded.

See the [Getting Started](getting_started.md) guide for a detailed tutorial. The [Vignettes](vignettes.md) page has 48 worked examples covering every solver, approximator, and `ecorecipes` integration workflow; for the complete reference, browse the [Solvers](solvers.md) and [Approximators](approximators.md) pages.
