# PartiallySpecifiedModels.jl

[![CI](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/PartiallySpecifiedModels.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Julia package for fitting **partially specified dynamical models** with nonparametric functional responses.

## Overview

**Partially specified models (PSMs)** are dynamical systems (ODEs, DDEs, or discrete-time maps) in which one or more functional responses are left unspecified and estimated directly from data. Instead of assuming a fixed parametric form for processes like density dependence or predation rates, PSMs replace these unknown functions with flexible nonparametric approximators — penalized B-splines, Gaussian processes, or neural networks — and fit them jointly with the model dynamics.

This approach is particularly valuable in **ecology**, where the form of key biological processes (e.g., Holling-type functional responses, density-dependent growth, transmission rates) is often uncertain or debated. PSMs allow researchers to let the data inform the shape of these relationships, combining mechanistic understanding of system structure with statistical flexibility for unknown components.

PartiallySpecifiedModels.jl provides a unified interface for specifying and fitting PSMs using two complementary approximation strategies:

- **Basis function approximators** (B-splines, shape-constrained splines, Gaussian processes): fewer parameters, automatic smoothing via LAML/GCV, interpretable, and easy to constrain (monotonicity, convexity, positivity).
- **Neural network approximators** (Lux.jl networks, COMONet): more flexible for high-dimensional or complex functional forms, compatible with gradient-based UDE-style training.

The package builds on the [SciML ecosystem](https://sciml.ai/) and supports 23 fitting algorithms, 11 approximator types, 5 likelihood families, and 14 shape constraint types.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/PartiallySpecifiedModels.jl")
```

Requires Julia ≥ 1.12.

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

PartiallySpecifiedModels.jl provides 23 solvers spanning penalized likelihood, gradient matching, probabilistic numerics, and Bayesian inference:

| Solver | Method | ODE-free? | Bayesian? | Reference |
|--------|--------|:---------:|:---------:|-----------|
| `LAML` | Penalized IRLS + LAML smoothing | No | No | Wood et al. (2016) |
| `GCVSolver` | Penalized IRLS + GCV smoothing | No | No | Wood (2001) |
| `CollocationLAML` | Generalized profiling | No | No | Ramsay et al. (2007) |
| `GradientMatching` | Smooth then match derivatives | Yes | No | — |
| `TwoStageSolver` | Smooth then match (simple) | Yes | No | Varah (1982) |
| `BNGSolver` | Ensemble Bayesian gradient matching | Yes | Yes | Bonnaffé & Coulson (2023) |
| `AdaptiveGradientMatching` | GP product-of-experts | Yes | No | Macdonald & Husmeier (2015) |
| `FGPGMSolver` | GP product-of-experts MCMC over states + parameters | Yes | Yes | Wenk et al. (2019) |
| `AdamSolver` | Adam through ODE (UDE-style) | No | No | Rackauckas et al. (2020) |
| `MultipleShootingSolver` | Multiple shooting + Adam | No | No | Turan & Jäschke (2021) |
| `DerivativeFreeSolver` | NelderMead / Particle Swarm | No | No | — |
| `RodeoSolver` | Probabilistic ODE (Kalman) | No | No | Wu & Lysy (2024); Tronarp et al. (2022) for `:fenrir` |
| `DaltonSolver` | Data-adaptive Kalman likelihood | No | No | Wu & Lysy (2024) |
| `MCMCSolver` | HMC/NUTS posterior sampling | No | Yes | — |
| `MagiSolver` | Manifold-constrained GP inference | No | Yes | Yang et al. (2021) |
| `PseudoMarginalSolver` | Pseudo-marginal MCMC (probabilistic ODE likelihood) | No | Yes | Andrieu & Roberts (2009) |
| `VariationalSolver` | Mean-field variational inference | No | Yes | — |
| `ABCSolver` | ABC-SMC (likelihood-free) | No | Yes | — |
| `IntegralMatchingSolver` | Integral matching (smooth, then match integrals) | Yes | No | Dattner & Klaassen (2015) |
| `ProfileLikelihoodSolver` | Profile likelihood identifiability/CIs | No | No | Simpson & Maclaren (2023) |
| `EnsembleKalmanSolver` | Ensemble Kalman inversion | No | No | Iglesias et al. (2013) |
| `ODINSolver` | ODIN-style Mahalanobis gradient matching | Yes | No | Wenk et al. (2020) |
| `RKHSSolver` | Trajectory-in-RKHS gradient matching | Yes | No | González et al. (2014) |

## Approximators

| Approximator | Description | Parameters |
|-------------|-------------|------------|
| `BSplineApproximator` | Cubic B-spline basis | Spline coefficients |
| `ShapeConstrainedBSplineApproximator` | SCOP-spline (Pya & Wood 2015) | Constrained coefficients |
| `TensorBSplineApproximator` | Bivariate tensor-product spline `f(x, y)` | Surface values on the knot grid |
| `SingleIndexApproximator` | `f(u₁,…,u_p) = s(z)`, `z` the standardized index `aᵀu`: nested inner direction + outer smooth | Loadings + outer coefficients |
| `TransformedCovariateApproximator` | `f(t) = s(z(t))`, `z` a standardized learned transform (adaptive exponential smoothing or distributed lag) of an exogenous covariate | Transform parameters + outer coefficients |
| `SPDEApproximator` | Matérn SPDE penalty (Lindgren et al. 2011) | Mesh node values |
| `ShapeConstrainedSPDEApproximator` | SPDE + shape constraints | Constrained mesh values |
| `NeuralApproximator` | Lux.jl neural network | Network weights |
| `GPApproximator` | Gaussian process | GP hyperparameters |
| `ShapeConstrainedGPApproximator` | GP + SCOP shape constraints | Constrained inducing values |
| `COMONetApproximator` | Constrained monotone network | exp(W) weights |

## Features

### Likelihoods

- **`Gaussian()`** — Gaussian errors with identity link (default)
- **`Poisson()`** — Count data, fitted on the response scale (identity link)
- **`NegativeBinomial()`** — Overdispersed counts with fixed dispersion θ, identity link
- **`TruncatedNormal()`** — Continuous data bounded below (e.g. non-negative densities)
- **`CustomLikelihood(loglik_scalar)`** — User-defined likelihood

### Missing / Masked Observations

Mark a missing cell as `NaN` in `data_values` **and** `0.0` in `data_weights`:

```julia
data_values[3, 1]  = NaN
data_weights[3, 1] = 0.0
```

Masked cells are excluded from the objective, the reported loss, the scale
estimate, every denominator and the residual diagnostics — no reshaping of
`data_times`/`data_values` needed. Marking both ways matters: `0 * NaN = NaN`,
so a zero weight alone does not neutralise a `NaN` value.

All solvers support masking **except** `RodeoSolver`, `DaltonSolver`,
`PseudoMarginalSolver` and `EnsembleKalmanSolver`, whose likelihood lives
inside a Kalman/particle recursion with no per-cell mask; they raise a clear
error rather than silently corrupting the filter. Drop the masked rows before
building the problem to use those four.

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

Following Pya & Wood (2015), the monotone and monotone-plus-curvature constraints (`:increasing`, `:decreasing`, `:inc_*`, `:dec_*`) carry a free level, so the fitted function may cross zero. Use `:positive` or `:dec_positive` when positivity is required.

## Vignettes

The `vignettes/` directory contains 35 worked examples:

| # | Vignette | Description |
|---|----------|-------------|
| 01 | Getting Started | Basic PSM workflow with logistic growth |
| 02 | Likelihoods | Gaussian, Poisson, Negative Binomial, and custom likelihoods |
| 03 | Lotka–Volterra | Hare–lynx predator-prey with LAML and collocation |
| 04 | Copepod | 11-stage structured population model with multiple unknown functions |
| 05 | Neural Networks | Comparing B-spline, GP, and neural network approximators on SIR |
| 06 | Solver Comparison | Side-by-side comparison of eleven solvers |
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
| 22 | Two-Stage | Redirect to vignette 09, which covers `TwoStageSolver` alongside the other integration-free methods |
| 23 | Derivative-Free | Nelder-Mead and Particle Swarm optimization |
| 24 | Variational | Fast approximate Bayesian inference via variational methods |
| 25 | ABC | Likelihood-free inference with ABC-SMC |
| 26 | SPDE | Matérn SPDE approximator with shape constraints |
| 27 | Predator–Prey | Predator-prey functional response with confidence intervals |
| 28 | Fisheries | Stock-recruitment dynamics with Poisson counts |
| 29 | Bootstrap | Bootstrap confidence intervals for unknown functions |
| 30 | Model Selection | Model selection with marginal likelihood |
| 31 | Integral Matching | Noise-robust integration-free estimation |
| 32 | Profile Likelihood | Identifiability analysis and likelihood-ratio CIs |
| 33 | Ensemble Kalman | Derivative-free estimation via ensemble Kalman inversion |
| 34 | ODIN | ODE-informed Gaussian process regression |
| 35 | RKHS | Trajectory-in-RKHS gradient matching |

## References

- Wood, S.N. (2001). "Partially specified ecological models." *Ecological Monographs*, 71(1), 1–25.
- Wood, S.N., Pya, N. & Säfken, B. (2016). "Smoothing parameter and model selection for general smooth models." *JASA*, 111(516), 1548–1575.
- Ramsay, J.O., Hooker, G., Campbell, D. & Cao, J. (2007). "Parameter estimation for differential equations: a generalized smoothing approach." *JRSS-B*, 69(5), 741–796.
- Pya, N. & Wood, S.N. (2015). "Shape constrained additive models." *Statistics and Computing*, 25(3), 543–559.
- Lindgren, F., Rue, H. & Lindström, J. (2011). "An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach." *JRSS-B*, 73(4), 423–498.
- Rackauckas, C. et al. (2020). "Universal differential equations for scientific machine learning." *arXiv:2001.04385*.
- Macdonald, B. & Husmeier, D. (2015). "Gradient matching methods for computational inference in mechanistic models for systems biology: a review and comparative analysis." *Frontiers in Bioengineering and Biotechnology*, 3, 180.
- Turan, E.M. & Jäschke, J. (2021). "Multiple shooting for training neural differential equations on time series." *IEEE Control Systems Letters*, 6, 1897–1902.
- Yang, S., Wong, S.W.K. & Kou, S.C. (2021). "Inference of dynamic systems from noisy and sparse data via manifold-constrained Gaussian processes." *PNAS*, 118(15).
- Bonnaffé, W., Sheldon, B.C. & Coulson, T. (2021). "Neural ordinary differential equations for ecological and evolutionary time-series analysis." *Methods in Ecology and Evolution*, 12, 1301–1315.
- Bonnaffé, W. & Coulson, T. (2023). "Fast fitting of neural ordinary differential equations by Bayesian neural gradient matching to infer ecological interactions from time-series data." *Methods in Ecology and Evolution*, 14, 1543–1563.
- Varah, J.M. (1982). "A spline least squares method for numerical parameter estimation in differential equations." *SIAM Journal on Scientific and Statistical Computing*, 3(1), 28–46.
- Andrieu, C. & Roberts, G.O. (2009). "The pseudo-marginal approach for efficient Monte Carlo computations." *Annals of Statistics*, 37(2), 697–725.
- Wu, M. & Lysy, M. (2024). "Data-adaptive probabilistic likelihood approximation for ordinary differential equations." *AISTATS*, PMLR 238.
- Tronarp, F., Bosch, N. & Hennig, P. (2022). "Fenrir: Physics-enhanced regression for initial value problems." *ICML*.
- Dattner, I. & Klaassen, C.A.J. (2015). "Optimal rate of direct estimators in systems of ordinary differential equations linear in functions of the parameters." *Electronic Journal of Statistics*, 9(2), 1939–1973.
- Simpson, M.J. & Maclaren, O.J. (2023). "Profile-wise analysis: a profile likelihood-based workflow for identifiability analysis, estimation, and prediction with mechanistic mathematical models." *PLOS Computational Biology*, 19(9), e1011515.
- Iglesias, M.A., Law, K.J.H. & Stuart, A.M. (2013). "Ensemble Kalman methods for inverse problems." *Inverse Problems*, 29(4), 045001.
- Wenk, P., Abbati, G., Osborne, M.A., Schölkopf, B., Krause, A. & Bauer, S. (2020). "ODIN: ODE-informed regression for parameter and state inference in time-continuous dynamical systems." *AAAI*.
- González, J., Vujačić, I. & Wit, E. (2014). "Reproducing kernel Hilbert space based estimation of systems of ordinary differential equations." *Pattern Recognition Letters*, 45, 26–32.

## License

MIT — see [LICENSE](LICENSE).
