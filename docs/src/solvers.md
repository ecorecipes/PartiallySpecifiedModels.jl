# Solvers

PartiallySpecifiedModels.jl provides 17 solvers for fitting partially specified models. Solvers are passed as the second argument to `solve`:

```@example solvers
using PartiallySpecifiedModels # hide
using PartiallySpecifiedModels: solve # hide
using OrdinaryDiffEq, Random # hide
Random.seed!(42) # hide
function growth!(du, u, p, t) # hide
    du[1] = p.r(u[1]) * u[1] # hide
end # hide
approx_r = BSplineApproximator(:r, (0.0, 12.0), 8; initial = x -> 0.3) # hide
r0, K, N0 = 0.5, 10.0, 0.5 # hide
true_sol = OrdinaryDiffEq.solve(ODEProblem((du,u,p,t) -> du[1] = r0*(1-u[1]/K)*u[1], [N0], (0.0,15.0)), Tsit5(); saveat=0.5) # hide
obs = max.([true_sol.u[i][1] + 0.3*randn() for i in 1:length(true_sol.t)], 0.01) # hide
prob = PSMProblem(growth!, [N0], (0.0,15.0), [approx_r]; data_times=collect(true_sol.t), data_values=reshape(obs,:,1), obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5()) # hide
sol = solve(prob, LAML())
println("Data loss: ", round(sol.data_loss, digits=4), ", EDF: ", round(sol.edf, digits=2))
```

## Penalized Likelihood Solvers

### LAML

Penalized Iteratively Reweighted Least Squares (P-IRLS) with **Laplace Approximate Marginal Likelihood** for automatic smoothing parameter selection. The default and recommended solver for B-spline approximators. For Gaussian data, LAML is equivalent to REML.

```@docs
LAML
```

### GCVSolver

Penalized IRLS with **Generalized Cross-Validation** for smoothing parameter selection. An alternative to LAML that minimizes leave-one-out prediction error.

```@docs
GCVSolver
```

### CollocationLAML

**Generalized profiling** (collocation) approach. Fits spline approximations to the state variables first, then optimizes the unknown function parameters to match the implied derivatives. Can be more robust than direct ODE fitting for stiff or chaotic systems.

```@docs
CollocationLAML
```

## Gradient Matching Solvers

These solvers avoid numerical ODE integration entirely by matching derivatives of smoothed data to the model equations.

### GradientMatching

Smooth the observed data, compute numerical derivatives, then fit the unknown functions to match. Requires good data coverage and low noise.

```@docs
GradientMatching
```

### TwoStageSolver

The simplest gradient matching approach: smooth the data with splines, differentiate, then regress the unknown functions on the derivatives. A useful baseline for comparison.

```@docs
TwoStageSolver
```

### AdaptiveGradientMatching

GP-based gradient matching using a **product-of-experts** formulation. Iteratively refines the GP fit and the parameter estimates. More robust than simple gradient matching.

```@docs
AdaptiveGradientMatching
```

### BNGSolver

**Bayesian Neural Gradient matching**. Uses neural networks for gradient matching with Bayesian regularization. Fast and suitable for complex dynamics.

```@docs
BNGSolver
```

## Optimization-Based Solvers

### AdamSolver

Gradient-based optimization through the ODE solver using the **Adam** optimizer. This is the standard approach for Universal Differential Equations (UDEs). Works with all approximator types including neural networks.

```@docs
AdamSolver
```

### MultipleShootingSolver

**Multiple shooting** with Adam optimization. Divides the time span into segments and optimizes continuity constraints. More robust than single shooting for long time series or chaotic dynamics.

```@docs
MultipleShootingSolver
```

### DerivativeFreeSolver

Derivative-free optimization using **Nelder-Mead** or **Particle Swarm** methods. A robust fallback when gradient-based methods fail. Slower but more reliable for non-smooth or multimodal objectives.

```@docs
DerivativeFreeSolver
```

## Probabilistic Numerics Solvers

### RodeoSolver

**Probabilistic ODE solver** based on Kalman filtering/smoothing. Provides uncertainty estimates from the numerical integration itself, not just from parameter uncertainty.

```@docs
RodeoSolver
```

### DaltonSolver

**Data-adaptive likelihood** with transformed observations using Kalman filtering. Combines probabilistic ODE solving with a likelihood that adapts to the data structure.

```@docs
DaltonSolver
```

## Bayesian Solvers

### MCMCSolver

Full Bayesian inference using **Hamiltonian Monte Carlo (HMC)** or **No-U-Turn Sampler (NUTS)**. Provides posterior distributions over all parameters including unknown function coefficients.

```@docs
MCMCSolver
```

### MagiSolver

**Manifold-constrained Gaussian Process Inference**. Uses GPs to represent both state trajectories and unknown functions, with the ODE constraints enforced on a manifold. Handles partially observed systems well.

```@docs
MagiSolver
```

### PseudoMarginalSolver

Combines **probabilistic ODE solving** with Bayesian MCMC via pseudo-marginal methods. The ODE solver uncertainty is marginalized out, providing fully Bayesian inference that accounts for numerical error.

```@docs
PseudoMarginalSolver
```

### VariationalSolver

**Mean-field variational inference** for fast approximate Bayesian inference. Approximates the posterior with a factorized Gaussian distribution. Much faster than MCMC but may underestimate uncertainty.

```@docs
VariationalSolver
```

### ABCSolver

**Approximate Bayesian Computation with Sequential Monte Carlo** (ABC-SMC). Likelihood-free inference that only requires the ability to simulate from the model. Useful when the likelihood is intractable.

```@docs
ABCSolver
```

## Choosing a Solver

| Use case | Recommended solver |
|----------|-------------------|
| Default / first try | [`LAML`](@ref) |
| Automatic smoothing comparison | [`GCVSolver`](@ref) |
| Stiff or chaotic systems | [`CollocationLAML`](@ref) |
| Quick baseline | [`TwoStageSolver`](@ref) |
| Neural network approximators | [`AdamSolver`](@ref) |
| Robust neural fitting | [`MultipleShootingSolver`](@ref) |
| When gradients fail | [`DerivativeFreeSolver`](@ref) |
| Uncertainty quantification | [`MCMCSolver`](@ref) or [`MagiSolver`](@ref) |
| Fast approximate Bayesian | [`VariationalSolver`](@ref) |
| Intractable likelihood | [`ABCSolver`](@ref) |
| Probabilistic numerics | [`RodeoSolver`](@ref) or [`DaltonSolver`](@ref) |
