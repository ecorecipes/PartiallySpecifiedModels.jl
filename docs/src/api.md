# API Reference

## Problem and Solution Types

```@docs
PSMProblem
PSMSolution
```

## Approximator Types

Detailed documentation for each approximator is on the [Approximators](approximators.md) page.

```@docs
AbstractApproximator
```

- [`BSplineApproximator`](@ref)
- [`ShapeConstrainedBSplineApproximator`](@ref)
- [`SPDEApproximator`](@ref)
- [`ShapeConstrainedSPDEApproximator`](@ref)
- [`NeuralApproximator`](@ref)
- [`GPApproximator`](@ref)
- [`COMONetApproximator`](@ref)

### Shape-Constraint Constants

The supported constraint symbols for the shape-constrained approximators:

```@docs
SHAPE_CONSTRAINTS
COMONET_CONSTRAINTS
```

## Solver Types

Detailed documentation for each solver is on the [Solvers](solvers.md) page.

- [`LAML`](@ref)
- [`GCVSolver`](@ref)
- [`CollocationLAML`](@ref)
- [`GradientMatching`](@ref)
- [`TwoStageSolver`](@ref)
- [`BNGSolver`](@ref)
- [`AdaptiveGradientMatching`](@ref)
- [`AdamSolver`](@ref)
- [`MultipleShootingSolver`](@ref)
- [`DerivativeFreeSolver`](@ref)
- [`RodeoSolver`](@ref)
- [`DaltonSolver`](@ref)
- [`MCMCSolver`](@ref)
- [`MagiSolver`](@ref)
- [`PseudoMarginalSolver`](@ref)
- [`VariationalSolver`](@ref)
- [`ABCSolver`](@ref)
- [`IntegralMatchingSolver`](@ref)
- [`ProfileLikelihoodSolver`](@ref)
- [`EnsembleKalmanSolver`](@ref)
- [`ODINSolver`](@ref)
- [`RKHSSolver`](@ref)

## Likelihood Types

```@docs
AbstractLikelihood
Gaussian
Poisson
NegativeBinomial
TruncatedNormal
CustomLikelihood
```

## Core Functions

```@docs
solve
simulate
predict
```

## Approximator Functions

```@docs
nparams
initial_params
penalty_matrix
spline_penalty_matrix
optimize_spde_range
with_range_param
```

## Bootstrap Confidence Intervals

```@docs
bootstrap
BootstrapResult
confidence_band
```

## Diagnostics

```@docs
appraise
deviance_residuals
residual_diagnostics
durbin_watson
residual_acf
semivariogram
```
