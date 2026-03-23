# API Reference

## Problem and Solution Types

```@docs
PSMProblem
PSMSolution
```

## Approximator Types

```@docs
AbstractApproximator
BSplineApproximator
ShapeConstrainedBSplineApproximator
NeuralApproximator
GPApproximator
COMONetApproximator
```

## Solver Types

```@docs
LAML
GCVSolver
CollocationLAML
GradientMatching
TwoStageSolver
BNGSolver
AdaptiveGradientMatching
AdamSolver
MultipleShootingSolver
DerivativeFreeSolver
RodeoSolver
DaltonSolver
MCMCSolver
MagiSolver
PseudoMarginalSolver
VariationalSolver
ABCSolver
```

## Likelihood Types

```@docs
AbstractLikelihood
Gaussian
Poisson
NegativeBinomial
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
```

## Diagnostics

```@docs
residual_diagnostics
durbin_watson
residual_acf
semivariogram
```
