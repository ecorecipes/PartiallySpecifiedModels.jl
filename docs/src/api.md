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
- [`TensorBSplineApproximator`](@ref)
- [`SingleIndexApproximator`](@ref)
- [`TransformedCovariateApproximator`](@ref)
- [`ShapeConstrainedBSplineApproximator`](@ref)
- [`SPDEApproximator`](@ref)
- [`ShapeConstrainedSPDEApproximator`](@ref)
- [`NeuralApproximator`](@ref)
- [`KANApproximator`](@ref)
- [`GPApproximator`](@ref)
- [`ShapeConstrainedGPApproximator`](@ref)
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
- [`FGPGMSolver`](@ref)
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

## Function Uncertainty

`confidence_band` uses a fitted parameter covariance; `bootstrap` resamples
and refits using the supplied algorithm. Both accept
`uf_points=Dict(:g => points)` for physical query coordinates, one sample per
row and one input per column (a vector also works for unary inputs).
Explicit points support KANs and tensor surfaces without treating unvisited
regions as identified. Single-index and transformed-covariate queries
evaluate the full response; their automatic grids still describe the outer
standardized curve.

The default intervals are pointwise. Opt into per-function, finite-grid
simultaneous bands with:

```julia
using Random
conditional = confidence_band(sol, prob; interval=:simultaneous,
    uf_points=Dict(:r => x), nsim=10_000, rng=Random.Xoshiro(42))
bs = bootstrap(sol, prob, algorithm; uf_points=Dict(:r => x))
full_refit = confidence_band(bs, sol, prob; interval=:simultaneous)
```

The covariance method uses the maximum standardized Gaussian/delta
deviation. The bootstrap method uses the maximum standardized refit
deviation around the original fitted function; it is not a simultaneous
percentile interval or bootstrap-t with replicate-specific standard errors.
Its complete function draws preserve dependence across query points.
`uf_indices=Dict(:r => ids)` selects a stored bootstrap grid before
calibration. Returned metadata identifies critical values, scope,
conditioning and draw counts.

Each function's grid is handled separately. This is neither joint coverage
across different unknown functions nor a continuum guarantee. Default
covariance intervals condition on the selected smoothing parameters.
Bootstrap reselects smoothing
only if its algorithm does. For KANs, architecture, grids and cached
initialization stay fixed. `BootstrapResult.uf_success` reports usable
replicate counts per point; fewer than three yields NaN endpoints.

### Analytic smoothing-parameter uncertainty

For **Gaussian LAML** fits, opt into smoothing-uncertainty correction
independently of the pointwise/simultaneous target:

```julia
adjusted = confidence_band(sol, prob; unconditional=true,
    uf_points=Dict(:r => x))
adjusted_joint = confidence_band(sol, prob; unconditional=true,
    interval=:simultaneous, uf_points=Dict(:r => x),
    nsim=10_000, rng=Random.Xoshiro(42))
correction = smoothing_covariance_correction(sol, prob)
C = correction.covariance  # already scaled; do NOT multiply by sigma2 again
```

This implements both terms of Wood, Pya and Saefken (2016), equation 7:
the covariance of the coefficient-mean response to log smoothing, and
the inverse-Cholesky covariance-root derivative term. The latter remains
important when fitted components are near zero; this is not just the
mean-only Kass--Steffey correction.

The log-smoothing covariance comes from analytic profiled-REML curvature
of the final **local linear Gaussian working model**, including coefficient
and profiled-scale responses. Data Jacobians/weights are frozen, and the
reported coefficient dispersion is held fixed for root differentiation.
It does not integrate dispersion, GP kernel parameters, architecture, grids,
or arbitrary model selection, and does not differentiate a nonlinear ODE's
full higher-order likelihood curvature. The fitted mean does not change.

The default adds **no** covariance-only prior precision. Unidentified or
indefinite log-smoothing curvature raises an error rather than falling
back to conditional bands. `rho_regularization=0.1`, for example, explicitly
adds that isotropic precision to the smoothing Hessian for **both** terms;
it does not refit the smoothing mean or establish an optimum. This differs
from mgcv's separate regularization of the root correction, so it has the
same purpose as `gratia::confint.gam(unconditional=TRUE)`, not a claim of
bit-for-bit equivalence.

For a fixed/external smoothing refit, supply `rho_covariance` explicitly.
With a predeclared multiplicative undersmoothing factor,
`log(lambda_final) = log(lambda_selected) + log(factor)`, so the
selection-stage log-smoothing covariance carries through unchanged:

```julia
Vrho = smoothing_covariance_correction(selection_fit, selection_prob).rho_covariance
adjusted_refit = confidence_band(final_fit, final_prob; unconditional=true,
    rho_covariance=Vrho, uf_points=Dict(:r => x))
```

The sensitivities are evaluated at `final_fit`, not at the selection fit.
Penalty-block ordering must match. Do not infer a smoothing covariance
from an intentionally fixed-lambda fit, and do not add this correction
to a full-refit bootstrap a second time. Non-Gaussian/other-solver fits,
old fits without saved working information, and automatic estimates on
an optimization bound are not supported by the default correction.

Corrected bands identify `conditioning=:smoothing_corrected` and
`covariance_method=:wps_local_gaussian`. The helper exposes each additive
matrix, the smoothing Hessian/covariance, coefficient sensitivities,
coefficient score/step, and stationarity diagnostics. Numerical availability
does not establish local normality, a stationary fit, or nominal coverage.
The [formula and retained-data assessment](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/SMOOTHING-COVARIANCE.md)
documents independent derivative comparisons, correction availability,
width changes and difficult cases where undercoverage persists.
The subsequent [stable-profile diagnostics](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/SMOOTHING-PROFILES.md)
also identify numerical sensitivity in smoothing selection for weak KAN
directions. The covariance correction does not reselect smoothing or
repair that selection criterion; profile-selected conditional bands still
undercover in the tested difficult conditions.

Interval availability is not empirical calibration. The
[fixed-grid KAN coverage study](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/UNCERTAINTY-CALIBRATION.md)
compares covariance and bootstrap intervals against a spline baseline
under repeated datasets, reporting widths, failures and Monte Carlo
uncertainty separately for in-range and extrapolative queries.
In its nonlinear fixture, nominal 95% KAN intervals covered only about
69% (covariance) and 55% (percentile bootstrap) of in-range targets on average.
This is a measured limitation of that specified fitting procedure, not a
universal coverage assessment of every supported approximator.
The [follow-on comparison](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/UNCERTAINTY-METHODS.md)
shows that those KAN figures depend strongly on the example's nondefault
affine penalty: the default unpenalized affine component gives about 91%
covariance coverage in an exploratory replay. Changed-prior bootstrap
coverage is now addressed by the
[fresh-data study](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/FRESH-COVERAGE.md):
on its nonlinear truth, free-affine KAN coverage was 91.8% for covariance
and 77.5% for recomputed percentile-bootstrap intervals. Removing the
affine penalty helps but does not restore nominal 95% coverage.
The [targeted bias investigation](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/BIAS-UNDERSMOOTHING.md)
then evaluated a predeclared two-stage undersmoothing procedure on
independent data. It improved coverage with wider intervals, but does not
justify a universal multiplier or a default change. Its bootstrap repeats
both smoothing selection and the fixed-weight coefficient refit.
The [robustness continuation](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/COVERAGE-ROBUSTNESS.md)
shows why this is not a universal rule: gains persist, but coverage can
remain far below nominal with higher noise, sparse observations or a more
localized response. Conditions and interval widths must be assessed separately.

The [pointwise/simultaneous assessment](https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/main/benchmarks/kan/SIMULTANEOUS-BANDS.md)
scores whole-grid containment separately from average pointwise containment
and explicitly labels conditional-covariance versus full-refit bootstrap
uncertainty.
In its reference setting, quarter-strength KAN simultaneous bands covered
all 76 in-range query points in 30/30 datasets (conditional Gaussian) and
28/30 (full-refit bootstrap). The corresponding pointwise bands covered
the whole grid in only 23/30 and 17/30. Simultaneous coverage nevertheless
remained poor under higher noise and some response shapes; neither target
nor smoothing-aware refitting provides an automatic coverage guarantee.

```@docs
bootstrap
BootstrapResult
confidence_band
smoothing_covariance_correction
```

## Diagnostics

```@docs
appraise
deviance_residuals
residual_diagnostics
durbin_watson
residual_acf
semivariogram
check_constraints
kan_edge_curves
kan_activation_diagnostics
```
