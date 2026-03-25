# Approximators

PartiallySpecifiedModels.jl provides seven approximator types for representing unknown functions in dynamical systems. Each approximator is a callable object that maps scalar inputs to scalar outputs and is fitted as part of the model.

## BSplineApproximator

Cubic B-spline basis with automatic smoothing via penalized least squares. The default and most commonly used approximator.

```@example approx
using PartiallySpecifiedModels # hide
approx = BSplineApproximator(:f, (0.0, 10.0), 8;
                              initial = x -> 0.3)
```

- **`:f`**: Symbol used to access the function in the parameter struct (`p.f(x)`)
- **`(0.0, 10.0)`**: Domain of the input variable
- **`8`**: Number of evenly-spaced knots (controls flexibility)
- **`initial`**: Function or constant for initial coefficient values

The smoothing parameter ``\lambda`` is estimated automatically by LAML or GCV.

```@docs
BSplineApproximator
```

## ShapeConstrainedBSplineApproximator

SCOP-spline (Shape-Constrained P-spline) that enforces shape constraints on the fitted function via reparameterization of the B-spline coefficients. Supports 14 constraint types.

```@example approx
approx_sc = ShapeConstrainedBSplineApproximator(:f, (0.0, 10.0), 8, :increasing)
```

### Available constraints

| Constraint | Description |
|------------|-------------|
| `:increasing` | Monotonically increasing |
| `:decreasing` | Monotonically decreasing |
| `:convex` | Convex (curves upward) |
| `:concave` | Concave (curves downward) |
| `:inc_convex` | Increasing and convex |
| `:inc_concave` | Increasing and concave |
| `:dec_convex` | Decreasing and convex |
| `:dec_concave` | Decreasing and concave |
| `:positive` | Non-negative everywhere |
| `:dec_positive` | Decreasing and non-negative |
| `:inc_zero_left` | Increasing, zero at left endpoint |
| `:inc_zero_right` | Increasing, zero at right endpoint |
| `:dec_zero_left` | Decreasing, zero at left endpoint |
| `:dec_zero_right` | Decreasing, zero at right endpoint |

Zero-at-endpoint constraints fix one knot coefficient to zero, reducing the parameter count by 1.

```@docs
ShapeConstrainedBSplineApproximator
```

## SPDEApproximator

Matérn SPDE (Stochastic Partial Differential Equation) approximator following [Lindgren et al. (2011)](https://doi.org/10.1111/j.1467-9868.2011.00777.x). Uses a finite element discretization of the Matérn SPDE to build a precision-based penalty matrix, providing an interpretable alternative to B-spline penalties.

```@example approx
approx_spde = SPDEApproximator(:f, (0.0, 10.0), 10;
                                nu = 1.5,
                                range_param = 3.0,
                                initial = x -> 0.0)
```

- **`nu`**: Matérn smoothness parameter — 0.5 (rough), 1.5 (once-differentiable, default), or 2.5 (twice-differentiable)
- **`range_param`**: Spatial correlation length ``\rho``. Controls the scale over which the function varies. Defaults to 1/3 of the domain width.
- The overall smoothing strength ``\tau^2`` is still estimated automatically via LAML/GCV, separately from the range.

Use [`optimize_spde_range`](@ref) to select the range parameter automatically via profile GCV.

```@docs
SPDEApproximator
```

## ShapeConstrainedSPDEApproximator

Combines the Matérn SPDE penalty with SCOP-spline reparameterization to enforce shape constraints at mesh nodes. Supports all 14 constraint types (same as [`ShapeConstrainedBSplineApproximator`](@ref)).

```@example approx
approx_scspde = ShapeConstrainedSPDEApproximator(:f, (0.0, 10.0), 10, :increasing;
                                                  nu = 1.5, range_param = 3.0)
```

Constraints are enforced at mesh nodes via a cumulative-sum reparameterization through `softplus`. The cubic spline interpolation between nodes may slightly overshoot; use more basis functions to reduce this.

!!! tip
    Simple constraints (`:increasing`, `:decreasing`, `:concave`) tend to converge more reliably than combined constraints (`:inc_concave`, `:dec_positive`) which can trap the optimizer in local optima.

```@docs
ShapeConstrainedSPDEApproximator
```

## NeuralApproximator

Neural network approximator using [Lux.jl](https://github.com/LuxDL/Lux.jl). Suitable for complex or high-dimensional unknown functions. Compatible with gradient-based solvers (`AdamSolver`, `MultipleShootingSolver`).

```@example approx
import Lux
chain = Lux.Chain(Lux.Dense(1, 16, tanh), Lux.Dense(16, 1))
approx_nn = NeuralApproximator(:f, chain)
```

The network weights are fitted as part of the optimization. Network architecture is specified as a standard Lux `Chain`.

```@docs
NeuralApproximator
```

## GPApproximator

Gaussian process approximator. Uses a GP prior over the unknown function with learnable hyperparameters (lengthscale, signal variance). Particularly useful with gradient matching solvers.

```@example approx
approx_gp = GPApproximator(:f, (0.0, 10.0), 20)
```

```@docs
GPApproximator
```

## COMONetApproximator

Constrained Monotone Network — a neural network architecture that guarantees monotonicity by construction using `exp(W)` weight parameterization. Supports a subset of shape constraints.

```@example approx
approx_comon = COMONetApproximator(:f, (0.0, 10.0), (16, 16), :increasing)
```

### Available constraints

| Constraint | Description |
|------------|-------------|
| `:increasing` | Monotonically increasing |
| `:decreasing` | Monotonically decreasing |
| `:convex` | Convex |
| `:concave` | Concave |
| `:inc_convex` | Increasing and convex |
| `:inc_concave` | Increasing and concave |
| `:dec_convex` | Decreasing and convex |
| `:dec_concave` | Decreasing and concave |
| `:positive` | Non-negative output |

```@docs
COMONetApproximator
```

## Choosing an Approximator

| Criterion | BSpline | ShapeConstrained | SPDE | Neural | GP | COMONet |
|-----------|:-------:|:----------------:|:----:|:------:|:--:|:-------:|
| Few parameters | ✓ | ✓ | ✓ | | | |
| Automatic smoothing | ✓ | ✓ | ✓ | | | |
| Shape constraints | | ✓ | ✓ | | | ✓ |
| Interpretable range | | | ✓ | | | |
| High flexibility | | | | ✓ | ✓ | |
| Gradient matching | ✓ | ✓ | ✓ | | ✓ | |
| UDE-style training | | | | ✓ | | ✓ |
| Interpretability | ✓ | ✓ | ✓ | | | |

**General guidance:**
- Start with [`BSplineApproximator`](@ref) — it works well in most cases with automatic smoothing.
- Use [`SPDEApproximator`](@ref) for an interpretable correlation-length parameter and Matérn-based smoothing.
- Use [`ShapeConstrainedBSplineApproximator`](@ref) or [`ShapeConstrainedSPDEApproximator`](@ref) when you have prior knowledge about monotonicity or convexity.
- Use [`NeuralApproximator`](@ref) when the unknown function may be complex or when using UDE-style solvers.
- Use [`GPApproximator`](@ref) with gradient matching or MAGI solvers.
- Use [`COMONetApproximator`](@ref) for neural network flexibility with monotonicity guarantees.
