# Approximators

PartiallySpecifiedModels.jl provides five approximator types for representing unknown functions in dynamical systems. Each approximator is a callable object that maps scalar inputs to scalar outputs and is fitted as part of the model.

## BSplineApproximator

Cubic B-spline basis with automatic smoothing via penalized least squares. The default and most commonly used approximator.

```julia
approx = BSplineApproximator(:name, (lower, upper), nknots;
                              initial = x -> 0.0)
```

- **`:name`**: Symbol used to access the function in the parameter struct (`p.name(x)`)
- **`(lower, upper)`**: Domain of the input variable
- **`nknots`**: Number of evenly-spaced knots (controls flexibility)
- **`initial`**: Function or constant for initial coefficient values

The smoothing parameter ``\lambda`` is estimated automatically by LAML or GCV.

```@docs
BSplineApproximator
```

## ShapeConstrainedBSplineApproximator

SCOP-spline (Shape-Constrained P-spline) that enforces shape constraints on the fitted function via reparameterization of the B-spline coefficients. Supports 14 constraint types.

```julia
approx = ShapeConstrainedBSplineApproximator(:name, (lower, upper), nknots, :increasing)
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

## NeuralApproximator

Neural network approximator using [Lux.jl](https://github.com/LuxDL/Lux.jl). Suitable for complex or high-dimensional unknown functions. Compatible with gradient-based solvers (`AdamSolver`, `MultipleShootingSolver`).

```julia
using Lux
chain = Chain(Dense(1, 16, tanh), Dense(16, 1))
approx = NeuralApproximator(:name, chain)
```

The network weights are fitted as part of the optimization. Network architecture is specified as a standard Lux `Chain`.

```@docs
NeuralApproximator
```

## GPApproximator

Gaussian process approximator. Uses a GP prior over the unknown function with learnable hyperparameters (lengthscale, signal variance). Particularly useful with gradient matching solvers.

```julia
approx = GPApproximator(:name, (lower, upper))
```

```@docs
GPApproximator
```

## COMONetApproximator

Constrained Monotone Network — a neural network architecture that guarantees monotonicity by construction using `exp(W)` weight parameterization. Supports a subset of shape constraints.

```julia
approx = COMONetApproximator(:name, (lower, upper);
                              constraint = :increasing,
                              hidden_dims = [16, 16])
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

| Criterion | BSpline | ShapeConstrained | Neural | GP | COMONet |
|-----------|:-------:|:----------------:|:------:|:--:|:-------:|
| Few parameters | ✓ | ✓ | | | |
| Automatic smoothing | ✓ | ✓ | | | |
| Shape constraints | | ✓ | | | ✓ |
| High flexibility | | | ✓ | ✓ | |
| Gradient matching | ✓ | ✓ | | ✓ | |
| UDE-style training | | | ✓ | | ✓ |
| Interpretability | ✓ | ✓ | | | |

**General guidance:**
- Start with [`BSplineApproximator`](@ref) — it works well in most cases with automatic smoothing.
- Use [`ShapeConstrainedBSplineApproximator`](@ref) when you have prior knowledge about monotonicity or convexity.
- Use [`NeuralApproximator`](@ref) when the unknown function may be complex or when using UDE-style solvers.
- Use [`GPApproximator`](@ref) with gradient matching or MAGI solvers.
- Use [`COMONetApproximator`](@ref) for neural network flexibility with monotonicity guarantees.
