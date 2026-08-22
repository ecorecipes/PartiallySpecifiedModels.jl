# ─── Approximator extension interface ─────────────────────────────
#
# `build_evaluator` is the single dispatch point through which every solver
# turns an approximator plus its coefficient block into a callable unknown
# function. Together with `nparams`, `initial_params`, and `penalty_matrix`
# it forms the complete interface a custom approximator type must implement;
# see docs/src/extending.md ("Custom approximators") for the contracts and a
# worked example.

"""
    build_evaluator(approx, params_k) -> callable

Construct the callable evaluator for an approximator's unknown function from
its parameter block `params_k`. Solvers call this to build the object that
the dynamics function receives as `p.<name>`, so the returned value must be
callable on a scalar, `f = build_evaluator(a, θ); f(x)::Number` (or on as
many scalar arguments as the dynamics pass it — `TensorBSplineApproximator`
returns a two-argument callable `f(x, y)`).

This is the extension point for user-defined approximator types. To add a
new approximator, define a struct subtyping `AbstractApproximator` (with at
least a `name::Symbol` field) and implement four methods:

- `nparams(a)` — number of free coefficients.
- `initial_params(a)` — starting coefficient vector, length `nparams(a)`.
- `penalty_matrix(a)` — roughness penalty matrix `S` (`nparams × nparams`;
  possibly a zero matrix) used in penalties `λ βᵀSβ`, or `nothing` for no
  quadratic penalty.
- `build_evaluator(a, params_k)` — this function.

`params_k` is the approximator's contiguous slice of the flat coefficient
vector maintained by the solver; it may be a `Vector{Float64}`, a view, or a
vector of `ForwardDiff.Dual` numbers, so the evaluator must be eltype-generic
(no hard-coded `Float64`) to work with the autodiff-based solvers
(`AdamSolver`, `MultipleShootingSolver`, `TwoStageSolver`, `BNGSolver`,
`IntegralMatchingSolver`, …).

Methods are provided for the eight built-in types:

| Approximator | Evaluator |
|:---|:---|
| `BSplineApproximator` | `build_bspline_evaluator` on a uniform knot grid over `domain` |
| `TensorBSplineApproximator` | `build_tensor_bspline_evaluator` (bivariate tensor product of the univariate construction; the callable takes TWO arguments, `f(x, y)`) |
| `NeuralApproximator` | `build_neural_evaluator` (Dual-safe MLP path + Lux fallback) |
| `GPApproximator` | `build_gp_evaluator` (kernel interpolation) |
| `ShapeConstrainedBSplineApproximator` | `build_constrained_bspline_evaluator` (SCOP reparameterization) |
| `COMONetApproximator` | `build_comonet_evaluator` (constrained monotone net) |
| `SPDEApproximator` | `build_spde_evaluator` on `mesh_points` |
| `ShapeConstrainedSPDEApproximator` | `build_constrained_spde_evaluator` |

For any other type the fallback method throws an error describing the
required interface.
"""
function build_evaluator(approx, params_k)
    error("build_evaluator: no method for approximator type " *
          "$(typeof(approx)). To use a custom approximator with " *
          "PartiallySpecifiedModels, implement the four interface functions " *
          "`nparams`, `initial_params`, `penalty_matrix`, and " *
          "`build_evaluator` for it — see the \"Custom approximators\" " *
          "page of the documentation (docs/src/extending.md).")
end

function build_evaluator(approx::BSplineApproximator, params_k)
    knots_x = collect(range(approx.domain[1], approx.domain[2],
                            length=approx.nknots))
    build_bspline_evaluator(knots_x, params_k)
end

# Two-argument callable f(x, y) — tensor product of the univariate
# construction; Dual-safe in params and in (x, y).
build_evaluator(approx::TensorBSplineApproximator, params_k) =
    build_tensor_bspline_evaluator(approx, params_k)

# Dual-safe, eltype-generic (see neural_evaluator.jl) — required for
# autodiff Jacobians in stiff ODE solvers and for gradients of any
# objective w.r.t. β.
build_evaluator(approx::NeuralApproximator, params_k) =
    build_neural_evaluator(approx, params_k)

build_evaluator(approx::GPApproximator, params_k) =
    build_gp_evaluator(approx, params_k)

build_evaluator(approx::ShapeConstrainedBSplineApproximator, params_k) =
    build_constrained_bspline_evaluator(approx, params_k)

build_evaluator(approx::COMONetApproximator, params_k) =
    build_comonet_evaluator(approx, params_k)

build_evaluator(approx::SPDEApproximator, params_k) =
    build_spde_evaluator(approx.mesh_points, params_k)

build_evaluator(approx::ShapeConstrainedSPDEApproximator, params_k) =
    build_constrained_spde_evaluator(approx, params_k)
