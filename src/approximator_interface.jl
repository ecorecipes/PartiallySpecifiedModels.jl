# ─── Approximator extension interface ─────────────────────────────
#
# `build_evaluator` is the single dispatch point through which every solver
# turns an approximator plus its coefficient block into a callable unknown
# function. Together with `nparams`, `initial_params`, and `penalty_matrix`
# it forms the complete interface a custom approximator type must implement
# (plus the OPTIONAL fifth function `penalty_blocks`, below, for types that
# carry several independently-smoothed penalty blocks); see
# docs/src/extending.md ("Custom approximators") for the contracts and a
# worked example.

# The ten built-in approximator types. Six solvers (TwoStage,
# IntegralMatching, MAGI, AGM, Rodeo, Dalton) assemble their smoothing
# penalties through per-type whitelists whose built-in treatment is
# historical and deliberately NOT unified (the lists differ — e.g.
# TwoStage never penalized COMONet — and AGM/Rodeo/Dalton build their
# BSpline penalty on DOMAIN knots where `penalty_matrix` uses unit-interval
# knots, a different scale; changing either would change existing fits).
# Each of those sites uses this Union to give any NON-built-in type the
# generic `penalty_matrix(approx)` treatment instead of silently leaving
# it unpenalized.
const _BUILTIN_APPROX_TYPES = Union{
    BSplineApproximator, ShapeConstrainedBSplineApproximator,
    SPDEApproximator, ShapeConstrainedSPDEApproximator,
    GPApproximator, ShapeConstrainedGPApproximator,
    NeuralApproximator, COMONetApproximator,
    TensorBSplineApproximator, SingleIndexApproximator}

"""
    build_evaluator(approx, params_k) -> callable

Construct the callable evaluator for an approximator's unknown function from
its parameter block `params_k`. Solvers call this to build the object that
the dynamics function receives as `p.<name>`, so the returned value must be
callable on a scalar, `f = build_evaluator(a, θ); f(x)::Number` (or on as
many scalar arguments as the dynamics pass it — `TensorBSplineApproximator`
returns a two-argument callable `f(x, y)`, and `SingleIndexApproximator` a
`p`-argument callable `f(u₁, …, u_p)`).

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

Methods are provided for the ten built-in types:

| Approximator | Evaluator |
|:---|:---|
| `BSplineApproximator` | `build_bspline_evaluator` on a uniform knot grid over `domain` |
| `TensorBSplineApproximator` | `build_tensor_bspline_evaluator` (bivariate tensor product of the univariate construction; the callable takes TWO arguments, `f(x, y)`) |
| `SingleIndexApproximator` | `build_single_index_evaluator` (learned direction composed with a univariate outer smooth; the callable takes `p` arguments, `f(u₁, …, u_p)`) |
| `NeuralApproximator` | `build_neural_evaluator` (Dual-safe MLP path + Lux fallback) |
| `GPApproximator` | `build_gp_evaluator` (kernel interpolation) |
| `ShapeConstrainedGPApproximator` | `build_constrained_gp_evaluator` (SCOP reparameterization + kernel interpolation) |
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

# p-argument callable f(u₁, …, u_p) = s((aᵀu − aᵀμ̂)/√(aᵀΣ̂a)) — Dual-safe
# in params and in the states.
build_evaluator(approx::SingleIndexApproximator, params_k) =
    build_single_index_evaluator(approx, params_k)

# Dual-safe, eltype-generic (see neural_evaluator.jl) — required for
# autodiff Jacobians in stiff ODE solvers and for gradients of any
# objective w.r.t. β.
build_evaluator(approx::NeuralApproximator, params_k) =
    build_neural_evaluator(approx, params_k)

build_evaluator(approx::GPApproximator, params_k) =
    build_gp_evaluator(approx, params_k)

build_evaluator(approx::ShapeConstrainedGPApproximator, params_k) =
    build_constrained_gp_evaluator(approx, params_k)

build_evaluator(approx::ShapeConstrainedBSplineApproximator, params_k) =
    build_constrained_bspline_evaluator(approx, params_k)

build_evaluator(approx::COMONetApproximator, params_k) =
    build_comonet_evaluator(approx, params_k)

build_evaluator(approx::SPDEApproximator, params_k) =
    build_spde_evaluator(approx.mesh_points, params_k)

build_evaluator(approx::ShapeConstrainedSPDEApproximator, params_k) =
    build_constrained_spde_evaluator(approx, params_k)

"""
    penalty_blocks(approx) -> Vector{Tuple{Matrix{Float64}, UnitRange{Int}}}

Quadratic penalty blocks of an approximator as `(S, local_range)` pairs.
Under the penalized-likelihood solvers (`LAML`, `GCVSolver`,
`CollocationLAML`, `GradientMatching`, `ProfileLikelihoodSolver`) each
block receives its OWN smoothing parameter — the total penalty is
`Σ_k λ_k β[r_k]' S_k β[r_k]` with every `λ_k` estimated jointly by
LAML/GCV. This is the optional fifth function of the approximator
extension protocol; types with a single roughness penalty need only
`penalty_matrix` and get the default below.

Contract: ranges are LOCAL to the approximator's own coefficient block
(within `1:nparams(approx)`) and MUST be pairwise disjoint — the
per-block generalized determinant `log|S_λ|₊` in laml.jl is exact only
for non-overlapping blocks; `build_penalty_matrices` validates this and
throws an `ArgumentError` otherwise. Each `S` must be
`length(range) × length(range)`, symmetric positive semi-definite
(possibly a zero matrix).

Default: the single block `(penalty_matrix(approx), 1:nparams(approx))`,
or no blocks (`[]`) when `penalty_matrix` returns `nothing` OR
`nparams(approx) < 3`. The `np ≥ 3` inclusion gate was historically
applied inside `build_penalty_matrices`; it lives in this default method
so the default path is unchanged while custom multi-block types remain
free to declare small blocks (e.g. a 2-parameter ridge).

Types overriding `penalty_blocks` should usually also provide a
CONSISTENT `penalty_matrix` — the block-diagonal merge of the blocks
with fixed weights — because the single-λ consumers (the
gradient-matching/probabilistic-numerics per-type penalty sites and the
MCMC/VI/ABC prior builder) read `penalty_matrix` only and apply ONE
weight to the whole approximator.
"""
function penalty_blocks(approx)
    np = nparams(approx)
    S = penalty_matrix(approx)
    (S === nothing || np < 3) &&
        return Tuple{Matrix{Float64}, UnitRange{Int}}[]
    Tuple{Matrix{Float64}, UnitRange{Int}}[(S, 1:np)]
end
