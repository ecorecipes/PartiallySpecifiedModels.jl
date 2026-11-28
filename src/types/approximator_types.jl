# Core type definitions for PartiallySpecifiedModels.jl

# ─── Approximator types ────────────────────────────────────────────

"""
Abstract base type for unknown function approximators.
Subtypes must implement:
- `nparams(a)`: number of parameters
- `build_evaluator(a, params_k)`: construct a callable `x -> value` given coefficient slice `params_k`
- `penalty_matrix(a)`: smoothing penalty matrix (or nothing for unpenalized)
- `initial_params(a)`: default initial parameter vector
"""
abstract type AbstractApproximator end

# Shared domain sanity check: every approximator needs an oriented,
# non-degenerate domain. An inverted domain silently produced reversed
# knot/mesh grids downstream; a degenerate one (lo == hi) produced NaNs
# (e.g. COMONet divides by the domain width when normalizing inputs).
function _validate_domain(ctor::AbstractString, domain::Tuple{Float64, Float64})
    (isfinite(domain[1]) && isfinite(domain[2]) && domain[2] > domain[1]) || throw(ArgumentError(
        "$ctor: domain must be finite and satisfy hi > lo, got ($(domain[1]), $(domain[2]))"))
    nothing
end

"""
    BSplineApproximator(name, domain, nknots; initial=nothing)

Cubic B-spline approximator with automatic smoothing penalty.
Knots are evenly spaced over `domain`. The penalty matrix
penalizes the integrated squared second derivative.
"""
struct BSplineApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    nknots::Int
    initial_func::Function
end

function BSplineApproximator(name::Union{Symbol,String},
                             domain::Tuple{Real,Real},
                             nknots::Int;
                             initial=nothing)
    name = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("BSplineApproximator", d)
    # CubicSpline interpolation needs at least 3 knots; fewer used to fail
    # much later with a raw BoundsError from inside DataInterpolations.
    nknots >= 3 || throw(ArgumentError(
        "BSplineApproximator needs nknots ≥ 3 for cubic spline " *
        "interpolation (got $nknots)"))
    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end
    BSplineApproximator(name, d, nknots, init_func)
end

"""
    CyclicBSplineApproximator(name, domain, nknots; initial=nothing)

Periodic (C²-continuous) cubic spline approximator on `domain = (lo, hi)`
with period `P = hi - lo` and `nknots ≥ 4` distinct periodic knot values
`β[k] = f(lo + (k - 1) * P / nknots)`.

Enforces exact periodicity `f(lo) == f(hi)`, `f'(lo) == f'(hi)`, `f''(lo) == f''(hi)`
and wraps any input `x ∈ ℝ` modulo `P` into `[lo, hi)`, making it ideal for
seasonal biological forcing (transmission, birth pulses, temperature-driven
rates) without boundary-jump artifacts.

Its roughness penalty `S_cyc = ∫₀¹ (f''(u))² du` is circulant PSD with rank
`nknots - 1` and a 1D constant null space (`λ → ∞` shrinks smoothly to a
constant function).
"""
struct CyclicBSplineApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    nknots::Int
    initial_func::Function
end

function CyclicBSplineApproximator(name::Union{Symbol,String},
                                   domain::Tuple{Real,Real},
                                   nknots::Int;
                                   initial=nothing)
    name = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("CyclicBSplineApproximator", d)
    nknots >= 4 || throw(ArgumentError(
        "CyclicBSplineApproximator needs nknots ≥ 4 for periodic cubic spline " *
        "interpolation (got $nknots)"))
    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end
    CyclicBSplineApproximator(name, d, nknots, init_func)
end

nparams(a::CyclicBSplineApproximator) = a.nknots

function initial_params(a::CyclicBSplineApproximator)
    lo, hi = a.domain
    h = (hi - lo) / a.nknots
    Float64[a.initial_func(lo + (k - 1) * h) for k in 1:a.nknots]
end

"""
    RandomEffectApproximator(name, n_groups; initial=0.0)

Group-level Gaussian random effect approximator (`b_g ~ N(0, λ⁻¹)`) for
hierarchical / multi-trajectory PSMs, matching `mgcv`'s `s(group, bs="re")`.

Callable as `p.<name>(g)` for integer group index `g ∈ 1:n_groups`. Its
penalty matrix is the identity `I_{n_groups}` (full-rank ridge penalty), so
LAML automatically estimates the random-effect precision `λ = σ² / σ_b²`.
"""
struct RandomEffectApproximator <: AbstractApproximator
    name::Symbol
    n_groups::Int
    initial_val::Float64
end

function RandomEffectApproximator(name::Union{Symbol,String}, n_groups::Int;
                                  initial::Real=0.0)
    n_groups >= 1 || throw(ArgumentError(
        "RandomEffectApproximator: n_groups must be ≥ 1 (got $n_groups)"))
    RandomEffectApproximator(Symbol(name), n_groups, Float64(initial))
end

nparams(a::RandomEffectApproximator) = a.n_groups
initial_params(a::RandomEffectApproximator) = fill(a.initial_val, a.n_groups)

"""
    TensorBSplineApproximator(name, domain_x, domain_y, nknots_x, nknots_y;
                              anisotropy=1.0, initial=nothing)

Bivariate tensor-product cubic-spline approximator for an unknown function
of TWO state variables — e.g. the predation response `g(P, N)` of Wood
(2001), "Partially specified ecological models", whose form depends on both
predator and prey densities. In the dynamics the fitted surface is a
two-argument callable, `du[1] = ... p.g(u[1], u[2]) ...`.

Coefficients are function values on the `nknots_x × nknots_y` grid of
evenly spaced knots over `domain_x × domain_y` when `anisotropy` is a positive
real scalar (column-major `vec(C)` with single Kronecker-sum penalty
`S = I_ny ⊗ S_x + anisotropy · (S_y ⊗ I_nx)`).

Passing `anisotropy = :estimate` activates Wood (2006)'s scale-invariant `t2`
orthogonal decomposition into three pairwise-disjoint penalized blocks
(`[λ_x, λ_y, λ_xy]`) plus the 4D bilinear null space (`a + bx + cy + dxy`),
letting LAML estimate independent smoothing parameters per axis and interaction.

# Arguments
- `name`: symbol for the unknown function (accessed as `p.name(x, y)`)
- `domain_x`, `domain_y`: `(lo, hi)` ranges of the first and second argument
- `nknots_x`, `nknots_y`: knots per margin (≥ 3 each; `nparams = nknots_x · nknots_y`)
- `anisotropy`: positive relative weight of the y-roughness penalty (default `1.0`),
  or `:estimate` for Wood (2006) multi-λ scale-invariant `t2` smoothing
- `initial`: optional initial surface — a two-argument function `(x, y) -> z`
  or a constant (default 0)

# Example
```julia
uf = TensorBSplineApproximator(:g, (0.0, 4.0), (0.0, 6.0), 6, 6; anisotropy=:estimate)
predprey!(du, u, p, t) = begin
    du[1] = p.r * u[1] - p.g(u[1], u[2])
    du[2] = p.e * p.g(u[1], u[2]) - p.m * u[2]
end
```

For pointwise function intervals, supply physical query coordinates to
`confidence_band` or `bootstrap` with `uf_points=Dict(:g => points)`,
where `points` has one sample per row and two columns. There is no
automatic univariate grid for a tensor surface.
"""
struct TensorBSplineApproximator <: AbstractApproximator
    name::Symbol
    domain_x::Tuple{Float64, Float64}
    domain_y::Tuple{Float64, Float64}
    nknots_x::Int
    nknots_y::Int
    anisotropy::Union{Float64, Symbol}
    initial_func::Function
end

function TensorBSplineApproximator(name::Union{Symbol,String},
                                   domain_x::Tuple{Real,Real},
                                   domain_y::Tuple{Real,Real},
                                   nknots_x::Int,
                                   nknots_y::Int;
                                   anisotropy::Union{Real,Symbol}=1.0,
                                   initial=nothing)
    name = Symbol(name)
    dx = (Float64(domain_x[1]), Float64(domain_x[2]))
    dy = (Float64(domain_y[1]), Float64(domain_y[2]))
    _validate_domain("TensorBSplineApproximator (x margin)", dx)
    _validate_domain("TensorBSplineApproximator (y margin)", dy)
    nknots_x >= 3 || throw(ArgumentError(
        "TensorBSplineApproximator needs nknots_x ≥ 3 for cubic spline " *
        "interpolation (got $nknots_x)"))
    nknots_y >= 3 || throw(ArgumentError(
        "TensorBSplineApproximator needs nknots_y ≥ 3 for cubic spline " *
        "interpolation (got $nknots_y)"))
    aniso = if anisotropy isa Symbol
        anisotropy === :estimate || throw(ArgumentError(
            "TensorBSplineApproximator: symbol anisotropy must be :estimate, got :$anisotropy"))
        :estimate
    else
        v = Float64(anisotropy)
        (isfinite(v) && v > 0.0) || throw(ArgumentError(
            "TensorBSplineApproximator: anisotropy must be positive and finite (or :estimate), " *
            "got $anisotropy"))
        v
    end
    init_func = if initial === nothing
        (x, y) -> 0.0
    elseif initial isa Function
        initial
    else
        (x, y) -> Float64(initial)
    end
    TensorBSplineApproximator(name, dx, dy, nknots_x, nknots_y, aniso,
                              init_func)
end

nparams(a::TensorBSplineApproximator) = a.nknots_x * a.nknots_y

function initial_params(a::TensorBSplineApproximator)
    xs = range(a.domain_x[1], a.domain_x[2], length=a.nknots_x)
    ys = range(a.domain_y[1], a.domain_y[2], length=a.nknots_y)
    # Column-major vec of the (nknots_x × nknots_y) value grid — x index
    # fastest, matching build_tensor_bspline_evaluator's reshape.
    c0 = vec(Float64[a.initial_func(x, y) for x in xs, y in ys])
    if a.anisotropy === :estimate
        Q, _ = _tensor_t2_decomp(a.nknots_x, a.nknots_y)
        return Q' * c0
    end
    c0
end

"""
    NeuralApproximator(name, model; penalty_weight=0.0, domain=nothing)

Neural network approximator using a Lux.jl model.

# Arguments
- `name`: symbol for the unknown function
- `model`: a Lux.jl model (Chain, Dense, etc.)
- `penalty_weight`: L2 regularization weight (>0 enables LAML smoothing)
- `domain`: optional `(lo, hi)` for input normalization to `[0, 1]`

# Evaluation and autodiff
For a `Lux.Chain` of `Lux.Dense` layers (or a bare `Lux.Dense`), the
network is evaluated through a hand-rolled, eltype-generic MLP path
(`build_neural_evaluator`) that is fully compatible with
`ForwardDiff.Dual` numbers — stiff ODE solvers with autodiff Jacobians
(e.g. `TRBDF2()`, `Rosenbrock23()`) and gradient-based solvers work.
Other architectures fall back to `Lux.apply` with the input and
parameters promoted to a common element type; that fallback is only as
Dual-safe as the Lux kernels of the layers involved, so exotic layers
may not support ForwardDiff through the dynamics.
"""
struct NeuralApproximator <: AbstractApproximator
    name::Symbol
    model::Any  # Lux.AbstractLuxLayer
    penalty_weight::Float64
    domain::Union{Nothing, Tuple{Float64, Float64}}
    rng_seed::Union{Nothing, Int}
end

function NeuralApproximator(name::Union{Symbol,String}, model;
                            penalty_weight::Float64=0.0,
                            domain::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
                            rng_seed::Union{Nothing, Int}=nothing)
    (isfinite(penalty_weight) && penalty_weight >= 0.0) || throw(ArgumentError(
        "NeuralApproximator: penalty_weight must be finite and ≥ 0, got $penalty_weight"))
    d = domain === nothing ? nothing : (Float64(domain[1]), Float64(domain[2]))
    d === nothing || _validate_domain("NeuralApproximator", d)
    NeuralApproximator(Symbol(name), model, penalty_weight, d, rng_seed)
end

"""
    KANApproximator(name, model; input_domains=nothing, penalty=:none,
                   nullspace_penalty=0.0, rng_seed=42)

Fixed-grid Kolmogorov-Arnold network for a scalar unknown response, called
as `p.f(x)` or `p.g(x, y, ...)` inside the dynamics.

Load `FluxKAN` to enable its optional backend. Supported models are a
`FluxKAN.LuxKANLinear` or a flat `Lux.Chain` of those layers, with cubic
splines, `standalone_spline_scale=false`, and one final output. Parameters
and grids are initialized once and stored as Float64 snapshots. Evaluation
uses the same edge functions and coefficient layout through a PSM-native
scalar spline path compatible with differentiation through the RHS.

`input_domains` is a tuple/vector containing one finite `(lo, hi)` pair per
input. It maps each physical input affinely onto the first layer's logical
grid range. `nothing` leaves inputs in the backend's raw coordinates.
Inputs are not clamped: outside spline support, the residual/base branch
still applies. Grids never adapt during a solve.

`penalty=:none` returns no quadratic penalty; `:ridge` returns an identity
matrix in raw coefficient coordinates. `:edge_curvature` integrates the
squared second derivative of each COMPLETE edge function (base activation
plus spline, including cross terms), in its layer's normalized logical
coordinate. It supplies one disjoint smoothing block per layer.
Curvature currently supports the default SiLU, `tanh`, and `identity`
base activations. It regularizes the edge representation, not the
curvature of the entire composed response.

For `:edge_curvature`, `nullspace_penalty >= 0` optionally adds a declared
quadratic penalty on each edge's affine coefficient null space. The default
is zero. The solver supplies or estimates the overall block weights:
`AdamSolver(penalty_weight=...)` uses a fixed weight, whereas LAML/GCV
select one per layer. Grids remain fixed; adaptive grids are not implemented.

`initial_params` returns a copy of the stored initialization, even when
`rng_seed=nothing`. `confidence_band` supports pointwise covariance
projection through the fixed-grid parameter gradient; a fit must supply
`V_beta` and `sigma2` (for example, LAML). Both it and `bootstrap` accept
physical `uf_points`, with one query per row and one column per input.
A unary input domain also permits an automatic grid. These intervals
condition on the architecture and grids, not on their identification.
Bootstrap keeps the cached KAN initialization and reselects smoothing
only when its fitting algorithm does.

[`kan_edge_curves`](@ref) extracts base/spline/total edge curves.
[`kan_activation_diagnostics`](@ref) reports fixed-grid exposure on supplied
samples without adapting grids. Neither establishes unique edge
interpretability, joint-state support, or uncertainty coverage.

# References
- Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M.,
  Hou, T.Y. & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks.
  arXiv:2404.19756.
- Backend: FluxKAN.jl (`LuxKANLinear`), https://github.com/cometscome/FluxKAN.jl —
  loaded through the `PartiallySpecifiedModelsFluxKANExt` package extension.
"""
struct KANApproximator{M,S,A,L,D} <: AbstractApproximator
    name::Symbol
    model::M
    state::S
    param_axes::A
    layers::L
    input_domains::D
    logical_domain::Tuple{Float64,Float64}
    input_dim::Int
    initial_parameters::Vector{Float64}
    penalty::Symbol
    nullspace_penalty::Float64
    curvature_blocks::Vector{Tuple{Matrix{Float64},UnitRange{Int}}}
    rng_seed::Union{Nothing,Int}
end

"""
    nparams(approx::AbstractApproximator) -> Int

Return the number of free parameters for the given approximator.
"""
nparams(a::BSplineApproximator) = a.nknots
nparams(a::NeuralApproximator) = Lux.parameterlength(a.model)

"""
    initial_params(approx::AbstractApproximator) -> Vector{Float64}

Return a vector of initial parameter values for the given approximator.
"""
function initial_params(a::BSplineApproximator)
    xs = range(a.domain[1], a.domain[2], length=a.nknots)
    Float64[a.initial_func(x) for x in xs]
end

function initial_params(a::NeuralApproximator)
    rng = a.rng_seed !== nothing ? Random.Xoshiro(a.rng_seed) : Random.default_rng()
    ps, st = Lux.setup(rng, a.model)
    return Float64.(collect(ComponentArrays.ComponentArray(ps)))
end

# ─── GP Approximator ──────────────────────────────────────────────

"""
    GPApproximator(name, domain, n_inducing; kernel=:sqexp, lengthscale=nothing,
                   variance=1.0, initial=nothing)

Kernel-interpolation approximator. Parameters are function values at
uniformly spaced inducing points, evaluated between points through the
kernel weights `k(x,X)K⁻¹f_X` (the GP predictive-mean formula). Note this
is a basis expansion, not full GP inference: no predictive variance is
produced.

Kernel hyperparameters are handled two ways:
- `lengthscale=nothing` (default): adaptive. During LAML (after `warmup`
  iterations) and GCV fits, `(lengthscale, variance)` are re-estimated by
  empirical Bayes — maximizing the GP marginal likelihood of the current
  inducing values over a lengthscale × variance grid — and `K`/`K⁻¹` are
  rebuilt. Because the coefficients are function *values* at the inducing
  points, re-fitting the kernel changes only the between-point behavior of
  the interpolant, not the values it passes through.
- `lengthscale=<value>`: fixed. The supplied hyperparameters are used
  as-is and never touched by any solver.

The LAML/GCV penalty is a spline-style second-derivative penalty on the
inducing values — not the theoretical GP prior precision `K⁻¹`, whose
narrow eigenvalue spectrum makes smoothing-parameter estimation
unreliable (see `penalty_matrix(::GPApproximator)` in approximators.jl).

# Kernels
- `:sqexp`  — Squared exponential: `k(r) = σ² exp(-r²/(2ℓ²))`
- `:matern32` — Matérn 3/2: `k(r) = σ²(1 + √3r/ℓ) exp(-√3r/ℓ)`
- `:matern52` — Matérn 5/2: `k(r) = σ²(1 + √5r/ℓ + 5r²/(3ℓ²)) exp(-√5r/ℓ)`

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `n_inducing`: number of inducing points (like nknots for splines)
- `kernel`: kernel type (default `:sqexp`)
- `lengthscale`: kernel lengthscale. `nothing` (default) starts at
  `domain_span / (n_inducing - 1)` and adapts during the fit; an explicit
  value is fixed for the whole fit
- `variance`: signal variance σ² (default: 1.0)
- `initial`: optional initial function `x -> y`
"""
mutable struct GPApproximator <: AbstractApproximator
    # Mutable so the fit loop can adapt (lengthscale, variance) — and the
    # derived K/K_inv — as the fitted function evolves (empirical Bayes on
    # the inducing values). `adapt` is set when the user did NOT supply a
    # lengthscale explicitly; solvers only mutate when it is true.
    # Bootstrap replicates deepcopy the approximators to avoid races.
    const name::Symbol
    const domain::Tuple{Float64, Float64}
    const n_inducing::Int
    const inducing_points::Vector{Float64}
    const kernel::Symbol
    lengthscale::Float64
    variance::Float64
    const initial_func::Function
    K::Matrix{Float64}        # kernel matrix at inducing points
    K_inv::Matrix{Float64}    # inverse kernel matrix
    const adapt::Bool
end

"""
    _gp_default_lengthscale(kernel, d, n_inducing) -> Float64

Default lengthscale: normalize so adjacent inducing points have similar
correlation (~0.6) regardless of kernel type. SqExp at h gives
exp(-0.5) ≈ 0.607; Matérn kernels need longer ℓ to achieve the same
correlation due to their heavier tails.
"""
function _gp_default_lengthscale(kernel::Symbol, d::Tuple{Float64, Float64},
                                 n_inducing::Int)
    h = Float64((d[2] - d[1]) / max(n_inducing - 1, 1))
    if kernel == :matern32
        2.0 * h     # (1+√3·h/(2h))·exp(-√3·h/(2h)) ≈ 0.60
    elseif kernel == :matern52
        1.9 * h     # (1+√5·h/(1.9h)+5/(3·1.9²))·exp(-√5·h/(1.9h)) ≈ 0.61
    else
        h           # SqExp: exp(-0.5) ≈ 0.607
    end
end

"""
    _gp_kernel_matrices(kernel, ℓ, σ², x_ind) -> (K, K_inv)

Build the inducing-point kernel matrix and its (jittered, symmetrized)
inverse. Factor with scaled jitter: squared-exponential Gram matrices are
notoriously ill-conditioned as n_inducing grows; a fixed 1e-8 jitter plus
explicit inv() produced large oscillatory weights K⁻¹f between inducing
points. Scale the jitter to the kernel magnitude and escalate until the
Cholesky succeeds.
"""
function _gp_kernel_matrices(kernel::Symbol, ℓ::Float64, σ²::Float64,
                             x_ind::Vector{Float64})
    kfunc = _kernel_func(kernel, ℓ, σ²)
    K = _build_kernel_matrix(kfunc, x_ind)
    scale = max(maximum(abs, K), 1.0)
    K_inv = nothing
    for jit in (1e-8, 1e-6, 1e-4)
        F = cholesky(Symmetric(K + jit * scale * I), check=false)
        if issuccess(F)
            K_inv = Matrix(inv(F))
            break
        end
    end
    K_inv === nothing && (K_inv = pinv(K + 1e-4 * scale * I))
    K_inv = 0.5 * (K_inv + K_inv')  # symmetrize
    K, K_inv
end

function GPApproximator(name::Union{Symbol,String},
                        domain::Tuple{<:Real, <:Real},
                        n_inducing::Int;
                        kernel::Symbol=:sqexp,
                        lengthscale::Union{Nothing, Real}=nothing,
                        variance::Real=1.0,
                        initial=nothing)
    name_s = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("GPApproximator", d)
    σ² = Float64(variance)
    ℓ = lengthscale === nothing ? _gp_default_lengthscale(kernel, d, n_inducing) :
        Float64(lengthscale)

    # Inducing points uniformly spaced in domain
    n_inducing >= 2 ||
        throw(ArgumentError("GPApproximator needs n_inducing ≥ 2 (got $n_inducing)"))
    x_ind = collect(range(d[1], d[2], length=n_inducing))

    # Kernel matrix and jittered inverse
    K, K_inv = _gp_kernel_matrices(kernel, ℓ, σ², x_ind)

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end

    GPApproximator(name_s, d, n_inducing, x_ind, kernel, ℓ, σ², init_func,
                   K, K_inv, lengthscale === nothing)
end

nparams(a::GPApproximator) = a.n_inducing

function initial_params(a::GPApproximator)
    Float64[a.initial_func(x) for x in a.inducing_points]
end

# ─── SPDE (Matérn) approximator ───────────────────────────────────

"""
Matérn SPDE approximator using finite element basis functions.

Represents an unknown function as a linear combination of piecewise linear
hat functions on a 1D mesh, with a Matérn SPDE penalty derived from the
stochastic PDE `(κ² - Δ)^(α/2) τu = W` (Lindgren et al. 2011).

The penalty matrix is `κ⁴C + 2κ²G + G₂` where C is the FEM mass matrix,
G is the stiffness matrix, and G₂ = G C⁻¹ G approximates the biharmonic
operator. The overall smoothing parameter τ² is estimated via LAML/GCV.

The correlation range `ρ = √(8ν)/κ` controls the effective smoothing scale,
providing a physically interpretable alternative to the abstract smoothing
parameter of B-spline penalties.

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `n_basis`: number of mesh nodes (basis functions)
- `nu`: Matérn smoothness parameter (default: 1.5, i.e. Matérn 3/2)
  - 0.5: rough (exponential covariance, penalty = κ²C + G)
  - 1.5: moderate smoothness (Matérn 3/2, penalty = κ⁴C + 2κ²G + G₂)
  - 2.5: smooth (Matérn 5/2, penalty uses G₃ — requires α=3)
- `range_param`: correlation range ρ (default: 1/3 of domain width).
  Larger values → smoother functions; κ = √(8ν)/ρ.
- `initial`: optional initial function `x -> y`

# References
- Lindgren, Rue & Lindström (2011). An explicit link between Gaussian fields
  and Gaussian Markov random fields. JRSS-B 73(4):423–498.
- Miller, Glennie & Seaton (2020). Understanding the SPDE approach to smoothing.
  JABES.
"""
struct SPDEApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    n_basis::Int
    nu::Float64
    kappa::Float64
    range_param::Float64
    mesh_points::Vector{Float64}
    initial_func::Function
end

"""Supported Matérn smoothness values for SPDE approximator."""
const SPDE_SMOOTHNESS = (0.5, 1.5, 2.5)

function SPDEApproximator(name::Union{Symbol,String},
                          domain::Tuple{<:Real, <:Real},
                          n_basis::Int;
                          nu::Real=1.5,
                          range_param::Union{Nothing, Real}=nothing,
                          initial=nothing)
    name_s = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("SPDEApproximator", d)
    ν = Float64(nu)
    ν ∈ SPDE_SMOOTHNESS || error("nu must be one of $SPDE_SMOOTHNESS, got $ν")
    n_basis >= 3 || error("n_basis must be ≥ 3, got $n_basis")

    # Default range: 1/3 of domain width
    ρ = if range_param === nothing
        (d[2] - d[1]) / 3.0
    else
        Float64(range_param)
    end
    ρ > 0 || error("range_param must be positive, got $ρ")
    κ = sqrt(8.0 * ν) / ρ

    mesh = collect(range(d[1], d[2], length=n_basis))

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end

    SPDEApproximator(name_s, d, n_basis, ν, κ, ρ, mesh, init_func)
end

nparams(a::SPDEApproximator) = a.n_basis

function initial_params(a::SPDEApproximator)
    Float64[a.initial_func(x) for x in a.mesh_points]
end

# ─── Shape-constrained SPDE approximator ──────────────────────────

"""
    ShapeConstrainedSPDEApproximator(name, domain, n_basis, constraint;
                                     nu=1.5, range_param=nothing, initial=nothing,
                                     penalty=:gamma_matern)

SPDE (Matérn) approximator with a shape constraint enforced via the SCOP-spline
reparameterization of Pya & Wood (2015).

Combines the Matérn SPDE penalty (interpretable range and smoothness parameters)
with shape constraints (monotonicity, positivity, convexity, etc.). Parameters
are stored in unconstrained space (γ). During evaluation, mesh node values are
computed as `β = Σ * d(γ)` (free level/slope components pass through linearly; the rest are softplus'd) where Σ is a constraint matrix, then
interpolated with a cubic spline.

**Note — constraints hold at the mesh nodes, NOT between them.** The cubic
spline interpolant's cardinal functions take negative values (measured
minimum −0.17), so the constraint satisfied by the node values does not
transfer to the interpolant: on a `:positive` fixture whose 10 node values
were all positive (alternating ≈5 / ≈0.007) the evaluator dipped to −0.121.
Every one of the 14 constraints is affected
(`ShapeConstrainedGPApproximator` shares the problem at larger magnitude;
`ShapeConstrainedBSplineApproximator` is exact everywhere by the B-spline
convex-hull property). Adding mesh nodes does NOT cure it — the worst-case
dip per unit node value grows slightly with `n_basis` (measured
0.24 → 0.27 for n = 6 → 40); more nodes help only indirectly, by letting
the fitted values vary more smoothly relative to the spacing. Audit any
fitted result with [`check_constraints`](@ref); use
`ShapeConstrainedBSplineApproximator` when the constraint must hold
exactly.

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `n_basis`: number of mesh nodes (≥ 4)
- `constraint`: one of `SHAPE_CONSTRAINTS`
- `nu`: Matérn smoothness parameter (0.5, 1.5, or 2.5)
- `range_param`: correlation length ρ (default: 1/3 of domain width)
- `initial`: optional initial function `x -> y` or constant
- `penalty`: `:gamma_matern` (default) applies the Matérn precision to the
  unconstrained γ as `Σᵀ P_β Σ`; `:difference` applies the Pya & Wood (2015)
  SCOP first-difference penalty on γ with the free level/slope in the null
  space. See the `penalty_matrix(::ShapeConstrainedSPDEApproximator)`
  docstring for the trade-offs — in particular, the default's λ→∞ limit is NOT a
  maximally smooth member of the constraint family.

# Example
```julia
# Monotone increasing functional response (Holling-type)
approx = ShapeConstrainedSPDEApproximator(:g, (0.0, 5.0), 10, :inc_concave;
    nu=1.5, initial=x -> 0.1*x)
```
"""
struct ShapeConstrainedSPDEApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    n_basis::Int
    nu::Float64
    kappa::Float64
    range_param::Float64
    mesh_points::Vector{Float64}
    constraint::Symbol
    Sigma::Matrix{Float64}
    initial_func::Function
    penalty::Symbol
end

function ShapeConstrainedSPDEApproximator(name::Union{Symbol,String},
                                          domain::Tuple{<:Real, <:Real},
                                          n_basis::Int,
                                          constraint::Symbol;
                                          nu::Real=1.5,
                                          range_param::Union{Nothing, Real}=nothing,
                                          initial=nothing,
                                          penalty::Symbol=:gamma_matern)
    name_s = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("ShapeConstrainedSPDEApproximator", d)
    ν = Float64(nu)
    ν ∈ SPDE_SMOOTHNESS || error("nu must be one of $SPDE_SMOOTHNESS, got $ν")
    n_basis >= 4 || error("n_basis must be ≥ 4 for shape-constrained SPDE, got $n_basis")
    constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $SHAPE_CONSTRAINTS"))
    penalty in (:gamma_matern, :difference) || throw(ArgumentError(
        "Unknown penalty :$penalty. Must be :gamma_matern or :difference"))

    ρ = if range_param === nothing
        (d[2] - d[1]) / 3.0
    else
        Float64(range_param)
    end
    ρ > 0 || error("range_param must be positive, got $ρ")
    κ = sqrt(8.0 * ν) / ρ

    mesh = collect(range(d[1], d[2], length=n_basis))
    Sig = _build_sigma_matrix(constraint, n_basis)

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end

    ShapeConstrainedSPDEApproximator(name_s, d, n_basis, ν, κ, ρ, mesh,
                                     constraint, Sig, init_func, penalty)
end

function nparams(a::ShapeConstrainedSPDEApproximator)
    a.constraint in _ZERO_ENDPOINT_CONSTRAINTS ? a.n_basis - 1 : a.n_basis
end

function initial_params(a::ShapeConstrainedSPDEApproximator)
    beta_target = Float64[a.initial_func(x) for x in a.mesh_points]
    d = a.Sigma \ beta_target
    lin = _linear_param_indices(a.constraint)
    # Free (linear) components pass through; nonnegative ones get a small
    # positive floor (large floors visibly distorted flat initial functions
    # into ramps) and are inverted through softplus.
    return [i in lin ? d[i] :
            (v = max(d[i], 1e-4); v > 20.0 ? v : log(exp(v) - 1.0))
            for i in eachindex(d)]
end

# ─── Shape-constrained B-spline approximator ──────────────────────

"""
Supported shape constraints for B-spline approximators.

**Basic monotonicity:**
- `:increasing`    — monotone increasing: f'(x) ≥ 0
- `:decreasing`    — monotone decreasing: f'(x) ≤ 0

**Curvature:**
- `:convex`        — convex: f''(x) ≥ 0
- `:concave`       — concave: f''(x) ≤ 0

**Combined monotonicity + curvature:**
- `:inc_convex`    — monotone increasing and convex
- `:inc_concave`   — monotone increasing and concave
- `:dec_convex`    — monotone decreasing and convex
- `:dec_concave`   — monotone decreasing and concave

**Positivity:**
- `:positive`      — f(x) ≥ 0 everywhere
- `:dec_positive`  — monotone decreasing and positive

**Zero at endpoint (nparams = nknots - 1):**
- `:inc_zero_left`  — increasing with f(x_min) = 0  (SCAM: miso)
- `:dec_zero_right` — decreasing with f(x_max) = 0  (SCAM: mifo-like)
- `:inc_zero_right` — increasing with f(x_max) = 0
- `:dec_zero_left`  — decreasing with f(x_min) = 0

Note: following Pya & Wood (2015), the monotone and combined
monotone/curvature constraints carry a *free* level parameter (β₁ = γ₁),
so e.g. an `:increasing` function may cross zero; use `:positive` or
`:dec_positive` when positivity itself is required.
"""
const SHAPE_CONSTRAINTS = (
    :increasing, :decreasing, :convex, :concave,
    :inc_convex, :inc_concave, :dec_convex, :dec_concave,
    :positive, :dec_positive,
    :inc_zero_left, :dec_zero_right, :inc_zero_right, :dec_zero_left,
)

# Zero-at-endpoint constraints have one fewer free parameter
const _ZERO_ENDPOINT_CONSTRAINTS = (
    :inc_zero_left, :dec_zero_right, :inc_zero_right, :dec_zero_left,
)

"""
    ShapeConstrainedBSplineApproximator(name, domain, nknots, constraint; initial=nothing)

B-spline approximator with a shape constraint enforced via the SCOP-spline
reparameterization of Pya & Wood (2015).

Parameters are stored in unconstrained space (γ). During evaluation, knot
values are computed as `β = Σ * d(γ)` (free level/slope components pass through linearly; the rest are softplus'd) where Σ is a constraint matrix
(cumulative sum for monotonicity, second-order cumsum for convexity, etc.).

For zero-at-endpoint constraints, one knot value is fixed at 0 and
`nparams = nknots - 1` (Σ is q × (q-1)).

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `nknots`: number of basis knots (≥ 4)
- `constraint`: one of `SHAPE_CONSTRAINTS`
- `initial`: optional initial function `x -> y` or constant

# Example
```julia
# Monotone decreasing transmission rate
approx = ShapeConstrainedBSplineApproximator(:β, (0.0, 0.15), 8, :decreasing;
    initial=0.4)

# Decreasing to zero at carrying capacity
approx = ShapeConstrainedBSplineApproximator(:r, (0.0, 1.0), 10, :dec_zero_right;
    initial=x -> 0.5*(1-x))
```

# Reference
Pya, N. & Wood, S.N. (2015). Shape constrained additive models.
Statistics and Computing, 25, 543–559.
"""
struct ShapeConstrainedBSplineApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    nknots::Int
    constraint::Symbol
    Sigma::Matrix{Float64}   # constraint reparameterization matrix (q × np)
    initial_func::Function
end

function ShapeConstrainedBSplineApproximator(name::Union{Symbol,String},
                                             domain::Tuple{<:Real, <:Real},
                                             nknots::Int,
                                             constraint::Symbol;
                                             initial=nothing)
    name = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("ShapeConstrainedBSplineApproximator", d)
    constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $SHAPE_CONSTRAINTS"))
    nknots >= 4 || throw(ArgumentError("Need nknots ≥ 4, got $nknots"))

    Sig = _build_sigma_matrix(constraint, nknots)

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end
    ShapeConstrainedBSplineApproximator(name, d, nknots, constraint, Sig, init_func)
end

function nparams(a::ShapeConstrainedBSplineApproximator)
    a.constraint in _ZERO_ENDPOINT_CONSTRAINTS ? a.nknots - 1 : a.nknots
end

function initial_params(a::ShapeConstrainedBSplineApproximator)
    # Sample the target at the GREVILLE abscissae ξᵢ = mean of the 3
    # interior knots supporting coefficient i (cubic B-splines): de Boor
    # coefficients approximate function values at ξᵢ, not at a uniform
    # domain grid — a uniform grid reproduced even a linear target with
    # O(slope·spacing) error.
    xk = _scam_knot_vector(a.domain, a.nknots)
    xs = [(xk[i+1] + xk[i+2] + xk[i+3]) / 3 for i in 1:a.nknots]
    # Evaluate at the UNCLAMPED Greville points: the boundary abscissae lie
    # slightly outside the domain, and clamping them there broke the exact
    # linear reproduction this initialization is built on (max error 0.067
    # on 1+2x with 8 knots). Only if the user's function throws outside its
    # domain do we fall back, per point, to the clamped value extrapolated
    # linearly with a one-sided finite-difference slope.
    lo, hi = a.domain
    h = 1e-6 * (hi - lo)
    beta_target = Vector{Float64}(undef, a.nknots)
    for (i, x) in enumerate(xs)
        beta_target[i] = if lo <= x <= hi
            Float64(a.initial_func(x))
        else
            try
                Float64(a.initial_func(x))
            catch e
                # Only an out-of-domain THROW justifies extrapolating. A
                # MethodError/BoundsError/TypeError inside the user's
                # function is a bug in that function; masking it as an
                # extrapolation would silently initialize the spline from
                # fabricated values. Rethrow those, and say out loud when
                # the fallback does fire.
                _is_program_error(e) && rethrow()
                @warn("initial_func threw at the Greville abscissa " *
                      "x=$x, just outside the domain $(a.domain); " *
                      "falling back to a linear extrapolation from the " *
                      "clamped point.", exception = e, maxlog = 1)
                xc = clamp(x, lo, hi)
                slope = x < lo ?
                    (a.initial_func(xc + h) - a.initial_func(xc)) / h :
                    (a.initial_func(xc) - a.initial_func(xc - h)) / h
                Float64(a.initial_func(xc) + slope * (x - xc))
            end
        end
    end
    # Solve Σ * d = β_target for the coefficient vector d.
    # For square Σ: direct solve; for rectangular (q × np): least-squares.
    d = a.Sigma \ beta_target
    lin = _linear_param_indices(a.constraint)
    # Linear (free) components pass through unchanged; nonnegative components
    # get a small positive floor (a large floor turned flat initial functions
    # into visible ramps) and are inverted through softplus.
    return [i in lin ? d[i] :
            (v = max(d[i], 1e-4); v > 20.0 ? v : log(exp(v) - 1.0))
            for i in eachindex(d)]
end

# ─── Shape-constrained GP approximator ────────────────────────────

"""
    ShapeConstrainedGPApproximator(name, domain, n_inducing, constraint;
                                   kernel=:sqexp, lengthscale=nothing,
                                   variance=1.0, initial=nothing)

Kernel-interpolation (GP predictive-mean) approximator with a shape
constraint enforced via the SCOP-spline reparameterization of Pya & Wood
(2015) — the same construction used by `ShapeConstrainedBSplineApproximator`
and `ShapeConstrainedSPDEApproximator`.

Parameters are stored in unconstrained space (γ). During evaluation the
inducing-point values are computed as `β = Σ * d(γ)` (free level/slope
components pass through linearly; the rest are softplus'd), then
interpolated with the GP predictive-mean formula `k(x,X)K⁻¹β` exactly as
`GPApproximator` does. Because the constraint holds by construction in γ,
every solver fits this approximator through the ordinary `build_evaluator`
protocol with no constraint handling of its own.

**Note — constraints hold at the inducing values, NOT between them.** The
kernel interpolant's cardinal functions take negative values (measured
minimum −0.26 for the default `:sqexp` kernel at the spacing-matched
default lengthscale), so the constraint satisfied by the inducing values
does not transfer to the interpolant: on a `:positive` fixture whose 10
inducing values were all positive (alternating ≈5 / ≈0.007) the evaluator
dipped to −0.505 against a maximum of 5.6 — a 9% violation, not a slight
overshoot. Every one of the 14 constraints is affected (the
`ShapeConstrainedSPDEApproximator` cubic interpolation shares the problem
at smaller magnitude; `ShapeConstrainedBSplineApproximator` is exact
everywhere by the B-spline convex-hull property). Adding inducing points
does NOT cure it — the worst-case dip per unit inducing value GROWS with
`n_inducing` (measured 0.35 → 0.57 for n = 6 → 40 at the default kernel
settings); more points help only indirectly, by letting the fitted values
vary more smoothly relative to the spacing. What measurably reduces the
violation on the fixture above: a rougher kernel (`kernel=:matern32` at
its default lengthscale cut the dip from −0.505 to −0.023) or a shorter
explicit `lengthscale` (0.5× the default eliminated it), both at the cost
of wigglier interpolation — note an explicit `lengthscale` is also needed
to keep empirical-Bayes adaptation from re-lengthening it during a fit.
Audit any fitted result with [`check_constraints`](@ref); use
`ShapeConstrainedBSplineApproximator` when the constraint must hold
exactly.

Kernel hyperparameters follow `GPApproximator`: `lengthscale=nothing`
(default) starts at the spacing-matched default and is re-estimated by
empirical Bayes during LAML (post-warmup) and GCV fits — applied to the
IMPLIED inducing values `β = Σ·d(γ)`, which a kernel change does not move —
while an explicit `lengthscale` is fixed for the whole fit.

The LAML/GCV penalty is the Pya & Wood (2015) SCOP first-difference penalty
on γ, matching `penalty_matrix(::ShapeConstrainedBSplineApproximator)`: the
free level (and the slope-like component for curvature constraints) lies in
the penalty null space, so λ→∞ shrinks toward a maximally smooth member of
the constraint family without biasing the level.

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `n_inducing`: number of inducing points (≥ 4)
- `constraint`: one of `SHAPE_CONSTRAINTS`
- `kernel`: kernel type — `:sqexp` (default), `:matern32`, or `:matern52`
- `lengthscale`: kernel lengthscale (`nothing` = adaptive, as above)
- `variance`: signal variance σ² (default: 1.0)
- `initial`: optional initial function `x -> y` or constant

# Example
```julia
# Monotone increasing predation (functional) response
approx = ShapeConstrainedGPApproximator(:g, (0.0, 5.0), 10, :increasing;
    initial=x -> 0.1*x)
```
"""
mutable struct ShapeConstrainedGPApproximator <: AbstractApproximator
    # Mutable for the same reason as GPApproximator: the fit loop adapts
    # (lengthscale, variance) — and the derived K/K_inv — by empirical Bayes
    # on the implied inducing values β = Σ·d(γ) when the user did not supply
    # a lengthscale. A kernel change moves only the between-point behavior;
    # the interpolant still passes through β (up to the K⁻¹ jitter,
    # ~1e-8 relative). Bootstrap replicates deepcopy
    # the approximators to avoid races.
    const name::Symbol
    const domain::Tuple{Float64, Float64}
    const n_inducing::Int
    const inducing_points::Vector{Float64}
    const kernel::Symbol
    lengthscale::Float64
    variance::Float64
    const constraint::Symbol
    const Sigma::Matrix{Float64}
    const initial_func::Function
    K::Matrix{Float64}        # kernel matrix at inducing points
    K_inv::Matrix{Float64}    # inverse kernel matrix
    const adapt::Bool
end

function ShapeConstrainedGPApproximator(name::Union{Symbol,String},
                                        domain::Tuple{<:Real, <:Real},
                                        n_inducing::Int,
                                        constraint::Symbol;
                                        kernel::Symbol=:sqexp,
                                        lengthscale::Union{Nothing, Real}=nothing,
                                        variance::Real=1.0,
                                        initial=nothing)
    name_s = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("ShapeConstrainedGPApproximator", d)
    constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $SHAPE_CONSTRAINTS"))
    n_inducing >= 4 || throw(ArgumentError(
        "ShapeConstrainedGPApproximator needs n_inducing ≥ 4 (got $n_inducing)"))
    σ² = Float64(variance)
    ℓ = lengthscale === nothing ? _gp_default_lengthscale(kernel, d, n_inducing) :
        Float64(lengthscale)

    x_ind = collect(range(d[1], d[2], length=n_inducing))
    K, K_inv = _gp_kernel_matrices(kernel, ℓ, σ², x_ind)
    Sig = _build_sigma_matrix(constraint, n_inducing)

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end

    ShapeConstrainedGPApproximator(name_s, d, n_inducing, x_ind, kernel, ℓ, σ²,
                                   constraint, Sig, init_func, K, K_inv,
                                   lengthscale === nothing)
end

function nparams(a::ShapeConstrainedGPApproximator)
    a.constraint in _ZERO_ENDPOINT_CONSTRAINTS ? a.n_inducing - 1 : a.n_inducing
end

function initial_params(a::ShapeConstrainedGPApproximator)
    beta_target = Float64[a.initial_func(x) for x in a.inducing_points]
    d = a.Sigma \ beta_target
    lin = _linear_param_indices(a.constraint)
    # Free (linear) components pass through; nonnegative ones get a small
    # positive floor (large floors visibly distorted flat initial functions
    # into ramps) and are inverted through softplus.
    return [i in lin ? d[i] :
            (v = max(d[i], 1e-4); v > 20.0 ? v : log(exp(v) - 1.0))
            for i in eachindex(d)]
end

# ─── COMONet shape-constrained neural network approximator ────────

"""
Supported shape constraints for COMONet approximators.

COMONet (Constrained Monotone Network) enforces shape constraints
architecturally using `exp(W)` weights and specialized activations.
Constraints are guaranteed everywhere by construction — not just at
knot points.

Each constraint uses the architecture that makes it the network's actual
function class (a positive-weight ReLU network is always convex, so it
cannot serve as a general monotone or general positive class):

**Monotonicity** (positive weights + saturating tanh hidden units — the
class includes sigmoids and other monotone-nonconvex shapes):
- `:increasing`    — f'(x) ≥ 0
- `:decreasing`    — f'(x) ≤ 0 (negate input)

**Curvature only** (two-branch input-convex form g₁(x) + g₂(−x): convex
and possibly non-monotone, e.g. U-shapes; twice the parameters):
- `:convex`        — f''(x) ≥ 0
- `:concave`       — f''(x) ≤ 0 (negated sum)

**Monotone + curvature** (positive weights + ReLU/softplus or their
negated concave forms):
- `:inc_convex`    — increasing + convex
- `:inc_concave`   — increasing + concave
- `:dec_convex`    — decreasing + convex
- `:dec_concave`   — decreasing + concave

**Positivity** (unconstrained tanh MLP inside exp(·) — humps allowed):
- `:positive`      — f(x) > 0
"""
const COMONET_CONSTRAINTS = (
    :increasing, :decreasing,
    :convex, :concave,
    :inc_convex, :inc_concave,
    :dec_convex, :dec_concave,
    :positive,
)

"""
    COMONetApproximator(name, domain, hidden_sizes, constraint;
                        penalty_weight=0.01, activation=:relu)

Shape-constrained neural network approximator using the COMONet architecture.
Constraints are enforced architecturally via `exp(W)` weights and specialized
activations — guaranteed to hold everywhere, not just at sample points.

Unlike `ShapeConstrainedBSplineApproximator` (which uses B-spline basis
functions), COMONet uses a neural network that can represent more complex
functions while still guaranteeing shape constraints.

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` tuple for input normalization to `[0, 1]`
- `hidden_sizes`: tuple of hidden layer widths, e.g. `(16, 16)`
- `constraint`: one of `COMONET_CONSTRAINTS`
- `penalty_weight`: L2 regularization on unconstrained weights (for LAML)
- `activation`: `:relu` (default, piecewise linear C⁰) or `:softplus` (smooth C∞).
  Both preserve the convexity/concavity guarantees. Use `:softplus` when smooth
  derivatives are needed.

  **This argument applies only to the curvature-constrained classes** —
  `:convex`, `:concave`, `:inc_convex`, `:inc_concave`, `:dec_convex`,
  `:dec_concave` — which are built from `_comonet_branch`. The purely monotone
  classes `:increasing` and `:decreasing` use a **tanh** network
  (`_comonet_monotone`) and **ignore `activation` entirely**; `:positive` uses a
  tanh MLP inside `exp(·)`. Note that the monotone network's output layer is
  `exp(W̃)·h + b` with an unconstrained bias `b` and `h ∈ (−1,1)`, so its range is
  sign-unrestricted — monotonicity is guaranteed, but there is no zero floor.

# Example
```julia
uf = COMONetApproximator(:f, (0.0, 100.0), (16, 16), :increasing)
uf_smooth = COMONetApproximator(:f, (0.0, 100.0), (16, 16), :increasing;
                                activation=:softplus)
```
"""
const COMONET_ACTIVATIONS = (:relu, :softplus)

"""
    COMONetApproximator <: AbstractApproximator

Shape-constrained neural network approximator (COMONet architecture).
See [`COMONetApproximator(name, domain, hidden_sizes, constraint)`](@ref) for constructor docs.
"""
struct COMONetApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    hidden_sizes::Tuple{Vararg{Int}}
    constraint::Symbol
    penalty_weight::Float64
    activation::Symbol
    rng_seed::Union{Nothing, Int}
end

function COMONetApproximator(name::Union{Symbol,String},
                             domain::Tuple{<:Real, <:Real},
                             hidden_sizes::Union{Tuple{Vararg{Int}}, Vector{Int}},
                             constraint::Symbol;
                             penalty_weight::Float64=0.01,
                             activation::Symbol=:relu,
                             rng_seed::Union{Nothing, Int}=nothing)
    name = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    _validate_domain("COMONetApproximator", d)
    constraint in COMONET_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $COMONET_CONSTRAINTS"))
    activation in COMONET_ACTIVATIONS || throw(ArgumentError(
        "Unknown activation :$activation. Must be one of $COMONET_ACTIVATIONS"))
    hs = hidden_sizes isa Vector ? Tuple(hidden_sizes) : hidden_sizes
    length(hs) >= 1 || throw(ArgumentError("Need at least 1 hidden layer"))
    all(>(0), hs) || throw(ArgumentError("COMONetApproximator: hidden_sizes must all be > 0, got $hs"))
    (isfinite(penalty_weight) && penalty_weight >= 0.0) || throw(ArgumentError(
        "COMONetApproximator: penalty_weight must be finite and ≥ 0, got $penalty_weight"))
    COMONetApproximator(name, d, hs, constraint, penalty_weight, activation,
                        rng_seed)
end

function _comonet_nparams_single(hidden_sizes)
    np = 0
    prev = 1  # single input (1D)
    for h in hidden_sizes
        np += prev * h + h  # W (prev×h) + b (h)
        prev = h
    end
    np + prev + 1  # output layer: W (prev×1) + b (1)
end

function nparams(a::COMONetApproximator)
    n1 = _comonet_nparams_single(a.hidden_sizes)
    # Two-branch input-convex constraints carry a pair of networks
    a.constraint in (:convex, :concave) ? 2 * n1 : n1
end

function initial_params(a::COMONetApproximator)
    rng = a.rng_seed === nothing ? Random.default_rng() :
                                   Random.Xoshiro(a.rng_seed)
    np = nparams(a)
    # W̃ ≈ 0 corresponds to fan-in-scaled unit-gain weights exp(0)/fanin
    # (see _comonet_branch), so a small-scale init starts near a tame,
    # roughly linear network.
    params = 0.1 .* randn(rng, np)

    # Activation-aware bias initialization. ReLU branches only have
    # gradient where their pre-activation is positive; with b ≈ 0 the
    # units whose input is negated (x ↦ −x_norm ∈ [−1, 0]) start almost
    # entirely dead and never recover under gradient descent.
    _set_hidden_biases! = function (θ, lo_idx, bias_mean)
        idx = lo_idx
        prev = 1
        for h in a.hidden_sizes
            idx += prev * h                      # skip weights
            θ[idx:idx+h-1] .= bias_mean .+ 0.1 .* randn(rng, h)
            idx += h
            prev = h
        end
        idx + prev                                # start of output bias block
    end
    if a.constraint in (:inc_concave, :dec_concave)
        # -relu(-z) needs z < 0 to be active: negative biases
        _set_hidden_biases!(params, 1, -1.0)
        params[end] = 1.0                         # output bias compensates
    elseif a.constraint == :dec_convex
        # relu on negated input (z = W(−x)+b ≤ b): positive biases
        _set_hidden_biases!(params, 1, 1.0)
    elseif a.constraint in (:convex, :concave)
        # two-branch g₁(x) + g₂(−x): only the g₂ branch sees negated input
        nh = length(params) ÷ 2
        _set_hidden_biases!(params, nh + 1, 1.0)
    end

    return params
end

# ─── Single-index approximator ────────────────────────────────────

"""
    SingleIndexApproximator(name, p, nknots; anchor=1, constraint=:none,
                            xi=2.0, index_stats=nothing, inner_ridge=1e-4,
                            states=nothing, initial=nothing)

Unknown function of SEVERAL states through ONE learned direction:

    f(u₁, …, u_p) = s(z),    z = (aᵀu − aᵀμ̂) / √(aᵀΣ̂a)

with the loadings `a` and the univariate outer smooth `s` estimated
JOINTLY. This is the nested-effects construction of Fasiolo et al.
(arXiv:2511.19234; R package *gamFactory*): a smooth composed with a
learned inner transformation, both fitted under LAML with separate
smoothing parameters (see [`penalty_blocks`](@ref)).

# Why a single index rather than a tensor surface

[`TensorBSplineApproximator`](@ref) fits a full interaction surface
`f(x, y)`, but a dynamical orbit is a thin 1-D manifold in state space:
the data identify the surface only along the trajectory, and off-orbit
the fit is unconstrained (worse still as the number of arguments grows).
A single index needs variation only ALONG the learned direction, which
orbit data supply. The loadings are directly interpretable and the outer
smooth is univariate, so [`confidence_band`](@ref) serves it (evaluating
the OUTER curve over the standardized index) where the tensor surface has
no univariate band at all. Explicit `uf_points` instead evaluates the
full function at physical state coordinates, including loading sensitivity.

# The standardization statistics are FIXED

`(μ̂, Σ̂)` are reference statistics of the index variables, computed ONCE
and never touched again. The paper standardizes the inner transform over
the covariate sample; here the "covariate" of a state-dependent `f` is
the TRAJECTORY, which moves during fitting — standardizing against it
would add derivative paths through the standardization and let the
geometry of `z` drift under the optimizer. Instead:

- `index_stats = (mu, Sigma)` fixes them explicitly (`mu` of length `p`,
  `Sigma` a `p × p` symmetric positive semi-definite matrix with positive
  trace). Use this whenever you know the operating range.
- `index_stats = nothing` (default) leaves them UNRESOLVED on the bare
  approximator, and `PSMProblem` construction fills them in from the data
  (see below), returning a resolved copy. The struct is immutable, so once
  resolved the statistics cannot change — no in-loop adaptation, and
  `bootstrap`'s `deepcopy` of the approximators carries the same values.
  Building an evaluator from an unresolved approximator is an error rather
  than a silent default.

Resolution from data smooths each observed state column with the package's
GCV smoothing spline (`smooth_and_differentiate`) and takes the mean and
sample covariance of the smoothed index states. A state that carries no
observation contributes its `u0` value as the mean and a diffuse scale
`max(|μ̂ⱼ|, 1)` as its standard deviation, with zero cross-covariance. The
same diffuse fallback also applies to an OBSERVED state whose empirical
spread is degenerate (a near-constant column): its variance becomes
`max(|μ̂ⱼ|, 1)²`, which for a state observed near 100 means ≈1e4 and so
flattens that coordinate's contribution to the index by ~100×. Pass
`index_stats` explicitly if you want a near-constant state to carry its
measured scale instead.
`aᵀΣ̂a` is additionally floored at `1e-8 · tr(Σ̂)/p · ‖a‖²` inside the
evaluator — a floor proportional to `‖a‖²` so the guard cannot break the
exact scale invariance below.

`states` names the state indices the `p` arguments correspond to; it is
used ONLY to derive `(μ̂, Σ̂)` from data and defaults to `1:p`. If your
dynamics pass a different selection — `p.f(u[2], u[4])` — pass
`states=[2, 4]` (or supply `index_stats` directly).

# Identification

`z` is invariant under `a → c·a` for `c > 0`, so the radial direction of
the loadings is EXACTLY flat — poison for the LAML log-determinant and for
any Newton step. The default `anchor = 1` removes it in the classical
single-index way: `a[anchor] ≡ 1` is not a parameter, the block stores the
remaining `p − 1` free loadings, and both the scale AND the sign of the
index are fixed. **The anchor variable must genuinely load on the index**
— anchoring a variable whose true loading is ~0 forces the other loadings
to blow up. Pass `anchor = k` to anchor a different argument.

`anchor = nothing` keeps all `p` loadings and leans on the inner ridge to
regularize the flat direction; `index_loadings` then reports `a` scaled to
`‖a‖ = 1` with a positive-first-loading sign convention.

!!! warning "Free mode degenerates under LAML/GCVSolver"
    The flat direction is exactly flat in the DATA term, so a ridge on `a`
    is minimized by `‖a‖ → 0`. Under a solver that also estimates λ this
    collapses: a measured `LAML` fit went from `‖a‖ = 1.41` at
    initialization to `‖a‖ = 5.7e-6`, with an inner λ of 6.3e5 and a data
    loss indistinguishable from the anchored fit. Free mode is for the
    flat-objective solvers (`AdamSolver`, `DerivativeFreeSolver`,
    `MCMCSolver`); `LAML` and `GCVSolver` emit a warning if they meet it.
    Anchored mode is the default for this reason.

The loadings also need the state path to CHANGE DIRECTION. What defeats
identification is collinearity — states moving along an affinely straight
line, where rescaling `a` is absorbed into `s`. Monotonicity alone does
not: on decay fixtures where every coordinate and the index itself
decrease monotonically, the loadings were still recovered to ~2%, because
a curved path keeps changing its tangent direction.

# Arguments
- `name`: symbol for the unknown function, called as `p.name(u1, …, up)`
- `p`: number of index arguments (≥ 2)
- `nknots`: knots of the outer smooth (≥ 3; ≥ 4 when `constraint ≠ :none`)
- `anchor`: index in `1:p` pinned to 1, or `nothing` (default `1`)
- `constraint`: `:none` (default) or any of [`SHAPE_CONSTRAINTS`](@ref) —
  the outer smooth is then the SCOP-spline construction of Pya & Wood
  (2015), reusing `ShapeConstrainedBSplineApproximator` verbatim
- `xi`: the outer smooth's knots span `[−xi, xi]` in STANDARDIZED units
  (default 2.0), with linear extrapolation outside, as everywhere else in
  the package
- `index_stats`: `(mu, Sigma)` or `nothing` (resolve from data)
- `inner_ridge`: fixed weight of the inner ridge in the merged
  `penalty_matrix` read by the single-λ consumers (default 1e-4). Under
  `LAML`/`GCVSolver` the inner ridge is a penalty BLOCK with its own
  estimated λ, so this value is irrelevant there
- `states`: state indices of the `p` arguments (default `1:p`)
- `initial`: optional initial outer curve `z -> s(z)` or a constant
- `initial_loadings`: optional starting direction, a length-`p` vector.
  Default `nothing` starts from EQUAL weights (`a = 1` everywhere,
  including the anchor). In anchored mode the vector is rescaled by its
  anchor entry, which must be non-zero

# On initializing the direction from the data

Seeding `a` by regressing smoothed state derivatives on the states — the
average-derivative estimator, which is consistent for `y = s(aᵀx)` — was
implemented and measured, and it is NOT the default because in a PSM the
state derivative is a SUM of known terms and the unknown function, so the
regression is contaminated. On the two recovery fixtures in the test
suite it produced a WRONG-SIGNED direction on one and, on the other,
turned an `AdamSolver` fit of loss 0.24 into one of loss 8.84; under
`LAML` it changed nothing (identical fits from either start). Equal
weights is therefore the default, and `initial_loadings` is there for the
case where you actually know a plausible direction.

# Example
```julia
uf = SingleIndexApproximator(:g, 2, 8)          # g(N, P) = s(a₁N + a₂P)
predprey!(du, u, p, t) = begin
    g = p.g(u[1], u[2])
    du[1] = u[1] * (1 - u[1] / 6) - g
    du[2] = p.e * g - p.m * u[2]
end
```

# Reference
Fasiolo, M. et al. Nested effects for generalized additive models.
arXiv:2511.19234.
"""
struct SingleIndexApproximator{O<:AbstractApproximator} <: AbstractApproximator
    name::Symbol
    p::Int
    nknots::Int
    anchor::Union{Int, Nothing}
    constraint::Symbol
    xi::Float64
    inner_ridge::Float64
    # Domain of the STANDARDIZED index (−xi, xi). Present so the generic
    # `confidence_band` / bootstrap gridding machinery, which reads
    # `approx.domain`, grids the outer curve.
    domain::Tuple{Float64, Float64}
    states::Vector{Int}
    outer::O
    # Reference statistics — `nothing` until resolved. IMMUTABLE once set.
    mu::Union{Nothing, Vector{Float64}}
    Sigma::Union{Nothing, Matrix{Float64}}
    # Optional user-supplied starting direction, stored as the FREE loadings
    # (length `_si_n_inner`); `nothing` means equal weights.
    loadings_init::Union{Nothing, Vector{Float64}}
end

"""
    _si_validate_index_stats(name, p, index_stats) -> (mu, Sigma)

Validate a user-supplied `(mu, Sigma)` pair: correct sizes, symmetric,
positive semi-definite, and with strictly positive trace (a zero Σ̂ would
make every standardized index infinite).
"""
function _si_validate_index_stats(name::Symbol, p::Int, index_stats)
    (index_stats isa Tuple && length(index_stats) == 2) || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats must be a tuple " *
        "(mu::Vector, Sigma::Matrix), got $(typeof(index_stats))"))
    mu = Float64.(collect(index_stats[1]))
    Sig = Matrix{Float64}(index_stats[2])
    length(mu) == p || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats mu has length " *
        "$(length(mu)) but p = $p"))
    size(Sig) == (p, p) || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats Sigma is " *
        "$(size(Sig, 1))×$(size(Sig, 2)) but p = $p"))
    scale = max(maximum(abs, Sig), 1.0)
    maximum(abs.(Sig .- Sig')) <= 1e-8 * scale || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats Sigma is not symmetric"))
    Sig = (Sig .+ Sig') ./ 2
    minimum(eigvals(Symmetric(Sig))) >= -1e-8 * scale || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats Sigma is not positive " *
        "semi-definite"))
    tr(Sig) > 0 || throw(ArgumentError(
        "SingleIndexApproximator(:$name): index_stats Sigma has zero trace; " *
        "the standardized index would be undefined"))
    mu, Sig
end

function SingleIndexApproximator(name::Union{Symbol,String},
                                 p::Int,
                                 nknots::Int;
                                 anchor::Union{Int,Nothing}=1,
                                 constraint::Symbol=:none,
                                 xi::Real=2.0,
                                 index_stats=nothing,
                                 inner_ridge::Real=1e-4,
                                 states::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                                 initial=nothing,
                                 initial_loadings::Union{Nothing,AbstractVector{<:Real}}=nothing)
    name_s = Symbol(name)
    p >= 2 || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s) needs p ≥ 2 index arguments " *
        "(got $p); a single-argument unknown function is an ordinary " *
        "BSplineApproximator"))
    constraint == :none || constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): unknown constraint :$constraint. " *
        "Must be :none or one of $SHAPE_CONSTRAINTS"))
    nknots >= 3 || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s) needs nknots ≥ 3 for the outer " *
        "cubic smooth (got $nknots)"))
    constraint == :none || nknots >= 4 || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s) needs nknots ≥ 4 for a " *
        "shape-constrained outer smooth (got $nknots)"))
    ξ = Float64(xi)
    (isfinite(ξ) && ξ > 0) || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): xi must be positive and finite, " *
        "got $xi"))
    anchor === nothing || (1 <= anchor <= p) || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): anchor must lie in 1:$p or be " *
        "nothing, got $anchor"))
    ridge = Float64(inner_ridge)
    (isfinite(ridge) && ridge >= 0) || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): inner_ridge must be finite and " *
        "non-negative, got $inner_ridge"))
    st = states === nothing ? collect(1:p) : Int.(collect(states))
    length(st) == p || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): states has $(length(st)) entries " *
        "but p = $p"))
    (all(>=(1), st) && allunique(st)) || throw(ArgumentError(
        "SingleIndexApproximator(:$name_s): states must be distinct positive " *
        "state indices, got $st"))

    dom = (-ξ, ξ)
    outer = if constraint == :none
        BSplineApproximator(name_s, dom, nknots; initial=initial)
    else
        ShapeConstrainedBSplineApproximator(name_s, dom, nknots, constraint;
                                            initial=initial)
    end

    mu, Sig = index_stats === nothing ? (nothing, nothing) :
              _si_validate_index_stats(name_s, p, index_stats)

    lo = if initial_loadings === nothing
        nothing
    else
        v = Float64.(collect(initial_loadings))
        length(v) == p || throw(ArgumentError(
            "SingleIndexApproximator(:$name_s): initial_loadings has " *
            "$(length(v)) entries but p = $p"))
        all(isfinite, v) || throw(ArgumentError(
            "SingleIndexApproximator(:$name_s): initial_loadings must be finite"))
        if anchor === nothing
            v
        else
            v[anchor] != 0 || throw(ArgumentError(
                "SingleIndexApproximator(:$name_s): initial_loadings[$anchor] " *
                "is zero, but the anchored parameterization fixes " *
                "a[$anchor] = 1 and rescales the direction by it"))
            v = v ./ v[anchor]
            [v[i] for i in 1:p if i != anchor]
        end
    end

    SingleIndexApproximator(name_s, p, nknots, anchor, constraint, ξ, ridge,
                            dom, st, outer, mu, Sig, lo)
end

"""Number of FREE inner loadings: `p − 1` when anchored, `p` otherwise."""
_si_n_inner(a::SingleIndexApproximator) = a.anchor === nothing ? a.p : a.p - 1

nparams(a::SingleIndexApproximator) = _si_n_inner(a) + nparams(a.outer)

function initial_params(a::SingleIndexApproximator)
    ni = _si_n_inner(a)
    # Equal weights (a = 1 in every coordinate, including the anchor) unless
    # the user supplied `initial_loadings`. See the constructor docstring for
    # why the data-driven average-derivative seed is NOT the default.
    inner = a.loadings_init === nothing ? ones(ni) : copy(a.loadings_init)
    vcat(inner, initial_params(a.outer))
end

# ─── Resolving the fixed index statistics at problem construction ──

"""
    _resolve_index_stats(approx, u0, data_times, data_values, data_weights,
                         obs_to_state) -> approx

Hook called once per approximator by the `PSMProblem` constructor. Every
type but [`SingleIndexApproximator`](@ref) is returned untouched; a
single-index approximator whose reference statistics are still `nothing`
is replaced by a RESOLVED COPY carrying `(μ̂, Σ̂)` derived from the data.
Already-resolved approximators are returned unchanged, so re-wrapping a
problem (as `bootstrap` does) is idempotent.
"""
_resolve_index_stats(approx::AbstractApproximator, u0, data_times,
                     data_values, data_weights, obs_to_state) = approx

function _resolve_index_stats(a::SingleIndexApproximator, u0,
                              data_times::Vector{Float64},
                              data_values::Matrix{Float64},
                              data_weights::Matrix{Float64},
                              obs_to_state::Vector{Int})
    a.mu === nothing || return a          # user-supplied: never overwritten

    p = a.p
    u0v = u0 isa AbstractVector ? Float64.(collect(u0)) : nothing
    K = max(maximum(obs_to_state), maximum(a.states),
            u0v === nothing ? 0 : length(u0v))
    observed = Set(obs_to_state)
    # An index state outside the model's state vector would silently index
    # past u0 below; name it here instead.
    u0v === nothing || maximum(a.states) <= length(u0v) || throw(ArgumentError(
        "SingleIndexApproximator(:$(a.name)): states $(a.states) reference " *
        "a state index beyond the $(length(u0v))-element u0"))
    any(s -> s in observed, a.states) || u0v !== nothing || throw(ArgumentError(
        "SingleIndexApproximator(:$(a.name)): none of the index states " *
        "$(a.states) is observed and u0 is a function, so the reference " *
        "statistics (μ̂, Σ̂) cannot be derived. Pass index_stats=(mu, Sigma)."))

    # Smooth the observed columns once — the same GCV smoothing spline the
    # gradient-matching solvers use. Noise in the raw data would inflate Σ̂
    # and shrink every standardized index toward zero.
    ys, _ = try
        smooth_and_differentiate(data_times, data_values, obs_to_state, K;
                                 weights=data_weights)
    catch e
        _is_program_error(e) && rethrow()
        @warn("SingleIndexApproximator(:$(a.name)): could not smooth the data " *
              "to derive the reference statistics (μ̂, Σ̂); falling back to " *
              "u0-centred diffuse statistics. Pass index_stats=(mu, Sigma) " *
              "to control them.", exception = e, maxlog = 1)
        nothing, nothing
    end

    mu = zeros(p)
    M = ys === nothing ? nothing : Matrix{Float64}(undef, size(ys, 1), p)
    for j in 1:p
        s = a.states[j]
        if ys !== nothing && s in observed
            M[:, j] = ys[:, s]
            mu[j] = mean(@view M[:, j])
        else
            v = u0v === nothing ? 0.0 : u0v[s]
            mu[j] = v
            M === nothing || (M[:, j] .= v)
        end
    end

    Sig = if M !== nothing && size(M, 1) >= 2
        Matrix(cov(M))
    else
        zeros(p, p)
    end
    # Any index coordinate with no (or degenerate) empirical spread gets a
    # diffuse scale and no cross-covariance: an unobserved state must not
    # make Σ̂ singular in its direction.
    for j in 1:p
        if !(isfinite(Sig[j, j])) || Sig[j, j] <= 1e-12 * max(abs(mu[j])^2, 1.0)
            Sig[j, :] .= 0.0
            Sig[:, j] .= 0.0
            Sig[j, j] = max(abs(mu[j]), 1.0)^2
        end
    end
    all(isfinite, Sig) || (Sig = Matrix(Diagonal([max(abs(m), 1.0)^2 for m in mu])))
    Sig = (Sig .+ Sig') ./ 2

    SingleIndexApproximator(a.name, a.p, a.nknots, a.anchor, a.constraint,
                            a.xi, a.inner_ridge, a.domain, a.states, a.outer,
                            mu, Sig, a.loadings_init)
end

# ─── Transformed-covariate approximator ────────────────────────────

"""
    _tc_logistic(z)

The logistic function written as `(1 + tanh(z/2))/2`. Algebraically
identical to `1/(1 + exp(-z))` but with no overflow branch: `exp(-z)`
overflows to `Inf` for `z ≲ -710` and then produces a `NaN` derivative,
while `tanh` saturates cleanly. Eltype-generic and branch-free, so it is
Dual-safe in the exponential-smoothing scan below.
"""
_tc_logistic(z) = (one(z) + tanh(z / 2)) / 2

"""
    _tc_linterp(ts, vals, t)

Linear interpolation of `vals` over the strictly increasing grid `ts`,
held CONSTANT at the end values outside `[ts[1], ts[end]]`. Eltype-generic
in `vals` (so `ForwardDiff.Dual` node values propagate) and in `t`.

The ODE solver asks for the unknown function at arbitrary `t`, so the
transformed covariate has to be continuous everywhere — a step function
read off the nearest grid cell would give the integrator a discontinuous
right-hand side and wreck adaptive stepping. Piecewise-linear is
continuous but only C⁰: it has kinks at the covariate times, which is why
the docstring of [`TransformedCovariateApproximator`](@ref) recommends
`jac=:forwarddiff`.
"""
function _tc_linterp(ts::AbstractVector{Float64}, vals::AbstractVector, t)
    n = length(ts)
    t <= ts[1] && return vals[1] * one(t)
    t >= ts[n] && return vals[n] * one(t)
    i = searchsortedlast(ts, t)
    i >= n && return vals[n] * one(t)
    w = (t - ts[i]) / (ts[i + 1] - ts[i])
    vals[i] + w * (vals[i + 1] - vals[i])
end

"""
    TransformedCovariateApproximator(name, times, X; trans, nknots=8, lags=0,
                                     anchor=1, constraint=:none, xi=2.0,
                                     inner_ridge=1e-4, initial=nothing)

Unknown function of TIME driven by an EXOGENOUS covariate through a
LEARNED inner transformation:

    f(t) = s(z(t)),    z = (s̃ − mean s̃) / sd s̃,    s̃ = T_a(x)

where `x` is a user-supplied covariate series sampled at `times`, `T_a` is
one of two parametric transformations with free parameters `a`, and `s` is
a univariate smooth. Both are estimated JOINTLY, with SEPARATE smoothing
parameters (see [`penalty_blocks`](@ref)). This is the nested-effects
construction of Fasiolo et al. (arXiv:2511.19234; R package *gamFactory*),
and it is the sibling of [`SingleIndexApproximator`](@ref): same outer
smooth on a standardized inner statistic, different inner transformation.

In the dynamics the fitted object is a ONE-ARGUMENT callable of time, the
same convention every other time-varying unknown function in the package
uses (`examples/sir_spline.jl`, `examples/copepod.jl`):

```julia
sir!(du, u, p, t) = begin
    du[1] = -p.beta(t) * u[1] * u[2] / p.N
    du[2] =  p.beta(t) * u[1] * u[2] / p.N - p.gamma * u[2]
end
```

# The two inner transformations

`trans = :expsm` — **adaptive exponential smoothing**. A causal
exponentially weighted mean of the covariate whose INERTIA is learned:

    s̃₁ = x₁,    s̃ᵢ = ωᵢ s̃ᵢ₋₁ + (1 − ωᵢ) xᵢ,    ωᵢ = logistic(w̃ᵢᵀa)

With a single covariate column (the default) `w̃ᵢ ≡ 1` and there is ONE
parameter, a constant inertia `ω = logistic(a₁)`. Extra columns of `X` are
AUXILIARY covariates that enter `w̃ᵢ = [1, X[i, 2], …]`, making the inertia
adaptive. The scientific case: transmission responding to temperature with
a learned thermal lag, where `ω` is the memory of the environment rather
than of the pathogen.

`trans = :lagindex` — **distributed lag**. A weighted sum over a lag
window of length `lags`:

    s̃ᵢ = Σ_{ℓ=0}^{lags−1} a_{ℓ+1} · x(tᵢ − ℓΔ),   Δ = (t_end − t₁)/(n − 1)

`Δ` is the mean sample spacing and `x` is read off the covariate by the
same linear interpolation used for `z(t)`, held constant before `t₁`. The
lag matrix is FIXED data, so it is built once in the constructor and the
transformation is then exactly LINEAR in `a`.

# The standardization is EASY here — and that is the point

The paper standardizes the transformed series to mean 0 and variance 1
over the covariate sample, and here that recipe ports VERBATIM: the
covariate is fixed data, so `mean s̃` and `sd s̃` are smooth, differentiable
functions of `a` alone with no feedback from the fitted trajectory.

Contrast [`SingleIndexApproximator`](@ref), whose "covariate" is the
trajectory itself: it standardizes against reference statistics `(μ̂, Σ̂)`
resolved ONCE from the data at `PSMProblem` construction and then frozen,
precisely to keep the geometry of `z` from drifting under the optimizer.
This type needs NONE of that machinery — no `index_stats` keyword, no
unresolved state, no `_resolve_index_stats` hook, no `PSMProblem`
involvement at all. It is fully self-contained from construction, so
`build_evaluator` works on a bare approximator and `bootstrap`'s
`deepcopy` is trivially faithful.

`sd s̃` is computed as `√(v + κ)` with `v` the sample variance and
`κ = 1e-10 · max(max|x|, 1)²` a fixed, data-derived floor. The floor is
ADDITIVE rather than a `max(...)` clamp so that `z` stays smooth in `a`
everywhere — including the `ω → 1` limit of `:expsm`, where `s̃` collapses
to the constant `x₁` and `z → 0`, i.e. infinite thermal inertia means a
constant response. That limit is a legitimate model, not a numerical
failure, and it should not produce a kink or a `NaN`.

Two exact consequences of the additive form, both benign and both asserted
in the test suite. The standardized variance is `v/(v + κ)`, so it
approaches 1 strictly from BELOW. The size of the deficit is governed by
the transform parameters, NOT by the covariate's scaling: over the
parameter range used here it is ≤ 2e-8, but it grows as the transformed
series flattens — ~2.5e-2 at an inertia parameter of 10, and → 1 beyond
about 20, which is the documented infinite-inertia limit where `z → 0`. And the `a → c·a` invariance of `:lagindex` holds only to
`≈ κ/v ~ 1e-10` relative, where [`SingleIndexApproximator`](@ref) keeps its
analogous invariance EXACT by making its floor proportional to `‖a‖²`. The
difference does not matter here because the anchor, not the floor, is what
removes that direction — whereas smoothness in `a` is needed everywhere.

# Identification

`z` is invariant under `a → c·a` (`c > 0`) for `:lagindex`, exactly the
flat direction [`SingleIndexApproximator`](@ref) has, so `:lagindex` uses
the SAME fix: `anchor = 1` pins `a[anchor] ≡ 1` (by default the lag-0
weight), it is not a parameter, and the block stores the remaining
`lags − 1` free weights. This fixes both the scale and the sign of the
index. There is no `anchor = nothing` free mode: N1 measured that mode
collapsing to `‖a‖ → 0` under any λ-estimating solver, and nothing here
would make it behave better.

`:expsm` has no such invariance — `ω` enters through a logistic, so every
`a` gives a genuinely different weighting — and therefore takes no anchor.
Its inner penalty is a plain ridge, which shrinks the inertia toward
`ω = 1/2`.

!!! note "ω is identified through PHASE, not amplitude"
    Standardization removes the amplitude damping that different `ω` apply
    to a periodic covariate, so what is left to identify `ω` is the PHASE
    LAG of the smoothed series. On a seasonal driver that is real
    information, but it is second-order compared with the response shape,
    and the inner ridge pulls toward `ω = 1/2`. Expect the outer curve to
    be recovered much more sharply than the inertia; see the
    `TransformedCovariateApproximator` recovery testset for measured
    numbers.

# Arguments
- `name`: symbol for the unknown function, called as `p.name(t)`
- `times`: strictly increasing covariate sample times (≥ 3 of them). They
  should span the problem's `tspan`; outside them `z` is held constant at
  its end values
- `X`: `length(times) × m` covariate matrix. Column 1 is the DRIVER `x`;
  for `:expsm` columns `2:m` are auxiliary inertia covariates, and for
  `:lagindex` `m` must be 1. An `AbstractVector` is accepted and treated
  as a single column
- `trans`: `:expsm` or `:lagindex` (required)
- `nknots`: knots of the outer smooth (≥ 3; ≥ 4 when `constraint ≠ :none`)
- `lags`: lag-window length for `:lagindex` (≥ 1, ≤ `length(times)`); must
  be left at 0 for `:expsm`
- `anchor`: `:lagindex` only — which lag is pinned to 1, in `1:lags`
  (default 1, the contemporaneous term)
- `constraint`: `:none` (default) or any of [`SHAPE_CONSTRAINTS`](@ref);
  the outer smooth is then the SCOP-spline construction of Pya & Wood
  (2015), reusing `ShapeConstrainedBSplineApproximator` verbatim
- `xi`: the outer smooth's knots span `[−xi, xi]` in STANDARDIZED units
  (default 2.0), with linear extrapolation outside. **`domain` is
  therefore standardized-covariate units, NOT time** — which is what makes
  [`confidence_band`](@ref) report the band of the RESPONSE CURVE `s(z)`
  rather than of `f(t)` by default. Explicit `uf_points` supplies times
  and evaluates the full `f(t)`, including inner-parameter sensitivity
- `inner_ridge`: fixed weight of the inner penalty in the merged
  [`penalty_matrix`](@ref) read by the single-λ consumers (default 1e-4).
  Under `LAML`/`GCVSolver` the inner penalty is a block with its own
  estimated λ, so this value is irrelevant there
- `initial`: optional initial outer curve `z -> s(z)`, or a constant

# Fitting notes: `jac`, and where to start λ

`f(t)` is piecewise linear in `t` between covariate times, so its
derivative is discontinuous there. That is a property of the ARGUMENT, not
of the parameters — the evaluator is perfectly smooth in `a` — but the
kinks still inject adaptive-step noise into the finite-difference
prediction Jacobian. Prefer `LAML(jac=:forwarddiff)` (likewise
`GCVSolver`, `CollocationLAML`), or sample the covariate finely enough
that the kinks are small. On the SIR fixture in the test suite the default
Jacobian settles at a badly worse optimum than `jac=:forwarddiff`'s
(objective 480.98, `λ_inner` 11.56, converged) — by 130 to 1500 nats
depending on the environment — and, in the default run configuration,
does so while REPORTING CONVERGENCE. The size and the reported status
are not stable enough to quote: the finite-difference optimum is chaotic
in the exact arithmetic, moving ~130 nats and six orders of magnitude in
`λ_inner` between BLAS thread counts, and it exits `:maxiters` under
`--check-bounds=yes` where it claims `:converged_tol` without it. Treat
it as "the fd path can land far away and may call it success", not as a
reproducible number.

Composing the `:expsm` scan with a SHAPE-CONSTRAINED outer smooth is
strongly nonlinear, and `LAML`'s `initial_lambda` matters there. Starting
from a very light penalty can fail to converge: on that same fixture
`initial_lambda = 0.01` exits on `:maxiters` with a β-RMSE of 0.031,
where `initial_lambda = 1.0` (or `0.1` with a longer `warmup`) converges
to objective 479.89, `λ = [11.18, 6.07]`, `edf` 5.44. Always check
`sol.convergence.converged`: the unconverged fit still returns finite,
plausible-looking numbers, and its λ and `edf` can look entirely
reasonable — the convergence flag is the reliable signal, not the
magnitudes. The unconstrained outer smooth is not sensitive this way.

# Example
```julia
temp  = 15.0 .+ 8.0 .* sin.(2π .* (0:120) ./ 365)
uf_β  = TransformedCovariateApproximator(:beta, collect(0.0:120.0), temp;
                                         trans=:expsm, nknots=8,
                                         constraint=:increasing)
```

# Reference
Fasiolo, M. et al. Nested effects for generalized additive models.
arXiv:2511.19234.
"""
struct TransformedCovariateApproximator{O<:AbstractApproximator} <: AbstractApproximator
    name::Symbol
    trans::Symbol
    times::Vector{Float64}
    X::Matrix{Float64}
    nknots::Int
    lags::Int
    anchor::Union{Int, Nothing}
    constraint::Symbol
    xi::Float64
    inner_ridge::Float64
    # Domain of the STANDARDIZED covariate (−xi, xi) — NOT time. Present so
    # the generic `confidence_band` / bootstrap gridding machinery, which
    # reads `approx.domain`, grids the OUTER response curve.
    domain::Tuple{Float64, Float64}
    outer::O
    # Fixed data, precomputed once: for :expsm the n × n_inner design of the
    # inertia linear predictor, `[1 aux…]`; for :lagindex the n × lags matrix
    # of lagged covariate values, so `s̃ = inner_design * a` exactly.
    inner_design::Matrix{Float64}
    # Additive variance floor keeping `sd s̃` smooth at a degenerate s̃.
    var_floor::Float64
end

const _TC_TRANSFORMS = (:expsm, :lagindex)

function TransformedCovariateApproximator(name::Union{Symbol,String},
                                          times::AbstractVector,
                                          X::AbstractMatrix;
                                          trans::Symbol,
                                          nknots::Int=8,
                                          lags::Int=0,
                                          anchor::Int=1,
                                          constraint::Symbol=:none,
                                          xi::Real=2.0,
                                          inner_ridge::Real=1e-4,
                                          initial=nothing)
    name_s = Symbol(name)
    pre = "TransformedCovariateApproximator(:$name_s)"
    trans in _TC_TRANSFORMS || throw(ArgumentError(
        "$pre: unknown trans :$trans. Must be one of $_TC_TRANSFORMS"))
    constraint == :none || constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "$pre: unknown constraint :$constraint. Must be :none or one of " *
        "$SHAPE_CONSTRAINTS"))
    nknots >= 3 || throw(ArgumentError(
        "$pre needs nknots ≥ 3 for the outer cubic smooth (got $nknots)"))
    constraint == :none || nknots >= 4 || throw(ArgumentError(
        "$pre needs nknots ≥ 4 for a shape-constrained outer smooth " *
        "(got $nknots)"))
    ξ = Float64(xi)
    (isfinite(ξ) && ξ > 0) || throw(ArgumentError(
        "$pre: xi must be positive and finite, got $xi"))
    ridge = Float64(inner_ridge)
    (isfinite(ridge) && ridge >= 0) || throw(ArgumentError(
        "$pre: inner_ridge must be finite and non-negative, got $inner_ridge"))

    ts = Float64.(collect(times))
    Xm = Matrix{Float64}(X)
    n = length(ts)
    n >= 3 || throw(ArgumentError(
        "$pre needs at least 3 covariate sample times (got $n)"))
    size(Xm, 1) == n || throw(ArgumentError(
        "$pre: X has $(size(Xm, 1)) rows but times has $n entries; the " *
        "covariate must be aligned to the sample times"))
    size(Xm, 2) >= 1 || throw(ArgumentError(
        "$pre: X needs at least one column (the driver covariate)"))
    all(isfinite, ts) || throw(ArgumentError("$pre: times must be finite"))
    all(isfinite, Xm) || throw(ArgumentError("$pre: X must be finite"))
    all(>(0), diff(ts)) || throw(ArgumentError(
        "$pre: times must be strictly increasing"))

    # Per-transformation structure. `inner_design` is fixed data in both
    # cases — the simplification the exogenous covariate buys us.
    inner_design, lags_out, anchor_out = if trans === :lagindex
        lags >= 1 || throw(ArgumentError(
            "$pre: trans=:lagindex needs lags ≥ 1 (got $lags)"))
        lags <= n || throw(ArgumentError(
            "$pre: lags = $lags exceeds the $n covariate samples; the lag " *
            "window cannot be longer than the record"))
        size(Xm, 2) == 1 || throw(ArgumentError(
            "$pre: trans=:lagindex takes a single covariate column (got " *
            "$(size(Xm, 2))); auxiliary columns are only meaningful for " *
            "trans=:expsm, where they modulate the inertia"))
        (1 <= anchor <= lags) || throw(ArgumentError(
            "$pre: anchor must lie in 1:$lags (lag 0 is index 1), got $anchor"))
        Δ = (ts[n] - ts[1]) / (n - 1)
        L = Matrix{Float64}(undef, n, lags)
        x1 = @view Xm[:, 1]
        for ℓ in 0:(lags - 1), i in 1:n
            L[i, ℓ + 1] = _tc_linterp(ts, x1, ts[i] - ℓ * Δ)
        end
        L, lags, anchor
    else
        lags == 0 || throw(ArgumentError(
            "$pre: lags is a :lagindex setting; leave it at 0 for " *
            "trans=:expsm, whose window length is set by the learned " *
            "inertia ω (got lags = $lags)"))
        # Same reasoning for `anchor`: :expsm has no scale-invariant
        # direction to pin (its inertia enters through a logistic link), so
        # silently dropping a user-supplied anchor would hide a modelling
        # misunderstanding.
        anchor == 1 || throw(ArgumentError(
            "$pre: anchor is a :lagindex setting; leave it at its default " *
            "for trans=:expsm, which has no scale-invariant loading " *
            "direction to anchor (got anchor = $anchor)"))
        hcat(ones(n), Xm[:, 2:end]), 0, nothing
    end

    dom = (-ξ, ξ)
    outer = if constraint == :none
        BSplineApproximator(name_s, dom, nknots; initial=initial)
    else
        ShapeConstrainedBSplineApproximator(name_s, dom, nknots, constraint;
                                            initial=initial)
    end

    var_floor = 1e-10 * max(maximum(abs, @view Xm[:, 1]), 1.0)^2

    TransformedCovariateApproximator(name_s, trans, ts, Xm, nknots, lags_out,
                                     anchor_out, constraint, ξ, ridge, dom,
                                     outer, inner_design, var_floor)
end

TransformedCovariateApproximator(name::Union{Symbol,String},
                                 times::AbstractVector,
                                 x::AbstractVector; kwargs...) =
    TransformedCovariateApproximator(name, times,
                                     reshape(Float64.(collect(x)), :, 1);
                                     kwargs...)

"""
Number of FREE inner parameters: the inertia linear predictor's width for
`:expsm`, and `lags − 1` (one anchored) for `:lagindex`.
"""
_tc_n_inner(a::TransformedCovariateApproximator) =
    a.trans === :lagindex ? a.lags - 1 : size(a.inner_design, 2)

nparams(a::TransformedCovariateApproximator) =
    _tc_n_inner(a) + nparams(a.outer)

function initial_params(a::TransformedCovariateApproximator)
    ni = _tc_n_inner(a)
    # :expsm starts at a = 0, i.e. the neutral inertia ω = 1/2; :lagindex at
    # equal weights (a = 1 everywhere including the anchor), matching the
    # equal-weight start SingleIndexApproximator uses for its loadings.
    inner = a.trans === :lagindex ? ones(ni) : zeros(ni)
    vcat(inner, initial_params(a.outer))
end

