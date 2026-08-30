# Core type definitions for PartiallySpecifiedModels.jl

# ─── Approximator types ────────────────────────────────────────────

"""
Abstract base type for unknown function approximators.
Subtypes must implement:
- `nparams(a)`: number of parameters
- `evaluate(a, params, x)`: evaluate at point x given parameter vector
- `penalty_matrix(a)`: smoothing penalty matrix (or nothing for unpenalized)
- `initial_params(a)`: default initial parameter vector
"""
abstract type AbstractApproximator end

# Shared domain sanity check: every approximator needs an oriented,
# non-degenerate domain. An inverted domain silently produced reversed
# knot/mesh grids downstream; a degenerate one (lo == hi) produced NaNs
# (e.g. COMONet divides by the domain width when normalizing inputs).
function _validate_domain(ctor::AbstractString, domain::Tuple{Float64, Float64})
    domain[2] > domain[1] || throw(ArgumentError(
        "$ctor: domain must satisfy hi > lo, got ($(domain[1]), $(domain[2]))"))
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
    TensorBSplineApproximator(name, domain_x, domain_y, nknots_x, nknots_y;
                              anisotropy=1.0, initial=nothing)

Bivariate tensor-product cubic-spline approximator for an unknown function
of TWO state variables — e.g. the predation response `g(P, N)` of Wood
(2001), "Partially specified ecological models", whose form depends on both
predator and prey densities. In the dynamics the fitted surface is a
two-argument callable, `du[1] = ... p.g(u[1], u[2]) ...`.

Coefficients are function values on the `nknots_x × nknots_y` grid of
evenly spaced knots over `domain_x × domain_y`, stored column-major as
`vec(C)` with `C[i, j] = f(x_i, y_j)` (the x index varies fastest).
Evaluation is the tensor product of the same univariate natural-cubic
interpolation `BSplineApproximator` uses, with linear extrapolation per
margin outside the domain.

The roughness penalty is the Kronecker sum
`S = I_ny ⊗ S_x + anisotropy · (S_y ⊗ I_nx)` of the univariate
`∫(f'')²` marginal penalties. The package's penalty interface carries a
single smoothing parameter λ per approximator, so LAML/GCV scale this whole
matrix by one λ; `anisotropy` sets the FIXED relative weight of y-roughness
against x-roughness (Wood's independent per-margin smoothing parameters
would require multi-λ penalty support, which is out of scope). The penalty
null space is the bilinear family `a + b·x + c·y + d·x·y`.

# Arguments
- `name`: symbol for the unknown function (accessed as `p.name(x, y)`)
- `domain_x`, `domain_y`: `(lo, hi)` ranges of the first and second argument
- `nknots_x`, `nknots_y`: knots per margin (≥ 3 each; `nparams = nknots_x · nknots_y`)
- `anisotropy`: positive relative weight of the y-roughness penalty (default 1.0)
- `initial`: optional initial surface — a two-argument function `(x, y) -> z`
  or a constant (default 0)

# Example
```julia
uf = TensorBSplineApproximator(:g, (0.0, 4.0), (0.0, 6.0), 6, 6)
predprey!(du, u, p, t) = begin
    du[1] = p.r * u[1] - p.g(u[1], u[2])
    du[2] = p.e * p.g(u[1], u[2]) - p.m * u[2]
end
```

Note: `confidence_band` and the bootstrap unknown-function bands are
univariate and do not support tensor surfaces.
"""
struct TensorBSplineApproximator <: AbstractApproximator
    name::Symbol
    domain_x::Tuple{Float64, Float64}
    domain_y::Tuple{Float64, Float64}
    nknots_x::Int
    nknots_y::Int
    anisotropy::Float64
    initial_func::Function
end

function TensorBSplineApproximator(name::Union{Symbol,String},
                                   domain_x::Tuple{Real,Real},
                                   domain_y::Tuple{Real,Real},
                                   nknots_x::Int,
                                   nknots_y::Int;
                                   anisotropy::Real=1.0,
                                   initial=nothing)
    name = Symbol(name)
    dx = (Float64(domain_x[1]), Float64(domain_x[2]))
    dy = (Float64(domain_y[1]), Float64(domain_y[2]))
    _validate_domain("TensorBSplineApproximator (x margin)", dx)
    _validate_domain("TensorBSplineApproximator (y margin)", dy)
    # Same floor as BSplineApproximator: cubic spline interpolation along
    # each margin needs at least 3 knots.
    nknots_x >= 3 || throw(ArgumentError(
        "TensorBSplineApproximator needs nknots_x ≥ 3 for cubic spline " *
        "interpolation (got $nknots_x)"))
    nknots_y >= 3 || throw(ArgumentError(
        "TensorBSplineApproximator needs nknots_y ≥ 3 for cubic spline " *
        "interpolation (got $nknots_y)"))
    aniso = Float64(anisotropy)
    (isfinite(aniso) && aniso > 0.0) || throw(ArgumentError(
        "TensorBSplineApproximator: anisotropy must be positive and finite, " *
        "got $anisotropy"))
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
    vec(Float64[a.initial_func(x, y) for x in xs, y in ys])
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
    d = domain === nothing ? nothing : (Float64(domain[1]), Float64(domain[2]))
    d === nothing || _validate_domain("NeuralApproximator", d)
    NeuralApproximator(Symbol(name), model, penalty_weight, d, rng_seed)
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
  Both preserve monotonicity/convexity guarantees. Use `:softplus` when smooth
  derivatives are needed.

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
    hs = hidden_sizes isa Vector ? Tuple(hidden_sizes...) : hidden_sizes
    length(hs) >= 1 || throw(ArgumentError("Need at least 1 hidden layer"))
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
no univariate band at all.

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
  rather than of `f(t)`
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

# ─── Likelihood types ──────────────────────────────────────────────

"""
Abstract type for exponential family likelihoods used in LAML/IRLS.

Each subtype must support:
- `log_likelihood(fam, y, mu, w)`: total log-likelihood
- `irls_weights(fam, y, mu, w)`: IRLS working weights
"""
abstract type AbstractLikelihood end

"""Gaussian likelihood with identity link (unknown σ² profiled out in REML)."""
struct Gaussian <: AbstractLikelihood end

"""Poisson likelihood, fitted on the response scale (identity link)."""
struct Poisson <: AbstractLikelihood end

"""Negative Binomial likelihood, fitted on the response scale (identity link)."""
struct NegativeBinomial <: AbstractLikelihood
    theta::Float64  # overdispersion: Var = μ + μ²/θ
end
NegativeBinomial() = NegativeBinomial(1.0)

"""
    TruncatedNormal(lower=0.0, sigma=1.0)

Truncated Normal likelihood for non-negative continuous data.  The distribution
is a Normal(μ, σ²) truncated to `[lower, ∞)`.  Scale parameter σ is fixed
(not profiled like the Gaussian σ²).

Useful for continuous measurements that are bounded below (e.g., population
densities, concentrations).

# Fields
- `lower::Float64`: lower truncation point (default 0.0)
- `sigma::Float64`: standard deviation (default 1.0)
"""
struct TruncatedNormal <: AbstractLikelihood
    lower::Float64
    sigma::Float64
end
TruncatedNormal(; lower::Float64=0.0, sigma::Float64=1.0) =
    TruncatedNormal(lower, sigma)

"""
    CustomLikelihood(loglik_scalar)

User-defined likelihood. `loglik_scalar(y, μ)` returns scalar log-likelihood
for one observation. IRLS weights derived via ForwardDiff.
"""
struct CustomLikelihood <: AbstractLikelihood
    loglik_scalar::Function
end

# ─── Algorithm types ───────────────────────────────────────────────

"""
    LAML(; maxiters=100, tol=1e-6, verbose=false, initial_lambda=nothing,
           warmup=3, sigma2_init=nothing, criterion=:working, jac=:fd)

Laplace Approximate Marginal Likelihood algorithm.
Equivalent to REML for Gaussian data.
Uses Fellner-Schall + Newton for smoothing parameter estimation.

# Keyword arguments
- `maxiters::Int=100`: maximum IRLS+LAML iterations
- `tol::Float64=1e-6`: convergence tolerance on penalized objective
- `verbose::Bool=false`: print iteration diagnostics
- `initial_lambda::Union{Nothing,Float64}=nothing`: initial smoothing parameter
  for all terms.  Default (`nothing`) uses the data-driven `θ = 1/tr(S)` per
  term (values below λ ≈ 4.5e-5 are floored there when LAML starts, to keep
  the criterion away from its flat tiny-λ region).  For strongly nonlinear
  problems, a higher value (e.g. `10.0`) combined with `warmup` helps the
  IRLS converge to a good basin before LAML refinement.
- `warmup::Int=3`: number of IRLS iterations to run with fixed smoothing before
  engaging LAML estimation.  Allows the coefficient estimates to stabilise
  before the smoothing parameters are adapted.  Increase for strongly
  nonlinear models (e.g. `warmup=10`).
- `sigma2_init::Union{Nothing,Float64}=nothing`: cap on the profiled σ² used in
  the Fellner-Schall smoothing update during the warmup phase.  When provided,
  σ² is clamped to `min(profiled_σ², sigma2_init)`, preventing the large
  residual variance from an early poor fit from driving oversmoothing.  After
  the warmup phase the cap is progressively relaxed.  Set to a value reflecting
  your prior belief about observation noise variance (e.g. `sigma2_init=25.0`
  for ±5 measurement error).  With `criterion=:laplace` and a count family,
  a cap below 1 transiently scales the warmup FS update below the unit
  dispersion the Newton phase optimizes — leave `sigma2_init` unset (or ≥ 1)
  there; it is a Gaussian-noise prior, not a count-model control.
- `criterion::Symbol=:working`: smoothing-parameter selection criterion for
  **non-Gaussian** likelihoods.
  - `:working` (default, previous behavior): PQL-flavored selection — the
    Fellner-Schall update is calibrated by the Pearson dispersion φ̂ of the
    working model (floored at 1), and the Newton refinement of the marginal
    likelihood is skipped.
  - `:laplace`: the true Laplace-approximate marginal likelihood of the actual
    family (Wood 2011; Wood, Pya & Säfken 2016). At the penalized MLE β̂_λ the
    criterion is `ℓ(β̂) − ½β̂'S_λβ̂ + ½log|S_λ|₊ − ½log|H + S_λ| + (Mp/2)log 2π`
    with `H = J'W̃J` the GLM-style **expected** (Fisher) Hessian at the fitted
    mean, and it is optimized by the generalized Fellner-Schall update of
    Wood & Fasiolo (2017) — i.e. FS with the family working weights and **unit
    dispersion** — followed by Newton refinement of the same criterion.
    Prefer `:laplace` for genuinely non-Gaussian data, especially counts with
    low means (Poisson μ ≲ 10), where the Gaussian working approximation
    behind `:working` is at its worst. Dispersion caveats: the criterion
    assumes the family's dispersion is FIXED — exact for `Poisson`, and
    conditional on the supplied `theta` for `NegativeBinomial` and on
    (`lower`, `sigma`) for `TruncatedNormal` (these are NOT re-estimated;
    a misspecified fixed dispersion biases λ̂). `CustomLikelihood` is
    rejected with an error: its `loglik_scalar` may be an unnormalized
    kernel with no declared dispersion, so the criterion value and its FS
    calibration would be off by unknown amounts — use `:working` there.
    For `Gaussian` likelihoods `:laplace` reduces exactly to the current
    profiled-REML criterion (identical code path), so results are identical
    to `:working`.
- `jac::Symbol=:fd`: backend for the working-model Jacobian of the
  predictions w.r.t. the coefficients (`compute_jacobian!`).
  - `:fd` (default, historical behavior, byte-identical): adaptive central
    finite differences — 2·n_p full model solves per IRLS iteration, with
    truncation + integration-noise error in each column.
  - `:forwarddiff`: forward-mode AD (`ForwardDiff.jacobian`) straight
    through the ODE/map/DDE solve — exact to solver precision and cheaper
    (n_p/chunk Dual solves instead of 2·n_p). The chunked configuration is
    built once per solve. If the Dual-valued solve fails numerically or
    returns non-finite entries, that iteration falls back to the `:fd`
    path (with a `@debug` note) rather than erroring — mirroring the FD
    path's own per-column degradation. Works for continuous ODEs (in-place
    and out-of-place dynamics), discrete maps, `u0`-as-function problems,
    and DDEs (`MethodOfSteps` is Dual-safe; a failure there also lands on
    the per-iteration FD fallback). Requires the dynamics and any custom
    evaluators to be eltype-generic (all built-in approximators are).

# Convergence info
`sol.convergence` is a NamedTuple `(V_beta, sigma2, converged, iterations,
reason, laml_failures, criterion, laml, stationarity,
smoothing_advanced)`: the posterior covariance and σ̂² used
by [`confidence_band`](@ref), plus the standard honest-convergence keys (see
[`PSMSolution`](@ref)), `laml_failures::Int` (the number of iterations in
which the LAML smoothing-parameter update failed and θ was kept),
`criterion::Symbol` (which selection criterion ran), and `laml::Float64` (the
LAML criterion value V at the returned fit — profiled REML for Gaussian, the
full Laplace criterion above for other families; a MAXIMIZED quantity, `NaN`
when no penalized term exists or the evaluation fails).

## Stationarity diagnostics: what `converged` does NOT tell you

`converged` is a STABILITY test — the penalized objective and the data loss
stopped changing. A fit stops changing for several distinct reasons and
`converged` cannot tell them apart: it is `true` for a genuine optimum, for a
search that stalled because a non-smooth model made the finite-difference
Jacobian too noisy to build an accepted step from, and for a fit whose
smoothing parameters never moved at all. The two keys below separate those
cases. **They are additive diagnostics; they do not change `converged`,
`reason`, or any fitted quantity, and they are present on every LAML solution,
converged or not.**

- `smoothing_advanced::Bool`: whether λ̂ ever moved off its initialization
  (`initial_lambda`, or the data-driven `1/tr(S)` default). `false` means the
  reported λ̂ **is** the initial value — smoothing selection never took
  effect, because every Fellner-Schall proposal was rejected by the step
  accept test or because it never ran (no penalized term, or
  `maxiters ≤ warmup`). It is `false` for a model with no penalized term.
  This is a plain fact about the search, not a statistical judgement — the
  comparison against the initialization uses a 1e-10 RELATIVE tolerance,
  which guards against pure round-off rather than being a calibrated
  cutoff, so there is no magnitude to tune. A `false` here with
  `converged == true` is the strongest available signal that the reported λ̂
  carries no information from the data. It does NOT mean the fit is wrong —
  the initialization can happen to be reasonable — only that nothing chose
  it. Across the package's own test suite this is `false` for 17 of 163 LAML
  solves, 8 of which nonetheless report `converged == true`. It is genuinely
  complementary to `stationarity` rather than a proxy for it: one of those 17
  is an unpenalized model whose residual is `0.0` by convention, and the
  other 16 span residuals from 0.016 to 8.7, so a fit can sit at a perfectly
  ordinary residual and still never have moved its λ̂.

- `stationarity::Float64`: how far the returned λ̂ is from a stationary point
  of the LAML criterion, as
  `maxₖ |∂V/∂ρₖ| / (½·rank(Sₖ))` with `ρ = log λ`, evaluated at the returned
  `(β̂, λ̂)`. The normalization makes it dimensionless and comparable across
  bases and resolutions — and, for Gaussian data, across data scales; under
  a non-Gaussian family the `λₖβ̂'Sₖβ̂` term carries no `σ̂²` divisor and so
  does scale with the data. The `½·rank(Sₖ)` and
  `½·tr(H⁻¹λₖSₖ)` terms of `∂V/∂ρₖ` are both bounded by `½·rank(Sₖ)`, so the
  ratio measures the gradient against the natural scale of the block, and `0`
  is exactly stationary. It is **not** bounded above by 1 — the
  `λₖβ̂'Sₖβ̂/σ̂²` term has no upper bound, and the largest value observed
  across this package's test suite is 8.7. `0.0` when the model has no
  penalized term (nothing to be stationary in); `NaN` when the criterion or
  its gradient could not be evaluated.

  It does NOT measure whether the fit is *good*, or whether λ̂ is at a global
  rather than a local optimum — it is a first-order condition on the
  smoothing parameters, evaluated at the returned β̂, not a gradient through
  the ODE solve. It is nonzero when λ̂ is off its conditional optimum AND
  when β̂ is off the penalized least-squares solution at λ̂ (at a true fixed
  point of the IRLS loop those coincide, so it tests both).

  One further caveat for NON-GAUSSIAN families under the default
  `criterion = :working`: the residual is the gradient of the FULL LAPLACE
  criterion (that is what `laml_gradient` returns for those families), while
  the Fellner-Schall update was calibrated on the Gaussian working model. A
  nonzero value can therefore reflect that mismatch rather than a stalled
  search — the same caveat the `laml` key already carries. Under
  `criterion = :laplace` the two agree and the residual is unambiguous.

### How to read `stationarity`

**There is deliberately no `stationary::Bool`, because the observed
distribution does not support one.** Measured on all 163 LAML solves the test
suite performs, the residual is an unbroken continuum rather than two
clusters. Its quantiles are p25 = 1.6e-7, p50 = 8.5e-6, p75 = 1.6e-2,
p90 = 0.29, max = 8.7, and across the whole decision-relevant region (1e-3
to 3) the largest ratio between consecutive sorted values is 1.52 below 1
and 2.98 across the whole region — nothing resembling the
orders-of-magnitude separation a threshold would need, so there is
no gap anywhere a threshold could sit. A cutoff at 0.1 would have flagged 23
of those 163 solves (14.1%), splitting a continuum of otherwise ordinary
`:converged_tol` fits.

So treat it as a magnitude, calibrated against your own problem class rather
than an absolute scale. For orientation, on this suite 95 of 163 solves sit
below 1e-4; a deliberately non-smooth fixture (a discrete map applying `floor`
to a coefficient-dependent quantity, which makes the default `jac=:fd`
Jacobian noise) reads 0.29, against 9.3e-7 for the identical map, basis and
data with the `floor` removed — a factor of 3.2e5 between two fits that
`converged` describes identically. A residual orders of magnitude larger than
comparable fits of yours is the signal; a particular number is not. The usual
causes of a large value are a non-smooth dynamics function
(`floor`/`clamp`/`round` on a coefficient-dependent quantity) — for which
`jac=:forwarddiff` is the fix — or too few `maxiters`.

**One regime where it is ill-conditioned, and must not be read at all:** for
`Gaussian` data the gradient contains `−½λₖβ̂'Sₖβ̂/σ̂²` with `σ̂²` the profiled
REML scale. On a fit that interpolates its data — noiseless synthetic
fixtures, most obviously — `σ̂²` underflows toward zero and that term becomes
a ratio of two near-zero quantities, so the residual carries no information
about the search and can read large on a fit that is exact to round-off.
Measured: the same noiseless growth fixture reads 0.91 under both `jac=:fd`
and `jac=:forwarddiff` — with λ̂ agreeing to machine precision, EDF to ~1e-4
relative, `σ̂² ≈ 2e-19` and a data loss of ~3e-18 — an alarming-looking
residual on an essentially exact fit. This is less a defect in the
residual than a true statement about the fit — with no residual variance,
REML does not identify λ and λ̂ is set by round-off — but it does mean
`stationarity` is uninformative there. Judge it only on fits with meaningful
residual variance.

`converged && smoothing_advanced`, plus a `stationarity` in line with
comparable fits, is the combination that means what users generally read
`converged` alone to mean.
"""
struct LAML
    maxiters::Int
    tol::Float64
    verbose::Bool
    initial_lambda::Union{Nothing,Float64}
    warmup::Int
    sigma2_init::Union{Nothing,Float64}
    criterion::Symbol
    jac::Symbol
end

function LAML(; maxiters::Int=100, tol::Float64=1e-6, verbose::Bool=false,
                initial_lambda::Union{Nothing,Float64}=nothing,
                warmup::Int=3,
                sigma2_init::Union{Nothing,Float64}=nothing,
                criterion::Symbol=:working,
                jac::Symbol=:fd)
    criterion in (:working, :laplace) ||
        throw(ArgumentError("LAML: criterion must be :working or :laplace " *
                            "(got $(repr(criterion)))"))
    jac in (:fd, :forwarddiff) ||
        throw(ArgumentError("LAML: jac must be :fd or :forwarddiff " *
                            "(got $(repr(jac)))"))
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, jac)
end

# Backward-compatible positional constructor (pre-jac arity).
LAML(maxiters::Int, tol::Float64, verbose::Bool,
     initial_lambda::Union{Nothing,Float64}, warmup::Int,
     sigma2_init::Union{Nothing,Float64}, criterion::Symbol) =
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, :fd)

"""
    CollocationLAML(; kwargs...)

Collocation-based LAML solver using the generalized profiling / parameter
cascading approach of Ramsay et al. (2007).

Instead of integrating the ODE at each step (as `LAML` does), this solver
represents the **state trajectories as free parameters** and penalizes
deviation from the ODE.  A continuation schedule gradually increases the
ODE compliance penalty `λ_ode`, transitioning from a data-fitting problem
(flexible states) to a model-constrained problem (states satisfy ODE).

This approach is much more robust for **highly nonlinear or oscillatory
models** (e.g., Lotka–Volterra) where the standard IRLS linearization
fails because the ODE trajectory is extremely sensitive to parameter
changes.  See Fasiolo, Pya & Wood (2016), Statistical Science 31(1).

The data-fidelity block honors `prob.likelihood` (all built-in families
and `CustomLikelihood`) via PIRLS working weights on the identity link,
with a deviance-scale objective and a Pearson-dispersion Fellner–Schall
scale mirroring `LAML`'s default `:working` criterion; the ODE-compliance
rows stay Gaussian in the derivative residuals. `Gaussian` data follow
the historical pure-least-squares path unchanged. (Before this, the
likelihood was silently ignored — every family was fitted by Gaussian
least squares.)

# Keyword arguments
- `maxiters::Int=50`: IRLS iterations per continuation level
- `tol::Float64=1e-6`: convergence tolerance
- `verbose::Bool=false`: print diagnostics
- `lambda_ode_start::Float64=0.01`: initial ODE compliance penalty
- `lambda_ode_end::Float64=1e4`: final ODE compliance penalty
- `n_continuation::Int=8`: number of log-spaced continuation levels
- `sigma2_init::Union{Nothing,Float64}=nothing`: σ² cap for Fellner-Schall
- `jac::Symbol=:fd`: backend for the differentiated parts of the
  collocation Jacobian — the pointwise state Jacobian ∂F/∂x, the
  coefficient block ∂F/∂β, and the Fellner–Schall EDF Jacobian of the
  simulated predictions.
  - `:fd` (default, historical behavior, byte-identical): finite
    differences.
  - `:forwarddiff`: forward-mode AD. The failure-sentinel convention is
    preserved EXACTLY (see the comment block in `collocation_solver.jl`):
    the residual at a failed collocation point still holds the large
    `_COLLOC_FAIL_SENTINEL`, the sentinel value itself is NEVER part of
    the differentiated computation (failed points are skipped inside the
    Dual sweep), and the Jacobian rows/columns at failed points are forced
    to zero exactly as the FD path forces them — large residual, zero
    Jacobian. If a Dual-valued sweep fails outright it falls back to the
    FD path for that evaluation (`@debug` note). One behavioral nuance,
    strictly an improvement: at a point whose BASE evaluation succeeds,
    forward mode returns the exact derivative and never perturbs the
    state, so FD's defensive zero column for "perturbation crossed a
    domain boundary" cannot arise (that zero existed only to protect FD
    from its own perturbation).

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The collocation objective
    calls the dynamics through the 4-argument ODE signature and has no way
    to supply the delayed history, so a DDE would silently be fitted as
    though it had no delays. Use [`LAML`](@ref) for DDEs.

# Convergence info
`sol.convergence` is a NamedTuple `(ode_compliance, lambda_ode_final,
converged, iterations, reason, iterations_total)`. `converged`/`reason`
describe the final continuation level's inner loop, `iterations` counts its
inner iterations, and `iterations_total` accumulates inner iterations across
all continuation levels (see [`PSMSolution`](@ref) for the key taxonomy).
"""
struct CollocationLAML
    maxiters::Int
    tol::Float64
    verbose::Bool
    lambda_ode_start::Float64
    lambda_ode_end::Float64
    n_continuation::Int
    sigma2_init::Union{Nothing,Float64}
    jac::Symbol
end

function CollocationLAML(; maxiters::Int=50, tol::Float64=1e-6,
                           verbose::Bool=false,
                           lambda_ode_start::Float64=0.01,
                           lambda_ode_end::Float64=1e4,
                           n_continuation::Int=8,
                           sigma2_init::Union{Nothing,Float64}=nothing,
                           jac::Symbol=:fd)
    jac in (:fd, :forwarddiff) ||
        throw(ArgumentError("CollocationLAML: jac must be :fd or " *
                            ":forwarddiff (got $(repr(jac)))"))
    CollocationLAML(maxiters, tol, verbose, lambda_ode_start, lambda_ode_end,
                    n_continuation, sigma2_init, jac)
end

# Backward-compatible positional constructor (pre-jac arity).
CollocationLAML(maxiters::Int, tol::Float64, verbose::Bool,
                lambda_ode_start::Float64, lambda_ode_end::Float64,
                n_continuation::Int,
                sigma2_init::Union{Nothing,Float64}) =
    CollocationLAML(maxiters, tol, verbose, lambda_ode_start, lambda_ode_end,
                    n_continuation, sigma2_init, :fd)

"""
    GradientMatching(; maxiters=500, tol=1e-6, verbose=false, sigma2_init=nothing)

Two-step gradient matching solver inspired by NODEBNGM (Bonnaffé et al. 2023):

1. Smooth observed data with cubic splines to obtain ŷ(t) and dŷ/dt
2. Fit unknown function parameters by matching ODE derivatives:
   minimize ||dŷ/dt - f(ŷ, p, t)||² + penalty

Avoids ODE integration entirely, making it far more robust for neural network
approximators where the IRLS linearization of the ODE trajectory is poor.

Uses Gauss-Newton for penalized approximators (B-spline, GP) and Adam optimizer
for unpenalized approximators (neural networks).

Requires that all state variables are observed (no latent states).

# Arguments
- `maxiters::Int=500`: maximum iterations (Adam needs more than GN)
- `tol::Float64=1e-6`: convergence tolerance
- `verbose::Bool=false`: print iteration details
- `sigma2_init::Union{Nothing,Float64}=nothing`: σ² cap for Fellner-Schall

!!! note "σ² for the Fellner-Schall update"
    The residual variance driving the smoothing-parameter update is
    `σ̂² = RSS / #{residuals with nonzero weight}`, not `RSS / #residuals`.
    Unobserved-state rows are zero-weighted; counting them biased σ̂² low
    and over-smoothed the unknown functions.
"""
struct GradientMatching
    maxiters::Int
    tol::Float64
    verbose::Bool
    sigma2_init::Union{Nothing,Float64}
    lr::Float64
    refine_iters::Int
end

GradientMatching(; maxiters::Int=500, tol::Float64=1e-6, verbose::Bool=false,
                   sigma2_init::Union{Nothing,Float64}=nothing,
                   lr::Float64=0.01, refine_iters::Int=0) =
    GradientMatching(maxiters, tol, verbose, sigma2_init, lr, refine_iters)

"""
    AdamSolver(; maxiters=300, lr=0.01, verbose=false, loss=:auto,
                 penalty_weight=0.0, autodiff=true, sensealg=nothing)

Adam optimizer that trains unknown function parameters through ODE integration.

For neural networks: uses a ForwardDiff-compatible MLP evaluator (bypassing Lux)
so that exact gradients can be computed through the ODE solve. This matches the
approach used in Universal Differential Equations (UDEs).

For B-splines and GPs: also supported, uses the standard evaluators.

The loss function computes `solve(ODEProblem(...), solver)` at each step and
compares with data. ForwardDiff computes exact gradients through the ODE solve.

Unlike `LAML`/`GCVSolver`, this solver performs no automatic smoothing-parameter
selection: with the default `penalty_weight = 0` the fit is *unpenalized*
(spline coefficients are free). Set `penalty_weight > 0` to add the fixed
quadratic roughness penalty `penalty_weight · Σₖ βₖ' Sₖ βₖ` to the loss.

# Arguments
- `maxiters::Int=300`: maximum Adam iterations
- `lr::Float64=0.01`: learning rate (with cosine annealing)
- `verbose::Bool=false`: print iteration details
- `loss::Symbol=:auto`: loss function; `:auto` selects `:mse` for `Gaussian`
  and `:poisson` for `Poisson` likelihoods (other families are not supported
  by this solver — use `LAML`). Explicit `:mse`/`:poisson` override with a
  warning on mismatch.
- `penalty_weight::Float64=0.0`: fixed weight of the quadratic smoothing
  penalty added to the loss (0 disables)
- `autodiff::Bool=true`: use ForwardDiff (true) or finite differences (false);
  ignored when `sensealg` is set
- `sensealg=nothing`: opt-in adjoint-sensitivity gradient backend. `nothing`
  (default) keeps the ForwardDiff path. Set to `:auto` (a robust default,
  `InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))`) or to a concrete
  SciMLSensitivity continuous adjoint algorithm (`InterpolatingAdjoint(...)`
  is the validated choice; `GaussAdjoint`/`QuadratureAdjoint` were observed
  to mis-differentiate spline-parameterized dynamics — validate against
  ForwardDiff before trusting them) to compute gradients via adjoint
  sensitivities instead. **Requires `using SciMLSensitivity`**
  (the package extension implements this path; without it, `solve` throws an
  informative error). Adjoints make the gradient cost roughly independent of
  the parameter count, which pays off for `NeuralApproximator` models with
  many weights; ForwardDiff is usually faster below a few dozen parameters.
  Only continuous ODE problems are supported (no discrete maps, no DDEs);
  the loss families are the same as the ForwardDiff path (`:mse`,
  `:poisson`).

# Convergence info
`sol.convergence` is a NamedTuple `(optimizer, method, converged, iterations,
reason, final_grad_norm, backend)` with the standard honest-convergence keys
(see [`PSMSolution`](@ref)) plus `final_grad_norm::Float64`, the Euclidean
norm of the last computed gradient, and `backend`, the gradient backend that
ran (`:forwarddiff`, `:finitediff`, or `:adjoint`). `reason == :plateau` is
only reported while the cosine-annealed learning rate is still above 5% of
the base `lr` — a plateau that appears merely because the schedule has driven
the step size to zero is reported as `:maxiters`, not convergence.
"""
struct AdamSolver
    maxiters::Int
    lr::Float64
    verbose::Bool
    loss::Symbol
    penalty_weight::Float64
    autodiff::Bool
    sensealg::Any
end

AdamSolver(; maxiters::Int=300, lr::Float64=0.01, verbose::Bool=false,
             loss::Symbol=:auto, penalty_weight::Float64=0.0,
             autodiff::Bool=true, sensealg=nothing) =
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff, sensealg)

# Backward-compatible positional constructor (pre-sensealg arity).
AdamSolver(maxiters::Int, lr::Float64, verbose::Bool, loss::Symbol,
           penalty_weight::Float64, autodiff::Bool) =
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff, nothing)

"""
    MultipleShootingSolver(; n_intervals=10, maxiters_inner=100, maxiters_outer=20,
                             rho_init=10.0, rho_max=1e6, loss=:auto,
                             penalty_weight=0.0, verbose=false)

Multiple shooting solver for training neural differential equations, following
Turan & Jäschke (2021). Partitions the time span into intervals with shooting
variables at boundaries. Uses an augmented Lagrangian (L-BFGS inner subproblems) to enforce continuity.

Advantages over single shooting (AdamSolver):
- Better initial fits: shooting variables initialized from data
- Avoids "flattened trajectory" failure mode for oscillatory systems
- Shorter integration intervals improve gradient quality

# Arguments
- `n_intervals::Int=10`: number of shooting intervals
- `maxiters_inner::Int=100`: L-BFGS iterations per augmented Lagrangian step
- `maxiters_outer::Int=20`: augmented Lagrangian outer iterations
- `rho_init::Float64=10.0`: initial penalty parameter for shooting constraints
- `rho_max::Float64=1e6`: maximum penalty parameter
- `loss::Symbol=:auto`: data-fit loss. `:auto` follows `prob.likelihood`
  (Gaussian → `:mse` weighted SSE, Poisson → `:poisson` weighted negative
  log-likelihood kernel; other families error). Explicit `:mse`/`:poisson`
  are honored with a warning on mismatch, matching `AdamSolver`.
- `penalty_weight::Float64=0.0`: fixed quadratic smoothing penalty
  `penalty_weight · Σₖ βₖ'Sₖβₖ` added to the training objective (0 = no
  penalty; there is no smoothing-parameter selection here)
- `verbose::Bool=false`: print iteration details

# Convergence info
`sol.convergence` is a NamedTuple `(optimizer, method, n_intervals, converged,
iterations, reason, max_gap, rho_final)` with the standard honest-convergence
keys (see [`PSMSolution`](@ref); `iterations` counts outer augmented-Lagrangian
iterations) plus `max_gap::Float64`, the final maximum shooting-gap magnitude,
and `rho_final::Float64`, the final penalty parameter.
"""
struct MultipleShootingSolver
    n_intervals::Int
    maxiters_inner::Int
    maxiters_outer::Int
    rho_init::Float64
    rho_max::Float64
    loss::Symbol
    penalty_weight::Float64
    verbose::Bool
end

MultipleShootingSolver(; n_intervals::Int=10, maxiters_inner::Int=100,
                         maxiters_outer::Int=20,
                         rho_init::Float64=10.0, rho_max::Float64=1e6,
                         loss::Symbol=:auto, penalty_weight::Float64=0.0,
                         verbose::Bool=false) =
    MultipleShootingSolver(n_intervals, maxiters_inner, maxiters_outer,
                           rho_init, rho_max, loss, penalty_weight, verbose)

"""
    AdaptiveGradientMatching(; maxiters=200, verbose=false, gamma_init=1.0,
                               fit_gamma=true, kernel=:rbf, n_samples=0,
                               n_chains=10, rng_seed=nothing)

Adaptive Gradient Matching solver using the product-of-experts objective of
Dondelinger et al. (2013). Gaussian processes smooth the data and supply
gradient estimates with uncertainty; the ODE-predicted gradients are matched
under the product-of-experts likelihood.

Two modes:

- **MAP (default, `n_samples=0`)**: GP hyperparameters from a
  marginal-likelihood grid search, latent states fixed at the GP posterior
  mean, and (β, log γ) optimized once by L-BFGS. Fast; no posterior samples.
- **Population MCMC (`n_samples > 0`)**: the tempered population-MCMC
  sampler of Dondelinger et al./deGradInfer. `n_chains` chains at
  temperatures `t_c = ((c−1)/(C−1))⁵` jointly sample the latent states X,
  the parameters β, and the mismatch variances γ by adaptive blockwise
  random-walk Metropolis with exchange moves between adjacent chains; the
  ODE-mismatch likelihood is raised to `t_c`, interpolating from pure GP
  regression (t=0) to the fully coupled model (t=1). The cold chain
  (t=1) provides the posterior: reported unknown functions use the
  posterior-mean β, and `sol.convergence.beta_samples` /
  `gamma_samples` hold the draws. GP kernel hyperparameters stay fixed
  at their grid-search values (a documented simplification — the paper
  also samples them). Continuous-time problems only.

The (log-)likelihood term for each state k is:
    L_k = -0.5 (f_k - m_k)ᵀ (A_k + γ_k I)⁻¹ (f_k - m_k) - 0.5 log|A_k + γ_k I|

where:
- f_k = ODE-predicted gradients for state k
- m_k = GP gradient mean ('K K⁻¹ x_k, conditioned on the states)
- A_k = GP gradient covariance = K** - K*(K + σ²I)⁻¹K*ᵀ
- γ_k = mismatch parameter controlling ODE-GP coupling

# Arguments
- `maxiters::Int=200`: maximum L-BFGS iterations (MAP mode); MCMC sweeps
  are `2 n_samples` (half burn-in) in sampling mode
- `verbose::Bool=false`: print iteration details
- `gamma_init::Float64=1.0`: initial mismatch parameter (per state)
- `fit_gamma::Bool=true`: optimize γ or keep fixed (MAP mode)
- `kernel::Symbol=:rbf`: GP kernel (:rbf, :matern32)
- `n_samples::Int=0`: cold-chain posterior draws; 0 = MAP mode
- `n_chains::Int=10`: number of tempered chains (sampling mode; ≥ 2)
- `rng_seed`: seed for the sampler (default `nothing` = non-reproducible)

!!! note "GP centering"
    Each state's GP prior is centered: observed states on their data mean,
    unobserved states on their initial condition, and every `x` entering
    the GP quadratic forms and the gradient mean is the CENTERED state
    `x − m_k`. A zero-mean GP prior on uncentered data pulls the
    trajectory toward zero, which biased the fitted unknown functions on
    data far from the origin.
"""
struct AdaptiveGradientMatching
    maxiters::Int
    verbose::Bool
    gamma_init::Float64
    fit_gamma::Bool
    kernel::Symbol
    n_samples::Int
    n_chains::Int
    rng_seed::Union{Nothing, Int}
end

AdaptiveGradientMatching(; maxiters::Int=200, verbose::Bool=false,
                           gamma_init::Float64=1.0, fit_gamma::Bool=true,
                           kernel::Symbol=:rbf, n_samples::Int=0,
                           n_chains::Int=10,
                           rng_seed::Union{Nothing, Int}=nothing) =
    AdaptiveGradientMatching(maxiters, verbose, gamma_init, fit_gamma, kernel,
                             n_samples, n_chains, rng_seed)

"""
    RodeoSolver

Probabilistic ODE solver (rodeo) for parameter inference.

Uses Kalman filtering with an integrated Brownian motion prior to
approximate the ODE solution and compute an approximate marginal likelihood.

# Fields
- `n_steps`: number of solver discretization steps (default: 200)
- `n_deriv`: number of derivatives in IBM prior (default: 3)
- `sigma`: IBM scale parameters (one per state variable, or nothing for auto)
- `obs_var`: observation noise variance (or nothing for auto)
- `method`: likelihood approximation (`:basic` or `:fenrir`)
- `interrogate`: interrogation method (`:kramer` or `:schober`)
- `sqrt_filter`: run the Kalman recursions in square-root (Cholesky/QR)
  form (default: false). Numerically stabler at high `n_deriv` and small
  steps, where the standard covariance recursion can lose
  positive-semidefiniteness; with `false` the standard covariance
  recursion is used unchanged.
- `maxiters`: max L-BFGS iterations (default: 200)
- `verbose`: print progress (default: false)
"""
struct RodeoSolver
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    method::Symbol
    interrogate::Symbol
    maxiters::Int
    verbose::Bool
    sqrt_filter::Bool
end

function RodeoSolver(; n_steps::Int=200, n_deriv::Int=3,
                       sigma::Union{Nothing, Vector{Float64}}=nothing,
                       obs_var::Union{Nothing, Float64}=nothing,
                       method::Symbol=:basic,
                       interrogate::Symbol=:kramer,
                       maxiters::Int=200, verbose::Bool=false,
                       sqrt_filter::Bool=false)
    method in (:basic, :fenrir) ||
        throw(ArgumentError("RodeoSolver: method must be :basic or :fenrir " *
                            "(got :$method)"))
    interrogate in (:kramer, :schober) ||
        throw(ArgumentError("RodeoSolver: interrogate must be :kramer or " *
                            ":schober (got :$interrogate)"))
    # The IBM prior needs at least value + first derivative per state;
    # n_deriv=1 BoundsErrors inside the Kalman filter selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("RodeoSolver: n_deriv must be ≥ 2 (got $n_deriv)"))
    RodeoSolver(n_steps, n_deriv, sigma, obs_var, method, interrogate, maxiters,
                verbose, sqrt_filter)
end

"""
    MCMCSolver(; n_samples=1000, n_warmup=500, n_chains=1, target_accept=0.8,
                 prior_scale=1.0, obs_sigma=nothing, sample_smoothing=false,
                 rng_seed=nothing, verbose=false)

Full Bayesian inference via Hamiltonian Monte Carlo (NUTS).
Uses LogDensityProblems.jl + AdvancedHMC.jl.

# Arguments
- `n_samples`: number of posterior samples per chain (after warmup)
- `n_warmup`: number of warmup/adaptation steps
- `n_chains`: number of independent chains
- `target_accept`: target acceptance rate for NUTS adaptation (0.6–0.95)
- `prior_scale`: scale for Gaussian prior on parameters (larger = weaker prior).
  When penalty matrices exist (B-spline, GP), uses the penalty; otherwise N(0, prior_scale²).
- `obs_sigma`: observation noise std dev (Gaussian likelihoods only; errors
  otherwise). If `nothing`, sampled as a parameter for Gaussian data; non-
  Gaussian families sample no σ (their dispersion is fixed in the family
  object). The data term follows `prob.likelihood` in all cases.
- `sample_smoothing`: if `true`, jointly sample log(λ) for each smooth term
  with a weakly informative N(log(λ_init), 2²) hyperprior. This gives wider,
  more honest credible intervals for the unknown functions. Default: `false`.
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress
"""
struct MCMCSolver
    n_samples::Int
    n_warmup::Int
    n_chains::Int
    target_accept::Float64
    prior_scale::Float64
    obs_sigma::Union{Nothing, Float64}
    sample_smoothing::Bool
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

MCMCSolver(; n_samples::Int=1000, n_warmup::Int=500, n_chains::Int=1,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             obs_sigma::Union{Nothing, Float64}=nothing,
             sample_smoothing::Bool=false,
             rng_seed::Union{Nothing, Int}=nothing,
             verbose::Bool=false) =
    MCMCSolver(n_samples, n_warmup, n_chains, target_accept, prior_scale,
               obs_sigma, sample_smoothing, rng_seed, verbose)

"""
    MagiSolver(; n_samples=1000, n_warmup=500, n_gridpoints=200,
                 sigma=nothing, obs_var=nothing, target_accept=0.8,
                 prior_scale=1.0, preoptimize=true, verbose=false)

Manifold-constrained Gaussian process inference (MAGI) for ODE systems.
Gaussian likelihoods only (the manifold-constrained posterior assumes
Gaussian observation noise); non-Gaussian `prob.likelihood` errors.

MAGI places a Matérn-3/2 Gaussian-process prior on each state (the MAGI
paper uses a generalized Matérn kernel with ν ≈ 2.01; with ν = 3/2 the
implied derivative process is rougher — mean-square continuous but not
mean-square differentiable) and constrains
the GP-implied derivative to the ODE vector field through the conditional
derivative covariance `K* = ''K − 'K C⁻¹ ('K)ᵀ`. The state values on the
discretization grid are sampled jointly with the unknown-function
parameters θ via NUTS/HMC (the manifold-constrained posterior).

**Key advantage**: Handles partially observed systems naturally (unobserved
state components are inferred through the ODE constraint).

Returns an `MCMCChains.Chains` object with posterior samples.

# Fields
- `n_samples`: number of posterior samples after warmup
- `n_warmup`: warmup/adaptation iterations
- `n_gridpoints`: number of discretization grid points for the GP/manifold
  constraint
- `sigma`: per-state observation noise standard deviations (one entry per
  state component). When supplied, these fixed SDs are used in the data
  term instead of auto-estimation; mutually exclusive with `obs_var`.
- `obs_var`: observation noise variance shared across components;
  `nothing` (default) estimates it from the data via a local-linear
  residual estimator (a fixed scale-blind default distorted the posterior
  whenever the data's units differed from O(0.1))
- `target_accept`: NUTS target acceptance rate
- `prior_scale`: scale for Gaussian prior on parameters
- `preoptimize`: run a short MAP pre-optimization to initialize the
  sampler (default `true`)
- `verbose`: print progress

!!! note "Changed"
    The former `n_deriv` field was removed: it was never read anywhere
    (the Matérn-3/2 GP prior has no derivative-order setting), so keeping
    it only suggested a control that did not exist.

# References
- Yang, Wong & Kou (2021) PNAS 118(15): "Inference of dynamic systems
  from noisy and sparse data via manifold-constrained Gaussian processes"
"""
struct MagiSolver
    n_samples::Int
    n_warmup::Int
    n_gridpoints::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    target_accept::Float64
    prior_scale::Float64
    preoptimize::Bool
    verbose::Bool
end

MagiSolver(; n_samples::Int=1000, n_warmup::Int=500,
             n_gridpoints::Int=200,
             sigma::Union{Nothing, Vector{Float64}}=nothing,
             obs_var::Union{Nothing, Float64}=nothing,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             preoptimize::Bool=true, verbose::Bool=false) =
    MagiSolver(n_samples, n_warmup, n_gridpoints, sigma, obs_var,
               target_accept, prior_scale, preoptimize, verbose)

# ─── BNG solver (Bonnaffé et al. 2023) ────────────────────────────

"""
    BNGSolver

Ensemble Bayesian gradient matching (Bonnaffé & Coulson 2023). Avoids
ODE integration entirely: observed series are smoothed with GCV splines,
and unknown-function parameters are fit by matching the smoothed
derivatives under a variance-marginalized log-posterior
`(n/2)log(1 + SSR/2) + (|β|/2)log(1 + Σ(β/prior_sd)²/2)` (the paper's
"Bayesian regularisation": observation and prior variances are
marginalized, not supplied).

Uncertainty comes from a `k_obs × k_proc` fit ensemble: `k_obs`
observation resamples (residual bootstrap of the smoother; the first is
the original data) times `k_proc` restarts from perturbed
initialisations. Reported unknown functions are ensemble means;
`sol.convergence.ensemble_std[name]` gives the pointwise ensemble
standard deviation.

# Fields
- `k_obs`: observation-ensemble size (default 10)
- `k_proc`: process fits per observation ensemble (default 3)
- `prior_sd`: prior scale of the marginalized Gaussian parameter prior
  (default 10.0)
- `maxiters`: Adam iterations per ensemble member (default 2000)
- `lr`: Adam learning rate (default 0.01)
- `lambda_smooth`: extra smoothing penalty on approximator coefficients
  (default 1.0)
- `rng_seed`: seed for the bootstrap and restarts (default `nothing` =
  non-reproducible)
- `verbose`: print progress

# Convergence info
`sol.convergence` carries the standard honest-convergence keys (see
[`PSMSolution`](@ref)): `converged`/`reason` report the BEST ensemble
member's outcome (`:plateau` when its loss stagnated, `:maxiters`
otherwise), `iterations` is the total Adam iterations summed over all
ensemble members, and `member_converged::Vector{Bool}` records each
member's plateau flag (alongside the existing `n_ensemble`,
`member_losses`, `member_weights`, `ensemble_std` keys).

# References
- Bonnaffé & Coulson (2023), "Fast fitting of neural ordinary
  differential equations by Bayesian neural gradient matching to infer
  ecological interactions from time-series data", Methods Ecol Evol 14
"""
struct BNGSolver
    k_obs::Int
    k_proc::Int
    prior_sd::Float64
    maxiters::Int
    lr::Float64
    lambda_smooth::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

BNGSolver(; k_obs::Int=10, k_proc::Int=3, prior_sd::Float64=10.0,
            maxiters::Int=2000, lr::Float64=0.01,
            lambda_smooth::Float64=1.0,
            rng_seed::Union{Nothing, Int}=nothing, verbose::Bool=false) =
    BNGSolver(k_obs, k_proc, prior_sd, maxiters, lr, lambda_smooth,
              rng_seed, verbose)

# ─── Dalton solver (Wu & Lysy 2024) ───────────────────────────────

"""
    DaltonSolver

Data-Adaptive Likelihood with Transformed Observations (DALTON) solver.

Extends the probabilistic ODE approach (RODEO) with a data-adaptive
marginal likelihood: p(Y|Z) = p(Y,Z)/p(Z), computed via two Kalman
filter passes — one joint (ODE + observations) and one marginal (ODE only).

# Fields
- `n_steps`: number of discretization steps (default 200)
- `n_deriv`: IBM prior derivative order (default 3)
- `sigma`: IBM scale parameters (nothing = auto-estimate)
- `obs_var`: observation noise variance. `nothing` (the default)
  auto-estimates it from the data at solve time as 1% of the mean
  per-column data variance (the same estimator `RodeoSolver` and
  `PseudoMarginalSolver` use). A fixed numeric value is honored exactly;
  note a fixed value must be chosen relative to the data scale — a value
  appropriate for data of order 1 badly misfits data of order 1000.
- `interrogate`: interrogation method `:kramer` or `:schober` (default `:kramer`)
- `sqrt_filter`: run both DALTON Kalman passes in square-root
  (Cholesky/QR) form (default: false). Numerically stabler at high
  `n_deriv` and small steps; the quasi-MLE diffusion calibration is
  computed from mathematically equivalent quantities. With `false` the
  standard covariance recursion is used unchanged.
- `maxiters`: optimization iterations (default 200)
- `verbose`: print progress
"""
struct DaltonSolver
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    interrogate::Symbol
    maxiters::Int
    verbose::Bool
    sqrt_filter::Bool
end

function DaltonSolver(; n_steps::Int=200, n_deriv::Int=3,
                        sigma::Union{Nothing, Vector{Float64}}=nothing,
                        obs_var::Union{Nothing, Float64}=nothing,
                        interrogate::Symbol=:kramer,
                        maxiters::Int=200, verbose::Bool=false,
                        sqrt_filter::Bool=false)
    interrogate in (:kramer, :schober) ||
        throw(ArgumentError("DaltonSolver: interrogate must be :kramer or " *
                            ":schober (got :$interrogate)"))
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("DaltonSolver: n_deriv must be ≥ 2 (got $n_deriv)"))
    DaltonSolver(n_steps, n_deriv, sigma, obs_var, interrogate, maxiters,
                 verbose, sqrt_filter)
end

# ─── Pseudo-marginal solver (Chkrebtii et al. 2016) ───────────────

"""
    PseudoMarginalSolver

Pseudo-marginal MCMC using a probabilistic ODE solver for likelihood estimation.

Uses RODEO/fenrir as an inner solver to compute an unbiased estimate of the
marginal likelihood p(Y|θ), then samples from the posterior p(θ|Y) via
adaptive random-walk Metropolis (the proposal scale is tuned toward
`target_accept` during warmup).

# Fields
- `n_samples`: number of posterior samples (default 1000)
- `n_warmup`: warmup/adaptation samples (default 500)
- `n_steps`: discretization steps for inner probabilistic solver (default 200)
- `n_deriv`: IBM prior derivative order (default 3)
- `sigma`: IBM scale parameters (nothing = auto-estimate)
- `obs_var`: observation noise variance (default `nothing` = estimate it
  from the data alongside the other nuisance quantities)
- `target_accept`: target acceptance rate for the adaptive random-walk
  Metropolis proposal-scale adaptation (default 0.8)
- `prior_scale`: prior VARIANCE on parameters (default 1.0). Used directly
  as the variance of the Gaussian smoothing prior
  `exp(−½ βₖ'Sₖβₖ / prior_scale)` and of the weak ridge
  `exp(−½ ‖β‖² / (100·prior_scale))` — it is not squared.
- `inner_method`: likelihood estimator — `:ffbs` (default; unbiased FFBS
  Monte-Carlo average, giving genuine pseudo-marginal MCMC), `:fenrir`
  (deterministic Fenrir evidence; the chain is then plain adaptive RWM on
  an approximate likelihood), or `:dalton` (deterministic DALTON
  data-adaptive likelihood, likewise plain RWM)
- `rng_seed`: seed for the chain's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `sqrt_filter`: run the inner probabilistic-ODE Kalman recursions in
  square-root (Cholesky/QR) form (default: false). Numerically stabler
  at high `n_deriv` and small steps. Supported by all three
  `inner_method`s — under `:ffbs` the backward draws use the
  PSD-by-construction covariance factors, so the jitter escalation of
  the standard sampler is never needed. With `false` the standard
  covariance recursion is used unchanged.
- `verbose`: print progress
"""
struct PseudoMarginalSolver
    n_samples::Int
    n_warmup::Int
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    target_accept::Float64
    prior_scale::Float64
    inner_method::Symbol
    initial_params::Union{Nothing, Vector{Float64}}
    rng_seed::Union{Nothing, Int}
    verbose::Bool
    sqrt_filter::Bool
end

function PseudoMarginalSolver(; n_samples::Int=1000, n_warmup::Int=500,
                                n_steps::Int=200, n_deriv::Int=3,
                                sigma::Union{Nothing, Vector{Float64}}=nothing,
                                obs_var::Union{Nothing, Float64}=nothing,
                                target_accept::Float64=0.8, prior_scale::Float64=1.0,
                                inner_method::Symbol=:ffbs,
                                initial_params::Union{Nothing, Vector{Float64}}=nothing,
                                rng_seed::Union{Nothing, Int}=nothing,
                                verbose::Bool=false,
                                sqrt_filter::Bool=false)
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("PseudoMarginalSolver: n_deriv must be ≥ 2 " *
                            "(got $n_deriv)"))
    PseudoMarginalSolver(n_samples, n_warmup, n_steps, n_deriv, sigma, obs_var,
                          target_accept, prior_scale, inner_method,
                          initial_params, rng_seed, verbose, sqrt_filter)
end

# ─── GCV solver (Wood 2001 / ddefit504) ────────────────────────────

"""
    GCVSolver

Cross-validation solver for smoothing parameter selection.

Uses GCV score minimization (via golden-section search on log(λ)) as an
alternative to LAML/REML. Simpler and faster than LAML, but typically
produces slightly less smooth estimates. With `criterion=:ncv` the GCV
score is replaced by neighbourhood cross-validation (NCV; Wood 2024,
arXiv:2404.16490): each point is predicted from a penalized fit that
omits a temporal *neighbourhood* around it, not just the point itself.
Because dynamical-systems residuals are typically short-range
autocorrelated, ordinary GCV/LOOCV sees the correlated neighbours make
the left-out point look predictable and undersmooths badly; NCV removes
exactly those neighbours and is the recommended criterion whenever
residual autocorrelation is suspected (check `residual_diagnostics`).

# Fields
- `n_grid`: number of grid points for initial λ search (default 50)
- `maxiters`: maximum IRLS iterations (default 50)
- `tol`: convergence tolerance (default 1e-6)
- `gamma`: GCV inflation factor (default 1.4, >1 guards against
  under-smoothing; `gamma=1.0` reproduces the classical unmodified GCV of
  Wood (2001) / ddefit). **Ignored when `criterion=:ncv`**: NCV needs no
  inflation factor — leaving out the neighbourhood is itself the guard
  against the undersmoothing that `gamma` patches over (Wood 2024 uses
  γ has no role in the base NCV criterion as implemented here).
- `criterion`: `:gcv` (default, classical GCV) or `:ncv`
  (neighbourhood cross-validation)
- `ncv_width`: half-width of the NCV deletion neighbourhood, in
  time-index steps (default 2). For observation i of a component, the
  neighbourhood δ(i) is every usable observation of the SAME component
  whose time index is within `ncv_width` of i's (i itself included).
  Must be ≥ 0; `ncv_width=0` gives exact leave-one-out CV. Choose it
  to cover the residual correlation length (e.g. the lag at which the
  residual ACF dies off). Ignored when `criterion=:gcv`.
- `search`: `:direct` (default) evaluates every λ in the GCV search with
  its own O(p³) penalized solve — the historical behavior, byte-identical
  to before this option existed. `:reuse` is the classical ddefit /
  Demmler–Reinsch speed trick (gcv.c `EasySmooth`/`EScv`): one expensive
  whitening + eigendecomposition per IRLS iteration, after which every λ
  in the grid + golden-section search costs only O(np) diagonal
  arithmetic. It is a pure reformulation, not an approximation — scores
  match `:direct` to ~1e-12 relative everywhere λ selection actually
  happens (the stability ridge of the direct scorer is replicated
  exactly; only at the essentially-infinite-smoothing extreme,
  λ ≳ 1e15 with rank-deficient penalties, does agreement degrade to
  ~1e-6, where the direct path's own trA/β are conditioning-limited
  too), so the selected λ̂ agrees within the
  search tolerance, and the returned fit always goes through the same
  final PCLS solve as `:direct`. When `A = J'WJ` (plus any fixed
  penalties) is near-singular the fast path falls back to the direct
  scorer for that IRLS iteration (debug-level log note). With multiple
  approximators the coordinate-descent λ search reuses one
  decomposition per coordinate per sweep. Only `criterion=:gcv` is
  supported: the NCV neighbourhood downdates need `A(λ)⁻¹` at every λ,
  which is genuinely incompatible with the one-decomposition trick, so
  `search=:reuse` with `criterion=:ncv` is rejected at construction.
- `jac`: backend for the working-model Jacobian of predictions w.r.t. the
  coefficients — `:fd` (default; adaptive central finite differences,
  historical behavior, byte-identical) or `:forwarddiff` (forward-mode AD
  through the ODE/map/DDE solve; exact to solver precision, one chunked
  Dual sweep instead of 2·n_p perturbed solves, config built once per
  solve; falls back to `:fd` for an iteration — `@debug` note — whenever
  the Dual-valued solve fails or returns non-finite entries). See the
  `LAML` docstring for the full description; the two solvers share
  `compute_jacobian!`.
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple
`(converged, iterations, reason, criterion, gcv, ncv)` with the standard
honest-convergence keys (see [`PSMSolution`](@ref)) plus:
- `criterion::Symbol`: which selection criterion ran (`:gcv` or `:ncv`)
- `gcv::Float64`: the final GCV score when `criterion=:gcv`; `NaN`
  otherwise (also NaN when no smooth terms are present)
- `ncv::Float64`: the final NCV score when `criterion=:ncv`; `NaN`
  otherwise
"""
struct GCVSolver
    n_grid::Int
    maxiters::Int
    tol::Float64
    gamma::Float64
    criterion::Symbol
    ncv_width::Int
    search::Symbol
    verbose::Bool
    jac::Symbol
end

function GCVSolver(; n_grid::Int=50, maxiters::Int=50, tol::Float64=1e-6,
                     gamma::Float64=1.4, criterion::Symbol=:gcv,
                     ncv_width::Int=2, search::Symbol=:direct,
                     verbose::Bool=false, jac::Symbol=:fd)
    criterion in (:gcv, :ncv) ||
        throw(ArgumentError("GCVSolver: criterion must be :gcv or :ncv " *
                            "(got $(repr(criterion)))"))
    ncv_width >= 0 ||
        throw(ArgumentError("GCVSolver: ncv_width must be ≥ 0 " *
                            "(got $ncv_width; 0 = leave-one-out CV)"))
    search in (:direct, :reuse) ||
        throw(ArgumentError("GCVSolver: search must be :direct or :reuse " *
                            "(got $(repr(search)))"))
    !(search === :reuse && criterion === :ncv) ||
        throw(ArgumentError("GCVSolver: search=:reuse supports only " *
                            "criterion=:gcv — the NCV neighbourhood downdates " *
                            "need A(λ)⁻¹ at every λ, which is incompatible " *
                            "with the one-decomposition reuse trick; use " *
                            "search=:direct with criterion=:ncv"))
    jac in (:fd, :forwarddiff) ||
        throw(ArgumentError("GCVSolver: jac must be :fd or :forwarddiff " *
                            "(got $(repr(jac)))"))
    GCVSolver(n_grid, maxiters, tol, gamma, criterion, ncv_width, search,
              verbose, jac)
end

# Backward-compatible positional constructor (pre-jac arity).
GCVSolver(n_grid::Int, maxiters::Int, tol::Float64, gamma::Float64,
          criterion::Symbol, ncv_width::Int, search::Symbol,
          verbose::Bool) =
    GCVSolver(n_grid, maxiters, tol, gamma, criterion, ncv_width, search,
              verbose, :fd)

# ─── Two-stage solver (Wood 2001 / deGradInfer) ───────────────────

"""
    TwoStageSolver

Two-stage smooth-then-differentiate solver (simplest baseline).

Stage 1: Smooth each observed state independently via spline + GCV/penalty.
Stage 2: Numerically differentiate smoothed curves, then match ODE RHS
          to derivatives via least squares to infer unknown function params.

This is the original approach from Wood (2001) / deGradInfer (Macdonald & Husmeier 2015).

# Fields
- `n_basis_smooth`: upper limit on the number of B-spline coefficients used
  by the stage-1 smoother; the actual basis size is
  `clamp(n − 2, 4, n_basis_smooth)` for `n` data points (default 15).

  !!! note "Changed"
      This field was previously dead: it was documented (default 20) but
      never read — the smoother always capped the basis at 15. It is now
      wired through; the default was set to 15 so default fits are
      byte-identical to before.
- `lambda_smooth`: smoothing penalty for initial data fit (default 1.0)
- `maxiters`: max iterations for parameter matching (default 1000)
- `lr`: learning rate for Adam optimization in matching step (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
with the standard honest-convergence keys (see [`PSMSolution`](@ref));
`converged=true` with `reason=:plateau` when the matching loss stagnated
over a 30-iteration window, `reason=:maxiters` otherwise.
"""
struct TwoStageSolver
    n_basis_smooth::Int
    lambda_smooth::Float64
    maxiters::Int
    lr::Float64
    verbose::Bool
end

function TwoStageSolver(; n_basis_smooth::Int=15, lambda_smooth::Float64=1.0,
                          maxiters::Int=1000, lr::Float64=0.01, verbose::Bool=false)
    n_basis_smooth >= 4 ||
        throw(ArgumentError("TwoStageSolver: n_basis_smooth must be ≥ 4 " *
                            "(cubic B-spline basis; got $n_basis_smooth)"))
    TwoStageSolver(n_basis_smooth, lambda_smooth, maxiters, lr, verbose)
end

# ─── Derivative-free solver (stochastic + NelderMead) ──────────────

"""
    DerivativeFreeSolver

Derivative-free optimization solver using NelderMead or particle swarm.

The objective includes the quadratic roughness penalty
`0.5 · penalty_weight · Σₖ βₖ'Sₖβₖ` (note the ½ factor; default
`penalty_weight=1.0`; set `penalty_weight=0` for an unpenalized fit —
there is no smoothing-parameter selection here).

Useful as a robust fallback when gradient-based methods fail (non-smooth
objectives, stiff dynamics, poor conditioning). Uses simulation-based
loss without requiring autodiff through ODE solves.

# Fields
- `method`: optimization method — `:nelder_mead` or `:particle_swarm` (default `:nelder_mead`)
- `maxiters`: maximum function evaluations (default 10000)
- `n_particles`: particle count for swarm methods (default 20)
- `loss`: loss type (default `:auto`). `:auto` follows `prob.likelihood`
  — Gaussian → `:mse` (weighted SSE), any other family → `:likelihood`
  (the family's negative log-likelihood). Explicit `:mse` or
  `:likelihood` is honored regardless of the likelihood family.
- `penalty_weight`: weight of the roughness penalty above (default 1.0)
- `verbose`: print progress
"""
struct DerivativeFreeSolver
    method::Symbol
    maxiters::Int
    n_particles::Int
    loss::Symbol
    penalty_weight::Float64
    verbose::Bool
end

DerivativeFreeSolver(; method::Symbol=:nelder_mead, maxiters::Int=10000,
                       n_particles::Int=20, loss::Symbol=:auto,
                       penalty_weight::Float64=1.0,
                       verbose::Bool=false) =
    DerivativeFreeSolver(method, maxiters, n_particles, loss, penalty_weight,
                         verbose)

# ─── Variational inference solver ──────────────────────────────────

"""
    VariationalSolver

Variational inference solver using mean-field Gaussian approximation.

Approximates the posterior p(θ|Y) with a factored Gaussian q(θ) = ∏ N(μᵢ, σᵢ²)
by maximizing the evidence lower bound (ELBO). Much faster than MCMC while
providing uncertainty estimates.

# Fields
- `maxiters`: max ELBO optimization iterations (default 2000)
- `lr`: learning rate for Adam on ELBO (default 0.01)
- `n_elbo_samples`: Monte Carlo samples for ELBO gradient (default 10)
- `prior_scale`: prior VARIANCE on parameters (default 1.0). Used directly
  as the variance of the Gaussian smoothing prior — the prior precision is
  `Λ = ridge·I + Σₖ Sₖ / prior_scale` — it is not squared.
- `obs_noise_var`: Gaussian observation-noise variance (default `nothing`
  = estimate from the data). Gaussian likelihoods only; errors otherwise.
  Non-Gaussian families use their own pointwise log-likelihood in the ELBO.
- `rng_seed`: seed for the ELBO Monte-Carlo draws (default `42`, which
  reproduces the historical hard-coded behavior; `nothing` = fresh
  non-reproducible stream). Does not touch the global RNG.
- `verbose`: print progress

!!! note "Reported EDF"
    `sol.edf` is the Laplace effective degrees of freedom at the
    variational mean, `tr((H + Λ)⁻¹H)` with the Gauss–Newton information
    `H = JᵀWJ/σ²_obs` — a real measure of model complexity that responds
    to the smoothing level. It is `NaN` (honestly missing) when the model
    cannot be simulated at the variational mean or the linear system is
    singular; it is never a fabricated constant.
"""
struct VariationalSolver
    maxiters::Int
    lr::Float64
    n_elbo_samples::Int
    prior_scale::Float64
    obs_noise_var::Union{Nothing, Float64}
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

# rng_seed defaults to 42, NOT nothing: the solver historically hard-coded
# Xoshiro(42) for its ELBO Monte-Carlo draws, and the default must keep
# reproducing that. `nothing` opts into a fresh non-reproducible stream.
VariationalSolver(; maxiters::Int=2000, lr::Float64=0.01,
                    n_elbo_samples::Int=10, prior_scale::Float64=1.0,
                    obs_noise_var::Union{Nothing, Float64}=nothing,
                    rng_seed::Union{Nothing, Int}=42,
                    verbose::Bool=false) =
    VariationalSolver(maxiters, lr, n_elbo_samples, prior_scale, obs_noise_var,
                      rng_seed, verbose)

# ─── ABC solver (Approximate Bayesian Computation) ─────────────────

"""
    ABCSolver

Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

Likelihood-free inference using simulation-based rejection sampling with
adaptive tolerance scheduling. Works for any simulator, including those
where the likelihood is intractable.

**Prior** (`prior` keyword): `:smoothness` (default) is a Gaussian GMRF
built from the approximators' roughness penalties, centered on their
initial values with overall spread `prior_scale` — matching the prior
structure of `MCMCSolver`/`VariationalSolver`, so wiggly coefficient
vectors are a-priori discouraged. `:box` restores the legacy uniform box
of half-width `prior_scale` (posterior then depends strongly on the
initialization; treat as approximate).

# Fields
- `n_particles`: number of ABC particles (default 500)
- `n_generations`: number of SMC generations (default 10)
- `summary_fn`: summary statistic function, or `:auto` for MSE-based (default `:auto`)
- `prior`: `:smoothness` (default, GMRF) or `:box` (legacy uniform)
- `prior_scale`: prior spread — GMRF scale, or box half-width (default 2.0)
- `quantile_eps`: quantile for tolerance schedule (default 0.5)
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress
"""
struct ABCSolver
    n_particles::Int
    n_generations::Int
    summary_fn::Union{Symbol, Function}
    prior::Symbol
    prior_scale::Float64
    quantile_eps::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

ABCSolver(; n_particles::Int=500, n_generations::Int=10,
            summary_fn::Union{Symbol, Function}=:auto,
            prior::Symbol=:smoothness,
            prior_scale::Float64=2.0, quantile_eps::Float64=0.5,
            rng_seed::Union{Nothing, Int}=nothing,
            verbose::Bool=false) =
    ABCSolver(n_particles, n_generations, summary_fn, prior, prior_scale,
              quantile_eps, rng_seed, verbose)

# ─── Integral matching solver (Dattner & Klaassen 2015) ────────────

"""
    IntegralMatchingSolver

Integral matching estimator: integrates both sides of the ODE so that
parameter estimation requires neither derivative estimation nor repeated
ODE integration.  Compares smoothed trajectory increments ŷ(tᵢ)−ŷ(t₁)
against the cumulative trapezoidal integral of f(ŷ(s),p,s).

# Fields
- `lambda_smooth`: smoothing penalty weight (default 1.0)
- `maxiters`: Adam iterations (default 1000)
- `lr`: learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
with the standard honest-convergence keys (see [`PSMSolution`](@ref));
`converged=true` with `reason=:plateau` when the matching loss stagnated
over a 30-iteration window, `reason=:maxiters` otherwise.

# References
- Dattner & Klaassen (2015), EJS 9(2), 1939–1973
- R package `simode` (Yaari & Dattner)
"""
struct IntegralMatchingSolver
    lambda_smooth::Float64
    maxiters::Int
    lr::Float64
    verbose::Bool
end

IntegralMatchingSolver(; lambda_smooth::Float64=1.0, maxiters::Int=1000,
                         lr::Float64=0.01, verbose::Bool=false) =
    IntegralMatchingSolver(lambda_smooth, maxiters, lr, verbose)

# ─── Profile likelihood solver ─────────────────────────────────────

"""
    ProfileLikelihoodSolver

Profile likelihood for identifiability analysis and confidence intervals.

For each unknown-function parameter βⱼ, sweeps βⱼ over a grid while
optimising all other parameters (β₋ⱼ) at each grid point.  Returns the
profile likelihood curve Lₚ(βⱼ) and a likelihood-ratio CI.

The profile is taken through the penalized objective at the fitted
smoothing parameters λ̂ (a penalized spline is not identified through its
raw RSS); the statistic ΔPenSS/σ̂² is referenced against χ²₁, and CI
endpoints are interpolated between grid points. Nuisance coefficients
are re-optimised at each grid point by a long Nelder–Mead run,
warm-started from the previous grid point. Gaussian likelihoods only.

# Fields
- `n_profile_points`: grid points per parameter (default 20); the exact
  MLE value is inserted, so the evaluated grid may have one extra point
- `ci_level`: confidence level for LR-based CI (default 0.95)
- `param_indices`: which parameter indices to profile (default `nothing` = all)
- `verbose`: print progress
- `base_alg`: the `LAML` fit the profile is taken around (default
  `LAML(verbose=false)`). The returned solution reports THIS fit's
  convergence keys, so a deliberately truncated base — e.g.
  `LAML(maxiters=1)` — surfaces as `converged = false` rather than being
  laundered into a converged profile-likelihood result.

# References
- Simpson & Maclaren (2023), PLOS Comp Biol
"""
struct ProfileLikelihoodSolver
    n_profile_points::Int
    ci_level::Float64
    param_indices::Union{Nothing, Vector{Int}}
    verbose::Bool
    base_alg::LAML
end

ProfileLikelihoodSolver(; n_profile_points::Int=20, ci_level::Float64=0.95,
                           param_indices::Union{Nothing, Vector{Int}}=nothing,
                           verbose::Bool=false,
                           base_alg::LAML=LAML(verbose=false)) =
    ProfileLikelihoodSolver(n_profile_points, ci_level, param_indices, verbose,
                            base_alg)

# ─── Ensemble Kalman inversion solver ──────────────────────────────

"""
    EnsembleKalmanSolver

Ensemble Kalman Inversion (EKI) for batch parameter estimation.

Uses an ensemble of parameter particles, propagates each through the
forward model (ODE simulation), and updates via the Kalman gain
K = Cov(θ, G(θ)) [Cov(G(θ), G(θ)) + Γ]⁻¹.  Iterates until the ensemble
collapses around the MAP estimate — mean particle std below 1e-3 of its
initial value, reported as `converged = true, reason =
:ensemble_collapse` — or until `n_iterations` is exhausted, reported
honestly as `converged = false, reason = :maxiters`.

Discrete problems are propagated with `simulate_discrete` (unit steps
over `tspan`, data times snapped to the nearest integer step), so EKI
sees the same trajectory as every other solver.

# Fields
- `n_ensemble`: ensemble size (default 50)
- `n_iterations`: EKI iterations (default 30)
- `noise_scale`: observation noise std for regularisation (default 0.1).
  This is a FIXED scale, not auto-estimated from the data: set it
  relative to the magnitude of your observations (e.g. roughly the
  expected noise std). The default suits data of order 1; for data of
  order 1000 use a correspondingly larger value.
- `rng_seed`: seed for the solver's own RNG stream (default `42`, which
  reproduces the historical hard-coded behavior; `nothing` = fresh
  non-reproducible stream). Does not touch the global RNG.
- `verbose`: print progress

# References
- Iglesias, Law & Stuart (2013), Inverse Problems
- Schillings & Stuart (2017), SIAM J Numer Anal
"""
struct EnsembleKalmanSolver
    n_ensemble::Int
    n_iterations::Int
    noise_scale::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

# rng_seed defaults to 42, NOT nothing: the solver historically hard-coded
# Xoshiro(42) for ensemble initialization and perturbations, and the default
# must keep reproducing that. `nothing` opts into a fresh stream.
EnsembleKalmanSolver(; n_ensemble::Int=50, n_iterations::Int=30,
                       noise_scale::Float64=0.1,
                       rng_seed::Union{Nothing, Int}=42,
                       verbose::Bool=false) =
    EnsembleKalmanSolver(n_ensemble, n_iterations, noise_scale, rng_seed, verbose)

# ─── ODIN solver (ODE-Informed regression) ─────────────────────────

"""
    ODINSolver

ODE-informed regression (Wenk, Abbati et al. 2020). GP hyperparameters
are first estimated per observed state by marginal likelihood, then the
states `X` and the unknown-function parameters `θ` are optimised
*jointly* against the ODIN risk

    R(X, θ) = Σ_k [ ‖y_k − x_k‖²/σ_{n,k}² + x̃_kᵀ K_k⁻¹ x̃_k
                    + (f_k(X,θ) − D_k x̃_k)ᵀ A_k⁻¹ (f_k(X,θ) − D_k x̃_k) ],

where `x̃_k` is the centered state, `D_k = 'K K⁻¹` maps states to their GP
conditional-mean derivative, and `A_k = ''K − 'K K⁻¹ 'Kᵀ + γI` is the GP
conditional derivative covariance — so the ODE mismatch is trusted only
in directions the GP actually determines. Unobserved states are included
as free variables (GP prior + ODE terms, no data term), so partially
observed systems are supported.

# Fields
- `maxiters`: outer iterations; 20 Adam steps each (default 50)
- `gp_lengthscale`, `gp_variance`: `nothing` (default) estimates the RBF
  hyperparameters per state by GP marginal likelihood. Supplying BOTH
  fixes them for all states (with noise assumed at `0.01 * gp_variance`);
  supplying only one is an error
- `ode_weight`: extra multiplier on the ODE-mismatch term (default 1.0 =
  the natural Mahalanobis weighting of the paper)
- `lr`: Adam learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method,
gp_hyperparams)` with the standard honest-convergence keys (see
[`PSMSolution`](@ref)). `iterations` counts the joint Adam steps actually
performed (the budget is `20 * maxiters` steps); `converged=true` with
`reason=:plateau` when the risk stagnated over a 30-step window while the
cosine-annealed learning rate was still above 5% of the base `lr`,
`reason=:maxiters` otherwise.

`sol.data_loss` is the WEIGHTED residual sum of squares over usable cells
(`weighted_data_loss`), the same convention as every other solver — so
masked (`data_weights == 0`) and down-weighted observations contribute to
the reported misfit exactly in proportion to their weight.

# References
- Wenk, Abbati et al. (2020), AAAI — ODIN
- Wenk et al. (2019), AISTATS — FGPGM
"""
struct ODINSolver
    maxiters::Int
    gp_lengthscale::Union{Nothing, Float64}
    gp_variance::Union{Nothing, Float64}
    ode_weight::Float64
    lr::Float64
    verbose::Bool
end

ODINSolver(; maxiters::Int=50,
             gp_lengthscale::Union{Nothing, Real}=nothing,
             gp_variance::Union{Nothing, Real}=nothing,
             ode_weight::Float64=1.0,
             lr::Float64=0.01, verbose::Bool=false) =
    ODINSolver(maxiters,
               gp_lengthscale === nothing ? nothing : Float64(gp_lengthscale),
               gp_variance === nothing ? nothing : Float64(gp_variance),
               ode_weight, lr, verbose)

# ─── FGPGM solver ──────────────────────────────────────────────────

"""
    FGPGMSolver(; n_samples=1000, n_warmup=500, gamma=0.1,
                  gp_lengthscale=nothing, gp_variance=nothing,
                  obs_sigma=nothing, prior_scale=1.0, target_accept=0.234,
                  rng_seed=nothing, verbose=false)

Fast Gaussian process based gradient matching (FGPGM; Wenk, Gotovos,
Bauer, Gorbach, Krause & Buhmann, AISTATS 2019). The latent states `X`
(at the observation times, as in the paper) and the unknown-function
parameters `θ` are sampled JOINTLY from one product-of-experts density

    p(X, θ | y) ∝ Π_k [ N(y_k | x_k, σ_{n,k}² I) · N(x_k | m_k, C_k)
                        · N(f_k(X, θ) | D_k x̃_k, A_k + γI) ] · p(θ)

with `D_k = 'C_k C_k⁻¹` and `A_k = ''C_k − 'C_k C_k⁻¹ 'C_kᵀ` (the GP
conditional derivative map and covariance), by single-chain adaptive
Metropolis-within-Gibbs. GP hyperparameters are fixed beforehand by
per-state marginal likelihood — the paper's key simplification.

**When to prefer FGPGM**: over [`AdaptiveGradientMatching`](@ref)'s
population MCMC when you want genuine posterior samples without the
temperature ladder and γ sampling (one chain, fixed hyperparameters —
cheaper and better-behaved); over [`ODINSolver`](@ref) (the same
authors' later pure-optimisation formulation) when you need posterior
uncertainty rather than a point estimate; over [`MagiSolver`](@ref)
when the observation times are dense enough to carry the latent states
(MAGI discretizes on a finer grid and uses NUTS — heavier per sweep).
Gaussian likelihoods only; continuous-time problems only;
`NeuralApproximator` is rejected (use [`AdamSolver`](@ref)).

# Fields
- `n_samples`: retained posterior draws after warmup (default 1000)
- `n_warmup`: warmup sweeps; proposal-scale adaptation happens here ONLY
  and the scales are frozen afterwards (default 500)
- `gamma`: the paper's model-mismatch variance γ added to the ODE
  expert's covariance `A_k + γI` (default 0.1). The effective slack also
  includes `σ_n²/ℓ²`, the derivative-scale observation noise, as in
  `ODINSolver`. Must be positive
- `gp_lengthscale`, `gp_variance`: `nothing` (default) estimates the RBF
  hyperparameters per state by GP marginal likelihood. Supplying BOTH
  fixes them for all states (with noise assumed at `0.01 * gp_variance`);
  supplying only one is an error (ODIN convention)
- `obs_sigma`: observation-noise SD overriding the marginal-likelihood
  estimate σ_n in the data expert (default `nothing` = per-state estimate)
- `prior_scale`: scale of the θ smoothing prior (larger = weaker), as in
  `MCMCSolver`/`MagiSolver` (default 1.0)
- `target_accept`: per-block acceptance rate targeted by the warmup
  adaptation (default 0.234, the classic random-walk optimum)
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress

# Convergence info
`sol.convergence` carries `chains` (`MCMCChains.Chains` of the θ draws),
`beta_samples`, `state_mean`, `gp_hyperparams`, post-warmup
`accept_rates` per block, and the honest keys
`converged=false`/`reason=:maxiters`/`iterations` — a fixed-budget
sampler has no stopping criterion (MagiSolver convention), so judge the
run by the R̂/ESS of `convergence.chains`.

# References
- Wenk, Gotovos, Bauer, Gorbach, Krause & Buhmann (2019), "Fast
  Gaussian process based gradient matching for parameter identification
  in systems of nonlinear ODEs", AISTATS 89:1351-1360.
"""
struct FGPGMSolver
    n_samples::Int
    n_warmup::Int
    gamma::Float64
    gp_lengthscale::Union{Nothing, Float64}
    gp_variance::Union{Nothing, Float64}
    obs_sigma::Union{Nothing, Float64}
    prior_scale::Float64
    target_accept::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

function FGPGMSolver(; n_samples::Int=1000, n_warmup::Int=500,
                       gamma::Real=0.1,
                       gp_lengthscale::Union{Nothing, Real}=nothing,
                       gp_variance::Union{Nothing, Real}=nothing,
                       obs_sigma::Union{Nothing, Real}=nothing,
                       prior_scale::Real=1.0,
                       target_accept::Real=0.234,
                       rng_seed::Union{Nothing, Int}=nothing,
                       verbose::Bool=false)
    n_samples >= 1 ||
        throw(ArgumentError("FGPGMSolver: n_samples must be ≥ 1 (got $n_samples)"))
    n_warmup >= 0 ||
        throw(ArgumentError("FGPGMSolver: n_warmup must be ≥ 0 (got $n_warmup)"))
    gamma > 0 ||
        throw(ArgumentError("FGPGMSolver: gamma must be positive (got $gamma) " *
                            "— it is the variance of the ODE-mismatch slack " *
                            "in N(f_k | D_k x̃_k, A_k + γI)"))
    0 < target_accept < 1 ||
        throw(ArgumentError("FGPGMSolver: target_accept must lie in (0, 1) " *
                            "(got $target_accept)"))
    prior_scale > 0 ||
        throw(ArgumentError("FGPGMSolver: prior_scale must be positive " *
                            "(got $prior_scale)"))
    obs_sigma === nothing || obs_sigma > 0 ||
        throw(ArgumentError("FGPGMSolver: obs_sigma must be positive " *
                            "(got $obs_sigma)"))
    (gp_lengthscale === nothing) == (gp_variance === nothing) ||
        throw(ArgumentError(
            "FGPGMSolver: supply BOTH gp_lengthscale and gp_variance to fix " *
            "the GP hyperparameters, or neither to estimate them per state " *
            "by marginal likelihood."))
    FGPGMSolver(n_samples, n_warmup, Float64(gamma),
                gp_lengthscale === nothing ? nothing : Float64(gp_lengthscale),
                gp_variance === nothing ? nothing : Float64(gp_variance),
                obs_sigma === nothing ? nothing : Float64(obs_sigma),
                Float64(prior_scale), Float64(target_accept),
                rng_seed, verbose)
end

# ─── RKHS solver ───────────────────────────────────────────────────

"""
    RKHSSolver

Trajectory-RKHS estimator (González et al. 2014): the *state trajectory*
is placed in a time-kernel RKHS, `x_k(t) = m_k + Σᵢ b_{k,i} k(t, tᵢ)`
with centers at the data times, so `ẋ_k(t)` is available analytically.
The objective — data fit + RKHS norm `λ bᵀKb` + ODE-gradient match
`ρ Σ(ẋ − f(x,θ))²` on a collocation grid — is minimized by alternating a
*linear* solve for the trajectory coefficients (with `f` frozen at the
previous iterate, Picard linearization) and Adam steps on the
unknown-function parameters θ. Continuous-time problems only.

# Fields
- `kernel`: time-kernel type `:rbf`, `:matern32`, `:matern52` (default `:rbf`)
- `lengthscale`: time-kernel lengthscale; ≤ 0 (default) auto-scales to
  `time_span / 8`
- `lambda_rkhs`: RKHS-norm penalty λ on the trajectory (default 0.01)
- `lambda_ode`: weight ρ of the ODE-gradient term (default 1.0)
- `n_repr_points`: size of the ODE collocation grid (default 30)
- `maxiters`: outer alternations; 10 θ Adam steps each (default 200)
- `lr`: θ learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method,
kernel, lengthscale)` with the standard honest-convergence keys (see
[`PSMSolution`](@ref)); `reason` is `:converged_tol` when the relative
objective change fell below tolerance, `:maxiters` otherwise.

# References
- González et al. (2014), Pattern Recognition Letters
"""
struct RKHSSolver
    kernel::Symbol
    lengthscale::Float64      # ≤ 0 means auto-scale from the time span
    lambda_rkhs::Float64
    lambda_ode::Float64
    n_repr_points::Int
    maxiters::Int
    lr::Float64
    verbose::Bool
end

RKHSSolver(; kernel::Symbol=:rbf, lengthscale::Float64=-1.0,
             lambda_rkhs::Float64=0.01, lambda_ode::Float64=1.0,
             n_repr_points::Int=30,
             maxiters::Int=200, lr::Float64=0.01, verbose::Bool=false) =
    RKHSSolver(kernel, lengthscale, lambda_rkhs, lambda_ode, n_repr_points,
               maxiters, lr, verbose)

# ─── Problem and solution types ────────────────────────────────────

"""
    PSMProblem

A partially specified model fitting problem.

# Fields
- `dynamics!`: right-hand side function.
  - Continuous (ODE): `f!(du, u, p, t)` — computes derivatives du/dt
  - Discrete: `f!(u_next, u, p, t)` — computes next state u(t+1)
  In both cases, `p` is a NamedTuple with callable unknown functions and
  known parameters.
- `u0`: initial conditions (vector or function `(params) -> u0`)
- `tspan`: time span `(t0, tf)`
- `approximators`: vector of `AbstractApproximator` for unknown functions
- `data_times`: observation times
- `data_values`: observations matrix `(n_times × n_obs)`
- `data_weights`: weight matrix (same shape as data_values)
- `obs_to_state`: maps observation column j → state variable index
- `known_params`: NamedTuple of known (fixed) parameters
- `likelihood`: likelihood family
- `ode_solver`: ODE/discrete solver (e.g. `Tsit5()` for continuous,
  `FunctionMap()` for discrete, or `nothing` for solvers that don't integrate)
- `ode_kwargs`: additional solver keyword arguments
- `discrete`: whether this is a discrete-time model
- `delays`: delay values for DDE problems (empty for ODE/discrete)
- `history`: history function `h(p, t)` for DDE problems (nothing for ODE)
"""
struct PSMProblem{D, U, L<:AbstractLikelihood, S}
    dynamics!::D
    u0::U
    tspan::Tuple{Float64, Float64}
    approximators::Vector{<:AbstractApproximator}
    data_times::Vector{Float64}
    data_values::Matrix{Float64}
    data_weights::Matrix{Float64}
    obs_to_state::Vector{Int}
    known_params::NamedTuple
    likelihood::L
    ode_solver::S
    ode_kwargs::Dict{Symbol, Any}
    discrete::Bool
    delays::Vector{Float64}
    history::Union{Nothing, Function}
end

"""
    PSMProblem(dynamics!, u0, tspan, approximators; kwargs...)

Construct a PSM fitting problem.

# Keyword arguments
- `data_times`: observation times
- `data_values`: observation matrix (n_times × n_obs)
- `data_weights=nothing`: optional weight matrix
- `obs_to_state`: maps observation columns to state indices
- `known_params=NamedTuple()`: fixed parameter values
- `likelihood=Gaussian()`: likelihood family
- `solver=Tsit5()`: ODE/discrete solver
- `discrete=false`: set `true` for discrete-time models where `dynamics!`
  computes `u(t+1) = f(u(t), p, t)` instead of `du/dt`
- `delays=Float64[]`: delay values for DDE problems
- `history=nothing`: history function `h(p, t)` for DDE problems
- `solver_kwargs...`: passed to the ODE/discrete solver
"""
function PSMProblem(dynamics!, u0, tspan,
                    approximators::Vector{<:AbstractApproximator};
                    data_times::AbstractVector,
                    data_values::AbstractMatrix,
                    data_weights::Union{Nothing, AbstractMatrix}=nothing,
                    obs_to_state::Vector{Int}=collect(1:size(data_values, 2)),
                    known_params::NamedTuple=NamedTuple(),
                    likelihood::AbstractLikelihood=Gaussian(),
                    solver=Tsit5(),
                    discrete::Bool=false,
                    delays::Vector{Float64}=Float64[],
                    history::Union{Nothing, Function}=nothing,
                    solver_kwargs...)
    n_times = length(data_times)
    n_obs = size(data_values, 2)
    size(data_values, 1) == n_times ||
        throw(ArgumentError("data_values has $(size(data_values, 1)) rows " *
                            "but data_times has $n_times entries"))
    length(obs_to_state) == n_obs ||
        throw(ArgumentError("obs_to_state has $(length(obs_to_state)) " *
                            "entries but data_values has $n_obs columns"))
    # A known parameter sharing a name with an approximator would silently
    # SHADOW the fitted function in the merged parameter NamedTuple
    # (later entries win in merge), producing baffling non-fits.
    for approx in approximators
        haskey(known_params, approx.name) &&
            throw(ArgumentError("known_params has an entry :$(approx.name) " *
                                "that collides with an approximator of the " *
                                "same name; rename one of them"))
    end
    # Two approximators sharing a name would silently misalign: each still
    # consumes its own β slice, but the parameter NamedTuple keeps only the
    # LAST evaluator, so earlier ones become dead weight.
    approx_names = Symbol[approx.name for approx in approximators]
    if !allunique(approx_names)
        dups = unique(n for n in approx_names if count(==(n), approx_names) > 1)
        throw(ArgumentError("duplicate approximator name(s) $(dups); " *
                            "each approximator needs a unique name — " *
                            "rename the duplicates"))
    end

    w = if data_weights === nothing
        ones(Float64, n_times, n_obs)
    else
        size(data_weights) == size(data_values) ||
            throw(ArgumentError("data_weights has size $(size(data_weights)) " *
                                "but data_values has size $(size(data_values)); " *
                                "they must match"))
        Float64.(data_weights)
    end

    kwargs = Dict{Symbol, Any}(pairs(solver_kwargs)...)

    t_f = Float64.(data_times)
    y_f = Float64.(data_values)

    # Types carrying FIXED reference statistics derived from the data (so
    # far only SingleIndexApproximator) resolve them here, exactly once,
    # and are replaced by resolved copies. Every other type is returned
    # untouched by the hook, so the approximator vector — and its element
    # type — is unchanged for all existing models.
    approximators = if any(a -> a isa SingleIndexApproximator, approximators)
        [_resolve_index_stats(a, u0, t_f, y_f, w, obs_to_state)
         for a in approximators]
    else
        approximators
    end

    PSMProblem(dynamics!, u0,
               (Float64(tspan[1]), Float64(tspan[2])),
               approximators,
               t_f,
               y_f,
               w,
               obs_to_state,
               known_params,
               likelihood,
               solver,
               kwargs,
               discrete,
               delays,
               history)
end

# ─── Constructors from SciML problem types ───────────────────────

# Wrap an out-of-place dynamics function f(u, p, t) -> result
# into the in-place form f!(out, u, p, t) expected by PSM solvers.
function _wrap_oop(f)
    (out, u, p, t) -> (out .= f(u, p, t); nothing)
end

"""
    PSMProblem(prob::ODEProblem, approximators; kwargs...)

Construct a PSM fitting problem from an `ODEProblem`. The dynamics function,
initial conditions, and time span are extracted from the ODE problem.
Both in-place `f!(du, u, p, t)` and out-of-place `f(u, p, t) -> du`
formulations are supported. Defaults to `solver=Tsit5()`.

# Example
```julia
# In-place
f!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
ode = ODEProblem(f!, [1.0], (0.0, 10.0))

# Out-of-place
f(u, p, t) = [p.r(u[1]) * u[1]]
ode = ODEProblem(f, [1.0], (0.0, 10.0))

psm = PSMProblem(ode, [BSplineApproximator(:r, (0.0, 10.0), 8)];
                 data_times=..., data_values=...)
```
"""
function PSMProblem(prob::SciMLBase.AbstractODEProblem,
                    approximators::Vector{<:AbstractApproximator};
                    solver=Tsit5(),
                    kwargs...)
    dynamics! = if SciMLBase.isinplace(prob)
        prob.f.f
    else
        _wrap_oop(prob.f.f)
    end
    PSMProblem(dynamics!, prob.u0, prob.tspan, approximators;
               solver=solver, discrete=false, kwargs...)
end

"""
    PSMProblem(prob::DiscreteProblem, approximators; kwargs...)

Construct a PSM fitting problem from a `DiscreteProblem`. The dynamics function,
initial conditions, and time span are extracted from the discrete problem.
Both in-place `f!(u_next, u, p, t)` and out-of-place `f(u, p, t) -> u_next`
formulations are supported. Defaults to `solver=nothing` (explicit iteration).

# Example
```julia
# In-place
ricker!(u_next, u, p, t) = (u_next[1] = u[1] * exp(p.g(u[1])); nothing)
disc = DiscreteProblem(ricker!, [20.0], (0.0, 40.0))

# Out-of-place
ricker(u, p, t) = [u[1] * exp(p.g(u[1]))]
disc = DiscreteProblem(ricker, [20.0], (0.0, 40.0))

psm = PSMProblem(disc, [BSplineApproximator(:g, (0.0, 150.0), 10)];
                 data_times=..., data_values=...)
```
"""
function PSMProblem(prob::SciMLBase.AbstractDiscreteProblem,
                    approximators::Vector{<:AbstractApproximator};
                    solver=nothing,
                    kwargs...)
    dynamics! = if SciMLBase.isinplace(prob)
        prob.f.f
    else
        _wrap_oop(prob.f.f)
    end
    PSMProblem(dynamics!, prob.u0, prob.tspan, approximators;
               solver=solver, discrete=true, kwargs...)
end

"""
    PSMSolution

Result of fitting a PSM.

# Fields
- `parameters`: ComponentArray with sections for each approximator
- `objective`: final penalized objective value
- `data_loss`: unpenalized data misfit — the WEIGHTED residual sum of
  squares `Σ wᵢⱼ (yᵢⱼ − ŷᵢⱼ)²`, computed by `weighted_data_loss` over the
  usable cells only (positive weight and finite datum, so one masked-out
  missing observation cannot turn it into `NaN`). This is the convention
  for EVERY likelihood family and every solver: no solver reports a
  deviance here, so for non-Gaussian families `data_loss` is a descriptive
  SSE and NOT the quantity the solver optimised (that is `objective`)
- `edf`: estimated degrees of freedom
- `smoothing_params`: vector of estimated smoothing parameters λ. For the
  penalized-likelihood solvers (LAML, GCVSolver, CollocationLAML,
  GradientMatching) there is ONE entry per penalty BLOCK, in the
  enumeration order of `build_penalty_matrices` (approximators in problem
  order; within an approximator, blocks in the order its `penalty_blocks`
  returns them). With only single-block approximators — the default —
  this is one entry per penalized approximator, as historically; an
  unpenalized approximator (no blocks) contributes no entry, and a
  multi-block approximator contributes one entry per block.
  For `LAML`, `GCVSolver` and `CollocationLAML`, λ̂ and `parameters` are a
  MATCHED PAIR: λ̂ is the λ whose penalty `B(λ̂) = Σ λ̂ₗ Sₗ` produced β̂ — the
  last accepted penalized step was taken under it — and every other
  θ-dependent quantity the solution carries is evaluated at that same λ̂
  (`objective` and `edf` for all three; `convergence.V_beta`,
  `convergence.sigma2` and `convergence.laml` for `LAML`, which alone
  reports them). Both `LAML` and `CollocationLAML` compute one further λ
  proposal after their last coefficient update — `LAML` from the next
  Fellner-Schall step of the IRLS loop, `CollocationLAML` at the end of its
  final continuation level, where the proposal exists only to seed a next
  level that never runs. Those proposals are not reported, because β̂ was
  never fitted under them. For `LAML` on a CONVERGED fit, β̂ is therefore a
  fixed point of the penalized step at λ̂ and refitting reproduces
  `parameters`; on a truncated fit (`convergence.converged == false`) it is
  not — the step contraction may have accepted a partial step — but the
  pairing is still honest. `CollocationLAML` gives the weaker guarantee:
  its inner loop stops on an objective tolerance inside a continuation
  schedule rather than at a stationary point, so β̂ is near but not exactly
  a fixed point at λ̂ (measured: refitting moves β̂ by up to 5e-5 on
  converged fits). The λ̂ it reports is still the one β̂ was fitted under. `RodeoSolver` and `DaltonSolver` also populate this field; their
  pairing has not been audited to the same standard.
- `fitted_values`: predicted values at data times (n_times × n_obs)
- `unknown_functions`: Dict of name => callable evaluator
- `convergence`: convergence information. For the iterative optimisers
  (LAML, GCVSolver, CollocationLAML, AdamSolver, MultipleShootingSolver,
  TwoStageSolver, IntegralMatchingSolver, BNGSolver, ODINSolver, RKHSSolver)
  this is a NamedTuple containing at least
  - `converged::Bool` — `true` only when a genuine stopping criterion fired
    (never when the loop merely exhausted its iteration budget),
  - `iterations::Int` — iterations actually performed,
  - `reason::Symbol` — why the loop stopped: `:converged_tol` (tolerance
    criterion met), `:plateau` (objective stagnated over a window / no
    improving step existed), `:maxiters` (budget exhausted without a
    criterion firing), or `:early_break` (internal failure such as a
    simulation error or singular linear system),
  plus solver-specific extras documented on each solver type.
"""
struct PSMSolution
    parameters::ComponentArray
    objective::Float64
    data_loss::Float64
    edf::Float64
    smoothing_params::Vector{Float64}
    fitted_values::Matrix{Float64}
    data_values::Matrix{Float64}
    data_times::Vector{Float64}
    unknown_functions::Dict{Symbol, Any}
    convergence::Any
end
