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
    ℓ = if lengthscale === nothing
        # Default lengthscale: normalize so adjacent inducing points have
        # similar correlation (~0.6) regardless of kernel type.
        # SqExp at h gives exp(-0.5) ≈ 0.607; Matérn kernels need longer ℓ
        # to achieve the same correlation due to their heavier tails.
        h = Float64((d[2] - d[1]) / max(n_inducing - 1, 1))
        if kernel == :matern32
            2.0 * h     # (1+√3·h/(2h))·exp(-√3·h/(2h)) ≈ 0.60
        elseif kernel == :matern52
            1.9 * h     # (1+√5·h/(1.9h)+5/(3·1.9²))·exp(-√5·h/(1.9h)) ≈ 0.61
        else
            h           # SqExp: exp(-0.5) ≈ 0.607
        end
    else
        Float64(lengthscale)
    end

    # Kernel function
    kfunc = _kernel_func(kernel, ℓ, σ²)

    # Inducing points uniformly spaced in domain
    n_inducing >= 2 ||
        throw(ArgumentError("GPApproximator needs n_inducing ≥ 2 (got $n_inducing)"))
    x_ind = collect(range(d[1], d[2], length=n_inducing))

    # Build kernel matrix
    K = _build_kernel_matrix(kfunc, x_ind)

    # Factor with scaled jitter. Squared-exponential Gram matrices are
    # notoriously ill-conditioned as n_inducing grows; a fixed 1e-8 jitter
    # plus explicit inv() produced large oscillatory weights K⁻¹f between
    # inducing points. Scale the jitter to the kernel magnitude and escalate
    # until the Cholesky succeeds.
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

**Note:** Shape constraints are enforced at mesh nodes. The cubic spline
interpolation between nodes can slightly overshoot, so constraints hold
approximately (not exactly) between nodes. Use more basis functions to
reduce overshoot.

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
           warmup=3, sigma2_init=nothing)

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
  for ±5 measurement error).

# Convergence info
`sol.convergence` is a NamedTuple `(V_beta, sigma2, converged, iterations,
reason, laml_failures)`: the posterior covariance and σ̂² used by
[`confidence_band`](@ref), plus the standard honest-convergence keys (see
[`PSMSolution`](@ref)) and `laml_failures::Int`, the number of iterations in
which the LAML smoothing-parameter update failed and θ was kept.
"""
struct LAML
    maxiters::Int
    tol::Float64
    verbose::Bool
    initial_lambda::Union{Nothing,Float64}
    warmup::Int
    sigma2_init::Union{Nothing,Float64}
end

LAML(; maxiters::Int=100, tol::Float64=1e-6, verbose::Bool=false,
       initial_lambda::Union{Nothing,Float64}=nothing,
       warmup::Int=3,
       sigma2_init::Union{Nothing,Float64}=nothing) =
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init)

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

# Keyword arguments
- `maxiters::Int=50`: IRLS iterations per continuation level
- `tol::Float64=1e-6`: convergence tolerance
- `verbose::Bool=false`: print diagnostics
- `lambda_ode_start::Float64=0.01`: initial ODE compliance penalty
- `lambda_ode_end::Float64=1e4`: final ODE compliance penalty
- `n_continuation::Int=8`: number of log-spaced continuation levels
- `sigma2_init::Union{Nothing,Float64}=nothing`: σ² cap for Fellner-Schall

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
end

CollocationLAML(; maxiters::Int=50, tol::Float64=1e-6, verbose::Bool=false,
                  lambda_ode_start::Float64=0.01, lambda_ode_end::Float64=1e4,
                  n_continuation::Int=8,
                  sigma2_init::Union{Nothing,Float64}=nothing) =
    CollocationLAML(maxiters, tol, verbose, lambda_ode_start, lambda_ode_end,
                    n_continuation, sigma2_init)

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
                 penalty_weight=0.0, autodiff=true)

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
- `autodiff::Bool=true`: use ForwardDiff (true) or finite differences (false)

# Convergence info
`sol.convergence` is a NamedTuple `(optimizer, method, converged, iterations,
reason, final_grad_norm)` with the standard honest-convergence keys (see
[`PSMSolution`](@ref)) plus `final_grad_norm::Float64`, the Euclidean norm of
the last computed gradient. `reason == :plateau` is only reported while the
cosine-annealed learning rate is still above 5% of the base `lr` — a plateau
that appears merely because the schedule has driven the step size to zero is
reported as `:maxiters`, not convergence.
"""
struct AdamSolver
    maxiters::Int
    lr::Float64
    verbose::Bool
    loss::Symbol
    penalty_weight::Float64
    autodiff::Bool
end

AdamSolver(; maxiters::Int=300, lr::Float64=0.01, verbose::Bool=false,
             loss::Symbol=:auto, penalty_weight::Float64=0.0,
             autodiff::Bool=true) =
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff)

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
end

function RodeoSolver(; n_steps::Int=200, n_deriv::Int=3,
                       sigma::Union{Nothing, Vector{Float64}}=nothing,
                       obs_var::Union{Nothing, Float64}=nothing,
                       method::Symbol=:basic,
                       interrogate::Symbol=:kramer,
                       maxiters::Int=200, verbose::Bool=false)
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
    RodeoSolver(n_steps, n_deriv, sigma, obs_var, method, interrogate, maxiters, verbose)
end

"""
    MCMCSolver(; n_samples=1000, n_warmup=500, n_chains=1, target_accept=0.8,
                 prior_scale=1.0, obs_sigma=nothing, sample_smoothing=false, verbose=false)

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
    verbose::Bool
end

MCMCSolver(; n_samples::Int=1000, n_warmup::Int=500, n_chains::Int=1,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             obs_sigma::Union{Nothing, Float64}=nothing,
             sample_smoothing::Bool=false,
             verbose::Bool=false) =
    MCMCSolver(n_samples, n_warmup, n_chains, target_accept, prior_scale,
               obs_sigma, sample_smoothing, verbose)

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
end

function DaltonSolver(; n_steps::Int=200, n_deriv::Int=3,
                        sigma::Union{Nothing, Vector{Float64}}=nothing,
                        obs_var::Union{Nothing, Float64}=nothing,
                        interrogate::Symbol=:kramer,
                        maxiters::Int=200, verbose::Bool=false)
    interrogate in (:kramer, :schober) ||
        throw(ArgumentError("DaltonSolver: interrogate must be :kramer or " *
                            ":schober (got :$interrogate)"))
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("DaltonSolver: n_deriv must be ≥ 2 (got $n_deriv)"))
    DaltonSolver(n_steps, n_deriv, sigma, obs_var, interrogate, maxiters, verbose)
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
    verbose::Bool
end

function PseudoMarginalSolver(; n_samples::Int=1000, n_warmup::Int=500,
                                n_steps::Int=200, n_deriv::Int=3,
                                sigma::Union{Nothing, Vector{Float64}}=nothing,
                                obs_var::Union{Nothing, Float64}=nothing,
                                target_accept::Float64=0.8, prior_scale::Float64=1.0,
                                inner_method::Symbol=:ffbs,
                                initial_params::Union{Nothing, Vector{Float64}}=nothing,
                                verbose::Bool=false)
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("PseudoMarginalSolver: n_deriv must be ≥ 2 " *
                            "(got $n_deriv)"))
    PseudoMarginalSolver(n_samples, n_warmup, n_steps, n_deriv, sigma, obs_var,
                          target_accept, prior_scale, inner_method,
                          initial_params, verbose)
end

# ─── GCV solver (Wood 2001 / ddefit504) ────────────────────────────

"""
    GCVSolver

Generalized Cross-Validation solver for smoothing parameter selection.

Uses GCV score minimization (via golden-section search on log(λ)) as an
alternative to LAML/REML. Simpler and faster than LAML, but typically
produces slightly less smooth estimates.

# Fields
- `n_grid`: number of grid points for initial λ search (default 50)
- `maxiters`: maximum IRLS iterations (default 50)
- `tol`: convergence tolerance (default 1e-6)
- `gamma`: GCV inflation factor (default 1.4, >1 guards against
  under-smoothing; `gamma=1.0` reproduces the classical unmodified GCV of
  Wood (2001) / ddefit)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, gcv)` with
the standard honest-convergence keys (see [`PSMSolution`](@ref)) plus
`gcv::Float64`, the last GCV score (NaN when no smooth terms are present).
"""
struct GCVSolver
    n_grid::Int
    maxiters::Int
    tol::Float64
    gamma::Float64
    verbose::Bool
end

GCVSolver(; n_grid::Int=50, maxiters::Int=50, tol::Float64=1e-6,
            gamma::Float64=1.4, verbose::Bool=false) =
    GCVSolver(n_grid, maxiters, tol, gamma, verbose)

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
    verbose::Bool
end

VariationalSolver(; maxiters::Int=2000, lr::Float64=0.01,
                    n_elbo_samples::Int=10, prior_scale::Float64=1.0,
                    obs_noise_var::Union{Nothing, Float64}=nothing,
                    verbose::Bool=false) =
    VariationalSolver(maxiters, lr, n_elbo_samples, prior_scale, obs_noise_var, verbose)

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
- `verbose`: print progress
"""
struct ABCSolver
    n_particles::Int
    n_generations::Int
    summary_fn::Union{Symbol, Function}
    prior::Symbol
    prior_scale::Float64
    quantile_eps::Float64
    verbose::Bool
end

ABCSolver(; n_particles::Int=500, n_generations::Int=10,
            summary_fn::Union{Symbol, Function}=:auto,
            prior::Symbol=:smoothness,
            prior_scale::Float64=2.0, quantile_eps::Float64=0.5,
            verbose::Bool=false) =
    ABCSolver(n_particles, n_generations, summary_fn, prior, prior_scale,
              quantile_eps, verbose)

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

# References
- Simpson & Maclaren (2023), PLOS Comp Biol
"""
struct ProfileLikelihoodSolver
    n_profile_points::Int
    ci_level::Float64
    param_indices::Union{Nothing, Vector{Int}}
    verbose::Bool
end

ProfileLikelihoodSolver(; n_profile_points::Int=20, ci_level::Float64=0.95,
                           param_indices::Union{Nothing, Vector{Int}}=nothing,
                           verbose::Bool=false) =
    ProfileLikelihoodSolver(n_profile_points, ci_level, param_indices, verbose)

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
- `verbose`: print progress

# References
- Iglesias, Law & Stuart (2013), Inverse Problems
- Schillings & Stuart (2017), SIAM J Numer Anal
"""
struct EnsembleKalmanSolver
    n_ensemble::Int
    n_iterations::Int
    noise_scale::Float64
    verbose::Bool
end

EnsembleKalmanSolver(; n_ensemble::Int=50, n_iterations::Int=30,
                       noise_scale::Float64=0.1, verbose::Bool=false) =
    EnsembleKalmanSolver(n_ensemble, n_iterations, noise_scale, verbose)

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

    PSMProblem(dynamics!, u0,
               (Float64(tspan[1]), Float64(tspan[2])),
               approximators,
               Float64.(data_times),
               Float64.(data_values),
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
- `smoothing_params`: vector of estimated smoothing parameters λ
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
