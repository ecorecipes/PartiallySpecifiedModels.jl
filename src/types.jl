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

Gaussian process approximator. Parameters are function values at uniformly
spaced inducing points; the penalty matrix is the inverse kernel matrix K⁻¹,
which corresponds to the GP prior `f ~ N(0, K/θ)`.

This provides automatic smoothing via LAML (like B-splines) but with
kernel-based correlation structure instead of polynomial smoothness.

# Kernels
- `:sqexp`  — Squared exponential: `k(r) = σ² exp(-r²/(2ℓ²))`
- `:matern32` — Matérn 3/2: `k(r) = σ²(1 + √3r/ℓ) exp(-√3r/ℓ)`
- `:matern52` — Matérn 5/2: `k(r) = σ²(1 + √5r/ℓ + 5r²/(3ℓ²)) exp(-√5r/ℓ)`

# Arguments
- `name`: symbol for the unknown function
- `domain`: `(lo, hi)` range of the input variable
- `n_inducing`: number of inducing points (like nknots for splines)
- `kernel`: kernel type (default `:sqexp`)
- `lengthscale`: kernel lengthscale (default: `domain_span / (n_inducing - 1)`)
- `variance`: signal variance σ² (default: 1.0)
- `initial`: optional initial function `x -> y`
"""
struct GPApproximator <: AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    n_inducing::Int
    inducing_points::Vector{Float64}
    kernel::Symbol
    lengthscale::Float64
    variance::Float64
    initial_func::Function
    K::Matrix{Float64}        # kernel matrix at inducing points
    K_inv::Matrix{Float64}    # inverse kernel matrix (penalty)
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
    x_ind = collect(range(d[1], d[2], length=n_inducing))

    # Build kernel matrix
    K = _build_kernel_matrix(kfunc, x_ind)

    # Invert with jitter for numerical stability
    K_jitter = K + 1e-8 * I
    K_inv = inv(K_jitter)
    K_inv = 0.5 * (K_inv + K_inv')  # symmetrize

    init_func = if initial === nothing
        x -> 0.0
    elseif initial isa Function
        initial
    else
        x -> Float64(initial)
    end

    GPApproximator(name_s, d, n_inducing, x_ind, kernel, ℓ, σ², init_func,
                   K, K_inv)
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
                                     nu=1.5, range_param=nothing, initial=nothing)

SPDE (Matérn) approximator with a shape constraint enforced via the SCOP-spline
reparameterization of Pya & Wood (2015).

Combines the Matérn SPDE penalty (interpretable range and smoothness parameters)
with shape constraints (monotonicity, positivity, convexity, etc.). Parameters
are stored in unconstrained space (γ). During evaluation, mesh node values are
computed as `β = Σ * softplus(γ)` where Σ is a constraint matrix, then
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
end

function ShapeConstrainedSPDEApproximator(name::Union{Symbol,String},
                                          domain::Tuple{<:Real, <:Real},
                                          n_basis::Int,
                                          constraint::Symbol;
                                          nu::Real=1.5,
                                          range_param::Union{Nothing, Real}=nothing,
                                          initial=nothing)
    name_s = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    ν = Float64(nu)
    ν ∈ SPDE_SMOOTHNESS || error("nu must be one of $SPDE_SMOOTHNESS, got $ν")
    n_basis >= 4 || error("n_basis must be ≥ 4 for shape-constrained SPDE, got $n_basis")
    constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $SHAPE_CONSTRAINTS"))

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
                                     constraint, Sig, init_func)
end

function nparams(a::ShapeConstrainedSPDEApproximator)
    a.constraint in _ZERO_ENDPOINT_CONSTRAINTS ? a.n_basis - 1 : a.n_basis
end

function initial_params(a::ShapeConstrainedSPDEApproximator)
    beta_target = Float64[a.initial_func(x) for x in a.mesh_points]
    ν = a.Sigma \ beta_target
    ν = max.(ν, 0.01)
    return [v > 20.0 ? v : log(exp(v) - 1.0) for v in ν]
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

Note: `:increasing` already implies f(x) > 0 since softplus increments are positive.
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
values are computed as `β = Σ * softplus(γ)` where Σ is a constraint matrix
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
    xs = range(a.domain[1], a.domain[2], length=a.nknots)
    beta_target = Float64[a.initial_func(x) for x in xs]
    np = nparams(a)
    # Solve Σ * ν = β_target for ν = softplus(γ) > 0
    # For square Σ: direct solve; for rectangular (q × np): least-squares
    ν = a.Sigma \ beta_target
    ν = max.(ν, 0.01)
    # softplus_inv(ν) = log(exp(ν) - 1) for ν > 0
    return [v > 20.0 ? v : log(exp(v) - 1.0) for v in ν]
end

# ─── COMONet shape-constrained neural network approximator ────────

"""
Supported shape constraints for COMONet approximators.

COMONet (Constrained Monotone Network) enforces shape constraints
architecturally using `exp(W)` weights and specialized activations.
Constraints are guaranteed everywhere by construction — not just at
knot points.

**Monotonicity:**
- `:increasing`    — f'(x) ≥ 0 (exp(W) weights + ReLU)
- `:decreasing`    — f'(x) ≤ 0 (negate input)

**Curvature:**
- `:convex`        — f''(x) ≥ 0 (exp(W) weights + ReLU)
- `:concave`       — f''(x) ≤ 0 (-ReLU(-·))

**Combined:**
- `:inc_convex`    — increasing + convex
- `:inc_concave`   — increasing + concave
- `:dec_convex`    — decreasing + convex
- `:dec_concave`   — decreasing + concave
- `:positive`      — f(x) ≥ 0 (exp output)
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
end

function COMONetApproximator(name::Union{Symbol,String},
                             domain::Tuple{<:Real, <:Real},
                             hidden_sizes::Union{Tuple{Vararg{Int}}, Vector{Int}},
                             constraint::Symbol;
                             penalty_weight::Float64=0.01,
                             activation::Symbol=:relu)
    name = Symbol(name)
    d = (Float64(domain[1]), Float64(domain[2]))
    constraint in COMONET_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $COMONET_CONSTRAINTS"))
    activation in COMONET_ACTIVATIONS || throw(ArgumentError(
        "Unknown activation :$activation. Must be one of $COMONET_ACTIVATIONS"))
    hs = hidden_sizes isa Vector ? Tuple(hidden_sizes...) : hidden_sizes
    length(hs) >= 1 || throw(ArgumentError("Need at least 1 hidden layer"))
    COMONetApproximator(name, d, hs, constraint, penalty_weight, activation)
end

function nparams(a::COMONetApproximator)
    # Count parameters: input→h1, h1→h2, ..., hL→1 (weights + biases)
    np = 0
    prev = 1  # single input (1D)
    for h in a.hidden_sizes
        np += prev * h + h  # W (prev×h) + b (h)
        prev = h
    end
    np += prev + 1  # output layer: W (prev×1) + b (1)
    return np
end

function initial_params(a::COMONetApproximator)
    np = nparams(a)
    # Xavier-like initialization scaled for exp(W) transform
    params = 0.1 .* randn(np)

    # For concave activations (min(0,z)), biases must be initialized negatively
    # so that pre-activations z = exp(W)*x + b can be negative.
    # With b≈0 and x∈[0,1], z>0 always, so min(0,z)=0 (dead neurons).
    # Also set output bias positive since hidden outputs are ≤ 0.
    use_concave = a.constraint in (:concave, :inc_concave, :dec_concave)
    if use_concave
        idx = 1
        prev = 1
        for h in a.hidden_sizes
            n_w = prev * h
            idx += n_w  # skip weights
            params[idx:idx+h-1] .= -1.0 .+ 0.1 .* randn(h)  # negative biases
            idx += h
            prev = h
        end
        # Output bias: set positive to compensate for negative hidden outputs
        params[end] = 1.0
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

"""Poisson likelihood with log link."""
struct Poisson <: AbstractLikelihood end

"""Negative Binomial likelihood with log link."""
struct NegativeBinomial <: AbstractLikelihood
    theta::Float64  # overdispersion: Var = μ + μ²/θ
end
NegativeBinomial() = NegativeBinomial(1.0)

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
           warmup=0, sigma2_init=nothing)

Laplace Approximate Marginal Likelihood algorithm.
Equivalent to REML for Gaussian data.
Uses Fellner-Schall + Newton for smoothing parameter estimation.

# Keyword arguments
- `maxiters::Int=100`: maximum IRLS+LAML iterations
- `tol::Float64=1e-6`: convergence tolerance on penalized objective
- `verbose::Bool=false`: print iteration diagnostics
- `initial_lambda::Union{Nothing,Float64}=nothing`: initial smoothing parameter
  for all terms.  Default (`nothing`) uses `θ = 1.0`, which gives moderate
  initial smoothing.  For strongly nonlinear problems, a higher value
  (e.g. `10.0`) combined with `warmup` helps the IRLS converge to a good
  basin before LAML refinement.
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
    AdamSolver(; maxiters=300, lr=0.01, verbose=false, loss=:mse, autodiff=true)

Adam optimizer that trains unknown function parameters through ODE integration.

For neural networks: uses a ForwardDiff-compatible MLP evaluator (bypassing Lux)
so that exact gradients can be computed through the ODE solve. This matches the
approach used in Universal Differential Equations (UDEs).

For B-splines and GPs: also supported, uses the standard evaluators.

The loss function computes `solve(ODEProblem(...), solver)` at each step and
compares with data. ForwardDiff computes exact gradients through the ODE solve.

# Arguments
- `maxiters::Int=300`: maximum Adam iterations
- `lr::Float64=0.01`: learning rate (with cosine annealing)
- `verbose::Bool=false`: print iteration details
- `loss::Symbol=:mse`: loss function (:mse or :poisson)
- `autodiff::Bool=true`: use ForwardDiff (true) or finite differences (false)
"""
struct AdamSolver
    maxiters::Int
    lr::Float64
    verbose::Bool
    loss::Symbol
    autodiff::Bool
end

AdamSolver(; maxiters::Int=300, lr::Float64=0.01, verbose::Bool=false,
             loss::Symbol=:mse, autodiff::Bool=true) =
    AdamSolver(maxiters, lr, verbose, loss, autodiff)

"""
    MultipleShootingSolver(; n_intervals=10, maxiters_inner=100, maxiters_outer=20,
                             lr=0.01, rho_init=1.0, rho_max=1e6, verbose=false)

Multiple shooting solver for training neural differential equations, following
Turan & Jäschke (2021). Partitions the time span into intervals with shooting
variables at boundaries. Uses augmented Lagrangian to enforce continuity.

Advantages over single shooting (AdamSolver):
- Better initial fits: shooting variables initialized from data
- Avoids "flattened trajectory" failure mode for oscillatory systems
- Shorter integration intervals improve gradient quality

# Arguments
- `n_intervals::Int=10`: number of shooting intervals
- `maxiters_inner::Int=100`: Adam iterations per augmented Lagrangian step
- `maxiters_outer::Int=20`: augmented Lagrangian outer iterations
- `lr::Float64=0.01`: Adam learning rate
- `rho_init::Float64=10.0`: initial penalty parameter for shooting constraints
- `rho_max::Float64=1e6`: maximum penalty parameter
- `verbose::Bool=false`: print iteration details
- `autodiff::Bool=true`: use ForwardDiff (true) or finite differences (false)
"""
struct MultipleShootingSolver
    n_intervals::Int
    maxiters_inner::Int
    maxiters_outer::Int
    lr::Float64
    rho_init::Float64
    rho_max::Float64
    verbose::Bool
    autodiff::Bool
end

MultipleShootingSolver(; n_intervals::Int=10, maxiters_inner::Int=100,
                         maxiters_outer::Int=20, lr::Float64=0.01,
                         rho_init::Float64=10.0, rho_max::Float64=1e6,
                         verbose::Bool=false, autodiff::Bool=true) =
    MultipleShootingSolver(n_intervals, maxiters_inner, maxiters_outer,
                           lr, rho_init, rho_max, verbose, autodiff)

"""
    AdaptiveGradientMatching(; maxiters=200, verbose=false, gamma_init=1.0,
                               fit_gamma=true, kernel=:rbf)

Adaptive Gradient Matching solver following Dondelinger et al. (2013) and the
deGradInfer R package. Uses Gaussian processes to smooth data and compute
gradient estimates with uncertainty, then matches ODE-predicted gradients
using a "product of experts" formulation.

For partially specified models, the unknown function coefficients are optimized
jointly with the mismatch parameter γ via L-BFGS.

The loss function for each state k is:
    L_k = -0.5 (f_k - m_k)ᵀ (A_k + γ_k I)⁻¹ (f_k - m_k) - 0.5 log|A_k + γ_k I|

where:
- f_k = ODE-predicted gradients for state k
- m_k = GP gradient mean = K*(K + σ²I)⁻¹x_k
- A_k = GP gradient covariance = K** - K*(K + σ²I)⁻¹K*ᵀ
- γ_k = mismatch parameter controlling ODE-GP coupling

# Arguments
- `maxiters::Int=200`: maximum L-BFGS iterations
- `verbose::Bool=false`: print iteration details
- `gamma_init::Float64=1.0`: initial mismatch parameter (per state)
- `fit_gamma::Bool=true`: optimize γ or keep fixed
- `kernel::Symbol=:rbf`: GP kernel (:rbf, :matern32, :matern52)
"""
struct AdaptiveGradientMatching
    maxiters::Int
    verbose::Bool
    gamma_init::Float64
    fit_gamma::Bool
    kernel::Symbol
end

AdaptiveGradientMatching(; maxiters::Int=200, verbose::Bool=false,
                           gamma_init::Float64=1.0, fit_gamma::Bool=true,
                           kernel::Symbol=:rbf) =
    AdaptiveGradientMatching(maxiters, verbose, gamma_init, fit_gamma, kernel)

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

RodeoSolver(; n_steps::Int=200, n_deriv::Int=3,
              sigma::Union{Nothing, Vector{Float64}}=nothing,
              obs_var::Union{Nothing, Float64}=nothing,
              method::Symbol=:basic,
              interrogate::Symbol=:kramer,
              maxiters::Int=200, verbose::Bool=false) =
    RodeoSolver(n_steps, n_deriv, sigma, obs_var, method, interrogate, maxiters, verbose)

"""
    MCMCSolver(; n_samples=1000, n_warmup=500, n_chains=1, target_accept=0.8,
                 prior_scale=1.0, obs_sigma=nothing, verbose=false)

Full Bayesian inference via Hamiltonian Monte Carlo (NUTS).
Uses LogDensityProblems.jl + AdvancedHMC.jl.

# Arguments
- `n_samples`: number of posterior samples per chain (after warmup)
- `n_warmup`: number of warmup/adaptation steps
- `n_chains`: number of independent chains
- `target_accept`: target acceptance rate for NUTS adaptation (0.6–0.95)
- `prior_scale`: scale for Gaussian prior on parameters (larger = weaker prior).
  When penalty matrices exist (B-spline, GP), uses the penalty; otherwise N(0, prior_scale²).
- `obs_sigma`: observation noise std dev. If `nothing`, estimated as a parameter.
- `verbose`: print progress
"""
struct MCMCSolver
    n_samples::Int
    n_warmup::Int
    n_chains::Int
    target_accept::Float64
    prior_scale::Float64
    obs_sigma::Union{Nothing, Float64}
    verbose::Bool
end

MCMCSolver(; n_samples::Int=1000, n_warmup::Int=500, n_chains::Int=1,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             obs_sigma::Union{Nothing, Float64}=nothing,
             verbose::Bool=false) =
    MCMCSolver(n_samples, n_warmup, n_chains, target_accept, prior_scale, obs_sigma, verbose)

"""
    MagiSolver(; n_samples=1000, n_warmup=500, n_deriv=3, n_gridpoints=200,
                 sigma=nothing, obs_var=0.01, target_accept=0.8,
                 prior_scale=1.0, verbose=false)

Manifold-constrained Gaussian process inference (MAGI) for ODE systems.

MAGI models each ODE state as integrated Brownian motion and uses a Kalman
filter to evaluate the log-likelihood of the ODE constraint. States are
marginalized out — only ODE parameters θ and unknown function parameters
are sampled via NUTS/HMC.

**Key advantage**: Handles partially observed systems naturally (unobserved
state components are inferred through the ODE constraint).

Returns an `MCMCChains.Chains` object with posterior samples.

# Fields
- `n_samples`: number of posterior samples after warmup
- `n_warmup`: warmup/adaptation iterations
- `n_deriv`: derivatives in IBM prior (2 or 3)
- `n_gridpoints`: number of time discretization points for Kalman filter
- `sigma`: IBM scale per state (auto-estimated if `nothing`)
- `obs_var`: observation noise variance
- `target_accept`: NUTS target acceptance rate
- `prior_scale`: scale for Gaussian prior on parameters
- `verbose`: print progress

# References
- Yang, Wong & Kou (2021) PNAS 118(15): "Inference of dynamic systems
  from noisy and sparse data via manifold-constrained Gaussian processes"
"""
struct MagiSolver
    n_samples::Int
    n_warmup::Int
    n_deriv::Int
    n_gridpoints::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Float64
    target_accept::Float64
    prior_scale::Float64
    preoptimize::Bool
    verbose::Bool
end

MagiSolver(; n_samples::Int=1000, n_warmup::Int=500, n_deriv::Int=3,
             n_gridpoints::Int=200,
             sigma::Union{Nothing, Vector{Float64}}=nothing,
             obs_var::Float64=0.01,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             preoptimize::Bool=true, verbose::Bool=false) =
    MagiSolver(n_samples, n_warmup, n_deriv, n_gridpoints, sigma, obs_var,
               target_accept, prior_scale, preoptimize, verbose)

# ─── BNG solver (Bonnaffé et al. 2023) ────────────────────────────

"""
    BNGSolver

Bayesian Neural Gradient matching solver (Bonnaffé et al. 2023).

Two-step approach that avoids ODE integration entirely:
1. Smooth observed time series to get interpolated states and derivatives
2. Fit unknown functions by matching the smoothed derivatives

# Fields
- `n_basis`: number of spline basis functions for data smoothing (default 20)
- `maxiters`: maximum optimization iterations for step 2 (default 2000)
- `lr`: learning rate for Adam optimizer (default 0.01)
- `lambda_smooth`: smoothing penalty for data interpolation (default 1.0)
- `verbose`: print progress
"""
struct BNGSolver
    n_basis::Int
    maxiters::Int
    lr::Float64
    lambda_smooth::Float64
    verbose::Bool
end

BNGSolver(; n_basis::Int=20, maxiters::Int=2000, lr::Float64=0.01,
            lambda_smooth::Float64=1.0, verbose::Bool=false) =
    BNGSolver(n_basis, maxiters, lr, lambda_smooth, verbose)

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
- `obs_var`: observation noise variance (default 0.01)
- `interrogate`: interrogation method `:kramer` or `:schober` (default `:kramer`)
- `maxiters`: optimization iterations (default 200)
- `verbose`: print progress
"""
struct DaltonSolver
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Float64
    interrogate::Symbol
    maxiters::Int
    verbose::Bool
end

DaltonSolver(; n_steps::Int=200, n_deriv::Int=3,
               sigma::Union{Nothing, Vector{Float64}}=nothing,
               obs_var::Float64=0.01,
               interrogate::Symbol=:kramer,
               maxiters::Int=200, verbose::Bool=false) =
    DaltonSolver(n_steps, n_deriv, sigma, obs_var, interrogate, maxiters, verbose)

# ─── Pseudo-marginal solver (Chkrebtii et al. 2016) ───────────────

"""
    PseudoMarginalSolver

Pseudo-marginal MCMC using a probabilistic ODE solver for likelihood estimation.

Uses RODEO/fenrir as an inner solver to compute an unbiased estimate of the
marginal likelihood p(Y|θ), then samples from the posterior p(θ|Y) via NUTS.

# Fields
- `n_samples`: number of posterior samples (default 1000)
- `n_warmup`: warmup/adaptation samples (default 500)
- `n_steps`: discretization steps for inner probabilistic solver (default 200)
- `n_deriv`: IBM prior derivative order (default 3)
- `sigma`: IBM scale parameters (nothing = auto-estimate)
- `obs_var`: observation noise variance (default 0.01)
- `target_accept`: NUTS target acceptance rate (default 0.8)
- `prior_scale`: prior standard deviation on parameters (default 1.0)
- `inner_method`: inner likelihood method `:fenrir` or `:dalton` (default `:fenrir`)
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

PseudoMarginalSolver(; n_samples::Int=1000, n_warmup::Int=500,
                       n_steps::Int=200, n_deriv::Int=3,
                       sigma::Union{Nothing, Vector{Float64}}=nothing,
                       obs_var::Union{Nothing, Float64}=nothing,
                       target_accept::Float64=0.8, prior_scale::Float64=1.0,
                       inner_method::Symbol=:fenrir,
                       initial_params::Union{Nothing, Vector{Float64}}=nothing,
                       verbose::Bool=false) =
    PseudoMarginalSolver(n_samples, n_warmup, n_steps, n_deriv, sigma, obs_var,
                          target_accept, prior_scale, inner_method,
                          initial_params, verbose)

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
- `gamma`: GCV inflation factor (default 1.4, >1 guards against under-smoothing)
- `verbose`: print progress
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
- `n_basis_smooth`: spline basis functions for data smoothing (default 20)
- `lambda_smooth`: smoothing penalty for initial data fit (default 1.0)
- `maxiters`: max iterations for parameter matching (default 1000)
- `lr`: learning rate for Adam optimization in matching step (default 0.01)
- `verbose`: print progress
"""
struct TwoStageSolver
    n_basis_smooth::Int
    lambda_smooth::Float64
    maxiters::Int
    lr::Float64
    verbose::Bool
end

TwoStageSolver(; n_basis_smooth::Int=20, lambda_smooth::Float64=1.0,
                 maxiters::Int=1000, lr::Float64=0.01, verbose::Bool=false) =
    TwoStageSolver(n_basis_smooth, lambda_smooth, maxiters, lr, verbose)

# ─── Derivative-free solver (stochastic + NelderMead) ──────────────

"""
    DerivativeFreeSolver

Derivative-free optimization solver using NelderMead or particle swarm.

Useful as a robust fallback when gradient-based methods fail (non-smooth
objectives, stiff dynamics, poor conditioning). Uses simulation-based
loss without requiring autodiff through ODE solves.

# Fields
- `method`: optimization method — `:nelder_mead`, `:particle_swarm`, `:cmaes` (default `:nelder_mead`)
- `maxiters`: maximum function evaluations (default 10000)
- `n_particles`: particle count for swarm methods (default 20)
- `loss`: loss type `:mse` or `:likelihood` (default `:mse`)
- `verbose`: print progress
"""
struct DerivativeFreeSolver
    method::Symbol
    maxiters::Int
    n_particles::Int
    loss::Symbol
    verbose::Bool
end

DerivativeFreeSolver(; method::Symbol=:nelder_mead, maxiters::Int=10000,
                       n_particles::Int=20, loss::Symbol=:mse,
                       verbose::Bool=false) =
    DerivativeFreeSolver(method, maxiters, n_particles, loss, verbose)

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
- `prior_scale`: prior std on parameters (default 1.0)
- `verbose`: print progress
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

# Fields
- `n_particles`: number of ABC particles (default 500)
- `n_generations`: number of SMC generations (default 10)
- `summary_fn`: summary statistic function, or `:auto` for MSE-based (default `:auto`)
- `prior_scale`: prior half-width on parameters (default 2.0)
- `quantile_eps`: quantile for tolerance schedule (default 0.5)
- `verbose`: print progress
"""
struct ABCSolver
    n_particles::Int
    n_generations::Int
    summary_fn::Union{Symbol, Function}
    prior_scale::Float64
    quantile_eps::Float64
    verbose::Bool
end

ABCSolver(; n_particles::Int=500, n_generations::Int=10,
            summary_fn::Union{Symbol, Function}=:auto,
            prior_scale::Float64=2.0, quantile_eps::Float64=0.5,
            verbose::Bool=false) =
    ABCSolver(n_particles, n_generations, summary_fn, prior_scale, quantile_eps, verbose)

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
    @assert size(data_values, 1) == n_times "data_values rows must match data_times length"
    @assert length(obs_to_state) == n_obs

    w = if data_weights === nothing
        ones(Float64, n_times, n_obs)
    else
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
- `data_loss`: unpenalized data loss (SS for Gaussian, deviance for others)
- `edf`: estimated degrees of freedom
- `smoothing_params`: vector of estimated smoothing parameters λ
- `fitted_values`: predicted values at data times (n_times × n_obs)
- `unknown_functions`: Dict of name => callable evaluator
- `convergence`: convergence information
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
