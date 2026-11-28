# ─── Likelihood types ──────────────────────────────────────────────

"""
Abstract type for exponential family likelihoods used in LAML/IRLS.

Each subtype must support:
- `log_likelihood(fam, y, mu, w)`: total log-likelihood
- `irls_weights(fam, y, mu, w)`: IRLS working weights
"""
abstract type AbstractLikelihood end

"""Gaussian likelihood with identity link (unknown σ² profiled out in REML)."""
struct GaussianLikelihood <: AbstractLikelihood end

"""Poisson likelihood, fitted on the response scale (identity link)."""
struct PoissonLikelihood <: AbstractLikelihood end

"""Negative Binomial likelihood, fitted on the response scale (identity link)."""
struct NegativeBinomialLikelihood <: AbstractLikelihood
    theta::Float64  # overdispersion: Var = μ + μ²/θ
    function NegativeBinomialLikelihood(theta::Real)
        θ = Float64(theta)
        (isfinite(θ) && θ > 0.0) || throw(ArgumentError(
            "NegativeBinomial: theta must be finite and > 0, got $theta"))
        new(θ)
    end
end
NegativeBinomialLikelihood() = NegativeBinomialLikelihood(1.0)

"""
    TruncatedNormal(lower=0.0, sigma=1.0)

Truncated Normal likelihood for non-negative continuous data.  The distribution
is a Normal(μ, σ²) truncated to `[lower, ∞)`.  Scale parameter σ is fixed
(not profiled like the Gaussian σ²).

The model predicts the latent location μ, which may lie below `lower` and
is not the truncated response mean. The likelihood, score, information
and Pearson variance use shared stable normal-tail calculations.

Useful for continuous measurements that are bounded below (e.g., population
densities, concentrations).

# Fields
- `lower::Float64`: lower truncation point (default 0.0)
- `sigma::Float64`: standard deviation (default 1.0)
"""
struct TruncatedNormalLikelihood <: AbstractLikelihood
    lower::Float64
    sigma::Float64
    function TruncatedNormalLikelihood(lower::Real, sigma::Real)
        lo = Float64(lower)
        σ = Float64(sigma)
        isfinite(lo) || throw(ArgumentError("TruncatedNormal: lower must be finite, got $lower"))
        (isfinite(σ) && σ > 0.0) || throw(ArgumentError(
            "TruncatedNormal: sigma must be finite and > 0, got $sigma"))
        new(lo, σ)
    end
end
TruncatedNormalLikelihood(; lower::Real=0.0, sigma::Real=1.0) =
    TruncatedNormalLikelihood(lower, sigma)

"""
    GammaLikelihood(phi=1.0)

Gamma likelihood for strictly positive continuous data with constant
coefficient of variation (`Var(Y) = phi * μ²`, shape `ν = 1 / phi`).
Identity-link exponential family on the response scale.
"""
struct GammaLikelihood <: AbstractLikelihood
    phi::Float64
    function GammaLikelihood(phi::Real)
        ϕ = Float64(phi)
        (isfinite(ϕ) && ϕ > 0.0) || throw(ArgumentError(
            "GammaLikelihood: dispersion phi must be finite and > 0, got $phi"))
        new(ϕ)
    end
end
GammaLikelihood() = GammaLikelihood(1.0)

"""
    LogNormalLikelihood(sigma=0.1)

Log-normal likelihood for strictly positive multiplicative-noise data:
`log(Y) ~ Normal(log(μ), sigma²)`, where `μ > 0` is the model trajectory
(median) and `sigma > 0` is the log-scale standard deviation.
"""
struct LogNormalLikelihood <: AbstractLikelihood
    sigma::Float64
    function LogNormalLikelihood(sigma::Real)
        σ = Float64(sigma)
        (isfinite(σ) && σ > 0.0) || throw(ArgumentError(
            "LogNormalLikelihood: sigma must be finite and > 0, got $sigma"))
        new(σ)
    end
end
LogNormalLikelihood() = LogNormalLikelihood(0.1)

"""
    BinomialLikelihood(n_trials=1)

Binomial likelihood for bounded success counts `y ∈ [0, n_trials]` where the
dynamical model predicts the success probability `p = μ ∈ (0, 1)` (e.g.,
seroprevalence or occupancy probability).
"""
struct BinomialLikelihood <: AbstractLikelihood
    n_trials::Int
    function BinomialLikelihood(n_trials::Integer)
        n = Int(n_trials)
        n >= 1 || throw(ArgumentError(
            "BinomialLikelihood: n_trials must be ≥ 1, got $n_trials"))
        new(n)
    end
end
BinomialLikelihood() = BinomialLikelihood(1)

"""
    StudentTLikelihood(df=4.0, sigma=1.0)

Heavy-tailed Student-t likelihood for robust continuous regression in the
presence of ecological outliers: `(Y - μ) / sigma ~ t_df`.
Uses strictly positive EM/IRLS robustness weights `w_i (df + 1) / (df * sigma² + r_i²)`.
"""
struct StudentTLikelihood <: AbstractLikelihood
    df::Float64
    sigma::Float64
    function StudentTLikelihood(df::Real, sigma::Real)
        ν = Float64(df); σ = Float64(sigma)
        (isfinite(ν) && ν > 0.0) || throw(ArgumentError(
            "StudentTLikelihood: df must be finite and > 0, got $df"))
        (isfinite(σ) && σ > 0.0) || throw(ArgumentError(
            "StudentTLikelihood: sigma must be finite and > 0, got $sigma"))
        new(ν, σ)
    end
end
StudentTLikelihood(; df::Real=4.0, sigma::Real=1.0) = StudentTLikelihood(df, sigma)

"""
    CustomLikelihood(loglik_scalar)

User-defined likelihood. `loglik_scalar(y, μ)` returns scalar log-likelihood
for one observation. IRLS weights derived via ForwardDiff.
"""
struct CustomLikelihood <: AbstractLikelihood
    loglik_scalar::Function
end

# Hook enabling optional Distributions.jl coercion in PSMProblem(...; likelihood=...)
_coerce_likelihood(lik::AbstractLikelihood) = lik
function _coerce_likelihood(x)
    throw(ArgumentError(
        "PSMProblem: unsupported likelihood of type $(typeof(x)). Pass a " *
        "PartiallySpecifiedModels AbstractLikelihood (GaussianLikelihood, " *
        "PoissonLikelihood, NegativeBinomialLikelihood, TruncatedNormalLikelihood, " *
        "GammaLikelihood, LogNormalLikelihood, BinomialLikelihood, StudentTLikelihood, " *
        "CustomLikelihood) or load Distributions.jl to pass a supported " *
        "Distributions.Distribution directly."))
end

