# Likelihood families for LAML/IRLS
#
# Each family provides:
#   log_likelihood(fam, y, mu, w) — total log-likelihood
#   irls_weights(fam, y, mu, w)  — IRLS working weights W̃
#
# Reference: Wood, Pya & Säfken (2016), Section 2.

# ─── Standard normal helpers (avoid Distributions.jl dependency) ────

const _INV_SQRT_2PI = 1.0 / sqrt(2π)

"""Standard normal PDF φ(x)."""
_normpdf(x::Real) = _INV_SQRT_2PI * exp(-0.5 * x^2)

"""Standard normal CDF Φ(x) via rational approximation (Abramowitz & Stegun 26.2.17)."""
function _normcdf(x::Real)
    # Accuracy: |ε| < 7.5e-8
    if x >= 0
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 +
               t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        1.0 - _normpdf(x) * poly
    else
        # Compute the small tail DIRECTLY: 1 − Φ(−x) cancels catastrophically
        # below x ≈ −7.5 (Φ underflows against 1), which inflated the
        # TruncatedNormal information ~300× in the far tail.
        t = 1.0 / (1.0 - 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 +
               t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        _normpdf(x) * poly
    end
end

"""Standard normal log-CDF log Φ(x), stable for large negative x."""
function _normlogcdf(x::Real)
    if x > -6.0
        log(_normcdf(x))
    else
        # Asymptotic expansion: log Φ(x) ≈ -½x² - log(-x√(2π))
        -0.5 * x^2 - log(-x) - 0.5 * log(2π)
    end
end

# ─── log Γ (Lanczos; avoids a SpecialFunctions dependency) ──────────

"""Log-gamma via the Lanczos approximation (g=7), accurate to ~1e-13 for x>0."""
function _loggamma(x::Real)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(π / abs(sin(π * x))) - _loggamma(1 - x)
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 2:9
        a += c[i] / (x + (i - 1))
    end
    0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end

# ─── Log-likelihood functions ───────────────────────────────────────

"""
    log_likelihood(fam, y, mu, w)

Total weighted log-likelihood: Σ_i w_i ℓ(y_i, μ_i).

Poisson, NegativeBinomial, and TruncatedNormal include their full
normalizing constants and are mutually comparable (e.g. for AIC). The
**Gaussian** value is the kernel −½Σw(y−μ)² only — σ² is profiled out
elsewhere, so the −(n/2)log(2πσ²) term is omitted and Gaussian values are
NOT comparable across families.
"""
function log_likelihood(::Gaussian, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        ll -= 0.5 * w[i] * (y[i] - mu[i])^2
    end
    ll
end

function log_likelihood(::Poisson, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        mu_i = max(mu[i], 1e-10)
        kern = y[i] > 0 ? y[i] * log(mu_i) - mu_i : -mu_i
        ll += w[i] * (kern - _loggamma(y[i] + 1))   # − log(y!)
    end
    ll
end

function log_likelihood(fam::NegativeBinomial, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    θ = fam.theta
    lgθ = _loggamma(θ)
    ll = 0.0
    for i in eachindex(y)
        mu_i = max(mu[i], 1e-10)
        kern = y[i] * log(mu_i / (mu_i + θ)) + θ * log(θ / (mu_i + θ))
        norm = _loggamma(y[i] + θ) - lgθ - _loggamma(y[i] + 1)
        ll += w[i] * (kern + norm)
    end
    ll
end

function log_likelihood(fam::TruncatedNormal, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    σ = fam.sigma
    a = fam.lower
    ll = 0.0
    for i in eachindex(y)
        z = (y[i] - mu[i]) / σ
        # log f(y|μ,σ,a) = -½z² - log(σ) - ½log(2π) - log Φ((μ-a)/σ)
        ll += w[i] * (-0.5 * z^2 - log(σ) - 0.5 * log(2π) -
                       _normlogcdf((mu[i] - a) / σ))
    end
    ll
end

function log_likelihood(fam::CustomLikelihood, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        ll += w[i] * fam.loglik_scalar(y[i], mu[i])
    end
    ll
end

# ─── Pointwise log-likelihood (single observation) ──────────────────

"""
    loglik_pointwise(fam, y, mu)

Pointwise log-likelihood ℓ(y, μ) for a single observation; the caller
multiplies in any observation weight. Term-for-term identical to what
`log_likelihood` accumulates, so summing `w_i · loglik_pointwise(fam,
y_i, μ_i)` reproduces `log_likelihood(fam, y, mu, w)` exactly. The
**Gaussian** method is the σ-free kernel −½(y−μ)² (σ² is a nuisance
parameter handled by each solver), matching the `log_likelihood`
convention. AD-safe: `μ` may be a `ForwardDiff.Dual`.
"""
loglik_pointwise(::Gaussian, y::Real, mu::Real) = -0.5 * (y - mu)^2

function loglik_pointwise(::Poisson, y::Real, mu::Real)
    mu_c = max(mu, 1e-10)
    kern = y > 0 ? y * log(mu_c) - mu_c : -mu_c
    kern - _loggamma(y + 1)
end

function loglik_pointwise(fam::NegativeBinomial, y::Real, mu::Real)
    θ = fam.theta
    mu_c = max(mu, 1e-10)
    kern = y * log(mu_c / (mu_c + θ)) + θ * log(θ / (mu_c + θ))
    kern + _loggamma(y + θ) - _loggamma(θ) - _loggamma(y + 1)
end

function loglik_pointwise(fam::TruncatedNormal, y::Real, mu::Real)
    σ = fam.sigma
    a = fam.lower
    z = (y - mu) / σ
    -0.5 * z^2 - log(σ) - 0.5 * log(2π) - _normlogcdf((mu - a) / σ)
end

loglik_pointwise(fam::CustomLikelihood, y::Real, mu::Real) =
    fam.loglik_scalar(y, mu)

# ─── IRLS working weights ──────────────────────────────────────────

"""
    irls_weights(fam, y, mu, w)

Compute IRLS working weights W̃ = w / V(μ) for identity-link Fisher scoring.

The PSM solver operates on the response scale (identity link), so
the working weight is the inverse variance: W̃_i = w_i / V(μ_i).
"""
function irls_weights(::Gaussian, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    copy(w)
end

function irls_weights(::Poisson, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Identity link, V(μ) = μ → W̃ = w / μ
    wt = similar(w)
    for i in eachindex(w)
        mu_i = max(abs(mu[i]), 1e-6)
        wt[i] = w[i] / mu_i
    end
    wt
end

function irls_weights(fam::NegativeBinomial, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Identity link, V(μ) = μ + μ²/θ → W̃ = w / V(μ)
    θ = fam.theta
    wt = similar(w)
    for i in eachindex(w)
        mu_i = max(abs(mu[i]), 1e-6)
        wt[i] = w[i] / (mu_i + mu_i^2 / θ)
    end
    wt
end

function irls_weights(fam::TruncatedNormal, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # For TruncatedNormal(a, σ), the observed (= Fisher) information is:
    #   I(μ) = (1/σ²)(1 - ξ·λ(ξ) - λ(ξ)²)
    # where ξ = (μ-a)/σ, λ(ξ) = φ(ξ)/Φ(ξ) (inverse Mills ratio), from
    # -∂²ℓ/∂μ² = (1 + λ'(ξ))/σ² with λ'(ξ) = -ξλ - λ².
    # Analytically I(μ) ∈ (0, 1/σ²); the floor guards roundoff at ξ ≪ 0.
    # Working weight: W̃ = w × I(μ)
    σ = fam.sigma
    a = fam.lower
    wt = similar(w)
    for i in eachindex(w)
        ξ = (mu[i] - a) / σ
        Φξ = max(_normcdf(ξ), 1e-15)
        λξ = _normpdf(ξ) / Φξ          # inverse Mills ratio
        info = (1.0 - ξ * λξ - λξ^2) / σ^2
        wt[i] = w[i] * max(info, 1e-10)
    end
    wt
end

function irls_weights(fam::CustomLikelihood, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Derive via ForwardDiff: w̃_i = w_i × (-∂²ℓ/∂μ²)
    wt = similar(w)
    for i in eachindex(w)
        yi = y[i]
        neg_d2l = -ForwardDiff.derivative(
            μ -> ForwardDiff.derivative(μ2 -> fam.loglik_scalar(yi, μ2), μ),
            mu[i]
        )
        wt[i] = w[i] * max(neg_d2l, 1e-10)
    end
    wt
end

