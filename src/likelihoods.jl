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
        1.0 - _normcdf(-x)
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

Total weighted log-likelihood: Σ_i w_i ℓ(y_i, μ_i). Includes the full
normalizing constants so the value is comparable across models/families
(e.g. for AIC), not just up to an additive constant.
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
    # For TruncatedNormal(a, σ), the Fisher information is:
    #   I(μ) = (1/σ²)(1 + ξ·λ(ξ) - λ(ξ)²)
    # where ξ = (μ-a)/σ, λ(ξ) = φ(ξ)/Φ(ξ) (inverse Mills ratio).
    # Working weight: W̃ = w × I(μ) = w × (-∂²ℓ/∂μ²)
    σ = fam.sigma
    a = fam.lower
    wt = similar(w)
    for i in eachindex(w)
        ξ = (mu[i] - a) / σ
        Φξ = max(_normcdf(ξ), 1e-15)
        λξ = _normpdf(ξ) / Φξ          # inverse Mills ratio
        info = (1.0 + ξ * λξ - λξ^2) / σ^2
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

# ─── IRLS pseudo-data ──────────────────────────────────────────────

"""
    irls_pseudodata(fam, y, mu, w)

Compute IRLS working response z̃_i for each observation.
For Gaussian (identity link): z̃ = y.
For log-link families: z̃ = log(μ) + (y - μ)/μ.
"""
function irls_pseudodata(::Gaussian, y::AbstractVector,
                         mu::AbstractVector, w::AbstractVector)
    copy(y)
end

function irls_pseudodata(::Poisson, y::AbstractVector,
                         mu::AbstractVector, w::AbstractVector)
    z = similar(y)
    for i in eachindex(y)
        mu_i = max(mu[i], 1e-10)
        z[i] = log(mu_i) + (y[i] - mu_i) / mu_i
    end
    z
end

function irls_pseudodata(fam::NegativeBinomial, y::AbstractVector,
                         mu::AbstractVector, w::AbstractVector)
    z = similar(y)
    for i in eachindex(y)
        mu_i = max(mu[i], 1e-10)
        z[i] = log(mu_i) + (y[i] - mu_i) / mu_i
    end
    z
end

function irls_pseudodata(fam::TruncatedNormal, y::AbstractVector,
                         mu::AbstractVector, w::AbstractVector)
    # Identity link: z = y (same as Gaussian)
    copy(y)
end

function irls_pseudodata(fam::CustomLikelihood, y::AbstractVector,
                         mu::AbstractVector, w::AbstractVector)
    # Numerical: z_i = η_i + (y_i - μ_i) / (∂μ/∂η)
    # For identity link, this reduces to y
    z = similar(y)
    for i in eachindex(y)
        yi = y[i]
        dl = ForwardDiff.derivative(μ -> fam.loglik_scalar(yi, μ), mu[i])
        neg_d2l = -ForwardDiff.derivative(
            μ -> ForwardDiff.derivative(μ2 -> fam.loglik_scalar(yi, μ2), μ),
            mu[i]
        )
        z[i] = mu[i] + dl / max(neg_d2l, 1e-10)
    end
    z
end
