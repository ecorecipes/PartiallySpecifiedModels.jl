# Likelihood families for LAML/IRLS
#
# Each family provides:
#   log_likelihood(fam, y, mu, w) — total log-likelihood
#   irls_weights(fam, y, mu, w)  — IRLS working weights W̃
#
# Reference: Wood, Pya & Säfken (2016), Section 2.

# ─── Log-likelihood functions ───────────────────────────────────────

"""
    log_likelihood(fam, y, mu, w)

Total weighted log-likelihood: Σ_i w_i ℓ(y_i, μ_i).
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
        ll += w[i] * (y[i] > 0 ? y[i] * log(mu_i) - mu_i : -mu_i)
    end
    ll
end

function log_likelihood(fam::NegativeBinomial, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    θ = fam.theta
    ll = 0.0
    for i in eachindex(y)
        mu_i = max(mu[i], 1e-10)
        ll += w[i] * (y[i] * log(mu_i / (mu_i + θ)) + θ * log(θ / (mu_i + θ)))
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
