# ─── Kalman Filter/Smoother ──────────────────────────────────────────
#
# Standard Kalman filtering and smoothing for time-varying linear Gaussian
# state-space models:
#   x_n = Q x_{n-1} + R^{1/2} ε_n           (state transition)
#   z_n = W x_n + d + V^{1/2} η_n           (measurement)
#
# These operate on a SINGLE state variable (one block).
# The probabilistic ODE solver applies them independently to each variable.
#
# Reference: standard Kalman filter, Rauch-Tung-Striebel smoother


"""
    kalman_update(μ_pred, Σ_pred, z, d, W, V)

One update step: p(x_n | z_{0:n}) from p(x_n | z_{0:n-1}).

- `z`: measurement vector
- `d`: measurement offset
- `W`: measurement matrix
- `V`: measurement noise variance

Returns `(μ_filt, Σ_filt)`.
"""
function kalman_update(μ_pred::AbstractVector, Σ_pred::AbstractMatrix,
                       z::AbstractVector, d::AbstractVector,
                       W::AbstractMatrix, V::AbstractMatrix)
    # Innovation
    ν = z - W * μ_pred - d
    S = W * Σ_pred * W' + V  # innovation covariance
    # Kalman gain
    K = Σ_pred * W' / S
    # Update
    μ_filt = μ_pred + K * ν
    Σ_filt = Σ_pred - K * W * Σ_pred
    # Symmetrize
    Σ_filt = 0.5 * (Σ_filt + Σ_filt')
    μ_filt, Σ_filt
end

"""
    kalman_forecast(μ_pred, Σ_pred, d, W, V)

Forecast the measurement: p(z_n | z_{0:n-1}).

Returns `(μ_fore, Σ_fore)`.
"""
function kalman_forecast(μ_pred::AbstractVector, Σ_pred::AbstractMatrix,
                         d::AbstractVector, W::AbstractMatrix,
                         V::AbstractMatrix)
    μ_fore = W * μ_pred + d
    Σ_fore = W * Σ_pred * W' + V
    μ_fore, Σ_fore
end


"""
    logpdf_mvn(x, μ, Σ)

Log-density of the multivariate normal N(μ, Σ) evaluated at `x`, computed
over the FULL dimension `n = length(x)` (a fixed reference measure).

A small adaptive jitter is added to `Σ` to keep it positive-definite for the
Cholesky factorization. Evaluating every call at the same dimension `n` is
essential when log-densities are subtracted (e.g. the DALTON joint − marginal
identity): a previous eigen-truncation to a data-dependent rank `k` made the
`-k/2·log(2π)` and log-determinant terms fail to cancel.
"""
function logpdf_mvn(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix)
    n = length(x)
    r = x - μ
    Σs = Symmetric(Matrix(Σ))
    # Adaptive jitter scaled to the matrix magnitude.
    base = max(maximum(diag(Σs)), 1.0)
    jit = 0.0
    C = nothing
    for _ in 1:8
        F = cholesky(Σs + jit * I, check=false)
        if issuccess(F)
            C = F
            break
        end
        jit = jit == 0.0 ? 1e-12 * base : jit * 10
    end
    if C === nothing
        # Last resort: eigen-based pseudo-density on the full dimension.
        w, v = eigen(Σs)
        w = max.(w, 1e-12 * base)
        z = v' * r
        return -0.5 * sum(z .^ 2 ./ w) - 0.5 * sum(log.(w)) - 0.5 * n * log(2π)
    end
    logdet_Σ = 2 * sum(log, diag(C.U))
    quad = sum(abs2, C.L \ r)
    -0.5 * quad - 0.5 * logdet_Σ - 0.5 * n * log(2π)
end
