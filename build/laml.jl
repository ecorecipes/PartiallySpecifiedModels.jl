# Laplace Approximate Marginal Likelihood (LAML) for smoothing parameter estimation
#
# Implements the method from:
#   Wood, Pya & Säfken (2016) "Smoothing parameter and model selection
#   for general smooth models", JASA 111(516), 1548-1575.
#   Wood & Fasiolo (2017) "A generalized Fellner-Schall method for smoothing
#   parameter optimization", Statistics and Computing 27(3), 759-774.
#
# For Gaussian data with unknown σ², this is equivalent to profiled REML:
#   V_REML(ρ) = -(n-Mp)/2 log(σ̂²) + ½ log|S^λ|_+ - ½ log|H| + const
#
# For general likelihoods (Poisson, NegBin, Custom):
#   V(ρ) = ℓ(β̂) - ½ β̂'S^λ β̂ + ½ log|S^λ|_+ - ½ log|H| + Mp/2 log(2π)

const RHO_MIN = -20.0
const RHO_MAX = 40.0   # exp(40) ≈ 2.4e17

# Variance function V(μ) for each family (used in Pearson dispersion)
_variance_function(::Gaussian, mu) = 1.0
_variance_function(::Poisson, mu) = mu
_variance_function(fam::NegativeBinomial, mu) = mu + mu^2 / fam.theta
function _variance_function(fam::CustomLikelihood, mu)
    # V(μ) = 1/(-∂²ℓ/∂μ²)
    neg_d2l = -ForwardDiff.derivative(
        μ -> ForwardDiff.derivative(μ2 -> fam.loglik_scalar(0.0, μ2), μ), mu)
    max(1.0 / max(neg_d2l, 1e-20), 1e-10)
end

# ─── Penalty matrix assembly ──────────────────────────────────────

"""
    build_S_lambda(S_list, offsets, nknots_list, rho, n_p)

Build total penalty S^λ = Σ_k exp(ρ_k) S_k embedded in n_p × n_p.
"""
function build_S_lambda(S_list::Vector{Matrix{Float64}},
                        offsets::Vector{Int}, nknots_list::Vector{Int},
                        rho::AbstractVector, n_p::Int)
    lambda = exp.(rho)
    S_lambda = zeros(n_p, n_p)
    for l in eachindex(rho)
        nk = nknots_list[l]
        off = offsets[l]
        for i in 1:nk, j in 1:nk
            S_lambda[off + i, off + j] += lambda[l] * S_list[l][i, j]
        end
    end
    S_lambda
end

# ─── LAML objective ───────────────────────────────────────────────

"""
    laml_objective(family, beta, J, W_irls, w_data, y, mu, S_list, offsets, nknots_list, rho, n_p)

Compute LAML objective V(ρ).

For Gaussian (profiled REML):
  V = -(n-Mp)/2 log(σ̂²) + ½ log|S^λ|_+ - ½ log|H|

For non-Gaussian:
  V = ℓ(β̂) - ½ β̂'S^λ β̂ + ½ log|S^λ|_+ - ½ log|H| + Mp/2 log(2π)

Returns `(V, H, S_lambda, sigma2)`.
"""
function laml_objective(family::AbstractLikelihood,
                        beta::AbstractVector, J::AbstractMatrix,
                        W_irls::AbstractVector, w_data::AbstractVector,
                        y::AbstractVector, mu::AbstractVector,
                        S_list::Vector{Matrix{Float64}},
                        offsets::Vector{Int}, nknots_list::Vector{Int},
                        rho::AbstractVector, n_p::Int)
    n = length(y)

    S_lambda = build_S_lambda(S_list, offsets, nknots_list, rho, n_p)

    # Working Hessian: H = J'W̃J + S^λ  (W̃ = IRLS weights, pre-computed)
    JWJ = J' * Diagonal(W_irls) * J
    H = JWJ + S_lambda

    pen = dot(beta, S_lambda * beta)

    log_det_S_plus = _log_det_plus(S_lambda)
    log_det_H = _log_det_pd(H)

    # Number of unpenalized parameters
    total_rank = sum(_rank_penalty(S_list[l]) for l in eachindex(S_list))
    Mp = n_p - total_rank

    if family isa Gaussian
        RSS = sum(w_data[i] * (y[i] - mu[i])^2 for i in 1:n)
        sigma2 = max((RSS + pen) / n, 1e-30)
        n_eff = n - Mp
        V = -0.5 * n_eff * log(sigma2) + 0.5 * log_det_S_plus - 0.5 * log_det_H
    else
        ll = log_likelihood(family, y, mu, w_data)
        sigma2 = 1.0
        V = ll - 0.5 * pen + 0.5 * log_det_S_plus - 0.5 * log_det_H + 0.5 * Mp * log(2π)
    end

    V, H, S_lambda, sigma2
end

# ─── LAML gradient ────────────────────────────────────────────────

"""
    laml_gradient(family, beta, J, W, y, mu, S_list, offsets, nknots_list,
                  rho, n_p, H, S_lambda, sigma2)

Gradient ∂V/∂ρ_k.

For Gaussian:  ∂V/∂ρ_k = -½ λ_k β̂'S_k β̂/σ̂² + ½ rank(S_k) - ½ tr(H⁻¹ λ_k S_k)
For others:    ∂V/∂ρ_k = -½ λ_k β̂'S_k β̂ + ½ rank(S_k) - ½ tr(H⁻¹ λ_k S_k)
"""
function laml_gradient(family::AbstractLikelihood,
                       beta::AbstractVector,
                       S_list::Vector{Matrix{Float64}},
                       offsets::Vector{Int}, nknots_list::Vector{Int},
                       rho::AbstractVector, n_p::Int,
                       H::AbstractMatrix, sigma2::Float64)
    m = length(rho)
    lambda = exp.(rho)
    grad = zeros(m)
    H_inv = _safe_inv(H)

    for k in 1:m
        nk = nknots_list[k]
        off = offsets[k]

        beta_k = @view beta[off+1:off+nk]
        bSb = dot(beta_k, S_list[k] * beta_k)
        rk = _rank_penalty(S_list[k])

        # tr(H⁻¹ λ_k S_k)
        tr_val = 0.0
        for i in 1:nk, j in 1:nk
            tr_val += H_inv[off + i, off + j] * S_list[k][j, i]
        end
        tr_val *= lambda[k]

        if family isa Gaussian
            grad[k] = -0.5 * lambda[k] * bSb / sigma2 + 0.5 * rk - 0.5 * tr_val
        else
            grad[k] = -0.5 * lambda[k] * bSb + 0.5 * rk - 0.5 * tr_val
        end
    end
    grad
end

# ─── LAML Hessian ─────────────────────────────────────────────────

"""
    laml_hessian(family, beta, S_list, offsets, nknots_list,
                 rho, n_p, H, sigma2)

Expected Hessian ∂²V/∂ρ_j∂ρ_k (Wood et al. approximation).
"""
function laml_hessian(family::AbstractLikelihood,
                      beta::AbstractVector,
                      S_list::Vector{Matrix{Float64}},
                      offsets::Vector{Int}, nknots_list::Vector{Int},
                      rho::AbstractVector, n_p::Int,
                      H::AbstractMatrix, sigma2::Float64)
    m = length(rho)
    lambda = exp.(rho)
    hess = zeros(m, m)
    H_inv = _safe_inv(H)

    # Precompute H⁻¹ λ_k S_k (full)
    HinvS = Vector{Matrix{Float64}}(undef, m)
    for k in 1:m
        nk = nknots_list[k]
        off = offsets[k]
        S_k_full = zeros(n_p, n_p)
        for i in 1:nk, j in 1:nk
            S_k_full[off + i, off + j] = lambda[k] * S_list[k][i, j]
        end
        HinvS[k] = H_inv * S_k_full
    end

    for j in 1:m, k in j:m
        tr_val = sum(HinvS[j][i, l] * HinvS[k][l, i] for i in 1:n_p, l in 1:n_p)

        if j == k
            nk = nknots_list[k]
            off = offsets[k]
            beta_k = @view beta[off+1:off+nk]
            bSb = dot(beta_k, S_list[k] * beta_k)
            tr_single = tr(HinvS[k])
            if family isa Gaussian
                hess[k, k] = -0.5 * lambda[k] * bSb / sigma2 - 0.5 * tr_single + 0.5 * tr_val
            else
                hess[k, k] = -0.5 * lambda[k] * bSb - 0.5 * tr_single + 0.5 * tr_val
            end
        else
            hess[j, k] = 0.5 * tr_val
            hess[k, j] = hess[j, k]
        end
    end
    hess
end

# ─── Fellner-Schall + Newton solver ──────────────────────────────

"""
    estimate_smoothing_params(J, W_irls, w_data, y, mu, beta, S_list, offsets, nknots_list, n_p;
                              family, rho_init, maxiter, tol, verbose)

Estimate smoothing parameters by maximizing LAML using two phases:
1. Fellner-Schall EM-type updates (globally stable)
2. Newton refinement with regularized Hessian

`W_irls` are the pre-computed IRLS working weights (= w_data / V(μ)).
`w_data` are the original data weights (for log-likelihood evaluation).

Returns `(lambda, edf)` where lambda[k] are the smoothing parameters.
"""
function estimate_smoothing_params(J::AbstractMatrix, W_irls::AbstractVector,
                                   w_data::AbstractVector,
                                   y::AbstractVector, mu::AbstractVector,
                                   beta::AbstractVector,
                                   S_list::Vector{Matrix{Float64}},
                                   offsets::Vector{Int}, nknots_list::Vector{Int},
                                   n_p::Int;
                                   family::AbstractLikelihood=Gaussian(),
                                   rho_init::Union{Nothing,Vector{Float64}}=nothing,
                                   sigma2_max::Float64=Inf,
                                   maxiter::Int=50, tol::Float64=1e-6,
                                   verbose::Bool=false)
    m = length(S_list)
    n = length(y)

    # Initialize ρ.  Fellner-Schall can converge to under-smoothing local
    # minima when started from very negative ρ (tiny λ).  Reset to ρ=0
    # (λ=1) when the initial values are all very small, providing a
    # moderate starting point.
    if rho_init !== nothing
        rho = clamp.(rho_init, RHO_MIN, RHO_MAX)
        if all(r -> r < -10.0, rho)
            rho .= 0.0
        end
    else
        rho = zeros(m)
    end

    if verbose
        println("LAML init: ρ = ", round.(rho, digits=3))
    end

    ranks = [_rank_penalty(S_list[k]) for k in 1:m]
    total_rank = sum(ranks)
    Mp = n_p - total_rank

    # ─── Phase 1: Fellner-Schall ────────────────────────────────
    n_fs = min(maxiter, 30)
    lambda = exp.(rho)

    for fs_iter in 1:n_fs
        lambda_old = copy(lambda)

        S_lambda = build_S_lambda(S_list, offsets, nknots_list, log.(lambda), n_p)
        JWJ = J' * Diagonal(W_irls) * J
        H = JWJ + S_lambda
        H_inv = _safe_inv(H)

        # Profiled scale for Gaussian; unit scale for non-Gaussian.
        # When sigma2_max is finite, cap σ² to prevent oversmoothing from
        # a poor initial fit that inflates the profiled residual variance.
        sigma2 = if family isa Gaussian
            RSS = sum(w_data[i] * (y[i] - mu[i])^2 for i in 1:n)
            pen = dot(beta, S_lambda * beta)
            profiled = max((RSS + pen) / n, 1e-30)
            min(profiled, sigma2_max)
        else
            1.0
        end

        all_converged = true
        for k in 1:m
            nk = nknots_list[k]
            off = offsets[k]
            beta_k = @view beta[off+1:off+nk]
            bSb = dot(beta_k, S_list[k] * beta_k)

            # τ_k = λ_k tr(H⁻¹ S_k)
            tau_k = 0.0
            for i in 1:nk, j in 1:nk
                tau_k += H_inv[off + i, off + j] * S_list[k][j, i]
            end
            tau_k *= lambda[k]

            edf_k = ranks[k] - tau_k

            lambda_new = if bSb > 1e-30 && edf_k > 0
                sigma2 * edf_k / bSb
            else
                lambda[k]
            end
            lambda_new = clamp(lambda_new, exp(RHO_MIN), exp(RHO_MAX))

            if abs(log(lambda_new) - log(max(lambda[k], 1e-30))) > tol
                all_converged = false
            end
            lambda[k] = lambda_new
        end

        if verbose && (fs_iter <= 5 || fs_iter % 10 == 0 || all_converged)
            sigma_str = family isa Gaussian ? @sprintf(" σ̂²=%.3e", sigma2) : ""
            println("LAML-FS iter $fs_iter:$sigma_str λ = ",
                    [round(l, sigdigits=4) for l in lambda])
        end

        if all_converged
            if verbose; println("LAML-FS converged at iteration $fs_iter"); end
            break
        end
    end

    rho .= clamp.(log.(lambda), RHO_MIN, RHO_MAX)

    # ─── Phase 2: Newton refinement ─────────────────────────────
    MAX_STEP = 5.0
    V_prev = -Inf
    n_newton = max(0, maxiter - n_fs)

    for iter in 1:min(n_newton, 20)
        V, H, S_lambda, sigma2 = laml_objective(family, beta, J, W_irls, w_data, y, mu,
                                                 S_list, offsets, nknots_list, rho, n_p)
        if !isfinite(V)
            if verbose; println("LAML-Newton: non-finite V, stopping"); end
            break
        end

        grad = laml_gradient(family, beta, S_list, offsets,
                             nknots_list, rho, n_p, H, sigma2)
        hess = laml_hessian(family, beta, S_list, offsets,
                            nknots_list, rho, n_p, H, sigma2)

        if verbose
            println("LAML-Newton iter $iter: V=$(@sprintf("%.6e", V)) " *
                    "|grad|=$(@sprintf("%.3e", norm(grad)))")
        end

        if norm(grad) < tol; break; end
        if iter > 1 && abs(V - V_prev) < tol * max(1.0, abs(V)); break; end
        V_prev = V

        # Regularized Newton step
        neg_hess = -hess
        min_eig = minimum(eigvals(Symmetric(neg_hess)))
        if min_eig < 1e-8
            ridge = max(1e-8 - min_eig, 1e-6 * norm(neg_hess))
            for i in 1:m; neg_hess[i, i] += ridge; end
        end

        delta = try
            Symmetric(neg_hess) \ grad
        catch
            0.1 * grad
        end

        for k in 1:m
            delta[k] = clamp(delta[k], -MAX_STEP, MAX_STEP)
        end

        # Line search
        step = 1.0
        rho_new = clamp.(rho .+ step .* delta, RHO_MIN, RHO_MAX)
        V_new = try
            v, _, _, _ = laml_objective(family, beta, J, W_irls, w_data, y, mu,
                                         S_list, offsets, nknots_list, rho_new, n_p)
            v
        catch; -Inf end

        for _ in 1:20
            if isfinite(V_new) && V_new >= V; break; end
            step *= 0.5
            rho_new = clamp.(rho .+ step .* delta, RHO_MIN, RHO_MAX)
            V_new = try
                v, _, _, _ = laml_objective(family, beta, J, W_irls, w_data, y, mu,
                                             S_list, offsets, nknots_list, rho_new, n_p)
                v
            catch; -Inf end
        end

        if !isfinite(V_new) || V_new < V; break; end
        rho .= rho_new
    end

    # Final EDF computation
    theta = exp.(clamp.(rho, RHO_MIN, RHO_MAX))
    S_lambda_final = build_S_lambda(S_list, offsets, nknots_list, log.(theta), n_p)
    H_final = J' * Diagonal(W_irls) * J + S_lambda_final
    H_inv = _safe_inv(H_final)
    edf = tr(H_inv * (J' * Diagonal(W_irls) * J))

    theta, edf
end

# ─── Helper functions ─────────────────────────────────────────────

"""Log of product of positive eigenvalues of symmetric matrix."""
function _log_det_plus(S::AbstractMatrix)
    evals = eigvals(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(evals)), 1e-14)
    pos = filter(e -> e > tol, evals)
    isempty(pos) ? 0.0 : sum(log, pos)
end

"""Log determinant of a positive definite matrix (with regularization)."""
function _log_det_pd(H::AbstractMatrix)
    n = size(H, 1)
    maxd = maximum(abs.(diag(H)))
    H_reg = copy(H)
    for i in 1:n
        H_reg[i, i] += 1e-10 * maxd + 1e-15
    end
    try
        C = cholesky(Symmetric(H_reg))
        2.0 * sum(log.(diag(C.U)))
    catch
        evals = eigvals(Symmetric(H_reg))
        pos = filter(e -> e > 0, evals)
        isempty(pos) ? 0.0 : sum(log, pos)
    end
end

"""Rank of a penalty matrix (number of positive eigenvalues)."""
function _rank_penalty(S::AbstractMatrix)
    evals = eigvals(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(evals)), 1e-14)
    count(e -> e > tol, evals)
end

"""Safe matrix inverse via Cholesky with regularization fallback."""
function _safe_inv(H::AbstractMatrix)
    n = size(H, 1)
    maxd = maximum(abs.(diag(H)))
    H_reg = copy(H)
    for i in 1:n
        H_reg[i, i] += 1e-10 * maxd + 1e-15
    end
    try
        inv(cholesky(Symmetric(H_reg)))
    catch
        pinv(H_reg)
    end
end
