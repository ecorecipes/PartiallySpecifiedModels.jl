# Laplace Approximate Marginal Likelihood (LAML) for smoothing parameter estimation
#
# Implements the method from:
#   Wood, Pya & Säfken (2016) "Smoothing parameter and model selection
#   for general smooth models", JASA 111(516), 1548-1575.
#   Wood & Fasiolo (2017) "A generalized Fellner-Schall method for smoothing
#   parameter optimization", Biometrics 73(4), 1071-1081.
#
# For Gaussian data with unknown σ², this is equivalent to profiled REML:
#   V_REML(ρ) = -(n-Mp)/2 log(σ̂²) + ½ log|S^λ|_+ - ½ log|H| + const
#
# For general likelihoods (Poisson, NegBin, Custom):
#   V(ρ) = ℓ(β̂) - ½ β̂'S^λ β̂ + ½ log|S^λ|_+ - ½ log|H| + Mp/2 log(2π)

const RHO_MIN = -20.0
const RHO_MAX = 40.0   # exp(40) ≈ 2.4e17

# Variance function V(μ) for each family (used in Pearson dispersion)
_variance_function(::GaussianLikelihood, mu) = 1.0
_variance_function(::PoissonLikelihood, mu) = mu
_variance_function(fam::NegativeBinomialLikelihood, mu) = mu + mu^2 / fam.theta
_variance_function(fam::GammaLikelihood, mu) = fam.phi * max(abs(mu), 1e-6)^2
function _variance_function(fam::LogNormalLikelihood, mu)
    s2 = fam.sigma^2
    m = max(abs(mu), 1e-6)
    (exp(s2) - 1.0) * exp(s2) * m^2
end
function _variance_function(fam::BinomialLikelihood, mu)
    p = clamp(mu, 1e-6, 1.0 - 1e-6)
    fam.n_trials * p * (1.0 - p)
end
function _variance_function(fam::StudentTLikelihood, mu)
    ν = fam.df; s2 = fam.sigma^2
    ν > 2.0 ? s2 * ν / (ν - 2.0) : s2 * (ν + 1.0) / ν
end
function _variance_function(fam::TruncatedNormalLikelihood, mu)
    _, variance = _truncated_normal_moments(fam, mu)
    fam.sigma^2 * variance
end
# Mean of the RESPONSE, E[Y | μ] — the centre the Pearson dispersion must
# subtract.
#
# For every family whose mean parameter IS the mean this is the identity.
# `TruncatedNormal` is the exception: there `μ` is the LATENT normal
# location, and for Y ~ N(μ, σ²) truncated to Y ≥ a,
#     E[Y | Y ≥ a] = μ + σ·λ(ξ),   ξ = (μ − a)/σ,  λ(ξ) = φ(ξ)/Φ(ξ)
# (the inverse Mills ratio), while `_variance_function` above already
# returns the matching TRUNCATED variance σ²(1 − ξλ − λ²). Before this
# accessor the two halves of the Pearson statistic were on different
# footings — a truncated denominator under a latent-centred numerator —
# which added the non-negative bias term Σ wᵢσ²λ(ξᵢ)²/V(μᵢ) and so inflated
# φ̂ one-directionally, over-smoothing through the Fellner–Schall fixed
# point λ* = φ̂·edf/βᵀSβ. Since E[Y] → μ as ξ → ∞, the correction is a no-op
# on weakly truncated fits and only bites where the bound actually bites.
#
# `CustomLikelihood` keeps the identity deliberately: the mean of an
# arbitrary user kernel is not recoverable from `loglik_scalar`.
_family_mean(::GaussianLikelihood, mu) = mu
_family_mean(::PoissonLikelihood, mu) = mu
_family_mean(::NegativeBinomialLikelihood, mu) = mu
_family_mean(::GammaLikelihood, mu) = max(abs(mu), 1e-6)
_family_mean(fam::LogNormalLikelihood, mu) = max(abs(mu), 1e-6) * exp(0.5 * fam.sigma^2)
_family_mean(fam::BinomialLikelihood, mu) = fam.n_trials * clamp(mu, 1e-6, 1.0 - 1e-6)
_family_mean(::StudentTLikelihood, mu) = mu
_family_mean(::CustomLikelihood, mu) = mu
function _family_mean(fam::TruncatedNormalLikelihood, mu)
    first(_truncated_normal_moments(fam, mu))
end

function _variance_function(fam::CustomLikelihood, mu)
    # V(μ) ≈ 1/(-∂²ℓ/∂μ²) evaluated at y = μ. Evaluating the curvature at
    # the mean recovers the exact variance function for exponential-family
    # kernels (e.g. Poisson: -∂²ℓ|_{y=μ} = 1/μ ⇒ V = μ); the old fixed
    # y = 0 gave y-dependent curvatures a wrong or degenerate answer.
    neg_d2l = -ForwardDiff.derivative(
        μ -> ForwardDiff.derivative(μ2 -> fam.loglik_scalar(mu, μ2), μ), mu)
    clamp(1.0 / max(neg_d2l, 1e-20), 1e-10, 1e10)
end

# A truncated-normal location can legitimately be negative; reflecting it
# changes the distribution rather than regularizing an invalid count mean.
_pearson_variance(fam::AbstractLikelihood, mu) = _variance_function(fam, abs(mu))
_pearson_variance(fam::TruncatedNormalLikelihood, mu) = _variance_function(fam, mu)
_pearson_variance(fam::BinomialLikelihood, mu) = _variance_function(fam, mu)
_pearson_variance(fam::StudentTLikelihood, mu) = _variance_function(fam, mu)

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

For mixed problems containing unpenalized approximator blocks (e.g. a
`NeuralApproximator` with `penalty_weight = 0`), the Gaussian restricted dof
counts those blocks by the RANK of their (weighted) design columns rather
than by their raw parameter count: `n − Mp_null − rank(√W·J[:, U])`. This is
the classical REML restricted dof `n − rank(X)`; it discounts columns that
carry no independent information (the raw count charges a full dof to a
column that is identically zero or an exact copy of another). It is a
function of the design only — NOT of ρ — so `laml_gradient` remains the
exact derivative of this objective. Pure-spline problems are unaffected.

Returns `(V, H, S_lambda, sigma2)`.
"""
function laml_objective(family::AbstractLikelihood,
                        beta::AbstractVector, J::AbstractMatrix,
                        W_irls::AbstractVector, w_data::AbstractVector,
                        y::AbstractVector, mu::AbstractVector,
                        S_list::Vector{Matrix{Float64}},
                        offsets::Vector{Int}, nknots_list::Vector{Int},
                        rho::AbstractVector, n_p::Int)
    # n is the REML sample size and must count USABLE cells only (positive
    # weight, non-NaN datum) — masked cells carry no information, so
    # counting them inflates n_eff = n − Mp and biases σ̂² = (RSS+pen)/n_eff
    # low, which in turn biases λ. `length(y)` counts the masked
    # placeholders the callers leave in the flattened vectors.
    # Identical to `length(y)` for complete data.
    n = _n_usable(y, w_data)

    S_lambda = build_S_lambda(S_list, offsets, nknots_list, rho, n_p)

    # Working Hessian: H = J'W̃J + S^λ  (W̃ = IRLS weights, pre-computed)
    JWJ = J' * Diagonal(W_irls) * J
    H = JWJ + S_lambda

    pen = dot(beta, S_lambda * beta)

    # Exact for non-overlapping penalty blocks (guaranteed: by default one
    # block per approximator, and `build_penalty_matrices` rejects any
    # overlapping ranges a multi-block `penalty_blocks` type could declare):
    #   log|S_λ|₊ = Σ_k [ r_k·ρ_k + log|S_k|₊ ].
    # Eigen-decomposing the COMBINED S_λ with a relative tolerance let the
    # effective rank drop when λ's across blocks differed by ≳1e10,
    # de-synchronizing the objective from the fixed-rank gradient exactly
    # where the Newton line search explores.
    log_det_S_plus = 0.0
    for l in eachindex(S_list)
        log_det_S_plus += _rank_penalty(S_list[l]) * rho[l] +
                          _log_det_plus(S_list[l])
    end
    log_det_H = _log_det_pd(H)

    # Number of unpenalized parameters
    total_rank = sum(_rank_penalty(S_list[l]) for l in eachindex(S_list))
    Mp = n_p - total_rank
    # The non-Gaussian (Mp/2)·log 2π term below uses this raw Mp, while the
    # Gaussian branch's n_eff uses Mp_eff. They differ only when unpenalized
    # design columns are rank-deficient (mixed spline+NN); the difference is
    # ρ-constant, so λ̂ is unaffected — only the reported criterion VALUE
    # carries a constant offset in that corner.
    # …and the same count with unpenalized blocks charged by design rank
    # (equals Mp whenever those columns are of full rank; see below).
    Mp_eff = _restricted_dof_Mp(J, W_irls, offsets, nknots_list, n_p, total_rank)

    if family isa GaussianLikelihood
        # This branch IS the full Laplace criterion of the non-Gaussian
        # branch, specialized to Gaussian and with σ² profiled out — which is
        # why `LAML(criterion=:laplace)` needs no separate Gaussian path and
        # reduces EXACTLY to the current REML criterion. Proof: for
        # y ~ N(Jβ, φW⁻¹) with prior precision S̃ = S_λ/φ (the REML/mgcv
        # convention that makes β̂ = argmin RSS + β'S_λβ independent of φ),
        # the generic Laplace expression
        #   V = ℓ(β̂) − ½β̂'S̃β̂ + ½log|S̃|₊ − ½log|J'WJ/φ + S̃| + (Mp/2)log 2π
        # expands, using ℓ = −RSS/(2φ) − (n/2)log(2πφ), rank(S_λ) = p − Mp,
        # log|S_λ/φ|₊ = log|S_λ|₊ − (p−Mp)log φ and
        # log|(J'WJ+S_λ)/φ| = log|H| − p·log φ, to
        #   V = −(RSS+pen)/(2φ) − ((n−Mp)/2)log φ + ½log|S_λ|₊ − ½log|H|
        #       − ((n−Mp)/2)log 2π.
        # ∂V/∂φ = 0 gives φ̂ = (RSS+pen)/(n−Mp) — exactly `sigma2` below —
        # and substituting yields, up to ρ-independent constants,
        #   V = −((n−Mp)/2)log σ̂² + ½log|S_λ|₊ − ½log|H|,
        # the expression computed here (with n−Mp → n_eff as documented).
        RSS = 0.0
        for i in eachindex(y)
            _usable(y[i], w_data[i]) || continue
            RSS += w_data[i] * (y[i] - mu[i])^2
        end
        # Profiled REML scale uses the restricted dof (n − Mp), NOT n. Using
        # /n (the ML scale) leaves the analytic gradient inconsistent with V
        # by a factor (n−Mp)/n on the penalty term and biases toward
        # undersmoothing. (Wood 2011, "Fast stable REML".)
        #
        # Mixed-approximator correction: Mp = n_p − total_rank counts EVERY
        # parameter without a penalty block (e.g. NeuralApproximator weights
        # with penalty_weight = 0) as a REML fixed effect, including columns
        # that carry no independent information. Count the unpenalized block
        # by the rank of its weighted design instead — the classical REML
        # n − rank(X). See `_restricted_dof_Mp`.
        #
        # An EDF-based count, Σ_unpen diag(H⁻¹J'WJ), was tried and reverted:
        # `build_S_lambda` leaves the unpenalized columns of S_λ identically
        # zero, so H⁻¹J'WJ = I − H⁻¹S_λ has EXACTLY 1 on those diagonals and
        # the whole expression collapses to n − Mp in exact arithmetic. Any
        # departure from that came solely from the 1e-10 ridge inside
        # `_safe_inv` acting on a near-singular H — a numerical artifact,
        # not a dof — and it made n_eff a function of ρ, which
        # `laml_gradient` (which omits −½·(dn_eff/dρ)·(log σ̂² − 1)) does not
        # account for. The rank form is ρ-independent, so objective and
        # gradient stay consistent.
        #
        # The floor is applied ONCE and shared by σ̂² and V. Previously V used
        # the raw `n − Mp` while σ̂² used `max(n − Mp, 1)`; when the raw value
        # dropped below 1 the two disagreed and `laml_gradient` (which is
        # exact only when the two coincide) silently became wrong by their
        # ratio.
        n_eff = max(n - Mp_eff, 1.0)
        sigma2 = max((RSS + pen) / n_eff, 1e-30)
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
function _laml_objective_precomputed(family::AbstractLikelihood,
                                     beta::AbstractVector,
                                     JWJ::AbstractMatrix,
                                     S_list::Vector{Matrix{Float64}},
                                     offsets::Vector{Int},
                                     nknots_list::Vector{Int},
                                     ranks::Vector{Int},
                                     log_det_plus_k::Vector{Float64},
                                     rho::AbstractVector, n_p::Int,
                                     n_eff::Float64, Mp::Int,
                                     data_term::Float64)
    S_lambda = build_S_lambda(S_list, offsets, nknots_list, rho, n_p)
    H = JWJ + S_lambda
    pen = dot(beta, S_lambda * beta)

    log_det_S_plus = 0.0
    @inbounds for l in eachindex(S_list)
        log_det_S_plus += ranks[l] * rho[l] + log_det_plus_k[l]
    end
    log_det_H = _log_det_pd(H)

    if family isa GaussianLikelihood
        sigma2 = max((data_term + pen) / n_eff, 1e-30)
        V = -0.5 * n_eff * log(sigma2) + 0.5 * log_det_S_plus - 0.5 * log_det_H
    else
        sigma2 = 1.0
        V = data_term - 0.5 * pen + 0.5 * log_det_S_plus - 0.5 * log_det_H + 0.5 * Mp * log(2π)
    end
    V, H, S_lambda, sigma2
end

function laml_gradient(family::AbstractLikelihood,
                       beta::AbstractVector,
                       S_list::Vector{Matrix{Float64}},
                       offsets::Vector{Int}, nknots_list::Vector{Int},
                       rho::AbstractVector, n_p::Int,
                       H::AbstractMatrix, sigma2::Float64;
                       H_inv::Union{Nothing,AbstractMatrix}=nothing,
                       ranks::Union{Nothing,Vector{Int}}=nothing)
    m = length(rho)
    lambda = exp.(rho)
    grad = zeros(m)
    Hi = H_inv === nothing ? _safe_inv(H) : H_inv

    for k in 1:m
        nk = nknots_list[k]
        off = offsets[k]

        beta_k = @view beta[off+1:off+nk]
        bSb = dot(beta_k, S_list[k] * beta_k)
        rk = ranks === nothing ? _rank_penalty(S_list[k]) : ranks[k]

        # tr(H⁻¹ λ_k S_k)
        tr_val = 0.0
        Sk = S_list[k]
        @inbounds for i in 1:nk, j in 1:nk
            tr_val += Hi[off + i, off + j] * Sk[j, i]
        end
        tr_val *= lambda[k]

        if family isa GaussianLikelihood
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
Evaluated block-sparsely using O(m² nk³) active sub-blocks of `H⁻¹` instead
of materializing dense `n_p × n_p` penalty blocks.
"""
function laml_hessian(family::AbstractLikelihood,
                      beta::AbstractVector,
                      S_list::Vector{Matrix{Float64}},
                      offsets::Vector{Int}, nknots_list::Vector{Int},
                      rho::AbstractVector, n_p::Int,
                      H::AbstractMatrix, sigma2::Float64;
                      H_inv::Union{Nothing,AbstractMatrix}=nothing)
    m = length(rho)
    lambda = exp.(rho)
    hess = zeros(m, m)
    Hi = H_inv === nothing ? _safe_inv(H) : H_inv

    # Block-sparse trace: tr((H⁻¹ λ_j S_j)(H⁻¹ λ_k S_k))
    # = λ_j λ_k tr( (H⁻¹[r_k, r_j] S_j) (H⁻¹[r_j, r_k] S_k) )
    for j in 1:m
        nj = nknots_list[j]
        oj = offsets[j]
        rj = (oj + 1):(oj + nj)
        Sj = S_list[j]
        for k in j:m
            nk = nknots_list[k]
            ok = offsets[k]
            rk = (ok + 1):(ok + nk)
            Sk = S_list[k]

            if j == k
                Mjj = @view(Hi[rj, rj]) * Sj
                tr_single = lambda[j] * tr(Mjj)
                tr_sq = 0.0
                @inbounds for a in 1:nj, b in 1:nj
                    tr_sq += Mjj[a, b] * Mjj[b, a]
                end
                tr_val = lambda[j]^2 * tr_sq
                beta_k = @view beta[rj]
                bSb = dot(beta_k, Sj * beta_k)
                if family isa GaussianLikelihood
                    hess[j, j] = -0.5 * lambda[j] * bSb / sigma2 - 0.5 * tr_single + 0.5 * tr_val
                else
                    hess[j, j] = -0.5 * lambda[j] * bSb - 0.5 * tr_single + 0.5 * tr_val
                end
            else
                Mkj = @view(Hi[rk, rj]) * Sj   # nk × nj
                Mjk = @view(Hi[rj, rk]) * Sk   # nj × nk
                tr_sq = 0.0
                @inbounds for a in 1:nk, b in 1:nj
                    tr_sq += Mkj[a, b] * Mjk[b, a]
                end
                hess[j, k] = 0.5 * lambda[j] * lambda[k] * tr_sq
                hess[k, j] = hess[j, k]
            end
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

`criterion` selects the smoothing criterion for NON-Gaussian families
(Gaussian always takes the profiled-REML path, identical under both):
- `:working` (default): PQL-flavored — FS calibrated by the Pearson
  dispersion φ̂ of the working model (floored at 1), Newton skipped.
- `:laplace`: the full Laplace-approximate marginal likelihood of the
  actual family. The FS update becomes the generalized Fellner-Schall of
  Wood & Fasiolo (2017) — the SAME update with unit dispersion (φ = 1),
  since these families have fixed dispersion — and the Newton phase runs
  on the non-Gaussian branch of `laml_objective`/`laml_gradient` (unit
  scale), which is exactly the criterion FS ascends, so the two phases
  are consistent.

Returns `(lambda, edf)` where lambda[k] are the smoothing parameters.
"""
# Exact implicit-derivative Newton Hessian (Wood 2011, Theorem 1) when β̂(ρ)
# is re-solved at each ρ: adds the ∂β̂/∂ρ = −H⁻¹(λ_j S_j β̂) and (for Gaussian)
# profiled ∂σ̂²/∂ρ_j = (λ_j β̂'S_j β̂)/n_eff curvature terms to the frozen-β Hessian.
function _laml_newton_hessian(family::AbstractLikelihood,
                              beta::AbstractVector,
                              S_list::Vector{Matrix{Float64}},
                              offsets::Vector{Int}, nknots_list::Vector{Int},
                              rho::AbstractVector, n_p::Int,
                              H::AbstractMatrix, sigma2::Float64,
                              n_eff::Float64;
                              H_inv::Union{Nothing,AbstractMatrix}=nothing)
    Hi = H_inv === nothing ? _safe_inv(H) : H_inv
    hess = laml_hessian(family, beta, S_list, offsets, nknots_list, rho, n_p,
                        H, sigma2; H_inv=Hi)
    m = length(rho)
    lambda = exp.(rho)
    # g_k = λ_k S_k β̂ (embedded in ℝ^{n_p}, nonzero only on block k)
    # and bSb_k = λ_k β̂' S_k β̂.
    g_vecs = Vector{Vector{Float64}}(undef, m)
    lam_bSb = Vector{Float64}(undef, m)
    for k in 1:m
        nk = nknots_list[k]; off = offsets[k]
        rk = (off + 1):(off + nk)
        bk = @view beta[rk]
        Sk_bk = S_list[k] * bk
        g_vecs[k] = lambda[k] .* Sk_bk
        lam_bSb[k] = lambda[k] * dot(bk, Sk_bk)
    end
    scale_inv = family isa GaussianLikelihood ? (1.0 / sigma2) : 1.0
    for j in 1:m
        nj = nknots_list[j]; oj = offsets[j]; rj = (oj + 1):(oj + nj)
        for k in j:m
            nk = nknots_list[k]; ok = offsets[k]; rk = (ok + 1):(ok + nk)
            # Implicit ∂β̂/∂ρ term: + (1/σ̂²) g_jᵀ H⁻¹[rj, rk] g_k
            imp_beta = scale_inv * dot(g_vecs[j], @view(Hi[rj, rk]) * g_vecs[k])
            imp_sig = family isa GaussianLikelihood ?
                      (0.5 / (n_eff * sigma2^2)) * lam_bSb[j] * lam_bSb[k] : 0.0
            hess[j, k] += imp_beta + imp_sig
            if j != k
                hess[k, j] = hess[j, k]
            end
        end
    end
    hess
end

function estimate_smoothing_params(J::AbstractMatrix, W_irls::AbstractVector,
                                   w_data::AbstractVector,
                                   y::AbstractVector, mu::AbstractVector,
                                   beta::AbstractVector,
                                   S_list::Vector{Matrix{Float64}},
                                   offsets::Vector{Int}, nknots_list::Vector{Int},
                                   n_p::Int;
                                   family::AbstractLikelihood=GaussianLikelihood(),
                                   rho_init::Union{Nothing,Vector{Float64}}=nothing,
                                   sigma2_max::Float64=Inf,
                                   maxiter::Int=50, tol::Float64=1e-6,
                                   criterion::Symbol=:working,
                                   fixed_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
                                   verbose::Bool=false)
    m = length(S_list)
    # USABLE-cell count, not `length(y)` — see the note in `laml_objective`.
    # The two functions must agree exactly: `laml_objective` computes the
    # Newton-phase V from n − Mp_eff and this function computes the
    # Fellner–Schall scale from the same denominator.
    n = _n_usable(y, w_data)

    # Initialize ρ.  Fellner-Schall can converge to under-smoothing local
    # minima when started from very negative ρ (tiny λ). Clamp to a floor
    # of ρ = −10 (λ ≈ 4.5e-5) rather than resetting to ρ = 0: the old reset
    # silently replaced any deliberately small initial λ — including the
    # data-driven default λ = 1/tr(S) — with λ = 1, discarding the caller's
    # initialization entirely.
    if rho_init !== nothing
        rho = clamp.(rho_init, max(RHO_MIN, -10.0), RHO_MAX)
    else
        rho = zeros(m)
    end

    if verbose
        println("LAML init: ρ = ", round.(rho, digits=3))
    end

    ranks = [_rank_penalty(S_list[k]) for k in 1:m]
    total_rank = sum(ranks)
    # Parameters outside every penalty block (e.g. unpenalized NN weights)
    # are charged to the REML fixed-effect count by the RANK of their design
    # columns, not by their raw number (see `_restricted_dof_Mp`). This is
    # ρ-independent, so the FS scale here and the Newton objective in
    # `laml_objective` use the identical constant denominator. Equals Mp for
    # pure-spline problems, whose code path — and results — are unchanged.
    Mp_eff = _restricted_dof_Mp(J, W_irls, offsets, nknots_list, n_p, total_rank)
    n_eff = max(n - Mp_eff, 1.0)
    # Gated on the Gaussian branch: `n_eff` is read ONLY by the Gaussian
    # REML scale below. Non-Gaussian families take the Pearson dispersion
    # path, whose denominator is `n − sum(ranks)` and never touches
    # `n_eff` — so warning there described a quantity the fit does not
    # use, and a healthy Poisson fit could emit an alarming irrelevant
    # message about an unidentified σ̂².
    if family isa GaussianLikelihood && Mp_eff >= n
        @warn "LAML: the unpenalized (fixed-effect) part of the model has " *
              "rank $(Mp_eff) ≥ n = $n data points, so the restricted " *
              "residual degrees of freedom are exhausted and the REML " *
              "scale denominator is held at its floor of 1. σ̂² is NOT " *
              "identified here and the resulting smoothing parameters are " *
              "not trustworthy; reduce the number of unpenalized " *
              "parameters, add data, or set penalty_weight > 0." maxlog=1
    end

    # ─── Phase 1: Fellner-Schall ────────────────────────────────
    n_fs = min(maxiter, 30)
    lambda = exp.(rho)

    # Working-model quantities for the coupled FS iteration: each λ update
    # is followed by a re-solve of the working-model coefficients
    # β̂(λ) = (J'WJ + S_λ)⁻¹ J'Wz (Wood & Fasiolo 2017 alternate FS and β̂
    # updates; iterating λ to a fixed point against a FROZEN β̂ solves a
    # different — frozen-β — criterion and can drift far from REML).
    # (z − η) is the family's working residual, NOT `y − μ`: the two differ
    # for TruncatedNormal, where using `y − μ` inverts the sign of the step
    # wherever the bound bites. `_working_residual` is `y .- mu` for every
    # other family, so this line is bit-identical off the truncated path.
    z_work = _working_residual(family, y, mu, w_data) .+ J * beta
    W_eff = similar(W_irls)
    z_eff = similar(z_work)
    @inbounds for i in eachindex(W_irls)
        if _usable(y[i], w_data[i]) && isfinite(W_irls[i]) && W_irls[i] > 0.0 && isfinite(z_work[i])
            W_eff[i] = W_irls[i]
            z_eff[i] = W_irls[i] * z_work[i]
        else
            W_eff[i] = 0.0
            z_eff[i] = 0.0
        end
    end
    JWJ = J' * Diagonal(W_eff) * J
    JWz = J' * z_eff
    beta_fs = copy(beta)

    for fs_iter in 1:n_fs
        lambda_old = copy(lambda)

        S_lambda = build_S_lambda(S_list, offsets, nknots_list, log.(lambda), n_p)
        H = JWJ + S_lambda
        H_inv = _safe_inv(H)
        # β̂ at the CURRENT λ (working model)
        beta_fs = H_inv * JWz

        # Profiled scale for Gaussian; for non-Gaussian the scale depends on
        # the criterion:
        # - :working — Pearson dispersion. For non-Gaussian families with
        #   identity link, IRLS weights (1/V(μ)) can be very small for large
        #   counts, causing λ to collapse.  The Pearson dispersion φ̂ acts as
        #   an effective scale that keeps the Fellner-Schall update
        #   well-calibrated in the quasi-likelihood sense.
        # - :laplace — unit dispersion. These families (Poisson; NegBin and
        #   TruncatedNormal conditional on their fixed shape parameters) have
        #   φ ≡ 1, and with φ = 1 the update below IS the generalized
        #   Fellner-Schall update of Wood & Fasiolo (2017) for the full
        #   Laplace criterion: λ_k ← (r_k − λ_k tr(H⁻¹S_k)) / β̂'S_kβ̂ with the
        #   family working weights W̃ inside H. It therefore (approximately)
        #   ascends exactly the objective the Newton phase refines.
        # When sigma2_max is finite, cap to prevent oversmoothing.
        sigma2 = if family isa GaussianLikelihood
            r_work = z_work .- J * beta_fs
            RSS = 0.0
            for i in eachindex(y)
                _usable(y[i], w_data[i]) || continue
                RSS += w_data[i] * r_work[i]^2
            end
            pen = dot(beta_fs, S_lambda * beta_fs)
            # Restricted dof n − Mp_eff, the SAME constant the Newton-phase
            # objective uses (see `laml_objective`).
            profiled = max((RSS + pen) / n_eff, 1e-30)   # REML scale
            min(profiled, sigma2_max)
        elseif criterion === :laplace
            min(1.0, sigma2_max)
        else
            pearson = 0.0
            for i in eachindex(y)
                _usable(y[i], w_data[i]) || continue
                # Centre on E[Y|μ], NOT on μ: the denominator is the family's
                # variance function, so the numerator must use the family's
                # MEAN. Identity for every family but `TruncatedNormal`
                # (see `_family_mean`), where μ is the latent location.
                pearson += w_data[i] * (y[i] - _family_mean(family, mu[i]))^2 /
                           max(_pearson_variance(family, mu[i]), 1e-10)
            end
            phi = max(pearson / max(n - sum(ranks), 1), 1.0)
            min(phi, sigma2_max)
        end

        all_converged = true
        for k in 1:m
            if fixed_mask !== nothing && fixed_mask[k]
                continue
            end
            nk = nknots_list[k]
            off = offsets[k]
            beta_k = @view beta_fs[off+1:off+nk]
            bSb = dot(beta_k, S_list[k] * beta_k)

            # τ_k = λ_k tr(H⁻¹ S_k)
            tau_k = 0.0
            for i in 1:nk, j in 1:nk
                tau_k += H_inv[off + i, off + j] * S_list[k][j, i]
            end
            tau_k *= lambda[k]

            edf_k = ranks[k] - tau_k

            # ── Degenerate-update policy (mgcv's directions, bounded) ──
            # The Fellner–Schall fixed point is λ*_k = φ·edf_k / β'S_kβ, so
            # each degenerate case NAMES A LIMIT rather than an absence of
            # information:
            #   edf_k ≤ 0, β'S_kβ > 0  ⇒ λ* ≤ 0     — release the penalty
            #   edf_k > 0, β'S_kβ ≈ 0  ⇒ λ* = +∞    — tighten it
            #   both degenerate        ⇒ 0/0        — genuinely no information
            # mgcv's `efsudr` (R/gam.fit4.r) resolves them as r = 0 (λ → 0),
            # r = Inf caught by `r[!is.finite(r)] <- 1e6` (λ ×= 1e6), and
            # `r[a==0&bSb==0] <- 1` (λ unchanged). This code used to FREEZE λ
            # in the first two cases, which stops the search dead at whatever
            # point the numerics happened to degenerate.
            #
            # We adopt mgcv's DIRECTION where the fixed point it points at is
            # finite, and keep the hold where it is not. Concretely:
            #
            # `edf_k ≤ 0, β'S_kβ > 0` — ADOPTED, bounded to one decade.
            #   λ* is finite and BELOW the current λ_k, so stepping down both
            #   moves toward a real optimum and terminates there. mgcv jumps
            #   straight to λ = 0; we step a decade, because every `efsudr`
            #   step is followed by a refit and an `if (fit$REML<=old.reml)`
            #   test that halves the multiplier on failure, whereas this loop
            #   iterates λ internally against a FROZEN working model (J, W̃,
            #   z_work fixed for the whole call) and re-scores nothing — and
            #   `edf_k ≤ 0` is reachable from roundoff in tr(H⁻¹S_k) alone.
            #   Adding mgcv's re-score is unavailable only on the NON-GAUSSIAN
            #   `criterion = :working` path, where the FS update deliberately
            #   uses the Pearson dispersion while `laml_objective` uses unit
            #   scale (for Gaussian both use the same profiled σ̂² and the
            #   same n_eff, so a re-score IS available there and was simply
            #   not attempted — worth revisiting if this branch is ever
            #   widened), and scoring a step against a criterion the update is not
            #   ascending is the exact failure the Newton phase is skipped to
            #   avoid (see the note above Phase 2). One decade costs no
            #   reach: n_fs ≤ 30 steps × ln 10 = 69.1 exceeds
            #   RHO_MAX − RHO_MIN = 60. And the update is a fixed-point MAP
            #   on λ, not an increment, so the next ordinary step recomputes
            #   λ from scratch and a single bounded misstep is erased rather
            #   than compounded — measured on the unit fixture in the F4
            #   testset, one decade is enough to unstick the block and the
            #   ordinary update then walks λ back down on its own (1e6 → 1e5
            #   → … → 3.78e4). Note it has NOT converged there: λ is still
            #   falling ~1.5%/step at the n_fs = 30 cap, so the value the
            #   test pins is the iterate at the cap, not a fixed point.
            #
            # `edf_k > 0, β'S_kβ ≈ 0` — NOT ADOPTED; λ_k is held.
            #   mgcv sends this to r = Inf → λ ×= 1e6. The criterion does
            #   prefer a larger λ_k here (the gradient is ½·edf_k > 0 through
            #   the ½log|S_λ|₊ − ½log|H| terms), but there is NO finite fixed
            #   point to walk to, and the walk buys nothing while costing
            #   accuracy: with β'S_kβ = 0 the penalty λ_k·β'S_kβ is zero for
            #   EVERY λ_k, so the fitted β cannot improve, while the walk
            #   took λ into a regime where our own linear algebra lost the
            #   fit. That mechanism was the PCLS truncation's RELATIVE
            #   singular-value test (`src/pcls.jl`) back when it referenced
            #   σmax of the AUGMENTED matrix: σmax_aug grows like √λ·‖C‖
            #   while the data block stays O(1), so past λ ≈ 1e14 it started
            #   zeroing DATA-INFORMED directions in the penalty null space —
            #   exactly where a fit with β'S_kβ = 0 lives (measured then: 0
            #   directions truncated at 1.9e13, 1 at 1e14, 2 at 1e16).
            #   F8 RETARGETED that reference to the λ-free σ_max(W^½J), so
            #   this particular mechanism is gone: on the exponential-growth
            #   null(S) fixture the number of data-informed directions the
            #   solve retains is now 2 at every λ from 1e8 to 1e17, and
            #   ‖β̂ − β_true‖ is flat at 0.0016441 (spread 4.6e-8) across all
            #   nine decades — see the "F8 — PCLS truncation reference and
            #   the step trust region" testset. The escalation costs quoted
            #   further down were measured under the OLD reference and have
            #   NOT been re-measured; the branch is still held, but on the
            #   modelling argument below rather than on the linear algebra.
            #   NOT the `_safe_inv` ridge, which an earlier draft of this
            #   comment blamed: β never passes through `_safe_inv` (it comes
            #   from the augmented SVD solve, already the mgcv-style
            #   augmented least-squares form), and replacing that ridge with
            #   an absolute one while escalating makes F2 (a) 21,700× WORSE,
            #   because λ then climbs to 2.0e14 instead of stopping. The
            #   ridge is still worth fixing, but for the `stationarity` /
            #   gradient / Hessian diagnostics, not for fit quality.
            #   An earlier draft of this comment concluded that relaxing the
            #   truncation to 1e-13·σmax makes the escalating policy
            #   reproduce the hold policy to 8 significant figures, "so this
            #   branch is blocked by ONE CONSTANT, not by anything
            #   fundamental". F5 measured that and BOTH halves are wrong;
            #   see the "F8 — PCLS truncation reference and the step trust
            #   region" testset for the pinned numbers.
            #   (i) 1e-13 is NOT inert. It is the constant that defeats the
            #   guard the truncation exists for: at the `x -> 0` default
            #   initialization with u0 placed off a knot (so the dead
            #   Jacobian columns are ~1e-10 rather than exactly 0, which is
            #   why an on-knot fixture hides this), it restores ‖β̂‖ = 2.3e9
            #   against 0.28 at 1e-7 — the unrescuable step, in full.
            #   (ii) The λ-coupling was not ONLY a bug: it was silently doing
            #   three jobs, so the reference could not simply be swapped.
            #   Job 1, the rank guard, survives the change untouched — the
            #   guard was always a statement about the DATA block's
            #   conditioning. Job 2, a subspace trust region at high λ, was
            #   real: the λ-free reference alone failed 3 suite assertions
            #   ("LAML mixed spline+NN", data_loss 18534 against a < 500
            #   gate) and sent 3 of 12 seeds of a Lotka-Volterra GCV
            #   ensemble from data_loss 0.72–1.29 to 26.9–29.0. F8 gave that
            #   job its own explicit mechanism — a Levenberg-Marquardt
            #   damping ladder on the STEP, engaged only when the undamped
            #   step multiplies the penalized objective by more than 10, in
            #   `_pcls_step_contract`. Job 3, an implicit brake on
            #   λ escalation, turned out NOT to be a separate mechanism but
            #   an artifact of F5's rejected two-candidate rule: under the
            #   corrected reference alone the SCOP-spline SIR fixture in
            #   "known_params mixed with an approximator" reports λ̂ = 9.65e-6
            #   and β̂(0.05) = 0.38431, unchanged to 5 significant figures
            #   from the augmented reference; with the trust region as well,
            #   6.049e-7 and 0.39595 (truth 0.38940; deterministic across
            #   seeds and BLAS thread counts). With both in place the
            #   mixed spline+NN fit lands at data_loss 0.90 and all 12 GCV
            #   seeds in 0.19–2.26.
            #   The branch is nevertheless still held, for the modelling
            #   reason that always applied as well: even with exact linear
            #   algebra mgcv's branch parks λ_k on the RHO_MAX rail with
            #   edf → 5.6e-6 — the right REML answer when the truth lies in
            #   null(S), but a boundary solution carrying no usable
            #   uncertainty. That is a modelling judgement, and it should be
            #   made deliberately rather than inherited.
            #   Measured cost of escalating anyway (one decade per step, the
            #   tamest version), on fixtures whose truth lies in the penalty
            #   null space so this branch fires AT the optimum:
            #     - F2 (a), truth r ≡ 0.1: λ̂ 1.87e4 → 1.87e13, edf 2.00 →
            #       0.037, data loss 7.68e-4 → 3.60e-2 (47×), r(1) error
            #       0.05% → 13%, stationarity 2.9e-7 → 3.6;
            #     - F2 (c) smooth map, truth linear: stationarity
            #       1.5e-6 → 2.5e32;
            #     - the collocation linear map: data loss 1.04e-16 →
            #       2.93e-13, g(1) error 1.4e-9 → 7.8e-8.
            #   Holding parks λ_k at the last FINITE Fellner–Schall fixed
            #   point, which is where the criterion's gradient in this
            #   coordinate is already ~0: on F2 (a) the reported
            #   `stationarity` there is 2.9e-7, i.e. |∂V/∂ρ_k| ≈ 4e-7 against
            #   its ½·rank(S_k) = 1.5 normalizer.
            #
            # `both degenerate` — mgcv's `r[a==0&bSb==0] <- 1`: hold. Same
            # in both implementations.
            #
            # mgcv is itself inconsistent about the aggressive branches,
            # which is why they are treated as a direction rather than a
            # specification: the sibling `efsud` (gam.fit5 path, same file)
            # floors BOTH quantities at `tiny = eps^0.5` — `a <- pmax(tiny,
            # ...)`, `r <- a/pmax(tiny, bSb)` — and so also declines the 0/∞
            # step, and Wood left `## NOTE: double check scaling here` on
            # `efsudr`'s φ line.
            lambda_new = if bSb > 1e-30
                if edf_k > 0
                    lam_fs = sigma2 * edf_k / bSb
                    # mgcv's `r[!is.finite(r)] <- 1e6` guards the same hole
                    # (we hold rather than multiply, per the note above).
                    # Letting a non-finite value through would SILENTLY leave
                    # λ_k = NaN: every comparison against NaN is false, so
                    # the convergence test below would declare success on it.
                    isfinite(lam_fs) ? lam_fs : lambda[k]
                else
                    lambda[k] / 10.0          # λ* ≤ 0 — release, bounded
                end
            else
                lambda[k]                     # no finite λ*, or 0/0 — hold
            end
            lambda_new = clamp(lambda_new, exp(RHO_MIN), exp(RHO_MAX))

            if abs(log(lambda_new) - log(max(lambda[k], 1e-30))) > tol
                all_converged = false
            end
            lambda[k] = lambda_new
        end

        if verbose && (fs_iter <= 5 || fs_iter % 10 == 0 || all_converged)
            sigma_str = family isa GaussianLikelihood ? @sprintf(" σ̂²=%.3e", sigma2) : ""
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
    # Under :working, skip Newton for non-Gaussian: the Fellner-Schall update
    # uses Pearson dispersion φ̂ as effective scale, but the LAML gradient uses
    # unit scale. Running Newton would push λ back toward the unit-scale
    # optimum, undoing the FS calibration (the T11 lesson: never let the
    # monitored/refined objective and the update disagree about ρ).
    # For Gaussian, FS and Newton are consistent because both use the
    # profiled σ²; under :laplace they are consistent because both use unit
    # dispersion, so Newton runs there too.
    MAX_STEP = 5.0
    V_prev = -Inf
    n_newton = (family isa GaussianLikelihood || criterion === :laplace) ?
               max(0, maxiter - n_fs) : 0

    # The objective must be evaluated at the SAME coefficients β̂_fs as its
    # penalty term: passing the stale outer-loop μ paired the data term at
    # β_outer with pen(β_fs). Shift μ to the working-model fit so that
    # y − μ_fs = z_work − J β̂_fs, exactly the FS phase's working residuals.
    # For Gaussian this feeds the RSS; for non-Gaussian (:laplace) it is the
    # linearized mean fed to ℓ(y, μ) — an approximation to the nonlinear
    # model's mean at β̂_fs, polished by the outer IRLS re-linearization
    # (identical in spirit to holding J and W̃ frozen within this call).
    mu_fs = mu .+ J * (beta_fs .- beta)

    # ─── Block-orthogonal penalty eigenbasis (Wood 2011 §3 / Sl.repara) ──
    # Diagonalize each penalty block S_k = Q_k diag(d_k) Q_k' once so that
    # H_Q = Q'(JWJ)Q + diag(w_λ) keeps the penalty strictly diagonal. Zeroing
    # null-space eigenvalues ensures diagonal equilibration leaves null-space
    # coordinates at O(1) while scaling penalized coordinates by √λ_k, keeping
    # κ(D⁻¹ H_Q D⁻¹) = O(1) uniformly all the way to λ = e^30 (~10^13) with
    # zero Float64 cancellation in log|H|, H⁻¹, β̂, gradient, Hessian, and EDF.
    Q = Matrix{Float64}(I, n_p, n_p)
    d_blocks = Vector{Vector{Float64}}(undef, m)
    log_det_plus_k = zeros(Float64, m)
    for k in 1:m
        nk = nknots_list[k]
        off = offsets[k]
        rk = (off + 1):(off + nk)
        ef = eigen(Symmetric(S_list[k]))
        Q[rk, rk] .= ef.vectors
        dmax = maximum(abs, ef.values)
        dtol = max(1e-10 * dmax, 1e-14)
        dk = Float64[v > dtol ? v : 0.0 for v in ef.values]
        d_blocks[k] = dk
        s_ld = 0.0
        @inbounds for v in dk
            v > 0.0 && (s_ld += log(v))
        end
        log_det_plus_k[k] = s_ld
    end
    A_Q = Matrix{Float64}(Symmetric(Q' * JWJ * Q))
    b_Q = Q' * JWz
    J_Q = J * Q
    mu_base = mu .- J * beta

    function _solve_HQ(rho_cand::AbstractVector)
        w_lam = zeros(Float64, n_p)
        @inbounds for k in 1:m
            lam_k = exp(rho_cand[k])
            off = offsets[k]
            dk = d_blocks[k]
            for j in eachindex(dk)
                w_lam[off + j] = lam_k * dk[j]
            end
        end
        H_Q = copy(A_Q)
        @inbounds for i in 1:n_p
            H_Q[i, i] += w_lam[i]
        end
        Hs, d_sqrt = _equilibrate_pd(H_Q; rel_ridge=0.0)
        log_scale = 2.0 * sum(log, d_sqrt)
        C = cholesky(Symmetric(Hs), check=false)
        Hs_inv = nothing
        ld_s = 0.0
        if issuccess(C)
            Hs_inv = inv(C)
            ld_s = 2.0 * sum(log, diag(C.U))
        else
            ef_h = eigen(Symmetric(Hs))
            emax = maximum(abs, ef_h.values)
            etol = max(1e-10 * emax, 1e-12)
            pos = filter(e -> e > etol, ef_h.values)
            ld_s = isempty(pos) ? 0.0 : sum(log, pos)
            inv_vals = Float64[v > etol ? 1.0 / v : 0.0 for v in ef_h.values]
            Hs_inv = ef_h.vectors * Diagonal(inv_vals) * ef_h.vectors'
        end
        H_Q_inv = Matrix{Float64}(undef, n_p, n_p)
        @inbounds for j in 1:n_p
            dj = d_sqrt[j]
            for i in 1:n_p
                H_Q_inv[i, j] = Hs_inv[i, j] / (d_sqrt[i] * dj)
            end
        end
        return H_Q_inv, log_scale + ld_s, w_lam
    end

    if n_newton > 0
        # Phase 2 re-solves working β̂(ρ) = (JWJ + S_λ(ρ))⁻¹ JWz at every ρ
        # in the orthogonal Q-basis and uses Wood (2011) Theorem 1's exact
        # implicit-derivative Hessian.
        Mp_raw = n_p - total_rank
        free_idx = fixed_mask === nothing ? collect(1:m) : findall(!, fixed_mask)

        function _eval_working_V(rho_cand)
            HQi, ldH, w_lam = _solve_HQ(rho_cand)
            beta_Q = HQi * b_Q
            mu_c = mu_base .+ J_Q * beta_Q
            dt_c = if family isa GaussianLikelihood
                rss_c = 0.0
                @inbounds for i in eachindex(y)
                    _usable(y[i], w_data[i]) || continue
                    rss_c += w_data[i] * (y[i] - mu_c[i])^2
                end
                rss_c
            else
                log_likelihood(family, y, mu_c, w_data)
            end
            pen_blocks = zeros(Float64, m)
            pen_c = 0.0
            lds = 0.0
            @inbounds for l in 1:m
                off = offsets[l]
                nk = nknots_list[l]
                pl = 0.0
                for j in 1:nk
                    idx = off + j
                    pl += w_lam[idx] * beta_Q[idx]^2
                end
                pen_blocks[l] = pl
                pen_c += pl
                lds += ranks[l] * rho_cand[l] + log_det_plus_k[l]
            end
            if family isa GaussianLikelihood
                sig2_c = max((dt_c + pen_c) / n_eff, 1e-30)
                V_c = -0.5 * n_eff * log(sig2_c) + 0.5 * lds - 0.5 * ldH
            else
                sig2_c = 1.0
                V_c = dt_c - 0.5 * pen_c + 0.5 * lds - 0.5 * ldH + 0.5 * Mp_raw * log(2π)
            end
            V_c, HQi, w_lam, beta_Q, pen_blocks, sig2_c
        end

        for iter in 1:min(n_newton, 20)
            isempty(free_idx) && break
            V, HQi, w_lam, beta_Q, pen_blocks, sigma2 = _eval_working_V(rho)
            if !isfinite(V)
                if verbose; println("LAML-Newton: non-finite V, stopping"); end
                break
            end

            # Exact gradient and Wood (2011) Theorem 1 Hessian in the Q-basis
            scale_inv = family isa GaussianLikelihood ? (1.0 / sigma2) : 1.0
            grad_full = zeros(Float64, m)
            tau_vec = zeros(Float64, m)
            u_blocks = Vector{Vector{Float64}}(undef, m)
            for k in 1:m
                off = offsets[k]
                nk = nknots_list[k]
                tk = 0.0
                uk = Vector{Float64}(undef, nk)
                @inbounds for j in 1:nk
                    idx = off + j
                    tk += w_lam[idx] * HQi[idx, idx]
                    uk[j] = w_lam[idx] * beta_Q[idx]
                end
                tau_vec[k] = tk
                u_blocks[k] = uk
                grad_full[k] = 0.5 * (ranks[k] - tk - scale_inv * pen_blocks[k])
            end

            hess_full = zeros(Float64, m, m)
            for j in 1:m
                oj = offsets[j]; nj = nknots_list[j]
                uj = u_blocks[j]
                for k in j:m
                    ok = offsets[k]; nk = nknots_list[k]
                    uk = u_blocks[k]
                    # tr(H⁻¹ λ_j S_j H⁻¹ λ_k S_k) = Σ_{a∈Bj, b∈Bk} HQi[a,b]² w_lam[a] w_lam[b]
                    tr_jk = 0.0
                    imp_b = 0.0
                    @inbounds for a in 1:nj
                        ia = oj + a
                        wa = w_lam[ia]
                        wa == 0.0 && continue
                        ua = uj[a]
                        for b in 1:nk
                            ib = ok + b
                            wb = w_lam[ib]
                            wb == 0.0 && continue
                            hij = HQi[ia, ib]
                            tr_jk += (hij * hij) * wa * wb
                            imp_b += ua * hij * uk[b]
                        end
                    end
                    hval = 0.5 * tr_jk + scale_inv * imp_b
                    if j == k
                        hval -= 0.5 * tau_vec[j] + 0.5 * scale_inv * pen_blocks[j]
                    end
                    if family isa GaussianLikelihood
                        hval += (0.5 / (n_eff * sigma2^2)) * pen_blocks[j] * pen_blocks[k]
                    end
                    hess_full[j, k] = hval
                    hess_full[k, j] = hval
                end
            end

            grad = grad_full[free_idx]
            hess = hess_full[free_idx, free_idx]

            if verbose
                println("LAML-Newton iter $iter: V=$(@sprintf("%.6e", V)) " *
                        "|grad|=$(@sprintf("%.3e", norm(grad)))")
            end

            if norm(grad) < tol; break; end
            if iter > 1 && abs(V - V_prev) < tol * max(1.0, abs(V)); break; end
            V_prev = V

            # Regularized Newton step on free coordinates
            neg_hess = -hess
            min_eig = minimum(eigvals(Symmetric(neg_hess)))
            if min_eig < 1e-8
                ridge = max(1e-8 - min_eig, 1e-6 * norm(neg_hess))
                for i in eachindex(free_idx); neg_hess[i, i] += ridge; end
            end

            delta_free = try
                Symmetric(neg_hess) \ grad
            catch
                0.1 * grad
            end

            delta = zeros(m)
            for (fi, k) in enumerate(free_idx)
                delta[k] = clamp(delta_free[fi], -MAX_STEP, MAX_STEP)
            end

            # Line search with re-solved working β̂(ρ_new)
            step = 1.0
            rho_new = clamp.(rho .+ step .* delta, RHO_MIN, RHO_MAX)
            V_new = try; first(_eval_working_V(rho_new)); catch; -Inf end

            for _ in 1:20
                if isfinite(V_new) && V_new >= V; break; end
                step *= 0.5
                rho_new = clamp.(rho .+ step .* delta, RHO_MIN, RHO_MAX)
                V_new = try; first(_eval_working_V(rho_new)); catch; -Inf end
            end

            if !isfinite(V_new) || V_new < V; break; end
            rho .= rho_new
        end
    end

    # Final EDF computation in the orthogonal Q-basis (exact even at λ ~ e^30)
    rho_final = clamp.(rho, RHO_MIN, RHO_MAX)
    theta = exp.(rho_final)
    HQi_final, _, _ = _solve_HQ(rho_final)
    edf = 0.0
    @inbounds for j in 1:n_p, i in 1:n_p
        edf += HQi_final[i, j] * A_Q[j, i]
    end

    theta, edf
end

# ─── Helper functions ─────────────────────────────────────────────

"""
    _unpenalized_indices(offsets, nknots_list, n_p)

Indices of parameters not covered by any penalty block: approximators whose
`penalty_matrix` is `nothing` (e.g. `NeuralApproximator` with
`penalty_weight = 0`) or whose block was dropped for having fewer than 3
parameters (the `np < 3` gate in the default `penalty_blocks` method), plus
any local indices a multi-block `penalty_blocks` type leaves outside every
declared range. Returns an empty vector for pure-spline problems.
"""
function _unpenalized_indices(offsets::Vector{Int}, nknots_list::Vector{Int},
                              n_p::Int)
    covered = falses(n_p)
    for l in eachindex(offsets)
        covered[offsets[l]+1:offsets[l]+nknots_list[l]] .= true
    end
    findall(!, covered)
end

"""
    _restricted_dof_Mp(J, W_irls, offsets, nknots_list, n_p, total_rank) → Int

REML fixed-effect count used in the Gaussian restricted dof `n − Mp`.

`Mp = n_p − total_rank` charges one degree of freedom to EVERY parameter
outside a penalty block, including design columns that carry no independent
information (an identically-zero column, or an exact copy of another). This
returns instead

    Mp_eff = (penalty null-space dimension)  +  rank(√W · J[:, U]),

with `U` the unpenalized indices — the classical REML `n − rank(X)`. It
equals `Mp` exactly whenever the unpenalized columns are of full rank, so
pure-spline problems and well-conditioned mixed problems are unaffected.

Being a function of the design and the IRLS weights only, it does NOT depend
on ρ; `laml_gradient` is therefore the exact derivative of the objective that
uses it.
"""
function _restricted_dof_Mp(J::AbstractMatrix, W_irls::AbstractVector,
                            offsets::Vector{Int}, nknots_list::Vector{Int},
                            n_p::Int, total_rank::Int)
    unpen = _unpenalized_indices(offsets, nknots_list, n_p)
    isempty(unpen) && return n_p - total_rank
    Mp_null = (n_p - length(unpen)) - total_rank
    JU = Diagonal(sqrt.(max.(W_irls, 0.0))) * view(J, :, unpen)
    rank_U = try
        rank(JU)
    catch
        length(unpen)
    end
    Mp_null + rank_U
end

"""Log of product of positive eigenvalues of symmetric matrix."""
function _log_det_plus(S::AbstractMatrix)
    evals = eigvals(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(evals)), 1e-14)
    pos = filter(e -> e > tol, evals)
    isempty(pos) ? 0.0 : sum(log, pos)
end

"""
    _equilibrate_pd(H; rel_ridge=1e-12) -> (H_scaled, d_sqrt)

Symmetrically diagonally equilibrate `H` via `D = diag(sqrt(max(|H_ii|, 1e-30)))`
so that `H_scaled = D⁻¹ H D⁻¹ + rel_ridge * I` has unit diagonal scale.
Prevents an O(1e14) heavily-penalized smooth block from contaminating O(1)
blocks via an isotropic `1e-10 * maximum(diag(H))` ridge.
"""
function _equilibrate_pd(H::AbstractMatrix; rel_ridge::Float64=1e-12)
    n = size(H, 1)
    d_sqrt = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        hii = abs(Float64(H[i, i]))
        # Leave dead/zero diagonal columns unscaled (1.0) so rank-deficient
        # unpenalized blocks (e.g. inactive NN weights) are never amplified by
        # 1/1e-15 into O(1e14) roundoff blowups in `_safe_inv`.
        d_sqrt[i] = hii > 1e-14 ? sqrt(hii) : 1.0
    end
    Hs = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n
        dj = d_sqrt[j]
        for i in 1:n
            Hs[i, j] = Float64(H[i, j]) / (d_sqrt[i] * dj)
        end
        Hs[j, j] += rel_ridge
    end
    return Hs, d_sqrt
end

"""Log determinant of a positive definite matrix (diagonally equilibrated)."""
function _log_det_pd(H::AbstractMatrix)
    n = size(H, 1)
    n == 0 && return 0.0
    # Try unridged equilibrated Cholesky FIRST (Wood 2011): preserves small
    # spline null-space eigenvalues when λ_k ~ 1e12 without inflating log|H|.
    Hs, d_sqrt = _equilibrate_pd(H; rel_ridge=0.0)
    log_scale = 2.0 * sum(log, d_sqrt)
    C = cholesky(Symmetric(Hs), check=false)
    if issuccess(C)
        return log_scale + 2.0 * sum(log, diag(C.U))
    end
    for rr in (1e-14, 1e-12, 1e-10, 1e-8)
        Hs2 = copy(Hs)
        @inbounds for i in 1:n
            Hs2[i, i] += rr
        end
        C2 = cholesky(Symmetric(Hs2), check=false)
        if issuccess(C2)
            return log_scale + 2.0 * sum(log, diag(C2.U))
        end
    end
    evals = eigvals(Symmetric(Hs))
    pos = filter(e -> e > 0, evals)
    log_scale + (isempty(pos) ? 0.0 : sum(log, pos))
end

"""Rank of a penalty matrix (number of positive eigenvalues)."""
function _rank_penalty(S::AbstractMatrix)
    evals = eigvals(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(evals)), 1e-14)
    count(e -> e > tol, evals)
end

"""Safe matrix inverse via diagonally equilibrated Cholesky with eigendecomp pseudoinverse fallback."""
function _safe_inv(H::AbstractMatrix)
    n = size(H, 1)
    n == 0 && return zeros(0, 0)
    Hs, d_sqrt = _equilibrate_pd(H; rel_ridge=0.0)
    C = cholesky(Symmetric(Hs), check=false)
    Hs_inv = if issuccess(C)
        inv(C)
    else
        # Rank-deficient or numerically indefinite equilibrated matrix (e.g.,
        # dead unpenalized NN weights): invert strictly positive eigenmodes and
        # zero out dead null-space directions cleanly rather than dividing
        # roundoff noise by a 1e-14 ridge.
        ef = eigen(Symmetric(Hs))
        emax = maximum(abs, ef.values)
        etol = max(1e-10 * emax, 1e-12)
        inv_vals = Float64[v > etol ? 1.0 / v : 0.0 for v in ef.values]
        ef.vectors * Diagonal(inv_vals) * ef.vectors'
    end
    H_inv = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n
        dj = d_sqrt[j]
        for i in 1:n
            H_inv[i, j] = Hs_inv[i, j] / (d_sqrt[i] * dj)
        end
    end
    return H_inv
end
