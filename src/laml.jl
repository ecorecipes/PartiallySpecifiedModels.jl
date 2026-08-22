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
_variance_function(::Gaussian, mu) = 1.0
_variance_function(::Poisson, mu) = mu
_variance_function(fam::NegativeBinomial, mu) = mu + mu^2 / fam.theta
function _variance_function(fam::TruncatedNormal, mu)
    # V(μ) = σ²(1 - δ) where δ = λ(ξ)(ξ + λ(ξ)), ξ = (μ-a)/σ, λ = φ/Φ
    σ = fam.sigma; a = fam.lower
    ξ = (mu - a) / σ
    Φξ = max(_normcdf(ξ), 1e-15)
    λξ = _normpdf(ξ) / Φξ
    σ^2 * max(1.0 - λξ * (ξ + λξ), 0.01)
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

    # Exact for non-overlapping penalty blocks (one block per approximator):
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

    if family isa Gaussian
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
                                   criterion::Symbol=:working,
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
    if family isa Gaussian && Mp_eff >= n
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
    z_work = y .- mu .+ J * beta
    JWJ = J' * Diagonal(W_irls) * J
    JWz = J' * (W_irls .* z_work)
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
        sigma2 = if family isa Gaussian
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
                pearson += w_data[i] * (y[i] - mu[i])^2 /
                           max(_variance_function(family, abs(mu[i])), 1e-10)
            end
            phi = max(pearson / max(n - sum(ranks), 1), 1.0)
            min(phi, sigma2_max)
        end

        all_converged = true
        for k in 1:m
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
    n_newton = (family isa Gaussian || criterion === :laplace) ?
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

    for iter in 1:min(n_newton, 20)
        # Evaluate at the working-model optimum β̂(λ) from the FS phase —
        # not the stale outer-loop β. (Within Newton the β̂ is held fixed:
        # a small approximation, polished by the outer IRLS re-linearization.)
        V, H, S_lambda, sigma2 = laml_objective(family, beta_fs, J, W_irls, w_data, y, mu_fs,
                                                 S_list, offsets, nknots_list, rho, n_p)
        if !isfinite(V)
            if verbose; println("LAML-Newton: non-finite V, stopping"); end
            break
        end

        grad = laml_gradient(family, beta_fs, S_list, offsets,
                             nknots_list, rho, n_p, H, sigma2)
        hess = laml_hessian(family, beta_fs, S_list, offsets,
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
            v, _, _, _ = laml_objective(family, beta_fs, J, W_irls, w_data, y, mu_fs,
                                         S_list, offsets, nknots_list, rho_new, n_p)
            v
        catch; -Inf end

        for _ in 1:20
            if isfinite(V_new) && V_new >= V; break; end
            step *= 0.5
            rho_new = clamp.(rho .+ step .* delta, RHO_MIN, RHO_MAX)
            V_new = try
                v, _, _, _ = laml_objective(family, beta_fs, J, W_irls, w_data, y, mu_fs,
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

"""
    _unpenalized_indices(offsets, nknots_list, n_p)

Indices of parameters not covered by any penalty block: approximators whose
`penalty_matrix` is `nothing` (e.g. `NeuralApproximator` with
`penalty_weight = 0`) or whose block was dropped for having fewer than 3
parameters (see `build_penalty_matrices`). Returns an empty vector for
pure-spline problems.
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
