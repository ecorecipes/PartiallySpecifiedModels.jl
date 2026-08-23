# GCV solver — smoothing parameter selection via Generalized Cross-Validation
#
# An alternative to LAML (Fellner-Schall + Newton) that selects smoothing
# parameters λ by minimizing the GCV score:
#
#   GCV(λ) = n ‖W^½(z − Jβ̂)‖² / (n − γ·tr(A))²
#
# where A = J(J'WJ + S^λ)⁻¹J'W is the influence/hat matrix and γ ≥ 1 is
# an inflation factor guarding against under-smoothing (default 1.4,
# following Kim & Gu, 2004).
#
# The algorithm uses the same IRLS outer loop as the LAML solver but
# replaces the Fellner-Schall/Newton inner loop with golden-section search
# on log(λ) to minimize GCV.
#
# With `criterion=:ncv` the GCV score is replaced by neighbourhood
# cross-validation (NCV; Wood 2024, arXiv:2404.16490, eq. 2 with m=n,
# δ(k)={k}, α(k)=nei(k)): each usable observation i is predicted from the
# penalized working-model fit computed WITHOUT the rows in a temporal
# neighbourhood δ(i) around i (same observed component, time index within
# `ncv_width` of i's, i itself included):
#
#   NCV(λ) = (1/n_usable) Σ_i w_i (z_i − ẑ_i^{−δ(i)})²
#
# For the quadratic working model the omitted-neighbourhood coefficients
# are available exactly by a rank-|δ| Woodbury downdate of
# A = J'WJ + S^λ (the one-Newton-step update of Wood 2024 eq. 3-4, which
# is exact for a quadratic loss):
#
#   β^{−δ} = β̂ + A⁻¹ J_δ' (W_δ⁻¹ − J_δ A⁻¹ J_δ')⁻¹ (J_δ β̂ − z_δ)
#
# so ẑ_i^{−δ(i)} = J_i β^{−δ(i)} costs O(p·|δ|) triangular-solve work
# per point (two solves per column of J_δ'), reusing the Cholesky
# factor of A. NCV uses no γ inflation
# (`gamma` is ignored): leaving out the correlated neighbours is itself
# the guard against undersmoothing that γ patches over.

using LinearAlgebra: Diagonal, dot, tr, Symmetric, eigvals, cholesky, norm, eigen

# ─── GCV score computation ────────────────────────────────────────

"""
    gcv_score(J, W_irls, z, S_lambda, n_p, gamma)

Compute the GCV score for a given total penalty matrix `S_lambda`.

Returns `(gcv, beta_hat, rss_w)`:
- `gcv`: the GCV criterion value
- `beta_hat`: the penalized LS solution
- `rss_w`: weighted residual sum of squares
"""
function _gcv_score(J::AbstractMatrix, W_irls::AbstractVector,
                    z::AbstractVector, S_lambda::AbstractMatrix,
                    n::Int, gamma::Float64)
    JWJ = J' * Diagonal(W_irls) * J
    H = JWJ + S_lambda

    # Regularize for numerical stability
    maxd = maximum(abs.(diag(H)))
    H_reg = copy(H)
    n_p = size(H, 1)
    for i in 1:n_p
        H_reg[i, i] += 1e-12 * maxd + 1e-15
    end

    # Solve penalized LS: β̂ = (J'WJ + S^λ)⁻¹ J'Wz
    beta_hat = try
        cholesky(Symmetric(H_reg)) \ (J' * (W_irls .* z))
    catch
        H_reg \ (J' * (W_irls .* z))
    end

    # Weighted RSS: ||W^½(z - Jβ̂)||²
    # Iterate over the FULL residual vector (`n` is now the count of usable
    # cells, which is smaller than `length(r)` under masking and would
    # silently truncate the sum), and skip rows whose working weight is 0.
    # Skipping — rather than relying on the multiply — is what keeps a masked
    # row's pseudo-datum `z[i]` from contributing `0 * NaN = NaN`.
    r = z .- J * beta_hat
    rss_w = 0.0
    for i in eachindex(r)
        W_irls[i] > 0 || continue
        rss_w += W_irls[i] * r[i]^2
    end

    # tr(A) where A = J (J'WJ + S^λ)⁻¹ J'W
    H_inv = try
        inv(cholesky(Symmetric(H_reg)))
    catch
        pinv(H_reg)
    end
    trA = tr(H_inv * JWJ)

    # GCV = n * RSS_w / (n - γ·tr(A))²
    denom = n - gamma * trA
    if denom <= 0.0
        # Denominator non-positive ⟹ model saturated; return large score
        return (Inf, beta_hat, rss_w, trA)
    end
    gcv = n * rss_w / denom^2

    (gcv, beta_hat, rss_w, trA)
end

# ─── NCV score computation (Wood 2024) ───────────────────────────

"""
    _ncv_neighbourhood(i, n_times, W_irls, width) -> Vector{Int}

Flattened indices of the NCV deletion neighbourhood δ(i).

The data vectors are flattened obs-major (component-major): index
`k = (oi−1)·n_times + ti` holds time `ti` of observed component `oi`, so
each component's time series occupies one contiguous block and
index distance within the block IS time-index distance. δ(i) is every
cell of the SAME component whose time index is within `width` steps of
i's, restricted to usable rows (working weight > 0; masked cells never
enter a neighbourhood). The point i itself is a member whenever it is
usable — which is the only case in which callers ask for δ(i).
"""
function _ncv_neighbourhood(i::Int, n_times::Int,
                            W_irls::AbstractVector, width::Int)
    ti = mod1(i, n_times)         # time index within the component block
    base = i - ti                 # block offset of i's component
    delta = Int[]
    for t in max(1, ti - width):min(n_times, ti + width)
        k = base + t
        W_irls[k] > 0 && push!(delta, k)
    end
    delta
end

"""
    _ncv_loo_predictions(J, W_irls, z, A_chol, beta_hat, n_times, width)

Exact leave-neighbourhood-out predictions `ẑ_i^{−δ(i)}` for every usable
row i, via the rank-|δ| Woodbury downdate of `A = J'WJ + S^λ` (whose
Cholesky factorization is `A_chol`):

    β^{−δ} = β̂ + A⁻¹ J_δ' (W_δ⁻¹ − J_δ A⁻¹ J_δ')⁻¹ (J_δ β̂ − z_δ)
    ẑ_i^{−δ(i)} = J_i β^{−δ(i)}

This equals brute-force drop-the-rows refitting exactly (the one-step
Newton update of Wood 2024 eq. 3 is exact for the quadratic working
model). Returns a vector with `NaN` at masked rows, or `nothing` when a
downdated system is singular/indefinite (model saturated at this λ —
the caller scores that λ as `Inf`, mirroring `_gcv_score`'s denominator
guard).
"""
function _ncv_loo_predictions(J::AbstractMatrix, W_irls::AbstractVector,
                              z::AbstractVector, A_chol,
                              beta_hat::AbstractVector,
                              n_times::Int, width::Int)
    fitted = J * beta_hat
    zhat = fill(NaN, length(z))
    for i in eachindex(z)
        W_irls[i] > 0 || continue
        delta = _ncv_neighbourhood(i, n_times, W_irls, width)
        Jd = J[delta, :]                       # |δ| × p
        Q = A_chol \ Matrix(Jd')               # p × |δ| = A⁻¹ J_δ'
        P = Jd * Q                             # |δ| × |δ| = J_δ A⁻¹ J_δ'
        M = Diagonal(1.0 ./ W_irls[delta]) - P # W_δ⁻¹ − J_δ A⁻¹ J_δ'
        rhs = fitted[delta] .- z[delta]        # J_δ β̂ − z_δ
        s = try
            M \ rhs
        catch
            return nothing
        end
        pos = findfirst(==(i), delta)
        # J_i β^{−δ} = ẑ_i + (J_i A⁻¹ J_δ') s, and J_i A⁻¹ J_δ' is row
        # `pos` of P.
        v = fitted[i] + dot(view(P, pos, :), s)
        isfinite(v) || return nothing
        zhat[i] = v
    end
    zhat
end

"""
    _ncv_score(J, W_irls, z, S_lambda, n_times, width)

Compute the NCV score for a given total penalty matrix `S_lambda`:

    NCV(λ) = (1/n_usable) Σ_i w_i (z_i − ẑ_i^{−δ(i)})²

summed over usable rows only (working weight > 0); see
`_ncv_loo_predictions` for the exact Woodbury computation of
`ẑ_i^{−δ(i)}`. Returns `(ncv, beta_hat, rss_w, trA)` — the same shape as
`_gcv_score`, so the golden-section/grid/coordinate search machinery
drives either criterion unchanged.
"""
function _ncv_score(J::AbstractMatrix, W_irls::AbstractVector,
                    z::AbstractVector, S_lambda::AbstractMatrix,
                    n_times::Int, width::Int)
    JWJ = J' * Diagonal(W_irls) * J
    H = JWJ + S_lambda

    # Same numerical-stability ridge as _gcv_score
    maxd = maximum(abs.(diag(H)))
    H_reg = copy(H)
    n_p = size(H, 1)
    for i in 1:n_p
        H_reg[i, i] += 1e-12 * maxd + 1e-15
    end

    A_chol = try
        cholesky(Symmetric(H_reg))
    catch
        nothing
    end
    if A_chol === nothing
        # Not PD even after the ridge: saturated/degenerate at this λ.
        beta_hat = H_reg \ (J' * (W_irls .* z))
        return (Inf, beta_hat, NaN, NaN)
    end
    beta_hat = A_chol \ (J' * (W_irls .* z))

    # Weighted RSS and tr(A) kept for the uniform return shape (verbose
    # reporting); masked-row skipping as in _gcv_score.
    r = z .- J * beta_hat
    rss_w = 0.0
    for i in eachindex(r)
        W_irls[i] > 0 || continue
        rss_w += W_irls[i] * r[i]^2
    end
    trA = tr(A_chol \ JWJ)

    zhat = _ncv_loo_predictions(J, W_irls, z, A_chol, beta_hat,
                                n_times, width)
    zhat === nothing && return (Inf, beta_hat, rss_w, trA)

    V = 0.0
    cnt = 0
    for i in eachindex(z)
        W_irls[i] > 0 || continue
        V += W_irls[i] * (z[i] - zhat[i])^2
        cnt += 1
    end
    cnt == 0 && return (Inf, beta_hat, rss_w, trA)
    (V / cnt, beta_hat, rss_w, trA)
end

# ─── Golden-section search on log(λ) ─────────────────────────────

"""
    _golden_section_gcv(scorer, S_list, offsets, nknots_list, n_p,
                        lo, hi, tol; maxiter)

Minimize a CV score over a shared log(λ) using golden-section search.

`scorer(S_lambda)` must return `(score, beta, rss, trA)` — either
`_gcv_score` or `_ncv_score` partially applied to the current working
model. All approximator penalties are scaled by the same λ = exp(rho).
Returns `(best_rho, best_beta, best_score, best_trA)`.
"""
function _golden_section_gcv(scorer,
                             S_list::Vector{Matrix{Float64}},
                             offsets::Vector{Int}, nknots_list::Vector{Int},
                             n_p::Int,
                             lo::Float64, hi::Float64, tol::Float64;
                             maxiter::Int=100)
    gr = (sqrt(5.0) + 1.0) / 2.0  # golden ratio

    function eval_gcv(rho)
        rho_vec = fill(rho, length(S_list))
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rho_vec, n_p)
        gcv, beta, rss, trA = scorer(S_lam)
        (gcv, beta, rss, trA)
    end

    a, b = lo, hi
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    gc, betac, _, trAc = eval_gcv(c)
    gd, betad, _, trAd = eval_gcv(d)

    for _ in 1:maxiter
        if abs(b - a) < tol
            break
        end
        if gc < gd
            b = d
            d = c
            gd = gc
            betad = betac
            trAd = trAc
            c = b - (b - a) / gr
            gc, betac, _, trAc = eval_gcv(c)
        else
            a = c
            c = d
            gc = gd
            betac = betad
            trAc = trAd
            d = a + (b - a) / gr
            gd, betad, _, trAd = eval_gcv(d)
        end
    end

    # Return the best of c and d
    if gc <= gd
        return (c, betac, gc, trAc)
    else
        return (d, betad, gd, trAd)
    end
end

"""
    _grid_then_refine_gcv(scorer, S_list, offsets, nknots_list,
                          n_p, n_grid, tol)

Initial coarse grid search over log(λ) ∈ [RHO_MIN, RHO_MAX], then
golden-section refinement around the best grid point. `scorer` as in
`_golden_section_gcv`.

Returns `(best_rho, best_beta, best_score, best_trA)`.
"""
function _grid_then_refine_gcv(scorer,
                               S_list::Vector{Matrix{Float64}},
                               offsets::Vector{Int}, nknots_list::Vector{Int},
                               n_p::Int,
                               n_grid::Int, tol::Float64)
    rho_grid = range(RHO_MIN, RHO_MAX, length=n_grid)
    best_gcv = Inf
    best_idx = 1
    best_beta = zeros(n_p)
    best_trA = 0.0

    for (idx, rho) in enumerate(rho_grid)
        rho_vec = fill(rho, length(S_list))
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rho_vec, n_p)
        gcv, beta, _, trA = scorer(S_lam)
        if gcv < best_gcv
            best_gcv = gcv
            best_idx = idx
            best_beta = beta
            best_trA = trA
        end
    end

    # Refine with golden section around the best grid interval
    step = (RHO_MAX - RHO_MIN) / (n_grid - 1)
    lo = max(RHO_MIN, rho_grid[best_idx] - step)
    hi = min(RHO_MAX, rho_grid[best_idx] + step)

    rho_opt, beta_opt, gcv_opt, trA_opt = _golden_section_gcv(
        scorer, S_list, offsets, nknots_list, n_p, lo, hi, tol)

    # Keep the better of grid and refinement
    if gcv_opt < best_gcv
        return (rho_opt, beta_opt, gcv_opt, trA_opt)
    else
        return (Float64(rho_grid[best_idx]), best_beta, best_gcv, best_trA)
    end
end

"""
    _coordinate_gcv(scorer, S_list, offsets, nknots_list, n_p,
                    rho0, tol; sweeps=3)

Per-approximator CV score: coordinate descent over the vector ρ, minimizing
each component by golden section while the others are held fixed. Wood (2001)
treats λ as a VECTOR with one smoothing parameter per unknown function
(as does ddefit's gcv.c); a single shared λ mis-smooths whenever the
functions differ in scale or wiggliness. Started from the shared-λ optimum,
2–3 sweeps typically converge. `scorer` as in `_golden_section_gcv`
(either GCV or NCV — the multi-λ path is criterion-agnostic).
"""
function _coordinate_gcv(scorer,
                         S_list::Vector{Matrix{Float64}},
                         offsets::Vector{Int}, nknots_list::Vector{Int},
                         n_p::Int,
                         rho0::Vector{Float64}, tol::Float64;
                         sweeps::Int=3)
    m = length(S_list)
    rho = copy(rho0)
    gr = (sqrt(5.0) + 1.0) / 2.0

    eval_vec = function (rv)
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rv, n_p)
        scorer(S_lam)
    end

    best_gcv, best_beta, _, best_trA = eval_vec(rho)
    for _ in 1:sweeps
        improved = false
        for k in 1:m
            a, b = RHO_MIN, RHO_MAX
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            rc = copy(rho); rc[k] = c
            rd = copy(rho); rd[k] = d
            gc_, _, _, _ = eval_vec(rc)
            gd_, _, _, _ = eval_vec(rd)
            for _ in 1:60
                abs(b - a) < tol && break
                if gc_ < gd_
                    b, d, gd_ = d, c, gc_
                    c = b - (b - a) / gr
                    rc[k] = c
                    gc_, _, _, _ = eval_vec(rc)
                else
                    a, c, gc_ = c, d, gd_
                    d = a + (b - a) / gr
                    rd[k] = d
                    gd_, _, _, _ = eval_vec(rd)
                end
            end
            rho_k_new = (a + b) / 2
            rtrial = copy(rho); rtrial[k] = rho_k_new
            g_new, beta_new, _, trA_new = eval_vec(rtrial)
            if g_new < best_gcv - 1e-12
                rho = rtrial
                best_gcv, best_beta, best_trA = g_new, beta_new, trA_new
                improved = true
            end
        end
        improved || break
    end
    rho, best_beta, best_gcv, best_trA
end

# ─── Main GCV solve function ─────────────────────────────────────

"""
    SciMLBase.solve(prob::PSMProblem, alg::GCVSolver)

Fit a partially specified model using IRLS with GCV smoothing parameter
selection.

# Algorithm
For each IRLS iteration:
1. Evaluate model and compute finite-difference Jacobian
2. Form pseudodata z = y − f + J·β
3. Compute IRLS weights from current predictions
4. Select λ by minimizing the CV criterion — GCV(λ) by default, NCV(λ)
   when `alg.criterion == :ncv` — via grid search + golden-section
   refinement (and per-approximator coordinate descent when there are
   multiple smooth terms; both criteria drive the identical search
   machinery)
5. Solve penalized LS at optimal λ
6. Step contraction (backtracking)
7. Repeat until convergence

Returns a `PSMSolution`. `sol.convergence` is a NamedTuple
`(converged, iterations, reason, criterion, gcv, ncv)` — see the
`GCVSolver` and `PSMSolution` docstrings for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::GCVSolver)
    _validate_problem(prob, "GCVSolver")
    maxiters = alg.maxiters
    verbose  = alg.verbose
    gamma    = alg.gamma
    n_grid   = alg.n_grid
    tol      = alg.tol
    criterion = alg.criterion
    ncv_width = alg.ncv_width
    crit_name = uppercase(String(criterion))

    n_times = length(prob.data_times)
    n_obs   = length(prob.obs_to_state)
    n_data  = n_times * n_obs
    n_p     = n_total_params(prob)

    # Build penalty matrices per approximator
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)

    # Initialize λ (moderate default)
    theta = ones(m)

    # Flatten data into vectors (obs-major order: obs 1 times, obs 2 times, …),
    # enforcing the package masking convention exactly as the LAML solver
    # does: usable iff weight > 0 AND datum non-NaN; masked cells get weight
    # 0 and a finite placeholder. Every downstream use (penalized_objective →
    # log_likelihood, the IRLS pseudo-data, the GCV score) multiplies by the
    # weight, and IEEE `0 * NaN = NaN` would otherwise poison all of them.
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y = prob.data_values[ti, oi]
        wv = prob.data_weights[ti, oi]
        if _usable(y, wv)
            y_vec[k] = y
            w_vec[k] = wv
        end   # else keep the 0.0 placeholder with weight 0.0
        k += 1
    end
    # The GCV score's sample size must count USABLE cells only. GCV(λ) =
    # n·RSS_w/(n − γ·tr(A))² is a per-observation criterion: counting masked
    # cells in n inflates the residual dof and systematically undersmooths.
    # Equals n_data for complete data, so complete-data fits are unchanged.
    n_gcv = count(>(0), w_vec)
    n_gcv == 0 && error("GCVSolver: every observation is masked (all " *
        "data_weights are 0 or all data_values are NaN); there is nothing to fit.")

    # Evaluate model → flattened predictions
    function eval_model(p_eval)
        pred = simulate(prob, p_eval)
        f_tmp = zeros(n_data)
        local k = 1
        for oi in 1:n_obs, ti in 1:n_times
            f_tmp[k] = pred[ti, oi]
            k += 1
        end
        f_tmp, pred
    end

    # Build total penalty B = Σ θ_k S_k (embedded in n_p × n_p)
    function build_B(th)
        B = zeros(n_p, n_p)
        for l in 1:m
            off = uf_offsets[l]
            nk = uf_nk[l]
            for i in 1:nk, j in 1:nk
                B[off+i, off+j] += th[l] * S_list[l][i, j]
            end
        end
        B
    end

    # Penalized objective: -ℓ(y,μ) + ½β'Bβ
    function penalized_objective(p_eval, B)
        f_tmp, _ = try; eval_model(p_eval); catch; return Inf; end
        neg_ll = -log_likelihood(prob.likelihood, y_vec, f_tmp, w_vec)
        neg_ll + 0.5 * dot(p_eval, B * p_eval)
    end

    # PCLS step: truncated-SVD solve of the augmented system
    # [W^½J; C] β = [W^½z; 0] — see _pcls_augmented_solve in pcls.jl.
    # (Shared with the LAML solver; the SVD truncation guards against
    # exploding coefficients along numerically-null Jacobian directions
    # at poor initializations, and equals the plain QR solve when the
    # system is well-conditioned.)
    function pcls_step(J_mat, z_pseudo, th, w_irls)
        B = build_B(th)
        _pcls_augmented_solve(J_mat, z_pseudo, B, w_irls), B
    end

    # Step contraction: backtracking with explosive-step rescue — see
    # _pcls_step_contract in pcls.jl.
    step_contract(a_old, a_new, B) =
        _pcls_step_contract(penalized_objective, a_old, a_new, B)

    # Initialize
    beta  = build_initial_params(prob)
    J     = zeros(n_data, n_p)
    f_vec = zeros(n_data)
    dam   = fill(1e-8, n_p)

    if verbose
        extra = criterion === :ncv ? "width=$ncv_width (γ ignored)" : "γ=$gamma"
        println("$crit_name solver: $n_p params, $n_data data, " *
                "$m smooth terms, $extra")
    end

    f_vec, _ = eval_model(beta)
    compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)

    prev_obj = Inf
    gcv_val  = NaN

    # Honest convergence reporting: defaults describe loop exhaustion.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0

    for iter in 0:(maxiters - 1)
        conv_iters = iter + 1
        # Adapt GP kernel hyperparameters to the evolving fit
        iter >= 2 && _adapt_gp_approximators!(prob, beta)
        # Re-evaluate model + Jacobian
        f_vec_new, _ = try; eval_model(beta); catch e
            if verbose; println("Iter $iter: simulation failed ($e)"); end
            conv_reason = :early_break
            break
        end
        f_vec .= f_vec_new
        compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)

        # Compute IRLS weights from current predictions
        w_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)

        # Form pseudodata z = y − f + J·β
        z_pseudo = y_vec .- f_vec .+ J * beta

        # ── GCV/NCV smoothing parameter selection ──
        if m > 0
            # Partially apply the criterion to the current working model:
            # both scores are pure functions of S_lambda given (J, W, z),
            # so the same grid/golden-section/coordinate machinery drives
            # either. NCV ignores `gamma` (see GCVSolver docstring).
            scorer = criterion === :ncv ?
                (S_lam -> _ncv_score(J, w_irls, z_pseudo, S_lam,
                                     n_times, ncv_width)) :
                (S_lam -> _gcv_score(J, w_irls, z_pseudo, S_lam,
                                     n_gcv, gamma))

            best_rho, beta_gcv, gcv_val, trA = _grid_then_refine_gcv(
                scorer, S_list, uf_offsets, uf_nk,
                n_p, n_grid, tol)

            if m == 1
                theta .= exp(best_rho)
            else
                # Per-approximator λ (Wood 2001 treats λ as a vector):
                # refine each component by coordinate descent from the
                # shared-λ optimum.
                rho_vec, beta_gcv, gcv_val, trA = _coordinate_gcv(
                    scorer, S_list, uf_offsets, uf_nk,
                    n_p, fill(best_rho, m), tol)
                theta .= exp.(rho_vec)
            end

            if verbose && (iter <= 4 || iter % 10 == 0)
                println("  $crit_name iter $iter: λ=$(round.(theta, sigdigits=4)), " *
                        "$crit_name=$(round(gcv_val, sigdigits=6)), " *
                        "tr(A)=$(round(trA, digits=2))")
            end
        end

        # PCLS step at current θ; step_contract already returns the
        # penalized objective at the accepted point (a full ODE solve),
        # so reuse it for convergence tracking instead of recomputing.
        beta_new_pcls, B_new = pcls_step(J, z_pseudo, theta, w_irls)
        beta_new, curr_obj = step_contract(beta, beta_new_pcls, B_new)

        if verbose && (iter <= 4 || iter % 10 == 0)
            data_ss = sum(w_vec[i] * (y_vec[i] - f_vec[i])^2 for i in 1:n_data)
            println("Iter $iter: obj=$(round(curr_obj, sigdigits=6)), " *
                    "SS=$(round(data_ss, sigdigits=6)), " *
                    "θ=$(round.(theta, sigdigits=3))")
        end

        beta .= beta_new

        # Check convergence
        if iter >= 3 && abs(curr_obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
            if verbose; println("Converged at iter $iter (objective stable)"); end
            conv_converged = true
            conv_reason = :converged_tol
            break
        end
        prev_obj = curr_obj
    end

    # ── Build solution ──
    p_opt = copy(beta)
    pred  = simulate(prob, p_opt)

    # Data loss (weighted SS)
    data_loss = weighted_data_loss(prob, pred)

    # Final EDF via hat matrix
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        f_vec[k] = pred[ti, oi]
        k += 1
    end
    compute_jacobian!(J, prob, p_opt, f_vec, n_times, n_obs; dam=dam)

    B_final  = build_B(theta)
    W_irls   = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
    JWJ      = J' * Diagonal(W_irls) * J
    H_final  = JWJ + B_final
    maxd = maximum(abs.(diag(H_final)))
    for i in 1:n_p
        H_final[i, i] += 1e-12 * maxd + 1e-15
    end
    edf = try
        tr(cholesky(Symmetric(H_final)) \ JWJ)
    catch
        tr(H_final \ JWJ)
    end

    pen_ss  = dot(p_opt, B_final * p_opt)
    obj_val = 0.5 * (data_loss + pen_ss)

    # Build ComponentArray for parameter access
    uf_syms = Symbol[a.name for a in prob.approximators]
    uf_vals = Vector{Float64}[]
    offset  = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(uf_vals, Float64.(p_opt[offset+1:offset+np]))
        offset += np
    end
    params = ComponentArray(NamedTuple{Tuple(uf_syms)}(Tuple(uf_vals)))

    # Build unknown function evaluators for the solution
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = p_opt[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    if verbose
        println("\nGCV final: data_loss=$(round(data_loss, sigdigits=6)), " *
                "penalty=$(round(pen_ss, sigdigits=6)), " *
                "EDF=$(round(edf, digits=2))")
        println("Final θ: ", [round(t, sigdigits=4) for t in theta])
        if isfinite(gcv_val)
            println("Final $crit_name: $(round(gcv_val, sigdigits=6))")
        end
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, criterion=criterion,
                 gcv=(criterion === :gcv ? gcv_val : NaN),
                 ncv=(criterion === :ncv ? gcv_val : NaN)))
end
