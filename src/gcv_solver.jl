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

# ─── Golden-section search on log(λ) ─────────────────────────────

"""
    _golden_section_gcv(J, W_irls, z, S_list, offsets, nknots_list, n_p, n,
                        gamma, lo, hi, tol; maxiter)

Minimize GCV over a shared log(λ) using golden-section search.

All approximator penalties are scaled by the same λ = exp(rho).
Returns `(best_rho, best_beta, best_gcv, best_trA)`.
"""
function _golden_section_gcv(J::AbstractMatrix, W_irls::AbstractVector,
                             z::AbstractVector,
                             S_list::Vector{Matrix{Float64}},
                             offsets::Vector{Int}, nknots_list::Vector{Int},
                             n_p::Int, n::Int, gamma::Float64,
                             lo::Float64, hi::Float64, tol::Float64;
                             maxiter::Int=100)
    gr = (sqrt(5.0) + 1.0) / 2.0  # golden ratio

    function eval_gcv(rho)
        rho_vec = fill(rho, length(S_list))
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rho_vec, n_p)
        gcv, beta, rss, trA = _gcv_score(J, W_irls, z, S_lam, n, gamma)
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
    _grid_then_refine_gcv(J, W_irls, z, S_list, offsets, nknots_list,
                          n_p, n, gamma, n_grid, tol)

Initial coarse grid search over log(λ) ∈ [RHO_MIN, RHO_MAX], then
golden-section refinement around the best grid point.

Returns `(best_rho, best_beta, best_gcv, best_trA)`.
"""
function _grid_then_refine_gcv(J::AbstractMatrix, W_irls::AbstractVector,
                               z::AbstractVector,
                               S_list::Vector{Matrix{Float64}},
                               offsets::Vector{Int}, nknots_list::Vector{Int},
                               n_p::Int, n::Int, gamma::Float64,
                               n_grid::Int, tol::Float64)
    rho_grid = range(RHO_MIN, RHO_MAX, length=n_grid)
    best_gcv = Inf
    best_idx = 1
    best_beta = zeros(n_p)
    best_trA = 0.0

    for (idx, rho) in enumerate(rho_grid)
        rho_vec = fill(rho, length(S_list))
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rho_vec, n_p)
        gcv, beta, _, trA = _gcv_score(J, W_irls, z, S_lam, n, gamma)
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
        J, W_irls, z, S_list, offsets, nknots_list,
        n_p, n, gamma, lo, hi, tol)

    # Keep the better of grid and refinement
    if gcv_opt < best_gcv
        return (rho_opt, beta_opt, gcv_opt, trA_opt)
    else
        return (Float64(rho_grid[best_idx]), best_beta, best_gcv, best_trA)
    end
end

"""
    _coordinate_gcv(J, W_irls, z, S_list, offsets, nknots_list, n_p, n,
                    gamma, rho0, tol; sweeps=3)

Per-approximator GCV: coordinate descent over the vector ρ, minimizing each
component by golden section while the others are held fixed. Wood (2001)
treats λ as a VECTOR with one smoothing parameter per unknown function
(as does ddefit's gcv.c); a single shared λ mis-smooths whenever the
functions differ in scale or wiggliness. Started from the shared-λ optimum,
2–3 sweeps typically converge.
"""
function _coordinate_gcv(J::AbstractMatrix, W_irls::AbstractVector,
                         z::AbstractVector,
                         S_list::Vector{Matrix{Float64}},
                         offsets::Vector{Int}, nknots_list::Vector{Int},
                         n_p::Int, n::Int, gamma::Float64,
                         rho0::Vector{Float64}, tol::Float64;
                         sweeps::Int=3)
    m = length(S_list)
    rho = copy(rho0)
    gr = (sqrt(5.0) + 1.0) / 2.0

    eval_vec = function (rv)
        S_lam = build_S_lambda(S_list, offsets, nknots_list, rv, n_p)
        _gcv_score(J, W_irls, z, S_lam, n, gamma)
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
4. Select λ by minimizing GCV(λ) via grid search + golden-section refinement
5. Solve penalized LS at optimal λ
6. Step contraction (backtracking)
7. Repeat until convergence

Returns a `PSMSolution`. `sol.convergence` is a NamedTuple
`(converged, iterations, reason, gcv)` — see the `GCVSolver` and
`PSMSolution` docstrings for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::GCVSolver)
    _validate_problem(prob, "GCVSolver")
    maxiters = alg.maxiters
    verbose  = alg.verbose
    gamma    = alg.gamma
    n_grid   = alg.n_grid
    tol      = alg.tol

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
        println("GCV solver: $n_p params, $n_data data, $m smooth terms, γ=$gamma")
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

        # ── GCV smoothing parameter selection ──
        if m > 0
            best_rho, beta_gcv, gcv_val, trA = _grid_then_refine_gcv(
                J, w_irls, z_pseudo,
                S_list, uf_offsets, uf_nk,
                n_p, n_gcv, gamma,
                n_grid, tol)

            if m == 1
                theta .= exp(best_rho)
            else
                # Per-approximator λ (Wood 2001 treats λ as a vector):
                # refine each component by coordinate descent from the
                # shared-λ optimum.
                rho_vec, beta_gcv, gcv_val, trA = _coordinate_gcv(
                    J, w_irls, z_pseudo,
                    S_list, uf_offsets, uf_nk,
                    n_p, n_gcv, gamma,
                    fill(best_rho, m), tol)
                theta .= exp.(rho_vec)
            end

            if verbose && (iter <= 4 || iter % 10 == 0)
                println("  GCV iter $iter: λ=$(round.(theta, sigdigits=4)), " *
                        "GCV=$(round(gcv_val, sigdigits=6)), " *
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
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            uf_evals[approx.name] = build_neural_evaluator(approx, params_k)
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] = build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
        elseif approx isa SPDEApproximator
            uf_evals[approx.name] = build_spde_evaluator(approx.mesh_points, params_k)
        elseif approx isa ShapeConstrainedSPDEApproximator
            uf_evals[approx.name] = build_constrained_spde_evaluator(approx, params_k)
        end
    end

    if verbose
        println("\nGCV final: data_loss=$(round(data_loss, sigdigits=6)), " *
                "penalty=$(round(pen_ss, sigdigits=6)), " *
                "EDF=$(round(edf, digits=2))")
        println("Final θ: ", [round(t, sigdigits=4) for t in theta])
        if isfinite(gcv_val)
            println("Final GCV: $(round(gcv_val, sigdigits=6))")
        end
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, gcv=gcv_val))
end
