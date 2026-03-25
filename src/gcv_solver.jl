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
    r = z .- J * beta_hat
    rss_w = sum(W_irls[i] * r[i]^2 for i in 1:n)

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
            gc = gc
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

Returns a `PSMSolution`.
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

    # Flatten data into vectors (obs-major order: obs 1 times, obs 2 times, …)
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y_vec[k] = prob.data_values[ti, oi]
        w_vec[k] = prob.data_weights[ti, oi]
        k += 1
    end

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

    # PCLS step: augmented system [W^½J; C] β = [W^½z; 0]
    function pcls_step(J_mat, z_pseudo, th, w_irls)
        B = build_B(th)
        C = penalty_sqrt_matrix(B)
        n_pen = size(C, 1)
        W_sqrt = sqrt.(max.(w_irls, 1e-15))
        F_aug = vcat(Diagonal(W_sqrt) * J_mat, C)
        z_aug = vcat(W_sqrt .* z_pseudo, zeros(n_pen))
        F_aug \ z_aug, B
    end

    # Step contraction: backtracking line search
    function step_contract(a_old, a_new, B)
        f_old = penalized_objective(a_old, B)
        direction = a_new .- a_old

        best_f = f_old
        best_a = copy(a_old)

        for k in 0:15
            α = 2.0^(-k)
            a_try = a_old .+ α .* direction
            f_try = penalized_objective(a_try, B)
            if f_try < best_f
                best_f = f_try
                best_a = copy(a_try)
            end
        end
        best_a, best_f
    end

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

    for iter in 0:(maxiters - 1)
        # Re-evaluate model + Jacobian
        f_vec_new, _ = try; eval_model(beta); catch e
            if verbose; println("Iter $iter: simulation failed ($e)"); end
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
                n_p, n_data, gamma,
                n_grid, tol)

            # Convert shared rho to per-approximator theta
            theta .= exp(best_rho)

            if verbose && (iter <= 4 || iter % 10 == 0)
                println("  GCV iter $iter: ρ=$(round(best_rho, digits=3)), " *
                        "λ=$(round(exp(best_rho), sigdigits=4)), " *
                        "GCV=$(round(gcv_val, sigdigits=6)), " *
                        "tr(A)=$(round(trA, digits=2))")
            end
        end

        # PCLS step at current θ
        beta_new_pcls, B_new = pcls_step(J, z_pseudo, theta, w_irls)
        beta_new, obj_new = step_contract(beta, beta_new_pcls, B_new)

        # Track penalized objective for convergence
        curr_obj = penalized_objective(beta_new, B_new)

        if verbose && (iter <= 4 || iter % 10 == 0)
            data_ss = sum(w_vec[i] * (y_vec[i] - f_vec[i])^2 for i in 1:n_data)
            println("Iter $iter: obj=$(round(curr_obj, sigdigits=6)), " *
                    "SS=$(round(data_ss, sigdigits=6)), " *
                    "θ=$(round.(theta, sigdigits=3))")
        end

        beta .= beta_new

        # Check convergence
        if iter >= 3 && abs(curr_obj - prev_obj) < 1e-6 * max(abs(prev_obj), 1.0)
            if verbose; println("Converged at iter $iter (objective stable)"); end
            break
        end
        prev_obj = curr_obj
    end

    # ── Build solution ──
    p_opt = copy(beta)
    pred  = simulate(prob, p_opt)

    # Data loss (weighted SS)
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

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
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(Lux.initialparameters(rng2, approx.model)))
            ps_final = similar(ps_ca)
            ps_final .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model, Float32.(reshape([xn], :, 1)), ps_final, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
            end
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
                Float64.(prob.data_times), uf_evals, nothing)
end
