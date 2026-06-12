# ─── Gradient matching solver ─────────────────────────────────────
#
# Two-step approach inspired by NODEBNGM (Bonnaffé et al.):
#   Step 1: Smooth observed data with cubic splines to get ŷ(t) and dŷ/dt
#   Step 2: Fit unknown functions by matching ODE derivatives:
#           minimize ||dŷ/dt - f(ŷ, p, t)||²
#
# Key advantage: no ODE integration in the optimization loop, making it
# far more robust for neural network approximators.
#
# Reference: Bonnaffé, Sheldon & Bhatt (2023), Methods in Ecology and Evolution

using LinearAlgebra
using Statistics

# ─── Step 1: Data smoothing ──────────────────────────────────────

"""
    _smoothing_spline(t, y) -> (value, derivative)

Fit a penalized cubic regression spline (P-spline: cubic B-spline basis +
second-order difference penalty) to `(t, y)` with the smoothing parameter
chosen by Generalized Cross-Validation, and return callables for the fitted
value and its first derivative.

This is a genuine SMOOTHER (it does not interpolate the noisy data), which
is what gradient matching requires: differentiating an interpolant amplifies
observation noise, whereas the penalized fit suppresses it (Wood 2001;
Varah 1982). Shared by the gradient-matching, two-stage, and BNG solvers.
"""
function _smoothing_spline(t::AbstractVector{Float64}, y::AbstractVector{Float64})
    n = length(t)
    a, b = minimum(t), maximum(t)
    if b <= a || n < 4
        ȳ = sum(y) / max(n, 1)
        return (x -> ȳ), (x -> 0.0)
    end
    q = clamp(n - 2, 4, 15)                     # number of B-spline coefficients
    knots = _scam_knot_vector((a, b), q)
    B = zeros(n, q)
    for i in 1:n
        B[i, :] = _bspline_basis_vector(t[i], knots, 4)
    end
    # Second-order difference penalty D'D.
    D = zeros(q - 2, q)
    for i in 1:(q - 2)
        D[i, i] = 1.0; D[i, i+1] = -2.0; D[i, i+2] = 1.0
    end
    P = D' * D
    BtB = B' * B; Bty = B' * y
    best_gcv = Inf; β = BtB \ Bty
    for logλ in range(-6.0, 6.0, length=40)
        λ = 10.0^logλ
        F = cholesky(Symmetric(BtB + λ * P + 1e-10 * I), check=false)
        issuccess(F) || continue
        βλ = F \ Bty
        resid = y - B * βλ
        trH = tr(B * (F \ B'))
        denom = (n - trH)^2
        gcv = denom > 1e-8 ? n * sum(abs2, resid) / denom : Inf
        if gcv < best_gcv
            best_gcv = gcv; β = βλ
        end
    end
    h = (b - a) * 1e-6
    value = x -> dot(_bspline_basis_vector(clamp(Float64(x), a, b), knots, 4), β)
    deriv = x -> begin
        xc = clamp(Float64(x), a + h, b - h)
        (dot(_bspline_basis_vector(xc + h, knots, 4), β) -
         dot(_bspline_basis_vector(xc - h, knots, 4), β)) / (2h)
    end
    value, deriv
end

"""
Smooth observed data with a penalized (GCV) smoothing spline and compute
time derivatives — see [`_smoothing_spline`](@ref).

Returns:
- `y_smooth`: smoothed state values (n_times × K)
- `dydt`: time derivatives from the smoother (n_times × K)
"""
function smooth_and_differentiate(times::Vector{Float64},
                                  data::Matrix{Float64},
                                  obs_to_state::Vector{Int},
                                  K::Int)
    T = length(times)
    n_obs = size(data, 2)
    y_smooth = zeros(T, K)
    dydt = zeros(T, K)

    for j in 1:n_obs
        sk = obs_to_state[j]
        val, der = _smoothing_spline(times, data[:, j])
        for i in 1:T
            y_smooth[i, sk] = val(times[i])
            dydt[i, sk] = der(times[i])
        end
    end

    # For unobserved states, leave as zero (user must observe all states
    # for gradient matching to work)
    y_smooth, dydt
end

# ─── Step 2: Derivative matching ─────────────────────────────────

"""
Evaluate ODE RHS at all time points using smoothed state values.
Returns (T × K) matrix of f(ŷ(t), p, t).
"""
function eval_rhs_at_smooth(prob::PSMProblem, times::Vector{Float64},
                            y_smooth::Matrix{Float64}, beta::Vector{Float64})
    T, K = size(y_smooth)
    F = zeros(T, K)
    p = build_param_struct(prob, beta)
    du = zeros(K)

    for i in 1:T
        u = y_smooth[i, :]
        try
            prob.dynamics!(du, u, p, times[i])
        catch
            du .= 1e6
        end
        F[i, :] .= du
    end
    F
end

"""
Compute gradient-matching residual and Jacobian.

Residual: r[i,k] = √w × (dŷ_k/dt(t_i) - f_k(ŷ(t_i), p, t_i))
Jacobian: ∂r/∂β via finite differences
"""
function gm_residual_jacobian(prob::PSMProblem, times::Vector{Float64},
                              y_smooth::Matrix{Float64}, dydt::Matrix{Float64},
                              beta::Vector{Float64}, w::Vector{Float64})
    T, K = size(y_smooth)
    n_beta = length(beta)
    n_match = prob.discrete ? T - 1 : T

    F = eval_rhs_at_smooth(prob, times, y_smooth, beta)

    # Residual: √w × (dydt - F)
    resid = zeros(n_match * K)
    for k in 1:K, i in 1:n_match
        idx = (k - 1) * n_match + i
        wi = idx <= length(w) ? sqrt(w[idx]) : 1.0
        resid[idx] = wi * (dydt[i, k] - F[i, k])
    end

    # Jacobian via FD
    J = zeros(n_match * K, n_beta)
    eps = 1e-5
    for b in 1:n_beta
        beta_p = copy(beta)
        step = max(eps, abs(beta[b]) * eps)
        beta_p[b] += step
        F_p = eval_rhs_at_smooth(prob, times, y_smooth, beta_p)
        for k in 1:K, i in 1:n_match
            idx = (k - 1) * n_match + i
            wi = idx <= length(w) ? sqrt(w[idx]) : 1.0
            J[idx, b] = -wi * (F_p[i, k] - F[i, k]) / step
        end
    end

    resid, J
end

# ─── Main solver ─────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::GradientMatching)

Fit a partially specified model using the smooth-then-match approach.
Observed data are first smoothed to estimate state trajectories and their
derivatives; the unknown-function parameters are then found by matching
the smoothed derivatives to the model right-hand side.

# Algorithm
1. Smooth each observed state with cubic splines (continuous) or compute
   forward differences (discrete).
2. Build a gradient-matching objective: ∑ₜ ‖x′(t) − f(x(t), uf(t; β))‖².
3. Minimise with `Optim.NelderMead` (or user-specified method).
4. Reconstruct the unknown functions at the fitted parameters.

# References
- Varah (1982), "A Spline Least Squares Method for Numerical Parameter
  Estimation in Differential Equations", SIAM J. Sci. Stat. Comput.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::GradientMatching)
    _validate_problem(prob, "GradientMatching")
    times = Float64.(prob.data_times)
    T_pts = length(times)
    K = length(prob.u0)
    n_obs = size(prob.data_values, 2)

    verbose = alg.verbose

    # Step 1: Smooth data and compute derivatives / forward differences
    if verbose; println("Step 1: Smoothing data and computing derivatives..."); end

    if prob.discrete
        # For discrete models, smooth data first (just like continuous) then
        # use smoothed next-state values as matching targets.
        # Without smoothing, noisy data→noisy targets produces poor recovery.
        y_raw = zeros(T_pts, K)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            y_raw[:, sk] .= prob.data_values[:, j]
        end
        # Smooth each observed state with a cubic spline
        y_smooth = zeros(T_pts, K)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            itp = CubicSpline(prob.data_values[:, j], times;
                              extrapolation=ExtrapolationType.Extension)
            for i in 1:T_pts
                y_smooth[i, sk] = itp(times[i])
            end
        end
        # Target: smoothed next-state value u[t+1] = f(u[t], p, t)
        dydt = zeros(T_pts, K)
        for k in 1:K
            for i in 1:(T_pts-1)
                dydt[i, k] = y_smooth[i+1, k]
            end
            dydt[T_pts, k] = y_smooth[T_pts, k]  # unused; last point has no forward diff
        end
    else
        y_smooth, dydt = smooth_and_differentiate(times, Float64.(prob.data_values),
                                                   prob.obs_to_state, K)
    end

    # Initialize unknown function parameters
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Build penalty matrices
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)

    # Smoothing parameters
    theta = if m > 0
        Float64[1.0 / max(tr(S_list[l]), 1e-10) for l in 1:m]
    else
        Float64[]
    end

    # Per-state weights for Gauss-Newton: normalize so each equation contributes
    # equally. Without this, states with large |dy/dt| dominate the penalty
    # update. For Adam (NNs), uniform weights work better since the NN
    # architecture already provides implicit regularization.
    n_match = prob.discrete ? T_pts - 1 : T_pts
    w = ones(n_match * K)
    if m > 0  # only weight for penalized (Gauss-Newton) branch
        for k in 1:K
            dydt_k = @view dydt[1:n_match, k]
            scale_k = max(std(dydt_k), 1e-10)
            for i in 1:n_match
                idx = (k - 1) * n_match + i
                w[idx] = 1.0 / scale_k^2
            end
        end
    end

    if verbose
        println("Step 2: Gradient matching — $(n_beta) params, $(n_match) match points, $(K) states")
        println("  Penalty terms: $m, optimizer: $(m > 0 ? "Gauss-Newton" : "Adam")")
    end

    # Loss function for derivative/map matching (with per-state weights)
    function gm_loss(β_eval)
        F = eval_rhs_at_smooth(prob, times, y_smooth, β_eval)
        loss_val = 0.0
        for k in 1:K, i in 1:n_match
            idx = (k - 1) * n_match + i
            loss_val += w[idx] * (dydt[i, k] - F[i, k])^2
        end
        # Add penalty
        if m > 0
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                beta_k = β_eval[off+1:off+nk]
                loss_val += theta[l] * dot(beta_k, S_list[l] * beta_k)
            end
        end
        loss_val
    end

    if m > 0
        # ─── Gauss-Newton for penalized approximators (B-spline, GP) ───
        prev_obj = Inf
        for iter in 1:alg.maxiters
            resid, J = gm_residual_jacobian(prob, times, y_smooth, dydt, beta, w)

            B = zeros(n_beta, n_beta)
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                B[off+1:off+nk, off+1:off+nk] .+= theta[l] .* S_list[l]
            end

            resid_ss = sum(resid.^2)
            pen_ss = dot(beta, B * beta)
            obj = resid_ss + pen_ss

            if verbose && (iter <= 3 || iter % 10 == 0)
                println("  iter $iter: deriv_SS=$(round(resid_ss, sigdigits=5)) pen=$(round(pen_ss, sigdigits=4))")
            end

            if iter > 1 && abs(obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
                if verbose; println("  Converged at iter $iter"); end
                break
            end
            prev_obj = obj

            JtJ = J' * J + B
            neg_Jtr = -(J' * resid)
            delta = try; JtJ \ neg_Jtr; catch; try; (JtJ + 1e-6 * I) \ neg_Jtr; catch; break; end; end

            best_obj = obj; best_step = 0.0
            for k in 0:8
                ss = 0.5^k
                beta_new = beta .+ ss .* delta
                obj_new = gm_loss(beta_new)
                if obj_new < best_obj; best_obj = obj_new; best_step = ss; end
            end
            if best_step == 0.0; if verbose; println("  No improvement, stopping"); end; break; end
            beta .= beta .+ best_step .* delta

            # Update smoothing params
            if iter % 5 == 0
                sigma2 = resid_ss / (T_pts * K)
                if alg.sigma2_init !== nothing; sigma2 = min(sigma2, alg.sigma2_init); end
                for l in 1:m
                    off = uf_offsets[l]; nk = uf_nk[l]
                    bSb = dot(beta[off+1:off+nk], S_list[l] * beta[off+1:off+nk])
                    rank_k = max(nk - 2, 1)
                    if bSb > 1e-30; theta[l] = clamp(sigma2 * rank_k / bSb, 1e-20, 1e20); end
                end
            end
        end
    else
        # ─── Adam optimizer for unpenalized approximators (NN) ─────────
        # Use ForwardDiff for exact gradients (avoids Float32 precision loss
        # from Lux evaluation that plagues finite-difference gradients)
        function gm_loss_ad(β_eval)
            p = build_autodiff_param_struct(prob, β_eval)
            T_pts_loc = size(y_smooth, 1)
            du = zeros(eltype(β_eval), K)
            loss_val = zero(eltype(β_eval))
            for i in 1:n_match
                u = eltype(β_eval).(y_smooth[i, :])
                try
                    prob.dynamics!(du, u, p, times[i])
                catch
                    du .= eltype(β_eval)(1e6)
                end
                for k in 1:K
                    loss_val += (dydt[i, k] - du[k])^2
                end
            end
            loss_val
        end

        # Adam state
        lr = alg.lr
        β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
        m_adam = zeros(n_beta)
        v_adam = zeros(n_beta)
        best_beta = copy(beta)
        best_loss = Inf
        loss_window = fill(Inf, 20)

        for iter in 1:alg.maxiters
            grad = ForwardDiff.gradient(gm_loss_ad, beta)
            loss_val = gm_loss_ad(beta)

            if loss_val < best_loss
                best_loss = loss_val
                best_beta .= beta
            end
            loss_window[mod1(iter, 20)] = loss_val

            # Cosine learning rate annealing (floor at 10% of initial lr)
            lr_t = lr * (0.1 + 0.45 * (1 + cos(π * iter / alg.maxiters)))

            # Adam update
            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
            m_hat = m_adam ./ (1 - β1_adam^iter)
            v_hat = v_adam ./ (1 - β2_adam^iter)
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

            if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
                println("  Adam iter $iter: loss=$(round(loss_val, sigdigits=5)) lr=$(round(lr_t, sigdigits=3))")
            end

            # Convergence: check if loss plateau (relative change < tol over window)
            if iter > 50
                recent_min = minimum(loss_window)
                recent_max = maximum(loss_window)
                if (recent_max - recent_min) / max(abs(recent_min), 1.0) < alg.tol
                    if verbose; println("  Converged at iter $iter (loss plateau)"); end
                    break
                end
            end
        end
        beta .= best_beta
        if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end
    end

    # ─── Optional shooting refinement ────────────────────────────
    # GM fits derivatives/maps only; it can miss the true function at regions
    # where |dy/dt| is small.  A few Adam shooting steps through the actual
    # model corrects this by using trajectory-level information.
    if alg.refine_iters > 0
        if verbose; println("\nStep 3: Shooting refinement ($(alg.refine_iters) iters)..."); end

        # Reuse the proven adam_loss_mse (with data weights, correct ODE/discrete setup)
        refine_loss = β_eval -> adam_loss_mse(prob, β_eval)

        # Adam refinement — start with moderate lr, comparable to AdamSolver
        lr_refine = min(alg.lr, 0.01)
        m_r = zeros(n_beta)
        v_r = zeros(n_beta)
        best_r = copy(beta)
        best_rl = refine_loss(beta)

        for iter in 1:alg.refine_iters
            result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
            ForwardDiff.gradient!(result, refine_loss, beta)
            rl = DiffResults.value(result)
            grad_r = DiffResults.gradient(result)

            if rl < best_rl; best_rl = rl; best_r .= beta; end

            lr_t = lr_refine * (0.1 + 0.45 * (1 + cos(π * iter / alg.refine_iters)))
            m_r .= 0.9 .* m_r .+ 0.1 .* grad_r
            v_r .= 0.999 .* v_r .+ 0.001 .* grad_r.^2
            m_hat = m_r ./ (1 - 0.9^iter)
            v_hat = v_r ./ (1 - 0.999^iter)
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ 1e-8)

            if verbose && (iter <= 3 || iter % 20 == 0)
                println("  Refine iter $iter: loss=$(round(rl, sigdigits=5)) lr=$(round(lr_t, sigdigits=3))")
            end
        end
        beta .= best_r
        if verbose; println("  Best refine loss: $(round(best_rl, sigdigits=5))"); end
    end

    # Build solution
    # For discrete models, simulate forward to get actual trajectory predictions
    # (instead of using smoothed states which give misleading data_loss=0)
    if prob.discrete
        p_sim = build_param_struct(prob, beta)
        u_sim = Float64.(prob.u0)
        u_next_sim = similar(u_sim)
        sim_states = zeros(T_pts, K)
        sim_states[1, :] .= u_sim
        for step in 1:(T_pts-1)
            try
                prob.dynamics!(u_next_sim, u_sim, p_sim, times[step])
            catch
                u_next_sim .= 1e6
            end
            u_sim = copy(u_next_sim)
            sim_states[step+1, :] .= u_sim
        end
        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[:, j] .= sim_states[:, sk]
        end
    else
        # Continuous: use smoothed states as predictions
        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[:, j] .= y_smooth[:, sk]
        end
    end

    # Data loss (against original data, not derivatives)
    data_loss = 0.0
    for j in 1:n_obs, i in 1:T_pts
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Derivative matching loss
    F_final = eval_rhs_at_smooth(prob, times, y_smooth, beta)
    deriv_loss = sum((dydt .- F_final).^2)

    # Build evaluators
    p_opt = build_param_struct(prob, beta)
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
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

    # EDF and parameters
    edf = Float64(n_beta)  # conservative
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))
    pen_ss = m > 0 ? sum(theta[l] * dot(beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]],
                  S_list[l] * beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]]) for l in 1:m) : 0.0
    obj_val = data_loss + pen_ss

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "deriv_SS=$(round(deriv_loss, sigdigits=5)) EDF=$(round(edf, digits=1))")
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (deriv_loss=deriv_loss, method=:gradient_matching))
end
