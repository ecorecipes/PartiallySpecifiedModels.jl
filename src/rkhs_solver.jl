# ─── RKHS solver (trajectory in a reproducing kernel Hilbert space) ─
#
# González et al. (2014): the state trajectory is placed in a
# time-kernel RKHS, x_k(t) = m_k + Σᵢ b_{k,i} k(t, tᵢ), so its
# derivative ẋ_k(t) = Σᵢ b_{k,i} k̇(t, tᵢ) is available analytically.
# Estimation alternates a joint LINEAR Gauss–Newton solve for all
# states' trajectory coefficients B (data fit + RKHS norm + ODE-gradient
# term with f linearized around the previous trajectory) with gradient
# steps on the unknown-function parameters θ.
#
# Reference: González et al. (2014), Pattern Recognition Letters —
#            "Reproducing kernel Hilbert space based estimation of
#             systems of ordinary differential equations"
#            Schölkopf & Smola (2002), Learning with Kernels

using LinearAlgebra: dot, norm, Symmetric, cholesky, I

"""
    solve(prob::PSMProblem, alg::RKHSSolver)

Fit a partially specified model with the state trajectory represented in
a time-kernel RKHS (González et al. 2014).

Each state is expanded over the data times, `x_k(t) = m_k + Σᵢ b_{k,i}
k(t, tᵢ)`, giving the analytic derivative `ẋ_k(t) = Σᵢ b_{k,i} k̇(t, tᵢ)`.
The objective

    J(B, θ) = Σ_k [ ‖y_k − x_k(t_data)‖² + λ b_kᵀ K b_k ]
              + ρ Σ_k Σ_g (ẋ_k(t_g) − f_k(x(t_g), θ))² + θᵀSθ

(data fit + RKHS norm + ODE-gradient match on a collocation grid +
approximator smoothing penalty) is minimized by alternating:

1. **B-step** — one Gauss–Newton step on ALL states' coefficients
   jointly: `f` is linearized around the previous trajectory using the
   grid Jacobian `∂f/∂x`, making every equation's mismatch linear in
   every state's coefficients, and the joint normal equations (size
   `n_times · n_vars`) are solved directly. The Jacobian coupling is what
   identifies unobserved states.
2. **θ-step** — Adam on the ODE-gradient mismatch with the trajectory
   fixed.

Unobserved states have no data term (`w_k = 0`) and are identified
through the ODE term alone. All approximator types are supported.

# Returns
`PSMSolution`: `params` are the approximator coefficients θ, the fitted
values are the RKHS trajectory at the data times.
"""
function SciMLBase.solve(prob::PSMProblem, alg::RKHSSolver)
    _validate_problem(prob, "RKHSSolver"; require_continuous=true,
                      reject_delays=true)
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    u0_vec = Float64.(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_vars = length(u0_vec)
    n_obs = size(prob.data_values, 2)
    obs_of_state = Dict{Int, Vector{Int}}()
    for j in 1:n_obs
        push!(get!(obs_of_state, prob.obs_to_state[j], Int[]), j)
    end

    # ── Time kernel and its derivative ∂k/∂t₁ ────────────────────────
    ℓ = alg.lengthscale
    if ℓ <= 0.0
        ℓ = (times[end] - times[1]) / 8.0
        if verbose; println("RKHSSolver: auto time-kernel lengthscale ℓ=$(round(ℓ, sigdigits=3))"); end
    end
    kernel_fn, dkernel_fn = if alg.kernel == :rbf
        ((t1, t2) -> exp(-0.5 * (t1 - t2)^2 / ℓ^2),
         (t1, t2) -> -(t1 - t2) / ℓ^2 * exp(-0.5 * (t1 - t2)^2 / ℓ^2))
    elseif alg.kernel == :matern32
        ((t1, t2) -> begin
             r = abs(t1 - t2) / ℓ
             (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
         end,
         (t1, t2) -> -3 * (t1 - t2) / ℓ^2 * exp(-sqrt(3) * abs(t1 - t2) / ℓ))
    elseif alg.kernel == :matern52
        ((t1, t2) -> begin
             r = abs(t1 - t2) / ℓ
             (1 + sqrt(5) * r + 5/3 * r^2) * exp(-sqrt(5) * r)
         end,
         (t1, t2) -> begin
             r = abs(t1 - t2) / ℓ
             -5/3 * (t1 - t2) / ℓ^2 * (1 + sqrt(5) * r) * exp(-sqrt(5) * r)
         end)
    else
        error("Unknown kernel: $(alg.kernel). Use :rbf, :matern32, or :matern52.")
    end

    # Expansion centers = data times; ODE collocation grid across the data
    # window (n_repr_points controls its resolution).
    centers = times
    m = n_times
    n_g = max(alg.n_repr_points, 2)
    t_grid = collect(range(times[1], times[end], length=n_g))

    K_cc = [kernel_fn(centers[i], centers[j]) for i in 1:m, j in 1:m]
    Φd = K_cc                                     # kernel at data times × centers
    Φg = [kernel_fn(t_grid[i], centers[j]) for i in 1:n_g, j in 1:m]
    dΦg = [dkernel_fn(t_grid[i], centers[j]) for i in 1:n_g, j in 1:m]

    λ = alg.lambda_rkhs
    ρ = alg.lambda_ode
    A_data = Φd' * Φd
    # 1e-8 jitter regularizes the solves; full_objective tracks the exact
    # λ bᵀKb, so tracked J and minimized J differ by a negligible 1e-8‖b‖².
    A_pen = Symmetric(K_cc + 1e-8 * I)
    # The ODE weight is ramped in over the first fifth of the iterations:
    # starting data-dominant prevents the Picard alternation from locking
    # onto the degenerate flat-trajectory fixed point before the coupling
    # between observed and unobserved states has taken hold.
    n_ramp = max(1, alg.maxiters ÷ 5)

    # Usable rows per observation column; `rkhs_all_usable` selects the
    # original (bit-for-bit unchanged) vectorized paths below.
    rkhs_keep = Dict{Int,Vector{Int}}(j => usable_rows(prob, j)
                                      for j in 1:n_obs)
    rkhs_all_usable = all(length(rkhs_keep[j]) == n_times for j in 1:n_obs)

    # ── Centers m_k and initial trajectory (kernel ridge on the data) ──
    m_center = zeros(n_vars)
    B = zeros(m, n_vars)                          # trajectory coefficients
    for k in 1:n_vars
        if haskey(obs_of_state, k)
            # Pool over replicate columns, USABLE cells only: `mean` over a
            # column holding a NaN is NaN, and that center then makes the
            # whole trajectory-coefficient block NaN.
            vals = Float64[prob.data_values[i, jc] for jc in obs_of_state[k]
                           for i in axes(prob.data_values, 1)
                           if usable_cell(prob, i, jc)]
            isempty(vals) && error("RKHSSolver: state $k has no usable " *
                "observations (all masked).")
            m_center[k] = length(vals) == n_times * length(obs_of_state[k]) ?
                          mean(prob.data_values[:, obs_of_state[k]]) : mean(vals)
            jc1 = obs_of_state[k][1]
            keep1 = usable_rows(prob, jc1)
            if length(keep1) == n_times
                y_k = prob.data_values[:, jc1]
                B[:, k] = Symmetric(A_data + λ * A_pen) \ (Φd' * (y_k .- m_center[k]))
            else
                Φk = Φd[keep1, :]
                y_k = Float64.(prob.data_values[keep1, jc1])
                B[:, k] = Symmetric(Φk' * Φk + λ * A_pen) \ (Φk' * (y_k .- m_center[k]))
            end
        else
            m_center[k] = u0_vec[k]               # flat at IC; ODE term moves it
        end
    end

    # ── Initialise unknown-function parameters θ ─────────────────────
    beta = Float64[]
    for approx in prob.approximators
        if approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) :
                  Random.default_rng()
            append!(beta, neural_init_params(approx, rng))
        else
            append!(beta, initial_params(approx))
        end
    end
    n_beta = length(beta)

    if verbose
        println("RKHSSolver: $m centers/state × $n_vars states, $n_beta θ, " *
                "grid $n_g, λ=$λ ρ=$ρ")
    end

    # Trajectory and derivative on the collocation grid for given B
    X_g(Bmat) = (Φg * Bmat) .+ m_center'
    dX_g(Bmat) = dΦg * Bmat

    # RHS of the ODE at the grid, along the given trajectory
    function rhs_on_grid(Xg, β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)
        F = Matrix{T_el}(undef, n_g, n_vars)
        for i in 1:n_g
            u = Vector{T_el}(@view Xg[i, :])
            try
                prob.dynamics!(du, u, p, t_grid[i])
            catch e
                _is_program_error(e) && rethrow()
                du .= T_el(1e6)
            end
            F[i, :] .= du
        end
        F
    end

    # θ objective: ODE mismatch on the grid + smoothing penalty
    function theta_loss(β_eval, Xg, dXg, ρ_use=ρ)
        T_el = eltype(β_eval)
        F = rhs_on_grid(Xg, β_eval)
        loss = zero(T_el)
        for k in 1:n_vars
            for i in 1:n_g
                loss += ρ_use * (dXg[i, k] - F[i, k])^2
            end
        end
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            pk = @view β_eval[offset+1:offset+np]
            offset += np
            S = penalty_matrix(approx)
            S !== nothing && (loss += dot(pk, S * pk))
        end
        loss
    end

    # Full objective J(B, θ) for convergence tracking
    function full_objective(Bmat, β_eval)
        J = 0.0
        for k in 1:n_vars
            if haskey(obs_of_state, k)
                xk = Φd * Bmat[:, k] .+ m_center[k]
                for j in obs_of_state[k]
                    # Usable cells only — one NaN otherwise makes the tracked
                    # objective NaN, so the convergence test never fires and
                    # `sol.objective` is reported as NaN.
                    rows = rkhs_keep[j]
                    if length(rows) == n_times
                        J += sum(abs2, prob.data_values[:, j] .- xk)
                    else
                        for i in rows
                            J += (prob.data_values[i, j] - xk[i])^2
                        end
                    end
                end
            end
            J += λ * dot(Bmat[:, k], K_cc * Bmat[:, k])
        end
        J + theta_loss(β_eval, X_g(Bmat), dX_g(Bmat))
    end

    # ── Alternating optimisation ─────────────────────────────────────
    n_inner = 10                                  # θ Adam steps per outer iter
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_beta); v_adam = zeros(n_beta)
    adam_t = 0
    result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))

    # Grid Jacobians G[i,:,:] = ∂f/∂x at the current trajectory, needed so
    # the B-step sees every equation's dependence on every state (without
    # them, a state entering only OTHER equations' right-hand sides — e.g.
    # an unobserved velocity in x1' = x2 — would never be pulled by the
    # ODE terms and the alternation locks onto a degenerate trajectory).
    function grid_jacobians(Xg, p_now)
        G = Array{Float64, 3}(undef, n_g, n_vars, n_vars)
        du_buf = zeros(n_vars)
        for i in 1:n_g
            u_i = Vector{Float64}(@view Xg[i, :])
            Ji = try
                ForwardDiff.jacobian((du, u) -> prob.dynamics!(du, u, p_now, t_grid[i]),
                                     du_buf, u_i)
            catch e
                _is_program_error(e) && rethrow()
                zeros(n_vars, n_vars)
            end
            G[i, :, :] = Ji
        end
        G
    end

    J_prev = Inf
    final_iter = alg.maxiters
    converged = false
    n_B = m * n_vars
    for iter in 1:alg.maxiters
        ρ_eff = ρ * min(1.0, iter / n_ramp)^2
        # B-step: one Gauss–Newton step on the trajectory coefficients.
        # Linearize f around the previous trajectory,
        #   f_k(x) ≈ F_prev_k + Σⱼ G_kj ∘ (Φg bⱼ − Φg bⱼ_prev),
        # so each equation's mismatch ẋ_k − f_k is linear in ALL states'
        # coefficients, and solve the joint normal equations (size m·n_vars).
        Xg_prev = X_g(B)
        p_float = build_autodiff_param_struct(prob, beta)
        F_prev = rhs_on_grid(Xg_prev, beta)
        G = grid_jacobians(Xg_prev, p_float)
        H = zeros(n_B, n_B)
        rhs = zeros(n_B)
        blk(j) = ((j-1)*m + 1):(j*m)
        for j in 1:n_vars
            w_j = haskey(obs_of_state, j) ? Float64(length(obs_of_state[j])) : 0.0
            # `w_j .* A_data` is a shortcut for Σ_jc Φd'Φd that is only valid
            # when EVERY row of every replicate column carries a usable datum.
            # Under masking the design must lose those rows, and the rhs must
            # not read their NaN values (`Φd' * NaN` contaminates the whole
            # block, and the weight never multiplies it away here at all).
            if w_j > 0 && !rkhs_all_usable
                Hj = zeros(m, m)
                for jc in obs_of_state[j]
                    Φk = @view Φd[rkhs_keep[jc], :]
                    Hj .+= Φk' * Φk
                    rhs[blk(j)] .+= Φk' *
                        (Float64.(prob.data_values[rkhs_keep[jc], jc]) .- m_center[j])
                end
                H[blk(j), blk(j)] .+= Hj .+ λ .* A_pen
            else
                H[blk(j), blk(j)] .+= w_j .* A_data .+ λ .* A_pen
                if w_j > 0
                    for jc in obs_of_state[j]
                        rhs[blk(j)] .+= Φd' * (prob.data_values[:, jc] .- m_center[j])
                    end
                end
            end
        end
        for k in 1:n_vars
            # c_k: linearization constant of eq k
            c_k = copy(F_prev[:, k])
            for j in 1:n_vars
                c_k .-= G[:, k, j] .* (Φg * B[:, j])
            end
            Ms = Vector{Matrix{Float64}}(undef, n_vars)
            for j in 1:n_vars
                Mkj = -(G[:, k, j] .* Φg)
                j == k && (Mkj .+= dΦg)
                Ms[j] = Mkj
            end
            for j1 in 1:n_vars, j2 in 1:n_vars
                H[blk(j1), blk(j2)] .+= ρ_eff .* (Ms[j1]' * Ms[j2])
            end
            for j in 1:n_vars
                rhs[blk(j)] .+= ρ_eff .* (Ms[j]' * c_k)
            end
        end
        B_new = reshape(Symmetric(H + 1e-10 * I) \ rhs, m, n_vars)
        B .= 0.5 .* B .+ 0.5 .* B_new      # damped step for stability

        # θ-step: Adam on the ODE mismatch with the trajectory fixed
        Xg_now = X_g(B); dXg_now = dX_g(B)
        loss_fixed = β -> theta_loss(β, Xg_now, dXg_now, ρ_eff)
        for _ in 1:n_inner
            adam_t += 1
            ForwardDiff.gradient!(result, loss_fixed, beta)
            grad = DiffResults.gradient(result)
            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad .^ 2
            m_hat = m_adam ./ (1 - β1_adam^adam_t)
            v_hat = v_adam ./ (1 - β2_adam^adam_t)
            beta .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
        end

        J = full_objective(B, beta)
        if verbose && (iter <= 3 || iter % 25 == 0)
            println("  iter $iter: J=$(round(J, sigdigits=6))")
        end
        if iter > n_ramp &&
           isfinite(J_prev) && abs(J_prev - J) / max(abs(J_prev), 1e-12) < 1e-9
            converged = true
            final_iter = iter
            if verbose; println("  Converged at iter $iter"); end
            break
        end
        J_prev = J
    end
    J_final = full_objective(B, beta)

    # ── Build solution ───────────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] = Φd * B[:, sk] .+ m_center[sk]
    end
    # Weighted sum of squares over usable cells, matching the data_loss
    # convention of the other solvers (masked / NaN cells are skipped so
    # one missing observation does not turn the reported loss into NaN).
    data_loss = weighted_data_loss(prob, pred)

    # Sentinel accounting on the COLLOCATION GRID, which is where
    # `rhs_on_grid` evaluates the dynamics — see `_dynamics_failure_verdict`
    # (solver.jl). Pre-fix this solver stamped `converged = true,
    # reason = :converged_tol` on a fit in which every grid evaluation had
    # fallen back to the sentinel.
    #
    # KNOWN LIMITATION — and it is a CLASS boundary, not an RKHS quirk.
    #
    # The count is COMPLETE for the smooth-then-match solvers
    # (GradientMatching, TwoStageSolver, IntegralMatchingSolver, BNGSolver):
    # they count over `y_smooth`, which is β-INDEPENDENT, so this final sweep
    # evaluates exactly the states every iteration evaluated. It is
    # INCOMPLETE here and in ODINSolver, because both estimate the trajectory
    # JOINTLY with β — so the final sweep sees a different trajectory from
    # the one the optimisation ran on, and the sentinel can push the states
    # off the throwing region entirely, leaving the count at 0.
    #
    # The count is therefore UNSTABLE for this solver. On a logistic fixture
    # whose dynamics raise for 2.5 ≤ u ≤ 7, the independent review of this
    # change measured 9 (maxiters, n_repr_points) settings: 7 read
    # `n_dynamics_failures = 0` with objectives from 7.6e7 to 1.9e12, and 2
    # read 1. Re-measured here on the runtests.jl `sentinel substitutions
    # block convergence` fixture (`Random.Xoshiro(11)`, maxiters=50):
    # `n_dynamics_failures = 0`, `objective = 8.9e9`, `reason = :maxiters`.
    #
    # A DIVERGENT OBJECTIVE IS NOT A USABLE RULE: across the same probes the
    # objective on a garbage fit ranged from 13990 to 7.8e10, so no fixed
    # figure separates the cases. What DID hold across all 19 settings the
    # review tried is that a garbage fit never reached `converged = true`:
    # every `converged = true` coincided with a genuinely clean fit (function
    # RMSE 0.1133, matching the clean control), and every garbage case
    # reported `converged = false, reason = :maxiters`.
    #
    # FOLLOW-UP, not implemented: a principled non-magic guard is available
    # here, because the sentinel is a known constant. With `n_g` grid points
    # each contributing at most a `1e6` residual to a clean fit,
    # `objective > 1e6 * n_g` is a threshold derived from the sentinel rather
    # than tuned to a fixture.
    n_dyn_fail = _count_dynamics_failures(prob, t_grid, X_g(B), beta)
    conv_reason_rkhs = converged ? :converged_tol : :maxiters
    converged, conv_reason_rkhs = _dynamics_failure_verdict(
        "RKHSSolver", n_dyn_fail, n_g, converged, conv_reason_rkhs)

    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    edf = Float64(n_beta)   # number of θ parameters (trajectory dof excluded)

    PSMSolution(params, J_final, data_loss, edf, Float64[λ, ρ],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=converged, iterations=final_iter,
                 reason=conv_reason_rkhs, method=:rkhs,
                 n_dynamics_failures=n_dyn_fail,
                 kernel=alg.kernel, lengthscale=ℓ))
end
