# ─── Integral Matching solver (Dattner & Klaassen 2015) ──────────────
#
# Instead of matching derivatives (noisy) or integrating the ODE (expensive),
# integrate both sides of the ODE:
#   x(t) - x(t₀) = ∫_{t₀}^{t} f(x(s), p, s) ds
#
# Algorithm:
#   Stage 1 — Smooth observed states with cubic splines → ŷ(t)
#   Stage 2 — For each time point tᵢ, compute:
#             LHS: ŷ(tᵢ) - ŷ(t₁)           (from smoothed data)
#             RHS: ∫_{t₁}^{tᵢ} f(ŷ(s), p(β), s) ds   (trapezoidal rule)
#             Minimize Σ_k Σ_i ||LHS_k(tᵢ) - RHS_k(tᵢ)||²
#
# Key advantage: avoids both derivative estimation AND full ODE integration.
# More robust than gradient matching because integrals smooth out noise.
#
# Reference: Dattner & Klaassen (2015), EJS 9(2), 1939-1973
#            R package `simode` (Yaari & Dattner)

using LinearAlgebra: dot, norm

"""
    solve(prob::PSMProblem, alg::IntegralMatchingSolver)

Fit a partially specified model by integral matching (Dattner & Klaassen 2015).

Instead of matching noisy derivatives, this method integrates both sides of the
ODE, comparing the smoothed trajectory increments with numerical quadrature of
the right-hand side. This avoids both derivative estimation and repeated ODE
integration, providing a robust and computationally efficient estimator.

# Algorithm
1. Smooth each observed state with cubic splines → ŷ(t).
2. Compute trajectory increments: Δᵢₖ = ŷₖ(tᵢ) − ŷₖ(t₁).
3. Compute cumulative integrals of the ODE RHS using the trapezoidal rule
   evaluated at the smoothed states: Iᵢₖ = ∫₁ⁱ fₖ(ŷ(s), p(β), s) ds.
4. Minimize L(β) = Σₖ Σᵢ (Δᵢₖ − Iᵢₖ)² + λ β'Sβ w.r.t. β using Adam.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
— see the `IntegralMatchingSolver` docstring for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::IntegralMatchingSolver)
    _validate_problem(prob, "IntegralMatchingSolver"; require_continuous=true,
                      reject_delays=true)
    _require_gaussian_likelihood(prob, "IntegralMatchingSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    u0_init = _resolve_u0_init(prob)
    n_vars = length(u0_init)
    n_obs = size(prob.data_values, 2)

    # ── Stage 1: Smooth data ─────────────────────────────────────
    if verbose; println("IntegralMatchingSolver Stage 1: Smoothing data..."); end

    y_smooth = zeros(n_times, n_vars)
    t_mid = n_times > 1 ? [0.5 * (times[i] + times[i+1]) for i in 1:(n_times - 1)] : Float64[]
    y_mid = zeros(max(n_times - 1, 0), n_vars)
    observed_states = Set{Int}()

    # Penalized GCV smoother — an interpolating spline evaluated at its own
    # knots returns the raw data (a no-op "smoothing" step). Evaluate on both
    # data times and midpoints so Stage 2 uses 4th-order composite Simpson
    # quadrature instead of 2nd-order trapezoidal rule.
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        push!(observed_states, sk)
        # Masked rows must be dropped from the smoother's normal equations,
        # not merely ignored afterwards: one NaN makes every spline
        # coefficient NaN, so `y_smooth` — and hence the integral-matching
        # residual — is NaN everywhere.
        sval, _ = _smoothing_spline_masked(times,
                                           Float64.(prob.data_values[:, j]),
                                           @view(prob.data_weights[:, j]))
        for i in 1:n_times
            y_smooth[i, sk] = sval(times[i])
        end
        for i in 1:(n_times - 1)
            y_mid[i, sk] = sval(t_mid[i])
        end
    end

    # Unobserved states: hold at initial condition
    for k in 1:n_vars
        if k ∉ observed_states
            y_smooth[:, k] .= u0_init[k]
            if n_times > 1
                y_mid[:, k] .= u0_init[k]
            end
        end
    end

    # Which (time, state) cells carry a usable observation. Masked rows are
    # dropped from the smoother above, so `y_smooth` there is a pure
    # interpolation with no data behind it: including it in the loss would
    # fit the model to the smoother's own extrapolation.
    match_usable = falses(n_times, n_vars)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:n_times
            usable_cell(prob, i, j) && (match_usable[i, sk] = true)
        end
    end
    usable_idx = Vector{Vector{Int}}(undef, n_vars)
    for k in 1:n_vars
        if k in observed_states
            idx_k = findall(@view match_usable[:, k])
            isempty(idx_k) && error("IntegralMatchingSolver: state $k has no " *
                "usable observation (every cell is masked — zero weight or " *
                "non-finite value); there is nothing to match.")
            usable_idx[k] = idx_k
        else
            usable_idx[k] = Int[]
        end
    end

    # Dattner & Klaassen (2015) time-mean centering eliminates the unknown
    # initial state x_k(t_0) without anchoring every residual to the noisy
    # boundary estimate ŷ_k(t_1): match (ŷ_k(t_i) - mean(ŷ_k)) against
    # (I_k(t_i) - mean(I_k)) across all usable times.
    y_centered = zeros(n_times, n_vars)
    for k in observed_states
        idx_k = usable_idx[k]
        ybar_k = sum(y_smooth[i, k] for i in idx_k) / length(idx_k)
        for i in idx_k
            y_centered[i, k] = y_smooth[i, k] - ybar_k
        end
    end

    if verbose
        println("  Smoothed $(length(observed_states))/$n_vars states, $n_times points")
    end

    # ── Stage 2: Integral matching via Adam ──────────────────────
    # Initialize parameters
    beta = Float64[]

    for approx in prob.approximators
        if approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta, neural_init_params(approx, rng))
        else
            append!(beta, initial_params(approx))
        end
    end
    n_beta = length(beta)

    if verbose
        println("IntegralMatchingSolver Stage 2: Integral matching — $n_beta params")
        println("  maxiters=$(alg.maxiters), lr=$(alg.lr), lambda=$(alg.lambda_smooth)")
    end

    lambda_smooth = alg.lambda_smooth
    im_mat_pens = Tuple{Int,Int,Matrix{Float64}}[]
    im_l2_pens = Tuple{Int,Int,Float64}[]
    let off = 0
        for approx in prob.approximators
            np = nparams(approx)
            if approx isa COMONetApproximator
                approx.penalty_weight > 0.0 && push!(im_l2_pens, (off, np, approx.penalty_weight))
            elseif !(approx isa NeuralApproximator)
                S = penalty_matrix(approx)
                S !== nothing && push!(im_mat_pens, (off, np, S))
            end
            off += np
        end
    end

    # Pre-compute time step widths for composite Simpson quadrature
    dt = diff(times)

    function integral_loss(β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)

        # Evaluate ODE RHS at smoothed data times and midpoints
        F = zeros(T_el, n_times, n_vars)
        for i in 1:n_times
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch e
                _is_program_error(e) && rethrow()
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                F[i, k] = du[k]
            end
        end
        F_mid = zeros(T_el, max(n_times - 1, 0), n_vars)
        for i in 1:(n_times - 1)
            u_m = T_el.(y_mid[i, :])
            try
                prob.dynamics!(du, u_m, p, t_mid[i])
            catch e
                _is_program_error(e) && rethrow()
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                F_mid[i, k] = du[k]
            end
        end

        # Cumulative integral via composite Simpson's rule:
        # I[i,k] = ∫_{t_1}^{t_i} f_k(ŷ(s), p, s) ds
        I_cum = zeros(T_el, n_times, n_vars)
        for k in 1:n_vars
            for i in 2:n_times
                I_cum[i, k] = I_cum[i-1, k] +
                    (F[i-1, k] + 4 * F_mid[i-1, k] + F[i, k]) * dt[i-1] / 6
            end
        end

        # Dattner & Klaassen (2015) time-mean-centered integral matching loss
        loss_val = zero(T_el)
        for k in 1:n_vars
            k in observed_states || continue
            idx_k = usable_idx[k]
            nk = length(idx_k)
            I_bar = zero(T_el)
            for i in idx_k
                I_bar += I_cum[i, k]
            end
            I_bar /= nk
            for i in idx_k
                loss_val += (y_centered[i, k] - (I_cum[i, k] - I_bar))^2
            end
        end

        # Smoothing penalty (precomputed)
        for (off, np, w_l2) in im_l2_pens
            pk = @view β_eval[off+1:off+np]
            loss_val += w_l2 * sum(abs2, pk)
        end
        for (off, np, S) in im_mat_pens
            pk = @view β_eval[off+1:off+np]
            loss_val += lambda_smooth * dot(pk, S * pk)
        end

        loss_val
    end

    # Adam optimizer
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_beta)
    v_adam = zeros(n_beta)
    best_beta = copy(beta)
    best_loss = Inf
    loss_window = fill(Inf, 30)
    final_iter = alg.maxiters
    # Honest convergence reporting: converged only when the plateau
    # criterion actually fires; otherwise the loop exhausted maxiters.
    conv_converged = false
    conv_reason = :maxiters
    result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))

    for iter in 1:alg.maxiters
        final_iter = iter
        ForwardDiff.gradient!(result, integral_loss, beta)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)

        if loss_val < best_loss
            best_loss = loss_val
            best_beta .= beta
        end
        loss_window[mod1(iter, 30)] = loss_val

        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
            println("  iter $iter: loss=$(round(loss_val, sigdigits=5)) " *
                    "lr=$(round(lr_t, sigdigits=3))")
        end

        # Plateau convergence, guarded against the cosine lr schedule
        # manufacturing a plateau as lr_t → 0 near maxiters: only trust the
        # criterion while the step size is still meaningful.
        # `best_loss < 1e9` is AdamSolver's failure-sentinel guard: the
        # dynamics fall back to `du .= 1e6` when the RHS throws, so a run
        # pinned at that sentinel is a stuck solver, not a converged one.
        if iter > 60 && best_loss < 1e9 && lr_t > 0.05 * lr
            recent_min = minimum(loss_window)
            recent_max = maximum(loss_window)
            if (recent_max - recent_min) / max(abs(recent_min), 1.0) < 1e-6
                if verbose; println("  Converged at iter $iter (loss plateau)"); end
                final_iter = iter
                conv_converged = true
                conv_reason = :plateau
                break
            end
        end
    end
    beta .= best_beta

    if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end

    # ── Build solution ───────────────────────────────────────────
    # Sentinel accounting at the fitted parameters — see
    # `_dynamics_failure_verdict` (solver.jl). It runs BEFORE the reporting
    # simulation below, because it ERRORS when nothing was fitted at all:
    # simulating a non-fit first emitted a warning promising to report the
    # smoother on a call that then threw. It is also the only thing here that
    # forces `converged`/`reason`; the simulation fallback sets `sim_ok` and
    # nothing else, so the two never conflict.
    n_dyn_fail = _count_dynamics_failures(prob, times, y_smooth, beta)
    conv_converged, conv_reason = _dynamics_failure_verdict(
        "IntegralMatchingSolver", n_dyn_fail, n_times, conv_converged,
        conv_reason)

    # Fitted values: the trajectory the fitted model produces, integrated from
    # `prob.u0` at the fitted beta.
    #
    # Before B7 this projected the stage-1 smoother (`y_smooth[:, sk]`), which
    # made `fitted_values` and `data_loss` functions of the DATA ALONE:
    # measured, two problems differing only in u0 (1.0 vs 4.0) returned bitwise
    # identical `data_loss` (0.09625512373) while their simulated losses were
    # 0.18993396 and 205.61006, a factor of 1083.
    sim_ok = true
    pred = try
        simulate_psm(prob, beta)
    catch e
        _is_program_error(e) && rethrow()
        sim_ok = false
        @warn "IntegralMatchingSolver: integrating the fitted dynamics from " *
              "u0 failed ($(sprint(showerror, e))). Reporting the stage-1 " *
              "data smoother as `fitted_values`; `data_loss` therefore " *
              "measures the SMOOTHER, not the model fit. See " *
              "`convergence.simulation_failed`."
        pr = zeros(n_times, n_obs)
        for j in 1:n_obs
            pr[:, j] .= y_smooth[:, prob.obs_to_state[j]]
        end
        pr
    end

    data_loss = weighted_data_loss(prob, pred)

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

    edf = Float64(n_beta)

    PSMSolution(params, best_loss, data_loss, edf, Float64[lambda_smooth],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=conv_converged, iterations=final_iter,
                 reason=conv_reason, method=:integral_matching,
                 n_dynamics_failures=n_dyn_fail,
                 simulation_failed=!sim_ok))
end
