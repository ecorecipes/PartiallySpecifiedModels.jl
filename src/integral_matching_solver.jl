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
    _validate_problem(prob, "IntegralMatchingSolver"; require_continuous=true)
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_obs = size(prob.data_values, 2)

    # ── Stage 1: Smooth data ─────────────────────────────────────
    if verbose; println("IntegralMatchingSolver Stage 1: Smoothing data..."); end

    y_smooth = zeros(n_times, n_vars)
    observed_states = Set{Int}()

    # Penalized GCV smoother — an interpolating spline evaluated at its own
    # knots returns the raw data (a no-op "smoothing" step).
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        push!(observed_states, sk)
        # Masked rows must be dropped from the smoother's normal equations,
        # not merely ignored afterwards: one NaN makes every spline
        # coefficient NaN, so `y_smooth` — and hence the integral-matching
        # residual `delta` — is NaN everywhere. There is no finiteness
        # sentinel in `integral_loss`, so the NaN reaches the Adam moments
        # and the solver silently returns its initial parameters.
        sval, _ = _smoothing_spline_masked(times,
                                           Float64.(prob.data_values[:, j]),
                                           @view(prob.data_weights[:, j]))
        for i in 1:n_times
            y_smooth[i, sk] = sval(times[i])
        end
    end

    # Unobserved states: hold at initial condition
    for k in 1:n_vars
        if k ∉ observed_states
            u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                           prob.u0[k])
            y_smooth[:, k] .= u0_k
        end
    end

    # Which (time, state) cells carry a usable observation, and which row
    # each observed state measures its increment FROM. Masked rows are
    # dropped from the smoother above, so `y_smooth` there is a pure
    # interpolation with no data behind it: including it in the loss would
    # fit the model to the smoother's own extrapolation. The three sibling
    # gradient-matching solvers (TwoStage, BNG, ODIN) all gate; this one
    # ran over every time index ungated.
    match_usable = falses(n_times, n_vars)
    base_row = ones(Int, n_vars)          # baseline row per state
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:n_times
            usable_cell(prob, i, j) && (match_usable[i, sk] = true)
        end
    end
    for k in 1:n_vars
        k in observed_states || continue
        # The baseline ŷ_k(t_b) must itself be anchored on data: with row 1
        # masked, `y_smooth[1, k]` is an extrapolation and every increment
        # in the column inherits its error.
        b = findfirst(@view match_usable[:, k])
        b === nothing && error("IntegralMatchingSolver: state $k has no " *
            "usable observation (every cell is masked — zero weight or " *
            "non-finite value); there is nothing to match.")
        base_row[k] = b
    end

    # Trajectory increments: Δ[i,k] = ŷ_k(t_i) - ŷ_k(t_b(k))
    delta = zeros(n_times, n_vars)
    for k in 1:n_vars
        for i in 1:n_times
            delta[i, k] = y_smooth[i, k] - y_smooth[base_row[k], k]
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

    # Pre-compute time step widths for the composite trapezoidal quadrature
    dt = diff(times)

    function integral_loss(β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)

        # Evaluate ODE RHS at all smooth points
        F = zeros(T_el, n_times, n_vars)
        for i in 1:n_times
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                F[i, k] = du[k]
            end
        end

        # Cumulative integral via composite trapezoidal rule:
        # I[i,k] = ∫_{t_1}^{t_i} f_k(ŷ(s), p, s) ds
        I_cum = zeros(T_el, n_times, n_vars)
        for k in 1:n_vars
            for i in 2:n_times
                # Trapezoidal: I[i] = I[i-1] + (f[i-1] + f[i]) * dt / 2
                I_cum[i, k] = I_cum[i-1, k] + (F[i-1, k] + F[i, k]) * dt[i-1] / 2
            end
        end

        # Loss over OBSERVED states (unobserved states hold a fabricated
        # constant, so their delta ≡ 0 would push the unknown functions to
        # zero the RHS along a fictitious path) and USABLE rows only.
        #
        # `I_cum` is still cumulative from t_1, so the increment matching
        # the baseline-shifted `delta[i,k] = ŷ(t_i) − ŷ(t_b)` is
        # `I_cum[i,k] − I_cum[b,k]`. For complete data b = 1 and
        # `I_cum[1,k] = 0`, so this is bit-for-bit the previous expression.
        loss_val = zero(T_el)
        for k in 1:n_vars
            k in observed_states || continue
            b = base_row[k]
            for i in 1:n_times
                (i != b && match_usable[i, k]) || continue
                loss_val += (delta[i, k] - (I_cum[i, k] - I_cum[b, k]))^2
            end
        end

        # Smoothing penalty
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            params_k = β_eval[offset+1:offset+np]
            offset += np

            if approx isa BSplineApproximator || approx isa GPApproximator ||
               approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator ||
               approx isa TensorBSplineApproximator
                S = penalty_matrix(approx)
                if S !== nothing
                    loss_val += lambda_smooth * dot(params_k, S * params_k)
                end
            elseif approx isa ShapeConstrainedBSplineApproximator
                S = penalty_matrix(approx)
                if S !== nothing
                    loss_val += lambda_smooth * dot(params_k, S * params_k)
                end
            elseif approx isa COMONetApproximator
                loss_val += approx.penalty_weight * sum(abs2, params_k)
            end
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

    for iter in 1:alg.maxiters
        final_iter = iter
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
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
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= y_smooth[:, sk]
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
                 reason=conv_reason, method=:integral_matching))
end
