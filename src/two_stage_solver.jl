# ─── Two-stage solver (Wood 2001 / deGradInfer pattern) ─────────────
#
# Simplest baseline for partially specified models:
#   Stage 1 — Smooth: Fit cubic splines to each observed state independently,
#             obtaining ŷ_k(t) and their derivatives dŷ_k/dt.
#   Stage 2 — Match:  Optimize unknown function parameters β by minimizing
#             derivative mismatch + smoothing penalty:
#             L(β) = Σ_k Σ_i ||dŷ_k/dt(t_i) - f_k(ŷ(t_i), p(β), t_i)||²
#                  + Σ_j λ_j β_j' S_j β_j
#
# Uses Adam optimizer with ForwardDiff gradients (same as BNGSolver / AdamSolver).

using LinearAlgebra: dot, norm

# ─── Main two-stage solver ──────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::TwoStageSolver)

Fit a partially specified model using a two-stage estimation procedure.
In stage 1 the data are smoothed to estimate state trajectories and
derivatives; in stage 2 the unknown-function parameters are estimated by
nonlinear least squares against those smoothed derivatives.

# Algorithm
1. Smooth each observed state with a penalized (GCV) smoothing spline and evaluate derivatives
   at the data time points.
2. Hold unobserved states constant at their initial-condition values and
   MASK them out of the matching loss (their fabricated targets carry no
   information).
3. Minimise ∑ₜ ‖x′(t) − f(x(t), uf(t; β))‖² plus the smoothing penalty with
   respect to β using Adam with ForwardDiff gradients.
4. Reconstruct the unknown functions at the fitted parameters.

# References
- Voss & Feng (2008), "Modelling Non-linear Differential Equations:
  a Two-Stage Method", JRSS-C.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
— see the `TwoStageSolver` docstring for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::TwoStageSolver)
    _validate_problem(prob, "TwoStageSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)

    # ── Stage 1: Smooth data and compute derivatives ─────────────

    if verbose; println("TwoStageSolver Stage 1: Smoothing data with cubic splines..."); end

    y_smooth = zeros(n_times, n_vars)
    dydt = zeros(n_times, n_vars)

    observed_states = Set{Int}()

    if prob.discrete
        # For discrete models: smooth data, then use forward differences
        # as the matching target (next-state prediction)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            val, _ = _smoothing_spline_masked(times, Float64.(prob.data_values[:, j]),
                                              @view(prob.data_weights[:, j]);
                                              max_basis=alg.n_basis_smooth)
            for i in 1:n_times
                y_smooth[i, sk] = val(times[i])
            end
        end
        # Target: smoothed next-state y_smooth[i+1, k]
        for k in 1:n_vars
            for i in 1:(n_times - 1)
                dydt[i, k] = y_smooth[i + 1, k]
            end
            dydt[n_times, k] = y_smooth[n_times, k]
        end
    else
        # For continuous models: penalized (GCV) smoothing spline + derivative
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            val, der = _smoothing_spline_masked(times, Float64.(prob.data_values[:, j]),
                                                @view(prob.data_weights[:, j]);
                                                max_basis=alg.n_basis_smooth)
            for i in 1:n_times
                y_smooth[i, sk] = val(times[i])
                dydt[i, sk] = der(times[i])
            end
        end
    end

    # Handle unobserved states: constant at initial condition value
    for k in 1:n_vars
        if k ∉ observed_states
            u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                           prob.u0[k])
            y_smooth[:, k] .= u0_k
            dydt[:, k] .= 0.0
        end
    end

    if verbose
        println("  Smoothed $(length(observed_states))/$n_vars observed states, " *
                "$n_times time points")
    end

    # ── Stage 2: Optimize β via Adam with derivative matching ────

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

    # Number of time points used for matching
    n_match = prob.discrete ? n_times - 1 : n_times

    # Per-(time, state) usability. The smoother above now drops masked cells
    # from its fit, so `y_smooth`/`dydt` at a masked time are extrapolations
    # rather than data; matching against them would invent information. A
    # (time, state) pair counts if ANY observation column mapping to that
    # state is usable at that time. All-true for complete data, so the loss
    # below is unchanged there.
    match_usable = falses(n_times, n_vars)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:n_times
            usable_cell(prob, i, j) && (match_usable[i, sk] = true)
        end
    end
    any(match_usable) || error("TwoStageSolver: every observation is masked " *
        "(all data_weights are 0 or all data_values are NaN); there is " *
        "nothing to fit.")

    if verbose
        println("TwoStageSolver Stage 2: Derivative matching — $n_beta params, " *
                "$n_match match points, $n_vars states")
        println("  maxiters=$(alg.maxiters), lr=$(alg.lr), " *
                "lambda_smooth=$(alg.lambda_smooth)")
    end

    # Derivative matching loss (ForwardDiff-compatible)
    lambda_smooth = alg.lambda_smooth

    function twostage_loss(β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)
        loss_val = zero(T_el)

        for i in 1:n_match
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                # Only observed states carry real targets; unobserved ones
                # were filled with a constant state and zero derivative,
                # and matching those fabricated values biases the fit.
                k in observed_states || continue
                # …and only at times where that state actually has a usable
                # observation (see `match_usable`).
                match_usable[i, k] || continue
                loss_val += (dydt[i, k] - du[k])^2
            end
        end

        # Penalty terms from approximators
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            params_k = β_eval[offset+1:offset+np]
            offset += np

            # Non-built-in (custom protocol) types take the generic branch
            # too; built-in treatment is unchanged (see _BUILTIN_APPROX_TYPES).
            if approx isa BSplineApproximator || approx isa GPApproximator || approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator || approx isa ShapeConstrainedGPApproximator || approx isa TensorBSplineApproximator || approx isa SingleIndexApproximator || approx isa TransformedCovariateApproximator || !(approx isa _BUILTIN_APPROX_TYPES)
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
            # NeuralApproximator: no explicit penalty (implicit regularization)
        end

        loss_val
    end

    # Adam optimizer state
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
        # Compute gradient via ForwardDiff
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        ForwardDiff.gradient!(result, twostage_loss, beta)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)

        if loss_val < best_loss
            best_loss = loss_val
            best_beta .= beta
        end
        loss_window[mod1(iter, 30)] = loss_val

        # Cosine learning rate annealing
        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        # Adam update
        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
            println("  iter $iter: loss=$(round(loss_val, sigdigits=5)) " *
                    "lr=$(round(lr_t, sigdigits=3))")
        end

        # Convergence: loss plateau over window. Guard against SPURIOUS
        # plateaus manufactured by the cosine lr schedule: near maxiters
        # lr_t → 0, so the loss stops moving no matter how far from an
        # optimum we are. Only declare plateau-convergence while the step
        # size is still meaningful (lr_t > 5% of the base lr).
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

    # Fitted values: use smoothed states projected to observed variables
    # (no ODE integration needed — this is the two-stage advantage)
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= y_smooth[:, sk]
    end

    # Data loss against original observations
    data_loss = weighted_data_loss(prob, pred)

    # Build evaluators for each approximator
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    # Build ComponentArray of fitted parameters
    edf = Float64(n_beta)
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "deriv_SS=$(round(best_loss, sigdigits=5)) EDF=$(round(edf, digits=1))")
    end

    PSMSolution(params, best_loss, data_loss, edf, Float64[lambda_smooth],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=conv_converged, iterations=final_iter,
                 reason=conv_reason, method=:two_stage))
end
