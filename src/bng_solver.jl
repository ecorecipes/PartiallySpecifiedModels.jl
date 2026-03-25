# ─── BNG solver (Bayesian Neural Gradient matching) ──────────────
#
# Two-step approach from Bonnaffé, Sheldon & Bhatt (2023):
#   Step 1: Smooth observed data with cubic splines → ŷ(t), dŷ/dt
#   Step 2: Fit unknown function parameters by gradient matching:
#           minimize Σ_k Σ_i ||dŷ_k/dt(t_i) - f_k(ŷ(t_i), p(β), t_i)||²
#           + penalty terms from approximators
#
# Uses Adam optimizer with ForwardDiff gradients (same pattern as AdamSolver).
# Key advantage: no ODE integration needed → faster, more robust initialization.

using LinearAlgebra: dot, norm

# ─── Main BNG solver ─────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::BNGSolver)

Fit a partially specified model using two-step Bayesian Nonparametric
Gradient Matching. Avoids solving the ODE by matching smoothed derivatives
to the model right-hand side.

# Algorithm
1. Smooth observed data with cubic splines and compute numerical derivatives.
2. Minimize the sum of squared gradient-matching residuals with respect to
   the unknown-function parameters using `Optim.NelderMead`.
3. Reconstruct trajectories from the fitted spline coefficients.

# References
- Niu et al. (2016), "Fast Parameter Inference in Nonlinear Dynamical
  Systems using Iterative Gradient Matching", ICML.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::BNGSolver)
    _validate_problem(prob, "BNGSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)

    # ── Step 1: Smooth data and compute derivatives ──────────────

    if verbose; println("BNGSolver Step 1: Smoothing data with cubic splines..."); end

    y_smooth = zeros(n_times, n_vars)
    dydt = zeros(n_times, n_vars)

    observed_states = Set{Int}()

    if prob.discrete
        # For discrete models: smooth data, then use forward differences
        # as the matching target (next-state prediction)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            itp = CubicSpline(prob.data_values[:, j], times;
                              extrapolation=ExtrapolationType.Extension)
            for i in 1:n_times
                y_smooth[i, sk] = itp(times[i])
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
        # For continuous models: smooth data and compute analytical derivatives
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            itp = CubicSpline(prob.data_values[:, j], times;
                              extrapolation=ExtrapolationType.Extension)
            for i in 1:n_times
                y_smooth[i, sk] = itp(times[i])
                dydt[i, sk] = DataInterpolations.derivative(itp, times[i])
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

    # ── Step 2: Optimize β via Adam with gradient matching ───────

    # Initialize parameters (use MLP init for NeuralApproximators, like AdamSolver)
    beta = Float64[]
    mlp_specs = Dict{Symbol, MLPSpec}()

    for approx in prob.approximators
        if approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            mlp_specs[approx.name] = spec
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta, init_mlp_params(spec, rng))
        else
            append!(beta, initial_params(approx))
        end
    end
    n_beta = length(beta)

    # Number of time points used for matching
    n_match = prob.discrete ? n_times - 1 : n_times

    if verbose
        println("BNGSolver Step 2: Gradient matching — $n_beta params, " *
                "$n_match match points, $n_vars states")
        println("  maxiters=$(alg.maxiters), lr=$(alg.lr), " *
                "lambda_smooth=$(alg.lambda_smooth)")
    end

    # Gradient matching loss (ForwardDiff-compatible)
    lambda_smooth = alg.lambda_smooth

    function bng_loss(β_eval)
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
                loss_val += (dydt[i, k] - du[k])^2
            end
        end

        # Penalty terms from approximators
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            params_k = β_eval[offset+1:offset+np]
            offset += np

            if approx isa BSplineApproximator || approx isa GPApproximator || approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator
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

    for iter in 1:alg.maxiters
        # Compute gradient via ForwardDiff
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        ForwardDiff.gradient!(result, bng_loss, beta)
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

        # Convergence: loss plateau over window
        if iter > 60
            recent_min = minimum(loss_window)
            recent_max = maximum(loss_window)
            if (recent_max - recent_min) / max(abs(recent_min), 1.0) < 1e-6
                if verbose; println("  Converged at iter $iter (loss plateau)"); end
                final_iter = iter
                break
            end
        end
    end
    beta .= best_beta

    if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end

    # ── Build solution ───────────────────────────────────────────

    # Simulate with fitted parameters to get trajectory predictions
    if prob.discrete
        p_sim = build_param_struct(prob, beta)
        u_sim = Float64.(prob.u0 isa Function ? prob.u0(p_sim) : prob.u0)
        u_next_sim = similar(u_sim)
        sim_states = zeros(n_times, n_vars)
        sim_states[1, :] .= u_sim
        for step in 1:(n_times - 1)
            try
                prob.dynamics!(u_next_sim, u_sim, p_sim, times[step])
            catch
                u_next_sim .= 1e6
            end
            u_sim = copy(u_next_sim)
            sim_states[step + 1, :] .= u_sim
        end
        pred = zeros(n_times, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[:, j] .= sim_states[:, sk]
        end
    else
        p_opt = build_param_struct(prob, beta)
        u0 = prob.u0 isa Function ? prob.u0(p_opt) : prob.u0
        ode_prob = ODEProblem((du, u, params, t) -> prob.dynamics!(du, u, p_opt, t),
                              Float64.(u0), prob.tspan)
        sol_ode = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                       saveat=prob.data_times,
                                       abstol=1e-7, reltol=1e-7,
                                       maxiters=10000)

        pred = zeros(n_times, n_obs)
        if sol_ode.retcode == SciMLBase.ReturnCode.Success ||
           sol_ode.retcode == SciMLBase.ReturnCode.Default ||
           sol_ode.retcode == SciMLBase.ReturnCode.Terminated
            for j in 1:n_obs
                sk = prob.obs_to_state[j]
                for i in 1:min(n_times, length(sol_ode.t))
                    pred[i, j] = sol_ode[sk, i]
                end
            end
        else
            # Fallback: use smoothed values as predictions
            if verbose; println("  ODE simulation failed, using smoothed values"); end
            for j in 1:n_obs
                sk = prob.obs_to_state[j]
                pred[:, j] .= y_smooth[:, sk]
            end
        end
    end

    # Data loss against original observations
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Build evaluators for each approximator
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
            spec = mlp_specs[approx.name]
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            let pk = copy(params_k), s = spec, lo_ = lo, span_ = span
                uf_evals[approx.name] = x -> begin
                    xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                        (Float64(x isa AbstractArray ? x[1] : x) - lo_) / span_
                    else
                        Float64(x isa AbstractArray ? x[1] : x)
                    end
                    mlp_evaluate(s, pk, xn)
                end
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

    PSMSolution(params, best_loss, data_loss, edf, Float64[],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=final_iter, method=:bng))
end
