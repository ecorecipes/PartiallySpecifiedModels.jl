# ─── Rodeo Solver for PSMProblem ─────────────────────────────────────
#
# Uses the probabilistic ODE solver (rodeo) for parameter inference
# in partially specified models. The key idea:
#
# 1. The probabilistic solver approximates the ODE solution AND provides
#    uncertainty estimates via Kalman filtering/smoothing
# 2. The approximate loglikelihood p(Y|β) is differentiable w.r.t.
#    unknown function parameters β
# 3. L-BFGS optimizes β to maximize the loglikelihood
#
# Supports both "basic" (plug-in mean) and "fenrir" (marginal) likelihood
# approximations.
#
# Reference: Wu & Lysy (2024), Tronarp et al (2022)

"""
    solve(prob::PSMProblem, alg::RodeoSolver)

Fit a partially specified model using the RODEO (Reverse-mode ODE
Observation) probabilistic numerics solver. An integrated Brownian motion
(IBM) prior is placed over the state and Kalman filtering/smoothing is
used to condition on both the observations and the ODE constraints.

# Algorithm
1. Discretise the time span on a fine grid (`n_steps` points).
2. Set up the IBM prior of order `n_deriv` for each state variable.
3. Forward Kalman filter: propagate the IBM prior and assimilate data
   observations and ODE residual pseudo-observations.
4. Rauch–Tung–Striebel backward smoother to obtain the posterior state.
5. Optimise unknown-function parameters by maximising the marginal
   log-likelihood of the Kalman filter.

# References
- Tronarp et al. (2022), "Fenrir: Physics-Enhanced Regression for Initial
  Value Problems", ICML.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::RodeoSolver)
    _validate_problem(prob, "RodeoSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)

    if verbose
        println("RodeoSolver: n_steps=$(alg.n_steps), n_deriv=$(alg.n_deriv), " *
                "method=$(alg.method), interrogate=$(alg.interrogate)")
    end

    # Initialize parameters
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Sigma parameters (IBM scale): one per state variable
    # Auto-scale from data range
    sigma = if alg.sigma === nothing
        sig = Float64[]
        for k in 1:n_vars
            # Check if this state is observed
            obs_idx = findfirst(j -> prob.obs_to_state[j] == k, 1:n_obs)
            if obs_idx !== nothing
                data_range = maximum(prob.data_values[:, obs_idx]) -
                             minimum(prob.data_values[:, obs_idx])
                push!(sig, max(data_range * 0.01, 0.01))
            else
                push!(sig, 1.0)
            end
        end
        sig
    else
        alg.sigma
    end

    # Observation noise variance (auto-estimate from data if not provided)
    obs_var = if alg.obs_var === nothing
        # Rough estimate: 1% of data variance
        total_var = 0.0
        for j in 1:n_obs
            total_var += var(prob.data_values[:, j])
        end
        max(total_var / n_obs * 0.01, 1e-4)
    else
        alg.obs_var
    end

    if verbose
        println("  σ (IBM scale): $(round.(sigma, sigdigits=3))")
        println("  obs_var: $(round(obs_var, sigdigits=3))")
        println("  $(n_beta) approximator params")
    end

    # Choose loglikelihood method
    loglik_fn = alg.method == :fenrir ? fenrir_loglik : basic_loglik

    # Compute smoothing penalty matrices for B-spline approximators
    smooth_mats = Matrix{Float64}[]
    smooth_offsets = Int[]
    offset_acc = 0
    for approx in prob.approximators
        np = nparams(approx)
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            S = spline_penalty_matrix(knots_x)
            push!(smooth_mats, S)
            push!(smooth_offsets, offset_acc)
        end
        offset_acc += np
    end

    # Auto-scale smoothing from data variance
    smooth_lambda = 0.1 / max(obs_var, 1e-6)

    # Optimization: maximize loglikelihood w.r.t. beta
    function neg_loglik(β_)
        p_ = build_autodiff_param_struct(prob, β_)

        function ode_rhs!(du, u, p_unused, t)
            prob.dynamics!(du, u, p_, t)
        end

        try
            ll = loglik_fn(ode_rhs!, nothing, Float64.(prob.u0), prob.tspan,
                           alg.n_steps, alg.n_deriv, sigma,
                           Float64.(prob.data_values), Float64.(prob.data_times),
                           prob.obs_to_state, obs_var;
                           interrogate=alg.interrogate)

            # Add smoothing penalty for B-splines
            penalty = 0.0
            for (S, off) in zip(smooth_mats, smooth_offsets)
                np = size(S, 1)
                β_k = β_[off+1:off+np]
                penalty += smooth_lambda * dot(β_k, S * β_k)
            end

            return -ll + penalty
        catch e
            return 1e10
        end
    end

    if verbose; println("\nStage 1: Nelder-Mead (derivative-free)..."); end

    # Stage 1: Nelder-Mead for robust initial exploration
    result_nm = Optim.optimize(
        neg_loglik,
        beta,
        Optim.NelderMead(),
        Optim.Options(
            iterations=alg.maxiters,
            show_trace=verbose,
            show_every=max(1, alg.maxiters ÷ 5),
            f_reltol=1e-8,
        )
    )
    beta_nm = Optim.minimizer(result_nm)

    if verbose
        println("  NM loss: $(round(Optim.minimum(result_nm), sigdigits=5))")
        println("\nStage 2: L-BFGS refinement...")
    end

    # Stage 2: L-BFGS refinement from NM solution
    # Use central finite differences for gradient
    function fd_gradient(β_)
        g = zeros(length(β_))
        ε = 1e-5
        for i in eachindex(β_)
            β_p = copy(β_); β_p[i] += ε
            β_m = copy(β_); β_m[i] -= ε
            fp = neg_loglik(β_p)
            fm = neg_loglik(β_m)
            if isfinite(fp) && isfinite(fm)
                g[i] = (fp - fm) / (2ε)
            else
                g[i] = 0.0
            end
        end
        gnorm = norm(g)
        if gnorm > 1e6
            g .*= 1e6 / gnorm
        end
        g
    end

    result = Optim.optimize(
        neg_loglik,
        fd_gradient,
        beta_nm,
        Optim.LBFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(
            iterations=alg.maxiters,
            show_trace=verbose,
            show_every=max(1, alg.maxiters ÷ 10),
            g_tol=1e-6,
            f_reltol=1e-10,
        );
        inplace=false
    )
    beta_opt = Optim.minimizer(result)

    if verbose
        println("  Converged: $(Optim.converged(result))")
        println("  Final -loglik: $(round(Optim.minimum(result), sigdigits=5))")
    end

    # Build solution
    p_opt = build_param_struct(prob, beta_opt)

    # Run probabilistic solver at optimal parameters for solution + uncertainty
    function ode_rhs_opt!(du, u, p_unused, t)
        prob.dynamics!(du, u, p_opt, t)
    end

    μ_smooth, Σ_smooth, times = probsolve(ode_rhs_opt!, nothing, Float64.(prob.u0),
                                           prob.tspan, alg.n_steps, alg.n_deriv, sigma;
                                           interrogate=alg.interrogate)

    # Extract solution at data times and compute data loss
    data_loss = 0.0
    pred = zeros(n_t, n_obs)
    for i in 1:n_t
        idx = searchsortedfirst(times, prob.data_times[i])
        idx = clamp(idx, 1, length(times))
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[i, j] = μ_smooth[idx][sk][1]
            data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
        end
    end

    # Build evaluators
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_opt[offset+1:offset+np]
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
        end
    end

    edf = Float64(n_beta)
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta_opt[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    # Extract solution uncertainty at observation times
    sol_var = zeros(n_t, n_vars)
    for i in 1:n_t
        idx = searchsortedfirst(times, prob.data_times[i])
        idx = clamp(idx, 1, length(times))
        for k in 1:n_vars
            sol_var[i, k] = Σ_smooth[idx][k][1, 1]  # variance of zeroth derivative
        end
    end

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "-loglik=$(round(Optim.minimum(result), sigdigits=5))")
    end

    PSMSolution(
        params,                           # parameters
        -Optim.minimum(result),           # objective (loglik)
        data_loss,                        # data_loss
        edf,                              # edf
        Float64[],                        # smoothing_params (not applicable)
        pred,                             # fitted_values
        Float64.(prob.data_values),       # data_values
        Float64.(prob.data_times),        # data_times
        uf_evals,                         # unknown_functions
        (                                 # convergence
            converged=Optim.converged(result),
            iterations=Optim.iterations(result),
            neg_loglik=Optim.minimum(result),
            method=alg.method,
            obs_var=obs_var,
            sigma=sigma,
            sol_variance=sol_var,
        )
    )
end
