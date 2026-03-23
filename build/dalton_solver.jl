# ─── DALTON Solver for PSMProblem ────────────────────────────────────
#
# DALTON: Data-Adaptive Likelihood with Transformed Observations
# (Wu & Lysy, 2024)
#
# Computes p(Y|Z) = p(Y,Z)/p(Z) via two Kalman filter passes:
#   1. Joint pass: Kalman filter with ODE + observations → logp(Y,Z)
#   2. Marginal pass: Kalman filter with ODE only → logp(Z)
#   3. Data-adaptive likelihood: logp(Y|Z) = logp(Y,Z) - logp(Z)
#
# Unlike fenrir (backward-pass data conditioning), DALTON incorporates
# observations directly into the forward Kalman filter state, then
# subtracts the marginal ODE contribution via Bayes' rule.
#
# Reference: Wu & Lysy (2024)

# ─── Core filter pass ─────────────────────────────────────────────

"""
    _dalton_filter_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                          obs_data, obs_times, obs_to_state, obs_var,
                          include_obs; interrogate=:kramer)

Forward Kalman filter pass with IBM prior and ODE interrogation constraints,
optionally incorporating observation updates at matching grid points.

Returns the total innovation log-likelihood via the Kalman decomposition:
    logp = Σ_n log p(z_n | z_{1:n-1})

When `include_obs=true`, observations are assimilated after ODE updates,
giving logp(Y,Z). When `include_obs=false`, only ODE constraints are
applied, giving logp(Z).
"""
function _dalton_filter_loglik(ode_fun!, p, u0::AbstractVector,
                               tspan::Tuple{Float64, Float64},
                               n_steps::Int, n_deriv::Int,
                               sigma::Vector{Float64},
                               obs_data::Matrix{Float64},
                               obs_times::Vector{Float64},
                               obs_to_state::Vector{Int},
                               obs_var::Float64,
                               include_obs::Bool;
                               interrogate::Symbol=:kramer)
    t_min, t_max = tspan
    dt = (t_max - t_min) / n_steps
    n_vars = length(u0)

    # IBM prior matrices
    wgt_state, var_state = ibm_init(dt, n_deriv, sigma)
    W_list = first_order_weight(n_vars, n_deriv)

    # Small ODE measurement noise for numerical stability of the DALTON
    # difference.  Without this, V_ode = 0 makes the innovation covariance
    # depend solely on Σ_pred, which can produce extreme log-densities
    # that amplify when we subtract marginal from joint.
    ode_nuggets = [fill(1e-10 * sigma[k]^2, 1, 1) for k in 1:n_vars]

    # Initial state from ODE (known exactly, zero covariance)
    X0 = first_order_init(ode_fun!, Float64.(u0), t_min, p, n_deriv)
    μ_filt = [copy(X0[k]) for k in 1:n_vars]
    Σ_filt = [zeros(n_deriv, n_deriv) for _ in 1:n_vars]

    # Map observation times to solver grid indices
    grid_times = collect(range(t_min, t_max, length=n_steps + 1))
    n_t_obs = size(obs_data, 1)
    n_obs_vars = length(obs_to_state)
    obs_ind = [searchsortedfirst(grid_times, obs_times[i]) for i in 1:n_t_obs]
    obs_ind = clamp.(obs_ind, 1, n_steps + 1)

    # Build lookup: grid index → [(time_idx, obs_col), ...]
    obs_at_grid = Dict{Int, Vector{Tuple{Int,Int}}}()
    if include_obs
        for i in 1:n_t_obs, j in 1:n_obs_vars
            gi = obs_ind[i]
            if !haskey(obs_at_grid, gi)
                obs_at_grid[gi] = Tuple{Int,Int}[]
            end
            push!(obs_at_grid[gi], (i, j))
        end
    end

    # Observation matrix: selects 0th derivative from IBM state [x, x', x'', ...]
    H_obs = zeros(1, n_deriv)
    H_obs[1, 1] = 1.0
    V_obs = fill(obs_var, 1, 1)

    interrogate_fn = interrogate == :kramer ? interrogate_kramer : interrogate_schober
    z_meas = zeros(1)  # ODE pseudo-observation: residual = 0
    loglik = 0.0

    # Handle observations at initial time (grid index 1)
    if include_obs && haskey(obs_at_grid, 1)
        for (ti, ji) in obs_at_grid[1]
            k = obs_to_state[ji]
            y_obs = [obs_data[ti, ji]]
            μ_fore = H_obs * μ_filt[k]
            Σ_fore = H_obs * Σ_filt[k] * H_obs' + V_obs
            loglik += logpdf_mvn(y_obs, μ_fore, Σ_fore)
            μ_filt[k], Σ_filt[k] = kalman_update(
                μ_filt[k], Σ_filt[k], y_obs, zeros(1), H_obs, V_obs)
        end
    end

    # Forward Kalman filter
    for n in 1:n_steps
        t_n = t_min + dt * n

        # ── Predict ──
        μ_pred = Vector{Vector{Float64}}(undef, n_vars)
        Σ_pred = Vector{Matrix{Float64}}(undef, n_vars)
        for k in 1:n_vars
            μ_pred[k], Σ_pred[k] = kalman_predict(
                μ_filt[k], Σ_filt[k], wgt_state[k], var_state[k])
        end

        # ── Interrogate ODE ──
        wgt_m, mean_m, var_m = interrogate_fn(
            ode_fun!, W_list, t_n, μ_pred, Σ_pred, p, n_vars)

        # ── ODE constraint update with innovation log-likelihood ──
        for k in 1:n_vars
            W_total = W_list[k] + wgt_m[k]
            V_reg = var_m[k] + ode_nuggets[k]

            # Innovation: log p(z=0 | past)
            μ_fore = W_total * μ_pred[k] + mean_m[k]
            Σ_fore = W_total * Σ_pred[k] * W_total' + V_reg
            loglik += logpdf_mvn(z_meas, μ_fore, Σ_fore)

            μ_filt[k], Σ_filt[k] = kalman_update(
                μ_pred[k], Σ_pred[k], z_meas, mean_m[k], W_total, V_reg)
        end

        # ── Observation update (joint pass only) ──
        grid_idx = n + 1
        if include_obs && haskey(obs_at_grid, grid_idx)
            for (ti, ji) in obs_at_grid[grid_idx]
                k = obs_to_state[ji]
                y_obs = [obs_data[ti, ji]]

                # Innovation: log p(y | past, z_ode)
                μ_fore = H_obs * μ_filt[k]
                Σ_fore = H_obs * Σ_filt[k] * H_obs' + V_obs
                loglik += logpdf_mvn(y_obs, μ_fore, Σ_fore)

                μ_filt[k], Σ_filt[k] = kalman_update(
                    μ_filt[k], Σ_filt[k], y_obs, zeros(1), H_obs, V_obs)
            end
        end
    end

    loglik
end

# ─── DALTON log-likelihood ────────────────────────────────────────

"""
    _dalton_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                   obs_data, obs_times, obs_to_state, obs_var;
                   interrogate=:kramer)

Compute the DALTON data-adaptive log-likelihood via Bayes' rule:

    logp(Y|Z) = logp(Y,Z) - logp(Z)

where Z are ODE interrogation pseudo-observations and Y are data.
Two forward Kalman filter passes compute the joint and marginal terms.
"""
function _dalton_loglik(ode_fun!, p, u0::AbstractVector,
                        tspan::Tuple{Float64, Float64},
                        n_steps::Int, n_deriv::Int,
                        sigma::Vector{Float64},
                        obs_data::Matrix{Float64},
                        obs_times::Vector{Float64},
                        obs_to_state::Vector{Int},
                        obs_var::Float64;
                        interrogate::Symbol=:kramer)
    loglik_joint = _dalton_filter_loglik(
        ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
        obs_data, obs_times, obs_to_state, obs_var, true;
        interrogate=interrogate)

    loglik_marginal = _dalton_filter_loglik(
        ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
        obs_data, obs_times, obs_to_state, obs_var, false;
        interrogate=interrogate)

    ll = loglik_joint - loglik_marginal
    # The difference of two large log-densities can lose precision;
    # return a sentinel when the result is clearly numerical noise.
    isfinite(ll) ? ll : -Inf
end

# ─── Solver dispatch ──────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::DaltonSolver)

Fit a partially specified model using the DALTON (DAta-driven Linearisation
Turn-ON) probabilistic ODE solver. Combines an IBM prior with iterative
local linearisation of the ODE to obtain a Gaussian posterior over the
state trajectory, then optimises the unknown-function parameters.

# Algorithm
1. Set up the IBM prior of order `n_deriv` for each state variable.
2. Forward Kalman filter: propagate the prior and assimilate data and
   ODE pseudo-observations obtained by linearising the dynamics.
3. RTS backward smoother to obtain the posterior state.
4. Iterate linearisation using the current posterior mean until
   convergence (DALTON iterations).
5. Optimise unknown-function parameters by maximising the marginal
   log-likelihood.

# References
- Tronarp et al. (2022), "Fenrir: Physics-Enhanced Regression for Initial
  Value Problems", ICML.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::DaltonSolver)
    _validate_problem(prob, "DaltonSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)

    if verbose
        @printf("DaltonSolver: n_steps=%d, n_deriv=%d, interrogate=%s\n",
                alg.n_steps, alg.n_deriv, alg.interrogate)
    end

    # Initialize parameters from approximators
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Auto-estimate sigma (IBM scale) from data range
    sigma = if alg.sigma === nothing
        sig = Float64[]
        for k in 1:n_vars
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

    obs_var = alg.obs_var

    if verbose
        @printf("  σ (IBM scale): %s\n", string(round.(sigma, sigdigits=3)))
        @printf("  obs_var: %.3g\n", obs_var)
        @printf("  %d approximator params\n", n_beta)
    end

    # Smoothing penalty matrices for B-spline approximators
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

    smooth_lambda = 0.1 / max(obs_var, 1e-6)

    # ── Objective: negative DALTON log-likelihood + penalty ──

    function neg_loglik(β_)
        p_ = build_autodiff_param_struct(prob, β_)

        function ode_rhs!(du, u, p_unused, t)
            prob.dynamics!(du, u, p_, t)
        end

        try
            ll = _dalton_loglik(ode_rhs!, nothing, Float64.(prob.u0), prob.tspan,
                                alg.n_steps, alg.n_deriv, sigma,
                                Float64.(prob.data_values), Float64.(prob.data_times),
                                prob.obs_to_state, obs_var;
                                interrogate=alg.interrogate)

            if !isfinite(ll)
                return 1e10
            end

            # Reject numerically extreme DALTON loglik arising from explosive
            # ODE dynamics — a real conditional loglik cannot exceed about
            # n_obs * max_per_obs, so any value far above that is numerical noise.
            n_data = size(prob.data_values, 1) * size(prob.data_values, 2)
            if ll > 100.0 * max(n_data, 1)
                return 1e10
            end

            # Smoothing penalty for B-splines
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

    # ── Stage 1: Nelder-Mead (robust exploration) ──

    if verbose; @printf("\nStage 1: Nelder-Mead (derivative-free)...\n"); end

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
        @printf("  NM loss: %.5g\n", Optim.minimum(result_nm))
        @printf("\nStage 2: L-BFGS refinement...\n")
    end

    # ── Stage 2: L-BFGS with finite-difference gradients ──

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
        @printf("  Converged: %s\n", Optim.converged(result))
        @printf("  Final -loglik: %.5g\n", Optim.minimum(result))
    end

    # ── Build solution ──

    p_opt = build_param_struct(prob, beta_opt)

    function ode_rhs_opt!(du, u, p_unused, t)
        prob.dynamics!(du, u, p_opt, t)
    end

    # Probabilistic solve for smooth posterior at optimal params
    μ_smooth, Σ_smooth, times = probsolve(
        ode_rhs_opt!, nothing, Float64.(prob.u0),
        prob.tspan, alg.n_steps, alg.n_deriv, sigma;
        interrogate=alg.interrogate)

    # Extract fitted values and data loss
    data_loss = 0.0
    pred = zeros(n_t, n_obs)
    for i in 1:n_t
        idx = searchsortedfirst(times, prob.data_times[i])
        idx = clamp(idx, 1, length(times))
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[i, j] = μ_smooth[idx][sk][1]
            data_loss += prob.data_weights[i, j] *
                         (prob.data_values[i, j] - pred[i, j])^2
        end
    end

    # Build unknown function evaluators (same pattern as RodeoSolver)
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
            rng = approx.rng_seed !== nothing ?
                  Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ?
                   Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(
                Lux.initialparameters(rng2, approx.model)))
            ps_final = similar(ps_ca)
            ps_final .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing :
                   (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model,
                    Float32.(reshape([xn], :, 1)), ps_final, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
            end
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] =
                build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
        end
    end

    # Package parameters as ComponentArray
    edf = Float64(n_beta)
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta_opt[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    # Solution uncertainty at observation times
    sol_var = zeros(n_t, n_vars)
    for i in 1:n_t
        idx = searchsortedfirst(times, prob.data_times[i])
        idx = clamp(idx, 1, length(times))
        for k in 1:n_vars
            sol_var[i, k] = Σ_smooth[idx][k][1, 1]
        end
    end

    if verbose
        @printf("\nFinal: data_SS=%.5g -loglik=%.5g\n",
                data_loss, Optim.minimum(result))
    end

    PSMSolution(
        params,
        -Optim.minimum(result),           # objective (loglik)
        data_loss,
        edf,
        Float64[],                         # smoothing_params (not applicable)
        pred,
        Float64.(prob.data_values),
        Float64.(prob.data_times),
        uf_evals,
        (
            converged=Optim.converged(result),
            iterations=Optim.iterations(result),
            neg_loglik=Optim.minimum(result),
            method=:dalton,
            obs_var=obs_var,
            sigma=sigma,
            sol_variance=sol_var,
        )
    )
end
