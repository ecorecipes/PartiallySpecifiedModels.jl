# ─── BNG solver (Bayesian Neural Gradient matching) ──────────────
#
# Bonnaffé & Coulson (2023), Methods Ecol Evol: ensemble Bayesian
# gradient matching.
#   Step 1: Smooth observed data with a penalized (GCV) spline.
#   Step 2: Build K_o observation ensembles by residual bootstrap of the
#           smoother, and for each fit the unknown-function parameters
#           K_p times from perturbed initialisations by minimizing the
#           variance-marginalized log-posterior of the gradient match.
#   Step 3: The K_o × K_p fits form the ensemble: point estimates are
#           ensemble means, uncertainty is the ensemble spread.
#
# Uses Adam with ForwardDiff gradients (same pattern as AdamSolver).
# Key advantage: no ODE integration needed → fast, robust initialization.

using LinearAlgebra: dot, norm

# ─── Main BNG solver ─────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::BNGSolver)

Fit a partially specified model by ensemble Bayesian gradient matching
(Bonnaffé & Coulson 2023).

# Algorithm
1. Smooth each observed series with a penalized (GCV) spline; keep the
   fitted values and residuals.
2. Build `k_obs` observation ensembles: the first is the original data,
   the rest add column-wise resampled residuals to the smoother fit
   (residual bootstrap). Re-smooth each to get state and derivative
   targets.
3. For each ensemble, run `k_proc` fits from perturbed initialisations
   (fresh random weights for neural approximators, jittered coefficients
   otherwise), minimizing the variance-marginalized log-posterior

       L(β) = (n/2)·log(1 + SSR/2) + (|β|/2)·log(1 + Σ(β/prior_sd)²/2)
              + λ_smooth · Σ βₖᵀ S βₖ,

   where SSR is the gradient-matching sum of squares. Marginalizing the
   observation and prior variances (rather than fixing them) is the
   "Bayesian regularisation" of the cited paper: the two log terms
   self-balance, so no noise variance needs to be supplied.
4. The `k_obs × k_proc` fits form the posterior ensemble. Reported
   unknown functions are ensemble means; `sol.convergence.ensemble_std`
   maps each function name to a pointwise standard-deviation function,
   and `sol.params`/`sol.objective` come from the best (lowest-loss)
   member.

# Returns
`PSMSolution`. Fitted values re-simulate the ODE/map with the best
member's parameters.
"""
function SciMLBase.solve(prob::PSMProblem, alg::BNGSolver)
    _validate_problem(prob, "BNGSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    # ── Step 1: base smooth of the original data ─────────────────
    if verbose; println("BNGSolver Step 1: smoothing data (GCV splines)..."); end

    observed_states = Set{Int}(prob.obs_to_state)
    base_fit = zeros(n_times, n_obs)             # smoother fit per data column
    resid = zeros(n_times, n_obs)                # residuals for the bootstrap
    for j in 1:n_obs
        val, _ = _smoothing_spline(times, Float64.(prob.data_values[:, j]))
        for i in 1:n_times
            base_fit[i, j] = val(times[i])
        end
        resid[:, j] = prob.data_values[:, j] .- base_fit[:, j]
    end

    # Smooth a data matrix into (states, derivative targets)
    function smooth_targets(data_matrix)
        y_smooth = zeros(n_times, n_vars)
        dydt = zeros(n_times, n_vars)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            val, der = _smoothing_spline(times, data_matrix[:, j])
            for i in 1:n_times
                y_smooth[i, sk] = val(times[i])
                if !prob.discrete
                    dydt[i, sk] = der(times[i])
                end
            end
        end
        if prob.discrete
            # Discrete maps: target the smoothed next state
            for k in 1:n_vars
                for i in 1:(n_times - 1)
                    dydt[i, k] = y_smooth[i + 1, k]
                end
                dydt[n_times, k] = y_smooth[n_times, k]
            end
        end
        # Unobserved states: constant at the initial condition
        for k in 1:n_vars
            if k ∉ observed_states
                u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                               prob.u0[k])
                y_smooth[:, k] .= u0_k
                dydt[:, k] .= 0.0
            end
        end
        y_smooth, dydt
    end

    # ── Parameter initialisation ─────────────────────────────────
    mlp_specs = Dict{Symbol, MLPSpec}()
    for approx in prob.approximators
        approx isa NeuralApproximator &&
            (mlp_specs[approx.name] = mlp_spec_from_lux(approx.model))
    end

    # kp = 1 uses the default initialisation (a seeded neural approximator
    # keeps its own seed; unseeded ones draw fresh weights); kp > 1 perturbs
    # it — fresh random weights for neural approximators, jittered
    # coefficients otherwise — so the process ensemble explores distinct
    # basins.
    function init_beta(kp::Int)
        beta = Float64[]
        for approx in prob.approximators
            if approx isa NeuralApproximator
                r = kp == 1 && approx.rng_seed !== nothing ?
                    Random.Xoshiro(approx.rng_seed) : rng
                append!(beta, init_mlp_params(mlp_specs[approx.name], r))
            else
                b0 = Float64.(initial_params(approx))
                kp > 1 && (b0 .+= 0.2 .* (abs.(b0) .+ 1.0) .* randn(rng, length(b0)))
                append!(beta, b0)
            end
        end
        beta
    end
    n_beta = length(init_beta(1))

    n_match = prob.discrete ? n_times - 1 : n_times
    n_resid = n_match * length(observed_states)
    lambda_smooth = alg.lambda_smooth
    prior_sd = alg.prior_sd
    K_total = alg.k_obs * alg.k_proc

    if verbose
        println("BNGSolver Step 2: $(alg.k_obs) obs-ensembles × " *
                "$(alg.k_proc) process fits, $n_beta params, " *
                "$n_match match points; maxiters=$(alg.maxiters), " *
                "prior_sd=$prior_sd")
    end

    # Variance-marginalized gradient-matching log-posterior
    function bng_loss(β_eval, y_smooth, dydt)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)
        ssr = zero(T_el)
        for i in 1:n_match
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                # Match only observed states (unobserved targets are
                # fabricated constants — see two_stage_solver.jl).
                k in observed_states || continue
                ssr += (dydt[i, k] - du[k])^2
            end
        end
        loss_val = (n_resid / 2) * log1p(ssr / 2) +
                   (n_beta / 2) * log1p(sum(abs2, β_eval ./ prior_sd) / 2)
        # Smoothing penalties from approximators
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            params_k = β_eval[offset+1:offset+np]
            offset += np
            if approx isa COMONetApproximator
                loss_val += approx.penalty_weight * sum(abs2, params_k)
            else
                S = penalty_matrix(approx)
                S !== nothing &&
                    (loss_val += lambda_smooth * dot(params_k, S * params_k))
            end
        end
        loss_val
    end

    # One Adam fit against fixed targets
    function fit_member(beta0, y_smooth, dydt)
        beta = copy(beta0)
        β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
        m_adam = zeros(n_beta); v_adam = zeros(n_beta)
        best_beta = copy(beta)
        best_loss = Inf
        loss_window = fill(Inf, 30)
        loss_fn = β -> bng_loss(β, y_smooth, dydt)
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        iters_used = alg.maxiters
        for iter in 1:alg.maxiters
            ForwardDiff.gradient!(result, loss_fn, beta)
            loss_val = DiffResults.value(result)
            grad = DiffResults.gradient(result)
            if loss_val < best_loss
                best_loss = loss_val
                best_beta .= beta
            end
            loss_window[mod1(iter, 30)] = loss_val
            lr_t = alg.lr * 0.5 * (1 + cos(π * iter / alg.maxiters))
            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
            m_hat = m_adam ./ (1 - β1_adam^iter)
            v_hat = v_adam ./ (1 - β2_adam^iter)
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
            if iter > 60
                rmin, rmax = extrema(loss_window)
                if (rmax - rmin) / max(abs(rmin), 1.0) < 1e-6
                    iters_used = iter
                    break
                end
            end
        end
        best_beta, best_loss, iters_used
    end

    # ── Step 2/3: the K_o × K_p ensemble ─────────────────────────
    member_betas = Vector{Vector{Float64}}()
    member_losses = Float64[]
    total_iters = 0
    for ko in 1:alg.k_obs
        data_ko = if ko == 1
            Float64.(prob.data_values)
        else
            # Residual bootstrap: smoother fit + resampled residual rows
            # (rows resampled jointly across columns, preserving any
            # cross-series residual correlation)
            idx_boot = rand(rng, 1:n_times, n_times)
            base_fit .+ resid[idx_boot, :]
        end
        y_smooth, dydt = smooth_targets(data_ko)
        for kp in 1:alg.k_proc
            beta_fit, loss_fit, it = fit_member(init_beta(kp), y_smooth, dydt)
            push!(member_betas, beta_fit)
            push!(member_losses, loss_fit)
            total_iters += it
            if verbose
                println("  member (obs $ko, proc $kp): " *
                        "loss=$(round(loss_fit, sigdigits=5)) [$it iters]")
            end
        end
    end
    best_idx = argmin(member_losses)
    beta = copy(member_betas[best_idx])
    best_loss = member_losses[best_idx]

    if verbose
        println("  Best member: $best_idx/$K_total, " *
                "loss=$(round(best_loss, sigdigits=5))")
    end

    # ── Build solution ───────────────────────────────────────────

    # Simulate with the best member's parameters for trajectory predictions
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
            # Fallback: use the base smoother fit as predictions
            if verbose; println("  ODE simulation failed, using smoothed values"); end
            for j in 1:n_obs
                pred[:, j] .= base_fit[:, j]
            end
        end
    end

    # Data loss against original observations
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Per-member evaluators for one approximator's parameter block
    function build_eval(approx, params_k)
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            spec = mlp_specs[approx.name]
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            let pk = copy(params_k), s = spec, lo_ = lo, span_ = span
                x -> begin
                    xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                        (Float64(x isa AbstractArray ? x[1] : x) - lo_) / span_
                    else
                        Float64(x isa AbstractArray ? x[1] : x)
                    end
                    mlp_evaluate(s, pk, xn)
                end
            end
        elseif approx isa GPApproximator
            build_gp_evaluator(approx, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            build_comonet_evaluator(approx, params_k)
        elseif approx isa SPDEApproximator
            build_spde_evaluator(approx.mesh_points, params_k)
        elseif approx isa ShapeConstrainedSPDEApproximator
            build_constrained_spde_evaluator(approx, params_k)
        end
    end

    # Posterior weights. Losses are only comparable WITHIN an observation
    # ensemble (each bootstrap has its own data), so exp(−Δloss) weights are
    # formed over the k_proc restarts of each obs-ensemble — zeroing
    # divergent restarts — and the obs-ensembles then count equally.
    # Non-finite losses are masked out entirely.
    w_members = zeros(K_total)
    for ko in 1:alg.k_obs
        idx = ((ko - 1) * alg.k_proc + 1):(ko * alg.k_proc)
        ls = member_losses[idx]
        ok = isfinite.(ls)
        any(ok) || continue
        lmin = minimum(ls[ok])
        w = [ok[i] ? exp(-(ls[i] - lmin)) : 0.0 for i in eachindex(ls)]
        w_members[idx] = w ./ sum(w)
    end
    sum(w_members) > 0 ||
        error("BNGSolver: every ensemble member diverged (non-finite loss)")
    w_members ./= sum(w_members)

    # Weighted ensemble-mean unknown functions + pointwise-sd companions.
    # Averaging in FUNCTION space (not parameter space) is essential for
    # neural members, whose weights are not comparable across restarts.
    uf_evals = Dict{Symbol, Any}()
    ensemble_std = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        member_evals = [build_eval(approx, mb[offset+1:offset+np])
                        for mb in member_betas]
        offset += np
        let evs = member_evals, w = w_members
            uf_evals[approx.name] =
                x -> sum(w[i] * evs[i](x) for i in eachindex(evs))
            ensemble_std[approx.name] = x -> begin
                vals = [ev(x) for ev in evs]
                μ = sum(w .* vals)
                sqrt(max(sum(w .* (vals .- μ).^2), 0.0))
            end
        end
    end

    # Parameters reported from the best member (parameter-space ensemble
    # means are meaningless for neural approximators)
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
                "objective=$(round(best_loss, sigdigits=5)) " *
                "ensemble=$(K_total) members")
    end

    PSMSolution(params, best_loss, data_loss, edf, Float64[],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=total_iters, method=:bng,
                 n_ensemble=K_total, member_losses=member_losses,
                 member_weights=w_members, ensemble_std=ensemble_std))
end
