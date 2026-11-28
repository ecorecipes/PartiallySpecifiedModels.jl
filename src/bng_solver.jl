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
member's parameters. `sol.convergence` carries the honest-convergence keys
`(converged, iterations, reason)` plus per-member `member_converged` — see
the `BNGSolver` docstring for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::BNGSolver)
    # DELIBERATE: BNGSolver keeps failing loudly on DDEs rather than being
    # "repaired" at its final-prediction block (`:333`, an ODEProblem built
    # from the 4-argument signature). Its FIT is gradient matching — `bng_loss`
    # evaluates the right-hand side on smoothed states with no history
    # mechanism at all — so fixing only the reporting site would turn a hard
    # error into a SILENT wrong answer, which is strictly worse. Real support
    # needs a history-from-the-smoother rework; that is a feature, not a fix.
    _validate_problem(prob, "BNGSolver"; reject_delays=true)
    _require_gaussian_likelihood(prob, "BNGSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    u0_init = _resolve_u0_init(prob)
    n_vars = length(u0_init)
    n_obs = size(prob.data_values, 2)
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    # ── Step 1: base smooth of the original data ─────────────────
    if verbose; println("BNGSolver Step 1: smoothing data (GCV splines)..."); end

    observed_states = Set{Int}(prob.obs_to_state)
    base_fit = zeros(n_times, n_obs)             # smoother fit per data column
    resid = zeros(n_times, n_obs)                # residuals for the bootstrap
    for j in 1:n_obs
        val, _ = _smoothing_spline_masked(times,
                                          Float64.(prob.data_values[:, j]),
                                          @view(prob.data_weights[:, j]))
        for i in 1:n_times
            base_fit[i, j] = val(times[i])
        end
        # Residual bootstrap pool: only usable cells yield real residuals.
        # A masked cell's "residual" is NaN (or a fabricated number at
        # weight 0) and would be resampled into random rows of every
        # replicate, spreading the contamination across the ensemble.
        for i in 1:n_times
            resid[i, j] = usable_cell(prob, i, j) ?
                          prob.data_values[i, j] - base_fit[i, j] : 0.0
        end
    end

    # Smooth a data matrix into (states, derivative targets)
    function smooth_targets(data_matrix)
        y_smooth = zeros(n_times, n_vars)
        dydt = zeros(n_times, n_vars)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            # `data_matrix` is the raw data on the first pass and a
            # bootstrap resample of base_fit + resid afterwards; the
            # former can hold NaN, so mask by the original weights.
            val, der = _smoothing_spline_masked(times,
                                                Float64.(data_matrix[:, j]),
                                                @view(prob.data_weights[:, j]))
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
                y_smooth[:, k] .= u0_init[k]
                dydt[:, k] .= 0.0
            end
        end
        y_smooth, dydt
    end

    # ── Parameter initialisation ─────────────────────────────────
    # kp = 1 uses the default initialisation (a seeded neural approximator
    # keeps its own seed; unseeded ones draw fresh weights); kp > 1 perturbs
    # it — fresh random weights for neural approximators, jittered
    # coefficients otherwise — so the process ensemble explores distinct
    # basins.
    function init_beta(kp::Int)
        beta_init = Float64[]
        for approx in prob.approximators
            if approx isa NeuralApproximator
                r = kp == 1 && approx.rng_seed !== nothing ?
                    Random.Xoshiro(approx.rng_seed) : rng
                append!(beta_init, neural_init_params(approx, r))
            else
                b0 = Float64.(initial_params(approx))
                kp > 1 && (b0 .+= 0.2 .* (abs.(b0) .+ 1.0) .* randn(rng, length(b0)))
                append!(beta_init, b0)
            end
        end
        beta_init
    end
    n_beta = length(init_beta(1))

    n_match = prob.discrete ? n_times - 1 : n_times
    # Per-(time, state) usability, mirroring TwoStageSolver: a match point
    # counts if ANY observation column mapping to that state is usable
    # there. All-true for complete data.
    match_usable = falses(n_times, n_vars)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:n_times
            usable_cell(prob, i, j) && (match_usable[i, sk] = true)
        end
    end
    # n_resid is the effective sample size in the variance-marginalized
    # log-posterior below, `(n_resid/2)*log1p(ssr/2)`. Counting masked
    # cells over-weights the data term against the parameter prior and
    # breaks the self-balancing property this solver advertises. Equals
    # n_match * |observed_states| for complete data.
    n_resid = count(match_usable[i, k] for i in 1:n_match, k in 1:n_vars)
    n_resid == 0 && error("BNGSolver: every observation is masked; " *
        "there is nothing to fit.")
    lambda_smooth = alg.lambda_smooth
    prior_sd = alg.prior_sd
    K_total = alg.k_obs * alg.k_proc

    if verbose
        println("BNGSolver Step 2: $(alg.k_obs) obs-ensembles × " *
                "$(alg.k_proc) process fits, $n_beta params, " *
                "$n_match match points; maxiters=$(alg.maxiters), " *
                "prior_sd=$prior_sd")
    end

    # Precompute smoothing penalties once before the ensemble Adam fits.
    bng_mat_pens = Tuple{Int,Int,Matrix{Float64}}[]
    bng_l2_pens = Tuple{Int,Int,Float64}[]
    let off = 0
        for approx in prob.approximators
            np = nparams(approx)
            if approx isa COMONetApproximator
                approx.penalty_weight > 0.0 && push!(bng_l2_pens, (off, np, approx.penalty_weight))
            else
                S = penalty_matrix(approx)
                S !== nothing && push!(bng_mat_pens, (off, np, S))
            end
            off += np
        end
    end

    # Standardize observed state & time scales (Bonnaffé & Coulson 2023) so
    # the variance-marginalized `log1p(ssr / 2)` term is scale-invariant and
    # multi-scale states contribute equitably.
    time_scale = prob.discrete ? 1.0 : max(times[end] - times[1], 1e-6)
    deriv_scale = ones(n_vars)
    for k in observed_states
        vals_k = Float64[prob.data_values[i, j] for j in 1:n_obs if prob.obs_to_state[j] == k
                         for i in 1:n_times if usable_cell(prob, i, j)]
        sd_k = length(vals_k) > 1 ? std(vals_k) : 1.0
        state_scale_k = (isfinite(sd_k) && sd_k > 1e-8) ? sd_k : 1.0
        deriv_scale[k] = state_scale_k / time_scale
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
            catch e
                _is_program_error(e) && rethrow()
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                k in observed_states || continue
                match_usable[i, k] || continue
                ssr += ((dydt[i, k] - du[k]) / deriv_scale[k])^2
            end
        end
        loss_val = (n_resid / 2) * log1p(ssr / 2) +
                   (n_beta / 2) * log1p(sum(abs2, β_eval ./ prior_sd) / 2)
        for (off, np, w_l2) in bng_l2_pens
            pk = @view β_eval[off+1:off+np]
            loss_val += w_l2 * sum(abs2, pk)
        end
        for (off, np, S) in bng_mat_pens
            pk = @view β_eval[off+1:off+np]
            loss_val += lambda_smooth * dot(pk, S * pk)
        end
        loss_val
    end

    # One Adam fit against fixed targets (explicit `local` declarations prevent
    # Julia from boxing outer-scope names like `best_loss` across threads).
    function fit_member(beta0, y_smooth, dydt)
        local beta_cur = copy(beta0)
        local β1_adam = 0.9
        local β2_adam = 0.999
        local eps_adam = 1e-8
        local m_adam = zeros(n_beta)
        local v_adam = zeros(n_beta)
        local best_beta_m = copy(beta_cur)
        local best_loss_m = Inf
        local loss_window = fill(Inf, 30)
        local loss_fn = β -> bng_loss(β, y_smooth, dydt)
        local result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        local iters_used = alg.maxiters
        local plateaued = false
        for iter in 1:alg.maxiters
            ForwardDiff.gradient!(result, loss_fn, beta_cur)
            local loss_val = DiffResults.value(result)
            local grad = DiffResults.gradient(result)
            if loss_val < best_loss_m
                best_loss_m = loss_val
                best_beta_m .= beta_cur
            end
            loss_window[mod1(iter, 30)] = loss_val
            local lr_t = alg.lr * 0.5 * (1 + cos(π * iter / alg.maxiters))
            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
            local m_hat = m_adam ./ (1 - β1_adam^iter)
            local v_hat = v_adam ./ (1 - β2_adam^iter)
            beta_cur .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
            if iter > 60 && best_loss_m < 1e9 && lr_t > 0.05 * alg.lr
                rmin, rmax = extrema(loss_window)
                if (rmax - rmin) / max(abs(rmin), 1.0) < 1e-6
                    iters_used = iter
                    plateaued = true
                    break
                end
            end
        end
        best_beta_m, best_loss_m, iters_used, plateaued
    end

    # ── Step 2/3: the K_o × K_p ensemble ─────────────────────────
    # Pre-generate all bootstrap targets & initializations deterministically so
    # `parallel=true` yields bit-identical ensembles regardless of thread scheduling.
    ys_list = Vector{Matrix{Float64}}(undef, alg.k_obs)
    dy_list = Vector{Matrix{Float64}}(undef, alg.k_obs)
    for ko in 1:alg.k_obs
        data_ko = if ko == 1
            Float64.(prob.data_values)
        else
            idx_boot = rand(rng, 1:n_times, n_times)
            base_fit .+ resid[idx_boot, :]
        end
        ys_list[ko], dy_list[ko] = smooth_targets(data_ko)
    end
    y_smooth_base = ys_list[1]
    dydt_base = dy_list[1]

    inits = Vector{Vector{Float64}}(undef, K_total)
    ko_of = Vector{Int}(undef, K_total)
    kp_of = Vector{Int}(undef, K_total)
    let idx = 0
        for ko in 1:alg.k_obs, kp in 1:alg.k_proc
            idx += 1
            ko_of[idx] = ko; kp_of[idx] = kp
            inits[idx] = init_beta(kp)
        end
    end

    member_betas = Vector{Vector{Float64}}(undef, K_total)
    member_losses = Vector{Float64}(undef, K_total)
    member_plateaued = Vector{Bool}(undef, K_total)
    member_iters = Vector{Int}(undef, K_total)

    if alg.parallel && Threads.nthreads() > 1
        Threads.@threads for idx in 1:K_total
            ko = ko_of[idx]
            bf, lf, it, plat = fit_member(inits[idx], ys_list[ko], dy_list[ko])
            member_betas[idx] = bf
            member_losses[idx] = lf
            member_plateaued[idx] = plat
            member_iters[idx] = it
        end
    else
        for idx in 1:K_total
            ko = ko_of[idx]; kp = kp_of[idx]
            bf, lf, it, plat = fit_member(inits[idx], ys_list[ko], dy_list[ko])
            member_betas[idx] = bf
            member_losses[idx] = lf
            member_plateaued[idx] = plat
            member_iters[idx] = it
            if verbose
                println("  member (obs $ko, proc $kp): " *
                        "loss=$(round(lf, sigdigits=5)) [$it iters]")
            end
        end
    end
    total_iters = sum(member_iters)
    # Evaluate all members on the SAME unbootstrapped base smooth rather than
    # comparing uncalibrated losses across different bootstrap resamples.
    base_losses = [bng_loss(b, y_smooth_base, dydt_base) for b in member_betas]
    best_idx = argmin(base_losses)
    beta = copy(member_betas[best_idx])
    best_loss = base_losses[best_idx]

    if verbose
        println("  Best member: $best_idx/$K_total, " *
                "loss=$(round(best_loss, sigdigits=5))")
    end

    # Sentinel accounting at the best member's parameters on the
    # unbootstrapped smoothed trajectory — see `_dynamics_failure_verdict`
    # (solver.jl). `bng_loss` substitutes a 1e6 sentinel wherever the dynamics
    # raise, so without this a model that never evaluates is reported as an
    # ordinary ensemble fit. Taken BEFORE the prediction block below, whose
    # ODE solve re-enters the same dynamics and would propagate the user's
    # exception verbatim (loud, but uninformative) before this ran.
    bng_converged = member_plateaued[best_idx]
    bng_reason = bng_converged ? :plateau : :maxiters
    n_dyn_fail = _count_dynamics_failures(prob, times, y_smooth_base, beta)
    bng_converged, bng_reason = _dynamics_failure_verdict(
        "BNGSolver", n_dyn_fail, n_times, bng_converged, bng_reason)

    # ── Build solution ───────────────────────────────────────────

    # Simulate with the best member's parameters for trajectory predictions
    # Set by the continuous fallback below when the ODE solve fails; the
    # discrete path builds `pred` by stepping and cannot fail this way.
    sim_failed = false
    if prob.discrete
        p_sim = build_param_struct(prob, beta)
        u_sim = Float64.(prob.u0 isa Function ? prob.u0(p_sim) : prob.u0)
        u_next_sim = similar(u_sim)
        sim_states = zeros(n_times, n_vars)
        sim_states[1, :] .= u_sim
        for step in 1:(n_times - 1)
            try
                prob.dynamics!(u_next_sim, u_sim, p_sim, times[step])
            catch e
                _is_program_error(e) && rethrow()
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
        bng_kw = Dict{Symbol,Any}(:saveat => prob.data_times, :abstol => 1e-7,
                                  :reltol => 1e-7, :maxiters => 10000,
                                  :verbose => _quiet_solve())
        for (k, v) in _normalise_ode_kwargs(prob.ode_kwargs)
            bng_kw[k] = v
        end
        sol_ode = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver; bng_kw...)

        pred = zeros(n_times, n_obs)
        if sol_ode.retcode == SciMLBase.ReturnCode.Success ||
           sol_ode.retcode == SciMLBase.ReturnCode.Default
            for j in 1:n_obs
                sk = prob.obs_to_state[j]
                for i in 1:min(n_times, length(sol_ode.t))
                    pred[i, j] = sol_ode[sk, i]
                end
            end
        else
            # Fifth member of the family B7 fixed in four solvers: on a
            # non-Success retcode the reported `fitted_values` silently
            # became the base smoother, so `data_loss` measured the SMOOTHER
            # rather than the model — and it was announced only under
            # `verbose`, i.e. never, by default. Same warning and same
            # `convergence.simulation_failed` flag as the rest of the family.
            sim_failed = true
            @warn "BNGSolver: integrating the fitted dynamics from u0 " *
                  "failed (retcode $(sol_ode.retcode)). Reporting the base " *
                  "data smoother as `fitted_values`; `data_loss` therefore " *
                  "measures the SMOOTHER, not the model fit. See " *
                  "`convergence.simulation_failed`."
            for j in 1:n_obs
                pred[:, j] .= base_fit[:, j]
            end
        end
    end

    # Data loss against original observations
    data_loss = weighted_data_loss(prob, pred)

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
        member_evals = [build_evaluator(approx, mb[offset+1:offset+np])
                        for mb in member_betas]
        offset += np
        let evs = member_evals, w = w_members
            # Varargs so multi-argument evaluators (TensorBSplineApproximator's
            # f(x, y)) pass through; single-argument behavior is unchanged.
            uf_evals[approx.name] =
                (x...) -> sum(w[i] * evs[i](x...) for i in eachindex(evs))
            ensemble_std[approx.name] = (x...) -> begin
                vals = [ev(x...) for ev in evs]
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
                (converged=bng_converged, iterations=total_iters,
                 reason=bng_reason,
                 method=:bng, n_dynamics_failures=n_dyn_fail,
                 n_ensemble=K_total, member_losses=member_losses,
                 member_weights=w_members, member_converged=member_plateaued,
                 simulation_failed=sim_failed,
                 ensemble_std=ensemble_std))
end
