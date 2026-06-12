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

# ─── Core: frozen-linearization joint filter ──────────────────────
#
# Correctness note (the heart of DALTON):
#   logp(Y|Z) = logp(Y,Z) − logp(Z)
# holds ONLY if both evidences are computed under the SAME, data-INDEPENDENT
# linear-Gaussian model.  The ODE pseudo-observation Z is a linearization
# of W·X − f(X) about a reference trajectory; if that linearization point
# moves when data are assimilated (as in a naive joint filter), the two
# passes use different models and the Z-terms do not cancel.  We therefore
# compute the linearization (Hₙ, bₙ) ONCE from an ODE-only "reference"
# filter and FREEZE it, then reuse those exact measurement models in the
# joint pass.  Both passes then share one model and the identity is exact.

"""
    _dalton_reference(ode_fun!, p, u0, tspan, n_steps, q, sigma; interrogate)

ODE-only joint Kalman filter. Returns the frozen interrogation models
`(Hs, bs)` (one per step) and the marginal ODE log-evidence `logp(Z)`.
"""
function _dalton_reference(ode_fun!, p, u0::AbstractVector,
                           tspan::Tuple{Float64,Float64},
                           n_steps::Int, q::Int, sigma::Vector{Float64};
                           interrogate::Symbol=:kramer)
    t_min, t_max = tspan
    n_vars = length(u0); D = n_vars * q
    A, Qmat = _joint_ibm((t_max - t_min) / n_steps, q, sigma)
    E0, E1 = _joint_selectors(n_vars, q)
    V = Matrix(1e-10 * I, n_vars, n_vars)

    μf = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    Σf = zeros(D, D)
    Hs = Vector{Matrix{Float64}}(undef, n_steps)
    bs = Vector{Vector{Float64}}(undef, n_steps)
    logZ = 0.0
    ztarget = zeros(n_vars)

    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps
        μp = A * μf; Σp = A * Σf * A' + Qmat; Σp = 0.5 * (Σp + Σp')
        H, b = _joint_interrogate(ode_fun!, E0, E1, t_n, μp, p, n_vars;
                                  method=interrogate)
        Hs[n] = H; bs[n] = b
        zmean = H * μp + b
        S = H * Σp * H' + V; S = 0.5 * (S + S')
        logZ += logpdf_mvn(ztarget, zmean, S)
        Sf = cholesky(Symmetric(S), check=false)
        K = (Σp * H') * (issuccess(Sf) ? inv(Sf) : pinv(S))
        μf = μp - K * zmean; Σf = Σp - K * H * Σp; Σf = 0.5 * (Σf + Σf')
    end
    Hs, bs, logZ
end

"""
    _dalton_joint_evidence(ode_fun!, p, u0, tspan, n_steps, q, sigma, Hs, bs,
                           obs_data, obs_times, obs_to_state, obs_var)

Joint Kalman filter using the FROZEN interrogation models `(Hs, bs)`,
assimilating both the ODE pseudo-observations and the data. Returns the
joint log-evidence `logp(Y,Z)`.
"""
function _dalton_joint_evidence(ode_fun!, p, u0::AbstractVector,
                                tspan::Tuple{Float64,Float64},
                                n_steps::Int, q::Int, sigma::Vector{Float64},
                                Hs::Vector{Matrix{Float64}}, bs::Vector{Vector{Float64}},
                                obs_data::Matrix{Float64}, obs_times::Vector{Float64},
                                obs_to_state::Vector{Int}, obs_var::Float64)
    t_min, t_max = tspan
    n_vars = length(u0); D = n_vars * q
    A, Qmat = _joint_ibm((t_max - t_min) / n_steps, q, sigma)
    V = Matrix(1e-10 * I, n_vars, n_vars)

    μf = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    Σf = zeros(D, D)
    times = collect(range(t_min, t_max, length = n_steps + 1))
    n_t_obs = size(obs_data, 1); n_obs_vars = length(obs_to_state)
    obs_ind = clamp.([searchsortedfirst(times, obs_times[i]) for i in 1:n_t_obs],
                     1, n_steps + 1)
    Dmats = [reshape([(c == (obs_to_state[j]-1)*q + 1) ? 1.0 : 0.0 for c in 1:D], 1, D)
             for j in 1:n_obs_vars]
    Vobs = fill(obs_var, 1, 1)
    ztarget = zeros(n_vars)
    logEv = 0.0

    function assimilate_data!(gi)
        for i in 1:n_t_obs
            obs_ind[i] == gi || continue
            for j in 1:n_obs_vars
                Dj = Dmats[j]; y = [obs_data[i, j]]
                μfo, Σfo = kalman_forecast(μf, Σf, zeros(1), Dj, Vobs)
                logEv += logpdf_mvn(y, μfo, Σfo)
                μf, Σf = kalman_update(μf, Σf, y, zeros(1), Dj, Vobs)
            end
        end
    end

    assimilate_data!(1)
    for n in 1:n_steps
        μp = A * μf; Σp = A * Σf * A' + Qmat; Σp = 0.5 * (Σp + Σp')
        H = Hs[n]; b = bs[n]
        zmean = H * μp + b
        S = H * Σp * H' + V; S = 0.5 * (S + S')
        logEv += logpdf_mvn(ztarget, zmean, S)
        Sf = cholesky(Symmetric(S), check=false)
        K = (Σp * H') * (issuccess(Sf) ? inv(Sf) : pinv(S))
        μf = μp - K * zmean; Σf = Σp - K * H * Σp; Σf = 0.5 * (Σf + Σf')
        assimilate_data!(n + 1)
    end
    logEv
end

# ─── DALTON log-likelihood ────────────────────────────────────────

"""
    _dalton_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                   obs_data, obs_times, obs_to_state, obs_var;
                   interrogate=:kramer)

DALTON data-conditional log-likelihood `logp(Y|Z) = logp(Y,Z) − logp(Z)`
(Wu & Lysy 2024). Computed with a frozen, data-independent ODE
linearization so the identity is exact (see note above).
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
    q = n_deriv
    Hs, bs, logZ = _dalton_reference(ode_fun!, p, u0, tspan, n_steps, q, sigma;
                                     interrogate=interrogate)
    logYZ = _dalton_joint_evidence(ode_fun!, p, u0, tspan, n_steps, q, sigma,
                                   Hs, bs, obs_data, obs_times, obs_to_state, obs_var)
    ll = logYZ - logZ
    isfinite(ll) ? ll : -Inf
end

# ─── Solver dispatch ──────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::DaltonSolver)

Fit a partially specified model using the DALTON (Data-Adaptive Likelihood
with Transformed ObservatioNs) probabilistic ODE solver of Wu & Lysy
(2024). The data-conditional likelihood `logp(Y|Z) = logp(Y,Z) − logp(Z)`
is evaluated with two joint Kalman-filter passes that share a single,
data-independent ODE linearization, then the unknown-function parameters
are optimized to maximize it.

# Algorithm
1. Set up the IBM prior of order `n_deriv` for each state variable.
2. Reference pass: ODE-only joint filter giving the marginal evidence
   `logp(Z)` and the frozen EKF1 linearization models `(Hₙ, bₙ)`.
3. Joint pass: re-filter with the same frozen models, additionally
   assimilating the data, giving `logp(Y,Z)`.
4. `logp(Y|Z) = logp(Y,Z) − logp(Z)`; optimize the parameters.

# References
- Wu, M. & Lysy, M. (2024), "Data-adaptive probabilistic likelihood
  approximation for ordinary differential equations", AISTATS.

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
        elseif approx isa SPDEApproximator
            S = penalty_matrix(approx)
            if S !== nothing
                push!(smooth_mats, S)
                push!(smooth_offsets, offset_acc)
            end
        elseif approx isa ShapeConstrainedSPDEApproximator
            S = penalty_matrix(approx)
            if S !== nothing
                push!(smooth_mats, S)
                push!(smooth_offsets, offset_acc)
            end
        end
        offset_acc += np
    end

    n_smooth = length(smooth_mats)
    smooth_lambdas = fill(0.1 / max(obs_var, 1e-6), n_smooth)

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

            # Smoothing penalty for B-splines
            penalty = 0.0
            for (k, (S, off)) in enumerate(zip(smooth_mats, smooth_offsets))
                np = size(S, 1)
                β_k = β_[off+1:off+np]
                penalty += smooth_lambdas[k] * dot(β_k, S * β_k)
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

    # ── Fellner-Schall λ refinement (autodiff Jacobian) ──
    if n_smooth > 0
        for fs_cycle in 1:3
            function pred_vec_d(β_)
                p_ = build_autodiff_param_struct(prob, β_)
                u0_ = prob.u0 isa Function ? prob.u0(p_) : prob.u0
                ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
                    (du, u, params, t) -> prob.dynamics!(du, u, p_, t))
                sol_ad = OrdinaryDiffEq.solve(
                    ODEProblem(ode_fn, eltype(β_).(u0_), prob.tspan, nothing),
                    prob.ode_solver;
                    saveat=prob.data_times, abstol=1e-7, reltol=1e-7,
                    maxiters=100000, verbose=false)
                pred = eltype(β_)[]
                for j in 1:n_obs
                    sk = prob.obs_to_state[j]
                    for i in 1:n_t
                        push!(pred, i <= length(sol_ad.t) ? sol_ad[sk, i] : eltype(β_)(0))
                    end
                end
                pred
            end

            J_ad = try
                ForwardDiff.jacobian(pred_vec_d, beta_opt)
            catch
                pred_b = pred_vec_d(beta_opt)
                nd = length(pred_b)
                J_fd = zeros(nd, n_beta)
                for j in 1:n_beta
                    bp = copy(beta_opt); bp[j] += 1e-6
                    J_fd[:, j] .= (pred_vec_d(bp) .- pred_b) ./ 1e-6
                end
                J_fd
            end

            y_vec = Float64[prob.data_values[i, j] for j in 1:n_obs for i in 1:n_t]
            mu_vec = Float64.(pred_vec_d(beta_opt))
            n_data = length(y_vec)
            JWJ = J_ad' * J_ad
            σ²_hat = sum((y_vec .- mu_vec).^2) / max(n_data - sum(s -> size(s,1), smooth_mats), 1.0)

            for (k, (S, off)) in enumerate(zip(smooth_mats, smooth_offsets))
                np_k = size(S, 1)
                β_k = beta_opt[off+1:off+np_k]
                bSb = max(dot(β_k, S * β_k), 1e-20)
                S_full = zeros(n_beta, n_beta)
                for (kk, (Sk, offk)) in enumerate(zip(smooth_mats, smooth_offsets))
                    npk = size(Sk, 1); idx = (offk+1):(offk+npk)
                    S_full[idx, idx] .+= smooth_lambdas[kk] .* Sk
                end
                H_hat = JWJ + S_full
                maxd = maximum(abs.(diag(H_hat)))
                for i in 1:n_beta; H_hat[i,i] += 1e-12*maxd; end
                H_inv = try; inv(cholesky(Symmetric(H_hat))); catch; pinv(H_hat); end
                idx_k = (off+1):(off+np_k)
                edf_k = clamp(tr(H_inv[idx_k, idx_k] * JWJ[idx_k, idx_k]), 0.01, np_k - 0.01)
                smooth_lambdas[k] = clamp(σ²_hat * edf_k / bSb, exp(RHO_MIN), exp(RHO_MAX))
            end

            if verbose; @printf("  FS cycle %d: λ = %s\n", fs_cycle, round.(smooth_lambdas, sigdigits=3)); end

            result_re = Optim.optimize(neg_loglik, fd_gradient, beta_opt,
                Optim.LBFGS(linesearch=LineSearches.BackTracking()),
                Optim.Options(iterations=alg.maxiters ÷ 2, g_tol=1e-6, f_reltol=1e-10, show_trace=false);
                inplace=false)
            beta_opt = Optim.minimizer(result_re)
        end
        if verbose; @printf("  Final λ: %s\n", round.(smooth_lambdas, sigdigits=3)); end
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
        elseif approx isa SPDEApproximator
            uf_evals[approx.name] = build_spde_evaluator(approx.mesh_points, params_k)
        elseif approx isa ShapeConstrainedSPDEApproximator
            uf_evals[approx.name] = build_constrained_spde_evaluator(approx, params_k)
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
        Float64.(smooth_lambdas),            # smoothing_params (Fellner-Schall)
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
