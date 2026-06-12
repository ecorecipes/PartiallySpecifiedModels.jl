# ─── Pseudo-Marginal MCMC Solver ─────────────────────────────────────
#
# A genuine pseudo-marginal sampler (Andrieu & Roberts 2009): random-walk
# Metropolis whose acceptance ratio uses an UNBIASED, NONNEGATIVE, stochastic
# estimate of the marginal likelihood.
#
# The estimate is Monte-Carlo over the probabilistic ODE solver's posterior:
# the joint Gauss–Markov filter is run conditioned on the ODE, then complete
# state trajectories are drawn by forward-filter / backward-sampling (FFBS);
# the data likelihood is averaged over the draws,
#
#     L̂(θ, u) = (1/N) Σ_{s=1}^{N} Π_i p(y_i | x^{(s)}(t_i)),   x^{(s)} ~ p(x|θ),
#
# which is unbiased for the marginal likelihood ∫ p(Y|x) p(x|θ) dx.  Fresh
# auxiliary randomness u is drawn every iteration (standard pseudo-marginal),
# and the noisy log-estimate of the CURRENT state is retained between
# iterations — this is what makes the chain target the exact posterior
# despite the likelihood being estimated rather than evaluated.
#
# Reference: Andrieu & Roberts (2009); Chkrebtii et al (2016).

using MCMCChains

# ─── unbiased likelihood estimator via FFBS ──────────────────────────

_pm_logsumexp(v) = (m = maximum(v); isfinite(m) ? m + log(sum(x -> exp(x - m), v)) : m)

function _pm_mvn_sample(μ::Vector{Float64}, Σ::Matrix{Float64}, rng)
    D = length(μ)
    base = maximum(diag(Σ))
    base <= 0 && return copy(μ)                      # degenerate (e.g. t=0)
    F = cholesky(Symmetric(Σ + 1e-12 * max(base, 1.0) * I), check=false)
    issuccess(F) ? μ .+ F.L * randn(rng, D) : copy(μ)
end

"""
    _pm_sample_traj(filt_out, rng) -> Vector{Vector{Float64}}

Draw one complete joint state trajectory from the probabilistic solver's
Gauss–Markov posterior by backward sampling (FFBS).
"""
function _pm_sample_traj(filt_out::Dict, rng)
    μ_filt = filt_out["μ_filt"]; Σ_filt = filt_out["Σ_filt"]
    μ_pred = filt_out["μ_pred"]; Σ_pred = filt_out["Σ_pred"]; A = filt_out["A"]
    N = length(μ_filt) - 1
    X = Vector{Vector{Float64}}(undef, N + 1)
    X[N+1] = _pm_mvn_sample(μ_filt[N+1], Σ_filt[N+1], rng)
    for n in N:-1:1
        Σpf = cholesky(Symmetric(Σ_pred[n+1]) + 1e-12 * I, check=false)
        G = issuccess(Σpf) ? (Σ_filt[n] * A') / Σpf : (Σ_filt[n] * A') * pinv(Σ_pred[n+1])
        m = μ_filt[n] + G * (X[n+1] - μ_pred[n+1])
        Cov = Σ_filt[n] - G * Σ_pred[n+1] * G'
        Cov = 0.5 * (Cov + Cov')
        X[n] = _pm_mvn_sample(m, Matrix(Cov), rng)
    end
    X
end

"""
    _pm_loglik_hat(ode_rhs!, u0, tspan, n_steps, n_deriv, sigma,
                   data, dtimes, obs_to_state, obs_var, n_particles, rng;
                   interrogate=:kramer) -> Float64

Log of an unbiased Monte-Carlo estimate of the marginal likelihood.
"""
function _pm_loglik_hat(ode_rhs!, u0, tspan, n_steps, n_deriv, sigma,
                        data, dtimes, obs_to_state, obs_var, n_particles, rng;
                        interrogate::Symbol=:kramer)
    filt = probsolve_filter(ode_rhs!, nothing, u0, tspan, n_steps, n_deriv, sigma;
                            interrogate=interrogate)
    times = filt["times"]; q = filt["q"]
    n_obs = size(data, 2)
    obs_ind = clamp.([searchsortedfirst(times, dtimes[i]) for i in 1:length(dtimes)],
                     1, n_steps + 1)
    logws = Vector{Float64}(undef, n_particles)
    c = -0.5 * log(2π * obs_var)
    for s in 1:n_particles
        X = _pm_sample_traj(filt, rng)
        ll = 0.0
        for i in 1:length(dtimes)
            gi = obs_ind[i]
            for j in 1:n_obs
                val = X[gi][(obs_to_state[j]-1)*q + 1]
                ll += c - 0.5 * (data[i, j] - val)^2 / obs_var
            end
        end
        logws[s] = ll
    end
    _pm_logsumexp(logws) - log(n_particles)
end

function _pm_logprior(beta, penalties, offsets, prior_scale)
    lp = 0.0
    for (idx, S) in enumerate(penalties)
        np = size(S, 1); off = offsets[idx]
        bk = @view beta[off+1:off+np]
        lp -= 0.5 / prior_scale * dot(bk, S * bk)
    end
    lp -= 0.5 * sum(abs2, beta) / (100.0 * prior_scale)
    lp
end

# ─── Solve method ────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::PseudoMarginalSolver)

Fit a partially specified model using pseudo-marginal MCMC. A probabilistic
ODE solver provides an UNBIASED, stochastic estimate of the marginal
likelihood (Monte-Carlo over the solver posterior via FFBS), which drives a
random-walk Metropolis sampler that targets the exact posterior over the
unknown-function parameters.

# References
- Andrieu, C. & Roberts, G.O. (2009), "The pseudo-marginal approach for
  efficient Monte Carlo computations", Ann. Statist. 37(2), 697–725.
- Chkrebtii, O. et al. (2016), "Bayesian solution uncertainty quantification
  for differential equations", Bayesian Analysis 11(4), 1239–1267.

# Returns
`PSMSolution`; `convergence` holds the MCMC chain (`MCMCChains.Chains`).
"""
function SciMLBase.solve(prob::PSMProblem, alg::PseudoMarginalSolver)
    _validate_problem(prob, "PseudoMarginalSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)
    rng = Random.default_rng()

    if alg.initial_params !== nothing
        beta0 = copy(alg.initial_params)
    else
        beta0 = Float64[]
        for approx in prob.approximators
            if approx isa NeuralApproximator
                spec = mlp_spec_from_lux(approx.model)
                rng0 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
                append!(beta0, init_mlp_params(spec, rng0))
            else
                append!(beta0, initial_params(approx))
            end
        end
    end
    n_beta = length(beta0)

    sigma = if alg.sigma === nothing
        sig = Float64[]
        for k in 1:n_vars
            oi = findfirst(j -> prob.obs_to_state[j] == k, 1:n_obs)
            if oi !== nothing
                dr = maximum(prob.data_values[:, oi]) - minimum(prob.data_values[:, oi])
                push!(sig, max(dr * 0.01, 0.01))
            else
                push!(sig, 1.0)
            end
        end
        sig
    else
        alg.sigma
    end

    obs_var = if alg.obs_var === nothing
        tv = 0.0
        for j in 1:n_obs; tv += var(prob.data_values[:, j]); end
        max(tv / n_obs * 0.01, 1e-4)
    else
        alg.obs_var
    end

    penalties, offsets, _ = _build_penalty_info(prob)
    n_particles = 32  # Monte-Carlo draws per likelihood estimate

    data = Float64.(prob.data_values)
    dtimes = Float64.(prob.data_times)
    u0 = Float64.(prob.u0)

    function loglik_hat(beta)
        p = build_param_struct(prob, beta)
        rhs!(du, u, pu, t) = prob.dynamics!(du, u, p, t)
        try
            _pm_loglik_hat(rhs!, u0, prob.tspan, alg.n_steps, alg.n_deriv, sigma,
                           data, dtimes, prob.obs_to_state, obs_var, n_particles, rng)
        catch
            -Inf
        end
    end

    logpost(beta, ll_hat) = ll_hat + _pm_logprior(beta, penalties, offsets, alg.prior_scale)

    if verbose
        println("PseudoMarginalSolver (random-walk PM-MCMC): $n_beta params, " *
                "$n_particles particles, n_steps=$(alg.n_steps)")
        println("  σ=$(round.(sigma, sigdigits=3)) obs_var=$(round(obs_var, sigdigits=3))")
    end

    # ── Random-walk Metropolis with adaptive global scale ──
    base_sd = max.(0.1 .* abs.(beta0), 0.05)
    logscale = 0.0
    target = alg.target_accept
    n_total = alg.n_warmup + alg.n_samples

    θ = copy(beta0)
    ll_cur = loglik_hat(θ)
    lp_cur = logpost(θ, ll_cur)
    # Guard against a -Inf start: jitter until finite.
    tries = 0
    while !isfinite(lp_cur) && tries < 50
        θ .= beta0 .+ base_sd .* randn(rng, n_beta)
        ll_cur = loglik_hat(θ); lp_cur = logpost(θ, ll_cur); tries += 1
    end

    samples = zeros(alg.n_samples, n_beta)
    n_accept = 0
    for it in 1:n_total
        prop = θ .+ exp(logscale) .* base_sd .* randn(rng, n_beta)
        ll_prop = loglik_hat(prop)
        lp_prop = logpost(prop, ll_prop)
        α = isfinite(lp_prop) ? min(1.0, exp(lp_prop - lp_cur)) : 0.0
        if rand(rng) < α
            θ = prop; lp_cur = lp_prop; ll_cur = ll_prop
            it > alg.n_warmup && (n_accept += 1)
        end
        # Robbins–Monro adaptation of the global proposal scale during warmup.
        if it <= alg.n_warmup
            logscale += (α - target) / sqrt(it)
        end
        if it > alg.n_warmup
            samples[it - alg.n_warmup, :] .= θ
        end
    end

    acc_rate = n_accept / max(alg.n_samples, 1)
    if verbose
        println("  acceptance (post-warmup): $(round(acc_rate, sigdigits=3))")
    end

    pnames = _param_names(prob, false)
    chain = MCMCChains.Chains(samples, pnames)

    # Posterior mean parameters for the point estimate / evaluators.
    map_beta = vec(mean(samples, dims=1))
    p_opt = build_param_struct(prob, map_beta)
    ode_rhs_opt!(du, u, pu, t) = prob.dynamics!(du, u, p_opt, t)
    μ_smooth, Σ_smooth, times = probsolve(ode_rhs_opt!, nothing, u0,
                                          prob.tspan, alg.n_steps, alg.n_deriv, sigma;
                                          interrogate=:kramer)

    data_loss = 0.0
    pred = zeros(n_t, n_obs)
    for i in 1:n_t
        idx = clamp(searchsortedfirst(times, prob.data_times[i]), 1, length(times))
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[i, j] = μ_smooth[idx][sk][1]
            data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
        end
    end

    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = map_beta[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2], length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
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
        elseif approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            let pk = copy(params_k), s = spec, lo_ = lo, span_ = span
                uf_evals[approx.name] = x -> begin
                    xn = (lo_ !== nothing && span_ !== nothing && span_ > 0) ?
                         (Float64(x isa AbstractArray ? x[1] : x) - lo_) / span_ :
                         Float64(x isa AbstractArray ? x[1] : x)
                    mlp_evaluate(s, pk, xn)
                end
            end
        end
    end

    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => map_beta[offset+1:offset+np])
        offset += np
    end
    ca = ComponentArray(NamedTuple(ca_entries))

    sol_var = zeros(n_t, n_vars)
    for i in 1:n_t
        idx = clamp(searchsortedfirst(times, prob.data_times[i]), 1, length(times))
        for k in 1:n_vars
            sol_var[i, k] = Σ_smooth[idx][k][1, 1]
        end
    end

    PSMSolution(
        ca,
        ll_cur,
        data_loss,
        Float64(n_beta),
        Float64[],
        pred,
        Float64.(prob.data_values),
        Float64.(prob.data_times),
        uf_evals,
        chain,
    )
end
