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
    # Escalate jitter until the Cholesky succeeds: silently returning the
    # mean would be a degenerate draw, breaking the FFBS estimator's
    # unbiasedness in exactly the near-singular cases where it matters.
    for jit in (1e-12, 1e-9, 1e-6)
        F = cholesky(Symmetric(Σ + jit * max(base, 1.0) * I), check=false)
        issuccess(F) && return μ .+ F.L * randn(rng, D)
    end
    F = cholesky(Symmetric(Σ + 1e-4 * max(base, 1.0) * I))
    μ .+ F.L * randn(rng, D)
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
    obs_ind = _nearest_grid_indices(times, dtimes)
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

# Masked data

This solver does NOT support masked observations and raises an error if
any data cell is masked (`data_weights == 0` or a non-finite
`data_values` entry). Its likelihood is evaluated inside a Kalman /
particle recursion with no per-cell mask: honouring a mask means skipping
the FILTER UPDATE, not merely the density term, for the masked cells.
Left unguarded the masked cells corrupt the filter state while the run
still looks like an ordinary converged fit, so it fails loudly instead.
Drop the masked rows from `data_times`/`data_values`, or use one of the
masking-capable solvers (`LAML`, `GCVSolver`, `CollocationLAML`,
`GradientMatching`, `TwoStageSolver`, `BNGSolver`, `ODINSolver`,
`RKHSSolver`, `IntegralMatchingSolver`, `AdamSolver`,
`MultipleShootingSolver`, `DerivativeFreeSolver`, `MCMCSolver`,
`MagiSolver`, `VariationalSolver`, `ABCSolver`,
`ProfileLikelihoodSolver`).
"""
function SciMLBase.solve(prob::PSMProblem, alg::PseudoMarginalSolver)
    _validate_problem(prob, "PseudoMarginalSolver"; require_continuous=true)
    _reject_masked_data(prob, "PseudoMarginalSolver")
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
                rng0 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
                append!(beta0, neural_init_params(approx, rng0))
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

    # Likelihood estimator selected by alg.inner_method:
    #   :ffbs   — unbiased FFBS Monte-Carlo average (pseudo-marginal MCMC
    #             in the Andrieu & Roberts 2009 sense; the default)
    #   :fenrir — deterministic Fenrir conditional evidence (Tronarp et al.
    #             2022); the chain is then plain adaptive RWM on an
    #             approximate likelihood
    #   :dalton — deterministic DALTON data-adaptive likelihood (Wu & Lysy
    #             2024); likewise plain RWM
    alg.inner_method in (:ffbs, :fenrir, :dalton) ||
        error("PseudoMarginalSolver: inner_method must be :ffbs, :fenrir, " *
              "or :dalton (got :$(alg.inner_method))")
    function loglik_hat(beta)
        p = build_param_struct(prob, beta)
        rhs!(du, u, pu, t) = prob.dynamics!(du, u, p, t)
        try
            if alg.inner_method == :fenrir
                fenrir_loglik(rhs!, nothing, u0, prob.tspan, alg.n_steps,
                              alg.n_deriv, sigma, data, dtimes,
                              prob.obs_to_state, obs_var)
            elseif alg.inner_method == :dalton
                _dalton_loglik(rhs!, nothing, u0, prob.tspan, alg.n_steps,
                               alg.n_deriv, sigma, data, dtimes,
                               prob.obs_to_state, obs_var)
            else
                _pm_loglik_hat(rhs!, u0, prob.tspan, alg.n_steps, alg.n_deriv,
                               sigma, data, dtimes, prob.obs_to_state,
                               obs_var, n_particles, rng)
            end
        catch e
            _is_program_error(e) && rethrow()
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

    pred = zeros(n_t, n_obs)
    for i in 1:n_t
        idx = _nearest_grid_index(times, prob.data_times[i])
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[i, j] = μ_smooth[idx][sk][1]
        end
    end
    data_loss = weighted_data_loss(prob, pred)

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
            uf_evals[approx.name] = build_neural_evaluator(approx, params_k)
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
        idx = _nearest_grid_index(times, prob.data_times[i])
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
