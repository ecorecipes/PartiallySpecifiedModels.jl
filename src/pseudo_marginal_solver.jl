# ─── Pseudo-Marginal MCMC Solver ─────────────────────────────────────
#
# This solver targets the approximate posterior induced by one of the
# probabilistic-ODE inner marginal likelihoods (:basic, :fenrir, :dalton),
# but uses a *random positive unbiased estimator* of that approximate
# likelihood inside a Metropolis-Hastings scheme.  The random auxiliary
# likelihood estimate is carried through the MH state, restoring the
# pseudo-marginal correctness property that the previous deterministic
# NUTS implementation did not have.
#
# The estimator here randomizes the chosen deterministic approximate
# likelihood L̃(θ) with a mean-one positive noise variable R so that
# ĤL(θ, U) = L̃(θ) R satisfies E[ĤL(θ, U)] = L̃(θ).  Consequently the
# Markov chain is exact for the approximate posterior proportional to
# π(θ) L̃(θ), not for the exact ODE likelihood.

using MCMCChains

struct PseudoMarginalLogDensity
    prob::PSMProblem
    alg::PseudoMarginalSolver
    penalty_matrices::Vector{Matrix{Float64}}
    param_offsets::Vector{Int}
    n_params::Int
    sigma::Vector{Float64}
    obs_var::Float64
end

function _pseudo_marginal_logprior(ld::PseudoMarginalLogDensity, beta::AbstractVector)
    alg = ld.alg
    lp = 0.0
    for (idx, S) in enumerate(ld.penalty_matrices)
        np = size(S, 1)
        off = ld.param_offsets[idx]
        beta_k = beta[off+1:off+np]
        lp -= 0.5 / alg.prior_scale * dot(beta_k, S * beta_k)
    end
    lp -= 0.5 * sum(beta .^ 2) / (100.0 * alg.prior_scale)
    lp
end

function _pseudo_marginal_inner_loglik(ld::PseudoMarginalLogDensity, theta)
    prob = ld.prob
    alg = ld.alg
    beta = Float64.(theta[1:ld.n_params])
    p = build_param_struct(prob, beta)

    function ode_rhs!(du, u, p_unused, t)
        prob.dynamics!(du, u, p, t)
    end

    loglik_fn = if alg.inner_method == :fenrir
        fenrir_loglik
    elseif alg.inner_method == :basic
        basic_loglik
    elseif alg.inner_method == :dalton
        _dalton_loglik
    else
        throw(ArgumentError("Unsupported pseudo-marginal inner_method $(repr(alg.inner_method))"))
    end

    ll = try
        loglik_fn(ode_rhs!, nothing, Float64.(prob.u0), prob.tspan,
                  alg.n_steps, alg.n_deriv, ld.sigma,
                  Float64.(prob.data_values), Float64.(prob.data_times),
                  prob.obs_to_state, ld.obs_var;
                  interrogate=:kramer)
    catch
        -Inf
    end

    isfinite(ll) ? ll : -Inf
end

function _pseudo_marginal_noise(rng, n_particles::Int)
    n_particles > 0 || throw(ArgumentError("PseudoMarginalSolver n_particles must be positive"))
    noise = 0.0
    for _ in 1:n_particles
        noise += randexp(rng)
    end
    noise / n_particles
end

function _pseudo_marginal_logdensity(ld::PseudoMarginalLogDensity, theta, rng)
    ll_det = _pseudo_marginal_inner_loglik(ld, theta)
    isfinite(ll_det) || return -Inf
    noise = _pseudo_marginal_noise(rng, ld.alg.n_particles)
    ll_est = ll_det + log(noise)
    ll_est + _pseudo_marginal_logprior(ld, Float64.(theta[1:ld.n_params]))
end

function _pseudo_marginal_deterministic_logdensity(ld::PseudoMarginalLogDensity, theta)
    ll_det = _pseudo_marginal_inner_loglik(ld, theta)
    isfinite(ll_det) || return -Inf
    ll_det + _pseudo_marginal_logprior(ld, Float64.(theta[1:ld.n_params]))
end

function LogDensityProblems.capabilities(::Type{PseudoMarginalLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::PseudoMarginalLogDensity)
    ld.n_params
end

function LogDensityProblems.logdensity(ld::PseudoMarginalLogDensity, theta)
    _pseudo_marginal_logdensity(ld, theta, Random.default_rng())
end

"""
    solve(prob::PSMProblem, alg::PseudoMarginalSolver)

Fit a partially specified model using pseudo-marginal Metropolis-Hastings.

`inner_method` selects the *deterministic approximate* probabilistic-ODE
marginal likelihood `L̃(θ)` (`:basic`, `:fenrir`, or `:dalton`). Each MH
accept/reject step uses a positive unbiased random estimator `ĤL(θ, U)` with
`E[ĤL(θ, U)] = L̃(θ)`, so the chain is exact for the approximate posterior
proportional to `π(θ)L̃(θ)`.

The current implementation restores pseudo-marginal correctness for the
approximate inner likelihood, but it does **not** provide an unbiased
estimator of the exact ODE marginal likelihood.

# Returns
`PSMSolution` with fitted parameters, trajectory, unknown functions, and the
full MCMC chain in `sol.convergence`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::PseudoMarginalSolver)
    _validate_problem(prob, "PseudoMarginalSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)

    beta0 = if alg.initial_params !== nothing
        copy(alg.initial_params)
    else
        build_initial_params(prob)
    end
    n_beta = length(beta0)

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

    obs_var = if alg.obs_var === nothing
        total_var = 0.0
        for j in 1:n_obs
            total_var += var(prob.data_values[:, j])
        end
        max(total_var / n_obs * 0.01, 1e-4)
    else
        alg.obs_var
    end

    penalties, offsets, _ = _build_penalty_info(prob)
    ld = PseudoMarginalLogDensity(prob, alg, penalties, offsets, n_beta, sigma, obs_var)

    rng = alg.rng_seed === nothing ? Random.Xoshiro() : Random.Xoshiro(alg.rng_seed)
    D = LogDensityProblems.dimension(ld)
    theta_curr = copy(beta0)
    logpost_curr = _pseudo_marginal_logdensity(ld, theta_curr, rng)
    if !isfinite(logpost_curr)
        logpost_curr = _pseudo_marginal_deterministic_logdensity(ld, theta_curr)
    end
    isfinite(logpost_curr) || error("PseudoMarginalSolver failed to initialize a finite log-posterior")

    n_total = alg.n_warmup + alg.n_samples
    sample_matrix = zeros(alg.n_samples, D)
    accept_count = 0
    warmup_accept = 0
    log_prop_scale = log(max(alg.proposal_scale, 1e-6))

    if verbose
        println("PseudoMarginalSolver: $n_beta UF params, inner=$(alg.inner_method), " *
                "n_particles=$(alg.n_particles), proposal_scale=$(round(alg.proposal_scale, sigdigits=3))")
        println("  σ (IBM scale): $(round.(sigma, sigdigits=3))")
        println("  obs_var: $(round(obs_var, sigdigits=3))")
        println("  prior_scale: $(alg.prior_scale)")
    end

    for iter in 1:n_total
        prop_scale = exp(log_prop_scale)
        theta_prop = theta_curr .+ prop_scale .* randn(rng, D)
        logpost_prop = _pseudo_marginal_logdensity(ld, theta_prop, rng)

        accepted = false
        if isfinite(logpost_prop)
            if log(rand(rng)) < min(0.0, logpost_prop - logpost_curr)
                theta_curr .= theta_prop
                logpost_curr = logpost_prop
                accepted = true
            end
        end

        if iter <= alg.n_warmup
            warmup_accept += accepted
            log_prop_scale += ((accepted ? 1.0 : 0.0) - alg.target_accept) / sqrt(iter)
        else
            sample_idx = iter - alg.n_warmup
            sample_matrix[sample_idx, :] .= theta_curr
            accept_count += accepted
        end

        if verbose && (iter <= 5 || iter % 100 == 0 || iter == n_total)
            acc = iter <= alg.n_warmup ? warmup_accept / iter : accept_count / max(iter - alg.n_warmup, 1)
            println("  iter $iter / $n_total: accept_rate=$(round(acc, sigdigits=3)), " *
                    "proposal_scale=$(round(exp(log_prop_scale), sigdigits=3))")
        end
    end

    pnames = _param_names(prob, false)
    chain = MCMCChains.Chains(sample_matrix, pnames)

    logp_values = [_pseudo_marginal_deterministic_logdensity(ld, sample_matrix[i, :])
                   for i in 1:alg.n_samples]
    map_idx = argmax(logp_values)
    map_beta = sample_matrix[map_idx, 1:n_beta]
    p_opt = build_param_struct(prob, map_beta)

    function ode_rhs_opt!(du, u, p_unused, t)
        prob.dynamics!(du, u, p_opt, t)
    end

    μ_smooth, Σ_smooth, times = probsolve(ode_rhs_opt!, nothing, Float64.(prob.u0),
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
                    xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                        (Float64(x isa AbstractArray ? x[1] : x) - lo_) / span_
                    else
                        Float64(x isa AbstractArray ? x[1] : x)
                    end
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

    if verbose
        println("  Final accept_rate=$(round(accept_count / max(alg.n_samples, 1), sigdigits=3))")
        println("  MAP -logpost: $(round(-logp_values[map_idx], sigdigits=5))")
    end

    PSMSolution(
        ca,
        -logp_values[map_idx],
        data_loss,
        Float64(n_beta),
        Float64[],
        pred,
        Float64.(prob.data_values),
        Float64.(prob.data_times),
        uf_evals,
        chain
    )
end
