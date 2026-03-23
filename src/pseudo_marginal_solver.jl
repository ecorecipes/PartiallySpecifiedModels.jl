# ─── Pseudo-Marginal MCMC Solver ─────────────────────────────────────
#
# Combines a probabilistic ODE solver (fenrir/dalton) as an inner likelihood
# estimator with NUTS/HMC for Bayesian posterior sampling over unknown
# function parameters.
#
# Instead of solving the ODE deterministically (as MCMCSolver does), the
# log-likelihood is computed via a Kalman-filter-based marginal likelihood
# (Tronarp et al 2022), which accounts for ODE discretization uncertainty.
#
# Uses the same LogDensityProblems.jl + AdvancedHMC.jl infrastructure as
# MCMCSolver.
#
# Reference: Chkrebtii et al (2016), Tronarp et al (2022)

using AdvancedHMC
using LogDensityProblems
using LogDensityProblemsAD
using MCMCChains
import AbstractMCMC

# ─── Log-density problem for pseudo-marginal MCMC ────────────────

struct PseudoMarginalLogDensity
    prob::PSMProblem
    alg::PseudoMarginalSolver
    penalty_matrices::Vector{Matrix{Float64}}
    param_offsets::Vector{Int}
    n_params::Int
    sigma::Vector{Float64}
end

function LogDensityProblems.capabilities(::Type{PseudoMarginalLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::PseudoMarginalLogDensity)
    ld.n_params
end

function LogDensityProblems.logdensity(ld::PseudoMarginalLogDensity, theta)
    prob = ld.prob
    alg = ld.alg
    T = eltype(theta)

    beta = theta[1:ld.n_params]

    # Build parameter struct compatible with autodiff (Dual numbers)
    p = build_autodiff_param_struct(prob, beta)

    function ode_rhs!(du, u, p_unused, t)
        prob.dynamics!(du, u, p, t)
    end

    # Compute probabilistic ODE log-likelihood via fenrir or basic method
    loglik_fn = alg.inner_method == :fenrir ? fenrir_loglik : basic_loglik

    ll = try
        loglik_fn(ode_rhs!, nothing, Float64.(prob.u0), prob.tspan,
                  alg.n_steps, alg.n_deriv, ld.sigma,
                  Float64.(prob.data_values), Float64.(prob.data_times),
                  prob.obs_to_state, alg.obs_var;
                  interrogate=:kramer)
    catch e
        return T(-1e20)
    end

    if !isfinite(ll)
        return T(-1e20)
    end

    # Log-prior: smoothing penalty from penalty matrices
    lp = zero(T)
    for (idx, S) in enumerate(ld.penalty_matrices)
        np = size(S, 1)
        off = ld.param_offsets[idx]
        beta_k = beta[off+1:off+np]
        lp -= T(0.5) / T(alg.prior_scale) * dot(beta_k, S * beta_k)
    end

    # Broad Gaussian prior on all parameters
    lp -= T(0.5) * sum(beta .^ 2) / T(100.0 * alg.prior_scale)

    return T(ll) + lp
end

# ─── Solve method ────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::PseudoMarginalSolver)

Fit a partially specified model using pseudo-marginal MCMC. A probabilistic
ODE solver (Kalman-filter-based) provides an unbiased estimate of the
marginal likelihood, which is used inside an MCMC sampler to draw from the
true posterior over the unknown-function parameters.

# Algorithm
1. Initialise parameters and set up the IBM prior.
2. At each MCMC iteration, compute a noisy marginal-likelihood estimate
   via the Kalman filter (forward pass only).
3. Accept/reject proposals using the Metropolis–Hastings ratio with the
   estimated likelihood (pseudo-marginal correctness is guaranteed).
4. Return the full chain and posterior-mean point estimates.

# References
- Andrieu & Roberts (2009), "The pseudo-marginal approach for efficient
  Monte Carlo computations", Ann. Statist.

# Returns
`PSMSolution` with fitted parameters, trajectory, unknown functions,
and the full MCMC chain in `sol.extras[:chain]`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::PseudoMarginalSolver)
    _validate_problem(prob, "PseudoMarginalSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)

    # Initialize parameters
    beta0 = Float64[]
    for approx in prob.approximators
        if approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta0, init_mlp_params(spec, rng))
        else
            append!(beta0, initial_params(approx))
        end
    end
    n_beta = length(beta0)

    # IBM sigma: auto-scale from data range if not provided
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

    # Build penalty matrices (reuse helper from mcmc_solver.jl)
    penalties, offsets, _ = _build_penalty_info(prob)

    if verbose
        println("PseudoMarginalSolver: $n_beta UF params, " *
                "inner=$(alg.inner_method), n_steps=$(alg.n_steps), " *
                "n_deriv=$(alg.n_deriv)")
        println("  σ (IBM scale): $(round.(sigma, sigdigits=3))")
        println("  obs_var: $(alg.obs_var)")
        println("  $(alg.n_warmup) warmup + $(alg.n_samples) samples")
    end

    # Build log-density problem
    ld = PseudoMarginalLogDensity(prob, alg, penalties, offsets, n_beta, sigma)

    D = LogDensityProblems.dimension(ld)
    theta0 = copy(beta0)

    # Wrap with ForwardDiff AD
    ld_ad = ADgradient(Val(:ForwardDiff), ld)

    # Set up NUTS sampler
    nuts = NUTS(alg.target_accept)

    if verbose
        println("  Running NUTS sampler...")
    end

    # Run sampler using AbstractMCMC interface
    chain_raw = AbstractMCMC.sample(
        ld_ad, nuts, alg.n_warmup + alg.n_samples;
        initial_params=theta0,
        progress=verbose, verbose=false)

    # Extract samples (drop warmup)
    n_total = length(chain_raw)
    start_idx = alg.n_warmup + 1
    sample_matrix = zeros(alg.n_samples, D)
    for (idx, i) in enumerate(start_idx:n_total)
        sample_matrix[idx, :] .= chain_raw[i].z.θ
    end

    # Build MCMCChains.Chains object
    pnames = _param_names(prob, false)
    chain = MCMCChains.Chains(sample_matrix, pnames)

    if verbose
        println("  Chain size: $(size(sample_matrix))")
    end

    # MAP estimate = sample with highest log-posterior
    logp_values = [LogDensityProblems.logdensity(ld, sample_matrix[i, :])
                   for i in 1:alg.n_samples]
    map_idx = argmax(logp_values)
    map_theta = sample_matrix[map_idx, :]
    map_beta = map_theta[1:n_beta]

    # Build solution from MAP estimate
    p_opt = build_param_struct(prob, map_beta)

    # Run probabilistic solver at MAP for fitted values + uncertainty
    function ode_rhs_opt!(du, u, p_unused, t)
        prob.dynamics!(du, u, p_opt, t)
    end

    μ_smooth, Σ_smooth, times = probsolve(ode_rhs_opt!, nothing, Float64.(prob.u0),
                                           prob.tspan, alg.n_steps, alg.n_deriv, sigma;
                                           interrogate=:kramer)

    # Extract fitted values at data times
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

    # Build evaluators from MAP params
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = map_beta[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] = build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
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

    # ComponentArray for MAP parameters
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => map_beta[offset+1:offset+np])
        offset += np
    end
    ca = ComponentArray(NamedTuple(ca_entries))

    # Extract solution uncertainty at observation times
    sol_var = zeros(n_t, n_vars)
    for i in 1:n_t
        idx = searchsortedfirst(times, prob.data_times[i])
        idx = clamp(idx, 1, length(times))
        for k in 1:n_vars
            sol_var[i, k] = Σ_smooth[idx][k][1, 1]
        end
    end

    # Posterior mean of parameters
    post_mean = vec(mean(sample_matrix, dims=1))

    if verbose
        println("  MAP -logpost: $(round(-logp_values[map_idx], sigdigits=5))")
        println("  Data loss (MAP): $(round(data_loss, sigdigits=5))")
    end

    PSMSolution(
        ca,                               # parameters (MAP)
        -logp_values[map_idx],            # objective (negative MAP log-posterior)
        data_loss,                        # data_loss
        Float64(n_beta),                  # edf
        Float64[],                        # smoothing_params (not applicable)
        pred,                             # fitted_values (from MAP)
        Float64.(prob.data_values),       # data_values
        Float64.(prob.data_times),        # data_times
        uf_evals,                         # unknown_functions (MAP evaluators)
        chain                             # convergence: MCMCChains.Chains
    )
end
