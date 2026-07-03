# MCMC/HMC solver for fully Bayesian inference
#
# Uses AdvancedHMC.jl (NUTS sampler) with LogDensityProblems.jl interface.
# Returns PSMSolution with MCMCChains.Chains in the convergence field.

using AdvancedHMC
using LogDensityProblems
using LogDensityProblemsAD
using MCMCChains
import AbstractMCMC

# ─── Log-density problem ─────────────────────────────────────────

struct PSMLogDensity
    prob::PSMProblem
    penalty_matrices::Vector{Matrix{Float64}}
    param_offsets::Vector{Int}
    n_params::Int
    prior_scale::Float64
    obs_sigma::Union{Nothing, Float64}  # nothing = estimate; Float64 = fixed
    sample_smoothing::Bool
    n_smooths::Int  # number of smooth terms (penalty matrices)
    log_lambda_init::Vector{Float64}  # initial log(λ) for hyperprior center
end

function LogDensityProblems.capabilities(::Type{PSMLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::PSMLogDensity)
    d = ld.n_params
    if ld.obs_sigma === nothing && ld.prob.likelihood isa Gaussian; d += 1; end
    if ld.sample_smoothing; d += ld.n_smooths; end
    d
end

function LogDensityProblems.logdensity(ld::PSMLogDensity, theta)
    prob = ld.prob
    T = eltype(theta)

    # Parse theta: [beta..., (log_sigma)?, (log_lambda_1, ..., log_lambda_m)?]
    idx = ld.n_params
    beta = theta[1:idx]

    estimate_sigma = ld.obs_sigma === nothing && prob.likelihood isa Gaussian

    if estimate_sigma
        idx += 1
        log_sigma = theta[idx]
        sigma2 = exp(2 * log_sigma)
    else
        log_sigma = nothing
        sigma2 = ld.obs_sigma === nothing ? nothing : T(ld.obs_sigma^2)
    end

    if ld.sample_smoothing && ld.n_smooths > 0
        log_lambdas = theta[idx+1:idx+ld.n_smooths]
    else
        log_lambdas = nothing
    end

    # --- Log-likelihood: simulate and compare to data ---
    p = build_autodiff_param_struct(prob, beta)

    if prob.discrete
        pred = adam_simulate_discrete(prob, p)
    else
        u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
        u0_T = T.(u0)
        ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
            (du, u, params, t) -> prob.dynamics!(du, u, params, t))
        ode_prob = ODEProblem(ode_fn, u0_T, prob.tspan, p)
        sol = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                   saveat=prob.data_times,
                                   abstol=1e-7, reltol=1e-7,
                                   maxiters=10000)

        if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
            return T(-1e20)
        end

        n_t = length(prob.data_times)
        n_obs = size(prob.data_values, 2)
    end

    n_t = size(prob.data_values, 1)
    n_obs = size(prob.data_values, 2)
    pred_mat = if prob.discrete
        pred
    else
        pred_tmp = zeros(T, n_t, n_obs)
        for j in 1:n_obs, i in 1:n_t
            pred_tmp[i, j] = i <= length(sol.t) ? sol[prob.obs_to_state[j], i] : T(0)
        end
        pred_tmp
    end
    ll = observation_loglikelihood(prob.likelihood,
                                   T.(prob.data_values),
                                   pred_mat,
                                   T.(prob.data_weights);
                                   sigma2=sigma2)

    # --- Log-prior: penalty matrices + broad prior ---
    lp = zero(T)
    for (k, S) in enumerate(ld.penalty_matrices)
        np = size(S, 1)
        off = ld.param_offsets[k]
        beta_k = beta[off+1:off+np]
        # Use sampled λ if available, otherwise fixed prior_scale
        lambda_k = if log_lambdas !== nothing
            exp(log_lambdas[k])
        else
            T(1.0) / T(ld.prior_scale)
        end
        lp -= T(0.5) * lambda_k * dot(beta_k, S * beta_k)
    end

    # Broad Gaussian prior on all params
    lp -= T(0.5) * sum(beta .^ 2) / T(100.0 * ld.prior_scale)

    # Jeffrey's prior on sigma via log transform
    if log_sigma !== nothing
        lp += log_sigma
    end

    # Weakly informative hyperprior on log(λ): N(log(λ_init), 2²)
    if log_lambdas !== nothing
        for k in 1:ld.n_smooths
            lp -= T(0.5) * (log_lambdas[k] - T(ld.log_lambda_init[k]))^2 / T(4.0)
        end
    end

    return ll + lp
end

# ─── Build penalty matrix info ───────────────────────────────────

function _build_penalty_info(prob::PSMProblem)
    penalties = Matrix{Float64}[]
    offsets = Int[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        S = penalty_matrix(approx)
        if S !== nothing
            push!(penalties, S)
            push!(offsets, offset)
        end
        offset += np
    end
    penalties, offsets, offset
end

# ─── Parameter names for MCMCChains ──────────────────────────────

function _param_names(prob::PSMProblem, estimate_sigma::Bool;
                     sample_smoothing::Bool=false)
    names = String[]
    for approx in prob.approximators
        np = nparams(approx)
        sym = string(approx.name)
        for i in 1:np
            push!(names, "$(sym)[$i]")
        end
    end
    if estimate_sigma
        push!(names, "log_σ")
    end
    if sample_smoothing
        for (k, approx) in enumerate(prob.approximators)
            S = penalty_matrix(approx)
            if S !== nothing
                push!(names, "log_λ[$(approx.name)]")
            end
        end
    end
    return names
end

# ─── Solve method ────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::MCMCSolver)

Fit a partially specified model using Markov Chain Monte Carlo sampling.
Supports Hamiltonian Monte Carlo (HMC), the No-U-Turn Sampler (NUTS),
and Metropolis–Hastings (MH), providing full posterior distributions
over the unknown-function parameters.

# Algorithm
1. Initialise parameters and define the log-posterior (log-likelihood +
   optional smoothing-penalty prior).
2. Run the selected MCMC sampler (`AdvancedMH.jl`) for `n_samples`
   iterations with `n_warmup` adaptation/burn-in steps.
3. Compute posterior mean parameters and reconstruct trajectories.
4. Return the full chain alongside point estimates.

# References
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC.

# Returns
`PSMSolution` with fitted parameters, trajectory, unknown functions,
and the full MCMC chain in `sol.convergence`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MCMCSolver)
    _validate_problem(prob, "MCMCSolver")
    verbose = alg.verbose

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

    # Build penalty matrices
    penalties, offsets, _ = _build_penalty_info(prob)

    # Build log-density problem
    n_smooths = length(penalties)
    log_lambda_init = if alg.sample_smoothing && n_smooths > 0
        [log(1.0 / alg.prior_scale) for _ in 1:n_smooths]
    else
        Float64[]
    end
    ld = PSMLogDensity(prob, penalties, offsets, n_beta, alg.prior_scale,
                       alg.obs_sigma, alg.sample_smoothing, n_smooths,
                       log_lambda_init)

    estimate_sigma = alg.obs_sigma === nothing && prob.likelihood isa Gaussian
    D = LogDensityProblems.dimension(ld)

    # Initial point: beta0 + optional log_sigma + optional log_lambda
    theta0 = copy(beta0)
    if estimate_sigma
        sig_init = std(prob.data_values) * 0.1
        push!(theta0, log(max(sig_init, 0.01)))
    end
    if alg.sample_smoothing && n_smooths > 0
        append!(theta0, log_lambda_init)
    end

    if verbose
        println("MCMCSolver: $n_beta UF params" *
                (estimate_sigma ? " + 1 noise param" : "") *
                (alg.sample_smoothing ? " + $n_smooths smoothing params" : "") *
                ", $(alg.n_warmup) warmup + $(alg.n_samples) samples")
    end

    # Wrap with ForwardDiff AD
    ld_ad = ADgradient(Val(:ForwardDiff), ld)

    # Set up NUTS sampler with target acceptance rate
    nuts = NUTS(alg.target_accept)

    # Run sampler using AbstractMCMC interface (handles adaptation internally)
    # Suppress AdvancedHMC "Verbosity toggle: max_iters" warnings
    chain_raw = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        AbstractMCMC.sample(
            ld_ad, nuts, alg.n_warmup + alg.n_samples;
            initial_params=theta0,
            progress=verbose, verbose=false)
    end

    # Extract samples as matrix (drop warmup)
    n_total = length(chain_raw)
    start_idx = alg.n_warmup + 1
    sample_matrix = zeros(alg.n_samples, D)
    for (idx, i) in enumerate(start_idx:n_total)
        sample_matrix[idx, :] .= chain_raw[i].z.θ
    end

    # Build MCMCChains.Chains object
    pnames = _param_names(prob, estimate_sigma;
                          sample_smoothing=alg.sample_smoothing)
    chain = MCMCChains.Chains(sample_matrix, pnames)

    if verbose
        println("  Chain size: $(size(sample_matrix))")
    end

    # MAP estimate = sample with highest log-posterior
    logp_values = [LogDensityProblems.logdensity(ld, sample_matrix[i, :]) for i in 1:alg.n_samples]
    map_idx = argmax(logp_values)
    map_theta = sample_matrix[map_idx, :]
    map_beta = map_theta[1:n_beta]

    # Build solution from MAP estimate
    p_opt = build_param_struct(prob, map_beta)

    n_t = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    if prob.discrete
        p_ad = build_autodiff_param_struct(prob, map_beta)
        pred = Float64.(adam_simulate_discrete(prob, p_ad))
    else
        u0 = prob.u0 isa Function ? prob.u0(p_opt) : prob.u0
        ode_prob = ODEProblem((du, u, params, t) -> prob.dynamics!(du, u, p_opt, t),
                              Float64.(u0), prob.tspan)
        sol_ode = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                       saveat=prob.data_times,
                                       abstol=1e-7, reltol=1e-7,
                                       maxiters=10000)
        pred = zeros(n_t, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            for i in 1:min(n_t, length(sol_ode.t))
                pred[i, j] = sol_ode[sk, i]
            end
        end
    end

    data_loss = sum(prob.data_weights .* (prob.data_values .- pred) .^ 2)

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

    # ComponentArray for MAP parameters
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => map_beta[offset+1:offset+np])
        offset += np
    end
    ca = ComponentArray(NamedTuple(ca_entries))

    sigma_map = estimate_sigma ? exp(map_theta[end]) : alg.obs_sigma
    smoothing = sigma_map === nothing ? Float64[] : [sigma_map]

    PSMSolution(ca, -logp_values[map_idx], data_loss, Float64(n_beta),
                smoothing, pred, prob.data_values, collect(prob.data_times),
                uf_evals, chain)
end
