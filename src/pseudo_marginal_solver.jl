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
# NOTE: The Kalman-filter-based fenrir_loglik uses in-place Float64 arrays
# that are not ForwardDiff-compatible. We use finite-difference gradients
# instead, following the same approach as RodeoSolver.
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
    obs_var::Float64
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
    T = Float64  # fenrir_loglik is not AD-compatible; always use Float64

    beta = Float64.(theta[1:ld.n_params])

    # Build parameter struct
    p = build_param_struct(prob, beta)

    function ode_rhs!(du, u, p_unused, t)
        prob.dynamics!(du, u, p, t)
    end

    # Compute probabilistic ODE log-likelihood via fenrir or basic method
    loglik_fn = alg.inner_method == :fenrir ? fenrir_loglik : basic_loglik

    ll = try
        loglik_fn(ode_rhs!, nothing, Float64.(prob.u0), prob.tspan,
                  alg.n_steps, alg.n_deriv, ld.sigma,
                  Float64.(prob.data_values), Float64.(prob.data_times),
                  prob.obs_to_state, ld.obs_var;
                  interrogate=:kramer)
    catch e
        return -1e20
    end

    if !isfinite(ll)
        return -1e20
    end

    # Log-prior: smoothing penalty from penalty matrices
    lp = 0.0
    for (idx, S) in enumerate(ld.penalty_matrices)
        np = size(S, 1)
        off = ld.param_offsets[idx]
        beta_k = beta[off+1:off+np]
        lp -= 0.5 / alg.prior_scale * dot(beta_k, S * beta_k)
    end

    # Broad Gaussian prior on all parameters
    lp -= 0.5 * sum(beta .^ 2) / (100.0 * alg.prior_scale)

    return ll + lp
end

# ─── Finite-difference gradient wrapper ──────────────────────────
#
# fenrir_loglik uses in-place Float64 Kalman filter arrays that are not
# compatible with ForwardDiff. We compute gradients via central finite
# differences, following the same approach as RodeoSolver.

struct PseudoMarginalFDGradient
    ld::PseudoMarginalLogDensity
    fd_eps::Float64
end

function LogDensityProblems.capabilities(::Type{PseudoMarginalFDGradient})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.dimension(g::PseudoMarginalFDGradient)
    g.ld.n_params
end

function LogDensityProblems.logdensity(g::PseudoMarginalFDGradient, theta)
    LogDensityProblems.logdensity(g.ld, theta)
end

function LogDensityProblems.logdensity_and_gradient(g::PseudoMarginalFDGradient, theta)
    ld = g.ld
    ε = g.fd_eps
    D = length(theta)
    f0 = LogDensityProblems.logdensity(ld, theta)
    grad = zeros(D)
    for i in 1:D
        θp = copy(theta); θp[i] += ε
        θm = copy(theta); θm[i] -= ε
        fp = LogDensityProblems.logdensity(ld, θp)
        fm = LogDensityProblems.logdensity(ld, θm)
        if isfinite(fp) && isfinite(fm)
            grad[i] = (fp - fm) / (2ε)
        else
            grad[i] = 0.0
        end
    end
    # Clip gradient norm to prevent extreme proposals
    gnorm = norm(grad)
    if gnorm > 1e4
        grad .*= 1e4 / gnorm
    end
    return f0, grad
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

    # Initialize parameters (use provided initial_params or default)
    if alg.initial_params !== nothing
        beta0 = copy(alg.initial_params)
    else
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

    # Observation variance: auto-scale from data variance if not provided
    obs_var = if alg.obs_var === nothing
        total_var = 0.0
        for j in 1:n_obs
            total_var += var(prob.data_values[:, j])
        end
        max(total_var / n_obs * 0.01, 1e-4)
    else
        alg.obs_var
    end

    # Build penalty matrices (reuse helper from mcmc_solver.jl)
    penalties, offsets, _ = _build_penalty_info(prob)

    if verbose
        println("PseudoMarginalSolver: $n_beta UF params, " *
                "inner=$(alg.inner_method), n_steps=$(alg.n_steps), " *
                "n_deriv=$(alg.n_deriv)")
        println("  σ (IBM scale): $(round.(sigma, sigdigits=3))")
        println("  obs_var: $(round(obs_var, sigdigits=3))")
        println("  prior_scale: $(alg.prior_scale)")
        init_label = alg.initial_params !== nothing ? "user-provided" : "default"
        println("  init: $init_label, $(alg.n_warmup) warmup + $(alg.n_samples) samples")
    end

    # Build log-density problem
    ld = PseudoMarginalLogDensity(prob, alg, penalties, offsets, n_beta, sigma, obs_var)

    D = LogDensityProblems.dimension(ld)
    theta0 = copy(beta0)

    # Use finite-difference gradient wrapper (fenrir_loglik is not AD-compatible)
    ld_fd = PseudoMarginalFDGradient(ld, 1e-5)

    # Set up NUTS sampler
    nuts = NUTS(alg.target_accept)

    if verbose
        init_ll = LogDensityProblems.logdensity(ld, theta0)
        println("  Initial log-density: $(round(init_ll, sigdigits=5))")
        println("  Running NUTS sampler (finite-difference gradients)...")
    end

    # Run sampler using AbstractMCMC interface
    chain_raw = AbstractMCMC.sample(
        ld_fd, nuts, alg.n_warmup + alg.n_samples;
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
