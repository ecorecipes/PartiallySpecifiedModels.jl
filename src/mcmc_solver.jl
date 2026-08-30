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
    # The σ nuisance parameter exists only for Gaussian observations;
    # other families carry their own (fixed) dispersion.
    if ld.obs_sigma === nothing && ld.prob.likelihood isa Gaussian
        d += 1
    end
    if ld.sample_smoothing; d += ld.n_smooths; end
    d
end

function LogDensityProblems.logdensity(ld::PSMLogDensity, theta)
    try
        return _psm_logdensity(ld, theta)
    catch e
        # A numerical failure inside the user dynamics should reject the
        # proposal, not abort the whole chain.
        _is_program_error(e) && rethrow()
        return eltype(theta)(-1e20)
    end
end

function _psm_logdensity(ld::PSMLogDensity, theta)
    prob = ld.prob
    T = eltype(theta)

    # Parse theta: [beta..., (log_sigma)?, (log_lambda_1, ..., log_lambda_m)?]
    idx = ld.n_params
    beta = theta[1:idx]

    fam = prob.likelihood
    if fam isa Gaussian
        if ld.obs_sigma === nothing
            idx += 1
            log_sigma = theta[idx]
            sigma2 = exp(2 * log_sigma)
        else
            log_sigma = nothing
            sigma2 = T(ld.obs_sigma^2)
        end
    else
        # Non-Gaussian families have no σ nuisance parameter (Poisson has
        # none; NegBin θ and TruncatedNormal σ are fixed in the family
        # object). σ² = 1 decouples the penalty prior from the data scale.
        log_sigma = nothing
        sigma2 = one(T)
    end

    if ld.sample_smoothing && ld.n_smooths > 0
        log_lambdas = theta[idx+1:idx+ld.n_smooths]
    else
        log_lambdas = nothing
    end

    # --- Log-likelihood: simulate and compare to data ---
    #
    # `_variational_simulate` (variational_solver.jl) is the shared simulator
    # that already covers all three problem classes — discrete map, ODE and
    # DDE. Using it here closes a sibling-parity gap: this block previously
    # branched on `prob.discrete` vs ODE only and built a 4-ARGUMENT
    # `ODEFunction` closure, so a PSM DDE (whose dynamics have the 5-argument
    # signature `f!(du, u, h, p, t)`) raised a raw `MethodError` from inside
    # NUTS after ~9 s of setup, while `VariationalSolver` — which dispatches
    # on `!isempty(prob.delays)` — handled the same problem.
    #
    # The ODE branch is byte-for-byte the code deleted here (the same
    # `ODEFunction{true, FullSpecialize}` wrapper, the same
    # `abstol=reltol=1e-7`, `maxiters=10000`, the same retcode test), and the
    # DDE branch goes through `adam_solve_dde`, which uses those same
    # tolerances. Measured: on this fixture the ODE path's reported
    # `r(5.0)`, `objective`, `data_loss` and `fitted_values` are unchanged
    # to the last printed digit.
    pred = _variational_simulate(prob, beta)
    pred === nothing && return T(-1e20)

    # Observation log-likelihood, dispatched through prob.likelihood
    # (accumulated as a scalar to preserve Dual type). Weight-zero and NaN
    # entries are masked/missing points and are skipped for every family.
    n_t = size(prob.data_values, 1)
    n_obs = size(prob.data_values, 2)
    ll = zero(T)
    n_eff = 0
    for j in 1:n_obs
        for i in 1:n_t
            w = prob.data_weights[i, j]
            y = prob.data_values[i, j]
            _usable(y, w) || continue
            n_eff += 1
            # `_variational_simulate` returns an (n_t × n_obs) matrix already
            # projected onto the observation mapping and zero-filled past the
            # end of a short solve, so the former `prob.discrete` branch here
            # (and its `sol[obs_to_state[j], i]` indexing) is gone.
            pred_ij = pred[i, j]
            if fam isa Gaussian
                ll -= T(0.5) * w * (pred_ij - T(y))^2 / sigma2
            else
                ll += T(w) * loglik_pointwise(fam, y, pred_ij)
            end
        end
    end
    if fam isa Gaussian
        # Normalizer counts only weight-carrying finite observations:
        # counting masked points in n_data biases the estimated σ² low.
        ll -= T(0.5) * n_eff * log(T(2π) * sigma2)
    end

    # --- Log-prior: penalty (Gaussian GMRF prior) + broad prior ---
    # The roughness penalty is the prior precision λS/σ²; the prior is
    # p(β|λ,σ²) ∝ (λ/σ²)^{rank(S)/2} exp(−(λ/2σ²) βᵀSβ).  The σ² coupling
    # (data and penalty share the scale) and the (λ/σ²) normalizer are both
    # required for the hierarchical λ/σ² posterior to be valid.
    lp = zero(T)
    for (k, S) in enumerate(ld.penalty_matrices)
        np = size(S, 1)
        off = ld.param_offsets[k]
        beta_k = beta[off+1:off+np]
        lambda_k = if log_lambdas !== nothing
            exp(log_lambdas[k])
        else
            T(1.0) / T(ld.prior_scale)
        end
        rk = _rank_penalty(S)
        lp += T(0.5) * rk * (log(lambda_k) - log(sigma2)) -
              T(0.5) * lambda_k / sigma2 * dot(beta_k, S * beta_k)
    end

    # Broad Gaussian prior on all params
    lp -= T(0.5) * sum(beta .^ 2) / T(100.0 * ld.prior_scale)

    # Sampling s = log σ with a FLAT prior on σ: the +log σ term is the
    # log-transform Jacobian |dσ/ds| = σ. (A Jeffreys prior p(σ) ∝ 1/σ
    # would contribute no term in log-σ space.)
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
Sampling is by the No-U-Turn Sampler (NUTS) with ForwardDiff gradients;
NUTS is the only sampler — `MCMCSolver` has no sampler-selection field,
and only `target_accept` tunes the adaptation. This provides full
posterior distributions over the unknown-function parameters.

The observation model follows `prob.likelihood`. Gaussian data get a
sampled (or fixed, via `obs_sigma`) noise parameter σ; Poisson,
NegativeBinomial, TruncatedNormal, and CustomLikelihood use the family's
pointwise log-likelihood with `data_weights` multiplied in and sample no
σ nuisance (dispersion parameters such as NegBin's θ are fixed in the
family object). Passing `obs_sigma` with a non-Gaussian family errors.

# Algorithm
1. Initialise parameters and define the log-posterior (log-likelihood +
   optional smoothing-penalty prior).
2. Run NUTS (`AdvancedHMC.jl`) for `n_samples` iterations with
   `n_warmup` adaptation/burn-in steps, over `n_chains` chains.
3. Take the MAP draw (the pooled sample maximising the log-posterior)
   as the point estimate and reconstruct trajectories from it.
4. Return the full chain alongside that point estimate.

# References
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC.

# Returns
`PSMSolution` whose `parameters` are the MAP draw, with trajectory and
unknown functions built from it. `sol.convergence` **is** the full
`MCMCChains.Chains` object (iterations × parameters × chains) — there is
no `extras` field on `PSMSolution`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MCMCSolver)
    _validate_problem(prob, "MCMCSolver")
    if !(prob.likelihood isa Gaussian) && alg.obs_sigma !== nothing
        error("MCMCSolver: obs_sigma is the Gaussian observation-noise " *
              "standard deviation, but prob.likelihood is " *
              "$(typeof(prob.likelihood)), which has no σ parameter " *
              "(Poisson has none; NegativeBinomial θ and TruncatedNormal σ " *
              "are fixed in the family object).")
    end
    verbose = alg.verbose

    # Initialize parameters
    beta0 = Float64[]
    for approx in prob.approximators
        if approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta0, neural_init_params(approx, rng))
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

    # σ is sampled only for Gaussian observations with unspecified obs_sigma
    estimate_sigma = alg.obs_sigma === nothing && prob.likelihood isa Gaussian
    D = LogDensityProblems.dimension(ld)

    # Initial point: beta0 + optional log_sigma + optional log_lambda
    theta0 = copy(beta0)
    if estimate_sigma
        # Usable cells only. Over ALL cells a single masked/NaN datum made
        # std() NaN → log(max(NaN, 0.01)) = NaN → NaN in theta0 → the whole
        # chain NaN, reported as objective=NaN, data_loss=NaN, no error.
        usable_vals = [prob.data_values[i, j]
                       for j in 1:size(prob.data_values, 2)
                       for i in 1:size(prob.data_values, 1)
                       if usable_cell(prob, i, j)]
        isempty(usable_vals) &&
            error("MCMCSolver: every observation is masked (data_weights " *
                  "zero) or NaN, so the observation noise σ cannot be " *
                  "initialized and there is nothing to fit.")
        sig_init = length(usable_vals) >= 2 ? std(usable_vals) * 0.1 : 0.0
        isfinite(sig_init) || (sig_init = 0.0)
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

    # Solver-owned RNG stream (AGM/BNG convention): seeded when rng_seed is
    # given, otherwise randomly seeded — never the global RNG.
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    # Run sampler via AbstractMCMC. n_adapts must equal n_warmup: without it
    # AdvancedHMC adapts for only min(N/10, 1000) steps, so the retained
    # draws could include iterations taken while step size and mass matrix
    # were still adapting (not valid posterior samples).
    # Suppress AdvancedHMC "Verbosity toggle: max_iters" warnings
    n_ch = max(alg.n_chains, 1)
    all_samples = zeros(alg.n_samples, D, n_ch)
    for c in 1:n_ch
        # Overdisperse the later chains' starts slightly so R̂-style
        # diagnostics on the returned Chains object are meaningful.
        θ0c = c == 1 ? theta0 : theta0 .+ 0.1 .* randn(rng, length(theta0))
        chain_raw = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            AbstractMCMC.sample(
                rng, ld_ad, nuts, alg.n_warmup + alg.n_samples;
                n_adapts=alg.n_warmup,
                initial_params=θ0c,
                progress=verbose, verbose=false)
        end
        # Extract samples (drop warmup)
        start_idx = alg.n_warmup + 1
        for (idx, i) in enumerate(start_idx:length(chain_raw))
            all_samples[idx, :, c] .= chain_raw[i].z.θ
        end
    end
    # Pool chains for point estimates
    sample_matrix = reshape(permutedims(all_samples, (1, 3, 2)),
                            alg.n_samples * n_ch, D)

    # Build MCMCChains.Chains object (iterations × params × chains)
    pnames = _param_names(prob, estimate_sigma;
                          sample_smoothing=alg.sample_smoothing)
    chain = MCMCChains.Chains(all_samples, pnames)

    if verbose
        println("  Chain size: $(size(all_samples))")
    end

    # MAP estimate = pooled sample with highest log-posterior
    logp_values = [LogDensityProblems.logdensity(ld, sample_matrix[i, :])
                   for i in 1:size(sample_matrix, 1)]
    map_idx = argmax(logp_values)
    map_theta = sample_matrix[map_idx, :]
    map_beta = map_theta[1:n_beta]

    # Build solution from MAP estimate
    n_t = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    # Same shared simulator as the log-density above, for the same reason:
    # the ODE-only branch this replaces raised a raw `MethodError` on a DDE.
    # `build_autodiff_param_struct` and `build_param_struct` have identical
    # bodies, so for a Float64 `map_beta` this builds the same parameter
    # struct the deleted block used.
    #
    # BEHAVIOUR CHANGE, on the FAILED-SOLVE path only. The block this
    # replaces had no retcode test: a solve that stopped early at the MAP
    # left `pred` partially filled and reported it as `fitted_values`.
    # `_variational_simulate` returns `nothing` on any non-Success retcode,
    # so that case now warns and reports zeros instead. On Success — which
    # is every in-tree fixture, and why the ODE path is bitwise unchanged —
    # the two are identical; "bitwise identical" does NOT cover this path.
    pred_map = _variational_simulate(prob, Float64.(map_beta))
    pred = if pred_map === nothing
        @warn "MCMCSolver: simulation at the MAP parameters failed " *
              "(non-Success retcode); fitted_values are reported as zeros " *
              "and data_loss does not measure the model fit."
        zeros(n_t, n_obs)
    else
        Float64.(pred_map)
    end

    # Masked/NaN cells are skipped (0 * NaN = NaN would make data_loss NaN).
    data_loss = weighted_data_loss(prob, pred)

    # Build evaluators from MAP params
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = map_beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
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

    # Layout is [β; log σ; log λ...]: with sample_smoothing the LAST entry
    # is a smoothing parameter, so index log σ by position, not by end.
    # Non-Gaussian families sample no σ, so there is nothing to report.
    sigma_map = estimate_sigma ? exp(map_theta[n_beta + 1]) : alg.obs_sigma
    smoothing = sigma_map === nothing ? Float64[] : [sigma_map]

    PSMSolution(ca, -logp_values[map_idx], data_loss, Float64(n_beta),
                smoothing, pred, prob.data_values, collect(prob.data_times),
                uf_evals, chain)
end
