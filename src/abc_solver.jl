# abc_solver.jl — Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC)
#
# Likelihood-free inference via population Monte Carlo.  Particles are
# iteratively resampled, perturbed, simulated, and accepted/rejected
# against a shrinking tolerance schedule driven by distance quantiles.

# ── helpers (module-private) ─────────────────────────────────────────

"""
    _abc_weighted_sample(weights) → Int

Sample an index from `weights` using inverse-CDF (no StatsBase dependency).
"""
function _abc_weighted_sample(weights::AbstractVector{Float64})
    cumw = cumsum(weights)
    r = rand() * cumw[end]
    for i in eachindex(cumw)
        r <= cumw[i] && return i
    end
    return length(weights)
end

"""
    _abc_quantile(v, q) → Float64

Compute the `q`-th quantile of the finite values in `v`.
"""
function _abc_quantile(v::AbstractVector{<:Real}, q::Float64)
    sv = sort(filter(isfinite, v))
    isempty(sv) && return Inf
    idx = clamp(ceil(Int, q * length(sv)), 1, length(sv))
    return Float64(sv[idx])
end

"""
    _abc_build_uf_dict(prob, beta) → Dict{Symbol, Any}

Build a dictionary mapping each approximator name to its evaluator
constructed from parameter vector `beta`.
"""
function _abc_build_uf_dict(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    uf = Dict{Symbol, Any}()
    for approx in prob.approximators
        uf[approx.name] = getfield(p, approx.name)
    end
    return uf
end

# ── main solver ──────────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::ABCSolver)

Fit a partially specified model using Approximate Bayesian Computation
with Sequential Monte Carlo (ABC-SMC). A likelihood-free method that
accepts parameter proposals when simulated data are sufficiently close
to observed data under a summary statistic.

# Algorithm
1. Draw `n_particles` parameter vectors from the prior (uniform around
   the initial guess).
2. For each ABC-SMC generation, propagate particles through a Gaussian
   kernel, simulate the model, and compute the summary statistic distance.
3. Accept particles within a shrinking tolerance schedule.
4. Return posterior-mean point estimates and the particle ensemble.

# References
- Toni et al. (2009), "Approximate Bayesian computation scheme for
  parameter inference and model selection", JRSS-B.

# Returns
`PSMSolution` with fitted parameters, trajectory, unknown functions,
and the particle ensemble in `sol.extras[:particles]`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::ABCSolver)
    _validate_problem(prob, "ABCSolver")
    n_p = n_total_params(prob)
    beta0 = build_initial_params(prob)
    N = alg.n_particles

    # ── summary statistic ────────────────────────────────────────────
    summary_fn = if alg.summary_fn === :auto
        (sim_data, obs_data) -> sqrt(sum((sim_data .- obs_data).^2) / length(obs_data))
    else
        alg.summary_fn
    end

    # ── prior over coefficients ──────────────────────────────────────
    # :smoothness (default): β ~ N(β₀, prior_scale²·(S̄ + εI)⁻¹), where S̄
    # is the block-diagonal roughness structure with each block scaled to
    # unit mean diagonal — the ABC analogue of the GMRF priors used by
    # MCMCSolver/VariationalSolver. :box: legacy Uniform(β₀ ± prior_scale).
    alg.prior in (:smoothness, :box) ||
        error("ABCSolver: prior must be :smoothness or :box (got :$(alg.prior))")
    prior_lo = beta0 .- alg.prior_scale
    prior_hi = beta0 .+ alg.prior_scale
    local prior_sample, prior_logpdf, in_support
    if alg.prior == :smoothness
        penalties, offsets, _ = _build_penalty_info(prob)
        # Unit ridge: smooth (penalty-null) directions get sd exactly
        # prior_scale, rough directions less — comparable overall spread to
        # the legacy box, with smoothness preference on top.
        P = Matrix{Float64}(1.0I, n_p, n_p)
        for (k, S) in enumerate(penalties)
            np_k = size(S, 1); off = offsets[k]
            sc = max(tr(S) / np_k, 1e-12)              # unit mean diagonal
            P[off+1:off+np_k, off+1:off+np_k] .+= S ./ sc
        end
        P ./= alg.prior_scale^2
        Lp = cholesky(Symmetric(P)).L
        prior_sample = () -> beta0 .+ (Lp' \ randn(n_p))
        prior_logpdf = β -> (d = β .- beta0; -0.5 * dot(d, P * d))
        in_support   = β -> true
    else
        prior_volume = prod(prior_hi .- prior_lo)
        prior_sample = () -> prior_lo .+ (prior_hi .- prior_lo) .* rand(n_p)
        prior_logpdf = β -> (prior_volume > 0 ? -log(prior_volume) : 0.0)
        in_support   = β -> !(any(β .< prior_lo) || any(β .> prior_hi))
    end

    # ── generation 0: sample from prior, accept all ──────────────────
    particles = Vector{Vector{Float64}}(undef, N)
    distances = Vector{Float64}(undef, N)
    weights   = fill(1.0 / N, N)

    for i in 1:N
        particles[i] = prior_sample()
        try
            pred = simulate(prob, particles[i])
            distances[i] = summary_fn(pred, prob.data_values)
        catch
            distances[i] = Inf
        end
    end

    epsilon = _abc_quantile(distances, 0.75)
    tolerance_history = Float64[epsilon]

    if alg.verbose
        n_finite = count(isfinite, distances)
        @printf("ABC-SMC gen 0: ε = %.4e  finite = %d/%d\n", epsilon, n_finite, N)
    end

    # ── SMC generations ──────────────────────────────────────────────
    for gen in 1:alg.n_generations
        new_particles = Vector{Vector{Float64}}(undef, N)
        new_distances = fill(Inf, N)
        accepted = 0

        # Kernel bandwidth: 2× weighted standard deviation per dimension
        particle_mat = hcat(particles...)          # n_p × N
        wmean = particle_mat * weights             # n_p
        wvar  = zeros(n_p)
        for j in 1:N
            diff = @view(particle_mat[:, j]) .- wmean
            wvar .+= weights[j] .* diff .* diff
        end
        kernel_std = 2.0 .* sqrt.(max.(wvar, 1e-12))

        accepted_slots = Int[]
        failed_slots   = Int[]
        for i in 1:N
            found = false
            for _attempt in 1:1000
                # (a) resample from previous generation
                j = _abc_weighted_sample(weights)
                # (b) perturb with Gaussian kernel
                theta_star = particles[j] .+ kernel_std .* randn(n_p)

                # check prior support (no-op for the Gaussian prior)
                in_support(theta_star) || continue

                # (c-d) simulate and compute distance
                local d::Float64
                try
                    pred = simulate(prob, theta_star)
                    d = summary_fn(pred, prob.data_values)
                catch
                    continue
                end

                # (e) accept / reject
                if d < epsilon
                    new_particles[i] = theta_star
                    new_distances[i] = d
                    accepted += 1
                    found = true
                    break
                end
            end
            push!(found ? accepted_slots : failed_slots, i)
        end

        # Slots that never produced an accepted draw within the attempt
        # budget must NOT carry forward the previous generation's particle
        # (it was accepted under the larger previous tolerance, so it is not
        # a draw from the current ABC posterior and would bias the target).
        # Instead fill each failed slot by resampling — with multiplicity —
        # from THIS generation's accepted particles, which are valid draws at
        # the current tolerance.  If nothing was accepted at all, the
        # tolerance is too aggressive for the budget: keep the previous
        # population and tolerance unchanged rather than corrupt it.
        if isempty(accepted_slots)
            if alg.verbose
                @printf("ABC-SMC gen %d: 0 accepted at ε = %.4e; holding population.\n",
                        gen, epsilon)
            end
            push!(tolerance_history, epsilon)
            continue
        end
        for i in failed_slots
            src = accepted_slots[rand(1:length(accepted_slots))]
            new_particles[i] = copy(new_particles[src])
            new_distances[i] = new_distances[src]
        end

        # ── (f) importance weights ───────────────────────────────────
        # w_i ∝ π(θ_i) / Σ_j w_j^{t-1} K(θ_i | θ_j^{t-1})
        new_weights = zeros(N)
        # Common constants in π(θ) cancel after normalization; shift by the
        # max log-density to avoid underflow of exp().
        lps = [in_support(new_particles[i]) ? prior_logpdf(new_particles[i]) :
               -Inf for i in 1:N]
        lp_max = maximum(lps)
        for i in 1:N
            isfinite(lps[i]) || continue
            kernel_sum = 0.0
            for j in 1:N
                diff = new_particles[i] .- particles[j]
                log_k = -0.5 * sum((diff ./ kernel_std).^2)
                kernel_sum += weights[j] * exp(log_k)
            end
            new_weights[i] = kernel_sum > 0.0 ?
                exp(lps[i] - lp_max) / kernel_sum : 0.0
        end

        wsum = sum(new_weights)
        if wsum > 0.0
            new_weights ./= wsum
        else
            new_weights .= 1.0 / N
        end

        # ── update state for next generation ─────────────────────────
        particles = new_particles
        distances = new_distances
        weights   = new_weights

        # (3) shrink tolerance
        epsilon = _abc_quantile(distances, alg.quantile_eps)
        push!(tolerance_history, epsilon)

        if alg.verbose
            @printf("ABC-SMC gen %d: ε = %.4e  accepted = %d/%d\n",
                    gen, epsilon, accepted, N)
        end
    end

    # ── assemble results ─────────────────────────────────────────────
    best_idx  = argmin(distances)
    best_beta = particles[best_idx]

    # Weighted posterior mean
    mean_beta = zeros(n_p)
    for i in 1:N
        mean_beta .+= weights[i] .* particles[i]
    end

    # Fitted values and unknown functions from the SAME point estimate the
    # solution reports (`params` = weighted posterior mean); mixing the
    # posterior mean with best-particle functions returned inconsistent
    # summaries.
    pred = try
        simulate(prob, mean_beta)
    catch e
        _is_program_error(e) && rethrow()
        fill(NaN, size(prob.data_values))
    end

    # Sum of squares, matching the data_loss convention of the other solvers
    data_loss = sum((pred .- prob.data_values).^2)

    uf_evals = _abc_build_uf_dict(prob, mean_beta)

    # ComponentArray of weighted posterior mean
    ca_entries = Pair{Symbol,Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => mean_beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    PSMSolution(
        params,
        Float64(distances[best_idx]),
        Float64(data_loss),
        Float64(n_p),
        Float64[],
        Float64.(pred),
        Float64.(prob.data_values),
        Float64.(prob.data_times),
        uf_evals,
        (method            = :abc_smc,
         particles         = particles,
         distances         = distances,
         weights           = weights,
         tolerance_history = tolerance_history,
         best_idx          = best_idx,
         n_generations     = alg.n_generations),
    )
end
