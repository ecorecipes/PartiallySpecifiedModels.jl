# abc_solver.jl — Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC)
#
# Likelihood-free inference via population Monte Carlo.  Particles are
# iteratively resampled, perturbed, simulated, and accepted/rejected
# against a shrinking tolerance schedule driven by distance quantiles.

# ── helpers (module-private) ─────────────────────────────────────────

"""
    _abc_weighted_sample(rng, weights) → Int

Sample an index from `weights` using inverse-CDF (no StatsBase dependency).
"""
function _abc_weighted_sample(rng::Random.AbstractRNG,
                              weights::AbstractVector{Float64})
    cumw = cumsum(weights)
    r = rand(rng) * cumw[end]
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
    _abc_importance_weights(new_particles, prev_particles, prev_weights,
                            kernel_std, lps; gen=0) → (weights, ess)

Compute normalized ABC-SMC importance weights

    w_i ∝ π(θ_i) / Σ_j w_j^{t-1} K(θ_i | θ_j^{t-1})

with the proposal-mixture denominator evaluated fully in log space:

    log_denom_i = logsumexp_j( log w_j^{t-1} + log K(θ_i | θ_j^{t-1}) ).

A particle far from every previous-generation particle has a tiny
proposal density; in linear arithmetic `Σ wⱼ exp(log_k)` underflows to
exactly 0, and the old code then set the particle's weight to 0 — the
exact opposite of the correct limit (weight ∝ π/denominator should be
LARGE when the denominator is small).  `lps[i]` is the (unnormalized)
log prior density of `new_particles[i]`, `-Inf` if outside support.

Also returns the effective sample size `ESS = 1/Σ wᵢ²` of the
normalized weights and warns (once per call, i.e. once per generation)
when ESS < N/2, signalling weight degeneracy.
"""
function _abc_importance_weights(new_particles::AbstractVector,
                                 prev_particles::AbstractVector,
                                 prev_weights::AbstractVector{Float64},
                                 kernel_std::AbstractVector{Float64},
                                 lps::AbstractVector{Float64};
                                 gen::Int=0)
    N = length(new_particles)
    M = length(prev_particles)
    log_wprev = log.(prev_weights)          # zero weights → -Inf (dropped below)
    new_log_w = fill(-Inf, N)
    log_terms = Vector{Float64}(undef, M)
    for i in 1:N
        isfinite(lps[i]) || continue
        m = -Inf
        for j in 1:M
            diff = new_particles[i] .- prev_particles[j]
            lt = log_wprev[j] - 0.5 * sum(abs2, diff ./ kernel_std)
            log_terms[j] = lt
            lt > m && (m = lt)
        end
        # prev_weights sum to 1, so at least one term is finite and m > -Inf
        log_denom = m + log(sum(t -> exp(t - m), log_terms))
        new_log_w[i] = lps[i] - log_denom
    end

    lw_max = maximum(new_log_w)
    weights = if isfinite(lw_max)
        w = [isfinite(lw) ? exp(lw - lw_max) : 0.0 for lw in new_log_w]
        w ./ sum(w)
    else
        fill(1.0 / N, N)
    end

    ess = 1.0 / sum(abs2, weights)
    if ess < N / 2
        @warn "ABC-SMC gen $gen: effective sample size $(round(ess, digits=1)) " *
              "< N/2 = $(N / 2); importance weights are highly concentrated"
    end
    return weights, ess
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
1. Draw `n_particles` parameter vectors from the prior. The default
   `prior=:smoothness` is a GMRF roughness prior
   `β ~ N(β₀, prior_scale²·(S̄ + εI)⁻¹)`, where `S̄` is the block-diagonal
   penalty structure with each block scaled to unit mean diagonal —
   penalty-null (smooth) directions get standard deviation exactly
   `prior_scale`, rougher directions less. `prior=:box` is the legacy
   `Uniform(β₀ ± prior_scale)`.
2. For each ABC-SMC generation, propagate particles through a Gaussian
   perturbation kernel, simulate the model, and compute the summary
   statistic distance. The kernel bandwidth is **2× the weighted standard
   deviation** of the current population, per dimension — note this is
   twice the *standard deviation*, deliberately wider than the textbook
   twice-the-*variance* (i.e. √2× std) choice.
3. Accept particles within a shrinking tolerance schedule.
4. Return posterior-mean point estimates and the particle ensemble.

# References
- Toni et al. (2009), "Approximate Bayesian computation scheme for
  parameter inference and model selection", JRSS-B.

# Returns
`PSMSolution` whose `parameters` are the weighted posterior mean, with
trajectory and unknown functions built from it. There is no `extras`
field on `PSMSolution`: the ensemble is in `sol.convergence`, a
`NamedTuple` carrying `method`, `particles`, `distances`, `weights`,
`tolerance_history`, `best_idx` and `n_generations`. `sol.objective` is
the best particle's summary-statistic distance.
"""
function SciMLBase.solve(prob::PSMProblem, alg::ABCSolver)
    _validate_problem(prob, "ABCSolver")
    n_p = n_total_params(prob)
    beta0 = build_initial_params(prob)
    N = alg.n_particles

    # Solver-owned RNG stream (AGM/BNG convention): seeded when rng_seed is
    # given, otherwise randomly seeded — never the global RNG.
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    # ── summary statistic ────────────────────────────────────────────
    # :auto is a weighted RMS distance over the USABLE cells only: cells
    # with zero weight or a non-finite observation are skipped, matching
    # the masking convention used across the package.  (Previously one
    # NaN observation made every particle's distance NaN; NaN is never
    # `< ε`, so nothing was ever accepted and the solver returned the
    # untouched prior population.)  Normalizing by the total weight of
    # the cells used keeps the scale comparable to the old
    # all-cells RMS when weights are unit and data complete.
    summary_fn = if alg.summary_fn === :auto
        let W = prob.data_weights
            (sim_data, obs_data) -> begin
                ss = 0.0
                wsum = 0.0
                @inbounds for k in eachindex(obs_data)
                    w = W[k]
                    y = obs_data[k]
                    (w > 0 && isfinite(y)) || continue
                    ss += w * (sim_data[k] - y)^2
                    wsum += w
                end
                wsum > 0 ? sqrt(ss / wsum) : Inf
            end
        end
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
        prior_sample = () -> beta0 .+ (Lp' \ randn(rng, n_p))
        prior_logpdf = β -> (d = β .- beta0; -0.5 * dot(d, P * d))
        in_support   = β -> true
    else
        prior_volume = prod(prior_hi .- prior_lo)
        prior_sample = () -> prior_lo .+ (prior_hi .- prior_lo) .* rand(rng, n_p)
        prior_logpdf = β -> (prior_volume > 0 ? -log(prior_volume) : 0.0)
        in_support   = β -> !(any(β .< prior_lo) || any(β .> prior_hi))
    end

    # ── generation 0: sample from prior, accept all ──────────────────
    # Counts simulations that failed for MODEL reasons (a bad draw whose
    # trajectory blows up). Program errors are rethrown, not counted — see
    # the catches below. Reported so a run where most draws never simulated
    # cannot look like a healthy posterior.
    n_sim_failed = 0
    particles = Vector{Vector{Float64}}(undef, N)
    distances = Vector{Float64}(undef, N)
    weights   = fill(1.0 / N, N)

    for i in 1:N
        particles[i] = prior_sample()
        try
            pred = simulate(prob, particles[i])
            distances[i] = summary_fn(pred, prob.data_values)
        catch e
            # A genuine integration failure means this particle is a bad
            # draw, and Inf is the right distance. A PROGRAM error (typo,
            # MethodError, BoundsError in the user's dynamics) is not a bad
            # draw — scoring it as "infinitely bad" makes every particle
            # look rejected and the solver reports a converged-looking fit
            # built entirely on nothing. Same policy as LAML and
            # DerivativeFreeSolver: bugs raise, model failures score.
            _is_program_error(e) && rethrow()
            distances[i] = Inf
            n_sim_failed += 1
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
                j = _abc_weighted_sample(rng, weights)
                # (b) perturb with Gaussian kernel
                theta_star = particles[j] .+ kernel_std .* randn(rng, n_p)

                # check prior support (no-op for the Gaussian prior)
                in_support(theta_star) || continue

                # (c-d) simulate and compute distance
                local d::Float64
                try
                    pred = simulate(prob, theta_star)
                    d = summary_fn(pred, prob.data_values)
                catch e
                    # Same policy as the prior-sampling loop above: a bug in
                    # the dynamics must raise rather than be silently
                    # skipped as a rejected proposal.
                    _is_program_error(e) && rethrow()
                    n_sim_failed += 1
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
            src = accepted_slots[rand(rng, 1:length(accepted_slots))]
            new_particles[i] = copy(new_particles[src])
            new_distances[i] = new_distances[src]
        end

        # ── (f) importance weights ───────────────────────────────────
        # w_i ∝ π(θ_i) / Σ_j w_j^{t-1} K(θ_i | θ_j^{t-1}), computed fully
        # in log space by _abc_importance_weights (which also monitors the
        # effective sample size).  Common constants in π(θ) cancel after
        # normalization.
        lps = Float64[in_support(new_particles[i]) ?
                      prior_logpdf(new_particles[i]) : -Inf for i in 1:N]
        new_weights, _ = _abc_importance_weights(new_particles, particles,
                                                 weights, kernel_std, lps;
                                                 gen=gen)

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

    # Weighted sum of squares over usable cells, matching the data_loss
    # convention of the other solvers (masked / NaN cells are skipped so
    # one missing observation does not turn the reported loss into NaN)
    data_loss = weighted_data_loss(prob, pred)

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
         n_sim_failed      = n_sim_failed,
         particles         = particles,
         distances         = distances,
         weights           = weights,
         tolerance_history = tolerance_history,
         best_idx          = best_idx,
         n_generations     = alg.n_generations),
    )
end
