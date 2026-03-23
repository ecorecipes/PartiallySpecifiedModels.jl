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

    # ── prior: Uniform(beta0 ± prior_scale) ──────────────────────────
    prior_lo = beta0 .- alg.prior_scale
    prior_hi = beta0 .+ alg.prior_scale
    prior_volume = prod(prior_hi .- prior_lo)

    # ── generation 0: sample from prior, accept all ──────────────────
    particles = Vector{Vector{Float64}}(undef, N)
    distances = Vector{Float64}(undef, N)
    weights   = fill(1.0 / N, N)

    for i in 1:N
        particles[i] = prior_lo .+ (prior_hi .- prior_lo) .* rand(n_p)
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

        for i in 1:N
            found = false
            for _attempt in 1:1000
                # (a) resample from previous generation
                j = _abc_weighted_sample(weights)
                # (b) perturb with Gaussian kernel
                theta_star = particles[j] .+ kernel_std .* randn(n_p)

                # check prior support
                if any(theta_star .< prior_lo) || any(theta_star .> prior_hi)
                    continue
                end

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

            # if no proposal accepted, carry forward old particle
            if !found
                new_particles[i] = copy(particles[i])
                new_distances[i] = distances[i]
            end
        end

        # ── (f) importance weights ───────────────────────────────────
        # w_i ∝ π(θ_i) / Σ_j w_j^{t-1} K(θ_i | θ_j^{t-1})
        new_weights = zeros(N)
        prior_density = prior_volume > 0 ? 1.0 / prior_volume : 1.0

        for i in 1:N
            if any(new_particles[i] .< prior_lo) || any(new_particles[i] .> prior_hi)
                new_weights[i] = 0.0
                continue
            end
            kernel_sum = 0.0
            for j in 1:N
                diff = new_particles[i] .- particles[j]
                log_k = -0.5 * sum((diff ./ kernel_std).^2)
                kernel_sum += weights[j] * exp(log_k)
            end
            new_weights[i] = kernel_sum > 0.0 ? prior_density / kernel_sum : 0.0
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

    # Fitted values from the best particle
    pred = try
        simulate(prob, best_beta)
    catch
        fill(NaN, size(prob.data_values))
    end

    data_loss = sum((pred .- prob.data_values).^2) / length(prob.data_values)

    # Unknown-function evaluators (from best particle)
    uf_evals = _abc_build_uf_dict(prob, best_beta)

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
