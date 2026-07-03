# ─── Variational inference solver ────────────────────────────────────
#
# Mean-field Gaussian variational inference for PSM problems.
# Approximates the posterior p(θ|Y) ≈ q(θ) = ∏_i N(μ_i, σ_i²) by
# maximizing the ELBO via Adam with reparameterization-trick gradients.

"""
    _variational_simulate(prob, beta)

Simulate model with parameter vector `beta`, returning predictions matrix.
Uses the ForwardDiff-compatible `build_autodiff_param_struct` when available,
falling back to `simulate` otherwise. Returns `nothing` on solver failure.
"""
function _variational_simulate(prob::PSMProblem, beta)
    T = eltype(beta)

    p = build_autodiff_param_struct(prob, beta)

    if prob.discrete
        return adam_simulate_discrete(prob, p)
    end

    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    u0_T = T.(u0)

    sol = if !isempty(prob.delays)
        adam_solve_dde(prob, beta)
    else
        ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
            (du, u, params, t) -> prob.dynamics!(du, u, params, t))
        ode_prob = ODEProblem(ode_fn, u0_T, prob.tspan, p)
        OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                             saveat=prob.data_times,
                             abstol=1e-7, reltol=1e-7,
                             maxiters=10000)
    end

    if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
        return nothing
    end

    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)
    pred = zeros(T, n_t, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:min(n_t, length(sol.t))
            pred[i, j] = sol[sk, i]
        end
    end
    pred
end

"""
    _kl_gaussian(mu, log_sigma, prior_scale)

Analytical KL divergence: KL(q || p) where q = N(μ, σ²), p = N(0, τ²).
`log_sigma` contains log(σ_i), `prior_scale` = τ.

    KL = Σ_i [log(τ/σ_i) + (σ_i² + μ_i²)/(2τ²) - 1/2]
"""
function _kl_gaussian(mu, log_sigma, prior_scale)
    T = promote_type(eltype(mu), eltype(log_sigma))
    τ = T(prior_scale)
    τ² = τ * τ
    log_τ = log(τ)
    kl = zero(T)
    for i in eachindex(mu)
        σ_i = exp(log_sigma[i])
        kl += log_τ - log_sigma[i] + (σ_i^2 + mu[i]^2) / (2 * τ²) - T(0.5)
    end
    kl
end

"""
    _compute_elbo(prob, mu, log_sigma, prior_scale, epsilons, obs_noise_var)

Compute the ELBO using the reparameterization trick.

    ELBO = (1/S) Σ_s log p(Y|θ_s) - KL(q||p)
    θ_s = μ + exp(log_σ) ⊙ ε_s

`epsilons` is a matrix of size (n_params, n_samples).
"""
function _compute_elbo(prob::PSMProblem, mu, log_sigma, prior_scale,
                       epsilons, obs_noise_var)
    T = promote_type(eltype(mu), eltype(log_sigma))
    n_samples = size(epsilons, 2)
    sigma = exp.(log_sigma)

    avg_ll = zero(T)
    n_valid = 0

    for s in 1:n_samples
        theta_s = mu .+ sigma .* epsilons[:, s]

        pred = try
            _variational_simulate(prob, theta_s)
        catch
            nothing
        end

        if pred === nothing
            continue
        end

        avg_ll += observation_loglikelihood(prob.likelihood,
                                            prob.data_values,
                                            pred,
                                            prob.data_weights;
                                            sigma2=obs_noise_var)
        n_valid += 1
    end

    if n_valid == 0
        return T(-1e10)
    end
    avg_ll /= n_valid

    kl = _kl_gaussian(mu, log_sigma, prior_scale)

    avg_ll - kl
end

# ─── Main variational solver ────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::VariationalSolver)

Fit a partially specified model using mean-field variational inference.
The posterior over unknown-function parameters is approximated by a
diagonal Gaussian, optimised by maximising the evidence lower bound (ELBO).

# Algorithm
1. Initialise the variational mean μ from the model's initial parameters
   and set log-σ to a small value.
2. At each iteration draw `n_samples` reparametrised samples from
   q(β) = N(μ, diag(σ²)).
3. Estimate the ELBO gradient via the reparametrisation trick and update
   (μ, log σ) with Adam.
4. Return the posterior mean as point estimate and the variational
   parameters in `sol.convergence`.

# References
- Blei, Kucukelbir & McAuliffe (2017), "Variational Inference: A Review
  for Statisticians", JASA.

# Returns
`PSMSolution` with fitted parameters, trajectory, unknown functions,
and variational parameters `μ`, `σ` in `sol.convergence`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::VariationalSolver)
    _validate_problem(prob, "VariationalSolver")
    verbose = alg.verbose

    # Initialize variational parameters
    beta0 = build_initial_params(prob)
    n_p = length(beta0)
    mu = copy(beta0)
    log_sigma = fill(-2.0, n_p)  # σ ≈ 0.135

    # Observation noise variance: user-specified or estimated from data
    obs_noise_var = if prob.likelihood isa Gaussian && alg.obs_noise_var !== nothing
        alg.obs_noise_var
    elseif prob.likelihood isa Gaussian
        # Estimate from short-range variability in data (successive differences)
        # This is more robust than the data range heuristic
        n_t = size(prob.data_values, 1)
        n_obs = size(prob.data_values, 2)
        if n_t >= 3
            total_var = 0.0
            count = 0
            for j in 1:n_obs
                for i in 2:n_t-1
                    # Second differences estimate noise (removes trend)
                    dd = prob.data_values[i-1, j] - 2*prob.data_values[i, j] + prob.data_values[i+1, j]
                    total_var += dd^2
                    count += 1
                end
            end
            max(total_var / (6 * count), 1e-6)  # Var(Δ²y) = 6σ² for white noise
        else
            data_range = maximum(prob.data_values) - minimum(prob.data_values)
            max((0.05 * data_range)^2, 1e-6)
        end
    else
        nothing
    end

    if verbose
        println("VariationalSolver: $n_p params, $(alg.maxiters) max iters, " *
                "lr=$(alg.lr), S=$(alg.n_elbo_samples)")
        obs_msg = obs_noise_var === nothing ? "n/a" : string(round(obs_noise_var, sigdigits=3))
        println("  prior_scale=$(alg.prior_scale), obs_noise_var=$obs_msg")
    end

    # Concatenated variational parameters: φ = [μ; log_σ]
    n_phi = 2 * n_p
    phi = vcat(mu, log_sigma)

    # Adam optimizer state
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_phi)
    v_adam = zeros(n_phi)

    best_phi = copy(phi)
    best_elbo = -Inf
    elbo_history = Float64[]

    rng = Random.Xoshiro(42)

    for iter in 1:alg.maxiters
        # Draw shared noise samples (fixed across gradient computation)
        epsilons = randn(rng, n_p, alg.n_elbo_samples)

        # Define ELBO as function of φ for ForwardDiff
        function neg_elbo(phi_vec)
            mu_v = phi_vec[1:n_p]
            ls_v = phi_vec[n_p+1:end]
            -_compute_elbo(prob, mu_v, ls_v, alg.prior_scale, epsilons,
                           obs_noise_var)
        end

        # Compute gradient via ForwardDiff
        local elbo_val
        grad = try
            result = DiffResults.MutableDiffResult(0.0, (zeros(n_phi),))
            ForwardDiff.gradient!(result, neg_elbo, phi)
            elbo_val = -DiffResults.value(result)
            neg_grad = DiffResults.gradient(result)
            neg_grad
        catch e
            if verbose && iter <= 5
                println("  iter $iter: gradient failed ($(typeof(e))), using zeros")
            end
            elbo_val = -Inf
            zeros(n_phi)
        end

        # Clip gradient for stability
        grad_norm = norm(grad)
        if grad_norm > 100.0
            grad .*= 100.0 / grad_norm
        end

        push!(elbo_history, elbo_val)

        if elbo_val > best_elbo
            best_elbo = elbo_val
            best_phi .= phi
        end

        # Cosine learning rate annealing
        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        # Adam update (minimize negative ELBO)
        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        phi .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 100 == 0 || iter == alg.maxiters)
            @printf("  iter %d: ELBO=%.4f  lr=%.5f  |∇|=%.3f\n",
                    iter, elbo_val, lr_t, grad_norm)
        end

        # Convergence check: ELBO plateau over last 50 iterations
        if iter > 100
            window = max(1, length(elbo_history) - 49):length(elbo_history)
            recent = elbo_history[window]
            recent_range = maximum(recent) - minimum(recent)
            if recent_range / max(abs(mean(recent)), 1.0) < 1e-4
                if verbose
                    println("  Converged at iter $iter (ELBO plateau)")
                end
                break
            end
        end
    end

    # Recover best variational parameters
    phi .= best_phi
    mu_opt = phi[1:n_p]
    log_sigma_opt = phi[n_p+1:end]
    sigma_opt = exp.(log_sigma_opt)

    if verbose
        println("  Best ELBO: $(round(best_elbo, sigdigits=5))")
        println("  Posterior std range: [$(round(minimum(sigma_opt), sigdigits=3)), " *
                "$(round(maximum(sigma_opt), sigdigits=3))]")
    end

    # Build solution using posterior mean
    pred = try
        p = simulate(prob, mu_opt)
        Float64.(p)
    catch
        zeros(length(prob.data_times), size(prob.data_values, 2))
    end

    n_t = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_t
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Approximate effective degrees of freedom from posterior variance:
    # Parameters with small posterior variance relative to prior are well-determined
    edf = sum(1.0 .- (sigma_opt .^ 2) ./ (alg.prior_scale^2))
    edf = clamp(edf, 1.0, Float64(n_p))

    # Build unknown function evaluators using posterior mean
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = mu_opt[offset+1:offset+np]
        offset += np

        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing :
                   (approx.domain[2] - approx.domain[1])
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
        end
    end

    # Build ComponentArray for parameters
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => mu_opt[offset+1:offset+np])
        offset += np
    end
    params_ca = ComponentArray(NamedTuple(ca_entries))

    convergence = Dict{Symbol, Any}(
        :method => :variational,
        :elbo_history => elbo_history,
        :final_elbo => best_elbo,
        :posterior_mean => copy(mu_opt),
        :posterior_std => copy(sigma_opt),
        :obs_noise_var => obs_noise_var,
        :n_iters => length(elbo_history),
    )

    PSMSolution(params_ca, -best_elbo, data_loss, edf,
                [alg.prior_scale],
                pred, Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                convergence)
end
