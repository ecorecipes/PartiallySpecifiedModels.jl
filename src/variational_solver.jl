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
        return adam_simulate_discrete(prob, p, T)
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
    _gaussian_loglik(pred, data, weights, obs_noise_var)

Gaussian log-likelihood: -0.5 Σ w_ij (pred_ij - data_ij)² / σ²_obs.

Zero-weight and NaN cells are skipped rather than multiplied by their
weight: `0 * NaN = NaN`, which would poison the whole ELBO (and with it
every gradient) from a single masked datum.
"""
function _gaussian_loglik(pred, data, weights, obs_noise_var)
    T = eltype(pred)
    ll = zero(T)
    n_t, n_obs = size(data)
    for j in 1:n_obs
        for i in 1:n_t
            w = weights[i, j]
            y = data[i, j]
            _usable(y, w) || continue
            ll -= w * (pred[i, j] - y)^2 / (2 * obs_noise_var)
        end
    end
    ll
end

"""
    _vi_loglik(fam, pred, data, weights, obs_noise_var)

Observation log-likelihood dispatched on the likelihood family. Gaussian
keeps the σ²-scaled quadratic form (`obs_noise_var` is the noise
variance); every other family uses its own pointwise log-likelihood with
`data_weights` multiplied in (`obs_noise_var` is ignored — dispersion
parameters live in the family object).
"""
_vi_loglik(::Gaussian, pred, data, weights, obs_noise_var) =
    _gaussian_loglik(pred, data, weights, obs_noise_var)

function _vi_loglik(fam::AbstractLikelihood, pred, data, weights, obs_noise_var)
    T = eltype(pred)
    ll = zero(T)
    n_t, n_obs = size(data)
    for j in 1:n_obs
        for i in 1:n_t
            w = weights[i, j]
            y = data[i, j]
            _usable(y, w) || continue
            ll += w * loglik_pointwise(fam, y, pred[i, j])
        end
    end
    ll
end

"""
    _kl_gaussian_penalized(mu, log_sigma, Λ, logdetΛ)

Analytical KL(q ‖ p) where q = N(μ, diag(σ²)) and the prior p = N(0, Λ⁻¹)
has precision Λ. The prior precision is the roughness-penalty GMRF prior
`λS` (plus a broad ridge on the null space), so — unlike an isotropic
N(0, τ²) prior — the ELBO actually contains the smoothing penalty `μᵀΛμ`
that defines a partially specified model.

    KL = ½[ tr(Λ Σ_q) + μᵀΛμ − k − log|Λ| − log|Σ_q| ]
"""
function _kl_gaussian_penalized(mu, log_sigma, Λ, logdetΛ)
    T = promote_type(eltype(mu), eltype(log_sigma))
    k = length(mu)
    dΛ = diag(Λ)
    tr_term = zero(T); quad_part = Λ * mu
    for i in 1:k
        tr_term += dΛ[i] * exp(2 * log_sigma[i])
    end
    quad = dot(mu, quad_part)
    logdetΣq = 2 * sum(log_sigma)
    T(0.5) * (tr_term + quad - k - T(logdetΛ) - logdetΣq)
end

"""
    _compute_elbo(prob, mu, log_sigma, Λ, logdetΛ, epsilons, obs_noise_var)

Compute the ELBO using the reparameterization trick, with the
penalty-induced Gaussian prior (precision `Λ`).

    ELBO = (1/S) Σ_s log p(Y|θ_s) - KL(q||p),   θ_s = μ + exp(log_σ) ⊙ ε_s
"""
function _compute_elbo(prob::PSMProblem, mu, log_sigma, Λ, logdetΛ,
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
        catch e
            _is_program_error(e) && rethrow()
            nothing
        end

        if pred === nothing
            # A failed simulation is a draw of effectively zero likelihood.
            # Averaging only over the successes would bias the ELBO toward
            # the non-failing region and hide infeasible posterior mass, so
            # count the failure with a large finite log-likelihood penalty.
            avg_ll += T(-1e8)
            n_valid += 1
            continue
        end

        avg_ll += _vi_loglik(prob.likelihood, pred, prob.data_values,
                             prob.data_weights, obs_noise_var)
        n_valid += 1
    end

    if n_valid == 0
        return T(-1e10)
    end
    avg_ll /= n_valid

    kl = _kl_gaussian_penalized(mu, log_sigma, Λ, logdetΛ)

    avg_ll - kl
end

"""
    _vi_edf(prob, mu_opt, Λ, obs_noise_var, n_p) → Float64

Effective degrees of freedom of the Gaussian (Laplace) approximation at the
variational posterior mean: `tr((H + Λ)⁻¹ H)` with the Gauss–Newton
information `H = JᵀWJ`, `J = ∂μ̂/∂β` by forward finite differences, and
`Λ` the variational prior precision.

`W` is the FAMILY's Gauss–Newton weight for the identity link, matching
`irls_weights` and the LAML/GCV curvature:

- Gaussian: `W = w / σ²_obs` (`obs_noise_var`), the classical form.
- everything else: `W = w / V(μ̂)` with `V` the family's variance function
  (`_variance_function`), evaluated at the fitted mean.

`obs_noise_var` is read for Gaussian data ONLY. Non-Gaussian families
carry their dispersion in the family object and `obs_noise_var` is
hard-set to 1.0 by the caller, so the old unconditional `H = JᵀWJ/σ²_obs`
silently reported a Poisson/NB/TruncatedNormal EDF computed with unit
observation variance — a curvature off by the factor `V(μ̂)`, which for a
Poisson fit with counts in the hundreds is two orders of magnitude.

Only usable observation cells (`usable_cell`: positive weight, finite
value) enter `W`. Returns `NaN` when the model cannot be simulated at
`mu_opt` or the linear system is singular — an honest missing value
rather than a fabricated constant.
"""
function _vi_edf(prob::PSMProblem, mu_opt::Vector{Float64},
                 Λ::Matrix{Float64}, obs_noise_var::Float64, n_p::Int)
    n_t = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    keep = [(i, j) for j in 1:n_obs for i in 1:n_t
            if usable_cell(prob, i, j)]
    isempty(keep) && return NaN

    base = try
        Float64.(simulate(prob, mu_opt))
    catch e
        _is_program_error(e) && rethrow()
        return NaN
    end
    all(isfinite, base) || return NaN

    J = zeros(length(keep), n_p)
    for b in 1:n_p
        step = max(1e-6, abs(mu_opt[b]) * 1e-6)
        bp = copy(mu_opt); bp[b] += step
        pert = try
            Float64.(simulate(prob, bp))
        catch e
            _is_program_error(e) && rethrow()
            return NaN
        end
        all(isfinite, pert) || return NaN
        for (r, (i, j)) in enumerate(keep)
            J[r, b] = (pert[i, j] - base[i, j]) / step
        end
    end

    fam = prob.likelihood
    w_gn = if fam isa Gaussian
        [Float64(prob.data_weights[i, j]) / obs_noise_var for (i, j) in keep]
    else
        # CALL `irls_weights` rather than re-deriving the weight from
        # `_variance_function`. The docstring above promises this curvature
        # "matches `irls_weights` and the LAML/GCV curvature", and the
        # hand-rolled `w / V(μ̂)` did not, for two families:
        #
        #  * TruncatedNormal. `irls_weights` is w·I(μ) with the Fisher
        #    information I = Var/σ⁴, NOT w/Var. The two differ by k² where
        #    k = 1 − ξλ(ξ) − λ(ξ)², so the EDF was wrong by that factor
        #    exactly where the truncation bites.
        #  * CustomLikelihood. `_variance_function` evaluates the curvature
        #    at y = μ (an EXPECTED variance); `irls_weights` evaluates it at
        #    the observed yᵢ, which is what the IRLS loop actually uses.
        #    Identical whenever the curvature is y-free (any Gaussian-shaped
        #    kernel), different exactly when it is not.
        #
        # Poisson and NegativeBinomial are unchanged for |μ| ≥ 1e-6 (both
        # forms reduce to w/V); below that `irls_weights` floors μ at 1e-6
        # where this floored V at 1e-10 — a regime no fit survives anyway,
        # and agreeing with the IRLS loop is the point.
        yk = [Float64(prob.data_values[i, j]) for (i, j) in keep]
        muk = [base[i, j] for (i, j) in keep]
        wk = [Float64(prob.data_weights[i, j]) for (i, j) in keep]
        irls_weights(fam, yk, muk, wk)
    end
    all(isfinite, w_gn) || return NaN
    H = J' * Diagonal(w_gn) * J
    edf = try
        tr((H .+ Λ) \ H)
    catch
        return NaN
    end
    isfinite(edf) || return NaN
    clamp(edf, 0.0, Float64(n_p))
end

# ─── Main variational solver ────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::VariationalSolver)

Fit a partially specified model using mean-field variational inference.
The posterior over unknown-function parameters is approximated by a
diagonal Gaussian, optimised by maximising the evidence lower bound (ELBO).

The ELBO's data term follows `prob.likelihood`: Gaussian data use the
σ²-scaled quadratic form with `obs_noise_var` (user-supplied or
estimated); Poisson, NegativeBinomial, TruncatedNormal, and
CustomLikelihood use the family's pointwise log-likelihood with
`data_weights` multiplied in and no Gaussian noise nuisance. Passing
`obs_noise_var` with a non-Gaussian family errors.

# Algorithm
1. Initialise the variational mean μ from the model's initial parameters
   and set log-σ to a small value.
2. At each iteration draw `n_samples` reparametrised samples from
   q(β) = N(μ, diag(σ²)).
3. Estimate the ELBO gradient via the reparametrisation trick and update
   (μ, log σ) with Adam.
4. Return the variational mean μ as point estimate, with the variational
   parameters in `sol.convergence`.

# References
- Blei, Kucukelbir & McAuliffe (2017), "Variational Inference: A Review
  for Statisticians", JASA.

# Returns
`PSMSolution` with fitted parameters, trajectory and unknown functions.
There is no `extras` field on `PSMSolution`: the variational parameters
live in `sol.convergence`, a `Dict{Symbol,Any}` with `:method`,
`:posterior_mean` (μ), `:posterior_std` (σ), `:final_elbo`,
`:elbo_history`, `:obs_noise_var` and `:n_iters`. `sol.objective` is the
negated best ELBO.
"""
function SciMLBase.solve(prob::PSMProblem, alg::VariationalSolver)
    _validate_problem(prob, "VariationalSolver")
    gaussian_obs = prob.likelihood isa Gaussian
    if !gaussian_obs && alg.obs_noise_var !== nothing
        error("VariationalSolver: obs_noise_var is the Gaussian " *
              "observation-noise variance, but prob.likelihood is " *
              "$(typeof(prob.likelihood)), which has no σ² parameter " *
              "(dispersion parameters are fixed in the family object).")
    end
    verbose = alg.verbose

    # Initialize variational parameters
    beta0 = build_initial_params(prob)
    n_p = length(beta0)
    mu = copy(beta0)
    log_sigma = fill(-2.0, n_p)  # σ ≈ 0.135

    # Observation noise variance (Gaussian families only): user-specified
    # or estimated from data. Non-Gaussian families carry their own
    # dispersion, so the value is never read (see `_vi_loglik`).
    obs_noise_var = if !gaussian_obs
        1.0
    elseif alg.obs_noise_var !== nothing
        alg.obs_noise_var
    else
        # Estimate from short-range variability in data (successive differences)
        # This is more robust than the data range heuristic
        n_t = size(prob.data_values, 1)
        n_obs = size(prob.data_values, 2)
        # Usable cells only: a masked/NaN datum must not poison σ̂².
        usable(i, j) = usable_cell(prob, i, j)
        total_var = 0.0
        n_dd = 0
        if n_t >= 3
            for j in 1:n_obs
                for i in 2:n_t-1
                    # Second differences estimate noise (removes trend)
                    (usable(i-1, j) && usable(i, j) && usable(i+1, j)) || continue
                    dd = prob.data_values[i-1, j] - 2*prob.data_values[i, j] + prob.data_values[i+1, j]
                    total_var += dd^2
                    n_dd += 1
                end
            end
        end
        if n_dd > 0
            max(total_var / (6 * n_dd), 1e-6)  # Var(Δ²y) = 6σ² for white noise
        else
            vals = [prob.data_values[i, j] for j in 1:n_obs for i in 1:n_t
                    if usable(i, j)]
            data_range = isempty(vals) ? 0.0 : maximum(vals) - minimum(vals)
            max((0.05 * data_range)^2, 1e-6)
        end
    end

    if verbose
        println("VariationalSolver: $n_p params, $(alg.maxiters) max iters, " *
                "lr=$(alg.lr), S=$(alg.n_elbo_samples)")
        println("  prior_scale=$(alg.prior_scale)" *
                (gaussian_obs ?
                 ", obs_noise_var=$(round(obs_noise_var, sigdigits=3))" :
                 ", likelihood=$(typeof(prob.likelihood))"))
    end

    # Prior precision Λ = roughness penalty (λS per smooth term) + broad
    # ridge on the null space.  This is what carries the smoothing penalty
    # into the ELBO (an isotropic prior would drop it entirely).
    Λ = zeros(n_p, n_p)
    ridge = 1.0 / (100.0 * alg.prior_scale)
    for i in 1:n_p; Λ[i, i] += ridge; end
    let (pens, offs, _) = _build_penalty_info(prob)
        for (k, S) in enumerate(pens)
            npk = size(S, 1); idx = (offs[k]+1):(offs[k]+npk)
            Λ[idx, idx] .+= (1.0 / alg.prior_scale) .* S
        end
    end
    logdetΛ = logdet(cholesky(Symmetric(Λ)))

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

    # rng_seed defaults to 42 (historical hard-coded stream); nothing = fresh.
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    for iter in 1:alg.maxiters
        # Draw shared noise samples (fixed across gradient computation)
        epsilons = randn(rng, n_p, alg.n_elbo_samples)

        # Define ELBO as function of φ for ForwardDiff
        function neg_elbo(phi_vec)
            mu_v = phi_vec[1:n_p]
            ls_v = phi_vec[n_p+1:end]
            -_compute_elbo(prob, mu_v, ls_v, Λ, logdetΛ, epsilons,
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
            _is_program_error(e) && rethrow()
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

        # Convergence check: ELBO plateau over last 50 iterations, guarded
        # against the cosine lr schedule manufacturing a plateau: near
        # maxiters lr_t → 0, so the ELBO stops moving however far from the
        # optimum φ still is. Only stop while the step is meaningful.
        #
        # `best_elbo > -1e9` is AdamSolver's failure-sentinel guard in ELBO
        # (maximization) form: `_vi_elbo` returns −1e10 when the dynamics
        # cannot be evaluated, and a run pinned at that sentinel is a stuck
        # solver, not a converged one — it must keep iterating rather than
        # stop early on the flat sentinel.
        if iter > 100 && isfinite(best_elbo) && best_elbo > -1e9 &&
           lr_t > 0.05 * lr
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

    # A non-finite best ELBO means no iteration ever produced a usable
    # objective, so `best_phi` is still the initialization: returning it
    # would present the initial guess as a fit. Fail loudly instead.
    if !isfinite(best_elbo)
        n_nan = count(!isfinite, elbo_history)
        error("VariationalSolver: the ELBO was non-finite at every one of " *
              "$(length(elbo_history)) iterations ($n_nan non-finite), so " *
              "the returned variational parameters would be the untouched " *
              "initialization rather than a fit. Common causes: NaN or Inf " *
              "in data_values at cells with nonzero data_weights, a " *
              "likelihood undefined at the initial predictions, or dynamics " *
              "that produce NaN. Mask unusable observations by setting " *
              "their data_weights entry to 0.")
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
    catch e
        _is_program_error(e) && rethrow()
        fill(NaN, length(prob.data_times), size(prob.data_values, 2))
    end

    n_t = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    # Masked/NaN cells are skipped (0 * NaN = NaN would make data_loss NaN).
    data_loss = weighted_data_loss(prob, pred)

    # Effective degrees of freedom of the Gaussian (Laplace) approximation
    # taken at the variational posterior mean:
    #
    #     edf = tr((H + Λ)⁻¹ H),
    #
    # with H the Gauss–Newton observed information of the data term,
    # H = JᵀWJ (W = w/σ²_obs for Gaussian data, w/V(μ̂) for every other
    # family — see `_vi_edf`) and J = ∂μ̂/∂β by finite differences, and Λ the SAME
    # prior precision (ridge + S/prior_scale) that defines the variational
    # prior above. This is the standard penalized-regression EDF and, being
    # built from the model Jacobian rather than from q's scale parameters,
    # it responds to `prior_scale` as a dof must.
    #
    # It replaces Σᵢ (1 − σ_q,ᵢ² Λᵢᵢ), which is the correct per-coordinate
    # shrinkage factor ONLY at the exact mean-field optimum, where
    # σ_q,ᵢ² = 1/(Hᵢᵢ + Λᵢᵢ). Adam does not drive log σ that far in a
    # finite iteration budget, so in practice σ_q,ᵢ² Λᵢᵢ > 1 for most
    # coordinates, every term hit its clamp, and the reported EDF was the
    # constant 1.0 for any smoothing level (measured: edf = 1.0 at
    # prior_scale = 1e-4 AND at prior_scale = 1).
    #
    # If the Jacobian cannot be built (simulation failure at the posterior
    # mean) the EDF is reported as NaN rather than fabricated.
    edf = _vi_edf(prob, mu_opt, Λ, obs_noise_var, n_p)

    # Build unknown function evaluators using posterior mean
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = mu_opt[offset+1:offset+np]
        offset += np

        uf_evals[approx.name] = build_evaluator(approx, params_k)
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
        :obs_noise_var => gaussian_obs ? obs_noise_var : nothing,
        :n_iters => length(elbo_history),
    )

    PSMSolution(params_ca, -best_elbo, data_loss, edf,
                [alg.prior_scale],
                pred, Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                convergence)
end
