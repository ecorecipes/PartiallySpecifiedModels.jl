# ─── Adaptive Gradient Matching solver ───────────────────────────────
#
# Implementation of adaptive gradient matching for parameter inference in
# ODE systems, following Dondelinger et al. (2013) and the deGradInfer R
# package (Macdonald & Husmeier, 2017).
#
# Key idea: Use Gaussian processes to smooth observed data AND quantify
# gradient uncertainty. Match ODE-predicted gradients to GP-inferred
# gradients via a "product of experts" formulation with mismatch parameter γ.
#
# Extended for partially specified models: unknown function coefficients
# (B-spline, GP, NN) are optimized jointly with the mismatch parameter γ.
#
# References:
# - Dondelinger et al. (2013), JMLR 14:3015-3043
# - Calderhead et al. (2009), PNAS 106(16):6461-6466
# - Macdonald & Husmeier (2017), deGradInfer R package

using LinearAlgebra: Symmetric, cholesky, logdet, dot, norm, tr, I

# ─── GP kernel derivatives for gradient matching ────────────────

"""
Compute the RBF kernel matrix K(T, T) and its derivatives K*, K**.

For k(t, t') = σ² exp(-(t-t')²/(2ℓ²)):
- K[i,j]   = σ² exp(-d²/(2ℓ²))
- K*[i,j]  = ∂k/∂t  = -(σ²/ℓ²)(t_i - t_j) exp(-d²/(2ℓ²))
- K**[i,j] = ∂²k/∂t∂t' = (σ²/ℓ²)(1 - (t_i-t_j)²/ℓ²) exp(-d²/(2ℓ²))
"""
function rbf_kernel_with_derivs(times::Vector{Float64}, σ²::Float64, ℓ::Float64)
    n = length(times)
    K = zeros(n, n)
    Kstar = zeros(n, n)    # ∂K/∂t_i  (derivative w.r.t. first argument)
    Kstarstar = zeros(n, n) # ∂²K/∂t_i∂t_j

    inv_ℓ² = 1.0 / ℓ^2

    for j in 1:n, i in 1:n
        d = times[i] - times[j]
        d² = d^2
        kval = σ² * exp(-0.5 * d² * inv_ℓ²)

        K[i, j] = kval
        Kstar[i, j] = -inv_ℓ² * d * kval
        Kstarstar[i, j] = inv_ℓ² * (1.0 - d² * inv_ℓ²) * kval
    end

    K, Kstar, Kstarstar
end

"""
Compute Matérn 3/2 kernel and its derivatives.
k(t, t') = σ²(1 + √3|d|/ℓ) exp(-√3|d|/ℓ)
"""
function matern32_kernel_with_derivs(times::Vector{Float64}, σ²::Float64, ℓ::Float64)
    n = length(times)
    K = zeros(n, n)
    Kstar = zeros(n, n)
    Kstarstar = zeros(n, n)

    sqrt3 = sqrt(3.0)
    inv_ℓ = 1.0 / ℓ

    for j in 1:n, i in 1:n
        d = times[i] - times[j]
        r = abs(d)
        sr = sqrt3 * r * inv_ℓ
        exp_sr = exp(-sr)

        K[i, j] = σ² * (1.0 + sr) * exp_sr

        # ∂k/∂t_i = -3σ²/ℓ² · d · exp(-√3|d|/ℓ)
        Kstar[i, j] = -3.0 * σ² * inv_ℓ^2 * d * exp_sr

        # ∂²k/∂t_i∂t_j = 3σ²/ℓ² · (1 - √3|d|/ℓ) · exp(-√3|d|/ℓ)
        Kstarstar[i, j] = 3.0 * σ² * inv_ℓ^2 * (1.0 - sr) * exp_sr
    end

    K, Kstar, Kstarstar
end

# ─── GP hyperparameter optimization ─────────────────────────────

"""
Optimize GP hyperparameters (σ², ℓ, σ_n²) by maximizing the log marginal
likelihood for a single state variable.

Returns: (σ², ℓ, σ_n²)
"""
function optimize_gp_hyperparams(times::Vector{Float64}, y::Vector{Float64},
                                  kernel::Symbol; verbose::Bool=false)
    n = length(times)
    time_span = times[end] - times[1]
    y_var = max(var(y), 1e-10)

    # Initial guess: σ² = data variance, ℓ = time_span/5, σ_n² = 0.01*y_var
    best_σ² = y_var
    best_ℓ = time_span / 5.0
    best_σn² = 0.01 * y_var
    best_nll = Inf

    # Grid search over lengthscales and noise levels
    for ℓ_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        ℓ_try = time_span * ℓ_frac
        for σn_frac in [1e-4, 1e-3, 1e-2, 5e-2, 0.1]
            σn²_try = σn_frac * y_var
            K, _, _ = if kernel == :matern32
                matern32_kernel_with_derivs(times, y_var, ℓ_try)
            else
                rbf_kernel_with_derivs(times, y_var, ℓ_try)
            end
            Ky = K + σn²_try * I(n)
            try
                C = cholesky(Symmetric(Ky))
                α = C \ y
                nll = 0.5 * dot(y, α) + 0.5 * logdet(C) + 0.5 * n * log(2π)
                if nll < best_nll
                    best_nll = nll
                    best_σ² = y_var
                    best_ℓ = ℓ_try
                    best_σn² = σn²_try
                end
            catch
            end
        end
    end

    if verbose
        println("  GP hyperparams: σ²=$(round(best_σ², sigdigits=3)) " *
                "ℓ=$(round(best_ℓ, sigdigits=3)) σ_n²=$(round(best_σn², sigdigits=3))")
    end

    best_σ², best_ℓ, best_σn²
end

# ─── GP gradient inference ───────────────────────────────────────

"""
Compute GP-inferred gradient mean and covariance for a single state variable.

Given observations y at times t with GP hyperparameters (σ², ℓ, σ_n²):
- Gradient mean:  m = K* (K + σ_n² I)⁻¹ y
- Gradient covariance: A = K** - K* (K + σ_n² I)⁻¹ K*ᵀ

Also returns the smoothed state: x = K (K + σ_n² I)⁻¹ y
"""
function gp_gradient_inference(times::Vector{Float64}, y::Vector{Float64},
                                σ²::Float64, ℓ::Float64, σn²::Float64,
                                kernel::Symbol)
    n = length(times)

    K, Kstar, Kstarstar = if kernel == :matern32
        matern32_kernel_with_derivs(times, σ², ℓ)
    else
        rbf_kernel_with_derivs(times, σ², ℓ)
    end

    # K_y = K + σ_n² I
    Ky = K + σn² * I(n)
    C = cholesky(Symmetric(Ky + 1e-10 * I(n)))

    # Smoothed state: x = K (K + σ_n² I)⁻¹ y
    α = C \ y  # α = (K + σ_n² I)⁻¹ y
    x_smooth = K * α

    # Gradient mean: m = K* α = K* (K + σ_n² I)⁻¹ y
    grad_mean = Kstar * α

    # Gradient covariance: A = K** - K* (K + σ_n² I)⁻¹ K*ᵀ
    V = C.L \ Kstar'  # L⁻¹ K*ᵀ
    grad_cov = Kstarstar - V' * V

    # Ensure positive definiteness
    grad_cov = Symmetric(grad_cov)
    eig = eigen(grad_cov)
    eig_vals = max.(eig.values, 1e-10)
    grad_cov = Symmetric(eig.vectors * Diagonal(eig_vals) * eig.vectors')

    x_smooth, grad_mean, grad_cov
end

# ─── Adaptive Gradient Matching loss ─────────────────────────────

"""
Compute the product-of-experts gradient matching loss.

For each state k (with a Gamma prior on the mismatch variance γ_k):
  L_k = -0.5 (f_k - m_k)ᵀ (A_k + γ_k I)⁻¹ (f_k - m_k) - 0.5 log|A_k + γ_k I|
        + log γ_k - γ_k/scale_k

Total loss = -Σ_k L_k  (negative because we minimize)
"""
function agm_loss(prob::PSMProblem, beta::AbstractVector,
                  log_gamma::AbstractVector,
                  times::Vector{Float64}, x_smooth::Matrix{Float64},
                  grad_means::Matrix{Float64},
                  A_eigvals::Vector{Vector{Float64}},
                  A_eigvecs::Vector{Matrix{Float64}};
                  smoothing_lambda::Float64=1.0)
    K_states = size(x_smooth, 2)
    T_pts = length(times)
    T = promote_type(eltype(beta), eltype(log_gamma))

    gamma = exp.(log_gamma)

    # Build parameter struct and evaluate ODE RHS (use autodiff-compatible version)
    p = build_autodiff_param_struct(prob, beta)
    F = zeros(T, T_pts, K_states)
    du = zeros(T, K_states)
    for i in 1:T_pts
        u = T.(x_smooth[i, :])
        try
            prob.dynamics!(du, u, p, times[i])
        catch
            return T(1e10)
        end
        F[i, :] .= du
    end

    # Gradient matching loss using pre-computed eigendecomposition of A
    total_loss = zero(T)
    for k in 1:K_states
        f_k = F[:, k]
        m_k = T.(grad_means[:, k])
        λ_A = A_eigvals[k]
        V = A_eigvecs[k]

        # Shifted eigenvalues: λ_A + γ (clamp for numerical safety)
        shifted_eig = T.(λ_A) .+ gamma[k]
        shifted_eig = max.(shifted_eig, T(1e-8))

        residual = f_k - m_k
        Vt_r = T.(V') * residual

        # rᵀ(A+γI)⁻¹r = Σ (Vᵀr)²/(λ+γ), log|A+γI| = Σ log(λ+γ)
        quad_form = sum(Vt_r .^ 2 ./ shifted_eig)
        log_det = sum(log.(shifted_eig))

        ll_k = -T(0.5) * quad_form - T(0.5) * log_det

        # Weakly-informative Gamma(shape=2, rate=1/scale) prior on the
        # gradient-mismatch variance γ_k (Dondelinger et al. 2013;
        # Calderhead et al. 2009 place a Gamma prior on γ precisely to keep
        # it away from the degenerate γ→0 limit, where the GP-derivative
        # constraint becomes infinitely tight and overfits). The scale is set
        # to the mean eigenvalue of A_k, the natural variance scale, so the
        # prior is uninformative relative to the data term.
        γ_scale = max(sum(λ_A) / length(λ_A), T(1e-6))
        ll_k += log(gamma[k]) - gamma[k] / T(γ_scale)
        total_loss -= ll_k
    end

    # Smoothing penalty for B-spline approximators
    if smoothing_lambda > 0
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            beta_k = beta[offset+1:offset+np]
            offset += np
            if approx isa BSplineApproximator
                knots_x = collect(range(approx.domain[1], approx.domain[2],
                                        length=approx.nknots))
                S = spline_penalty_matrix(knots_x)
                total_loss += T(smoothing_lambda) * dot(beta_k, S * beta_k)
            elseif approx isa ShapeConstrainedBSplineApproximator
                S = penalty_matrix(approx)
                total_loss += T(smoothing_lambda) * dot(beta_k, S * beta_k)
            elseif approx isa COMONetApproximator
                S = penalty_matrix(approx)
                total_loss += T(smoothing_lambda) * dot(beta_k, S * beta_k)
            elseif approx isa SPDEApproximator
                S = penalty_matrix(approx)
                total_loss += T(smoothing_lambda) * dot(beta_k, S * beta_k)
            elseif approx isa ShapeConstrainedSPDEApproximator
                S = penalty_matrix(approx)
                total_loss += T(smoothing_lambda) * dot(beta_k, S * beta_k)
            end
        end
    end

    total_loss
end

# ─── Main solver ─────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::AdaptiveGradientMatching)

Fit a partially specified model using adaptive gradient matching with a
product-of-experts Gaussian process formulation. A GP prior on the state
is combined with a GP-based likelihood derived from the ODE to form a
joint posterior, avoiding explicit numerical integration.

# Algorithm
1. Fit a GP to each observed state to obtain smoothed trajectories and
   derivative statistics (mean and covariance).
2. Construct the product-of-experts posterior by combining the data GP
   with the ODE-implied GP likelihood.
3. Optimise unknown-function parameters by maximising the joint marginal
   likelihood with respect to the combined model.
4. Optionally re-estimate GP hyperparameters (γ) to adapt the state prior.

# References
- Dondelinger et al. (2013), "ODE parameter inference using adaptive
  gradient matching with Gaussian processes", AISTATS.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::AdaptiveGradientMatching)
    _validate_problem(prob, "AdaptiveGradientMatching")
    verbose = alg.verbose
    times = Float64.(prob.data_times)
    T_pts = length(times)
    K_states = length(prob.u0)
    n_obs = size(prob.data_values, 2)

    if verbose
        println("AdaptiveGradientMatching: kernel=$(alg.kernel), " *
                "fit_gamma=$(alg.fit_gamma)")
    end

    # Step 1: Fit GP to each observed state and compute gradient statistics
    if verbose; println("\nStep 1: GP smoothing and gradient inference"); end

    x_smooth = zeros(T_pts, K_states)
    grad_means = zeros(T_pts, K_states)
    grad_covs = Vector{Matrix{Float64}}(undef, K_states)

    gp_hyperparams = Vector{Tuple{Float64,Float64,Float64}}(undef, K_states)

    if prob.discrete
        # For discrete models: GP-smooth the states, and use forward-shifted
        # smoothed values as "gradient targets" (i.e., match f(x[t]) to x[t+1])
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            y = Float64.(prob.data_values[:, j])

            σ², ℓ, σn² = optimize_gp_hyperparams(times, y, alg.kernel; verbose=verbose)
            gp_hyperparams[sk] = (σ², ℓ, σn²)

            xs, _, gc = gp_gradient_inference(times, y, σ², ℓ, σn², alg.kernel)
            x_smooth[:, sk] .= xs

            # For discrete: "gradient_mean" = next state value (forward shift)
            for i in 1:(T_pts-1)
                grad_means[i, sk] = xs[i+1]
            end
            grad_means[T_pts, sk] = xs[T_pts]  # unused padding

            # Use GP state covariance as gradient covariance proxy
            # (uncertainty in predicting next state)
            grad_covs[sk] = gc  # Reuse derivative covariance as proxy
        end
    else
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            y = Float64.(prob.data_values[:, j])

            σ², ℓ, σn² = optimize_gp_hyperparams(times, y, alg.kernel; verbose=verbose)
            gp_hyperparams[sk] = (σ², ℓ, σn²)

            xs, gm, gc = gp_gradient_inference(times, y, σ², ℓ, σn², alg.kernel)
            x_smooth[:, sk] .= xs
            grad_means[:, sk] .= gm
            grad_covs[sk] = gc
        end
    end

    # For unobserved states, initialize from u0 (constant)
    for k in 1:K_states
        if !isassigned(grad_covs, k)
            x_smooth[:, k] .= prob.u0[k]
            grad_means[:, k] .= 0.0
            grad_covs[k] = Matrix(1e6 * I(T_pts))  # Large uncertainty
        end
    end

    # Pre-compute eigendecompositions of gradient covariance matrices
    # (these are constant during optimization — only gamma changes)
    A_eigvals = Vector{Vector{Float64}}(undef, K_states)
    A_eigvecs = Vector{Matrix{Float64}}(undef, K_states)
    for k in 1:K_states
        E = eigen(Symmetric(grad_covs[k]))
        A_eigvals[k] = E.values
        A_eigvecs[k] = E.vectors
        if verbose
            min_eig = minimum(E.values)
            println("  State $k: min eigenvalue of A = $(round(min_eig, sigdigits=3))")
        end
    end

    # Step 2: Initialize parameters
    if verbose; println("\nStep 2: Initializing parameters"); end

    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Compute initial gradient mismatch to set gamma adaptively
    p_init = build_param_struct(prob, beta)
    du_init = zeros(K_states)
    mismatch_var = zeros(K_states)
    for k in 1:K_states
        resids = zeros(T_pts)
        for i in 1:T_pts
            u = x_smooth[i, :]
            prob.dynamics!(du_init, u, p_init, times[i])
            resids[i] = du_init[k] - grad_means[i, k]
        end
        mismatch_var[k] = max(var(resids), 1e-4)
    end

    # Initialize gamma from mismatch variance (start tight, let optimizer loosen)
    gamma_init_vals = mismatch_var .* 0.1
    log_gamma = log.(gamma_init_vals)
    n_gamma = alg.fit_gamma ? K_states : 0

    if verbose
        println("  Initial γ: ", round.(gamma_init_vals, sigdigits=3))
    end

    # Smoothing lambda: auto-scale from gradient variance
    smoothing_lambda = 0.01 * mean(mismatch_var)

    # Optimization variable: z = [beta; log_gamma]
    z = alg.fit_gamma ? vcat(beta, log_gamma) : copy(beta)
    n_z = length(z)

    if verbose
        println("  $(n_beta) approximator params + $(n_gamma) mismatch params = $(n_z) total")
        println("  Smoothing λ = $(round(smoothing_lambda, sigdigits=3))")
    end

    # Step 3: Optimize using L-BFGS
    if verbose; println("\nStep 3: L-BFGS optimization"); end

    function loss_fn(z_)
        β_ = z_[1:n_beta]
        lg_ = alg.fit_gamma ? z_[n_beta+1:end] : log_gamma
        agm_loss(prob, β_, lg_, times, x_smooth, grad_means, A_eigvals, A_eigvecs;
                 smoothing_lambda=smoothing_lambda)
    end

    result = Optim.optimize(
        loss_fn,
        z_ -> ForwardDiff.gradient(loss_fn, z_),
        z,
        Optim.LBFGS(),
        Optim.Options(
            iterations=alg.maxiters,
            show_trace=verbose,
            show_every=max(1, alg.maxiters ÷ 10),
            g_tol=1e-8,
            f_reltol=1e-12,
        );
        inplace=false
    )
    z_opt = Optim.minimizer(result)

    beta_opt = z_opt[1:n_beta]
    gamma_opt = alg.fit_gamma ? exp.(z_opt[n_beta+1:end]) : exp.(log_gamma)

    if verbose
        println("  Converged: $(Optim.converged(result))")
        println("  Final loss: $(round(Optim.minimum(result), sigdigits=5))")
        println("  γ per state: $(round.(gamma_opt, sigdigits=3))")
    end

    # Step 4: Build solution
    # Compute data-space predictions from smoothed states
    pred = zeros(T_pts, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= x_smooth[:, sk]
    end

    data_loss = 0.0
    for j in 1:n_obs, i in 1:T_pts
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Derivative matching loss
    p_opt = build_param_struct(prob, beta_opt)
    du = zeros(K_states)
    F_final = zeros(T_pts, K_states)
    for i in 1:T_pts
        prob.dynamics!(du, x_smooth[i, :], p_opt, times[i])
        F_final[i, :] .= du
    end
    deriv_loss = sum((grad_means .- F_final).^2)

    # Build evaluators
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_opt[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(Lux.initialparameters(rng2, approx.model)))
            ps_final = similar(ps_ca)
            ps_final .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model, Float32.(reshape([xn], :, 1)), ps_final, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
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

    edf = Float64(n_beta)
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta_opt[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "deriv_SS=$(round(deriv_loss, sigdigits=5))")
    end

    PSMSolution(params, Optim.minimum(result), data_loss, edf,
                gamma_opt,
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (method=:adaptive_gradient_matching,
                 gp_hyperparams=gp_hyperparams,
                 gamma=gamma_opt,
                 deriv_loss=deriv_loss))
end
