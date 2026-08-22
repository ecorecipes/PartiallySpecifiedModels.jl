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

    # Grid search over lengthscale, signal variance, and noise level
    # (all three are optimized against the marginal likelihood; σ² was
    # previously pinned at var(y) despite the docstring saying otherwise)
    kernel in (:rbf, :matern32) ||
        error("AdaptiveGradientMatching: kernel must be :rbf or " *
              ":matern32 (got :$(kernel)); :matern52 is not implemented")
    for ℓ_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        ℓ_try = time_span * ℓ_frac
        for σ_mult in [0.5, 1.0, 2.0]
            σ²_try = σ_mult * y_var
            for σn_frac in [1e-4, 1e-3, 1e-2, 5e-2, 0.1]
                σn²_try = σn_frac * y_var
                K, _, _ = if kernel == :matern32
                    matern32_kernel_with_derivs(times, σ²_try, ℓ_try)
                else
                    rbf_kernel_with_derivs(times, σ²_try, ℓ_try)
                end
                Ky = K + σn²_try * I(n)
                try
                    C = cholesky(Symmetric(Ky))
                    α = C \ y
                    nll = 0.5 * dot(y, α) + 0.5 * logdet(C) + 0.5 * n * log(2π)
                    if nll < best_nll
                        best_nll = nll
                        best_σ² = σ²_try
                        best_ℓ = ℓ_try
                        best_σn² = σn²_try
                    end
                catch
                end
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
                                kernel::Symbol;
                                keep::Union{Nothing,AbstractVector{Int}}=nothing)
    n = length(times)

    K, Kstar, Kstarstar = if kernel == :matern32
        matern32_kernel_with_derivs(times, σ², ℓ)
    else
        rbf_kernel_with_derivs(times, σ², ℓ)
    end

    # `keep` restricts the CONDITIONING set to the usable observations while
    # the prediction grid stays the full `times`: x = K(t, t_keep) (K(t_keep,
    # t_keep) + σ_n² I)⁻¹ y_keep. Passing the raw column instead would put a
    # NaN into α and make the smoothed state, the gradient mean and the
    # gradient covariance NaN everywhere (or throw from `cholesky`/`eigen`).
    # `nothing` (or a full index set) takes the original code path unchanged.
    if keep !== nothing && length(keep) < n
        Kkk = K[keep, keep]
        C = cholesky(Symmetric(Kkk + σn² * I(length(keep)) + 1e-10 * I(length(keep))))
        α = C \ y
        x_smooth = K[:, keep] * α
        grad_mean = Kstar[:, keep] * α
        V = C.L \ Kstar[:, keep]'
        grad_cov = Kstarstar - V' * V
        grad_cov = Symmetric(grad_cov)
        eig = eigen(grad_cov)
        eig_vals = max.(eig.values, 1e-10)
        return x_smooth, grad_mean,
               Symmetric(eig.vectors * Diagonal(eig_vals) * eig.vectors')
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

# ─── Population MCMC (Dondelinger et al. 2013) ───────────────────
#
# Tempered chains at t_c = ((c−1)/(C−1))^5 jointly sample the latent
# states X, the parameters β, and the mismatch variances γ. The ODE
# product-of-experts likelihood is raised to t_c, so chain 1 is pure GP
# regression and chain C is the fully coupled model; exchange moves let
# information flow across the ladder. GP kernel hyperparameters are held
# fixed at their marginal-likelihood grid-search values.

function _agm_population_mcmc(prob::PSMProblem, alg::AdaptiveGradientMatching)
    prob.discrete && error("AdaptiveGradientMatching: population MCMC " *
                           "(n_samples > 0) requires a continuous-time problem")
    any(a -> a isa NeuralApproximator, prob.approximators) &&
        error("AdaptiveGradientMatching population MCMC does not support " *
              "NeuralApproximator; use the MAP mode or AdamSolver.")
    alg.n_chains >= 2 ||
        throw(ArgumentError("AdaptiveGradientMatching: population MCMC needs " *
                            "n_chains ≥ 2 (got $(alg.n_chains))"))
    verbose = alg.verbose
    times = Float64.(prob.data_times)
    T_pts = length(times)
    K_states = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    u0_vec = Float64.(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)
    obs_of_state = Dict{Int, Vector{Int}}()
    for j in 1:n_obs
        push!(get!(obs_of_state, prob.obs_to_state[j], Int[]), j)
    end

    # ── Fixed GP structure per state ─────────────────────────────
    m_center = zeros(K_states)
    P = Vector{Matrix{Float64}}(undef, K_states)      # prior precision K⁻¹
    Lp = Vector{Matrix{Float64}}(undef, K_states)     # chol(K).L for proposals
    D = Vector{Matrix{Float64}}(undef, K_states)      # 'K K⁻¹
    A_vals = Vector{Vector{Float64}}(undef, K_states)
    A_vecs = Vector{Matrix{Float64}}(undef, K_states)
    sigma_n2 = fill(NaN, K_states)
    gp_hyperparams = fill((0.0, 0.0, 0.0), K_states)
    x_init = zeros(T_pts, K_states)

    function mcmc_state_matrices(σ²::Float64, ℓ::Float64)
        K, Kstar, Kstarstar = alg.kernel == :matern32 ?
            matern32_kernel_with_derivs(times, σ², ℓ) :
            rbf_kernel_with_derivs(times, σ², ℓ)
        jitter = 1e-6 * σ²
        C = cholesky(Symmetric(K + jitter * I))
        Pk = Matrix(inv(C))
        Dk = Kstar * Pk
        A = Kstarstar - Kstar * (C \ Kstar') 
        E = eigen(Symmetric(0.5 * (A + A')))
        Pk, Matrix(C.L), Dk, max.(E.values, 1e-10), E.vectors
    end

    obs_states = sort(collect(keys(obs_of_state)))
    for sk in obs_states
        # Hyperparameters, center and initialization from the USABLE rows
        # only — `mean`/`optimize_gp_hyperparams` return NaN otherwise, and
        # the NaN then reaches every state matrix and the initial trajectory.
        jc1 = obs_of_state[sk][1]
        keep_sk = usable_rows(prob, jc1)
        isempty(keep_sk) && error("AdaptiveGradientMatching: observation " *
            "column $jc1 is entirely masked; state $sk has no usable data.")
        y = Float64.(prob.data_values[keep_sk, jc1])
        σ², ℓ, σn² = optimize_gp_hyperparams(times[keep_sk], y .- mean(y), alg.kernel;
                                             verbose=verbose)
        gp_hyperparams[sk] = (σ², ℓ, σn²)
        sigma_n2[sk] = σn²
        m_center[sk] = mean(y)
        P[sk], Lp[sk], D[sk], A_vals[sk], A_vecs[sk] = mcmc_state_matrices(σ², ℓ)
        # Initialise at the GP posterior mean, conditioned on the usable rows
        # and evaluated on the full grid via the cross-covariance K(t, t_keep).
        K, _, _ = alg.kernel == :matern32 ?
            matern32_kernel_with_derivs(times, σ², ℓ) :
            rbf_kernel_with_derivs(times, σ², ℓ)
        if length(keep_sk) == length(times)
            x_init[:, sk] = m_center[sk] .+
                            K * (cholesky(Symmetric(K + σn² * I)) \ (y .- m_center[sk]))
        else
            x_init[:, sk] = m_center[sk] .+
                K[:, keep_sk] * (cholesky(Symmetric(K[keep_sk, keep_sk] + σn² * I)) \
                                 (y .- m_center[sk]))
        end
    end
    ℓ_bar = mean(gp_hyperparams[sk][2] for sk in obs_states)
    σ²_bar = mean(gp_hyperparams[sk][1] for sk in obs_states)
    for k in 1:K_states
        haskey(obs_of_state, k) && continue
        gp_hyperparams[k] = (σ²_bar, ℓ_bar, 0.0)
        m_center[k] = u0_vec[k]
        P[k], Lp[k], D[k], A_vals[k], A_vecs[k] = mcmc_state_matrices(σ²_bar, ℓ_bar)
        x_init[:, k] .= u0_vec[k]
    end
    γ_scale = [max(mean(A_vals[k]), 1e-6) for k in 1:K_states]

    # ── β structure ──────────────────────────────────────────────
    beta0 = Float64[]
    for approx in prob.approximators
        append!(beta0, initial_params(approx))
    end
    n_beta = length(beta0)
    S_blocks = Tuple{UnitRange{Int}, Matrix{Float64}}[]
    let offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            S = penalty_matrix(approx)
            S !== nothing && push!(S_blocks, (offset+1:offset+np, Matrix{Float64}(S)))
            offset += np
        end
    end

    # Match the MAP path's smoothing scale (0.01 × mean mismatch variance)
    p_init = build_param_struct(prob, beta0)
    du_init = zeros(K_states)
    mm_var = zeros(K_states)
    for k in 1:K_states
        resids = zeros(T_pts)
        for i in 1:T_pts
            prob.dynamics!(du_init, x_init[i, :], p_init, times[i])
            resids[i] = du_init[k] - dot(D[k][i, :], x_init[:, k] .- m_center[k])
        end
        mm_var[k] = max(var(resids), 1e-4)
    end
    smoothing_lambda = 0.01 * mean(mm_var)
    lg0 = log.(max.(0.1 .* alg.gamma_init .* mm_var, 1e-8))

    # ── Log-density pieces ───────────────────────────────────────
    # ODE product-of-experts term (untempered; also drives exchange moves).
    # A proposal where the dynamics throw gets ode = −Inf; note t_c·(−Inf)
    # is NaN for the t=0 chain, and NaN comparisons reject — so every chain
    # (including t=0) is implicitly restricted to the ODE-evaluable support,
    # consistently across the ladder, which keeps exchange moves valid.
    function ode_loglik(X, beta, lg)
        p = build_param_struct(prob, beta)
        du = zeros(K_states)
        F = zeros(T_pts, K_states)
        for i in 1:T_pts
            try
                prob.dynamics!(du, X[i, :], p, times[i])
            catch
                return -Inf
            end
            F[i, :] .= du
        end
        ll = 0.0
        for k in 1:K_states
            γ = exp(lg[k])
            r = F[:, k] .- D[k] * (X[:, k] .- m_center[k])
            Vt_r = A_vecs[k]' * r
            shifted = A_vals[k] .+ γ
            ll += -0.5 * sum(abs2.(Vt_r) ./ shifted) - 0.5 * sum(log.(shifted))
        end
        isfinite(ll) ? ll : -Inf
    end

    agm_keep = Dict{Int,Vector{Int}}(j => usable_rows(prob, j)
                                     for j in 1:size(prob.data_values, 2))

    # Data + GP prior + γ prior (in log space, incl. Jacobian) + β smoothing.
    # Conventions (shared with the MAP path): with several data columns per
    # state, hyperparameters/σn²/init come from the first column while all
    # columns enter the data term; per-cell data_weights MAGNITUDES are not
    # applied here (AGM weights by state through sigma_n2), but the
    # zero-weight / NaN mask IS honored — see `agm_keep`.
    function base_logp(X, beta, lg)
        lp = 0.0
        for k in 1:K_states
            xc = X[:, k] .- m_center[k]
            lp -= 0.5 * dot(xc, P[k] * xc)
            if haskey(obs_of_state, k)
                for j in obs_of_state[k]
                    # Usable cells only. One NaN makes `lp` NaN, every MH
                    # test then fails, and the finiteness guard downstream
                    # aborts with a misleading "the dynamics return
                    # non-finite values" message.
                    rows = agm_keep[j]
                    if length(rows) == size(prob.data_values, 1)
                        lp -= 0.5 / sigma_n2[k] *
                              sum(abs2, prob.data_values[:, j] .- X[:, k])
                    else
                        acc = 0.0
                        for i in rows
                            acc += (prob.data_values[i, j] - X[i, k])^2
                        end
                        lp -= 0.5 / sigma_n2[k] * acc
                    end
                end
            end
            # Gamma(2, scale) prior on γ, sampled as lg = log γ:
            # log p(γ) + log|dγ/dlg| = 2 lg − exp(lg)/scale (+ const)
            lp += 2 * lg[k] - exp(lg[k]) / γ_scale[k]
        end
        for (idx, S) in S_blocks
            bk = @view beta[idx]
            lp -= smoothing_lambda * dot(bk, S * bk)
        end
        lp
    end

    # ── Chains ───────────────────────────────────────────────────
    C = alg.n_chains
    temps = [((c - 1) / (C - 1))^5 for c in 1:C]
    X_ch = [copy(x_init) for _ in 1:C]
    beta_ch = [copy(beta0) for _ in 1:C]
    lg_ch = [copy(lg0) for _ in 1:C]
    ode_ch = [ode_loglik(X_ch[c], beta_ch[c], lg_ch[c]) for c in 1:C]
    base_ch = [base_logp(X_ch[c], beta_ch[c], lg_ch[c]) for c in 1:C]
    isfinite(base_ch[C] + ode_ch[C]) ||
        error("AdaptiveGradientMatching: the target density is not finite at " *
              "the initial state — the dynamics return non-finite values on " *
              "the GP-smoothed trajectory. Check the dynamics/approximator " *
              "domains.")
    # Adaptive step sizes per chain: states (per state), β, log γ
    sx = [fill(0.05, K_states) for _ in 1:C]
    sb = fill(0.05, C)
    sg = fill(0.3, C)
    acc = zeros(4); tot = zeros(4)                 # X, β, γ, swap

    n_sweeps = 2 * alg.n_samples                   # first half is burn-in
    burnin = alg.n_samples
    beta_samples = Matrix{Float64}(undef, alg.n_samples, n_beta)
    gamma_samples = Matrix{Float64}(undef, alg.n_samples, K_states)
    X_mean = zeros(T_pts, K_states)

    if verbose
        println("AGM population MCMC: $C chains, $n_sweeps sweeps " *
                "($(burnin) burn-in), $n_beta β, $K_states states")
    end

    for sweep in 1:n_sweeps
        adapting = sweep <= burnin
        for c in 1:C
            t_c = temps[c]
            # X blocks: GP-correlated proposal per state
            for k in 1:K_states
                Xp = copy(X_ch[c])
                Xp[:, k] .+= sx[c][k] .* (Lp[k] * randn(rng, T_pts))
                ode_p = ode_loglik(Xp, beta_ch[c], lg_ch[c])
                base_p = base_logp(Xp, beta_ch[c], lg_ch[c])
                if log(rand(rng)) < (base_p + t_c * ode_p) -
                                    (base_ch[c] + t_c * ode_ch[c])
                    X_ch[c] = Xp; ode_ch[c] = ode_p; base_ch[c] = base_p
                    c == C && (acc[1] += 1)
                    adapting && (sx[c][k] *= exp(0.05))
                else
                    adapting && (sx[c][k] *= exp(-0.0167))
                end
                c == C && (tot[1] += 1)
            end
            # β block
            bp = beta_ch[c] .+ sb[c] .* randn(rng, n_beta)
            ode_p = ode_loglik(X_ch[c], bp, lg_ch[c])
            base_p = base_logp(X_ch[c], bp, lg_ch[c])
            if log(rand(rng)) < (base_p + t_c * ode_p) -
                                (base_ch[c] + t_c * ode_ch[c])
                beta_ch[c] = bp; ode_ch[c] = ode_p; base_ch[c] = base_p
                c == C && (acc[2] += 1)
                adapting && (sb[c] *= exp(0.05))
            else
                adapting && (sb[c] *= exp(-0.0167))
            end
            c == C && (tot[2] += 1)
            # log γ block
            lgp = lg_ch[c] .+ sg[c] .* randn(rng, K_states)
            ode_p = ode_loglik(X_ch[c], beta_ch[c], lgp)
            base_p = base_logp(X_ch[c], beta_ch[c], lgp)
            if log(rand(rng)) < (base_p + t_c * ode_p) -
                                (base_ch[c] + t_c * ode_ch[c])
                lg_ch[c] = lgp; ode_ch[c] = ode_p; base_ch[c] = base_p
                c == C && (acc[3] += 1)
                adapting && (sg[c] *= exp(0.05))
            else
                adapting && (sg[c] *= exp(-0.0167))
            end
            c == C && (tot[3] += 1)
        end
        # Exchange move between a random adjacent pair (the untempered ODE
        # term is the only tempered piece, so it alone enters the swap ratio)
        c = rand(rng, 1:(C - 1))
        Δ = (temps[c] - temps[c + 1]) * (ode_ch[c + 1] - ode_ch[c])
        tot[4] += 1
        if log(rand(rng)) < Δ
            acc[4] += 1
            X_ch[c], X_ch[c + 1] = X_ch[c + 1], X_ch[c]
            beta_ch[c], beta_ch[c + 1] = beta_ch[c + 1], beta_ch[c]
            lg_ch[c], lg_ch[c + 1] = lg_ch[c + 1], lg_ch[c]
            ode_ch[c], ode_ch[c + 1] = ode_ch[c + 1], ode_ch[c]
            base_ch[c], base_ch[c + 1] = base_ch[c + 1], base_ch[c]
        end
        if sweep > burnin
            s_idx = sweep - burnin
            beta_samples[s_idx, :] = beta_ch[C]
            gamma_samples[s_idx, :] = exp.(lg_ch[C])
            X_mean .+= X_ch[C]
        end
        if verbose && sweep % max(1, n_sweeps ÷ 10) == 0
            println("  sweep $sweep/$n_sweeps  cold-chain logp=" *
                    "$(round(base_ch[C] + ode_ch[C], sigdigits=6))")
        end
    end
    X_mean ./= alg.n_samples
    beta_mean = vec(mean(beta_samples, dims=1))
    gamma_mean = vec(mean(gamma_samples, dims=1))

    if verbose
        println("  acceptance: X=$(round(acc[1]/tot[1], digits=2)) " *
                "β=$(round(acc[2]/tot[2], digits=2)) " *
                "γ=$(round(acc[3]/tot[3], digits=2)) " *
                "swap=$(round(acc[4]/tot[4], digits=2))")
    end

    # ── Build solution from the cold chain ───────────────────────
    pred = zeros(T_pts, n_obs)
    for j in 1:n_obs
        pred[:, j] .= X_mean[:, prob.obs_to_state[j]]
    end
    data_loss = weighted_data_loss(prob, pred)

    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_mean[offset+1:offset+np]
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
        end
    end

    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta_mean[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    PSMSolution(params, -(base_ch[C] + ode_ch[C]), data_loss,
                Float64(n_beta), gamma_mean,
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (method=:adaptive_gradient_matching,
                 sampler=:population_mcmc,
                 gp_hyperparams=gp_hyperparams,
                 gamma=gamma_mean,
                 beta_samples=beta_samples,
                 gamma_samples=gamma_samples,
                 n_chains=C,
                 temperatures=temps,
                 accept_rates=(x=acc[1]/max(tot[1],1), beta=acc[2]/max(tot[2],1),
                               gamma=acc[3]/max(tot[3],1), swap=acc[4]/max(tot[4],1))))
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

With `n_samples > 0` the MAP fit is replaced by the paper's tempered
population-MCMC sampler over states, parameters, and mismatch variances
(see the `AdaptiveGradientMatching` docstring).

# References
- Dondelinger et al. (2013), "ODE parameter inference using adaptive
  gradient matching with Gaussian processes", AISTATS.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::AdaptiveGradientMatching)
    _validate_problem(prob, "AdaptiveGradientMatching")
    alg.n_samples > 0 && return _agm_population_mcmc(prob, alg)
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

    # Initialize so unobserved states report zeros rather than undef garbage
    gp_hyperparams = fill((0.0, 0.0, 0.0), K_states)

    if prob.discrete
        # For discrete models: GP-smooth the states, and use forward-shifted
        # smoothed values as "gradient targets" (i.e., match f(x[t]) to x[t+1])
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            # Fit the GP on the USABLE rows only (see `_usable`): `mean`,
            # `optimize_gp_hyperparams` and `gp_gradient_inference` are all
            # NaN-poisoned by a single masked cell, and the results are then
            # NaN for every time point. `keep_j` is every row for complete
            # data, so this path is unchanged there.
            keep_j = usable_rows(prob, j)
            isempty(keep_j) && error("AdaptiveGradientMatching: observation " *
                "column $j is entirely masked; there is nothing to smooth.")
            y = Float64.(prob.data_values[keep_j, j])
            times_j = times[keep_j]

            # Center the data before GP fitting (zero-mean GP prior); add the
            # mean back to the smoothed states. Without centering, data with
            # large mean levels get shrunk toward 0 (cf. the MCMC path).
            ȳ = mean(y)
            σ², ℓ, σn² = optimize_gp_hyperparams(times_j, y .- ȳ, alg.kernel; verbose=verbose)
            gp_hyperparams[sk] = (σ², ℓ, σn²)

            xs, _, gc = gp_gradient_inference(times, y .- ȳ, σ², ℓ, σn²,
                                              alg.kernel; keep=keep_j)
            xs = xs .+ ȳ
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
            # Fit the GP on the USABLE rows only (see `_usable`): `mean`,
            # `optimize_gp_hyperparams` and `gp_gradient_inference` are all
            # NaN-poisoned by a single masked cell, and the results are then
            # NaN for every time point. `keep_j` is every row for complete
            # data, so this path is unchanged there.
            keep_j = usable_rows(prob, j)
            isempty(keep_j) && error("AdaptiveGradientMatching: observation " *
                "column $j is entirely masked; there is nothing to smooth.")
            y = Float64.(prob.data_values[keep_j, j])
            times_j = times[keep_j]

            # Center the data before GP fitting (zero-mean GP prior); add the
            # mean back to the smoothed states. Derivative estimates need no
            # offset (d/dt of a constant is zero). Cf. the MCMC path, which
            # centers via m_center[sk].
            ȳ = mean(y)
            σ², ℓ, σn² = optimize_gp_hyperparams(times_j, y .- ȳ, alg.kernel; verbose=verbose)
            gp_hyperparams[sk] = (σ², ℓ, σn²)

            xs, gm, gc = gp_gradient_inference(times, y .- ȳ, σ², ℓ, σn²,
                                               alg.kernel; keep=keep_j)
            x_smooth[:, sk] .= xs .+ ȳ
            grad_means[:, sk] .= gm
            grad_covs[sk] = gc
        end
    end

    # For unobserved states, initialize from u0 (constant)
    for k in 1:K_states
        if !isassigned(grad_covs, k)
            x_smooth[:, k] .= (prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)[k]
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
    # gamma_init scales the data-driven default (0.1·Var of the GP-implied
    # derivative mismatch); the field was previously accepted but ignored.
    gamma_init_vals = 0.1 .* alg.gamma_init .* mismatch_var
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

    data_loss = weighted_data_loss(prob, pred)

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
            uf_evals[approx.name] = build_neural_evaluator(approx, params_k)
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
