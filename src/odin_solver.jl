# ─── ODIN solver (ODE-Informed regression) ─────────────────────────
#
# Joint state-and-parameter optimisation under the ODIN risk functional:
# GP hyperparameters are pre-trained per state by marginal likelihood,
# then states X and unknown-function parameters θ are optimised together.
#
# Reference: Wenk, Abbati et al. (2020), AAAI — ODIN
#            Wenk et al. (2019), AISTATS — FGPGM

using LinearAlgebra: dot, norm, Symmetric, cholesky, logdet, tr, I

"""
    solve(prob::PSMProblem, alg::ODINSolver)

Fit a partially specified model using ODE-Informed regression (ODIN;
Wenk & Abbati et al. 2020).

Stage 1 — per observed state, GP hyperparameters `(σ², ℓ, σ_n²)` are
estimated by maximising the GP marginal likelihood of the (centered)
data (or taken from the solver if supplied). From these, three fixed
matrices are built at the data times: the prior precision `K⁻¹`, the
derivative map `D = 'K K⁻¹` (states ↦ GP conditional-mean derivative),
and the conditional derivative covariance `A = ''K − 'K K⁻¹ 'Kᵀ + γI`.

Stage 2 — the states `X` (all of them, including unobserved) and the
unknown-function parameters `θ` are optimised **jointly** by Adam on

    R(X, θ) = Σ_k [ ‖y_k − x_k‖²/σ_{n,k}²      (observed states only)
                    + x̃_kᵀ K_k⁻¹ x̃_k            (GP prior, x̃ centered)
                    + (f_k(X,θ) − D_k x̃_k)ᵀ A_k⁻¹ (f_k(X,θ) − D_k x̃_k) ]
                    + θ smoothing penalty,

so the ODE mismatch is weighted by how well the GP determines the
derivative (tight where data are dense, loose where sparse), and states
may move away from the GP posterior mean when the ODE demands it.
Unobserved states carry a GP prior with hyperparameters borrowed from
the observed states (mean lengthscale/variance), centered at their
initial condition, and are identified through the ODE terms alone.

# Returns
`PSMSolution` with fitted parameters, the jointly optimised trajectory,
and unknown functions. `sol.convergence.gp_hyperparams` records the
per-state `(σ², ℓ, σ_n²)`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::ODINSolver)
    _validate_problem(prob, "ODINSolver"; require_continuous=true)
    if (alg.gp_lengthscale === nothing) != (alg.gp_variance === nothing)
        throw(ArgumentError(
            "ODINSolver: supply BOTH gp_lengthscale and gp_variance to fix " *
            "the GP hyperparameters, or neither to estimate them per state " *
            "by marginal likelihood."))
    end
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    u0_vec = Float64.(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_vars = length(u0_vec)
    n_obs = size(prob.data_values, 2)
    obs_of_state = Dict{Int, Vector{Int}}()   # state ↦ data columns
    for j in 1:n_obs
        push!(get!(obs_of_state, prob.obs_to_state[j], Int[]), j)
    end

    if verbose; println("ODINSolver: $n_obs observed states, $n_times time points"); end

    # ── Stage 1: GP hyperparameters and fixed matrices per state ────
    # For each state k we need: center m_k, prior precision P_k = K⁻¹,
    # derivative map D_k = 'K K⁻¹, mismatch precision Ainv_k, and (for
    # observed states) the data weight 1/σ_n².
    m_center = zeros(n_vars)
    P = Vector{Matrix{Float64}}(undef, n_vars)
    D = Vector{Matrix{Float64}}(undef, n_vars)
    Ainv = Vector{Matrix{Float64}}(undef, n_vars)
    data_w = zeros(n_vars)                    # 1/σ_n² (0 for unobserved)
    hyper = Vector{NamedTuple}(undef, n_vars)
    x_init = zeros(n_times, n_vars)

    function state_matrices(σ²::Float64, ℓ::Float64, σn²::Float64)
        K, dK, d2K = rbf_kernel_with_derivs(times, σ², ℓ)
        # Noiseless RBF Grams are severely ill-conditioned; regularize with
        # a small jitter before factorizing.
        jitter = 1e-6 * σ²
        C = cholesky(Symmetric(K + jitter * I))
        Pk = Matrix(inv(C))
        Dk = dK * Pk
        # γ slack (the γ of Wenk & Abbati's risk functional): model-mismatch
        # tolerance. σn²/ℓ² is the derivative-scale noise induced by
        # observation noise over one lengthscale — without it the mismatch
        # precision dwarfs the data precision and the states abandon the
        # data to satisfy f = Dx̃ exactly.
        γ = σn² / ℓ^2 + 1e-6 * (tr(d2K) / n_times) + 1e-10
        A = d2K - dK * (C \ dK') + γ * I
        Ak_inv = Matrix(inv(cholesky(Symmetric(0.5 * (A + A')))))
        Pk, Dk, Ak_inv
    end

    obs_states = sort(collect(keys(obs_of_state)))
    for sk in obs_states
        y_k = prob.data_values[:, obs_of_state[sk][1]]
        m_center[sk] = mean(y_k)
        yc = y_k .- m_center[sk]
        # Fixed-hyperparameter path assumes 1% observation noise; with
        # several data columns per state, hyperparameters and the center
        # come from the first column (replicates enter only the data term).
        σ², ℓ, σn² = if alg.gp_lengthscale !== nothing && alg.gp_variance !== nothing
            (alg.gp_variance, alg.gp_lengthscale, 0.01 * alg.gp_variance)
        else
            optimize_gp_hyperparams(times, yc, :rbf; verbose=verbose)
        end
        hyper[sk] = (σ²=σ², ℓ=ℓ, σn²=σn², observed=true)
        data_w[sk] = 1.0 / σn²
        P[sk], D[sk], Ainv[sk] = state_matrices(σ², ℓ, σn²)
        # Initialise at the GP posterior mean
        K, _, _ = rbf_kernel_with_derivs(times, σ², ℓ)
        x_init[:, sk] = m_center[sk] .+ K * (cholesky(Symmetric(K + σn² * I)) \ yc)
    end

    # Unobserved states: free variables with a borrowed GP prior centered
    # at the initial condition; identified through the ODE terms.
    ℓ_bar = mean(hyper[sk].ℓ for sk in obs_states)
    σ²_bar = mean(hyper[sk].σ² for sk in obs_states)
    σn²_bar = mean(hyper[sk].σn² for sk in obs_states)
    for k in 1:n_vars
        haskey(obs_of_state, k) && continue
        m_center[k] = u0_vec[k]
        hyper[k] = (σ²=σ²_bar, ℓ=ℓ_bar, σn²=σn²_bar, observed=false)
        P[k], D[k], Ainv[k] = state_matrices(σ²_bar, ℓ_bar, σn²_bar)
        x_init[:, k] .= u0_vec[k]
    end

    # ── Initialise unknown-function parameters ───────────────────
    beta = Float64[]
    mlp_specs = Dict{Symbol, MLPSpec}()

    for approx in prob.approximators
        if approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            mlp_specs[approx.name] = spec
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta, init_mlp_params(spec, rng))
        else
            append!(beta, initial_params(approx))
        end
    end
    n_beta = length(beta)
    n_state = n_times * n_vars

    if verbose
        println("  joint optimisation over $n_state state values + " *
                "$n_beta unknown-function parameters")
    end

    # ── Stage 2: joint Adam on z = [vec(X); β] ──────────────────────
    ode_weight = alg.ode_weight
    lr = alg.lr

    function odin_risk(z)
        T_el = eltype(z)
        X = reshape(@view(z[1:n_state]), n_times, n_vars)
        β_eval = @view z[n_state+1:end]
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)
        F = Matrix{T_el}(undef, n_times, n_vars)   # ODE RHS along X
        for i in 1:n_times
            u = Vector{T_el}(@view X[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            F[i, :] .= du
        end
        loss = zero(T_el)
        for k in 1:n_vars
            xc = @view(X[:, k]) .- m_center[k]
            loss += dot(xc, P[k] * xc)                              # GP prior
            resid = @view(F[:, k]) .- D[k] * xc                     # ODE mismatch
            loss += ode_weight * dot(resid, Ainv[k] * resid)
            if haskey(obs_of_state, k)                              # data terms
                for j in obs_of_state[k]
                    r = @view(prob.data_values[:, j]) .- @view(X[:, k])
                    loss += data_w[k] * sum(abs2, r)
                end
            end
        end
        # Smoothing penalty on the unknown-function parameters
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            pk = @view β_eval[offset+1:offset+np]
            offset += np
            S = penalty_matrix(approx)
            S !== nothing && (loss += dot(pk, S * pk))
        end
        loss
    end

    z = vcat(vec(x_init), beta)
    n_z = length(z)
    best_z = copy(z)
    best_loss = Inf
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_z); v_adam = zeros(n_z)
    n_total = alg.maxiters * 20
    result = DiffResults.MutableDiffResult(0.0, (zeros(n_z),))
    for step in 1:n_total
        ForwardDiff.gradient!(result, odin_risk, z)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)
        lr_t = lr * 0.5 * (1 + cos(π * step / n_total))
        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad .^ 2
        # Record the incumbent BEFORE stepping: loss_val is the risk at the
        # pre-step z, so best_z must be captured at that same point.
        if loss_val < best_loss
            best_loss = loss_val; best_z .= z
        end
        m_hat = m_adam ./ (1 - β1_adam^step)
        v_hat = v_adam ./ (1 - β2_adam^step)
        z .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
        if verbose && (step <= 3 || step % 100 == 0 || step == n_total)
            println("  step $step: risk=$(round(loss_val, sigdigits=5))")
        end
    end
    X_fit = reshape(best_z[1:n_state], n_times, n_vars)
    beta = best_z[n_state+1:end]

    # ── Build solution ───────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        pred[:, j] .= X_fit[:, prob.obs_to_state[j]]
    end

    data_loss = sum(abs2, prob.data_values .- pred)

    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            spec = mlp_specs[approx.name]
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
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    edf = Float64(n_beta)

    PSMSolution(params, best_loss, data_loss, edf, Float64[ode_weight],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=alg.maxiters, method=:odin,
                 gp_hyperparams=[hyper[k] for k in 1:n_vars]))
end
