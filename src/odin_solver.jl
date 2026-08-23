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
per-state `(σ², ℓ, σ_n²)`; `sol.convergence` also carries the honest
convergence keys `(converged, iterations, reason)` — see the `ODINSolver`
docstring.
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
        # Stage 1 (GP smoothing) must see only USABLE rows: `mean` over a
        # column holding a NaN is NaN, and that NaN propagates through the
        # hyperparameter search, the state initialization and every matrix
        # derived from it. `keep_k` is every row for complete data, so the
        # values below are then unchanged.
        jc = obs_of_state[sk][1]
        keep_k = usable_rows(prob, jc)
        isempty(keep_k) && error("ODINSolver: observation column $jc is " *
            "entirely masked; state $sk has no usable data.")
        y_k = Float64.(prob.data_values[keep_k, jc])
        m_center[sk] = mean(y_k)
        yc = y_k .- m_center[sk]
        times_k = times[keep_k]
        # Fixed-hyperparameter path assumes 1% observation noise; with
        # several data columns per state, hyperparameters and the center
        # come from the first column (replicates enter only the data term).
        σ², ℓ, σn² = if alg.gp_lengthscale !== nothing && alg.gp_variance !== nothing
            (alg.gp_variance, alg.gp_lengthscale, 0.01 * alg.gp_variance)
        else
            optimize_gp_hyperparams(times_k, yc, :rbf; verbose=verbose)
        end
        hyper[sk] = (σ²=σ², ℓ=ℓ, σn²=σn², observed=true)
        data_w[sk] = 1.0 / σn²
        P[sk], D[sk], Ainv[sk] = state_matrices(σ², ℓ, σn²)
        # Initialise at the GP posterior mean, conditioned on the usable rows
        # and evaluated on the FULL time grid via the cross-covariance
        # K(t, t_keep). Reduces to the previous K(t,t)-based expression when
        # every row is usable.
        K_full, _, _ = rbf_kernel_with_derivs(times, σ², ℓ)
        if length(keep_k) == n_times
            # Unmasked: the original expression, unchanged bit-for-bit.
            x_init[:, sk] = m_center[sk] .+
                            K_full * (cholesky(Symmetric(K_full + σn² * I)) \ yc)
        else
            K_cross = K_full[:, keep_k]
            K_kk = K_full[keep_k, keep_k]
            x_init[:, sk] = m_center[sk] .+
                            K_cross * (cholesky(Symmetric(K_kk + σn² * I)) \ yc)
        end
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

    for approx in prob.approximators
        if approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(beta, neural_init_params(approx, rng))
        else
            append!(beta, initial_params(approx))
        end
    end
    n_beta = length(beta)
    # Usable row indices per observation column, precomputed once for the
    # risk closure (see the data term inside `odin_risk`).
    data_rows = Dict{Int,Vector{Int}}(j => usable_rows(prob, j)
                                      for j in 1:size(prob.data_values, 2))
    any(!isempty, values(data_rows)) || error("ODINSolver: every " *
        "observation is masked; there is nothing to fit.")

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
                    # Accumulate over usable rows only. The whole-column form
                    # `sum(abs2, data[:, j] .- X[:, k])` turns NaN on one
                    # masked cell, and there is no finiteness sentinel in this
                    # risk: the NaN reaches the Adam moments, every parameter
                    # becomes NaN, `loss_val < best_loss` is false forever, and
                    # the solver returns its initialization as if converged.
                    # (Per-cell `data_weights` magnitudes remain unapplied here
                    # — ODIN weights by state via `data_w[k] = 1/σn²`; only the
                    # zero/NaN mask is honored.)
                    rows = data_rows[j]
                    if length(rows) == n_times
                        # Unmasked: the original vectorized expression, whose
                        # pairwise summation order this loop would otherwise
                        # perturb at the 1e-16 level.
                        r = @view(prob.data_values[:, j]) .- @view(X[:, k])
                        loss += data_w[k] * sum(abs2, r)
                    else
                        acc = zero(eltype(X))
                        for i in rows
                            acc += (prob.data_values[i, j] - X[i, k])^2
                        end
                        loss += data_w[k] * acc
                    end
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
    # Honest convergence reporting: objective-plateau check following the
    # gradient-matching family convention (30-step window, relative range
    # < 1e-6 after step 60). Defaults describe loop exhaustion.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    loss_window = fill(Inf, 30)
    for step in 1:n_total
        conv_iters = step
        ForwardDiff.gradient!(result, odin_risk, z)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)
        loss_window[mod1(step, 30)] = loss_val
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
        # Plateau convergence, guarded against the cosine lr schedule
        # manufacturing a plateau as lr_t → 0 near n_total.
        # `best_loss < 1e9` is AdamSolver's failure-sentinel guard: the
        # dynamics fall back to `du .= 1e6` when the RHS throws, so a run
        # pinned at that sentinel is a stuck solver, not a converged one.
        if step > 60 && best_loss < 1e9 && lr_t > 0.05 * lr
            rmin, rmax = extrema(loss_window)
            if (rmax - rmin) / max(abs(rmin), 1.0) < 1e-6
                if verbose; println("  Converged at step $step (loss plateau)"); end
                conv_converged = true
                conv_reason = :plateau
                break
            end
        end
    end
    X_fit = reshape(best_z[1:n_state], n_times, n_vars)
    beta = best_z[n_state+1:end]

    # ── Build solution ───────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        pred[:, j] .= X_fit[:, prob.obs_to_state[j]]
    end

    # Masked cells are skipped — `0 * NaN = NaN` would otherwise make the
    # reported loss NaN. Every other solver reports via this helper.
    data_loss = weighted_data_loss(prob, pred)

    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
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
                (converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, method=:odin,
                 gp_hyperparams=[hyper[k] for k in 1:n_vars]))
end
