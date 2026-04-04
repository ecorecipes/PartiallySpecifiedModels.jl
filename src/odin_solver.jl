# ─── ODIN solver (ODE-Informed regression) ─────────────────────────
#
# GP smoothing with ODE-informed structure: iterates between fitting
# a Gaussian process to states and optimising unknown-function parameters
# via the ODE mismatch on the GP posterior mean.
#
# Reference: Wenk, Abbati et al. (2020), AAAI — ODIN
#            Wenk et al. (2019), AISTATS — FGPGM

using LinearAlgebra: dot, norm, Symmetric, cholesky, logdet, I

"""
    solve(prob::PSMProblem, alg::ODINSolver)

Fit a partially specified model using ODE-Informed regression (ODIN).

Alternates between:
1. **GP step**: Fit a Gaussian process to each observed state using an
   RBF kernel, with the ODE residual as an additional penalty on the
   GP marginal likelihood.
2. **ODE step**: Optimise unknown-function parameters β to minimise the
   ODE mismatch at the GP mean trajectory.

# Returns
`PSMSolution` with fitted parameters, GP-smoothed trajectory, and
unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::ODINSolver)
    _validate_problem(prob, "ODINSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_obs = size(prob.data_values, 2)

    if verbose; println("ODINSolver: $n_obs observed states, $n_times time points"); end

    # ── Build RBF kernel matrix ──────────────────────────────────
    function rbf_kernel(t1, t2, ℓ, σ²)
        σ² * exp(-0.5 * (t1 - t2)^2 / ℓ^2)
    end

    function build_K(times, ℓ, σ², noise_var)
        n = length(times)
        K = Matrix{Float64}(undef, n, n)
        for i in 1:n, j in 1:n
            K[i, j] = rbf_kernel(times[i], times[j], ℓ, σ²)
        end
        K + noise_var * I
    end

    # Build derivative kernel: ∂K/∂t₂ for gradient matching
    function build_dKdt(times, ℓ, σ²)
        n = length(times)
        dK = Matrix{Float64}(undef, n, n)
        for i in 1:n, j in 1:n
            dK[i, j] = -σ² * (times[i] - times[j]) / ℓ^2 *
                        exp(-0.5 * (times[i] - times[j])^2 / ℓ^2)
        end
        dK
    end

    # ── GP-smooth each observed state ────────────────────────────
    ℓ = alg.gp_lengthscale
    σ² = alg.gp_variance
    noise_var = 0.01 * σ²  # observation noise

    y_smooth = zeros(n_times, n_vars)
    dydt = zeros(n_times, n_vars)
    observed_states = Set{Int}()

    K_mat = build_K(times, ℓ, σ², noise_var)
    K_inv = inv(Symmetric(K_mat))
    dK_mat = build_dKdt(times, ℓ, σ²)

    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        push!(observed_states, sk)
        y_j = prob.data_values[:, j]

        # GP posterior mean: μ = K * K⁻¹ * y = K_clean * K_noisy⁻¹ * y
        alpha_gp = K_inv * y_j
        K_clean = build_K(times, ℓ, σ², 0.0)  # no noise
        y_smooth[:, sk] = K_clean * alpha_gp

        # GP derivative: dμ/dt = dK/dt * K⁻¹ * y
        dydt[:, sk] = dK_mat * alpha_gp
    end

    # Unobserved states: hold at IC
    for k in 1:n_vars
        if k ∉ observed_states
            u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                           prob.u0[k])
            y_smooth[:, k] .= u0_k
        end
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

    if verbose
        println("  $n_beta unknown-function parameters, $(alg.maxiters) outer iterations")
    end

    # ── Alternating optimisation ─────────────────────────────────
    ode_weight = alg.ode_weight
    lr = alg.lr
    best_beta = copy(beta)
    best_loss = Inf

    for outer in 1:alg.maxiters
        # ── ODE step: optimise β to match GP derivatives ─────────
        # Adam inner loop (20 steps per outer iteration)
        β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
        m_adam = zeros(n_beta)
        v_adam = zeros(n_beta)
        n_inner = 20

        function odin_loss(β_eval)
            T_el = eltype(β_eval)
            p = build_autodiff_param_struct(prob, β_eval)
            du = zeros(T_el, n_vars)
            loss = zero(T_el)

            for i in 1:n_times
                u = T_el.(y_smooth[i, :])
                try
                    prob.dynamics!(du, u, p, times[i])
                catch
                    du .= T_el(1e6)
                end
                for k in 1:n_vars
                    loss += ode_weight * (dydt[i, k] - du[k])^2
                end
            end

            # Smoothing penalty
            offset = 0
            for approx in prob.approximators
                np = nparams(approx)
                pk = β_eval[offset+1:offset+np]
                offset += np
                if approx isa BSplineApproximator || approx isa GPApproximator ||
                   approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator
                    S = penalty_matrix(approx)
                    if S !== nothing; loss += dot(pk, S * pk); end
                elseif approx isa ShapeConstrainedBSplineApproximator
                    S = penalty_matrix(approx)
                    if S !== nothing; loss += dot(pk, S * pk); end
                end
            end
            loss
        end

        for inner in 1:n_inner
            result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
            ForwardDiff.gradient!(result, odin_loss, beta)
            loss_val = DiffResults.value(result)
            grad = DiffResults.gradient(result)

            step = outer * n_inner + inner
            lr_t = lr * 0.5 * (1 + cos(π * step / (alg.maxiters * n_inner)))

            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
            m_hat = m_adam ./ (1 - β1_adam^(inner))
            v_hat = v_adam ./ (1 - β2_adam^(inner))
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

            if loss_val < best_loss
                best_loss = loss_val
                best_beta .= beta
            end
        end

        # ── GP step: re-smooth with ODE-informed penalty ─────────
        # Update GP with ODE residual as additional penalty
        p_cur = build_param_struct(prob, beta)
        du_tmp = zeros(n_vars)

        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            y_j = prob.data_values[:, j]

            # ODE residuals at current β
            ode_resid = zeros(n_times)
            for i in 1:n_times
                u = y_smooth[i, :]
                prob.dynamics!(du_tmp, u, p_cur, times[i])
                ode_resid[i] = dydt[i, sk] - du_tmp[sk]
            end

            # Adjust GP: increase effective noise where ODE residual is small
            # (trust ODE more) — simple heuristic: tighten noise_var
            residual_scale = mean(abs2, ode_resid)
            adjusted_noise = max(noise_var * min(residual_scale / max(σ², 1e-10), 1.0), 1e-8)

            K_adj = build_K(times, ℓ, σ², adjusted_noise)
            K_adj_inv = inv(Symmetric(K_adj))
            alpha_gp = K_adj_inv * y_j
            K_clean = build_K(times, ℓ, σ², 0.0)
            y_smooth[:, sk] = K_clean * alpha_gp
            dydt[:, sk] = dK_mat * alpha_gp
        end

        if verbose && (outer <= 3 || outer % 10 == 0 || outer == alg.maxiters)
            println("  outer $outer: loss=$(round(best_loss, sigdigits=5))")
        end
    end
    beta .= best_beta

    # ── Build solution ───────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= y_smooth[:, sk]
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
                 gp_lengthscale=ℓ, gp_variance=σ²))
end
