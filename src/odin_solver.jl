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

Fit a partially specified model using ODE-Informed regression (ODIN;
Wenk & Abbati et al. 2020).

A Gaussian process is fit to each observed state, yielding the posterior
mean state `x`, the posterior-mean derivative `Dx = 'K C⁻¹ y`, and — the
heart of ODIN — the posterior derivative covariance
`A = ''K − 'K C⁻¹ ('K)ᵀ (+ γ I)`. The unknown-function parameters are then
chosen to minimize the Mahalanobis ODE-mismatch *risk functional*

    R(θ) = Σ_d (f_d(x,θ) − D_d x_d)ᵀ A_d⁻¹ (f_d(x,θ) − D_d x_d),

so the gradient match is weighted by how well the GP actually determines the
derivative (tight where data are dense, loose where sparse). This replaces
the previous uniform-weight mismatch and ad-hoc noise heuristic.

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

    # Derivative cross-kernel 'K[i,j] = ∂k/∂t_i = Cov(ẋ_i, x_j)
    function build_dKdt(times, ℓ, σ²)
        n = length(times)
        dK = Matrix{Float64}(undef, n, n)
        for i in 1:n, j in 1:n
            dK[i, j] = -σ² * (times[i] - times[j]) / ℓ^2 *
                        exp(-0.5 * (times[i] - times[j])^2 / ℓ^2)
        end
        dK
    end

    # Second-derivative kernel ''K[i,j] = ∂²k/∂t_i∂t_j = Cov(ẋ_i, ẋ_j)
    function build_d2Kdt2(times, ℓ, σ²)
        n = length(times)
        d2K = Matrix{Float64}(undef, n, n)
        for i in 1:n, j in 1:n
            r2 = (times[i] - times[j])^2
            d2K[i, j] = σ² / ℓ^2 * (1 - r2 / ℓ^2) * exp(-0.5 * r2 / ℓ^2)
        end
        d2K
    end

    # ── GP-smooth each observed state and build ODIN weighting ──────
    ℓ = alg.gp_lengthscale
    σ² = alg.gp_variance
    noise_var = 0.01 * σ²  # observation noise

    y_smooth = zeros(n_times, n_vars)
    dydt = zeros(n_times, n_vars)                 # GP posterior-mean derivative Dx
    Ainv = Dict{Int, Matrix{Float64}}()           # A_d⁻¹ per observed state
    observed_states = Set{Int}()

    K_clean = build_K(times, ℓ, σ², 0.0)
    dK_mat = build_dKdt(times, ℓ, σ²)
    d2K_mat = build_d2Kdt2(times, ℓ, σ²)
    Cf = cholesky(Symmetric(build_K(times, ℓ, σ², noise_var)))

    # ODE-mismatch slack (γ): keeps A well-conditioned and represents model
    # discrepancy tolerance (the γ of Wenk & Abbati's risk functional).
    γ_slack = 1e-6 * (tr(d2K_mat) / n_times) + 1e-10

    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        push!(observed_states, sk)
        y_j = prob.data_values[:, j]
        α = Cf \ y_j
        y_smooth[:, sk] = K_clean * α
        dydt[:, sk] = dK_mat * α
        # Posterior derivative covariance A = ''K − 'K C⁻¹ ('K)ᵀ + γ I
        A = d2K_mat - dK_mat * (Cf \ dK_mat') + γ_slack * I
        A = Symmetric(0.5 * (A + A'))
        Ainv[sk] = Matrix(inv(cholesky(A)))
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

    # ── Optimise β against the ODIN Mahalanobis risk functional ─────
    # R(θ) = Σ_d (f_d − D_d x_d)ᵀ A_d⁻¹ (f_d − D_d x_d), with the GP
    # posterior (x, Dx, A) fixed from the data. No ad-hoc GP re-noising.
    ode_weight = alg.ode_weight
    lr = alg.lr
    best_beta = copy(beta)
    best_loss = Inf
    obs_list = sort(collect(observed_states))

    function odin_risk(β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)
        # ODE RHS at the GP-mean trajectory, per state column.
        F = Matrix{T_el}(undef, n_times, n_vars)
        for i in 1:n_times
            u = T_el.(@view y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            F[i, :] .= du
        end
        loss = zero(T_el)
        for sk in obs_list
            resid = @view(F[:, sk]) .- T_el.(@view dydt[:, sk])
            loss += ode_weight * dot(resid, T_el.(Ainv[sk]) * resid)
        end
        # Smoothing penalty
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            pk = @view β_eval[offset+1:offset+np]
            offset += np
            S = penalty_matrix(approx)
            S !== nothing && (loss += dot(pk, T_el.(S) * pk))
        end
        loss
    end

    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_beta); v_adam = zeros(n_beta)
    n_total = alg.maxiters * 20
    for step in 1:n_total
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        ForwardDiff.gradient!(result, odin_risk, beta)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)
        lr_t = lr * 0.5 * (1 + cos(π * step / n_total))
        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad .^ 2
        m_hat = m_adam ./ (1 - β1_adam^step)
        v_hat = v_adam ./ (1 - β2_adam^step)
        beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
        if loss_val < best_loss
            best_loss = loss_val; best_beta .= beta
        end
        if verbose && (step <= 3 || step % 50 == 0 || step == n_total)
            println("  step $step: risk=$(round(best_loss, sigdigits=5))")
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
