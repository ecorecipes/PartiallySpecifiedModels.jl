# ─── Integral Matching solver (Dattner & Klaassen 2015) ──────────────
#
# Instead of matching derivatives (noisy) or integrating the ODE (expensive),
# integrate both sides of the ODE:
#   x(t) - x(t₀) = ∫_{t₀}^{t} f(x(s), p, s) ds
#
# Algorithm:
#   Stage 1 — Smooth observed states with cubic splines → ŷ(t)
#   Stage 2 — For each time point tᵢ, compute:
#             LHS: ŷ(tᵢ) - ŷ(t₁)           (from smoothed data)
#             RHS: ∫_{t₁}^{tᵢ} f(ŷ(s), p(β), s) ds   (trapezoidal rule)
#             Minimize Σ_k Σ_i ||LHS_k(tᵢ) - RHS_k(tᵢ)||²
#
# Key advantage: avoids both derivative estimation AND full ODE integration.
# More robust than gradient matching because integrals smooth out noise.
#
# Reference: Dattner & Klaassen (2015), EJS 9(2), 1939-1973
#            R package `simode` (Yaari & Dattner)

using LinearAlgebra: dot, norm

"""
    solve(prob::PSMProblem, alg::IntegralMatchingSolver)

Fit a partially specified model by integral matching (Dattner & Klaassen 2015).

Instead of matching noisy derivatives, this method integrates both sides of the
ODE, comparing the smoothed trajectory increments with numerical quadrature of
the right-hand side. This avoids both derivative estimation and repeated ODE
integration, providing a robust and computationally efficient estimator.

# Algorithm
1. Smooth each observed state with cubic splines → ŷ(t).
2. Compute trajectory increments: Δᵢₖ = ŷₖ(tᵢ) − ŷₖ(t₁).
3. Compute cumulative integrals of the ODE RHS using the trapezoidal rule
   evaluated at the smoothed states: Iᵢₖ = ∫₁ⁱ fₖ(ŷ(s), p(β), s) ds.
4. Minimize L(β) = Σₖ Σᵢ (Δᵢₖ − Iᵢₖ)² + λ β'Sβ w.r.t. β using Adam.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::IntegralMatchingSolver)
    _validate_problem(prob, "IntegralMatchingSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_obs = size(prob.data_values, 2)

    # ── Stage 1: Smooth data ─────────────────────────────────────
    if verbose; println("IntegralMatchingSolver Stage 1: Smoothing data..."); end

    y_smooth = zeros(n_times, n_vars)
    observed_states = Set{Int}()

    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        push!(observed_states, sk)
        itp = CubicSpline(prob.data_values[:, j], times;
                          extrapolation=ExtrapolationType.Extension)
        for i in 1:n_times
            y_smooth[i, sk] = itp(times[i])
        end
    end

    # Unobserved states: hold at initial condition
    for k in 1:n_vars
        if k ∉ observed_states
            u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                           prob.u0[k])
            y_smooth[:, k] .= u0_k
        end
    end

    # Trajectory increments: Δ[i,k] = ŷ_k(t_i) - ŷ_k(t_1)
    delta = zeros(n_times, n_vars)
    for k in 1:n_vars
        for i in 2:n_times
            delta[i, k] = y_smooth[i, k] - y_smooth[1, k]
        end
    end

    if verbose
        println("  Smoothed $(length(observed_states))/$n_vars states, $n_times points")
    end

    # ── Stage 2: Integral matching via Adam ──────────────────────
    # Initialize parameters
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
        println("IntegralMatchingSolver Stage 2: Integral matching — $n_beta params")
        println("  maxiters=$(alg.maxiters), lr=$(alg.lr), lambda=$(alg.lambda_smooth)")
    end

    lambda_smooth = alg.lambda_smooth
    use_simpson = n_times >= 3

    # Pre-compute time step widths for trapezoidal/Simpson quadrature
    dt = diff(times)

    function integral_loss(β_eval)
        T_el = eltype(β_eval)
        p = build_autodiff_param_struct(prob, β_eval)
        du = zeros(T_el, n_vars)

        # Evaluate ODE RHS at all smooth points
        F = zeros(T_el, n_times, n_vars)
        for i in 1:n_times
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            for k in 1:n_vars
                F[i, k] = du[k]
            end
        end

        # Cumulative integral via composite trapezoidal rule:
        # I[i,k] = ∫_{t_1}^{t_i} f_k(ŷ(s), p, s) ds
        I_cum = zeros(T_el, n_times, n_vars)
        for k in 1:n_vars
            for i in 2:n_times
                # Trapezoidal: I[i] = I[i-1] + (f[i-1] + f[i]) * dt / 2
                I_cum[i, k] = I_cum[i-1, k] + (F[i-1, k] + F[i, k]) * dt[i-1] / 2
            end
        end

        # Loss: ||delta - I_cum||²
        loss_val = zero(T_el)
        for k in 1:n_vars, i in 2:n_times
            loss_val += (delta[i, k] - I_cum[i, k])^2
        end

        # Smoothing penalty
        offset = 0
        for approx in prob.approximators
            np = nparams(approx)
            params_k = β_eval[offset+1:offset+np]
            offset += np

            if approx isa BSplineApproximator || approx isa GPApproximator ||
               approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator
                S = penalty_matrix(approx)
                if S !== nothing
                    loss_val += lambda_smooth * dot(params_k, S * params_k)
                end
            elseif approx isa ShapeConstrainedBSplineApproximator
                S = penalty_matrix(approx)
                if S !== nothing
                    loss_val += lambda_smooth * dot(params_k, S * params_k)
                end
            elseif approx isa COMONetApproximator
                loss_val += approx.penalty_weight * sum(abs2, params_k)
            end
        end

        loss_val
    end

    # Adam optimizer
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_beta)
    v_adam = zeros(n_beta)
    best_beta = copy(beta)
    best_loss = Inf
    loss_window = fill(Inf, 30)
    final_iter = alg.maxiters

    for iter in 1:alg.maxiters
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
        ForwardDiff.gradient!(result, integral_loss, beta)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)

        if loss_val < best_loss
            best_loss = loss_val
            best_beta .= beta
        end
        loss_window[mod1(iter, 30)] = loss_val

        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
            println("  iter $iter: loss=$(round(loss_val, sigdigits=5)) " *
                    "lr=$(round(lr_t, sigdigits=3))")
        end

        if iter > 60
            recent_min = minimum(loss_window)
            recent_max = maximum(loss_window)
            if (recent_max - recent_min) / max(abs(recent_min), 1.0) < 1e-6
                if verbose; println("  Converged at iter $iter (loss plateau)"); end
                final_iter = iter
                break
            end
        end
    end
    beta .= best_beta

    if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end

    # ── Build solution ───────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= y_smooth[:, sk]
    end

    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

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

    PSMSolution(params, best_loss, data_loss, edf, Float64[lambda_smooth],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=final_iter, method=:integral_matching))
end
