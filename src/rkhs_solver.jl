# ─── RKHS solver (Reproducing Kernel Hilbert Space) ────────────────
#
# Represents the unknown function as f(x) = Σᵢ αᵢ K(x, xᵢ) where K is
# a kernel and xᵢ are representative points.  Solves a penalised
# least-squares problem with RKHS norm penalty ‖f‖²_H = α' K α.
#
# The approach is kernel ridge regression embedded in an ODE fitting loop:
#   Stage 1 — Smooth states with cubic splines → ŷ(t), dŷ/dt
#   Stage 2 — Fit kernel weights α by gradient-matching + RKHS penalty
#
# Reference: Gonzalez et al. (2014), Pattern Recognition Letters
#            Schölkopf & Smola (2002), Learning with Kernels

using LinearAlgebra: dot, norm, Symmetric

"""
    solve(prob::PSMProblem, alg::RKHSSolver)

Fit a partially specified model using an RKHS representation for the
unknown functions.

Instead of B-spline basis expansions, represents each unknown function
as a weighted sum of kernel evaluations at representative points:
f(x) = Σᵢ αᵢ K(x, xᵢ).  The RKHS norm ‖f‖² = α'Kα serves as the
smoothing penalty (analogous to β'Sβ for splines).

# Algorithm
1. Smooth observed data with cubic splines.
2. Place n_repr_points representative points across each UF's domain.
3. Build kernel matrix K and derivative-matching loss.
4. Optimise kernel weights α using Adam with RKHS penalty.

# Returns
`PSMSolution` with kernel-based unknown function evaluators.
"""
function SciMLBase.solve(prob::PSMProblem, alg::RKHSSolver)
    _validate_problem(prob, "RKHSSolver")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    n_vars = length(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_obs = size(prob.data_values, 2)

    # ── Stage 1: Smooth data ─────────────────────────────────────
    if verbose; println("RKHSSolver Stage 1: Smoothing data..."); end

    y_smooth = zeros(n_times, n_vars)
    dydt = zeros(n_times, n_vars)
    observed_states = Set{Int}()

    if prob.discrete
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            itp = CubicSpline(prob.data_values[:, j], times;
                              extrapolation=ExtrapolationType.Extension)
            for i in 1:n_times
                y_smooth[i, sk] = itp(times[i])
            end
        end
        for k in 1:n_vars
            for i in 1:(n_times - 1)
                dydt[i, k] = y_smooth[i + 1, k]
            end
            dydt[n_times, k] = y_smooth[n_times, k]
        end
    else
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            push!(observed_states, sk)
            itp = CubicSpline(prob.data_values[:, j], times;
                              extrapolation=ExtrapolationType.Extension)
            for i in 1:n_times
                y_smooth[i, sk] = itp(times[i])
                dydt[i, sk] = DataInterpolations.derivative(itp, times[i])
            end
        end
    end

    for k in 1:n_vars
        if k ∉ observed_states
            u0_k = Float64(prob.u0 isa Function ? prob.u0(prob.known_params)[k] :
                           prob.u0[k])
            y_smooth[:, k] .= u0_k
        end
    end

    n_match = prob.discrete ? n_times - 1 : n_times

    # ── Stage 2: Build kernel representations ────────────────────
    if verbose; println("RKHSSolver Stage 2: Building kernel representations..."); end

    # Kernel function — auto-scale lengthscale from domain if ≤ 0
    ℓ = alg.lengthscale
    if ℓ <= 0.0
        # Compute domain span from first approximator with a domain
        domain_span = 1.0
        for approx in prob.approximators
            if approx isa BSplineApproximator
                domain_span = approx.domain[2] - approx.domain[1]
                break
            elseif approx isa GPApproximator
                domain_span = maximum(approx.inducing_points) - minimum(approx.inducing_points)
                break
            elseif approx isa SPDEApproximator
                domain_span = maximum(approx.mesh_points) - minimum(approx.mesh_points)
                break
            end
        end
        ℓ = domain_span / 3.0   # ~3 effective kernel widths across domain
        if verbose; println("  Auto lengthscale: ℓ=$(round(ℓ, sigdigits=3)) (domain span=$(round(domain_span, sigdigits=3)))"); end
    end
    kernel_fn = if alg.kernel == :rbf
        (x1, x2) -> exp(-0.5 * (x1 - x2)^2 / ℓ^2)
    elseif alg.kernel == :matern32
        (x1, x2) -> begin
            r = abs(x1 - x2) / ℓ
            (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
        end
    elseif alg.kernel == :matern52
        (x1, x2) -> begin
            r = abs(x1 - x2) / ℓ
            (1 + sqrt(5) * r + 5/3 * r^2) * exp(-sqrt(5) * r)
        end
    else
        error("Unknown kernel: $(alg.kernel). Use :rbf, :matern32, or :matern52.")
    end

    # For each approximator, build representative points and kernel matrix
    n_repr = alg.n_repr_points
    repr_info = []

    total_alpha = 0
    for approx in prob.approximators
        if approx isa BSplineApproximator || approx isa GPApproximator ||
           approx isa SPDEApproximator
            domain = if approx isa BSplineApproximator
                approx.domain
            elseif approx isa GPApproximator
                (minimum(approx.inducing_points), maximum(approx.inducing_points))
            else
                (minimum(approx.mesh_points), maximum(approx.mesh_points))
            end
            x_repr = collect(range(domain[1], domain[2], length=n_repr))
            K_repr = Matrix{Float64}(undef, n_repr, n_repr)
            for i in 1:n_repr, j in 1:n_repr
                K_repr[i, j] = kernel_fn(x_repr[i], x_repr[j])
            end
            K_repr .+= 1e-6 * Matrix{Float64}(I, n_repr, n_repr)  # jitter
            push!(repr_info, (name=approx.name, x_repr=x_repr, K_repr=K_repr,
                              domain=domain, n_alpha=n_repr))
            total_alpha += n_repr
        else
            # For neural/other approximators, fall back to nparams
            np = nparams(approx)
            push!(repr_info, (name=approx.name, x_repr=nothing, K_repr=nothing,
                              domain=nothing, n_alpha=np, approx=approx))
            total_alpha += np
        end
    end

    # Initialise kernel weights from approximator initial values
    alpha = zeros(total_alpha)
    off = 0
    for (ri, info) in enumerate(repr_info)
        np = info.n_alpha
        if info.K_repr !== nothing
            # Use the approximator's initial_params to get target values
            approx = prob.approximators[ri]
            ip = initial_params(approx)
            init_val = sum(ip) / length(ip)  # mean initial value
            f_target = fill(init_val, np)
            alpha[off+1:off+np] = info.K_repr \ f_target
        end
        off += np
    end
    n_alpha = total_alpha

    if verbose
        println("  $n_alpha kernel weights, $(alg.maxiters) iterations, λ=$(alg.lambda_rkhs)")
    end

    lambda_rkhs = alg.lambda_rkhs

    # Kernel evaluator: given weights α and repr points, evaluate at x
    function kernel_evaluate(x, x_repr, alpha_k)
        val = 0.0
        for i in eachindex(x_repr)
            val += alpha_k[i] * kernel_fn(x, x_repr[i])
        end
        val
    end

    # Only include observed states in gradient-matching loss
    obs_states = sort(unique(prob.obs_to_state))

    # ── Loss function ────────────────────────────────────────────
    function rkhs_loss(α_eval)
        T_el = eltype(α_eval)

        # Build callable unknown functions from kernel weights
        uf_entries = Pair{Symbol, Any}[]
        off = 0
        for (ri, info) in enumerate(repr_info)
            np = info.n_alpha
            ak = α_eval[off+1:off+np]
            off += np

            if info.x_repr !== nothing
                let xr = info.x_repr, a = ak
                    push!(uf_entries, info.name => (x -> begin
                        val = zero(T_el)
                        for i in eachindex(xr)
                            val += a[i] * kernel_fn(x, xr[i])
                        end
                        val
                    end))
                end
            end
        end

        p = merge(NamedTuple(uf_entries), prob.known_params)
        du = zeros(T_el, n_vars)
        loss = zero(T_el)

        for i in 1:n_match
            u = T_el.(y_smooth[i, :])
            try
                prob.dynamics!(du, u, p, times[i])
            catch
                du .= T_el(1e6)
            end
            for k in obs_states
                loss += (dydt[i, k] - du[k])^2
            end
        end

        # RKHS norm penalty: α' K α for each kernel UF
        off = 0
        for info in repr_info
            np = info.n_alpha
            ak = α_eval[off+1:off+np]
            off += np
            if info.K_repr !== nothing
                loss += lambda_rkhs * dot(ak, info.K_repr * ak)
            end
        end

        loss
    end

    # ── Adam optimisation ────────────────────────────────────────
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_alpha)
    v_adam = zeros(n_alpha)
    best_alpha = copy(alpha)
    best_loss = Inf
    loss_window = fill(Inf, 30)
    final_iter = alg.maxiters

    for iter in 1:alg.maxiters
        result = DiffResults.MutableDiffResult(0.0, (zeros(n_alpha),))
        ForwardDiff.gradient!(result, rkhs_loss, alpha)
        loss_val = DiffResults.value(result)
        grad = DiffResults.gradient(result)

        if loss_val < best_loss
            best_loss = loss_val
            best_alpha .= alpha
        end
        loss_window[mod1(iter, 30)] = loss_val

        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        alpha .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
            println("  iter $iter: loss=$(round(loss_val, sigdigits=5)) " *
                    "lr=$(round(lr_t, sigdigits=3))")
        end

        if iter > 60
            recent_min = minimum(loss_window)
            recent_max = maximum(loss_window)
            if (recent_max - recent_min) / max(abs(recent_min), 1.0) < 1e-6
                if verbose; println("  Converged at iter $iter"); end
                final_iter = iter
                break
            end
        end
    end
    alpha .= best_alpha

    if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end

    # ── Build solution ───────────────────────────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= y_smooth[:, sk]
    end

    data_loss = sum(abs2, prob.data_values .- pred)

    uf_evals = Dict{Symbol, Any}()
    off = 0
    for info in repr_info
        np = info.n_alpha
        ak = alpha[off+1:off+np]
        off += np
        if info.x_repr !== nothing
            let xr = copy(info.x_repr), a = copy(ak), kf = kernel_fn
                uf_evals[info.name] = x -> begin
                    val = 0.0
                    for i in eachindex(xr)
                        val += a[i] * kf(Float64(x isa AbstractArray ? x[1] : x), xr[i])
                    end
                    val
                end
            end
        end
    end

    ca_entries = Pair{Symbol, Any}[]
    off = 0
    for info in repr_info
        np = info.n_alpha
        push!(ca_entries, info.name => alpha[off+1:off+np])
        off += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    edf = Float64(n_alpha)

    PSMSolution(params, best_loss, data_loss, edf, Float64[lambda_rkhs],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=final_iter, method=:rkhs,
                 kernel=alg.kernel, lengthscale=ℓ))
end
