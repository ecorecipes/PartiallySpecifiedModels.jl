# ─── Adam solver with autodiff through ODE ──────────────────────────
#
# Trains unknown function parameters using Adam optimizer with gradients
# computed via ForwardDiff through the ODE solve. This is equivalent to
# the UDE approach and matches the reference PSM implementation.
#
# For neural networks, we bypass Lux and use a lightweight MLP evaluator
# that is fully compatible with ForwardDiff's Dual numbers.

using LinearAlgebra: norm, dot
using ForwardDiff

# The ForwardDiff-compatible MLP machinery (MLPSpec, mlp_spec_from_lux,
# mlp_evaluate, init_mlp_params) lives in neural_evaluator.jl, shared by
# all solvers. Solvers reach it only through the guarded entry points
# `neural_init_params` (initialisation) and `build_neural_evaluator`
# (evaluation): `mlp_spec_from_lux` ERRORS on architectures it cannot
# describe, so calling it directly turns an exotic-but-valid Lux model
# into a hard failure at solve() instead of a fallback.

# ─── ForwardDiff-compatible param struct builder ─────────────────

"""
Build parameter NamedTuple where evaluators preserve ForwardDiff Dual types.
"""
function build_autodiff_param_struct(prob::PSMProblem, beta)
    offset = 0
    uf_entries = Pair{Symbol, Any}[]

    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np

        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            # Build B-spline evaluator that preserves Dual type in coefficients
            evaluator = build_bspline_evaluator(knots_x, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa NeuralApproximator
            # Shared Dual-safe evaluator (closure captures params_k, which
            # may be Dual-valued)
            push!(uf_entries, approx.name => build_neural_evaluator(approx, params_k))
        elseif approx isa GPApproximator
            evaluator = build_gp_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa ShapeConstrainedBSplineApproximator
            # Use the SAME coefficient-basis evaluator as every other solver.
            # (The old AD path treated β = Σ·softplus(γ) as interpolation
            # knot VALUES and natural-cubic-splined through them — a
            # different function from the de Boor coefficient spline the
            # returned solution reports, it could violate the constraint
            # between knots, and it ignored the free linear parameters.)
            # _softplus and _bspline_basis_vector are Dual-safe.
            evaluator = build_constrained_bspline_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa COMONetApproximator
            evaluator = build_comonet_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa SPDEApproximator
            evaluator = build_spde_evaluator(approx.mesh_points, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa ShapeConstrainedSPDEApproximator
            # Route through gamma_to_mesh_values so the free linear
            # components are NOT softplus'd — the optimized function must be
            # the same one initial_params inverts and the solution reports.
            mesh_values = gamma_to_mesh_values(approx, params_k)
            evaluator = build_spde_evaluator(approx.mesh_points, mesh_values)
            push!(uf_entries, approx.name => evaluator)
        end
    end

    uf_nt = NamedTuple(uf_entries)
    merge(uf_nt, prob.known_params)
end

# ─── Loss functions ──────────────────────────────────────────────

"""
    adam_simulate_discrete(prob, p, T)

Discrete-time simulation for AdamSolver, compatible with ForwardDiff Dual
numbers. `T` is the state element type — pass `eltype(beta)` so Dual
parameters propagate (the old trial-evaluation inference failed whenever the
first state's derivative did not touch an approximator).
Returns matrix of predictions at data_times (n_times × n_obs).
"""
function adam_simulate_discrete(prob::PSMProblem, p, ::Type{T}) where {T}
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0

    n_vars = length(u0)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    t_start = prob.tspan[1]
    t_end = prob.tspan[2]
    all_times = collect(t_start:1.0:t_end)

    # Map data_times to step indices
    data_time_set = Dict{Float64, Vector{Int}}()
    for (di, dt) in enumerate(prob.data_times)
        t_nearest = round(dt)
        if !haskey(data_time_set, t_nearest)
            data_time_set[t_nearest] = Int[]
        end
        push!(data_time_set[t_nearest], di)
    end

    # Allocate prediction matrix in the caller-supplied element type
    pred = zeros(T, n_times, n_obs)

    u = T.(u0)
    u_next = zeros(T, n_vars)

    # Record initial condition
    t = t_start
    if haskey(data_time_set, t)
        for di in data_time_set[t]
            for j in 1:n_obs
                pred[di, j] = u[prob.obs_to_state[j]]
            end
        end
    end

    for step in 1:(length(all_times)-1)
        t = all_times[step]
        prob.dynamics!(u_next, u, p, t)
        u = copy(u_next)
        t_now = all_times[step + 1]

        if haskey(data_time_set, t_now)
            for di in data_time_set[t_now]
                for j in 1:n_obs
                    pred[di, j] = u[prob.obs_to_state[j]]
                end
            end
        end
    end

    pred
end

"""
    adam_penalty(prob, beta, w)

Fixed quadratic smoothing penalty `w · Σₖ βₖ' Sₖ βₖ` over all approximators
with a penalty matrix. Returns `zero(eltype(beta))` when `w == 0`.
"""
function adam_penalty(prob::PSMProblem, beta, w::Float64)
    T = eltype(beta)
    w == 0.0 && return zero(T)
    pen = zero(T)
    off = 0
    for approx in prob.approximators
        np = nparams(approx)
        S = penalty_matrix(approx)
        if S !== nothing
            bk = @view beta[off+1:off+np]
            pen += dot(bk, S * bk)
        end
        off += np
    end
    T(w) * pen
end

function adam_loss_mse(prob::PSMProblem, beta, penalty_w::Float64=0.0)
    p = build_autodiff_param_struct(prob, beta)
    T = eltype(beta)

    if prob.discrete
        pred = adam_simulate_discrete(prob, p, T)
        loss = zero(T)
        n_obs = size(prob.data_values, 2)
        n_t = length(prob.data_times)
        for j in 1:n_obs
            for i in 1:n_t
                loss += prob.data_weights[i, j] * (pred[i, j] - prob.data_values[i, j])^2
            end
        end
        total = loss + adam_penalty(prob, beta, penalty_w)
        return _all_finite(total) ? total : T(1e10)
    end

    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    u0_T = T.(u0)

    # Dispatch to DDE or ODE solve
    sol = try
        if !isempty(prob.delays)
            adam_solve_dde(prob, beta)
        else
            ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}((du, u, params, t) -> prob.dynamics!(du, u, params, t))
            ode_prob = ODEProblem(ode_fn, u0_T, prob.tspan, p)
            OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                 saveat=prob.data_times,
                                 abstol=get(prob.ode_kwargs, :abstol, 1e-7),
                                 reltol=get(prob.ode_kwargs, :reltol, 1e-7),
                                 maxiters=get(prob.ode_kwargs, :maxiters, 10000))
        end
    catch e
        # Blow-up regions can throw from inside spline evaluators (NaN Dual
        # state reaching DataInterpolations) before the integrator aborts
        # with a retcode; convert to the failure sentinel instead of
        # crashing the optimization.
        _is_program_error(e) && rethrow()
        return T(1e10)
    end

    if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
        return T(1e10)
    end

    loss = zero(T)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:min(n_t, length(sol.t))
            pred = sol[sk, i]
            obs = prob.data_values[i, j]
            loss += prob.data_weights[i, j] * (pred - obs)^2
        end
    end
    total = loss + adam_penalty(prob, beta, penalty_w)
    _all_finite(total) ? total : T(1e10)
end

function adam_loss_poisson(prob::PSMProblem, beta, penalty_w::Float64=0.0)
    p = build_autodiff_param_struct(prob, beta)
    T = eltype(beta)

    if prob.discrete
        pred = adam_simulate_discrete(prob, p, T)
        loss = zero(T)
        n_obs = size(prob.data_values, 2)
        n_t = length(prob.data_times)
        for j in 1:n_obs
            for i in 1:n_t
                mu = max(pred[i, j], T(1e-10))
                y = prob.data_values[i, j]
                loss -= prob.data_weights[i, j] * (y * log(mu) - mu)
            end
        end
        total = loss + adam_penalty(prob, beta, penalty_w)
        return _all_finite(total) ? total : T(1e10)
    end

    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    u0_T = T.(u0)

    # Dispatch to DDE or ODE solve
    sol = try
        if !isempty(prob.delays)
            adam_solve_dde(prob, beta)
        else
            ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}((du, u, params, t) -> prob.dynamics!(du, u, params, t))
            ode_prob = ODEProblem(ode_fn, u0_T, prob.tspan, p)
            OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                 saveat=prob.data_times,
                                 abstol=get(prob.ode_kwargs, :abstol, 1e-7),
                                 reltol=get(prob.ode_kwargs, :reltol, 1e-7),
                                 maxiters=get(prob.ode_kwargs, :maxiters, 10000))
        end
    catch e
        # Blow-up regions can throw from inside spline evaluators (NaN Dual
        # state reaching DataInterpolations) before the integrator aborts
        # with a retcode; convert to the failure sentinel instead of
        # crashing the optimization.
        _is_program_error(e) && rethrow()
        return T(1e10)
    end

    if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
        return T(1e10)
    end

    loss = zero(T)
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:min(n_t, length(sol.t))
            mu = max(sol[sk, i], T(1e-10))
            y = prob.data_values[i, j]
            loss -= prob.data_weights[i, j] * (y * log(mu) - mu)
        end
    end
    total = loss + adam_penalty(prob, beta, penalty_w)
    _all_finite(total) ? total : T(1e10)
end

# ─── Main Adam solver ────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::AdamSolver)

Fit a partially specified model using the Adam optimizer with automatic
differentiation through the ODE/map solver via `ForwardDiff.jl`.

# Algorithm
1. Initialize neural-network or spline parameters.
2. Define a differentiable loss: simulate the model, compute weighted
   residuals, and optionally add smoothing penalties.
3. Iterate Adam updates on the full parameter vector with learning-rate
   scheduling and optional gradient clipping.
4. Return the parameters at the lowest observed loss.

# References
- Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization", ICLR.
- Rackauckas et al. (2020), "Universal Differential Equations", arXiv:2001.04385.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
`sol.convergence` is a NamedTuple `(optimizer, method, converged, iterations,
reason, final_grad_norm)` — see the `AdamSolver` docstring for the key
taxonomy and the guarded plateau criterion.
"""
function SciMLBase.solve(prob::PSMProblem, alg::AdamSolver)
    _validate_problem(prob, "AdamSolver")
    verbose = alg.verbose

    # Initialize parameters
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

    # Select loss function. :auto follows prob.likelihood; an explicit
    # choice is honored but warned about on mismatch.
    loss_sym = alg.loss
    if loss_sym == :auto
        loss_sym = if prob.likelihood isa Gaussian
            :mse
        elseif prob.likelihood isa Poisson
            :poisson
        else
            error("AdamSolver has no loss for $(typeof(prob.likelihood)); " *
                  "it supports Gaussian (:mse) and Poisson (:poisson). " *
                  "Use LAML for other likelihood families.")
        end
    else
        expected = prob.likelihood isa Poisson ? :poisson : :mse
        if !(prob.likelihood isa Gaussian || prob.likelihood isa Poisson)
            @warn "AdamSolver: prob.likelihood is $(typeof(prob.likelihood)), " *
                  "which this solver cannot honor; fitting with loss=$loss_sym instead"
        elseif loss_sym != expected
            @warn "AdamSolver: loss=$loss_sym does not match " *
                  "prob.likelihood=$(typeof(prob.likelihood)) (expected :$expected)"
        end
    end
    loss_fn = if loss_sym == :poisson
        β -> adam_loss_poisson(prob, β, alg.penalty_weight)
    else
        β -> adam_loss_mse(prob, β, alg.penalty_weight)
    end

    if verbose
        println("AdamSolver: $(n_beta) params, $(alg.maxiters) max iters, lr=$(alg.lr)")
        println("  Loss: $(loss_sym) (from $(alg.loss)), autodiff: $(alg.autodiff)")
    end

    # Adam state
    lr = alg.lr
    β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
    m_adam = zeros(n_beta)
    v_adam = zeros(n_beta)
    best_beta = copy(beta)
    best_loss = Inf
    loss_window = fill(Inf, 30)

    # Honest convergence reporting: defaults describe loop exhaustion.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    final_grad_norm = NaN

    for iter in 1:alg.maxiters
        conv_iters = iter
        # Compute gradient
        local loss_val
        if alg.autodiff
            # ForwardDiff gradient
            result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
            ForwardDiff.gradient!(result, loss_fn, beta)
            loss_val = DiffResults.value(result)
            grad = DiffResults.gradient(result)
        else
            # Finite difference gradient
            loss_val = loss_fn(beta)
            grad = zeros(n_beta)
            eps = 1e-5
            for i in 1:n_beta
                h = max(eps, abs(beta[i]) * eps)
                beta[i] += h
                grad[i] = (loss_fn(beta) - loss_val) / h
                beta[i] -= h
            end
        end

        final_grad_norm = norm(grad)

        if loss_val < best_loss
            best_loss = loss_val
            best_beta .= beta
        end
        loss_window[mod1(iter, 30)] = loss_val

        # Cosine learning rate annealing
        lr_t = lr * 0.5 * (1 + cos(π * iter / alg.maxiters))

        # Adam update
        m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
        v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
        m_hat = m_adam ./ (1 - β1_adam^iter)
        v_hat = v_adam ./ (1 - β2_adam^iter)
        beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

        if verbose && (iter <= 5 || iter % 25 == 0 || iter == alg.maxiters)
            println("  iter $iter: loss=$(round(loss_val, sigdigits=5)) lr=$(round(lr_t, sigdigits=3))")
        end

        # Convergence: loss plateau. A plateau at the 1e10 failure sentinel
        # is a stuck solver, not convergence — keep iterating (the Adam
        # moments may still walk out of the failing region).
        # Guard against SPURIOUS plateaus manufactured by the cosine lr
        # schedule: near maxiters lr_t → 0, so the loss stops moving no matter
        # how far from an optimum we are. Only declare plateau-convergence
        # while the step size is still meaningful (lr_t > 5% of the base lr).
        if iter > 60 && best_loss < 1e9 && lr_t > 0.05 * lr
            recent_min = minimum(loss_window)
            recent_max = maximum(loss_window)
            if (recent_max - recent_min) / max(abs(recent_min), 1.0) < 1e-4
                if verbose; println("  Converged at iter $iter (loss plateau)"); end
                conv_converged = true
                conv_reason = :plateau
                break
            end
        end
    end
    beta .= best_beta

    if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end

    # Build solution with best parameters
    T_pts = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    # Simulate with best params to get predictions
    p_opt = build_autodiff_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p_opt) : prob.u0

    if prob.discrete
        pred = adam_simulate_discrete(prob, p_opt, Float64)
        pred = Float64.(pred)
    elseif !isempty(prob.delays)
        sol_dde = adam_solve_dde_final(prob, p_opt, Float64.(u0))
        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            for i in 1:T_pts
                pred[i, j] = sol_dde[sk, i]
            end
        end
    else
        ode_prob = ODEProblem((du, u, params, t) -> prob.dynamics!(du, u, p_opt, t),
                              Float64.(u0), prob.tspan)
        sol_ode = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                       saveat=prob.data_times,
                                       abstol=get(prob.ode_kwargs, :abstol, 1e-7),
                                       reltol=get(prob.ode_kwargs, :reltol, 1e-7),
                                       maxiters=get(prob.ode_kwargs, :maxiters, 10000))
        (sol_ode.retcode == SciMLBase.ReturnCode.Success ||
         sol_ode.retcode == :Success) && length(sol_ode.u) >= T_pts ||
            error("final ODE solve at the fitted parameters failed " *
                  "(retcode $(sol_ode.retcode))")

        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            for i in 1:T_pts
                pred[i, j] = sol_ode[sk, i]
            end
        end
    end

    data_loss = weighted_data_loss(prob, pred)

    # Build evaluators for plotting
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
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) EDF=$(round(edf, digits=1))")
    end

    PSMSolution(params, best_loss, data_loss, edf, Float64[],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (optimizer=:adam, method=:adam_ode,
                 converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, final_grad_norm=final_grad_norm))
end
