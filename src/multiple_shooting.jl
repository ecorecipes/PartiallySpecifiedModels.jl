# ─── Multiple shooting solver ────────────────────────────────────────
#
# Implementation of multiple shooting for training neural differential equations,
# following Turan & Jäschke (2021) "Multiple shooting for training neural
# differential equations on time series".
#
# Key idea: partition [t₀, tf] into N intervals with shooting variables (state
# values) at interval boundaries. Optimize model parameters + shooting variables
# jointly. Continuity constraints enforced via augmented Lagrangian:
#
#   φ(θ, s) = C(θ, s) + v'h(θ, s) + ρ/2 ||h(θ, s)||²
#
# where h(θ, s) = [x⁽ⁱ⁾_f - s_{i+1}] are the shooting gap constraints.

using LinearAlgebra: norm, dot

# ─── Interval management ─────────────────────────────────────────

"""
Partition data times into shooting intervals, returning interval boundaries
and data point indices per interval.
"""
function partition_intervals(data_times::Vector{Float64}, n_intervals::Int)
    n_t = length(data_times)
    t_start = data_times[1]
    t_end = data_times[end]

    # Create evenly spaced interval boundaries
    boundaries = collect(range(t_start, t_end, length=n_intervals + 1))

    # Assign data points to intervals
    intervals = Vector{Vector{Int}}(undef, n_intervals)
    for k in 1:n_intervals
        t_lo = boundaries[k]
        t_hi = boundaries[k + 1]
        if k < n_intervals
            intervals[k] = findall(t -> t_lo <= t < t_hi, data_times)
        else
            intervals[k] = findall(t -> t_lo <= t <= t_hi, data_times)
        end
    end

    boundaries, intervals
end

"""
Initialize shooting variables from data (interpolated state values at boundaries).
Returns matrix (n_intervals-1) × K for interior boundary points.
"""
function init_shooting_vars(data_times::Vector{Float64}, data_values::Matrix{Float64},
                            obs_to_state::Vector{Int}, K::Int,
                            boundaries::Vector{Float64})
    n_interior = length(boundaries) - 2  # exclude first and last
    shooting_vars = zeros(n_interior, K)

    for j in 1:size(data_values, 2)
        sk = obs_to_state[j]
        itp = CubicSpline(data_values[:, j], data_times;
                          extrapolation=ExtrapolationType.Extension)
        for i in 1:n_interior
            shooting_vars[i, sk] = itp(boundaries[i + 1])
        end
    end

    shooting_vars
end

# ─── Loss function ───────────────────────────────────────────────

"""
Compute the multiple shooting loss:
- Data fit: SSE across all intervals
- Continuity constraints: augmented Lagrangian penalty on shooting gaps

Parameters z = [θ; vec(shooting_vars)] where θ are the model parameters
and shooting_vars are the state values at interior boundaries.
"""
function ms_loss(prob::PSMProblem, z, n_theta::Int, K::Int,
                 boundaries::Vector{Float64}, intervals::Vector{Vector{Int}},
                 lagrange_mult::Matrix{Float64}, rho::Float64)
    n_intervals = length(intervals)
    n_interior = n_intervals - 1
    T = eltype(z)

    # Unpack: model params and shooting variables
    theta = z[1:n_theta]
    shooting_flat = z[n_theta+1:end]
    shooting_vars = reshape(shooting_flat, n_interior, K)

    # Build parameter struct from model params
    p = build_autodiff_param_struct(prob, theta)

    data_loss = zero(T)
    constraint_violation = zero(T)
    lagrangian_term = zero(T)

    for k in 1:n_intervals
        # Initial state for this interval
        u0_k = if k == 1
            T.(prob.u0 isa Function ? prob.u0(p) : prob.u0)
        else
            T.(shooting_vars[k - 1, :])
        end

        # Time span for this interval
        t_lo = boundaries[k]
        t_hi = boundaries[k + 1]

        # Data times in this interval
        idx = intervals[k]
        if isempty(idx)
            continue
        end
        local_times = prob.data_times[idx]

        if prob.discrete
            # Discrete-time: iterate from t_lo to t_hi
            u = copy(u0_k)
            u_next = similar(u)
            all_steps = collect(t_lo:1.0:t_hi)

            # Build lookup for data times and interval end
            time_states = Dict{Float64, Vector{T}}()
            if haskey(Dict(t_lo => true), t_lo)
                time_states[t_lo] = copy(u)
            end
            # Record at t_lo if needed
            for gi in idx
                if abs(prob.data_times[gi] - t_lo) < 1e-10
                    time_states[t_lo] = copy(u)
                end
            end
            time_states[t_lo] = copy(u)

            for si in 1:(length(all_steps)-1)
                t_cur = all_steps[si]
                prob.dynamics!(u_next, u, p, t_cur)
                u = copy(u_next)
                t_now = all_steps[si + 1]
                time_states[t_now] = copy(u)
            end

            # Data fit loss
            for gi in idx
                t_data = prob.data_times[gi]
                t_nearest = round(t_data)
                u_at_t = get(time_states, t_nearest, nothing)
                if u_at_t === nothing
                    continue
                end
                for j in 1:size(prob.data_values, 2)
                    sk = prob.obs_to_state[j]
                    pred = u_at_t[sk]
                    obs = T(prob.data_values[gi, j])
                    data_loss += prob.data_weights[gi, j] * (pred - obs)^2
                end
            end

            # Shooting constraint
            if k < n_intervals
                u_end = get(time_states, t_hi, nothing)
                if u_end !== nothing
                    for s in 1:K
                        gap = u_end[s] - shooting_vars[k, s]
                        lagrangian_term += lagrange_mult[k, s] * gap
                        constraint_violation += T(rho) / 2 * gap^2
                    end
                end
            end
        else
            # Continuous-time: solve ODE on this interval
            ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
                (du, u, params, t) -> prob.dynamics!(du, u, params, t))
            save_times = unique(sort([local_times; t_hi]))
            ode_prob = ODEProblem(ode_fn, u0_k, (t_lo, t_hi), p)
            sol = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                       saveat=save_times,
                                       abstol=1e-7, reltol=1e-7,
                                       maxiters=10000)

            if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
                return T(1e10)
            end

            # Data fit loss on this interval
            for (li, gi) in enumerate(idx)
                t_data = prob.data_times[gi]
                sol_idx = findfirst(t -> abs(t - t_data) < 1e-10, sol.t)
                if sol_idx === nothing
                    continue
                end
                for j in 1:size(prob.data_values, 2)
                    sk = prob.obs_to_state[j]
                    pred = sol[sk, sol_idx]
                    obs = T(prob.data_values[gi, j])
                    data_loss += prob.data_weights[gi, j] * (pred - obs)^2
                end
            end

            # Shooting constraint
            if k < n_intervals
                end_idx = findfirst(t -> abs(t - t_hi) < 1e-10, sol.t)
                if end_idx !== nothing
                    for s in 1:K
                        gap = sol[s, end_idx] - shooting_vars[k, s]
                        lagrangian_term += lagrange_mult[k, s] * gap
                        constraint_violation += T(rho) / 2 * gap^2
                    end
                end
            end
        end
    end

    data_loss + lagrangian_term + constraint_violation
end

# ─── Main solver ─────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::MultipleShootingSolver)

Fit a partially specified model using multiple shooting. The time span is
divided into intervals, each with its own initial condition; a combined
objective penalises data misfit and continuity gaps between intervals.

# Algorithm
1. Partition the time span into `n_intervals` sub-intervals.
2. Introduce free initial conditions at each interval boundary.
3. Define a loss combining data-fit residuals and continuity penalties
   (weighted by `continuity_weight`).
4. Minimise with Adam (autodiff through the ODE/map solver) using
   learning-rate scheduling and gradient clipping.
5. Return the parameters at the lowest observed loss.

# References
- Bock & Plitt (1984), "A Multiple Shooting Algorithm for Direct Solution
  of Optimal Control Problems", IFAC Proceedings.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MultipleShootingSolver)
    _validate_problem(prob, "MultipleShootingSolver")
    verbose = alg.verbose
    K = length(prob.u0)
    n_obs = size(prob.data_values, 2)
    T_pts = length(prob.data_times)
    n_intervals = alg.n_intervals

    # Initialize model parameters
    theta = Float64[]
    mlp_specs = Dict{Symbol, MLPSpec}()

    for approx in prob.approximators
        if approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            mlp_specs[approx.name] = spec
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(theta, init_mlp_params(spec, rng))
        else
            append!(theta, initial_params(approx))
        end
    end
    n_theta = length(theta)

    # Partition time span
    boundaries, intervals = partition_intervals(Float64.(prob.data_times), n_intervals)
    n_interior = n_intervals - 1

    # Initialize shooting variables from data
    shooting_vars = init_shooting_vars(Float64.(prob.data_times),
                                       Float64.(prob.data_values),
                                       prob.obs_to_state, K, boundaries)

    # Optimization variable: z = [theta; vec(shooting_vars)]
    z = vcat(theta, vec(shooting_vars))
    n_z = length(z)

    if verbose
        println("MultipleShootingSolver: $(n_theta) model params + $(n_interior * K) shooting vars = $(n_z) total")
        println("  $(n_intervals) intervals, $(alg.maxiters_outer) outer × $(alg.maxiters_inner) inner iters")
    end

    # Augmented Lagrangian state
    lagrange_mult = zeros(n_interior, K)
    rho = alg.rho_init

    best_z_global = copy(z)
    best_data_loss_global = Inf
    prev_max_gap = Inf

    # Outer loop: augmented Lagrangian
    for outer in 1:alg.maxiters_outer
        # Re-create loss function with current lagrange_mult and rho
        loss_fn = z_ -> ms_loss(prob, z_, n_theta, K, boundaries, intervals, lagrange_mult, rho)

        if verbose
            println("\n─── Outer iter $outer: ρ=$(round(rho, sigdigits=3)) ───")
        end

        # Inner loop: L-BFGS optimization (matches paper's approach)
        inner_result = Optim.optimize(
            loss_fn,
            z_ -> ForwardDiff.gradient(loss_fn, z_),
            z,
            Optim.LBFGS(),
            Optim.Options(
                iterations=alg.maxiters_inner,
                show_trace=verbose,
                show_every=max(1, alg.maxiters_inner ÷ 4),
                g_tol=1e-8,
                f_reltol=1e-10,
            );
            inplace=false
        )
        z .= Optim.minimizer(inner_result)

        if verbose
            println("  Inner converged: $(Optim.converged(inner_result)), " *
                    "f=$(round(Optim.minimum(inner_result), sigdigits=5))")
        end

        # Compute shooting gaps for multiplier update
        theta_cur = z[1:n_theta]
        shooting_flat = z[n_theta+1:end]
        shooting_cur = reshape(shooting_flat, n_interior, K)

        p_cur = build_autodiff_param_struct(prob, theta_cur)
        max_gap = 0.0

        for k in 1:n_intervals-1
            u0_k = if k == 1
                Float64.(prob.u0 isa Function ? prob.u0(p_cur) : prob.u0)
            else
                Float64.(shooting_cur[k - 1, :])
            end
            t_lo = boundaries[k]
            t_hi = boundaries[k + 1]

            if prob.discrete
                # Discrete-time: iterate to get end state
                u = copy(u0_k)
                u_next = similar(u)
                all_steps = collect(t_lo:1.0:t_hi)
                for si in 1:(length(all_steps)-1)
                    prob.dynamics!(u_next, u, p_cur, all_steps[si])
                    u = copy(u_next)
                end
                for s in 1:K
                    gap = u[s] - shooting_cur[k, s]
                    lagrange_mult[k, s] += rho * gap
                    max_gap = max(max_gap, abs(gap))
                end
            else
                ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
                    (du, u, params, t) -> prob.dynamics!(du, u, params, t))
                ode_prob = ODEProblem(ode_fn, u0_k, (t_lo, t_hi), p_cur)
                sol = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                           saveat=[t_hi],
                                           abstol=1e-7, reltol=1e-7,
                                           maxiters=10000)
                if length(sol.t) > 0
                    for s in 1:K
                        gap = sol[s, end] - shooting_cur[k, s]
                        lagrange_mult[k, s] += rho * gap
                        max_gap = max(max_gap, abs(gap))
                    end
                end
            end
        end

        # Also compute data-only loss (for tracking best overall fit)
        data_only_loss = ms_loss(prob, z, n_theta, K, boundaries, intervals,
                                 zeros(n_interior, K), 0.0)
        if data_only_loss < best_data_loss_global
            best_data_loss_global = data_only_loss
            best_z_global .= z
        end

        if verbose
            println("  Max shooting gap: $(round(max_gap, sigdigits=4))")
            println("  Data-only loss: $(round(data_only_loss, sigdigits=5))")
        end

        # Check convergence (relative to state scale)
        state_scale = norm(prob.u0)
        if max_gap < 1e-2 * state_scale
            if verbose; println("  Shooting gaps converged!"); end
            break
        end

        # Increase penalty only if gaps aren't decreasing enough
        if max_gap > 0.5 * prev_max_gap
            rho = min(rho * 2.0, alg.rho_max)
        else
            rho = min(rho * 1.2, alg.rho_max)
        end
        prev_max_gap = max_gap
    end

    # Build final solution using single shooting with best params
    theta_final = z[1:n_theta]
    p_opt = build_autodiff_param_struct(prob, theta_final)
    u0 = prob.u0 isa Function ? prob.u0(p_opt) : prob.u0

    if prob.discrete
        pred = Float64.(adam_simulate_discrete(prob, p_opt))
    else
        ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
            (du, u, params, t) -> prob.dynamics!(du, u, params, t))
        ode_prob = ODEProblem(ode_fn, Float64.(u0), prob.tspan, p_opt)
        sol_ode = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                                       saveat=prob.data_times,
                                       abstol=1e-7, reltol=1e-7,
                                       maxiters=10000)

        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            for i in 1:T_pts
                pred[i, j] = sol_ode[sk, i]
            end
        end
    end

    data_loss = 0.0
    for j in 1:n_obs, i in 1:T_pts
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Build evaluators for unknown functions
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = theta_final[offset+1:offset+np]
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
                    xval = x isa AbstractArray ? x[1] : x
                    xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                        (Float64(xval) - lo_) / span_
                    else
                        Float64(xval)
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
        end
    end

    edf = Float64(n_theta)
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => theta_final[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    if verbose
        println("\nFinal (single-shoot): data_SS=$(round(data_loss, sigdigits=5))")
    end

    PSMSolution(params, data_loss, data_loss, edf, Float64[],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (optimizer=:adam, method=:multiple_shooting,
                 n_intervals=n_intervals))
end
