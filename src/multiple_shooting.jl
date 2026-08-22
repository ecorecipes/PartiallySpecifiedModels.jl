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
using ForwardDiff

# ─── Interval management ─────────────────────────────────────────

"""
Partition data times into shooting intervals, returning interval boundaries
and data point indices per interval.
"""
function partition_intervals(data_times::Vector{Float64}, n_intervals::Int;
                             t0::Float64=data_times[1],
                             discrete::Bool=false)
    n_t = length(data_times)
    # Start segments at t0 (the time where u0 is defined), not at the first
    # observation: anchoring u0 at data_times[1] when tspan[1] < data_times[1]
    # made the training objective disagree with the final single-shoot fit.
    t_start = min(t0, data_times[1])
    t_end = data_times[end]

    # Create evenly spaced interval boundaries
    boundaries = collect(range(t_start, t_end, length=n_intervals + 1))

    if discrete
        # Discrete maps advance on integer steps; fractional boundaries made
        # the step grid miss both the data times (zero data loss) and the
        # segment ends (continuity constraints silently dropped).
        boundaries = unique(round.(boundaries) .+ 0.0)   # .+ 0.0 folds -0.0 into 0.0
        if length(boundaries) - 1 < n_intervals
            @warn "MultipleShooting: reduced n_intervals from $n_intervals " *
                  "to $(length(boundaries) - 1) after snapping interval " *
                  "boundaries to integer time steps"
            n_intervals = length(boundaries) - 1
        end
        n_intervals >= 1 ||
            error("MultipleShooting: fewer than one interval after snapping " *
                  "boundaries to integer steps; use single shooting instead")
    end

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
                            boundaries::Vector{Float64};
                            data_weights::Union{Nothing,AbstractMatrix}=nothing)
    n_interior = length(boundaries) - 2  # exclude first and last
    shooting_vars = zeros(n_interior, K)

    for j in 1:size(data_values, 2)
        sk = obs_to_state[j]
        # Interpolate through the USABLE rows only. `CubicSpline` solves a
        # tridiagonal system for its coefficients, so one NaN in the column
        # makes the interpolant NaN EVERYWHERE — the shooting variables would
        # all start NaN and the optimizer would begin from a NaN point,
        # independently of any masking in the loss. Complete data keeps every
        # row, so the interpolant is bit-for-bit the same.
        keep = [i for i in axes(data_values, 1)
                if isfinite(data_values[i, j]) &&
                   (data_weights === nothing || data_weights[i, j] > 0)]
        length(keep) < 2 && continue   # leave this state at 0.0; too little data
        itp = CubicSpline(data_values[keep, j], data_times[keep];
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
- Data fit across all intervals: weighted SSE (`loss_sym == :mse`) or the
  weighted Poisson negative log-likelihood kernel `−Σ w(y log μ − μ)`
  (`loss_sym == :poisson`, matching `adam_loss_poisson`)
- Continuity constraints: augmented Lagrangian penalty on shooting gaps
- Optional fixed smoothing penalty `adam_penalty(prob, θ, penalty_w)`

Parameters z = [θ; vec(shooting_vars)] where θ are the model parameters
and shooting_vars are the state values at interior boundaries.
"""
# Value AND derivative finiteness: Optim's HagerZhang line search asserts
# isfinite(phi) && isfinite(dphi), and a Dual can carry a finite value with
# Inf/NaN partials (e.g. squared huge-but-finite states).
_all_finite(x::Real) = isfinite(x)
_all_finite(x::ForwardDiff.Dual) =
    isfinite(ForwardDiff.value(x)) && all(isfinite, ForwardDiff.partials(x))

function ms_loss(prob::PSMProblem, z, n_theta::Int, K::Int,
                 boundaries::Vector{Float64}, intervals::Vector{Vector{Int}},
                 lagrange_mult::Matrix{Float64}, rho::Float64,
                 loss_sym::Symbol=:mse, penalty_w::Float64=0.0)
    try
        return _ms_loss_inner(prob, z, n_theta, K, boundaries, intervals,
                              lagrange_mult, rho, loss_sym, penalty_w)
    catch e
        # A blow-up region can throw from inside the spline evaluators
        # (NaN Dual state reaching DataInterpolations) before the
        # integrator can abort with a retcode; the exception would
        # otherwise escape Optim.optimize and kill the whole solve.
        _is_program_error(e) && rethrow()
        return eltype(z)(1e10)
    end
end

function _ms_loss_inner(prob::PSMProblem, z, n_theta::Int, K::Int,
                 boundaries::Vector{Float64}, intervals::Vector{Vector{Int}},
                 lagrange_mult::Matrix{Float64}, rho::Float64,
                 loss_sym::Symbol=:mse, penalty_w::Float64=0.0)
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

        # Data times in this interval. An interval with NO data still
        # carries its continuity constraint: skipping it entirely removed
        # those shooting gaps from the training objective while the outer
        # multiplier update kept measuring them, escalating ρ forever
        # without effect (common since segments start at tspan[1], which
        # can precede the first observation by several intervals).
        idx = intervals[k]
        local_times = prob.data_times[idx]

        if prob.discrete
            # Discrete-time: iterate from t_lo to t_hi. Boundaries are
            # snapped to integers in partition_intervals, so the unit-step
            # grid lands exactly on t_hi and on rounded data times.
            u = copy(u0_k)
            u_next = similar(u)
            all_steps = collect(t_lo:1.0:t_hi)

            time_states = Dict{Float64, Vector{T}}()
            time_states[t_lo] = copy(u)

            for si in 1:(length(all_steps)-1)
                t_cur = all_steps[si]
                prob.dynamics!(u_next, u, p, t_cur)
                u = copy(u_next)
                # Guard map blow-up: a NaN/Inf state would otherwise reach
                # the spline evaluators (which cannot bracket a NaN knot
                # position) deep inside the AD-driven line search.
                all(_all_finite, u) || return T(1e10)
                t_now = all_steps[si + 1]
                time_states[t_now] = copy(u)
            end

            # Data fit loss. A missing entry indicates an internal
            # inconsistency between the boundary grid and the data times —
            # fail loudly rather than silently dropping observations.
            for gi in idx
                t_data = prob.data_times[gi]
                t_nearest = round(t_data)
                u_at_t = get(time_states, t_nearest, nothing)
                u_at_t === nothing &&
                    error("MultipleShooting internal error: no state " *
                          "recorded at t=$t_nearest for observation at " *
                          "t=$t_data in interval [$t_lo, $t_hi]")
                for j in 1:size(prob.data_values, 2)
                    # Masked cells contribute nothing. `0 * NaN = NaN`,
                    # and the `_all_finite` sentinel would then flatten
                    # the whole augmented Lagrangian to the constant
                    # 1e10 — L-BFGS makes no progress, the shooting gaps
                    # never close, and the outer loop exits at maxiters.
                    usable_cell(prob, gi, j) || continue
                    sk = prob.obs_to_state[j]
                    pred = u_at_t[sk]
                    obs = T(prob.data_values[gi, j])
                    if loss_sym == :poisson
                        mu = max(pred, T(1e-10))
                        data_loss -= prob.data_weights[gi, j] *
                                     (obs * log(mu) - mu)
                    else
                        data_loss += prob.data_weights[gi, j] * (pred - obs)^2
                    end
                end
            end

            # Shooting constraint: t_hi is always on the step grid.
            if k < n_intervals
                u_end = time_states[t_hi]
                for s in 1:K
                    gap = u_end[s] - shooting_vars[k, s]
                    lagrangian_term += lagrange_mult[k, s] * gap
                    constraint_violation += T(rho) / 2 * gap^2
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
                # As in the discrete branch: a data time with no matching
                # saved point indicates an internal inconsistency between
                # the interval partition and `saveat` — fail loudly rather
                # than silently dropping observations from the objective.
                sol_idx = findfirst(t -> abs(t - t_data) < 1e-10, sol.t)
                sol_idx === nothing &&
                    error("MultipleShooting internal error: the ODE " *
                          "solution on [$t_lo, $t_hi] has no saved point " *
                          "at the observation time t=$t_data " *
                          "(saved times: $(sol.t))")
                for j in 1:size(prob.data_values, 2)
                    # Masked cells contribute nothing. `0 * NaN = NaN`,
                    # and the `_all_finite` sentinel would then flatten
                    # the whole augmented Lagrangian to the constant
                    # 1e10 — L-BFGS makes no progress, the shooting gaps
                    # never close, and the outer loop exits at maxiters.
                    usable_cell(prob, gi, j) || continue
                    sk = prob.obs_to_state[j]
                    pred = sol[sk, sol_idx]
                    obs = T(prob.data_values[gi, j])
                    if loss_sym == :poisson
                        mu = max(pred, T(1e-10))
                        data_loss -= prob.data_weights[gi, j] *
                                     (obs * log(mu) - mu)
                    else
                        data_loss += prob.data_weights[gi, j] * (pred - obs)^2
                    end
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

    total = data_loss + lagrangian_term + constraint_violation +
            adam_penalty(prob, theta, penalty_w)
    _all_finite(total) ? total : T(1e10)
end

# ─── Main solver ─────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::MultipleShootingSolver)

Fit a partially specified model using multiple shooting. The time span is
divided into intervals, each with its own initial condition; a combined
objective penalises data misfit and continuity gaps between intervals.

# Algorithm
1. Partition the time span into `n_intervals` sub-intervals (integer
   boundaries for discrete-time models).
2. Introduce free initial conditions at each interval boundary, optimized
   jointly with the model parameters.
3. Form the augmented Lagrangian: data-fit loss + v'h + (ρ/2)‖h‖²
   over the continuity gaps h, with multiplier updates v ← v + ρh.
   The data-fit loss follows `prob.likelihood` via `loss=:auto`
   (Gaussian → weighted SSE, Poisson → weighted negative log-likelihood
   kernel, matching `AdamSolver`; other families error). A fixed
   smoothing penalty `penalty_weight · Σₖ βₖ'Sₖβₖ` is added when
   `penalty_weight > 0`.
4. Minimise each subproblem with L-BFGS (ForwardDiff gradients through
   the ODE/map solve), escalating ρ while gaps stagnate.
5. Return the final iterate (which best satisfies the continuity
   constraints); fitted values are produced by a final single-shoot
   simulation at those parameters.

DDE problems are not supported (segment initial states cannot supply the
delayed history).

# References
- Bock & Plitt (1984), "A Multiple Shooting Algorithm for Direct Solution
  of Optimal Control Problems", IFAC Proceedings.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
`objective` is the final single-shoot data-fit loss in the training metric
(weighted SSE for `:mse`, the weighted Poisson NLL kernel for `:poisson`)
plus the smoothing penalty when `penalty_weight > 0`; `data_loss` is
always the descriptive weighted SSE. `sol.convergence` is a NamedTuple
`(optimizer, method, n_intervals, converged, iterations, reason, max_gap,
rho_final)` — see the `MultipleShootingSolver` docstring for the key
taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MultipleShootingSolver)
    _validate_problem(prob, "MultipleShootingSolver")
    isempty(prob.delays) ||
        error("MultipleShootingSolver does not support DDE problems: " *
              "segment initial states cannot supply the delayed history. " *
              "Use LAML or AdamSolver for DDEs.")
    verbose = alg.verbose
    n_obs = size(prob.data_values, 2)
    T_pts = length(prob.data_times)
    n_intervals = alg.n_intervals

    # Select the data-fit loss. :auto follows prob.likelihood; an explicit
    # choice is honored but warned about on mismatch (same policy as
    # AdamSolver).
    loss_sym = alg.loss
    if loss_sym == :auto
        loss_sym = if prob.likelihood isa Gaussian
            :mse
        elseif prob.likelihood isa Poisson
            :poisson
        else
            error("MultipleShootingSolver has no loss for " *
                  "$(typeof(prob.likelihood)); it supports Gaussian (:mse) " *
                  "and Poisson (:poisson). Use LAML for other likelihood " *
                  "families.")
        end
    elseif loss_sym in (:mse, :poisson)
        expected = prob.likelihood isa Poisson ? :poisson : :mse
        if !(prob.likelihood isa Gaussian || prob.likelihood isa Poisson)
            @warn "MultipleShootingSolver: prob.likelihood is " *
                  "$(typeof(prob.likelihood)), which this solver cannot " *
                  "honor; fitting with loss=$loss_sym instead"
        elseif loss_sym != expected
            @warn "MultipleShootingSolver: loss=$loss_sym does not match " *
                  "prob.likelihood=$(typeof(prob.likelihood)) (expected :$expected)"
        end
    else
        error("MultipleShootingSolver: unknown loss :$loss_sym. " *
              "Supported: :auto, :mse, :poisson.")
    end
    penalty_w = alg.penalty_weight

    # Initialize model parameters
    theta = Float64[]

    for approx in prob.approximators
        if approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            append!(theta, neural_init_params(approx, rng))
        else
            append!(theta, initial_params(approx))
        end
    end
    n_theta = length(theta)

    # u0 may be a function of the parameters; evaluate it at the initial
    # parameters to obtain the state dimension and scale.
    u0_init = prob.u0 isa Function ?
        prob.u0(build_autodiff_param_struct(prob, theta)) : prob.u0
    K = length(u0_init)

    # Partition time span (anchored at tspan[1], where u0 lives; integer
    # boundaries for discrete maps)
    boundaries, intervals = partition_intervals(Float64.(prob.data_times),
                                                n_intervals;
                                                t0=Float64(prob.tspan[1]),
                                                discrete=prob.discrete)
    n_intervals = length(boundaries) - 1   # may shrink after integer snapping
    n_interior = n_intervals - 1

    # Initialize shooting variables from data
    shooting_vars = init_shooting_vars(Float64.(prob.data_times),
                                       Float64.(prob.data_values),
                                       prob.obs_to_state, K, boundaries;
                                       data_weights=prob.data_weights)

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

    prev_max_gap = Inf

    # Honest convergence reporting: defaults describe outer-loop exhaustion.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    final_max_gap = NaN

    # Outer loop: augmented Lagrangian
    for outer in 1:alg.maxiters_outer
        conv_iters = outer
        # Re-create loss function with current lagrange_mult and rho
        loss_fn = z_ -> ms_loss(prob, z_, n_theta, K, boundaries, intervals,
                                lagrange_mult, rho, loss_sym, penalty_w)

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

        if verbose
            data_only_loss = ms_loss(prob, z, n_theta, K, boundaries,
                                     intervals, zeros(n_interior, K), 0.0,
                                     loss_sym, 0.0)
            println("  Max shooting gap: $(round(max_gap, sigdigits=4))")
            println("  Data-only loss: $(round(data_only_loss, sigdigits=5))")
        end

        # Check convergence (relative to state scale)
        final_max_gap = max_gap
        state_scale = norm(u0_init)
        if max_gap < 1e-2 * state_scale
            if verbose; println("  Shooting gaps converged!"); end
            conv_converged = true
            conv_reason = :converged_tol
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
        pred = Float64.(adam_simulate_discrete(prob, p_opt, Float64))
    else
        ode_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
            (du, u, params, t) -> prob.dynamics!(du, u, params, t))
        ode_prob = ODEProblem(ode_fn, Float64.(u0), prob.tspan, p_opt)
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

    # Objective in the TRAINING metric (matching AdamSolver's reporting):
    # weighted SSE for :mse, the weighted Poisson negative log-likelihood
    # kernel for :poisson, plus the fixed smoothing penalty when active.
    # data_loss above stays the descriptive weighted SSE in both cases.
    final_obj = if loss_sym == :poisson
        obj = 0.0
        for j in 1:n_obs, i in 1:T_pts
            usable_cell(prob, i, j) || continue
            mu = max(pred[i, j], 1e-10)
            y = prob.data_values[i, j]
            obj -= prob.data_weights[i, j] * (y * log(mu) - mu)
        end
        obj
    else
        data_loss
    end
    final_obj += adam_penalty(prob, theta_final, penalty_w)

    # Build evaluators for unknown functions
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = theta_final[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
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

    PSMSolution(params, final_obj, data_loss, edf, Float64[],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (optimizer=:lbfgs, method=:multiple_shooting,
                 n_intervals=n_intervals,
                 converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, max_gap=final_max_gap, rho_final=rho))
end
