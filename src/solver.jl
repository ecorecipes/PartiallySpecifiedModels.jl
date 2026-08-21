# Main solver: IRLS loop with LAML smoothing parameter estimation
#
# Algorithm (per iteration):
# 1. Evaluate model f(β) and compute Jacobian J by finite differences
# 2. Form pseudodata z = y - f + J*β
# 3. Solve penalized LS: min ||W^½(z - Jβ)||² + β'S^λβ
# 4. Step contraction (backtrack to ensure decrease)
# 5. Re-estimate smoothing parameters λ via LAML (Fellner-Schall + Newton)
# 6. Repeat until convergence

using LinearAlgebra: Diagonal, dot, tr, Symmetric, eigvals, cholesky, norm, eigen

# ─── Input validation ─────────────────────────────────────────────

"""
    _validate_problem(prob, solver_name; require_continuous=false)

Common input validation for all solve methods. Checks data dimensions,
approximator configuration, and observation mapping consistency.
"""
function _validate_problem(prob::PSMProblem, solver_name::String;
                           require_continuous::Bool=false)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    n_times == 0 && error("$solver_name: data_times is empty")
    issorted(prob.data_times) ||
        error("$solver_name: data_times must be sorted increasing (the " *
              "Kalman-filter and profiling paths assume monotone " *
              "observation indices)")
    size(prob.data_values, 1) != n_times &&
        error("$solver_name: data_values has $(size(prob.data_values, 1)) rows " *
              "but data_times has $n_times entries")
    size(prob.data_weights, 1) != n_times &&
        error("$solver_name: data_weights row count does not match data_times")
    length(prob.obs_to_state) != n_obs &&
        error("$solver_name: obs_to_state has $(length(prob.obs_to_state)) entries " *
              "but data_values has $n_obs columns")
    isempty(prob.approximators) &&
        error("$solver_name: no approximators specified")
    # Resolve u0 — may be a function of parameters (e.g., copepod model)
    if !(prob.u0 isa Function)
        any(s -> s < 1 || s > length(prob.u0), prob.obs_to_state) &&
            error("$solver_name: obs_to_state contains indices outside " *
                  "range 1:$(length(prob.u0))")
    end
    if require_continuous && prob.discrete
        error("$solver_name does not support discrete-time models. " *
              "The probabilistic ODE solver is designed for continuous ODEs. " *
              "Use LAML, GradientMatching, AdamSolver, BNGSolver, GCVSolver, " *
              "TwoStageSolver, DerivativeFreeSolver, or ABCSolver instead.")
    end
    nothing
end

# ─── Parameter layout ─────────────────────────────────────────────

"""Total number of parameters across all approximators."""
function n_total_params(prob::PSMProblem)
    sum(nparams(a) for a in prob.approximators)
end

"""Build initial parameter vector by concatenating approximator initial params."""
function build_initial_params(prob::PSMProblem)
    vcat([initial_params(a) for a in prob.approximators]...)
end

"""
    build_param_struct(prob, beta)

Build the parameter NamedTuple that the dynamics function receives.
Contains callable unknown functions and known parameters.
"""
function build_param_struct(prob::PSMProblem, beta::AbstractVector)
    offset = 0
    uf_entries = Pair{Symbol, Any}[]

    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np

        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            evaluator = build_bspline_evaluator(knots_x, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa NeuralApproximator
            # Dual-safe, eltype-generic (see neural_evaluator.jl) — required
            # for autodiff Jacobians in stiff ODE solvers and for gradients
            # of any objective w.r.t. β.
            evaluator = build_neural_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa GPApproximator
            evaluator = build_gp_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa ShapeConstrainedBSplineApproximator
            evaluator = build_constrained_bspline_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa COMONetApproximator
            evaluator = build_comonet_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa SPDEApproximator
            evaluator = build_spde_evaluator(approx.mesh_points, params_k)
            push!(uf_entries, approx.name => evaluator)
        elseif approx isa ShapeConstrainedSPDEApproximator
            evaluator = build_constrained_spde_evaluator(approx, params_k)
            push!(uf_entries, approx.name => evaluator)
        end
    end

    # Merge unknown function evaluators with known params
    uf_nt = NamedTuple(uf_entries)
    merge(uf_nt, prob.known_params)
end

"""
    _is_program_error(e)

Classify an exception caught inside a solver objective. Programming
errors (wrong method signature, out-of-bounds indexing, type bugs) must
be rethrown so they surface immediately; only genuinely numerical
failures (domain errors, singular factorizations, solver blow-ups) may
be converted into a large-but-finite penalty for the optimizer.

`InexactError` is special-cased: converting a NaN/Inf (e.g. inside
DataInterpolations when a diverged trajectory reaches a spline) is a
numerical failure the optimizer must survive, while converting a finite
value (`Int(3.7)`) is a genuine bug.
"""
function _is_program_error(e::InexactError)
    # Julia ≥ 1.12 stores the offending value as the last element of e.args;
    # earlier versions expose it as e.val.
    val = hasfield(InexactError, :val) ? e.val : e.args[end]
    val isa Number && isfinite(val)
end
_is_program_error(e) = e isa Union{MethodError, BoundsError, UndefVarError,
                                   TypeError, KeyError, DimensionMismatch,
                                   UndefRefError}

"""
    _adapt_gp_approximators!(prob, beta) -> Bool

Run the empirical-Bayes hyperparameter update for every `GPApproximator`
with `adapt=true`, using its slice of the current coefficient vector.
Returns whether any kernel changed (callers should re-evaluate the model).
"""
function _adapt_gp_approximators!(prob::PSMProblem, beta::AbstractVector)
    off = 0
    changed = false
    for a in prob.approximators
        np = nparams(a)
        if a isa GPApproximator && a.adapt
            changed |= _adapt_gp_hyperparams!(a, Float64.(beta[off+1:off+np]))
        end
        off += np
    end
    changed
end

# ─── Simulation ───────────────────────────────────────────────────

"""
    simulate(prob, beta)

Simulate the model with parameter vector β.
Returns predicted values at data times as matrix (n_times × n_obs).
Dispatches to ODE integration (continuous) or explicit iteration (discrete).
"""
function simulate(prob::PSMProblem, beta::AbstractVector)
    if !isempty(prob.delays)
        return simulate_dde(prob, beta)
    elseif prob.discrete
        return simulate_discrete(prob, beta)
    end
    return simulate_continuous(prob, beta)
end

"""
    simulate_continuous(prob, beta)

Simulate a continuous-time (ODE) model.
"""
function simulate_continuous(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0

    function ode_rhs!(du, u, params, t)
        prob.dynamics!(du, u, p, t)
    end

    ode_prob = ODEProblem(ode_rhs!, u0, prob.tspan)

    solve_kwargs = Dict{Symbol, Any}(
        :saveat => prob.data_times,
        :abstol => get(prob.ode_kwargs, :abstol, 1e-8),
        :reltol => get(prob.ode_kwargs, :reltol, 1e-8),
        :maxiters => get(prob.ode_kwargs, :maxiters, 1_000_000),
        :verbose => get(prob.ode_kwargs, :verbose, false),
    )
    merge!(solve_kwargs, prob.ode_kwargs)

    sol = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver; solve_kwargs...)

    # Check for solver failure (e.g. maxiters exceeded, instability)
    if sol.retcode != SciMLBase.ReturnCode.Success &&
       sol.retcode != SciMLBase.ReturnCode.Default &&
       sol.retcode != SciMLBase.ReturnCode.Terminated
        error("ODE solve failed: $(sol.retcode)")
    end

    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    pred = zeros(eltype(beta), n_times, n_obs)

    length(sol.u) >= n_times ||
        error("solve terminated after $(length(sol.u)) of $n_times save " *
              "points (retcode $(sol.retcode)); cannot form predictions " *
              "at all data times")
    for i in 1:n_times
        u_i = sol.u[i]
        for j in 1:n_obs
            pred[i, j] = u_i[prob.obs_to_state[j]]
        end
    end
    pred
end

"""
    simulate_discrete(prob, beta)

Simulate a discrete-time model by explicit iteration.
The dynamics function `f!(u_next, u, p, t)` computes `u(t+1)` from `u(t)`.

Iterates through all integer time steps from `tspan[1]` to `tspan[2]`,
recording state at `data_times`.
"""
function simulate_discrete(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    T = eltype(beta)

    n_vars = length(u0)
    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    pred = zeros(T, n_times, n_obs)

    t_start = prob.tspan[1]
    t_end = prob.tspan[2]

    # Build sorted set of all times we need to visit
    # (integer steps from tspan[1] to tspan[2])
    all_times = collect(t_start:1.0:t_end)

    # Map data_times to indices in all_times (allow non-integer data_times
    # by finding nearest time step)
    data_time_set = Dict{Float64, Vector{Int}}()
    for (di, dt) in enumerate(prob.data_times)
        # Round to nearest time step
        t_nearest = round(dt)
        if !haskey(data_time_set, t_nearest)
            data_time_set[t_nearest] = Int[]
        end
        push!(data_time_set[t_nearest], di)
    end

    u = T.(u0)
    u_next = similar(u)

    # Record initial condition if it's a data time
    t = t_start
    if haskey(data_time_set, t)
        for di in data_time_set[t]
            for j in 1:n_obs
                pred[di, j] = u[prob.obs_to_state[j]]
            end
        end
    end

    # Iterate forward
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
    predict(sol::PSMSolution, prob::PSMProblem)

Predict at data times using the fitted solution.
"""
function predict(sol::PSMSolution, prob::PSMProblem)
    sol.fitted_values
end

# ─── Finite-difference Jacobian ───────────────────────────────────

"""
    compute_jacobian!(J, prob, beta, f0, n_times, n_obs; dam)

Compute Jacobian of model predictions w.r.t. parameters using
central finite differences with adaptive step sizes.

J is (n_data × n_params), f0 is the flattened prediction vector.
`dam` contains adaptive fractional FD intervals per parameter.
"""
function compute_jacobian!(J::AbstractMatrix, prob::PSMProblem,
                           beta::AbstractVector, f0::AbstractVector,
                           n_times::Int, n_obs::Int;
                           dam::Vector{Float64})
    n_p = length(beta)
    n_data = n_times * n_obs
    p_pert = copy(beta)
    fp = zeros(n_data)
    fb = zeros(n_data)

    # Absolute FD step floor. DDEfit's fully relative step (da = dam·|β|,
    # floored at 1e-8·dam ≈ 1e-16 for β = 0) was safe there because Wood
    # (2001, p.11) reuses the SAME integration time steps for the perturbed
    # and unperturbed trajectories, so integrator error cancels in the
    # difference; simulate() re-solves adaptively per perturbation, so a
    # 1e-16 step measures pure solver noise. Tie the floor to the solver
    # tolerance instead: differences must exceed integration error.
    reltol_ode = Float64(get(prob.ode_kwargs, :reltol, 1e-8))
    abs_floor = max(100.0 * reltol_ode, 1e-7)

    for j in 1:n_p
        da = max(dam[j] * abs(beta[j]), abs_floor)

        # Forward perturbation
        p_pert[j] = beta[j] + da
        pred_fwd = try
            simulate(prob, p_pert)
        catch e
            _is_program_error(e) && rethrow()
            p_pert[j] = beta[j]
            J[:, j] .= 0.0   # don't leak the previous iteration's column
            continue
        end
        p_pert[j] = beta[j]

        # Backward perturbation
        p_pert[j] = beta[j] - da
        pred_bwd = try
            simulate(prob, p_pert)
        catch
            # Fall back to forward differences
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                J[k, j] = (pred_fwd[ti, oi] - f0[k]) / da
                k += 1
            end
            p_pert[j] = beta[j]
            continue
        end
        p_pert[j] = beta[j]

        # Flatten and compute central differences
        k = 1
        mean_te = 0.0
        mean_ce = 0.0
        for oi in 1:n_obs, ti in 1:n_times
            fp[k] = pred_fwd[ti, oi]
            fb[k] = pred_bwd[ti, oi]
            J[k, j] = (fp[k] - fb[k]) / (2.0 * da)
            mean_te += 0.5 * (fp[k] - 2.0 * f0[k] + fb[k]) / da
            mean_ce += 2.0 * max(abs(f0[k]), abs(fp[k])) * 1e-15 / da
            k += 1
        end

        # Adapt step size
        if dam[j] >= 1e-10 && abs(mean_te) > 10.0 * abs(mean_ce)
            dam[j] /= 10.0
        end
        if dam[j] <= 0.001 && abs(mean_ce) > 10.0 * abs(mean_te)
            dam[j] *= 10.0
        end
    end
end

# ─── Penalty matrix assembly ─────────────────────────────────────

"""
    build_penalty_matrices(prob)

Build per-approximator penalty matrices (unit smoothing parameter).
Returns `(S_list, offsets, nknots_list)`.
"""
function build_penalty_matrices(prob::PSMProblem)
    S_list = Matrix{Float64}[]
    offsets = Int[]
    nknots_list = Int[]

    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        S = penalty_matrix(approx)
        if S !== nothing && np >= 3
            push!(S_list, S)
            push!(offsets, offset)
            push!(nknots_list, np)
        end
        offset += np
    end
    S_list, offsets, nknots_list
end

"""
    penalty_sqrt_matrix(S)

Compute C such that C'C = S (via eigendecomposition of PSD matrix).
"""
function penalty_sqrt_matrix(S::AbstractMatrix)
    eig = eigen(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(eig.values)), 1e-14)
    pos_idx = findall(eig.values .> tol)
    if isempty(pos_idx)
        return zeros(0, size(S, 1))
    end
    Diagonal(sqrt.(eig.values[pos_idx])) * eig.vectors[:, pos_idx]'
end

# ─── Main solve function ─────────────────────────────────────────

"""
    SciMLBase.solve(prob::PSMProblem, alg::LAML)

Fit a partially specified model using IRLS with LAML smoothing.

# Algorithm
For each IRLS iteration:
1. Evaluate model and compute FD Jacobian
2. Form pseudodata z = y - f + J*β
3. Solve penalized least squares (augmented system)
4. Step contraction (backtracking)
5. Re-estimate smoothing parameters via Fellner-Schall + Newton

Returns a `PSMSolution`. `sol.convergence` is a NamedTuple
`(V_beta, sigma2, converged, iterations, reason, laml_failures)` — see the
`LAML` and `PSMSolution` docstrings for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::LAML)
    _validate_problem(prob, "LAML")
    maxiters = alg.maxiters
    verbose = alg.verbose

    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    n_data = n_times * n_obs
    n_p = n_total_params(prob)

    # Build penalty matrices per approximator
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)

    # Mixed-approximator dof advisory: parameters without a penalty block
    # (e.g. NeuralApproximator weights with penalty_weight = 0) are not REML
    # fixed effects in any useful sense when they rival the data size. The
    # LAML scale estimate replaces the rank-based restricted dof n − Mp with
    # an EDF-based count in that case (see laml.jl); warn when the rank-based
    # count would have been (nearly) exhausted so users know the model is
    # heavily over-parameterized relative to the data.
    n_unpenalized = n_p - sum(uf_nk; init=0)
    if m > 0 && n_unpenalized > 0
        total_rank = sum(_rank_penalty(S_list[l]) for l in 1:m)
        if n_data - (n_p - total_rank) < 10
            @warn "LAML: $n_unpenalized unpenalized parameters (e.g. neural network " *
                  "weights) leave at most $(n_data - (n_p - total_rank)) rank-based " *
                  "residual degrees of freedom for the n=$n_data data points. The " *
                  "Gaussian scale σ̂² uses the EDF-based restricted dof instead; " *
                  "consider more data, a smaller network, or penalty_weight > 0."
        end
    end

    # Initialize smoothing: user-specified or data-driven default.
    # The penalty matrices are computed on a normalised [0,1] domain,
    # so their eigenvalue spectrum is stable across problems.
    # Default: θ = 1/tr(S) ≈ 3.7e-5 (light initial smoothing).  LAML will
    # quickly adjust this once the warmup phase is complete.  For strongly
    # nonlinear problems, use initial_lambda=10.0 + warmup=5 or higher.
    if alg.initial_lambda !== nothing
        theta = fill(alg.initial_lambda, m)
    else
        theta = Float64[1.0 / max(tr(S_list[l]), 1e-10) for l in 1:m]
    end

    # Build total penalty B = Σ θ_k S_k (embedded in n_p × n_p)
    function build_B(th)
        B = zeros(n_p, n_p)
        for l in 1:m
            off = uf_offsets[l]
            nk = uf_nk[l]
            for i in 1:nk, j in 1:nk
                B[off+i, off+j] += th[l] * S_list[l][i, j]
            end
        end
        B
    end

    # Flatten data, enforcing the package's masking convention: a cell is
    # usable only if its weight is positive AND its datum is non-NaN.
    # Masked cells get weight 0 and a finite placeholder value — every
    # downstream use multiplies by the weight, but IEEE `0 * NaN = NaN`,
    # so leaving a NaN datum in y_vec would silently poison the objective
    # (the optimizer would then reject every step and return the initial
    # coefficients unchanged, without any error).
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y = prob.data_values[ti, oi]
        wv = prob.data_weights[ti, oi]
        if wv > 0 && !isnan(y)
            y_vec[k] = y
            w_vec[k] = wv
        end   # else keep the 0.0 placeholder with weight 0.0
        k += 1
    end

    # Evaluate model, return flattened predictions
    function eval_model(p_eval)
        pred = simulate(prob, p_eval)
        f_tmp = zeros(n_data)
        local k = 1
        for oi in 1:n_obs, ti in 1:n_times
            f_tmp[k] = pred[ti, oi]
            k += 1
        end
        f_tmp, pred
    end

    # Penalized objective: -ℓ(y,μ) + ½β'Bβ
    # For Gaussian this equals ½(RSS + penalty); for other families uses the
    # actual log-likelihood, ensuring correct step comparisons.
    function penalized_objective(p_eval, B)
        f_tmp, _ = try; eval_model(p_eval); catch; return Inf; end
        neg_ll = -log_likelihood(prob.likelihood, y_vec, f_tmp, w_vec)
        neg_ll + 0.5 * dot(p_eval, B * p_eval)
    end

    # PCLS step: truncated-SVD solve of the augmented system
    # [W^½J; C] β = [W^½z; 0] — see _pcls_augmented_solve in pcls.jl.
    # Uses IRLS weights that depend on the current predictions.
    function pcls_step(J_mat, z_pseudo, th, w_irls)
        B = build_B(th)
        _pcls_augmented_solve(J_mat, z_pseudo, B, w_irls), B
    end

    # Step contraction: backtracking with explosive-step rescue — see
    # _pcls_step_contract in pcls.jl.
    step_contract(a_old, a_new, B) =
        _pcls_step_contract(penalized_objective, a_old, a_new, B)

    # Initialize
    beta = build_initial_params(prob)
    J = zeros(n_data, n_p)
    f_vec = zeros(n_data)
    dam = fill(1e-8, n_p)

    if verbose
        println("IRLS+LAML: $n_p params, $n_data data, $m smooth terms")
        println("Initial θ: ", [round(t, sigdigits=4) for t in theta])
    end

    f_vec, _ = eval_model(beta)
    compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)

    # ─── Gaussian warm-start for non-Gaussian likelihoods ─────────
    # For Poisson/NegBin with identity link, the IRLS weights (1/V(μ))
    # can create local minima when the initial fit is poor.  We run a
    # full Gaussian IRLS+LAML solve (unit weights, profiled σ²) to find
    # good starting coefficients AND smoothing parameters, then switch
    # to the actual likelihood.  This mimics mgcv's initialization.
    if !(prob.likelihood isa Gaussian) && m > 0
        gw_otheta = copy(theta)
        prev_gw_obj = Inf

        for gw_iter in 1:50
            f_vec_new, _ = try; eval_model(beta); catch; break; end
            f_vec .= f_vec_new
            compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)

            w_gauss = copy(w_vec)
            z_pseudo = y_vec .- f_vec .+ J * beta

            # Try step with current θ
            a0_pcls, _ = pcls_step(J, z_pseudo, gw_otheta, w_gauss)
            a0, f01 = step_contract(beta, a0_pcls, build_B(gw_otheta))

            # After warmup iters, also estimate θ via Gaussian LAML
            if gw_iter > 3
                theta_new, _ = try
                    rho0 = log.(max.(gw_otheta, 1e-20))
                    estimate_smoothing_params(J, w_gauss, w_vec,
                        y_vec, f_vec, beta, S_list, uf_offsets, uf_nk, n_p;
                        family=Gaussian(), rho_init=rho0, verbose=false)
                catch; (copy(gw_otheta), NaN); end

                # Try step with new θ
                a1_pcls, B1 = pcls_step(J, z_pseudo, theta_new, w_gauss)
                a1, f11 = step_contract(beta, a1_pcls, B1)

                # Accept new θ only if it improves the Gaussian data fit.
                # Evaluate each candidate's model ONCE (a full ODE solve) —
                # the previous generator re-solved the ODE per data point.
                f_a0 = try; first(eval_model(a0)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                f_a1 = try; first(eval_model(a1)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                ss_a0 = sum((y_vec[i] - f_a0[i])^2 * w_vec[i] for i in 1:n_data)
                ss_a1 = sum((y_vec[i] - f_a1[i])^2 * w_vec[i] for i in 1:n_data)

                if ss_a1 <= ss_a0
                    beta .= a1
                    gw_otheta .= theta_new
                    f_vec .= f_a1
                else
                    beta .= a0
                    f_vec .= f_a0
                end
            else
                beta .= a0
                f_new = try; first(eval_model(a0)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                f_vec .= f_new
            end

            # Convergence monitor uses the CURRENT step's fit (the old code
            # scored f_vec from before the step — a one-iteration-stale
            # objective).
            gw_ss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
            gw_obj = 0.5 * (gw_ss + dot(beta, build_B(gw_otheta) * beta))

            if verbose && (gw_iter <= 3 || gw_iter % 10 == 0)
                println("Gauss-warmup $gw_iter: SS=$(round(gw_ss, sigdigits=6)), " *
                        "θ=$(round.(gw_otheta, sigdigits=3))")
            end

            if gw_iter > 5 && abs(gw_obj - prev_gw_obj) < 1e-6 * max(abs(prev_gw_obj), 1.0)
                if verbose; println("Gauss-warmup converged at iter $gw_iter"); end
                break
            end
            prev_gw_obj = gw_obj
        end

        theta .= gw_otheta
    end

    otheta = copy(theta)
    prev_obj = Inf  # Track penalized objective for convergence
    prev_data_loss = Inf  # Track data loss for non-Gaussian convergence

    # Honest convergence reporting (see PSMSolution docs): defaults describe
    # loop exhaustion; the breaks below overwrite them with the actual outcome.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    laml_failures = 0

    for iter in 0:(maxiters-1)
        conv_iters = iter + 1
        # Adapt GP kernel hyperparameters to the evolving fit (before the
        # model evaluation so f/J/W below are consistent with the new kernel)
        if iter >= alg.warmup
            _adapt_gp_approximators!(prob, beta)
        end
        # Re-evaluate model + Jacobian
        f_vec_new, _ = try; eval_model(beta); catch e
            if verbose; println("Iter $iter: simulation failed ($e)"); end
            conv_reason = :early_break
            break
        end
        f_vec .= f_vec_new
        compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)

        # Compute IRLS weights from current predictions
        w_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)

        # Form pseudodata z = y - f + J*β
        z_pseudo = y_vec .- f_vec .+ J * beta

        # PCLS with current (accepted) θ
        a0_pcls, _ = pcls_step(J, z_pseudo, otheta, w_irls)
        a0, f01 = step_contract(beta, a0_pcls, build_B(otheta))

        stop = false
        obj_prev = penalized_objective(beta, build_B(otheta))

        if iter > 0 && m > 0
            # PCLS with new θ (from LAML)
            a1_pcls, B_new = pcls_step(J, z_pseudo, theta, w_irls)
            a1, f11 = step_contract(beta, a1_pcls, B_new)

            f10 = penalized_objective(beta, B_new)

            # Compare old-θ step vs new-θ step using DATA LOSS (not penalized
            # objective).  Penalized objective is biased: lower θ → lower
            # penalty → lower objective even if the fit is worse.  Data loss
            # is θ-independent and gives an unbiased comparison.
            dl_a0 = -log_likelihood(prob.likelihood, y_vec,
                        (try; first(eval_model(a0)); catch; f_vec; end), w_vec)
            dl_a1 = -log_likelihood(prob.likelihood, y_vec,
                        (try; first(eval_model(a1)); catch; f_vec; end), w_vec)
            dl_curr = -log_likelihood(prob.likelihood, y_vec, f_vec, w_vec)

            if f11 < f10 && dl_a1 <= dl_a0
                # New theta + step is best (data loss confirms)
                f1_vec, _ = try; eval_model(a1); catch; (f_vec, nothing); end
                beta .= a1
                f_vec .= f1_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)
                otheta .= theta
            elseif f01 < obj_prev
                # Old theta step improved at old theta
                f0_vec, _ = try; eval_model(a0); catch; (f_vec, nothing); end
                beta .= a0
                f_vec .= f0_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)
                # Also accept new theta if it didn't make data loss worse
                if dl_a1 < dl_curr
                    otheta .= theta
                end
            elseif f11 < f10
                # New theta step improved within new theta's metric
                f1_vec, _ = try; eval_model(a1); catch; (f_vec, nothing); end
                beta .= a1
                f_vec .= f1_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)
                otheta .= theta
            else
                # No improvement from either
                if iter >= 10
                    stop = true
                end
                theta .= otheta
            end
        else
            # First iteration: accept a0 and update otheta
            f0_vec, _ = try; eval_model(a0); catch; (f_vec, nothing); end
            beta .= a0
            f_vec .= f0_vec
            compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam)
        end

        # Track penalized objective for convergence monitoring
        curr_obj = penalized_objective(beta, build_B(theta))

        if verbose && (iter <= 4 || iter % 10 == 0)
            curr_data_ss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
            println("Iter $iter: obj=$(round(curr_obj, sigdigits=6)), " *
                    "SS=$(round(curr_data_ss, sigdigits=6)), " *
                    "θ=$(round.(theta, sigdigits=3))")
        end

        # Check convergence: relative change in penalized objective AND data fit.
        # For non-Gaussian likelihoods, the penalized objective can appear stable
        # (large |obj| makes relative tolerance easy to meet) while the data fit
        # is still poor.  Require both objective stability AND small relative
        # change in data loss.
        # Don't converge before warmup is complete — the smoothing parameters
        # haven't been optimised yet and the objective may improve further.
        min_conv_iter = max(3, alg.warmup + 3)
        curr_data_loss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
        # alg.tol governs the penalized-objective test (as documented); the
        # data-loss test uses a proportionally looser threshold.
        obj_stable = abs(curr_obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
        dl_stable = prev_data_loss < Inf &&
                    abs(curr_data_loss - prev_data_loss) <
                        100 * alg.tol * max(prev_data_loss, 1.0)
        if iter >= min_conv_iter && obj_stable && dl_stable
            if verbose; println("Converged at iter $iter (objective stable)"); end
            conv_converged = true
            conv_reason = :converged_tol
            break
        end
        prev_obj = curr_obj
        prev_data_loss = curr_data_loss

        if stop && iter >= min_conv_iter
            if verbose; println("Converged at iter $iter (no improvement)"); end
            conv_converged = true
            conv_reason = :plateau
            break
        end

        # Re-estimate smoothing parameters via LAML.
        # Use theta (latest), NOT otheta, for warm-start so Fellner-Schall
        # doesn't restart from scratch.
        w_irls_for_laml = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
        if m > 0 && iter >= alg.warmup
            # sigma2_init caps the FS dispersion during early iterations to
            # prevent runaway smoothing while the fit is still poor; as
            # documented, the cap relaxes (×10 per iteration past warmup)
            # so it cannot permanently bias λ downward.
            s2cap = if alg.sigma2_init === nothing
                Inf
            else
                alg.sigma2_init * 10.0^clamp(iter - alg.warmup, 0, 300)
            end
            theta_new, _ = try
                rho_init = log.(max.(theta, 1e-20))
                estimate_smoothing_params(J, w_irls_for_laml, w_vec,
                                         y_vec, f_vec, beta,
                                         S_list, uf_offsets, uf_nk, n_p;
                                         family=prob.likelihood,
                                         rho_init=rho_init,
                                         sigma2_max=s2cap,
                                         verbose=verbose)
            catch e
                if verbose; println("LAML failed: $e, keeping theta"); end
                laml_failures += 1
                (copy(theta), NaN)
            end
            theta .= theta_new
        end
    end

    # Build solution
    p_opt = copy(beta)
    pred = simulate(prob, p_opt)

    # Compute data loss (masked cells — zero weight or NaN datum — are
    # skipped; `0 * NaN = NaN` would otherwise contaminate the total)
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        wv = prob.data_weights[i,j]
        y = prob.data_values[i,j]
        (wv > 0 && !isnan(y)) || continue
        data_loss += wv * (y - pred[i,j])^2
    end

    # EDF from hat matrix
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        f_vec[k] = pred[ti, oi]
        k += 1
    end
    compute_jacobian!(J, prob, p_opt, f_vec, n_times, n_obs; dam=dam)

    B_final = build_B(theta)
    W_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
    JWJ = J' * Diagonal(W_irls) * J
    H_final = JWJ + B_final
    maxd = maximum(abs.(diag(H_final)))
    for i in 1:n_p
        H_final[i,i] += 1e-12 * maxd + 1e-15
    end
    edf = try
        tr(cholesky(Symmetric(H_final)) \ JWJ)
    catch
        tr(H_final \ JWJ)
    end

    pen_ss = dot(p_opt, B_final * p_opt)
    obj_val = 0.5 * (data_loss + pen_ss)

    # Build ComponentArray for nice parameter access
    uf_syms = Symbol[a.name for a in prob.approximators]
    uf_vals = Vector{Float64}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(uf_vals, Float64.(p_opt[offset+1:offset+np]))
        offset += np
    end
    params = ComponentArray(NamedTuple{Tuple(uf_syms)}(Tuple(uf_vals)))

    # Build unknown function evaluators for the solution
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = p_opt[offset+1:offset+np]
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

    if verbose
        println("\nFinal: data_loss = $(round(data_loss, sigdigits=6)), " *
                "penalty = $(round(pen_ss, sigdigits=6)), " *
                "EDF = $(round(edf, digits=2))")
        println("Final θ: ", [round(t, sigdigits=4) for t in theta])
    end

    # Compute Bayesian posterior covariance V_β = (J'WJ + S^λ)⁻¹
    # This gives "across-the-function" CIs with near-nominal coverage
    # (Nychka 1988, Wood 2006 §4.8)
    V_beta = try
        inv(cholesky(Symmetric(H_final)))
    catch
        try; inv(Symmetric(H_final)); catch; nothing; end
    end

    # Estimate σ² for Gaussian (needed for CI scaling)
    sigma2_hat = if prob.likelihood isa Gaussian
        data_loss / max(n_data - edf, 1.0)
    else
        1.0  # non-Gaussian: V_β already on natural scale
    end

    convergence_info = (V_beta=V_beta, sigma2=sigma2_hat,
                        converged=conv_converged, iterations=conv_iters,
                        reason=conv_reason, laml_failures=laml_failures)

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals, convergence_info)
end
