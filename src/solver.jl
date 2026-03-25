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
    any(s -> s < 1 || s > length(prob.u0), prob.obs_to_state) &&
        error("$solver_name: obs_to_state contains indices outside " *
              "range 1:$(length(prob.u0))")
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
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(Lux.initialparameters(rng2, approx.model)))
            ps_vec = similar(ps_ca)
            ps_vec .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            evaluator = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model, Float32.(reshape([xn], :, 1)), ps_vec, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
            end
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

    for j in 1:n_p
        da = dam[j] * abs(beta[j])
        if da < 1e-8 * dam[j]
            da = 1e-8 * dam[j]
        end

        # Forward perturbation
        p_pert[j] = beta[j] + da
        pred_fwd = try
            simulate(prob, p_pert)
        catch
            p_pert[j] = beta[j]
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

Returns a `PSMSolution`.
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

    # Flatten data
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y_vec[k] = prob.data_values[ti, oi]
        w_vec[k] = prob.data_weights[ti, oi]
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

    # PCLS step: augmented system [W^½J; C] β = [W^½z; 0]
    # Uses IRLS weights that depend on the current predictions.
    function pcls_step(J_mat, z_pseudo, th, w_irls)
        B = build_B(th)
        C = penalty_sqrt_matrix(B)
        n_pen = size(C, 1)
        W_sqrt = sqrt.(max.(w_irls, 1e-15))
        F_aug = vcat(Diagonal(W_sqrt) * J_mat, C)
        z_aug = vcat(W_sqrt .* z_pseudo, zeros(n_pen))
        F_aug \ z_aug, B
    end

    # Step contraction: backtracking line search with exponential step sizes
    # Tries α = 1, 0.5, 0.25, ..., 2^(-15) ≈ 3e-5 to find a step that
    # reduces the penalized objective. This handles highly nonlinear models
    # where the full PCLS step overshoots badly.
    function step_contract(a_old, a_new, B)
        f_old = penalized_objective(a_old, B)
        direction = a_new .- a_old

        best_f = f_old
        best_a = copy(a_old)

        for k in 0:15
            α = 2.0^(-k)
            a_try = a_old .+ α .* direction
            f_try = penalized_objective(a_try, B)
            if f_try < best_f
                best_f = f_try
                best_a = copy(a_try)
            end
        end
        best_a, best_f
    end

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

    otheta = copy(theta)
    prev_obj = Inf  # Track penalized objective for convergence

    for iter in 0:(maxiters-1)
        # Re-evaluate model + Jacobian
        f_vec_new, _ = try; eval_model(beta); catch e
            if verbose; println("Iter $iter: simulation failed ($e)"); end
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

        # Check convergence: relative change in penalized objective.
        # Don't converge before warmup is complete — the smoothing parameters
        # haven't been optimised yet and the objective may improve further.
        min_conv_iter = max(3, alg.warmup + 3)
        if iter >= min_conv_iter && abs(curr_obj - prev_obj) < 1e-6 * max(abs(prev_obj), 1.0)
            if verbose; println("Converged at iter $iter (objective stable)"); end
            break
        end
        prev_obj = curr_obj

        if stop && iter >= min_conv_iter
            if verbose; println("Converged at iter $iter (no improvement)"); end
            break
        end

        # Re-estimate smoothing parameters via LAML.
        # Use theta (latest), NOT otheta, for warm-start so Fellner-Schall
        # doesn't restart from scratch.
        w_irls_for_laml = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
        if m > 0 && iter >= alg.warmup
            s2cap = alg.sigma2_init !== nothing ? alg.sigma2_init : Inf
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
                (copy(theta), NaN)
            end
            theta .= theta_new
        end
    end

    # Build solution
    p_opt = copy(beta)
    pred = simulate(prob, p_opt)

    # Compute data loss
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i,j] * (prob.data_values[i,j] - pred[i,j])^2
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
            rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(Lux.initialparameters(rng2, approx.model)))
            ps_final = similar(ps_ca)
            ps_final .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model, Float32.(reshape([xn], :, 1)), ps_final, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
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

    if verbose
        println("\nFinal: data_loss = $(round(data_loss, sigdigits=6)), " *
                "penalty = $(round(pen_ss, sigdigits=6)), " *
                "EDF = $(round(edf, digits=2))")
        println("Final θ: ", [round(t, sigdigits=4) for t in theta])
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals, nothing)
end
