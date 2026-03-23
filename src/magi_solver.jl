# ─── MAGI Solver ─────────────────────────────────────────────────────
#
# Manifold-constrained Gaussian process inference for ODE systems.
#
# Algorithm:
#   1. Discretize time on a fine grid
#   2. Model each state component as Integrated Brownian Motion (IBM)
#   3. Use Kalman filter to evaluate log p(ODE constraint | θ)
#      - States are marginalized out (only θ is sampled)
#   4. Condition on data via observation updates in the Kalman filter
#   5. Sample θ via NUTS/HMC
#
# Reference: Yang, Wong & Kou (2021) PNAS 118(15)

using LinearAlgebra

# ─── MAGI log-density via Kalman filter ─────────────────────────────

"""
    MAGILogDensity

LogDensityProblems.jl interface for the MAGI log-posterior.
The Kalman filter marginalizes states; only θ (UF params) is sampled.
"""
struct MAGILogDensity{P <: PSMProblem}
    prob::P
    grid_times::Vector{Float64}
    obs_indices::Vector{Int}         # grid indices closest to data times
    n_deriv::Int
    obs_var::Float64
    prior_scale::Float64
    sigma::Vector{Float64}           # IBM scale per state
    n_vars::Int                      # number of ODE state variables
    n_params::Int                    # total number of sampled parameters
end

function LogDensityProblems.dimension(ld::MAGILogDensity)
    return ld.n_params
end

function LogDensityProblems.capabilities(::Type{<:MAGILogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

"""
    _magi_kalman_loglik(ld, theta)

Run Kalman filter forward pass and return log-likelihood of the ODE
constraint + data observations. This is the core MAGI computation.

For each time step on the grid:
1. PREDICT: IBM transition x_{n} from x_{n-1}
2. ODE CONSTRAINT: Measure derivative x'(t) = f(x, θ, t)
3. DATA UPDATE: If this grid point has observations, condition on them
"""
function _magi_kalman_loglik(ld::MAGILogDensity, theta::AbstractVector{T}) where T
    prob = ld.prob
    n_vars = ld.n_vars
    n_deriv = ld.n_deriv
    grid = ld.grid_times
    n_grid = length(grid)

    # Build parameter struct from theta
    offset = 0
    uf_entries = Pair{Symbol, Any}[]
    for approx in prob.approximators
        np = nparams(approx)
        params_k = theta[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            push!(uf_entries, approx.name => build_bspline_evaluator(knots_x, params_k))
        elseif approx isa ShapeConstrainedBSplineApproximator
            push!(uf_entries, approx.name => build_constrained_bspline_evaluator(approx, params_k))
        elseif approx isa COMONetApproximator
            push!(uf_entries, approx.name => build_comonet_evaluator(approx, params_k))
        elseif approx isa GPApproximator
            push!(uf_entries, approx.name => build_gp_evaluator(approx, params_k))
        elseif approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            let pk = params_k, s = spec, lo_ = lo, span_ = span
                evaluator = x -> begin
                    xval = x isa AbstractArray ? x[1] : x
                    xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                        (xval - lo_) / span_
                    else
                        xval
                    end
                    mlp_evaluate(s, pk, xn)
                end
                push!(uf_entries, approx.name => evaluator)
            end
        end
    end
    p_nt = merge(NamedTuple(uf_entries), prob.known_params)

    # IBM matrices for each state variable
    obs_var_mat = T.(fill(ld.obs_var, 1, 1))
    D = zeros(T, 1, n_deriv)
    D[1, 1] = one(T)  # observation picks out x^(0)

    # W picks out x^(1) (first derivative) for ODE constraint
    W = zeros(T, 1, n_deriv)
    W[1, 2] = one(T)

    # Small positive noise for ODE constraint (softened manifold constraint)
    # Prevents covariance collapse and improves NUTS mixing
    V_ode = T(1e-4) * ones(T, 1, 1)

    # Initialize Kalman states
    u0_vec = T.(prob.u0 isa Function ? prob.u0(p_nt) : prob.u0)
    du0 = zeros(T, n_vars)
    prob.dynamics!(du0, u0_vec, p_nt, grid[1])

    # μ[k] = [x_k, ẋ_k, 0, ...], Σ[k] = small diagonal
    μ_filt = [zeros(T, n_deriv) for _ in 1:n_vars]
    Σ_filt = [Matrix{T}(T(1e-4) * I, n_deriv, n_deriv) for _ in 1:n_vars]
    for k in 1:n_vars
        μ_filt[k][1] = u0_vec[k]
        if n_deriv >= 2
            μ_filt[k][2] = du0[k]
        end
    end

    loglik = zero(T)
    obs_idx_ptr = 1  # pointer into obs_indices

    # Process data at grid[1] (initial time) if present
    if obs_idx_ptr <= length(ld.obs_indices) && ld.obs_indices[obs_idx_ptr] == 1
        data_row = obs_idx_ptr
        for j in 1:size(prob.data_values, 2)
            state_idx = prob.obs_to_state[j]
            y_raw = prob.data_values[data_row, j]
            if !isnan(y_raw)
                y_obs = T(y_raw)
                μ_pred_obs = (D * μ_filt[state_idx])[1]
                S_pred_obs = (D * Σ_filt[state_idx] * D')[1, 1] + T(ld.obs_var)
                res = y_obs - μ_pred_obs
                loglik += T(-0.5) * (log(max(S_pred_obs, T(1e-300))) + res^2 / S_pred_obs)

                z_data = T[y_obs]
                d_zero = zeros(T, 1)
                μ_filt[state_idx], Σ_filt[state_idx] = kalman_update(
                    μ_filt[state_idx], Σ_filt[state_idx],
                    z_data, d_zero, T.(D), T.(obs_var_mat))
            end
        end
        obs_idx_ptr += 1
    end

    for n in 2:n_grid
        dt = grid[n] - grid[n-1]
        t_n = grid[n]

        for k in 1:n_vars
            # IBM transition matrices
            Q_k, R_k = ibm_state(Float64(dt), n_deriv, Float64(ld.sigma[k]))
            Q_T = T.(Q_k)
            R_T = T.(R_k)

            # PREDICT
            μ_pred = Q_T * μ_filt[k]
            Σ_pred = Q_T * Σ_filt[k] * Q_T' + R_T
            Σ_pred = T(0.5) * (Σ_pred + Σ_pred')
            # Numerical floor to prevent singular covariance
            for d in 1:n_deriv
                Σ_pred[d, d] = max(Σ_pred[d, d], T(1e-12))
            end

            μ_filt[k] = μ_pred
            Σ_filt[k] = Σ_pred
        end

        # ODE CONSTRAINT UPDATE: measure x'(t) = f(x, θ, t)
        # Extract current state estimates for ODE evaluation
        # Clamp to prevent extreme values that could cause NaN in evaluators
        u_current = T[clamp(μ_filt[k][1], T(-1e6), T(1e6)) for k in 1:n_vars]

        du_current = zeros(T, n_vars)
        prob.dynamics!(du_current, u_current, p_nt, t_n)

        # Clamp ODE RHS to prevent extreme innovations
        for k in 1:n_vars
            du_current[k] = clamp(du_current[k], T(-1e6), T(1e6))
        end

        for k in 1:n_vars
            # Innovation: ODE says ẋ_k = du_current[k], Kalman predicts ẋ_k = W*μ
            z_ode = T[du_current[k]]
            ν = z_ode - W * μ_filt[k]
            S = W * Σ_filt[k] * W' + V_ode
            S_val = S[1, 1]

            if abs(S_val) > T(1e-20)
                # Log-likelihood contribution (clamped for numerical safety)
                loglik += T(-0.5) * (log(max(abs(S_val), T(1e-300))) + ν[1]^2 / S_val)

                # Kalman update
                K_gain = Σ_filt[k] * W' / S
                # Clamp gain to prevent explosive updates
                K_gain = clamp.(K_gain, T(-100), T(100))
                μ_filt[k] = μ_filt[k] + K_gain * ν
                Σ_filt[k] = Σ_filt[k] - K_gain * W * Σ_filt[k]
                Σ_filt[k] = T(0.5) * (Σ_filt[k] + Σ_filt[k]')
                # Ensure positive diagonal
                for d in 1:n_deriv
                    Σ_filt[k][d, d] = max(Σ_filt[k][d, d], T(1e-12))
                end
            end
        end

        # DATA UPDATE: if this grid point has observations
        if obs_idx_ptr <= length(ld.obs_indices) && n == ld.obs_indices[obs_idx_ptr]
            data_row = obs_idx_ptr
            for j in 1:size(prob.data_values, 2)
                state_idx = prob.obs_to_state[j]
                y_raw = prob.data_values[data_row, j]
                if !isnan(y_raw)
                    y_obs = T(y_raw)
                    # Data log-likelihood from PREDICTED (pre-update) distribution
                    μ_pred_obs = (D * μ_filt[state_idx])[1]
                    S_pred_obs = (D * Σ_filt[state_idx] * D')[1, 1] + T(ld.obs_var)
                    res = y_obs - μ_pred_obs
                    loglik += T(-0.5) * (log(max(S_pred_obs, T(1e-300))) + res^2 / S_pred_obs)

                    # Kalman update (condition on observation)
                    z_data = T[y_obs]
                    d_zero = zeros(T, 1)
                    μ_filt[state_idx], Σ_filt[state_idx] = kalman_update(
                        μ_filt[state_idx], Σ_filt[state_idx],
                        z_data, d_zero, T.(D), T.(obs_var_mat))
                end
            end
            obs_idx_ptr += 1
        end
    end

    return loglik
end

function LogDensityProblems.logdensity(ld::MAGILogDensity, theta::AbstractVector)
    T = eltype(theta)

    # Log-likelihood from Kalman filter
    ll = _magi_kalman_loglik(ld, theta)

    # Prior: Gaussian on all parameters
    prior = zero(T)
    offset = 0
    for approx in ld.prob.approximators
        np = nparams(approx)
        params_k = theta[offset+1:offset+np]
        offset += np
        # Penalty matrix prior if available
        if approx isa BSplineApproximator || approx isa ShapeConstrainedBSplineApproximator ||
           approx isa GPApproximator || approx isa COMONetApproximator
            S = penalty_matrix(approx)
            prior += T(-0.5) / T(ld.prior_scale) * dot(params_k, T.(S) * params_k)
        end
        # Broad Gaussian prior on all params
        prior += T(-0.5) / T(100.0 * ld.prior_scale) * dot(params_k, params_k)
    end

    # Return -Inf for NaN, clamp extreme values (invalid parameter region)
    result = ll + prior
    fval = ForwardDiff.value(result)
    return isfinite(fval) ? result : T(-1e10)
end

# ─── Main solve method ──────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::MagiSolver)

Fit a partially specified model using the MAGI (MAnifold-constrained
Gaussian process Inference) method. Places an integrated Brownian motion
prior on the state and uses Kalman filtering/smoothing to jointly infer
the trajectory and unknown-function parameters.

# Algorithm
1. Auto-estimate IBM σ from the data range (or use user-supplied value).
2. Set up the IBM prior of order `n_deriv` for each state variable.
3. Forward Kalman filter with data and ODE pseudo-observations.
4. RTS backward smoother to obtain the posterior state.
5. Optimise parameters by maximising the marginal log-likelihood,
   optionally using multiple restarts and grid-search initialisation.

# References
- Yang, Wong & Kou (2021), "Inference of dynamic systems from noisy and
  sparse data via manifold-constrained Gaussian processes", PNAS.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MagiSolver)
    _validate_problem(prob, "MagiSolver"; require_continuous=true)
    verbose = alg.verbose
    n_vars = length(prob.u0 isa Function ? prob.u0((;)) : prob.u0)

    # Auto-estimate IBM sigma from data range
    # Use a moderate scale — too large causes numerical overflow in IBM matrices
    sigma = if alg.sigma !== nothing
        alg.sigma
    else
        sig = Float64[]
        for j in 1:size(prob.data_values, 2)
            col = prob.data_values[:, j]
            valid = col[.!isnan.(col)]
            s = isempty(valid) ? 1.0 : max(std(valid), 0.1)
            push!(sig, clamp(s, 0.1, 10.0))
        end
        # Extend to n_vars if fewer observations than states
        while length(sig) < n_vars
            push!(sig, 1.0)
        end
        sig
    end

    if verbose
        println("MAGI: n_vars=$n_vars, n_deriv=$(alg.n_deriv), " *
                "n_grid=$(alg.n_gridpoints), sigma=$(round.(sigma, digits=3))")
    end

    # Build fine time grid
    t0, tf = prob.tspan
    grid_times = collect(range(t0, tf, length=alg.n_gridpoints))

    # Map data times to nearest grid indices
    obs_indices = Int[]
    for td in prob.data_times
        _, idx = findmin(abs.(grid_times .- td))
        push!(obs_indices, idx)
    end

    # Total parameters = sum of all approximator params
    n_params = sum(nparams(a) for a in prob.approximators)

    # Initialize parameters
    theta0 = Float64[]
    for approx in prob.approximators
        append!(theta0, initial_params(approx))
    end

    # Build log-density
    ld = MAGILogDensity(prob, grid_times, obs_indices,
                        alg.n_deriv, alg.obs_var, alg.prior_scale,
                        sigma, n_vars, n_params)

    # Pre-optimization: find MAP estimate as NUTS starting point
    if alg.preoptimize
        if verbose
            println("MAGI: Pre-optimizing with Nelder-Mead...")
        end
        neg_logdens = x -> begin
            v = -LogDensityProblems.logdensity(ld, x)
            isfinite(v) ? v : 1e10
        end
        try
            opt_result = Optim.optimize(neg_logdens, theta0, Optim.NelderMead(),
                Optim.Options(iterations=2000, show_trace=false))
            if Optim.minimum(opt_result) < neg_logdens(theta0)
                theta0 = Optim.minimizer(opt_result)
                if verbose
                    println("MAGI: Pre-optimization improved logdensity to " *
                            "$(round(-Optim.minimum(opt_result), digits=2))")
                end
            end
        catch e
            if verbose
                println("MAGI: Pre-optimization failed ($(typeof(e))), using initial params")
            end
        end
    end

    # Wrap with ForwardDiff for gradients
    ld_ad = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ld)

    # NUTS sampler
    n_total = alg.n_warmup + alg.n_samples
    nuts = AdvancedHMC.NUTS(alg.target_accept)

    if verbose
        println("MAGI: Starting NUTS sampling ($n_total iterations, " *
                "$(alg.n_warmup) warmup)...")
    end

    # Sample
    chain_raw = AbstractMCMC.sample(ld_ad, nuts, n_total;
                                    initial_params=theta0,
                                    progress=verbose)

    # Extract posterior samples (discard warmup)
    n_keep = alg.n_samples
    sample_matrix = zeros(n_keep, n_params)
    for i in 1:n_keep
        sample_matrix[i, :] = chain_raw[alg.n_warmup + i].z.θ
    end

    # Build parameter names
    param_names = String[]
    for approx in prob.approximators
        np = nparams(approx)
        for j in 1:np
            push!(param_names, "$(approx.name)[$j]")
        end
    end

    chains = MCMCChains.Chains(sample_matrix, Symbol.(param_names))

    if verbose
        println("MAGI: Sampling complete.")
    end

    # Build MAP evaluators from posterior mean
    map_beta = vec(mean(sample_matrix, dims=1))
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = map_beta[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] = build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa NeuralApproximator
            spec = mlp_spec_from_lux(approx.model)
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
        end
    end

    # Build ComponentArray for parameters
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => map_beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    # Return solution with chains in convergence field
    PSMSolution(params, 0.0, 0.0, Float64(n_params), Float64[],
                zeros(length(prob.data_times), max(size(prob.data_values, 2), 1)),
                Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (method=:magi, chains=chains))
end
