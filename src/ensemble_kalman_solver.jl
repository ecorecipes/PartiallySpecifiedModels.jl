# ─── Ensemble Kalman Inversion (EKI) solver ────────────────────────
#
# Derivative-free ensemble method for parameter estimation.
# Maintains an ensemble of parameter particles, propagates each through
# the forward model, and updates via the Kalman gain.
#
# Reference: Iglesias, Law & Stuart (2013), Inverse Problems
#            Schillings & Stuart (2017), SIAM J Numer Anal
#            Kovachki & Stuart (2019), Inverse Problems

using LinearAlgebra: dot, norm, Symmetric, pinv

"""
    solve(prob::PSMProblem, alg::EnsembleKalmanSolver)

Fit a partially specified model using Ensemble Kalman Inversion (EKI).

Uses an ensemble of parameter particles that are iteratively updated via
the Kalman gain to match observations.  The method is derivative-free
and naturally handles non-smooth or stiff forward models.

# Algorithm
1. Initialise J ensemble members θ⁽ʲ⁾ from a prior (centred on initial params).
2. For each iteration n:
   a. Evaluate forward model G(θ⁽ʲ⁾) for each particle.
   b. Compute ensemble covariances: Cθg, Cgg.
   c. Update: θ⁽ʲ⁾ₙ₊₁ = θ⁽ʲ⁾ₙ + Cθg (Cgg + Γ)⁻¹ (y + ξ⁽ʲ⁾ − G(θ⁽ʲ⁾ₙ))
3. Return ensemble mean as the point estimate.

# Returns
`PSMSolution` with fitted parameters and `convergence` containing
`:ensemble_spread` (final ensemble std) and `:ensemble_history`.
"""
function SciMLBase.solve(prob::PSMProblem, alg::EnsembleKalmanSolver)
    _validate_problem(prob, "EnsembleKalmanSolver")
    verbose = alg.verbose

    J = alg.n_ensemble
    n_iter = alg.n_iterations
    σ_obs = alg.noise_scale

    # Observation vector
    y_obs = vec(prob.data_values)
    n_data = length(y_obs)

    # Initial parameters
    beta0 = build_initial_params(prob)
    n_beta = length(beta0)

    if verbose
        println("EnsembleKalmanSolver: $J particles, $n_iter iterations, $n_beta params")
    end

    # ── Forward model: θ → G(θ) (predicted observations) ────────
    function forward_model(theta::Vector{Float64})
        p = build_param_struct(prob, theta)
        pred = zeros(length(prob.data_times), size(prob.data_values, 2))

        try
            if prob.discrete
                u = Float64.(prob.u0 isa Function ? prob.u0(p) : prob.u0)
                n_vars = length(u)
                du = zeros(n_vars)
                for i in 1:length(prob.data_times)
                    for j in 1:size(prob.data_values, 2)
                        sk = prob.obs_to_state[j]
                        pred[i, j] = u[sk]
                    end
                    if i < length(prob.data_times)
                        prob.dynamics!(du, u, p, prob.data_times[i])
                        u = copy(du)
                    end
                end
            else
                ode_u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
                ode_prob = ODEProblem(prob.dynamics!, ode_u0, prob.tspan, p)
                solver = prob.ode_solver === nothing ? Tsit5() : prob.ode_solver
                ode_sol = OrdinaryDiffEq.solve(ode_prob, solver;
                            saveat=prob.data_times, prob.ode_kwargs...)
                if ode_sol.retcode != :Success && ode_sol.retcode != SciMLBase.ReturnCode.Success
                    return fill(1e6, n_data)
                end
                for i in 1:length(prob.data_times)
                    for j in 1:size(prob.data_values, 2)
                        sk = prob.obs_to_state[j]
                        pred[i, j] = ode_sol.u[i][sk]
                    end
                end
            end
        catch
            return fill(1e6, n_data)
        end

        vec(pred)
    end

    # ── Initialise ensemble ──────────────────────────────────────
    rng = Random.Xoshiro(42)
    ensemble = Matrix{Float64}(undef, n_beta, J)
    for j in 1:J
        ensemble[:, j] = beta0 .+ 0.5 .* randn(rng, n_beta) .* max.(abs.(beta0), 0.1)
    end

    # Observation noise covariance
    Γ = σ_obs^2 * Matrix{Float64}(I, n_data, n_data)

    spread_history = Float64[]

    # ── EKI iterations ───────────────────────────────────────────
    for iter in 1:n_iter
        # Evaluate forward model for each particle
        G = Matrix{Float64}(undef, n_data, J)
        for j in 1:J
            G[:, j] = forward_model(ensemble[:, j])
        end

        # Ensemble means
        θ_mean = vec(mean(ensemble, dims=2))
        G_mean = vec(mean(G, dims=2))

        # Ensemble anomalies
        Δθ = ensemble .- θ_mean
        ΔG = G .- G_mean

        # Cross-covariance Cθg and auto-covariance Cgg
        Cθg = (Δθ * ΔG') / (J - 1)
        Cgg = (ΔG * ΔG') / (J - 1)

        # Kalman gain: K = Cθg * (Cgg + Γ)⁻¹
        # Use pseudo-inverse for numerical stability
        K = Cθg * pinv(Cgg + Γ)

        # Update each particle
        for j in 1:J
            ξ = σ_obs .* randn(rng, n_data)
            innovation = y_obs .+ ξ .- G[:, j]
            ensemble[:, j] .+= K * innovation
        end

        # Track ensemble spread
        spread = mean(std(ensemble, dims=2))
        push!(spread_history, spread)

        if verbose && (iter <= 3 || iter % 5 == 0 || iter == n_iter)
            misfit = mean(abs2, G_mean .- y_obs)
            println("  iter $iter: misfit=$(round(misfit, sigdigits=4)) " *
                    "spread=$(round(spread, sigdigits=4))")
        end
    end

    # ── Build solution from ensemble mean ────────────────────────
    beta_final = vec(mean(ensemble, dims=2))
    ensemble_std = vec(std(ensemble, dims=2))

    # Simulate at ensemble mean for fitted values
    pred_vec = forward_model(beta_final)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)
    pred = reshape(pred_vec, n_times, n_obs)

    data_loss = sum(abs2, prob.data_values .- pred)

    # Build UF evaluators
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_final[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] = build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa SPDEApproximator
            uf_evals[approx.name] = build_spde_evaluator(approx.mesh_points, params_k)
        elseif approx isa ShapeConstrainedSPDEApproximator
            uf_evals[approx.name] = build_constrained_spde_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
        elseif approx isa NeuralApproximator
            rng_nn = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng_nn, approx.model)
            rng_nn2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(Lux.initialparameters(rng_nn2, approx.model)))
            ps_vec = similar(ps_ca)
            ps_vec .= params_k
            lo = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing : (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model, Float32.(reshape([xn], :, 1)), ps_vec, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
            end
        end
    end

    # ComponentArray parameters
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta_final[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    edf = Float64(n_beta)

    PSMSolution(params, data_loss, data_loss, edf, Float64[alg.noise_scale],
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=true, iterations=n_iter, method=:ensemble_kalman,
                 ensemble_spread=spread_history,
                 ensemble_std=ensemble_std))
end
