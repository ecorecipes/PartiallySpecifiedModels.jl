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

`sol.convergence.converged` is honest: it is `true` only when the
ensemble actually collapsed (mean particle std fell to 1e-3 of its
initial value, `reason = :ensemble_collapse`). Exhausting
`n_iterations` reports `converged = false, reason = :maxiters`, and a
failed forward solve at the ensemble mean — which makes the fitted
values `NaN` and `data_loss` `Inf` — reports
`reason = :final_solve_failed`.

Note the discrete-time convention: discrete problems are propagated with
`simulate_discrete` (unit steps over `tspan`, data times snapped to the
nearest integer step), the same trajectory every other solver sees.

# Masked data

This solver does NOT support masked observations and raises an error if
any data cell is masked (`data_weights == 0` or a non-finite
`data_values` entry). Its likelihood is evaluated inside a Kalman /
particle recursion with no per-cell mask: honouring a mask means skipping
the FILTER UPDATE, not merely the density term, for the masked cells.
Left unguarded the masked cells corrupt the filter state while the run
still looks like an ordinary converged fit, so it fails loudly instead.
Drop the masked rows from `data_times`/`data_values`, or use one of the
masking-capable solvers (`LAML`, `GCVSolver`, `CollocationLAML`,
`GradientMatching`, `TwoStageSolver`, `BNGSolver`, `ODINSolver`,
`RKHSSolver`, `IntegralMatchingSolver`, `AdamSolver`,
`MultipleShootingSolver`, `DerivativeFreeSolver`, `MCMCSolver`,
`MagiSolver`, `VariationalSolver`, `ABCSolver`,
`ProfileLikelihoodSolver`).
"""
function SciMLBase.solve(prob::PSMProblem, alg::EnsembleKalmanSolver)
    _validate_problem(prob, "EnsembleKalmanSolver")
    _reject_masked_data(prob, "EnsembleKalmanSolver")
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
        pred = zeros(length(prob.data_times), size(prob.data_values, 2))

        try
            if prob.discrete
                # Use the package-canonical discrete simulation (unit steps
                # over tspan, data times snapped to the nearest integer step)
                # so EKI sees the same trajectory as every other solver even
                # when data times have gaps.
                pred = simulate_discrete(prob, theta)
            else
                p = build_param_struct(prob, theta)
                ode_u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
                ode_prob = ODEProblem(prob.dynamics!, ode_u0, prob.tspan, p)
                solver = prob.ode_solver === nothing ? Tsit5() : prob.ode_solver
                ode_sol = OrdinaryDiffEq.solve(ode_prob, solver;
                            saveat=prob.data_times, prob.ode_kwargs...)
                if ode_sol.retcode != :Success && ode_sol.retcode != SciMLBase.ReturnCode.Success
                    return nothing
                end
                for i in 1:length(prob.data_times)
                    for j in 1:size(prob.data_values, 2)
                        sk = prob.obs_to_state[j]
                        pred[i, j] = ode_sol.u[i][sk]
                    end
                end
            end
        catch e
            _is_program_error(e) && rethrow()
            return nothing
        end

        # Diverged particles (e.g. an exploding discrete map) yield Inf/NaN
        # without throwing; treat them as failures like any other.
        all(isfinite, pred) || return nothing
        vec(pred)
    end

    # ── Initialise ensemble ──────────────────────────────────────
    # rng_seed defaults to 42 (historical hard-coded stream); nothing = fresh.
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)
    ensemble = Matrix{Float64}(undef, n_beta, J)
    for j in 1:J
        ensemble[:, j] = beta0 .+ 0.5 .* randn(rng, n_beta) .* max.(abs.(beta0), 0.1)
    end

    # Observation noise covariance
    Γ = σ_obs^2 * Matrix{Float64}(I, n_data, n_data)

    spread_history = Float64[]

    # Honest convergence reporting: EKI has a real stopping test (ensemble
    # collapse); exhausting n_iterations is NOT convergence.
    conv_converged = false
    conv_reason = :maxiters
    iters_used = n_iter

    # ── EKI iterations ───────────────────────────────────────────
    for iter in 1:n_iter
        iters_used = iter
        # Evaluate forward model for each particle; failures return nothing.
        # (A 1e6 sentinel column would inflate the ensemble covariances by
        # ~1e12 and drag the whole ensemble in a single update.)
        Gcols = Vector{Union{Nothing, Vector{Float64}}}(undef, J)
        for j in 1:J
            Gcols[j] = forward_model(ensemble[:, j])
        end
        valid = findall(!isnothing, Gcols)
        length(valid) >= 3 ||
            error("EnsembleKalmanSolver: only $(length(valid)) of $J " *
                  "particles produced a successful forward solve at " *
                  "iteration $iter; the problem or the initial ensemble " *
                  "spread is likely ill-posed")
        G = reduce(hcat, (Gcols[j] for j in valid))
        θv = ensemble[:, valid]

        # Ensemble statistics over the valid particles only
        θ_mean = vec(mean(θv, dims=2))
        G_mean = vec(mean(G, dims=2))
        Δθ = θv .- θ_mean
        ΔG = G .- G_mean
        Jv = length(valid)
        Cθg = (Δθ * ΔG') / max(Jv - 1, 1)
        Cgg = (ΔG * ΔG') / max(Jv - 1, 1)

        # Kalman gain: K = Cθg * (Cgg + Γ)⁻¹
        # Use pseudo-inverse for numerical stability
        K = Cθg * pinv(Cgg + Γ)

        # Update valid particles with the perturbed-observation EKI step
        for (k, j) in enumerate(valid)
            ξ = σ_obs .* randn(rng, n_data)
            innovation = y_obs .+ ξ .- G[:, k]
            ensemble[:, j] .+= K * innovation
        end
        # Resample failed particles from the updated valid ensemble with a
        # small jitter, pulling them back into the feasible region.
        θ_std = vec(std(ensemble[:, valid], dims=2)) .+ 1e-8
        for j in 1:J
            j in valid && continue
            src = valid[rand(rng, 1:Jv)]
            ensemble[:, j] .= ensemble[:, src] .+ 0.1 .* θ_std .* randn(rng, n_beta)
        end

        # Track ensemble spread
        spread = mean(std(ensemble, dims=2))
        push!(spread_history, spread)

        if verbose && (iter <= 3 || iter % 5 == 0 || iter == n_iter)
            misfit = mean(abs2, G_mean .- y_obs)
            println("  iter $iter: misfit=$(round(misfit, sigdigits=4)) " *
                    "spread=$(round(spread, sigdigits=4))")
        end

        # Stopping test: ensemble collapse. EKI contracts the ensemble
        # geometrically; once the mean particle std is a thousandth of its
        # initial value the particles agree and further iterations only
        # shrink an already degenerate ensemble. Without this test the
        # solver had NO stopping criterion at all, yet reported
        # converged=true unconditionally.
        if spread <= 1e-3 * spread_history[1]
            conv_converged = true
            conv_reason = :ensemble_collapse
            if verbose
                println("  Ensemble collapsed at iter $iter " *
                        "(spread=$(round(spread, sigdigits=3)))")
            end
            break
        end
    end

    # ── Build solution from ensemble mean ────────────────────────
    beta_final = vec(mean(ensemble, dims=2))
    ensemble_std = vec(std(ensemble, dims=2))

    # Simulate at ensemble mean for fitted values (the mean of a converged
    # ensemble should solve; if not, report NaN fitted values rather than
    # fabricating numbers)
    pred_vec = forward_model(beta_final)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)
    pred = pred_vec === nothing ? fill(NaN, n_times, n_obs) :
                                  reshape(pred_vec, n_times, n_obs)

    # NaN fitted values and an infinite loss are a failed fit, not a
    # converged one — override whatever the iteration reported.
    if pred_vec === nothing
        conv_converged = false
        conv_reason = :final_solve_failed
    end

    # Weighted sum of squares over usable cells, matching the data_loss
    # convention of the other solvers (masked / NaN cells are skipped so
    # one missing observation does not turn the reported loss into NaN).
    data_loss = pred_vec === nothing ? Inf : weighted_data_loss(prob, pred)

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
            uf_evals[approx.name] = build_neural_evaluator(approx, params_k)
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
                (converged=conv_converged, iterations=iters_used,
                 reason=conv_reason, method=:ensemble_kalman,
                 ensemble_spread=spread_history,
                 ensemble_std=ensemble_std))
end
