# derivative_free_solver.jl — Derivative-free optimization for PSMs
#
# Uses gradient-free optimizers (Nelder-Mead, Particle Swarm) with a
# simulation-based loss function. No autodiff through the ODE is needed,
# making this a robust fallback for stiff/discontinuous models.

"""
    SciMLBase.solve(prob::PSMProblem, alg::DerivativeFreeSolver)

Fit a partially specified model using derivative-free optimization.

# Algorithm
1. Define a loss function that simulates the model and computes MSE or
   negative log-likelihood against observed data.
2. Optionally add smoothing penalties: `Σ λ_j β_j' S_j β_j`.
3. Use `Optim.NelderMead()` or `Optim.ParticleSwarm()` to minimize the loss.
4. Return a `PSMSolution` with fitted parameters.

Simulation failures (e.g. ODE divergence) are handled gracefully by returning
a large loss value, guiding the optimizer away from bad regions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::DerivativeFreeSolver)
    _validate_problem(prob, "DerivativeFreeSolver")
    verbose = alg.verbose

    # ── Initialize parameters ──
    beta0 = build_initial_params(prob)
    n_p = length(beta0)

    n_times = length(prob.data_times)
    n_obs   = length(prob.obs_to_state)
    n_data  = n_times * n_obs

    # Flatten data and weights into vectors (obs-major: obs 1 times, obs 2 times, …)
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y_vec[k] = prob.data_values[ti, oi]
        w_vec[k] = prob.data_weights[ti, oi]
        k += 1
    end

    # Build penalty matrices for optional regularization
    S_list   = Matrix{Float64}[]
    offsets  = Int[]
    nk_list  = Int[]
    offset   = 0
    for approx in prob.approximators
        np = nparams(approx)
        S = penalty_matrix(approx)
        if S !== nothing
            push!(S_list, S)
            push!(offsets, offset)
            push!(nk_list, np)
        end
        offset += np
    end

    if verbose
        println("DerivativeFreeSolver: $n_p params, $n_data data points")
        println("  method=$(alg.method), loss=$(alg.loss), maxiters=$(alg.maxiters)")
    end

    # ── Loss function ──
    function loss_fn(beta::AbstractVector)
        # Simulate model
        pred = try
            simulate(prob, beta)
        catch
            return 1e20
        end

        # Check for NaN/Inf in predictions
        any(x -> !isfinite(x), pred) && return 1e20

        # Data loss
        data_loss = if alg.loss == :mse
            s = 0.0
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                s += w_vec[k] * (y_vec[k] - pred[ti, oi])^2
                k += 1
            end
            s / n_data
        else  # :likelihood — negative log-likelihood
            f_vec = zeros(n_data)
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                f_vec[k] = pred[ti, oi]
                k += 1
            end
            -log_likelihood(prob.likelihood, y_vec, f_vec, w_vec)
        end

        !isfinite(data_loss) && return 1e20

        # Optional smoothing penalty: Σ_j λ_j β_j' S_j β_j
        pen = 0.0
        for l in eachindex(S_list)
            off = offsets[l]
            nk  = nk_list[l]
            bk  = @view beta[off+1:off+nk]
            pen += dot(bk, S_list[l] * bk)
        end

        data_loss + 0.5 * pen
    end

    # ── Choose optimizer ──
    if alg.method == :particle_swarm
        # ParticleSwarm needs bounds
        lower = fill(-10.0, n_p)
        upper = fill( 10.0, n_p)
        # Widen bounds if initial params fall outside [-10, 10]
        for i in 1:n_p
            lower[i] = min(lower[i], beta0[i] - 5.0)
            upper[i] = max(upper[i], beta0[i] + 5.0)
        end
        optimizer = Optim.ParticleSwarm(; lower=lower, upper=upper,
                                          n_particles=alg.n_particles)
        result = Optim.optimize(loss_fn, beta0, optimizer,
                                Optim.Options(iterations=alg.maxiters,
                                              show_trace=verbose))
    elseif alg.method == :nelder_mead
        optimizer = Optim.NelderMead()
        result = Optim.optimize(loss_fn, beta0, optimizer,
                                Optim.Options(iterations=alg.maxiters,
                                              show_trace=verbose))
    else
        # Fallback: Nelder-Mead
        optimizer = Optim.NelderMead()
        result = Optim.optimize(loss_fn, beta0, optimizer,
                                Optim.Options(iterations=alg.maxiters,
                                              show_trace=verbose))
    end

    beta_opt = Optim.minimizer(result)
    obj_val  = Optim.minimum(result)

    if verbose
        println("  Optimization complete: f=$(round(obj_val, sigdigits=6)), " *
                "converged=$(Optim.converged(result)), " *
                "iters=$(Optim.iterations(result))")
    end

    # ── Simulate with optimal parameters for fitted values ──
    pred = simulate(prob, beta_opt)

    # Data loss (weighted sum of squares)
    data_loss = 0.0
    for j in 1:n_obs, i in 1:n_times
        data_loss += prob.data_weights[i, j] *
                     (prob.data_values[i, j] - pred[i, j])^2
    end

    # ── Build ComponentArray of fitted parameters ──
    uf_syms = Symbol[a.name for a in prob.approximators]
    uf_vals = Vector{Float64}[]
    offset  = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(uf_vals, Float64.(beta_opt[offset+1:offset+np]))
        offset += np
    end
    params = ComponentArray(NamedTuple{Tuple(uf_syms)}(Tuple(uf_vals)))

    # ── Build unknown function evaluators ──
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_opt[offset+1:offset+np]
        offset += np
        if approx isa BSplineApproximator
            knots_x = collect(range(approx.domain[1], approx.domain[2],
                                    length=approx.nknots))
            uf_evals[approx.name] = build_bspline_evaluator(knots_x, params_k)
        elseif approx isa NeuralApproximator
            rng = approx.rng_seed !== nothing ?
                  Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            _, st = Lux.setup(rng, approx.model)
            rng2 = approx.rng_seed !== nothing ?
                   Random.Xoshiro(approx.rng_seed) : Random.default_rng()
            ps_ca = Float64.(ComponentArray(
                Lux.initialparameters(rng2, approx.model)))
            ps_final = similar(ps_ca)
            ps_final .= params_k
            lo   = approx.domain === nothing ? nothing : approx.domain[1]
            span = approx.domain === nothing ? nothing :
                   (approx.domain[2] - approx.domain[1])
            uf_evals[approx.name] = x -> begin
                xn = if lo !== nothing && span !== nothing && span > 0
                    (Float64(x isa AbstractArray ? x[1] : x) - lo) / span
                else
                    Float64(x isa AbstractArray ? x[1] : x)
                end
                out, _ = Lux.apply(approx.model,
                                   Float32.(reshape([xn], :, 1)),
                                   ps_final, st)
                length(out) == 1 ? Float64(out[1]) : Float64.(out)
            end
        elseif approx isa GPApproximator
            uf_evals[approx.name] = build_gp_evaluator(approx, params_k)
        elseif approx isa ShapeConstrainedBSplineApproximator
            uf_evals[approx.name] =
                build_constrained_bspline_evaluator(approx, params_k)
        elseif approx isa COMONetApproximator
            uf_evals[approx.name] = build_comonet_evaluator(approx, params_k)
        elseif approx isa SPDEApproximator
            uf_evals[approx.name] = build_spde_evaluator(approx.mesh_points, params_k)
        elseif approx isa ShapeConstrainedSPDEApproximator
            uf_evals[approx.name] = build_constrained_spde_evaluator(approx, params_k)
        end
    end

    if verbose
        println("  data_loss=$(round(data_loss, sigdigits=6)), " *
                "n_params=$n_p")
    end

    # EDF = n_p (all parameters are "free"; no penalized smoothing selection)
    edf = Float64(n_p)

    PSMSolution(params, obj_val, data_loss, edf,
                Float64[],   # no smoothing parameters selected
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (converged=Optim.converged(result),
                 method=alg.method,
                 iterations=Optim.iterations(result),
                 f_calls=Optim.f_calls(result),
                 minimum=Optim.minimum(result)))
end
