# derivative_free_solver.jl — Derivative-free optimization for PSMs
#
# Uses gradient-free optimizers (Nelder-Mead, Particle Swarm) with a
# simulation-based loss function. No autodiff through the ODE is needed,
# making this a robust fallback for stiff/discontinuous models.

"""
    SciMLBase.solve(prob::PSMProblem, alg::DerivativeFreeSolver)

Fit a partially specified model using derivative-free optimization.

# Algorithm
1. Define a loss function that simulates the model and computes weighted
   SSE (`:mse`) or the negative log-likelihood of `prob.likelihood`
   (`:likelihood`). The default `loss=:auto` picks `:mse` for Gaussian
   data and `:likelihood` for any other family, so a declared
   non-Gaussian likelihood is honored rather than silently ignored.
2. Add the smoothing penalty `0.5 · penalty_weight · Σ_j β_j' S_j β_j`
   (default `penalty_weight=1.0`; set 0 for an unpenalized fit).
3. Use `Optim.NelderMead()` or `Optim.ParticleSwarm()` to minimize the loss.
4. Return a `PSMSolution` with fitted parameters.

Simulation failures (e.g. ODE divergence) are handled gracefully by returning
a large loss value, guiding the optimizer away from bad regions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::DerivativeFreeSolver)
    _validate_problem(prob, "DerivativeFreeSolver")
    verbose = alg.verbose

    # Resolve the loss: :auto follows prob.likelihood (Gaussian → :mse,
    # anything else → :likelihood); explicit :mse/:likelihood is honored.
    loss_sym = if alg.loss == :auto
        prob.likelihood isa Gaussian ? :mse : :likelihood
    elseif alg.loss in (:mse, :likelihood)
        alg.loss
    else
        error("DerivativeFreeSolver: unknown loss :$(alg.loss). " *
              "Supported: :auto, :mse, :likelihood.")
    end

    # ── Initialize parameters ──
    beta0 = build_initial_params(prob)
    n_p = length(beta0)

    n_times = length(prob.data_times)
    n_obs   = length(prob.obs_to_state)
    n_data  = n_times * n_obs

    # Flatten data and weights into vectors (obs-major: obs 1 times, obs 2 times, …),
    # enforcing the package masking convention: a cell is usable only if its
    # weight is positive AND its datum is non-NaN. Masked cells get weight 0
    # and a finite placeholder value. Masking HERE (rather than in the two
    # loss branches below) fixes both at once: the :mse branch forms
    # `w_vec[k] * (y_vec[k] - pred)^2` and the :likelihood branch calls
    # `log_likelihood`, and IEEE `0 * NaN = NaN` would poison either one.
    # A poisoned loss is not merely a NaN report here: `!isfinite(data_loss)
    # && return 1e20` below turns it into a CONSTANT objective, so Nelder-Mead
    # / particle swarm see a flat surface, never move, and the solver returns
    # its initialization while reporting ordinary-looking convergence.
    # n_usable is the sample size for any per-observation denominator.
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y = prob.data_values[ti, oi]
        wv = prob.data_weights[ti, oi]
        if _usable(y, wv)
            y_vec[k] = y
            w_vec[k] = wv
        end   # else keep the 0.0 placeholder with weight 0.0
        k += 1
    end
    # The package-wide count (`solver.jl`), not a re-derived one, so
    # this can never drift from the predicate the flatten applied.
    n_usable_cells = n_usable(prob)
    n_usable_cells == 0 && error("DerivativeFreeSolver: every observation is " *
        "masked (every weight is 0 or every value is non-finite); there " *
        "is nothing to fit.")

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
        println("  method=$(alg.method), loss=$loss_sym (from $(alg.loss)), " *
                "maxiters=$(alg.maxiters)")
    end

    # ── Loss function ──
    function loss_fn(beta::AbstractVector)
        # Simulate model
        pred = try
            simulate(prob, beta)
        catch e
            _is_program_error(e) && rethrow()
            return 1e20
        end

        # Check for NaN/Inf in predictions
        any(x -> !isfinite(x), pred) && return 1e20

        # Data loss
        data_loss = if loss_sym == :mse
            s = 0.0
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                s += w_vec[k] * (y_vec[k] - pred[ti, oi])^2
                k += 1
            end
            # Unscaled SS: dividing by n here (but not in :likelihood mode,
            # nor in the penalty) made the effective smoothing strength
            # depend on sample size and flip by a factor n between modes.
            s
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

        data_loss + 0.5 * alg.penalty_weight * pen
    end

    # ── Choose optimizer ──
    if alg.method == :particle_swarm
        # ParticleSwarm needs bounds. Scale them to the initial values —
        # fixed [-10, 10] boxes silently clamped coefficients for data on
        # larger scales, excluding the optimum entirely.
        span = max(10.0, 10.0 * maximum(abs, beta0))
        lower = beta0 .- span
        upper = beta0 .+ span
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
        error("DerivativeFreeSolver: unknown method :$(alg.method). " *
              "Supported: :nelder_mead, :particle_swarm. (:cmaes is not " *
              "implemented; it previously fell back to NelderMead silently.)")
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
    data_loss = weighted_data_loss(prob, pred)

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
        uf_evals[approx.name] = build_evaluator(approx, params_k)
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
