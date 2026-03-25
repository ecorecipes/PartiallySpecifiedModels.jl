"""
    with_range_param(approx, range_param)

Create a copy of an SPDE approximator with a new range parameter.
Non-SPDE approximators are returned unchanged.
"""
with_range_param(a::AbstractApproximator, ::Float64) = a

function with_range_param(a::SPDEApproximator, range_param::Float64)
    SPDEApproximator(a.name, a.domain, a.n_basis;
        nu=a.nu, range_param=range_param, initial=a.initial_func)
end

function with_range_param(a::ShapeConstrainedSPDEApproximator, range_param::Float64)
    ShapeConstrainedSPDEApproximator(a.name, a.domain, a.n_basis, a.constraint;
        nu=a.nu, range_param=range_param, initial=a.initial_func)
end

"""
    optimize_spde_range(prob, alg; kwargs...) -> NamedTuple

Optimize the SPDE range parameter via profile GCV.

Runs the solver for each candidate range parameter on a log-scale grid,
and returns the solution with the best GCV score. All SPDE approximators
in the problem share the same range multiplier.

# Keyword Arguments
- `range_multipliers::AbstractVector`: explicit multipliers of the default
  range to try (overrides `n_grid`).
- `n_grid::Int=10`: number of log-spaced grid points from 0.1× to 10×
  the default range.
- `verbose::Bool=false`: print progress for each grid point.

# Returns
A `NamedTuple` with fields:
- `solution::PSMSolution` — best solution found
- `range_param::Float64` — optimal range parameter (of the first SPDE approximator)
- `gcv_scores::Vector{Float64}` — GCV score at each grid point (`Inf` if failed)
- `range_values::Vector{Float64}` — range parameter values tried
"""
function optimize_spde_range(prob::PSMProblem, alg;
    range_multipliers::Union{Nothing, AbstractVector}=nothing,
    n_grid::Int=10,
    verbose::Bool=false)

    # Identify SPDE approximators and their default ranges
    spde_indices = Int[]
    default_ranges = Float64[]
    for (i, a) in enumerate(prob.approximators)
        if a isa SPDEApproximator || a isa ShapeConstrainedSPDEApproximator
            push!(spde_indices, i)
            push!(default_ranges, a.range_param)
        end
    end
    isempty(spde_indices) && error("No SPDE approximators found in problem")

    # Build grid of multipliers
    mults = if range_multipliers !== nothing
        collect(Float64, range_multipliers)
    else
        exp.(range(log(0.1), log(10.0), length=n_grid))
    end

    n_data = length(prob.data_times) * length(prob.obs_to_state)

    best_gcv = Inf
    best_sol = nothing
    best_range = 0.0
    gcv_scores = Float64[]
    range_values = Float64[]

    for mult in mults
        # Rebuild approximators with scaled range
        new_approxs = AbstractApproximator[
            if i in spde_indices
                idx = findfirst(==(i), spde_indices)
                with_range_param(prob.approximators[i], default_ranges[idx] * mult)
            else
                prob.approximators[i]
            end
            for i in eachindex(prob.approximators)
        ]

        range_val = default_ranges[1] * mult
        push!(range_values, range_val)

        # Rebuild problem with new approximators
        new_prob = PSMProblem(prob.dynamics!, prob.u0, prob.tspan, new_approxs;
            data_times=prob.data_times,
            data_values=prob.data_values,
            data_weights=prob.data_weights,
            obs_to_state=prob.obs_to_state,
            known_params=prob.known_params,
            likelihood=prob.likelihood,
            solver=prob.ode_solver,
            discrete=prob.discrete,
            delays=prob.delays,
            history=prob.history)

        # Run solver
        sol = try
            solve(new_prob, alg)
        catch e
            verbose && @printf("  range=%6.3f (×%.2f): FAILED (%s)\n",
                range_val, mult, sprint(showerror, e))
            push!(gcv_scores, Inf)
            continue
        end

        # GCV score: n * data_loss / (n - edf)²
        denom = max(n_data - sol.edf, 1.0)
        gcv = n_data * sol.data_loss / denom^2
        push!(gcv_scores, gcv)

        if verbose
            @printf("  range=%6.3f (×%.2f): GCV=%.4f, loss=%.4f, edf=%.1f\n",
                range_val, mult, gcv, sol.data_loss, sol.edf)
        end

        if gcv < best_gcv
            best_gcv = gcv
            best_sol = sol
            best_range = range_val
        end
    end

    best_sol === nothing && error("All range parameter values failed")

    if verbose
        @printf("Best range=%.3f, GCV=%.4f\n", best_range, best_gcv)
    end

    (solution=best_sol, range_param=best_range,
     gcv_scores=gcv_scores, range_values=range_values)
end
