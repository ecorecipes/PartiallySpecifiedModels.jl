# Bootstrap confidence intervals for partially specified models
#
# Implements residual-based bootstrap following Wood (2001, 2006) and the
# ddefit504 reference implementation.  Supports parametric (Gaussian residuals),
# nonparametric (resampled residuals), and case (resampled observations) bootstrap.
#
# The key function is `bootstrap(sol, prob, alg; nboot, method, ...)` which
# returns a BootstrapResult with pointwise CIs on fitted values and unknown
# function evaluations.

using Random: default_rng, shuffle!
using Statistics: quantile, mean, std

# ─── Bootstrap result type ────────────────────────────────────────

"""
    BootstrapResult

Result of bootstrap resampling for a PSM solution.

# Fields
- `coefs::Matrix{Float64}`: B × p matrix of bootstrap coefficient vectors
- `fitted_values::Array{Float64,3}`: n_times × n_obs × B array of fitted trajectories
- `uf_values::Dict{Symbol, Matrix{Float64}}`: unknown function evaluations on a grid,
   each n_grid × B matrix
- `uf_grid::Dict{Symbol, Vector{Float64}}`: evaluation grid for each UF
- `ci_fitted::NamedTuple`: `(lower, upper)` matrices (n_times × n_obs) at given level
- `ci_uf::Dict{Symbol, NamedTuple}`: `(lower, upper)` vectors for each UF
- `level::Float64`: confidence level (e.g., 0.95)
- `n_success::Int`: number of successful bootstrap replicates (out of B attempted)
"""
struct BootstrapResult
    coefs::Matrix{Float64}
    fitted_values::Array{Float64, 3}
    uf_values::Dict{Symbol, Matrix{Float64}}
    uf_grid::Dict{Symbol, Vector{Float64}}
    ci_fitted::NamedTuple{(:lower, :upper), Tuple{Matrix{Float64}, Matrix{Float64}}}
    ci_uf::Dict{Symbol, NamedTuple{(:lower, :upper), Tuple{Vector{Float64}, Vector{Float64}}}}
    level::Float64
    n_success::Int
end

# ─── Bootstrap methods ────────────────────────────────────────────

"""
    bootstrap(sol, prob, alg; nboot=200, method=:parametric, level=0.95,
              uf_ngrid=100, rng=default_rng(), verbose=false)

Compute bootstrap confidence intervals for a PSM solution.

# Arguments
- `sol::PSMSolution`: the original fitted solution
- `prob::PSMProblem`: the problem definition
- `alg`: the solver algorithm (e.g., `LAML(...)`)
- `nboot::Int=200`: number of bootstrap replicates
- `method::Symbol=:parametric`: bootstrap method
  - `:parametric` — resample from N(0, σ̂) per observed state
  - `:nonparametric` — resample residuals with replacement per state
  - `:case` — resample entire observations (rows) with replacement
- `level::Float64=0.95`: confidence level for CIs
- `uf_ngrid::Int=100`: number of grid points for unknown function CIs
- `rng`: random number generator
- `verbose::Bool=false`: print progress

# Returns
A `BootstrapResult` with coefficient samples, fitted value CIs, and
unknown function CIs.

# Example
```julia
sol = solve(prob, LAML(maxiters=80))
bs = bootstrap(sol, prob, LAML(maxiters=80); nboot=200, verbose=true)

# Plot fitted values with 95% CI ribbon
plot(sol.data_times, sol.fitted_values[:, 1], lw=2)
plot!(sol.data_times, bs.ci_fitted.lower[:, 1], fillrange=bs.ci_fitted.upper[:, 1],
      alpha=0.2, label="95% CI")

# Plot unknown function with CI
plot(bs.uf_grid[:λ], bs.ci_uf[:λ].lower, fillrange=bs.ci_uf[:λ].upper,
     alpha=0.2, label="95% CI")
```
"""
function bootstrap(sol::PSMSolution, prob::PSMProblem, alg;
                   nboot::Int=200,
                   method::Symbol=:parametric,
                   level::Float64=0.95,
                   uf_ngrid::Int=100,
                   rng=default_rng(),
                   verbose::Bool=false)

    method in (:parametric, :nonparametric, :case) ||
        error("bootstrap: method must be :parametric, :nonparametric, or :case")
    0.0 < level < 1.0 || error("bootstrap: level must be in (0, 1)")

    n_times = length(sol.data_times)
    n_obs = size(sol.data_values, 2)
    fitted = sol.fitted_values  # n_times × n_obs
    resid = sol.data_values .- fitted

    # Compute residual statistics per observed state
    σ_hat = Float64[std(resid[:, j]; corrected=true) for j in 1:n_obs]

    # Build UF evaluation grids
    uf_grids = Dict{Symbol, Vector{Float64}}()
    for approx in prob.approximators
        lo, hi = approx.domain
        uf_grids[approx.name] = collect(range(lo, hi, length=uf_ngrid))
    end

    # Storage for bootstrap samples
    n_p = length(sol.parameters)
    coef_samples = zeros(nboot, n_p)
    fitted_samples = zeros(n_times, n_obs, nboot)
    uf_samples = Dict{Symbol, Matrix{Float64}}(
        name => zeros(uf_ngrid, nboot) for name in keys(uf_grids))

    n_success = 0

    for b in 1:nboot
        if verbose && (b <= 3 || b % 50 == 0 || b == nboot)
            println("Bootstrap replicate $b / $nboot")
        end

        # Generate pseudo-data
        y_boot = _resample_data(method, fitted, resid, σ_hat, n_times, n_obs, rng)

        # Build new problem with pseudo-data
        prob_boot = PSMProblem(prob.dynamics!, prob.u0, prob.tspan,
            prob.approximators;
            data_times=prob.data_times,
            data_values=y_boot,
            data_weights=prob.data_weights,
            obs_to_state=prob.obs_to_state,
            known_params=prob.known_params,
            likelihood=prob.likelihood,
            solver=prob.ode_solver,
            discrete=prob.discrete,
            delays=prob.delays,
            history=prob.history,
            prob.ode_kwargs...)

        # Refit
        sol_boot = try
            solve(prob_boot, alg)
        catch e
            if verbose; println("  Replicate $b failed: $e"); end
            continue
        end

        # Check for valid solution
        if !all(isfinite, sol_boot.fitted_values)
            if verbose; println("  Replicate $b: non-finite fitted values"); end
            continue
        end

        n_success += 1
        coef_samples[n_success, :] .= sol_boot.parameters
        fitted_samples[:, :, n_success] .= sol_boot.fitted_values

        # Evaluate unknown functions on grids
        for (name, grid) in uf_grids
            if haskey(sol_boot.unknown_functions, name)
                f = sol_boot.unknown_functions[name]
                for (k, x) in enumerate(grid)
                    val = try; f(x); catch; NaN; end
                    uf_samples[name][k, n_success] = val
                end
            end
        end
    end

    if n_success < 3
        error("bootstrap: only $n_success / $nboot replicates succeeded. " *
              "Check model stability or increase nboot.")
    end

    if verbose
        println("Bootstrap complete: $n_success / $nboot successful")
    end

    # Trim to successful replicates
    coef_samples = coef_samples[1:n_success, :]
    fitted_samples = fitted_samples[:, :, 1:n_success]
    for name in keys(uf_samples)
        uf_samples[name] = uf_samples[name][:, 1:n_success]
    end

    # Compute quantiles
    α_lo = (1 - level) / 2
    α_hi = 1 - α_lo

    ci_lower = zeros(n_times, n_obs)
    ci_upper = zeros(n_times, n_obs)
    for j in 1:n_obs, i in 1:n_times
        vals = fitted_samples[i, j, :]
        ci_lower[i, j] = quantile(vals, α_lo)
        ci_upper[i, j] = quantile(vals, α_hi)
    end

    ci_uf = Dict{Symbol, NamedTuple{(:lower, :upper), Tuple{Vector{Float64}, Vector{Float64}}}}()
    for (name, mat) in uf_samples
        lo = [quantile(mat[k, :], α_lo) for k in 1:uf_ngrid]
        hi = [quantile(mat[k, :], α_hi) for k in 1:uf_ngrid]
        ci_uf[name] = (lower=lo, upper=hi)
    end

    BootstrapResult(
        coef_samples, fitted_samples,
        uf_samples, uf_grids,
        (lower=ci_lower, upper=ci_upper), ci_uf,
        level, n_success)
end

# ─── Resampling methods ──────────────────────────────────────────

function _resample_data(method::Symbol, fitted::Matrix{Float64},
                        resid::Matrix{Float64}, σ_hat::Vector{Float64},
                        n_times::Int, n_obs::Int, rng)
    y_boot = similar(fitted)

    if method == :parametric
        # Parametric: fitted + N(0, σ̂_j) per state j
        for j in 1:n_obs
            for i in 1:n_times
                y_boot[i, j] = fitted[i, j] + σ_hat[j] * randn(rng)
            end
        end
    elseif method == :nonparametric
        # Nonparametric: fitted + resampled residuals (per state)
        for j in 1:n_obs
            idx = rand(rng, 1:n_times, n_times)
            for i in 1:n_times
                y_boot[i, j] = fitted[i, j] + resid[idx[i], j]
            end
        end
    elseif method == :case
        # Case resampling: resample entire rows
        idx = rand(rng, 1:n_times, n_times)
        for j in 1:n_obs
            for i in 1:n_times
                y_boot[i, j] = (fitted .+ resid)[idx[i], j]
            end
        end
    end

    y_boot
end
