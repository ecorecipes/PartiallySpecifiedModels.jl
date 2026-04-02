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
  - `:parametric` — sample from the fitted distribution (Gaussian, Poisson,
    NegBin, TruncatedNormal — uses the problem's likelihood family)
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
        y_boot = _resample_data(method, prob.likelihood, fitted, resid, σ_hat,
                                n_times, n_obs, rng)

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

"""
    _resample_data(method, family, fitted, resid, σ_hat, n_times, n_obs, rng)

Generate bootstrap pseudo-data.

For `:parametric`, the sampling distribution depends on the likelihood family:
- `Gaussian()`:  y* ~ N(μ̂, σ̂)
- `Poisson()`:   y* ~ Poisson(μ̂)
- `NegativeBinomial(θ)`: y* ~ NegBin(μ̂, θ)  (Gamma-Poisson mixture)
- `TruncatedNormal(lower, σ)`: y* ~ TruncNorm(μ̂, σ, lower)
- Other: falls back to Gaussian residuals

For `:nonparametric`, residuals are resampled with replacement per state.
For `:case`, entire observation rows are resampled.
"""
function _resample_data(method::Symbol, family::AbstractLikelihood,
                        fitted::Matrix{Float64},
                        resid::Matrix{Float64}, σ_hat::Vector{Float64},
                        n_times::Int, n_obs::Int, rng)
    y_boot = similar(fitted)

    if method == :parametric
        _parametric_resample!(y_boot, family, fitted, σ_hat, n_times, n_obs, rng)
    elseif method == :nonparametric
        for j in 1:n_obs
            idx = rand(rng, 1:n_times, n_times)
            for i in 1:n_times
                y_boot[i, j] = fitted[i, j] + resid[idx[i], j]
            end
        end
    elseif method == :case
        idx = rand(rng, 1:n_times, n_times)
        for j in 1:n_obs, i in 1:n_times
            y_boot[i, j] = resid[idx[i], j] + fitted[idx[i], j]
        end
    end

    y_boot
end

# ─── Parametric samplers per likelihood family ────────────────────

function _parametric_resample!(y::Matrix, ::Gaussian, fitted::Matrix,
                               σ_hat::Vector, n_t::Int, n_obs::Int, rng)
    for j in 1:n_obs, i in 1:n_t
        y[i, j] = fitted[i, j] + σ_hat[j] * randn(rng)
    end
end

function _parametric_resample!(y::Matrix, ::Poisson, fitted::Matrix,
                               σ_hat::Vector, n_t::Int, n_obs::Int, rng)
    for j in 1:n_obs, i in 1:n_t
        μ = max(fitted[i, j], 1e-10)
        y[i, j] = Float64(_sample_poisson(μ, rng))
    end
end

function _parametric_resample!(y::Matrix, fam::NegativeBinomial, fitted::Matrix,
                               σ_hat::Vector, n_t::Int, n_obs::Int, rng)
    θ = fam.theta
    for j in 1:n_obs, i in 1:n_t
        μ = max(fitted[i, j], 1e-10)
        # Gamma-Poisson mixture: G ~ Gamma(θ, μ/θ), then Y ~ Poisson(G)
        g = _sample_gamma(θ, μ / θ, rng)
        y[i, j] = Float64(_sample_poisson(g, rng))
    end
end

function _parametric_resample!(y::Matrix, fam::TruncatedNormal, fitted::Matrix,
                               σ_hat::Vector, n_t::Int, n_obs::Int, rng)
    σ = fam.sigma
    lo = fam.lower
    for j in 1:n_obs, i in 1:n_t
        # Rejection sampling from N(μ, σ²) truncated to [lower, ∞)
        μ = fitted[i, j]
        for _ in 1:1000
            z = μ + σ * randn(rng)
            if z >= lo
                y[i, j] = z
                @goto next_tn
            end
        end
        y[i, j] = max(μ, lo)  # fallback
        @label next_tn
    end
end

# Fallback: use Gaussian residuals for unknown likelihood families
function _parametric_resample!(y::Matrix, ::AbstractLikelihood, fitted::Matrix,
                               σ_hat::Vector, n_t::Int, n_obs::Int, rng)
    for j in 1:n_obs, i in 1:n_t
        y[i, j] = fitted[i, j] + σ_hat[j] * randn(rng)
    end
end

# ─── Distribution samplers (no Distributions.jl dependency) ───────

"""Sample from Poisson(μ) using inverse CDF (Knuth) for μ ≤ 30, normal approx for μ > 30."""
function _sample_poisson(μ::Real, rng)
    if μ <= 30.0
        L = exp(-μ)
        k = 0; p = 1.0
        while true
            k += 1
            p *= rand(rng)
            p <= L && return k - 1
        end
    else
        # Normal approximation for large μ
        max(0, round(Int, μ + sqrt(μ) * randn(rng)))
    end
end

"""Sample from Gamma(shape, scale) using Marsaglia & Tsang (2000)."""
function _sample_gamma(shape::Real, scale::Real, rng)
    if shape < 1.0
        # Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
        return _sample_gamma(shape + 1.0, scale, rng) * rand(rng)^(1.0 / shape)
    end
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn(rng)
        v = (1.0 + c * x)^3
        v <= 0.0 && continue
        u = rand(rng)
        if u < 1.0 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
            return d * v * scale
        end
    end
end
