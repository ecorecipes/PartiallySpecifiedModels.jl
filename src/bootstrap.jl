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
    (Gaussian likelihoods only: additive residuals are invalid pseudo-data
    for count or truncated families)

Each replicate is refit from scratch with `alg`, so smoothing parameters are
re-estimated per replicate; the intervals therefore include smoothing-
selection variability (unlike a fixed-λ conditional bootstrap).
- `level::Float64=0.95`: confidence level for CIs
- `uf_ngrid::Int=100`: number of grid points for unknown function CIs
- `rng`: random number generator
- `parallel::Bool=false`: use multi-threading (`Threads.@threads`) for replicates.
  Requires Julia started with `JULIA_NUM_THREADS > 1`.
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
                   parallel::Bool=false,
                   verbose::Bool=false)

    method == :case &&
        error("bootstrap: the :case method has been removed. Resampling " *
              "observation rows onto the original time stamps destroys the " *
              "temporal structure of trajectory data, so its confidence " *
              "intervals had no statistical meaning for dynamical models. " *
              "Use :parametric (any likelihood) or :nonparametric (Gaussian).")
    method in (:parametric, :nonparametric) ||
        error("bootstrap: method must be :parametric or :nonparametric")
    method == :nonparametric && !(prob.likelihood isa Gaussian) &&
        error("bootstrap: :nonparametric residual resampling produces " *
              "invalid pseudo-data for $(typeof(prob.likelihood)) " *
              "(negative/non-integer counts, values below a truncation " *
              "bound). Use method=:parametric, which samples from the " *
              "fitted distribution.")
    0.0 < level < 1.0 || error("bootstrap: level must be in (0, 1)")

    n_times = length(sol.data_times)
    n_obs = size(sol.data_values, 2)
    fitted = sol.fitted_values  # n_times × n_obs
    resid = sol.data_values .- fitted

    # Residual scale per observed state, corrected for the effective model
    # degrees of freedom (a flexible smooth absorbs part of the noise, so a
    # raw n−1 denominator understates σ and narrows parametric CIs). The
    # total edf is allocated evenly across observed states.
    edf_j = sol.edf / n_obs
    σ_hat = Float64[sqrt(sum(abs2, resid[:, j]) /
                         max(n_times - edf_j, 1.0)) for j in 1:n_obs]

    # Build UF evaluation grids (approximators without a domain — e.g. a
    # NeuralApproximator constructed without one — cannot be gridded)
    uf_grids = Dict{Symbol, Vector{Float64}}()
    for approx in prob.approximators
        if approx.domain === nothing
            @warn "bootstrap: approximator :$(approx.name) has no domain; " *
                  "skipping its unknown-function confidence band"
            continue
        end
        lo, hi = approx.domain
        uf_grids[approx.name] = collect(range(lo, hi, length=uf_ngrid))
    end
    uf_names = collect(keys(uf_grids))

    n_p = length(sol.parameters)

    if parallel && Threads.nthreads() > 1
        # ─── Threaded bootstrap ───────────────────────────────────
        # Pre-generate all pseudo-data with independent RNGs per replicate
        # to ensure reproducibility regardless of thread scheduling.
        rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nboot]
        boot_data = [_resample_data(method, prob.likelihood, fitted, resid,
                                    σ_hat, n_times, n_obs, rngs[b])
                     for b in 1:nboot]

        # Per-replicate result storage (avoid races)
        results = Vector{Union{Nothing, NamedTuple}}(nothing, nboot)

        if verbose
            println("Bootstrap: $nboot replicates on $(Threads.nthreads()) threads")
        end

        Threads.@threads for b in 1:nboot
            prob_boot = PSMProblem(prob.dynamics!, prob.u0, prob.tspan,
                prob.approximators;
                data_times=prob.data_times,
                data_values=boot_data[b],
                data_weights=prob.data_weights,
                obs_to_state=prob.obs_to_state,
                known_params=prob.known_params,
                likelihood=prob.likelihood,
                solver=prob.ode_solver,
                discrete=prob.discrete,
                delays=prob.delays,
                history=prob.history,
                prob.ode_kwargs...)

            sol_boot = try
                solve(prob_boot, alg)
            catch
                nothing
            end

            if sol_boot !== nothing && all(isfinite, sol_boot.fitted_values)
                # Evaluate UFs on grid
                uf_vals = Dict{Symbol, Vector{Float64}}()
                for name in uf_names
                    grid = uf_grids[name]
                    if haskey(sol_boot.unknown_functions, name)
                        f = sol_boot.unknown_functions[name]
                        uf_vals[name] = Float64[(try; f(x); catch; NaN; end) for x in grid]
                    end
                end
                results[b] = (coefs=collect(sol_boot.parameters),
                              fitted=copy(sol_boot.fitted_values),
                              uf_vals=uf_vals)
            end
        end

        # Collect successful results
        successful = filter(!isnothing, results)
        n_success = length(successful)

        if n_success < 3
            error("bootstrap: only $n_success / $nboot replicates succeeded. " *
                  "Check model stability or increase nboot.")
        end

        coef_samples = zeros(n_success, n_p)
        fitted_samples = zeros(n_times, n_obs, n_success)
        # NaN marks replicates where a UF was absent or failed to evaluate;
        # the quantile step below skips NaN columns instead of treating
        # them as zeros.
        uf_samples = Dict{Symbol, Matrix{Float64}}(
            name => fill(NaN, uf_ngrid, n_success) for name in uf_names)

        for (k, r) in enumerate(successful)
            coef_samples[k, :] .= r.coefs
            fitted_samples[:, :, k] .= r.fitted
            for name in uf_names
                if haskey(r.uf_vals, name)
                    uf_samples[name][:, k] .= r.uf_vals[name]
                end
            end
        end

        if verbose
            println("Bootstrap complete: $n_success / $nboot successful")
        end
    else
        # ─── Sequential bootstrap ─────────────────────────────────
        coef_samples = zeros(nboot, n_p)
        fitted_samples = zeros(n_times, n_obs, nboot)
        # NaN marks absent/failed UF evaluations (see threaded path)
        uf_samples = Dict{Symbol, Matrix{Float64}}(
            name => fill(NaN, uf_ngrid, nboot) for name in uf_names)
        n_success = 0

        for b in 1:nboot
            if verbose && (b <= 3 || b % 50 == 0 || b == nboot)
                println("Bootstrap replicate $b / $nboot")
            end

            y_boot = _resample_data(method, prob.likelihood, fitted, resid,
                                    σ_hat, n_times, n_obs, rng)

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

            sol_boot = try
                solve(prob_boot, alg)
            catch e
                if verbose; println("  Replicate $b failed: $e"); end
                continue
            end

            if !all(isfinite, sol_boot.fitted_values)
                if verbose; println("  Replicate $b: non-finite fitted values"); end
                continue
            end

            n_success += 1
            coef_samples[n_success, :] .= sol_boot.parameters
            fitted_samples[:, :, n_success] .= sol_boot.fitted_values

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
        for name in uf_names
            uf_samples[name] = uf_samples[name][:, 1:n_success]
        end
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

    # UF bands: skip NaN entries (absent UF in a replicate, or a failed
    # pointwise evaluation) — quantile() throws on NaN, which previously
    # killed an entire completed bootstrap run at this final step.
    ci_uf = Dict{Symbol, NamedTuple{(:lower, :upper), Tuple{Vector{Float64}, Vector{Float64}}}}()
    for (name, mat) in uf_samples
        lo = Vector{Float64}(undef, uf_ngrid)
        hi = Vector{Float64}(undef, uf_ngrid)
        for k in 1:uf_ngrid
            vals = filter(!isnan, view(mat, k, :))
            if isempty(vals)
                lo[k] = NaN; hi[k] = NaN
            else
                lo[k] = quantile(vals, α_lo)
                hi[k] = quantile(vals, α_hi)
            end
        end
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

For `:nonparametric`, residuals are resampled with replacement per state
(valid for Gaussian likelihoods; rejected earlier for other families).
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
