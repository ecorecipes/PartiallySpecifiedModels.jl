# Bootstrap confidence intervals for partially specified models
#
# Implements residual-based bootstrap following Wood (2001, 2006) and the
# ddefit504 reference implementation.  Supports parametric (Gaussian residuals),
# Methods: :parametric (per-likelihood samplers) and :nonparametric
# (Gaussian residual resampling).
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

`prob.data_weights` are honoured by BOTH methods. They are precision weights
(`Var(yᵢ) = σ²/wᵢ`, the convention used throughout the package), so each
sampler narrows the cell it draws by `wᵢ`: the Gaussian and fallback samplers
draw at `σ̂/√w`, `TruncatedNormal` at `σ/√w`, and the count families read `w`
as a replicate count (`Poisson(wμ)/w`, `NegBin(wμ, wθ)/w`, both with variance
`V(μ)/w`), which can make pseudo-data non-integer when `w ≠ 1`.
`:nonparametric` resamples STANDARDIZED residuals `√wᵢ·rᵢ` — the raw
residuals are not exchangeable under unequal weights — and rescales each draw
by `1/√wᵢ`. With uniform weights all of this reduces to the unweighted
sampler exactly.

For every family EXCEPT `TruncatedNormal` this gives a `w = 4` cell exactly
half the spread of a `w = 1` cell, i.e. `Var = V(μ)/w`. `TruncatedNormal` does
NOT satisfy that, deliberately: scaling `σ` also moves
`ξ = (μ − a)/σ_eff`, so the truncation bites LESS at high `w`, and the draw's
sd is `σ_eff·√(1 − ξλ(ξ) − λ(ξ)²)`, not `σ_eff`. At `μ = 0.30, σ = 0.20,
a = 0` (measured): `w = 1` gives `ξ = 1.5`, `λ = 0.13878975`, shape
`0.87894981`, sd `0.17578996`; `w = 4` gives `ξ = 3.0`, `λ = 0.00443784`,
shape `0.99331102`, sd `0.09933110` — a ratio of `0.56505560`, and 13.0%
WIDER than `Var = V(μ)/w` would require. The response mean moves too, from
`μ + σ_eff·λ(ξ)` = `0.32775795` toward the latent `μ = 0.30`
(`0.30044378` at `w = 4`). Reweighting a truncated normal cannot preserve
both its mean and `V(μ)/w`; this sampler holds the LATENT location fixed and
scales the latent σ, which is the only reading under which the family's own
`σ` still means what `TruncatedNormal(lower, σ)` says it means. The count
families take the other branch (replicate count) because there `V(μ)/w`
is reachable inside the family.

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

    # Cells that actually informed the fit: positive weight AND non-NaN
    # datum.  The rest of the package (LAML, MCMC, profile likelihood)
    # masks data this way; a single NaN cell previously propagated into
    # σ̂ (making it NaN), which made every pseudo-dataset all-NaN — each
    # replicate then "fit" pure-NaN data without moving from the initial
    # coefficients and all CIs silently collapsed to zero width.
    valid = BitMatrix(undef, n_times, n_obs)
    for j in 1:n_obs, i in 1:n_times
        valid[i, j] = _usable(sol.data_values[i, j], prob.data_weights[i, j])
    end

    # Residual scale per observed state, over usable cells only and with
    # the data weights applied (w is a precision weight: Var(yᵢ) = σ²/wᵢ),
    # corrected for the effective model degrees of freedom (a flexible
    # smooth absorbs part of the noise, so a raw n−1 denominator
    # understates σ and narrows parametric CIs). The total edf is
    # allocated evenly across observed states.
    edf_j = sol.edf / n_obs
    σ_hat = Vector{Float64}(undef, n_obs)
    for j in 1:n_obs
        nv = count(view(valid, :, j))
        nv > 0 || error("bootstrap: observed column $j has no usable cells " *
                        "(every entry is NaN or has zero weight)")
        ss = 0.0
        for i in 1:n_times
            valid[i, j] || continue
            ss += prob.data_weights[i, j] * abs2(resid[i, j])
        end
        σ_hat[j] = sqrt(ss / max(nv - edf_j, 1.0))
    end

    # Build UF evaluation grids (approximators without a domain — e.g. a
    # NeuralApproximator constructed without one — cannot be gridded)
    uf_grids = Dict{Symbol, Vector{Float64}}()
    # Approximators whose FITTED callable is not the univariate function the
    # band grids. A SingleIndexApproximator's callable takes p arguments and
    # a TransformedCovariateApproximator's takes TIME while its `domain` is
    # the STANDARDIZED covariate range, so for both the band is taken from
    # the OUTER curve — recorded here as (approximator, coefficient range) so
    # each replicate can rebuild that curve from its own parameters.
    uf_special = Dict{Symbol, Tuple{Any, UnitRange{Int}}}()
    _uf_offset = 0
    for approx in prob.approximators
        _uf_range = (_uf_offset + 1):(_uf_offset + nparams(approx))
        _uf_offset = last(_uf_range)
        (approx isa SingleIndexApproximator ||
         approx isa TransformedCovariateApproximator) &&
            (uf_special[approx.name] = (approx, _uf_range))
        if approx isa TensorBSplineApproximator
            # Bivariate surface: the UF band machinery grids one variable
            # and calls the evaluator with one argument. Skip its band
            # (parameter/trajectory bootstrap still covers it).
            @warn "bootstrap: approximator :$(approx.name) is bivariate; " *
                  "skipping its unknown-function confidence band " *
                  "(only univariate functions can be gridded)"
            continue
        end
        # `band_domain`, NOT `approx.domain`: a `domain` field is NOT part of
        # the documented approximator protocol (which is `nparams`,
        # `initial_params`, `penalty_matrix`, `build_evaluator`, plus the
        # optional `penalty_blocks`). Reading the field directly threw a hard
        # `FieldError` on a protocol-conforming custom type, and it threw HERE
        # — in the grid loop, before a single replicate was fitted — so the
        # entire bootstrap was unavailable, not merely the band. The default
        # `band_domain` method still reads `a.domain` where it exists, so
        # every built-in path is unchanged; a custom type opts into a band by
        # defining the method.
        dom = band_domain(approx)
        if dom === nothing
            @warn "bootstrap: approximator :$(approx.name) has no domain; " *
                  "skipping its unknown-function confidence band"
            continue
        end
        lo, hi = dom
        uf_grids[approx.name] = collect(range(lo, hi, length=uf_ngrid))
    end
    uf_names = collect(keys(uf_grids))

    # One-argument band curve for `name` in a bootstrap replicate.
    _boot_uf_curve(name, sol_b) =
        haskey(uf_special, name) ?
            (x -> _eval_approx_at(uf_special[name][1],
                                  Float64.(sol_b.parameters[uf_special[name][2]]),
                                  x)) :
            sol_b.unknown_functions[name]

    n_p = length(sol.parameters)

    if parallel && Threads.nthreads() > 1
        # ─── Threaded bootstrap ───────────────────────────────────
        # Pre-generate all pseudo-data with independent RNGs per replicate
        # to ensure reproducibility regardless of thread scheduling.
        rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nboot]
        boot_data = [_resample_data(method, prob.likelihood, fitted, resid,
                                    σ_hat, valid, prob.data_values,
                                    prob.data_weights, n_times, n_obs, rngs[b])
                     for b in 1:nboot]

        # Per-replicate result storage (avoid races)
        results = Vector{Union{Nothing, NamedTuple}}(nothing, nboot)

        if verbose
            println("Bootstrap: $nboot replicates on $(Threads.nthreads()) threads")
        end

        Threads.@threads for b in 1:nboot
            prob_boot = PSMProblem(prob.dynamics!, prob.u0, prob.tspan,
                deepcopy(prob.approximators);   # isolate adaptive (mutable) state per replicate
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
                        f = _boot_uf_curve(name, sol_boot)
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
                                    σ_hat, valid, prob.data_values,
                                    prob.data_weights, n_times, n_obs, rng)

            prob_boot = PSMProblem(prob.dynamics!, prob.u0, prob.tspan,
                deepcopy(prob.approximators);   # isolate adaptive (mutable) state per replicate
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
                    f = _boot_uf_curve(name, sol_boot)
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
    _resample_data(method, family, fitted, resid, σ_hat, valid, orig, w,
                   n_times, n_obs, rng)

Generate bootstrap pseudo-data.

`w` is the problem's `data_weights` matrix. The package convention throughout
is that these are PRECISION weights, `Var(yᵢ) = σ²/wᵢ` (stated where `σ̂` is
estimated above and enforced by `irls_weights`, `W̃ᵢ = wᵢ/V(μᵢ)`), so every
sampler below scales the cell it draws by `wᵢ`. `σ̂` itself is the UNIT-weight
scale — it is estimated with the weights applied — so applying it unscaled to
a `w = 4` cell would simulate that cell `√w = 2×` too wide.

For `:parametric`, the sampling distribution depends on the likelihood family
(μ̂ = `fitted`, w = the cell's precision weight):
- `Gaussian()`:  y* ~ N(μ̂, σ̂²/w)
- `Poisson()`:   w·y* ~ Poisson(w·μ̂), i.e. y* = Poisson(w·μ̂)/w
- `NegativeBinomial(θ)`: y* = NegBin(mean = w·μ̂, shape = w·θ)/w
  (Gamma-Poisson mixture with `G ~ Gamma(wθ, μ̂/θ)`)
- `TruncatedNormal(lower, σ)`: y* ~ TruncNorm(μ̂, σ/√w, lower)
- Other: falls back to Gaussian residuals, N(μ̂, σ̂²/w)

The count families read `w` as a REPLICATE COUNT, which is the only reading
that both reproduces the working variance `V(μ)/w` that `irls_weights`
already commits to AND stays inside the family: `Poisson(wμ)/w` has mean `μ`
and variance `μ/w`; `NegBin(wμ, wθ)/w` has mean `μ` and variance
`(μ + μ²/θ)/w = V(μ)/w`. Its one cost is non-integer pseudo-data when
`w ≠ 1`, which is harmless here — the package's Poisson/NegBin
`log_likelihood` and `loglik_pointwise` use `_loggamma(y+1)` and accept
real `y`.

With uniform weights (`w ≡ 1`) every expression below reduces to the
unweighted one EXACTLY — `x/sqrt(1.0) === x` and `1.0*μ === μ` — so the
unweighted bootstrap is bitwise unchanged.

For `:nonparametric`, residuals are resampled with replacement per state
(valid for Gaussian likelihoods; rejected earlier for other families),
drawing only from cells marked usable in `valid` (positive weight,
non-NaN datum).

Cells NOT in `valid` keep their original value from `orig`: replicates
must present the same missingness pattern as the real data (they are
refit with the same `data_weights`, so masked cells never enter the
replicate objective), and perturbing a NaN cell would either fabricate
data or propagate NaN into the replicate fit. Such cells typically have
`w = 0`, and `1/√0 = Inf` would poison the draw BEFORE that overwrite, so
every sampler floors the weight it uses at 1.0 (`wi = w > 0 ? w : 1.0`).
"""
function _resample_data(method::Symbol, family::AbstractLikelihood,
                        fitted::Matrix{Float64},
                        resid::Matrix{Float64}, σ_hat::Vector{Float64},
                        valid::BitMatrix, orig::Matrix{Float64},
                        w::AbstractMatrix, n_times::Int, n_obs::Int, rng)
    y_boot = similar(fitted)

    if method == :parametric
        _parametric_resample!(y_boot, family, fitted, σ_hat, w,
                              n_times, n_obs, rng)
    elseif method == :nonparametric
        for j in 1:n_obs
            # Pool of usable residuals for this column, CENTERED before
            # resampling: a penalized fit need not have mean-zero
            # residuals (the roughness penalty biases the fit toward
            # smoothness, leaving a systematic offset in the residuals),
            # and resampling an off-center pool would shift every
            # pseudo-dataset by that same offset instead of representing
            # pure noise around the fit.
            #
            # The pool holds STANDARDIZED residuals √wᵢ·rᵢ, not raw ones.
            # Under Var(yᵢ) = σ²/wᵢ the raw residuals are NOT exchangeable —
            # their scale is 1/√wᵢ — so a pool of them is a mixture and
            # assigns every cell the same mixed spread regardless of its own
            # weight. The standardized residuals share the common scale σ;
            # draw from those, then push the draw back onto the target cell's
            # scale with 1/√wᵢ. (Measured on the suite's own D1 fixture —
            # `d1_sd_ratio(:nonparametric, Gaussian(), fill(1.0,17,1),
            # [0.034254])`, w = 1 on rows 1:9 and w = 4 on rows 10:17,
            # 20 000 draws: the raw pool gave replicate sd 0.67208× the
            # model-implied σ/√w at w = 1 and 1.34383× at w = 4 — the SAME
            # absolute spread everywhere, group ratio 0.99976 where 0.5 is
            # correct. The standardized pool gives 0.78065× at BOTH weights,
            # group ratio 0.49999. The common 0.78 shortfall is the finite
            # 17-residual pool and the n-vs-n−edf deficiency of residual
            # resampling, not the weighting; the group RATIO is the
            # discriminating statistic, which is why the test gates on it.)
            pool = Float64[sqrt(w[i, j]) * resid[i, j]
                           for i in 1:n_times if valid[i, j]]
            pool .-= mean(pool)
            np = length(pool)
            for i in 1:n_times
                wi = w[i, j] > 0 ? w[i, j] : 1.0
                y_boot[i, j] = fitted[i, j] + pool[rand(rng, 1:np)] / sqrt(wi)
            end
        end
    end

    # Restore the original values (including the NaN pattern) at masked cells
    for j in 1:n_obs, i in 1:n_times
        valid[i, j] || (y_boot[i, j] = orig[i, j])
    end

    y_boot
end

# ─── Parametric samplers per likelihood family ────────────────────

# Every sampler floors the precision weight it uses at 1.0. A masked cell
# normally carries w = 0 and is overwritten from `orig` by the caller, but
# that overwrite happens AFTER the draw, and `1/√0 = Inf` (or `Poisson(0)`)
# would already have poisoned it. `w ≡ 1` makes each expression bitwise the
# unweighted one.
_boot_wi(w, i, j) = w[i, j] > 0 ? Float64(w[i, j]) : 1.0

function _parametric_resample!(y::Matrix, ::Gaussian, fitted::Matrix,
                               σ_hat::Vector, w::AbstractMatrix,
                               n_t::Int, n_obs::Int, rng)
    # σ̂ is the unit-weight scale; this cell's sd is σ̂/√w.
    for j in 1:n_obs, i in 1:n_t
        y[i, j] = fitted[i, j] + σ_hat[j] / sqrt(_boot_wi(w, i, j)) * randn(rng)
    end
end

function _parametric_resample!(y::Matrix, ::Poisson, fitted::Matrix,
                               σ_hat::Vector, w::AbstractMatrix,
                               n_t::Int, n_obs::Int, rng)
    # Replicate-count reading of the precision weight: y* = Poisson(wμ)/w has
    # mean μ and variance μ/w = V(μ)/w, the working variance `irls_weights`
    # already assumes.
    for j in 1:n_obs, i in 1:n_t
        wi = _boot_wi(w, i, j)
        μ = max(fitted[i, j], 1e-10)
        y[i, j] = Float64(_sample_poisson(wi * μ, rng)) / wi
    end
end

function _parametric_resample!(y::Matrix, fam::NegativeBinomial, fitted::Matrix,
                               σ_hat::Vector, w::AbstractMatrix,
                               n_t::Int, n_obs::Int, rng)
    θ = fam.theta
    for j in 1:n_obs, i in 1:n_t
        wi = _boot_wi(w, i, j)
        μ = max(fitted[i, j], 1e-10)
        # Gamma-Poisson mixture: G ~ Gamma(wθ, μ/θ), then Y ~ Poisson(G),
        # divided by w. That is NegBin(mean = wμ, shape = wθ)/w, which has
        # mean μ and variance (wμ + (wμ)²/(wθ))/w² = (μ + μ²/θ)/w = V(μ)/w.
        g = _sample_gamma(wi * θ, μ / θ, rng)
        y[i, j] = Float64(_sample_poisson(g, rng)) / wi
    end
end

function _parametric_resample!(y::Matrix, fam::TruncatedNormal, fitted::Matrix,
                               σ_hat::Vector, w::AbstractMatrix,
                               n_t::Int, n_obs::Int, rng)
    σ = fam.sigma
    lo = fam.lower
    for j in 1:n_obs, i in 1:n_t
        # Rejection sampling from N(μ, σ²/w) truncated to [lower, ∞). The
        # family's σ is fixed, so the precision weight enters as σ_eff = σ/√w.
        σ_eff = σ / sqrt(_boot_wi(w, i, j))
        μ = fitted[i, j]
        for _ in 1:1000
            z = μ + σ_eff * randn(rng)
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
function _parametric_resample!(y::Matrix, fam::AbstractLikelihood, fitted::Matrix,
                               σ_hat::Vector, w::AbstractMatrix,
                               n_t::Int, n_obs::Int, rng)
    @warn "bootstrap: no parametric sampler for $(typeof(fam)); " *
          "falling back to Gaussian residual sampling" maxlog=1
    for j in 1:n_obs, i in 1:n_t
        y[i, j] = fitted[i, j] + σ_hat[j] / sqrt(_boot_wi(w, i, j)) * randn(rng)
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
