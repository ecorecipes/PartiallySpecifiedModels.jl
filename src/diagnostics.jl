"""
Residual diagnostics for PSM solutions.

Provides tools to assess fit quality and detect oversmoothing:
- `appraise` — 4-panel diagnostic data (QQ, residuals vs fitted, histogram, obs vs fitted)
- `deviance_residuals` — per-observation deviance residuals for each likelihood family
- `durbin_watson` — Durbin-Watson statistic for residual autocorrelation
- `residual_acf` — empirical autocorrelation function (ACF)
- `semivariogram` — empirical semivariogram
"""

using Statistics: mean, std

# ─── Inverse standard normal CDF ─────────────────────────────────────

"""
    _qnorm(p)

Inverse standard normal CDF (quantile function) via Acklam's rational
approximation.  Accuracy: |ε| < 1.15×10⁻⁹ for 0 < p < 1.
"""
function _qnorm(p::Real)
    # Peter Acklam's rational approximation
    a = (-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = (7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low
        q = sqrt(-2.0 * log(p))
        (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
         ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) * q /
         (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
          ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0)
    end
end

# ─── Deviance residuals ──────────────────────────────────────────────

"""
    deviance_residuals(family, y, mu)

Compute per-observation deviance residuals: ``r_i^D = \\text{sign}(y_i - \\mu_i) \\sqrt{d_i}``
where ``d_i`` is the unit deviance contribution.

For a well-specified model, deviance residuals are approximately standard normal.
"""
function deviance_residuals(::Gaussian, y::AbstractVector, mu::AbstractVector)
    y .- mu
end

function deviance_residuals(::Poisson, y::AbstractVector, mu::AbstractVector)
    [begin
        yi, mi = y[i], max(mu[i], 1e-10)
        d = yi > 0 ? 2.0 * (yi * log(yi / mi) - (yi - mi)) : 2.0 * mi
        sign(yi - mi) * sqrt(max(d, 0.0))
    end for i in eachindex(y)]
end

function deviance_residuals(fam::NegativeBinomial, y::AbstractVector, mu::AbstractVector)
    k = fam.theta
    [begin
        yi, mi = y[i], max(mu[i], 1e-10)
        d_y = yi > 0 ? yi * log(yi / mi) : 0.0
        d_k = (yi + k) * log((yi + k) / (mi + k))
        sign(yi - mi) * sqrt(max(2.0 * (d_y - d_k), 0.0))
    end for i in eachindex(y)]
end

function deviance_residuals(::TruncatedNormal, y::AbstractVector, mu::AbstractVector)
    y .- mu  # same as Gaussian for response-scale residuals
end

# Fallback for unknown families — use raw residuals
function deviance_residuals(::AbstractLikelihood, y::AbstractVector, mu::AbstractVector)
    y .- mu
end

# ─── Appraise (4-panel diagnostic data) ──────────────────────────────

"""
    appraise(sol::PSMSolution; family=nothing)

Compute diagnostic data for a standard 4-panel goodness-of-fit display,
following the pattern of R's `gratia::appraise()` for GAMs.

Returns a named tuple with fields:
- `residuals`: standardized residuals (deviance if `family` given, else response/σ̂)
- `fitted`: fitted values (vectorized across all observed states)
- `observed`: observed values (vectorized)
- `qq_theoretical`: theoretical normal quantiles
- `qq_sample`: sorted standardized residuals
- `durbin_watson`: DW statistic per observed state

## Example

```julia
diag = appraise(sol)

# 4-panel plot with Plots.jl:
p_qq = scatter(diag.qq_theoretical, diag.qq_sample, title="QQ Plot")
p_rf = scatter(diag.fitted, diag.residuals, title="Residuals vs Fitted")
p_hist = histogram(diag.residuals, title="Histogram of Residuals")
p_of = scatter(diag.observed, diag.fitted, title="Observed vs Fitted")
plot(p_qq, p_rf, p_hist, p_of, layout=(2,2))
```
"""
function appraise(sol::PSMSolution; family::Union{Nothing, AbstractLikelihood}=nothing)
    y = vec(sol.data_values)
    mu = vec(sol.fitted_values)

    if family !== nothing && !(family isa Gaussian)
        r = deviance_residuals(family, y, mu)
        # Standardize by estimated scale (median absolute deviance residual)
        sc = median_abs(r)
        if sc > 1e-10
            r_std = r ./ sc
        else
            r_std = copy(r)
        end
    else
        r = y .- mu
        σ = std(r; corrected=true)
        r_std = σ > 1e-10 ? r ./ σ : copy(r)
    end

    # QQ data: sorted standardized residuals vs normal quantiles
    n = length(r_std)
    sorted = sort(r_std)
    theoretical = [_qnorm((i - 0.5) / n) for i in 1:n]

    # DW per observed state
    resid_mat = sol.data_values .- sol.fitted_values
    dw = durbin_watson(resid_mat)

    (residuals=r_std, fitted=mu, observed=y,
     qq_theoretical=theoretical, qq_sample=sorted,
     durbin_watson=dw)
end

"""Median of absolute values (robust scale estimator, avoids StatsBase dependency)."""
function median_abs(x::AbstractVector)
    ax = abs.(x)
    sort!(ax)
    n = length(ax)
    n == 0 && return 0.0
    isodd(n) ? ax[(n+1)÷2] : 0.5 * (ax[n÷2] + ax[n÷2+1])
end

"""
    durbin_watson(residuals)

Compute the Durbin–Watson statistic for temporal autocorrelation in residuals.

- DW ≈ 2: no autocorrelation (good fit)
- DW < 2: positive autocorrelation (oversmoothing — systematic patterns remain)
- DW > 2: negative autocorrelation (overfitting — alternating residuals)

For multivariate data, computes per-column and returns a vector.
"""
function durbin_watson(r::AbstractVector)
    n = length(r)
    n < 2 && return NaN
    num = sum((r[i] - r[i-1])^2 for i in 2:n)
    den = sum(r[i]^2 for i in 1:n)
    den < 1e-30 ? NaN : num / den
end

function durbin_watson(r::AbstractMatrix)
    [durbin_watson(r[:, j]) for j in axes(r, 2)]
end

"""
    residual_acf(residuals; maxlag=10)

Compute empirical autocorrelation function (ACF) of residuals at lags 1:maxlag.

Returns a vector of autocorrelations. Values significantly different from zero
at low lags indicate the model is over- or under-smoothing.
"""
function residual_acf(r::AbstractVector; maxlag::Int=10)
    n = length(r)
    maxlag = min(maxlag, n - 1)
    r_centered = r .- mean(r)
    var_r = sum(r_centered .^ 2)
    var_r < 1e-30 && return fill(NaN, maxlag)
    [sum(r_centered[i] * r_centered[i+h] for i in 1:n-h) / var_r for h in 1:maxlag]
end

function residual_acf(r::AbstractMatrix; maxlag::Int=10)
    hcat([residual_acf(r[:, j]; maxlag) for j in axes(r, 2)]...)
end

"""
    semivariogram(times, residuals; maxlag=nothing, nbins=15)

Compute empirical semivariogram γ(h) = (1/2|N(h)|) Σ (r(t) - r(t+h))².

Returns `(lag_centers, gamma)`. For a well-fitted model with independent
residuals, γ(h) should be approximately constant (≈ σ²) across all lags.
A rising semivariogram at small lags indicates positive autocorrelation
(oversmoothing).
"""
function semivariogram(times::AbstractVector, r::AbstractVector;
                       maxlag=nothing, nbins::Int=15)
    n = length(r)
    n < 3 && return (Float64[], Float64[])

    # Compute all pairwise squared differences and lags
    dists = Float64[]
    sqdiffs = Float64[]
    for i in 1:n, j in (i+1):n
        push!(dists, abs(times[j] - times[i]))
        push!(sqdiffs, (r[j] - r[i])^2)
    end

    if maxlag === nothing
        maxlag = maximum(dists) / 2
    end

    # Bin into equal-width bins
    edges = range(0.0, maxlag, length=nbins+1)
    centers = Float64[]
    gamma = Float64[]
    for b in 1:nbins
        lo, hi = edges[b], edges[b+1]
        mask = lo .< dists .<= hi
        count = sum(mask)
        if count > 0
            push!(centers, (lo + hi) / 2)
            push!(gamma, 0.5 * sum(sqdiffs[mask]) / count)
        end
    end
    (centers, gamma)
end

"""
    residual_diagnostics(sol::PSMSolution)

Compute a suite of residual diagnostics for a PSM solution.

Returns a named tuple with:
- `residuals`: raw residuals (observed - fitted)
- `durbin_watson`: DW statistic per observed state (2.0 = no autocorrelation)
- `acf`: autocorrelation at lags 1:10 per state
- `semivariogram`: `(lags, gamma)` per observed state
"""
function residual_diagnostics(sol::PSMSolution)
    resid = sol.data_values .- sol.fitted_values
    dw = durbin_watson(resid)
    acf = residual_acf(resid)
    times = sol.data_times

    svgs = [(lags=Float64[], gamma=Float64[]) for _ in axes(resid, 2)]
    for j in axes(resid, 2)
        c, g = semivariogram(times, resid[:, j])
        svgs[j] = (lags=c, gamma=g)
    end

    (residuals=resid, durbin_watson=dw, acf=acf, semivariogram=svgs)
end

# ─── Bayesian confidence bands from LAML posterior ────────────────

"""
    confidence_band(sol::PSMSolution, prob::PSMProblem;
                    level=0.95, uf_ngrid=100)

Compute Bayesian confidence/credible bands for the unknown functions
using the posterior covariance from the LAML fit.

The posterior covariance is `V_β = σ̂² (J'WJ + S^λ)⁻¹`, and the
pointwise standard error of `f(x)` is `se(x) = √(b(x)' V_β b(x))`,
where `b(x)` is the basis vector mapping coefficients to function value.

These "across-the-function" intervals (Nychka 1988, Wood 2006 §4.8)
account for smoothing bias and typically achieve near-nominal coverage,
unlike bootstrap CIs which only capture sampling variability.

Returns a Dict mapping each unknown function name to a NamedTuple
`(grid, fitted, lower, upper, se)`.

## Example

```julia
sol = solve(prob, LAML(maxiters=80))
bands = confidence_band(sol, prob)

# Plot unknown function with 95% credible band
plot(bands[:λ].grid, bands[:λ].fitted, lw=2, label="Estimated λ")
plot!(bands[:λ].grid, bands[:λ].lower,
      fillrange=bands[:λ].upper, fillalpha=0.2, label="95% CI")
```
"""
function confidence_band(sol::PSMSolution, prob::PSMProblem;
                         level::Float64=0.95, uf_ngrid::Int=100)
    # Check that V_beta is available
    conv = sol.convergence
    if conv === nothing || !hasproperty(conv, :V_beta) || conv.V_beta === nothing
        error("confidence_band: posterior covariance V_β not available. " *
              "Only LAML-fitted solutions support this. Refit with LAML().")
    end
    V_beta = conv.V_beta
    σ² = conv.sigma2

    z = _qnorm(1.0 - (1.0 - level) / 2.0)

    result = Dict{Symbol, NamedTuple{(:grid, :fitted, :lower, :upper, :se),
                  Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64},
                        Vector{Float64}, Vector{Float64}}}}()

    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        idx = (offset+1):(offset+np)
        V_k = σ² .* V_beta[idx, idx]

        grid = collect(range(approx.domain[1], approx.domain[2], length=uf_ngrid))
        f_est = Float64[sol.unknown_functions[approx.name](x) for x in grid]
        se = zeros(uf_ngrid)

        # Compute ∂f(x)/∂β via central finite differences for all approximator types.
        # This handles B-splines, shape-constrained, SPDE, GP uniformly.
        params_k = Float64.(sol.parameters[idx])
        for (k, x) in enumerate(grid)
            jac = zeros(np)
            for j in 1:np
                eps_fd = max(abs(params_k[j]) * 1e-4, 1e-5)
                p_plus = copy(params_k); p_plus[j] += eps_fd
                p_minus = copy(params_k); p_minus[j] -= eps_fd
                f_plus = _eval_approx_at(approx, p_plus, x)
                f_minus = _eval_approx_at(approx, p_minus, x)
                jac[j] = (f_plus - f_minus) / (2 * eps_fd)
            end
            se[k] = sqrt(max(dot(jac, V_k * jac), 0.0))
        end

        lower = f_est .- z .* se
        upper = f_est .+ z .* se

        result[approx.name] = (grid=grid, fitted=f_est, lower=lower, upper=upper, se=se)
        offset += np
    end

    result
end

"""Evaluate an approximator at point x given parameter vector p."""
function _eval_approx_at(approx::BSplineApproximator, p::Vector{Float64}, x::Real)
    knots = collect(range(approx.domain[1], approx.domain[2], length=approx.nknots))
    build_bspline_evaluator(knots, p)(x)
end

function _eval_approx_at(approx::ShapeConstrainedBSplineApproximator, p::Vector{Float64}, x::Real)
    build_constrained_bspline_evaluator(approx, p)(x)
end

function _eval_approx_at(approx::ShapeConstrainedSPDEApproximator, p::Vector{Float64}, x::Real)
    build_constrained_spde_evaluator(approx, p)(x)
end

function _eval_approx_at(approx::SPDEApproximator, p::Vector{Float64}, x::Real)
    build_spde_evaluator(approx.mesh_points, p)(x)
end

function _eval_approx_at(approx::GPApproximator, p::Vector{Float64}, x::Real)
    build_gp_evaluator(approx, p)(x)
end

function _eval_approx_at(approx, p::Vector{Float64}, x::Real)
    error("confidence_band: unsupported approximator type $(typeof(approx))")
end

"""Piecewise linear basis vector for SPDE mesh."""
function _piecewise_linear_basis(x::Real, mesh::Vector{Float64})
    n = length(mesh)
    b = zeros(n)
    if x <= mesh[1]
        b[1] = 1.0
    elseif x >= mesh[end]
        b[end] = 1.0
    else
        for i in 1:n-1
            if mesh[i] <= x <= mesh[i+1]
                t = (x - mesh[i]) / (mesh[i+1] - mesh[i])
                b[i] = 1.0 - t
                b[i+1] = t
                break
            end
        end
    end
    b
end
