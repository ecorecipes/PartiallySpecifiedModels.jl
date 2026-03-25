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
