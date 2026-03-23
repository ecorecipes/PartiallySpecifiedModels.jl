"""
Residual diagnostics for PSM solutions.

Provides tools to assess fit quality and detect oversmoothing:
- Durbin-Watson statistic for residual autocorrelation
- Empirical autocorrelation function (ACF)
- Empirical semivariogram
"""

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
