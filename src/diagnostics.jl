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

# ─── Cell mask for the diagnostics ───────────────────────────────────

"""
    _diag_usable(sol) -> BitMatrix

Which data cells participate in a residual diagnostic. `true` where the
stored observation is finite.

This is the diagnostics-side reading of the package-wide masking
convention (`usable_cell` in `solver.jl`: positive weight AND a usable
datum). `PSMSolution` stores `data_values` but NOT `data_weights`, so the
weight half of the predicate is not recoverable here; the value half is,
and it is the half that actually poisons the arithmetic — `NaN`
propagates through every mean, `std`, sum-of-squares and denominator
below, which is why a single masked cell used to turn EVERY standardized
residual, the Durbin–Watson statistic and the whole ACF into `NaN`.

LIMITATION (deliberate): a cell masked purely by a ZERO WEIGHT while
carrying a finite value cannot be detected from a `PSMSolution` and is
still counted by these diagnostics. Detecting it would require threading
`data_weights` into the solution struct — a change to a public type, out
of scope here. Every masking test and example in this package marks a
masked cell BOTH ways (`data_values[i,j] = NaN` and
`data_weights[i,j] = 0`), and marking the value is what these functions
key on; follow that convention if a cell must also drop out of the
residual diagnostics.
"""
_diag_usable(sol::PSMSolution) = isfinite.(sol.data_values)

"""Row indices of column `j` whose observation is usable (see `_diag_usable`)."""
_diag_rows(keep::AbstractMatrix{Bool}, j::Int) =
    [i for i in axes(keep, 1) if keep[i, j]]

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
- `usable`: the cell mask actually used (see `_diag_usable`)

Masked observations (a `NaN`/non-finite stored value) are dropped BEFORE
every statistic and every denominator, so `residuals`, `fitted`,
`observed` and the QQ vectors have length equal to the number of usable
cells rather than `length(sol.data_values)`. Without this a single masked
cell made the `std` — and hence every standardized residual — `NaN`. See
`_diag_usable` for the one masking case that cannot be detected from a
`PSMSolution`.

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
    keep = _diag_usable(sol)
    # `vec` is column-major, and so is `vec(keep)` — the linear indices line up.
    idx = findall(vec(keep))
    y = vec(sol.data_values)[idx]
    mu = vec(sol.fitted_values)[idx]

    if family !== nothing && !(family isa Gaussian)
        r = deviance_residuals(family, y, mu)
        # Robust scale: for N(0,1)-distributed residuals, median|r| ≈ 0.6745,
        # so dividing by the raw median|r| inflates the standardized
        # residuals ~1.48× against the N(0,1) QQ reference line.
        sc = median_abs(r) / 0.6745
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

    # QQ data: sorted standardized residuals vs normal quantiles.
    # `n` is the USABLE-cell count — the plotting-position denominator must
    # not count cells that contribute no residual.
    n = length(r_std)
    sorted = sort(r_std)
    theoretical = [_qnorm((i - 0.5) / n) for i in 1:n]

    # DW per observed state, over that state's usable rows only.
    resid_mat = sol.data_values .- sol.fitted_values
    dw = [durbin_watson(resid_mat[_diag_rows(keep, j), j])
          for j in axes(resid_mat, 2)]

    (residuals=r_std, fitted=mu, observed=y,
     qq_theoretical=theoretical, qq_sample=sorted,
     durbin_watson=dw, usable=keep)
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
- `residuals`: raw residuals (observed - fitted), full `(time × state)` shape
- `usable`: cell mask; `residuals` is non-finite exactly where this is `false`
- `durbin_watson`: DW statistic per observed state (2.0 = no autocorrelation)
- `acf`: autocorrelation at lags 1:maxlag per state
- `semivariogram`: `(lags, gamma)` per observed state

Masked observations are excluded from every statistic and every
denominator: each column's DW, ACF and semivariogram are computed over
that column's usable rows only (with the matching `data_times` subset for
the semivariogram lags). `residuals` keeps its rectangular shape, so a
masked cell shows up there as the `NaN` it is — read it through `usable`.
Before this, one masked cell made `durbin_watson`, the whole `acf` and
the semivariogram `NaN`. See `_diag_usable` for the masking case that
cannot be detected from a `PSMSolution`.
"""
function residual_diagnostics(sol::PSMSolution; maxlag::Int=10)
    resid = sol.data_values .- sol.fitted_values
    keep = _diag_usable(sol)
    times = sol.data_times
    ncol = size(resid, 2)

    rows_per_col = [_diag_rows(keep, j) for j in 1:ncol]

    # One common lag count so `acf` stays a matrix. `residual_acf` already
    # caps at `n - 1`; with masking the columns can have different usable
    # counts, so cap at the SHORTEST usable column rather than padding the
    # rest with NaN (which would put the poison straight back).
    n_min = ncol == 0 ? 0 : minimum(length.(rows_per_col))
    maxlag_eff = max(min(maxlag, n_min - 1), 0)

    dw = Float64[]
    acf_cols = Vector{Vector{Float64}}(undef, ncol)
    svgs = [(lags=Float64[], gamma=Float64[]) for _ in 1:ncol]
    for j in 1:ncol
        rows = rows_per_col[j]
        rj = resid[rows, j]
        push!(dw, durbin_watson(rj))
        acf_cols[j] = residual_acf(rj; maxlag=maxlag_eff)
        c, g = semivariogram(times[rows], rj)
        svgs[j] = (lags=c, gamma=g)
    end
    acf = ncol == 0 ? zeros(0, 0) : hcat(acf_cols...)

    (residuals=resid, usable=keep, durbin_watson=dw, acf=acf,
     semivariogram=svgs)
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
        # The band machinery grids a SINGLE input variable and evaluates the
        # unknown function with one argument; a bivariate tensor surface has
        # no univariate band. Reject loudly rather than erroring obscurely
        # on the missing `.domain` field below.
        approx isa TensorBSplineApproximator && throw(ArgumentError(
            "confidence_band: TensorBSplineApproximator (:$(approx.name)) " *
            "is a bivariate surface; univariate confidence bands are not " *
            "defined for it. Evaluate sol.unknown_functions[:" *
            "$(approx.name)](x, y) directly for point estimates."))
        np = nparams(approx)
        idx = (offset+1):(offset+np)
        V_k = σ² .* V_beta[idx, idx]

        grid = collect(range(approx.domain[1], approx.domain[2], length=uf_ngrid))
        # A single-index approximator's fitted callable takes p arguments, so
        # its band is the band of the OUTER curve s(z) over the STANDARDIZED
        # index z ∈ [−xi, xi] — the univariate payoff a tensor surface cannot
        # have. A transformed-covariate approximator is the same story with a
        # different inner statistic: its callable takes TIME while its domain
        # is the standardized covariate range, so calling it on the grid would
        # evaluate f at times −xi…xi. Everything else is evaluated through its
        # own callable.
        f_est = if approx isa SingleIndexApproximator ||
                   approx isa TransformedCovariateApproximator
            Float64[_eval_approx_at(approx, Float64.(sol.parameters[idx]), x)
                    for x in grid]
        else
            Float64[sol.unknown_functions[approx.name](x) for x in grid]
        end
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

function _eval_approx_at(approx::ShapeConstrainedGPApproximator, p::Vector{Float64}, x::Real)
    build_constrained_gp_evaluator(approx, p)(x)
end

"""
Single index: the band is over the OUTER smooth `s(z)` evaluated at the
standardized index `z`. The inner loadings do not enter `s` at a FIXED `z`,
so their finite-difference sensitivities are exactly zero and the band is
the uncertainty of the curve itself — not of where a given state maps onto
it.
"""
function _eval_approx_at(approx::SingleIndexApproximator, p::Vector{Float64},
                         x::Real)
    ni = _si_n_inner(approx)
    build_evaluator(approx.outer, p[(ni + 1):end])(x)
end

"""
Transformed covariate: the band is over the OUTER response curve `s(z)`
evaluated at the standardized covariate `z`, for the same reason as the
single index — the fitted callable is a function of TIME, but `domain` is
the range of `z`. The inner transformation parameters do not enter `s` at a
FIXED `z`, so their finite-difference sensitivities are exactly zero and
the band is the uncertainty of the response curve itself, not of where a
given date maps onto it.
"""
function _eval_approx_at(approx::TransformedCovariateApproximator,
                         p::Vector{Float64}, x::Real)
    ni = _tc_n_inner(approx)
    build_evaluator(approx.outer, p[(ni + 1):end])(x)
end

function _eval_approx_at(approx, p::Vector{Float64}, x::Real)
    error("confidence_band: unsupported approximator type $(typeof(approx))")
end

# ─── Post-fit shape-constraint audit ──────────────────────────────

"""
The SCOP-reparameterized approximator types whose constraints hold exactly
at their node/inducing values but only approximately between them
(`ShapeConstrainedSPDEApproximator`, `ShapeConstrainedGPApproximator`) —
plus `ShapeConstrainedBSplineApproximator`, whose B-spline convex-hull
property makes the constraints exact everywhere and which is included so a
mixed-approximator audit reports on every declared constraint.
"""
const _SHAPE_CHECKABLE = Union{ShapeConstrainedBSplineApproximator,
                               ShapeConstrainedSPDEApproximator,
                               ShapeConstrainedGPApproximator}

"""
    check_constraints(f, constraint, domain; grid_size=201, tol=1e-6)

Audit a fitted callable `f` against a declared shape constraint (one of
`SHAPE_CONSTRAINTS`) on a `grid_size`-point uniform grid over
`domain = (lo, hi)`.

Motivation: `ShapeConstrainedSPDEApproximator` and
`ShapeConstrainedGPApproximator` enforce their constraint only AT the mesh
nodes / inducing-point values; the interpolant between them (cubic spline /
GP kernel) has cardinal functions that take negative values, so the fitted
function can violate the constraint between nodes — measured dips of −0.505
(SCGP) and −0.121 (SCSPDE) on an all-positive-node `:positive` fixture with
node values alternating between ≈5 and ≈0. This checker makes such
violations visible after a fit. (`ShapeConstrainedBSplineApproximator` is
exact everywhere by the B-spline convex-hull property and should always
pass.)

Violations are measured per constrained quantity:
- value (`:positive`, `:dec_positive`): `max(0, -f)`;
- slope (monotone families): forward differences divided by the grid step;
- curvature (`:convex`/`:concave` families): second differences over step²;
- endpoint pin (zero-at-endpoint families): `|f(endpoint)|`.

Each is normalized by the maximum absolute value of the same quantity on
the grid (with a small floor), and the function fails the check when the
worst normalized violation exceeds `tol`. A violation at or below the
finite-difference round-off level of its own quantity
(`8·eps·yscale/h^k`, with `k = 0, 1, 2` for value, slope and curvature and
`h` the grid step) is treated as exactly zero, so a function that satisfies
its constraint exactly passes at any `grid_size`.

Returns a NamedTuple `(constraint, satisfied, worst_relative,
worst_absolute, worst_quantity, worst_location, grid_size, tol)` where
`worst_quantity` is `:value`, `:slope`, `:curvature`, `:endpoint`, or
`:none` and `worst_absolute` is in the units of that quantity (function
value, f′, f″, or endpoint value respectively).

Note the audit is grid-based: it can miss a violation narrower than the
grid spacing, so it is a diagnostic, not a certificate.
"""
function check_constraints(f, constraint::Symbol, domain::Tuple{<:Real,<:Real};
                           grid_size::Int=201, tol::Real=1e-6)
    constraint in SHAPE_CONSTRAINTS || throw(ArgumentError(
        "Unknown constraint :$constraint. Must be one of $SHAPE_CONSTRAINTS"))
    grid_size >= 3 || throw(ArgumentError("grid_size must be ≥ 3, got $grid_size"))
    lo, hi = Float64(domain[1]), Float64(domain[2])
    xs = collect(range(lo, hi, length=grid_size))
    h = xs[2] - xs[1]
    ys = Float64[f(x) for x in xs]
    d1 = diff(ys) ./ h        # forward-difference slope estimates
    d2 = diff(d1) ./ h        # second-difference curvature estimates
    yscale = maximum(abs, ys)
    # Absolute floor keeps a flat function's O(eps) finite-difference noise
    # from being magnified into a spurious relative violation.
    floorv = 1e-10 * (1.0 + yscale)
    # Per-quantity noise gate. `floorv` alone is not enough: it is in VALUE
    # units, but the slope and curvature violations are differenced 1 and 2
    # times, so their round-off noise is ≈ eps·yscale/h^k (k = differencing
    # order) — which exceeds any fixed value-unit floor and GROWS as the grid
    # is refined. Without this gate, exactly-satisfying functions report
    # violations that get worse with `grid_size`: measured on f(x) = x under
    # :convex, worst_relative 0.0074 (domain (0,2), grid 201) → 0.185 (grid
    # 1001), and 0.231 → 1.0 on domain (0,0.2); :increasing on
    # cos(x)^2+sin(x)^2 reported 1.67e-4. A violation at or below the noise
    # level for its own quantity is therefore treated as exactly zero.
    fd_noise(k) = 8 * eps(Float64) * yscale / h^k

    worst_rel = 0.0
    worst_abs = 0.0
    worst_q = :none
    worst_x = lo
    consider = (viol, scale, q, x, k) -> begin
        viol > fd_noise(k) || return nothing
        rel = viol / max(scale, floorv)
        if rel > worst_rel
            worst_rel = rel; worst_abs = viol; worst_q = q; worst_x = x
        end
        nothing
    end

    if constraint in (:increasing, :inc_convex, :inc_concave,
                      :inc_zero_left, :inc_zero_right)
        i = argmin(d1)
        consider(max(0.0, -d1[i]), maximum(abs, d1), :slope, xs[i], 1)
    elseif constraint in (:decreasing, :dec_convex, :dec_concave,
                          :dec_positive, :dec_zero_left, :dec_zero_right)
        i = argmax(d1)
        consider(max(0.0, d1[i]), maximum(abs, d1), :slope, xs[i], 1)
    end
    if constraint in (:convex, :inc_convex, :dec_convex)
        i = argmin(d2)
        consider(max(0.0, -d2[i]), maximum(abs, d2), :curvature, xs[i + 1], 2)
    elseif constraint in (:concave, :inc_concave, :dec_concave)
        i = argmax(d2)
        consider(max(0.0, d2[i]), maximum(abs, d2), :curvature, xs[i + 1], 2)
    end
    if constraint in (:positive, :dec_positive)
        i = argmin(ys)
        consider(max(0.0, -ys[i]), yscale, :value, xs[i], 0)
    end
    if constraint in _ZERO_ENDPOINT_CONSTRAINTS
        x0 = constraint in (:inc_zero_left, :dec_zero_left) ? lo : hi
        consider(abs(Float64(f(x0))), yscale, :endpoint, x0, 0)
    end

    (constraint=constraint, satisfied=(worst_rel <= tol),
     worst_relative=worst_rel, worst_absolute=worst_abs,
     worst_quantity=worst_q, worst_location=worst_x,
     grid_size=grid_size, tol=Float64(tol))
end

"""
    check_constraints(sol::PSMSolution, prob::PSMProblem;
                      grid_size=201, tol=1e-6, warn=true)

Audit every shape-constrained SCOP approximator of `prob`
(`ShapeConstrainedBSplineApproximator`, `ShapeConstrainedSPDEApproximator`,
`ShapeConstrainedGPApproximator`) against its declared constraint, using
the fitted evaluators in `sol.unknown_functions`. Other approximator types
are skipped.

Returns a `Dict{Symbol,NamedTuple}` keyed by approximator name with the
per-function results of the single-function method (see
[`check_constraints(f, constraint, domain)`](@ref check_constraints)).
With `warn=true` (default) a `@warn` is logged for each function whose
declared constraint is violated beyond `tol` — expected only for the SPDE
and GP variants, whose between-node interpolation does not preserve the
constraint that holds exactly at their node values.
"""
function check_constraints(sol::PSMSolution, prob::PSMProblem;
                           grid_size::Int=201, tol::Real=1e-6,
                           warn::Bool=true)
    results = Dict{Symbol,NamedTuple}()
    for a in prob.approximators
        a isa _SHAPE_CHECKABLE || continue
        haskey(sol.unknown_functions, a.name) || continue
        r = check_constraints(sol.unknown_functions[a.name], a.constraint,
                              a.domain; grid_size=grid_size, tol=tol)
        results[a.name] = r
        if warn && !r.satisfied
            @warn("Fitted function :$(a.name) violates its declared " *
                  ":$(a.constraint) constraint between nodes " *
                  "(worst relative violation $(round(r.worst_relative, sigdigits=3)) " *
                  "in $(r.worst_quantity) at x = " *
                  "$(round(r.worst_location, sigdigits=4))). " *
                  "$(nameof(typeof(a))) enforces the constraint at its " *
                  "node/inducing values only; see the approximator docstring " *
                  "for what reduces between-node violations.")
        end
    end
    results
end

