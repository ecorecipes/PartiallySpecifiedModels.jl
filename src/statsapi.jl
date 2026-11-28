# ─── StatsAPI interface for PSMSolution and PSMProblem ───────────
#
# Extends StatsAPI verbs (predict, coef, vcov, dof, loglikelihood, aic, bic,
# nobs, residuals, fitted, isconverged) so PSMSolution integrates cleanly with
# JuliaStats and ecorecipes packages (GAM.jl, DecisionAnalysis.jl, etc.) with
# zero export collisions.

"""
    StatsAPI.predict(sol::PSMSolution)
    StatsAPI.predict(sol::PSMSolution, prob::PSMProblem)

Return the fitted trajectory matrix `(n_times × n_obs)` at the observation times.
"""
StatsAPI.predict(sol::PSMSolution) = sol.fitted_values
StatsAPI.predict(sol::PSMSolution, ::PSMProblem) = sol.fitted_values

"""
    StatsAPI.fitted(sol::PSMSolution)

Return the fitted values matrix `(n_times × n_obs)`.
"""
StatsAPI.fitted(sol::PSMSolution) = sol.fitted_values

"""
    StatsAPI.residuals(sol::PSMSolution)

Return raw response residuals `data_values .- fitted_values`.
"""
StatsAPI.residuals(sol::PSMSolution) = sol.data_values .- sol.fitted_values

"""
    StatsAPI.coef(sol::PSMSolution) -> Vector{Float64}

Return the flattened fitted parameter vector.
"""
StatsAPI.coef(sol::PSMSolution) = Vector{Float64}(sol.parameters)

"""
    StatsAPI.dof(sol::PSMSolution) -> Float64

Return the effective degrees of freedom (`sol.edf`).
"""
StatsAPI.dof(sol::PSMSolution) = sol.edf

"""
    StatsAPI.nobs(sol::PSMSolution) -> Int
    StatsAPI.nobs(prob::PSMProblem) -> Int

Number of usable (finite, positive-weight) scalar observations.
"""
StatsAPI.nobs(sol::PSMSolution) = count(isfinite, sol.data_values)
function StatsAPI.nobs(prob::PSMProblem)
    c = 0
    @inbounds for k in eachindex(prob.data_values)
        _usable(prob.data_values[k], prob.data_weights[k]) && (c += 1)
    end
    c
end

"""
    StatsAPI.stderror(sol::PSMSolution) -> Vector{Float64}

Return standard errors `sqrt.(diag(StatsAPI.vcov(sol)))` of the fitted parameters.
"""
StatsAPI.stderror(sol::PSMSolution) = sqrt.(max.(diag(StatsAPI.vcov(sol)), 0.0))

"""
    StatsAPI.islinear(::PSMSolution) -> Bool

Partially specified dynamical models are generally nonlinear in parameters.
"""
StatsAPI.islinear(::PSMSolution) = false

"""
    StatsAPI.aicc(sol::PSMSolution, prob::PSMProblem) -> Float64

Small-sample corrected Akaike Information Criterion (`AICc`).
"""
function StatsAPI.aicc(sol::PSMSolution, prob::PSMProblem)
    n = StatsAPI.nobs(prob)
    k = StatsAPI.dof(sol) + (prob.likelihood isa GaussianLikelihood ? 1.0 : 0.0)
    a = StatsAPI.aic(sol, prob)
    n > k + 1 ? a + (2.0 * k * (k + 1.0)) / (n - k - 1.0) : Inf
end

"""
    StatsAPI.vcov(sol::PSMSolution) -> Matrix{Float64}

Return the estimated parameter covariance matrix. For LAML / penalized-likelihood
fits, returns the scaled conditional covariance `σ² * V_beta`. For variational
fits, returns `Diagonal(sigma .^ 2)`. For sample-based solvers (`MCMCSolver`,
`MagiSolver`, `FGPGMSolver`, `EnsembleKalmanSolver`, `BNGSolver`), returns the
empirical sample covariance matrix. Raises an `ArgumentError` if no covariance
estimate is available in `sol.convergence`.
"""
_conv_has(c::NamedTuple, k::Symbol) = haskey(c, k)
_conv_has(c::AbstractDict, k::Symbol) = haskey(c, k)
_conv_has(_, ::Symbol) = false
_conv_get(c::NamedTuple, k::Symbol) = getproperty(c, k)
_conv_get(c::AbstractDict, k::Symbol) = c[k]

function StatsAPI.vcov(sol::PSMSolution)
    conv = sol.convergence
    if _conv_has(conv, :V_beta) && _conv_get(conv, :V_beta) !== nothing
        σ² = _conv_has(conv, :sigma2) && _conv_get(conv, :sigma2) !== nothing ?
             Float64(_conv_get(conv, :sigma2)) : 1.0
        return Matrix{Float64}(σ² .* _conv_get(conv, :V_beta))
    elseif _conv_has(conv, :posterior_std) && _conv_get(conv, :posterior_std) isa AbstractVector
        return Matrix{Float64}(Diagonal(Float64.(_conv_get(conv, :posterior_std)) .^ 2))
    elseif _conv_has(conv, :sigma) && _conv_get(conv, :sigma) isa AbstractVector
        return Matrix{Float64}(Diagonal(Float64.(_conv_get(conv, :sigma)) .^ 2))
    elseif _conv_has(conv, :ensemble) && _conv_get(conv, :ensemble) isa AbstractMatrix
        return Matrix{Float64}(cov(_conv_get(conv, :ensemble); dims=2))
    elseif _conv_has(conv, :beta_samples) && _conv_get(conv, :beta_samples) isa AbstractMatrix
        return Matrix{Float64}(cov(_conv_get(conv, :beta_samples); dims=1))
    elseif conv isa MCMCChains.Chains || (_conv_has(conv, :chains) && _conv_get(conv, :chains) isa MCMCChains.Chains)
        ch = conv isa MCMCChains.Chains ? conv : _conv_get(conv, :chains)
        np = length(sol.parameters)
        arr = Array(ch)
        size(arr, 2) >= np && return Matrix{Float64}(cov(arr[:, 1:np]; dims=1))
    end
    throw(ArgumentError(
        "StatsAPI.vcov: this PSMSolution (convergence::$(typeof(conv))) does not " *
        "carry a parameter covariance matrix or posterior sample ensemble."))
end

"""
    StatsAPI.loglikelihood(sol::PSMSolution, prob::PSMProblem) -> Float64

Normalized observation log-likelihood evaluated at `sol.fitted_values`.
For `GaussianLikelihood`, includes the profiled normalizer
`-(n/2) * (log(2π * σ̂²_MLE) + 1)` with `σ̂²_MLE = RSS / n`, making `aic` and
`bic` comparable across models and likelihood families.
"""
function StatsAPI.loglikelihood(sol::PSMSolution, prob::PSMProblem)
    n_times, n_obs = size(prob.data_values)
    n_data = n_times * n_obs
    y_vec = Vector{Float64}(undef, n_data)
    f_vec = Vector{Float64}(undef, n_data)
    w_vec = Vector{Float64}(undef, n_data)
    k = 1
    @inbounds for j in 1:n_obs, i in 1:n_times
        y_vec[k] = prob.data_values[i, j]
        f_vec[k] = sol.fitted_values[i, j]
        w_vec[k] = prob.data_weights[i, j]
        k += 1
    end
    if prob.likelihood isa GaussianLikelihood
        n = _n_usable(y_vec, w_vec)
        n == 0 && return NaN
        rss = 0.0
        sum_log_w = 0.0
        @inbounds for i in 1:n_data
            _usable(y_vec[i], w_vec[i]) || continue
            rss += w_vec[i] * (y_vec[i] - f_vec[i])^2
            sum_log_w += log(w_vec[i])
        end
        sigma2_mle = max(rss / n, 1e-300)
        return -0.5 * n * (log(2π * sigma2_mle) + 1.0) + 0.5 * sum_log_w
    end
    log_likelihood(prob.likelihood, y_vec, f_vec, w_vec)
end

"""
    StatsAPI.aic(sol::PSMSolution, prob::PSMProblem) -> Float64

Akaike Information Criterion using effective degrees of freedom:
`-2 * loglikelihood(sol, prob) + 2 * dof(sol)` (plus 2 for Gaussian's
estimated variance parameter σ²).
"""
function StatsAPI.aic(sol::PSMSolution, prob::PSMProblem)
    k_extra = prob.likelihood isa GaussianLikelihood ? 1.0 : 0.0
    -2.0 * StatsAPI.loglikelihood(sol, prob) + 2.0 * (StatsAPI.dof(sol) + k_extra)
end

"""
    StatsAPI.bic(sol::PSMSolution, prob::PSMProblem) -> Float64

Bayesian Information Criterion using effective degrees of freedom:
`-2 * loglikelihood(sol, prob) + log(nobs(prob)) * dof(sol)`.
"""
function StatsAPI.bic(sol::PSMSolution, prob::PSMProblem)
    n = StatsAPI.nobs(prob)
    k_extra = prob.likelihood isa GaussianLikelihood ? 1.0 : 0.0
    -2.0 * StatsAPI.loglikelihood(sol, prob) + log(max(n, 1)) * (StatsAPI.dof(sol) + k_extra)
end
