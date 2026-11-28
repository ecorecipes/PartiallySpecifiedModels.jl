# ─── Model Comparison & Parametric Hypothesis Testing ───────────────
#
# 1. `compare_models`: AIC / AICc / BIC / Akaike-weight / LAML comparison table
#    across candidate PSMs (supports `PSMProblem` and `SDEPSMProblem`).
# 2. `test_parametric_hypothesis`: Wood (2013)-style Wald chi-square test of
#    whether an estimated nonparametric smooth `f(x)` departs significantly from
#    a parametric null function `h0(x)`, `:constant`, or `:linear`.

# ─── Self-contained regularized upper incomplete gamma Q(a, x) ──────
# Used for exact chi-square survival function P(χ²_df ≥ stat) = Q(df/2, stat/2)
# without requiring SpecialFunctions or Distributions.

function _gamma_p_series(a::Float64, x::Float64)
    ap = a
    term = 1.0 / a
    s = term
    for _ in 1:400
        ap += 1.0
        term *= x / ap
        s += term
        abs(term) <= abs(s) * 1e-14 && break
    end
    s * exp(-x + a * log(x) - _loggamma(a))
end

function _gamma_q_cf(a::Float64, x::Float64)
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in 1:400
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        abs(d) < fpmin && (d = fpmin)
        c = b + an / c
        abs(c) < fpmin && (c = fpmin)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) <= 1e-14 && break
    end
    exp(-x + a * log(x) - _loggamma(a)) * h
end

function _chisq_ccdf(df::Real, stat::Real)
    stat <= 0.0 && return 1.0
    (!isfinite(stat) || df <= 0.0) && return 0.0
    a = 0.5 * Float64(df)
    x = 0.5 * Float64(stat)
    val = x < a + 1.0 ? (1.0 - _gamma_p_series(a, x)) : _gamma_q_cf(a, x)
    clamp(val, 0.0, 1.0)
end

# ─── Model comparison table ─────────────────────────────────────────

"""
    ModelComparisonTable

Result of [`compare_models`](@ref): a ranked table of candidate PSM fits with
`name`, `nobs`, `edf`, `loglik`, `aic`, `aicc`, `bic`, `delta_aic`,
`akaike_weight`, `delta_aicc`, `aicc_weight`, `laml`, and `data_loss`.
Indexable and iterable over its `rows::Vector{NamedTuple}`.
"""
struct ModelComparisonTable
    rows::Vector{NamedTuple}
    sort_by::Symbol
end

Base.length(t::ModelComparisonTable) = length(t.rows)
Base.getindex(t::ModelComparisonTable, i) = t.rows[i]
Base.iterate(t::ModelComparisonTable, args...) = iterate(t.rows, args...)

function Base.show(io::IO, tbl::ModelComparisonTable)
    println(io, "ModelComparisonTable ($(length(tbl.rows)) models, sorted by :$(tbl.sort_by)):")
    println(io, "  Model                |  EDF  |    LogLik   |     AIC     |    AICc     |   ΔAICc  | Weight |     BIC")
    println(io, "  " * "-"^100)
    for r in tbl.rows
        nm = rpad(string(r.name), 20)
        @printf(io, "  %s | %5.2f | %11.3f | %11.3f | %11.3f | %8.3f | %6.3f | %11.3f\n",
                nm, r.edf, r.loglik, r.aic, r.aicc, r.delta_aicc, r.aicc_weight, r.bic)
    end
end

function _extract_model_metrics(sol::PSMSolution, prob, name::String)
    if prob isa SDEPSMProblem
        n = count(isfinite, prob.data_values)
        k = sol.edf
        ll = haskey(sol.convergence, :loglik) ? Float64(sol.convergence.loglik) : NaN
        aic_v = -2.0 * ll + 2.0 * k
        aicc_v = (n - k - 1.0) > 0.0 ? aic_v + (2.0 * k * (k + 1.0)) / (n - k - 1.0) : Inf
        bic_v = -2.0 * ll + k * log(max(n, 1))
        laml_v = NaN
    else
        n = StatsAPI.nobs(prob)
        k = StatsAPI.dof(sol)
        ll = StatsAPI.loglikelihood(sol, prob)
        aic_v = StatsAPI.aic(sol, prob)
        aicc_v = StatsAPI.aicc(sol, prob)
        bic_v = StatsAPI.bic(sol, prob)
        laml_v = (sol.convergence isa NamedTuple && haskey(sol.convergence, :laml)) ?
                 Float64(sol.convergence.laml) : NaN
    end
    (name=name, nobs=n, edf=k, loglik=ll, aic=aic_v, aicc=aicc_v, bic=bic_v,
     laml=laml_v, data_loss=sol.data_loss)
end

"""
    compare_models(pairs::Pair{<:PSMSolution, <:Any}...; names=nothing, sort_by=:aicc)
    compare_models(sols, probs; names=nothing, sort_by=:aicc)

Compare multiple fitted PSMs via AIC, AICc, BIC, Akaike weights, and LAML.
Accepts pairs `sol => prob` (where `prob` is a `PSMProblem` or `SDEPSMProblem`)
or matching vectors `sols`, `probs`.

# Example
```julia
tbl = compare_models(sol_lin => prob_lin, sol_flex => prob_flex;
                     names=["Linear", "Flexible spline"])
```
"""
function compare_models(sols::AbstractVector{<:PSMSolution},
                        probs::AbstractVector;
                        names::Union{Nothing, AbstractVector}=nothing,
                        sort_by::Symbol=:aicc)
    n_mod = length(sols)
    n_mod >= 1 || throw(ArgumentError("compare_models: need at least one model"))
    length(probs) == n_mod || throw(ArgumentError("compare_models: sols and probs must have same length"))
    sort_by in (:aic, :aicc, :bic, :laml) || throw(ArgumentError(
        "compare_models: sort_by must be :aic, :aicc, :bic, or :laml (got :$sort_by)"))

    mod_names = names === nothing ? ["Model_$i" for i in 1:n_mod] : [string(nm) for nm in names]
    length(mod_names) == n_mod || throw(ArgumentError("compare_models: names length must match models"))

    raw = [_extract_model_metrics(sols[i], probs[i], mod_names[i]) for i in 1:n_mod]
    min_aic = minimum(r.aic for r in raw)
    min_aicc = minimum(r.aicc for r in raw)
    d_aic = [r.aic - min_aic for r in raw]
    d_aicc = [r.aicc - min_aicc for r in raw]
    w_aic_raw = [isfinite(d) ? exp(-0.5 * d) : 0.0 for d in d_aic]
    w_aicc_raw = [isfinite(d) ? exp(-0.5 * d) : 0.0 for d in d_aicc]
    sum_wa = sum(w_aic_raw); sum_wc = sum(w_aicc_raw)
    w_aic = sum_wa > 0 ? w_aic_raw ./ sum_wa : fill(NaN, n_mod)
    w_aicc = sum_wc > 0 ? w_aicc_raw ./ sum_wc : fill(NaN, n_mod)

    rows = NamedTuple[
        merge(raw[i], (delta_aic=d_aic[i], akaike_weight=w_aic[i],
                       delta_aicc=d_aicc[i], aicc_weight=w_aicc[i]))
        for i in 1:n_mod
    ]
    if sort_by === :laml
        sort!(rows; by=r -> -r.laml)
    else
        sort!(rows; by=r -> getproperty(r, sort_by))
    end
    ModelComparisonTable(rows, sort_by)
end

function compare_models(pairs::Pair{<:PSMSolution, <:Any}...;
                        names::Union{Nothing, AbstractVector}=nothing,
                        sort_by::Symbol=:aicc)
    sols = PSMSolution[p.first for p in pairs]
    probs = Any[p.second for p in pairs]
    compare_models(sols, probs; names=names, sort_by=sort_by)
end

# ─── Parametric hypothesis test on a fitted smooth ──────────────────

"""
    test_parametric_hypothesis(sol, prob, name::Symbol, h0;
                               grid=nothing, grid_size=50,
                               unconditional=false) -> NamedTuple

Wald chi-square test (Wood 2013) of whether the nonparametric smooth `name`
departs significantly from a parametric null hypothesis `h0`:

- `h0::Function`: exact point-null `H₀: f(x) = h0(x)` over `grid`
- `h0 === :constant`: tests `H₀: f(x) = c` (constant function)
- `h0 === :linear`: tests `H₀: f(x) = c₀ + c₁ x` (linear function)

Computes the curve covariance `V_f = G Σ_k G'` (using `ForwardDiff` Jacobian
`G = ∂f/∂β_k` and conditional `σ² V_β` or unconditional Wood 2016 covariance
when `unconditional=true`), forms the Wald quadratic form `T_W = δ' V_f⁺ δ`
over the non-degenerate eigenmodes of `V_f`, and returns:

    (statistic, df, pvalue, reject_05, grid, fitted, h0_values,
     name, h0_type, unconditional)
"""
function test_parametric_hypothesis(sol::PSMSolution, prob::PSMProblem,
                                    name::Symbol, h0::Union{Function, Symbol};
                                    grid::Union{Nothing, AbstractVector{<:Real}}=nothing,
                                    grid_size::Int=50,
                                    unconditional::Bool=false)
    # Locate approximator and its parameter block
    idx_a = findfirst(a -> a.name === name, prob.approximators)
    idx_a === nothing && throw(ArgumentError(
        "test_parametric_hypothesis: approximator :$name not found in problem"))
    approx = prob.approximators[idx_a]
    off = sum(nparams(prob.approximators[j]) for j in 1:(idx_a - 1); init=0)
    np = nparams(approx)
    rk = (off + 1):(off + np)

    # Full parameter covariance Σ_β
    Σ_full = if unconditional
        smoothing_covariance_correction(sol, prob).covariance
    else
        Matrix{Float64}(StatsAPI.vcov(sol))
    end
    Σ_k = Matrix{Float64}(Σ_full[rk, rk])
    Σ_k = 0.5 .* (Σ_k .+ Σ_k')

    # Evaluation grid
    x_grid = if grid !== nothing
        Float64.(collect(grid))
    else
        dom = band_domain(approx)
        dom === nothing && throw(ArgumentError(
            "test_parametric_hypothesis: approximator :$name has no univariate domain; pass explicit `grid`"))
        collect(range(dom[1], dom[2], length=max(grid_size, 5)))
    end
    ng = length(x_grid)

    beta_all = Vector{Float64}(sol.parameters)
    beta_k = beta_all[rk]
    f_eval = build_evaluator(approx, beta_k)
    f_hat = Float64[f_eval(x) for x in x_grid]

    # Curve Jacobian G[i, j] = ∂f(x_i; β_k)/∂β_{k,j}
    G = Matrix{Float64}(undef, ng, np)
    for i in 1:ng
        xi = x_grid[i]
        G[i, :] .= ForwardDiff.gradient(b -> build_evaluator(approx, b)(xi), beta_k)
    end

    # Curve covariance V_f = G Σ_k G'
    Vf = G * Σ_k * G'
    Vf = 0.5 .* (Vf .+ Vf')
    eig = eigen(Symmetric(Vf))
    lam_max = maximum(eig.values)
    lam_max > 0.0 || throw(ArgumentError(
        "test_parametric_hypothesis: curve covariance V_f is numerically zero"))

    # Wood (2013, Biometrika 100(1):221-228): truncate V_f pseudo-inverse to
    # the block's effective degrees of freedom (r_eff) so heavily penalized
    # high-frequency eigenmodes with tiny variance do not amplify shrinkage bias.
    edf_k = Float64(np)
    conv = sol.convergence
    if conv isa NamedTuple && haskey(conv, :smoothing_state) && conv.smoothing_state !== nothing &&
       haskey(conv, :V_beta) && conv.V_beta !== nothing
        ss = conv.smoothing_state
        if hasproperty(ss, :information)
            F_hat = conv.V_beta * ss.information
            edf_k = clamp(sum(F_hat[i, i] for i in rk), 1.0, Float64(np))
        end
    elseif isfinite(sol.edf) && length(prob.approximators) == 1
        edf_k = clamp(Float64(sol.edf), 1.0, Float64(np))
    end

    # Retain top eigenmodes up to rounded effective rank above numerical floor
    pos_idx = findall(v -> v > 1e-6 * lam_max, eig.values)
    # eig.values are sorted ascending; take the largest r_target modes
    min_req = h0 === :linear ? 3 : (h0 === :constant ? 2 : 1)
    r_target = clamp(round(Int, edf_k), min_req, length(pos_idx))
    keep = pos_idx[(end - r_target + 1):end]
    r = length(keep)
    U_r = eig.vectors[:, keep]             # ng × r
    inv_sqrt_lam = 1.0 ./ sqrt.(eig.values[keep])

    # Whitened transform W = Λ_r^{-1/2} U_r' (r × ng)
    W = Diagonal(inv_sqrt_lam) * U_r'

    local h0_vals::Vector{Float64}
    local stat::Float64
    local df::Int
    local h0_type::Symbol

    if h0 isa Function
        h0_type = :function
        h0_vals = Float64[h0(x) for x in x_grid]
        δ = f_hat .- h0_vals
        z = W * δ
        stat = dot(z, z)
        df = r
    elseif h0 === :constant || h0 === :linear
        h0_type = h0
        X0 = h0 === :constant ? reshape(ones(ng), ng, 1) : hcat(ones(ng), x_grid)
        q0 = size(X0, 2)
        r > q0 || throw(ArgumentError(
            "test_parametric_hypothesis: effective curve rank ($r) must exceed null dimension ($q0)"))
        WX = W * X0
        Wf = W * f_hat
        c_gls = WX \ Wf
        h0_vals = X0 * c_gls
        resid_w = Wf .- WX * c_gls
        stat = dot(resid_w, resid_w)
        df = r - q0
    else
        throw(ArgumentError(
            "test_parametric_hypothesis: h0 must be a Function, :constant, or :linear (got :$h0)"))
    end

    pval = _chisq_ccdf(df, stat)
    (statistic=stat, df=df, pvalue=pval, reject_05=(pval < 0.05),
     grid=x_grid, fitted=f_hat, h0_values=h0_vals,
     name=name, h0_type=h0_type, unconditional=unconditional)
end
