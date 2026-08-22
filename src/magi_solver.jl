# ─── MAGI Solver ─────────────────────────────────────────────────────
#
# Manifold-constrained Gaussian process inference (Yang, Wong & Kou 2021,
# PNAS 118(15)).
#
# Each state component x_d(·) has a Gaussian-process prior.  On a fine grid
# I = (t₁,…,t_n) the GP induces a joint Gaussian over (x_d, ẋ_d):
#
#   x_d            ~ N(μ_d, C_d),                       C_d = K(I,I)
#   ẋ_d | x_d      ~ N( m_d (x_d − μ_d),  K*_d ),
#       m_d   = 'K C_d⁻¹                  (GP-implied derivative map)
#       K*_d  = ''K − 'K C_d⁻¹ ('K)ᵀ     (conditional derivative cov)
#
# The "manifold constraint" W requires the GP-implied derivative to match
# the ODE vector field, f_d(x(I), θ, I).  Conditioning on W = 0 gives the
# MAGI posterior over (θ, X(I)) (states are sampled jointly with θ, NOT
# marginalized):
#
#   log p(θ, X | Y) = log π(θ)
#       − ½ Σ_d (x_d−μ_d)ᵀ C_d⁻¹ (x_d−μ_d)                 [GP prior]
#       − ½ Σ_d (f_d − m_d(x_d−μ_d))ᵀ K*_d⁻¹ (f_d − m_d(x_d−μ_d))  [W]
#       − ½ Σ_obs (y − x)ᵀ Σ_obs⁻¹ (y − x)                 [data]
#
# C_d⁻¹, m_d and K*_d⁻¹ depend only on the grid and the GP hyperparameters,
# so they are precomputed once; each NUTS step is matrix–vector products
# plus one ODE-RHS evaluation per grid point.
#
# Reference: Yang, Wong & Kou (2021) PNAS 118(15)

using LinearAlgebra

# ─── Matérn-3/2 GP and its derivative covariances ───────────────────

"""
    _matern32_gp_matrices(t, ℓ, σ²) -> (C, K1, K2)

For the Matérn-3/2 kernel k(r)=σ²(1+√3 r/ℓ)e^{−√3 r/ℓ} on grid `t`, return
- `C[i,j]  = k(t_i,t_j)`                      Cov(x_i, x_j)
- `K1[i,j] = ∂k/∂t_i = Cov(ẋ_i, x_j)`         ('K)
- `K2[i,j] = ∂²k/∂t_i∂t_j = Cov(ẋ_i, ẋ_j)`    (''K)
"""
function _matern32_gp_matrices(t::Vector{Float64}, ℓ::Float64, σ2::Float64)
    n = length(t)
    s3 = sqrt(3.0)
    C  = zeros(n, n); K1 = zeros(n, n); K2 = zeros(n, n)
    for i in 1:n, j in 1:n
        d = t[i] - t[j]
        ad = abs(d)
        e = exp(-s3 * ad / ℓ)
        C[i, j]  = σ2 * (1 + s3 * ad / ℓ) * e
        K1[i, j] = -3σ2 / ℓ^2 * d * e
        K2[i, j] = 3σ2 / ℓ^2 * (1 - s3 * ad / ℓ) * e
    end
    C, K1, K2
end

"""
    _magi_fit_hyperparams(td, y, obs_var) -> (ℓ, σ²)

Estimate GP length-scale and signal variance for one observed component by
maximizing the Matérn-3/2 marginal likelihood on the data, with a fixed
observation noise `obs_var`. Falls back to heuristics on failure.
"""
function _magi_fit_hyperparams(td::Vector{Float64}, y::Vector{Float64}, obs_var::Float64)
    trange = maximum(td) - minimum(td)
    ybar = Statistics.mean(y)
    yc = y .- ybar
    σ2_0 = max(Statistics.var(y), 1e-3)
    nll = function (lp)
        ℓ = exp(lp[1]); σ2 = exp(lp[2])
        C, _, _ = _matern32_gp_matrices(td, ℓ, σ2)
        M = Symmetric(C + (obs_var + 1e-8) * I)
        F = cholesky(M, check=false)
        issuccess(F) || return 1e10
        0.5 * dot(yc, F \ yc) + sum(log, diag(F.U))
    end
    ℓ, σ2 = trange / 4, σ2_0
    try
        res = Optim.optimize(nll, [log(trange / 4), log(σ2_0)],
                             Optim.NelderMead(), Optim.Options(iterations=150))
        lp = Optim.minimizer(res)
        ℓ = clamp(exp(lp[1]), trange / 50, trange * 2)
        σ2 = clamp(exp(lp[2]), 1e-4, 1e6)
    catch
    end
    ℓ, σ2
end

_magi_linear_interp(xs, ys, xq) = begin
    if xq <= xs[1]; return ys[1]; end
    if xq >= xs[end]; return ys[end]; end
    j = searchsortedlast(xs, xq)
    j = clamp(j, 1, length(xs) - 1)
    w = (xq - xs[j]) / (xs[j+1] - xs[j])
    (1 - w) * ys[j] + w * ys[j+1]
end

# ─── MAGI log-density (samples θ and the state grid X jointly) ───────

"""
    MAGILogDensity

LogDensityProblems interface for the MAGI posterior over the stacked vector
`[θ ; vec(X)]`, where `X` is the n_grid × n_vars matrix of state values on
the discretization grid. The GP matrices are precomputed per component.
"""
struct MAGILogDensity{P <: PSMProblem}
    prob::P
    grid_times::Vector{Float64}
    obs_indices::Vector{Int}
    obs_var::Vector{Float64}           # per state component
    prior_scale::Float64
    n_vars::Int
    n_grid::Int
    n_params::Int
    Cinv::Vector{Matrix{Float64}}      # per component
    mmap::Vector{Matrix{Float64}}      # 'K C⁻¹
    Kstar_inv::Vector{Matrix{Float64}}
    μ::Vector{Vector{Float64}}         # GP prior mean per component
end

LogDensityProblems.dimension(ld::MAGILogDensity) = ld.n_params + ld.n_vars * ld.n_grid
LogDensityProblems.capabilities(::Type{<:MAGILogDensity}) = LogDensityProblems.LogDensityOrder{0}()

function _magi_build_uf(prob, theta)
    offset = 0
    uf_entries = Pair{Symbol, Any}[]
    for approx in prob.approximators
        np = nparams(approx)
        params_k = theta[offset+1:offset+np]
        offset += np
        push!(uf_entries, approx.name => build_evaluator(approx, params_k))
    end
    merge(NamedTuple(uf_entries), prob.known_params)
end

function _magi_logposterior(ld::MAGILogDensity, v::AbstractVector{T}) where T
    prob = ld.prob
    n_vars = ld.n_vars; n_grid = ld.n_grid; np = ld.n_params
    theta = @view v[1:np]
    Xflat = @view v[np+1:end]
    # X[i, d] = state d at grid point i
    X = reshape(Xflat, n_grid, n_vars)

    p_nt = _magi_build_uf(prob, theta)

    # ODE vector field at every grid point: F[i, d] = f_d(X[i,:], θ, t_i)
    F = Matrix{T}(undef, n_grid, n_vars)
    du = Vector{T}(undef, n_vars)
    uvec = Vector{T}(undef, n_vars)
    for i in 1:n_grid
        @inbounds for d in 1:n_vars
            uvec[d] = X[i, d]
        end
        prob.dynamics!(du, uvec, p_nt, ld.grid_times[i])
        @inbounds for d in 1:n_vars
            F[i, d] = du[d]
        end
    end

    lp = zero(T)
    for d in 1:n_vars
        xd = @view X[:, d]
        xc = xd .- ld.μ[d]
        # GP prior: −½ xcᵀ C⁻¹ xc
        lp += -T(0.5) * dot(xc, ld.Cinv[d] * xc)
        # Manifold constraint: −½ (F − m·xc)ᵀ K*⁻¹ (F − m·xc)
        resid = @view(F[:, d]) .- ld.mmap[d] * xc
        lp += -T(0.5) * dot(resid, ld.Kstar_inv[d] * resid)
    end

    # Data likelihood (Gaussian; per-state noise variance). Weight-zero
    # entries encode masked points and are excluded, matching the other
    # solvers' data_weights semantics.
    for i in 1:size(prob.data_values, 1)
        gi = ld.obs_indices[i]
        for j in 1:size(prob.data_values, 2)
            y = prob.data_values[i, j]
            w = prob.data_weights[i, j]
            _usable(y, w) || continue
            sk = prob.obs_to_state[j]
            r = T(y) - X[gi, sk]
            lp += -T(w) / (2 * ld.obs_var[sk]) * r^2
        end
    end

    # Parameter prior (smoothing penalty + broad ridge)
    offset = 0
    for approx in prob.approximators
        nk = nparams(approx)
        pk = @view theta[offset+1:offset+nk]
        offset += nk
        if approx isa BSplineApproximator || approx isa ShapeConstrainedBSplineApproximator ||
           approx isa GPApproximator || approx isa COMONetApproximator ||
           approx isa SPDEApproximator || approx isa ShapeConstrainedSPDEApproximator
            S = penalty_matrix(approx)
            if S !== nothing
                lp += -T(0.5) / T(ld.prior_scale) * dot(pk, T.(S) * pk)
            end
        end
        lp += -T(0.5) / T(100.0 * ld.prior_scale) * dot(pk, pk)
    end
    lp
end

function LogDensityProblems.logdensity(ld::MAGILogDensity, v::AbstractVector)
    T = eltype(v)
    val = _magi_logposterior(ld, v)
    isfinite(ForwardDiff.value(val)) ? val : T(-1e10)
end

# ─── Main solve method ──────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::MagiSolver)

Fit a partially specified model using MAGI (MAnifold-constrained Gaussian
process Inference; Yang, Wong & Kou 2021). A Matérn-3/2 GP prior is placed
on each state; the GP-implied derivative is constrained to the ODE vector
field through the conditional covariance `K* = ''K − 'K C⁻¹ ('K)ᵀ`, and the
states `X(I)` are sampled jointly with the unknown-function parameters θ via
NUTS.

Gaussian likelihoods only: the manifold-constrained posterior's data term
is a Gaussian quadratic form on the state grid, so non-Gaussian
`prob.likelihood` errors at entry. Observation noise is per state
component: `sigma` (explicit SDs) takes priority over `obs_var` (shared
variance; specifying both errors), which takes priority over
auto-estimation. `data_weights` scale each observation's contribution;
weight-zero entries are excluded like NaN.

# References
- Yang, Wong & Kou (2021), "Inference of dynamic systems from noisy and
  sparse data via manifold-constrained Gaussian processes", PNAS 118(15).

# Returns
`PSMSolution`; `convergence.chains` holds the posterior samples of θ.
`objective` is minus the mean log-posterior over the retained draws (the
only scalar objective a sampler has; negated for the package-wide
lower-is-better convention) and `data_loss` is the usual mask-aware
`weighted_data_loss` at the posterior-mean trajectory. `convergence`
also carries `converged`/`reason`; NUTS runs a fixed budget with no
stopping criterion, so these are always `false`/`:maxiters` — judge the
sampler by the R̂ and ESS of `convergence.chains`.

Masked observations (`NaN` value and/or zero weight) are supported: they
are dropped from the data term, the GP hyperparameter fit and the state
initialization.
"""
function SciMLBase.solve(prob::PSMProblem, alg::MagiSolver)
    _validate_problem(prob, "MagiSolver"; require_continuous=true)
    # The manifold-constrained GP posterior assumes Gaussian observation
    # noise (the data term is a Gaussian quadratic form on the state grid),
    # so refuse other families rather than silently fitting Gaussian.
    prob.likelihood isa Gaussian ||
        error("MagiSolver supports Gaussian likelihoods only (the " *
              "manifold-constrained GP posterior assumes Gaussian " *
              "observation noise); got $(typeof(prob.likelihood)). " *
              "Use MCMCSolver or LAML for other likelihood families.")
    verbose = alg.verbose
    n_vars = length(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)

    t0, tf = prob.tspan
    grid_times = collect(range(t0, tf, length=alg.n_gridpoints))
    n_grid = alg.n_gridpoints

    obs_indices = Int[]
    for td in prob.data_times
        _, idx = findmin(abs.(grid_times .- td))
        push!(obs_indices, idx)
    end

    # ── Observation noise variance (one per state component) ──
    # Priority: `sigma` (explicit per-state SDs) > `obs_var` (shared
    # variance) > auto-estimation. If unspecified, estimate per observed
    # column with a local-linear residual estimator and average across
    # columns. A fixed default is scale-blind: with data in units ≫ 0.1 it
    # made the data term overwhelm the GP and manifold terms, silently
    # distorting the posterior.
    if alg.sigma !== nothing
        length(alg.sigma) == n_vars ||
            error("MagiSolver: sigma must supply one observation SD per " *
                  "state component ($n_vars); got length $(length(alg.sigma)).")
        all(>(0.0), alg.sigma) ||
            error("MagiSolver: sigma entries must be positive.")
        alg.obs_var === nothing ||
            error("MagiSolver: specify observation noise via sigma " *
                  "(per-state SDs) or obs_var (shared variance), not both.")
        obs_var_vec = alg.sigma .^ 2
    elseif alg.obs_var !== nothing
        obs_var_vec = fill(alg.obs_var, n_vars)
    else
        # Gasser–Sroka–Jennen local-linear residual estimator: for each
        # interior point, ε̂ᵢ = yᵢ − aᵢy₍ᵢ₋₁₎ − bᵢy₍ᵢ₊₁₎ with the linear-
        # interpolation weights, σ̂² = Σ cᵢ²ε̂ᵢ²/(n−2), cᵢ² = 1/(aᵢ²+bᵢ²+1).
        # Valid for NON-equispaced times (reduces to Var(Δ²y)/6 when
        # equispaced, which the previous estimator assumed).
        ests = Float64[]
        t_all = Float64.(prob.data_times)
        for j in 1:size(prob.data_values, 2)
            y = Float64.(prob.data_values[:, j])
            # weight-0 points are masked: exclude them like NaN
            keep = _usable.(y, prob.data_weights[:, j])
            yv = y[keep]; tv = t_all[keep]
            n = length(yv)
            n >= 4 || continue
            acc = 0.0
            for i in 2:(n - 1)
                den = tv[i+1] - tv[i-1]
                den > 0 || continue
                a = (tv[i+1] - tv[i]) / den
                b = (tv[i] - tv[i-1]) / den
                ε = yv[i] - a * yv[i-1] - b * yv[i+1]
                acc += ε^2 / (a^2 + b^2 + 1)
            end
            push!(ests, max(acc / (n - 2), 1e-10))
        end
        obs_var = isempty(ests) ? 0.01 : Statistics.mean(ests)
        verbose && println("MAGI: estimated obs_var = $(round(obs_var, sigdigits=3))")
        obs_var_vec = fill(obs_var, n_vars)
    end

    # ── GP hyperparameters and precomputed matrices per component ──
    data_times = Float64.(prob.data_times)
    Cinv = Vector{Matrix{Float64}}(undef, n_vars)
    mmap = Vector{Matrix{Float64}}(undef, n_vars)
    Kstar_inv = Vector{Matrix{Float64}}(undef, n_vars)
    μ = Vector{Vector{Float64}}(undef, n_vars)
    Xinit = zeros(n_grid, n_vars)

    u0v = Float64.(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    for d in 1:n_vars
        # observation column mapping to this state, if any
        ocol = findfirst(j -> prob.obs_to_state[j] == d, 1:size(prob.data_values, 2))
        if ocol !== nothing
            yobs = Float64.(prob.data_values[:, ocol])
            # weight-0 points are masked: keep them out of the GP
            # hyperparameter fit and the state initialization too
            keep = _usable.(yobs, prob.data_weights[:, ocol])
            td = data_times[keep]; yv = yobs[keep]
            ℓ, σ2 = _magi_fit_hyperparams(td, yv, obs_var_vec[d])
            ybar = Statistics.mean(yv)
            for i in 1:n_grid
                Xinit[i, d] = _magi_linear_interp(td, yv, grid_times[i])
            end
        else
            ℓ = (tf - t0) / 4
            σ2 = 1.0
            ybar = u0v[d]
            Xinit[:, d] .= u0v[d]
        end
        C, K1, K2 = _matern32_gp_matrices(grid_times, ℓ, σ2)
        jitter = 1e-7 * σ2
        Cf = cholesky(Symmetric(C + jitter * I))
        Ci = inv(Cf)
        m = K1 * Ci
        Kstar = Symmetric(K2 - K1 * Ci * K1' + jitter * I)
        Cinv[d] = Matrix(Ci)
        mmap[d] = m
        Kstar_inv[d] = Matrix(inv(cholesky(Kstar)))
        μ[d] = fill(ybar, n_grid)
        if verbose
            println("MAGI: component $d  ℓ=$(round(ℓ,sigdigits=3)) σ²=$(round(σ2,sigdigits=3))")
        end
    end

    n_params = sum(nparams(a) for a in prob.approximators)
    theta0 = Float64[]
    for approx in prob.approximators
        append!(theta0, initial_params(approx))
    end
    v0 = vcat(theta0, vec(Xinit))

    ld = MAGILogDensity(prob, grid_times, obs_indices, obs_var_vec, alg.prior_scale,
                        n_vars, n_grid, n_params, Cinv, mmap, Kstar_inv, μ)

    ld_ad = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ld)

    # MAP pre-optimization (gradient-based) for a good NUTS init.
    if alg.preoptimize
        verbose && println("MAGI: MAP pre-optimization (L-BFGS)...")
        negf = x -> -LogDensityProblems.logdensity(ld, x)
        negg! = (g, x) -> begin
            _, grad = LogDensityProblems.logdensity_and_gradient(ld_ad, x)
            g .= .-grad
            g
        end
        try
            res = Optim.optimize(negf, negg!, v0, Optim.LBFGS(),
                                 Optim.Options(iterations=60, show_trace=false))
            if isfinite(Optim.minimum(res)) && Optim.minimum(res) < negf(v0)
                v0 = Optim.minimizer(res)
            end
        catch e
            verbose && println("MAGI: pre-optimization failed ($(typeof(e)))")
        end
    end

    n_total = alg.n_warmup + alg.n_samples
    nuts = AdvancedHMC.NUTS(alg.target_accept)
    verbose && println("MAGI: NUTS sampling ($n_total iterations, $(alg.n_warmup) warmup)...")
    # n_adapts must equal n_warmup so all discarded draws are adaptation
    # and all retained draws come from the frozen kernel.
    chain_raw = AbstractMCMC.sample(ld_ad, nuts, n_total;
                                    n_adapts=alg.n_warmup,
                                    initial_params=v0, progress=verbose)

    n_keep = alg.n_samples
    sample_matrix = zeros(n_keep, n_params)       # θ samples only
    state_mean = zeros(n_grid, n_vars)
    # Mean log-posterior over the retained draws. This is the only scalar
    # objective MAGI actually has: the sampler targets the joint posterior
    # over (θ, x) and there is no penalized-likelihood value to report.
    # Reported NEGATED below, so `sol.objective` keeps the package-wide
    # "lower is better" convention (MCMCSolver reports −log p likewise).
    mean_logpost = 0.0
    for i in 1:n_keep
        z = chain_raw[alg.n_warmup + i].z.θ
        sample_matrix[i, :] = z[1:n_params]
        state_mean .+= reshape(z[n_params+1:end], n_grid, n_vars) ./ n_keep
        mean_logpost += _magi_logposterior(ld, z) / n_keep
    end

    param_names = String[]
    for approx in prob.approximators, j in 1:nparams(approx)
        push!(param_names, "$(approx.name)[$j]")
    end
    chains = MCMCChains.Chains(sample_matrix, Symbol.(param_names))

    # Posterior mean of the pooled draws (a point summary; not a MAP)
    posterior_mean_beta = vec(Statistics.mean(sample_matrix, dims=1))
    map_beta = posterior_mean_beta  # retained name for downstream code
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = map_beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => map_beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    # Fitted values at observation times from the posterior-mean state grid.
    n_obs = size(prob.data_values, 2)
    pred = zeros(length(prob.data_times), max(n_obs, 1))
    for i in 1:length(prob.data_times)
        gi = obs_indices[i]
        for j in 1:n_obs
            pred[i, j] = state_mean[gi, prob.obs_to_state[j]]
        end
    end

    verbose && println("MAGI: sampling complete.")

    # Reported loss: the SAME weighted, mask-aware residual sum of squares
    # every other solver reports (`weighted_data_loss`), evaluated at the
    # posterior-mean state trajectory. A hard-coded 0.0 here claimed a
    # perfect fit for every MAGI run, which no diagnostic could distinguish
    # from a genuinely perfect one.
    data_loss = weighted_data_loss(prob, pred)

    # NUTS has no stopping criterion: it runs the requested budget and
    # stops. Reporting `converged = true` would be a fabrication, so this
    # is loop exhaustion — `:maxiters`, the same taxonomy the iterative
    # solvers use. Assess the sampler with the R̂/ESS diagnostics of
    # `convergence.chains`, not with this flag.
    PSMSolution(params, -mean_logpost, data_loss, Float64(n_params), Float64[],
                pred, Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (method=:magi, chains=chains, state_mean=state_mean,
                 mean_logposterior=mean_logpost,
                 converged=false, reason=:maxiters,
                 iterations=n_total))
end
