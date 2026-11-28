# Fitted PSMs as SciML problems, posterior parameter draws across UQ-capable
# solvers, and resilience of equilibria with uncertainty via
# DynamicalResilience.jl & ResilienceCore.jl.

"""
    to_ode_problem(prob::PSMProblem, β) -> ODEProblem or DiscreteProblem
    to_ode_problem(prob::PSMProblem, sol::PSMSolution)

The fitted model as a SciML problem, with the unknown functions evaluated at the
coefficient vector `β` (all approximators' coefficients concatenated in the
order of `prob.approximators`) or at `sol.parameters`.

- Continuous-time PSM → in-place `ODEProblem` `prob.dynamics!(du, u, p, t)`
- Discrete-time PSM (`discrete = true`) → in-place `DiscreteProblem` (also see
  [`to_discrete_problem`](@ref))
- Delay PSM → throws `ArgumentError` (infinite-dimensional spectrum)
"""
function to_ode_problem(prob::PSMProblem, beta::AbstractVector)
    isempty(prob.delays) ||
        throw(ArgumentError("to_ode_problem: this PSM has delays $(prob.delays), so it is a delay " *
                            "differential equation with infinitely many eigenvalues and no " *
                            "finite-dimensional ODEProblem; resilience at an equilibrium needs the " *
                            "characteristic equation of the DDE instead"))
    p = build_param_struct(prob, beta)
    u0 = !isempty(prob.estimate_u0) ? _resolve_u0(prob, beta) :
         (prob.u0 isa Function ? prob.u0(p) : prob.u0)
    Tel = promote_type(eltype(beta), eltype(u0))
    u0 = Tel === eltype(u0) ? u0 : Tel.(u0)
    if prob.discrete
        return DiscreteProblem{true}(prob.dynamics!, u0, prob.tspan, p)
    end
    return ODEProblem{true}(prob.dynamics!, u0, prob.tspan, p)
end

to_ode_problem(prob::PSMProblem, sol::PSMSolution) =
    to_ode_problem(prob, Vector{Float64}(sol.parameters))

"""
    to_discrete_problem(prob::PSMProblem, β) -> DiscreteProblem
    to_discrete_problem(prob::PSMProblem, sol::PSMSolution)

Construct the SciML `DiscreteProblem` for a discrete-time PSM (`prob.discrete == true`),
matching `ResilienceCore.jl` and `DynamicalResilience.jl` discrete-map conventions.
Throws `ArgumentError` if `prob` is continuous-time or has delays.
"""
function to_discrete_problem(prob::PSMProblem, beta::AbstractVector)
    prob.discrete || throw(ArgumentError(
        "to_discrete_problem: requires a discrete-time PSMProblem (discrete=true); " *
        "use to_ode_problem for continuous-time ODE models."))
    to_ode_problem(prob, beta)
end

to_discrete_problem(prob::PSMProblem, sol::PSMSolution) =
    to_discrete_problem(prob, Vector{Float64}(sol.parameters))

function _extract_chain_beta_draws(chain::MCMCChains.Chains, prob::PSMProblem, thin::Integer)
    thin >= 1 || throw(ArgumentError("thin must be at least 1, got $thin"))
    beta_names = [Symbol("$(a.name)[$i]") for a in prob.approximators for i in 1:nparams(a)]
    available = names(chain, :parameters)
    cols = map(beta_names) do nm
        j = findfirst(==(nm), available)
        j === nothing && throw(ArgumentError(
            "the chain has no parameter $nm: it was not sampled for this problem"))
        j
    end
    samples = Array(chain[available])
    [Vector{Float64}(samples[i, cols]) for i in 1:thin:size(samples, 1)]
end

function _mvn_draws(mu::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, n::Int, rng::AbstractRNG)
    mu_v = Vector{Float64}(mu)
    d = length(mu_v)
    S = Symmetric(0.5 .* (Matrix{Float64}(Σ) .+ Matrix{Float64}(Σ)'))
    eig = eigen(S)
    vals = max.(eig.values, 0.0)
    L = eig.vectors * Diagonal(sqrt.(vals))
    [mu_v .+ L * randn(rng, d) for _ in 1:n]
end

"""
    posterior_draws(bs::BootstrapResult) -> Vector{Vector{Float64}}
    posterior_draws(sol::PSMSolution, prob::PSMProblem; n=500, thin=1,
                    unconditional=false, rng=Random.default_rng()) -> Vector{Vector{Float64}}

Return coefficient vectors `β` representing posterior or sampling uncertainty
of a fitted PSM, each usable with [`to_ode_problem`](@ref) or
[`to_discrete_problem`](@ref):

- `BootstrapResult`: one draw per successful bootstrap replicate (`bs.coefs`).
- `MCMCSolver` / `PseudoMarginalSolver` (`MCMCChains.Chains`): pooled chain draws
  thinned by `thin`.
- `MagiSolver` / `FGPGMSolver` / `AdaptiveGradientMatching` (MCMC): pooled draws
  from `sol.convergence.chains` (or `beta_samples`) thinned by `thin`.
- `BNGSolver`: ensemble draws from `sol.convergence.beta_samples`.
- `EnsembleKalmanSolver`: ensemble particles from `sol.convergence.ensemble`.
- `ABCSolver`: `n` resampled draws from `sol.convergence.particles` weighted by
  `sol.convergence.weights`.
- `VariationalSolver`: `n` draws from `N(μ, diag(σ²))`.
- `LAML` / `CollocationLAML` / `GCVSolver` (covariance-producing fits): `n`
  Gaussian posterior draws from `N(β̂, Σ)`, using `σ² * V_beta` (conditional)
  or `smoothing_covariance_correction(sol, prob).covariance` when
  `unconditional=true`.
"""
posterior_draws(bs::BootstrapResult) = [Vector{Float64}(r) for r in eachrow(bs.coefs)]

function posterior_draws(sol::PSMSolution, prob::PSMProblem;
                         n::Integer = 500,
                         thin::Integer = 1,
                         unconditional::Bool = false,
                         rng::AbstractRNG = Random.default_rng())
    n >= 1 || throw(ArgumentError("posterior_draws: n must be at least 1, got $n"))
    thin >= 1 || throw(ArgumentError("posterior_draws: thin must be at least 1, got $thin"))
    conv = sol.convergence

    # 1. Direct MCMCChains.Chains (MCMCSolver, PseudoMarginalSolver)
    if conv isa MCMCChains.Chains
        return _extract_chain_beta_draws(conv, prob, thin)
    end

    if conv isa NamedTuple || conv isa AbstractDict
        _has(k) = haskey(conv, k)
        _get(k) = conv isa NamedTuple ? getproperty(conv, k) : conv[k]

        # 2. NamedTuple/Dict with MCMCChains.Chains (MagiSolver, FGPGMSolver, AGM-MCMC)
        if _has(:chains) && _get(:chains) isa MCMCChains.Chains
            return _extract_chain_beta_draws(_get(:chains), prob, thin)
        end
        # 3. NamedTuple/Dict with beta_samples (BNGSolver, etc.)
        if _has(:beta_samples) && _get(:beta_samples) !== nothing
            bsamp = _get(:beta_samples)
            if bsamp isa AbstractMatrix
                return [Vector{Float64}(bsamp[i, :]) for i in 1:thin:size(bsamp, 1)]
            elseif bsamp isa AbstractVector
                return [Vector{Float64}(bsamp[i]) for i in 1:thin:length(bsamp)]
            end
        end
        # 4. ABCSolver particles + weights
        if _has(:particles) && _has(:weights)
            parts = _get(:particles)
            wts = Float64.(_get(:weights))
            cumw = cumsum(wts)
            cumw ./= cumw[end]
            out = Vector{Vector{Float64}}(undef, n)
            for i in 1:n
                idx = clamp(searchsortedfirst(cumw, rand(rng)), 1, length(parts))
                out[i] = Vector{Float64}(parts[idx])
            end
            return out
        end
        # 5. EnsembleKalmanSolver ensemble matrix (n_beta × J)
        if _has(:ensemble) && _get(:ensemble) isa AbstractMatrix
            E = _get(:ensemble)
            return [Vector{Float64}(E[:, j]) for j in 1:thin:size(E, 2)]
        end
        # 6. VariationalSolver mean-field Gaussian (posterior_mean/posterior_std or mu/sigma)
        if _has(:posterior_mean) && _has(:posterior_std) &&
           _get(:posterior_mean) isa AbstractVector && _get(:posterior_std) isa AbstractVector
            mu = Float64.(_get(:posterior_mean))
            sig = Float64.(_get(:posterior_std))
            return [mu .+ sig .* randn(rng, length(mu)) for _ in 1:n]
        elseif _has(:mu) && _has(:sigma) &&
               _get(:mu) isa AbstractVector && _get(:sigma) isa AbstractVector
            mu = Float64.(_get(:mu))
            sig = Float64.(_get(:sigma))
            return [mu .+ sig .* randn(rng, length(mu)) for _ in 1:n]
        end
        # 7. Penalized-likelihood posterior covariance (LAML, CollocationLAML, GCV)
        if _has(:V_beta) && _get(:V_beta) !== nothing
            mu = Vector{Float64}(sol.parameters)
            Σ = if unconditional
                smoothing_covariance_correction(sol, prob).covariance
            else
                σ² = _has(:sigma2) && _get(:sigma2) !== nothing ?
                     Float64(_get(:sigma2)) : 1.0
                Matrix{Float64}(σ² .* _get(:V_beta))
            end
            return _mvn_draws(mu, Σ, Int(n), rng)
        end
    end

    throw(ArgumentError(
        "posterior_draws(sol, prob): this solution's convergence field " *
        "($(typeof(conv))) carries neither posterior samples nor a parameter " *
        "covariance matrix V_beta. Use posterior_draws(bootstrap(sol, prob, alg))."))
end

# ─── Direct DynamicalResilience & ResilienceCore dispatches ────────

"""
    DynamicalResilience.steady_state(prob::PSMProblem, sol::PSMSolution, u0; kwargs...)

Find an equilibrium of the fitted PSM `to_ode_problem(prob, sol)` starting from `u0`.
"""
function DynamicalResilience.steady_state(prob::PSMProblem, sol::PSMSolution, u0; kwargs...)
    DynamicalResilience.steady_state(to_ode_problem(prob, sol), u0; kwargs...)
end

"""
    ResilienceCore.linearise(prob::PSMProblem, sol::PSMSolution, u_star; kwargs...)

Linearise the fitted PSM `to_ode_problem(prob, sol)` around equilibrium `u_star`.
"""
function ResilienceCore.linearise(prob::PSMProblem, sol::PSMSolution, u_star; kwargs...)
    ResilienceCore.linearise(to_ode_problem(prob, sol), u_star; kwargs...)
end

"""
    DynamicalResilience.reproduction_number(prob::PSMProblem, sol::PSMSolution, dfe; kwargs...)

Compute the next-generation basic reproduction number R₀ of the fitted PSM at
disease-free equilibrium `dfe`.
"""
function DynamicalResilience.reproduction_number(prob::PSMProblem, sol::PSMSolution, dfe; kwargs...)
    DynamicalResilience.reproduction_number(to_ode_problem(prob, sol), dfe; kwargs...)
end

"""
    metrics_over(prob::PSMProblem, draws; u0, kwargs...)
    metrics_over(prob::PSMProblem, bs::BootstrapResult; u0, kwargs...)
    metrics_over(prob::PSMProblem, sol::PSMSolution; u0, n=500, thin=1,
                 unconditional=false, rng=Random.default_rng(), kwargs...)

Resilience metrics of the fitted model's equilibrium over parameter uncertainty
(`draws`, `BootstrapResult`, or any UQ-capable `PSMSolution` via
[`posterior_draws`](@ref)).
"""
function DynamicalResilience.metrics_over(prob::PSMProblem, draws; kwargs...)
    return DynamicalResilience.metrics_over(beta -> to_ode_problem(prob, beta), draws; kwargs...)
end

function DynamicalResilience.metrics_over(prob::PSMProblem, bs::BootstrapResult; kwargs...)
    return DynamicalResilience.metrics_over(prob, posterior_draws(bs); kwargs...)
end

function DynamicalResilience.metrics_over(prob::PSMProblem, sol::PSMSolution;
                                          n::Integer = 500,
                                          thin::Integer = 1,
                                          unconditional::Bool = false,
                                          rng::AbstractRNG = Random.default_rng(),
                                          kwargs...)
    draws = posterior_draws(sol, prob; n = n, thin = thin,
                            unconditional = unconditional, rng = rng)
    return DynamicalResilience.metrics_over(prob, draws; kwargs...)
end
