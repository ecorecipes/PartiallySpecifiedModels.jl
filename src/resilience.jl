# Fitted PSMs as SciML problems, and the resilience of their equilibria with uncertainty, through
# DynamicalResilience.jl (steady states, linearisation, metrics over parameter draws).

"""
    to_ode_problem(prob::PSMProblem, β) -> ODEProblem or DiscreteProblem
    to_ode_problem(prob::PSMProblem, sol::PSMSolution)

The fitted model as a SciML problem, with the unknown functions evaluated at the coefficient
vector `β` (all approximators' coefficients, concatenated in the order of
`prob.approximators`, as in a row of `BootstrapResult.coefs`) or at a solution's
`parameters`.

- A continuous-time PSM gives the in-place `ODEProblem` `prob.dynamics!(du, u, p, t)` over
  `prob.tspan`, with `p` the parameter struct the dynamics see (the unknown-function evaluators
  merged with `known_params`). This is exactly the problem [`simulate`](@ref) integrates.
- A discrete-time PSM (`discrete = true`) gives the in-place `DiscreteProblem` whose map is
  `prob.dynamics!(u_next, u, p, t)`, the convention `simulate` iterates.
- A delay PSM has no finite-dimensional linearisation, so it is an error.

With DynamicalResilience.jl the result gives the model's equilibria and their resilience:
`steady_state(to_ode_problem(prob, sol), u0)` and `linearise`; see also [`metrics_over`](@ref).
"""
function to_ode_problem(prob::PSMProblem, beta::AbstractVector)
    isempty(prob.delays) ||
        throw(ArgumentError("to_ode_problem: this PSM has delays $(prob.delays), so it is a delay " *
                            "differential equation with infinitely many eigenvalues and no " *
                            "finite-dimensional ODEProblem; resilience at an equilibrium needs the " *
                            "characteristic equation of the DDE instead"))
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    # Promote the state eltype when β carries ForwardDiff Duals (the jac=:forwarddiff
    # prediction-Jacobian path differentiates straight through `simulate`): the integrator's
    # in-place buffers take their eltype from u0, and writing a Dual du into a Float64 buffer is
    # a MethodError. A no-op for Float64 β.
    Tel = promote_type(eltype(beta), eltype(u0))
    u0 = Tel === eltype(u0) ? u0 : Tel.(u0)
    if prob.discrete
        return DiscreteProblem{true}(prob.dynamics!, u0, prob.tspan, p)
    end
    return ODEProblem{true}(prob.dynamics!, u0, prob.tspan, p)
end

to_ode_problem(prob::PSMProblem, sol::PSMSolution) = to_ode_problem(prob, collect(Float64, sol.parameters))

"""
    parameter_draws(bs::BootstrapResult) -> Vector{Vector{Float64}}
    parameter_draws(sol::PSMSolution, prob::PSMProblem; thin = 1) -> Vector{Vector{Float64}}

Coefficient vectors `β` describing the uncertainty of a fit, one per draw, each usable with
[`to_ode_problem`](@ref):

- from a [`bootstrap`](@ref) result, one per successful replicate (the rows of `bs.coefs`);
- from an `MCMCSolver` solution, one per posterior sample, pooled over chains and keeping every
  `thin`-th. Only the approximators' coefficients are taken from the chain; `log_σ` and any
  sampled smoothing parameters are dropped, since the dynamics do not depend on them.
"""
parameter_draws(bs::BootstrapResult) = [Vector{Float64}(r) for r in eachrow(bs.coefs)]

function parameter_draws(sol::PSMSolution, prob::PSMProblem; thin::Integer = 1)
    chain = sol.convergence
    chain isa MCMCChains.Chains ||
        throw(ArgumentError("parameter_draws(sol, prob) needs a solution from MCMCSolver, whose " *
                            "`convergence` holds the posterior draws as MCMCChains.Chains; this " *
                            "solution's `convergence` is a $(typeof(chain)). For other solvers use " *
                            "parameter_draws(bootstrap(sol, prob, alg))"))
    thin >= 1 || throw(ArgumentError("thin must be at least 1, got $thin"))
    beta_names = [Symbol("$(a.name)[$i]") for a in prob.approximators for i in 1:nparams(a)]
    available = names(chain, :parameters)
    cols = map(beta_names) do n
        j = findfirst(==(n), available)
        j === nothing && throw(ArgumentError("the chain has no parameter $n: it was not sampled for this problem"))
        j
    end
    samples = Array(chain[available])          # draws × parameters, chains appended
    return [Vector{Float64}(samples[i, cols]) for i in 1:thin:size(samples, 1)]
end

"""
    metrics_over(prob::PSMProblem, draws; u0, kwargs...)
    metrics_over(prob::PSMProblem, bs::BootstrapResult; u0, kwargs...)
    metrics_over(prob::PSMProblem, sol::PSMSolution; u0, thin = 1, kwargs...)

Resilience metrics of the fitted model's equilibrium over the uncertainty of the fit:
DynamicalResilience's `metrics_over` with `build = β -> to_ode_problem(prob, β)`, over coefficient
vectors `draws`, the replicates of a bootstrap, or the posterior draws of an `MCMCSolver` solution
(see [`parameter_draws`](@ref)).

Pass `u0` near the equilibrium of interest (for example the last fitted state). The default,
`prob.u0`, is the initial condition of the data, from which Newton's method may reach a
different equilibrium, such as one where a species is absent. Other keywords (`metrics`,
`quantiles`, `method`, `alg`, …) are those of DynamicalResilience's `metrics_over`; draws without
a converged or stable equilibrium are counted, not dropped.
"""
function DynamicalResilience.metrics_over(prob::PSMProblem, draws; kwargs...)
    return DynamicalResilience.metrics_over(beta -> to_ode_problem(prob, beta), draws; kwargs...)
end

function DynamicalResilience.metrics_over(prob::PSMProblem, bs::BootstrapResult; kwargs...)
    return DynamicalResilience.metrics_over(prob, parameter_draws(bs); kwargs...)
end

function DynamicalResilience.metrics_over(prob::PSMProblem, sol::PSMSolution; thin::Integer = 1, kwargs...)
    return DynamicalResilience.metrics_over(prob, parameter_draws(sol, prob; thin = thin); kwargs...)
end
