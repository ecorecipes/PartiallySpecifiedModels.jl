module PartiallySpecifiedModelsDecisionAnalysisExt

using PartiallySpecifiedModels
const PSM = PartiallySpecifiedModels
using DecisionAnalysis
using Random

function _coef_column_names(prob::PSM.PSMProblem)
    Symbol[Symbol("$(a.name)_$i") for a in prob.approximators for i in 1:PSM.nparams(a)]
end

function _draws_to_columntable(draws::Vector{Vector{Float64}}, prob::PSM.PSMProblem;
                               summary_fn=nothing)
    n = length(draws)
    n >= 1 || throw(ArgumentError("sample_belief: need at least one parameter draw"))
    cnames = _coef_column_names(prob)
    np = length(cnames)
    pairs_vec = Pair{Symbol, Any}[:beta => draws]
    for j in 1:np
        push!(pairs_vec, cnames[j] => Float64[draws[i][j] for i in 1:n])
    end
    if summary_fn !== nothing
        first_s = summary_fn(draws[1], prob)
        first_s isa NamedTuple || throw(ArgumentError(
            "sample_belief: summary_fn(beta, prob) must return a NamedTuple, got $(typeof(first_s))"))
        skeys = keys(first_s)
        svals = [Vector{Float64}(undef, n) for _ in skeys]
        for (k, v) in enumerate(values(first_s))
            svals[k][1] = Float64(v)
        end
        for i in 2:n
            si = summary_fn(draws[i], prob)
            for (k, v) in enumerate(values(si))
                svals[k][i] = Float64(v)
            end
        end
        for (k, sk) in enumerate(skeys)
            push!(pairs_vec, sk => svals[k])
        end
    end
    NamedTuple(pairs_vec)
end

function PSM._psm_sample_belief(sol::PSM.PSMSolution, prob::PSM.PSMProblem;
                                n::Integer=500, thin::Integer=1,
                                unconditional::Bool=false,
                                rng::AbstractRNG=Random.default_rng(),
                                summary_fn=nothing)
    draws = PSM.posterior_draws(sol, prob; n=n, thin=thin,
                                unconditional=unconditional, rng=rng)
    cols = _draws_to_columntable(draws, prob; summary_fn=summary_fn)
    DecisionAnalysis.SampleBelief(cols)
end

function PSM._psm_sample_belief(bs::PSM.BootstrapResult, prob::PSM.PSMProblem;
                                summary_fn=nothing)
    draws = PSM.posterior_draws(bs)
    cols = _draws_to_columntable(draws, prob; summary_fn=summary_fn)
    DecisionAnalysis.SampleBelief(cols)
end

# ─── DecisionAnalysis adapter verbs for ManagedPSM ───────────────

function DecisionAnalysis.with_parameters(m::PSM.ManagedPSM, θ::NamedTuple)
    beta = if haskey(θ, :beta)
        collect(Float64, θ.beta)
    else
        cnames = _coef_column_names(m.prob)
        Float64[Float64(getproperty(θ, cn)) for cn in cnames]
    end
    PSM.ManagedPSM(m.prob, m.action_fn, m.outcome_fn; beta=beta, action=m.action)
end

function DecisionAnalysis.apply_action(m::PSM.ManagedPSM, action)
    prob_acted = m.action_fn(m.prob, action)
    PSM.ManagedPSM(prob_acted, m.action_fn, m.outcome_fn; beta=m.beta, action=action)
end

function DecisionAnalysis.simulate(m::PSM.ManagedPSM; rng::AbstractRNG=Random.default_rng())
    m.beta === nothing && throw(ArgumentError(
        "DecisionAnalysis.simulate(::ManagedPSM): parameters not bound yet; call with_parameters first."))
    PSM.simulate_psm(m.prob, m.beta)
end

function DecisionAnalysis.outcome(m::PSM.ManagedPSM, traj)
    m.outcome_fn(traj, m.prob, m.action)
end

# DecisionAnalysis.parameter_draws adapter on BootstrapResult (Tables.jl column table)
function DecisionAnalysis.parameter_draws(bs::PSM.BootstrapResult)
    nc = size(bs.coefs, 2)
    NamedTuple{Tuple(Symbol("c$j") for j in 1:nc)}(
        Tuple(Vector{Float64}(bs.coefs[:, j]) for j in 1:nc))
end

end # module
