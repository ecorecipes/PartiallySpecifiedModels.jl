# ─── DecisionAnalysis.jl interface (weak-dependency bridge) ─────────
#
# Defines `ManagedPSM` and `sample_belief` so fitted PSMs plug directly into
# `DecisionAnalysis.jl` (`outcomes`, `payoff_table`, `decide`, `evpi`, `evppi`,
# `evsi`, `SampleBelief`) when `using DecisionAnalysis` is loaded.

"""
    ManagedPSM(prob::PSMProblem, action_fn, outcome_fn; beta=nothing)

Wraps a [`PSMProblem`](@ref) together with:
- `action_fn(prob, action) -> PSMProblem`: applies a management action to the
  model (e.g. modifying `known_params`, `u0`, or `dynamics!`), and
- `outcome_fn(traj, prob, action) -> NamedTuple` (or `Real`): extracts scalar
  management outcomes from the simulated trajectory `traj = simulate_psm(prob, beta)`.

When `DecisionAnalysis.jl` is loaded (`using DecisionAnalysis`), `ManagedPSM`
implements its four adapter verbs (`with_parameters`, `apply_action`,
`simulate`, `outcome`) so `DecisionAnalysis.outcomes(managed, belief, actions)`
works directly for EVPI / EVPPI / decision-rule analysis.
"""
struct ManagedPSM{P<:PSMProblem, A, O}
    prob::P
    action_fn::A
    outcome_fn::O
    beta::Union{Nothing, Vector{Float64}}
    action::Any
end

function ManagedPSM(prob::PSMProblem, action_fn, outcome_fn;
                    beta::Union{Nothing,AbstractVector}=nothing,
                    action=nothing)
    b = beta === nothing ? nothing : collect(Float64, beta)
    ManagedPSM{typeof(prob), typeof(action_fn), typeof(outcome_fn)}(
        prob, action_fn, outcome_fn, b, action)
end

# Hook implemented by PartiallySpecifiedModelsDecisionAnalysisExt
function _psm_sample_belief(source, prob; kwargs...)
    throw(ArgumentError(
        "sample_belief requires DecisionAnalysis.jl to be loaded (`using DecisionAnalysis`)."))
end

"""
    sample_belief(sol::PSMSolution, prob::PSMProblem; n=500, thin=1,
                  unconditional=false, rng=Random.default_rng(),
                  summary_fn=nothing)
    sample_belief(bs::BootstrapResult, prob::PSMProblem; summary_fn=nothing)

Convert parameter uncertainty from a fitted PSM (`PSMSolution` or
`BootstrapResult`) into a `DecisionAnalysis.SampleBelief` (requires
`using DecisionAnalysis`).

Each draw stores the full parameter vector under `:beta` (plus each
approximator's named coefficients `Symbol("\$(name)_\$i")` or any custom
interpretable summaries returned as a `NamedTuple` by `summary_fn(beta, prob)`),
allowing direct EVPPI computation over interpretable ecological quantities.
"""
sample_belief(sol::PSMSolution, prob::PSMProblem; kwargs...) =
    _psm_sample_belief(sol, prob; kwargs...)
sample_belief(bs::BootstrapResult, prob::PSMProblem; kwargs...) =
    _psm_sample_belief(bs, prob; kwargs...)
