# ─── Problem and solution types ────────────────────────────────────

"""
    TrajectoryObservation(fn)

Wrapper marking `obs_fn` in [`PSMProblem`](@ref) as a full-trajectory
observation operator `fn(U, times, p) -> Matrix` (size `n_times × n_obs`),
where `U` is the `n_times × n_vars` simulated state matrix. Enables exact
interval-aggregated observations such as weekly cumulative incidence differences
`C(t_i) - C(t_{i-1})` while staying fully ForwardDiff-differentiable.
"""
struct TrajectoryObservation{F}
    fn::F
end

"""
    PSMProblem

A partially specified model fitting problem.

# Fields
- `dynamics!`: right-hand side function.
  - Continuous (ODE): `f!(du, u, p, t)` — computes derivatives du/dt
  - Discrete: `f!(u_next, u, p, t)` — computes next state u(t+1)
  In both cases, `p` is a NamedTuple with callable unknown functions and
  known parameters.
- `u0`: initial conditions (vector or function `(params) -> u0`)
- `tspan`: time span `(t0, tf)`
- `approximators`: vector of `AbstractApproximator` for unknown functions
- `data_times`: observation times
- `data_values`: observations matrix `(n_times × n_obs)`
- `data_weights`: weight matrix (same shape as data_values)
- `obs_to_state`: maps observation column j → state variable index
- `known_params`: NamedTuple of known (fixed) parameters
- `likelihood`: likelihood family
- `ode_solver`: ODE/discrete solver (e.g. `Tsit5()` for continuous,
  `FunctionMap()` for discrete, or `nothing` for solvers that don't integrate)
- `ode_kwargs`: additional solver keyword arguments. The ODE and DDE forward
  solves default to `abstol = reltol = 1e-8` and `maxiters = 100_000` integrator
  steps; pass `maxiters` to the constructor if your trajectories need more
- `discrete`: whether this is a discrete-time model
- `delays`: delay values for DDE problems (empty for ODE/discrete)
- `history`: history function `h(p, t)` for DDE problems (nothing for ODE)
- `obs_fn`: optional observation operator `obs_fn(u, p, t)` or [`TrajectoryObservation`](@ref)
- `estimate_u0`: state indices whose initial values are estimated jointly with β
"""
struct PSMProblem{D, U, L<:AbstractLikelihood, S}
    dynamics!::D
    u0::U
    tspan::Tuple{Float64, Float64}
    approximators::Vector{<:AbstractApproximator}
    data_times::Vector{Float64}
    data_values::Matrix{Float64}
    data_weights::Matrix{Float64}
    obs_to_state::Vector{Int}
    known_params::NamedTuple
    likelihood::L
    ode_solver::S
    ode_kwargs::Dict{Symbol, Any}
    discrete::Bool
    delays::Vector{Float64}
    history::Union{Nothing, Function}
    obs_fn::Union{Nothing, Function, TrajectoryObservation}
    estimate_u0::Vector{Int}
end

# Backward-compatible 15-argument positional constructor
PSMProblem(dynamics!, u0, tspan, approximators, data_times, data_values,
           data_weights, obs_to_state, known_params, likelihood, ode_solver,
           ode_kwargs, discrete, delays, history) =
    PSMProblem(dynamics!, u0, tspan, approximators, data_times, data_values,
               data_weights, obs_to_state, known_params, likelihood, ode_solver,
               ode_kwargs, discrete, delays, history, nothing, Int[])

function PSMProblem(dynamics!, u0, tspan,
                    approximators::Vector{<:AbstractApproximator};
                    data_times::AbstractVector,
                    data_values::AbstractMatrix,
                    data_weights::Union{Nothing, AbstractMatrix}=nothing,
                    obs_to_state::Vector{Int}=collect(1:size(data_values, 2)),
                    known_params::NamedTuple=NamedTuple(),
                    likelihood=GaussianLikelihood(),
                    solver=Tsit5(),
                    discrete::Bool=false,
                    delays::Vector{Float64}=Float64[],
                    history::Union{Nothing, Function}=nothing,
                    obs_fn::Union{Nothing, Function, TrajectoryObservation}=nothing,
                    estimate_u0::Union{Bool, AbstractVector{<:Integer}}=false,
                    solver_kwargs...)
    lik = _coerce_likelihood(likelihood)
    n_times = length(data_times)
    n_obs = size(data_values, 2)
    size(data_values, 1) == n_times ||
        throw(ArgumentError("data_values has $(size(data_values, 1)) rows " *
                            "but data_times has $n_times entries"))
    if obs_fn === nothing
        length(obs_to_state) == n_obs ||
            throw(ArgumentError("obs_to_state has $(length(obs_to_state)) " *
                                "entries but data_values has $n_obs columns"))
    end
    est_u0 = if estimate_u0 === false
        Int[]
    elseif estimate_u0 === true
        u0 isa Function && throw(ArgumentError(
            "PSMProblem: estimate_u0=true cannot be combined with a function u0(p)"))
        collect(1:length(u0))
    else
        u0 isa Function && throw(ArgumentError(
            "PSMProblem: estimate_u0 cannot be combined with a function u0(p)"))
        idxs = Int.(collect(estimate_u0))
        allunique(idxs) || throw(ArgumentError("PSMProblem: estimate_u0 indices must be unique"))
        all(i -> 1 <= i <= length(u0), idxs) || throw(ArgumentError(
            "PSMProblem: estimate_u0 indices must lie in 1:$(length(u0)), got $idxs"))
        idxs
    end
    all(τ -> isfinite(τ) && τ > 0.0, delays) ||
        throw(ArgumentError("PSMProblem: all delays must be finite and > 0, got $delays"))
    # A known parameter sharing a name with an approximator would silently
    # SHADOW the fitted function in the merged parameter NamedTuple
    # (later entries win in merge), producing baffling non-fits.
    for approx in approximators
        haskey(known_params, approx.name) &&
            throw(ArgumentError("known_params has an entry :$(approx.name) " *
                                "that collides with an approximator of the " *
                                "same name; rename one of them"))
        !isempty(est_u0) && approx.name === :u0 &&
            throw(ArgumentError("PSMProblem: approximator name :u0 collides with estimate_u0"))
    end
    # Two approximators sharing a name would silently misalign: each still
    # consumes its own β slice, but the parameter NamedTuple keeps only the
    # LAST evaluator, so earlier ones become dead weight.
    approx_names = Symbol[approx.name for approx in approximators]
    if !allunique(approx_names)
        dups = unique(n for n in approx_names if count(==(n), approx_names) > 1)
        throw(ArgumentError("duplicate approximator name(s) $(dups); " *
                            "each approximator needs a unique name — " *
                            "rename the duplicates"))
    end

    w = if data_weights === nothing
        ones(Float64, n_times, n_obs)
    else
        size(data_weights) == size(data_values) ||
            throw(ArgumentError("data_weights has size $(size(data_weights)) " *
                                "but data_values has size $(size(data_values)); " *
                                "they must match"))
        Float64.(data_weights)
    end

    kwargs = Dict{Symbol, Any}(pairs(solver_kwargs)...)

    t_f = Float64.(data_times)
    y_f = Float64.(data_values)

    # Types carrying FIXED reference statistics derived from the data (so
    # far only SingleIndexApproximator) resolve them here, exactly once,
    # and are replaced by resolved copies. Every other type is returned
    # untouched by the hook, so the approximator vector — and its element
    # type — is unchanged for all existing models.
    approximators = if any(a -> a isa SingleIndexApproximator, approximators)
        [_resolve_index_stats(a, u0, t_f, y_f, w, obs_to_state)
         for a in approximators]
    else
        approximators
    end

    PSMProblem(dynamics!, u0,
               (Float64(tspan[1]), Float64(tspan[2])),
               approximators,
               t_f,
               y_f,
               w,
               obs_to_state,
               known_params,
               lik,
               solver,
               kwargs,
               discrete,
               delays,
               history,
               obs_fn,
               est_u0)
end

# ─── Constructors from SciML problem types ───────────────────────

# Wrap an out-of-place dynamics function f(u, p, t) -> result
# into the in-place form f!(out, u, p, t) expected by PSM solvers.
function _wrap_oop(f)
    (out, u, p, t) -> (out .= f(u, p, t); nothing)
end

"""
    PSMProblem(prob::ODEProblem, approximators; kwargs...)

Construct a PSM fitting problem from an `ODEProblem`. The dynamics function,
initial conditions, and time span are extracted from the ODE problem.
Both in-place `f!(du, u, p, t)` and out-of-place `f(u, p, t) -> du`
formulations are supported. Defaults to `solver=Tsit5()`.

# Example
```julia
# In-place
f!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
ode = ODEProblem(f!, [1.0], (0.0, 10.0))

# Out-of-place
f(u, p, t) = [p.r(u[1]) * u[1]]
ode = ODEProblem(f, [1.0], (0.0, 10.0))

psm = PSMProblem(ode, [BSplineApproximator(:r, (0.0, 10.0), 8)];
                 data_times=..., data_values=...)
```
"""
function PSMProblem(prob::SciMLBase.AbstractODEProblem,
                    approximators::Vector{<:AbstractApproximator};
                    solver=Tsit5(),
                    kwargs...)
    dynamics! = if SciMLBase.isinplace(prob)
        prob.f.f
    else
        _wrap_oop(prob.f.f)
    end
    PSMProblem(dynamics!, prob.u0, prob.tspan, approximators;
               solver=solver, discrete=false, kwargs...)
end

"""
    PSMProblem(prob::DiscreteProblem, approximators; kwargs...)

Construct a PSM fitting problem from a `DiscreteProblem`. The dynamics function,
initial conditions, and time span are extracted from the discrete problem.
Both in-place `f!(u_next, u, p, t)` and out-of-place `f(u, p, t) -> u_next`
formulations are supported. Defaults to `solver=nothing` (explicit iteration).

# Example
```julia
# In-place
ricker!(u_next, u, p, t) = (u_next[1] = u[1] * exp(p.g(u[1])); nothing)
disc = DiscreteProblem(ricker!, [20.0], (0.0, 40.0))

# Out-of-place
ricker(u, p, t) = [u[1] * exp(p.g(u[1]))]
disc = DiscreteProblem(ricker, [20.0], (0.0, 40.0))

psm = PSMProblem(disc, [BSplineApproximator(:g, (0.0, 150.0), 10)];
                 data_times=..., data_values=...)
```
"""
function PSMProblem(prob::SciMLBase.AbstractDiscreteProblem,
                    approximators::Vector{<:AbstractApproximator};
                    solver=nothing,
                    kwargs...)
    dynamics! = if SciMLBase.isinplace(prob)
        prob.f.f
    else
        _wrap_oop(prob.f.f)
    end
    PSMProblem(dynamics!, prob.u0, prob.tspan, approximators;
               solver=solver, discrete=true, kwargs...)
end

"""
    PSMSolution

Result of fitting a PSM.

# Fields
- `parameters`: ComponentArray with sections for each approximator
- `objective`: final penalized objective value
- `data_loss`: unpenalized data misfit — the WEIGHTED residual sum of
  squares `Σ wᵢⱼ (yᵢⱼ − ŷᵢⱼ)²`, computed by `weighted_data_loss` over the
  usable cells only (positive weight and finite datum, so one masked-out
  missing observation cannot turn it into `NaN`). This is the convention
  for EVERY likelihood family and every solver: no solver reports a
  deviance here, so for non-Gaussian families `data_loss` is a descriptive
  SSE and NOT the quantity the solver optimised (that is `objective`)
- `edf`: estimated degrees of freedom
- `smoothing_params`: vector of estimated smoothing parameters λ. For the
  penalized-likelihood solvers (LAML, GCVSolver, CollocationLAML,
  GradientMatching) there is ONE entry per penalty BLOCK, in the
  enumeration order of `build_penalty_matrices` (approximators in problem
  order; within an approximator, blocks in the order its `penalty_blocks`
  returns them). With only single-block approximators — the default —
  this is one entry per penalized approximator, as historically; an
  unpenalized approximator (no blocks) contributes no entry, and a
  multi-block approximator contributes one entry per block.
  For `LAML`, `GCVSolver` and `CollocationLAML`, λ̂ and `parameters` are a
  MATCHED PAIR: λ̂ is the λ whose penalty `B(λ̂) = Σ λ̂ₗ Sₗ` produced β̂ — the
  last accepted penalized step was taken under it — and every other
  θ-dependent quantity the solution carries is evaluated at that same λ̂
  (`objective` and `edf` for all three; `convergence.V_beta`,
  `convergence.sigma2` and `convergence.laml` for `LAML`, which alone
  reports them). Both `LAML` and `CollocationLAML` compute one further λ
  proposal after their last coefficient update — `LAML` from the next
  Fellner-Schall step of the IRLS loop, `CollocationLAML` at the end of its
  final continuation level, where the proposal exists only to seed a next
  level that never runs. Those proposals are not reported, because β̂ was
  never fitted under them. For `LAML` on a CONVERGED fit, β̂ is therefore a
  fixed point of the penalized step at λ̂ and refitting reproduces
  `parameters`; on a truncated fit (`convergence.converged == false`) it is
  not — the step contraction may have accepted a partial step — but the
  pairing is still honest. `CollocationLAML` gives the weaker guarantee:
  its inner loop stops on an objective tolerance inside a continuation
  schedule rather than at a stationary point, so β̂ is near but not exactly
  a fixed point at λ̂ (measured: refitting moves β̂ by up to 5e-5 on
  converged fits). The λ̂ it reports is still the one β̂ was fitted under. `RodeoSolver` and `DaltonSolver` also populate this field; their
  pairing has not been audited to the same standard.
- `fitted_values`: predicted values at data times (n_times × n_obs).

  For almost every solver this is the MODEL TRAJECTORY: the dynamics
  integrated from `u0` at the fitted parameters, evaluated at
  `data_times` and projected through `obs_to_state`. `data_loss` is its
  weighted residual sum of squares, so the number measures the fitted
  MODEL against the data and is comparable across those solvers.

  The exception is the solvers that estimate the state trajectory as a
  parameter, jointly with β — `CollocationLAML`, `ODINSolver`,
  `RKHSSolver`, `MagiSolver`, `FGPGMSolver`, and
  `AdaptiveGradientMatching`'s population-MCMC path (`n_samples > 0`).
  Those report that jointly-estimated trajectory, because it is the
  quantity their own objective penalises. It is β-coupled, so unlike the
  data smoother below it does move when the fit moves.

  Do NOT read it as a model-vs-data statistic, and do not treat it as a
  conservative one: the state block can absorb model error, so it is
  optimistic without bound. Measured on a 45-point logistic fixture with
  a right-hand side that CANNOT reproduce the data (`du/dt = -r(u)^2*u`,
  forced non-increasing, against data rising from 1 to 10) — `LAML`,
  which integrates, reports 2399.9513; `CollocationLAML` reports
  1.7043728e-06, understating that same model's own simulated loss by a
  factor of 1.4e9; `RKHSSolver` 0.69481787 (3454x understated);
  `ODINSolver` 110.75749 (21.67x). All three are also blind to `u0`: on
  the same data, moving `u0` from 1.0 to 4.0 changes their `data_loss` by
  a ratio of 1.000 (bitwise unchanged for `ODINSolver` and `RKHSSolver`)
  where `LAML` moves by 341x — the same insensitivity that motivated the
  change described below. Compare these numbers only across solvers of
  this kind, and call `simulate` yourself if you need to score the model.
  (`MagiSolver`, `FGPGMSolver` and the AGM-MCMC path were not measured.)

  A third convention used to ship under this same name and no longer
  does: `GradientMatching` (continuous), `TwoStageSolver`,
  `IntegralMatchingSolver` and `AdaptiveGradientMatching`'s deterministic
  path reported the STAGE-1 DATA SMOOTHER, into which β never entered.
  Their `data_loss` was consequently a function of the data alone —
  measured bitwise equal to `weighted_data_loss(prob, y_smooth)`, and
  identical between two problems differing only in `u0` whose true
  trajectories were a factor of 1237 apart in loss. They now simulate.

  If that reporting simulation fails, the solver warns, falls back to the
  smoother, and sets `convergence.simulation_failed = true`; `data_loss`
  then measures the smoother rather than the model. The flag is defined
  by those four solvers and ONLY those four, where it is `false` on every
  clean fit; on `AdaptiveGradientMatching`'s population-MCMC path it is
  present but always `false`, because that path reports the sampled state
  block and never runs a reporting simulation at all. Every other solver
  omits the key, so guard with `haskey(sol.convergence, :simulation_failed)`
  before reading it off an arbitrary `PSMSolution`. It is independent of
  `n_dynamics_failures`, which counts failure-sentinel substitutions
  DURING the fit.
- `unknown_functions`: Dict of name => callable evaluator
- `convergence`: convergence information. For the iterative optimisers
  (LAML, GCVSolver, CollocationLAML, AdamSolver, MultipleShootingSolver,
  TwoStageSolver, IntegralMatchingSolver, BNGSolver, ODINSolver, RKHSSolver)
  this is a NamedTuple containing at least
  - `converged::Bool` — `true` only when a genuine stopping criterion fired
    (never when the loop merely exhausted its iteration budget),
  - `iterations::Int` — iterations actually performed,
  - `reason::Symbol` — why the loop stopped: `:converged_tol` (tolerance
    criterion met), `:plateau` (objective stagnated over a window / no
    improving step existed), `:maxiters` (budget exhausted without a
    criterion firing), `:early_break` (internal failure such as a
    simulation error or singular linear system), or `:dynamics_failure`
    (see below),
  plus solver-specific additional keys documented on each solver type.

  The gradient-matching family (`GradientMatching`, `TwoStageSolver`,
  `IntegralMatchingSolver`, `ODINSolver`, `RKHSSolver`, `BNGSolver`)
  additionally reports
  - `n_dynamics_failures::Int` — how many evaluation points fell back to
    the large failure sentinel at the FITTED parameters, because the user
    dynamics raised there. `0` on a clean fit. When it is nonzero the fit
    is driven partly by fictitious residuals, so `converged` is forced to
    `false` and `reason` to `:dynamics_failure`, with a warning; when
    EVERY point fails, nothing was fitted and `solve` errors instead of
    returning the initial guess. Measured before this accounting existed:
    a plain ODE whose dynamics threw on 31% of the smoothed trajectory was
    reported as `converged = true, reason = :objective_tol` with a
    function-recovery RMSE 17× worse than the same fixture without the
    throwing band.
  A few solvers put a whole object here instead of a NamedTuple (for
  example `MCMCSolver` stores its `MCMCChains.Chains`). Note there is no
  `extras` field: solver-specific output lives in `convergence`.
"""
struct PSMSolution
    parameters::ComponentArray
    objective::Float64
    data_loss::Float64
    edf::Float64
    smoothing_params::Vector{Float64}
    fitted_values::Matrix{Float64}
    data_values::Matrix{Float64}
    data_times::Vector{Float64}
    unknown_functions::Dict{Symbol, Any}
    convergence::Any
end
