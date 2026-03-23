# ─── DDE (Delay Differential Equation) solver support ─────────────────
#
# Extends PartiallySpecifiedModels with DDE support via DelayDiffEq.jl.
# A DDE has dynamics f!(du, u, h, p, t) where h is the history function:
#   h(p, t-τ)       → full state vector at time t-τ
#   h(p, t-τ; idxs=i) → scalar state variable i at time t-τ
#
# Usage:
#   PSMProblem(dde_dynamics!, u0, tspan, approximators;
#              delays=[1.0], history=h, ...)
# or:
#   PSMProblem(DDEProblem(f!, u0, h, tspan; constant_lags=[τ]), approximators; ...)

using DelayDiffEq: DDEProblem, DDEFunction, MethodOfSteps

# ─── DDE simulation ──────────────────────────────────────────────

"""
    simulate_dde(prob::PSMProblem, beta::AbstractVector)

Simulate a DDE model with given parameters. Uses DelayDiffEq.jl with
`MethodOfSteps` wrapping the configured ODE solver.

The user's dynamics function must have signature `f!(du, u, h, p, t)`
where `h` is the DelayDiffEq history function.
"""
function simulate_dde(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0

    # Wrap dynamics: the DDEProblem passes its own `params` positional arg,
    # but we use our built `p` NamedTuple instead.
    dde_rhs! = let p_local = p, dyn! = prob.dynamics!
        (du, u, h, _params, t) -> dyn!(du, u, h, p_local, t)
    end

    # History function: user-provided or constant at u0
    h_func = _build_history(prob, p, u0)

    dde_prob = DDEProblem(dde_rhs!, u0, h_func, prob.tspan;
                          constant_lags=prob.delays)

    inner_solver = prob.ode_solver !== nothing ? prob.ode_solver : Tsit5()
    dde_solver = MethodOfSteps(inner_solver)

    solve_kwargs = Dict{Symbol, Any}(
        :saveat => prob.data_times,
        :abstol => get(prob.ode_kwargs, :abstol, 1e-8),
        :reltol => get(prob.ode_kwargs, :reltol, 1e-8),
        :maxiters => get(prob.ode_kwargs, :maxiters, 1_000_000),
        :verbose => get(prob.ode_kwargs, :verbose, false),
    )
    merge!(solve_kwargs, prob.ode_kwargs)

    sol = DelayDiffEq.solve(dde_prob, dde_solver; solve_kwargs...)

    if sol.retcode != SciMLBase.ReturnCode.Success &&
       sol.retcode != SciMLBase.ReturnCode.Default &&
       sol.retcode != SciMLBase.ReturnCode.Terminated
        error("DDE solve failed: $(sol.retcode)")
    end

    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    pred = zeros(eltype(beta), n_times, n_obs)

    for i in 1:n_times
        u_i = sol.u[i]
        for j in 1:n_obs
            pred[i, j] = u_i[prob.obs_to_state[j]]
        end
    end
    pred
end

"""
    _build_history(prob, p, u0)

Build a history function compatible with DelayDiffEq.jl.
If `prob.history` is provided, wraps it to accept the `idxs` keyword.
Otherwise returns a constant history at `u0`.
"""
function _build_history(prob::PSMProblem, p, u0)
    if prob.history !== nothing
        let hist = prob.history, p_local = p
            (params, t; idxs=nothing) -> begin
                h_val = hist(p_local, t)
                idxs === nothing ? h_val : h_val[idxs]
            end
        end
    else
        let u0_local = u0
            (params, t; idxs=nothing) -> idxs === nothing ? u0_local : u0_local[idxs]
        end
    end
end

# ─── ForwardDiff-compatible DDE solve (for AdamSolver) ───────────

"""
    adam_solve_dde(prob::PSMProblem, beta)

ForwardDiff-compatible DDE solve for Adam loss functions. Uses
`build_autodiff_param_struct` and `DDEFunction{true, FullSpecialize}`
to support Dual-typed parameters. Returns the DDE solution object.
"""
function adam_solve_dde(prob::PSMProblem, beta)
    p = build_autodiff_param_struct(prob, beta)
    T = eltype(beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    u0_T = T.(u0)

    dde_rhs! = let p_local = p, dyn! = prob.dynamics!
        (du, u, h, _params, t) -> dyn!(du, u, h, p_local, t)
    end

    h_func = _build_history(prob, p, u0_T)

    dde_fn = DDEFunction{true, SciMLBase.FullSpecialize}(dde_rhs!)
    dde_prob = DDEProblem(dde_fn, u0_T, h_func, prob.tspan;
                          constant_lags=prob.delays)

    inner_solver = prob.ode_solver !== nothing ? prob.ode_solver : Tsit5()
    dde_solver = MethodOfSteps(inner_solver)

    DelayDiffEq.solve(dde_prob, dde_solver;
                      saveat=prob.data_times,
                      abstol=1e-7, reltol=1e-7,
                      maxiters=10_000)
end

"""
    adam_solve_dde_final(prob::PSMProblem, p_opt, u0)

DDE solve for the final prediction in Adam solver's solution building.
Uses Float64 parameters (no Dual types).
"""
function adam_solve_dde_final(prob::PSMProblem, p_opt, u0)
    dde_rhs! = let p_local = p_opt, dyn! = prob.dynamics!
        (du, u, h, _params, t) -> dyn!(du, u, h, p_local, t)
    end

    h_func = _build_history(prob, p_opt, u0)

    dde_prob = DDEProblem(dde_rhs!, u0, h_func, prob.tspan;
                          constant_lags=prob.delays)

    inner_solver = prob.ode_solver !== nothing ? prob.ode_solver : Tsit5()
    dde_solver = MethodOfSteps(inner_solver)

    DelayDiffEq.solve(dde_prob, dde_solver;
                      saveat=prob.data_times,
                      abstol=1e-7, reltol=1e-7,
                      maxiters=10_000)
end

# ─── DDEProblem constructor for PSMProblem ───────────────────────

# Wrap an out-of-place DDE function f(u, h, p, t) -> du
# into in-place form f!(du, u, h, p, t).
function _wrap_oop_dde(f)
    (du, u, h, p, t) -> (du .= f(u, h, p, t); nothing)
end

"""
    PSMProblem(prob::DDEProblem, approximators; kwargs...)

Construct a PSM fitting problem from a `DDEProblem`. The dynamics function,
initial conditions, time span, delays, and history are extracted from the
DDE problem. Both in-place `f!(du, u, h, p, t)` and out-of-place
`f(u, h, p, t) -> du` formulations are supported.

# Example
```julia
function dde!(du, u, h, p, t)
    u_delayed = h(p, t - 1.0)
    du[1] = -p.f(u_delayed[1])
end
h(p, t) = [1.0]

dde_prob = DDEProblem(dde!, [1.0], h, (0.0, 10.0); constant_lags=[1.0])
psm = PSMProblem(dde_prob, [BSplineApproximator(:f, (0.0, 2.0), 8)];
                 data_times=..., data_values=...)
```
"""
function PSMProblem(prob::SciMLBase.AbstractDDEProblem,
                    approximators::Vector{<:AbstractApproximator};
                    solver=Tsit5(),
                    kwargs...)
    dynamics! = if SciMLBase.isinplace(prob)
        prob.f.f
    else
        _wrap_oop_dde(prob.f.f)
    end

    # Extract constant lags from the DDEProblem
    delays = if hasproperty(prob, :constant_lags) && prob.constant_lags !== nothing
        Float64.(collect(prob.constant_lags))
    else
        Float64[]
    end

    # Extract history function
    history = prob.h

    PSMProblem(dynamics!, prob.u0, prob.tspan, approximators;
               solver=solver, discrete=false,
               delays=delays, history=history,
               kwargs...)
end
