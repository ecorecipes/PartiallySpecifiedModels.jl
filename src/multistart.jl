# ─── Multistart wrapper ───────────────────────────────────────────

"""
    MultiStart(alg::LAML; n_starts=8, rng_seed=1, ode_maxiters=10_000)

Fit the same problem `n_starts` times with `alg`, each time from a different
random initialization of the problem's randomly initialized approximators,
and return the best fit. `LAML` is the only supported inner solver. Every
ODE solve in every start is capped at `ode_maxiters` integrator steps unless
the problem sets its own `maxiters` (see Cost below).

Start 1 is the problem as given. When the problem sets `maxiters`, or with
`ode_maxiters = nothing`, `n_starts = 1` therefore reproduces
`solve(prob, alg)` bit for bit apart from the extra `convergence.multistart`
record. Starts 2 to `n_starts` rebuild every randomly initialized
approximator (`NeuralApproximator`, `KANApproximator`, `COMONetApproximator`)
with a new `rng_seed`. The seeds are drawn in sequence from
`Random.Xoshiro(rng_seed)`, one per random approximator per start, so the
same `rng_seed` gives the same starts, and increasing `n_starts` keeps the
earlier starts and only adds new ones. Deterministically initialized
approximators (B-splines, GPs, SPDEs and their shape-constrained and
wrapped forms) start from the same values every time. A custom
approximator (see the extension protocol) is treated as deterministic. A
problem with no randomly initialized approximator raises an
`ArgumentError`, because every start would be identical. Each start fits
a deep copy of the approximators, so `prob` itself is never mutated (GP
approximators adapt their kernel during a fit).

# Why
`convergence.converged` cannot tell a poor local minimum from the global
one: it tests whether the fit stopped changing, not where it stopped.
Measured on an unpenalized `NeuralApproximator` growth-rate fixture
(`du/dt = r(u)·u`, `Dense(1,4,tanh) → Dense(4,1)`, 21 points of
`exp(0.1t)` plus noise of amplitude 0.01, `LAML(maxiters=100)`), 15 of 41
`rng_seed`s (1–40 and 42) ended at a local minimum with data loss 3.8 to 64,
and those fits reported `converged = true, reason = :converged_tol`. The
rest reached the noise floor (data loss ≈ 7.7e-4) or used up `maxiters`
close to it. No local convergence test we measured separates the two
groups, so the only remedy is to restart from somewhere else.

# Selection
A start is skipped if its solve throws a numerical error or returns a
non-finite `objective`. Program errors (`MethodError`, `BoundsError`, …)
and `ArgumentError`s are rethrown, because they would recur at every start.
Among the remaining starts:
- if every one has a finite LAML criterion (`convergence.laml`, a MAXIMIZED
  quantity), the start with the largest `laml` wins (`criterion = :laml`);
- otherwise, for example when the problem has no penalized term and `laml`
  is `NaN`, the start with the smallest `objective` wins
  (`criterion = :objective`).
Ties go to the earliest start. If every start is skipped, `solve` raises an
error that lists why each one failed.

# Cost
Each start is one full `solve(prob, alg)`, and the cost of a start varies a
great deal. On the fixture above, the first 3 LAML iterations took
0.3–2.8 s for most restart seeds, but 279 s for one and more than 20
minutes for another. In those starts the trial steps moved the network to
where the ODE became expensive to integrate, and every finite-difference
Jacobian column paid for it. With the default integrator step limit, 8-start
fits took 117–312 s, except one that took 53 minutes, and a 4-start fit
that met the 20-minute seed was abandoned after an hour.

So `MultiStart` caps every ODE solve at `ode_maxiters` integrator steps
(default 10_000; the fixture's converged trajectories need about 50). The
cap is added to the problem's solver keywords only when they do not already
set `maxiters`: a `PSMProblem(...; maxiters=...)` always wins, and
`ode_maxiters = nothing` turns the cap off. A problem whose trajectories
genuinely need more steps than the cap will see every solve fail; raise
`ode_maxiters` or set `maxiters` on the problem. The cap applied is recorded
as `convergence.multistart.ode_maxiters`. With the cap, over `rng_seed` 1:10 from
the poor seed, the median MultiStart solve took 8.6 s (range 7.5–14.3) for
`n_starts = 4` and 23.3 s (12.6–45.5) for `n_starts = 8`. All 20 returned
fits had data loss below 0.01, 19 of them within 2% of the noise floor and
one (`n_starts = 4`) at 1.5e-3. (Timings are from a machine with load
average around 30 on 10 cores.) The cap is not neutral: a trial solve that
hits it fails, which changes the path of some starts. One start moved from
the noise floor to the poor minimum, and another from data loss 49 to
0.019.

# Result
The chosen start's `PSMSolution`, with its `convergence` NamedTuple
extended by
`multistart = (n_starts, seeds, criterion, values, reasons, converged, chosen, ode_maxiters)`:
- `seeds[k]` holds the `rng_seed` of each randomly initialized approximator
  at start `k`, in problem order. Start 1 has the problem's own seeds, which
  may be `nothing`. To reproduce start `k`, pass these seeds to the
  approximators' constructors;
- `values[k]` is start `k`'s selection criterion (`NaN` when it was skipped);
- `reasons[k]` is its `convergence.reason`, or `:nonfinite_objective`, or
  `Symbol("threw_", <exception type>)`;
- `converged[k]` is its `convergence.converged` (`false` if it threw);
- `chosen` is the index of the returned start;
- `ode_maxiters` is the integrator step cap every start ran with: the
  problem's own `maxiters` if it set one, otherwise `ms.ode_maxiters`
  (`nothing` when uncapped).

Every other `convergence` key belongs to the chosen start. Warnings logged
during the solve can come from any start, so read the returned keys rather
than the log. Only the approximators' initializations differ between
starts, not their architecture or grids, so the solution can be used with
the original `prob` in `confidence_band`, `predict` and the other
post-fit functions. `bootstrap(sol, prob, ms)` runs the whole multistart
for every replicate.

# Example
```julia
sol = solve(prob, MultiStart(LAML(maxiters=100); n_starts=8))
sol.convergence.multistart.chosen   # which start was returned
sol.convergence.multistart.values   # each start's selection criterion
```
"""
struct MultiStart{A}
    alg::A
    n_starts::Int
    rng_seed::Int
    ode_maxiters::Union{Nothing,Int}
end

function MultiStart(alg::LAML; n_starts::Int=8, rng_seed::Integer=1,
                    ode_maxiters::Union{Nothing,Integer}=10_000)
    n_starts >= 1 ||
        throw(ArgumentError("MultiStart: n_starts must be at least 1 (got $n_starts)"))
    ode_maxiters === nothing || ode_maxiters >= 1 ||
        throw(ArgumentError("MultiStart: ode_maxiters must be at least 1 or nothing (got $ode_maxiters)"))
    MultiStart{LAML}(alg, n_starts, Int(rng_seed),
                     ode_maxiters === nothing ? nothing : Int(ode_maxiters))
end

# Rebuild a randomly initialized approximator with a new seed, or return
# `nothing` for a deterministically initialized one. Each method calls the
# type's own constructor, so derived state (KAN's cached initial parameters)
# is regenerated rather than patched.
_reseeded(::AbstractApproximator, ::Int) = nothing
_reseeded(a::NeuralApproximator, seed::Int) =
    NeuralApproximator(a.name, a.model; penalty_weight=a.penalty_weight,
                       domain=a.domain, rng_seed=seed)
_reseeded(a::COMONetApproximator, seed::Int) =
    COMONetApproximator(a.name, a.domain, a.hidden_sizes, a.constraint;
                        penalty_weight=a.penalty_weight,
                        activation=a.activation, rng_seed=seed)
_reseeded(a::KANApproximator, seed::Int) =
    KANApproximator(a.name, a.model; input_domains=a.input_domains,
                    penalty=a.penalty, nullspace_penalty=a.nullspace_penalty,
                    rng_seed=seed)

_is_random_init(a::AbstractApproximator) = _reseeded(a, 1) !== nothing

# Same problem, different approximators. The positional constructor is used
# so nothing the keyword constructor resolves from the data (index
# statistics) is recomputed.
_with_approximators(prob::PSMProblem, approximators, ode_kwargs) =
    PSMProblem(prob.dynamics!, prob.u0, prob.tspan, approximators,
               prob.data_times, prob.data_values, prob.data_weights,
               prob.obs_to_state, prob.known_params, prob.likelihood,
               prob.ode_solver, ode_kwargs, prob.discrete, prob.delays,
               prob.history)

# The problem's solver keywords with the MultiStart step cap added, unless
# the problem already sets `maxiters`. A copy, so `prob` is never mutated.
function _capped_ode_kwargs(prob::PSMProblem, ms::MultiStart)
    kw = copy(prob.ode_kwargs)
    (ms.ode_maxiters === nothing || haskey(kw, :maxiters)) && return kw
    kw[:maxiters] = ms.ode_maxiters
    kw
end

"""
    solve(prob::PSMProblem, ms::MultiStart)

Fit `prob` from `ms.n_starts` initializations and return the best fit. See
[`MultiStart`](@ref).
"""
function SciMLBase.solve(prob::PSMProblem, ms::MultiStart{LAML})
    random_idx = findall(_is_random_init, prob.approximators)
    isempty(random_idx) && throw(ArgumentError(
        "MultiStart: the problem has no randomly initialized approximator " *
        "(NeuralApproximator, KANApproximator or COMONetApproximator), so " *
        "every start would be identical. Use solve(prob, alg) directly."))

    n = ms.n_starts
    rng = Random.Xoshiro(ms.rng_seed)
    seeds = Vector{Vector{Union{Nothing,Int}}}(undef, n)
    seeds[1] = Union{Nothing,Int}[prob.approximators[i].rng_seed for i in random_idx]
    for k in 2:n
        seeds[k] = Union{Nothing,Int}[rand(rng, 1:Int(typemax(Int32)))
                                      for _ in random_idx]
    end

    ode_kwargs = _capped_ode_kwargs(prob, ms)
    sols = Vector{Union{Nothing,PSMSolution}}(nothing, n)
    reasons = Vector{Symbol}(undef, n)
    converged = falses(n)
    first_error = nothing
    for k in 1:n
        approximators = deepcopy(prob.approximators)
        if k > 1
            # `map` rather than assignment in place: a reseeded parametric
            # approximator (KAN) need not match a concrete element type.
            approximators = map(enumerate(approximators)) do (i, a)
                j = findfirst(==(i), random_idx)
                j === nothing ? a : _reseeded(a, seeds[k][j])
            end
        end
        sol = try
            solve(_with_approximators(prob, approximators, ode_kwargs), ms.alg)
        catch e
            (_is_program_error(e) || e isa ArgumentError) && rethrow()
            first_error === nothing && (first_error = e)
            reasons[k] = Symbol("threw_", nameof(typeof(e)))
            continue
        end
        reasons[k] = sol.convergence.reason
        converged[k] = sol.convergence.converged
        if isfinite(sol.objective)
            sols[k] = sol
        else
            reasons[k] = :nonfinite_objective
        end
    end

    ok = findall(!isnothing, sols)
    if isempty(ok)
        msg = "MultiStart: all $n starts failed (reasons: $(join(reasons, ", ")))."
        first_error === nothing ||
            (msg *= " First exception: " * sprint(showerror, first_error))
        error(msg)
    end

    use_laml = all(k -> isfinite(sols[k].convergence.laml), ok)
    criterion = use_laml ? :laml : :objective
    values = fill(NaN, n)
    for k in ok
        values[k] = use_laml ? sols[k].convergence.laml : sols[k].objective
    end
    chosen = first(ok)
    for k in ok
        better = use_laml ? values[k] > values[chosen] : values[k] < values[chosen]
        better && (chosen = k)
    end

    best = sols[chosen]
    record = (n_starts=n, seeds=seeds, criterion=criterion, values=values,
              reasons=reasons, converged=collect(converged), chosen=chosen,
              ode_maxiters=get(ode_kwargs, :maxiters, nothing))
    PSMSolution(best.parameters, best.objective, best.data_loss, best.edf,
                best.smoothing_params, best.fitted_values, best.data_values,
                best.data_times, best.unknown_functions,
                merge(best.convergence, (multistart=record,)))
end
