# Main solver: IRLS loop with LAML smoothing parameter estimation
#
# Algorithm (per iteration):
# 1. Evaluate model f(β) and compute Jacobian J by finite differences
# 2. Form pseudodata z = y - f + J*β
# 3. Solve penalized LS: min ||W^½(z - Jβ)||² + β'S^λβ
# 4. Step contraction (backtrack to ensure decrease)
# 5. Re-estimate smoothing parameters λ via LAML (Fellner-Schall + Newton)
# 6. Repeat until convergence

using LinearAlgebra: Diagonal, dot, tr, Symmetric, eigvals, cholesky, norm, eigen
using ForwardDiff   # jac=:forwarddiff prediction-Jacobian path

# ─── Input validation ─────────────────────────────────────────────

"""
    _warn_unanchored_index(prob, solver_name)

Warn once per solve when a `SingleIndexApproximator` with `anchor === nothing`
is fitted by a smoothing-parameter-estimating solver.

The data term is EXACTLY flat along `a -> c*a` (the index standardization
divides it out), so the inner ridge that makes free mode well posed for
flat-objective optimizers is minimized by `‖a‖ -> 0` under a criterion that
also drives λ. The fit then collapses to a near-zero loading vector with a
meaningless inner λ and an ill-conditioned penalized Hessian, while the data
loss looks normal — a silent-wrong path this warning exists to break.
"""
function _warn_unanchored_index(prob::PSMProblem, solver_name::String)
    for a in prob.approximators
        if a isa SingleIndexApproximator && a.anchor === nothing
            @warn "$solver_name: SingleIndexApproximator :$(a.name) has " *
                  "anchor=nothing. The data term is scale-invariant in the " *
                  "loadings, so the inner ridge is minimized by ‖a‖ → 0 and " *
                  "this fit will degenerate (near-zero loadings, meaningless " *
                  "inner λ) while the data loss still looks reasonable. Use " *
                  "the default anchor=<index> with $solver_name; free mode is " *
                  "for flat-objective solvers (AdamSolver, DerivativeFreeSolver, " *
                  "MCMCSolver)." maxlog=1
        end
    end
end

"""
    _validate_problem(prob, solver_name; require_continuous=false,
                      reject_delays=false)

Common input validation for all solve methods. Checks data dimensions,
approximator configuration, and observation mapping consistency.

`require_continuous=true` rejects discrete-time maps.

`reject_delays=true` rejects DDE problems, for the solvers that evaluate
the dynamics through the 4-argument ODE signature `f!(du, u, p, t)`. A
PSM DDE's `dynamics!` has the 5-argument signature `f!(du, u, h, p, t)`
(see `dde_solver.jl`), so those calls raise a `MethodError` at every
point. Measured before this guard existed: `GradientMatching`,
`TwoStageSolver`, `IntegralMatchingSolver`, `ODINSolver` and
`RKHSSolver` swallowed that `MethodError` into their `1e6` failure
sentinel and returned the initial guess as though it were a fit, while
seven other solvers surfaced a raw `MethodError` from deep inside their
inner loops. Supplying the delayed history to a gradient-matching
objective is a feature (the smoother would have to yield a history
callable, including for unobserved states), not a fix.
"""
function _validate_problem(prob::PSMProblem, solver_name::String;
                           require_continuous::Bool=false,
                           reject_delays::Bool=false,
                           delay_reason::String=
                               "It evaluates the dynamics through the " *
                               "4-argument ODE signature f!(du, u, p, t) on " *
                               "a smoothed trajectory.")
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    n_times == 0 && error("$solver_name: data_times is empty")
    issorted(prob.data_times) ||
        error("$solver_name: data_times must be sorted increasing (the " *
              "Kalman-filter and profiling paths assume monotone " *
              "observation indices)")
    size(prob.data_values, 1) != n_times &&
        error("$solver_name: data_values has $(size(prob.data_values, 1)) rows " *
              "but data_times has $n_times entries")
    size(prob.data_weights, 1) != n_times &&
        error("$solver_name: data_weights row count does not match data_times")
    length(prob.obs_to_state) != n_obs &&
        error("$solver_name: obs_to_state has $(length(prob.obs_to_state)) entries " *
              "but data_values has $n_obs columns")
    isempty(prob.approximators) &&
        error("$solver_name: no approximators specified")
    # Resolve u0 — may be a function of parameters (e.g., copepod model)
    if !(prob.u0 isa Function)
        any(s -> s < 1 || s > length(prob.u0), prob.obs_to_state) &&
            error("$solver_name: obs_to_state contains indices outside " *
                  "range 1:$(length(prob.u0))")
    end
    if reject_delays && !isempty(prob.delays)
        error("$solver_name does not support DDE problems. $delay_reason " *
              "A PSM DDE's dynamics function has the 5-argument signature " *
              "f!(du, u, h, p, t), so the delayed history can never be " *
              "supplied and the model would be fitted as though it had no " *
              "delays. Use LAML, GCVSolver, AdamSolver, VariationalSolver, " *
              "MCMCSolver, DerivativeFreeSolver or ProfileLikelihoodSolver " *
              "for DDEs.")
    end
    if require_continuous && prob.discrete
        error("$solver_name does not support discrete-time models. " *
              "The probabilistic ODE solver is designed for continuous ODEs. " *
              "Use LAML, GradientMatching, AdamSolver, BNGSolver, GCVSolver, " *
              "TwoStageSolver, DerivativeFreeSolver, or ABCSolver instead.")
    end
    nothing
end

# ─── Parameter layout ─────────────────────────────────────────────

"""Total number of parameters across all approximators."""
function n_total_params(prob::PSMProblem)
    sum(nparams(a) for a in prob.approximators)
end

"""Build initial parameter vector by concatenating approximator initial params."""
function build_initial_params(prob::PSMProblem)
    vcat([initial_params(a) for a in prob.approximators]...)
end

"""
    build_param_struct(prob, beta)

Build the parameter NamedTuple that the dynamics function receives.
Contains callable unknown functions and known parameters.
"""
function build_param_struct(prob::PSMProblem, beta::AbstractVector)
    offset = 0
    uf_entries = Pair{Symbol, Any}[]

    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np

        # Per-type construction lives in build_evaluator
        # (approximator_interface.jl) — the extension point for custom
        # approximator types.
        push!(uf_entries, approx.name => build_evaluator(approx, params_k))
    end

    # Merge unknown function evaluators with known params
    uf_nt = NamedTuple(uf_entries)
    merge(uf_nt, prob.known_params)
end

"""
    weighted_data_loss(prob, pred) -> Float64

The reported `PSMSolution.data_loss`: the weighted residual sum of
squares `Σ wᵢⱼ (yᵢⱼ − ŷᵢⱼ)²` taken over the USABLE data cells only.

A cell is usable when its weight is positive and its datum is finite.
Skipping the rest matters twice over: an unweighted sum silently ignores
`data_weights` (so a masked or down-weighted observation still counts),
and `0 * NaN = NaN` would otherwise let a single masked-out missing
observation turn the whole reported loss into `NaN`.
"""
function weighted_data_loss(prob::PSMProblem, pred::AbstractMatrix)
    dl = 0.0
    for k in eachindex(prob.data_values)
        # Same predicate as `usable_cell` / `_usable` — see `_usable`.
        _usable(prob.data_values[k], prob.data_weights[k]) || continue
        dl += prob.data_weights[k] * (pred[k] - prob.data_values[k])^2
    end
    dl
end

"""
    usable_cell(prob, i, j) -> Bool

Whether data cell `(i, j)` participates in objectives. The single
package-wide convention: positive weight AND FINITE value — the same
predicate as `_usable(y, w)` and `weighted_data_loss`. Solvers that
accumulate a data term cell-by-cell should gate on this rather than
relying on the weight alone — `0 * NaN = NaN`, so a zero weight does NOT
neutralize a NaN datum.
"""
@inline usable_cell(prob::PSMProblem, i::Int, j::Int) =
    _usable(prob.data_values[i, j], prob.data_weights[i, j])

"""
    usable_rows(prob, j) -> Vector{Int}

Row indices of column `j` that carry a usable observation. Used by the
solvers that pre-smooth a data column (`_smoothing_spline`, `CubicSpline`,
GP smoothers): those fits solve normal equations in which a single NaN
makes EVERY coefficient NaN, so the masked rows must be dropped from the
fit rather than merely down-weighted afterwards.
"""
usable_rows(prob::PSMProblem, j::Int) =
    [i for i in axes(prob.data_values, 1) if usable_cell(prob, i, j)]

"""
    n_usable(prob) -> Int

Total number of usable data cells — the sample size any denominator over
observations (σ̂²'s `n − edf`, a GCV `n`, a marginal-likelihood
normalizer) must use. Equals `length(prob.data_values)` for complete data.
"""
n_usable(prob::PSMProblem) =
    count(k -> _usable(prob.data_values[k], prob.data_weights[k]),
          eachindex(prob.data_values))

"""
    _reject_masked_data(prob, solver_name)

Raise a clear error if any data cell is masked, for solvers that do NOT
yet support masking.

These are the probabilistic-numerics and ensemble solvers whose data term
lives inside a Kalman/particle recursion (`probsolve.jl`'s `basic_loglik`
and Fenrir `condition!`, `_dalton_joint_evidence`'s `assimilate_data!`,
`_pm_loglik_hat`, and the EKI innovation update). Honoring the mask there
means threading a weight matrix through those signatures AND skipping the
filter update — not just the density term — for masked cells; that is a
structural change, deliberately out of scope here.

Failing loudly is the point. Left unguarded these solvers do not merely
report NaN: `dalton_solver.jl`'s `!isfinite(ll) && return 1e10` flattens
the objective to a constant, `pseudo_marginal_solver.jl` rejects every MH
proposal (α = 0), and the EKI update turns every ensemble member NaN on
the first iteration — each of which returns the initialization while
looking like an ordinary, converged fit.
"""
function _reject_masked_data(prob::PSMProblem, solver_name::String)
    # `n_usable` is the package-wide count, so the rejection predicate can
    # never drift from the predicate the masking solvers actually apply.
    n_masked = length(prob.data_values) - n_usable(prob)
    n_masked == 0 && return nothing
    error("$solver_name does not support masked observations, and $n_masked " *
          "of $(length(prob.data_values)) data cells are masked (weight 0 " *
          "or non-finite value). Its likelihood is evaluated inside a Kalman / " *
          "particle recursion that has no per-cell mask, so masked cells " *
          "would silently corrupt the filter state rather than being " *
          "skipped. Use LAML, GCVSolver, CollocationLAML, GradientMatching, " *
          "TwoStageSolver, BNGSolver, ODINSolver, RKHSSolver, " *
          "IntegralMatchingSolver, AdamSolver, MultipleShootingSolver, " *
          "DerivativeFreeSolver, MCMCSolver, MagiSolver, VariationalSolver, " *
          "ABCSolver or ProfileLikelihoodSolver for masked data, or drop " *
          "the masked rows from data_times/data_values before calling.")
end

"""
    _is_program_error(e)

Classify an exception caught inside a solver objective. Programming
errors (wrong method signature, out-of-bounds indexing, type bugs) must
be rethrown so they surface immediately; only genuinely numerical
failures (domain errors, singular factorizations, solver blow-ups) may
be converted into a large-but-finite penalty for the optimizer.

`InexactError` is special-cased: converting a NaN/Inf (e.g. inside
DataInterpolations when a diverged trajectory reaches a spline) is a
numerical failure the optimizer must survive, while converting a finite
value (`Int(3.7)`) is a genuine bug.

`FieldError` (Julia ≥ 1.12) is classified as a program error: it is what
`p.typo` raises when a user misspells an approximator name in their
dynamics function. Measured before this classification was added, such a
typo was absorbed into the `1e6` failure sentinel at every
gradient-matching site and the solver returned `initial_params` with a
healthy-looking objective and no warning.

On Julia ≤ 1.11 the same misspelling raises a plain `ErrorException`
("type NamedTuple has no field kk"), which cannot be distinguished from a
user's own `error(...)` without matching on message text, so it is
deliberately NOT classified there. The practical consequence is
cosmetic rather than a hole in the invariant: a never-evaluable
right-hand side is caught anyway by the total-failure branch of
[`_dynamics_failure_verdict`](@ref), which errors with "raised at every
one of the N evaluation points" instead of surfacing the underlying
`FieldError`. What is lost on those versions is the specific exception
type in the message, not the refusal to report a fit.
"""
function _is_program_error(e::InexactError)
    # Julia ≥ 1.12 stores the offending value as the last element of e.args;
    # earlier versions expose it as e.val.
    val = hasfield(InexactError, :val) ? e.val : e.args[end]
    val isa Number && isfinite(val)
end
const _PROGRAM_ERROR_TYPES = Union{MethodError, BoundsError, UndefVarError,
                                   TypeError, KeyError, DimensionMismatch,
                                   UndefRefError,
                                   # Not a "program error" as such, but it
                                   # must escape for the same reason: a
                                   # Ctrl-C landing inside user dynamics was
                                   # otherwise swallowed as a numerical
                                   # failure at ~20 rethrow sites, making
                                   # long solves uninterruptible.
                                   InterruptException}

# `FieldError` was added in Julia 1.12; the package supports ≥ 1.10, so the
# reference is resolved at load time rather than written literally (a bare
# `Base.FieldError` in the Union would be an UndefVarError on 1.10/1.11).
const _HAS_FIELD_ERROR = isdefined(Base, :FieldError)

@static if _HAS_FIELD_ERROR
    _is_program_error(e) = e isa _PROGRAM_ERROR_TYPES ||
                           e isa getfield(Base, :FieldError)
else
    _is_program_error(e) = e isa _PROGRAM_ERROR_TYPES
end

"""
    _count_dynamics_failures(prob, times, states, beta) -> Int

Evaluate the dynamics once per row of `states` at the FITTED parameters
and count how many points raise a (numerical) exception. Program errors
are rethrown, as everywhere else.

Why this exists. The gradient-matching family replaces a throwing
right-hand side with a `1e6` failure sentinel so the optimizer survives a
model that is undefined on part of its state range. Rethrowing program
errors alone does not make that safe: `DomainError` is correctly NOT a
program error, so a user's own validity guard (`sqrt`, `log`, a
non-negativity floor) is still absorbed. Measured on a plain ODE fixture
with a validity band, 392 of 1260 right-hand-side evaluations (31.1%)
fell back to the sentinel and `GradientMatching` reported
`converged = true, reason = :objective_tol` with a function-recovery
RMSE of 0.051973 against 0.003050 for the same fixture without the band.
Counting the substitutions is what makes that outcome reportable.

The count is taken over the FINAL evaluation at the fitted parameters,
not over the whole run: a transient failure at a trial step inside a line
search is not a defect in the reported fit.
"""
function _count_dynamics_failures(prob::PSMProblem, times::AbstractVector,
                                  states::AbstractMatrix, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    K = size(states, 2)
    du = zeros(K)
    n_failed = 0
    for i in axes(states, 1)
        u = Vector{Float64}(@view states[i, :])
        try
            prob.dynamics!(du, u, p, times[i])
        catch e
            _is_program_error(e) && rethrow()
            n_failed += 1
        end
    end
    n_failed
end

"""
    _dynamics_failure_verdict(solver_name, n_failed, n_points, converged, reason)
        -> (converged, reason)

The package-wide policy for a fit whose right-hand side fell back to the
failure sentinel, applied by every solver that carries one (see
[`_count_dynamics_failures`](@ref)):

- `n_failed == n_points` — not one dynamics evaluation succeeded, so
  nothing was fitted and the returned coefficients are bit-for-bit the
  initial guess. `error`, because there is no fit to report. Measured
  before this guard: a one-character typo (`p.kk` for `p.k`) in an
  ordinary ODE model returned a `PSMSolution` with no error and no
  warning, coefficients bitwise at `initial_params`, and
  `objective = 0.1000804433` — which is exactly the data smoother's
  residual sum of squares on that fixture, i.e. indistinguishable from a
  healthy fit.
- `0 < n_failed < n_points` — a partial fit. Warn, and force
  `converged = false, reason = :dynamics_failure`; a convergence
  criterion that fired against sentinel residuals is not evidence of a
  fit.
- `n_failed == 0` — untouched.

`n_dynamics_failures` is reported in the convergence `NamedTuple`
unconditionally (`0` on a clean fit) so the condition is machine-checkable.

ZERO TOLERANCE, deliberately. There is no threshold: ONE failing
evaluation point out of any number forces `converged = false`. (The
independent review of this change measured `RKHSSolver` firing the
verdict at `n_dynamics_failures = 1` of 30 grid points.) The rule is
strict because the count is reported alongside it — a caller who judges
one bad point in thirty acceptable can read `n_dynamics_failures` and
decide that for themselves; what the solver must not do is decide it for
them by stamping `converged = true`.

WHAT THIS DOES NOT CATCH. Detection is by EXCEPTION TYPE ONLY. A
right-hand side that returns a large finite derivative WITHOUT throwing
is invisible to this machinery — and "return a big penalty in the bad
region" is a common idiom, so this is not a hypothetical. Measured on the
fixture of the `sentinel substitutions block convergence` testset
(runtests.jl), with `throw(DomainError(...))` inside the 2.5 ≤ u ≤ 7 band
replaced by `du[1] = 1e6`: `GradientMatching` reports
`converged = true, reason = :objective_tol`, `n_dynamics_failures = 0`,
`objective = 0.030088129645966723` and function-recovery RMSE
0.0519726780631551 — bit-for-bit the objective and RMSE of the THROWING
version that this verdict now rejects, i.e. the pre-fix defect reproduced
exactly. The reason is structural, not an oversight in the classifier:
the sentinel substituted on a caught exception IS `1e6`, so a user
penalty of that magnitude is by construction indistinguishable from a
swallowed exception, and only the exception form is counted.

A `NaN` derivative returned without throwing is likewise not counted.
Measured on the same fixture: `GradientMatching` gives
`converged = false, reason = :singular_system`; `TwoStageSolver`,
`IntegralMatchingSolver` and `ODINSolver` give
`converged = false, reason = :maxiters` with `objective = Inf` — all four
with `n_dynamics_failures = 0`. Those are INCIDENTAL rescues by the
linear algebra, not by this guard, and must not be credited to it.
(`RKHSSolver` does error on that input, but through the total-failure
branch above: the NaN states drive the dynamics into raising at all 30
grid points, so there the exception path is reached after all.)
"""
function _dynamics_failure_verdict(solver_name::String, n_failed::Int,
                                   n_points::Int, converged::Bool,
                                   reason::Symbol)
    n_failed == 0 && return (converged, reason)
    if n_failed >= n_points
        error("$solver_name: the dynamics function raised at every one of " *
              "the $n_points evaluation points, so not a single right-hand " *
              "side evaluation succeeded and nothing was fitted — the " *
              "coefficients returned would be bit-for-bit the initial guess " *
              "and the reported objective would reflect only the data " *
              "smoother. Check the dynamics function: a misspelt " *
              "approximator name (`p.kk` where the approximator is `:k`), a " *
              "state that is outside the model's domain everywhere, or a " *
              "DDE passed to an ODE-only solver all produce this. Re-run " *
              "the dynamics by hand at one smoothed state to see the " *
              "underlying exception.")
    end
    @warn "$solver_name: the dynamics function raised at $n_failed of " *
          "$n_points evaluation points at the fitted parameters " *
          "($(round(100 * n_failed / n_points, digits=1))%); those points " *
          "used a large failure sentinel instead of a real right-hand side, " *
          "so the fit is driven partly by fictitious residuals. Reporting " *
          "converged=false, reason=:dynamics_failure. Widen the model's " *
          "domain (clamp rather than throw), restrict the approximator " *
          "domain, or drop the affected times."
    (false, :dynamics_failure)
end

"""
    _adapt_gp_approximators!(prob, beta) -> Bool

Run the empirical-Bayes hyperparameter update for every `GPApproximator`
and `ShapeConstrainedGPApproximator` with `adapt=true`, using its slice of
the current coefficient vector. The constrained type stores unconstrained
γ, so its slice is mapped to the implied inducing values β = Σ·d(γ) first —
the marginal likelihood is over function values.
Returns whether any kernel changed (callers should re-evaluate the model).
"""
function _adapt_gp_approximators!(prob::PSMProblem, beta::AbstractVector)
    off = 0
    changed = false
    for a in prob.approximators
        np = nparams(a)
        if a isa GPApproximator && a.adapt
            changed |= _adapt_gp_hyperparams!(a, Float64.(beta[off+1:off+np]))
        elseif a isa ShapeConstrainedGPApproximator && a.adapt
            changed |= _adapt_gp_hyperparams!(
                a, gamma_to_inducing_values(a, Float64.(beta[off+1:off+np])))
        end
        off += np
    end
    changed
end

# ─── Simulation ───────────────────────────────────────────────────

"""
    simulate(prob, beta)

Simulate the model with parameter vector β.
Returns predicted values at data times as matrix (n_times × n_obs).
Dispatches to ODE integration (continuous) or explicit iteration (discrete).
"""
function simulate(prob::PSMProblem, beta::AbstractVector)
    if !isempty(prob.delays)
        return simulate_dde(prob, beta)
    elseif prob.discrete
        return simulate_discrete(prob, beta)
    end
    return simulate_continuous(prob, beta)
end

# ─── Companion simulated loss for state-estimating solvers ──────────
"""
    _simulated_companion(prob, beta, reported_loss) -> (loss, failed, ratio)

For solvers whose `fitted_values` are an ESTIMATED STATE rather than a
simulation of the fitted dynamics — CollocationLAML (generalized profiling),
ODINSolver, RKHSSolver — `data_loss` measures how well that state fits the
data, which is NOT how well the model does. With a finite compliance weight
the state can sit arbitrarily far from any trajectory the dynamics admit, so
the reported loss can understate the model's fit without bound: measured on
an infeasible RHS, LAML 1×, ODIN 21.7×, RKHS 3454×, CollocationLAML 1.409e9×.

B7 fixed the analogous problem in the gradient-matching family by REPLACING
`fitted_values` with `simulate(prob, beta)`. That is deliberately NOT done
here: in generalized profiling the state IS the estimand, jointly estimated
with the parameters, so simulating would change what the method returns
rather than how honestly it reports. Instead the simulated loss is reported
ALONGSIDE, as `convergence.simulated_data_loss`, so the gap is visible.

Returns `(NaN, true, NaN)` when the fitted dynamics cannot be integrated —
which is itself the strongest possible statement about the reported loss.
"""
function _simulated_companion(prob::PSMProblem, beta::AbstractVector,
                              reported_loss::Real)
    sim = try
        simulate(prob, beta)
    catch e
        _is_program_error(e) && rethrow()
        return (NaN, true, NaN)
    end
    all(isfinite, sim) || return (NaN, true, NaN)
    sl = weighted_data_loss(prob, Float64.(sim))
    ratio = reported_loss > 0 ? sl / reported_loss : NaN
    (sl, false, ratio)
end

"""
    simulate_continuous(prob, beta)

Simulate a continuous-time (ODE) model.
"""
function simulate_continuous(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    # Promote the state eltype when β carries ForwardDiff Duals (the
    # jac=:forwarddiff Jacobian path differentiates straight through this
    # solve): the integrator's in-place buffers take their eltype from u0,
    # and writing a Dual du into a Float64 buffer is a MethodError. A no-op
    # for Float64 β (Tel === Float64 keeps the original array).
    Tel = promote_type(eltype(beta), eltype(u0))
    u0 = Tel === eltype(u0) ? u0 : Tel.(u0)

    function ode_rhs!(du, u, params, t)
        prob.dynamics!(du, u, p, t)
    end

    ode_prob = ODEProblem(ode_rhs!, u0, prob.tspan)

    solve_kwargs = Dict{Symbol, Any}(
        :saveat => prob.data_times,
        :abstol => get(prob.ode_kwargs, :abstol, 1e-8),
        :reltol => get(prob.ode_kwargs, :reltol, 1e-8),
        :maxiters => get(prob.ode_kwargs, :maxiters, 1_000_000),
        :verbose => get(prob.ode_kwargs, :verbose, false),
    )
    merge!(solve_kwargs, prob.ode_kwargs)

    sol = OrdinaryDiffEq.solve(ode_prob, prob.ode_solver; solve_kwargs...)

    # Check for solver failure (e.g. maxiters exceeded, instability)
    if sol.retcode != SciMLBase.ReturnCode.Success &&
       sol.retcode != SciMLBase.ReturnCode.Default &&
       sol.retcode != SciMLBase.ReturnCode.Terminated
        error("ODE solve failed: $(sol.retcode)")
    end

    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    pred = zeros(eltype(beta), n_times, n_obs)

    length(sol.u) >= n_times ||
        error("solve terminated after $(length(sol.u)) of $n_times save " *
              "points (retcode $(sol.retcode)); cannot form predictions " *
              "at all data times")
    for i in 1:n_times
        u_i = sol.u[i]
        for j in 1:n_obs
            pred[i, j] = u_i[prob.obs_to_state[j]]
        end
    end
    pred
end

"""
    simulate_discrete(prob, beta)

Simulate a discrete-time model by explicit iteration.
The dynamics function `f!(u_next, u, p, t)` computes `u(t+1)` from `u(t)`.

Iterates through all integer time steps from `tspan[1]` to `tspan[2]`,
recording state at `data_times`.
"""
function simulate_discrete(prob::PSMProblem, beta::AbstractVector)
    p = build_param_struct(prob, beta)
    u0 = prob.u0 isa Function ? prob.u0(p) : prob.u0
    T = eltype(beta)

    n_vars = length(u0)
    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    pred = zeros(T, n_times, n_obs)

    t_start = prob.tspan[1]
    t_end = prob.tspan[2]

    # Build sorted set of all times we need to visit
    # (integer steps from tspan[1] to tspan[2])
    all_times = collect(t_start:1.0:t_end)

    # Map data_times to indices in all_times (allow non-integer data_times
    # by finding nearest time step)
    data_time_set = Dict{Float64, Vector{Int}}()
    for (di, dt) in enumerate(prob.data_times)
        # Round to nearest time step
        t_nearest = round(dt)
        if !haskey(data_time_set, t_nearest)
            data_time_set[t_nearest] = Int[]
        end
        push!(data_time_set[t_nearest], di)
    end

    u = T.(u0)
    u_next = similar(u)

    # Record initial condition if it's a data time
    t = t_start
    if haskey(data_time_set, t)
        for di in data_time_set[t]
            for j in 1:n_obs
                pred[di, j] = u[prob.obs_to_state[j]]
            end
        end
    end

    # Iterate forward
    for step in 1:(length(all_times)-1)
        t = all_times[step]
        prob.dynamics!(u_next, u, p, t)
        u = copy(u_next)
        t_now = all_times[step + 1]

        if haskey(data_time_set, t_now)
            for di in data_time_set[t_now]
                for j in 1:n_obs
                    pred[di, j] = u[prob.obs_to_state[j]]
                end
            end
        end
    end

    pred
end

"""
    predict(sol::PSMSolution, prob::PSMProblem)

Predict at data times using the fitted solution.
"""
function predict(sol::PSMSolution, prob::PSMProblem)
    sol.fitted_values
end

# ─── Prediction Jacobian (finite differences / ForwardDiff) ───────

"""
    _fd_jacobian_config(n_p)

`ForwardDiff.JacobianConfig` for the `jac=:forwarddiff` prediction-Jacobian
path. Built ONCE per solve — the chunk size is a function of the parameter
count only, so there is no reason to rebuild it every IRLS iteration.
Tagless (`nothing` as the function) because each call wraps `simulate` in a
fresh closure; tag checking is disabled at the call sites accordingly.
"""
function _fd_jacobian_config(n_p::Int)
    x = zeros(n_p)
    ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk(x))
end

"""
    _forwarddiff_jacobian!(J, prob, beta, n_times, n_obs, cfg) -> Bool

`jac=:forwarddiff` backend for [`compute_jacobian!`](@ref): one Dual-valued
`simulate` sweep (n_p/chunk solves) instead of the FD path's 2·n_p perturbed
solves, and exact to solver precision instead of carrying FD truncation +
integration-noise error.

Returns `true` when `J` was filled with finite entries. Returns `false` —
leaving the caller to fall back to the finite-difference path for this
iteration — when the Dual-valued solve fails numerically or produces
non-finite entries; this mirrors how the FD path itself degrades per column
when a perturbed solve fails. Program errors are rethrown, exactly as in
the FD path.

Rows are produced for EVERY data cell, masked ones included, identical to
the FD path — masking is applied downstream through the weights, never
inside the Jacobian.
"""
function _forwarddiff_jacobian!(J::AbstractMatrix, prob::PSMProblem,
                                beta::AbstractVector,
                                n_times::Int, n_obs::Int, cfg)
    n_data = n_times * n_obs
    # Flattened prediction map in the same obs-major order the FD path uses.
    predmap = function (b)
        pred = simulate(prob, b)
        f = similar(b, n_data)
        k = 1
        for oi in 1:n_obs, ti in 1:n_times
            f[k] = pred[ti, oi]
            k += 1
        end
        f
    end
    try
        if cfg === nothing
            ForwardDiff.jacobian!(J, predmap, beta)
        else
            ForwardDiff.jacobian!(J, predmap, beta, cfg, Val{false}())
        end
    catch e
        _is_program_error(e) && rethrow()
        return false
    end
    all(isfinite, J)
end

"""
    compute_jacobian!(J, prob, beta, f0, n_times, n_obs; dam, jac=:fd,
                      fd_cfg=nothing, grow=true)

Compute Jacobian of model predictions w.r.t. parameters.

`jac=:fd` (default, historical behavior): central finite differences with
adaptive step sizes — one full model solve per perturbed column, 2·n_p
solves per call, plus one nudge solve that measures the model's re-solve
output jitter (adaptive integrators re-select their step sequence under a
tiny parameter perturbation, so two solves at nearly identical parameters
differ at far above machine precision). A column whose central-difference
signal falls below `_FD_SNR_TRIGGER`× that measured jitter — the
qualitative-garbage regime — is recomputed at grown steps (never shrunk on
noise) until its signal clears `_FD_SNR_TARGET`× the noise or the step
hits a cap, with each grown step validated against the previous column
(`_FD_GROW_TOL`) and a curvature guard (`_FD_CURV_MAX`) stopping growth
once truncation is resolved above the noise. Columns already at trigger
SNR or better keep the historical fixed-step behavior. `dam` contains the
adaptive fractional FD intervals per parameter and is updated in place by
the historical truncation/cancellation heuristic only — noise-driven
growth is per-call and never persisted, so J stays a pure function of
`(prob, beta, dam)`.

`grow=false` disables the noise measurement and growth loop entirely,
reproducing the historical fixed-step FD policy byte-for-byte. GCVSolver
passes it: its `search=:direct` vs `search=:reuse` equivalence contract
(agreement to 1e-6) requires a Jacobian that is CONTINUOUS in `beta` —
threshold-triggered growth flips with the chaotic re-solve jitter, and on
the W10 two-approximator fixture a single mid-search call whose jitter
spiked to 2.7e-7 grew three columns and drove the two searches to λ̂
pairs far apart (measured 0.756 vs 0.253 against a 1e-4 log gate).

`jac=:forwarddiff`: forward-mode AD through the model solve (see
[`_forwarddiff_jacobian!`](@ref)); `fd_cfg` is the per-solve
`ForwardDiff.JacobianConfig` from [`_fd_jacobian_config`](@ref) (or
`nothing` for an ad-hoc config). If the Dual-valued solve fails or returns
non-finite entries, the call falls back to the FD path for this iteration
(with a `@debug` note); `dam` is only consulted/updated on that fallback.

J is (n_data × n_params), f0 is the flattened prediction vector.
"""
function compute_jacobian!(J::AbstractMatrix, prob::PSMProblem,
                           beta::AbstractVector, f0::AbstractVector,
                           n_times::Int, n_obs::Int;
                           dam::Vector{Float64}, jac::Symbol=:fd,
                           fd_cfg=nothing, grow::Bool=true)
    if jac === :forwarddiff
        _forwarddiff_jacobian!(J, prob, beta, n_times, n_obs, fd_cfg) &&
            return
        @debug "compute_jacobian!: ForwardDiff Jacobian failed or returned " *
               "non-finite entries; falling back to finite differences for " *
               "this iteration"
    end
    n_p = length(beta)
    n_data = n_times * n_obs
    p_pert = copy(beta)
    fp = zeros(n_data)
    fb = zeros(n_data)
    jprev = zeros(n_data)   # last accepted column during validated growth

    # Absolute FD step floor. DDEfit's fully relative step (da = dam·|β|,
    # floored at 1e-8·dam ≈ 1e-16 for β = 0) was safe there because Wood
    # (2001, p.11) reuses the SAME integration time steps for the perturbed
    # and unperturbed trajectories, so integrator error cancels in the
    # difference; simulate() re-solves adaptively per perturbation, so a
    # 1e-16 step measures pure solver noise. Tie the floor to the solver
    # tolerance instead: differences must exceed integration error. The
    # floor is the STARTING step only — measured on a noise-free logistic
    # ODE fixture (default tolerances, zero initial coefficients) the
    # central-difference error is minimal exactly here (6.7e-11 at h=1e-6,
    # rising in both directions), while noisy fixtures are handled by the
    # signal-to-noise growth loop below, not by a larger floor.
    reltol_ode = Float64(get(prob.ode_kwargs, :reltol, 1e-8))
    abs_floor = max(100.0 * reltol_ode, 1e-7)

    # Measure the actual re-solve output jitter once per call: nudge every
    # parameter by an FD-invisible 1e-13·max(1,|βⱼ|) and re-solve. Adaptive
    # integrators re-select their accepted-step sequence under the nudge,
    # exposing the true perturbation-to-perturbation noise, which is far
    # above both machine precision and the naive tolerance guess (measured
    # on the exact-reference quadrature fixture at reltol=abstol=1e-8:
    # jitter 1.5e-6 ≈ 146·reltol; on the noise-free logistic fixture:
    # 6.7e-14). For noise-free paths (e.g. discrete maps) the measurement
    # collapses to ‖J‖·1e-13 — a harmless overestimate that still accepts
    # the floor step immediately in the growth loop below.
    eps_noise = 1e-15 * max(1.0, maximum(abs, f0))
    grow && let nudged = copy(beta)
        for j in 1:n_p
            nudged[j] += 1e-13 * max(1.0, abs(beta[j]))
        end
        pred_n = try
            simulate(prob, nudged)
        catch e
            _is_program_error(e) && rethrow()
            nothing
        end
        if pred_n !== nothing
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                d = abs(pred_n[ti, oi] - f0[k])
                isfinite(d) && (eps_noise = max(eps_noise, d))
                k += 1
            end
        end
    end

    for j in 1:n_p
        # Step cap: on the exact-reference quadrature fixture (‖J‖=1.196,
        # jitter 1.5e-6) the weakest columns need h up to ~1e-1 before
        # their signal clears the noise, and the measured error still
        # FALLS through that range (1.3e-5 at h=1e-1 vs 9.9e-5 at h=1e-2 —
        # the growth is only taken when smaller steps are provably noise,
        # so a large cap is safe). The cap must bound the STARTING step
        # too, not just the growth: dam is shared across IRLS/λ iterations
        # while β moves, so a fractional step persisted at a tiny |βⱼ|
        # replayed at a larger |βⱼ| would otherwise start (and accept)
        # far above the cap — measured on the quadrature fixture: dam
        # grown to 2.1045 at β₁=1e-3 would start the next call at 1.0523
        # for β₁=0.5, 10.5× the cap; the clamp starts it at 0.1.
        da_cap = 0.1 * max(1.0, abs(beta[j]))
        da = clamp(dam[j] * abs(beta[j]), abs_floor, da_cap)
        grew = false
        have_col = false
        adapt_col = false    # run dam adaptation only on a full central diff
        reverted = false     # a grown step was computed and rejected
        pending_validate = false   # this iteration recomputes a grown step
        da_prev = da
        da_used = da         # the step of the last SUCCESSFUL central diff
        while true
            # Forward perturbation
            p_pert[j] = beta[j] + da
            pred_fwd = try
                simulate(prob, p_pert)
            catch e
                _is_program_error(e) && rethrow()
                nothing
            end
            p_pert[j] = beta[j]
            if pred_fwd === nothing
                # Keep the last accepted (smaller-step) column if we have
                # one; otherwise don't leak the previous iteration's column.
                have_col || (J[:, j] .= 0.0)
                break
            end

            # Backward perturbation
            p_pert[j] = beta[j] - da
            pred_bwd = try
                simulate(prob, p_pert)
            catch
                nothing
            end
            p_pert[j] = beta[j]
            if pred_bwd === nothing
                if !have_col
                    # Fall back to forward differences (historical path;
                    # skips step adaptation like it always did)
                    k = 1
                    for oi in 1:n_obs, ti in 1:n_times
                        J[k, j] = (pred_fwd[ti, oi] - f0[k]) / da
                        k += 1
                    end
                end
                break
            end

            # Flatten and compute central differences
            k = 1
            signal = 0.0
            d2max = 0.0
            for oi in 1:n_obs, ti in 1:n_times
                fp[k] = pred_fwd[ti, oi]
                fb[k] = pred_bwd[ti, oi]
                J[k, j] = (fp[k] - fb[k]) / (2.0 * da)
                signal = max(signal, abs(fp[k] - fb[k]))
                d2max = max(d2max, abs(fp[k] - 2.0 * f0[k] + fb[k]))
                k += 1
            end
            # Validated growth: a grown step is kept only if the column
            # moved by no more than the noise it was supposed to remove.
            # When the smaller-step column was noise-limited, its error is
            # ≈ eps_noise/(2·da_prev), so the recomputed column should
            # differ by about that much; a change far beyond it means the
            # larger step introduced REAL error (truncation, or a
            # nonlinearity/clamp kink in the perturbed dynamics) and the
            # smaller-step column was already the better estimate — keep
            # it and stop. This is the safety net that no base-point
            # noise/curvature statistic can provide, because the jitter is
            # measured once at beta and can misclassify a mid-fit column.
            if pending_validate
                change = 0.0
                for k2 in 1:n_data
                    change = max(change, abs(J[k2, j] - jprev[k2]))
                end
                if change > _FD_GROW_TOL * eps_noise / (2.0 * da_prev)
                    @inbounds for k2 in 1:n_data
                        J[k2, j] = jprev[k2]
                    end
                    da_used = da_prev
                    reverted = true
                    break
                end
                pending_validate = false
            end
            have_col = true
            adapt_col = true
            da_used = da

            # Noise-domination check: the FD error in this column is
            # ≈ eps_noise/(2·da) = (eps_noise/signal)·max|J[:,j]|, so
            # requiring signal ≥ _FD_SNR_TARGET·eps_noise bounds the
            # column's noise-induced relative error by 1/_FD_SNR_TARGET.
            # When noise dominates, GROW the step (the old te/ce heuristic
            # misread noise in the second difference as truncation error
            # and shrank it — measured dam 1e-8 → 1e-9 on every column of
            # the quadrature fixture, error 0.645 on a scale-1.196 matrix).
            # Growth is additionally allowed only while the column is
            # PROVABLY noise-limited: once the second difference rises
            # clearly above the jitter (d2max > _FD_CURV_MAX·eps_noise)
            # the column is truncation-limited and a larger step trades
            # bounded noise error for unbounded truncation error — on a
            # stiff-sensitivity SIR fixture (‖J‖=2e4, truncation error
            # 0.295 at h=1e-5 rising to 1.5e4 at h=1e-2) unrestricted
            # growth collapsed whole LAML fits (edf → 0).
            # Growth is TRIGGERED only for catastrophically noise-dominated
            # columns (SNR below _FD_SNR_TRIGGER — the B1 defect regime,
            # where the column is qualitatively garbage); a column that
            # already resolves its signal at ≥ trigger SNR keeps the
            # historical floor-step behavior. Once triggered, the column
            # grows all the way to the _FD_SNR_TARGET accuracy.
            snr_gate = grew ? _FD_SNR_TARGET : _FD_SNR_TRIGGER
            (!grow || signal >= snr_gate * eps_noise || da >= da_cap ||
             d2max > _FD_CURV_MAX * eps_noise) && break
            # Jump toward the step that reaches the target SNR (signal
            # scales linearly in da once above the noise), at least ×10.
            @inbounds for k2 in 1:n_data
                jprev[k2] = J[k2, j]
            end
            da_prev = da
            pending_validate = true
            da = min(da_cap,
                     max(10.0 * da,
                         da * _FD_SNR_TARGET * eps_noise / max(signal, 1e-300)))
            grew = true
        end
        # After a revert, fp/fb hold the REJECTED step's values, so the
        # te/ce adaptation below would be inconsistent — skip it (dam is
        # left untouched, like the historical failure paths).
        (adapt_col && !reverted) || continue

        # Noise-grown steps are deliberately NOT persisted into dam: the
        # Jacobian must be a pure function of (prob, beta, dam-in) up to
        # the historical ±10× te/ce drift, because designed equivalence
        # contracts (GCVSolver search=:direct vs :reuse, suite W10) compare
        # solves whose compute_jacobian! call counts differ — a persisted
        # step makes J history-dependent and was measured to break that
        # equivalence (λ̂ log-mismatch ≥ 1e-4, fitted-value diff 1.96e-2
        # against 1e-6 gates on the two-approximator coordinate-descent
        # fixture). Each call re-pays its growth retries instead; the
        # starting-step clamp above stays as a guard on the te/ce drift.

        # Adapt step size — the HISTORICAL heuristic, byte-identical to
        # the pre-B1 code, and deliberately free of eps_noise: the jitter
        # measurement is itself chaotic in β (a re-solve step-sequence
        # artifact), so letting it into dam — state shared across IRLS/λ
        # iterations — injects that chaos into every subsequent Jacobian.
        # Measured on the W10 two-approximator GCV fixture: with the
        # measured noise in these gates the contractually equivalent
        # :direct and :reuse searches diverged to λ̂ pairs e^18 apart and
        # objectives 0.000159 vs 0.003618 against 1e-6 agreement gates;
        # with the historical gates they agree again. The B1 wrong-way
        # shrink (noise read as truncation, dam 1e-8 → 1e-9) is fixed one
        # level up instead: a column the SNR loop grew (`grew`) skips
        # adaptation entirely — its fp/fb sit at the grown step, and its
        # noise handling belongs to the growth loop, per call.
        grew && continue
        k = 1
        mean_te = 0.0
        mean_ce = 0.0
        for oi in 1:n_obs, ti in 1:n_times
            mean_te += 0.5 * (fp[k] - 2.0 * f0[k] + fb[k]) / da_used
            mean_ce += 2.0 * max(abs(f0[k]), abs(fp[k])) * 1e-15 / da_used
            k += 1
        end
        if dam[j] >= 1e-10 && abs(mean_te) > 10.0 * abs(mean_ce)
            dam[j] /= 10.0
        end
        if dam[j] <= 0.001 && abs(mean_ce) > 10.0 * abs(mean_te)
            dam[j] *= 10.0
        end
    end
end

# Target signal-to-noise ratio for an accepted central-difference column:
# the noise-induced relative error of the column is ≈ 1/_FD_SNR_TARGET.
# 1e4 measured on both calibration fixtures this session: the
# exact-reference quadrature fixture (O(1) sensitivities, jitter 1.5e-6)
# lands at steps 1e-2..1e-1 with max error 9.9e-5..1.3e-5 (≤ 8.3e-5 of
# scale), and the noise-free logistic/discrete fixtures accept the 1e-6
# floor step immediately (signal ~3e-6 ≫ 1e4·6.7e-14), leaving their
# measured 6.7e-11 / 8.2e-11 accuracy untouched.
const _FD_SNR_TARGET = 1.0e4

# Growth trigger: a column enters the growth loop only when its measured
# SNR at the starting step is below this — i.e. its noise-induced relative
# error exceeds ~10%, the qualitative-garbage regime the B1 fix exists
# for (the exact-reference fixture measures floor-step SNR of roughly 2–5
# across chaotic jitter draws, with 54%-of-scale error). Columns already
# at SNR ≥ 10 keep the historical floor-step behavior: measured on the
# SIR-Poisson warm-start fixture, growing such columns (e.g. col 8, floor
# SNR 2023, error 4.8e-3 → 5.8e-6 when grown — individually BETTER) still
# perturbs the chaotic warm-started IRLS path into a λ-clamp dead-zone
# basin (J ≡ 0, edf → 0, data_loss 5.2e5 vs 3342 for jac=:forwarddiff),
# so unnecessary growth is a stability hazard, not an accuracy win.
# KNOWN LIMITATION (measured, independent review of B1): the SNR is
# computed against a single base-point nudge jitter, which can
# underestimate the realized per-column ±h re-solve noise by ~6–11×. A
# noise-inflated signal can then clear the trigger and keep a garbage
# column: on a one-coefficient variant of the reference fixture
# (β₁ = 1e-3), columns at measured SNR 12.55 and 18.15 escaped the
# trigger with errors of 0.31 and 0.45 of column scale. Closing this
# needs per-column noise estimation (e.g. a second nudge or the ±h pair
# asymmetry) — see the review record before changing the constant alone.
const _FD_SNR_TRIGGER = 10.0

# Curvature guard for the growth loop: a column may only grow while its
# max second difference is within _FD_CURV_MAX× the measured jitter — at
# that level the second difference is indistinguishable from noise (it
# combines the noise of three solves), so a larger step provably reduces
# total error; above it, truncation is already resolved and growth would
# increase it. Calibrated on both regimes this session: noise-dominated
# columns that MUST grow show worst-col D2/jitter of 6.0–8.4 (quadrature
# fixture, every step 1e-6..1e-1) and 0.0 (weakest W11-zero column),
# while truncation-limited columns that must NOT grow show 82.6–3798 at
# the floor step (SIR-Poisson warm-start fixture, cols 1–4). 25 sits
# between with ~3.0× margin below and ~3.3× above.
const _FD_CURV_MAX = 25.0

# Validated-growth tolerance: a recomputed (grown) column is kept only if
# max|Δcolumn| ≤ _FD_GROW_TOL · eps_noise/(2·da_prev). Calibrated this
# session: on the quadrature fixture (growth MUST be accepted) the
# measured change/prediction ratio is 0.47–2.60 on the two columns probed
# across step decades 1e-6→1e-1, while on the truncation-limited
# SIR-Poisson columns (growth must be rejected) it is 40.9 → 4.5e7. 8
# sits between with 3.1× margin below and 5.1× above.
# KNOWN LIMITATION (measured, independent review of B1): the tolerance is
# denominated in the base-point nudge jitter, so when the realized ±h
# re-solve noise is much larger than the nudge measured, a LEGITIMATE
# growth can be rejected and the column falls back to the (garbage) floor
# step — pre-fix behavior, not a regression, but an incomplete repair: on
# the one-coefficient reference-fixture variant, a needed growth was
# rejected at measured change 0.95 vs tolerance 0.685, keeping a 0.95
# error on column scale 1.08. Same root cause and same fix direction as
# the _FD_SNR_TRIGGER limitation above.
const _FD_GROW_TOL = 8.0

# ─── Penalty matrix assembly ─────────────────────────────────────

"""
    build_penalty_matrices(prob)

Enumerate the quadratic penalty blocks of every approximator (unit
smoothing parameter), via [`penalty_blocks`](@ref) — by default one block
per penalized approximator, its `penalty_matrix` over the full coefficient
range; a type overriding `penalty_blocks` contributes one entry per block.
Every downstream smoothing-parameter machinery (LAML's Fellner–Schall and
Newton phases, GCV's coordinate descent, CollocationLAML's and
GradientMatching's FS updates) is per-ENTRY of these lists, so each block
receives its own λ.

Returns `(S_list, offsets, nknots_list)` where entry `l` is a penalty
matrix, its GLOBAL offset into the flat coefficient vector, and its block
size. `nknots_list` is the historical name: each entry is the SIZE of a
penalty block (`length(range)`), which coincided with the knot count when
blocks were whole approximators — every consumer treats it as a block
size.

Validates the blocks of each approximator (this is the single funnel all
block consumers go through): local ranges must be non-empty and lie
within `1:nparams(approx)`, be pairwise disjoint (the per-block
generalized determinant `log|S_λ|₊` in laml.jl is exact only for
non-overlapping blocks), and each `S` must be
`length(range) × length(range)`, numerically symmetric, and positive
semi-definite (a non-PSD block would otherwise pass silently and be
treated as unpenalized by the rank and log-determinant routines).
Violations throw an `ArgumentError` naming the approximator.
"""
function build_penalty_matrices(prob::PSMProblem)
    S_list = Matrix{Float64}[]
    offsets = Int[]
    nknots_list = Int[]

    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        covered = falses(np)
        for (S, r) in penalty_blocks(approx)
            # Contiguous ranges only: the enumeration below records a block
            # by (offset, length), so a strided or non-contiguous index set
            # would be silently reinterpreted as the contiguous range of
            # the same length.
            r isa AbstractUnitRange{<:Integer} || throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): range must be a " *
                "contiguous unit range (got $(typeof(r)): $r)"))
            isempty(r) && throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): empty range $r"))
            (1 <= first(r) && last(r) <= np) || throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): range $r lies outside " *
                "the approximator's coefficient block 1:$np"))
            size(S) == (length(r), length(r)) || throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): S is " *
                "$(size(S, 1))×$(size(S, 2)) but range $r has length " *
                "$(length(r))"))
            asym = maximum(abs.(S .- S'))
            asym <= 1e-8 * max(maximum(abs.(S)), 1.0) || throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): S for range $r is not " *
                "symmetric (max |S - S'| = $asym)"))
            # PSD, which the contract requires and nothing used to check.
            # Neither failure mode is caught downstream, and both are silent
            # (measured): a NEGATIVE-DEFINITE block gives _rank_penalty 0 and
            # _log_det_plus 0, i.e. a fully inert penalty — the user asked
            # for smoothing and got none. An INDEFINITE block is worse
            # because it looks alive: diag(-1,+1) gives rank 1 and logdet 0,
            # so the negative direction is silently dropped and the block
            # penalizes a subspace the user never specified. One
            # eigendecomposition per block per solve (every call site is
            # once-per-solve, and blocks are small), so the cost is noise.
            #
            # The tolerance is relative and generous: across 73 penalty
            # blocks from 69 configurations of all eleven built-in types,
            # the worst relative eigmin is -9.7e-17 (pure roundoff), so
            # -1e-8 leaves seven orders of margin over anything the package
            # itself produces.
            evmin, evmax = extrema(eigvals(Symmetric(Matrix(Float64.(S)))))
            evmin >= -1e-8 * max(evmax, 1.0) || throw(ArgumentError(
                "penalty_blocks(:$(approx.name)): S for range $r is not " *
                "positive semi-definite (smallest eigenvalue $evmin against " *
                "largest $evmax). A penalty must be PSD: a negative " *
                "direction would REWARD roughness, and downstream rank and " *
                "log-determinant routines would silently treat the block as " *
                "unpenalized rather than report the error."))
            for i in r
                covered[i] && throw(ArgumentError(
                    "penalty_blocks(:$(approx.name)): ranges overlap at " *
                    "local index $i. Blocks must be pairwise disjoint — " *
                    "the per-block generalized determinant log|S_λ|₊ in " *
                    "laml.jl is exact only for non-overlapping blocks."))
                covered[i] = true
            end
            push!(S_list, S)
            push!(offsets, offset + first(r) - 1)
            push!(nknots_list, length(r))
        end
        offset += np
    end
    S_list, offsets, nknots_list
end

"""
    penalty_sqrt_matrix(S)

Compute C such that C'C = S (via eigendecomposition of PSD matrix).
"""
function penalty_sqrt_matrix(S::AbstractMatrix)
    eig = eigen(Symmetric(S))
    tol = max(1e-10 * maximum(abs.(eig.values)), 1e-14)
    pos_idx = findall(eig.values .> tol)
    if isempty(pos_idx)
        return zeros(0, size(S, 1))
    end
    Diagonal(sqrt.(eig.values[pos_idx])) * eig.vectors[:, pos_idx]'
end

# ─── Main solve function ─────────────────────────────────────────

"""
    SciMLBase.solve(prob::PSMProblem, alg::LAML)

Fit a partially specified model using IRLS with LAML smoothing.

# Algorithm
For each IRLS iteration:
1. Evaluate model and compute the prediction Jacobian (finite differences
   by default; forward-mode AD with `LAML(jac=:forwarddiff)`)
2. Form pseudodata z = y - f + J*β
3. Solve penalized least squares (augmented system)
4. Step contraction (backtracking)
5. Re-estimate smoothing parameters via Fellner-Schall + Newton

Returns a `PSMSolution`. `sol.convergence` is a NamedTuple
`(V_beta, sigma2, converged, iterations, reason, laml_failures, criterion,
laml, stationarity, smoothing_advanced)` — see the `LAML` and `PSMSolution`
docstrings for the key taxonomy. Note in particular that `converged` is a
stability test, and that `stationarity`/`smoothing_advanced` are the additive
diagnostics that say whether the fit stopped at a smoothing optimum or merely
stopped moving.
"""
function SciMLBase.solve(prob::PSMProblem, alg::LAML)
    _validate_problem(prob, "LAML")
    _warn_unanchored_index(prob, "LAML")
    # The full Laplace criterion needs the actual family's NORMALIZED
    # log-likelihood with a KNOWN, fixed dispersion (its value enters the
    # criterion directly, and the generalized Fellner-Schall update assumes
    # unit dispersion). A CustomLikelihood's loglik_scalar may be an
    # arbitrary — possibly unnormalized, possibly free-dispersion — kernel,
    # so both the criterion value and its FS calibration would be off by
    # unknown amounts. Refuse loudly rather than fit with a silently wrong
    # criterion.
    if alg.criterion === :laplace && prob.likelihood isa CustomLikelihood
        error("LAML(criterion=:laplace) does not support CustomLikelihood: " *
              "the full Laplace criterion requires a normalized log-density " *
              "with fixed dispersion, which a user-supplied loglik_scalar " *
              "does not declare. Use LAML(criterion=:working) (the default) " *
              "or a built-in likelihood (Gaussian, Poisson, " *
              "NegativeBinomial, TruncatedNormal).")
    end
    maxiters = alg.maxiters
    verbose = alg.verbose

    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    n_data = n_times * n_obs
    n_p = n_total_params(prob)

    # jac=:forwarddiff — one JacobianConfig per solve (chunking depends only
    # on n_p), reused by every compute_jacobian! call below.
    fd_cfg = alg.jac === :forwarddiff ? _fd_jacobian_config(n_p) : nothing

    # Enumerate penalty blocks (default: one per penalized approximator; a
    # multi-block type contributes one entry — and hence one λ — per block)
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)

    # Mixed-approximator dof advisory: parameters without a penalty block
    # (e.g. NeuralApproximator weights with penalty_weight = 0) are REML
    # fixed effects, and when they rival the data size they exhaust the
    # restricted residual dof that the Gaussian scale σ̂² = (RSS+pen)/(n−Mp)
    # is estimated from. `laml.jl` charges them by design rank rather than
    # raw count (see `_restricted_dof_Mp`) and warns again if even that
    # exhausts the dof; warn here so the over-parameterization is visible
    # before any fitting happens.
    n_unpenalized = n_p - sum(uf_nk; init=0)
    if m > 0 && n_unpenalized > 0
        total_rank = sum(_rank_penalty(S_list[l]) for l in 1:m)
        if n_data - (n_p - total_rank) < 10
            @warn "LAML: $n_unpenalized unpenalized parameters (e.g. neural network " *
                  "weights) leave at most $(n_data - (n_p - total_rank)) rank-based " *
                  "residual degrees of freedom for the n=$n_data data points. The " *
                  "Gaussian scale σ̂² charges them by design rank instead, but the " *
                  "model is heavily over-parameterized: consider more data, a " *
                  "smaller network, or penalty_weight > 0."
        end
    end

    # Initialize smoothing: user-specified or data-driven default.
    # The penalty matrices are computed on a normalised [0,1] domain,
    # so their eigenvalue spectrum is stable across problems.
    # Default: θ = 1/tr(S) ≈ 3.7e-5 (light initial smoothing).  LAML will
    # quickly adjust this once the warmup phase is complete.  For strongly
    # nonlinear problems, use initial_lambda=10.0 + warmup=5 or higher.
    if alg.initial_lambda !== nothing
        theta = fill(alg.initial_lambda, m)
    else
        theta = Float64[1.0 / max(tr(S_list[l]), 1e-10) for l in 1:m]
    end
    # Snapshot of the INITIALIZATION, kept untouched for the whole solve so
    # `convergence.smoothing_advanced` can answer "did smoothing selection
    # ever move λ off this?". Compared against the FINAL reported θ (=
    # theta_fit), so both the Gaussian warm-start and the main-loop
    # Fellner-Schall count as advancement.
    theta_init = copy(theta)

    # Build total penalty B = Σ θ_k S_k (embedded in n_p × n_p)
    function build_B(th)
        B = zeros(n_p, n_p)
        for l in 1:m
            off = uf_offsets[l]
            nk = uf_nk[l]
            for i in 1:nk, j in 1:nk
                B[off+i, off+j] += th[l] * S_list[l][i, j]
            end
        end
        B
    end

    # Flatten data, enforcing the package's masking convention: a cell is
    # usable only if its weight is positive AND its datum is non-NaN.
    # Masked cells get weight 0 and a finite placeholder value — every
    # downstream use multiplies by the weight, but IEEE `0 * NaN = NaN`,
    # so leaving a NaN datum in y_vec would silently poison the objective
    # (the optimizer would then reject every step and return the initial
    # coefficients unchanged, without any error).
    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y = prob.data_values[ti, oi]
        wv = prob.data_weights[ti, oi]
        if _usable(y, wv)
            y_vec[k] = y
            w_vec[k] = wv
        end   # else keep the 0.0 placeholder with weight 0.0
        k += 1
    end

    # Evaluate model, return flattened predictions
    function eval_model(p_eval)
        pred = simulate(prob, p_eval)
        f_tmp = zeros(n_data)
        local k = 1
        for oi in 1:n_obs, ti in 1:n_times
            f_tmp[k] = pred[ti, oi]
            k += 1
        end
        f_tmp, pred
    end

    # Penalized objective: -ℓ(y,μ) + ½β'Bβ
    # For Gaussian this equals ½(RSS + penalty); for other families uses the
    # actual log-likelihood, ensuring correct step comparisons.
    function penalized_objective(p_eval, B)
        f_tmp, _ = try; eval_model(p_eval); catch; return Inf; end
        neg_ll = -log_likelihood(prob.likelihood, y_vec, f_tmp, w_vec)
        neg_ll + 0.5 * dot(p_eval, B * p_eval)
    end

    # PCLS step: truncated-SVD solve of the augmented system
    # [W^½J; C] β = [W^½z; 0] — see _pcls_factorize / _pcls_truncated_step in
    # pcls.jl.  Uses IRLS weights that depend on the current predictions.
    # The factorization is returned as well so `step_contract` can reuse it
    # for the trust region instead of decomposing again.
    function pcls_step(J_mat, z_pseudo, th, w_irls)
        B = build_B(th)
        fac = _pcls_factorize(J_mat, z_pseudo, B, w_irls)
        _pcls_truncated_step(fac), B, fac
    end

    # Step contraction: backtracking with explosive-step rescue, plus the
    # Levenberg-Marquardt trust region — see _pcls_step_contract in pcls.jl.
    step_contract(a_old, a_new, B, fac) =
        _pcls_step_contract(penalized_objective, a_old, a_new, B, fac)

    # Initialize
    beta = build_initial_params(prob)
    J = zeros(n_data, n_p)
    f_vec = zeros(n_data)
    dam = fill(1e-8, n_p)

    if verbose
        println("IRLS+LAML: $n_p params, $n_data data, $m smooth terms")
        println("Initial θ: ", [round(t, sigdigits=4) for t in theta])
    end

    f_vec, _ = eval_model(beta)
    compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                      jac=alg.jac, fd_cfg=fd_cfg)

    # ─── Gaussian warm-start for non-Gaussian likelihoods ─────────
    # For Poisson/NegBin with identity link, the IRLS weights (1/V(μ))
    # can create local minima when the initial fit is poor.  We run a
    # full Gaussian IRLS+LAML solve (unit weights, profiled σ²) to find
    # good starting coefficients AND smoothing parameters, then switch
    # to the actual likelihood.  This mimics mgcv's initialization.
    if !(prob.likelihood isa Gaussian) && m > 0
        gw_otheta = copy(theta)
        prev_gw_obj = Inf

        for gw_iter in 1:50
            f_vec_new, _ = try; eval_model(beta); catch; break; end
            f_vec .= f_vec_new
            compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                              jac=alg.jac, fd_cfg=fd_cfg)

            w_gauss = copy(w_vec)
            z_pseudo = y_vec .- f_vec .+ J * beta

            # Try step with current θ
            a0_pcls, B0, fac0 = pcls_step(J, z_pseudo, gw_otheta, w_gauss)
            a0, f01 = step_contract(beta, a0_pcls, B0, fac0)

            # After warmup iters, also estimate θ via Gaussian LAML
            if gw_iter > 3
                theta_new, _ = try
                    rho0 = log.(max.(gw_otheta, 1e-20))
                    estimate_smoothing_params(J, w_gauss, w_vec,
                        y_vec, f_vec, beta, S_list, uf_offsets, uf_nk, n_p;
                        family=Gaussian(), rho_init=rho0, verbose=false)
                catch; (copy(gw_otheta), NaN); end

                # Try step with new θ
                a1_pcls, B1, fac1 = pcls_step(J, z_pseudo, theta_new, w_gauss)
                a1, f11 = step_contract(beta, a1_pcls, B1, fac1)

                # Accept new θ only if it improves the Gaussian data fit.
                # Evaluate each candidate's model ONCE (a full ODE solve) —
                # the previous generator re-solved the ODE per data point.
                f_a0 = try; first(eval_model(a0)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                f_a1 = try; first(eval_model(a1)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                ss_a0 = sum((y_vec[i] - f_a0[i])^2 * w_vec[i] for i in 1:n_data)
                ss_a1 = sum((y_vec[i] - f_a1[i])^2 * w_vec[i] for i in 1:n_data)

                if ss_a1 <= ss_a0
                    beta .= a1
                    gw_otheta .= theta_new
                    f_vec .= f_a1
                else
                    beta .= a0
                    f_vec .= f_a0
                end
            else
                beta .= a0
                f_new = try; first(eval_model(a0)); catch e
                    _is_program_error(e) && rethrow(); f_vec; end
                f_vec .= f_new
            end

            # Convergence monitor uses the CURRENT step's fit (the old code
            # scored f_vec from before the step — a one-iteration-stale
            # objective).
            gw_ss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
            gw_obj = 0.5 * (gw_ss + dot(beta, build_B(gw_otheta) * beta))

            if verbose && (gw_iter <= 3 || gw_iter % 10 == 0)
                println("Gauss-warmup $gw_iter: SS=$(round(gw_ss, sigdigits=6)), " *
                        "θ=$(round.(gw_otheta, sigdigits=3))")
            end

            if gw_iter > 5 && abs(gw_obj - prev_gw_obj) < 1e-6 * max(abs(prev_gw_obj), 1.0)
                if verbose; println("Gauss-warmup converged at iter $gw_iter"); end
                break
            end
            prev_gw_obj = gw_obj
        end

        theta .= gw_otheta
    end

    otheta = copy(theta)
    # θ̂ REPORTED TO THE USER: the smoothing parameters β was actually fitted
    # under. Three θ vectors coexist in this loop and they are NOT
    # interchangeable:
    #   theta     — the newest Fellner-Schall PROPOSAL. Computed at the very
    #               end of each iteration and only tested at the start of the
    #               next one, so at loop exit it is always untested and may
    #               have been explicitly REJECTED by the accept block below.
    #   otheta    — the θ the NEXT PCLS step will use. The `f01 < obj_prev`
    #               branch advances it to `theta` AFTER fitting β at the old
    #               value, so it too can disagree with β at exit.
    #   theta_fit — the θ whose penalty B(θ) actually produced the current β.
    #               This is the only one for which β is the penalized MLE, so
    #               it is what `smoothing_params` and every θ-dependent
    #               quantity in the final block (B_final, edf, objective,
    #               V_beta, sigma2, the LAML criterion) are computed at.
    # Reporting `theta` instead made λ̂ and β̂ mutually inconsistent: refitting
    # β at the reported λ moved it by up to 78% of ‖β‖ on a CONVERGED
    # two-λ Lotka-Volterra fixture (measured), and the reported objective was
    # 7.3e7 against a data loss of 0.89 because the penalty was evaluated at a
    # λ the coefficients had never seen.
    theta_fit = copy(theta)
    prev_obj = Inf  # Track penalized objective for convergence
    prev_data_loss = Inf  # Track data loss for non-Gaussian convergence

    # Honest convergence reporting (see PSMSolution docs): defaults describe
    # loop exhaustion; the breaks below overwrite them with the actual outcome.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    laml_failures = 0

    for iter in 0:(maxiters-1)
        conv_iters = iter + 1
        # Adapt GP kernel hyperparameters to the evolving fit (before the
        # model evaluation so f/J/W below are consistent with the new kernel)
        if iter >= alg.warmup
            _adapt_gp_approximators!(prob, beta)
        end
        # Re-evaluate model + Jacobian
        f_vec_new, _ = try; eval_model(beta); catch e
            if verbose; println("Iter $iter: simulation failed ($e)"); end
            conv_reason = :early_break
            break
        end
        f_vec .= f_vec_new
        compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                          jac=alg.jac, fd_cfg=fd_cfg)

        # Compute IRLS weights from current predictions
        w_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)

        # Form pseudodata z = (z − η) + J*β. The working residual is the
        # family's, not `y − f`: they coincide for Gaussian/Poisson/NegBin
        # (bit-identical) and diverge for TruncatedNormal — see
        # `_working_residual`. The Gaussian warm-start above deliberately
        # keeps `y − f`: it IS a Gaussian solve whatever the declared family.
        z_pseudo = _working_residual(prob.likelihood, y_vec, f_vec, w_vec) .+ J * beta

        # PCLS with current (accepted) θ
        a0_pcls, B_old, fac0 = pcls_step(J, z_pseudo, otheta, w_irls)
        a0, f01 = step_contract(beta, a0_pcls, B_old, fac0)

        stop = false
        obj_prev = penalized_objective(beta, build_B(otheta))

        if iter > 0 && m > 0
            # PCLS with new θ (from LAML)
            a1_pcls, B_new, fac1 = pcls_step(J, z_pseudo, theta, w_irls)
            a1, f11 = step_contract(beta, a1_pcls, B_new, fac1)

            f10 = penalized_objective(beta, B_new)

            # Compare old-θ step vs new-θ step using DATA LOSS (not penalized
            # objective).  Penalized objective is biased: lower θ → lower
            # penalty → lower objective even if the fit is worse.  Data loss
            # is θ-independent and gives an unbiased comparison.
            dl_a0 = -log_likelihood(prob.likelihood, y_vec,
                        (try; first(eval_model(a0)); catch; f_vec; end), w_vec)
            dl_a1 = -log_likelihood(prob.likelihood, y_vec,
                        (try; first(eval_model(a1)); catch; f_vec; end), w_vec)
            dl_curr = -log_likelihood(prob.likelihood, y_vec, f_vec, w_vec)

            if f11 < f10 && dl_a1 <= dl_a0
                # New theta + step is best (data loss confirms)
                f1_vec, _ = try; eval_model(a1); catch; (f_vec, nothing); end
                beta .= a1
                f_vec .= f1_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                                  jac=alg.jac, fd_cfg=fd_cfg)
                otheta .= theta
                theta_fit .= theta      # β was fitted under B(theta)
            elseif f01 < obj_prev
                # Old theta step improved at old theta
                f0_vec, _ = try; eval_model(a0); catch; (f_vec, nothing); end
                beta .= a0
                f_vec .= f0_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                                  jac=alg.jac, fd_cfg=fd_cfg)
                # a0 came from PCLS at the OLD θ, so record it BEFORE the
                # otheta update below moves otheta onto the new proposal.
                theta_fit .= otheta
                # Also accept new theta if it didn't make data loss worse
                if dl_a1 < dl_curr
                    otheta .= theta
                end
            elseif f11 < f10
                # New theta step improved within new theta's metric
                f1_vec, _ = try; eval_model(a1); catch; (f_vec, nothing); end
                beta .= a1
                f_vec .= f1_vec
                compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                                  jac=alg.jac, fd_cfg=fd_cfg)
                otheta .= theta
                theta_fit .= theta      # β was fitted under B(theta)
            else
                # No improvement from either: β is unchanged, so theta_fit
                # stays on whatever θ produced it.
                if iter >= 10
                    stop = true
                end
                theta .= otheta
            end
        else
            # First iteration (or no penalized blocks): accept a0. Note this
            # branch does NOT update otheta — theta_fit therefore records
            # otheta, the θ a0 was actually stepped under.
            f0_vec, _ = try; eval_model(a0); catch; (f_vec, nothing); end
            beta .= a0
            f_vec .= f0_vec
            compute_jacobian!(J, prob, beta, f_vec, n_times, n_obs; dam=dam,
                              jac=alg.jac, fd_cfg=fd_cfg)
            theta_fit .= otheta     # a0 came from PCLS at otheta
        end

        # Track penalized objective for convergence monitoring.
        #
        # `theta`, NOT `theta_fit`, ON PURPOSE — and this is the one place in
        # the loop where that is true. F1 moved every REPORTED θ-dependent
        # quantity onto `theta_fit` (the θ β was actually fitted under) and
        # deliberately left this monitor alone. Tested here, in isolation,
        # on the 18-fixture characterization sweep: scoring at `theta_fit`
        # is a REGRESSION, not the completion of F1.
        #
        # The reason is that `curr_obj` is not a report, it is a STOPPING
        # SIGNAL, and it is the only term in the convergence test that can
        # see the smoothing parameters still moving. `theta` carries the
        # newest Fellner-Schall proposal, so B(theta) keeps changing while
        # smoothing selection is still working even when β has settled;
        # `theta_fit` lags a step behind and goes flat first. Freezing the
        # monitor on `theta_fit` therefore fires `:converged_tol` BEFORE FS
        # has advanced λ off its initialization.
        #
        # Measured (jac defaults, same fixtures, only this line changed):
        # the loop stopped earlier on 6 of 18 fixtures — exponential-growth
        # B-spline 9→7 iters, ODEProblem-converted growth 9→7, NegBin
        # logistic 11→7, two-λ Lotka-Volterra 18→7, floor-kink map 32→13,
        # clamp-kink map 22→20 — and three fits that are stationary today
        # became non-stationary stalls whose λ̂ never left the default
        # 1/tr(S). Worst case, two-λ Lotka-Volterra: EDF 4.00 → 11.99 and
        # λ̂ [1.9e7, 3.1e7] → [1.569e-4, 1.569e-4] (i.e. exactly the
        # initialization), with `converged == true` in both cases. Its
        # stationarity residual went 5.3e-6 → 0.9997.
        curr_obj = penalized_objective(beta, build_B(theta))

        if verbose && (iter <= 4 || iter % 10 == 0)
            curr_data_ss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
            println("Iter $iter: obj=$(round(curr_obj, sigdigits=6)), " *
                    "SS=$(round(curr_data_ss, sigdigits=6)), " *
                    "θ=$(round.(theta, sigdigits=3))")
        end

        # Check convergence: relative change in penalized objective AND data fit.
        # For non-Gaussian likelihoods, the penalized objective can appear stable
        # (large |obj| makes relative tolerance easy to meet) while the data fit
        # is still poor.  Require both objective stability AND small relative
        # change in data loss.
        # Don't converge before warmup is complete — the smoothing parameters
        # haven't been optimised yet and the objective may improve further.
        min_conv_iter = max(3, alg.warmup + 3)
        curr_data_loss = sum((y_vec[i] - f_vec[i])^2 * w_vec[i] for i in 1:n_data)
        # alg.tol governs the penalized-objective test (as documented); the
        # data-loss test uses a proportionally looser threshold.
        obj_stable = abs(curr_obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
        dl_stable = prev_data_loss < Inf &&
                    abs(curr_data_loss - prev_data_loss) <
                        100 * alg.tol * max(prev_data_loss, 1.0)
        if iter >= min_conv_iter && obj_stable && dl_stable
            if verbose; println("Converged at iter $iter (objective stable)"); end
            conv_converged = true
            conv_reason = :converged_tol
            break
        end
        prev_obj = curr_obj
        prev_data_loss = curr_data_loss

        if stop && iter >= min_conv_iter
            if verbose; println("Converged at iter $iter (no improvement)"); end
            conv_converged = true
            conv_reason = :plateau
            break
        end

        # Re-estimate smoothing parameters via LAML.
        #
        # WARM START FROM `otheta` — the smoothing parameters the search has
        # ACCEPTED — not from `theta`, the last (possibly REJECTED) proposal.
        #
        # Warm-starting from the proposal is self-referential, and on the
        # rejection path it is degenerate. `estimate_smoothing_params` leaves
        # its Fellner–Schall phase as soon as one internal step moves log λ by
        # less than its 1e-6 tolerance, so handing it back its own previous
        # output makes it a NEAR-IDENTITY MAP: it takes one step, sees no
        # movement, and returns what it was given. Measured on the F2 fixture
        # under `jac=:forwarddiff` (exponential growth, one 5-knot B-spline):
        # from iteration 4 onward it returned 31429.207404658788
        # bit-identically on every call, while β went on being fitted at
        # θ = 4.2641e-4 — the 1/tr(S) initialization — a 7.4e7× divergence
        # that repeated unchanged until the loop exited.
        #
        # That freeze is not cosmetic. `curr_obj` above is scored at `theta`
        # on purpose, because it is the only term in the convergence test that
        # can see smoothing selection still moving (see the comment there). A
        # frozen proposal makes that monitor blind, so `:converged_tol` fires
        # while λ̂ still sits on its initialization — and the loop never
        # reaches the `f11 < f10` branch, which is what accepts a new θ once β
        # stops improving at the old one. This is the mechanism behind LAML
        # solves that report `converged == true` with
        # `smoothing_advanced == false`.
        #
        # `otheta` is the θ the next PCLS step will use, so Fellner–Schall is
        # asked the question it exists to answer: given the current working
        # model, what should the CURRENT λ become? This is NOT "restarting
        # from scratch" (that would be `theta_init` — `initial_lambda`, or the
        # 1/tr(S) default); whenever a proposal IS accepted, `otheta == theta`
        # and the warm start is bit-identical to the old behaviour. The two
        # differ only on the rejection path, which is exactly where the old
        # behaviour was degenerate. The Gaussian pre-loop above already warm-
        # starts from its accepted `gw_otheta`; this line was the outlier.
        #
        # `theta_fit` was the other candidate and is subtly wrong here: on the
        # `f01 < obj_prev` branch that ALSO accepts the new θ, `theta_fit`
        # holds the PREVIOUS `otheta`, so warm-starting from it would discard
        # a λ the loop had just accepted. `theta_fit` is a REPORTING concept
        # (the θ at which β̂ is the penalized MLE); `otheta` is the search
        # STATE.
        #
        # NOT changed, deliberately: the `dl_a1 <= dl_a0` / `dl_a1 < dl_curr`
        # vetoes above. Data loss is monotone decreasing in model flexibility,
        # so ANY λ increase raises it and those vetoes are a one-way ratchet
        # that can only ever accept λ DECREASES — measured `dl_a1/dl_curr` at
        # the first rejected iteration of the three fixtures below: 1.0086,
        # 1.0163, 1.00078. (The comment above them says data loss is unbiased
        # because it is θ-independent; the FUNCTION is, but its argmin over β
        # moves toward λ → 0, so it is biased against λ increases exactly as
        # the penalized objective is biased for them.) The correct arbiter of
        # λ is the LAML criterion Fellner–Schall already ascends — but the
        # loop does not need one here, because the `f11 < f10` branch accepts
        # a new θ on its own penalized objective once β is exhausted at the
        # old θ. Restoring a live proposal is what lets that branch be
        # reached; widening the veto is a larger change with no measured need.
        #
        # Measured effect (λ̂ / EDF / `stationarity` / sup|f̂ − truth|, before →
        # after). All three reported `converged = true` with λ̂ frozen bit-
        # identically on its initialization before this change:
        #   exp growth, BSpline(:r,(0,5),5), jac=:forwarddiff, truth r ≡ 0.1
        #     4.2641e-4 → 3.502e4 | 3.222 → 2.000 | 0.391 → 4.95e-7 |
        #     1.447e-3 → 1.311e-4
        #   exp growth, BSpline(:r,(0,5),8), jac=:fd, truth r ≡ 0.1
        #     3.6584e-5 → 1.474e4 | 4.070 → 2.000 | 0.328 → 6.25e-7 |
        #     3.368e-3 → 1.312e-4
        #   logistic, BSpline(:g,(0,60),6), Poisson, truth g(N)=0.3(1−N/50)
        #     1.5691e-4 → 1.831e7 | 5.320 → 2.000 | 0.829 → 1.07e-8 |
        #     7.068e-2 → 2.129e-3
        # In all three the truth lies in null(S), so EDF 2 is the correct
        # answer: the frozen fits were UNDERSMOOTHED, not merely mislabelled.
        w_irls_for_laml = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
        if m > 0 && iter >= alg.warmup
            # sigma2_init caps the FS dispersion during early iterations to
            # prevent runaway smoothing while the fit is still poor; as
            # documented, the cap relaxes (×10 per iteration past warmup)
            # so it cannot permanently bias λ downward.
            s2cap = if alg.sigma2_init === nothing
                Inf
            else
                alg.sigma2_init * 10.0^clamp(iter - alg.warmup, 0, 300)
            end
            theta_new, _ = try
                rho_init = log.(max.(otheta, 1e-20))
                estimate_smoothing_params(J, w_irls_for_laml, w_vec,
                                         y_vec, f_vec, beta,
                                         S_list, uf_offsets, uf_nk, n_p;
                                         family=prob.likelihood,
                                         rho_init=rho_init,
                                         sigma2_max=s2cap,
                                         criterion=alg.criterion,
                                         verbose=verbose)
            catch e
                if verbose; println("LAML failed: $e, keeping theta"); end
                laml_failures += 1
                (copy(theta), NaN)
            end
            theta .= theta_new
        end
    end

    # Collapse onto the θ that β was actually fitted at (see the theta_fit
    # comment above). Everything from here down — B_final, edf, obj_val,
    # V_beta, sigma2, the LAML criterion, the verbose "Final θ" line and
    # `smoothing_params` itself — reads `theta`, so this single assignment is
    # what makes the whole reported solution self-consistent: β̂ IS the
    # penalized MLE at the λ̂ reported next to it.
    #
    # A truncated fit (`converged == false`) is still not stationary at ANY λ
    # — no choice of reported λ can fix that, and `convergence.reason` already
    # says so — but the λ/β PAIRING is now honest in every case.
    theta .= theta_fit

    # Build solution
    p_opt = copy(beta)
    pred = simulate(prob, p_opt)

    # Compute data loss (masked cells — zero weight or NaN datum — are
    # skipped; `0 * NaN = NaN` would otherwise contaminate the total)
    data_loss = 0.0
    n_used = 0          # cells that actually contributed to data_loss
    for j in 1:n_obs, i in 1:n_times
        wv = prob.data_weights[i,j]
        y = prob.data_values[i,j]
        _usable(y, wv) || continue
        data_loss += wv * (y - pred[i,j])^2
        n_used += 1
    end

    # EDF from hat matrix
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        f_vec[k] = pred[ti, oi]
        k += 1
    end
    compute_jacobian!(J, prob, p_opt, f_vec, n_times, n_obs; dam=dam,
                      jac=alg.jac, fd_cfg=fd_cfg)

    B_final = build_B(theta)
    W_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)
    JWJ = J' * Diagonal(W_irls) * J
    H_final = JWJ + B_final
    maxd = maximum(abs.(diag(H_final)))
    for i in 1:n_p
        H_final[i,i] += 1e-12 * maxd + 1e-15
    end
    edf = try
        tr(cholesky(Symmetric(H_final)) \ JWJ)
    catch
        tr(H_final \ JWJ)
    end

    pen_ss = dot(p_opt, B_final * p_opt)
    obj_val = 0.5 * (data_loss + pen_ss)

    # Build ComponentArray for nice parameter access
    uf_syms = Symbol[a.name for a in prob.approximators]
    uf_vals = Vector{Float64}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(uf_vals, Float64.(p_opt[offset+1:offset+np]))
        offset += np
    end
    params = ComponentArray(NamedTuple{Tuple(uf_syms)}(Tuple(uf_vals)))

    # Build unknown function evaluators for the solution
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = p_opt[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    if verbose
        println("\nFinal: data_loss = $(round(data_loss, sigdigits=6)), " *
                "penalty = $(round(pen_ss, sigdigits=6)), " *
                "EDF = $(round(edf, digits=2))")
        println("Final θ: ", [round(t, sigdigits=4) for t in theta])
    end

    # Compute Bayesian posterior covariance V_β = (J'WJ + S^λ)⁻¹
    # This gives "across-the-function" CIs with near-nominal coverage
    # (Nychka 1988, Wood 2006 §4.8)
    V_beta = try
        inv(cholesky(Symmetric(H_final)))
    catch
        try; inv(Symmetric(H_final)); catch; nothing; end
    end

    # Estimate σ² for Gaussian (needed for CI scaling)
    sigma2_hat = if prob.likelihood isa Gaussian
        # Divide the MASKED data_loss by the number of cells that actually
        # contributed to it, not by every cell in the data matrix: using
        # n_data biased σ̂² low (and CIs narrow) in proportion to the amount
        # of missing/zero-weight data. Matches profile_likelihood_solver.jl.
        data_loss / max(n_used - edf, 1.0)
    else
        1.0  # non-Gaussian: V_β already on natural scale
    end

    # LAML criterion value at the returned fit (a report, not an extra
    # optimization step): the same `laml_objective` the smoothing update
    # maximizes, evaluated at the final (β̂, μ̂, J, W̃, θ̂) — profiled REML
    # for Gaussian, the full Laplace criterion for the other families.
    # Reported under BOTH criteria (under :working for non-Gaussian it is a
    # diagnostic, not the quantity the FS update was calibrated to). NaN
    # when no penalized term exists (m == 0) or the evaluation fails.
    # `laml_objective` returns (V, H_laml, S_lambda, sigma2). H_laml and
    # sigma2 are exactly the quantities `laml_gradient` needs, so taking all
    # four here makes the reported criterion value and the stationarity
    # residual below two readings of the SAME evaluation — no second, subtly
    # different assembly of H or σ̂².
    laml_value, laml_H, laml_sigma2 = if m > 0
        try
            V, Hl, _, s2l = laml_objective(prob.likelihood, p_opt, J, W_irls,
                                           w_vec, y_vec, f_vec, S_list,
                                           uf_offsets, uf_nk,
                                           log.(max.(theta, 1e-300)), n_p)
            (V, Hl, s2l)
        catch
            (NaN, nothing, NaN)
        end
    else
        (NaN, nothing, NaN)
    end

    # ─── Stationarity of the smoothing parameters ────────────────────
    # `converged` above is a STABILITY test (the penalized objective and the
    # data loss stopped moving). Stability is not stationarity: an IRLS
    # search can stop moving because it reached the optimum, OR because the
    # steps it can construct from a noisy finite-difference Jacobian all fail
    # the accept test, OR because Fellner-Schall never engaged at all. Only
    # the first is convergence. `stationarity` distinguishes them by asking
    # the criterion directly.
    #
    #   ∂V/∂ρ_k = -½λ_k β̂'S_kβ̂/σ̂² + ½ r_k - ½ tr(H⁻¹λ_kS_k)
    #
    # is the exact gradient of the LAML criterion w.r.t. ρ = log λ at the
    # returned (β̂, λ̂). Dividing by ½ r_k = ½ rank(S_k) makes the residual
    # dimensionless and comparable across blocks, bases, resolutions and
    # data scales: the ½r_k and ½tr(H⁻¹λ_kS_k) terms are both bounded by
    # ½ r_k, so the ratio measures the gradient against the natural scale of
    # the block. It is NOT bounded by 1 — the β̂'S_kβ̂/σ̂² term has no upper
    # bound, and the largest value observed across this package's suite is
    # 8.72. `stationarity` is the max over blocks of that ratio.
    #
    # DELIBERATELY REPORTED AS A NUMBER, NOT A PASS/FAIL FLAG. An earlier
    # draft of this block carried a companion `stationary::Bool` at a
    # threshold of 0.1. It was dropped after measuring the residual on all
    # LAML solves the test suite performs (most recently the 163 solves of
    # the post-F6/F8 suite): the distribution is an unbroken continuum, not
    # two clusters. Quantiles are p25 = 1.6e-7, p50 = 8.5e-6, p75 = 1.6e-2,
    # p90 = 0.29, max = 8.72, and across the whole decision-relevant region
    # (1e-3 to 3) the largest ratio between consecutive sorted values is
    # 1.52 below 1 and 2.98 across the whole region — nothing resembling
    # the orders-of-magnitude separation a threshold would need, i.e. there
    # is no gap anywhere a threshold could sit. A cutoff at 0.1 would have
    # flagged 23 of 163 solves (14.1%), splitting a continuum of otherwise
    # ordinary `:converged_tol` fits spanning the same families and
    # Jacobian backends. A flag that fires on one fit in seven teaches
    # users to ignore it, which would destroy the value of the honest
    # signal this block exists to add.
    #
    # (During the original 151-solve calibration a partial sample of 94
    # solves appeared to show a 3.2x gap around 0.1; the remaining 57
    # filled it in completely. Recorded because it is the reason not to
    # calibrate a gate on a subsample.)
    #
    # Users who need a gate should threshold the float for their own problem
    # class — and should read the σ̂²→0 caveat in the LAML docstring first.
    stationarity = if m == 0
        # No penalized block ⇒ no smoothing parameter to be stationary in.
        # The empty gradient's max-norm is 0 by convention, not by luck.
        0.0
    elseif laml_H === nothing
        NaN
    else
        try
            g = laml_gradient(prob.likelihood, p_opt, S_list, uf_offsets,
                              uf_nk, log.(max.(theta, 1e-300)), n_p,
                              laml_H, laml_sigma2)
            maximum(abs(g[l]) / max(0.5 * _rank_penalty(S_list[l]), 1e-8)
                    for l in 1:m)
        catch
            NaN
        end
    end
    # Did smoothing selection ever move λ off its initialization? `false`
    # means the reported λ̂ IS the initial value: either every FS proposal
    # was rejected by the accept block, or FS never ran (m == 0, or
    # maxiters ≤ warmup). See the LAML docstring.
    smoothing_advanced = m > 0 && any(
        abs(theta[l] - theta_init[l]) >
            1e-10 * max(abs(theta_init[l]), 1e-300) for l in 1:m)

    # SAY SO OUT LOUD. A solve that finishes with smooth terms present but
    # λ̂ still sitting on its 1/tr(S) initialization has not done smoothing
    # selection at all, and everything downstream that depends on λ̂ — the
    # EDF, the posterior covariance `V_beta`, the intervals built from it —
    # describes a model nobody chose. Until now that was visible only to a
    # caller who went looking for `convergence.smoothing_advanced`.
    #
    # This is not hypothetical and it is not rare. On the package's own F6
    # fixture under the DEFAULT `jac=:fd`, k = 8 knots lands here while
    # k = 6, 7, 9, 10 do not: measured edf 4.074 against the correct 2.000
    # and a sup error of 3.39e-3 against 1.31e-4 — 26x worse — on a problem
    # whose truth lies in the penalty null space. Which knot counts trip is
    # platform-dependent (macOS CI trips k = 8; this machine trips k = 8 too
    # under --check-bounds=yes, and k = 9 without it), because the mechanism
    # is Fellner-Schall proposals being rejected when the finite-difference
    # Jacobian is noisy. Under `jac=:forwarddiff` every k in 6:12 succeeds.
    #
    # `:forwarddiff` is NOT recommended unconditionally in the message: it is
    # ~2.5x faster than `:fd` at 8 parameters but ~25x SLOWER at 12 (measured
    # 3.86 s vs 0.152 s), so it is the right escape hatch for a fit that has
    # actually hit this, not a blanket default.
    if m > 0 && !smoothing_advanced
        @warn "LAML: smoothing selection never moved λ̂ off its " *
              "initialization, so the reported λ̂, EDF and posterior " *
              "covariance describe the INITIAL smoothing, not a selected " *
              "one. Every Fellner–Schall proposal was rejected (or the " *
              "iteration budget was spent before any ran). This happens " *
              "when the working-model Jacobian is too noisy for the " *
              "proposals to be accepted; try `jac=:forwarddiff`, more " *
              "`maxiters`, or a different knot count. See " *
              "`convergence.smoothing_advanced`." maxlog=1
    end

    convergence_info = (V_beta=V_beta, sigma2=sigma2_hat,
                        converged=conv_converged, iterations=conv_iters,
                        reason=conv_reason, laml_failures=laml_failures,
                        criterion=alg.criterion, laml=laml_value,
                        stationarity=stationarity,
                        smoothing_advanced=smoothing_advanced)

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals, convergence_info)
end
