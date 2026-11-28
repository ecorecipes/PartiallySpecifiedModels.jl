# ─── Algorithm types ───────────────────────────────────────────────

const _FixedLambdaSpec = Union{Nothing, Float64, Vector{Union{Nothing, Float64}}}

function _normalize_fixed_lambda(fl)
    fl === nothing && return nothing
    if fl isa Real
        v = Float64(fl)
        (isfinite(v) && v > 0) ||
            throw(ArgumentError("LAML: fixed_lambda must be positive and finite"))
        return v
    elseif fl isa AbstractVector
        isempty(fl) && throw(ArgumentError("LAML: fixed_lambda vector must be non-empty"))
        out = Vector{Union{Nothing, Float64}}(undef, length(fl))
        for (i, x) in enumerate(fl)
            if x === nothing
                out[i] = nothing
            elseif x isa Real
                v = Float64(x)
                (isfinite(v) && v > 0) || throw(ArgumentError(
                    "LAML: fixed_lambda[$i] must be positive and finite (or nothing), got $x"))
                out[i] = v
            else
                throw(ArgumentError(
                    "LAML: fixed_lambda[$i] must be Real or nothing, got $(typeof(x))"))
            end
        end
        return out
    else
        throw(ArgumentError(
            "LAML: fixed_lambda must be nothing, a positive Real, or a vector of (Real | nothing)"))
    end
end

"""
    LAML(; maxiters=100, tol=1e-6, verbose=false, initial_lambda=nothing,
           warmup=3, sigma2_init=nothing, criterion=:working, jac=:auto,
           fixed_lambda=nothing, exploratory_maxiters=5_000)

Laplace Approximate Marginal Likelihood algorithm.
Equivalent to REML for Gaussian data.
Uses Fellner-Schall + exact Wood (2011) Theorem 1 Newton refinement for smoothing
parameter estimation.

# Keyword arguments
- `maxiters::Int=100`: maximum IRLS+LAML iterations
- `tol::Float64=1e-6`: convergence tolerance on penalized objective. An
  old-smoothing step below this precision no longer blocks an available
  finite step that descends the proposed smoothing's own objective.
- `verbose::Bool=false`: print iteration diagnostics
- `initial_lambda::Union{Nothing,Float64}=nothing`: initial smoothing parameter
  for all terms.  Default (`nothing`) uses the data-driven `θ = 1/tr(S)` per
  term (values below λ ≈ 4.5e-5 are floored there when LAML starts, to keep
  the criterion away from its flat tiny-λ region).  For strongly nonlinear
  problems, a higher value (e.g. `10.0`) combined with `warmup` helps the
  IRLS converge to a good basin before LAML refinement.
- `warmup::Int=3`: number of IRLS iterations to run with fixed smoothing before
  engaging LAML estimation.  Allows the coefficient estimates to stabilise
  before the smoothing parameters are adapted.  Increase for strongly
  nonlinear models (e.g. `warmup=10`).
- `fixed_lambda`: opt-in coefficient fitting at supplied positive, finite
  smoothing weights. Accepts `nothing` (default, automatic selection), a
  positive `Real` shared across all penalty blocks, or a `Vector{Union{Nothing,Real}}`
  with one entry per penalty block (fixing the numeric blocks while estimating
  the `nothing` blocks via LAML). Cannot be combined with `initial_lambda` when
  all blocks are fixed. When all blocks are fixed, `smoothing_fixed=true`
  distinguishes this intentional choice from stalled selection.
- `sigma2_init::Union{Nothing,Float64}=nothing`: cap on the profiled σ² used in
  the Fellner-Schall smoothing update during the warmup phase.
- `criterion::Symbol=:working`: smoothing-parameter selection criterion for
  **non-Gaussian** likelihoods (`:working` PQL-flavored default vs `:laplace`
  full Laplace-approximate marginal likelihood of Wood 2011 / Wood, Pya &
  Säfken 2016). For `GaussianLikelihood` data `:laplace` reduces exactly to
  profiled REML.
- `jac::Symbol=:auto`: backend for the working-model Jacobian (`compute_jacobian!`).
  - `:auto` (default): tries forward-mode AD (`ForwardDiff`) once on the first
    iteration and permanently latches to `:fd` if the dynamics are not
    Dual-compatible; exact and typically 5–19× faster than `:fd`.
  - `:forwarddiff`: force ForwardDiff with per-iteration FD fallback on numerical failure.
  - `:fd`: adaptive central finite differences (2·n_p model solves per IRLS step).
- `exploratory_maxiters::Union{Nothing,Int}=5_000`: ODE integrator step cap applied
  during exploratory trial steps to prevent catastrophic hangs on wild trial
  coefficients (automatically bypassed on the final solution evaluation and
  whenever the user explicitly specifies `:maxiters` in `ode_kwargs`; pass
  `nothing` to disable).

# Convergence info
`sol.convergence` is a NamedTuple `(V_beta, sigma2, converged, iterations,
reason, laml_failures, criterion, laml, stationarity,
smoothing_advanced)`: the posterior covariance and σ̂² used
by [`confidence_band`](@ref), plus the standard honest-convergence keys (see
[`PSMSolution`](@ref)), `laml_failures::Int` (the number of iterations in
which the LAML smoothing-parameter update failed and θ was kept),
`criterion::Symbol` (which selection criterion ran), and `laml::Float64` (the
LAML criterion value V at the returned fit — profiled REML for Gaussian, the
full Laplace criterion above for other families; a MAXIMIZED quantity, `NaN`
when no penalized term exists or the evaluation fails).

## Stationarity diagnostics: what `converged` does NOT tell you

`converged` is a STABILITY test — the penalized objective and the data loss
stopped changing. A fit stops changing for several distinct reasons and
`converged` cannot tell them apart: it is `true` for a genuine optimum, for a
search that stalled because a non-smooth model made the finite-difference
Jacobian too noisy to build an accepted step from, and for a fit whose
smoothing parameters never moved at all. The two keys below separate those
cases. **They are additive diagnostics; they do not change `converged`,
`reason`, or any fitted quantity, and they are present on every LAML solution,
converged or not.** Neither they nor `converged` can detect a poor local
minimum of a randomly initialized approximator. For that, refit from several
initializations with [`MultiStart`](@ref).

- `smoothing_advanced::Bool`: whether λ̂ ever moved off its initialization
  (`initial_lambda`, or the data-driven `1/tr(S)` default). `false` means the
  reported λ̂ **is** the initial value — smoothing selection never took
  effect, because every Fellner-Schall proposal was rejected by the step
  accept test or because it never ran (no penalized term, or
  `maxiters ≤ warmup`). It is `false` for a model with no penalized term.
  This is a plain fact about the search, not a statistical judgement — the
  comparison against the initialization uses a 1e-10 RELATIVE tolerance,
  which guards against pure round-off rather than being a calibrated
  cutoff, so there is no magnitude to tune. A `false` here with
  `converged == true` is the strongest available signal that the reported λ̂
  carries no information from the data. It does NOT mean the fit is wrong —
  the initialization can happen to be reasonable — only that nothing chose
  it. In an earlier test-suite characterization this was `false` for 17 of 163 LAML
  solves, 8 of which nonetheless report `converged == true`. It is genuinely
  complementary to `stationarity` rather than a proxy for it: one of those 17
  is an unpenalized model whose residual is `0.0` by convention, and the
  other 16 span residuals from 0.016 to 8.7, so a fit can sit at a perfectly
  ordinary residual and still never have moved its λ̂.
- `ridge::Bool`: the objective converged but one more PCLS step from the reported
  fit would still move the parameters by more than `1e-3` relative — the fit stopped on a flat
  direction. The fitted functions are usable; individual coefficients and
  their covariance are not pinned.
- `final_parameter_step::Float64`: relative size of that probe step. Reported for every
  fit; `ridge` itself is evaluated only for penalized approximators — an
  unpenalized one (`NeuralApproximator`, `KANApproximator`) has no pinned
  coefficients by construction, so it is never flagged.

- Proposal deferral is BOUNDED: while the step at the previous λ̂ still
  improves the penalized objective by more than `tol`, a live Fellner–Schall
  proposal is deferred in its favour — but for at most 5 consecutive
  iterations, after which the proposal is taken. Without the bound a fit
  whose coefficients converge slowly at the initial λ (the copepod vignette:
  45 coefficients, EDF 41 at λ₀) re-proposed and deferred the same λ̂ for
  ~130 iterations (18 with the bound, same fixed point). `verbose=true`
  prints which branch each iteration took, and
  `convergence.deferral_cap_hits` counts how often the bound decided.
- `stationarity::Float64`: how far the returned λ̂ is from a stationary point
  of the LAML criterion, as
  `maxₖ |∂V/∂ρₖ| / (½·rank(Sₖ))` with `ρ = log λ`, evaluated at the returned
  `(β̂, λ̂)`. The normalization makes it dimensionless and comparable across
  bases and resolutions — and, for Gaussian data, across data scales; under
  a non-Gaussian family the `λₖβ̂'Sₖβ̂` term carries no `σ̂²` divisor and so
  does scale with the data. The `½·rank(Sₖ)` and
  `½·tr(H⁻¹λₖSₖ)` terms of `∂V/∂ρₖ` are both bounded by `½·rank(Sₖ)`, so the
  ratio measures the gradient against the natural scale of the block, and `0`
  is exactly stationary. It is **not** bounded above by 1 — the
  `λₖβ̂'Sₖβ̂/σ̂²` term has no upper bound, and the largest value observed
  across this package's test suite is 8.7. `0.0` when the model has no
  penalized term (nothing to be stationary in); `NaN` when the criterion or
  its gradient could not be evaluated.

  It does NOT measure whether the fit is *good*, or whether λ̂ is at a global
  rather than a local optimum — it is a first-order condition on the
  smoothing parameters, evaluated at the returned β̂, not a gradient through
  the ODE solve. It is nonzero when λ̂ is off its conditional optimum AND
  when β̂ is off the penalized least-squares solution at λ̂ (at a true fixed
  point of the IRLS loop those coincide, so it tests both).

  One further caveat for NON-GAUSSIAN families under the default
  `criterion = :working`: the residual is the gradient of the FULL LAPLACE
  criterion (that is what `laml_gradient` returns for those families), while
  the Fellner-Schall update was calibrated on the Gaussian working model. A
  nonzero value can therefore reflect that mismatch rather than a stalled
  search — the same caveat the `laml` key already carries. Under
  `criterion = :laplace` the two agree and the residual is unambiguous.

### How to read `stationarity`

**There is deliberately no universal `stationary::Bool`.** An earlier
characterization of 163 LAML solves found an unbroken continuum rather than two
clusters. Its quantiles are p25 = 1.6e-7, p50 = 8.5e-6, p75 = 1.6e-2,
p90 = 0.29, max = 8.7, and across the whole decision-relevant region (1e-3
to 3) the largest ratio between consecutive sorted values is 1.52 below 1
and 2.98 across the whole region — nothing resembling the
orders-of-magnitude separation a threshold would need, so there is
no gap anywhere a threshold could sit. A cutoff at 0.1 would have flagged 23
of those 163 solves (14.1%), splitting a continuum of otherwise ordinary
`:converged_tol` fits.

Treat it as a magnitude, calibrated against your own problem class rather
than an absolute scale. In that characterization, 95 of 163 solves were
below 1e-4. These are historical observations, not bounds or promises for
current solver versions. In particular, a formerly stalled `floor`-based
map improves when sub-tolerance old-smoothing steps no longer veto a new
proposal. A non-smooth evaluator need not produce a large smoothing
residual, and a small residual does not establish a good nonlinear fit.

A residual orders of magnitude larger than comparable fits of yours is a
signal to investigate. Causes can include a noisy working-model Jacobian,
a stalled nonlinear search, or too few iterations. `jac=:forwarddiff` can
help when the dynamics are Dual-compatible; it is neither a guarantee for
non-smooth problems nor an unconditional performance recommendation.

**One regime where it is ill-conditioned, and must not be read at all:** for
`Gaussian` data the gradient contains `−½λₖβ̂'Sₖβ̂/σ̂²` with `σ̂²` the profiled
REML scale. On a fit that interpolates its data — noiseless synthetic
fixtures, most obviously — `σ̂²` underflows toward zero and that term becomes
a ratio of two near-zero quantities, so the residual carries no information
about the search and can read large on a fit that is exact to round-off.
Measured: the same noiseless growth fixture reads 0.91 under both `jac=:fd`
and `jac=:forwarddiff` — with λ̂ agreeing to machine precision, EDF to ~1e-4
relative, `σ̂² ≈ 2e-19` and a data loss of ~3e-18 — an alarming-looking
residual on an essentially exact fit. This is less a defect in the
residual than a true statement about the fit — with no residual variance,
REML does not identify λ and λ̂ is set by round-off — but it does mean
`stationarity` is uninformative there. Judge it only on fits with meaningful
residual variance.

`converged && smoothing_advanced`, plus a `stationarity` in line with
comparable fits, is the combination that means what users generally read
`converged` alone to mean.

Gaussian penalized fits save final working information in
`convergence.smoothing_state` for [`smoothing_covariance_correction`](@ref).
`confidence_band(sol, prob; unconditional=true)` uses that opt-in analytic
correction, including mean and covariance-root terms. The ordinary stored
`V_beta` and the default conditional bands are unchanged. This correction
is local-Gaussian, holds coefficient dispersion fixed, and requires
identifiable coefficient and log-smoothing curvature (or an explicitly
supplied log-smoothing covariance for fixed/external smoothing).
"""
struct LAML
    maxiters::Int
    tol::Float64
    verbose::Bool
    initial_lambda::Union{Nothing,Float64}
    warmup::Int
    sigma2_init::Union{Nothing,Float64}
    criterion::Symbol
    jac::Symbol
    fixed_lambda::_FixedLambdaSpec
    exploratory_maxiters::Union{Nothing,Int}
end

function LAML(; maxiters::Int=100, tol::Float64=1e-6, verbose::Bool=false,
                initial_lambda::Union{Nothing,Float64}=nothing,
                warmup::Int=3,
                sigma2_init::Union{Nothing,Float64}=nothing,
                criterion::Symbol=:working,
                jac::Symbol=:auto,
                fixed_lambda=nothing,
                exploratory_maxiters::Union{Nothing,Int}=5_000)
    criterion in (:working, :laplace) ||
        throw(ArgumentError("LAML: criterion must be :working or :laplace " *
                            "(got $(repr(criterion)))"))
    jac in (:auto, :fd, :forwarddiff) ||
        throw(ArgumentError("LAML: jac must be :auto, :fd, or :forwarddiff " *
                            "(got $(repr(jac)))"))
    fl = _normalize_fixed_lambda(fixed_lambda)
    if fl isa Float64 && initial_lambda !== nothing
        throw(ArgumentError("LAML: use fixed_lambda or initial_lambda, not both"))
    elseif fl isa Vector && all(!isnothing, fl) && initial_lambda !== nothing
        throw(ArgumentError("LAML: use fixed_lambda or initial_lambda, not both"))
    end
    exploratory_maxiters === nothing || exploratory_maxiters >= 1 ||
        throw(ArgumentError("LAML: exploratory_maxiters must be >= 1 or nothing"))
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, jac, fl, exploratory_maxiters)
end

# Backward-compatible positional constructors.
LAML(maxiters::Int, tol::Float64, verbose::Bool,
     initial_lambda::Union{Nothing,Float64}, warmup::Int,
     sigma2_init::Union{Nothing,Float64}, criterion::Symbol, jac::Symbol,
     fixed_lambda::_FixedLambdaSpec) =
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, jac, fixed_lambda, 5_000)

LAML(maxiters::Int, tol::Float64, verbose::Bool,
     initial_lambda::Union{Nothing,Float64}, warmup::Int,
     sigma2_init::Union{Nothing,Float64}, criterion::Symbol, jac::Symbol) =
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, jac, nothing, 5_000)

LAML(maxiters::Int, tol::Float64, verbose::Bool,
     initial_lambda::Union{Nothing,Float64}, warmup::Int,
     sigma2_init::Union{Nothing,Float64}, criterion::Symbol) =
    LAML(maxiters, tol, verbose, initial_lambda, warmup, sigma2_init,
         criterion, :auto, nothing, 5_000)

"""
    CollocationLAML(; kwargs...)

Collocation-based LAML solver using the generalized profiling / parameter
cascading approach of Ramsay et al. (2007).

Instead of integrating the ODE at each step (as `LAML` does), this solver
represents the **state trajectories as free parameters** and penalizes
deviation from the ODE.  A continuation schedule gradually increases the
ODE compliance penalty `λ_ode`, transitioning from a data-fitting problem
(flexible states) to a model-constrained problem (states satisfy ODE).

This approach is much more robust for **highly nonlinear or oscillatory
models** (e.g., Lotka–Volterra) where the standard IRLS linearization
fails because the ODE trajectory is extremely sensitive to parameter
changes.  See Fasiolo, Pya & Wood (2016), Statistical Science 31(1).

The data-fidelity block honors `prob.likelihood` (all built-in families
and `CustomLikelihood`) via PIRLS working weights on the identity link,
with a deviance-scale objective and a Pearson-dispersion Fellner–Schall
scale mirroring `LAML`'s default `:working` criterion; the ODE-compliance
rows stay Gaussian in the derivative residuals. `Gaussian` data follow
the historical pure-least-squares path unchanged. (Before this, the
likelihood was silently ignored — every family was fitted by Gaussian
least squares.)

# Keyword arguments
- `maxiters::Int=50`: IRLS iterations per continuation level
- `tol::Float64=1e-6`: convergence tolerance
- `verbose::Bool=false`: print diagnostics
- `lambda_ode_start::Float64=0.01`: initial ODE compliance penalty
- `lambda_ode_end::Float64=1e4`: final ODE compliance penalty. **Capped at
  100 for discrete-time problems** (`prob.discrete`), where the "ODE"
  residual is a one-step map residual coupling only consecutive pairs, so a
  1e4 weight overwhelms the data block without improving the dynamics fit.
  `sol.convergence.lambda_ode_final` reports the capped endpoint, not this
  setting.
- `n_continuation::Int=8`: number of log-spaced continuation levels
- `sigma2_init::Union{Nothing,Float64}=nothing`: σ² cap for Fellner-Schall
- `jac::Symbol=:fd`: backend for the differentiated parts of the
  collocation Jacobian — the pointwise state Jacobian ∂F/∂x, the
  coefficient block ∂F/∂β, and the Fellner–Schall EDF Jacobian of the
  simulated predictions.
  - `:fd` (default, historical behavior, byte-identical): finite
    differences.
  - `:forwarddiff`: forward-mode AD. The failure-sentinel convention is
    preserved EXACTLY (see the comment block in `collocation_solver.jl`):
    the residual at a failed collocation point still holds the large
    `_COLLOC_FAIL_SENTINEL`, the sentinel value itself is NEVER part of
    the differentiated computation (failed points are skipped inside the
    Dual sweep), and the Jacobian rows/columns at failed points are forced
    to zero exactly as the FD path forces them — large residual, zero
    Jacobian. If a Dual-valued sweep fails outright it falls back to the
    FD path for that evaluation (`@debug` note). One behavioral nuance,
    strictly an improvement: at a point whose BASE evaluation succeeds,
    forward mode returns the exact derivative and never perturbs the
    state, so FD's defensive zero column for "perturbation crossed a
    domain boundary" cannot arise (that zero existed only to protect FD
    from its own perturbation).

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The collocation objective
    calls the dynamics through the 4-argument ODE signature and has no way
    to supply the delayed history, so a DDE would silently be fitted as
    though it had no delays. Use [`LAML`](@ref) for DDEs.

# Convergence info
`sol.convergence` is a NamedTuple `(ode_compliance, lambda_ode_final,
converged, iterations, reason, iterations_total, ridge,
final_parameter_step)`. `converged`/`reason`
describe the final continuation level's inner loop, `iterations` counts its
inner iterations, and `iterations_total` accumulates inner iterations across
all continuation levels (see [`PSMSolution`](@ref) for the key taxonomy).
`ridge` and `final_parameter_step` have the [`LAML`](@ref) semantics — one
more UNCONTRACTED Gauss–Newton step on (α, β) from the reported fit, at the
final `lambda_ode_final`, weights and λ̂, measured on the coefficient part β:
`ridge` is `true` when the objective converged but that step would still move
β by more than 1e-3 relative (the fitted functions are usable; individual
coefficients and their covariance are not pinned). Penalized approximators
only; the step is reported for every fit.
`lambda_ode_final` is the ENDPOINT THE CONTINUATION SCHEDULE WAS BUILT TO —
the discrete cap above means it is `min(lambda_ode_end, 100.0)` for
`prob.discrete`, not necessarily the setting you passed. (The schedule's last
level is the round-tripped `exp(log(lambda_ode_final))`, which can differ in
the last bits — `100.00000000000004` for `100.0` — so this is the exact
endpoint rather than the exact multiplier of the final level.)
"""
struct CollocationLAML
    maxiters::Int
    tol::Float64
    verbose::Bool
    lambda_ode_start::Float64
    lambda_ode_end::Float64
    n_continuation::Int
    sigma2_init::Union{Nothing,Float64}
    jac::Symbol
end

function CollocationLAML(; maxiters::Int=50, tol::Float64=1e-6,
                           verbose::Bool=false,
                           lambda_ode_start::Float64=0.01,
                           lambda_ode_end::Float64=1e4,
                           n_continuation::Int=8,
                           sigma2_init::Union{Nothing,Float64}=nothing,
                           jac::Symbol=:fd)
    jac in (:fd, :forwarddiff) ||
        throw(ArgumentError("CollocationLAML: jac must be :fd or " *
                            ":forwarddiff (got $(repr(jac)))"))
    CollocationLAML(maxiters, tol, verbose, lambda_ode_start, lambda_ode_end,
                    n_continuation, sigma2_init, jac)
end

# Backward-compatible positional constructor (pre-jac arity).
CollocationLAML(maxiters::Int, tol::Float64, verbose::Bool,
                lambda_ode_start::Float64, lambda_ode_end::Float64,
                n_continuation::Int,
                sigma2_init::Union{Nothing,Float64}) =
    CollocationLAML(maxiters, tol, verbose, lambda_ode_start, lambda_ode_end,
                    n_continuation, sigma2_init, :fd)

"""
    GradientMatching(; maxiters=500, tol=1e-6, verbose=false, sigma2_init=nothing)

Two-step gradient matching solver inspired by NODEBNGM (Bonnaffé et al. 2023):

1. Smooth observed data with cubic splines to obtain ŷ(t) and dŷ/dt
2. Fit unknown function parameters by matching ODE derivatives:
   minimize ||dŷ/dt - f(ŷ, p, t)||² + penalty

Avoids ODE integration entirely, making it far more robust for neural network
approximators where the IRLS linearization of the ODE trajectory is poor.

Uses Gauss-Newton for penalized approximators (B-spline, GP) and Adam optimizer
for unpenalized approximators (neural networks).

Requires that all state variables are observed (no latent states).

# Arguments
- `maxiters::Int=500`: maximum iterations (Adam needs more than GN)
- `tol::Float64=1e-6`: convergence tolerance
- `verbose::Bool=false`: print iteration details
- `sigma2_init::Union{Nothing,Float64}=nothing`: σ² cap for Fellner-Schall

!!! note "σ² for the Fellner-Schall update"
    The residual variance driving the smoothing-parameter update is
    `σ̂² = RSS / #{residuals with nonzero weight}`, not `RSS / #residuals`.
    Unobserved-state rows are zero-weighted; counting them biased σ̂² low
    and over-smoothed the unknown functions.

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct GradientMatching
    maxiters::Int
    tol::Float64
    verbose::Bool
    sigma2_init::Union{Nothing,Float64}
    lr::Float64
    refine_iters::Int
end

GradientMatching(; maxiters::Int=500, tol::Float64=1e-6, verbose::Bool=false,
                   sigma2_init::Union{Nothing,Float64}=nothing,
                   lr::Float64=0.01, refine_iters::Int=0) =
    GradientMatching(maxiters, tol, verbose, sigma2_init, lr, refine_iters)

"""
    AdamSolver(; maxiters=300, lr=0.01, verbose=false, loss=:auto,
                 penalty_weight=0.0, autodiff=true, sensealg=nothing,
                 plateau_tol=1e-4, plateau_window=30, early_stopping=true)

Adam optimizer that trains unknown function parameters through ODE integration.

For neural networks: uses a ForwardDiff-compatible MLP evaluator (bypassing Lux)
so that exact gradients can be computed through the ODE solve. This matches the
approach used in Universal Differential Equations (UDEs).

For B-splines and GPs: also supported, uses the standard evaluators.

The loss function computes `solve(ODEProblem(...), solver)` at each step and
compares with data. ForwardDiff computes exact gradients through the ODE solve.

Unlike `LAML`/`GCVSolver`, this solver performs no automatic smoothing-parameter
selection: with the default `penalty_weight = 0` the fit is *unpenalized*
(spline coefficients are free). Set `penalty_weight > 0` to add the fixed
quadratic roughness penalty `penalty_weight · Σₖ βₖ' Sₖ βₖ` to the loss.

# Arguments
- `maxiters::Int=300`: maximum Adam iterations
- `lr::Float64=0.01`: learning rate (with cosine annealing)
- `verbose::Bool=false`: print iteration details
- `loss::Symbol=:auto`: loss function; `:auto` selects `:mse` for `Gaussian`
  and `:poisson` for `Poisson` likelihoods (other families are not supported
  by this solver — use `LAML`). Explicit `:mse`/`:poisson` override with a
  warning on mismatch.
- `penalty_weight::Float64=0.0`: fixed weight of the quadratic smoothing
  penalty added to the loss (0 disables)
- `autodiff::Bool=true`: use ForwardDiff (true) or finite differences (false);
  ignored when `sensealg` is set
- `sensealg=nothing`: opt-in adjoint-sensitivity gradient backend. `nothing`
  (default) keeps the ForwardDiff path. Set to `:auto` (a robust default,
  `InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))`) or to a concrete
  SciMLSensitivity continuous adjoint algorithm (`InterpolatingAdjoint(...)`
  is the validated choice; `GaussAdjoint`/`QuadratureAdjoint` were observed
  to mis-differentiate spline-parameterized dynamics — validate against
  ForwardDiff before trusting them) to compute gradients via adjoint
  sensitivities instead. **Requires `using SciMLSensitivity`**
  (the package extension implements this path; without it, `solve` throws an
  informative error). Adjoints make the gradient cost roughly independent of
  the parameter count, which pays off for `NeuralApproximator` models with
  many weights; ForwardDiff is usually faster below a few dozen parameters.
  Only continuous ODE problems are supported (no discrete maps, no DDEs);
  the loss families are the same as the ForwardDiff path (`:mse`,
  `:poisson`).
- `plateau_tol::Float64=1e-4`: stopping threshold for
  `(maximum(window)-minimum(window))/max(abs(minimum(window)),1)`.
  This is an absolute threshold for losses below one and a relative
  threshold above one. A smaller value requires a flatter loss window;
  zero disables this criterion.
- `plateau_window::Int=30`: number of recent losses used by the plateau
  criterion (at least two). Stopping is considered only after both 60
  iterations and a full window.
- `early_stopping::Bool=true`: enable the guarded plateau criterion.
  `false` uses the complete iteration budget; it does not turn budget
  exhaustion into convergence.

# Convergence info
`sol.convergence` is a NamedTuple `(optimizer, method, converged, iterations,
reason, final_grad_norm, backend)` with the standard honest-convergence keys
(see [`PSMSolution`](@ref)) plus `final_grad_norm::Float64`, the Euclidean
norm of the last computed gradient, and `backend`, the gradient backend that
ran (`:forwarddiff`, `:finitediff`, or `:adjoint`). `reason == :plateau` is
only reported while the cosine-annealed learning rate is still above 5% of
the base `lr` — a plateau that appears merely because the schedule has driven
the step size to zero is reported as `:maxiters`, not convergence.
"""
struct AdamSolver
    maxiters::Int
    lr::Float64
    verbose::Bool
    loss::Symbol
    penalty_weight::Float64
    autodiff::Bool
    sensealg::Any
    plateau_tol::Float64
    plateau_window::Int
    early_stopping::Bool
end

function AdamSolver(; maxiters::Int=300, lr::Float64=0.01, verbose::Bool=false,
             loss::Symbol=:auto, penalty_weight::Float64=0.0,
             autodiff::Bool=true, sensealg=nothing, plateau_tol::Float64=1e-4,
             plateau_window::Int=30, early_stopping::Bool=true)
    isfinite(plateau_tol) && plateau_tol >= 0 ||
        throw(ArgumentError("AdamSolver: plateau_tol must be finite and nonnegative"))
    plateau_window >= 2 ||
        throw(ArgumentError("AdamSolver: plateau_window must be at least two"))
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff, sensealg,
               plateau_tol, plateau_window, early_stopping)
end

# Backward-compatible positional constructor (before plateau controls).
AdamSolver(maxiters::Int, lr::Float64, verbose::Bool, loss::Symbol,
           penalty_weight::Float64, autodiff::Bool, sensealg) =
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff, sensealg,
               1e-4, 30, true)

# Backward-compatible positional constructor (pre-sensealg arity).
AdamSolver(maxiters::Int, lr::Float64, verbose::Bool, loss::Symbol,
           penalty_weight::Float64, autodiff::Bool) =
    AdamSolver(maxiters, lr, verbose, loss, penalty_weight, autodiff, nothing)

"""
    MultipleShootingSolver(; n_intervals=10, maxiters_inner=100, maxiters_outer=20,
                             rho_init=10.0, rho_max=1e6, loss=:auto,
                             penalty_weight=0.0, verbose=false)

Multiple shooting solver for training neural differential equations, following
Turan & Jäschke (2021). Partitions the time span into intervals with shooting
variables at boundaries. Uses an augmented Lagrangian (L-BFGS inner subproblems) to enforce continuity.

Advantages over single shooting (AdamSolver):
- Better initial fits: shooting variables initialized from data
- Avoids "flattened trajectory" failure mode for oscillatory systems
- Shorter integration intervals improve gradient quality

# Arguments
- `n_intervals::Int=10`: number of shooting intervals
- `maxiters_inner::Int=100`: L-BFGS iterations per augmented Lagrangian step
- `maxiters_outer::Int=20`: augmented Lagrangian outer iterations
- `rho_init::Float64=10.0`: initial penalty parameter for shooting constraints
- `rho_max::Float64=1e6`: maximum penalty parameter
- `loss::Symbol=:auto`: data-fit loss. `:auto` follows `prob.likelihood`
  (Gaussian → `:mse` weighted SSE, Poisson → `:poisson` weighted negative
  log-likelihood kernel; other families error). Explicit `:mse`/`:poisson`
  are honored with a warning on mismatch, matching `AdamSolver`.
- `penalty_weight::Float64=0.0`: fixed quadratic smoothing penalty
  `penalty_weight · Σₖ βₖ'Sₖβₖ` added to the training objective (0 = no
  penalty; there is no smoothing-parameter selection here)
- `verbose::Bool=false`: print iteration details

# Convergence info
`sol.convergence` is a NamedTuple `(optimizer, method, n_intervals, converged,
iterations, reason, max_gap, rho_final)` with the standard honest-convergence
keys (see [`PSMSolution`](@ref); `iterations` counts outer augmented-Lagrangian
iterations) plus `max_gap::Float64`, the final maximum shooting-gap magnitude,
and `rho_final::Float64`, the final penalty parameter.
"""
struct MultipleShootingSolver
    n_intervals::Int
    maxiters_inner::Int
    maxiters_outer::Int
    rho_init::Float64
    rho_max::Float64
    loss::Symbol
    penalty_weight::Float64
    verbose::Bool
end

MultipleShootingSolver(; n_intervals::Int=10, maxiters_inner::Int=100,
                         maxiters_outer::Int=20,
                         rho_init::Float64=10.0, rho_max::Float64=1e6,
                         loss::Symbol=:auto, penalty_weight::Float64=0.0,
                         verbose::Bool=false) =
    MultipleShootingSolver(n_intervals, maxiters_inner, maxiters_outer,
                           rho_init, rho_max, loss, penalty_weight, verbose)

"""
    AdaptiveGradientMatching(; maxiters=200, verbose=false, gamma_init=1.0,
                               fit_gamma=true, kernel=:rbf, n_samples=0,
                               n_chains=10, rng_seed=nothing)

Adaptive Gradient Matching solver using the product-of-experts objective of
Dondelinger et al. (2013). Gaussian processes smooth the data and supply
gradient estimates with uncertainty; the ODE-predicted gradients are matched
under the product-of-experts likelihood.

Two modes:

- **MAP (default, `n_samples=0`)**: GP hyperparameters from a
  marginal-likelihood grid search, latent states fixed at the GP posterior
  mean, and (β, log γ) optimized once by L-BFGS. Fast; no posterior samples.
- **Population MCMC (`n_samples > 0`)**: the tempered population-MCMC
  sampler of Dondelinger et al./deGradInfer. `n_chains` chains at
  temperatures `t_c = ((c−1)/(C−1))⁵` jointly sample the latent states X,
  the parameters β, and the mismatch variances γ by adaptive blockwise
  random-walk Metropolis with exchange moves between adjacent chains; the
  ODE-mismatch likelihood is raised to `t_c`, interpolating from pure GP
  regression (t=0) to the fully coupled model (t=1). The cold chain
  (t=1) provides the posterior: reported unknown functions use the
  posterior-mean β, and `sol.convergence.beta_samples` /
  `gamma_samples` hold the draws. GP kernel hyperparameters stay fixed
  at their grid-search values (a documented simplification — the paper
  also samples them). Continuous-time problems only.

The (log-)likelihood term for each state k is:
    L_k = -0.5 (f_k - m_k)ᵀ (A_k + γ_k I)⁻¹ (f_k - m_k) - 0.5 log|A_k + γ_k I|

where:
- f_k = ODE-predicted gradients for state k
- m_k = GP gradient mean ('K K⁻¹ x_k, conditioned on the states)
- A_k = GP gradient covariance = K** - K*(K + σ²I)⁻¹K*ᵀ
- γ_k = mismatch parameter controlling ODE-GP coupling

# Arguments
- `maxiters::Int=200`: maximum L-BFGS iterations (MAP mode); MCMC sweeps
  are `2 n_samples` (half burn-in) in sampling mode
- `verbose::Bool=false`: print iteration details
- `gamma_init::Float64=1.0`: initial mismatch parameter (per state)
- `fit_gamma::Bool=true`: optimize γ or keep fixed (MAP mode)
- `kernel::Symbol=:rbf`: GP kernel (:rbf, :matern32)
- `n_samples::Int=0`: cold-chain posterior draws; 0 = MAP mode
- `n_chains::Int=10`: number of tempered chains (sampling mode; ≥ 2)
- `rng_seed`: seed for the sampler (default `nothing` = non-reproducible)

!!! note "GP centering"
    Each state's GP prior is centered: observed states on their data mean,
    unobserved states on their initial condition, and every `x` entering
    the GP quadratic forms and the gradient mean is the CENTERED state
    `x − m_k`. A zero-mean GP prior on uncentered data pulls the
    trajectory toward zero, which biased the fitted unknown functions on
    data far from the origin.

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct AdaptiveGradientMatching
    maxiters::Int
    verbose::Bool
    gamma_init::Float64
    fit_gamma::Bool
    kernel::Symbol
    n_samples::Int
    n_chains::Int
    rng_seed::Union{Nothing, Int}
end

AdaptiveGradientMatching(; maxiters::Int=200, verbose::Bool=false,
                           gamma_init::Float64=1.0, fit_gamma::Bool=true,
                           kernel::Symbol=:rbf, n_samples::Int=0,
                           n_chains::Int=10,
                           rng_seed::Union{Nothing, Int}=nothing) =
    AdaptiveGradientMatching(maxiters, verbose, gamma_init, fit_gamma, kernel,
                             n_samples, n_chains, rng_seed)

"""
    RodeoSolver

Probabilistic ODE solver (rodeo) for parameter inference.

Uses Kalman filtering with an integrated Brownian motion prior to
approximate the ODE solution and compute an approximate marginal likelihood.

# Fields
- `n_steps`: number of solver discretization steps (default: 200)
- `n_deriv`: number of derivatives in IBM prior (default: 3)
- `sigma`: IBM scale parameters (one per state variable, or nothing for auto)
- `obs_var`: observation noise variance (or nothing for auto)
- `method`: likelihood approximation (`:basic` or `:fenrir`)
- `interrogate`: interrogation method (`:kramer` or `:schober`)
- `sqrt_filter`: run the Kalman recursions in square-root (Cholesky/QR)
  form (default: false). Numerically stabler at high `n_deriv` and small
  steps, where the standard covariance recursion can lose
  positive-semidefiniteness; with `false` the standard covariance
  recursion is used unchanged.
- `maxiters`: max L-BFGS iterations (default: 200)
- `verbose`: print progress (default: false)

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. It builds a probabilistic ODE / GP
    state-space manifold from the 4-argument ODE signature.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct RodeoSolver
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    method::Symbol
    interrogate::Symbol
    maxiters::Int
    verbose::Bool
    sqrt_filter::Bool
end

function RodeoSolver(; n_steps::Int=200, n_deriv::Int=3,
                       sigma::Union{Nothing, Vector{Float64}}=nothing,
                       obs_var::Union{Nothing, Float64}=nothing,
                       method::Symbol=:basic,
                       interrogate::Symbol=:kramer,
                       maxiters::Int=200, verbose::Bool=false,
                       sqrt_filter::Bool=false)
    method in (:basic, :fenrir) ||
        throw(ArgumentError("RodeoSolver: method must be :basic or :fenrir " *
                            "(got :$method)"))
    interrogate in (:kramer, :schober) ||
        throw(ArgumentError("RodeoSolver: interrogate must be :kramer or " *
                            ":schober (got :$interrogate)"))
    # The IBM prior needs at least value + first derivative per state;
    # n_deriv=1 BoundsErrors inside the Kalman filter selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("RodeoSolver: n_deriv must be ≥ 2 (got $n_deriv)"))
    RodeoSolver(n_steps, n_deriv, sigma, obs_var, method, interrogate, maxiters,
                verbose, sqrt_filter)
end

"""
    MCMCSolver(; n_samples=1000, n_warmup=500, n_chains=1, target_accept=0.8,
                 prior_scale=1.0, obs_sigma=nothing, sample_smoothing=false,
                 rng_seed=nothing, verbose=false)

Full Bayesian inference via Hamiltonian Monte Carlo, using the No-U-Turn
Sampler (NUTS; Hoffman & Gelman 2014) with dual-averaging step-size
adaptation, on the Hamiltonian dynamics of Neal (2011).
Implemented over LogDensityProblems.jl + AdvancedHMC.jl.

# Arguments
- `n_samples`: number of posterior samples per chain (after warmup)
- `n_warmup`: number of warmup/adaptation steps
- `n_chains`: number of independent chains
- `target_accept`: target acceptance rate for NUTS adaptation (0.6–0.95)
- `prior_scale`: scale for Gaussian prior on parameters (larger = weaker prior).
  When penalty matrices exist (B-spline, GP), uses the penalty; otherwise N(0, prior_scale²).
- `obs_sigma`: observation noise std dev (Gaussian likelihoods only; errors
  otherwise). If `nothing`, sampled as a parameter for Gaussian data; non-
  Gaussian families sample no σ (their dispersion is fixed in the family
  object). The data term follows `prob.likelihood` in all cases.
- `sample_smoothing`: if `true`, jointly sample log(λ) for each smooth term
  with a weakly informative N(log(λ_init), 2²) hyperprior. This gives wider,
  more honest credible intervals for the unknown functions. Default: `false`.
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress

!!! note "Problem classes"
    Discrete maps, ODEs and DDEs are all supported: the log-density and
    the MAP-prediction path both go through the shared
    `_variational_simulate`, which dispatches on `prob.discrete` and
    `prob.delays`. (Before that unification the ODE-only branch built a
    4-argument `ODEFunction` closure and a DDE raised a raw `MethodError`
    from inside NUTS.)
"""
struct MCMCSolver
    n_samples::Int
    n_warmup::Int
    n_chains::Int
    target_accept::Float64
    prior_scale::Float64
    obs_sigma::Union{Nothing, Float64}
    sample_smoothing::Bool
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

MCMCSolver(; n_samples::Int=1000, n_warmup::Int=500, n_chains::Int=1,
             target_accept::Float64=0.8, prior_scale::Float64=1.0,
             obs_sigma::Union{Nothing, Float64}=nothing,
             sample_smoothing::Bool=false,
             rng_seed::Union{Nothing, Int}=nothing,
             verbose::Bool=false) =
    MCMCSolver(n_samples, n_warmup, n_chains, target_accept, prior_scale,
               obs_sigma, sample_smoothing, rng_seed, verbose)

"""
    MagiSolver(; n_samples=1000, n_warmup=500, n_gridpoints=200,
                 sigma=nothing, obs_var=nothing, target_accept=0.8,
                 prior_scale=1.0, preoptimize=true, verbose=false)

Manifold-constrained Gaussian process inference (MAGI) for ODE systems.
Gaussian likelihoods only (the manifold-constrained posterior assumes
Gaussian observation noise); non-Gaussian `prob.likelihood` errors.

MAGI places a Matérn-3/2 Gaussian-process prior on each state (the MAGI
paper uses a generalized Matérn kernel with ν ≈ 2.01; with ν = 3/2 the
implied derivative process is rougher — mean-square continuous but not
mean-square differentiable) and constrains
the GP-implied derivative to the ODE vector field through the conditional
derivative covariance `K* = ''K − 'K C⁻¹ ('K)ᵀ`. The state values on the
discretization grid are sampled jointly with the unknown-function
parameters θ via NUTS/HMC (the manifold-constrained posterior).

**Key advantage**: Handles partially observed systems naturally (unobserved
state components are inferred through the ODE constraint).

Returns an `MCMCChains.Chains` object with posterior samples.

# Fields
- `n_samples`: number of posterior samples after warmup
- `n_warmup`: warmup/adaptation iterations
- `n_gridpoints`: number of discretization grid points for the GP/manifold
  constraint
- `sigma`: per-state observation noise standard deviations (one entry per
  state component). When supplied, these fixed SDs are used in the data
  term instead of auto-estimation; mutually exclusive with `obs_var`.
- `obs_var`: observation noise variance shared across components;
  `nothing` (default) estimates it from the data via a local-linear
  residual estimator (a fixed scale-blind default distorted the posterior
  whenever the data's units differed from O(0.1))
- `target_accept`: NUTS target acceptance rate
- `prior_scale`: scale for Gaussian prior on parameters
- `preoptimize`: run a short MAP pre-optimization to initialize the
  sampler (default `true`)
- `verbose`: print progress

!!! note "Changed"
    The former `n_deriv` field was removed: it was never read anywhere
    (the Matérn-3/2 GP prior has no derivative-order setting), so keeping
    it only suggested a control that did not exist.

# References
- Yang, Wong & Kou (2021) PNAS 118(15): "Inference of dynamic systems
  from noisy and sparse data via manifold-constrained Gaussian processes"

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. It builds a probabilistic ODE / GP
    state-space manifold from the 4-argument ODE signature.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct MagiSolver
    n_samples::Int
    n_warmup::Int
    n_gridpoints::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    target_accept::Float64
    prior_scale::Float64
    preoptimize::Bool
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

function MagiSolver(; n_samples::Int=1000, n_warmup::Int=500,
                    n_gridpoints::Int=200,
                    sigma::Union{Nothing, Vector{Float64}}=nothing,
                    obs_var::Union{Nothing, Float64}=nothing,
                    target_accept::Float64=0.8, prior_scale::Float64=1.0,
                    preoptimize::Bool=true,
                    rng_seed::Union{Nothing, Int}=nothing,
                    verbose::Bool=false)
    n_samples > 0 || throw(ArgumentError("MagiSolver: n_samples must be > 0"))
    n_warmup >= 0 || throw(ArgumentError("MagiSolver: n_warmup must be ≥ 0"))
    n_gridpoints >= 4 || throw(ArgumentError("MagiSolver: n_gridpoints must be ≥ 4"))
    (0.0 < target_accept < 1.0) || throw(ArgumentError("MagiSolver: target_accept must be in (0, 1)"))
    prior_scale > 0 || throw(ArgumentError("MagiSolver: prior_scale must be > 0"))
    MagiSolver(n_samples, n_warmup, n_gridpoints, sigma, obs_var,
               target_accept, prior_scale, preoptimize, rng_seed, verbose)
end

# ─── BNG solver (Bonnaffé et al. 2023) ────────────────────────────

"""
    BNGSolver

Ensemble Bayesian gradient matching (Bonnaffé & Coulson 2023). Avoids
ODE integration entirely: observed series are smoothed with GCV splines,
and unknown-function parameters are fit by matching the smoothed
derivatives under a variance-marginalized log-posterior
`(n/2)log(1 + SSR/2) + (|β|/2)log(1 + Σ(β/prior_sd)²/2)` (the paper's
"Bayesian regularisation": observation and prior variances are
marginalized, not supplied).

Uncertainty comes from a `k_obs × k_proc` fit ensemble: `k_obs`
observation resamples (residual bootstrap of the smoother; the first is
the original data) times `k_proc` restarts from perturbed
initialisations. Reported unknown functions are ensemble means;
`sol.convergence.ensemble_std[name]` gives the pointwise ensemble
standard deviation.

# Fields
- `k_obs`: observation-ensemble size (default 10)
- `k_proc`: process fits per observation ensemble (default 3)
- `prior_sd`: prior scale of the marginalized Gaussian parameter prior
  (default 10.0)
- `maxiters`: Adam iterations per ensemble member (default 2000)
- `lr`: Adam learning rate (default 0.01)
- `lambda_smooth`: extra smoothing penalty on approximator coefficients
  (default 1.0)
- `rng_seed`: seed for the bootstrap and restarts (default `nothing` =
  non-reproducible)
- `verbose`: print progress

# Convergence info
`sol.convergence` carries the standard honest-convergence keys (see
[`PSMSolution`](@ref)): `converged`/`reason` report the BEST ensemble
member's outcome (`:plateau` when its loss stagnated, `:maxiters`
otherwise), `iterations` is the total Adam iterations summed over all
ensemble members, and `member_converged::Vector{Bool}` records each
member's plateau flag (alongside the existing `n_ensemble`,
`member_losses`, `member_weights`, `ensemble_std` keys).

# References
- Bonnaffé & Coulson (2023), "Fast fitting of neural ordinary
  differential equations by Bayesian neural gradient matching to infer
  ecological interactions from time-series data", Methods Ecol Evol 14

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct BNGSolver
    k_obs::Int
    k_proc::Int
    prior_sd::Float64
    maxiters::Int
    lr::Float64
    lambda_smooth::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
    parallel::Bool
end

BNGSolver(; k_obs::Int=10, k_proc::Int=3, prior_sd::Float64=10.0,
            maxiters::Int=2000, lr::Float64=0.01,
            lambda_smooth::Float64=1.0,
            rng_seed::Union{Nothing, Int}=nothing, verbose::Bool=false,
            parallel::Bool=false) =
    BNGSolver(k_obs, k_proc, prior_sd, maxiters, lr, lambda_smooth,
              rng_seed, verbose, parallel)
BNGSolver(k_obs::Int, k_proc::Int, prior_sd::Float64, maxiters::Int,
          lr::Float64, lambda_smooth::Float64, rng_seed, verbose::Bool) =
    BNGSolver(k_obs, k_proc, prior_sd, maxiters, lr, lambda_smooth,
              rng_seed, verbose, false)

# ─── Dalton solver (Wu & Lysy 2024) ───────────────────────────────

"""
    DaltonSolver

Data-Adaptive Likelihood with Transformed Observations (DALTON) solver.

Extends the probabilistic ODE approach (RODEO) with a data-adaptive
marginal likelihood: p(Y|Z) = p(Y,Z)/p(Z), computed via two Kalman
filter passes — one joint (ODE + observations) and one marginal (ODE only).

# Fields
- `n_steps`: number of discretization steps (default 200)
- `n_deriv`: IBM prior derivative order (default 3)
- `sigma`: IBM scale parameters (nothing = auto-estimate)
- `obs_var`: observation noise variance. `nothing` (the default)
  auto-estimates it from the data at solve time as 1% of the mean
  per-column data variance (the same estimator `RodeoSolver` and
  `PseudoMarginalSolver` use). A fixed numeric value is honored exactly;
  note a fixed value must be chosen relative to the data scale — a value
  appropriate for data of order 1 badly misfits data of order 1000.
- `interrogate`: interrogation method `:kramer` or `:schober` (default `:kramer`)
- `sqrt_filter`: run both DALTON Kalman passes in square-root
  (Cholesky/QR) form (default: false). Numerically stabler at high
  `n_deriv` and small steps; the quasi-MLE diffusion calibration is
  computed from mathematically equivalent quantities. With `false` the
  standard covariance recursion is used unchanged.
- `maxiters`: optimization iterations (default 200)
- `verbose`: print progress

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. It builds a probabilistic ODE / GP
    state-space manifold from the 4-argument ODE signature.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct DaltonSolver
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    interrogate::Symbol
    maxiters::Int
    verbose::Bool
    sqrt_filter::Bool
end

function DaltonSolver(; n_steps::Int=200, n_deriv::Int=3,
                        sigma::Union{Nothing, Vector{Float64}}=nothing,
                        obs_var::Union{Nothing, Float64}=nothing,
                        interrogate::Symbol=:kramer,
                        maxiters::Int=200, verbose::Bool=false,
                        sqrt_filter::Bool=false)
    interrogate in (:kramer, :schober) ||
        throw(ArgumentError("DaltonSolver: interrogate must be :kramer or " *
                            ":schober (got :$interrogate)"))
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("DaltonSolver: n_deriv must be ≥ 2 (got $n_deriv)"))
    DaltonSolver(n_steps, n_deriv, sigma, obs_var, interrogate, maxiters,
                 verbose, sqrt_filter)
end

# ─── Pseudo-marginal solver (Chkrebtii et al. 2016) ───────────────

"""
    PseudoMarginalSolver

Pseudo-marginal MCMC using a probabilistic ODE solver for likelihood estimation.

Uses RODEO/fenrir as an inner solver to compute an unbiased estimate of the
marginal likelihood p(Y|θ), then samples from the posterior p(θ|Y) via
adaptive random-walk Metropolis (the proposal scale is tuned toward
`target_accept` during warmup).

# Fields
- `n_samples`: number of posterior samples (default 1000)
- `n_warmup`: warmup/adaptation samples (default 500)
- `n_steps`: discretization steps for inner probabilistic solver (default 200)
- `n_deriv`: IBM prior derivative order (default 3)
- `sigma`: IBM scale parameters (nothing = auto-estimate)
- `obs_var`: observation noise variance (default `nothing` = estimate it
  from the data alongside the other nuisance quantities)
- `target_accept`: target acceptance rate for the adaptive random-walk
  Metropolis proposal-scale adaptation (default 0.8)
- `prior_scale`: prior VARIANCE on parameters (default 1.0). Used directly
  as the variance of the Gaussian smoothing prior
  `exp(−½ βₖ'Sₖβₖ / prior_scale)` and of the weak ridge
  `exp(−½ ‖β‖² / (100·prior_scale))` — it is not squared.
- `inner_method`: likelihood estimator — `:ffbs` (default; unbiased FFBS
  Monte-Carlo average, giving genuine pseudo-marginal MCMC), `:fenrir`
  (deterministic Fenrir evidence; the chain is then plain adaptive RWM on
  an approximate likelihood), or `:dalton` (deterministic DALTON
  data-adaptive likelihood, likewise plain RWM)
- `rng_seed`: seed for the chain's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `sqrt_filter`: run the inner probabilistic-ODE Kalman recursions in
  square-root (Cholesky/QR) form (default: false). Numerically stabler
  at high `n_deriv` and small steps. Supported by all three
  `inner_method`s — under `:ffbs` the backward draws use the
  PSD-by-construction covariance factors, so the jitter escalation of
  the standard sampler is never needed. With `false` the standard
  covariance recursion is used unchanged.
- `verbose`: print progress

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. It builds a probabilistic ODE / GP
    state-space manifold from the 4-argument ODE signature.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct PseudoMarginalSolver
    n_samples::Int
    n_warmup::Int
    n_steps::Int
    n_deriv::Int
    sigma::Union{Nothing, Vector{Float64}}
    obs_var::Union{Nothing, Float64}
    target_accept::Float64
    prior_scale::Float64
    inner_method::Symbol
    initial_params::Union{Nothing, Vector{Float64}}
    rng_seed::Union{Nothing, Int}
    verbose::Bool
    sqrt_filter::Bool
end

function PseudoMarginalSolver(; n_samples::Int=1000, n_warmup::Int=500,
                                n_steps::Int=200, n_deriv::Int=3,
                                sigma::Union{Nothing, Vector{Float64}}=nothing,
                                obs_var::Union{Nothing, Float64}=nothing,
                                target_accept::Float64=0.234, prior_scale::Float64=1.0,
                                inner_method::Symbol=:ffbs,
                                initial_params::Union{Nothing, Vector{Float64}}=nothing,
                                rng_seed::Union{Nothing, Int}=nothing,
                                verbose::Bool=false,
                                sqrt_filter::Bool=false)
    # See RodeoSolver: n_deriv=1 BoundsErrors in the Kalman selectors.
    n_deriv >= 2 ||
        throw(ArgumentError("PseudoMarginalSolver: n_deriv must be ≥ 2 " *
                            "(got $n_deriv)"))
    n_samples > 0 || throw(ArgumentError("PseudoMarginalSolver: n_samples must be > 0"))
    n_warmup >= 0 || throw(ArgumentError("PseudoMarginalSolver: n_warmup must be ≥ 0"))
    (0.0 < target_accept < 1.0) || throw(ArgumentError("PseudoMarginalSolver: target_accept must be in (0, 1)"))
    prior_scale > 0 || throw(ArgumentError("PseudoMarginalSolver: prior_scale must be > 0"))
    PseudoMarginalSolver(n_samples, n_warmup, n_steps, n_deriv, sigma, obs_var,
                          target_accept, prior_scale, inner_method,
                          initial_params, rng_seed, verbose, sqrt_filter)
end

# ─── GCV solver (Wood 2001 / ddefit504) ────────────────────────────

"""
    GCVSolver

Cross-validation solver for smoothing parameter selection.

Uses GCV score minimization (via golden-section search on log(λ)) as an
alternative to LAML/REML. Simpler and faster than LAML, but typically
produces slightly less smooth estimates. With `criterion=:ncv` the GCV
score is replaced by neighbourhood cross-validation (NCV; Wood 2024,
arXiv:2404.16490): each point is predicted from a penalized fit that
omits a temporal *neighbourhood* around it, not just the point itself.
Because dynamical-systems residuals are typically short-range
autocorrelated, ordinary GCV/LOOCV sees the correlated neighbours make
the left-out point look predictable and undersmooths badly; NCV removes
exactly those neighbours and is the recommended criterion whenever
residual autocorrelation is suspected (check `residual_diagnostics`).

# Fields
- `n_grid`: number of grid points for initial λ search (default 50)
- `maxiters`: maximum IRLS iterations (default 50)
- `tol`: convergence tolerance (default 1e-6)
- `gamma`: GCV inflation factor (default 1.4, >1 guards against
  under-smoothing; `gamma=1.0` reproduces the classical unmodified GCV of
  Wood (2001) / ddefit). **Ignored when `criterion=:ncv`**: NCV needs no
  inflation factor — leaving out the neighbourhood is itself the guard
  against the undersmoothing that `gamma` patches over (Wood 2024 uses
  γ has no role in the base NCV criterion as implemented here).
- `criterion`: `:gcv` (default, classical GCV) or `:ncv`
  (neighbourhood cross-validation)
- `ncv_width`: half-width of the NCV deletion neighbourhood, in
  time-index steps (default 2). For observation i of a component, the
  neighbourhood δ(i) is every usable observation of the SAME component
  whose time index is within `ncv_width` of i's (i itself included).
  Must be ≥ 0; `ncv_width=0` gives exact leave-one-out CV. Choose it
  to cover the residual correlation length (e.g. the lag at which the
  residual ACF dies off). Ignored when `criterion=:gcv`.
- `search`: `:direct` (default) evaluates every λ in the GCV search with
  its own O(p³) penalized solve — the historical behavior, byte-identical
  to before this option existed. `:reuse` is the classical ddefit /
  Demmler–Reinsch speed trick (gcv.c `EasySmooth`/`EScv`): one expensive
  whitening + eigendecomposition per IRLS iteration, after which every λ
  in the grid + golden-section search costs only O(np) diagonal
  arithmetic. It is a pure reformulation, not an approximation — scores
  match `:direct` to ~1e-12 relative everywhere λ selection actually
  happens (the stability ridge of the direct scorer is replicated
  exactly; only at the essentially-infinite-smoothing extreme,
  λ ≳ 1e15 with rank-deficient penalties, does agreement degrade to
  ~1e-6, where the direct path's own trA/β are conditioning-limited
  too), so the selected λ̂ agrees within the
  search tolerance, and the returned fit always goes through the same
  final PCLS solve as `:direct`. When `A = J'WJ` (plus any fixed
  penalties) is near-singular the fast path falls back to the direct
  scorer for that IRLS iteration (debug-level log note). With multiple
  approximators the coordinate-descent λ search reuses one
  decomposition per coordinate per sweep. Only `criterion=:gcv` is
  supported: the NCV neighbourhood downdates need `A(λ)⁻¹` at every λ,
  which is genuinely incompatible with the one-decomposition trick, so
  `search=:reuse` with `criterion=:ncv` is rejected at construction.
- `jac`: backend for the working-model Jacobian of predictions w.r.t. the
  coefficients — `:fd` (default; adaptive central finite differences,
  historical behavior, byte-identical) or `:forwarddiff` (forward-mode AD
  through the ODE/map/DDE solve; exact to solver precision, one chunked
  Dual sweep instead of 2·n_p perturbed solves, config built once per
  solve; falls back to `:fd` for an iteration — `@debug` note — whenever
  the Dual-valued solve fails or returns non-finite entries). See the
  `LAML` docstring for the full description; the two solvers share
  `compute_jacobian!`.
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple
`(converged, iterations, reason, criterion, gcv, ncv, ridge,
final_parameter_step)` with the standard
honest-convergence keys (see [`PSMSolution`](@ref)) plus:
- `criterion::Symbol`: which selection criterion ran (`:gcv` or `:ncv`)
- `gcv::Float64`: the final GCV score when `criterion=:gcv`; `NaN`
  otherwise (also NaN when no smooth terms are present)
- `ncv::Float64`: the final NCV score when `criterion=:ncv`; `NaN`
  otherwise
- `ridge::Bool`, `final_parameter_step::Float64`: the [`LAML`](@ref) ridge
  probe — one more uncontracted PCLS step from the reported (λ̂, β̂) with the
  final Jacobian and weights; `ridge` is `true` when the objective converged
  but that step would still move β by more than 1e-3 relative. Penalized
  approximators only; the step is reported for every fit.
"""
struct GCVSolver
    n_grid::Int
    maxiters::Int
    tol::Float64
    gamma::Float64
    criterion::Symbol
    ncv_width::Int
    search::Symbol
    verbose::Bool
    jac::Symbol
end

function GCVSolver(; n_grid::Int=50, maxiters::Int=50, tol::Float64=1e-6,
                     gamma::Float64=1.4, criterion::Symbol=:gcv,
                     ncv_width::Int=2, search::Symbol=:direct,
                     verbose::Bool=false, jac::Symbol=:fd)
    criterion in (:gcv, :ncv) ||
        throw(ArgumentError("GCVSolver: criterion must be :gcv or :ncv " *
                            "(got $(repr(criterion)))"))
    ncv_width >= 0 ||
        throw(ArgumentError("GCVSolver: ncv_width must be ≥ 0 " *
                            "(got $ncv_width; 0 = leave-one-out CV)"))
    search in (:direct, :reuse) ||
        throw(ArgumentError("GCVSolver: search must be :direct or :reuse " *
                            "(got $(repr(search)))"))
    !(search === :reuse && criterion === :ncv) ||
        throw(ArgumentError("GCVSolver: search=:reuse supports only " *
                            "criterion=:gcv — the NCV neighbourhood downdates " *
                            "need A(λ)⁻¹ at every λ, which is incompatible " *
                            "with the one-decomposition reuse trick; use " *
                            "search=:direct with criterion=:ncv"))
    jac in (:fd, :forwarddiff) ||
        throw(ArgumentError("GCVSolver: jac must be :fd or :forwarddiff " *
                            "(got $(repr(jac)))"))
    GCVSolver(n_grid, maxiters, tol, gamma, criterion, ncv_width, search,
              verbose, jac)
end

# Backward-compatible positional constructor (pre-jac arity).
GCVSolver(n_grid::Int, maxiters::Int, tol::Float64, gamma::Float64,
          criterion::Symbol, ncv_width::Int, search::Symbol,
          verbose::Bool) =
    GCVSolver(n_grid, maxiters, tol, gamma, criterion, ncv_width, search,
              verbose, :fd)

# ─── Two-stage solver (Wood 2001 / deGradInfer) ───────────────────

"""
    TwoStageSolver

Two-stage smooth-then-differentiate solver (simplest baseline).

Stage 1: Smooth each observed state independently via spline + GCV/penalty.
Stage 2: Numerically differentiate smoothed curves, then match ODE RHS
          to derivatives via least squares to infer unknown function params.

This is the original approach from Wood (2001) / deGradInfer (Macdonald & Husmeier 2015).

# Fields
- `n_basis_smooth`: upper limit on the number of B-spline coefficients used
  by the stage-1 smoother; the actual basis size is
  `clamp(n − 2, 4, n_basis_smooth)` for `n` data points (default 15).

  !!! note "Changed"
      This field was previously dead: it was documented (default 20) but
      never read — the smoother always capped the basis at 15. It is now
      wired through; the default was set to 15 so default fits are
      byte-identical to before.
- `lambda_smooth`: smoothing penalty for initial data fit (default 1.0)
- `maxiters`: max iterations for parameter matching (default 1000)
- `lr`: learning rate for Adam optimization in matching step (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
with the standard honest-convergence keys (see [`PSMSolution`](@ref));
`converged=true` with `reason=:plateau` when the matching loss stagnated
over a 30-iteration window, `reason=:maxiters` otherwise.

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct TwoStageSolver
    n_basis_smooth::Int
    lambda_smooth::Float64
    maxiters::Int
    lr::Float64
    verbose::Bool
end

function TwoStageSolver(; n_basis_smooth::Int=15, lambda_smooth::Float64=1.0,
                          maxiters::Int=1000, lr::Float64=0.01, verbose::Bool=false)
    n_basis_smooth >= 4 ||
        throw(ArgumentError("TwoStageSolver: n_basis_smooth must be ≥ 4 " *
                            "(cubic B-spline basis; got $n_basis_smooth)"))
    TwoStageSolver(n_basis_smooth, lambda_smooth, maxiters, lr, verbose)
end

# ─── Derivative-free solver (stochastic + NelderMead) ──────────────

"""
    DerivativeFreeSolver

Derivative-free optimization solver using NelderMead or particle swarm.

The objective includes the quadratic roughness penalty
`0.5 · penalty_weight · Σₖ βₖ'Sₖβₖ` (note the ½ factor; default
`penalty_weight=1.0`; set `penalty_weight=0` for an unpenalized fit —
there is no smoothing-parameter selection here).

Useful as a robust fallback when gradient-based methods fail (non-smooth
objectives, stiff dynamics, poor conditioning). Uses simulation-based
loss without requiring autodiff through ODE solves.

# Fields
- `method`: optimization method — `:nelder_mead` or `:particle_swarm` (default `:nelder_mead`)
- `maxiters`: maximum function evaluations (default 10000)
- `n_particles`: particle count for swarm methods (default 20)
- `loss`: loss type (default `:auto`). `:auto` follows `prob.likelihood`
  — Gaussian → `:mse` (weighted SSE), any other family → `:likelihood`
  (the family's negative log-likelihood). Explicit `:mse` or
  `:likelihood` is honored regardless of the likelihood family.
- `penalty_weight`: weight of the roughness penalty above (default 1.0)
- `verbose`: print progress
"""
struct DerivativeFreeSolver
    method::Symbol
    maxiters::Int
    n_particles::Int
    loss::Symbol
    penalty_weight::Float64
    verbose::Bool
end

DerivativeFreeSolver(; method::Symbol=:nelder_mead, maxiters::Int=10000,
                       n_particles::Int=20, loss::Symbol=:auto,
                       penalty_weight::Float64=1.0,
                       verbose::Bool=false) =
    DerivativeFreeSolver(method, maxiters, n_particles, loss, penalty_weight,
                         verbose)

# ─── Variational inference solver ──────────────────────────────────

"""
    VariationalSolver

Variational inference solver using mean-field Gaussian approximation.

Approximates the posterior p(θ|Y) with a factored Gaussian q(θ) = ∏ N(μᵢ, σᵢ²)
by maximizing the evidence lower bound (ELBO). Much faster than MCMC while
providing uncertainty estimates.

# Fields
- `maxiters`: max ELBO optimization iterations (default 2000)
- `lr`: learning rate for Adam on ELBO (default 0.01)
- `n_elbo_samples`: Monte Carlo samples for ELBO gradient (default 10)
- `prior_scale`: prior VARIANCE on parameters (default 1.0). Used directly
  as the variance of the Gaussian smoothing prior — the prior precision is
  `Λ = ridge·I + Σₖ Sₖ / prior_scale` — it is not squared.
- `obs_noise_var`: Gaussian observation-noise variance (default `nothing`
  = estimate from the data). Gaussian likelihoods only; errors otherwise.
  Non-Gaussian families use their own pointwise log-likelihood in the ELBO.
- `rng_seed`: seed for the ELBO Monte-Carlo draws (default `42`, which
  reproduces the historical hard-coded behavior; `nothing` = fresh
  non-reproducible stream). Does not touch the global RNG.
- `verbose`: print progress

!!! note "Reported EDF"
    `sol.edf` is the Laplace effective degrees of freedom at the
    variational mean, `tr((H + Λ)⁻¹H)` with the Gauss–Newton information
    `H = JᵀWJ/σ²_obs` — a real measure of model complexity that responds
    to the smoothing level. It is `NaN` (honestly missing) when the model
    cannot be simulated at the variational mean or the linear system is
    singular; it is never a fabricated constant.
"""
struct VariationalSolver
    maxiters::Int
    lr::Float64
    n_elbo_samples::Int
    prior_scale::Float64
    obs_noise_var::Union{Nothing, Float64}
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

# rng_seed defaults to 42, NOT nothing: the solver historically hard-coded
# Xoshiro(42) for its ELBO Monte-Carlo draws, and the default must keep
# reproducing that. `nothing` opts into a fresh non-reproducible stream.
VariationalSolver(; maxiters::Int=2000, lr::Float64=0.01,
                    n_elbo_samples::Int=10, prior_scale::Float64=1.0,
                    obs_noise_var::Union{Nothing, Float64}=nothing,
                    rng_seed::Union{Nothing, Int}=42,
                    verbose::Bool=false) =
    VariationalSolver(maxiters, lr, n_elbo_samples, prior_scale, obs_noise_var,
                      rng_seed, verbose)

# ─── ABC solver (Approximate Bayesian Computation) ─────────────────

"""
    ABCSolver

Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

Likelihood-free inference using simulation-based rejection sampling with
adaptive tolerance scheduling. Works for any simulator, including those
where the likelihood is intractable.

**Prior** (`prior` keyword): `:smoothness` (default) is a Gaussian GMRF
built from the approximators' roughness penalties, centered on their
initial values with overall spread `prior_scale` — matching the prior
structure of `MCMCSolver`/`VariationalSolver`, so wiggly coefficient
vectors are a-priori discouraged. `:box` restores the legacy uniform box
of half-width `prior_scale` (posterior then depends strongly on the
initialization; treat as approximate).

# Fields
- `n_particles`: number of ABC particles (default 500)
- `n_generations`: number of SMC generations (default 10)
- `summary_fn`: summary statistic function, or `:auto` for MSE-based (default `:auto`)
- `prior`: `:smoothness` (default, GMRF) or `:box` (legacy uniform)
- `prior_scale`: prior spread — GMRF scale, or box half-width (default 2.0)
- `quantile_eps`: quantile for tolerance schedule (default 0.5)
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress
"""
struct ABCSolver
    n_particles::Int
    n_generations::Int
    summary_fn::Union{Symbol, Function}
    prior::Symbol
    prior_scale::Float64
    quantile_eps::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
    parallel::Bool
end

ABCSolver(; n_particles::Int=500, n_generations::Int=10,
            summary_fn::Union{Symbol, Function}=:auto,
            prior::Symbol=:smoothness,
            prior_scale::Float64=2.0, quantile_eps::Float64=0.5,
            rng_seed::Union{Nothing, Int}=nothing,
            verbose::Bool=false,
            parallel::Bool=false) =
    ABCSolver(n_particles, n_generations, summary_fn, prior, prior_scale,
              quantile_eps, rng_seed, verbose, parallel)
ABCSolver(n_particles::Int, n_generations::Int, summary_fn, prior::Symbol,
          prior_scale::Float64, quantile_eps::Float64, rng_seed, verbose::Bool) =
    ABCSolver(n_particles, n_generations, summary_fn, prior, prior_scale,
              quantile_eps, rng_seed, verbose, false)

# ─── Integral matching solver (Dattner & Klaassen 2015) ────────────

"""
    IntegralMatchingSolver

Integral matching estimator: integrates both sides of the ODE so that
parameter estimation requires neither derivative estimation nor repeated
ODE integration.  Compares smoothed trajectory increments ŷ(tᵢ)−ŷ(t₁)
against the cumulative trapezoidal integral of f(ŷ(s),p,s).

# Fields
- `lambda_smooth`: smoothing penalty weight (default 1.0)
- `maxiters`: Adam iterations (default 1000)
- `lr`: learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method)`
with the standard honest-convergence keys (see [`PSMSolution`](@ref));
`converged=true` with `reason=:plateau` when the matching loss stagnated
over a 30-iteration window, `reason=:maxiters` otherwise.

# References
- Dattner & Klaassen (2015), EJS 9(2), 1939–1973
- R package `simode` (Yaari & Dattner)

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct IntegralMatchingSolver
    lambda_smooth::Float64
    maxiters::Int
    lr::Float64
    verbose::Bool
end

IntegralMatchingSolver(; lambda_smooth::Float64=1.0, maxiters::Int=1000,
                         lr::Float64=0.01, verbose::Bool=false) =
    IntegralMatchingSolver(lambda_smooth, maxiters, lr, verbose)

# ─── Profile likelihood solver ─────────────────────────────────────

"""
    ProfileLikelihoodSolver

Profile likelihood for identifiability analysis and confidence intervals.

For each unknown-function parameter βⱼ, sweeps βⱼ over a grid while
optimising all other parameters (β₋ⱼ) at each grid point.  Returns the
profile likelihood curve Lₚ(βⱼ) and a likelihood-ratio CI.

The profile is taken through the penalized objective at the fitted
smoothing parameters λ̂ (a penalized spline is not identified through its
raw RSS); the statistic ΔPenSS/σ̂² is referenced against χ²₁, and CI
endpoints are interpolated between grid points. Nuisance coefficients
are re-optimised at each grid point by a long Nelder–Mead run,
warm-started from the previous grid point. Gaussian likelihoods only.

# Fields
- `n_profile_points`: grid points per parameter (default 20); the exact
  MLE value is inserted, so the evaluated grid may have one extra point
- `ci_level`: confidence level for LR-based CI (default 0.95)
- `param_indices`: which parameter indices to profile (default `nothing` = all)
- `verbose`: print progress
- `base_alg`: the `LAML` fit the profile is taken around (default
  `LAML(verbose=false)`). The returned solution reports THIS fit's
  convergence keys, so a deliberately truncated base — e.g.
  `LAML(maxiters=1)` — surfaces as `converged = false` rather than being
  laundered into a converged profile-likelihood result.

# References
- Simpson & Maclaren (2023), PLOS Comp Biol
"""
struct ProfileLikelihoodSolver
    n_profile_points::Int
    ci_level::Float64
    param_indices::Union{Nothing, Vector{Int}}
    verbose::Bool
    base_alg::LAML
    parallel::Bool
end

ProfileLikelihoodSolver(; n_profile_points::Int=20, ci_level::Float64=0.95,
                           param_indices::Union{Nothing, Vector{Int}}=nothing,
                           verbose::Bool=false,
                           base_alg::LAML=LAML(verbose=false),
                           parallel::Bool=false) =
    ProfileLikelihoodSolver(n_profile_points, ci_level, param_indices, verbose,
                            base_alg, parallel)
ProfileLikelihoodSolver(n_profile_points::Int, ci_level::Float64,
                        param_indices, verbose::Bool, base_alg::LAML) =
    ProfileLikelihoodSolver(n_profile_points, ci_level, param_indices, verbose,
                            base_alg, false)

# ─── Ensemble Kalman inversion solver ──────────────────────────────

"""
    EnsembleKalmanSolver

Ensemble Kalman Inversion (EKI) for batch parameter estimation.

Uses an ensemble of parameter particles, propagates each through the
forward model (ODE simulation), and updates via the Kalman gain
K = Cov(θ, G(θ)) [Cov(G(θ), G(θ)) + Γ]⁻¹.  Iterates until the ensemble
collapses around the MAP estimate — mean particle std below 1e-3 of its
initial value, reported as `converged = true, reason =
:ensemble_collapse` — or until `n_iterations` is exhausted, reported
honestly as `converged = false, reason = :maxiters`.

Discrete problems are propagated with `simulate_discrete` (unit steps
over `tspan`, data times snapped to the nearest unit step from `tspan[1]`),
so EKI sees the same trajectory as every other simulation-based solver.

# Fields
- `n_ensemble`: ensemble size (default 50)
- `n_iterations`: EKI iterations (default 30)
- `noise_scale`: observation noise std for regularisation (default 0.1).
  This is a FIXED scale, not auto-estimated from the data: set it
  relative to the magnitude of your observations (e.g. roughly the
  expected noise std). The default suits data of order 1; for data of
  order 1000 use a correspondingly larger value.
- `rng_seed`: seed for the solver's own RNG stream (default `42`, which
  reproduces the historical hard-coded behavior; `nothing` = fresh
  non-reproducible stream). Does not touch the global RNG.
- `verbose`: print progress

# References
- Iglesias, Law & Stuart (2013), Inverse Problems
- Schillings & Stuart (2017), SIAM J Numer Anal

!!! note "ODE and discrete problems only"
    DDE problems are REJECTED with an error: `EnsembleKalmanSolver` propagates
    ensemble particles through the 4-argument ODE (`f!(du, u, p, t)`) or
    discrete-map simulation loop, which does not supply delayed history `h`.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct EnsembleKalmanSolver
    n_ensemble::Int
    n_iterations::Int
    noise_scale::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
end

# rng_seed defaults to 42, NOT nothing: the solver historically hard-coded
# Xoshiro(42) for ensemble initialization and perturbations, and the default
# must keep reproducing that. `nothing` opts into a fresh stream.
EnsembleKalmanSolver(; n_ensemble::Int=50, n_iterations::Int=30,
                       noise_scale::Float64=0.1,
                       rng_seed::Union{Nothing, Int}=42,
                       verbose::Bool=false) =
    EnsembleKalmanSolver(n_ensemble, n_iterations, noise_scale, rng_seed, verbose)

# ─── ODIN solver (ODE-Informed regression) ─────────────────────────

"""
    ODINSolver

ODE-informed regression (Wenk, Abbati et al. 2020). GP hyperparameters
are first estimated per observed state by marginal likelihood, then the
states `X` and the unknown-function parameters `θ` are optimised
*jointly* against the ODIN risk

    R(X, θ) = Σ_k [ ‖y_k − x_k‖²/σ_{n,k}² + x̃_kᵀ K_k⁻¹ x̃_k
                    + (f_k(X,θ) − D_k x̃_k)ᵀ A_k⁻¹ (f_k(X,θ) − D_k x̃_k) ],

where `x̃_k` is the centered state, `D_k = 'K K⁻¹` maps states to their GP
conditional-mean derivative, and `A_k = ''K − 'K K⁻¹ 'Kᵀ + γI` is the GP
conditional derivative covariance — so the ODE mismatch is trusted only
in directions the GP actually determines. Unobserved states are included
as free variables (GP prior + ODE terms, no data term), so partially
observed systems are supported.

# Fields
- `maxiters`: outer iterations; 20 Adam steps each (default 50)
- `gp_lengthscale`, `gp_variance`: `nothing` (default) estimates the RBF
  hyperparameters per state by GP marginal likelihood. Supplying BOTH
  fixes them for all states (with noise assumed at `0.01 * gp_variance`);
  supplying only one is an error
- `ode_weight`: extra multiplier on the ODE-mismatch term (default 1.0 =
  the natural Mahalanobis weighting of the paper)
- `lr`: Adam learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method,
gp_hyperparams)` with the standard honest-convergence keys (see
[`PSMSolution`](@ref)). `iterations` counts the joint Adam steps actually
performed (the budget is `20 * maxiters` steps); `converged=true` with
`reason=:plateau` when the risk stagnated over a 30-step window while the
cosine-annealed learning rate was still above 5% of the base `lr`,
`reason=:maxiters` otherwise.

`sol.data_loss` is the WEIGHTED residual sum of squares over usable cells
(`weighted_data_loss`), the same convention as every other solver — so
masked (`data_weights == 0`) and down-weighted observations contribute to
the reported misfit exactly in proportion to their weight.

# References
- Wenk, Abbati et al. (2020), AAAI — ODIN
- Wenk et al. (2019), AISTATS — FGPGM

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct ODINSolver
    maxiters::Int
    gp_lengthscale::Union{Nothing, Float64}
    gp_variance::Union{Nothing, Float64}
    ode_weight::Float64
    lr::Float64
    verbose::Bool
    kernel::Symbol
    standardize::Bool
end

function ODINSolver(; maxiters::Int=50,
                    gp_lengthscale::Union{Nothing, Real}=nothing,
                    gp_variance::Union{Nothing, Real}=nothing,
                    ode_weight::Float64=1.0,
                    lr::Float64=0.01, verbose::Bool=false,
                    kernel::Symbol=:rbf,
                    standardize::Bool=true)
    kernel in (:rbf, :matern32, :matern52) ||
        throw(ArgumentError("ODINSolver: kernel must be :rbf, :matern32, or :matern52 (got :$kernel)"))
    ODINSolver(maxiters,
               gp_lengthscale === nothing ? nothing : Float64(gp_lengthscale),
               gp_variance === nothing ? nothing : Float64(gp_variance),
               ode_weight, lr, verbose, kernel, standardize)
end
ODINSolver(maxiters::Int, gp_l, gp_v, ode_weight::Float64, lr::Float64, verbose::Bool) =
    ODINSolver(maxiters, gp_l, gp_v, ode_weight, lr, verbose, :rbf, true)

# ─── FGPGM solver ──────────────────────────────────────────────────

"""
    FGPGMSolver(; n_samples=1000, n_warmup=500, gamma=0.1,
                  gp_lengthscale=nothing, gp_variance=nothing,
                  obs_sigma=nothing, prior_scale=1.0, target_accept=0.234,
                  rng_seed=nothing, verbose=false)

Fast Gaussian process based gradient matching (FGPGM; Wenk, Gotovos,
Bauer, Gorbach, Krause & Buhmann, AISTATS 2019). The latent states `X`
(at the observation times, as in the paper) and the unknown-function
parameters `θ` are sampled JOINTLY from one product-of-experts density

    p(X, θ | y) ∝ Π_k [ N(y_k | x_k, σ_{n,k}² I) · N(x_k | m_k, C_k)
                        · N(f_k(X, θ) | D_k x̃_k, A_k + γI) ] · p(θ)

with `D_k = 'C_k C_k⁻¹` and `A_k = ''C_k − 'C_k C_k⁻¹ 'C_kᵀ` (the GP
conditional derivative map and covariance), by single-chain adaptive
Metropolis-within-Gibbs: one GP-correlated random-walk block per state
and one joint random-walk block for θ whose proposal covariance is the
sample covariance of the warmup draws (adaptive Metropolis; Haario,
Saksman & Tamminen 2001). GP hyperparameters are fixed beforehand by
per-state marginal likelihood — the paper's key simplification.

**When to prefer FGPGM**: over [`AdaptiveGradientMatching`](@ref)'s
population MCMC when you want genuine posterior samples without the
temperature ladder and γ sampling (one chain, fixed hyperparameters —
cheaper and better-behaved); over [`ODINSolver`](@ref) (the same
authors' later pure-optimisation formulation) when you need posterior
uncertainty rather than a point estimate; over [`MagiSolver`](@ref)
when the observation times are dense enough to carry the latent states
(MAGI discretizes on a finer grid and uses NUTS — heavier per sweep).
Gaussian likelihoods only; continuous-time problems only;
`NeuralApproximator` and `KANApproximator` are rejected (use [`AdamSolver`](@ref)).

# Fields
- `n_samples`: retained posterior draws after warmup (default 1000)
- `n_warmup`: warmup sweeps; proposal adaptation happens here ONLY and
  the kernel is frozen afterwards (default 500). Two things adapt: the
  per-block scalar scales (Robbins-Monro toward `target_accept`) and,
  from the first fifth of warmup on, the θ proposal covariance (the
  running covariance of the warmup draws, so proposals follow the
  posterior's own axes — an isotropic walk over correlated B-spline
  coefficients mixes an order of magnitude worse). A few hundred warmup
  sweeps are needed for the covariance to settle; judge by ESS
- `gamma`: the paper's model-mismatch variance γ added to the ODE
  expert's covariance `A_k + γI` (default 0.1). The effective slack also
  includes `σ_n²/ℓ²`, the derivative-scale observation noise, as in
  `ODINSolver`. Must be positive
- `gp_lengthscale`, `gp_variance`: `nothing` (default) estimates the RBF
  hyperparameters per state by GP marginal likelihood. Supplying BOTH
  fixes them for all states (with noise assumed at `0.01 * gp_variance`);
  supplying only one is an error (ODIN convention)
- `obs_sigma`: observation-noise SD overriding the marginal-likelihood
  estimate σ_n in the data expert (default `nothing` = per-state estimate)
- `prior_scale`: scale of the θ smoothing prior (larger = weaker), as in
  `MCMCSolver`/`MagiSolver` (default 1.0)
- `target_accept`: per-block acceptance rate targeted by the warmup
  adaptation (default 0.234, the classic random-walk optimum)
- `rng_seed`: seed for the sampler's own RNG stream (default `nothing` =
  non-reproducible). Matches the `rng_seed` convention of
  `AdaptiveGradientMatching`/`BNGSolver`; does not touch the global RNG.
- `verbose`: print progress

# Convergence info
`sol.convergence` carries `chains` (`MCMCChains.Chains` of the θ draws),
`beta_samples`, `state_mean`, `gp_hyperparams`, post-warmup
`accept_rates` per block, and the honest keys
`converged=false`/`reason=:maxiters`/`iterations` — a fixed-budget
sampler has no stopping criterion (MagiSolver convention), so judge the
run by the R̂/ESS of `convergence.chains`.

# References
- Wenk, Gotovos, Bauer, Gorbach, Krause & Buhmann (2019), "Fast
  Gaussian process based gradient matching for parameter identification
  in systems of nonlinear ODEs", AISTATS 89:1351-1360.
- Haario, Saksman & Tamminen (2001), "An adaptive Metropolis algorithm",
  Bernoulli 7(2):223-242 — the warmup covariance adaptation of the θ block.

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct FGPGMSolver
    n_samples::Int
    n_warmup::Int
    gamma::Float64
    gp_lengthscale::Union{Nothing, Float64}
    gp_variance::Union{Nothing, Float64}
    obs_sigma::Union{Nothing, Float64}
    prior_scale::Float64
    target_accept::Float64
    rng_seed::Union{Nothing, Int}
    verbose::Bool
    kernel::Symbol
end

function FGPGMSolver(; n_samples::Int=1000, n_warmup::Int=500,
                       gamma::Real=0.1,
                       gp_lengthscale::Union{Nothing, Real}=nothing,
                       gp_variance::Union{Nothing, Real}=nothing,
                       obs_sigma::Union{Nothing, Real}=nothing,
                       prior_scale::Real=1.0,
                       target_accept::Real=0.234,
                       rng_seed::Union{Nothing, Int}=nothing,
                       verbose::Bool=false,
                       kernel::Symbol=:rbf)
    n_samples >= 1 ||
        throw(ArgumentError("FGPGMSolver: n_samples must be ≥ 1 (got $n_samples)"))
    n_warmup >= 0 ||
        throw(ArgumentError("FGPGMSolver: n_warmup must be ≥ 0 (got $n_warmup)"))
    gamma > 0 ||
        throw(ArgumentError("FGPGMSolver: gamma must be positive (got $gamma) " *
                            "— it is the variance of the ODE-mismatch slack " *
                            "in N(f_k | D_k x̃_k, A_k + γI)"))
    0 < target_accept < 1 ||
        throw(ArgumentError("FGPGMSolver: target_accept must lie in (0, 1) " *
                            "(got $target_accept)"))
    prior_scale > 0 ||
        throw(ArgumentError("FGPGMSolver: prior_scale must be positive " *
                            "(got $prior_scale)"))
    obs_sigma === nothing || obs_sigma > 0 ||
        throw(ArgumentError("FGPGMSolver: obs_sigma must be positive " *
                            "(got $obs_sigma)"))
    (gp_lengthscale === nothing) == (gp_variance === nothing) ||
        throw(ArgumentError(
            "FGPGMSolver: supply BOTH gp_lengthscale and gp_variance to fix " *
            "the GP hyperparameters, or neither to estimate them per state " *
            "by marginal likelihood."))
    kernel in (:rbf, :matern32, :matern52) ||
        throw(ArgumentError("FGPGMSolver: kernel must be :rbf, :matern32, or :matern52 (got :$kernel)"))
    FGPGMSolver(n_samples, n_warmup, Float64(gamma),
                gp_lengthscale === nothing ? nothing : Float64(gp_lengthscale),
                gp_variance === nothing ? nothing : Float64(gp_variance),
                obs_sigma === nothing ? nothing : Float64(obs_sigma),
                Float64(prior_scale), Float64(target_accept),
                rng_seed, verbose, kernel)
end
FGPGMSolver(n_samples::Int, n_warmup::Int, gamma::Float64, gp_l, gp_v, obs_s,
            prior_scale::Float64, target_accept::Float64, rng_seed, verbose::Bool) =
    FGPGMSolver(n_samples, n_warmup, gamma, gp_l, gp_v, obs_s,
                prior_scale, target_accept, rng_seed, verbose, :rbf)

# ─── RKHS solver ───────────────────────────────────────────────────

"""
    RKHSSolver

Trajectory-RKHS estimator inspired by González et al. (2014): each *state
trajectory* is expanded in a time-kernel RKHS over the data times,
`x_k(t) = m_k + Σᵢ b_{k,i} k(t, tᵢ)`, so `ẋ_k(t)` is available analytically.
(Unlike González et al. 2014's augmented representer basis containing
derivative-kernel atoms `k̇(·, t_g)`, this uses the compact data-center basis
`{k(·, tᵢ)}` paired with a joint Gauss–Newton linearization of `f` via `∂f/∂x`.)
The objective — data fit + RKHS norm `λ bᵀKb` + ODE-gradient match
`ρ Σ(ẋ − f(x,θ))²` on a collocation grid — is minimized by alternating a
joint Gauss–Newton solve for the trajectory coefficients (coupling observed
and unobserved states via `∂f/∂x`) and Adam steps on the unknown-function
parameters θ. Continuous-time problems only.

# Fields
- `kernel`: time-kernel type `:rbf`, `:matern32`, `:matern52` (default `:rbf`)
- `lengthscale`: time-kernel lengthscale; ≤ 0 (default) auto-scales to
  `time_span / 8`
- `lambda_rkhs`: RKHS-norm penalty λ on the trajectory (default 0.01)
- `lambda_ode`: weight ρ of the ODE-gradient term (default 1.0)
- `n_repr_points`: size of the ODE collocation grid (default 30)
- `maxiters`: outer alternations; 10 θ Adam steps each (default 200)
- `lr`: θ learning rate (default 0.01)
- `verbose`: print progress

# Convergence info
`sol.convergence` is a NamedTuple `(converged, iterations, reason, method,
kernel, lengthscale)` with the standard honest-convergence keys (see
[`PSMSolution`](@ref)); `reason` is `:converged_tol` when the relative
objective change fell below tolerance, `:maxiters` otherwise.

# References
- González et al. (2014), Pattern Recognition Letters

!!! note "ODE problems only"
    DDE problems are REJECTED with an error. The objective evaluates the dynamics
    through the 4-argument ODE signature on a smoothed trajectory.
    A PSM DDE's dynamics have the 5-argument signature
    `f!(du, u, h, p, t)`, so the delayed history can never be supplied
    and the model would be fitted as though it had no delays.
    Use [`LAML`](@ref), [`GCVSolver`](@ref), [`AdamSolver`](@ref),
    [`VariationalSolver`](@ref), [`MCMCSolver`](@ref),
    [`DerivativeFreeSolver`](@ref) or
    [`ProfileLikelihoodSolver`](@ref) for DDEs.
"""
struct RKHSSolver
    kernel::Symbol
    lengthscale::Float64      # ≤ 0 means auto-scale from the time span
    lambda_rkhs::Float64
    lambda_ode::Float64
    n_repr_points::Int
    maxiters::Int
    lr::Float64
    verbose::Bool
end

RKHSSolver(; kernel::Symbol=:rbf, lengthscale::Float64=-1.0,
             lambda_rkhs::Float64=0.01, lambda_ode::Float64=1.0,
             n_repr_points::Int=30,
             maxiters::Int=200, lr::Float64=0.01, verbose::Bool=false) =
    RKHSSolver(kernel, lengthscale, lambda_rkhs, lambda_ode, n_repr_points,
               maxiters, lr, verbose)

