# Custom approximators

PartiallySpecifiedModels.jl ships with nine approximator types, but the set is open: every solver constructs and consumes unknown functions through four generic functions, so adding your own approximator requires no changes to any solver file. Define a struct, implement the four methods, and pass it to `PSMProblem` like any built-in type.

## The interface

A custom approximator is a subtype of `AbstractApproximator` with (at least) a `name::Symbol` field — the symbol under which the fitted function appears in the dynamics parameters (`p.<name>`) and in `PSMSolution.unknown_functions`. It must implement four methods:

| Function | Contract |
|:---|:---|
| `nparams(a)` | Number of free coefficients. Determines the size of the approximator's slice of the flat coefficient vector. |
| `initial_params(a)` | Starting coefficient vector of length `nparams(a)`, `Float64`-convertible. |
| `penalty_matrix(a)` | The roughness penalty matrix `S` (size `nparams(a) × nparams(a)`, symmetric positive semi-definite, possibly all zeros), used in quadratic penalties `λ βᵀSβ`. Return `nothing` for no quadratic penalty. |
| `build_evaluator(a, params_k)` | A callable representing the function encoded by the coefficient block `params_k`. |

### Contracts in detail

**`params_k` is a view or slice of the flat coefficient vector.** Solvers keep all parameters in one flat vector with one contiguous block per approximator (in problem order); `build_evaluator` receives the approximator's block. Do not assume it is a `Vector{Float64}` — it may be a `SubArray` or a vector of `ForwardDiff.Dual` numbers.

**The evaluator must be callable on scalars.** The dynamics function calls `p.<name>(x)` with a scalar `x` and expects a scalar back. (If your dynamics call the function with several arguments, accept those too.)

**The evaluator must be ForwardDiff-Dual-safe if used with autodiff solvers.** `AdamSolver`, `MultipleShootingSolver`, `TwoStageSolver`, `BNGSolver`, `IntegralMatchingSolver`, and the stiff-ODE Jacobian paths differentiate through the evaluator with ForwardDiff. Write eltype-generic code: no `Float64[]` scratch arrays, no `convert(Float64, ...)` on values that depend on `params_k` or `x`.

**`penalty_matrix` acts on the raw coefficients.** Whatever basis your coefficients live in, `S` must be the penalty in that same basis, since solvers form `βᵀSβ` directly on the coefficient block.

## Worked example: a polynomial approximator

A global polynomial `f(x) = Σ βⱼ xʲ⁻¹` with an (optional) ridge penalty on the curvature-carrying coefficients:

```julia
using PartiallySpecifiedModels
import PartiallySpecifiedModels: nparams, initial_params, penalty_matrix,
                                build_evaluator

struct PolyApproximator <: PartiallySpecifiedModels.AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    degree::Int
end

nparams(a::PolyApproximator) = a.degree + 1
initial_params(a::PolyApproximator) = zeros(a.degree + 1)

function penalty_matrix(a::PolyApproximator)
    # Ridge on the coefficients of x², x³, … — constants and lines are
    # "maximally smooth" and unpenalized, mirroring the spline penalties.
    S = zeros(a.degree + 1, a.degree + 1)
    for j in 3:(a.degree + 1)
        S[j, j] = 1.0
    end
    S
end

function build_evaluator(a::PolyApproximator, params_k)
    # Horner evaluation; eltype-generic, so Dual-valued params_k works.
    coeffs = collect(params_k)
    function poly_eval(x)
        acc = coeffs[end]
        for j in (length(coeffs) - 1):-1:1
            acc = acc * x + coeffs[j]
        end
        acc
    end
    poly_eval
end
```

That is the whole extension. The approximator now works end-to-end:

```julia
decay!(du, u, p, t) = (du[1] = -p.f(u[1]))

# synthetic decay data with true rate f(u) = 0.35u
ts = collect(0.0:0.5:5.0)
obs = reshape(exp.(-0.35 .* ts), :, 1)

approx = PolyApproximator(:f, (0.0, 1.0), 2)
prob = PSMProblem(decay!, [1.0], (0.0, 5.0), [approx];
                  data_times=ts, data_values=obs, obs_to_state=[1],
                  likelihood=Gaussian(), solver=Tsit5())
sol = solve(prob, AdamSolver(maxiters=500))
fhat = sol.unknown_functions[:f]   # callable fitted polynomial
```

## Which solvers consume which functions

All 23 solvers use `nparams` (parameter layout), `initial_params` (starting values), and `build_evaluator` (turning coefficient blocks into the callables the dynamics receive and into `unknown_functions`). `penalty_matrix` is consumed by the solver families that apply explicit smoothing:

| Solver family | `penalty_matrix` used for |
|:---|:---|
| Penalized likelihood (`LAML`, `GCVSolver`, `CollocationLAML`) | Smoothing-parameter estimation (LAML/GCV); an SPD-restricted penalty is required for EDF computation |
| Gradient matching (`TwoStageSolver`, `AdaptiveGradientMatching`, `BNGSolver`, `IntegralMatchingSolver`) | Fixed-λ smoothing penalty added to the matching loss |
| Gradient matching, estimated λ (`GradientMatching`) | Smoothing-parameter estimation, like the penalized-likelihood family |
| Through-the-solver (`AdamSolver`, `MultipleShootingSolver`, `DerivativeFreeSolver`) | Optional penalty term (`penalty_weight`) |
| Probabilistic numerics (`RodeoSolver`, `DaltonSolver`) | Smoothing penalty inside the marginal-likelihood objective |
| Bayesian (`MCMCSolver`, `MagiSolver`, `VariationalSolver`, `PseudoMarginalSolver`, `ABCSolver`) | Gaussian prior precision on the coefficients |

Returning `nothing` from `penalty_matrix` is always safe: the approximator is then simply unpenalized (as `NeuralApproximator` is, relying on implicit regularization).

Three caveats for custom types:

- A handful of solver capabilities are gated on the built-in types (e.g. `AdaptiveGradientMatching`'s population-MCMC mode rejects `NeuralApproximator`; `optimize_spde_range` only operates on SPDE approximators). Custom types pass these gates like any non-listed type.
- Six solvers (`TwoStageSolver`, `IntegralMatchingSolver`, `MagiSolver`, `AdaptiveGradientMatching`, `RodeoSolver`, `DaltonSolver`) assemble their smoothing penalty through per-type code whose built-in treatment is historical and deliberately preserved (the lists differ between solvers, and some build spline penalties on domain rather than unit knots). Custom types are **not** affected by those quirks: every one of these sites falls back to the generic `penalty_matrix(approx)` for any non-built-in type, so a custom penalty flows in everywhere — as it always did in the penalized-likelihood, through-the-solver, and MCMC/VI/ABC families.
- The adaptive GP hyperparameter refitting inside `LAML`/`GCVSolver` is specific to `GPApproximator`; custom types keep whatever structure their four methods define.
- `confidence_band` (diagnostics) evaluates approximators through its own restricted mechanism and supports only the built-in basis-expansion types — custom types (like `NeuralApproximator`/`COMONetApproximator`) are not supported there and will error.

One further subtlety: the default penalty enumeration only includes a term
when the approximator has at least 3 parameters (the gate lives in the
default `penalty_blocks` method), so a 1–2 parameter custom type is
effectively unpenalized under `LAML`/`GCVSolver` even if its
`penalty_matrix` is nonzero — unless it overrides `penalty_blocks`, which
may declare blocks of any size.

## Optional: multiple penalty blocks (`penalty_blocks`)

A type whose coefficients naturally split into parts that should be
smoothed *independently* can implement the optional fifth protocol
function, `penalty_blocks(a)`, returning a vector of
`(S, local_range)` pairs. Under the penalized-likelihood solvers
(`LAML`, `GCVSolver`, `CollocationLAML`, `GradientMatching`,
`ProfileLikelihoodSolver`) each block then receives its **own** smoothing
parameter, estimated jointly, and `PSMSolution.smoothing_params` carries
one entry per block. Ranges are local to the approximator's coefficient
block and must be pairwise disjoint; each `S` must be
`length(range) × length(range)`, symmetric PSD. The default method
returns the single block `(penalty_matrix(a), 1:nparams(a))` (or none,
under the gate above), so single-penalty types need not care.

Types overriding `penalty_blocks` should usually also provide a
consistent `penalty_matrix` — the block-diagonal merge of the blocks with
fixed weights — because the single-λ consumers (the per-type penalty
sites in `TwoStageSolver`/`IntegralMatchingSolver`/`MagiSolver`/
`AdaptiveGradientMatching`/`RodeoSolver`/`DaltonSolver`, and the
MCMC/VI/ABC prior builder) read `penalty_matrix` only and apply one
weight to the whole approximator.

Two practical notes. Block validity (contiguous, in-range, correctly
sized, symmetric, pairwise disjoint) is checked when a penalized-
likelihood solver assembles the penalties, not at `PSMProblem`
construction — so a malformed `penalty_blocks` is simply inert under the
solvers that never read it. And if your evaluator is not smooth in its
argument — a kink from `floor`/`clamp` indexing, a branch on the input —
prefer `jac=:forwarddiff` on `LAML`/`GCVSolver`/`CollocationLAML`: the
default finite-difference prediction Jacobian picks up adaptive-step
noise at the kink, and the smoothing search can then stall well short of
the optimum while still reporting convergence. This is a property of
evaluator smoothness, independent of how many penalty blocks you
declare.

If you pass an approximator type that does not implement the interface, `build_evaluator` fails with an error listing the four required functions.

```@docs
build_evaluator
penalty_blocks
```
