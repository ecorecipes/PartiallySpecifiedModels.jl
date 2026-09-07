# Copilot instructions — PartiallySpecifiedModels.jl

## What this package is

Partially specified models (PSMs): ODEs, DDEs and discrete-time maps in which
one or more functional responses are left unspecified and estimated
nonparametrically — penalized splines, Gaussian processes, neural networks —
rather than assumed to have a fixed parametric form. The application domain is
ecology and epidemiology, where density dependence, functional responses and
transmission rates are often of uncertain shape.

Scale: ~25k lines of source, ~12.7k lines of tests (2760 assertions), 23
solvers, 11 approximators, 5 likelihoods, 40 vignettes.

## Architecture — four composable abstractions

1. **`PSMProblem`** (`src/types.jl`) — dynamics function, `u0`, tspan, the list
   of approximators, observed data (`data_times`, `data_values`,
   `obs_to_state`), a likelihood, a solver.
2. **Approximators** (`src/approximators.jl`) — the nonparametric stand-ins.
   In the dynamics an unknown function is a **callable parameter**:
   `du[1] = p.r(N) * N`, or `p.g(N, P)` for tensor/single-index types.
3. **Likelihoods** (`src/likelihoods.jl`) — `Gaussian`, `Poisson`,
   `NegativeBinomial`, `TruncatedNormal`, `CustomLikelihood`.
4. **Solvers** — `solve(prob, SolverType())` returns a `PSMSolution`.

Fitted parameters live in a `ComponentArray` with one named section per
approximator, which is why they are reachable as `p.<symbol>`.

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # NOT `julia test/runtests.jl`
julia --project=docs docs/make.jl              # docs
cd vignettes && quarto render NN_x/NN_x.qmd --to gfm
```

`Pkg.test()` defaults to `--check-bounds=yes`, which changes SIMD and
floating-point association. **Ad-hoc scripts must pass `--check-bounds=yes`
explicitly** or they will not reproduce suite behaviour. This has produced
wrong conclusions before.

## Reporting semantics — read before judging any fit

These are the package's deliberate conventions. Misreading them produces
false bug reports.

- **`converged` is a STABILITY test, not a verdict on fit quality.** It means
  the iteration stopped changing. A fit can report `converged = true` with a
  catastrophic error. The companion diagnostics are the quality signals:
  - `convergence.stationarity` — how close the smoothing criterion is to
    stationary. Values like 1e26 mean the fit is garbage, honestly reported.
  - `convergence.smoothing_advanced` — `false` means λ̂ never moved off its
    `1/tr(S)` initialization, so the EDF and posterior covariance describe the
    *initial* smoothing. The solver warns on this.
- **`data_loss` is the weighted RSS `Σw(y−μ)²`**, not −2·loglik. It is
  minimised by fitting `E[Y]`, so for `TruncatedNormal` a *defective* fit can
  post a *better* `data_loss`. Judge such fits by log-likelihood.
- For **state-estimating solvers** (`CollocationLAML`, `ODINSolver`,
  `RKHSSolver`) `fitted_values` are the estimated state, not a simulation, so
  `data_loss` measures the state's fit. `convergence.simulated_data_loss` is
  the companion that measures the model.
- Fixed-budget samplers (`FGPGMSolver`) always report
  `converged=false, reason=:maxiters`. That is correct, not a failure — judge
  them by R̂/ESS of `convergence.chains`.

## Jacobian modes

`jac=:fd` (default) makes no genericity demands on user dynamics.
`jac=:forwarddiff` is exact to solver precision and usually faster at small
parameter counts, but was measured **25× slower at 12 parameters**. Do not
propose switching the default on accuracy grounds alone.

## Test conventions

- **Every tolerance carries a comment citing a measured number and the
  headroom.** Do not add a bare numeric gate.
- **Never pin the output of a nonlinear optimisation tightly.** λ̂, EDF and
  iteration counts vary across platforms and dependency versions. CI failed
  for months because 26 assertions were calibrated on one machine, and ubuntu
  and macOS failed *disjoint* sets. Assert the property, not the number.
- **`Manifest.toml` is gitignored**, so CI resolves dependencies fresh. A test
  that only passes against the local manifest is broken.
- **Defect-presence assertions are only legitimate for deterministic
  defects.** One that pinned a platform-specific outlier as "present" failed
  on CI.
- After changing source, **revert the change and confirm the new tests fail.**
  Green tests prove nothing on their own; three tests written in this
  codebase passed while testing nothing.

## Known traps

- **Sibling parity.** The recurring defect class: a fix lands in one solver
  and not its siblings. When changing one solver, grep for the same pattern in
  the others. `src/solver.jl` holds an FS parity matrix.
- **Free-spline expressiveness.** With a free spline, a *wrong* fixed
  parameter or a structurally wrong RHS is not necessarily a worse model — the
  nonparametric term absorbs it. Do not assert "the true parameter fits
  better"; it often does not.
- **Bivariate and index approximators are only identified where the data go.**
  An ODE trajectory is a curve, not a filled region. Score surfaces on the
  realised path.
- **Shape-constrained GP/SPDE evaluators enforce constraints at the inducing
  points, not between them.** Use `check_constraints` to audit; do not claim a
  guarantee.
