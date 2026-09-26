# Second review brief

This file asks GitHub Copilot to review **PartiallySpecifiedModels.jl**
independently. It exists because the package has already had eight
correctness campaigns and one seven-dimension review; a second reviewer is
useful only if it looks where the first one did not, and does not re-report
what has already been checked and dismissed.

Read `.github/copilot-instructions.md` first — it carries the architecture,
the reporting semantics, and the traps. Most false findings against this
package come from misreading those semantics rather than from misreading code.

## How to use this file

Paste the section you want as a prompt, or point Copilot at this file and ask
it to work through **Scope** below in order. Every finding must satisfy the
**Evidence bar**. Report using the **Output format**.

## Evidence bar — this is the important part

The first review checked eight candidate findings and **discarded six**. The
discards were not sloppiness; they were the normal rate for a package this
heavily worked. Assume most of what looks wrong is not.

Before reporting anything:

1. **Run it.** Do not report a defect you have not reproduced. Use
   `--check-bounds=yes`, or you are not reproducing suite behaviour.
2. **Check the semantics table** in the Copilot instructions. `converged=true`
   next to a terrible fit, a *lower* `data_loss` for a worse model, and
   `reason=:maxiters` from a sampler are all **correct behaviour**.
3. **Prove it is load-bearing.** Revert your fix and confirm the failure
   returns. If output is byte-identical with and without your change, the
   change does nothing — say so and drop it.
4. **State the measurement, not the impression.** "Relative error 2.85 versus
   8.37e-5 at nk=10" is a finding; "the Jacobian looks imprecise" is not.
5. **Scope the claim.** If it fails at one parameter count and not others, say
   which. If it is platform-specific, say so — one defect here reproduces on
   macOS and not ubuntu.

## Scope, in priority order

1. **Correctness of the numerics.** `src/laml.jl` (Fellner–Schall,
   dispersion), `src/solver.jl` (IRLS, `compute_jacobian!`),
   `src/collocation_solver.jl`. Look for quantities computed one way and
   reported another.
2. **Cross-solver consistency.** 23 solvers, and the recurring defect is a fix
   applied to one and not its siblings. Pick an invariant — how `converged` is
   set, what `objective` means, whether `fitted_values` are a simulation — and
   check all 23 against it.
3. **Reporting honesty.** Anywhere a solver can return a plausible-looking
   solution from a failed fit without saying so.
4. **Test quality.** Assertions that would pass on broken code; tolerances
   pinned to one machine; missing round-trip or parameter-recovery coverage.
5. **Faithfulness to the cited literature.** Docstrings cite Wood, Ramsay,
   Wenk, Hoffman & Gelman and others. Check the implemented formula against
   the citation.
6. **Vignettes and docs.** All 41 vignettes render (41_kan is new). Four
   fits across 04, 28, 29 and 38 still emit the warning that λ̂ never left
   its initialization — honest output; see the known-open entry below for
   which causes are diagnosed and which are not.

## Already checked — do not re-report without new evidence

- `converged=true` on a diverged fit — by design; `stationarity` flags it.
- `data_loss` lower for a worse model — it is weighted RSS, minimised by
  fitting `E[Y]`.
- `FGPGMSolver` reporting `converged=false` — correct for a fixed-budget
  sampler, and documented.
- `PolyApproximator` "missing from src" — it is the extension-protocol example
  in `docs/src/extending.md`.
- DDE construction "undocumented" — `delays` is in the `PSMProblem` docstring
  and vignette 20 is a full worked example.
- `sigma2_init` "ignored" — its cap relaxes a decade per iteration after
  `warmup`, by design.
- `GradientMatching` posting a huge `data_loss` — it reports
  `converged=false, reason=:line_search_failure` alongside.

## Known open — describe better, do not rediscover

- **`jac=:fd` at nk=9 — RESOLVED 2026-09-16.** Three changes in
  `compute_jacobian!`, moved together: the curvature stop requires a signal
  above the noise AND curvature that is a real fraction of it
  (`_FD_CURV_FRAC`); the growth-validation revert compares only against a
  RESOLVED previous column; and the per-column noise estimate is refined at
  every grown step that the relative test classifies as noise (the re-solve
  jitter is step-dependent — 34x the floor on the offending column). nk=9
  went 5.91e-3 -> 4.63e-5 under BOTH package stacks; all sizes <= 1.33e-4;
  the stiff probe is unchanged at 1.5e-5. The suite asserts `rel < 1e-3`
  for nk in 5,6,7,8,9,10,12.

  **Trap found on the way:** the local `Manifest.toml` (gitignored) was
  stale — `Pkg.test` and CI resolve a fresh environment (OrdinaryDiffEq
  6.111 vs 6.108 locally), and the FD noise structure differs between the
  two. A fix measured only under `--project=.` passed locally and failed
  under `Pkg.test`. Measure FD claims under `Pkg.test`, or refresh the
  Manifest first.

- **Nine vignettes emit the smoothing warning — UPDATED 2026-09-15.** The
  dominant cause was found and fixed: LAML's accept block let a
  sub-tolerance old-θ improvement (measured 5.74e-13) veto a new-θ step
  that descended its own penalized objective, so λ̂ never advanced.
  `_laml_prefer_old_step` in `src/solver.jl` now requires the old-θ gain to
  be material (above `tol`). Re-rendering all nine, warnings went
  **12 → 4**: 03, 06, 12, 13, 18 cleared entirely; 28 and 29 each dropped
  from 2 to 1; 04 and 38 unchanged. So the veto explained 8 of 12.

  RESOLVED 2026-09-16 for three of the four. 28 and 29: the warning came
  from inside `bootstrap` (one replicate of 50); replicate fits are now run
  under a logger that drops the per-fit warnings, and `BootstrapResult`
  carries `n_smoothing_stalled` / `n_ridge` with one aggregate warning.
  04_copepod: neither cause — at λ₀ = 1/tr(S) (EDF 41) the 45 coefficients
  kept improving by more than `tol` for ~130 iterations and the accept block
  preferred that progress over the (identical, re-proposed) Fellner–Schall
  λ̂ every time; `maxiters=100` ran out first. Proposal deferral is now
  bounded (`_LAML_MAX_DEFER = 5`): 136 → 18 iterations, same fixed point
  (EDF 2.28, λ̂/λ₀ 1e23 — LAML says these counts support only near-linear
  trends; the fit reports `ridge = true` for the weakly determined
  null-space components, explained in the vignette).
  Deferral bound VALIDATED 2026-09-25: the same six-fixture script under
  `_LAML_MAX_DEFER = 5` and under an unbounded copy (1e9) — exp-growth
  (fd, forwarddiff), logistic Poisson, LV2 at both suite settings, Poisson
  SIR warm-start, copepod: bit-identical iterations/λ̂/EDF/data loss on five
  (`cap = 0`, 7–17 iterations); copepod 132 → 19 iterations, same fixed
  point (EDF 2.269 vs 2.275, loss 3.799e9 vs 3.7991e9). The bound changes
  nothing that did not defer past five iterations.
  SUPERSEDED 2026-09-25 — the "LAML says the copepod data support only
  near-linear trends" reading was WRONG. It was the optimiser: performance
  iteration (FS/Newton on a frozen linearisation) drove all three λ to
  RHO_MAX where the true LAML criterion is V = −987.5, against −951.9 at a
  common λ = 1e8 and −952.4 at the GCV solution (Wood's own criterion, which
  recovers the simulated recruitment pulse). Measured with `fixed_lambda`
  profiles and a direct `laml_objective` evaluation at the GCV fit (the
  instrument reproduces the solver's reported `laml` to all digits). Fix in
  `solve(::LAML)`: `_LAML_MAX_RHO_STEP` (≤ ×1e3 per iteration) plus an
  incumbent on the true V with backtrack-and-freeze
  (`_LAML_V_BACKTRACK = 3`, `_LAML_V_TOL = 0.5`, key
  `smoothing_backtracked`). Copepod now V = −953.0, EDF 7.85, interior λ̂.
  Remaining difference from GCV (bump in R vs U-shaped μ_j) is criterion
  flatness — 0.6 log-units — i.e. identifiability with 10 sampling times,
  not a defect. Wood's data: simulated, ADDITIVE NORMAL noise (SD 8 on his
  scale); Poisson/NegBin are misspecified for it and do not help.

- **NegativeBinomial(θ=25) on the copepod DIVERGED and reported
  `converged = true` — RESOLVED 2026-09-26** (was: EDF 0.00, data loss
  2.6e114, all λ at RHO_MAX, μ_a negative). Two changes in `solve(::LAML)`:
  (i) the true-criterion safeguard is refereed by the criterion the search
  actually optimises — the family LAML for Gaussian/`:laplace`, the
  Pearson-scaled WORKING-model REML for non-Gaussian `:working` (the family
  V vetoed every FS move on this fit, −2620 vs −2774, and froze λ at λ₀);
  now λ̂ = 2.02 ×3, EDF 12.9, backtracked at iteration 23, fitted values
  within the data's range. (ii) an exit DIVERGENCE GUARD: non-finite fitted
  values, or a data loss > 1000x the iteration-0 value, report
  `converged = false, reason = :diverged` (warning `:laml_diverged`,
  suppressed inside bootstrap). The guard has no natural trigger left in
  the suite — it is documented by the measurement, not by a fabricated
  fixture. Regression test: the NegBin(25) block in the copepod testset.
- **CollocationLAML walks the copepod to the λ boundary — KNOWN OPEN
  (measured 2026-09-26).** Same march as the main loop had: λ → [8e15,
  6e15, 3e18], EDF 1.00, `:plateau`, ridge, data loss 3.77e9. Its
  per-level FS ratio step is unbounded (clamp 1e-20..1e20, beyond
  `RHO_MAX`), and at those λ the verbose log shows NEGATIVE penalty values
  (β'S_λβ = −348 … −164200 — a PSD quadratic form overflowing). A direct
  port of the LAML safeguard (per-level cap log(1e3) + backtrack against
  the working criterion) fixed this fixture (λ̂ [1.8e5, 3.9e7, 6.3e8], EDF
  9.1, converged, no ridge) but REGRESSED the likelihood-aware fixtures:
  Gaussian λ̂ 574.8 → 175.9 (the FS fixed point needs more capped steps
  than the continuation has levels), Poisson counts non-convergent (a
  capped DOWNWARD step held λ at 1.3e-5 where the old full jump to an
  unpenalized fit converged; Pearson σ² 1.6e11). Reverted. Proposed design:
  cap UPWARD moves only, and iterate FS (with the backtrack) to its fixed
  point at the FINAL continuation level instead of once. Fixed on the way:
  a callable u0 crashed CollocationLAML with `length(::Function)`.
  GCVSolver needs no safeguard — it searches its criterion directly (grid
  + refine) and landed interior (λ_R = 353) on the same fixture.
- **LAML trust region and convergence — FOUND ON CI 2026-09-26.** With the
  cap, two smooths saturating it from a common λ₀ move identically, and the
  objective-stability test could fire mid-path: Julia 1.13/macOS reported
  both λ̂ tied at exactly 9.999999999999998 (TransformedCovariate SIR
  fixture). A clamped proposal now defers `converged_tol` to the next
  iteration.
  STILL OPEN (benign): 38_transformed_covariates — null-space collapse,
  `λ = 1`, `edf = 1.0000`, the documented HOLD branch (table below).

  The original entry follows, kept for the ruled-out table.

- **(original) Nine vignettes emit the smoothing warning** — 03_lotka_volterra,
  04_copepod, 06_solver_comparison, 12_discrete_time, 13_shape_constraints,
  18_dalton, 28_fisheries, 29_bootstrap, 38_transformed_covariates. Each
  contains at least one fit finishing with λ̂ still on its initialization.
  Partially investigated; here is exactly how far it got, so you neither
  redo it nor over-trust it.

  **Ruled out** on 38_transformed_covariates, the one fixture reconstructed
  in full (all four runs under `--check-bounds=yes`):

  | configuration           | advanced | edf    | λ | data_loss |
  |-------------------------|----------|--------|---|-----------|
  | jac=:fd, maxiters=40    | false    | 1.0000 | 1 | 183.75    |
  | jac=:fd, maxiters=300   | false    | 1.0018 | 1 | 183.71    |
  | jac=:forwarddiff, 40    | false    | 1.0000 | 1 | 183.78    |
  | jac=:fd, 40, warmup=10  | false    | 1.0000 | 1 | 183.75    |

  So it is NOT the `jac=:fd` noise cliff (forwarddiff does not help) and NOT
  an iteration budget (300 does not help). The signature there is `λ = 1`
  exactly with `edf = 1.0000`: the outer smooth has collapsed into the
  penalty null space, so `βᵀSβ ≈ 0` and LAML's documented HOLD branch
  correctly declines to move λ. The machinery is behaving as designed and
  that particular fit is degenerate.

  **Not established.** The nine are probably not one phenomenon: their
  printed EDFs span 2.0 to 10.0, and only 38 shows the degenerate `edf ≈ 1`,
  so the null-space explanation does NOT transfer to the other eight. Do not
  assume it does — generalising from this single fixture is precisely the
  error to avoid.

  **Two traps** if you pick this up:
   - The warning uses `maxlog=1`, so one warning means *at least one*
     affected fit, not that the vignette's headline fit is affected. Several
     of these documents run many solves, and the EDFs above may belong to
     healthy ones.
   - Identifying *which* solve warns needs the renders instrumented
     individually; reading the rendered `.md` is not enough, because the
     warning is not adjacent to the call that caused it.

  The open question worth answering: for each of the remaining eight, which
  fit warns, and is its cause the null-space hold (benign, correctly
  reported) or something else.
- **`_vi_edf`, Rodeo/Dalton FS, ABC/AGM/FGPGM convergence keys** were
  addressed in earlier campaigns; verify rather than assume.

- **The same LV2 fixture is Julia-VERSION sensitive.** On Julia 1.13.0 (what
  the CI matrix's `'1'` now resolves to) the fit exits on a non-stationary
  ridge — stationarity 0.392, `converged = true` by the stability semantics,
  λ̂ = [0.012, 1.43e7] — where 1.12 reaches stationarity 3.4e-3. Honestly
  reported, not a defect in the reporting; but a design question for the
  accept-block change: should a step be accepted that exits on a ridge?
  The refit-move proxy is now asserted only when stationarity < 1e-2.

- **Context-dependent optimum on the LV2 consistency fixture.** After the
  accept-block fix, the "λ̂ and β̂ are mutually consistent" fixture (hard-coded
  data) converges to a penalty/data-loss ratio of 1.18 under `Pkg.test` and
  0.002 under `--project=.`, both converged with the pairing invariant
  intact. The consistency test now gates the ratio at 100× (the F1 defect it
  guards sits at 8e7). Worth knowing before trusting any single-run λ̂ on a
  multi-smooth fixture; not itself a defect.

## Output format

For each finding:

- **Claim** — one sentence.
- **Evidence** — the command run and the numbers returned, before and after.
- **Scope** — which configurations, platforms, parameter counts.
- **Load-bearing** — what failed when you reverted the fix.
- **Severity** — silently wrong results > misreported diagnostics > docs.

An empty report is an acceptable outcome. A list of plausible-sounding
concerns without measurements is worse than nothing, because someone has to
spend real time disproving each one.
