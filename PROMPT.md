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
6. **Vignettes and docs.** All 40 vignettes render; nine currently emit a
   warning that λ̂ never left its initialization, which is honest output whose
   surrounding prose has not been updated.

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

- **`jac=:fd` at nk=9** on the quadrature fixture sits at 5.9e-3 relative
  where the best achievable step reaches 8.3e-6. Diagnosed: the curvature
  stop fires at SNR 5383 because `d2max ≈ h²f''` is compared against a fixed
  noise level, and relaxing it hands the column to the validation guard,
  which reverts it to the same step. Both guards would have to move together,
  and the stiff fixture that motivated them is **not in the suite**.
- **Nine vignettes emit the smoothing warning** — 03_lotka_volterra,
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
