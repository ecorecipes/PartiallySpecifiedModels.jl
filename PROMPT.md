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
- **Nine vignettes** fit with λ̂ stuck at its initialization.
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
