# LAML smoothing-priority correction

**The tensor smoothing stalls were an acceptance-priority defect, not
evidence that the tensor basis was incapable of fitting the data.**
After the correction, all forty retained tensor LAML fits advance
smoothing, compared with six before. Their median smoothing-stationarity
residuals fall from about 0.1 to roughly `1e-6`--`3e-6`.

Prediction improves substantially in three of the four tensor
case/split groups. The forecast single-index fixture remains difficult:
its median trajectory error barely changes. Correct smoothing selection
does not guarantee extrapolation quality or identify an unobserved response
surface.

## Cause and correction

LAML constructs two coefficient steps: one under the accepted smoothing
parameters and one under a new Fellner-Schall proposal. The old code gave
priority to **any** positive improvement at the old penalty if the new
penalty increased training loss. This could indefinitely defer the existing
new-penalty descent fallback, even after the old step was far smaller than
the tolerance at which the solver was willing to stop.

Instrumented copies reproduced the original seed-201 tensor fits exactly.
At the final rejected proposals:

| Local fixture | Old-penalty objective improvement | New-penalty own-objective improvement | Candidate LAML criterion improvement |
|---|---:|---:|---:|
| Single index | 5.74e-13 | 0.00683 | +9.21 |
| Interaction | 6.74e-14 | 0.00420 | +7.45 |

The new candidates slightly increase training RSS, as a legitimate increase
in smoothing can do. A likelihood-only veto therefore cannot establish
which smoothing parameter is preferable. Conversely, absolute penalized
objectives at different penalties are not comparable either.

The correction is deliberately narrow: when the old step's gain is at most
`tol * max(abs(old_objective), 1)`, it no longer blocks a **finite new step
that decreases its own penalized objective**. Material old-step progress
keeps its existing priority, and any improving old step remains available
when the new proposal has no finite descent.

This reuses the solver's existing convergence precision and fallback.
It does not change the Fellner-Schall/Newton formulas, Jacobian default,
accepted-state warm start, or pairing of reported coefficients with the
smoothing parameters that produced them. `converged` remains an
iteration-stability flag, not a fit-quality judgement.

### Related paths

| Path | Relationship to the correction |
|---|---|
| LAML main loop | Contains the competing old/new penalty steps and priority defect; corrected |
| Non-Gaussian Gaussian warm start | Separate initialization heuristic; unchanged |
| GCV | Selects smoothing then contracts one step at that penalty; no competing old-step veto |
| Collocation LAML | Advances smoothing between continuation levels; no equivalent priority chain |
| Gradient matching, Rodeo and Dalton FS | Do not use this acceptance chain; formulas unchanged |

## Same-protocol reassessment

Only the affected LAML tracks were rerun: forty constant-start forecast fits
and 240 local candidates using the existing eleven-start index grid.
Seeds, data, validation/test splits, starting configurations, 25-iteration
ceilings and package versions are unchanged.

All 280 candidates returned finite training/validation results, and all
160 selected-fit test trajectories completed finitely. The old archives,
and the Adam/GCV results, were not overwritten or silently relabeled as
new-source runs.

Errors below first combine the two held-out trajectories within seed, then
take the median across ten seeds. Paired wins compare the corrected and
original tensor fits on the same seed.

| Split | Truth | Smoothing advanced, before/after | Trajectory NRMSE, before/after | Response RMSE, before/after | Corrected trajectory wins |
|---|---|---:|---:|---:|---:|
| Forecast | Single index | 4/10 -> 10/10 | 0.05962 -> 0.05963 | 0.04417 -> 0.04132 | 4/10, one tie |
| Forecast | Interaction | 1/10 -> 10/10 | 0.02894 -> 0.01285 | 0.03374 -> 0.01294 | 9/10 |
| Local | Single index | 0/10 -> 10/10 | 0.01781 -> 0.00419 | 0.02082 -> 0.00969 | 10/10 |
| Local | Interaction | 1/10 -> 10/10 | 0.00584 -> 0.00314 | 0.02124 -> 0.00967 | 9/10 |

Tensor median stationarity changes:

| Split | Truth | Before | After |
|---|---|---:|---:|
| Forecast | Single index | 0.11395 | 2.73e-6 |
| Forecast | Interaction | 0.10014 | 1.24e-6 |
| Local | Single index | 0.10732 | 1.19e-6 |
| Local | Interaction | 0.11048 | 3.33e-6 |

The final reported LAML criterion has positive median changes in all four
groups, from +7.01 to +7.76. This is not a globally monotone nonlinear
optimization guarantee: one previously advanced forecast fit changes by
-0.000571. The correction merely removes the inconsistent priority that
prevented a valid smoothing step from being tried.

Selected tensor fit times remain similar: about 0.63--0.70 seconds after
the correction, versus about 0.60--0.83 before. These are warmed,
single-machine, serial-track measurements, not hardware-independent speed
claims.

### Index controls and remaining limitations

The forecast single-index controls are numerically unchanged and still fit
poorly from the constant response start. That is the separate
[initialization issue](LOCAL-ASSESSMENT.md#comparison-with-the-constant-initialized-native-fit),
not repaired by changing smoothing-step priority.

The local multistart index fits change little in accuracy: trajectory
medians remain about 0.00183 and 0.00238, and response medians remain
0.00785 and 0.00917. Some selected candidates and stationarity residuals
change, so their full tuning costs and provenance remain separate records.

The stronger tensor baseline narrows some earlier gaps but does not
establish a universal model ranking. The shared-Adam, native-solver and
initialization-search tracks still answer different questions. In
particular, the forecast single-index test trajectories remain outside the
declared joint training neighborhood.

## Regression evidence

A deterministic three-column linear model isolates the defect without
pinning a nonlinear optimizer output. Its old-penalty gain is `1.07e-8`
against the existing `1e-6` tolerance, while the proposed penalty gives an
own-objective gain of `0.00824` and a LAML criterion improvement of `5.34`,
despite increasing training RSS.

The permanent regression group covers this priority, non-finite and
non-descending alternatives, tensor recovery, and a model that cannot fit
its observations but correctly reports iteration stability. Reverting the
preference restores nine failures. A fifteen-fixture paired
characterization also covers likelihood families, FD/AD, DDEs, smooth/kinked
maps, multiple penalties and no penalties.

The formerly nonstationary kink fixture improves from a residual of
0.0737 to `1.33e-5` in that characterization. Its old defect-presence
assertions were therefore removed rather than requiring the solver to stay
defective; finite/nonnegative diagnostic checks and the deterministic
stable-but-bad fixture preserve the reporting contract.

The restored implementation passes 1,457 focused assertions and 340
selected existing LAML/Fellner-Schall/PCLS/Laplace and sibling assertions.
The full pre-change baseline was interrupted during unrelated Adam work
after 1,139 legacy assertions, in addition to its completed focused groups;
it is **not** reported as a completed full-suite run. The existing runner
now supports named legacy testset selection to avoid that unrelated cost:

```bash
julia --project=. --check-bounds=yes -e \
  'using Pkg; Pkg.test(test_args=["laml-stalls"])'

julia --project=. --check-bounds=yes -e \
  'using Pkg; Pkg.test(test_args=["--testset=LAML|Fellner|PCLS|Laplace"])'
```

The selector keeps shared imports/fixtures, executes complete matching
testsets and fails on an empty selection. An implicit module alias in one
legacy index testset was made explicit so that testset also works alone.

## Reproduction and provenance

Use fresh output directories:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --split=forecast --track=laml --models=tensor64,single_index9 \
  --starts=constant --iterations=25 \
  --output=benchmarks/kan/results/corrected-forecast-laml

julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --split=local --track=laml --models=tensor64,single_index9 \
  --starts=index-multistart --iterations=25 \
  --output=benchmarks/kan/results/corrected-local-laml

python3 -B benchmarks/kan/compare_laml.py
```

The [forecast](results/corrected-forecast-laml/) and
[local](results/corrected-local-laml/) archives retain every candidate,
selected weights, diagnostics, geometry and environment metadata.
[Paired results](results/laml-correction-analysis/paired.csv),
[summaries](results/laml-correction-analysis/summary.csv), and
[cross-source provenance](results/laml-correction-analysis/metadata.json)
keep the before/after sources explicit.

[Traces and the paired characterization](results/laml-stall-evidence/)
retain the investigation evidence. Their reproduction recipes require
`PSM_STALL_SOURCE_ROOT` and the matching Julia project to point at the
pre-correction source snapshot; they reject the corrected acceptance site.

The old fitting source remains in `results/source-e309b452.tar.gz`.
The corrected fitting source and benchmark manifest are preserved in
[`source-59084871.tar.gz`](results/source-59084871.tar.gz), with source hash
`59084871c9c6dd2fdff17759bf1e51815b86b0f1106b542492623743730c568e`
and archive SHA-256
`e252af8dfa694c7107d7c30c9860b549e1093dd6c6fffee13faca6f64c8f2223`.
