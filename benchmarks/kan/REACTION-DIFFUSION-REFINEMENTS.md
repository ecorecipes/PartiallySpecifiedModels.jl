# Reaction-diffusion optimizer sensitivity and mesh transfer

This continuation refines the [original spatial assessment](REACTION-DIFFUSION-ASSESSMENT.md).
The original fits and raw archives are immutable. The new studies distinguish
optimization sensitivity from spatial discretization; neither is a new KAN architecture.

**Outcome:** stricter stopping materially improves the composed KAN and the
MLP on crowding. The MLP has the lowest median error after validation-only
condition selection, while KAN wins six of ten paired seeds against it.
The original short-budget ranking was therefore optimizer-sensitive, not
an approximation-family verdict. No-refit mesh transfer preserves the
original median accuracy rankings but changes the fields by an amount
comparable to some fitting errors.

## Declared optimizer protocol

The nonlinear crowding case uses the original seeds 301--310 and all five
models: spline8, spline28, MLP28, shallow KAN28 and composed KAN28. Each
model/seed keeps the penalty `eta` selected by the original validation study.
Every condition restarts from the same seeded initialization, not from the
previous fitted coefficients.

| Condition | Initial learning rate | Plateau tolerance | Early stopping | Budget |
|---|---:|---:|---|---:|
| `default` | 0.03 | 1e-4 | yes | 400 |
| `tight` | 0.03 | 1e-6 | yes | 400 |
| `full` | 0.03 | 1e-4, inactive | no | 400 |
| `slow_tight` | 0.01 | 1e-6 | yes | 400 |

The window remains 30 iterations. A common 400-iteration budget holds the
cosine schedule fixed across stopping-rule comparisons. The original
150-iteration results are context, not a pure stopping-rule control: their
schedule also differs. These are 200 new fits, with no extra penalty search.
The small-loss plateau statistic is absolute below one, so changing it can
matter even when the original fit reported stability.

Every condition is reported, followed by a separate validation-only selection
among the four. Test fields and true reaction coefficients never select a
condition. Search cost includes all four new fits and separately records the
original penalty-search cost; selected-fit timing alone is not tuning cost.
The experiment does not retune penalties for each schedule, sample additional
network initializations, or establish best attainable family performance.

## Optimizer results

All 200 new candidates and all 400 condition-specific held-out profiles
scored finitely. All 50 validation selections were usable.

Median held-out field RMSE, pooling the two profiles within each seed:

| Model | Original 150 | Default 400 | Tight 400 | Full 400 | Slow/tight 400 | Validation-selected |
|---|---:|---:|---:|---:|---:|---:|
| Spline8 | 0.001481 | 0.001575 | 0.001526 | 0.001526 | 0.001547 | 0.001570 |
| Spline28 | 0.002450 | 0.002082 | 0.002276 | 0.002300 | 0.002398 | 0.002140 |
| MLP28 | 0.003302 | 0.002577 | 0.000869 | 0.000869 | 0.001103 | 0.000869 |
| Shallow KAN28 | 0.001771 | 0.001687 | 0.001321 | 0.001319 | 0.001520 | 0.001244 |
| Composed KAN28 | 0.002934 | 0.002572 | 0.001156 | 0.001176 | 0.001601 | 0.001140 |

The controlled stopping contrast is between the **400-step columns**, not
between 150 and 400. Relative to default stopping at the same 400-step
schedule, tight stopping improves the composed KAN in 9/10 seeds and the
MLP in 10/10; median within-seed error ratios are 0.376 and 0.351.
Spline28 improves in only 4/10 seeds and its median ratio is 1.081.
More optimization does not uniformly improve every model's generalization.

Default stopping triggers in all 50 fits. Median default iteration counts
range from 75 to 87 across models. Tight stopping triggers in 46/50 fits;
the composed KAN uses 179--240 iterations, median 204. Full-budget fits
all use 400 iterations and honestly report `:maxiters`. Merely retaining
the original plateau policy while raising the ceiling is not equivalent
to optimizing more thoroughly.

Compared with the original protocol, validation-selected median field error
falls by 61% for the composed KAN, 74% for the MLP and 30% for the shallow
KAN. Those changes combine schedule, stopping and validation selection;
they are not pure stopping-rule effect sizes.

| Selected model | Field RMSE median [seed IQR] | Reaction RMSE median | Composed-KAN field wins against it |
|---|---|---:|---:|
| Spline8 | 0.001570 [0.001023, 0.001885] | 0.001779 | 5/10 |
| Spline28 | 0.002140 [0.001397, 0.002662] | 0.003499 | 8/10 |
| MLP28 | 0.000869 [0.000760, 0.001035] | 0.000933 | 6/10 |
| Shallow KAN28 | 0.001244 [0.000988, 0.002001] | 0.002374 | 7/10 |
| Composed KAN28 | 0.001140 [0.000645, 0.001619] | 0.001311 | -- |

The IQR describes variability across noise/initialization seeds, **not**
an uncertainty interval for an individual fit. The MLP has the lower
marginal median but KAN wins a small majority of paired seeds; these
summaries need not agree. The selected KAN/MLP median paired error ratio is
0.943, and ten pairs do not establish superiority. Against spline8, wins
split evenly. The experiment supports optimizer-sensitive competitiveness,
not a universal KAN advantage.

Validation chooses full/tight stopping in all ten composed-KAN cases (six
full, four tight). It chooses four tight, four full and two slow/tight
MLPs. Settings are selected by noisy held-out validation data, not by the
test errors in the tables.

### Cost of the refined search

| Model | Median selected-fit seconds | Median four-condition search seconds | Median search including original penalty tuning |
|---|---:|---:|---:|
| Spline8 | 0.162 | 1.36 | 1.90 |
| Spline28 | 2.72 | 7.39 | 10.32 |
| MLP28 | 7.04 | 22.28 | 30.51 |
| Shallow KAN28 | 3.92 | 10.14 | 14.07 |
| Composed KAN28 | 5.83 | 15.08 | 19.59 |

These are warmed, serial CPU Float64 fits on the same Darwin arm64 machine,
Julia 1.12.7, with one BLAS thread. Setup, replay, scoring and file writing
are outside fit timing. The final column adds each seed's original recorded
penalty-search cost before taking medians; it is not the sum of separately
reported medians or a new contemporaneous timing of the old search.

The previous native LAML spline baselines remain a different optimization
track: crowding median field errors 0.001423/0.001467 at roughly
0.15/0.26 seconds per fit. The refined neural fits can have lower median
error here, but require much greater tuning cost. No native track was
silently folded into an approximation-only ranking.

## Declared mesh-transfer protocol

Replay all 140 original selections (100 shared-Adam, 40 native LAML) at
24, 48 and 96 cells, with **no refitting**. Rebuild the approximator from the
saved model seed and evaluate the unchanged coefficients. The initial
profiles are exact cell averages at each resolution, with the same no-flux
diffusion coefficient and physical reaction function.

Each replay first reproduces its archived training, validation, held-out field,
reaction and coefficient errors at 24 cells. Fitting-source and new
evaluation-source hashes remain separate; the new version is not presented
as the historical fitting code.

| Metric | Meaning |
|---|---|
| `field_rmse` | Fitted versus true field, both on the evaluation mesh |
| `projected_model_rmse` | Same comparison after conservative averaging onto 24 cells |
| `projected_to_finest_truth_rmse` | Projected fitted field versus projected 96-cell truth |
| `truth_mesh_rmse` | Projected true field at this mesh versus projected 96-cell truth |
| `learned_mesh_shift_rmse` | Projected learned field versus the original learned 24-cell field |
| `learned_to_finest_mesh_rmse` | Projected learned field versus its learned 96-cell counterpart |

The finest reference is still a finite discretization, not a continuum
solution. Model and mesh errors need not add in quadrature: they can cancel.
All space-time squared errors pool across the two held-out profiles within
each seed before summarizing seeds.

## Mesh-transfer results

All 840 transferred profile/mesh evaluations are finite. The largest
same-mesh discrepancy from the original archived training, validation,
field and response scores is `1.22e-17`. The inputs and coefficients remain
unchanged. This experiment uses the **original** selections, not the new
400-iteration weights from the optimizer experiment.

| Track | Model | Logistic field RMSE, 24 / 96 cells | Crowding field RMSE, 24 / 96 cells |
|---|---|---:|---:|
| Adam | Spline8 | 0.000732 / 0.000731 | 0.001481 / 0.001480 |
| Adam | Spline28 | 0.001479 / 0.001478 | 0.002450 / 0.002439 |
| Adam | MLP28 | 0.000821 / 0.000820 | 0.003302 / 0.003290 |
| Adam | Shallow KAN28 | 0.000827 / 0.000825 | 0.001771 / 0.001760 |
| Adam | Composed KAN28 | 0.000526 / 0.000526 | 0.002934 / 0.002928 |
| LAML | Spline8 | 0.000537 / 0.000538 | 0.001423 / 0.001421 |
| LAML | Spline28 | 0.000537 / 0.000537 | 0.001467 / 0.001466 |

Median model rankings are unchanged. Across individual model/seed
comparisons, the 96/24-cell same-mesh field-error ratio ranges from 0.9813
to 1.0037. This is successful portability of the learned reaction functions
in this fixture, not evidence that the predicted fields themselves remain
unchanged or that a continuum error has been measured.

The distinction matters. For the original composed KAN on logistic:

- Median same-mesh field error is 0.000526 on both grids.
- The learned 24-to-96-cell field shift, projected onto 24 cells, has median
  RMS 0.001013 across the two profiles.
- The original 24-cell prediction versus projected 96-cell truth has median
  error 0.001110. Transferring the same coefficients to 96 cells reduces that
  common-grid comparison to 0.000525.

Thus the discretization contribution can exceed the approximator error
even when the model's relative accuracy is stable under refinement. The
largest true-field 24-to-96-cell discrepancy on an individual profile is
0.001426 for logistic and 0.001096 for crowding, both on the fine-scale
profile. This is why the original same-mesh KAN error was not a
continuum-PDE accuracy claim.

## Reproduction

From the repository root, in the benchmark environment:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/refine_reaction_diffusion.jl
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/transfer_reaction_diffusion.jl
python benchmarks/kan/analyze_rd_followon.py
```

Use new `--output` directories to repeat runs; existing directories are
rejected. The optimizer runner also accepts `--input`, `--cases`, `--models`,
`--seeds` and `--iterations`. The mesh runner accepts `--inputs` (comma-separated
original archives) and `--cells` (increasing multiples of the fitting mesh).
Changing these makes a different declared experiment. The original spatial
runner also exposes `--plateau-tol`, `--plateau-window` and `--early-stopping`
for its Adam track, with unchanged defaults.

Raw outputs retain every candidate, validation choice, fitted coefficients,
stopping controls, same-mesh replay differences, per-profile scores, seed
aggregates, source snapshots and resolved environment metadata. The read-only
analyzer audits the saved choices and recomputes summaries without refitting.

Primary new archives: [optimizer candidates](results/reaction-diffusion-optimizer/),
[mesh transfer](results/reaction-diffusion-transfer/) and
[audited summaries](results/reaction-diffusion-followon-analysis/). The latter
includes all condition/seed distributions, paired KAN comparisons, search
costs and the six mesh estimands, including the intermediate 48-cell results.

The fitting/evaluation execution source is
`3985b006b8798feb2d172c397f216094429af7ae51752178a3c0b15298e778ed`,
separate from the original fitting source `475352ef...`.
The [execution snapshot](results/source-3985b006.tar.gz) includes source,
Julia harnesses, license and a resolved benchmark manifest with a relative
local package path. Archive SHA-256:
`339721c822baa24a79b91193f5a1878862748baf4fefd4170f0e544f782dfa2f`.
The final read-only analysis and its helper are separately snapshotted in
the summary directory; their digests and all input fingerprints are recorded
in `metadata.json`.

The accompanying [vignette](../../vignettes/41_kan/41_kan.md) demonstrates
the new pointwise covariance/bootstrap APIs, not empirical coverage of the
intervals in these assessment fixtures. Architecture/grids and KAN
initialization remain fixed, and no simultaneous-band or identification
guarantee is inferred from either the tutorial or these comparisons.
