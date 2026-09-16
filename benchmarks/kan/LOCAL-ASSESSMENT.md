# KAN near-trajectory assessment

**Solver update:** the subsequently identified tensor-LAML acceptance defect
is corrected in [LAML-STALL.md](LAML-STALL.md). That report reassesses the same
data and budgets; the original measurements below remain a historical record.

**The composed KAN is competitive near training trajectories, but improving
the native baseline initialization changes the comparison materially.**
Under shared Adam, KAN has the lowest median trajectory and response
errors in both fixtures. Its trajectory comparison with the much cheaper
nine-parameter single-index model splits 5/10 seeds each, however.
Validation-selected native single-index starts achieve lower median
trajectory errors than KAN in their separate solver tracks.

This supports offering KAN as an alternative, not replacing conventional
approximators by default. Optimizer, initialization, validation geometry
and total search cost matter alongside architecture.

## What changed

The [declared protocol](LOCAL.md) preserves the equations, training
trajectories, noise model, scales and five approximators from the
[forecast study](MULTIVARIATE-ASSESSMENT.md), but uses a distinct local
validation/test split. Test time ends at 6 rather than 8.

At the unchanged radius 0.05 in domain-scaled joint coordinates,
validation coverage is 100%; the four case/test-path combinations have
coverage 100%, 96.72%, 100% and 100%. This was measured from reference
geometry before fitting. It fixes the earlier experiment's lack of
near-training-cloud evaluation, **not** the broader identifiability
problem: a neighborhood of three sampled trajectories is not a filled
two-dimensional domain.

Seeds 201--210 are new. The shared-Adam study keeps its constant response
initialization, 250-iteration ceiling and learning rate 0.03, with
`eta = [0, 0.1, 10, 1000, 100000]`. Native LAML/GCV retain their
25-iteration ceilings and internal smoothing selection. Tensor fits keep
one constant start; index fits try the constant start and ten affine
outer-response/loading combinations, selected by noisy validation NRMSE.
No test errors select a model, penalty or initialization.

All **980 candidates** completed with finite training/validation results:
500 Adam, 240 LAML and 240 GCV. All 180 selected fits produced two finite
test trajectories and finite response scores. No selected model produced
negative attack values at the evaluated test points. These are numerical
completion counts, not guarantees of fit quality or positivity elsewhere.

## Shared Adam: accuracy and variability

Each seed first pools squared errors across its two equally sized held-out
trajectories. Tables then report the median and `[25th, 75th percentile]`
over ten seed values. Trajectory NRMSE uses state spans 3 and 2.5; response
RMSE is in attack-coefficient units. The two trajectories within a seed
are not treated as independent replicates.

| Case | Model | Parameters | Trajectory NRMSE, median [IQR] | Response RMSE, median [IQR] |
|---|---|---:|---:|---:|
| Single index | Tensor spline | 64 | 0.01162 [0.01130, 0.01380] | 0.04639 [0.02707, 0.04771] |
| Single index | Single-index spline | 9 | 0.00280 [0.00230, 0.00335] | 0.01066 [0.00946, 0.01094] |
| Single index | MLP, 2-16-1 | 65 | 0.00496 [0.00424, 0.00775] | 0.01417 [0.01052, 0.02024] |
| Single index | Additive KAN | 64 | 0.01745 [0.01649, 0.02439] | 0.03213 [0.03070, 0.07066] |
| Single index | Composed KAN, 2-3-1 | 63 | **0.00272 [0.00221, 0.00308]** | **0.00826 [0.00654, 0.01010]** |
| Interaction | Tensor spline | 64 | 0.00590 [0.00548, 0.00738] | 0.03290 [0.02839, 0.03449] |
| Interaction | Single-index spline | 9 | 0.00339 [0.00231, 0.00412] | 0.01199 [0.00847, 0.01934] |
| Interaction | MLP, 2-16-1 | 65 | 0.00512 [0.00427, 0.00624] | 0.02245 [0.02066, 0.02861] |
| Interaction | Additive KAN | 64 | 0.00637 [0.00534, 0.00695] | 0.01994 [0.01566, 0.02147] |
| Interaction | Composed KAN, 2-3-1 | 63 | **0.00280 [0.00207, 0.00342]** | **0.01080 [0.00860, 0.01392]** |

Near-support response medians in the single-index fixture are, in the same
model order, 0.04677, 0.01074, 0.01426, 0.03237 and 0.00833. They are
identical to the all-path values in the interaction fixture, which has
complete coverage. The conclusions do not depend on the two excluded
single-index test samples.

Paired-seed wins for the **composed KAN**, with ten comparisons per column:

| Baseline | Single-index trajectory wins | Single-index response wins | Interaction trajectory wins | Interaction response wins |
|---|---:|---:|---:|---:|
| Tensor spline | 10/10 | 10/10 | 9/10 | 10/10 |
| Single-index spline | 5/10 | 7/10 | 5/10 | 6/10 |
| MLP | 9/10 | 9/10 | 9/10 | 10/10 |
| Additive KAN | 10/10 | 10/10 | 10/10 | 9/10 |

The median paired KAN/index trajectory-error ratios are 0.989 and 0.975:
the small difference between these two methods is less compelling than
their ratios of separately computed medians might suggest. Against the
MLP, the paired trajectory-error ratios are 0.523 and 0.547.
These are descriptive comparisons, not significance tests.

## Cost under shared Adam

Cells give **single-index / interaction** case medians. Tuning includes all
five penalty candidates.

| Model | Selected-fit seconds | Total tuning seconds | Selected-fit cumulative allocated MiB |
|---|---:|---:|---:|
| Tensor spline | 2.855 / 3.274 | 14.32 / 18.03 | 8679 / 8548 |
| Single-index spline | 0.118 / 0.131 | 0.649 / 0.647 | 243 / 269 |
| MLP | 1.126 / 1.331 | 5.20 / 5.96 | 5111 / 6184 |
| Additive KAN | 0.886 / 0.843 | 6.31 / 4.37 | 184.6 / 184.6 |
| Composed KAN | 1.047 / 0.976 | 4.89 / 5.30 | 467 / 423 |

KAN retains its allocation advantage over this MLP implementation, with
comparable or lower fit/tuning time. The single-index spline is much
cheaper than either. These are warmed CPU Float64 measurements on
Darwin arm64, Julia 1.12.7, one BLAS thread, with tracks run serially.
Setup and archive-writing time are outside fit timing. Allocation volume
is cumulative, not peak resident memory.

## Native multistart: a stronger baseline

These are separate solver/initialization tracks, not additional rows in a
matched-optimizer or matched-compute ranking. Each index selection searches
eleven starts; the tensor has one.

| Track | Case | Model | Trajectory NRMSE, median [IQR] | Median response RMSE | Median fit seconds | Median total tuning seconds |
|---|---|---|---:|---:|---:|---:|
| LAML | Single index | Tensor spline | 0.01781 [0.00978, 0.02373] | 0.02082 | 0.800 | 0.800 |
| LAML | Single index | Single-index spline | 0.00183 [0.00166, 0.00244] | 0.00785 | 0.206 | **6.93** |
| LAML | Interaction | Tensor spline | 0.00584 [0.00341, 0.00825] | 0.02124 | 0.747 | 0.747 |
| LAML | Interaction | Single-index spline | 0.00238 [0.00169, 0.00258] | 0.00917 | 0.217 | **6.47** |
| GCV | Single index | Tensor spline | 0.00358 [0.00291, 0.00503] | 0.00944 | 0.265 | 0.265 |
| GCV | Single index | Single-index spline | 0.00227 [0.00185, 0.00272] | 0.01035 | 0.074 | **3.35** |
| GCV | Interaction | Tensor spline | 0.00300 [0.00218, 0.00364] | 0.00891 | 0.261 | 0.261 |
| GCV | Interaction | Single-index spline | 0.00249 [0.00163, 0.00349] | 0.00983 | 0.114 | **4.45** |

The selected native index fits are individually cheap, but quoting only
their selected-fit time would hide most of their cost. LAML's complete
index search costs more than the shared-Adam KAN search here. Conversely,
the single-start tensor GCV fits are inexpensive and competitive.

Every selected index initialization was nonconstant. Ten of the forty
selected native index starts had a **positive** initial outer slope,
despite both truths being decreasing in each input. This is a starting
point, not a shape constraint or the fitted response's slope; keeping both
signs was useful. The grid's loading values are starting directions, not
bounds on the fitted loading.

### Comparison with the constant-initialized native fit

The constant candidate is still a fitted model, not the unfitted constant
function 0.3. Its trained coefficients were saved during the same search
and replayed afterward without optimization. All eighty constant candidates
(including tensor fits) produced two finite replayed trajectories.

| Solver | Case | Constant-start trajectory NRMSE | Selected-start trajectory NRMSE | Trajectory wins | Constant-start response RMSE | Selected-start response RMSE |
|---|---|---:|---:|---:|---:|---:|
| LAML | Single index | 0.01905 | 0.00183 | 8/10 | 0.05770 | 0.00785 |
| LAML | Interaction | 0.02657 | 0.00238 | 10/10 | 0.06869 | 0.00917 |
| GCV | Single index | 0.18650 | 0.00227 | 8/10 | 0.61496 | 0.01035 |
| GCV | Interaction | 0.08708 | 0.00249 | 10/10 | 0.31864 | 0.00983 |

Entries are ten-seed medians for the index model. Response wins have the
same counts. The tensor's constant replay matches its selected result;
it did not receive additional starts.

This confirms the earlier initialization concern on new data, rather than
relying on the old seed-101 diagnostic. It also shows the limit of
validation selection: it improves the index test errors in 8/10, not all
ten, single-index seeds. GCV's constant-initialized index produced negative
attack predictions on 13/40 test trajectories; none of its selected
multistart fits did so at the sampled points.

## Remaining numerical and scientific limitations

**Tensor LAML smoothing stalls remain unresolved.** Nineteen of twenty
tensor fits never advanced smoothing from initialization, despite all
reporting iteration stability. Median stationarity residuals are 0.107
and 0.110. The reported EDF/covariance for those fits describes initial,
not selected, smoothing. No solver default or stopping rule was changed
to conceal this.

All twenty selected LAML index fits advanced smoothing, with median
stationarity residuals `5.09e-4` and `6.87e-4`. Even these diagnostics do
not establish a global optimum. One selected LAML index fit and two
selected GCV index fits exhausted their iteration budgets. The GCV index
interaction response error has a worst seed of 0.04921, against a median
of 0.00983: initialization search reduced, but did not eliminate, variability.

The expanded `eta=100000` candidate was selected only once, by an additive
KAN on the interaction case. The former tensor/additive boundary at 1000
is therefore no longer the upper edge of the declared search. Budget
sensitivity remains: 7/10 and 8/10 selected tensor Adam fits, and 7/10 and
9/10 selected MLP fits, reached 250 iterations. This is still not an
estimate of every family's best attainable performance.

The changed horizon, validation/test initial conditions, seeds and penalty
grid prevent attributing differences from the forecast study solely to
support. The older tensor/interaction advantage and this local KAN advantage
are different experimental results, not contradictory estimates of one
universal ranking. Good local index performance on the product truth also
does not make that truth globally single-index: fitting thin trajectory
neighborhoods is a much weaker requirement.

The next justified work is edge/activation-coverage diagnostics, investigation
of the native tensor smoothing stalls, broader optimizer budgets and an
independent dynamical system. Adaptive grids, additional bases, symbolic
extraction and broad superiority claims remain premature.

## Implementation and artifacts

`multivariate.jl` now has explicit split/start options, IC-aware validation
and scoring, pre-fit geometry records, full candidate weights, and selected
candidate provenance. The original forecast and constant-start defaults are
unchanged. `replay_multivariate.jl` preserves failed/unavailable baselines
as failures rather than dropping their seeds or silently refitting them.

The [protocol and commands](LOCAL.md) reproduce the experiment. Primary
archives: [Adam](results/local-adam/), [LAML](results/local-laml/) and
[GCV](results/local-gcv/). Each retains trials, selections, test scores,
candidate/selected coefficients, geometry, source snapshots and resolved
package versions.

The [seed distributions](results/local-analysis/model_summary.csv),
[paired comparisons](results/local-analysis/paired_comparisons.csv), and
[constant-baseline comparisons](results/local-analysis/constant_summary.csv)
are derived from those archives. The analysis audits the full declared
candidate grid, validation minimum, tuning-time accounting, weight/score
provenance and support, and independently reproduces the Julia seed metrics.
The [LAML](results/local-laml-constant/) and
[GCV](results/local-gcv-constant/) constant replays retain their own
input/source/environment hashes and explicitly record `optimizer_refit=false`.

All three primary runs use assessment-script hash
`ce00f9acfd69d6cb3c9534a54c3ab2c36b52561743897ac8dc4e6c836ababedf`
and the same package/extension/wrapper source hash as the forecast study,
`e309b4525899d04d30ab692f9479e7cd972a3f7cdb2e34d0ca63b0de49411032`.
The completed forecast result archives were not rewritten.
