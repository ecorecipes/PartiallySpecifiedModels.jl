# Near-trajectory KAN assessment protocol

This is a separate continuation of the [forecast assessment](MULTIVARIATE-ASSESSMENT.md).
Its purpose is local response recovery near the sampled training
trajectories, plus a less initialization-sensitive native single-index
baseline. It does not replace or revise the completed forecast study.
The completed results are in [LOCAL-ASSESSMENT.md](LOCAL-ASSESSMENT.md).

## Declared split and support

The predator-prey equations, two truths, three training initial conditions,
training observations, state scales and observation noise are unchanged
from [MULTIVARIATE.md](MULTIVARIATE.md). `--split=forecast` remains the
default and preserves the original data generation.

`--split=local` changes the separate validation initial condition to
`(1.45, 0.65)` and the two test initial conditions to `(0.45, 0.33)` and
`(2.5, 1.7)`. Test times are `0:0.1:6`, not the longer forecast horizon 8.
The initial conditions are distinct from each other and from training.
They are perturbations of training initial conditions, not exact restarts
along an observed orbit.

Before fitting any candidate, the harness writes `geometry.csv`, reporting
distance from clean validation/test states to the 63 clean training
observation states. The fixed metric divides coordinates by `[3, 2.5]`
and uses radius 0.05. Geometry-only measurements for the declared split:

| Case | Validation coverage | First test coverage | Second test coverage |
|---|---:|---:|---:|
| Single-index truth | 21/21 | 61/61 | 59/61 (96.72%) |
| Interaction truth | 21/21 | 61/61 | 61/61 |

The design target is at least 95% coverage for each trajectory at that
radius. The clean cloud is used only for geometry and evaluation, not to
fit parameters, choose regularization, or select a start. This is a local
trajectory-neighborhood experiment, **not a filled two-dimensional domain
or a guarantee of global response identification**.

The local split defaults to new seeds 201--210. Within a seed, training
noise is identical between splits; validation uses the same independent
noise stream at its new initial condition. Test references are noise-free.

## Shared Adam and native multistart

The five shared-Adam alternatives remain tensor64, single_index9, mlp65,
kan_additive64 and kan63. All retain the constant attack start 0.3, the
250-iteration ceiling and learning rate 0.03. The declared local study
adds one larger candidate to the regularization grid:
`eta = [0, 0.1, 10, 1000, 100000]`, because earlier selections frequently
hit 1000. This grid is fixed before the local candidate fits; no test error
is used to extend it further.

`--starts=index-multistart` is an explicit **native-only** option for
LAML/GCV with `--models=tensor64,single_index9`. It is rejected for the
shared-Adam track or unsupported models. Tensor64 retains its constant
start. Single-index9 tries eleven starts:

1. The original constant response 0.3 with loading `[1,1]`.
2. All ten combinations of anchored loading `[1,a]`, where
   `a` is `-2,-1,0,1,2`, and outer response `0.3 + b*z`, where
   `b` is `-0.04` or `0.04`.

Both slope signs and both loading signs are represented. No true loading,
response values, or test errors enter the grid. Every start shares the
same training-only index statistics, outer knots, penalties and parameter
count. Unlike the constant response, the affine outer starts give a
nonzero prediction sensitivity to the free loading.

Each native candidate has a 25-iteration ceiling and selects smoothing
internally. The winner minimizes noisy validation trajectory NRMSE.
Failed starts remain in `trials.csv`; total tuning time includes every
attempt. The selected initialization label, loading, slope, coefficients
and actual selected approximator are carried through scoring and saving.
Ties retain the first valid candidate.

`candidate_coefficients.toml` records every attempt's status and any
returned finite coefficient vector, linked by `candidate_id`. This permits
rescoring the constant native candidate after selection without refitting
it or changing the selected result. The loading/slope CSV columns apply
only to the single-index model and are `missing` for the other models.

This is not a matched-compute native comparison: the index receives eleven
starts while the tensor receives one. Better validation error is guaranteed
relative to its retained constant candidate when that candidate succeeds;
better test error, good smoothing stationarity and global optimization are
**not** guaranteed. Existing `converged`, `stationarity` and
`smoothing_advanced` semantics are unchanged.

## Declared runs

Use the [benchmark environment](README.md). Output directories must be new;
do not run timed tracks concurrently.

```bash
# 500 shared-Adam candidates: two cases, ten seeds, five models, five etas
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --split=local --etas=0,0.1,10,1000,100000 \
  --output=benchmarks/kan/results/local-adam

# 240 native candidates per track: twenty tensor and 220 index fits
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --split=local --track=laml --models=tensor64,single_index9 \
  --starts=index-multistart --iterations=25 \
  --output=benchmarks/kan/results/local-laml

julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --split=local --track=gcv --models=tensor64,single_index9 \
  --starts=index-multistart --iterations=25 \
  --output=benchmarks/kan/results/local-gcv
```

After the runs, derive paired-seed summaries and replay the constant native
candidate from its saved weights, without another optimization:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/replay_multivariate.jl \
  --input=benchmarks/kan/results/local-laml \
  --output=benchmarks/kan/results/local-laml-constant

julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/replay_multivariate.jl \
  --input=benchmarks/kan/results/local-gcv \
  --output=benchmarks/kan/results/local-gcv-constant

python3 benchmarks/kan/analyze_multivariate.py --prefix=local --constant-baselines
```

The analyzer requires Python 3.11+ with only standard-library modules.
The replay requires the recorded package/harness source, records its
environment, and preserves unavailable constant candidates as failures.
After source changes, use the [preserved fitting source](README.md#analysis-after-source-changes)
for model replay. Saved-score analysis does not execute models and remains
available from the current worktree; `--require-current-source` requests
the stricter source check.

Scoring still aggregates the two equally sized test trajectories within
seed before reporting seed medians/IQRs. Near-support errors use supported
point counts as weights. Report all failures, negative-response fractions,
variability, selected starts, full tuning costs and native diagnostics,
not only median fit accuracy.

The altered horizon, validation initial condition, seeds and wider
regularization grid mean differences from the forecast study are not a
causal estimate of support alone. Keep native and shared-Adam rankings
separate, and compare selected native starts with their retained constant
candidates on the same new data rather than with old seed-101 examples.
