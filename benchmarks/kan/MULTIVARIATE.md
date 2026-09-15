# Multitrajectory KAN assessment

This continuation evaluates a genuinely two-input unknown response in a
predator-prey ODE. It is separate from the earlier one-input growth-rate
pilot and keeps shared-optimizer and native-solver tracks distinct.
The completed ten-seed study and its limitations are in
[MULTIVARIATE-ASSESSMENT.md](MULTIVARIATE-ASSESSMENT.md).
A separate [local trajectory-neighborhood protocol](LOCAL.md) adds
`--split=local` and native single-index multistart selection. The original
split and constant initialization remain the defaults.

## Model and data

```math
\dot N=N(1-N/3)-g(N,P)NP,\qquad
\dot P=0.6g(N,P)NP-0.25P.
```

The unknown `g` is an attack coefficient; the known `NP` factors preserve
the zero-population boundaries.

| Case | True attack coefficient | Representational distinction |
|---|---|---|
| `single_index` | `0.2 + 0.6/(1 + exp(2*(N + 0.8P - 1.8)))` | Exactly a function of one linear index |
| `interaction` | `0.9/((1 + 0.6N)*(1 + 0.8P))` | Not a function of one linear index; compositional/product structure |

Training uses three complete trajectories, starting at `(0.4,0.3)`,
`(1.5,0.6)` and `(2.6,1.8)`. They are stacked as six ODE states sharing
one `g`, not fitted separately. Observations are at `0:0.3:6`.
Validation is an independent trajectory from `(1.1,0.9)`, not interleaved
points from the training trajectories.

The two held-out initial conditions are `(0.8,1.5)` and `(2.7,0.25)`, with
evaluation at `0:0.1:8`. Training and validation observations have Gaussian
noise of standard deviation 1% of each state's declared domain span; test
references are noise-free. The fixed spans are
3 for prey and 2.5 for predators; trajectory loss/NRMSE uses those same
scales. Fresh default seeds are 101 through 110.

## Approximators

| Model | Parameters | Configuration |
|---|---:|---|
| `tensor64` | 64 | 8-by-8 natural cubic tensor spline |
| `single_index9` | 9 | Anchored two-input index, eight-coefficient outer spline |
| `mlp65` | 65 | Two inputs, 16 tanh hidden units, one output |
| `kan_additive64` | 64 | Single KAN layer, two inputs, 28 grid intervals |
| `kan63` | 63 | Composed 2-3-1 KAN, three grid intervals per layer |

The single-layer KAN is **additive**, regardless of parameter count. It
cannot generally express the interaction case. The low-parameter
single-index model is deliberately included as an appropriate structural
baseline for the first case, not as a parameter-matched model.

The MLP is a benchmark-only two-input approximator, independently compared
with Lux's Dense evaluation. It does not use the existing unary
`NeuralApproximator` input path. The single-index reference statistics use
only noisy training observations and remain fixed; neither validation nor
test trajectories enter their construction.

All models start from attack coefficient 0.3, with seeded random hidden
features for the MLP and composed KAN. Native penalties are used; KAN has
complete-edge curvature plus the declared `1e-6` affine-null-space penalty.

## Tracks and scoring

`--track=adam` is the approximation comparison: a shared learning rate,
iteration ceiling and validation-selected `eta/tr(S)` penalty strength.
The default grid is `0,0.1,10,1000` and the default ceiling is 250 iterations.

`--track=laml` and `--track=gcv` are separate native-solver comparisons.
They select smoothing internally and reject `--etas`; their default
iteration ceiling is 25. Do not merge these results into an
approximation-only ranking. The initialization wrapper preserves the
underlying per-layer/per-component penalty blocks for these tracks.

Each held-out trajectory is scored independently, so failure on one does
not erase the other. Response RMSE is measured along the actual test
trajectory, independently of simulated-trajectory success.

Near-support errors use nearest-neighbor distance to the joint training
state cloud after division by the fixed domain spans. The default radius
is 0.05. A bounding rectangle is **not** treated as trajectory support.
Both all-path and near-support response errors are retained, together
with coverage and negative-response fractions. The clean reference cloud
is used only to describe evaluation support, never to fit or select a model.

## Run

Use the environment setup in [README.md](README.md), then:

```bash
# Small wiring run
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --seeds=101 --iterations=10 --etas=0 \
  --output=benchmarks/kan/results/multivariate-smoke

# Fresh-seed shared-Adam study
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --output=benchmarks/kan/results/multivariate-adam

# Separate native-solver baseline
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --track=laml --models=tensor64,single_index9 --iterations=25 \
  --output=benchmarks/kan/results/multivariate-laml
```

Other options are `--cases`, `--models`, `--lr`, `--support-radius`,
`--split=forecast|local`, `--starts=constant|index-multistart` and
`--seeds` (comma-separated). Every output directory must be new.
`trials.csv`, `selected.csv`, `test_trajectories.csv`, `coefficients.toml`
and `metadata.toml` retain settings, failures, individual trajectories,
weights, source hashes and the resolved package versions. Median summaries
must be read with their selection and trajectory success counts.

GCV uses the same native baseline selection:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/multivariate.jl \
  --track=gcv --models=tensor64,single_index9 --iterations=25 \
  --output=benchmarks/kan/results/multivariate-gcv
```

The current harness also retains `seed_metrics.csv` and an exact
`assessment-script.jl` snapshot. `geometry.csv` is written before fitting
and describes validation/test support for the selected split.
It rejects a run whose source changes
before completion. Trial/selection diagnostics include stability, EDF,
criterion value and, where supplied, LAML `stationarity`,
`smoothing_advanced` and failure counts. Unavailable diagnostics are
`missing`; finite selection status is not a fit-quality flag.

Error summaries combine squared errors across the two equally sized test
trajectories within seed, then report medians across complete seeds.
Near-support errors weight each trajectory by its supported point count.
Zero coverage is `NA`, not zero error. The original Adam/LAML archive
summaries predate this aggregation; the assessment report uses the
consistent, recomputed seed-level summaries for all tracks.
