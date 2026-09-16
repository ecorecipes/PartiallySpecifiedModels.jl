# KAN approximator assessment

This benchmark fits unknown per-capita growth responses through the ODE
`dN/dt = r(N) * N`. It compares conventional cubic splines, an SPDE
approximator, a fixed-kernel GP, an MLP, and shallow/composed KANs.
It is a controlled pilot, not a claim of universal KAN superiority.

## Run

From the repository root:

```bash
julia --project=benchmarks/kan -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/compare.jl
```

The default is three seeds, two response functions, seven models and four
regularization candidates per model/seed/case. All candidates get at most
250 Adam iterations at learning rate 0.03; the eta grid is
`0, 0.1, 10, 1000`. A small wiring run is:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/compare.jl \
  --seeds=1 --iterations=20 --etas=0 --cases=logistic \
  --models=spline8,mlp28,kan28 --output=benchmarks/kan/results/smoke
```

Options are `--seeds`, `--iterations`, `--lr`, `--etas`, `--cases`,
`--models`, and `--output`, all supplied as `--name=value`.
Output directories containing earlier results are rejected; use a fresh
`--output` directory for a new run.

The retained initial comparison and its limitations are discussed in
[ASSESSMENT.md](ASSESSMENT.md), with separate 250- and 1000-iteration results.
The local-support evaluator update and frozen-weight before/after replay
are documented in [PERFORMANCE.md](PERFORMANCE.md).
The two-input predator-prey continuation compares tensor splines,
single-index models, a genuine two-input MLP, and additive/composed KANs
over ten fresh seeds. See [the multitrajectory protocol](MULTIVARIATE.md)
and [measured assessment](MULTIVARIATE-ASSESSMENT.md), including separate
native LAML/GCV tracks and joint-state-support limitations.
The [local follow-on protocol](LOCAL.md) adds a separately declared
near-trajectory split and validation-selected native index starts.
Candidate coefficients can be replayed without refitting, and
`analyze_multivariate.py` derives consistent paired-seed summaries.
Its [completed assessment](LOCAL-ASSESSMENT.md) separates the KAN/MLP
comparison from the substantial effect of native index initialization.
The [edge and activation diagnostics](DIAGNOSTICS.md) inspect all retained
KAN fits, with frozen predictions, marginal grid exposure and effective
edge figures.
The [LAML smoothing-priority correction](LAML-STALL.md) resolves the
retained tensor smoothing stalls and reassesses the affected native tracks
without rewriting earlier results.
The [reaction-diffusion example](REACTION-DIFFUSION.md) adds an independent
spatial Fisher-KPP assessment with conservative method-of-lines dynamics,
separate mesh-error controls and [measured results](REACTION-DIFFUSION-ASSESSMENT.md).
The [spatial refinements](REACTION-DIFFUSION-REFINEMENTS.md) separately
compare Adam stopping/learning-rate controls with frozen validation-selected
penalties and transfer the original fitted reactions across meshes without
refitting. The [KAN vignette](../../vignettes/41_kan/41_kan.md) demonstrates
fitting, diagnostics and conditional/bootstrap pointwise uncertainty.
The [uncertainty-calibration study](UNCERTAINTY-CALIBRATION.md) compares
pointwise KAN and spline coverage across independent noisy datasets,
separating in-range queries from extrapolation, finite-budget bootstrap
effects, failure rates and dataset-clustered Monte Carlo uncertainty.
The [broader interval/prior comparison](UNCERTAINTY-METHODS.md) evaluates
five interval constructions and a two-by-two initialization/prior ablation.
It distinguishes coverage-method effects from the substantial effect of
jointly penalizing the KAN's affine component under LAML.
The [fresh-data continuation](FRESH-COVERAGE.md) recomputes bootstrap
refits under the default free-affine KAN prior and separates its larger
confirmation cohort from a matched, low-budget seven-model screen.
The [bias/undersmoothing investigation](BIAS-UNDERSMOOTHING.md) decomposes
retained-fit errors and evaluates a predeclared two-stage smoothing
procedure on independent datasets, rather than tuning interval widths
to known truth.
The [robustness continuation](COVERAGE-ROBUSTNESS.md) keeps that multiplier
fixed while changing observation density, noise and response shape, with
new datasets and matched controls.
The [pointwise/simultaneous comparison](SIMULTANEOUS-BANDS.md) separates
coverage targets and conditional-covariance versus full-refit-bootstrap
uncertainty. It reuses retained fits, preserving original smoothing
selection rather than presenting a fixed-penalty bootstrap as unconditional
inference.
The [analytic smoothing-covariance correction](SMOOTHING-COVARIANCE.md)
adds both mean and covariance-root terms for Gaussian LAML working models,
with explicit regularity policies and selection-covariance propagation
to fixed-smoothing refits. A retained-data assessment reports availability,
pointwise versus whole-grid coverage, and width inflation without claiming
that the correction removes bias or guarantees nominal coverage.
The [stable smoothing-profile investigation](SMOOTHING-PROFILES.md)
implements penalty/null-space refits with an explicit affine limit,
multiple coefficient paths, representation-error oracles and known-noise
controls. It identifies consequential native KAN criterion stabilization,
rescues several oversmoothed fits, and separates those numerical effects
from remaining bias and basis-resolution limits. Solver defaults are unchanged.
The broader [cross-approximator calibration suite](../calibration/README.md)
adds spline, GP-interpolant, SPDE, MLP and KAN variants, studentized
bootstrap bands, sample-split bias bounds, confidence-set references,
functionals and weak-identification experiments. Development selection is
frozen before independent confirmation, with explicit workload previews
and dataset sharding.

## Protocol

| Model | Parameters | Penalty |
|---|---:|---|
| `spline8` | 8 | Natural cubic curvature; practical low-capacity baseline |
| `spline28` | 28 | Natural cubic curvature; parameter-matched baseline |
| `spde28` | 28 | Native SPDE penalty, fixed range 0.4 |
| `gp28` | 28 | Native inducing-value curvature, fixed Matern-5/2 kernel/range 0.4 |
| `mlp28` | 28 | One tanh hidden layer of width 9; weight ridge |
| `kan_shallow28` | 28 | One cubic KAN edge, 24 grid intervals; edge curvature |
| `kan28` | 28 | Widths 1-2-1, three grid intervals per layer; edge curvature |

All approximators use the physical domain `[0, 2]`. KAN penalties include
an explicitly declared `1e-6` affine-null-space penalty. The scalar Adam
weight is `eta / tr(S)`, with `eta` selected independently for each
method/seed using validation error. This normalization is not a claim that
different parameterizations have identical priors.

The initialization targets the same constant rate 0.4 for every method.
A lightweight benchmark-only wrapper supplies explicit initial weights so
solver-specific neural initialization cannot replace them. MLP and KAN
hidden layers retain seeded random initial features; their output layer is
initialized to the constant. The realized initialization error is recorded,
including any GP interpolation error.

Training observations are at times `0:0.2:6` from initial state 0.2, with
Gaussian noise of standard deviation 0.02. Validation observations use an
independent noise draw at interleaved times `0.1:0.2:5.9`. Selection never
uses the test trajectory or true response values.

The test trajectory starts at state 0.5 and runs to time 8. It is scored
against a tight-tolerance noise-free reference. Functional-response RMSE is
evaluated along that test trajectory, restricted to states overlapping the
training trajectory's state range. `support_fraction` records this scope;
there is no rectangular/off-support surface claim. Response errors are
computed independently of forecast success, and the in-support trajectory
is scored separately if a later forecast fails.

Each model is warmed up before timing its candidates. `fit_seconds`,
allocation volume, GC time, and RHS evaluation counts are recorded.
`tuning_seconds` sums the candidate fitting times rather than presenting only
the selected fit's cost. `setup_seconds` is separate and may include
first-use compilation. BLAS uses one thread. This is not a GPU benchmark,
and allocation volume is not peak resident memory.

All attempted candidates are retained in `trials.csv`. Numerical failures
are recorded explicitly; programming/configuration errors interrupt the
run. `selected.csv` retains unsuccessful selections instead of silently
dropping them from success counts. Full-horizon medians in `summary.md`
are conditional on success and are accompanied by success counts; the
in-support and response columns use their own finite results.

The metadata records seeds, options, package versions and a hash of source
and extension/benchmark files. Selected weights are retained in
`coefficients.toml` for inspection or independent rescoring. Increase seed counts and budgets before drawing
broader conclusions. The same optimizer/schedule isolates one training
protocol; it does not establish each family's best attainable performance.

## Analysis after source changes

`analyze_multivariate.py` reads saved scores without executing current model
code. It checks that the supplied tracks share their recorded fitting
source, records the current worktree hash separately, and preserves the
old numerical summaries. `--require-current-source` makes the worktree
match mandatory.

Actual historical model replay must still use the recorded fitting source.
The forecast/local fitting version is preserved in
`results/source-e309b452.tar.gz`, including its resolved benchmark manifest
and relative `PartiallySpecifiedModels` path. Restore it into an empty
directory rather than overlaying the current worktree:

```bash
snapshot=$(mktemp -d)
tar -xzf benchmarks/kan/results/source-e309b452.tar.gz -C "$snapshot"
julia --project="$snapshot/benchmarks/kan" --check-bounds=yes -e \
  'include(joinpath(dirname(Base.active_project()), "multivariate.jl")); println(source_hash())'
```

The expected hash is
`e309b4525899d04d30ab692f9479e7cd972a3f7cdb2e34d0ca63b0de49411032`.
Use the recorded Julia version (1.12.7) for closest reproduction. If
dependencies are not cached, instantiate that restored environment.
Run its `replay_multivariate.jl` with an absolute `--input` path to the
retained archive and a new `--output` directory. New diagnostics instead
record their changed source explicitly and compare against pre-change
frozen predictions, as described in [DIAGNOSTICS.md](DIAGNOSTICS.md).
