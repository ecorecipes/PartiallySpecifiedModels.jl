# KAN edge and activation diagnostics

**The fitted KANs sometimes use spline padding, but none of the sampled
layer inputs loses all spline support.** The new diagnostics expose this
distinction without changing fitted functions, grids or solver defaults.
They also make the complete edge functions visible, rather than showing
only their spline components.

## Public interfaces

`kan_edge_curves(a, beta; npoints=101, extent=:grid)` returns one record per
edge with logical coordinates, optional first-layer physical coordinates,
base/spline/total values, and central/padded knot ranges. `extent=:support`
includes the padding.

`kan_activation_diagnostics(a, beta, samples)` accepts a matrix with one
sample per row and one physical input per column. A vector is also
accepted for a unary KAN. It returns layer input/output matrices,
predictions, central-interval occupancy, and zero-spline-basis counts.
Only the first layer applies physical-domain normalization.

Both return owned CPU Float64 snapshots. They are not differentiable
evaluators, uncertainty bands, adaptive-grid operations or pruning rules.
Use the relevant approximator's parameter block, for example
`sol.parameters.g`, rather than a whole mixed-model vector.
See the [API documentation](../../docs/src/approximators.md#edge-and-activation-diagnostics).

## What was inspected

The retained forecast and local shared-Adam studies each provide twenty
additive KAN64 and twenty composed KAN63 fits: two truths, ten seeds and
two architectures per study. The inspection covers all **80 fits**.

Reference evaluations were captured before modifying the evaluator's
input-normalization helper. They use clean training observation states,
clean validation states and both true held-out trajectories. The combined
**18,080 frozen evaluations** are model evaluations, not independent data
replicates. The maximum pre-change scalar/Lux discrepancy was `3.33e-16`.

The diagnostic forward traces reproduce all frozen scalar predictions
with **zero measured change**. Scalar evaluation allocation remains
0/0 bytes for the shallow model's primal/input derivative, and 80/96 bytes
for the composed unary model. Diagnostics allocate their own plotting and
trace arrays; these allocations are not inserted into the ODE RHS.

No optimization, coefficient update or new regularization selection was
performed. Fitting and diagnostic source hashes are recorded separately.

## Grid exposure

All first-layer samples stay inside their central fixed grids. The
composed models' hidden coordinates sometimes exceed the central
`[-1,1]` range, while remaining inside the padded support.

The table averages exposure over the three hidden input channels within
each fit, then takes the median over ten seeds. The maximum column is the
largest individual hidden-channel test fraction in the group.

| Study | Truth | Median training exposure outside central grid | Median test exposure outside central grid | Maximum single-channel test exposure outside central grid |
|---|---|---:|---:|---:|
| Forecast | Single index | 3.17% | 0.00% | 0.00% |
| Forecast | Interaction | 2.12% | 0.00% | 0.00% |
| Local | Single index | 3.70% | 4.51% | 15.57% |
| Local | Interaction | 1.85% | 2.05% | 17.21% |

There are 840 input-channel/partition records across layers, models and
training/validation/test partitions. Fifty-eight report some central-grid
exposure outside the interval; all are hidden channels. **None reports a
sample where every cubic basis is zero.** These records are related
measurements within fits, not 840 independent models.

Central-grid exposure and spline support answer different questions.
Cubic bases continue through the padded intervals, so crossing a central
limit does not automatically reduce an edge to its base branch. The
curvature penalty's integration interval remains the declared logical
domain; legal evaluation in padding is not a reason to ignore its behavior.

Within the composed layer, the median least-occupied hidden channel visits
one of three central intervals on the single-index truth and two of three
on the interaction truth, in both studies. For the much finer additive
grid, the least-covered physical input visits twelve of twenty-eight
central intervals during training.

Neither observation identifies unused coefficients: neighboring spline
bases overlap intervals, and penalties couple their coefficients. Hidden
channel labels also do not identify the same scientific component across
different seeded fits.

### Marginal grid coverage is not joint data support

The forecast single-index test paths had **zero** near-training-cloud
coverage at the original joint-distance radius, yet their first-layer
inputs all stay inside the KAN grids. Thus grid coverage cannot replace
the [joint-trajectory support assessment](MULTIVARIATE-ASSESSMENT.md#joint-state-support-is-limited).
It measures coordinate exposure, not identification of a bivariate law.

## Effective edge curves

Eight representative networks use the first declared seed of each study,
with both truths and both architectures. This is a fixed example-selection
rule, not a choice of the best test fit. Their 44 edges are sampled at
101 points across padded support, producing 4,444 curve rows.

The figure below shows the local single-index example, seed 201:

![Base, spline and total edge contributions for the local composed KAN](results/kan-diagnostics/plots/local-adam__single_index__kan63__201.svg)

Every horizontal axis is in logical layer coordinates, and vertical scales
are separate for each edge. Grey marks the central grid. Green is only the
min/max range of that input along training states; it is **not** a confidence
band or a joint-support region. Physical first-layer coordinates are retained
in the CSV for other plotting choices.

The base and spline can offset one another. Interpret the black **total**
curve when discussing an effective edge, not either component alone.
Even complete edges are non-unique network representations, not recovered
mechanistic sub-laws. No automatic pruning or symbolic extraction is implied.

The [plot directory](results/kan-diagnostics/plots/) contains all eight
examples and renderer/input fingerprints.

## Reproduce and inspect

From the repository root with the benchmark environment:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/diagnose.jl \
  --output=benchmarks/kan/results/kan-diagnostics-replay

python3 benchmarks/kan/plot_diagnostics.py \
  --input=benchmarks/kan/results/kan-diagnostics-replay
```

The inspector requires the frozen reference fingerprint and matching
model/data constructor. It records the current diagnostic source and
compares every prediction with the pre-change oracle; it does not pretend
that the current package is the original fitting version.
The renderer uses only Python standard-library modules.

Primary artifacts: [coverage](results/kan-diagnostics/coverage.csv),
[layer aggregates](results/kan-diagnostics/layers.csv),
[interval counts](results/kan-diagnostics/intervals.csv),
[edge curves](results/kan-diagnostics/edge_curves.csv),
[frozen-reference comparisons](results/kan-diagnostics/reference_comparison.csv)
and [metadata](results/kan-diagnostics/metadata.toml).

The original fitting source was preserved as
[`source-e309b452.tar.gz`](results/source-e309b452.tar.gz) before adding
diagnostics. Its source hash is
`e309b4525899d04d30ab692f9479e7cd972a3f7cdb2e34d0ca63b0de49411032`;
the archive SHA-256 is
`a8a262607516d03b0d0006038bc5e3370ab8d59229cb9a6f8126d30ce2e0955b`.
It includes the resolved benchmark manifest with a relative package path.
Restoration into an empty directory loads the historical package and
recovers that exact source hash.

As source evolves, `analyze_multivariate.py` can still summarize saved CSV
scores without evaluating a model. It requires consistent archive origins,
checks input fingerprints/provenance and records the current worktree hash
separately. Add `--require-current-source` when an exact worktree match is
required. Actual constant-candidate model replay retains its strict
matching-source guard; use the archived source for historical replay.

These observations establish a working inspection interface, not an
accuracy benefit from adaptive grids. Grid/range changes would still need
explicit coefficient remapping, penalty rebuilding and renewed solver
qualification. The tensor-LAML smoothing stalls and independent-system
assessment remain separate open work.

**Subsequent solver work:** [LAML-STALL.md](LAML-STALL.md) resolves the
identified tensor acceptance-priority stalls and records new native fits.
The edge/activation inspection above still refers to its original frozen
KAN weights.
