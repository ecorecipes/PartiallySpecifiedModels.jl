# Multitrajectory KANApproximator assessment

**Follow-on:** [LOCAL-ASSESSMENT.md](LOCAL-ASSESSMENT.md) reports a separate
near-training-trajectory study and native initialization search. The
forecast results below are preserved, not replaced by that different protocol.
The later [LAML smoothing-priority correction](LAML-STALL.md) separately
reassesses this forecast protocol's native baseline.

**Composition helps, but there is no universal winner.** On the
single-index truth, the composed 63-parameter KAN has the lowest median
held-out error under the shared Adam protocol. The tensor spline is more
accurate on the non-single-index interaction truth. KAN improves on the
MLP in all ten paired seeds on the first case, but only six on the second.
The nine-parameter single-index model remains a very cheap, competitive
baseline.

This is an exploratory two-problem, ten-seed CPU study, not an estimate of
each family's best attainable accuracy. Limited joint-state coverage and
native-solver initialization sensitivity materially qualify the results.
The ordinary spline default should not change on this evidence.

## Experiment and aggregation

The [protocol](MULTIVARIATE.md) fits the shared attack coefficient in

```math
\dot N=N(1-N/3)-g(N,P)NP,\qquad
\dot P=0.6g(N,P)NP-0.25P.
```

The cases are a nonlinear function of `N + 0.8P` and a product response
`0.9/((1+0.6N)*(1+0.8P))`. Three training trajectories share one fitted
response. A fourth, independently noisy trajectory selects Adam
regularization; two different initial conditions are held out for
noise-free evaluation through time 8, beyond the training horizon 6.
Training-only single-index statistics stay fixed during fitting and scoring.

Seeds 101--110 are new relative to the earlier univariate study. They
change observation noise and neural initialization, not the reference
dynamics or held-out initial conditions. All models start at attack
coefficient 0.3. The MLP is a genuine two-input, benchmark-only Lux-equivalent
evaluator; it does not use the public unary `NeuralApproximator` input path.

The shared Adam track uses a 250-iteration ceiling, learning rate 0.03,
and validation selection from `eta = [0, 0.1, 10, 1000]`, applied as
`eta/tr(S)`. LAML and GCV are separate tensor/single-index baseline tracks,
with 25-iteration ceilings, `jac=:forwarddiff`, and internal smoothing
selection; GCV uses `search=:reuse`.

There are **400 Adam candidates, 100 selected Adam fits, and 40 fits in
each native track**. All candidates returned finite training/validation
results, and all 360 selected-fit test simulations completed finitely.
Response evaluation also completed for all 360 trajectories. These counts
describe numerical completion, not fit quality or physical plausibility.

The experimental replicate is a **seed, not a trajectory**. For either
trajectory or response error, first combine the two equally sized
held-out trajectories within seed:

```math
E_s=\sqrt{(e_{s,1}^2+e_{s,2}^2)/2}.
```

Tables report the median and `[25th, 75th percentile]` over ten such seed
values, with linearly interpolated quartiles. No significance tests or
independence assumption for the two trajectories are used. In runs with
failures, a seed requires both trajectories for its all-path metric;
response completion is counted separately.

Trajectory NRMSE divides prey/predator errors by fixed spans 3 and 2.5.
Response RMSE is in unscaled attack-coefficient units, evaluated on the
true held-out paths rather than on fitted states or a rectangular grid.

## Shared Adam: held-out accuracy

| Case | Model | Parameters | Trajectory NRMSE, median [IQR] | Response RMSE, median [IQR] |
|---|---|---:|---:|---:|
| Single index | Tensor spline | 64 | 0.06116 [0.04430, 0.06842] | 0.06113 [0.05179, 0.06317] |
| Single index | Single-index spline | 9 | 0.01811 [0.01348, 0.02289] | 0.01264 [0.01045, 0.01361] |
| Single index | MLP, 2-16-1 | 65 | 0.04371 [0.03237, 0.05330] | 0.03810 [0.02125, 0.04762] |
| Single index | Additive KAN | 64 | 0.08451 [0.08138, 0.08927] | 0.08191 [0.07756, 0.08508] |
| Single index | Composed KAN, 2-3-1 | 63 | **0.01232 [0.00692, 0.01913]** | **0.00690 [0.00513, 0.01082]** |
| Interaction | Tensor spline | 64 | **0.00622 [0.00497, 0.00702]** | **0.00722 [0.00634, 0.00787]** |
| Interaction | Single-index spline | 9 | 0.02227 [0.02187, 0.02334] | 0.01699 [0.01659, 0.01789] |
| Interaction | MLP, 2-16-1 | 65 | 0.03165 [0.00894, 0.03550] | 0.03653 [0.01046, 0.04799] |
| Interaction | Additive KAN | 64 | 0.03256 [0.03093, 0.03347] | 0.03680 [0.03274, 0.04075] |
| Interaction | Composed KAN, 2-3-1 | 63 | 0.01857 [0.01194, 0.02001] | 0.01813 [0.01241, 0.02050] |

All entries use ten complete seeds. The composed KAN's lower median does
not mean it wins every seed against every competitor:

| Baseline | Single-index truth: KAN trajectory wins | Single-index truth: KAN response wins | Interaction truth: KAN trajectory wins | Interaction truth: KAN response wins |
|---|---:|---:|---:|---:|
| Tensor spline | 10/10 | 10/10 | 1/10 | 1/10 |
| Single-index spline | 5/10 | 7/10 | 9/10 | 4/10 |
| MLP | 10/10 | 10/10 | 6/10 | 6/10 |
| Additive KAN | 10/10 | 10/10 | 10/10 | 10/10 |

Here "KAN" means the composed model. Its comparison with the MLP on the
interaction case is mixed, not decisive; the MLP's trajectory IQR overlaps
the KAN's substantially. The cheap single-index model has slightly better
median response error than KAN even on the interaction case, although its
trajectory error is worse. These estimands need not rank models identically.

The single-layer KAN is additive in the two inputs. Both truths are
non-additive: even a nonlinear function of a linear index is generally not
a sum of separate input functions. Its poor result is evidence about this
restricted architecture, not all shallow networks or all KAN bases.

## Joint-state support is limited

Support is the nearest distance to the **63 clean training observation
states**, after dividing coordinates by `[3, 2.5]`, at the declared radius
0.05. It is a sampled joint-trajectory neighborhood, not a bounding rectangle
or a claim about exact continuous-curve coverage.

| Case | Test initial condition | Samples within radius |
|---|---|---:|
| Single index | (0.8, 1.5) | 0/81 |
| Single index | (2.7, 0.25) | 0/81 |
| Interaction | (0.8, 1.5) | 0/81 |
| Interaction | (2.7, 0.25) | 34/81 (41.98%) |

Thus the single-index results are an off-neighborhood generalization
comparison under this definition. They do **not** establish response
identification near observed joint states. Its near-support errors are
`NA`, not zero. The interaction case has only 34/162 supported samples
across both test trajectories; the following column pools only those
samples, with point-count weights:

| Model, shared Adam | Interaction near-support response RMSE, median [IQR] |
|---|---:|
| Tensor spline | **0.00314 [0.00211, 0.00393]** |
| Single-index spline | 0.01414 [0.01371, 0.01472] |
| MLP | 0.01151 [0.00918, 0.01479] |
| Additive KAN | 0.03925 [0.03608, 0.04524] |
| Composed KAN | 0.00985 [0.00487, 0.02122] |

No radius or initial condition was changed after observing this coverage.
A deliberately interpolative experiment must be a separate study.

## Shared Adam: computational cost

Each cell below gives **single-index / interaction** case medians.
Tuning time sums all four candidate fit times, not just the selected fit.

| Model | Selected-fit seconds | Total tuning seconds | Selected-fit cumulative allocated MiB |
|---|---:|---:|---:|
| Tensor spline | 2.762 / 2.903 | 11.04 / 15.16 | 8020 / 8530 |
| Single-index spline | 0.148 / 0.136 | 0.519 / 0.456 | 280 / 271 |
| MLP | 1.169 / 1.323 | 4.35 / 5.20 | 5033 / 5992 |
| Additive KAN | 0.762 / 0.859 | 5.40 / 3.56 | 156.6 / 184.6 |
| Composed KAN | 1.105 / 0.989 | 3.91 / 4.07 | 499.3 / 424.4 |

The optimized composed KAN has comparable or lower fitting cost than this
MLP, with much less cumulative allocation. The single-index spline remains
far cheaper. These are Julia 1.12.7, Darwin arm64, CPU Float64 measurements
with one BLAS thread and a two-iteration warm-up per model. Setup is recorded
separately; timings are descriptive measurements on a shared machine.
Allocated bytes are **not peak resident memory**.

## Native LAML/GCV tracks: not best-attainable baselines

These tracks change the optimizer and smoothing selection, so they must
not be merged into the approximation-only ranking above. Each entry again
uses ten complete seeds.

| Track | Case | Model | Trajectory NRMSE, median [IQR] | Median response RMSE | Worst seed response RMSE | Median fit seconds |
|---|---|---|---:|---:|---:|---:|
| LAML | Single index | Tensor spline | 0.05962 [0.05065, 0.09884] | 0.04417 | 0.12222 | 0.601 |
| LAML | Single index | Single-index spline | 0.14111 [0.13660, 0.15111] | 0.18430 | 0.20134 | 0.374 |
| LAML | Interaction | Tensor spline | 0.02894 [0.01996, 0.05106] | 0.03374 | 0.07134 | 0.828 |
| LAML | Interaction | Single-index spline | 0.22041 [0.20462, 0.22508] | 0.15697 | 0.22438 | 0.661 |
| GCV | Single index | Tensor spline | 0.06217 [0.05852, 0.06914] | 0.04149 | 0.05331 | 0.278 |
| GCV | Single index | Single-index spline | 0.02530 [0.01863, 0.07762] | 0.01152 | **5.80983** | 0.111 |
| GCV | Interaction | Tensor spline | 0.01883 [0.01069, 0.02166] | 0.01726 | 0.02440 | 0.231 |
| GCV | Interaction | Single-index spline | 0.21847 [0.18118, 0.22453] | 0.25752 | **3.72537** | 0.486 |

Finite completion hides important problems here. LAML never advanced
smoothing off initialization in **15/20 tensor fits**: six in the
single-index case and nine in the interaction case. All twenty nevertheless
reported iteration stability. Their median smoothing-stationarity residuals
were 0.114 and 0.100, respectively. Their EDF/covariance therefore cannot all
be interpreted as describing selected smoothing.

Conversely, the poor single-index LAML fits on the single-index truth had
median stationarity `1.40e-5` and all advanced smoothing. A small smoothing
gradient and stable iteration are not certificates of good response
recovery. All ten interaction single-index fits exhausted the LAML budget.
GCV single-index fits exhausted it in 3/10 single-index and 10/10 interaction
cases. GCV does not supply LAML stationarity/advancement diagnostics; those
fields are recorded as `missing`, not invented.

LAML single-index fits produced negative attack values on 2/80 native
test trajectories; GCV single-index fits did so on 14/80. No selected Adam
fit produced negative attack values at the evaluated test points. None
of these unconstrained approximators guarantees positivity elsewhere.

### Why the constant start matters

A separate, **post-hoc seed-101 diagnostic**, excluded from every ranking
above, compared the wrapped and bare single-index approximators. They gave
exactly identical LAML parameters in both cases, so these two discrepancies
are not caused by the initialization wrapper.

At the constant outer response, the prediction Jacobian's loading column
was exactly zero. The first LAML step moved the free loading from 1 to
0.5 on the single-index fixture and to 0 on the interaction fixture.
This start provides no local data information about the index direction.

Starting instead with the mildly decreasing outer response `0.3 - 0.04z`
(the same loading `[1,1]`, statistics, knots, and 25-iteration budget)
changed the single-index fixture as follows:

| Seed-101 LAML initialization | Validation NRMSE | Two-trajectory NRMSE | Response RMSE |
|---|---:|---:|---:|
| Constant 0.3 | 0.20783 | 0.13568 | 0.19859 |
| Mild decreasing slope | 0.01083 | 0.00839 | 0.00353 |

This demonstrates substantial initialization sensitivity, not a generally
validated replacement initialization. It did not cure the interaction
fixture: validation NRMSE changed from 0.29837 to 0.30623.
The native results must not be used to claim that single-index models are
incapable of fitting their own structural truth, or that KAN generally
outperforms a properly initialized native baseline.

## Limitations and priorities

The regularization search is not demonstrably wide enough: every additive
KAN selection and every interaction tensor selection chose the largest
candidate, `eta=1000`. All ten selected interaction tensor and MLP fits also
reached the Adam ceiling. These facts motivate new budget/grid studies;
they are not grounds for changing this study's settings retroactively.

Ten seeds are better than the earlier three, but there are still only two
equations and fixed trajectory splits. Initialization and data noise vary
together. Parameter counts do not equate representational capacity, priors,
optimization difficulty, or computational work. The edge penalty is not
curvature of the entire composed response, and learned edges are not a
uniquely identified scientific decomposition.

The next useful work is a separately declared interpolative trajectory
split, native initialization/multistart and smoothing-stall investigation,
and wider optimizer/regularization budgets selected without test feedback.
A further independent dynamical system should precede broad superiority
claims. Adaptive grids, alternative bases and symbolic extraction remain
deferred rather than being justified by this study.

## Artifacts and reproduction

Use [MULTIVARIATE.md](MULTIVARIATE.md) for fitting commands. The primary
archives are [Adam](results/multivariate-adam/),
[LAML](results/multivariate-laml/), and [GCV](results/multivariate-gcv/).
Each retains individual trials, selections, test trajectories, fitted
coefficients, package versions, source hashes, and its exact assessment
script snapshot. Snapshots are provenance records; run the current harness
from `benchmarks/kan`, not a relocated snapshot.

The original Adam and LAML `summary.md` files pooled twenty individual
trajectory errors. **Their medians are not the seed-level estimand used
in this report.** The unified [seed metrics](results/multivariate-analysis/seed_metrics.csv),
[distributions](results/multivariate-analysis/model_summary.csv), and
[paired comparisons](results/multivariate-analysis/paired_comparisons.csv)
provide consistent aggregation for all three tracks.

The retained analysis script checks each selected candidate against the
minimum valid validation error, reconstructs tuning time, checks coefficient
counts and archive hashes, and independently reproduces the GCV harness's
seed metrics exactly. From the repository root, with Python 3.11+:

```bash
python3 benchmarks/kan/analyze_multivariate.py --prefix=multivariate \
  --output=benchmarks/kan/results/multivariate-analysis-replay
```

The current analyzer summarizes saved scores without evaluating the current
package. Its archived predecessor required a matching working tree.
Use the [preserved fitting source](README.md#analysis-after-source-changes)
for actual historical model replay; the original analysis snapshots remain
provenance records.

LAML [diagnostics](results/multivariate-laml/diagnostics.csv) were recovered
in same-settings refits with exact coefficient and smoothing-parameter
agreement for all forty fits. Their provenance and the separate
[initialization probe](results/multivariate-laml/initialization-probe.csv)
are retained alongside them. No diagnostic refit replaced a primary result
or its timing. The shared package/extension/initialization-wrapper source
hash is `e309b4525899d04d30ab692f9479e7cd972a3f7cdb2e34d0ca63b0de49411032`;
each archive separately hashes its assessment-script version.
