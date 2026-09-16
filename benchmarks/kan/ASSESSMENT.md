# Initial KANApproximator assessment

**Performance update:** the cost measurements below describe the initial
evaluator. Local-support evaluation subsequently reduced the composed KAN's
nonlinear fit from 4.665 to 0.694 seconds and its allocation volume from
14.5 GiB to 309 MiB, with materially unchanged errors and selections.
See [PERFORMANCE.md](PERFORMANCE.md) for frozen-weight parity and the unchanged
experiment rerun. The accuracy conclusions below still apply; the old
compute-cost disadvantage against the MLP no longer does.

## Main conclusion

**KANApproximator is useful, but this experiment does not justify replacing
the conventional spline default.** Under a shared Adam training protocol,
KANs recover the nonlinear response substantially better than the matched
MLP. However, a conventional spline or GP is at least as competitive in
response recovery, usually better in longer-horizon prediction, and much
cheaper in the original implementation. The composed KAN has no consistent
accuracy advantage over the shallow KAN.

These are three-seed, two-problem CPU Float64 results. They compare one
specified initialization, optimizer, learning rate and penalty-selection
protocol, not each model family's best possible implementation or fit.

## What was compared

The fitted equation is `dN/dt = r(N)*N`. The two true responses are:

```math
r_{\mathrm{logistic}}(N)=0.8(1-N/2),
\qquad
r_{\mathrm{nonlinear}}(N)=0.8(1-N/2)[1+0.5\sin(3\pi N/2)].
```

Methods: an 8-coefficient practical cubic-spline baseline, a
parameter-matched 28-coefficient spline, 28-node SPDE and fixed-kernel GP
approximators, a 28-parameter tanh MLP, a single-edge 28-parameter KAN,
and a composed 1-2-1 KAN with 28 parameters.

All start from a rate of approximately 0.4. The model-specific hidden
features are seeded, while output-layer initialization makes the MLP/KAN
initial response constant. A wrapper prevents solver-specific
reinitialization from changing that protocol. GP interpolation introduces
a small, recorded initialization discrepancy.

Each model/seed/case selects its regularization from
`eta = [0, 0.1, 10, 1000]` using independent noisy validation observations.
The applied weight is `eta/tr(S)`. KAN uses the new complete-edge
curvature penalty with affine-null-space weight `1e-6`; the MLP uses
ridge, and conventional approximators use their native penalties.
This is not an assertion that these priors are equivalent.

Two runs use ceilings of 250 and 1000 Adam iterations at learning rate
0.03. Actual iteration counts can be lower because the solver retains its
normal stopping rules. There are 168 candidate fits per budget, 336 in
total. No candidate failed during training/validation; forecast failures
were retained in the results.

The training initial condition is 0.2, with noisy observations over
`0:0.2:6`; validation uses interleaved times and an independent noise draw.
The test initial condition is 0.5 and the test horizon extends to time 8.
Thus full-horizon error includes an extrapolation challenge.

## Larger-budget results

Numbers below are medians over three seeds at the 1000-iteration ceiling.
Full-trajectory medians are conditional on finite completion; success
counts must be read alongside them. "Success" means a finite completed
simulation, not that its fit is good.

| Method | Parameters | Logistic full RMSE | Finite forecasts | Nonlinear full RMSE | Finite forecasts | Nonlinear response RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Cubic spline | 8 | 0.00665 | 3/3 | 0.02121 | 3/3 | 0.02763 |
| Cubic spline | 28 | 0.03632 | 2/3 | 0.01186 | 3/3 | 0.01632 |
| SPDE | 28 | 0.22133 | 3/3 | 0.16722 | 3/3 | 0.02173 |
| GP, Matern-5/2 | 28 | 0.03669 | 2/3 | 0.01195 | 3/3 | 0.01631 |
| MLP, width 9 | 28 | 0.01107 | 3/3 | 0.14590 | 3/3 | 0.09433 |
| Shallow KAN | 28 | 0.00865 | 3/3 | 0.02802 | 3/3 | 0.01879 |
| Composed KAN, 1-2-1 | 28 | 0.01476 | 3/3 | 0.07532 | 3/3 | 0.02672 |

Response error is evaluated on true held-out trajectory states only where
those states overlap the training trajectory's state range. This covers
58.0% of logistic test samples and 63.0% of nonlinear test samples.
It is computed even if the longer forecast fails.

The in-support trajectory comparison tells a more qualified story:

| Method | Logistic in-support RMSE | Nonlinear in-support RMSE |
|---|---:|---:|
| Cubic spline, 8 | 0.00669 | 0.01384 |
| Cubic spline, 28 | 0.00803 | 0.01148 |
| SPDE, 28 | 0.00957 | 0.01774 |
| GP, 28 | 0.00803 | 0.01147 |
| MLP, 28 | 0.00517 | 0.10127 |
| Shallow KAN, 28 | 0.00713 | 0.01348 |
| Composed KAN, 28 | 0.00688 | 0.01040 |

The composed KAN is competitive on the supported part of the nonlinear
trajectory, but its full forecast is substantially worse than the spline
or GP. Low supported-trajectory error does not establish correct response
recovery or safe extrapolation.

## Cost

Median costs for the nonlinear problem at the larger budget:

| Method | Selected-fit seconds | Total tuning seconds | Selected-fit allocated MiB |
|---|---:|---:|---:|
| Cubic spline, 8 | 0.0184 | 0.0812 | 6.95 |
| Cubic spline, 28 | 0.3580 | 1.5175 | 124.59 |
| SPDE, 28 | 0.1204 | 1.1884 | 34.79 |
| GP, 28 | 1.1231 | 4.4779 | 92.44 |
| MLP, 28 | 0.6206 | 1.7847 | 1880.04 |
| Shallow KAN, 28 | 2.5137 | 12.080 | 8451.16 |
| Composed KAN, 28 | 4.6650 | 11.760 | 14835.12 |

Compilation warm-up is excluded from fit timing; constructor/setup time
is recorded separately. Allocation volume is cumulative, not peak memory.
BLAS uses one thread. These are single-machine timings in a shared
environment, not hardware-independent ratios.

The composed KAN's selected fit is about 7.5 times slower than the MLP and
13 times slower than the parameter-matched spline in this nonlinear case.
The large allocation volume is a concrete implementation bottleneck:
the current scalar evaluator repeatedly constructs full spline bases and
intermediate arrays. Parameter count alone hides this cost.

## Budget sensitivity and variability

At 250 iterations, the composed KAN's nonlinear full-trajectory and
response RMSE medians are 0.10430 and 0.05881; at 1000 they are 0.07532 and
0.02672. The shallow KAN improves from 0.04578 / 0.02541 to
0.02802 / 0.01879. Thus the smaller budget would understate the composed
KAN's response recovery.

The conventional 28-coefficient spline and GP also benefit substantially
from the larger budget. Both retained runs use the same four-point
regularization grid. Across all methods,
selected full-forecast failures fall from 7/42 at 250 iterations to 2/42
at 1000. The remaining failures are one logistic spline28 and one
logistic GP fit; both report `ODE solve failed: Unstable` on the held-out
forecast. Those failures are not dropped from the success counts.

Variability remains material. At the larger budget the composed KAN's
nonlinear full RMSE ranges from 0.04445 to 0.14605; the MLP ranges from
0.08935 to 0.35277. Three seeds are insufficient for statistical
superiority claims.

This is exploratory rather than a preregistered confirmation: an initial
pilot frequently selected the largest regularization candidate, so the
grid was broadened before both retained runs were performed. Independent
new seeds and problems are needed for confirmatory conclusions.

The SPDE and ordinary 28-coefficient spline currently share a cubic
interpolating evaluator; their distinction in this experiment is the
penalty. They are identical in the unpenalized smoke run. The GP kernel
and SPDE range were fixed, not tuned.

## Recommendation

Keep ordinary splines as the default for these small one-input ecological
PSMs. Offer KAN as an explicit alternative, particularly when a composed
representation may help on a genuinely multivariate response.

Priorities informed by this assessment:

1. Profile and reduce scalar-RHS allocations, using active/local basis
   support while retaining primal and derivative parity.
2. Add multivariate response problems, more independent trajectories and
   held-out initial conditions before judging the value of composition.
3. Include native LAML/GCV spline fits and alternative optimizer schedules
   as a separate best-practice comparison, not as an approximation-only
   experiment.
4. Increase seeds, expand regularization/range searches and assess
   validation rules that detect unsafe extrapolation.
5. Consider adaptive grids only after the fixed-grid cost and robustness
   picture is clearer.

## Reproduce and inspect

See [the protocol and commands](README.md).

- [250-iteration summary](results/pilot-250/summary.md)
- [1000-iteration summary](results/pilot-1000/summary.md)
- [All larger-budget trials](results/pilot-1000/trials.csv)
- [Larger-budget selections and failures](results/pilot-1000/selected.csv)
- [Selected coefficient vectors](results/pilot-1000/coefficients.toml)
- [Environment and source metadata](results/pilot-1000/metadata.toml)

These files record the exact settings and coefficients used. The result
is an initial implementation assessment, not a replication of the
KAN-ODE paper or a general result about KANs.
