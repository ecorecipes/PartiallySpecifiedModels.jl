# KAN reaction-diffusion assessment

**Composition helps on the constant-reaction control, but the practical
eight-coefficient spline is better on the nonlinear crowding reaction.**
The independent spatial example therefore reinforces the earlier conclusion:
KAN is a useful option, not a replacement for conventional spline defaults.
Native LAML spline fits are competitive and substantially cheaper here.

This is a fixed-mesh, ten-seed assessment with separate numerical
discretization controls. It is not a continuum-PDE accuracy claim or each
model family's best-attainable performance.

## Independent spatial problem

The [declared protocol](REACTION-DIFFUSION.md) learns `r(u)` in

```math
\partial_t u = 0.1\,\partial_{xx}u + u(1-u)r(u),\qquad x\in[0,6],
```

with zero boundary flux. The two reactions use `r(u)=1` and
`r(u)=0.4+1.2/(1+9u^2)`. A conservative, 24-cell finite-volume
method of lines supplies the dynamics to existing PSM ODE solvers.
It is not a separate neural PDE solver.

Two complete training profiles share one coefficient function. A third
profile selects regularization; two further profiles are held out through
time 4, beyond training time 2.5. Initial conditions are exact cell averages
of the declared cosine profiles, not noisy observations.

The study uses seeds 301--310, Gaussian measurement noise 0.01,
physical density domain `[0,1]` and common coefficient initialization 0.8.
Training loss averages over space and initial profiles at each time.
The shared-Adam track has a 150-iteration ceiling, learning rate 0.03 and
validation selection from `eta=[0,0.1,10,1000]`. A separate native LAML
track uses 25 iterations and internal smoothing selection for the two
conventional splines.

All **440 candidates** returned finite training/validation results:
400 Adam candidates and 40 native spline fits. All 140 selected fits
produced finite field and response scores for both held-out profiles.
No selected coefficient was negative at the evaluated true field values.
Finite completion does not certify fit quality or positivity elsewhere.

## Shared Adam: held-out field and reaction recovery

Errors pool squared space-time errors across the two profiles **within
seed** before taking the median and `[25th, 75th percentile]` across ten
seeds. Cells and times are not independent seed replicates.

Reaction RMSE scores `u(1-u)r(u)` along true test fields. The separately
retained coefficient error scores `r(u)` itself.

| Reaction | Model | Parameters | Field RMSE, median [IQR] | Reaction RMSE, median [IQR] |
|---|---|---:|---:|---:|
| Logistic | Spline | 8 | 0.000732 [0.000441, 0.000780] | 0.000717 [0.000458, 0.001307] |
| Logistic | Spline | 28 | 0.001479 [0.001103, 0.001845] | 0.002714 [0.001994, 0.003749] |
| Logistic | MLP | 28 | 0.000821 [0.000714, 0.000984] | 0.000991 [0.000814, 0.001265] |
| Logistic | Shallow KAN | 28 | 0.000827 [0.000756, 0.001067] | 0.001323 [0.001097, 0.001790] |
| Logistic | Composed KAN | 28 | **0.000526 [0.000321, 0.000732]** | **0.000481 [0.000329, 0.000641]** |
| Crowding | Spline | 8 | **0.001481 [0.001083, 0.002205]** | **0.001847 [0.001636, 0.002654]** |
| Crowding | Spline | 28 | 0.002450 [0.001769, 0.002915] | 0.005295 [0.004784, 0.005430] |
| Crowding | MLP | 28 | 0.003302 [0.002862, 0.004784] | 0.004261 [0.003789, 0.005922] |
| Crowding | Shallow KAN | 28 | 0.001771 [0.001673, 0.001964] | 0.003581 [0.003238, 0.004421] |
| Crowding | Composed KAN | 28 | 0.002934 [0.002767, 0.003144] | 0.004046 [0.003444, 0.004261] |

Paired wins for the **composed KAN**, out of ten seeds:

| Baseline | Logistic field wins | Logistic reaction wins | Crowding field wins | Crowding reaction wins |
|---|---:|---:|---:|---:|
| Spline8 | 8/10 | 7/10 | 2/10 | 1/10 |
| Spline28 | 10/10 | 10/10 | 2/10 | 9/10 |
| MLP28 | 8/10 | 9/10 | 7/10 | 6/10 |
| Shallow KAN28 | 8/10 | 10/10 | 1/10 | 4/10 |

The composed KAN is only modestly ahead of the MLP on crowding: its median
paired field-error ratio is 0.889. It is substantially behind the practical
spline and usually behind the shallow KAN on that fixture. Field prediction
and reaction recovery can rank methods differently, as the spline28
comparison illustrates.

### Coefficient recovery and density support

The known factors `u(1-u)` attenuate information about `r(u)` near zero and
one. Small field or reaction error therefore need not imply equally small
coefficient error.

| Model | Logistic coefficient RMSE | Crowding coefficient RMSE |
|---|---:|---:|
| Spline8 | 0.00570 | **0.01455** |
| Spline28 | 0.04419 | 0.03304 |
| MLP28 | 0.01115 | 0.03057 |
| Shallow KAN28 | 0.02444 | 0.02542 |
| Composed KAN28 | 0.00583 | 0.03290 |

The reverse-front test profile has 79.27% density-range overlap with training
on the logistic case and 76.83% on crowding; the fine-scale profile has 100%
overlap in both. The supported-reaction metrics are retained separately.
These are statements about the scalar density argument, not guarantees
about all spatial initial conditions or pattern scales.

## Computation and stopping behavior

Cells give **logistic / crowding** medians. Tuning includes all four
candidate fit times; allocation volume is cumulative, not peak memory.

| Model | Selected-fit seconds | Total tuning seconds | Selected-fit allocated MiB |
|---|---:|---:|---:|
| Spline8 | 0.100 / 0.109 | 0.500 / 0.533 | 26.4 / 29.5 |
| Spline28 | 0.526 / 0.552 | 2.71 / 2.81 | 95.6 / 105.5 |
| MLP28 | 1.230 / 1.427 | 8.21 / 8.14 | 3984 / 4640 |
| Shallow KAN28 | 0.791 / 0.731 | 3.92 / 3.99 | 93.9 / 96.6 |
| Composed KAN28 | 0.854 / 1.031 | 4.05 / 4.67 | 381.9 / 465.1 |

KAN again allocates less than this MLP implementation, but the practical
spline is much cheaper. These are warmed CPU Float64 measurements on one
Darwin arm64 machine, Julia 1.12.7, with one BLAS thread and serial tracks.
Setup and archive writing are outside fit timing.

All selected Adam fits stopped by the solver's plateau rule before the
150-iteration ceiling. Selected iteration counts range from 67 to 123 on
logistic and 70 to 107 on crowding. Merely raising the ceiling would not
necessarily change those fits. The separately declared
[optimizer refinements](REACTION-DIFFUSION-REFINEMENTS.md) address learning-rate
and stopping-rule sensitivity without post-hoc adjustment of this study.

## Native spline baseline

This track changes the optimization and smoothing selection, so it is not
an approximation-only ranking against Adam.

| Reaction | Model | Median field RMSE | Median reaction RMSE | Median coefficient RMSE | Median fit seconds |
|---|---|---:|---:|---:|---:|
| Logistic | Spline8 | 0.000537 | 0.000513 | 0.00442 | 0.147 |
| Logistic | Spline28 | 0.000537 | 0.000513 | 0.00443 | 0.275 |
| Crowding | Spline8 | 0.001423 | 0.001759 | 0.01287 | 0.147 |
| Crowding | Spline28 | 0.001467 | 0.001827 | 0.01320 | 0.262 |

All forty native fits advanced smoothing and reported iteration stability.
Median stationarity residuals range from `3.34e-8` to `8.54e-6`; no
initial-smoothing stall occurs among the selected native fits.
Those diagnostics are not global-optimum certificates.

The native logistic splines are comparable to the composed KAN's accuracy
at much lower search cost. Native splines also provide the strongest
crowding results in this experiment.

## Spatial error is a separate quantity

The fitting references use the **same 24-cell semidiscrete equation** as
the fitted models. Separate true-reaction refinements to 48 and 96 cells,
restricted conservatively to the coarse cells, give ratios 4.005--4.035,
consistent with second-order spatial convergence.

The largest 24-vs-48 field discrepancies are `0.001143` on the logistic
fine-scale profile and `0.000878` on crowding. These are comparable to or
larger than some fitting errors. In particular, a KAN field RMSE of
0.000526 on the logistic same-mesh problem is **not** a claim of that
accuracy against the continuum PDE. Finite-mesh approximation,
measurement noise, function estimation and optimization are distinct
error sources.

The [no-refit mesh-transfer continuation](REACTION-DIFFUSION-REFINEMENTS.md)
also evaluates the saved fitted reactions on finer meshes, separating
transferred-model error from the true-field discretization shift.

The pure-diffusion mode oracle and exact initial-average restriction are
independent of fitting. Deliberately removing `dx^-2` leaves shared
fitting/reference checks green but fails the analytic physics contracts.
This prevents a common-mode discretization error from masquerading as a
successful approximator comparison.

## Artifacts and reproduction

See [REACTION-DIFFUSION.md](REACTION-DIFFUSION.md) for the equation,
profiles, controls and commands. The benchmark generalizes the existing
univariate constructor with explicit domain/initial-rate keywords while
preserving its old defaults. It adds no PDE dependency or new PSM solver.

Primary archives: [Adam](results/reaction-diffusion-adam/) and
[LAML](results/reaction-diffusion-laml/). They retain every candidate,
selected parameters, independent profile scores, mesh controls,
diagnostics, source snapshots and resolved package versions.

The [seed distributions](results/reaction-diffusion-analysis/model_summary.csv)
and [paired comparisons](results/reaction-diffusion-analysis/paired_comparisons.csv)
are derived by checking the candidate grid, validation minimum, complete
profile sets, parameter provenance and tuning-time accounting:

```bash
python3 -B benchmarks/kan/analyze_reaction_diffusion.py \
  --output=benchmarks/kan/results/reaction-diffusion-analysis-replay
```

The analysis performs no model evaluation or refitting. Both fitting tracks
record source hash
`475352efbab2549bacf2512d7369689f37ce403fbeee036fc659389c610745aa`
and assessment-script hash
`1a14286a22e254c8e02dafe7239862dbb787a366d21b047b3e28989c86afed86`.
Earlier univariate, multivariate and corrected-LAML results are not rewritten.
The exact spatial fitting source, license and resolved benchmark manifest
are preserved in [`source-475352ef.tar.gz`](results/source-475352ef.tar.gz),
with archive SHA-256
`600cb1eb3c0ec550a963030b062a24273264f0e55c23ad653811d082ce15c7be`.
