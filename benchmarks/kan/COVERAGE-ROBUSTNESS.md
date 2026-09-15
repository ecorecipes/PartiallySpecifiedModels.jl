# Robustness of the fixed undersmoothing procedure

The [focused undersmoothing study](BIAS-UNDERSMOOTHING.md) found improved
coverage under a prespecified quarter-strength penalty, with wider
intervals. This continuation asks whether that tradeoff survives changes
to the observations and response shape. The multiplier is not retuned.

**Outcome:** quarter-strength smoothing improves average coverage in all
four conditions, but it is not a general calibration fix. KAN nominal
95% covariance coverage reaches 96.1% in the reference condition, 92.4%
with sparse observations and 90.6% for the localized response, yet only
67.3% when observation noise doubles. The reference-condition gain
reproduces; the higher-noise failure is substantial.

## Design declared before evaluation

Use spline8 and single-edge KAN12 with their previously declared free-affine
curvature penalties and initializations. Each procedure selects lambda
using LAML, then warm-starts a coefficient fit at either **1.0** or **0.25**
times its selected value. Every bootstrap repeats both stages.

All conditions retain two populations starting at `[0.2,0.7]`, time horizon
5, physical domain `[0,2.4]`, Gaussian observation errors, the same sixteen
query points and LAML iteration settings. Change one aspect at a time:

| Condition | Observation times | Noise SD | Response |
|---|---|---:|---|
| Reference | `0:0.25:5` (21 per population) | 0.015 | Original nonlinear response |
| Sparse | `0:0.5:5` (11 per population) | 0.015 | Same response |
| Noisy | `0:0.25:5` | 0.030 | Same response |
| Localized | `0:0.25:5` | 0.015 | Localized bump response |

The original nonlinear response is
`0.9(1-N/2)(1+0.35sin(2N))`. The new response is

```text
0.9*(1-N/2) * (1 + 0.5*exp(-((N-0.9)/0.25)^2))
```

The Gaussian-shaped multiplier is centred at 0.9 with scale 0.25 and peak
increment 0.5. The approximators receive neither formula. Their grids and
parameter counts stay fixed, so this also probes sensitivity to functional
resolution; it is not an architecture-selection experiment.

There are **30 independent datasets per condition**, seeds 4001--4030,
disjoint from earlier campaigns. Conditions use independent random streams.
Within each condition/seed, both models and both fractions receive exactly
the same observations. The same bootstrap RNG seeds align random shocks;
each procedure still resamples from its own fitted trajectory and scale.

Use 99 attempted parametric refits per original fit and evaluate nominal
90%/95% pointwise intervals. This gives 480 two-stage original fits and
47,520 two-stage bootstrap refits, up to 96,000 underlying solves—not
96,000 independent coverage replications.

## Accounting and limits

In-range/below-range/above-range labels are determined separately from each
condition's noiseless trajectory. All queries lie inside the declared
approximator domain. Preserve every requested interval, failed fit and
unavailable evaluation. Report conditional coverage, availability,
available-and-covers yield, width, response error and dataset-clustered
Monte Carlo uncertainty.

The reference, sparse and noisy conditions have the same noiseless
trajectories at common observation times. The localized reference is
also checked through independent separable-ODE inverse-time quadrature,
not just through the same fitting RHS.

No fraction is chosen using these coverage results. Thirty datasets give
limited Monte Carlo precision, especially for extrapolation and rare
failures. The goal is to expose robust gains or failure modes, not to
certify universal calibration or recommend the factor 0.25 for every model.

## Reproduction

```bash
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/coverage_robustness.jl \
  --models=spline8 --fractions=1 --output=benchmarks/kan/results/robustness-spline8-1
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/coverage_robustness.jl \
  --models=spline8 --fractions=0.25 --output=benchmarks/kan/results/robustness-spline8-quarter
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/coverage_robustness.jl \
  --models=kan12_free --fractions=1 --output=benchmarks/kan/results/robustness-kan12_free-1
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/coverage_robustness.jl \
  --models=kan12_free --fractions=0.25 --output=benchmarks/kan/results/robustness-kan12_free-quarter
python3 -B benchmarks/kan/analyze_coverage_robustness.py
```

Cohorts may run independently; each uses one Julia/BLAS thread and serial
bootstrap refits. Timings are not comparative performance claims. Outputs
reject overwriting and retain scenario specifications, observations,
selected/fixed lambdas, coefficient/covariance records, replicate identities,
both-stage diagnostics, interval-level scores and source/environment hashes.
The independent analyzer verifies response formulas, observation grids,
noise declarations, paired data and the exact fraction relation.

## Results on the new datasets

All 480 two-stage original fits and 47,520 two-stage refits are finite.
Both stages report iteration stability, and smoothing selection advances
in the first stage. Every evaluated interval is available. None of these
computational diagnostics establishes coverage quality.

Average in-range pointwise coverage of **nominal 95% intervals**.
Each cell shows same-strength control -> quarter-strength smoothing:

| Condition | Spline covariance | Spline percentile | KAN covariance | KAN percentile |
|---|---:|---:|---:|---:|
| Reference | 91.5% -> 95.2% | 80.0% -> 90.9% | 90.3% -> 96.1% | 75.8% -> 92.1% |
| Sparse observations | 67.3% -> 75.8% | 59.4% -> 72.1% | 81.5% -> 92.4% | 62.4% -> 82.1% |
| Double noise SD | 78.8% -> 87.6% | 71.8% -> 86.4% | 61.5% -> 67.3% | 56.1% -> 67.3% |
| Localized response | 59.7% -> 70.3% | 51.5% -> 59.1% | 73.6% -> 90.6% | 53.3% -> 80.9% |

The quarter-strength KAN covariance/percentile MCSEs, clustered by the
30 independent datasets, are 1.67/2.42 percentage points for reference,
2.54/3.74 pp for sparse, 6.76/5.91 pp for noisy and 2.20/2.70 pp for
localized. The full pointwise Wilson limits and regional MCSEs are retained.
These are not simultaneous bands or independent query-point replications.

KAN's paired percentile-coverage gains are 16.36 pp (MCSE 2.32 pp),
19.70 pp (2.09 pp), 11.21 pp (3.04 pp) and 27.58 pp (2.48 pp) in the
four conditions respectively. The gains are real features of this
experiment, but even a large gain can leave coverage far below 95%.

## Width and prediction tradeoffs

Mean in-range widths under quarter-strength smoothing:

| Condition | Spline covariance / percentile | KAN covariance / percentile |
|---|---:|---:|
| Reference | 0.04907 / 0.04196 | 0.04995 / 0.04123 |
| Sparse | 0.04838 / 0.04503 | 0.05662 / 0.04578 |
| Noisy | 0.07063 / 0.06426 | 0.05627 / 0.05139 |
| Localized | 0.06026 / 0.05320 | 0.06378 / 0.05572 |

The intervals widen relative to their controls in every condition. Point
estimation does not uniformly improve. KAN response RMSE changes from
0.01270 to 0.01233 (reference), 0.01858 to 0.01701 (sparse),
0.02672 to 0.02715 (noisy), and 0.02875 to 0.02302 (localized).
For spline8 the corresponding pairs are 0.01233/0.01233,
0.02261/0.02186, 0.02276/0.02298 and 0.03409/0.03220.

The localized response is harder for these fixed representations,
especially spline8. KAN12 has more parameters, a different basis and
different initialization, so this is not proof of an intrinsic KAN
advantage. Basis resolution, observation information and smoothing remain
distinct questions.

## Why the noisy condition is different

A post-hoc diagnostic finds 12/30 noisy KAN control fits with EDF below 3.
Ten remain below 3 after quartering lambda. This strongly smoothed subgroup
is absent from the KAN reference, sparse and localized controls.

Keep those twelve datasets in the primary results. Within that diagnostic
subgroup, quarter-strength covariance/percentile coverage is only
24.2%/30.3%; in the other eighteen datasets it is 96.0%/91.9%.
Removing the difficult subgroup would conceal the procedure's failure.

Dividing a very large selected penalty by four can leave the fitted
response essentially restricted to a low-dimensional regime. Moreover,
parametric bootstrap data are generated from the fitted response, which
may not reproduce features of the unknown true response that were smoothed
away. This is consistent with the observed failure, not proof of a solver
bug or a guarantee that a different multiplier would solve it.

Sparse spline8 has a related warning: 7/30 controls have EDF below 3,
and all seven remain there after quartering. Thus the problem is not
exclusive to KANs, and the dominant failure mechanism can change with
the observation design.

## Extrapolation remains uneven

Quarter-strength KAN percentile coverage below/above the training range
is 55.0%/96.7% for reference, 26.7%/92.2% for sparse,
36.7%/58.9% for noisy, and 71.7%/96.7% for localized.
Covariance intervals can have much higher extrapolative coverage by
becoming very wide: for localized KAN their mean widths are 0.5836 below
and 0.6338 above the training range.

Consequently, neither good in-range averages nor finite intervals provide
a general extrapolation guarantee. The limited 30-dataset replication is
also particularly important for rare tail events and apparent 100% coverage.

## Interpretation and evidence

The reference condition independently reproduces the earlier coverage
improvement. The broader result is more qualified: **undersmoothing helps,
but the fixed quarter-strength rule is not robustly nominal across noise
levels, observation density or response shapes**. Automatic selection of a
strongly smoothed regime, remaining response bias and finite representation
can matter more than a modest rescaling of lambda.

No multiplier was retuned, no failed or strongly smoothed fit was dropped,
and no solver default was changed. The next useful work is to understand
uncertainty about the fitted smoothing/response regime and separate
resolution limits from observation-information limits—not to keep lowering
the multiplier until this particular dataset achieves 95%.

Primary cohorts: [spline control](results/robustness-spline8-1/),
[spline quarter](results/robustness-spline8-quarter/),
[KAN control](results/robustness-kan12_free-1/) and
[KAN quarter](results/robustness-kan12_free-quarter/).
The [analysis](results/coverage-robustness-analysis/) retains all 76,800
evaluated intervals across five constructions, pointwise and regional
summaries, paired fraction effects, and both-stage diagnostics. Conditions
are never pooled into one coverage claim.

The [execution snapshot](results/source-coverage-robustness.tar.gz) has
SHA-256 `65e8913a7f640252a48f69963cf2b40c3c68fade5a4f646a19a242591cb2453d`.
The robustness harness hash is
`9f2d8e23e34f255427cd137dba422583ad6ca3e25852209221a302e6c6c9f303`;
the scenario-aware calibration helper hash is
`e1ee776cebbcb3ecd5ce5b545c89f9373c7e1d4707070c21cca50cc357aced42`.
Earlier studies remain unchanged.
