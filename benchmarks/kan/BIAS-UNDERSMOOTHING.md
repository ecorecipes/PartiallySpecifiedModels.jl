# Bias, variance and a focused undersmoothing experiment

The [fresh coverage study](FRESH-COVERAGE.md) established remaining
nonlinear undercoverage. This continuation first diagnoses the retained
fits without refitting, then evaluates a predeclared change on independent
datasets. Diagnostic knowledge of truth is never used to construct the
deployable intervals.

**Outcome:** the predeclared quarter-strength refit materially improves
nonlinear coverage on independent data. KAN covariance coverage rises from
93.5% to 97.3%, and percentile-bootstrap coverage from 75.5% to 91.5%.
The intervals widen, and some bootstrap undercoverage remains. This
supports a bias--variance explanation, not a universal quarter-lambda rule.

The [subsequent robustness study](COVERAGE-ROBUSTNESS.md) keeps the same
fraction and varies sampling density, noise and response shape. It
reproduces the reference-condition gain but finds marked failures,
especially for KAN under doubled observation noise. The results below
remain unchanged and should be read within their original scope.

## Retained-fit decomposition

At each physical query point, across the 100 fresh datasets, record:
mean response bias, empirical sampling SD, RMS covariance SE, RMS bootstrap
SD, bootstrap-estimated bias, and the bias/variability after subtracting
that bootstrap bias. Variance is measured around the empirical mean, not
around truth: otherwise squared bias would masquerade as sampling variance.

For nonlinear KAN12 at density 0.6:

| Quantity | Measured value |
|---|---:|
| Mean error | -0.01887 |
| Empirical sampling SD | 0.01056 |
| RMS covariance SE | 0.01532 |
| RMS bootstrap SD | 0.00976 |
| Mean bootstrap-estimated bias | -0.00600 |
| Remaining bias after bootstrap correction | -0.01288 |
| Sampling SD of the bias-corrected estimate | 0.01451 |

Thus covariance uncertainty is not simply too small: its RMS scale is
already larger than empirical sampling SD. Bias is substantial, the
bootstrap estimates only part of it, and correcting the centre increases
sampling variability. Keeping the original bootstrap SD after correcting
the centre is not automatically an adequate uncertainty calculation.

Spline8 exhibits the same issue, though less severely: at density 0.6,
bias is -0.01624, empirical SD 0.01228, covariance SE 0.01596 and bootstrap
SD 0.01169. Bootstrap correction leaves bias -0.01053 and increases
sampling SD to 0.01702.

A **truth-assisted diagnostic** subtracts the mean error from all the
*other* datasets at each point. This leave-one-dataset-out centering raises
KAN's average in-range covariance coverage from 91.8% to 98.1%, and
normal-bootstrap coverage from 85.7% to 96.1%. The corresponding spline
figures are 93.5% to 97.5% and 88.3% to 95.5%. These oracle calculations
are not an implementable correction and are not used by the experiment.
They help distinguish bias from a uniform lack of uncertainty scale.

## Predeclared focused procedure

Compare two procedures for spline8 and free-affine KAN12:

1. Select smoothing with the original `LAML(maxiters=40,jac=:forwarddiff)`.
2. Warm-start a coefficient refit at a fixed fraction of the selected
   lambda, retaining the same basis, initialization, priors and observations.
3. Compute covariance and bootstrap intervals from the returned refit.

The fractions are **1.0** (same-strength control) and **0.25**
(quarter-strength smoothing). They are fixed in advance, not selected
from observed coverage. Both procedures do the second coefficient fit;
otherwise extra optimization would be confounded with changing smoothing.

The explicit `LAML(fixed_lambda=...)` mode holds the penalty weight,
skips smoothing selection and GP adaptation, and retains ordinary
coefficient convergence. A long `warmup` is not an equivalent substitute:
it also prevents normal convergence checks. Final-stage
`smoothing_advanced=false` is intentional; first-stage selection diagnostics
are recorded separately.

Every bootstrap replicate repeats **both stages**, selecting its own
lambda and applying the same fixed fraction. It does not borrow the
original dataset's lambda or reuse old bootstrap draws. Each procedure
resamples from its own fitted trajectories and residual scale. Common
random seeds across procedures align random shocks, not necessarily the
resulting pseudo-observations.

## Independent evaluation

Use seeds **3001--3050**, disjoint from both earlier calibration campaigns,
for both logistic and nonlinear truths. The two populations, noise
standard deviation 0.015, observation times, physical domain and sixteen
query points are unchanged. There are 50 datasets per case, two models,
two fixed fractions, and 99 attempted bootstrap refits per original fit.
Nominal levels remain 90% and 95%.

This gives 400 two-stage original fits and 39,600 two-stage bootstrap
refits, up to 80,000 underlying coefficient/selection solves. It is not
80,000 independent coverage replications. Availability, failure counts,
widths, pointwise Wilson limits and dataset-clustered Monte Carlo errors
remain explicit. No failed or strongly smoothed fit is dropped to improve
the headline.

This is one motivated undersmoothing procedure, not a promise of nominal
coverage or a universal smoothing fraction. Lower bias may cost wider
intervals, more variance, poorer extrapolation, or different optimization
behavior. Those tradeoffs must be reported.

## Reproduction

```bash
python3 -B benchmarks/kan/decompose_coverage.py --output=benchmarks/kan/results/coverage-bias-variance-replay
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/undersmoothing.jl \
  --models=spline8 --fractions=1 --output=benchmarks/kan/results/undersmoothing-spline8-1
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/undersmoothing.jl \
  --models=spline8 --fractions=0.25 --output=benchmarks/kan/results/undersmoothing-spline8-quarter
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/undersmoothing.jl \
  --models=kan12_free --fractions=1 --output=benchmarks/kan/results/undersmoothing-kan12_free-1
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/undersmoothing.jl \
  --models=kan12_free --fractions=0.25 --output=benchmarks/kan/results/undersmoothing-kan12_free-quarter
python3 -B benchmarks/kan/analyze_undersmoothing.py
```

Archives retain both-stage diagnostics, selected and fixed lambdas,
coefficients, covariance, bootstrap evaluations, attempted identities,
observations and source/environment provenance. The analyzer requires
the exact fraction relation in every successful original and bootstrap
fit and compares only matched datasets. Earlier studies remain unchanged.

## Independent results

All 400 original two-stage fits and 39,600 two-stage bootstrap refits
produce finite results. Their selected and fixed lambdas satisfy the
declared fraction relation in every recorded successful fit. First-stage
selection and final fixed-weight convergence are reported separately.
The final-stage `smoothing_advanced=false` is intentional, not a stalled
selection.

Average in-range pointwise coverage of nominal 95% intervals, over
50 independent nonlinear datasets:

| Procedure | Spline8, fraction 1 | Spline8, fraction 0.25 | KAN12 free, fraction 1 | KAN12 free, fraction 0.25 |
|---|---:|---:|---:|---:|
| Covariance | 94.2% | 96.7% | 93.5% | 97.3% |
| Percentile bootstrap | 83.3% | 92.2% | 75.5% | 91.5% |
| Basic bootstrap | 82.7% | 89.6% | 82.5% | 88.2% |
| Normal bootstrap | 85.5% | 93.3% | 84.2% | 93.3% |
| Bias-corrected normal bootstrap | 84.4% | 91.6% | 84.7% | 91.1% |

The paired KAN percentile gain is **16.0 percentage points**, with
dataset-clustered Monte Carlo SE 1.67 pp. Its covariance gain is 3.82 pp
(MCSE 1.07 pp). Spline gains are 8.91 pp (1.71 pp) for percentile and
2.55 pp (0.74 pp) for covariance.

For the quarter-strength procedure, KAN covariance coverage has MCSE
1.17 pp and percentile coverage 1.75 pp. The latter remains below the
nominal level. A high regional average also does not guarantee uniform
pointwise coverage.

### Width and estimation costs

| Model | Mean covariance width, fraction 1 / 0.25 | Mean percentile width, fraction 1 / 0.25 | Response RMSE, fraction 1 / 0.25 |
|---|---:|---:|---:|
| Spline8 | 0.03878 / 0.05034 | 0.03170 / 0.04228 | 0.01204 / 0.01254 |
| KAN12 free | 0.03786 / 0.05093 | 0.03294 / 0.04204 | 0.01251 / 0.01247 |

KAN covariance intervals widen by about 35%, and percentile intervals
by 28%. Spline widths increase by about 30% and 33%. Point-estimation
RMSE does not uniformly improve: less smoothing exchanges bias for
variance, rather than making every criterion better.

At density 0.6, KAN mean bias decreases from -0.01950 to -0.01255,
while covariance width increases from 0.06001 to 0.09372. Pointwise
covariance coverage increases from 44/50 to 47/50 datasets; the latter's
95% Wilson MC limits are 83.8--97.9%. This is evidence for the mechanism,
not precise certification of 95% coverage at that point.

### Control truth and extrapolation

On the logistic control, KAN covariance coverage is 92.4%/92.5% and
percentile coverage 92.9%/94.9% for fractions 1/0.25. Spline covariance
coverage is 93.8%/92.7%, while percentile coverage is 95.3% for both.
Undersmoothing increases point-estimation RMSE on this affine truth:
KAN 0.00255 to 0.00279 and spline 0.00310 to 0.00391. The control therefore
does not support indiscriminate weakening of smoothing.

Nonlinear extrapolation improves but remains uneven. KAN percentile
coverage below the training range rises from 1% to 56%, and above it
from 72% to 94%. Covariance coverage reaches 97% below and 100% above,
with much wider mean intervals (0.3274 and 0.3034). The spline exhibits
the same qualitative tradeoff. Extrapolative percentile inference is
still unreliable below the observed range.

## Interpretation

The retained-fit decomposition showed that a large part of the deficit
was response bias, not merely insufficient reported covariance variance.
The independent quarter-strength experiment supports that explanation:
coverage improves after a prespecified reduction in smoothing, without
using truth to move interval endpoints.

Both the control and treatment perform a second coefficient fit. Each
bootstrap replicate selects its own smoothing and repeats the conditional
refit. The result is therefore not explained by simply giving one method
extra optimization or treating the original selected lambda as known
throughout resampling.

This is a bounded, encouraging result for two truths, two approximators,
one fixed fraction and these observation conditions. It does not establish
a universal correction, simultaneous coverage, a best smoothing fraction
or reliability for composed networks. The package's automatic smoothing
defaults have not changed. `LAML(fixed_lambda=...)` is an opt-in primitive;
using it alone is not the full two-stage bootstrap procedure.

The next investigations should assess the robustness of the bias--width
tradeoff under different observation designs and response shapes, not tune
the fraction to the coverage observed here.

## Retained evidence

The [audited decomposition](results/coverage-bias-variance-audited/)
retains dataset-level bias/scale records and clearly labelled
truth-assisted diagnostics. The [independent experiment](results/undersmoothing-analysis/)
retains every evaluated interval, pointwise Wilson limits, region and
dataset summaries, paired fraction effects and both-stage status counts.
Its four fitting cohorts are linked by the metadata and retain observations,
coefficients, covariance, bootstrap evaluations and selection diagnostics.

The execution source hash is
`49a5566bc7b76585e95caf70abd7c5449d4f8ca1aafc110d4ad3946a1ae8b061`.
The [source snapshot](results/source-undersmoothing.tar.gz) has SHA-256
`f1ceda1a9cf3144060177febd2fc636652bb6f8d3150ae04655c3ec818f249cb`;
the experimental refitter hash is
`0bb322bbf1793b0a6319602b9c1877a5a7f448da30b9462b25355c38a83d0b21`.
Earlier data, fits and calibration results have not been rewritten.
