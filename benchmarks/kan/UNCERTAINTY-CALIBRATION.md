# Pointwise uncertainty calibration for a fixed-grid KAN

This study asks whether the intervals demonstrated in
[vignette 41](../../vignettes/41_kan/41_kan.md) attain their nominal coverage
under repeated noisy datasets. Computing an interval correctly is not an
empirical coverage guarantee. The scope is deliberately a small unary KAN,
not every architecture supported by the API.

**Result:** the logistic control is reasonably close to nominal coverage,
but the nonlinear example exposes substantial undercoverage. For nominal
95% intervals, the KAN's average in-range coverage is **68.9% for covariance
intervals and 55.2% for the percentile bootstrap**. The corresponding spline
figures are 91.9% and 81.4%. These failures occur despite finite intervals,
iteration stability and advanced smoothing selection; they are not repaired
merely by computing more bootstrap replicates on the tested subset.

The [subsequent methods/prior investigation](UNCERTAINTY-METHODS.md)
compares five interval constructions and identifies an important scope
qualification: the original KAN used the nondefault, jointly scaled
`1e-6` affine penalty. Leaving that component unpenalized produces about
91% covariance coverage in an exploratory replay. The original numbers
below remain unchanged and must not be generalized to all KAN prior choices.

## Protocol declared before the study

Two populations share an unknown response, `N'_j = r(N_j) N_j`, starting
at `[0.2, 0.7]`. Observe both at times `0:0.25:5`, with independent additive
Gaussian noise of standard deviation 0.015.

| Case | True response |
|---|---|
| Logistic control | `0.9(1-N/2)` |
| Nonlinear | `0.9(1-N/2)(1+0.35sin(2N))` |

The logistic reference is also compared against its independent analytic
solution, rather than only against the same numerical RHS used for fitting.
References use Tsit5 at `1e-11` tolerances; fitting uses `1e-8` and an ODE
iteration guard of 10,000.

There are **100 independent datasets per case**, seeds 1001--1100. Both
models receive exactly the same observations for a given case/seed:

| Model | Parameters | Initialization and regularization |
|---|---:|---|
| `spline8` | 8 | Natural cubic spline on `[0,2.4]`, initial response 0.5 |
| `kan12` | 12 | One FluxKAN cubic edge, grid size 8, SiLU base; complete-edge curvature plus declared `1e-6` affine-null-space penalty |

KAN architecture, grids and initialization seed 42 are fixed across
datasets and bootstrap replicates. This is conditional calibration of that
procedure, not initialization/architecture-selection uncertainty. The
spline is a practical baseline, not a parameter-matched or identical-prior
control. No architecture, knot, initialization or hyperparameter search
uses the true response or the interval coverage.

Every fit uses `LAML(maxiters=40, jac=:forwarddiff)`. Its smoothing parameters
are reselected in bootstrap refits; the covariance intervals condition on
the smoothing selected in the original fit. Stability, smoothing progress
and stationarity are recorded, not used as truth-informed exclusion gates.
Finite fits can still be poor fits.

## Queries and interval methods

Evaluate **90% and 95% pointwise intervals** at sixteen fixed physical
densities: two below the noiseless training range (0.05, 0.1), eleven
in-range (`0.3:0.15:1.8`), and three above it (2.1, 2.25, 2.35).
All are inside the approximators' declared domain. Thus extrapolation
means beyond the training trajectory's range, not outside the KAN grid.
An in-range classification does not by itself establish identifiability.

The primary methods are the public covariance/delta-method intervals and
parametric percentile bootstrap intervals with **99 attempted refits**.
Seeds 1001--1020 are a predeclared tail-resolution sensitivity subset:
run 199 attempts and also report the 99-attempt prefix. This gives 400
original fits and up to 47,600 bootstrap refits, not 48,000 independent
coverage replicates.

The prefix is selected by **attempt identity**, not by taking the first
99 successful columns. Failed refits otherwise pull later attempts into
the smaller budget and silently change the procedure. The same per-dataset
bootstrap RNG seed is used across models; no bootstrap randomness chooses
the architecture or tuning settings.

Ninety-nine refits give only a few order statistics in a 2.5% tail. These
results characterize that finite-budget procedure; the 199-attempt subset
assesses sensitivity rather than proving an infinite-bootstrap limit.

## Failure-aware coverage and Monte Carlo error

Every requested dataset/method/level/query combination has a row, including
failed fits, unavailable covariances and insufficient bootstrap evaluations.
Availability requires finite ordered endpoints. A percentile interval can
remain available even if the original fitted point estimate is unavailable.
Programming/configuration errors interrupt execution; recognized numerical
failures and interval-quorum failures are recorded.

Report three distinct quantities:

- **Availability:** fraction of datasets producing a usable interval.
- **Coverage given availability:** containment among usable intervals.
- **Available-and-covers yield:** fraction of all datasets that produce an
  interval containing truth. Missing intervals cannot improve this number.

Pointwise Monte Carlo limits are 95% Wilson intervals over independent
datasets. With 100 datasets, nominal 95% coverage has binomial standard
error about 2.18 percentage points even before accounting for failures.
Widths, bias and response RMSE accompany coverage: an excessively wide
interval is not automatically a useful one.

Regional summaries weight the declared query points equally but cluster
uncertainty by dataset. If `C_i` and `A_i` are that dataset's fractions of
covered-and-available and available queries, regional conditional coverage
is `mean(C)/mean(A)`. Its delta-method Monte Carlo SE uses the dataset
influence scores `(C_i - coverage*A_i)/mean(A)`. Queries from one fitted
curve are **not** independent simulation replicates. A zero estimated MCSE
at a boundary is not certainty; pointwise Wilson limits remain nondegenerate.
No simultaneous confidence band or multiple-comparison claim is made.

## Reproduction and artifacts

From the repository root, using the existing benchmark environment:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_uncertainty.jl \
  --cases=logistic --models=spline8 \
  --output=benchmarks/kan/results/uncertainty-calibration-logistic-spline8
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_uncertainty.jl \
  --cases=logistic --models=kan12 \
  --output=benchmarks/kan/results/uncertainty-calibration-logistic-kan12
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_uncertainty.jl \
  --cases=nonlinear --models=spline8 \
  --output=benchmarks/kan/results/uncertainty-calibration-nonlinear-spline8
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_uncertainty.jl \
  --cases=nonlinear --models=kan12 \
  --output=benchmarks/kan/results/uncertainty-calibration-nonlinear-kan12
python3 -B benchmarks/kan/analyze_calibration.py
```

The four cohorts can run independently; bootstrap refits within each cohort
are serial and BLAS uses one thread. Recorded timings include first-call
compilation and possible machine contention: this is a coverage study,
not a comparative performance benchmark.

The runner also accepts `--seeds`, `--iterations`, `--levels`, `--nboot`,
`--sensitivity-seeds` and `--sensitivity-boot`; changing them declares a
different experiment. A single combined cohort is also supported, with
its directory supplied to the analyzer's `--inputs`. Existing outputs are
never overwritten.

Archives retain the design before fitting, observations, original
coefficients/covariances, bootstrap coefficients and query evaluations,
attempt identities, refit diagnostics, every interval, source snapshots and
resolved environment metadata. The independent Python analyzer reconstructs
normal/percentile endpoints, interval membership, failure denominators,
paired comparisons and dataset-clustered Monte Carlo summaries. Its plot's
error bars describe **Monte Carlo uncertainty in coverage**, not uncertainty
in a fitted response.

## Results

All 400 original fits and all 47,600 bootstrap refits produced finite
parameters and trajectories. Every fit stopped with `:converged_tol`, and
every fit advanced smoothing. All 28,160 requested intervals are available.
Consequently conditional coverage and available-and-covers yield coincide
in this run. This is a numerical-completion result, not evidence that the
uncertainty procedure is calibrated.

### In-range coverage

The table averages the eleven fixed in-range query points within each
dataset. Parentheses give **one dataset-clustered Monte Carlo standard error**
in percentage points, not a fitted-response interval.

| Truth | Model | Covariance: nominal 95% | Bootstrap99: nominal 95% | Mean covariance width | Mean bootstrap width |
|---|---|---:|---:|---:|---:|
| Logistic | Spline8 | 94.3% (1.3 pp) | 96.3% (1.0 pp) | 0.00909 | 0.01043 |
| Logistic | KAN12 | 96.4% (1.1 pp) | 93.5% (1.4 pp) | 0.00779 | 0.00743 |
| Nonlinear | Spline8 | 91.9% (1.5 pp) | 81.4% (1.9 pp) | 0.03826 | 0.03146 |
| Nonlinear | KAN12 | 68.9% (3.7 pp) | 55.2% (3.0 pp) | 0.02781 | 0.02864 |

The 90% intervals show the same qualitative distinction. Logistic in-range
coverage is 88.5%/91.4% (spline covariance/bootstrap) and 89.3%/88.0% (KAN).
For the nonlinear case it is 85.8%/73.8% for the spline and 63.5%/49.7% for
KAN. Near-nominal behavior on the affine control does not transfer to the
nonlinear response.

The paired nonlinear KAN-minus-spline in-range coverage differences are
-23.0 percentage points (MCSE 3.4 pp) for covariance intervals and
-26.2 pp (MCSE 2.5 pp) for the bootstrap. Both models saw the same datasets;
these are not comparisons between unpaired noise realizations.

Regional averages also hide substantial pointwise variation. At density
0.6 in the nonlinear KAN experiment, the nominal 95% covariance interval
covers in 56/100 datasets (95% Wilson MC limits 46.2--65.3%), while the
bootstrap covers in only 11/100 (6.3--18.6%). Across the eleven in-range
points, KAN coverage spans 56--91% for covariance and 11--88% for bootstrap.

![Empirical pointwise coverage with Wilson Monte Carlo limits](results/uncertainty-calibration-analysis/coverage.svg)

The error bars above quantify uncertainty in the measured coverage
proportions. They are not bands for the biological response. The green
range denotes exposure to the noiseless training trajectories, not an
identification certificate.

### Extrapolation is not automatically protected

Average nonlinear-case coverage for nominal 95% intervals:

| Model and method | Below training range | Above training range |
|---|---:|---:|
| Spline covariance | 66.5% | 99.3% |
| Spline bootstrap99 | 17.5% | 71.7% |
| KAN covariance | 24.0% | 70.7% |
| KAN bootstrap99 | 1.0% | 47.0% |

The high spline-covariance coverage above the range is accompanied by wide
intervals: mean width 0.2076, versus 0.1147 for KAN covariance intervals.
At density 2.35 the spline covariance interval's mean width is 0.3208.
Coverage alone does not imply informative inference, and favorable
extrapolation results for one truth do not generalize to another.

### Bias and smoothing behavior

Nonlinear in-range response RMSE, pooled across datasets and fixed queries,
is 0.02061 for KAN versus 0.01227 for the spline. At density 0.6, KAN's
mean signed response error is -0.02547, compared with mean full interval
widths 0.04239 (covariance) and 0.03077 (bootstrap). Substantial response
bias relative to interval width is visible; it is not a missing-replicate
artifact.

A **post-hoc descriptive diagnostic**, not an exclusion rule, finds two
well-separated KAN smoothing regimes. Twenty-nine nonlinear fits have EDF
1.99991--2.00014; the other 71 have EDF 4.323--5.052. An EDF split at 3 lies
inside the observed gap and identifies the near-affine subgroup. All these
fits advanced smoothing, and the largest original-fit stationarity
residual is only 0.001425.

The near-affine subgroup has 13.8%/12.5% in-range covariance/bootstrap
coverage. The other subgroup has 91.4%/72.6%. Removing the first group
would hide part of the specified procedure's failure, and the bootstrap
still undercovers in the remaining group. Small stationarity and iteration
stability do not diagnose response bias or establish coverage.

These observations identify a smoothing/initialization-sensitive inference
problem worth investigating; they do not by themselves prove an algebraic
covariance bug, identify the globally optimal fit, or justify replacing the
nominal level with a truth-tuned correction. No solver defaults, fitting
settings or interval endpoints were adjusted to improve these results.

### Finite bootstrap budget

On the predeclared twenty-dataset subset, increasing from 99 to 199
attempts changes nonlinear KAN in-range coverage by **+1.36 pp** (paired
MCSE 0.99 pp), and spline coverage by +0.91 pp (0.63 pp). The KAN's
199-attempt in-range coverage on that subset is 50.9%; below-range
coverage remains zero.

The subset must be compared with its own 99-attempt prefix, not with all
100 datasets. Its limited size also prevents a claim that bootstrap Monte
Carlo error is negligible. What it does show is that doubling this budget
does not resolve the observed nonlinear undercoverage.

## Practical conclusion and scope

The API now has a reproducible empirical assessment, not just algebra and
shape contracts. For this specified small KAN and fitting procedure:

- The affine control supports approximate nominal calibration.
- Nonlinear response bias and strongly smoothed fits can produce severe
  undercoverage, even at in-range points.
- Refitting smoothing in a percentile bootstrap does not automatically
  correct that bias.
- Extrapolation needs separate assessment; finite, narrow intervals are
  not evidence of identification.

These are conditional results for two truths, known initial states,
Gaussian observation errors, fixed initialization and a single-edge KAN.
They do not establish coverage for composed/multivariate KANs, correlated
or misspecified observations, architecture selection, or adaptive grids.
Addressing smoothing/initialization sensitivity and bias is more urgent
than presenting these nominal intervals as ready-made inferential guarantees.

## Retained evidence

Primary cohorts: [logistic spline](results/uncertainty-calibration-logistic-spline8/),
[logistic KAN](results/uncertainty-calibration-logistic-kan12/),
[nonlinear spline](results/uncertainty-calibration-nonlinear-spline8/) and
[nonlinear KAN](results/uncertainty-calibration-nonlinear-kan12/).
The [analysis directory](results/uncertainty-calibration-analysis/) contains
pointwise Wilson limits, region-level MCSEs, dataset-level scores, paired
comparisons, status accounting and the coverage figure. Observations and
all original/replicate coefficients and diagnostics remain in the cohorts.

The execution package source is
`3985b006b8798feb2d172c397f216094429af7ae51752178a3c0b15298e778ed`;
the calibration harness hash is
`949a7da1beca07bca93f7e6273190b96849979aa303d92cd8c6fe06b89bbb2c5`.
The [execution snapshot](results/source-uncertainty-calibration.tar.gz)
has SHA-256
`693d042118f3bdb2e9ab4dfdedd14b57fae81b6063d762f0097f407a72a1f9e1`.
The final read-only analysis is separately snapshotted with its own digest
and input fingerprints in the analysis directory. Prior KAN benchmark
archives have not been rewritten.

A post-study documentation-only edit removes an overbroad coverage claim
from the solver's internal covariance comment. The solver's parsed
expressions are unchanged; the executable fitting and interval calculations
used in the study have not been altered.
