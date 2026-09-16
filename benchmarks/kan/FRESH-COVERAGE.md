# Fresh-data coverage confirmation and family screen

This follow-on to [the prior/method investigation](UNCERTAINTY-METHODS.md)
uses new noise realizations and recomputes all bootstrap refits. It separates
a higher-replication confirmation from a deliberately low-budget family
screen. No configuration is selected using its measured coverage.

**Outcome:** fresh data confirm that free-affine KAN covariance intervals
are much closer to the spline baseline than the earlier jointly penalized
configuration, but they are still not fully calibrated. On the nonlinear
truth, nominal 95% in-range coverage is **91.8% for KAN covariance and
77.5% for its recomputed percentile bootstrap**, versus 93.5% and 82.7%
for spline8. The exploratory family screen identifies a serious failure
of the specified composed-KAN fitting procedure and shows why coverage
must be read alongside interval width and smoothing diagnostics.

## Declared confirmation

Use the original logistic and nonlinear two-population problems:
`N'_j=r(N_j)N_j`, initial states `[0.2,0.7]`, observations at `0:0.25:5`,
and independent Gaussian noise with standard deviation 0.015. Domain,
sixteen physical query points and in-range/extrapolation definitions are
unchanged. Dataset seeds **2001--2100** are disjoint from the original
1001--1100 study.

Compare spline8 with the twelve-parameter single-edge KAN using
**`nullspace_penalty=0`**. The KAN retains its cached initialization seed42;
the spline retains its initial response 0.5. This isolates the earlier
prior change without adding a data-dependent multistart selection.

There are 100 datasets per case, shared across these two procedures. Each
uses `LAML(maxiters=40,jac=:forwarddiff)`, 99 attempted parametric bootstrap
refits, and nominal 90/95% pointwise intervals. Seeds 2001--2020 additionally
receive 199 attempts for the predeclared sensitivity comparison. Prefixes
are defined by attempted replicate identity, including failures.

All bootstraps are recomputed under the new procedure. Draws from the old
joint-affine-penalty study are not reused.

## Declared exploratory screen

Five additional configurations use the first 20 **nonlinear** datasets
and 19 attempted bootstrap refits each. The core spline8/KAN12 fits are
restricted to those same 20 datasets and their first 19 attempted refits
for the seven-model table. Comparing all 100 core datasets or 99 attempts
against these smaller cohorts would be an unmatched comparison.

The smaller budget was chosen after timing-only development pilots:
composed-network refits are much more expensive. Nineteen refits provide
less than one expected observation in a 2.5% tail. This is a screening
result for a finite-budget procedure, not a precise assessment of an
ideal infinite-bootstrap interval. With 20 datasets, even pointwise 95%
coverage has binomial MCSE about 4.9 percentage points.

| Model | Parameters | Declared penalty and initialization |
|---|---:|---|
| Spline8 | 8 | Natural cubic curvature, free affine part; initial 0.5 |
| KAN12 free | 12 | Complete-edge curvature, zero affine penalty; cached seed 42 |
| Spline12 | 12 | Capacity-matched natural cubic curvature, free affine part; initial 0.5 |
| SPDE12 | 12 | Native full-rank Matérn precision, nu 1.5 and fixed range 0.8; initial node values 0.5 |
| GP12 fixed | 12 | Nodal spline-style curvature penalty; Matérn-5/2 interpolant with fixed lengthscale 0.4 and variance 1; initial inducing values 0.5 |
| MLP13 | 13 | Widths 1-4-1, tanh; full-rank weight/bias ridge; seed 42 hidden features, zero output weights and bias 0.5 |
| Composed KAN28 free | 28 | Widths 1-2-1, cubic grid size 3; free affine components and separate edge-curvature blocks per layer; seed 42 hidden features, constant output 0.5 |

These priors are **explicit, not identical**. SPDE and spline12 share the
same cubic interpolation of node values but have different penalties.
The GP is the package's finite kernel-interpolation approximator, not full
continuous GP inference: its intervals do not add conditional GP process
variance between inducing points. Its initial interpolant is not exactly
constant between nodes, and actual initial query values are retained.

MLP weight ridge is not function-curvature regularization. Composed KAN
edge curvature is not the curvature of the complete response. Free
affine edge components also introduce identification issues that a
single-edge KAN does not have. No comparison of marginal criterion values
across these prior conventions is used to select a winner.

## Shared accounting

Every requested interval is retained, including unavailable intervals and
failed refits. Report availability, coverage conditional on availability,
available-and-covers yield, interval width and response RMSE. Monte Carlo
uncertainty clusters query points within independent datasets. All
initializations, fixed kernel settings, penalty ranks and coefficient
layouts are recorded.

The existing covariance, percentile, basic, normal-bootstrap and
bias-corrected-normal constructions are evaluated on the same retained
draws. These are pointwise methods, not BCa or bootstrap-t, and no
simultaneous-band or posterior-mixing claim is made. Poor stability or
smoothing diagnostics are reported rather than silently excluded.

The default workload is 500 original fits and up to 49,500 bootstrap refits:
47,600 for confirmation and 1,900 for the additional screening models.
These are not 50,000 independent coverage replications.

## Reproduction

```bash
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_families.jl \
  --stage=confirm --models=spline8 --output=benchmarks/kan/results/coverage-fresh-spline8
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_families.jl \
  --stage=confirm --models=kan12_free --output=benchmarks/kan/results/coverage-fresh-kan12-free
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_families.jl \
  --stage=screen --models=kan28_free --output=benchmarks/kan/results/coverage-screen-kan28
julia --threads=1 --project=benchmarks/kan --check-bounds=yes benchmarks/kan/calibrate_families.jl \
  --stage=screen --models=spline12,spde12,gp12_fixed,mlp13 \
  --output=benchmarks/kan/results/coverage-screen-baselines
python3 -B benchmarks/kan/analyze_fresh_coverage.py
```

The four cohorts can execute independently, with one Julia/BLAS thread
each and serial refits. Timings include compilation and possible machine
contention; they are not comparative performance benchmarks.

Outputs reject overwriting and retain observations, coefficients,
covariances, bootstrap evaluations, attempt identities, refit diagnostics,
design/source/environment snapshots and file fingerprints. The analyzer
checks matched observations and builds separate full-confirmation and
matched-screen summaries rather than pooling their replication counts.

## Fresh confirmation results

All 400 confirmation fits and 47,600 refits are finite, report iteration
stability, and advance smoothing. All requested intervals are available.
Those diagnostics describe computational completion, not coverage quality.

Average **in-range pointwise coverage of nominal 95% intervals** across
100 datasets per case:

| Interval method | Logistic spline8 | Logistic KAN12 free | Nonlinear spline8 | Nonlinear KAN12 free |
|---|---:|---:|---:|---:|
| Conditional covariance | 94.5% | 94.5% | 93.5% | 91.8% |
| Percentile bootstrap99 | 96.6% | 95.6% | 82.7% | 77.5% |
| Basic/reflected bootstrap99 | 94.8% | 92.7% | 85.2% | 85.1% |
| Normal bootstrap99 | 97.1% | 96.4% | 88.3% | 85.7% |
| Bias-corrected normal bootstrap99 | 95.1% | 93.6% | 87.0% | 86.5% |

The nonlinear KAN covariance figure has dataset-clustered MCSE 1.44
percentage points, versus 1.19 pp for spline8. Their paired coverage
difference is -1.73 pp (MCSE 0.59 pp). Percentile-bootstrap MCSEs are
1.69/1.59 pp, with paired KAN-minus-spline difference -5.27 pp
(MCSE 0.84 pp).

Mean nonlinear in-range covariance widths are 0.03695 (KAN) and 0.03803
(spline8); percentile widths are 0.03176 and 0.03111. Response RMSE is
0.01230 versus 0.01165. Thus the KAN covariance result is close to the
spline baseline, not a demonstration of nominal calibration or superiority.
The logistic control remains broadly consistent with the nominal level.

Only one of the 100 fresh nonlinear KAN fits has EDF below 3, compared
with 29 under the earlier nondefault affine penalty. The earlier
exploratory free-affine covariance result of about 91% therefore
generalizes to new noise realizations, while the newly recomputed
bootstrap still materially undercovers.

On the predeclared 20-dataset sensitivity subset, increasing from 99 to
199 attempts changes nonlinear KAN in-range percentile coverage by
+0.91 pp (paired MCSE 1.12 pp); spline coverage changes by -0.91 pp
(0.91 pp). This is a paired prefix comparison, not a comparison of the
20-dataset subset with all 100 datasets. More refits do not resolve the
observed shortfall in this experiment.

### Fresh extrapolation

Nonlinear KAN covariance coverage averages 48.5% below the training range
and 98.7% above it; percentile coverage is only 6.0% and 71.0%. The
corresponding spline figures are 67.5%/98.7% for covariance and
15.0%/72.3% for percentile. High above-range covariance coverage comes
with wider intervals: mean widths 0.1768 for KAN and 0.2025 for spline8.
Being inside the declared approximator domain is not identification.

## Matched seven-model screen

This table uses **the same 20 nonlinear datasets and 19 attempted refits
for every model**, including the restricted core fits. It must not be
compared directly with the 100-dataset/99-refit table above.

Nominal 95% in-range pointwise intervals:

| Configuration | Covariance coverage | Percentile coverage | Mean covariance width | Mean percentile width |
|---|---:|---:|---:|---:|
| Spline8 | 92.7% | 75.9% | 0.03967 | 0.02881 |
| KAN12 free | 92.7% | 70.9% | 0.03857 | 0.02886 |
| Spline12 | 94.1% | 76.4% | 0.04290 | 0.02997 |
| GP12 fixed interpolant | 95.5% | 79.5% | 0.04278 | 0.03058 |
| SPDE12 | 96.4% | 91.4% | 0.12074 | 0.09124 |
| MLP13 | 81.4% | 65.5% | 0.02414 | 0.02096 |
| Composed KAN28 free | 33.6% | 76.8% | 2.11564 | 53.19788 |

Dataset-clustered covariance/percentile MCSEs are respectively:
spline8 2.43/3.69 pp, KAN12 2.43/4.10 pp, spline12 2.21/3.81 pp,
GP12 1.92/3.78 pp, SPDE12 1.22/1.92 pp, MLP13 4.35/4.87 pp and
composed KAN28 10.13/4.50 pp. Pointwise Wilson limits are retained in
the raw summaries. Small differences in this screen are not a reliable
family ranking.

SPDE's stronger coverage comes with intervals roughly three times as
wide as the spline/KAN12 intervals. The GP's results concern its finite
interpolant and nodal curvature prior, not a full GP posterior. MLP
undercoverage is a result for this fixed architecture, ridge prior,
initialization and LAML budget, not neural approximators generally.

### Composed-KAN results are not useful calibrated inference

The composed configuration's mean widths are dominated by severe
instabilities, but medians also expose problems. Across datasets, the
median mean in-range covariance width is only 0.000573, while the maximum
is 41.82. For percentile intervals, the median is 6.90 and the maximum
259.96. A higher bootstrap coverage percentage obtained with such widths
is not a scientific advantage.

Four of 20 original composed-KAN fits never advanced smoothing. Of 380
bootstrap refits, 126 exhausted the iteration budget and 94 never advanced
smoothing. Median original-fit stationarity is 1.85, with maximum 7.23,
and median EDF is about `2.7e-5`. These fits remain in the reported
population, rather than being removed to improve coverage.

The specified composed-model fitting procedure is therefore poorly
qualified here. This screen does not establish whether more appropriate
initialization, identification constraints, penalties or optimization
would make that architecture reliable. All other screening configurations
advanced smoothing in every original fit/refit; MLP still exhausted the
budget in 2/20 original fits and 32/380 refits.

## Implications and retained evidence

The fresh confirmation supports the earlier prior diagnosis: use of the
API-default free affine component largely removes the anomalous
near-affine KAN subgroup, but **percentile-bootstrap bias and some
covariance undercoverage remain**. Bias-corrected normal intervals improve
the KAN percentile result, without reaching 95%. No interval correction
or configuration was selected using the observed coverage.

The broader results argue against a single KAN-versus-alternatives
coverage claim. Prior conventions, fitting quality, interval construction,
support and width all matter. In particular, the composed architecture
needs fitting/identification work before its interval percentages can be
treated as useful uncertainty assessment. Larger replicated family
comparisons, better-resolved bootstrap tails and appropriately qualified
studentization/bias correction remain separate investigations.

The subsequent [bias/undersmoothing study](BIAS-UNDERSMOOTHING.md) uses
these retained fits for diagnosis and new datasets for a fixed,
predeclared intervention. It finds a substantial coverage gain accompanied
by wider intervals, without changing the results reported here.

Primary archives: [fresh spline](results/coverage-fresh-spline8/),
[fresh KAN12](results/coverage-fresh-kan12-free/),
[screened baselines](results/coverage-screen-baselines/) and
[composed-KAN screen](results/coverage-screen-kan28/).
The [analysis directory](results/coverage-fresh-analysis/) contains separate
`confirmation/` and `matched_screen/` interval-level records, pointwise
Wilson limits, regional MCSEs, paired-model differences and status counts.
All 500 original fits and 49,500 refits are finite; this does not imply
good fitting, particularly for the composed network.

The [execution snapshot](results/source-fresh-coverage.tar.gz) has SHA-256
`6cfef624dee76c1d74b849df7c67f0c1a504385b8ca05808146b1a88e90d903d`.
The family harness hash is
`042ee86f4c99c6f36f861a52d2c09729f0794cbbb871f10b58137fdc9778ffc2`;
the generic calibration helper hash is
`d6be60a24629670713b0fe5f2d4091236807dcc48288576ce68449db3ddf4c84`.
Previous source snapshots, calibration datasets and results remain unchanged.
