# Comparing interval constructions and initialization effects

The [first coverage study](UNCERTAINTY-CALIBRATION.md) compared two
approximators and two interval constructions under one LAML fitting
procedure. This follow-on separates two questions: whether the interval
construction explains undercoverage, and whether initialization explains
the nonlinear KAN's near-two-EDF fits.

**Main finding:** changing interval construction helps, but the larger
effect comes from the example's affine-prior specification. The original
nonlinear KAN covariance coverage of 68.9% rises to 91.0% when the affine
component is left unpenalized, as in the API default. This is an exploratory
result on reused datasets, not an independently validated remedy or a new
bootstrap calibration.

## Interval comparison

Reuse the original observations, fits and bootstrap draws. No models are
refitted for this comparison, and no interval setting is selected using
the true response. Retain 90%/95% levels, the 99-attempt primary budget,
the preselected 199-attempt sensitivity subset, physical query regions and
dataset-clustered Monte Carlo accounting.

For original estimate `t`, bootstrap replicates `t*`, empirical quantiles
`q`, bootstrap mean `m*`, sample standard deviation `s*` and normal
critical value `z`, compare:

| Method | Interval |
|---|---|
| Conditional covariance | Existing LAML delta-method interval |
| Percentile bootstrap | `[q_low, q_high]` |
| Basic/reflected bootstrap | `[2t-q_high, 2t-q_low]` |
| Normal bootstrap | `t +/- z*s*` |
| Bias-corrected normal bootstrap | `(2t-m*) +/- z*s*` |

“Bias-corrected normal” uses the bootstrap-estimated bias, not the known
truth. It is **not BCa**: there is no acceleration estimate. Likewise the
normal bootstrap is not bootstrap-t studentization. Those methods need
additional fitted quantities and cannot be reconstructed honestly from
these archived draws alone.

The existing covariance and percentile intervals remain unchanged. New
methods require a finite original estimate and at least three finite
replicates. Failed intervals remain in the requested population. Every
bootstrap budget refers to attempted replicate IDs, not a count of retained
successful columns. Original-estimator bias/RMSE remain separate from any
adjustment to an interval's centre.

## Initialization and prior diagnosis

For each of the 200 original KAN datasets, compare the original cached
initialization with a fixed constant response of 0.5. The latter is the
same initial response used by the spline baseline, not a value fitted to
truth. Cross these two starts with the originally declared
`nullspace_penalty=1e-6` and the API default `nullspace_penalty=0`.
Architecture, grids, curvature penalty, likelihood, observations and LAML
iteration settings remain unchanged. The affine penalty is a prior choice,
not merely a numerical stabilization.

Both starts use the same benchmark initialization wrapper; original-start
response and standard-error replay must agree with the retained calibration.
Interval reporting uses the underlying KAN's parameter-AD path. This
protects against accidentally comparing wrapper finite differences with
KAN AD or changing the penalty dispatch along with the initialization.

Select between starts **within each prior** using only the maximum finite
reported LAML criterion on training observations. Do not rank the proper
and intrinsic-prior criterion values against one another: their
normalizations and unpenalized dimensions differ. Preserve all four candidates and full
search cost. The comparison is exploratory, motivated by the completed
calibration; it is not independent validation of a newly calibrated
procedure. Existing bootstrap draws must not be attributed to changed
initializations. This stage computes only fresh covariance intervals.

This extension follows a diagnostic probe, not a change to the original
coverage experiment: a previously near-affine fit had *higher* LAML
evidence than its more flexible constant-start counterpart. Removing the
jointly scaled affine penalty changed that smoothing regime. The full
factorial replay investigates the association without selecting on truth.

## Reproduction

```bash
python3 -B benchmarks/kan/compare_uncertainty_methods.py
julia --threads=1 --project=benchmarks/kan --check-bounds=yes \
  benchmarks/kan/diagnose_uncertainty_initialization.jl
```

Both programs reject existing output directories. Original calibration
archives remain immutable. New interval-level records, paired Monte Carlo
summaries, candidate fits, replay discrepancies, coefficients, source
snapshots and input fingerprints are retained separately.

## Five interval methods on the same original fits

The comparison preserves all original fits and draws and evaluates 74,240
intervals, including both levels and the sensitivity subset. Average
in-range pointwise coverage of **nominal 95% intervals** is:

| Interval method | Logistic spline8 | Logistic KAN12 | Nonlinear spline8 | Nonlinear KAN12 |
|---|---:|---:|---:|---:|
| Conditional covariance | 94.3% | 96.4% | 91.9% | 68.9% |
| Percentile bootstrap | 96.3% | 93.5% | 81.4% | 55.2% |
| Basic/reflected bootstrap | 94.3% | 94.4% | 82.3% | 62.5% |
| Normal bootstrap | 96.7% | 95.5% | 85.5% | 66.0% |
| Bias-corrected normal bootstrap | 94.3% | 94.9% | 84.3% | 65.8% |

These are five **interval constructions**, not five fitting algorithms or
five approximator families. All fits here use LAML, and the KAN retains the
original `nullspace_penalty=1e-6` configuration.

For nonlinear KAN, basic intervals improve coverage over percentile by
7.27 percentage points (paired dataset-clustered MCSE 1.75 pp); normal
intervals improve it by 10.82 pp (1.18 pp), and bias-corrected normal
intervals by 10.64 pp (1.68 pp). None approaches nominal 95% coverage.
The mean widths are respectively 0.02864, 0.03246 and 0.03246, versus
0.02864 for percentile and 0.02781 for covariance intervals.

Reflection leaves the percentile interval's width unchanged but changes
its location. The normal methods widen it on average. Neither operation
can be assumed to eliminate smoothing bias or uncertainty about the fitted
response regime.

Extrapolation remains difficult. Below the nonlinear training range, KAN
coverage is 1% for percentile, 38% for basic, 18.5% for normal, 41.5% for
bias-corrected normal and 24% for covariance. Above the range the respective
figures are 47.0%, 65.3%, 67.3%, 68.3% and 70.7%. Improved coverage from
one construction is not a general extrapolation guarantee.

## The prior matters more than initialization alone

The two-by-two diagnosis completed 800 fits and 38,400 covariance intervals.
Original-start fitted responses and standard errors replayed **exactly**
for all 200 archived KAN datasets. Every diagnostic fit was finite.

Nonlinear-case results, nominal 95% covariance intervals:

| Affine penalty | Initialization/selection | Fits with EDF below 3 | In-range coverage | Mean width |
|---|---|---:|---:|---:|
| Jointly scaled `1e-6` | Original cached start | 29/100 | 68.9% | 0.02781 |
| Jointly scaled `1e-6` | Constant response 0.5 | 5/100 | 82.5% | 0.03202 |
| Jointly scaled `1e-6` | Best LAML of the two starts | 29/100 | 68.9% | 0.02781 |
| `0` (unpenalized affine part) | Original cached start | 0/100 | 91.0% | 0.03727 |
| `0` (unpenalized affine part) | Constant response 0.5 | 0/100 | 91.0% | 0.03727 |
| `0` (unpenalized affine part) | Best LAML of the two starts | 0/100 | 91.0% | 0.03727 |

In-range MCSE is 3.71 pp for the original joint-penalty procedure, 2.34 pp
for its constant-start variant, and 1.48 pp for the free-affine variants.
The free-affine original-start response RMSE is 0.01256, versus 0.02061
under the original joint-penalty procedure; the spline baseline is 0.01227.
Logistic-control coverage remains close to nominal under free-affine KAN,
at about 94.6%.

This is **not simply a worse local fit selected by a broken convergence
flag**. Of the 29 originally near-affine nonlinear fits, a constant start
finds a more flexible solution in 24, but those solutions have *lower*
reported LAML evidence. Their differences are material: the median
constant-minus-original criterion difference across the original
near-affine subgroup is -6.50. Selecting the higher criterion within that
same prior therefore retains the original poor-coverage subgroup.

### Why a small affine penalty is not harmless

For these twelve-parameter KANs, the measured penalty rank changes from
10 with `nullspace_penalty=0` to 12 with `1e-6`. The latter is a proper
prior on the affine coefficients as well as the curved component, and
the **same estimated smoothing parameter scales both**.

LAML contains the normalization term

$$
\tfrac12\log|S^\lambda|_+
= \tfrac12\,\operatorname{rank}(S)\log\lambda
  +\tfrac12\log|S|_+.
$$

Once the two added directions are counted as positive, they add a
`log(lambda)` term to this normalization, regardless of the small numeric
size of their eigenvalues. The Gaussian restricted-dof calculation also
changes because the unpenalized dimension changes from two to zero.
The quadratic penalty and penalized information change too.

Thus `1e-6` is a substantive prior specification, not mere numerical
jitter. The package default is already **zero**; the first coverage study
assessed a nondefault, explicitly declared prior from the example.
Raw criterion values from these two prior conventions must not be ranked
against each other as if their normalizations were identical.

This mechanism concerns automatic LAML smoothing selection. It is not a
demonstrated reranking of the earlier fixed-penalty Adam comparisons.

## What remains unresolved

The free-affine covariance result is much closer to the spline baseline,
but 91% is not 95%, and the diagnostic reused the original calibration
datasets. Extrapolation is still uneven: free-affine KAN covariance
coverage is 54.5% below the nonlinear training range and 97.7% above it,
with mean widths 0.1990 and 0.1800.

This diagnostic did not measure bootstrap coverage for the changed prior
or initialization: reusing old bootstrap draws would evaluate the wrong
procedure. The [fresh-data continuation](FRESH-COVERAGE.md) now recomputes
those refits and adds a separately labelled GP/SPDE/MLP/composed-KAN screen.
BCa, bootstrap-t and posterior-sampling coverage still require additional
work and uncertainty quantities.

The next defensible comparison is a fresh-dataset calibration with explicit,
comparable prior/smoothing conventions and recomputed bootstrap refits,
followed by additional approximator families. Neither a truth-tuned
interval correction nor silently discarding strongly smoothed fits is an
acceptable substitute.

## Evidence

The [five-method archive](results/uncertainty-method-comparison/) contains
every derived interval, pointwise Wilson limits, regional MCSEs, paired
method differences, formulas and input fingerprints. The
[factorial fitting archive](results/uncertainty-initialization/) retains
all candidates, selected starts, coefficients, original replay differences,
the declared two-by-two design and its execution snapshot. Its
[independent summaries](results/uncertainty-initialization-analysis/) keep
the two prior conventions separate.

The [source archive](results/source-uncertainty-methods.tar.gz) has SHA-256
`292b649d70880ec7da15e8a6de581f64eb392cfbf5d2f83331fd65a3b95bdad6`;
the diagnostic harness hash is
`657ddcb446e498561590684abab7d1ab5e9ed3276c1c53c01518b3d14522c808`.
All original calibration data and results remain unchanged.
