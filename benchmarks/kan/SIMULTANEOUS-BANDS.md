# Pointwise versus simultaneous function bands

This continuation makes the coverage target explicit. A pointwise interval
targets one function value; a simultaneous band targets all values on a
prespecified query grid in one dataset. Averaging pointwise coverage over
queries is not simultaneous coverage.

The distinction follows the Gaussian maximum-deviation construction used by
[`gratia::confint.gam(type="simultaneous")`](https://gavinsimpson.github.io/gratia/reference/confint.gam.html).
Smoothing-parameter uncertainty is a separate axis. `gratia` can request an
analytic smoothing-uncertainty-corrected covariance with `unconditional=TRUE`.
That covariance correction is **not implemented here** and is not being
relabelled as a bootstrap.

## API and constructions

```julia
# Existing default: pointwise conditional covariance intervals
point = confidence_band(sol, prob; uf_points=Dict(:r => x))

# Per-function Gaussian/delta band over these points
sim = confidence_band(sol, prob; uf_points=Dict(:r => x),
    interval=:simultaneous, nsim=10_000, rng=Random.Xoshiro(42))

# Reuse an actual full-refit bootstrap, without further refitting
bs = bootstrap(sol, prob, algorithm; nboot=200, uf_points=Dict(:r => x))
percentile = confidence_band(bs, sol, prob; interval=:pointwise)
boot_sim = confidence_band(bs, sol, prob; interval=:simultaneous)

# A smaller grid must be selected BEFORE its maximum is calibrated
regional = confidence_band(bs, sol, prob; interval=:simultaneous,
    uf_indices=Dict(:r => indices))
```

### Conditional covariance

Let `C` be the fitted parameter covariance and `J` the parameter Jacobian of
the response at the query points. Simulate centred Gaussian parameter
deviations, project them with `J`, and compute

```text
max_j abs(delta_f[j] / se[j])
```

for every draw. Its level-quantile is the simultaneous critical value;
the band is `fitted +/- critical*se`. The affine coefficients of the full
response are included. This is not a centred GAM term with an omitted
intercept. For nonlinear parameterizations it is a local delta-method band.

Covariance must be finite, symmetric and positive semidefinite. Only
factorization-scale negative roundoff is clamped. Factor-space norms avoid
quadratic-form cancellation; they agree with the pointwise standard errors
up to floating-point factorization error. The Gaussian critical value is
floored at the known marginal normal value so simulation noise cannot make
the simultaneous Gaussian band narrower through its critical value.

The covariance still conditions on the selected smoothing parameters.
There is no claim of an analytic unconditional correction or guaranteed
frequentist coverage.

### Full-refit bootstrap

The existing pointwise method remains the percentile interval. The new
simultaneous method instead uses

```text
max_j abs((f_boot[j] - fitted[j]) / bootstrap_sd[j])
```

with the original fitted curve as centre and the empirical bootstrap SD
at each point. The resulting symmetric maximum-deviation band is **not**
a simultaneous percentile interval, and is not bootstrap-t with a
different estimated standard error inside each replicate.

Whole bootstrap curves remain paired across coordinates. Every retained
refit must evaluate finitely at every selected point; otherwise this API
raises an error rather than silently removing its most extreme coordinates.
A zero-variance point must have exactly the original fitted value in every
draw. Unavailable bands stay unavailable in the assessment denominator.

This method includes whatever uncertainty the original refitting procedure
actually repeats. The retained bootstraps used below re-estimate smoothing
and coefficients in every replicate, including both stages of the
undersmoothing procedure. Fixed architecture, grids and specified priors
are not randomly selected.

Both empirical critical values use Julia's linearly interpolated quantile
(R type7), whereas the linked `gratia` source uses type8. The construction
is analogous, not a claim of bitwise equivalence to R.

## Scope and metadata

Each returned band is simultaneous over **one function's finite query
set**, not across every unknown function in a problem and not over a
continuum. Returned metadata identifies the construction, conditioning,
critical value, pointwise versus grid-wise scope and draw count.
Multivariate functions require physical query matrices; a rectangular
unobserved surface is not invented.

Existing `confidence_band(sol,prob)` defaults and `BootstrapResult` fields
are unchanged. Bootstrap-result bands use the stored grid/points, with
owned coordinate copies. They do not reuse a coefficient covariance while
claiming to integrate smoothing-selection uncertainty.

## Retained-fit assessment

Replay the [robustness campaign](COVERAGE-ROBUSTNESS.md): 480 original fits,
four conditions, two approximators, two fixed smoothing fractions and
30 independent datasets per condition. No optimizers are rerun. Original
and bootstrap coefficient vectors are reevaluated, and their archived
function values and original pointwise standard errors must agree.

For each fit, compare pointwise and simultaneous constructions using:

- The original query grid.
- A denser grid formed from 101 equally spaced locations plus every
  original query point.
- An in-range subset `[0.3,1.8]` and all queries `[0.05,2.35]`, calibrated
  separately.

Use 10,000 Gaussian simulations, nominal 90/95% levels and the first 99
attempted bootstrap refits. Identical Gaussian seeds across nested grids,
scopes and levels preserve the dependence needed for numerical nesting
comparisons. Bootstrap columns always retain their original replicate
identity.

For every dataset, retain both the fraction of covered points and the
indicator that **all** selected points are covered. Joint coverage is
summarized across independent datasets with Wilson Monte Carlo limits;
pointwise averages retain dataset-clustered uncertainty. Grid refinement
is a sensitivity check, not a proof of continuum coverage.

```bash
julia --threads=1 --project=benchmarks/kan --check-bounds=yes \
  benchmarks/kan/assess_simultaneous_bands.jl
python3 -B benchmarks/kan/analyze_simultaneous_bands.py
```

Existing output directories are rejected. Inputs remain immutable; the
new record includes band endpoints, critical values, scope, simulation
seeds, source/environment snapshots and input fingerprints.

## Results with the correct coverage target

All 480 original fits and their retained bootstrap draws replay without
refitting. The assessment contains 15,360 dataset/band combinations; all
are numerically available. The refined grid has 76 in-range points and
114 total points after adding the archived locations. The original grid
has 11 and 16 respectively.

For the quarter-strength KAN in the reference condition, nominal 95%
bands on the **76-point in-range grid** give:

| Construction | Average pointwise coverage | Whole-grid coverage | Mean width |
|---|---:|---:|---:|
| Pointwise conditional covariance intervals | 96.3% | 23/30 = 76.7% | 0.04955 |
| Simultaneous conditional Gaussian/delta band | 100.0% | 30/30 = 100.0% | 0.07334 |
| Pointwise full-refit percentile intervals | 91.9% | 17/30 = 56.7% | 0.04080 |
| Simultaneous full-refit maximum-deviation band | 98.9% | 28/30 = 93.3% | 0.06477 |

The median simultaneous critical values are 2.90 for conditional Gaussian
deviations and 2.99 for the bootstrap, versus the marginal normal critical
value 1.96. Wider bands are needed for the joint target. The 95% Wilson
Monte Carlo limits for 30/30 are 88.6--100%, and for 28/30 are 78.7--98.2%:
this is not proof of perfect or exact nominal calibration.

### Simultaneous construction is not a universal coverage correction

Whole-grid coverage of nominal 95% **simultaneous bands**, on the
76-point in-range grid, for quarter-strength procedures:

| Condition | Spline8 conditional covariance | Spline8 full-refit bootstrap | KAN12 conditional covariance | KAN12 full-refit bootstrap |
|---|---:|---:|---:|---:|
| Reference | 96.7% | 93.3% | 100.0% | 93.3% |
| Sparse observations | 66.7% | 60.0% | 90.0% | 83.3% |
| Double noise SD | 80.0% | 76.7% | 60.0% | 56.7% |
| Localized response | 3.3% | 3.3% | 86.7% | 73.3% |

These are dataset-level **all-points-covered** events, not averages of
pointwise containment. The low localized-spline joint coverage can coexist
with about 89% average pointwise coverage of its simultaneous bands:
missing a small part of the curve still fails the whole-grid event.

The noisy KAN failure also persists with the correct target and a full-refit
bootstrap. Its 18/30 conditional-Gaussian successes have Wilson limits
42.3--75.4%; its 17/30 bootstrap successes have limits 39.2--72.6%.
Reselecting smoothing and using a maximum statistic do not automatically
recover features absent from a strongly smoothed or weakly identified fit.

### Domain and grid choices matter

Extending the quarter-KAN target to all 114 points, including extrapolation,
changes simultaneous conditional-covariance/bootstrap joint coverage to:

- Reference: 96.7% / 80.0%.
- Sparse: 90.0% / 60.0%.
- Noisy: 56.7% / 43.3%.
- Localized: 86.7% / 80.0%.

The critical value is recalibrated for each scope. A wider-scope band can
therefore occasionally cover more datasets even though it protects more
locations; it is not the same band assessed on an expanded grid.

Grid refinement can also alter measured joint coverage. For example,
same-strength KAN bootstrap coverage in the reference condition changes
from 73.3% on the archived in-range grid to 60.0% on the denser grid.
The refined grids contain the archived points, and common simulated draws
ensure that the maximum-based critical values do not decrease. Changes
in coverage reflect the combination of more evaluation locations and a
recalibrated band, not a claim of continuous-domain certification.

## Interpretation

Pointwise and simultaneous intervals answer different questions. The
earlier pointwise coverage figures were not measurements of simultaneous
coverage. Proper simultaneous construction produces much stronger joint
protection in the reference setting, while difficult observation designs
and response shapes still expose substantial failures.

Conditional Gaussian bands and full-refit bootstrap bands also represent
different uncertainty constructions. The latter include the smoothing
selection actually repeated by the estimator; the former have no analytic
`unconditional=TRUE` covariance correction. This study does not establish
equivalence to that `gratia`/`mgcv` option.

The comparison is a reanalysis of retained calibration data, not an
independent new-data validation. No fit, smoothing multiplier, covariance
scale or critical value was selected using the known response to achieve
95% coverage. The source, grids, random seeds and empirical maximum rule
were declared before the full replay.

## Retained evidence

The [band assessment](results/simultaneous-band-assessment/) retains
per-dataset endpoints, standard errors, critical values, grid scopes,
Gaussian seeds, original-input fingerprints and source/environment snapshots.
The [independent analysis](results/simultaneous-band-analysis/) reconstructs
pointwise and all-points-covered events, retains unavailable-band
denominators, reports Wilson Monte Carlo limits and verifies grid/scope
nesting of the calibrated critical values.

The [execution source snapshot](results/source-simultaneous-bands.tar.gz)
has SHA-256
`1b2878ac8569dc179fe1ee023c4a00e0a93dbea12c0dc1fcccafd2b32ba572f0`.
The assessment harness hash is
`8598db8c1d68e90964857cf99942961f1d79f1d831d34817a45cc0926fcf6564`.
All prior fits and calibration archives remain unchanged.
