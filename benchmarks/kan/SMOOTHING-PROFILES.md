# Stable smoothing profiles and bias diagnostics

Implemented the diagnostic stage proposed after the
[analytic covariance correction](SMOOTHING-COVARIANCE.md): fully refitted
log-smoothing profiles, an explicit affine limit, representation-error
oracles, and density-wise bias/sampling-variance/noise-scale comparisons.
These are experimental benchmark tools, **not a change to LAML defaults
or a claim that calibrated intervals have been obtained**.

The main findings are distinct:

- Native coefficient refits and covariance recalculation at the same
  lambda barely change the results.
- The native smoothing criterion's numerical ridge materially affects
  the weak-curvature KAN direction, changing smoothing comparisons.
- Stable profile selection rescues seven of the twelve low-EDF noisy KAN
  fits, but some profiles genuinely prefer a nearly affine response.
- Representation error is negligible for the noisy nonlinear truth,
  but substantial for spline8 on the localized response.
- Better profile selection improves KAN coverage without restoring 95%;
  it can worsen spline coverage. Maximizing the criterion more faithfully
  is not itself a frequentist calibration procedure.

## Implementation and reproduction

`smoothing_profiles.jl` contains reusable, single-block Gaussian profile
refitting/scoring primitives. `assess_smoothing_profiles.jl` drives the
declared cohort, `analyze_smoothing_profiles.py` audits and decomposes its
results, and `plot_smoothing_profiles.jl` produces the figures.

```bash
julia --project=benchmarks/kan --check-bounds=yes \
  benchmarks/kan/assess_smoothing_profiles.jl \
  --output=benchmarks/kan/results/smoothing-profile-replay

python3 -B benchmarks/kan/analyze_smoothing_profiles.py \
  --input=benchmarks/kan/results/smoothing-profile-replay \
  --output=benchmarks/kan/results/smoothing-profile-replay-analysis

julia --project=vignettes --check-bounds=yes \
  benchmarks/kan/plot_smoothing_profiles.jl \
  --input=benchmarks/kan/results/smoothing-profile-replay \
  --analysis=benchmarks/kan/results/smoothing-profile-replay-analysis \
  --output=benchmarks/kan/results/smoothing-profile-replay-figures
```

Outputs refuse overwriting. Defaults read the retained robustness
observations; `--data=generated` explicitly generates observations instead,
so small runs and automated contracts do not require archived datasets.
`--cases`, `--models`, `--seeds`, `--rhos`, `--iterations`, `--tol`,
`--ode-tol` and `--output` are explicit options. Generated observations
are never silently substituted for missing archives.

All coefficient starts, failures, best coefficients, QR factors, penalty
frames, data, profile scores and numerical comparisons are retained.
The function-representation oracle is computed separately and never used
as a coefficient start or to select the ordinary profile maximum.

## Stable coordinates and a comparable affine limit

For the fixed-rank penalty decomposition
`S = U+ D U+'`, with orthonormal null-space basis `U0`, first whiten the
penalized coefficients. Then remove their affine function components
using a fixed 401-point design on the declared approximator domain:

```text
P = U+ D^(-1/2)
A = (Phi U0) \ (Phi P)
beta = [U0, (P - U0 A) exp(-rho/2)] theta
rho = log(lambda)
```

The affine shear has determinant one. It changes coordinates, not the
prior. The penalized sum becomes `sum(abs2, theta_penalized)`, and the
finite refits can reuse `LAML(fixed_lambda=1.0)` in transformed coordinates.
The recorded `rho`/`lambda`, not that internal unit penalty weight, is the
physical smoothing setting.

Plain whitening is insufficient: for a KAN linear-design example it
inflated the PCLS data-rank threshold enough to discard an informative
direction. Relative prediction disagreement with an independent QR solve
was 1.41e-5 before the shear, versus 2.49e-15 after it. The shear avoids
mistaking an artifact of the diagnostic parameterization for fit bias.

The profile determinant is calculated by column-equilibrated QR of
`[sqrt(W) J_theta; C]`, where `C` selects penalized coordinates. No numerical
ridge is added to identify null directions. The penalty determinant and
coordinate Jacobian cancel, giving the stable working criterion

```text
Q = RSS + sum(abs2, theta_penalized)
n_eff = n_used - nullity
V_profile = -(n_eff/2) log(Q/n_eff) - log|R|
```

At infinite lambda, refit the reduced `U0` model explicitly. Its criterion
has the **same within-model normalization** as the finite-lambda limit.
Across this cohort, scores at `rho=40` approach the explicit limit to
within 4.96e-8.

The effective penalty uses the existing fixed-rank eigenvalue convention.
Discarded eigenvalues and reconstruction discrepancies are recorded;
this is not an added affine penalty or a large covariance projection.
Absolute marginal criteria are not compared across spline and KAN
parameterizations.

This is a fully refitted **working-LAML** profile: coefficients and the
prediction Jacobian are recomputed at each lambda. It is not the exact
nonlinear marginal likelihood, a posterior density over log lambda,
or a likelihood-ratio confidence interval. In particular, a flat tail
cannot be turned into a proper posterior by silently choosing arbitrary
log-lambda integration bounds.

## Declared assessment

Use all 30 archived datasets in each of the **noisy** and **localized**
conditions, seeds 4001--4030, with spline8 and free-affine single-edge
KAN12. These are 60 distinct datasets and 120 native model fits.

The base log-lambda grid has 35 points: `-20,-18,-16`, every half-unit
from `-14` to `-2`, and `0,2,5,10,20,30,40`. Add the native selected value
and offsets of +/-0.1 and +/-0.2 for local curvature comparisons.
Each finite point has three coefficient paths: native-selected start,
ascending continuation from the cached initialization, and descending
continuation from the fitted affine limit. The limit itself has two starts.
Choose coefficients by the smallest **penalized objective**, not the
largest marginal criterion across imperfect coefficient fits.

This gives **14,640 coefficient attempts and 4,920 profile scores**.
All are finite/available; 49 coefficient attempts exhaust their iteration
budget and remain in the records. The largest normalized coefficient
decrement among selected profile fits is 9.19e-5; over all profile points
it is 1.63e-4. These are diagnostics, not a universal quality gate.

Native fits retain their 40-iteration, 1e-8 ODE-tolerance specification.
Diagnostic refits use 80 iterations, coefficient tolerance 1e-10, and
ODE tolerances 1e-10. The fixed-selected-lambda control distinguishes
extra optimization from changing smoothing. No new bootstrap was run.

Evaluate 76 in-range density points per fit and retain all six methods,
giving 54,720 function-query records. Conditional pointwise intervals
below use the reported residual/EDF dispersion; a separate known-scale
column replaces it with the generating variance. No simultaneous or
analytic-unconditional correction is applied in this diagnostic comparison.

## What changes the fits?

| Condition/model | Native low-EDF fits | Stable-profile low-EDF fits | Native low-EDF fits escaping that subgroup |
|---|---:|---:|---:|
| Noisy spline8 | 3/30 | 6/30 | 0 |
| Noisy KAN12 | 12/30 | 5/30 | 7 |
| Localized spline8 | 0/30 | 0/30 | 0 |
| Localized KAN12 | 0/30 | 0/30 | 0 |

EDF below three is the previously declared diagnostic subgroup, not a
new acceptance criterion. No fit is excluded because it falls there.

At the native selected lambda, the median **native-minus-stable
criterion difference on the same refitted coefficients** is:

| Condition | Spline8 | KAN12 |
|---|---:|---:|
| Noisy | -4.04e-8 | -0.249 |
| Localized | -3.19e-7 | -1.478 |

For localized KAN fits the range is -2.310 to -1.137 criterion units.
The numerical log-determinant ridge is consequential for the weak
curvature direction even at ordinary selected smoothing values. The
ridge used for reported covariance is smaller: replacing covariance at
the same fit changes coverage negligibly. The important numerical effect
here is on **smoothing selection**, not simply on interval scaling.

Large-lambda arithmetic also fails more dramatically. At `rho=40`, the
native raw quadratic penalty becomes negative and its profiled scale is
floored in all 120 evaluated cases. Stable penalties stay nonnegative and
converge to zero at the affine limit. For noisy KAN seed 4003, the native
formula on the `rho=40` fit reports a penalty near -2.87e5 and a criterion
near 1357, while the stable criterion is about 133.314. These artificial
tail scores are recorded, **never used to select the stable profile fit**.

The fully refitted profile need not have the same stationary point as
the frozen-Jacobian working criterion. For noisy spline seed 4003, the
native-lambda local gradient is about 5.0e-6, whereas the refitted-profile
gradient is about 0.081 (0.1 log-lambda difference step). Both numerical
conditioning and the working-model approximation matter.

![Fully refitted profiles, seed 4003](results/smoothing-profile-analysis-audited/figures/profiles-4003.svg)

Each scale criterion is centered at its own sampled maximum. Horizontal
lines show affine limits, not confidence cutoffs. Seed 4001's corresponding
[profile figure](results/smoothing-profile-analysis-audited/figures/profiles-4001.svg)
provides a less strongly smoothed noisy-data comparison.

## Bias, representation and coverage

The representation oracle is an unpenalized function-space projection on
801 uniformly spaced density points with trapezoidal weights, assessed on
an independent 1,601-point grid. It does not see noisy observations or
participate in fitting.

| Truth | Spline8 representation RMSE / maximum error | KAN12 representation RMSE / maximum error |
|---|---:|---:|
| Noisy-condition nonlinear response | 0.0000530 / 0.000110 | 0.0000303 / 0.0000598 |
| Localized response | 0.01742 / 0.04045 | 0.00460 / 0.00970 |

Representation is not the main obstacle for the noisy response. It is
already a significant obstacle for spline8 on the localized response.
The KAN design has rank nine on this restricted interval: its twelve
raw coefficients are not twelve independently observable function
directions there.

Nominal 95% **conditional pointwise** coverage, averaged over the same
in-range grid and datasets:

| Condition/model | Native | Stable profile-grid selection | Paired change / MCSE |
|---|---:|---:|---:|
| Noisy spline8 | 78.0% | 71.8% | -6.18 / 3.12 pp |
| Noisy KAN12 | 60.6% | 75.7% | +15.04 / 5.20 pp |
| Localized spline8 | 59.3% | 59.4% | +0.09 / 0.11 pp |
| Localized KAN12 | 73.6% | 80.3% | +6.71 / 1.32 pp |

These are not the previous ten-dataset analytic-correction figures,
not undersmoothed fits, and not simultaneous coverage. This is a reused
30-dataset diagnostic population, not fresh confirmation of a procedure.

For noisy KAN fits, RMS bias falls from 0.01683 to 0.01300, while RMS
sampling SD changes from 0.02013 to 0.01949. RMS reported conditional SE
is still smaller, increasing from 0.01272 to 0.01498. **Both bias and
unrepresented sampling variability remain** in this regime.
For localized KAN, RMS bias falls from 0.02557 to 0.02241, still much
larger than its representation floor.

Replacing only the native fit's noise scale with the known generating
scale barely helps: noisy KAN coverage is 60.5% instead of 60.6%.
Selecting lambda with the known-scale criterion and using known-scale
SEs gives 75.8% noisy KAN coverage and 80.9% localized KAN coverage.
Known variance is therefore a useful diagnostic, not a universal remedy.

![Bias and sampling variability](results/smoothing-profile-analysis-audited/figures/bias-decomposition.svg)

The gray region is +/-one empirical sampling SD, **not a confidence
band**. Bias and variance are calculated across datasets, not by treating
76 correlated query points as independent replications. Representation
error is subtracted as a function to obtain excess bias; the displayed RMS
magnitudes need not add, since the reporting and projection grids differ.

## Qualification and provenance

The current profile group has 104 assertions, including BigFloat
native-coordinate oracles, affine-limit normalization, KAN conditioning,
multiple starts, masking, ownership, generated-data archive schemas and
objective-overflow rejection. Existing fixed-smoothing and analytic
covariance contracts remain unchanged. Six Python profile-accounting
tests and four existing bias-decomposition tests pass.

BigFloat criterion discrepancies measured at most 7.11e-15, EDF at most
1.34e-15 and fitted linear responses at most 4.45e-16. Reversing the
affine shear and determinant term caused eight failures; replacing
sampling SD with RMS error caused two failures. Both reversals were
restored. A post-run guard also rejects finite coefficients/trajectories
with an infinite objective; none of the retained 14,640 attempts triggers it.

Raw evidence: `results/smoothing-profile-diagnostics/`.
Audited statistics and final figures:
`results/smoothing-profile-analysis-audited/`.
The stricter post-run audit reproduces the initial statistical CSVs
byte-for-byte. The numerical run, final input guard, and plotting layout
revisions are distinguished by their recorded script hashes.

- Package execution hash:
  `43941179726e2881921cdc5e930d00540e054028d87aec017ea71de0fdb43f72`.
- Execution-source archive `source-smoothing-profiles.tar.gz`:
  `634459d914fbf4f927cec679eec26e5d6a0d986f0f297165a72042c7ad8035df`.
- Executed profile primitives:
  `e553a7139d33451eef263c23c9e08d9a2e8aee2150c64df296743f57a6f94b7a`.
- Cohort harness:
  `33c5e9122a329a51fe0aa6d2209708eeacc572ad74b17bcca80916aec3ad04d2`.
- Julia 1.12.7, Darwin arm64, bounds checking enabled, one BLAS thread.

The evidence supports replacing numerically consequential smoothing
stabilization and addressing the remaining mode-selection/bias problem.
It does not support a blanket interval multiplier, an EDF exclusion rule,
or calling the profile-selected conditional bands calibrated.
