# Analytic smoothing-parameter covariance correction

The package now supports
`confidence_band(sol, prob; unconditional=true)` for **Gaussian LAML
local working models**, with either pointwise or finite-grid simultaneous
bands. Conditional bands remain the default.

The implementation includes **both** additions in Wood, Pya and Saefken
(2016), equation 7. It does not just add the covariance of the coefficient
mean, does not silently fall back to conditional covariance, and does not
claim nominal frequentist coverage.

## Formula, scale and scope

Let `rho = log(lambda)`, `A = J_data' W J_data`,
`B_k = lambda_k S_k`, `B = sum(B_k)`, and `H = A+B`. Freeze the final data
Jacobian and weights. Then the local coefficient sensitivity is

```text
u_k = B_k beta
D_k = -H^{-1} u_k
```

For an upper precision Cholesky `H_c = R'R`, let `T = R^{-1}`. With the
reported coefficient dispersion `phi` held fixed, the conditional
covariance is `phi T T'`. Define

```text
A_k = T' B_k T
E_k = upper(A_k), with its diagonal halved
T_k = -T E_k

C_mean = D V_rho D'
C_root = phi sum_{k,l} V_rho[k,l] T_k T_l'
C_corrected = C_conditional + C_mean + C_root
```

The root is the **inverse coefficient-precision Cholesky**, not the
Cholesky of a penalty, and not an arbitrary differentiated eigensystem
square root. The code forms the additions as sums of Gram products.
`smoothing_covariance_correction` returns each component and the already
scaled total; multiplying that total by `sigma2` again is wrong.

The automatic log-smoothing covariance uses the analytic negative
profiled-REML Hessian of the local working model. For
`Q = RSS + beta' B beta`, `q_k = beta' B_k beta`, and restricted residual
degrees of freedom `n_eff`,

```text
q_jk = delta_jk q_k - 2 u_j' H^{-1} u_k
t_jk = delta_jk tr(V_d B_k) - tr(V_d B_j V_d B_k)
K_jk = 0.5 n_eff (q_jk/Q - q_j q_k/Q^2) + 0.5 t_jk
V_rho = K^{-1}
```

Here `V_d` is the inverse precision used in the criterion's determinant.
Penalty blocks are disjoint, so their generalized log determinant is
linear in `rho` and contributes no second derivative. This is **not**
the existing `laml_hessian` approximation: that approximation omits
coefficient response and profiled-scale coupling.

The solver's conditional precision ridge and criterion determinant ridge
are retained, with their numerical values frozen for differentiation.
The coefficient optimum must be identifiable without a numerical ridge.
The correction preserves the original conditional covariance exactly;
its coefficient dispersion is `RSS/max(n_used-EDF,1)`, whereas the
profiled-REML scale in the smoothing Hessian is
`Q/(n_used-Mp_eff)`. These are deliberately different quantities.

This is analytic algebra for a local linear Gaussian model, not the full
higher derivatives of a nonlinear ODE likelihood. It holds dispersion,
GP kernel parameters, architecture, grids and other model choices fixed.
It does not correct smoothing bias or establish identifiability, local
normality or coefficient/smoothing stationarity. Those remain assumptions
to assess using the returned diagnostics and scientific context.

## Explicit regularity and fixed-smoothing policies

Default automatic correction requires a current Gaussian LAML fit,
advanced non-fixed smoothing, estimates away from optimization bounds,
identified coefficient information, and numerically positive log-smoothing
curvature. Unsupported requests and unavailable curvature raise errors.
No universal stationarity cutoff is imposed; coefficient score/step,
log-smoothing gradient, original stationarity and convergence flags are
exposed rather than replaced by a misleading quality flag.

`rho_regularization` is a default-zero, explicitly chosen isotropic
covariance-only precision. It shifts the Hessian before inversion, not the
fitted smoothing mean. Unlike mgcv's standard postprocessor, this
implementation uses **one** log-smoothing covariance for both terms, not
a positive-eigenspace mean correction and a separately regularized root
term. The public option therefore has the same purpose as
`gratia::confint.gam(unconditional=TRUE)`, not identical implementation
details or coverage.

An explicit finite PSD `rho_covariance` also supports fixed/external
smoothing. For a predeclared multiplier, adding `log(fraction)` to the
selected log lambda leaves its covariance unchanged. Supply that
selection-stage covariance to the final refit; coefficient and root
sensitivities are evaluated at the **final** coefficients and lambda.
The API never infers a selection covariance from an intentionally
nonstationary fixed-lambda fit. Other uncertainty in a two-stage estimator
is not thereby guaranteed to be captured.

## Independent numerical contracts

The focused correction group has 164 assertions, including signed
coefficient derivatives, the zero-coefficient case with a nonzero root
term, correlated/noncommuting multi-penalty examples, coefficient-coordinate
and dispersion scaling, masking, offsets, unsupported cases, optimization bounds,
no re-simulation/mutation, and both band targets.

Measured algebraic discrepancies:

| Independent comparison | Maximum absolute discrepancy |
|---|---:|
| Scalar inverse-Cholesky derivative | 1.39e-17 |
| Scalar mean/root covariance additions | 0 |
| Coefficient derivative versus AD of the exact linear estimator | 2.78e-17 |
| Profile Hessian versus AD of the profiled linear objective | 1.89e-15 |
| Profile gradient versus AD | 4.89e-15 |
| Root derivatives versus independently perturbed Cholesky factors | 5.79e-11 |
| Linear-ODE coefficient versus ridge-regression oracle | 1.89e-11 |

Controlled reversal of the sensitivity sign, full profile Hessian,
root correction and band covariance selection produced **19 failures**
in the then-156-assertion group. All changes were restored; the expanded
group and related conditional-band, KAN, fixed-smoothing, smoothing-step
and legacy uncertainty contracts pass. These are numerical/algebraic
contracts, not theorem-prover certificates.

The rendered [vignette](../../vignettes/41_kan/41_kan.md) exercises the public
pointwise and simultaneous options. Its mean pointwise SE increased by
5.84%, with the fitted function unchanged. Coefficient-space covariance
traces depend on parameterization and can be very large for correlated
KAN base/spline coefficients; use projected function uncertainty, not
cross-model comparisons of those raw traces.

## Retained-data assessment

Run from the repository root:

```bash
julia --project=benchmarks/kan --check-bounds=yes \
  benchmarks/kan/assess_smoothing_covariance.jl \
  --output=benchmarks/kan/results/analytic-smoothing-covariance-new
```

The retained default run uses the **first ten archived datasets per
condition**, seeds 4001--4010: 40 distinct datasets, each fitted with
spline8 and free-affine KAN12. It performs 80 automatic LAML fits and
80 warm-started quarter-lambda refits, with 40 iterations per fit.
It creates no new bootstrap replicates and does not modify old archives.

This is an implementation/availability investigation on **reused data**,
not fresh coverage confirmation. Each function is evaluated on the
76-point refined in-range grid; simultaneous calibration uses 10,000
Gaussian draws. Nominal level is 95%, with no curvature regularization.
All 160 model fits are finite. The analytic correction is available for
79/80 automatic fits and 79/80 quarter refits; 316/320 analytic bands and
320/320 conditional bands are available. Missing analytic bands remain
in the population, rather than being quietly excluded.

For KAN12, all ten datasets in each panel have both intervals available:

| Condition | Automatic pointwise: conditional -> corrected | Automatic simultaneous whole-grid: conditional -> corrected | Mean paired pointwise width increase |
|---|---:|---:|---:|
| Reference | 79.7% -> 85.9% | 8/10 -> 9/10 | 8.3% |
| Sparse | 75.5% -> 79.1% | 6/10 -> 6/10 | 23.4% |
| Noisy | 70.8% -> 73.2% | 7/10 -> 7/10 | 14.8% |
| Localized | 69.9% -> 77.6% | 2/10 -> 6/10 | 15.3% |

The dataset-paired pointwise coverage gains are 6.18, 3.55, 2.37 and
7.76 percentage points, with dataset-level MCSEs 2.10, 1.26, 1.19 and
2.37 points respectively. These small, reused panels are not substitutes
for the earlier 30/50/100-dataset populations.

After the quarter-lambda refit with explicitly propagated selection
covariance:

| Condition | KAN pointwise: conditional -> corrected | KAN simultaneous whole-grid: conditional -> corrected |
|---|---:|---:|
| Reference | 89.5% -> 90.5% | 10/10 -> 10/10 |
| Sparse | 84.1% -> 89.2% | 8/10 -> 8/10 |
| Noisy | 72.8% -> 73.8% | 7/10 -> 7/10 |
| Localized | 90.4% -> 92.0% | 7/10 -> 9/10 |

The difficult noisy regime remains undercovered. Analytic smoothing
uncertainty is not a universal bias/coverage repair.

There are also important warnings from the spline baseline. Sparse
seed 4001 has log-smoothing curvature about **-1.21e-4**, so automatic
correction is unavailable and its quarter refit cannot inherit a covariance.
Another sparse spline fit has log-smoothing variance about **16052**;
its pointwise band expands about **30-fold** (39-fold at quarter lambda).
A finite matrix plainly does not establish that a local Gaussian
approximation is informative. Neither case was silently clipped,
regularized, replaced or dropped.

Raw designs, fit/covariance components, dataset scores, summaries and input
fingerprints are under `results/analytic-smoothing-covariance/`.
The `status` in `fits.csv` describes correction availability, not fit
failure. Summary coverage yields retain unavailable bands in the
denominator; available-only widths must not be compared across unmatched
populations.

## Sources and execution provenance

- Wood, Pya and Saefken (2016), section 4, equation 7, and supplementary
  Appendix D: [arXiv 1511.03864v2](https://arxiv.org/abs/1511.03864v2);
  [author-hosted paper](https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf).
- Software comparison: `gam.fit3.r`, `Vb.corr` and `gam.fit3.post.proc`,
  at [mgcv commit 1b6a4c8374612da27e36420b4459e93acb183f2d](https://github.com/cran/mgcv/blob/1b6a4c8374612da27e36420b4459e93acb183f2d/R/gam.fit3.r).
  The magnitude-comparison appendix's coefficient sign is not copied:
  differentiating `H beta = b` independently gives the negative sign
  checked above.
- Julia 1.12.7, Darwin arm64, bounds checking enabled, one BLAS thread.
- Execution source SHA-256:
  `43941179726e2881921cdc5e930d00540e054028d87aec017ea71de0fdb43f72`.
- `source-smoothing-covariance.tar.gz` SHA-256:
  `ae4b8a6cd21040f6898ace5eca4b7a661e3840aa33ef67d0a62b6216aa42b742`.
- Assessment harness SHA-256:
  `ffb1e5965b8f6b783ca951ba0c77db87930d07cb2661b062a91edb631ff51619`.
- Original arXiv source archive SHA-256:
  `0e3086a1980add7c61379279380044d0ab2cbb58344014b0d8062dc1741e3de9`.
- Pinned TeX SHA-256:
  `173064c159822f3a6f7423341e9f5a2b2435165595b9561642e7fab79fde2826`.
