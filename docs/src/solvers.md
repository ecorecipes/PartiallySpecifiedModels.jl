# Solvers

PartiallySpecifiedModels.jl provides 23 solvers for fitting partially specified models. Solvers are passed as the second argument to `solve`:

```@example solvers
using PartiallySpecifiedModels # hide
using PartiallySpecifiedModels: solve # hide
using OrdinaryDiffEq, Random # hide
Random.seed!(42) # hide
function growth!(du, u, p, t) # hide
    du[1] = p.r(u[1]) * u[1] # hide
end # hide
approx_r = BSplineApproximator(:r, (0.0, 12.0), 8; initial = x -> 0.3) # hide
r0, K, N0 = 0.5, 10.0, 0.5 # hide
true_sol = OrdinaryDiffEq.solve(ODEProblem((du,u,p,t) -> du[1] = r0*(1-u[1]/K)*u[1], [N0], (0.0,15.0)), Tsit5(); saveat=0.5) # hide
obs = max.([true_sol.u[i][1] + 0.3*randn() for i in 1:length(true_sol.t)], 0.01) # hide
prob = PSMProblem(growth!, [N0], (0.0,15.0), [approx_r]; data_times=collect(true_sol.t), data_values=reshape(obs,:,1), obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5()) # hide
sol = solve(prob, LAML())
println("Data loss: ", round(sol.data_loss, digits=4), ", EDF: ", round(sol.edf, digits=2))
```

## Penalized Likelihood Solvers

### LAML

Penalized Iteratively Reweighted Least Squares (P-IRLS) with **Laplace Approximate Marginal Likelihood** for automatic smoothing parameter selection. The default and recommended solver for B-spline approximators. For Gaussian data, LAML is equivalent to REML.

For non-Gaussian likelihoods the default smoothing criterion (`criterion=:working`) is PQL-flavored: smoothing parameters are calibrated on the Gaussian working model of the IRLS loop via a Pearson-dispersion-scaled Fellner–Schall update. The opt-in `criterion=:laplace` instead maximizes the actual family's full Laplace-approximate marginal likelihood, `ℓ(β̂) − ½β̂ᵀS_λβ̂ + ½log|S_λ|₊ − ½log|JᵀW̃J + S_λ| + (Mp/2)log 2π` (Wood 2011; Wood, Pya & Säfken 2016), using the generalized Fellner–Schall update of Wood & Fasiolo (2017) plus Newton refinement. Prefer `:laplace` for count data with low means (Poisson μ ≲ 10), where the working-model approximation is most biased. It supports `Poisson`, `NegativeBinomial` (dispersion fixed at the supplied `theta`), and `TruncatedNormal` (fixed `lower`/`sigma`); `CustomLikelihood` is rejected because it declares no normalized density or dispersion. For `Gaussian` data `:laplace` reduces exactly to the profiled-REML criterion, so results are identical to the default. `sol.convergence.criterion` records which criterion ran and `sol.convergence.laml` the criterion value at the fit.

`sol.convergence.converged` is a **stability** test — it fires when the penalized objective and the data loss stop changing — and a fit stops changing for several distinct reasons that it cannot tell apart: a genuine optimum, a search that stalled because a non-smooth model made the finite-difference Jacobian too noisy to build an accepted step from, and a fit whose smoothing parameters never moved off their initialization at all. LAML therefore also reports two **additive** diagnostics that separate those cases, present on every solution whether converged or not:

- `smoothing_advanced::Bool` — whether λ̂ ever moved off `initial_lambda`/the `1/tr(S)` default. `false` means smoothing selection never took effect and the reported λ̂ carries no information from the data. It compares λ̂ against its initialization with a 1e-10 relative tolerance (guarding against pure round-off, not a calibrated cutoff); it is `false` for 17 of the 163 LAML solves in this package's test suite, 8 of which nonetheless report `converged == true`.
- `stationarity::Float64` — the LAML criterion's gradient w.r.t. `ρ = log λ` at the returned fit, normalized to `maxₖ |∂V/∂ρₖ| / (½·rank(Sₖ))` so that it is dimensionless and `0` means exactly stationary. It is not bounded above (largest observed on this suite: 8.7).

There is deliberately **no `stationary::Bool`**: measured across all 163 LAML solves in the test suite the residual is an unbroken continuum, not two clusters — quantiles p25 = 1.6e-7, p50 = 8.5e-6, p75 = 1.6e-2, p90 = 0.29, and across the whole decision-relevant region (1e-3 to 3) the largest ratio between consecutive sorted values is 1.52 below 1 and 2.98 across the whole region — nothing resembling the orders-of-magnitude separation a threshold would need, so there is no gap anywhere a threshold could sit. A cutoff at 0.1 would have flagged 23 of 163 (14.1%). Read `stationarity` as a magnitude calibrated against your own problem class: a value orders of magnitude larger than comparable fits of yours is the signal. The two keys are complementary rather than redundant — of the 17 `smoothing_advanced == false` fits, one is an unpenalized model whose residual is `0.0` by convention and the other 16 span residuals from 0.016 to 8.7, so a fit can sit at an ordinary residual and still never have moved its λ̂. See the [`LAML`](@ref) docstring for the precise definitions, what each key does *not* mean, and the one regime — near-interpolating Gaussian fits, where the profiled σ̂² underflows and the residual becomes a ratio of two near-zero quantities — in which `stationarity` must not be read at all.

Neither `GCVSolver` nor `CollocationLAML` reports these yet, and neither can reuse LAML's residual, because neither optimizes the LAML criterion. `GCVSolver` minimizes a GCV (or, with `criterion=:ncv`, a neighbourhood-CV) score by grid plus golden-section search over `log λ`, so its analogue would be the derivative of that score — and `smoothing_advanced` would be near-vacuous there, since a golden-section search always moves λ. `CollocationLAML` runs its own inline Fellner–Schall update against the collocation criterion (which carries the extra ODE-compliance term and its own `S_full`), so its analogue is the gradient of *that* criterion, not of `laml_objective`. Both need their own residual derivation and their own calibration sweep.

Both `LAML` and `GCVSolver` (and `CollocationLAML`) accept `jac=:forwarddiff` to compute the working-model Jacobian of the predictions by forward-mode AD (`ForwardDiff`) straight through the ODE/map/DDE solve instead of the default adaptive central finite differences (`jac=:fd`): exact to solver precision and roughly an order of magnitude faster per Jacobian (one chunked Dual sweep replaces 2·p perturbed solves), with an automatic per-iteration fallback to finite differences if the Dual-valued solve fails. For `CollocationLAML` the option also covers the pointwise state Jacobian and the ∂F/∂β block of the collocation residual, preserving the failure-sentinel convention exactly (failed collocation points keep a large residual and a zero Jacobian; the sentinel is never differentiated).

The `:fd` path carries a **residual noise floor** that `:forwarddiff` does not: adaptive integrators re-select their accepted-step sequence under each parameter perturbation, so two solves at nearly identical parameters differ by an integration jitter far above machine precision (measured 1.5e-6 on a Tsit5 quadrature fixture at the default `reltol=abstol=1e-8`). The FD path measures that jitter once per Jacobian with a nudge solve, and a column whose central-difference signal falls below 10× the measured jitter — the qualitatively-garbage regime, where the noise floor used to produce entries wrong by half their scale — is recomputed at grown steps until its signal clears 1e4× the noise, each grown step validated against the previous column and guarded against growing into truncation. On an exact-reference fixture with O(1) sensitivities this repairs the FD Jacobian from 54%-of-scale error to 5.7e-5 of scale (vs 2.2e-6 for `:forwarddiff`). Noise-free problems such as discrete maps keep the historical tolerance-tied floor step and their ~1e-10 accuracy; but on *noisy* problems a column that measures above the 10× trigger keeps the historical floor step **and its historical noise floor** — the trigger's jitter estimate comes from a single base-point nudge solve, which can underestimate the realized per-column re-solve noise by ~10×, and on a one-coefficient variant of the reference fixture untriggered columns at measured SNR 12–18 retained errors of ~0.3–0.95 of column scale. These bounds are as-measured on these fixtures, not a guarantee. One exception by design: `GCVSolver` pins the historical fixed-step FD policy (no growth) inside its λ search, because its `search=:direct` vs `search=:reuse` equivalence contract requires a Jacobian that responds continuously to the coefficients, which a jitter-triggered step policy does not. The consequence is that `GCVSolver`'s `:fd` path retains the original noise floor on noisy problems in full; prefer `jac=:forwarddiff` there whenever the dynamics accept Dual numbers. Prefer `jac=:forwarddiff` when the dynamics are generic enough to accept Dual numbers — it is exact to solver precision and roughly 9–19× faster per Jacobian in this package's microbenchmarks (measured 14.1× on a spline fixture at p=8, 9.0× on a neural fixture at p=13, and 19.1× on a noisy fixture where the FD path pays signal-growth retries); `:fd` remains the default because it makes no genericity demands on user dynamics.

```@docs
LAML
```

### GCVSolver

Penalized IRLS with **Generalized Cross-Validation** for smoothing parameter selection. An alternative to LAML that minimizes leave-one-out prediction error. With `criterion=:ncv` it instead minimizes **neighbourhood cross-validation** (NCV; Wood 2024), which leaves out a temporal neighbourhood of `ncv_width` time steps around each point — the recommended criterion when residuals are short-range autocorrelated, where ordinary GCV undersmooths. With `search=:reuse` the whole λ-search is evaluated from a single whitening + eigendecomposition per IRLS iteration (the classical ddefit / Demmler–Reinsch trick, several-fold faster for larger bases) instead of one O(p³) solve per candidate λ — an exact reformulation of the default `search=:direct` scores, with automatic fallback to the direct path when `J'WJ` is near-singular; GCV criterion only.

```@docs
GCVSolver
```

### CollocationLAML

**Generalized profiling** (collocation) approach. Fits spline approximations to the state variables first, then optimizes the unknown function parameters to match the implied derivatives. Can be more robust than direct ODE fitting for stiff or chaotic systems.

```@docs
CollocationLAML
```

## Gradient Matching Solvers

These solvers avoid numerical ODE integration entirely by matching derivatives of smoothed data to the model equations.

### GradientMatching

Smooth the observed data, compute numerical derivatives, then fit the unknown functions to match. Requires good data coverage and low noise.

```@docs
GradientMatching
```

### TwoStageSolver

The simplest gradient matching approach (Varah 1982): smooth the data with splines, differentiate, then regress the unknown functions on the derivatives. A useful baseline for comparison.

```@docs
TwoStageSolver
```

### AdaptiveGradientMatching

GP-based gradient matching using the **product-of-experts** formulation of Dondelinger et al. (2013). Default mode is a fast MAP fit; with `n_samples > 0` it runs the paper's tempered **population MCMC**, jointly sampling latent states, parameters, and mismatch variances across a temperature ladder with exchange moves, returning cold-chain posterior draws in `convergence.beta_samples`.

```@docs
AdaptiveGradientMatching
```

### FGPGMSolver

**Fast GP-based gradient matching** (Wenk et al. 2019). One product-of-experts density over the latent states AND parameters jointly — data expert, GP prior, and ODE expert `N(f_k | D_k x̃_k, A_k + γI)` — sampled by single-chain adaptive Metropolis-within-Gibbs with the GP hyperparameters fixed beforehand by per-state marginal likelihood. Sits between [`AdaptiveGradientMatching`](@ref)'s population MCMC (cheaper: no temperature ladder, no γ sampling) and [`ODINSolver`](@ref)'s pure optimisation (unlike ODIN, it returns genuine posterior samples in `convergence.chains`). Gaussian likelihoods only.

```@docs
FGPGMSolver
```

### BNGSolver

**Ensemble Bayesian gradient matching** (Bonnaffé & Coulson 2023). Smooth-then-match under a variance-marginalized log-posterior, repeated over `k_obs` residual-bootstrap resamples × `k_proc` restarts; unknown functions are posterior-weighted ensemble means with pointwise uncertainty in `convergence.ensemble_std`. Fast (no ODE integration) and suitable for complex dynamics.

```@docs
BNGSolver
```

## Optimization-Based Solvers

### AdamSolver

Gradient-based optimization through the ODE solver using the **Adam** optimizer. This is the standard approach for Universal Differential Equations (UDEs). Works with all approximator types including neural networks.

By default, gradients are computed with ForwardDiff through the ODE solve, whose cost grows with the number of parameters. For continuous ODE problems you can opt into **adjoint sensitivities** instead by loading SciMLSensitivity and setting `sensealg`:

```julia
using SciMLSensitivity   # activates the adjoint extension
sol = solve(prob, AdamSolver(sensealg=:auto))                 # validated default
sol = solve(prob, AdamSolver(                # compiled tape: branch-free MLP
          sensealg=InterpolatingAdjoint(     # dynamics ONLY — silently wrong
              autojacvec=ReverseDiffVJP(true))))   # on spline evaluators
```

`sensealg=:auto` uses `InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))`, which re-tapes the dynamics at every step and is therefore safe for spline evaluators (their knot-interval lookup branches on the state). For branch-free dynamics — pure `Lux.Dense`/`tanh` MLP approximators — the compiled tape `ReverseDiffVJP(true)` is several times faster and becomes competitive with (and beyond ~1000 parameters faster than) ForwardDiff; for small parameter counts ForwardDiff generally remains fastest. `GaussAdjoint`/`QuadratureAdjoint` were observed to mis-differentiate spline-parameterized dynamics in testing — validate gradients against ForwardDiff before trusting them, and avoid `EnzymeVJP()` here (the closure-based dynamics wrapper is not Enzyme-compatible). The adjoint backend supports continuous ODE problems only (not discrete maps or DDEs) and the same loss families as the default path; `sol.convergence.backend` records which gradient backend ran.

```@docs
AdamSolver
```

### MultipleShootingSolver

**Multiple shooting** with Adam optimization. Divides the time span into segments and optimizes continuity constraints. More robust than single shooting for long time series or chaotic dynamics.

```@docs
MultipleShootingSolver
```

### DerivativeFreeSolver

Derivative-free optimization using **Nelder-Mead** or **Particle Swarm** methods. A robust fallback when gradient-based methods fail. Slower but more reliable for non-smooth or multimodal objectives.

```@docs
DerivativeFreeSolver
```

## Probabilistic Numerics Solvers

All three Kalman-filter-based solvers in this family (`RodeoSolver`,
`DaltonSolver`, `PseudoMarginalSolver`) accept an opt-in
`sqrt_filter=true` keyword that runs the filter/smoother recursions in
square-root (Cholesky/QR) form — the numerically canonical formulation of
modern probabilistic ODE solvers (Krämer & Hennig, JMLR 2024). Covariances
are propagated as triangular factors, so they stay positive semidefinite
by construction; prefer it at high `n_deriv` (≥ 5) or with very fine step
grids. The default (`false`) uses the standard covariance recursion
unchanged.

### RodeoSolver

**Probabilistic ODE solver** based on Kalman filtering/smoothing (Wu & Lysy 2024; the `:fenrir` likelihood variant follows Tronarp et al. 2022). Provides uncertainty estimates from the numerical integration itself, not just from parameter uncertainty.

```@docs
RodeoSolver
```

### DaltonSolver

**Data-adaptive likelihood** with transformed observations using Kalman filtering. Combines probabilistic ODE solving with a likelihood that adapts to the data structure.

```@docs
DaltonSolver
```

## Bayesian Solvers

### MCMCSolver

Full Bayesian inference using **Hamiltonian Monte Carlo (HMC)** or **No-U-Turn Sampler (NUTS)**. Provides posterior distributions over all parameters including unknown function coefficients.

```@docs
MCMCSolver
```

### MagiSolver

**Manifold-constrained Gaussian Process Inference**. Uses GPs to represent both state trajectories and unknown functions, with the ODE constraints enforced on a manifold. Handles partially observed systems well.

```@docs
MagiSolver
```

### PseudoMarginalSolver

**Pseudo-marginal MCMC** (Andrieu & Roberts 2009): a Metropolis-Hastings sampler that targets the exact posterior using an unbiased likelihood estimate, here computed with a probabilistic ODE solver so that numerical integration error is accounted for.

```@docs
PseudoMarginalSolver
```

### VariationalSolver

**Mean-field variational inference** for fast approximate Bayesian inference. Approximates the posterior with a factorized Gaussian distribution. Much faster than MCMC but may underestimate uncertainty.

```@docs
VariationalSolver
```

### ABCSolver

**Approximate Bayesian Computation with Sequential Monte Carlo** (ABC-SMC). Likelihood-free inference that only requires the ability to simulate from the model. Useful when the likelihood is intractable.

```@docs
ABCSolver
```

## Additional Solvers

### IntegralMatchingSolver

**Integral matching** (Dattner & Klaassen 2015): integrates both sides of the ODE and matches smoothed trajectory increments against the cumulative integral of the right-hand side. Integration-free and more noise-robust than derivative matching.

```@docs
IntegralMatchingSolver
```

### ProfileLikelihoodSolver

**Profile likelihood** for identifiability analysis and confidence intervals (Simpson & Maclaren 2023). Each unknown-function parameter is swept over a grid; the profile is taken through the penalized objective at the fitted smoothing parameters λ̂ — a penalized spline is not identified through its raw RSS — and the statistic ΔPenSS/σ̂² is referenced against χ²₁, with CI endpoints interpolated between grid points. Nuisance coefficients are re-optimised at each grid point by a long Nelder–Mead run. Gaussian likelihoods only.

```@docs
ProfileLikelihoodSolver
```

### EnsembleKalmanSolver

**Ensemble Kalman Inversion** (Iglesias et al. 2013): propagates an ensemble of parameter particles through the forward model and updates them with the Kalman gain. Derivative-free batch estimation.

```@docs
EnsembleKalmanSolver
```

### ODINSolver

**ODE-informed regression** (ODIN; Wenk et al. 2020): GP hyperparameters are estimated per state by marginal likelihood, then states and unknown-function parameters are optimised *jointly* under a Mahalanobis risk that weights the ODE mismatch by the GP's conditional derivative covariance. Supports partially observed systems (unobserved states are identified through the ODE terms).

```@docs
ODINSolver
```

### RKHSSolver

**Trajectory-RKHS estimation** (González et al. 2014): the state trajectory is represented in a time-kernel RKHS so its derivative is analytic, and fitting alternates a linear Gauss–Newton solve for the trajectory coefficients (data + RKHS norm + linearized ODE-gradient term) with gradient steps on the unknown-function parameters. Supports partially observed systems; no ODE integration.

```@docs
RKHSSolver
```

## Choosing a Solver

| Use case | Recommended solver |
|----------|-------------------|
| Default / first try | [`LAML`](@ref) |
| Automatic smoothing comparison | [`GCVSolver`](@ref) |
| Stiff or chaotic systems | [`CollocationLAML`](@ref) |
| Quick baseline | [`TwoStageSolver`](@ref) |
| Integration-free derivative matching | [`GradientMatching`](@ref) |
| GP-based gradient matching (MAP or MCMC) | [`AdaptiveGradientMatching`](@ref) |
| GP gradient matching with posterior samples, one chain | [`FGPGMSolver`](@ref) |
| Ensemble gradient matching with uncertainty | [`BNGSolver`](@ref) |
| Neural network approximators | [`AdamSolver`](@ref) |
| Robust neural fitting | [`MultipleShootingSolver`](@ref) |
| When gradients fail | [`DerivativeFreeSolver`](@ref) |
| Uncertainty quantification | [`MCMCSolver`](@ref) or [`MagiSolver`](@ref) |
| Exact posterior accounting for ODE-solver error | [`PseudoMarginalSolver`](@ref) |
| Fast approximate Bayesian | [`VariationalSolver`](@ref) |
| Intractable likelihood | [`ABCSolver`](@ref) |
| Probabilistic numerics | [`RodeoSolver`](@ref) or [`DaltonSolver`](@ref) |
| Noisy data, integration-free | [`IntegralMatchingSolver`](@ref) |
| Identifiability analysis | [`ProfileLikelihoodSolver`](@ref) |
| Derivative-free batch estimation | [`EnsembleKalmanSolver`](@ref) |
| GP-weighted gradient matching | [`ODINSolver`](@ref) |
| Trajectory-in-RKHS gradient matching | [`RKHSSolver`](@ref) |

## Missing and Masked Observations

An observation cell is **masked** — excluded from every objective, every
reported loss and every denominator — when either

- its value is non-finite (`data_values[i, j] = NaN`), or
- its weight is zero (`data_weights[i, j] = 0.0`).

Marking a cell **both** ways is the recommended convention, and the one every
example and test in this package follows:

```julia
data_values[3, 1]  = NaN    # the observation is missing
data_weights[3, 1] = 0.0    # ...and carries no weight
```

A zero weight alone is *not* sufficient protection against a `NaN` value:
IEEE arithmetic gives `0 * NaN = NaN`, so an ungated weighted sum turns the
entire objective — and every gradient derived from it — into `NaN`. That is
why the package gates on the cell predicate (positive weight **and** finite
value) rather than on the weight alone.

Masking is useful for ragged panels (states observed on different schedules),
dropped or censored samples, and hold-out cells for cross-validation, without
having to reshape `data_times`/`data_values`.

### Which solvers support masking

Masked data is supported by **all but four** solvers: [`LAML`](@ref),
[`GCVSolver`](@ref), [`CollocationLAML`](@ref), [`GradientMatching`](@ref),
[`AdaptiveGradientMatching`](@ref), [`FGPGMSolver`](@ref),
[`TwoStageSolver`](@ref),
[`BNGSolver`](@ref), [`ODINSolver`](@ref), [`RKHSSolver`](@ref),
[`IntegralMatchingSolver`](@ref), [`AdamSolver`](@ref),
[`MultipleShootingSolver`](@ref), [`DerivativeFreeSolver`](@ref),
[`MCMCSolver`](@ref), [`MagiSolver`](@ref), [`VariationalSolver`](@ref),
[`ABCSolver`](@ref) and [`ProfileLikelihoodSolver`](@ref).

Four solvers **reject masked data with an error**:

- [`RodeoSolver`](@ref)
- [`DaltonSolver`](@ref)
- [`PseudoMarginalSolver`](@ref)
- [`EnsembleKalmanSolver`](@ref)

Their data term lives inside a Kalman-filter or particle recursion that has no
per-cell mask. Honouring a mask there means skipping the *filter update*, not
merely the density term, for the masked cells — a structural change to those
recursions. Left unguarded they do not merely report `NaN`: `DaltonSolver`
flattens its objective to a constant, `PseudoMarginalSolver` rejects every
Metropolis–Hastings proposal, and `EnsembleKalmanSolver` turns every ensemble
member `NaN` on the first iteration — each of which returns the initialization
while *looking* like an ordinary converged fit. Failing loudly is the point.

To use one of these four with incomplete data, drop the masked rows from
`data_times`/`data_values` before constructing the `PSMProblem`.

### Diagnostics

[`residual_diagnostics`](@ref) and [`appraise`](@ref) exclude masked cells
from every statistic. Because a `PSMSolution` stores `data_values` but not
`data_weights`, they detect masking from the stored **value** only — one more
reason to mark a masked cell with `NaN` as well as with a zero weight.
