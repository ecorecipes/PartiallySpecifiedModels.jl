# Migration Guide: v0.2 → v0.3

`PartiallySpecifiedModels.jl` v0.3.0 is a clean breaking release designed for
seamless interoperability across the `ecorecipes` ecosystem (`GAM.jl`,
`DynamicalResilience.jl`, `DecisionAnalysis.jl`), `Distributions.jl`, and
`StatsAPI.jl`, while upgrading to **Julia ≥ 1.13**.

## Breaking API Changes

### 1. Likelihood Type Renames (Eliminates `Distributions.jl` Collisions)

All built-in likelihood structs now carry the `*Likelihood` suffix so
`using PartiallySpecifiedModels, Distributions` works with **zero export
conflicts**:

| v0.2 | v0.3 | Also accepted via `DistributionsExt` |
|:---|:---|:---|
| `Gaussian()` | `GaussianLikelihood()` | `Normal(0, σ)` |
| `Poisson()` | `PoissonLikelihood()` | `Poisson()` |
| `NegativeBinomial(θ)` | `NegativeBinomialLikelihood(θ)` | `NegativeBinomial(r, p)` |
| `TruncatedNormal(lo, σ)` | `TruncatedNormalLikelihood(lo, σ)` | `truncated(Normal(0, σ), lo, Inf)` |
| — (new) | `GammaLikelihood(ϕ)` | `Gamma(α, θ)` |
| — (new) | `LogNormalLikelihood(σ)` | `LogNormal(0, σ)` |
| — (new) | `BinomialLikelihood(n)` | `Binomial(n, p)` |
| — (new) | `StudentTLikelihood(df, σ)` | `TDist(df)` / `LocationScale(0, σ, TDist(df))` |
| `CustomLikelihood(fn)` | `CustomLikelihood(fn)` *(unchanged)* | — |

When `Distributions` is loaded, pass a `Distribution` directly to `PSMProblem`:

```julia
using PartiallySpecifiedModels, Distributions
prob = PSMProblem(dynamics!, u0, tspan, [approx];
                  data_times=ts, data_values=Y, obs_to_state=[1],
                  likelihood=Normal(0.0, 0.3))
```

### 2. Function Renames & `StatsAPI` Alignment

| v0.2 | v0.3 | Reason |
|:---|:---|:---|
| `simulate(prob, beta)` | `simulate_psm(prob, beta)` | Avoids collision with `DynamicalResilience.simulate` & `DecisionAnalysis.simulate` |
| `appraise(sol)` | `psm_appraise(sol)` | Avoids collision with `GAM.appraise` |
| `parameter_draws(sol)` | `posterior_draws(sol)` | Clearer Bayesian/bootstrap semantics; now supports all Bayesian & bootstrap solutions |
| `predict(sol, prob, ...)` | Extends & re-exports `StatsAPI.predict` | Standard Julia statistics ecosystem interface |

In addition, v0.3 implements the standard `StatsAPI` model interface on
`PSMSolution`: `coef`, `vcov`, `dof`, `loglikelihood`, `aic`, `bic`, `nobs`,
`residuals`, `fitted`, `predict`, and `isconverged`.

### 3. Extension-Protocol Functions No Longer Exported

The low-level approximator extension functions are still public protocol API,
but are no longer exported at top level (preventing conflicts with `GAM.jl` and
`StatsAPI` packages):

```julia
# v0.3: import explicitly when implementing a custom AbstractApproximator
import PartiallySpecifiedModels: nparams, initial_params, penalty_matrix,
                                 penalty_blocks, build_evaluator, band_domain
```

### 4. `LAML` Default Improvements

- **`jac = :auto`** (was `:fd`): tries forward-mode AD (`ForwardDiff`) once on
  the first IRLS iteration and permanently latches to `:fd` if user dynamics are
  not Dual-compatible. Pass `LAML(jac=:fd)` to force finite differences.
- **`exploratory_maxiters = 5_000`** (was uncapped): protects trial steps
  against stiff/divergent ODE stalls while leaving the final evaluation and
  user-supplied `ode_kwargs[:maxiters]` untouched. Pass
  `exploratory_maxiters = nothing` to disable.
- **`fixed_lambda`** now accepts per-block vectors such as `[0.1, nothing]`
  (fixing block 1 while estimating block 2 via LAML).

### 5. Julia Version Requirement

`Project.toml` now requires **Julia ≥ 1.13** (enabling automatic resolution of
transitive unregistered `[sources]` dependencies across `ecorecipes`).

## Major New Features in v0.3.0

- **Periodic & anisotropic splines**: [`CyclicBSplineApproximator`](@ref) ($C^2$
  periodic splines for seasonal forcing) and
  `TensorBSplineApproximator(...; anisotropy=:estimate)` (Wood 2006 `t2`
  anisotropic tensor-product smooths with 3 smoothing parameters).
- **Observation operators & latent `u0`**: `obs_fn` (pointwise or
  [`TrajectoryObservation`](@ref) for interval incidence) and
  `estimate_u0=true` (or `estimate_u0=[1, 3]`) in [`PSMProblem`](@ref).
- **Four new likelihoods**: [`GammaLikelihood`](@ref),
  [`LogNormalLikelihood`](@ref), [`BinomialLikelihood`](@ref), and
  [`StudentTLikelihood`](@ref).
- **Multi-trajectory hierarchical PSMs**: [`MultiTrajectoryPSMProblem`](@ref) +
  [`RandomEffectApproximator`](@ref) for multi-replicate / multi-site experiments.
- **Process noise / SDE PSMs**: [`SDEPSMProblem`](@ref) + [`EKFSolver`](@ref)
  (continuous-discrete EKF with RTS smoother and joint process/observation noise
  estimation) + [`simulate_sde`](@ref) / [`ekf_loglik`](@ref).
- **Model comparison & hypothesis testing**: [`compare_models`](@ref)
  (AIC/AICc/BIC/Akaike weights/LAML) and [`test_parametric_hypothesis`](@ref)
  (Wood 2013 effective-rank Wald $\chi^2$ test against parametric null hypotheses).
- **`ecorecipes` bridges**:
  - `GAM.jl` extension (`GAMSmoothApproximator`: `cr`, `tp`, `cc`, `ps`, `bs`, `re`, `ds`, `fs`, `mrf`, `te`, `ti`, `t2`).
  - `DecisionAnalysis.jl` extension (`ManagedPSM`, `sample_belief` for EVPI/EVSI/adaptive management).
  - `DynamicalResilience.jl` direct dispatches (`steady_state(prob, sol)`, `linearise(prob, sol)`, `to_discrete_problem(prob, sol)`, broadened `posterior_draws`).
- **Opt-in deterministic multithreading** (`parallel=true`) on `MultiStart`,
  `ProfileLikelihoodSolver`, `optimize_spde_range`, `BNGSolver`, `ABCSolver`,
  and `MultiTrajectoryPSMProblem` (`LAML`).
