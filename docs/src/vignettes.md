# Vignettes

The `vignettes/` directory contains 48 worked examples covering every solver, approximator, likelihood family, and `ecorecipes` workflow. Each vignette is a self-contained [Quarto](https://quarto.org/) document; rendered tutorials are linked below.

## Getting Started

| # | Vignette | Description |
|:--|:---------|:------------|
| 01 | [Getting Started](tutorials/01_getting_started.md) | Basic PSM workflow with logistic growth |
| 02 | [Likelihoods](tutorials/02_likelihoods.md) | Gaussian, Poisson, Negative Binomial, Truncated Normal, and custom likelihoods |

## Ecological Models

| # | Vignette | Description |
|:--|:---------|:------------|
| 03 | [Lotka–Volterra](tutorials/03_lotka_volterra.md) | Hare–lynx predator-prey with LAML and collocation |
| 04 | [Copepod](tutorials/04_copepod.md) | Wood (2001) 13-state fixed-duration stage-structured DDE with `:positive` SCOP-splines |
| 08 | [Rosenzweig–MacArthur](tutorials/08_rosenzweig_macarthur.md) | Recovering functional responses in consumer-resource dynamics |
| 10 | [Chemostat](tutorials/10_chemostat.md) | Microbial dynamics recovering unknown Monod growth kinetics |
| 11 | [Count Data SIR](tutorials/11_count_data_sir.md) | SIR model with Poisson and Negative Binomial likelihoods |
| 12 | [Discrete Time](tutorials/12_discrete_time.md) | Ricker, Beverton–Holt, and discrete competition models |
| 20 | [DDE](tutorials/20_dde.md) | Delay differential equations with unknown functions |
| 27 | [Predator-Prey Functional Response](tutorials/27_predator_prey.md) | Holling Type II with shape constraints, Bayesian and bootstrap CIs |
| 28 | [Fisheries](tutorials/28_fisheries.md) | Discrete stock-recruitment with Poisson counts and shape constraints |

## Approximators

| # | Vignette | Description |
|:--|:---------|:------------|
| 05 | [Neural Networks](tutorials/05_neural_networks.md) | Comparing B-spline, GP, and neural network approximators |
| 13 | [Shape Constraints](tutorials/13_shape_constraints.md) | Monotonicity, convexity, and zero-at-endpoint constraints |
| 16 | [COMONet](tutorials/16_comonet.md) | Shape-constrained neural network approximators |
| 26 | [SPDE](tutorials/26_spde.md) | Matérn SPDE approximator with shape constraints and profile range optimization |
| 41 | [Kolmogorov-Arnold Networks](tutorials/41_kan.md) | Fixed-grid KAN responses, spline comparison, edge diagnostics and pointwise uncertainty |

## Solvers

| # | Vignette | Description |
|:--|:---------|:------------|
| 06 | [Solver Comparison](tutorials/06_solver_comparison.md) | Side-by-side comparison of eleven solvers |
| 07 | [Probabilistic Fitting](tutorials/07_probabilistic_fitting.md) | Probabilistic ODE fitting with uncertainty quantification |
| 09 | [Gradient Matching](tutorials/09_gradient_matching.md) | Integration-free inference with adaptive gradient matching |
| 14 | [MCMC](tutorials/14_mcmc.md) | Full Bayesian inference with HMC/NUTS posterior sampling |
| 15 | [MAGI](tutorials/15_magi.md) | Manifold-constrained Gaussian process inference |
| 17 | [BNG](tutorials/17_bng.md) | Bayesian neural gradient matching |
| 18 | [Dalton](tutorials/18_dalton.md) | Data-adaptive Kalman likelihood fitting |
| 19 | [Pseudo-Marginal](tutorials/19_pseudo_marginal.md) | Probabilistic ODE + Bayesian MCMC |
| 21 | [GCV](tutorials/21_gcv.md) | Generalized Cross-Validation vs LAML smoothing |
| 22 | [Two-Stage](tutorials/22_two_stage.md) | Redirect — see Vignette 09 |
| 23 | [Derivative-Free](tutorials/23_derivative_free.md) | Nelder-Mead and Particle Swarm optimization |
| 24 | [Variational](tutorials/24_variational.md) | Fast approximate Bayesian inference via variational methods |
| 25 | [ABC](tutorials/25_abc.md) | Likelihood-free inference with ABC-SMC |
| 31 | [Integral Matching](tutorials/31_integral_matching.md) | Noise-robust integration-free estimation |
| 33 | [Ensemble Kalman](tutorials/33_ensemble_kalman.md) | Derivative-free estimation via ensemble Kalman inversion |
| 34 | [ODIN](tutorials/34_odin.md) | ODE-informed Gaussian process regression |
| 35 | [RKHS](tutorials/35_rkhs.md) | Reproducing kernel Hilbert space estimation |
| 36 | [Tensor Smooths](tutorials/36_tensor_smooths.md) | Bivariate interaction surfaces with `TensorBSplineApproximator` — predator interference |
| 37 | [Single-Index](tutorials/37_single_index.md) | Rank-1 multivariate responses with `SingleIndexApproximator`, and when the direction is identifiable |
| 38 | [Transformed Covariates](tutorials/38_transformed_covariates.md) | Lagged environmental drivers with `TransformedCovariateApproximator` |
| 39 | [Shape-Constrained GP](tutorials/39_shape_constrained_gp.md) | Monotone Gaussian processes with `ShapeConstrainedGPApproximator`, audited with `check_constraints` |
| 40 | [FGPGM](tutorials/40_fgpgm.md) | Bayesian gradient matching with `FGPGMSolver` (Wenk et al. 2019) |

## Uncertainty Quantification

| # | Vignette | Description |
|:--|:---------|:------------|
| 29 | [Bootstrap](tutorials/29_bootstrap.md) | Parametric and nonparametric bootstrap CIs (Gaussian and Poisson) |
| 30 | [Model Selection](tutorials/30_model_selection.md) | LAML marginal likelihood for knot, approximator, and structural selection |
| 32 | [Profile Likelihood](tutorials/32_profile_likelihood.md) | Identifiability analysis and likelihood-ratio CIs |
| 42 | [Resilience](tutorials/42_resilience.md) | Resilience of a fitted equilibrium with bootstrap intervals: a spline functional response vs a misspecified Type II (DynamicalResilience.jl) |

## Advanced & Ecosystem Workflows (v0.3)

| # | Vignette | Description |
|:--|:---------|:------------|
| 43 | Gause *Paramecium* Competition | Multi-replicate competition with `MultiTrajectoryPSMProblem`, `RandomEffectApproximator`, `:decreasing` SCOP-splines, and Wood (2013) `test_parametric_hypothesis` |
| 44 | *Tribolium* LPA Beetle Chaos | Discrete-time cannibalism with `ShapeConstrainedBSplineApproximator` (`:dec_positive`) and `to_discrete_problem` (`DynamicalResilience.jl`) |
| 45 | Seasonal Measles Forcing | Periodic transmission with `CyclicBSplineApproximator`, `TrajectoryObservation` interval incidence, `Distributions.NegativeBinomial`, and `compare_models` |
| 46 | Fisheries Surplus Production & Decision Analysis | Surplus-production biomass (`LogNormalLikelihood`) bridged to `DecisionAnalysis.jl` (`ManagedPSM`, `sample_belief`, EVPI & EVPPI) |
| 47 | GAM.jl & DynamicalResilience Ecosystem Bridge | Embedding `GAM.jl` smooths (`GAMSmoothApproximator`) inside ODEs with equilibrium resilience propagation |
| 48 | Stochastic Predator–Prey with Process Noise | Separating environmental stochasticity from observation error via `SDEPSMProblem` and `EKFSolver` |

## Running Vignettes Locally

Each vignette is a Quarto `.qmd` file that can be rendered locally:

```bash
cd vignettes/01_getting_started
quarto render 01_getting_started.qmd --to html
```

Requires [Quarto](https://quarto.org/) and a Julia installation with the package dependencies.
