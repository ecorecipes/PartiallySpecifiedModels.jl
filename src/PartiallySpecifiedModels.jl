"""
    PartiallySpecifiedModels

A Julia package for fitting partially specified models (PSMs) — dynamical
systems (ODEs, DDEs, discrete maps, and continuous-discrete SDEs) where one
or more functional responses are modelled nonparametrically using penalized
B-splines, cyclic/tensor/shape-constrained splines, GAM.jl smooths, Matérn
SPDEs, Gaussian processes, random effects, or neural/KAN networks.

Provides 24 solvers (`LAML`, `MultiStart`, `EKFSolver`, `CollocationLAML`,
`GradientMatching`, `AdamSolver`, `MultipleShootingSolver`,
`AdaptiveGradientMatching`, `RodeoSolver`, `MCMCSolver`, `MagiSolver`,
`BNGSolver`, `DaltonSolver`, `PseudoMarginalSolver`, `GCVSolver`,
`TwoStageSolver`, `DerivativeFreeSolver`, `VariationalSolver`, `ABCSolver`,
`IntegralMatchingSolver`, `ProfileLikelihoodSolver`, `EnsembleKalmanSolver`,
`ODINSolver`, `FGPGMSolver`, and `RKHSSolver`) and 15 approximator types
(including optional `FluxKAN.jl` and `GAM.jl` backends).

Uses Laplace Approximate Marginal Likelihood (LAML) for automatic smoothing
parameter estimation, following:
- Wood (2001), Ecological Monographs 71(1), 1-25
- Wood (2011), JRSS-B 73(1), 3-36
- Wood, Pya & Säfken (2016), JASA 111(516), 1548-1575
- Wood & Fasiolo (2017), Biometrics 73(4), 1071-1081

Supports 9 likelihood families (`GaussianLikelihood`, `PoissonLikelihood`,
`NegativeBinomialLikelihood`, `TruncatedNormalLikelihood`, `GammaLikelihood`,
`LogNormalLikelihood`, `BinomialLikelihood`, `StudentTLikelihood`,
`CustomLikelihood`) plus direct `Distributions.jl` coercion, `StatsAPI` verbs,
multi-trajectory hierarchical PSMs (`MultiTrajectoryPSMProblem`), SDE process-noise
PSMs (`SDEPSMProblem`), model comparison & Wald hypothesis testing
(`compare_models`, `test_parametric_hypothesis`), and `ecorecipes` integration
(`GAM.jl`, `DynamicalResilience.jl`, `DecisionAnalysis.jl`).
"""
module PartiallySpecifiedModels

using LinearAlgebra
using Printf
using Random
using Statistics
using DataInterpolations
using ComponentArrays
using ForwardDiff
using DiffResults
using OrdinaryDiffEq
using SciMLBase
import SciMLBase: solve
import Lux
import Optim
import LineSearches
import MCMCChains
using DelayDiffEq
import StatsAPI
import ResilienceCore
import DynamicalResilience
import DynamicalResilience: metrics_over

# OrdinaryDiffEq 7 / SciMLBase 3 take a verbosity object for `verbose`, not a Bool
# (passing a Bool is an ArgumentError). The solvers run the ODE quietly by default,
# as `verbose=false` used to, and a user's `ode_kwargs[:verbose]::Bool` is mapped.
_quiet_solve() = OrdinaryDiffEq.DEVerbosity(OrdinaryDiffEq.SciMLLogging.None())
_ode_verbosity(v::Bool) = v ? OrdinaryDiffEq.DEVerbosity() : _quiet_solve()
_ode_verbosity(v) = v
function _normalise_ode_kwargs(kw::AbstractDict)
    haskey(kw, :verbose) || return kw
    out = Dict{Symbol, Any}(kw)
    out[:verbose] = _ode_verbosity(kw[:verbose])
    return out
end

# Core types
include("types.jl")

# B-spline approximator and penalty matrices
include("approximators.jl")

# Shared neural-network evaluator (Dual-safe MLP path + Lux fallback);
# must precede solver.jl, which uses build_neural_evaluator
include("neural_evaluator.jl")

# Generic evaluator construction (the approximator extension interface);
# must precede the solver files, which all call build_evaluator
include("approximator_interface.jl")

# Optional-backend, fixed-grid KAN evaluator and protocol methods.
include("kan_evaluator.jl")
include("kan_penalties.jl")
include("kan_diagnostics.jl")

# Likelihood families
include("likelihoods.jl")

# LAML smoothing parameter estimation
include("laml.jl")

# Shared penalized least-squares step (truncated SVD + step contraction)
include("pcls.jl")

# Main solver (IRLS + LAML)
include("solver.jl")

# Multistart wrapper around LAML
include("multistart.jl")

# Collocation-based solver
include("collocation_solver.jl")

# Gradient matching solver
include("gradient_matching.jl")

# Adam solver with autodiff through ODE
include("adam_solver.jl")

# Multiple shooting solver
include("multiple_shooting.jl")

# Adaptive gradient matching solver (deGradInfer-style)
include("adaptive_gradient_matching.jl")

# Probabilistic ODE solver (rodeo)
include("ibm_prior.jl")
include("kalman.jl")
include("probsolve.jl")
include("sqrt_kalman.jl")
include("rodeo_solver.jl")

# MCMC/HMC solver (Bayesian inference)
include("mcmc_solver.jl")

include("magi_solver.jl")

# BNG solver (Bayesian Neural Gradient matching)
include("bng_solver.jl")

# DALTON solver (data-adaptive likelihood)
include("dalton_solver.jl")

# Pseudo-marginal MCMC solver
include("pseudo_marginal_solver.jl")

# DDE solver support
include("dde_solver.jl")

# GCV solver (alternative smoothing parameter selection)
include("gcv_solver.jl")

# Two-stage smooth-then-differentiate solver
include("two_stage_solver.jl")

# Derivative-free solver (NelderMead / particle swarm)
include("derivative_free_solver.jl")

# Variational inference solver
include("variational_solver.jl")

# ABC-SMC solver (likelihood-free)
include("abc_solver.jl")

# Integral matching solver (Dattner & Klaassen 2015)
include("integral_matching_solver.jl")

# Profile likelihood solver (Simpson & Maclaren 2023)
include("profile_likelihood_solver.jl")

# Ensemble Kalman Inversion solver (Iglesias et al. 2013)
include("ensemble_kalman_solver.jl")

# ODIN solver (ODE-Informed regression, Wenk & Abbati 2020)
include("odin_solver.jl")

# FGPGM solver (fast GP-based gradient matching MCMC, Wenk et al. 2019)
include("fgpgm_solver.jl")

# RKHS solver (trajectory-in-RKHS gradient matching)
include("rkhs_solver.jl")

# Profile range parameter optimization for SPDE
include("profile_range.jl")

# Residual diagnostics
include("diagnostics.jl")

# Bootstrap confidence intervals
include("bootstrap.jl")

# Grid-wise simultaneous bands and post-bootstrap band construction
include("simultaneous_bands.jl")

# Opt-in Gaussian LAML smoothing-parameter covariance correction
include("smoothing_uncertainty.jl")

# Fitted models as SciML problems; resilience of their equilibria (DynamicalResilience.jl)
include("resilience.jl")

# StatsAPI verbs (predict, coef, vcov, dof, loglikelihood, aic, bic, nobs, residuals, fitted, isconverged)
include("statsapi.jl")

# Weak-dependency bridges: GAM.jl smooths and DecisionAnalysis.jl adapters
include("gam_approximator.jl")
include("decision_interface.jl")

# Flagship v0.3 modeling extensions: multi-trajectory hierarchical PSMs,
# SDE/process-noise CD-EKF PSMs, and model comparison / Wald hypothesis testing
include("multi_trajectory.jl")
include("sde_solver.jl")
include("model_comparison.jl")

# Exports — types
export AbstractApproximator, BSplineApproximator, CyclicBSplineApproximator,
       RandomEffectApproximator, NeuralApproximator, GPApproximator, SPDEApproximator
export KANApproximator, GAMSmoothApproximator
export TensorBSplineApproximator
export SingleIndexApproximator, index_loadings
export TransformedCovariateApproximator, lag_weights, smoothing_inertia,
       transformed_covariate
export ShapeConstrainedBSplineApproximator, ShapeConstrainedSPDEApproximator, SHAPE_CONSTRAINTS
export ShapeConstrainedGPApproximator
export COMONetApproximator, COMONET_CONSTRAINTS
export AbstractLikelihood, GaussianLikelihood, PoissonLikelihood,
       NegativeBinomialLikelihood, TruncatedNormalLikelihood,
       GammaLikelihood, LogNormalLikelihood, BinomialLikelihood,
       StudentTLikelihood, CustomLikelihood
export TrajectoryObservation
export PSMProblem, MultiTrajectoryPSMProblem, SDEPSMProblem, PSMSolution
export LAML, MultiStart, CollocationLAML, GradientMatching, EKFSolver
export AdamSolver, MultipleShootingSolver, AdaptiveGradientMatching
export RodeoSolver, MCMCSolver, MagiSolver
export BNGSolver, DaltonSolver, PseudoMarginalSolver
export GCVSolver, TwoStageSolver, DerivativeFreeSolver
export VariationalSolver, ABCSolver
export IntegralMatchingSolver, ProfileLikelihoodSolver
export EnsembleKalmanSolver, ODINSolver, RKHSSolver
export FGPGMSolver
export ManagedPSM, sample_belief
export ModelComparisonTable

# Exports — functions
export solve, simulate_psm, simulate_sde, ekf_loglik
export compare_models, test_parametric_hypothesis
using StatsAPI: predict
export predict
export to_ode_problem, to_discrete_problem, posterior_draws, metrics_over
export spline_penalty_matrix
export optimize_spde_range, with_range_param
export residual_diagnostics, durbin_watson, residual_acf, semivariogram
export check_constraints
export kan_edge_curves, kan_activation_diagnostics
export psm_appraise, deviance_residuals
export bootstrap, BootstrapResult
export confidence_band, smoothing_covariance_correction

# Re-export common ODE solvers and problem types
using OrdinaryDiffEq: Tsit5, Vern7, Vern9
# OrdinaryDiffEq 7 loads only its default solvers; the others come from sublibraries.
using OrdinaryDiffEqLowOrderRK: BS3
using OrdinaryDiffEqSDIRK: TRBDF2
using SciMLBase: ODEProblem, DiscreteProblem
export Tsit5, BS3, Vern7, Vern9, TRBDF2
export ODEProblem, DiscreteProblem
using DelayDiffEq: DDEProblem, MethodOfSteps
export DDEProblem, MethodOfSteps

end # module
