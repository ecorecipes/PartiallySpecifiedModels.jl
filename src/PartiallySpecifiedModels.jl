"""
    PartiallySpecifiedModels

A Julia package for fitting partially specified models (PSMs) — dynamical
systems where some functional responses are modelled nonparametrically using
penalized B-splines, Gaussian processes, or neural networks.

Provides 17 solvers (LAML, collocation, gradient matching, Adam, multiple
shooting, adaptive gradient matching, rodeo, MCMC/HMC, MAGI, BNG, DALTON,
pseudo-marginal, GCV, two-stage, derivative-free, variational, and ABC-SMC)
and 7 approximator types (B-spline, shape-constrained B-spline, SPDE,
shape-constrained SPDE, neural network, Gaussian process, and COMONet).

Uses Laplace Approximate Marginal Likelihood (LAML) for automatic smoothing
parameter estimation, following:
- Wood, Pya & Säfken (2016), JASA 111(516), 1548-1575
- Wood & Fasiolo (2017), Statistics and Computing 27(3), 759-774

Supports Gaussian, Poisson, Negative Binomial, and custom likelihoods.
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

# Core types
include("types.jl")

# B-spline approximator and penalty matrices
include("approximators.jl")

# Likelihood families
include("likelihoods.jl")

# LAML smoothing parameter estimation
include("laml.jl")

# Main solver (IRLS + LAML)
include("solver.jl")

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
include("interrogation.jl")
include("probsolve.jl")
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

# Profile range parameter optimization for SPDE
include("profile_range.jl")

# Residual diagnostics
include("diagnostics.jl")

# Bootstrap confidence intervals
include("bootstrap.jl")

# Exports — types
export AbstractApproximator, BSplineApproximator, NeuralApproximator, GPApproximator, SPDEApproximator
export ShapeConstrainedBSplineApproximator, ShapeConstrainedSPDEApproximator, SHAPE_CONSTRAINTS
export COMONetApproximator, COMONET_CONSTRAINTS
export AbstractLikelihood, Gaussian, Poisson, NegativeBinomial, TruncatedNormal,
       CustomLikelihood
export PSMProblem, PSMSolution, LAML, CollocationLAML, GradientMatching
export AdamSolver, MultipleShootingSolver, AdaptiveGradientMatching
export RodeoSolver, MCMCSolver, MagiSolver
export BNGSolver, DaltonSolver, PseudoMarginalSolver
export GCVSolver, TwoStageSolver, DerivativeFreeSolver
export VariationalSolver, ABCSolver

# Exports — functions
export solve, simulate, predict
export spline_penalty_matrix, penalty_matrix
export nparams, initial_params
export optimize_spde_range, with_range_param
export residual_diagnostics, durbin_watson, residual_acf, semivariogram
export appraise, deviance_residuals
export bootstrap, BootstrapResult
export confidence_band

# Re-export common ODE solvers and problem types
using OrdinaryDiffEq: Tsit5, BS3, Vern7, Vern9, TRBDF2
using SciMLBase: ODEProblem, DiscreteProblem
export Tsit5, BS3, Vern7, Vern9, TRBDF2
export ODEProblem, DiscreteProblem
using DelayDiffEq: DDEProblem, MethodOfSteps
export DDEProblem, MethodOfSteps

end # module
