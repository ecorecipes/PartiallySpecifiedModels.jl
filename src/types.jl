# Core type definitions for PartiallySpecifiedModels.jl
# Modularized into topic-focused files under src/types/:
#   1. approximator_types.jl  — AbstractApproximator & all built-in approximators
#   2. likelihood_types.jl    — AbstractLikelihood, exponential families & coercion
#   3. solver_types.jl        — Fitting algorithm structs & keyword constructors
#   4. problem_types.jl       — TrajectoryObservation, PSMProblem, PSMSolution & SciML bridges

include("types/approximator_types.jl")
include("types/likelihood_types.jl")
include("types/solver_types.jl")
include("types/problem_types.jl")
