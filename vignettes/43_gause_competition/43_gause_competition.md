# Gause Paramecium Competition: Hierarchical Multi-Trajectory PSMs
Simon Frost
2026-11-27

- [Overview](#overview)
- [Setup](#setup)
- [Multi-Replicate Microcosm Competition
  Data](#multi-replicate-microcosm-competition-data)
- [Hierarchical Multi-Trajectory
  PSM](#hierarchical-multi-trajectory-psm)
- [Testing Nonlinear Density
  Dependence](#testing-nonlinear-density-dependence)
- [Summary](#summary)

## Overview

In his foundational microcosm experiments (Gause 1934), G. F. Gause grew
*Paramecium aurelia* and *Paramecium caudatum* across replicated culture
tubes with differing initial inocula and medium conditions to study
competitive exclusion and density-dependent growth.

Rather than assuming a rigid linear Lotka–Volterra per-capita growth
rate $r(N) = r_0 (1 - N / K)$, we model the shared density-dependence
curve nonparametrically using
`ShapeConstrainedBSplineApproximator(..., :decreasing)` while pooling
information across multiple experimental replicate trajectories via
`MultiTrajectoryPSMProblem` and tube-level `RandomEffectApproximator`s.
Finally, we use `test_parametric_hypothesis` to test formally whether
density dependence deviates from linear logistic regulation.

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using Random
using Statistics
using Printf
```

## Multi-Replicate Microcosm Competition Data

We generate 4 replicated competition microcosms (starting from distinct
inocula and experiencing tube-level productivity random effects
$b_g \sim \mathcal{N}(0, \sigma_b^2)$) governed by nonlinear
theta-logistic self-limitation in *P. aurelia* ($N_1$) and interspecific
competition with *P. caudatum* ($N_2$):

$$\frac{dN_1}{dt} = \big(r_1(N_1) - \alpha_{12} N_2 + b_g\big) N_1, \qquad
\frac{dN_2}{dt} = r_2 \left(1 - \frac{N_2 + \alpha_{21} N_1}{K_2}\right) N_2.$$

``` julia
rng = Random.Xoshiro(4301)
ts = collect(range(0.0, 14.0, length=22))

# Nonlinear concave density dependence (theta-logistic, theta = 1.8)
r1_true(N) = 0.85 * (1.0 - (max(N, 0.0) / 250.0)^1.8)
alpha12_true = 0.0022
r2_true = 0.55; K2_true = 180.0; alpha21_true = 0.75

# Four replicate culture tubes with different inocula and tube random effects
u0_tubes = [[12.0, 10.0], [35.0, 15.0], [18.0, 40.0], [60.0, 25.0]]
b_tubes = [-0.07, -0.02, 0.03, 0.06]

y_tubes = Matrix{Float64}[]
for g in 1:4
    ode_g = ODEProblem((du, u, p, t) -> begin
        du[1] = (r1_true(u[1]) - alpha12_true * u[2] + b_tubes[g]) * u[1]
        du[2] = r2_true * (1.0 - (u[2] + alpha21_true * u[1]) / K2_true) * u[2]
    end, u0_tubes[g], (0.0, 14.0))
    sol_g = OrdinaryDiffEq.solve(ode_g, Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
    U_g = hcat(Float64[sol_g[1, i] for i in eachindex(ts)],
               Float64[sol_g[2, i] for i in eachindex(ts)])
    push!(y_tubes, max.(U_g .+ 3.0 .* randn(rng, size(U_g)), 0.1))
end

println("Generated $(length(y_tubes)) replicate tubes × $(length(ts)) days × 2 species")
```

    Generated 4 replicate tubes × 22 days × 2 species

## Hierarchical Multi-Trajectory PSM

All 4 tubes share the decreasing per-capita growth curve `r1(N1)` while
each tube receives a shrinkage-penalized productivity offset
`p.tube(p.group_id)` whose variance is selected automatically by LAML:

``` julia
function gause_ode!(du, u, p, t)
    du[1] = (p.r1(u[1]) - p.alpha12 * u[2] + p.tube(p.group_id)) * u[1]
    du[2] = p.r2 * (1.0 - (u[2] + p.alpha21 * u[1]) / p.K2) * u[2]
end

approxs = AbstractApproximator[
    ShapeConstrainedBSplineApproximator(:r1, (0.0, 280.0), 8, :decreasing;
        initial = N -> 0.8 * (1.0 - N / 250.0)),
    RandomEffectApproximator(:tube, 4; initial=0.0),
]

mp = MultiTrajectoryPSMProblem(
    gause_ode!, u0_tubes, (0.0, 14.0), approxs;
    data_times_list = fill(ts, 4),
    data_values_list = y_tubes,
    known_params_list = fill((alpha12=alpha12_true, r2=r2_true,
                              K2=K2_true, alpha21=alpha21_true), 4),
    likelihood = GaussianLikelihood())

sol_mt = solve(mp, LAML(maxiters=60))

println("Converged:        ", sol_mt.convergence.converged, " ($(sol_mt.convergence.reason))")
println("Total EDF:        ", round(sol_mt.edf, digits=2))
println("Smoothing λ:      ", [round(s, sigdigits=4) for s in sol_mt.smoothing_params])
println("True tube b_g:    ", round.(b_tubes, digits=3))
println("Fitted tube b_g:  ", round.(sol_mt.parameters.tube, digits=3))
```

    Converged:        true (converged_tol)
    Total EDF:        7.39
    Smoothing λ:      [36.73, 2558.0]
    True tube b_g:    [-0.07, -0.02, 0.03, 0.06]
    Fitted tube b_g:  [-0.07, -0.023, 0.023, 0.07]

## Testing Nonlinear Density Dependence

We compare the recovered monotone per-capita growth rate
$\hat{r}_1(N_1)$ against the true theta-logistic curve and test
$H_0: r_1(N_1)$ is linear (standard Lotka–Volterra) via
`test_parametric_hypothesis`:

``` julia
r1_hat = sol_mt.unknown_functions[:r1]
N_grid = range(20.0, 230.0, length=25)
rmse_r1 = sqrt(mean((r1_hat(N) - r1_true(N))^2 for N in N_grid))
@printf("Per-capita growth rate RMSE on [20, 230]: %.4f\n", rmse_r1)

wald_lin = test_parametric_hypothesis(sol_mt, mp.problems[1], :r1, :linear; grid=N_grid)
@printf("Wald test vs linear logistic: T = %.2f (df = %d), p = %.3e\n",
        wald_lin.statistic, wald_lin.df, wald_lin.pvalue)
```

    Per-capita growth rate RMSE on [20, 230]: 0.0142
    Wald test vs linear logistic: T = 237.83 (df = 5), p = 2.239e-49

## Summary

- `MultiTrajectoryPSMProblem` pools information across heterogeneous
  replicate experiments while sharing smooth mechanistic responses.
- `RandomEffectApproximator` provides `mgcv`-style ridge random effects
  whose variance component is estimated automatically by LAML.
- `test_parametric_hypothesis` formally evaluates whether data support
  nonlinear density dependence beyond classical Lotka–Volterra
  competition.
