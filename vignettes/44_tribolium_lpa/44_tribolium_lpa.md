# Tribolium Flour Beetle LPA Dynamics: Discrete Maps and Resilience
Simon Frost
2026-11-27

- [Overview](#overview)
- [Setup](#setup)
- [Simulating Biweekly LPA Census
  Data](#simulating-biweekly-lpa-census-data)
- [Discrete-Time PSM with Shape-Constrained Cannibalism
  Survival](#discrete-time-psm-with-shape-constrained-cannibalism-survival)
- [Discrete-Time Equilibrium & Resilience UQ
  (`DynamicalResilience.jl`)](#discrete-time-equilibrium--resilience-uq-dynamicalresiliencejl)
- [Summary](#summary)

## Overview

The *Tribolium castaneum* (flour beetle) Larva–Pupa–Adult (LPA) model
(Costantino et al. 1995, 1997, *Science*) is one of ecology’s iconic
discrete-time nonlinear systems, famous for exhibiting equilibria,
two-cycles, invariant loops, and chaos driven by cannibalism of eggs and
pupae by larvae and adults:

$$\begin{aligned}
L_{t+1} &= b \, A_t \, \exp\big(-c_{el} L_t - c_{ea} A_t\big) \\
P_{t+1} &= (1 - \mu_l) L_t \\
A_{t+1} &= P_t \, s_P(A_t) + (1 - \mu_a) A_t
\end{aligned}$$

Here we replace the parametric adult-cannibalism survival term
$\exp(-c_{pa} A_t)$ on pupal recruitment with a nonparametric
shape-constrained smooth $s_P(A_t) \in (0, 1)$ (decreasing positive via
`ShapeConstrainedBSplineApproximator`), construct the fitting problem
directly from a SciML `DiscreteProblem`, and propagate parameter
uncertainty into discrete-time equilibrium stability metrics via
`to_discrete_problem` and `DynamicalResilience.jl`.

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using DynamicalResilience
using ResilienceCore
using LinearAlgebra
using Random
using Statistics
using Printf
```

## Simulating Biweekly LPA Census Data

``` julia
rng = Random.Xoshiro(4401)
b_true = 6.598
cel_true = 0.01209
cea_true = 0.01155
cpa_true = 0.0047
mul_true = 0.2055
mua_true = 0.50

sP_true(A) = exp(-cpa_true * A)

n_steps = 30
U_true = zeros(n_steps + 1, 3)
U_true[1, :] = [120.0, 60.0, 40.0]
for k in 1:n_steps
    L, P, A = U_true[k, :]
    U_true[k + 1, 1] = b_true * A * exp(-cel_true * L - cea_true * A)
    U_true[k + 1, 2] = (1.0 - mul_true) * L
    U_true[k + 1, 3] = P * sP_true(A) + (1.0 - mua_true) * A
end

times = collect(0.0:1.0:Float64(n_steps))
Y_obs = max.(U_true .+ 2.5 .* randn(rng, size(U_true)), 0.1)
println("Tribolium LPA series: $(length(times)) biweekly censuses (L, P, A)")
```

    Tribolium LPA series: 31 biweekly censuses (L, P, A)

## Discrete-Time PSM with Shape-Constrained Cannibalism Survival

``` julia
function lpa_map!(u_next, u, p, t)
    L, P, A = u[1], u[2], u[3]
    u_next[1] = p.b * A * exp(-p.cel * L - p.cea * A)
    u_next[2] = (1.0 - p.mul) * L
    u_next[3] = P * p.sP(A) + (1.0 - p.mua) * A
    nothing
end

dp_template = DiscreteProblem(lpa_map!, [120.0, 60.0, 40.0], (0.0, Float64(n_steps)))

approx_sP = ShapeConstrainedBSplineApproximator(:sP, (0.0, 260.0), 8, :dec_positive;
    initial = A -> exp(-0.003 * A))

prob = PSMProblem(dp_template, [approx_sP];
    data_times = times,
    data_values = Y_obs,
    obs_to_state = [1, 2, 3],
    known_params = (b=b_true, cel=cel_true, cea=cea_true, mul=mul_true, mua=mua_true),
    likelihood = GaussianLikelihood())

sol = solve(prob, LAML(maxiters=60))
println("Converged: ", sol.convergence.converged, "  EDF: ", round(sol.edf, digits=2))

sP_hat = sol.unknown_functions[:sP]
A_grid = range(20.0, 200.0, length=25)
rmse_sP = sqrt(mean((sP_hat(A) - sP_true(A))^2 for A in A_grid))
@printf("Pupal cannibalism survival RMSE: %.4f\n", rmse_sP)
```

    ┌ Warning: LAML: the objective converged but the parameters did not — one more PCLS step from the reported fit moves β by 3.24 (relative), above 0.001. The fit stopped on a flat ridge of the penalized objective: the fitted FUNCTIONS are usable, but individual coefficients and their covariance are not pinned. See `convergence.ridge` and `convergence.final_parameter_step`.
    └ @ PartiallySpecifiedModels ~/Projects/psm/PartiallySpecifiedModels.jl/src/solver.jl:2451
    Converged: true  EDF: 1.0
    Pupal cannibalism survival RMSE: 0.0198

## Discrete-Time Equilibrium & Resilience UQ (`DynamicalResilience.jl`)

Using `to_discrete_problem` and the direct `DynamicalResilience`
bridges, we compute the fixed point, linearise the discrete map Jacobian
$J^*$, and propagate LAML parameter uncertainty to resilience metrics:

``` julia
dp_fit = to_discrete_problem(prob, sol)
eq = DynamicalResilience.steady_state(prob, sol, [100.0, 80.0, 100.0])
lin = ResilienceCore.linearise(prob, sol, eq.u)

println("Fitted equilibrium (L*, P*, A*): ", round.(eq.u, digits=2))
println("Spectral radius ρ(J*):           ", round(maximum(abs, eigvals(lin.J)), digits=4))

res_uq = metrics_over(prob, sol; u0=eq.u, n=40, rng=Random.Xoshiro(4402))
println("Posterior resilience ensemble size: ", res_uq.n)
```

    Fitted equilibrium (L*, P*, A*): [79.94, 63.51, 86.24]
    Spectral radius ρ(J*):           0.9649
    Posterior resilience ensemble size: 40

## Summary

- `PSMProblem(DiscreteProblem(...), ...)` wraps SciML discrete maps
  natively with `LAML`.
- `:dec_positive` enforces biologically valid non-negative,
  non-increasing cannibalism survival $s_P(A)$.
- `to_discrete_problem`, `steady_state`, `linearise`, and `metrics_over`
  connect discrete PSM fits directly to `DynamicalResilience.jl`
  stability analysis.
