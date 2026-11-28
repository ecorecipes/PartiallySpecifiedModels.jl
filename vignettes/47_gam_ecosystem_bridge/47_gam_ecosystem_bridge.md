# GAM.jl & DynamicalResilience.jl Ecosystem Bridge: Predator–Prey Interactions
Simon Frost
2026-11-27

- [Overview](#overview)
- [Setup](#setup)
- [Predator–Prey System with Unknown Functional
  Response](#predatorprey-system-with-unknown-functional-response)
- [Fitting with `GAMSmoothApproximator` (`GAM.jl`
  Extension)](#fitting-with-gamsmoothapproximator-gamjl-extension)
- [Equilibrium Resilience & Uncertainty Quantification
  (`DynamicalResilience.jl`)](#equilibrium-resilience--uncertainty-quantification-dynamicalresiliencejl)
- [Bivariate `GAM.t2` Smooth Preview](#bivariate-gamt2-smooth-preview)
- [Summary](#summary)

## Overview

Across the `ecorecipes` ecosystem, users routinely specify smooth
functional forms with [`GAM.jl`](https://github.com/ecorecipes/GAM.jl)
(thin-plate regression splines `:tp`, cyclic cubic splines `:cc`, and
Wood’s scale-invariant `t2` tensor products) and analyze ecosystem
stability with
[`DynamicalResilience.jl`](https://github.com/ecorecipes/DynamicalResilience.jl).

This vignette demonstrates how `GAMSmoothApproximator` embeds any
`GAM.jl` smooth directly inside a mechanistic ODE fitted by `LAML`, and
how `steady_state`, `linearise`, and `metrics_over` seamlessly propagate
smooth-curve uncertainty into equilibrium resilience metrics.

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using GAM
using DynamicalResilience
using ResilienceCore
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Statistics
using Printf
```

## Predator–Prey System with Unknown Functional Response

Consider a Rosenzweig–MacArthur predator–prey system where prey growth
is logistic and the predator functional response $g(N)$ is estimated
using a `GAM.jl` thin-plate regression spline `GAM.s(:N; bs=:tp, k=8)`:

$$\frac{dN}{dt} = r N \left(1 - \frac{N}{K}\right) - g(N) P, \qquad
\frac{dP}{dt} = e \, g(N) P - m P.$$

``` julia
rng = Random.Xoshiro(4701)
r_p = 1.0; K_p = 10.0; e_p = 0.5; m_p = 0.35
g_true(N) = 1.4 * N / (2.5 + N)   # Holling Type II

ts = collect(range(0.0, 18.0, length=37))
u0_pp = [4.0, 1.2]
sol_pp = OrdinaryDiffEq.solve(
    ODEProblem((du, u, p, t) -> begin
        g = g_true(u[1])
        du[1] = r_p * u[1] * (1.0 - u[1] / K_p) - g * u[2]
        du[2] = e_p * g * u[2] - m_p * u[2]
    end, u0_pp, (0.0, 18.0)),
    Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)

Y_pp = hcat(Float64[sol_pp[1, i] for i in eachindex(ts)],
            Float64[sol_pp[2, i] for i in eachindex(ts)]) .+
       0.05 .* randn(rng, length(ts), 2)
```

    37×2 Matrix{Float64}:
     4.00623   1.23087
     4.63174   1.21035
     5.35313   1.32641
     5.91449   1.38004
     6.31047   1.51964
     6.65975   1.67651
     6.90288   1.82976
     6.98885   1.93044
     6.98976   2.1863
     6.90344   2.33099
     ⋮         
     0.329379  2.28785
     0.41164   1.95419
     0.287025  1.76772
     0.410079  1.55456
     0.475348  1.30671
     0.50583   1.16413
     0.637867  1.05024
     0.889243  0.897949
     1.06856   0.89238

## Fitting with `GAMSmoothApproximator` (`GAM.jl` Extension)

``` julia
function pp_ode!(du, u, p, t)
    g = p.g(u[1])
    du[1] = p.r * u[1] * (1.0 - u[1] / p.K) - g * u[2]
    du[2] = p.e * g * u[2] - p.m * u[2]
    nothing
end

approx_gam = GAMSmoothApproximator(:g, GAM.s(:N; bs=:tp, k=8), (0.0, 10.0);
                                   initial = N -> 0.4 * N)

prob_pp = PSMProblem(pp_ode!, u0_pp, (0.0, 18.0), [approx_gam];
    data_times = ts,
    data_values = Y_pp,
    obs_to_state = [1, 2],
    known_params = (r=r_p, K=K_p, e=e_p, m=m_p),
    likelihood = GaussianLikelihood())

sol_gam = solve(prob_pp, LAML(maxiters=60))
println("Converged: ", sol_gam.convergence.converged, "  EDF: ", round(sol_gam.edf, digits=2))

g_hat = sol_gam.unknown_functions[:g]
N_eval = range(1.0, 7.0, length=25)
@printf("Holling Type II recovery RMSE via GAM.s(:tp): %.4f\n",
        sqrt(mean((g_hat(N) - g_true(N))^2 for N in N_eval)))
```

    Converged: true  EDF: 5.99
    Holling Type II recovery RMSE via GAM.s(:tp): 0.0063

## Equilibrium Resilience & Uncertainty Quantification (`DynamicalResilience.jl`)

Because `GAMSmoothApproximator`’s evaluator is Dual-safe in both
parameters and state coordinates, `DynamicalResilience.steady_state`,
`ResilienceCore.linearise`, and `metrics_over` differentiate directly
through the fitted `GAM.jl` smooth:

``` julia
eq_pp = DynamicalResilience.steady_state(prob_pp, sol_gam, [2.5, 2.0])
lin_pp = ResilienceCore.linearise(prob_pp, sol_gam, eq_pp.u)
dom_re = maximum(real, eigvals(lin_pp.J))

println("Coexistence equilibrium (N*, P*): ", round.(eq_pp.u, digits=3))
println("Dominant eigenvalue Re(λ₁):       ", round(dom_re, digits=4))

res_pp = metrics_over(prob_pp, sol_gam; u0=eq_pp.u, n=40, rng=Random.Xoshiro(4702))
println("Resilience UQ draws completed:    ", res_pp.n)
```

    Coexistence equilibrium (N*, P*): [2.48, 2.664]
    Dominant eigenvalue Re(λ₁):       0.065
    Resilience UQ draws completed:    40

## Bivariate `GAM.t2` Smooth Preview

`GAMSmoothApproximator` also wraps Wood’s multi-penalty `GAM.t2`
tensor-product smooths for bivariate ecological interactions such as
Beddington–DeAngelis predator interference $g(N, P)$:

``` julia
approx_bd = GAMSmoothApproximator(:g, GAM.t2(:N, :P; k=[4, 4]),
                                  ((0.0, 10.0), (0.0, 5.0)); initial=0.5)
println("GAM.t2 parameter count: ", PartiallySpecifiedModels.nparams(approx_bd),
        "  independent penalty blocks: ", length(PartiallySpecifiedModels.penalty_blocks(approx_bd)))
```

    GAM.t2 parameter count: 16  independent penalty blocks: 3

## Summary

- `GAMSmoothApproximator` brings the full `GAM.jl` smooth catalogue
  (`:tp`, `:cr`, `:cc`, `te`, `t2`) directly into mechanistic PSMs with
  automatic LAML smoothing.
- `DynamicalResilience.jl` and `ResilienceCore.jl` consume the fitted
  PSM seamlessly for equilibrium stability and posterior resilience UQ.
