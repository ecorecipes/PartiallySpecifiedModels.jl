# GCV Smoothing Parameter Selection
Simon Frost
2026-04-02

- [Overview](#overview)
- [Logistic Growth with Unknown Per-Capita
  Rate](#logistic-growth-with-unknown-per-capita-rate)
- [GCV vs LAML Comparison](#gcv-vs-laml-comparison)
  - [Trajectory Fit](#trajectory-fit)
  - [Recovered Unknown Function](#recovered-unknown-function)
  - [Summary](#summary)
- [Effect of GCV Inflation Factor γ](#effect-of-gcv-inflation-factor-γ)
- [Diagnostic Plots](#diagnostic-plots)
- [When to Use GCV vs LAML](#when-to-use-gcv-vs-laml)

## Overview

The `GCVSolver` uses **Generalized Cross-Validation** (GCV) for
automatic smoothing parameter selection — a simpler alternative to LAML
(Laplace Approximate Marginal Likelihood). While LAML uses a Laplace
approximation to the marginal likelihood with Fellner-Schall updates,
GCV minimizes a leave-one-out cross-validation score:

$$\text{GCV}(\lambda) = \frac{n \| W^{1/2}(z - J\hat\beta) \|^2}{(n - \gamma \cdot \text{tr}(A))^2}$$

where $A = J(J'WJ + S^\lambda)^{-1}J'W$ is the hat matrix and
$\gamma \geq 1$ is an inflation factor that guards against
under-smoothing (default 1.4).

``` julia
using PartiallySpecifiedModels
using OrdinaryDiffEq
using Plots
using Random
Random.seed!(123)
```

    Precompiling packages...
        PartiallySpecifiedModels Being precompiled by another process (pid: 36853, pidfile: /Users/username/.julia/compiled/v1.12/PartiallySpecifiedModels/tWtwA_lLwID.ji.pidfile)
      19702.6 ms  ✓ PartiallySpecifiedModels
      1 dependency successfully precompiled in 32 seconds. 387 already precompiled.

    TaskLocalRNG()

## Logistic Growth with Unknown Per-Capita Rate

We model logistic growth $dN/dt = r(N) \cdot N$ where
$r(N) = 0.5(1 - N/10)$ is unknown.

``` julia
r_true(N) = 0.5 * (1.0 - N / 10.0)

function logistic!(du, u, p, t)
    N = u[1]
    du[1] = p.r(N) * N
end

sol_true = solve(ODEProblem(logistic!, [1.0], (0.0, 15.0), (; r=r_true)),
                 Tsit5(); saveat=0.5)
t_data = collect(sol_true.t)
data_N = [sol_true.u[i][1] + 0.1 * randn() for i in 1:length(t_data)]
data_matrix = reshape(max.(data_N, 0.01), :, 1)
```

    31×1 Matrix{Float64}:
      0.9354269327896023
      1.1022379499081711
      1.3859208539670416
      1.8825231543204026
      2.3689191995037016
      2.892534072992916
      3.33227680373001
      4.055139873380624
      4.374401031201666
      5.173094818257388
      ⋮
      9.517594615011413
      9.760412380623237
      9.79268472010831
      9.856463164828625
      9.872187590739777
      9.935785227284114
      9.903174071008431
     10.115692712241504
      9.998761920177897

## GCV vs LAML Comparison

``` julia
uf = BSplineApproximator(:r, (0.0, 12.0), 10)

prob = PSMProblem(logistic!, [1.0], (0.0, 15.0), [uf];
    data_times=t_data, data_values=Float64.(data_matrix),
    obs_to_state=[1], known_params=NamedTuple())

sol_gcv = solve(prob, GCVSolver(maxiters=50, gamma=1.4, verbose=false))
sol_laml = solve(prob, LAML(maxiters=50, verbose=false))
```

    PSMSolution((r = [0.5048302484826799, 0.4362630715311124, 0.367642219459838, 0.2990970498832144, 0.231112276182824, 0.16405318966501814, 0.09823441381242196, 0.03393922690935286, -0.029212667929864967, -0.09235165473684903]), 0.1032156623501261, 0.1992839931941202, 3.0733221362588052, [1.4435541602090602], [1.0; 1.2505546606982674; … ; 9.969999186255793; 9.986515470396709;;], [0.9354269327896023; 1.1022379499081711; … ; 10.115692712241504; 9.998761920177897;;], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5  …  10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0], Dict{Symbol, Any}(:r => DataInterpolations.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}([0.5048302484826799, 0.4362630715311124, 0.367642219459838, 0.2990970498832144, 0.231112276182824, 0.16405318966501814, 0.09823441381242196, 0.03393922690935286, -0.029212667929864967, -0.09235165473684903], [0.0, 1.3333333333333333, 2.6666666666666665, 4.0, 5.333333333333333, 6.666666666666667, 8.0, 9.333333333333334, 10.666666666666666, 12.0], Float64[], DataInterpolations.CubicSplineParameterCache{Vector{Float64}}(Float64[], Float64[]), [0.0, 1.3333333333333333, 1.3333333333333333, 1.3333333333333335, 1.333333333333333, 1.333333333333334, 1.333333333333333, 1.333333333333334, 1.3333333333333321, 1.333333333333334], [0.0, -4.233914458022046e-5, -1.1796950689859511e-5, 0.00034495536678612413, 0.000523311565832395, 0.0006859926111071669, 0.0009187664848214652, 0.0007810541542610611, -0.00018437238636796622, 0.0], DataInterpolations.ExtrapolationType.Extension, DataInterpolations.ExtrapolationType.Extension, FindFirstFunctions.Guesser{Vector{Float64}}([0.0, 1.3333333333333333, 2.6666666666666665, 4.0, 5.333333333333333, 6.666666666666667, 8.0, 9.333333333333334, 10.666666666666666, 12.0], Base.RefValue{Int64}(1), true), false, false)), nothing)

### Trajectory Fit

``` julia
p1 = plot(t_data, data_matrix[:, 1], seriestype=:scatter, label="Data",
          xlabel="Time", ylabel="N", title="Trajectory Comparison", ms=3, alpha=0.6)
plot!(p1, t_data, sol_gcv.fitted_values[:, 1], label="GCV (loss=$(round(sol_gcv.data_loss, digits=2)))", lw=2)
plot!(p1, t_data, sol_laml.fitted_values[:, 1], label="LAML (loss=$(round(sol_laml.data_loss, digits=2)))", lw=2, ls=:dash)
p1
```

![](21_gcv_files/figure-commonmark/cell-5-output-1.svg)

### Recovered Unknown Function

``` julia
r_gcv = sol_gcv.unknown_functions[:r]
r_laml = sol_laml.unknown_functions[:r]
N_grid = range(0.0, 12.0, length=100)

p2 = plot(N_grid, r_true.(N_grid), label="True r(N)", lw=2, color=:black,
          xlabel="N", ylabel="r(N)", title="Per-Capita Growth Rate")
plot!(p2, N_grid, [r_gcv(n) for n in N_grid], label="GCV (edf=$(round(sol_gcv.edf, digits=1)))", lw=2)
plot!(p2, N_grid, [r_laml(n) for n in N_grid], label="LAML (edf=$(round(sol_laml.edf, digits=1)))", lw=2, ls=:dash)
p2
```

![](21_gcv_files/figure-commonmark/cell-6-output-1.svg)

### Summary

``` julia
println("GCV:  data_loss=$(round(sol_gcv.data_loss, digits=3)), edf=$(round(sol_gcv.edf, digits=1)), " *
        "r(5)=$(round(r_gcv(5.0), digits=3)) (true=$(round(r_true(5.0), digits=3)))")
println("LAML: data_loss=$(round(sol_laml.data_loss, digits=3)), edf=$(round(sol_laml.edf, digits=1)), " *
        "r(5)=$(round(r_laml(5.0), digits=3)) (true=$(round(r_true(5.0), digits=3)))")
```

    GCV:  data_loss=73.522, edf=9.8, r(5)=35.254 (true=0.25)
    LAML: data_loss=0.199, edf=3.1, r(5)=0.248 (true=0.25)

## Effect of GCV Inflation Factor γ

The inflation factor $\gamma$ controls the bias-variance tradeoff.
$\gamma = 1$ is standard GCV; $\gamma > 1$ penalizes model complexity
more, producing smoother fits.

``` julia
gammas = [1.0, 1.4, 2.0]
p3 = plot(N_grid, r_true.(N_grid), label="True", lw=2, color=:black,
          xlabel="N", ylabel="r(N)", title="Effect of γ on Smoothing")
for γ in gammas
    sol_g = solve(prob, GCVSolver(maxiters=50, gamma=γ, verbose=false))
    r_g = sol_g.unknown_functions[:r]
    plot!(p3, N_grid, [r_g(n) for n in N_grid],
          label="γ=$(γ) (edf=$(round(sol_g.edf, digits=1)))", lw=2)
end
p3
```

![](21_gcv_files/figure-commonmark/cell-8-output-1.svg)

## Diagnostic Plots

A standard 4-panel diagnostic display assesses residual behaviour. The
QQ plot checks normality of standardized residuals, “Residuals vs
Fitted” detects systematic patterns, the histogram visualises the
residual distribution, and “Observed vs Fitted” checks overall
calibration.

``` julia
using PartiallySpecifiedModels: appraise

diag = appraise(sol_gcv)

p_qq = scatter(diag.qq_theoretical, diag.qq_sample,
    xlabel="Theoretical quantiles", ylabel="Sample quantiles",
    title="QQ Plot of Residuals", ms=3, legend=false, color=:steelblue)
mn, mx = extrema(vcat(diag.qq_theoretical, diag.qq_sample))
plot!(p_qq, [mn, mx], [mn, mx], color=:red, ls=:dash, label="")

p_rf = scatter(diag.fitted, diag.residuals,
    xlabel="Fitted values", ylabel="Residuals",
    title="Residuals vs Fitted", ms=3, legend=false, color=:steelblue)
hline!(p_rf, [0], color=:gray, ls=:dot)

p_hist = histogram(diag.residuals, normalize=:pdf,
    xlabel="Residuals", ylabel="Density",
    title="Histogram of Residuals", legend=false, color=:steelblue, alpha=0.7)

p_of = scatter(diag.observed, diag.fitted,
    xlabel="Observed", ylabel="Fitted",
    title="Observed vs Fitted", ms=3, legend=false, color=:steelblue)
mn2, mx2 = extrema(vcat(diag.observed, diag.fitted))
plot!(p_of, [mn2, mx2], [mn2, mx2], color=:red, ls=:dash, label="")

plot(p_qq, p_rf, p_hist, p_of, layout=(2, 2), size=(700, 600))
```

![](21_gcv_files/figure-commonmark/cell-9-output-1.svg)

    Durbin-Watson: 0.39

## When to Use GCV vs LAML

- **GCV** is simpler, faster (no Hessian needed), and works well with
  abundant data
- **LAML** is more principled (approximate marginal likelihood), handles
  complex penalty structures better
- GCV with $\gamma = 1.4$ often gives similar results to LAML
- For non-Gaussian likelihoods, LAML is generally preferred
