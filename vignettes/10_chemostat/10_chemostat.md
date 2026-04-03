# Chemostat Dynamics: Recovering Microbial Growth Kinetics
Simon Frost
2026-04-03

- [Overview](#overview)
- [Setup](#setup)
- [The Chemostat Model](#the-chemostat-model)
  - [Visualise Monod kinetics](#visualise-monod-kinetics)
  - [Generate data](#generate-data)
- [Define and Fit the PSM](#define-and-fit-the-psm)
  - [LAML fit](#laml-fit)
  - [RodeoSolver fit (with
    uncertainty)](#rodeosolver-fit-with-uncertainty)
- [Results](#results)
  - [Fitted trajectories](#fitted-trajectories)
  - [Recovered growth kinetics](#recovered-growth-kinetics)
- [Residual Diagnostics](#residual-diagnostics)
- [Substrate Inhibition: What If Monod Is
  Wrong?](#substrate-inhibition-what-if-monod-is-wrong)
- [Diagnostic Plots](#diagnostic-plots)
- [Key Takeaways](#key-takeaways)

## Overview

The **chemostat** is a fundamental model in microbial ecology and
biotechnology. In a continuous-flow bioreactor, microorganisms grow on a
limiting substrate. The specific growth rate $\mu(S)$ — how fast
microbes grow as a function of substrate concentration — is a key
unknown quantity.

The classical **Monod kinetics** assumes
$\mu(S) = \mu_{\max} S / (K_s + S)$, analogous to Michaelis–Menten
enzyme kinetics. However, in practice:

- Substrate inhibition may cause growth to decline at high $S$
- Multiple limiting substrates may create more complex dependencies
- Overflow metabolism or maintenance requirements may alter the
  relationship

A PSM approach lets us **estimate $\mu(S)$ directly from time series
data**, without assuming a specific functional form.

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using Plots
using Statistics
using Random
Random.seed!(42)
```

    TaskLocalRNG()

## The Chemostat Model

$$\begin{aligned}
\frac{dS}{dt} &= D(S_{\text{in}} - S) - \frac{\mu(S) \cdot X}{Y} \\
\frac{dX}{dt} &= \mu(S) \cdot X - D \cdot X
\end{aligned}$$

where:

| Parameter       | Description                  | Value       |
|-----------------|------------------------------|-------------|
| $D$             | Dilution rate                | 0.3 h⁻¹     |
| $S_{\text{in}}$ | Feed substrate concentration | 10 g/L      |
| $Y$             | Yield coefficient            | 0.5 g/g     |
| $\mu(S)$        | Specific growth rate         | **Unknown** |

The true growth kinetics follow Monod:
$\mu(S) = \frac{\mu_{\max} S}{K_s + S} = \frac{S}{2 + S}$ with
$\mu_{\max} = 1.0$ h⁻¹ and $K_s = 2.0$ g/L.

### Visualise Monod kinetics

``` julia
S_grid = range(0, 12, length=200)
μ_true = [S / (2.0 + S) for S in S_grid]

plot(S_grid, μ_true, lw=3, color=:black,
     xlabel="Substrate S (g/L)", ylabel="μ(S) (h⁻¹)",
     title="True Monod growth kinetics",
     label="μ(S) = S/(2+S)", legend=:bottomright)
hline!([1.0], ls=:dot, color=:gray, label="μmax = 1.0")
vline!([2.0], ls=:dot, color=:gray, label="Ks = 2.0")
```

![](10_chemostat_files/figure-commonmark/cell-3-output-1.svg)

### Generate data

We simulate a chemostat experiment: starting with high substrate and low
biomass, the system approaches a steady state as the microbes consume
the substrate.

``` julia
function chemo_true!(du, u, p, t)
    S, X = u
    μ = 1.0 * S / (2.0 + S)
    du[1] = 0.3 * (10.0 - S) - μ * X / 0.5
    du[2] = μ * X - 0.3 * X
end

u0 = [10.0, 0.5]
tspan = (0.0, 30.0)
sol_ode = OrdinaryDiffEq.solve(ODEProblem(chemo_true!, u0, tspan), Tsit5(), saveat=0.5)

data_t = sol_ode.t
σ_S, σ_X = 0.3, 0.1
data = max.(hcat(sol_ode[1,:], sol_ode[2,:]) .+
            hcat(σ_S .* randn(length(data_t)), σ_X .* randn(length(data_t))), 0.01)

p1 = plot(sol_ode.t, sol_ode[1,:], label="True S", lw=2, color=:purple, ls=:dash)
scatter!(p1, data_t, data[:, 1], label="S (obs)", ms=3, alpha=0.6, color=:purple)
p2 = plot(sol_ode.t, sol_ode[2,:], label="True X", lw=2, color=:teal, ls=:dash)
scatter!(p2, data_t, data[:, 2], label="X (obs)", ms=3, alpha=0.6, color=:teal)
plot(p1, p2, layout=(1, 2), size=(800, 350),
     xlabel="Time (h)", ylabel="Concentration (g/L)")
```

![](10_chemostat_files/figure-commonmark/cell-4-output-1.svg)

## Define and Fit the PSM

``` julia
function chemostat!(du, u, p, t)
    S, X = u
    μ_val = p.μ(max(S, 0.01))
    du[1] = 0.3 * (10.0 - S) - max(μ_val, 0.0) * X / 0.5
    du[2] = max(μ_val, 0.0) * X - 0.3 * X
end

approx_μ = BSplineApproximator(:μ, (0.0, 12.0), 8; initial=S -> 0.5*S/(2.0+S))

prob = PSMProblem(chemostat!, u0, tspan, [approx_μ];
    data_times=data_t, data_values=data,
    obs_to_state=[1, 2],
    known_params=(D=0.3, Sin=10.0, Y=0.5),
    solver=Tsit5())
```

    PSMProblem{typeof(chemostat!), Vector{Float64}, Gaussian, Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}}(chemostat!, [10.0, 0.5], (0.0, 30.0), BSplineApproximator[BSplineApproximator(:μ, (0.0, 12.0), 8, var"#5#6"())], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5  …  25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0], [9.890992755564467 0.49480188331659125; 9.632637153300509 0.6972316752750717; … ; 0.8887669549405476 4.425115965910029; 0.5675476653902087 4.555125423173031], [1.0 1.0; 1.0 1.0; … ; 1.0 1.0; 1.0 1.0], [1, 2], (D = 0.3, Sin = 10.0, Y = 0.5), Gaussian(), Tsit5{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}(OrdinaryDiffEqCore.trivial_limiter!, OrdinaryDiffEqCore.trivial_limiter!, static(false)), Dict{Symbol, Any}(), false, Float64[], nothing)

### LAML fit

    IRLS+LAML: 8 params, 122 data, 1 smooth terms
    Initial θ: [3.658e-5]
    Iter 0: obj=722.853, SS=1445.69, θ=[3.66e-5]
    Iter 1: obj=667.295, SS=1334.55, θ=[3.66e-5]
    Iter 2: obj=617.141, SS=1234.26, θ=[3.66e-5]
    Iter 3: obj=575.415, SS=1150.82, θ=[3.66e-5]
    LAML init: ρ = [0.0]
    LAML-FS iter 1: σ̂²=1.265e+01 λ = [0.05475]
    LAML-FS iter 2: σ̂²=9.609e+00 λ = [0.07487]
    LAML-FS iter 3: σ̂²=9.674e+00 λ = [0.07225]
    LAML-FS iter 4: σ̂²=9.665e+00 λ = [0.07254]
    LAML-FS iter 5: σ̂²=9.666e+00 λ = [0.07251]
    LAML-FS iter 8: σ̂²=9.666e+00 λ = [0.07251]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-1.507829e+02 |grad|=1.134e-07
    Iter 4: obj=543.815, SS=1061.69, θ=[0.0725]
    LAML init: ρ = [-2.624]
    LAML-FS iter 1: σ̂²=8.915e+00 λ = [0.1097]
    LAML-FS iter 2: σ̂²=9.024e+00 λ = [0.1049]
    LAML-FS iter 3: σ̂²=9.010e+00 λ = [0.1054]
    LAML-FS iter 4: σ̂²=9.011e+00 λ = [0.1053]
    LAML-FS iter 5: σ̂²=9.011e+00 λ = [0.1053]
    LAML-FS iter 7: σ̂²=9.011e+00 λ = [0.1053]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-1.548572e+02 |grad|=1.379e-07
    LAML init: ρ = [-2.251]
    LAML-FS iter 1: σ̂²=8.381e+00 λ = [0.06986]
    LAML-FS iter 2: σ̂²=8.290e+00 λ = [0.07422]
    LAML-FS iter 3: σ̂²=8.301e+00 λ = [0.07358]
    LAML-FS iter 4: σ̂²=8.299e+00 λ = [0.07368]
    LAML-FS iter 5: σ̂²=8.300e+00 λ = [0.07366]
    LAML-FS iter 8: σ̂²=8.300e+00 λ = [0.07366]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-1.391420e+02 |grad|=1.046e-07
    LAML init: ρ = [-2.608]
    LAML-FS iter 1: σ̂²=7.280e+00 λ = [0.05497]
    LAML-FS iter 2: σ̂²=7.240e+00 λ = [0.05864]
    LAML-FS iter 3: σ̂²=7.248e+00 λ = [0.05782]
    LAML-FS iter 4: σ̂²=7.246e+00 λ = [0.058]
    LAML-FS iter 5: σ̂²=7.246e+00 λ = [0.05796]
    LAML-FS iter 10: σ̂²=7.246e+00 λ = [0.05797]
    LAML-FS converged at iteration 10
    LAML-Newton iter 1: V=-1.299039e+02 |grad|=7.481e-08
    LAML init: ρ = [-2.848]
    LAML-FS iter 1: σ̂²=6.427e+00 λ = [0.0789]
    LAML-FS iter 2: σ̂²=6.463e+00 λ = [0.07476]
    LAML-FS iter 3: σ̂²=6.455e+00 λ = [0.07548]
    LAML-FS iter 4: σ̂²=6.457e+00 λ = [0.07536]
    LAML-FS iter 5: σ̂²=6.456e+00 λ = [0.07538]
    LAML-FS iter 9: σ̂²=6.456e+00 λ = [0.07537]
    LAML-FS converged at iteration 9
    LAML-Newton iter 1: V=-1.256018e+02 |grad|=6.649e-08
    LAML init: ρ = [-2.585]
    LAML-FS iter 1: σ̂²=5.861e+00 λ = [0.05086]
    LAML-FS iter 2: σ̂²=5.822e+00 λ = [0.05695]
    LAML-FS iter 3: σ̂²=5.832e+00 λ = [0.05512]
    LAML-FS iter 4: σ̂²=5.829e+00 λ = [0.05564]
    LAML-FS iter 5: σ̂²=5.830e+00 λ = [0.05549]
    LAML-FS iter 10: σ̂²=5.830e+00 λ = [0.05553]
    LAML-FS iter 12: σ̂²=5.830e+00 λ = [0.05553]
    LAML-FS converged at iteration 12
    LAML-Newton iter 1: V=-1.159811e+02 |grad|=1.161e-07
    LAML init: ρ = [-2.891]
    LAML-FS iter 1: σ̂²=4.977e+00 λ = [0.05534]
    LAML-FS iter 2: σ̂²=4.977e+00 λ = [0.05537]
    LAML-FS iter 3: σ̂²=4.977e+00 λ = [0.05537]
    LAML-FS iter 4: σ̂²=4.977e+00 λ = [0.05537]
    LAML-FS iter 5: σ̂²=4.977e+00 λ = [0.05537]
    LAML-FS iter 6: σ̂²=4.977e+00 λ = [0.05537]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=-1.044219e+02 |grad|=6.904e-08
    Iter 10: obj=153.006, SS=303.1, θ=[0.0554]
    LAML init: ρ = [-2.894]
    LAML-FS iter 1: σ̂²=2.508e+00 λ = [0.05273]
    LAML-FS iter 2: σ̂²=2.507e+00 λ = [0.05312]
    LAML-FS iter 3: σ̂²=2.507e+00 λ = [0.05306]
    LAML-FS iter 4: σ̂²=2.507e+00 λ = [0.05307]
    LAML-FS iter 5: σ̂²=2.507e+00 λ = [0.05307]
    LAML-FS iter 7: σ̂²=2.507e+00 λ = [0.05307]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-6.299206e+01 |grad|=4.869e-08
    LAML init: ρ = [-2.936]
    LAML-FS iter 1: σ̂²=1.150e+00 λ = [0.004162]
    LAML-FS iter 2: σ̂²=1.035e+00 λ = [0.006011]
    LAML-FS iter 3: σ̂²=1.040e+00 λ = [0.005552]
    LAML-FS iter 4: σ̂²=1.039e+00 λ = [0.005647]
    LAML-FS iter 5: σ̂²=1.039e+00 λ = [0.005627]
    LAML-FS iter 10: σ̂²=1.039e+00 λ = [0.00563]
    LAML-FS iter 11: σ̂²=1.039e+00 λ = [0.00563]
    LAML-FS converged at iteration 11
    LAML-Newton iter 1: V=-1.145035e+01 |grad|=5.697e-08
    LAML init: ρ = [-5.18]
    LAML-FS iter 1: σ̂²=5.215e-01 λ = [0.001331]
    LAML-FS iter 2: σ̂²=4.982e-01 λ = [0.001726]
    LAML-FS iter 3: σ̂²=5.004e-01 λ = [0.001644]
    LAML-FS iter 4: σ̂²=4.999e-01 λ = [0.001659]
    LAML-FS iter 5: σ̂²=5.000e-01 λ = [0.001656]
    LAML-FS iter 10: σ̂²=5.000e-01 λ = [0.001657]
    LAML-FS converged at iteration 10
    LAML-Newton iter 1: V=3.116338e+01 |grad|=8.014e-08
    LAML init: ρ = [-6.403]
    LAML-FS iter 1: σ̂²=1.714e-01 λ = [0.001163]
    LAML-FS iter 2: σ̂²=1.695e-01 λ = [0.001221]
    LAML-FS iter 3: σ̂²=1.698e-01 λ = [0.001212]
    LAML-FS iter 4: σ̂²=1.697e-01 λ = [0.001213]
    LAML-FS iter 5: σ̂²=1.697e-01 λ = [0.001213]
    LAML-FS iter 8: σ̂²=1.697e-01 λ = [0.001213]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=9.311779e+01 |grad|=7.977e-08
    LAML init: ρ = [-6.714]
    LAML-FS iter 1: σ̂²=7.298e-02 λ = [0.0006461]
    LAML-FS iter 2: σ̂²=7.134e-02 λ = [0.0007033]
    LAML-FS iter 3: σ̂²=7.150e-02 λ = [0.0006954]
    LAML-FS iter 4: σ̂²=7.148e-02 λ = [0.0006964]
    LAML-FS iter 5: σ̂²=7.148e-02 λ = [0.0006963]
    LAML-FS iter 8: σ̂²=7.148e-02 λ = [0.0006963]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=1.426717e+02 |grad|=1.088e-07
    LAML init: ρ = [-7.27]
    LAML-FS iter 1: σ̂²=5.456e-02 λ = [0.0005929]
    LAML-FS iter 2: σ̂²=5.426e-02 λ = [0.0006038]
    LAML-FS iter 3: σ̂²=5.429e-02 λ = [0.0006025]
    LAML-FS iter 4: σ̂²=5.429e-02 λ = [0.0006027]
    LAML-FS iter 5: σ̂²=5.429e-02 λ = [0.0006027]
    LAML-FS iter 7: σ̂²=5.429e-02 λ = [0.0006027]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.604398e+02 |grad|=7.583e-08
    LAML init: ρ = [-7.414]
    LAML-FS iter 1: σ̂²=4.375e-02 λ = [0.0002691]
    LAML-FS iter 2: σ̂²=4.191e-02 λ = [0.000285]
    LAML-FS iter 3: σ̂²=4.200e-02 λ = [0.0002837]
    LAML-FS iter 4: σ̂²=4.199e-02 λ = [0.0002838]
    LAML-FS iter 5: σ̂²=4.199e-02 λ = [0.0002838]
    LAML-FS iter 7: σ̂²=4.199e-02 λ = [0.0002838]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.721222e+02 |grad|=2.353e-08
    LAML init: ρ = [-8.167]
    LAML-FS iter 1: σ̂²=4.015e-02 λ = [0.0002511]
    LAML-FS iter 2: σ̂²=3.995e-02 λ = [0.0002531]
    LAML-FS iter 3: σ̂²=3.996e-02 λ = [0.000253]
    LAML-FS iter 4: σ̂²=3.996e-02 λ = [0.000253]
    LAML-FS iter 5: σ̂²=3.996e-02 λ = [0.000253]
    LAML-FS iter 6: σ̂²=3.996e-02 λ = [0.000253]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=1.765366e+02 |grad|=2.229e-08
    LAML init: ρ = [-8.282]
    LAML-FS iter 1: σ̂²=3.971e-02 λ = [0.0002841]
    LAML-FS iter 2: σ̂²=3.990e-02 λ = [0.0002834]
    LAML-FS iter 3: σ̂²=3.990e-02 λ = [0.0002834]
    LAML-FS iter 4: σ̂²=3.990e-02 λ = [0.0002834]
    LAML-FS iter 5: σ̂²=3.990e-02 λ = [0.0002834]
    LAML-FS converged at iteration 5
    LAML-Newton iter 1: V=1.676087e+02 |grad|=1.804e-09
    LAML init: ρ = [-8.169]
    LAML-FS iter 1: σ̂²=3.990e-02 λ = [0.0002336]
    LAML-FS iter 2: σ̂²=3.958e-02 λ = [0.0002377]
    LAML-FS iter 3: σ̂²=3.961e-02 λ = [0.0002373]
    LAML-FS iter 4: σ̂²=3.961e-02 λ = [0.0002374]
    LAML-FS iter 5: σ̂²=3.961e-02 λ = [0.0002374]
    LAML-FS iter 7: σ̂²=3.961e-02 λ = [0.0002374]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.767080e+02 |grad|=2.338e-08
    Iter 20: obj=2.41438, SS=4.647, θ=[0.000237]
    LAML init: ρ = [-8.346]
    LAML-FS iter 1: σ̂²=3.958e-02 λ = [0.0002068]
    LAML-FS iter 2: σ̂²=3.939e-02 λ = [0.0002099]
    LAML-FS iter 3: σ̂²=3.941e-02 λ = [0.0002095]
    LAML-FS iter 4: σ̂²=3.941e-02 λ = [0.0002096]
    LAML-FS iter 5: σ̂²=3.941e-02 λ = [0.0002096]
    LAML-FS iter 7: σ̂²=3.941e-02 λ = [0.0002096]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.794540e+02 |grad|=5.469e-08
    LAML init: ρ = [-8.47]
    LAML-FS iter 1: σ̂²=3.778e-02 λ = [0.0009419]
    LAML-FS iter 2: σ̂²=3.892e-02 λ = [0.0007965]
    LAML-FS iter 3: σ̂²=3.869e-02 λ = [0.0008114]
    LAML-FS iter 4: σ̂²=3.872e-02 λ = [0.0008097]
    LAML-FS iter 5: σ̂²=3.871e-02 λ = [0.0008099]
    LAML-FS iter 8: σ̂²=3.871e-02 λ = [0.0008099]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=1.804786e+02 |grad|=6.923e-08
    LAML init: ρ = [-7.119]
    LAML-FS iter 1: σ̂²=3.871e-02 λ = [0.0008233]
    LAML-FS iter 2: σ̂²=3.873e-02 λ = [0.0008218]
    LAML-FS iter 3: σ̂²=3.873e-02 λ = [0.000822]
    LAML-FS iter 4: σ̂²=3.873e-02 λ = [0.0008219]
    LAML-FS iter 5: σ̂²=3.873e-02 λ = [0.0008219]
    LAML-FS iter 6: σ̂²=3.873e-02 λ = [0.0008219]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=1.785994e+02 |grad|=5.816e-08
    LAML init: ρ = [-7.104]
    LAML-FS iter 1: σ̂²=3.874e-02 λ = [0.000794]
    LAML-FS iter 2: σ̂²=3.869e-02 λ = [0.000797]
    LAML-FS iter 3: σ̂²=3.870e-02 λ = [0.0007967]
    LAML-FS iter 4: σ̂²=3.870e-02 λ = [0.0007967]
    LAML-FS iter 5: σ̂²=3.870e-02 λ = [0.0007967]
    LAML-FS iter 6: σ̂²=3.870e-02 λ = [0.0007967]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=1.809265e+02 |grad|=1.092e-07
    LAML init: ρ = [-7.135]
    LAML-FS iter 1: σ̂²=3.869e-02 λ = [0.0008298]
    LAML-FS iter 2: σ̂²=3.874e-02 λ = [0.0008259]
    LAML-FS iter 3: σ̂²=3.873e-02 λ = [0.0008264]
    LAML-FS iter 4: σ̂²=3.874e-02 λ = [0.0008263]
    LAML-FS iter 5: σ̂²=3.874e-02 λ = [0.0008263]
    LAML-FS iter 6: σ̂²=3.874e-02 λ = [0.0008263]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=1.800745e+02 |grad|=2.094e-07
    Converged at iter 25 (objective stable)

    Final: data_loss = 4.56995, penalty = 0.150206, EDF = 6.04
    Final θ: [0.0007967]
    Data loss: 4.57
    EDF: 6.04

### RodeoSolver fit (with uncertainty)

    RodeoSolver: n_steps=200, n_deriv=3, method=basic, interrogate=kramer
      σ (IBM scale): [0.095, 0.0432]
      obs_var: 0.0357
      8 approximator params

    Stage 1: Nelder-Mead (derivative-free)...
    Iter     Function value    √(Σ(yᵢ-ȳ)²)/n 
    ------   --------------    --------------
         0     1.258035e+04     2.868692e+03
     * time: 0.012192964553833008
        40     3.101931e+00     5.604010e+01
     * time: 0.17432498931884766
        80    -1.629488e+01     1.630722e+00
     * time: 0.2613968849182129
       120    -2.070186e+01     3.235583e-01
     * time: 0.34677886962890625
       160    -2.265674e+01     1.214233e-01
     * time: 0.43083691596984863
       200    -2.306034e+01     2.151290e-02
     * time: 0.5047180652618408
      NM loss: -23.075

    Stage 2: L-BFGS refinement...
    Iter     Function value   Gradient norm 
         0    -2.307532e+01     1.958825e+01
     * time: 6.29425048828125e-5
        20    -2.624374e+01     9.067674e+00
     * time: 0.6796619892120361
        40    -2.726207e+01     1.226153e+01
     * time: 1.1329610347747803
      Converged: true
      Final -loglik: -27.333
      FS cycle 1: λ = [3.54]
      FS cycle 2: λ = [4.16]
      FS cycle 3: λ = [4.68]
      Final λ after FS: [4.68]

    Final: data_SS=4.5525 -loglik=-27.333
    Data loss: 4.55

## Results

### Fitted trajectories

``` julia
p1 = plot(sol_ode.t, sol_ode[1,:], label="True", lw=2, color=:black, ls=:dash,
          xlabel="Time (h)", ylabel="S (g/L)", title="Substrate")
scatter!(p1, data_t, data[:, 1], label="Data", ms=3, alpha=0.5, color=:gray)
plot!(p1, data_t, sol_laml.fitted_values[:, 1], label="LAML", lw=2, color=:blue)
plot!(p1, data_t, sol_rodeo.fitted_values[:, 1], label="Rodeo", lw=2, color=:green)

p2 = plot(sol_ode.t, sol_ode[2,:], label="True", lw=2, color=:black, ls=:dash,
          xlabel="Time (h)", ylabel="X (g/L)", title="Biomass")
scatter!(p2, data_t, data[:, 2], label="Data", ms=3, alpha=0.5, color=:gray)
plot!(p2, data_t, sol_laml.fitted_values[:, 2], label="LAML", lw=2, color=:blue)
plot!(p2, data_t, sol_rodeo.fitted_values[:, 2], label="Rodeo", lw=2, color=:green)

plot(p1, p2, layout=(1, 2), size=(800, 350))
```

![](10_chemostat_files/figure-commonmark/cell-8-output-1.svg)

### Recovered growth kinetics

``` julia
S_eval = range(0.5, 10.0, length=100)
μ_true_vals = [S / (2.0 + S) for S in S_eval]
μ_laml = [sol_laml.unknown_functions[:μ](S) for S in S_eval]
μ_rodeo = [sol_rodeo.unknown_functions[:μ](S) for S in S_eval]

plot(S_eval, μ_true_vals, label="True μ(S) = S/(2+S)", lw=3, color=:black, ls=:dash,
     xlabel="Substrate S (g/L)", ylabel="Specific growth rate μ(S) (h⁻¹)",
     title="Recovered growth kinetics", legend=:bottomright)
plot!(S_eval, μ_laml, label="LAML", lw=2, color=:blue)
plot!(S_eval, μ_rodeo, label="RodeoSolver", lw=2, color=:green)
hline!([1.0], ls=:dot, color=:gray, alpha=0.5, label="μmax")
```

![](10_chemostat_files/figure-commonmark/cell-9-output-1.svg)

Both solvers recover the saturating Monod curve without assuming any
parametric form. The key features are correctly identified:

1.  **Linear increase** at low substrate concentrations (first-order
    kinetics)
2.  **Saturation** approaching $\mu_{\max}$ at high concentrations
3.  **Half-saturation** correctly located near $K_s = 2$ g/L

## Residual Diagnostics

Good model fit can be verified using the built-in residual diagnostics:

    Residual diagnostics (LAML):
      RMSE S: 0.2528
      RMSE X: 0.1049
      Durbin-Watson: [1.352, 1.616]

``` julia
acf = diag.acf
nlags = size(acf, 1) - 1
p_acf = bar(0:nlags, acf[:, 1], label="S residual ACF", alpha=0.7, color=:purple,
            xlabel="Lag", ylabel="Autocorrelation",
            title="Residual autocorrelation")
bar!(0:nlags, acf[:, 2], label="X residual ACF", alpha=0.7, color=:teal)
hline!([1.96 / sqrt(length(data_t)), -1.96 / sqrt(length(data_t))],
       ls=:dash, color=:gray, label="95% CI")
p_acf
```

![](10_chemostat_files/figure-commonmark/cell-11-output-1.svg)

## Substrate Inhibition: What If Monod Is Wrong?

In many industrial fermentations, high substrate concentrations actually
**inhibit** growth. Let’s see what happens when the true kinetics
include inhibition:

``` julia
# Haldane kinetics: μ(S) = μmax*S / (Ks + S + S²/Ki)
function chemo_haldane!(du, u, p, t)
    S, X = u
    μ = 1.0 * S / (2.0 + S + S^2 / 15.0)  # Ki = 15
    du[1] = 0.3 * (10.0 - S) - μ * X / 0.5
    du[2] = μ * X - 0.3 * X
end

Random.seed!(123)
sol_haldane = OrdinaryDiffEq.solve(ODEProblem(chemo_haldane!, u0, tspan), Tsit5(), saveat=0.5)
data_haldane = max.(hcat(sol_haldane[1,:], sol_haldane[2,:]) .+
                    hcat(0.3 .* randn(length(sol_haldane.t)), 0.1 .* randn(length(sol_haldane.t))), 0.01)

prob_h = PSMProblem(chemostat!, u0, tspan,
    [BSplineApproximator(:μ, (0.0, 12.0), 10; initial=S -> 0.3*S/(2.0+S))];
    data_times=sol_haldane.t, data_values=data_haldane,
    obs_to_state=[1, 2], known_params=(D=0.3, Sin=10.0, Y=0.5), solver=Tsit5())

sol_h = solve(prob_h, LAML(maxiters=200, verbose=false))

S_eval_h = range(0.5, 10.0, length=100)
μ_haldane = [S / (2.0 + S + S^2 / 15.0) for S in S_eval_h]
μ_est_h = [sol_h.unknown_functions[:μ](S) for S in S_eval_h]

plot(S_eval_h, μ_haldane, label="True (Haldane)", lw=3, color=:black, ls=:dash,
     xlabel="Substrate S (g/L)", ylabel="μ(S) (h⁻¹)",
     title="Detecting substrate inhibition", legend=:topright)
plot!(S_eval_h, μ_est_h, label="PSM estimate", lw=2, color=:red)
```

![](10_chemostat_files/figure-commonmark/cell-12-output-1.svg)

The PSM successfully detects the **non-monotonic** growth kinetics — the
specific growth rate increases, peaks, and then declines — without any
assumption about inhibition.

## Diagnostic Plots

A standard 4-panel diagnostic display assesses residual behaviour. The
QQ plot checks normality of standardized residuals, “Residuals vs
Fitted” detects systematic patterns, the histogram visualises the
residual distribution, and “Observed vs Fitted” checks overall
calibration.

``` julia
using PartiallySpecifiedModels: appraise

diag = appraise(sol_laml)

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

![](10_chemostat_files/figure-commonmark/cell-13-output-1.svg)

    Durbin-Watson: 1.352, 1.616

## Key Takeaways

1.  **Chemostat dynamics** are a natural fit for PSMs — the growth
    kinetics are the main unknown
2.  **Both LAML and RodeoSolver** accurately recover Monod kinetics from
    noisy time series
3.  **Non-standard kinetics** (e.g., substrate inhibition) are detected
    automatically
4.  **Residual diagnostics** help verify model adequacy and detect
    systematic departures
