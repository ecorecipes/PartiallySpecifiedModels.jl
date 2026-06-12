# Chemostat Dynamics: Recovering Microbial Growth Kinetics
Simon Frost
2026-06-12

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
    LAML-FS iter 1: σ̂²=1.286e+01 λ = [0.05567]
    LAML-FS iter 2: σ̂²=9.772e+00 λ = [0.07598]
    LAML-FS iter 3: σ̂²=9.839e+00 λ = [0.07333]
    LAML-FS iter 4: σ̂²=9.830e+00 λ = [0.07363]
    LAML-FS iter 5: σ̂²=9.831e+00 λ = [0.07359]
    LAML-FS iter 8: σ̂²=9.831e+00 λ = [0.0736]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-1.517745e+02 |grad|=1.126e-07
    Iter 4: obj=543.711, SS=1061.1, θ=[0.0736]
    LAML init: ρ = [-2.609]
    LAML-FS iter 1: σ̂²=9.062e+00 λ = [0.07875]
    LAML-FS iter 2: σ̂²=9.077e+00 λ = [0.07798]
    LAML-FS iter 3: σ̂²=9.075e+00 λ = [0.07809]
    LAML-FS iter 4: σ̂²=9.075e+00 λ = [0.07808]
    LAML-FS iter 5: σ̂²=9.075e+00 λ = [0.07808]
    LAML-FS iter 7: σ̂²=9.075e+00 λ = [0.07808]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-1.469758e+02 |grad|=1.503e-07
    LAML init: ρ = [-2.55]
    LAML-FS iter 1: σ̂²=8.594e+00 λ = [0.07529]
    LAML-FS iter 2: σ̂²=8.586e+00 λ = [0.07574]
    LAML-FS iter 3: σ̂²=8.587e+00 λ = [0.07567]
    LAML-FS iter 4: σ̂²=8.587e+00 λ = [0.07568]
    LAML-FS iter 5: σ̂²=8.587e+00 λ = [0.07568]
    LAML-FS iter 7: σ̂²=8.587e+00 λ = [0.07568]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-1.420746e+02 |grad|=1.799e-07
    LAML init: ρ = [-2.581]
    LAML-FS iter 1: σ̂²=8.054e+00 λ = [0.07506]
    LAML-FS iter 2: σ̂²=8.053e+00 λ = [0.07516]
    LAML-FS iter 3: σ̂²=8.053e+00 λ = [0.07515]
    LAML-FS iter 4: σ̂²=8.053e+00 λ = [0.07515]
    LAML-FS iter 5: σ̂²=8.053e+00 λ = [0.07515]
    LAML-FS iter 6: σ̂²=8.053e+00 λ = [0.07515]
    LAML-FS converged at iteration 6
    LAML-Newton iter 1: V=-1.407447e+02 |grad|=2.016e-07
    LAML init: ρ = [-2.588]
    LAML-FS iter 1: σ̂²=7.542e+00 λ = [0.07314]
    LAML-FS iter 2: σ̂²=7.537e+00 λ = [0.07346]
    LAML-FS iter 3: σ̂²=7.538e+00 λ = [0.07341]
    LAML-FS iter 4: σ̂²=7.538e+00 λ = [0.07342]
    LAML-FS iter 5: σ̂²=7.538e+00 λ = [0.07342]
    LAML-FS iter 7: σ̂²=7.538e+00 λ = [0.07342]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-1.362850e+02 |grad|=1.076e-07
    LAML init: ρ = [-2.612]
    LAML-FS iter 1: σ̂²=7.024e+00 λ = [0.06913]
    LAML-FS iter 2: σ̂²=7.014e+00 λ = [0.0699]
    LAML-FS iter 3: σ̂²=7.016e+00 λ = [0.06976]
    LAML-FS iter 4: σ̂²=7.016e+00 λ = [0.06979]
    LAML-FS iter 5: σ̂²=7.016e+00 λ = [0.06978]
    LAML-FS iter 8: σ̂²=7.016e+00 λ = [0.06978]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-1.294832e+02 |grad|=1.080e-07
    LAML init: ρ = [-2.662]
    LAML-FS iter 1: σ̂²=6.545e+00 λ = [0.06721]
    LAML-FS iter 2: σ̂²=6.540e+00 λ = [0.06764]
    LAML-FS iter 3: σ̂²=6.541e+00 λ = [0.06756]
    LAML-FS iter 4: σ̂²=6.541e+00 λ = [0.06758]
    LAML-FS iter 5: σ̂²=6.541e+00 λ = [0.06757]
    LAML-FS iter 7: σ̂²=6.541e+00 λ = [0.06758]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=-1.267624e+02 |grad|=1.934e-07
    Iter 10: obj=363.38, SS=710.756, θ=[0.0676]
    LAML init: ρ = [-2.695]
    LAML-FS iter 1: σ̂²=6.056e+00 λ = [0.05642]
    LAML-FS iter 2: σ̂²=6.034e+00 λ = [0.05822]
    LAML-FS iter 3: σ̂²=6.038e+00 λ = [0.0579]
    LAML-FS iter 4: σ̂²=6.037e+00 λ = [0.05796]
    LAML-FS iter 5: σ̂²=6.037e+00 λ = [0.05795]
    LAML-FS iter 8: σ̂²=6.037e+00 λ = [0.05795]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-1.191912e+02 |grad|=1.674e-07
    LAML init: ρ = [-2.848]
    LAML-FS iter 1: σ̂²=5.298e+00 λ = [0.03947]
    LAML-FS iter 2: σ̂²=5.271e+00 λ = [0.04462]
    LAML-FS iter 3: σ̂²=5.279e+00 λ = [0.04303]
    LAML-FS iter 4: σ̂²=5.277e+00 λ = [0.0435]
    LAML-FS iter 5: σ̂²=5.277e+00 λ = [0.04336]
    LAML-FS iter 10: σ̂²=5.277e+00 λ = [0.04339]
    LAML-FS iter 12: σ̂²=5.277e+00 λ = [0.04339]
    LAML-FS converged at iteration 12
    LAML-Newton iter 1: V=-1.063106e+02 |grad|=1.592e-07
    LAML init: ρ = [-3.137]
    LAML-FS iter 1: σ̂²=2.129e+00 λ = [0.04943]
    LAML-FS iter 2: σ̂²=2.132e+00 λ = [0.04826]
    LAML-FS iter 3: σ̂²=2.131e+00 λ = [0.04847]
    LAML-FS iter 4: σ̂²=2.131e+00 λ = [0.04843]
    LAML-FS iter 5: σ̂²=2.131e+00 λ = [0.04844]
    LAML-FS iter 8: σ̂²=2.131e+00 λ = [0.04844]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=-5.265434e+01 |grad|=1.229e-07
    LAML init: ρ = [-3.028]
    LAML-FS iter 1: σ̂²=1.014e-01 λ = [0.002302]
    LAML-FS iter 2: σ̂²=7.539e-02 λ = [0.003093]
    LAML-FS iter 3: σ̂²=7.584e-02 λ = [0.002968]
    LAML-FS iter 4: σ̂²=7.577e-02 λ = [0.002985]
    LAML-FS iter 5: σ̂²=7.578e-02 λ = [0.002982]
    LAML-FS iter 9: σ̂²=7.578e-02 λ = [0.002983]
    LAML-FS converged at iteration 9
    LAML-Newton iter 1: V=1.418413e+02 |grad|=5.583e-08
    LAML init: ρ = [-5.815]
    LAML-FS iter 1: σ̂²=5.395e-02 λ = [0.002359]
    LAML-FS iter 2: σ̂²=5.361e-02 λ = [0.002424]
    LAML-FS iter 3: σ̂²=5.365e-02 λ = [0.002416]
    LAML-FS iter 4: σ̂²=5.364e-02 λ = [0.002417]
    LAML-FS iter 5: σ̂²=5.364e-02 λ = [0.002417]
    LAML-FS iter 7: σ̂²=5.364e-02 λ = [0.002417]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.614956e+02 |grad|=9.997e-08
    LAML init: ρ = [-6.025]
    LAML-FS iter 1: σ̂²=4.474e-02 λ = [0.001168]
    LAML-FS iter 2: σ̂²=4.351e-02 λ = [0.001262]
    LAML-FS iter 3: σ̂²=4.360e-02 λ = [0.001251]
    LAML-FS iter 4: σ̂²=4.359e-02 λ = [0.001252]
    LAML-FS iter 5: σ̂²=4.359e-02 λ = [0.001252]
    LAML-FS iter 8: σ̂²=4.359e-02 λ = [0.001252]
    LAML-FS converged at iteration 8
    LAML-Newton iter 1: V=1.715250e+02 |grad|=2.631e-08
    LAML init: ρ = [-6.683]
    LAML-FS iter 1: σ̂²=4.065e-02 λ = [0.0008575]
    LAML-FS iter 2: σ̂²=4.006e-02 λ = [0.0008935]
    LAML-FS iter 3: σ̂²=4.012e-02 λ = [0.0008895]
    LAML-FS iter 4: σ̂²=4.011e-02 λ = [0.0008899]
    LAML-FS iter 5: σ̂²=4.011e-02 λ = [0.0008899]
    LAML-FS iter 7: σ̂²=4.011e-02 λ = [0.0008899]
    LAML-FS converged at iteration 7
    LAML-Newton iter 1: V=1.710570e+02 |grad|=1.430e-07
    Converged at iter 17 (objective stable)

    Final: data_loss = 4.65533, penalty = 0.222401, EDF = 5.72
    Final θ: [0.001252]
    Data loss: 4.66
    EDF: 5.72

### RodeoSolver fit (with uncertainty)

    RodeoSolver: n_steps=200, n_deriv=3, method=basic, interrogate=kramer
      σ (IBM scale): [0.095, 0.0432]
      obs_var: 0.0357
      8 approximator params

    Stage 1: Nelder-Mead (derivative-free)...
    Iter     Function value    √(Σ(yᵢ-ȳ)²)/n 
    ------   --------------    --------------
         0     1.257937e+04     2.869004e+03
     * time: 0.01408696174621582
        40     3.312956e+00     5.600080e+01
     * time: 0.6055347919464111
        80    -1.476239e+01     1.415984e+00
     * time: 0.6924328804016113
       120    -2.216158e+01     4.058614e-01
     * time: 0.7925848960876465
       160    -2.316704e+01     4.087223e-02
     * time: 0.8776278495788574
       200    -2.339729e+01     1.731839e-02
     * time: 0.9714579582214355
      NM loss: -23.397

    Stage 2: L-BFGS refinement...
    Iter     Function value   Gradient norm 
         0    -2.339729e+01     2.423065e+01
     * time: 6.985664367675781e-5
        20    -2.632475e+01     2.222920e+00
     * time: 0.6870338916778564
        40    -2.725968e+01     4.498615e+00
     * time: 1.1474189758300781
        60    -2.731572e+01     2.133323e-02
     * time: 1.6098289489746094
      Converged: true
      Final -loglik: -27.316
      FS cycle 1: λ = [3.6]
      FS cycle 2: λ = [4.28]
      FS cycle 3: λ = [4.85]
      Final λ after FS: [4.85]

    Final: data_SS=4.5542 -loglik=-27.316
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
      RMSE S: 0.2558
      RMSE X: 0.1043
      Durbin-Watson: [1.322, 1.618]

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

    Durbin-Watson: 1.322, 1.618

## Key Takeaways

1.  **Chemostat dynamics** are a natural fit for PSMs — the growth
    kinetics are the main unknown
2.  **Both LAML and RodeoSolver** accurately recover Monod kinetics from
    noisy time series
3.  **Non-standard kinetics** (e.g., substrate inhibition) are detected
    automatically
4.  **Residual diagnostics** help verify model adequacy and detect
    systematic departures
