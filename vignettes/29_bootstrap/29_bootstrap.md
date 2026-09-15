# Bootstrap Confidence Intervals
Simon Frost
2026-09-15

- [Overview](#overview)
- [Setup](#setup)
- [The Model: SIR with Unknown Force of
  Infection](#the-model-sir-with-unknown-force-of-infection)
  - [Generate synthetic data](#generate-synthetic-data)
- [Section 1: Fit the PSM](#section-1-fit-the-psm)
  - [Fitted trajectory](#fitted-trajectory)
  - [Recovered unknown function](#recovered-unknown-function)
- [Section 2: Parametric Bootstrap](#section-2-parametric-bootstrap)
  - [Trajectory confidence intervals](#trajectory-confidence-intervals)
  - [Unknown function confidence
    intervals](#unknown-function-confidence-intervals)
- [Section 3: Nonparametric
  Bootstrap](#section-3-nonparametric-bootstrap)
  - [Trajectory confidence
    intervals](#trajectory-confidence-intervals-1)
  - [Unknown function confidence
    intervals](#unknown-function-confidence-intervals-1)
- [Section 4: Case Bootstrap](#section-4-case-bootstrap)
- [Section 5: Comparison of Bootstrap
  Methods](#section-5-comparison-of-bootstrap-methods)
  - [Side-by-side trajectory CIs](#side-by-side-trajectory-cis)
  - [Side-by-side unknown function
    CIs](#side-by-side-unknown-function-cis)
  - [Quantitative comparison](#quantitative-comparison)
- [Section 6: Bootstrap with Non-Gaussian
  Likelihoods](#section-6-bootstrap-with-non-gaussian-likelihoods)
  - [Generate Poisson data](#generate-poisson-data)
  - [Fit with Poisson likelihood](#fit-with-poisson-likelihood)
  - [Parametric bootstrap with Poisson
    likelihood](#parametric-bootstrap-with-poisson-likelihood)
- [Section 7: Diagnostic Plots](#section-7-diagnostic-plots)
- [Practical Guidance](#practical-guidance)
  - [Choosing a bootstrap method](#choosing-a-bootstrap-method)
  - [Tips](#tips)

## Overview

When fitting a partially specified model, point estimates of the unknown
function and fitted trajectories are rarely sufficient — we also need
**uncertainty quantification**. Bootstrap confidence intervals provide a
distribution-free (or distribution-aware) approach to estimating the
variability of both the fitted trajectories and the recovered unknown
functions.

`PartiallySpecifiedModels.jl` implements two bootstrap methods:

| Method | Description | Assumptions |
|----|----|----|
| `:parametric` | Simulate new data from the fitted likelihood (e.g., $N(\hat\mu, \hat\sigma)$ or $\text{Pois}(\hat\mu)$) | Correct likelihood family |
| `:nonparametric` | Resample residuals with replacement per state (Gaussian likelihood only) | Exchangeable residuals |

This vignette demonstrates both methods on an SIR epidemic model with a
nonparametric force of infection, compares the resulting confidence
intervals, and shows how the parametric bootstrap adapts to non-Gaussian
likelihoods.

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve, appraise
using OrdinaryDiffEq
using Plots
using Statistics
using Random
Random.seed!(7)
```

    TaskLocalRNG()

## The Model: SIR with Unknown Force of Infection

We consider an SIR epidemic model where the force of infection
$\lambda(I/N)$ is unknown. The true transmission follows a **power-law**
form:

$$\lambda(I/N) = \beta \left(\frac{I}{N}\right)^\alpha, \quad \beta = 0.5, \; \alpha = 0.9$$

This departs slightly from the standard mass-action
$\lambda = \beta \cdot I/N$ ($\alpha = 1$), which makes the
nonparametric recovery more interesting.

``` julia
function sir_true!(du, u, p, t)
    S, I, R = u
    N = 1000.0
    prev = I / N
    λ = 0.5 * prev^0.9
    du[1] = -λ * S
    du[2] =  λ * S - 0.25 * I
    du[3] =  0.25 * I
end
```

    sir_true! (generic function with 1 method)

### Generate synthetic data

We simulate the true model and observe $I(t)$ daily with Gaussian noise
($\sigma = 5$):

![](29_bootstrap_files/figure-commonmark/cell-4-output-1.svg)

The prevalence range determines the B-spline domain:

    Prevalence range: 0.005 – 0.2264
    B-spline domain: (0.0, 0.272)

## Section 1: Fit the PSM

We model $\lambda(I/N)$ with a shape-constrained B-spline (8 knots,
increasing with $\lambda(0) = 0$). The `inc_zero_left` constraint is
biologically motivated: the force of infection must be zero when there
are no infected individuals, and should increase with prevalence.

``` julia
function sir_psm!(du, u, p, t)
    S, I, R = u
    λ = p.λ(I / p.N)
    du[1] = -λ * S
    du[2] =  λ * S - p.γ * I
    du[3] =  p.γ * I
end
```

    sir_psm! (generic function with 1 method)

``` julia
approx_λ = ShapeConstrainedBSplineApproximator(:λ, foi_domain, 10,
    :inc_zero_left; initial = 0.4)

prob = PSMProblem(
    sir_psm!, u0, tspan, [approx_λ];
    data_times = data_times,
    data_values = reshape(I_obs, :, 1),
    obs_to_state = [2],
    known_params = (γ = 0.25, N = N_pop),
    likelihood = Gaussian(),
    solver = Tsit5()
)

sol = solve(prob, LAML(maxiters=100, verbose=false))
```

    PSMSolution((λ = [-3.45085705318858, -3.5590860444878976, -3.776742973515556, -3.9139147690008116, -3.9088836033041336, -3.81072440660502, -3.752630010227148, -3.7461194095364103, -3.7461194095364436]), 578.251482589099, 1100.24772067709, 2.945796738327827, [618.3115171665414], [10.0; 16.284206748645893; … ; 5.942005957949615; 5.048537052226073;;], [3.6696475510497084; 19.43055459411786; … ; 5.397424420239466; 10.498804562183398;;], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0  …  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0], Dict{Symbol, Any}(:λ => PartiallySpecifiedModels.var"#evaluator#build_constrained_bspline_evaluator##0"{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int64, Vector{Float64}, Vector{Float64}}(0.0306993726776959, 0.18997251162932013, 0.0306993726776959, 0.6005279237979639, 0.7629617964580723, 0.272, 0.0, 4, [-0.11657142857142856, -0.07771428571428571, -0.038857142857142854, 0.0, 0.038857142857142854, 0.07771428571428571, 0.11657142857142858, 0.15542857142857142, 0.19428571428571428, 0.23314285714285715, 0.272, 0.3108571428571429, 0.34971428571428576, 0.38857142857142857], [0.0, 0.031225800702057008, 0.05929303325794735, 0.08193197433330424, 0.10169754260328145, 0.12156182163048004, 0.143452601022485, 0.16663771225292218, 0.18997251162932027, 0.2133073110057176])), (V_beta = [0.0007556450409659419 -0.0002575240073422675 … 8.99321250057317e-7 8.993212496443135e-7; -0.0002575240073422675 0.00012761279269707154 … 3.1791340319679915e-5 3.179134030508008e-5; … ; 8.99321250057317e-7 3.1791340319679915e-5 … 0.0025469507505294134 0.0025469507493597536; 8.993212496443135e-7 3.179134030508008e-5 … 0.0025469507493597536 0.004164258483493708], sigma2 = 28.91264634057518, converged = true, iterations = 31, reason = :converged_tol, laml_failures = 0, criterion = :working, laml = -76.87967105803384, stationarity = 1.1911830243072075e-5, smoothing_advanced = true, smoothing_fixed = false, solver = :LAML, smoothing_state = (beta = [-3.45085705318858, -3.5590860444878976, -3.776742973515556, -3.9139147690008116, -3.9088836033041336, -3.81072440660502, -3.752630010227148, -3.7461194095364103, -3.7461194095364436], lambda = [618.3115171665414], penalties = [[1.0 -1.0 … 0.0 0.0; -1.0 2.0 … 0.0 0.0; … ; 0.0 0.0 … 2.0 -1.0; 0.0 0.0 … -1.0 1.0]], offsets = [0], sizes = [9], ranks = [8], information = [31858.457494765884 94009.4215295807 … 21.872235486901356 0.0; 94009.4215295807 282716.23254758987 … 207.7703657351071 0.0; … ; 21.872235486901356 207.7703657351071 … 7.451148392497197 0.0; 0.0 0.0 … 0.0 0.0], rss = 1100.24772067709, residual_dof = 40, covariance_ridge = 2.83952856581923e-7, coefficient_score = [-0.05196526553774561, -0.021995485391059333, -0.00942860381881161, 0.0019386820950586525, 0.002053950081325695, -0.0010815379568818173, 0.0007949108712459463, 0.00011555417507569388, -2.0589734594981003e-11], names = (:λ,), data_weights = [1.0; 1.0; … ; 1.0; 1.0;;], jac = :fd)))

    Data loss (SS): 1100.2
    EDF:            2.95
    Smoothing λ:    [618.3]

### Fitted trajectory

``` julia
plot(data_times, I_true, label="True I(t)", lw=2, color=:black, ls=:dash,
     xlabel="Time (days)", ylabel="Infected",
     title="PSM fit: SIR with unknown λ(I/N)")
scatter!(data_times, I_obs, label="Observed", ms=4, alpha=0.5, color=:steelblue)
plot!(data_times, sol.fitted_values[:, 1], label="PSM fit", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-9-output-1.svg)

### Recovered unknown function

``` julia
prev_grid = range(0.005, 0.14, length=100)
λ_true = [0.5 * p^0.9 for p in prev_grid]
λ_est = [sol.unknown_functions[:λ](p) for p in prev_grid]

plot(prev_grid, λ_true, label="True λ(I/N)", lw=2, color=:black, ls=:dash,
     xlabel="Prevalence (I/N)", ylabel="Force of infection λ",
     title="Recovered unknown function")
plot!(prev_grid, λ_est, label="Estimated λ(I/N)", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-10-output-1.svg)

## Section 2: Parametric Bootstrap

The **parametric bootstrap** generates pseudo-data by sampling from the
fitted likelihood:

$$I^*_t \sim N\bigl(\hat{I}(t),\; \hat\sigma^2\bigr)$$

Each replicate is refit with LAML, producing a distribution of fitted
trajectories and unknown function curves. Pointwise quantiles give the
confidence intervals.

``` julia
bs_param = bootstrap(sol, prob, LAML(maxiters=80, verbose=false);
    nboot=50, method=:parametric, rng=Random.Xoshiro(42), verbose=true)
```

    Bootstrap replicate 1 / 50
    Bootstrap replicate 2 / 50
    Bootstrap replicate 3 / 50
    ┌ Warning: LAML: smoothing selection never moved λ̂ off its initialization, so the reported λ̂, EDF and posterior covariance describe the INITIAL smoothing, not a selected one. Every Fellner–Schall proposal was rejected (or the iteration budget was spent before any ran). A noisy working-model Jacobian or a stalled nonlinear search can prevent acceptance; try `jac=:forwarddiff`, more `maxiters`, or a different knot count. See `convergence.smoothing_advanced`.
    └ @ PartiallySpecifiedModels ~/Projects/psm/PartiallySpecifiedModels.jl/src/solver.jl:2070
    Bootstrap replicate 50 / 50
    Bootstrap complete: 50 / 50 successful

    BootstrapResult([-3.5354652290936737 -3.5686324205961415 … -3.8691758908904017 -3.869175890890427; -3.4843453138614455 -3.55223688830137 … -3.856734528130992 -3.8567345281310295; … ; -3.491366023083381 -3.5582016780112715 … -3.876694812948638 -3.876694812948669; -3.5137762455136667 -3.539366268631411 … -3.774084692744404 -3.7740846927444274], [10.0; 15.87255475274929; … ; 5.762162550367454; 4.8753233772862945;;; 10.0; 16.19098429828677; … ; 5.8813420702837185; 4.990458525647471;;; 10.0; 15.91140017257584; … ; 5.726179329288117; 4.848005444622687;;; … ;;; 10.0; 16.07137350616337; … ; 6.167820963409456; 5.2346718756399815;;; 10.0; 16.117812782379435; … ; 5.9343013453090405; 5.035708124135503;;; 10.0; 16.160800360664965; … ; 5.749139191724543; 4.87133275780318], Dict(:λ => [0.0 0.0 … 0.0 0.0; 0.0019961599842011 0.00206205260833418 … 0.002048936455731471 0.0020472616446451375; … ; 0.15255709923057065 0.15337245570894104 … 0.15259055485361953 0.1568108452310603; 0.15401794655224646 0.15485141000170863 … 0.15404058373835777 0.15841578510807583]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 15.690798605020817; … ; 5.458588130574154; 4.614848430224123;;], upper = [10.0; 16.785627304245544; … ; 6.318787788693276; 5.40787464003493;;]), Dict(:λ => (lower = [0.0, 0.0019612559835097618, 0.003919326403169374, 0.005872589819219015, 0.00781705251579582, 0.009774470930068757, 0.011713072108446732, 0.013642770035447387, 0.015558418035674221, 0.017463907775108277  …  0.13659067921437976, 0.13788975626991773, 0.1391887024200249, 0.14048753402788017, 0.14178626745666234, 0.14308491906955031, 0.14438350522972287, 0.14568204230035894, 0.14698054664463725, 0.14827903462573674], upper = [0.0, 0.002191292141918197, 0.004357254447454811, 0.006496636016665951, 0.00860818594960774, 0.010690653346336313, 0.012742787306907755, 0.014763336931378237, 0.01675105131980386, 0.018704679572240722  …  0.16481372696055172, 0.16789533976218018, 0.17098889006951618, 0.1740928856943465, 0.17720583444845742, 0.1803262441436357, 0.18345262259166772, 0.1865834776043402, 0.18971731699343947, 0.19285264857075224])), 0.95, 50, Dict{Symbol, Matrix{Float64}}(), Dict(:λ => [50, 50, 50, 50, 50, 50, 50, 50, 50, 50  …  50, 50, 50, 50, 50, 50, 50, 50, 50, 50]))

### Trajectory confidence intervals

``` julia
plot(data_times, I_true, label="True I(t)", lw=2, color=:black, ls=:dash,
     xlabel="Time (days)", ylabel="Infected",
     title="Parametric bootstrap: trajectory CI")
scatter!(data_times, I_obs, label="Observed", ms=3, alpha=0.4, color=:steelblue)
plot!(data_times, sol.fitted_values[:, 1], label="PSM fit", lw=2, color=:red)
plot!(data_times, bs_param.ci_fitted.lower[:, 1],
      fillrange=bs_param.ci_fitted.upper[:, 1],
      fillalpha=0.2, color=:red, label="95% CI", ls=:dot, lw=0)
```

![](29_bootstrap_files/figure-commonmark/cell-12-output-1.svg)

### Unknown function confidence intervals

``` julia
uf_grid = bs_param.uf_grid[:λ]
plot(prev_grid, λ_true, label="True λ(I/N)", lw=2, color=:black, ls=:dash,
     xlabel="Prevalence (I/N)", ylabel="Force of infection λ",
     title="Parametric bootstrap: unknown function CI")
plot!(uf_grid, bs_param.ci_uf[:λ].lower,
      fillrange=bs_param.ci_uf[:λ].upper,
      fillalpha=0.2, color=:red, label="95% CI", ls=:dot, lw=0)
plot!(prev_grid, λ_est, label="Estimated λ(I/N)", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-13-output-1.svg)

    Parametric bootstrap: 50 / 50 replicates succeeded

## Section 3: Nonparametric Bootstrap

The **nonparametric bootstrap** resamples the residuals
$\hat{e}_t = I_t - \hat{I}(t)$ with replacement and adds them to the
fitted values to create pseudo-data:

$$I^*_t = \hat{I}(t) + \hat{e}_{\pi(t)}$$

where $\pi$ is a random permutation with replacement. This makes no
assumption about the error distribution — only that residuals are
exchangeable.

``` julia
bs_nonparam = bootstrap(sol, prob, LAML(maxiters=80, verbose=false);
    nboot=50, method=:nonparametric, rng=Random.Xoshiro(42), verbose=true)
```

    Bootstrap replicate 1 / 50
    Bootstrap replicate 2 / 50
    Bootstrap replicate 3 / 50
    Bootstrap replicate 50 / 50
    Bootstrap complete: 50 / 50 successful

    BootstrapResult([-3.4980804281088957 -3.5314633969558256 … -3.672737032513467 -3.672737032513517; -3.439364712081999 -3.5486213963826114 … -3.8705477110581157 -3.870547711058142; … ; -3.51017850199167 -3.5510045828144605 … -3.695720164471482 -3.6957201644715063; -3.532541294126057 -3.5820400350734714 … -4.009774613816553 -4.009774613816596], [10.0; 16.2769098022152; … ; 6.112061841169536; 5.196050891992233;;; 10.0; 16.40741119808494; … ; 5.681078578887394; 4.820689988561966;;; 10.0; 15.962867691558126; … ; 6.1724506112668704; 5.239692538234888;;; … ;;; 10.0; 15.71082489894971; … ; 5.629927094335434; 4.753399248244117;;; 10.0; 16.09155973118687; … ; 5.641870690580248; 4.776158127907325;;; 10.0; 15.795672171913177; … ; 5.560516904752431; 4.695838430274611], Dict(:λ => [0.0 0.0 … 0.0 0.0; 0.0020707096677425006 0.002110846402420038 … 0.0020383632552118357 0.001985127109500783; … ; 0.16009101025645345 0.15450965663905297 … 0.1606607597342933 0.14887983678736338; 0.1618649988244717 0.15596854046814176 … 0.1623949269677679 0.15015079755428207]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 15.797769990466414; … ; 5.5761341974086065; 4.70878961431775;;], upper = [10.0; 16.727561083240765; … ; 6.273108915228827; 5.3473970871498;;]), Dict(:λ => (lower = [0.0, 0.001984497302221032, 0.003962743057052607, 0.005932974581111397, 0.00789485660907876, 0.0098478471969346, 0.011791344330332885, 0.013724745994927644, 0.01564745017637289, 0.017558854860322597  …  0.13824779924828304, 0.1396199702369915, 0.14098013297859563, 0.1423245424144605, 0.14366886500379278, 0.14501311522101454, 0.1463434781097141, 0.14763573741518635, 0.14892795473484305, 0.150220151061592], upper = [0.0, 0.0021839923684972545, 0.00434096530173341, 0.006470014171917745, 0.008570234351259549, 0.010640721211968119, 0.012680570126252696, 0.014688876466322623, 0.016669084534180527, 0.018621608726199133  …  0.15124345088731297, 0.15334913141596873, 0.15545577460962612, 0.15756326013515998, 0.15967146765944504, 0.16178027684935614, 0.16388956737176805, 0.16599921889355562, 0.16810911108159352, 0.17021912360275665])), 0.95, 50, Dict{Symbol, Matrix{Float64}}(), Dict(:λ => [50, 50, 50, 50, 50, 50, 50, 50, 50, 50  …  50, 50, 50, 50, 50, 50, 50, 50, 50, 50]))

### Trajectory confidence intervals

``` julia
plot(data_times, I_true, label="True I(t)", lw=2, color=:black, ls=:dash,
     xlabel="Time (days)", ylabel="Infected",
     title="Nonparametric bootstrap: trajectory CI")
scatter!(data_times, I_obs, label="Observed", ms=3, alpha=0.4, color=:steelblue)
plot!(data_times, sol.fitted_values[:, 1], label="PSM fit", lw=2, color=:red)
plot!(data_times, bs_nonparam.ci_fitted.lower[:, 1],
      fillrange=bs_nonparam.ci_fitted.upper[:, 1],
      fillalpha=0.2, color=:blue, label="95% CI", ls=:dot, lw=0)
```

![](29_bootstrap_files/figure-commonmark/cell-16-output-1.svg)

### Unknown function confidence intervals

``` julia
plot(prev_grid, λ_true, label="True λ(I/N)", lw=2, color=:black, ls=:dash,
     xlabel="Prevalence (I/N)", ylabel="Force of infection λ",
     title="Nonparametric bootstrap: unknown function CI")
plot!(bs_nonparam.uf_grid[:λ], bs_nonparam.ci_uf[:λ].lower,
      fillrange=bs_nonparam.ci_uf[:λ].upper,
      fillalpha=0.2, color=:blue, label="95% CI", ls=:dot, lw=0)
plot!(prev_grid, λ_est, label="Estimated λ(I/N)", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-17-output-1.svg)

    Nonparametric bootstrap: 50 / 50 replicates succeeded

## Section 4: Case Bootstrap

> [!WARNING]
>
> ### Case bootstrap has been removed
>
> The **case bootstrap** resamples entire observation rows with
> replacement. For time series data from ODE models, this scrambles the
> temporal structure — a resampled dataset might place the peak
> observation at an early time point. This produces unreliable CIs and
> high failure rates. Case resampling is designed for cross-sectional
> (i.i.d.) data, not time-ordered dynamical systems. For this reason
> `method=:case` has been removed from the package (requesting it raises
> an error).
>
> For ODE-based PSMs, use **parametric** or **nonparametric** bootstrap
> instead.

## Section 5: Comparison of Bootstrap Methods

### Side-by-side trajectory CIs

``` julia
p_traj = plot(data_times, I_true, label="True", lw=2, color=:black, ls=:dash,
     xlabel="Time (days)", ylabel="Infected",
     title="Trajectory CI comparison", legend=:topright)
scatter!(p_traj, data_times, I_obs, label="Data", ms=2, alpha=0.3, color=:gray)

plot!(p_traj, data_times, bs_param.ci_fitted.lower[:, 1],
      fillrange=bs_param.ci_fitted.upper[:, 1],
      fillalpha=0.2, color=:red, label="Parametric", ls=:dot, lw=0)
plot!(p_traj, data_times, bs_nonparam.ci_fitted.lower[:, 1],
      fillrange=bs_nonparam.ci_fitted.upper[:, 1],
      fillalpha=0.2, color=:blue, label="Nonparametric", ls=:dot, lw=0)
plot!(p_traj, data_times, sol.fitted_values[:, 1], label="Fit", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-19-output-1.svg)

### Side-by-side unknown function CIs

``` julia
p_uf = plot(prev_grid, λ_true, label="True λ", lw=2, color=:black, ls=:dash,
     xlabel="Prevalence (I/N)", ylabel="λ(I/N)",
     title="Unknown function CI comparison", legend=:topleft)

plot!(p_uf, bs_param.uf_grid[:λ], bs_param.ci_uf[:λ].lower,
      fillrange=bs_param.ci_uf[:λ].upper,
      fillalpha=0.2, color=:red, label="Parametric", ls=:dot, lw=0)
plot!(p_uf, bs_nonparam.uf_grid[:λ], bs_nonparam.ci_uf[:λ].lower,
      fillrange=bs_nonparam.ci_uf[:λ].upper,
      fillalpha=0.2, color=:blue, label="Nonparametric", ls=:dot, lw=0)
plot!(p_uf, prev_grid, λ_est, label="Fit", lw=2, color=:red)
```

![](29_bootstrap_files/figure-commonmark/cell-20-output-1.svg)

### Quantitative comparison

    Method          | n_success | CI width at peak | Mean UF CI width | UF coverage
    -------------------------------------------------------------------------------------
    Parametric      | 50/50     | 9.6              | 0.00859          | 87.0%
    Nonparametric   | 50/50     | 6.8              | 0.0055           | 82.0%

**Interpretation:**

- **Parametric** CIs tend to be narrower because they assume the correct
  error model.
- **Nonparametric** CIs are slightly wider because resampled residuals
  capture any non-Gaussian features.

> [!IMPORTANT]
>
> ### Bootstrap coverage and smoothing bias
>
> The bootstrap CIs may show **less than nominal coverage** (e.g., 70%
> instead of 95%) for the unknown function $\lambda(I/N)$. This is a
> well-known limitation: the smoothing penalty introduces **bias** in
> the estimated function (pulling it towards linearity), and the
> bootstrap only captures **sampling variability** around the biased
> estimate — not the bias itself.
>
> This is analogous to the bias–variance tradeoff in kernel smoothing
> and GAMs (Nychka 1988, Wood 2006 §6.10). Two approaches give better
> coverage:
>
> 1.  **Bayesian credible intervals** from the LAML posterior covariance
>     (see [Vignette 14: MCMC](../14_mcmc/14_mcmc.qmd)) — these account
>     for smoothing uncertainty by design.
> 2.  **Undersmoothing** — using more knots or smaller λ reduces bias at
>     the cost of wider CIs.
>
> For **trajectory** CIs (fitted values at observed times), bootstrap
> coverage is typically much closer to nominal because the ODE
> integration integrates out local bias in the unknown function.

## Section 6: Bootstrap with Non-Gaussian Likelihoods

A key advantage of the parametric bootstrap is that it **respects the
likelihood family**. When fitting count data with `Poisson()`, the
parametric bootstrap samples $I^*_t \sim \text{Pois}(\hat{I}(t))$
instead of adding Gaussian noise. This naturally produces integer
pseudo-data with variance proportional to the mean.

### Generate Poisson data

![](29_bootstrap_files/figure-commonmark/cell-22-output-1.svg)

### Fit with Poisson likelihood

``` julia
prob_pois = PSMProblem(
    sir_psm!, u0, tspan,
    [ShapeConstrainedBSplineApproximator(:λ, foi_domain, 10, :inc_zero_left; initial = 0.4)];
    data_times = data_times,
    data_values = reshape(I_pois, :, 1),
    obs_to_state = [2],
    known_params = (γ = 0.25, N = N_pop),
    likelihood = Poisson(),
    solver = Tsit5()
)

sol_pois = solve(prob_pois, LAML(maxiters=100, verbose=false))
```

    PSMSolution((λ = [-3.5396699484531746, -3.579041917659837, -3.6909580261773396, -3.794870136612641, -3.876295435011491, -3.9158724471901074, -3.9233197052077116, -3.923527545755791, -3.9235275457558223]), 1213.433975429936, 2425.5319169674785, 2.336033139957409, [40.333076767680815], [10.0; 15.788726611103428; … ; 5.4132570048357636; 4.566332368328208;;], [6.0; 19.0; … ; 8.0; 7.0;;], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0  …  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0], Dict{Symbol, Any}(:λ => PartiallySpecifiedModels.var"#evaluator#build_constrained_bspline_evaluator##0"{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int64, Vector{Float64}, Vector{Float64}}(0.028428136150533228, 0.18241219761147542, 0.028428136150533228, 0.503853707769426, 0.7222604315946737, 0.272, 0.0, 4, [-0.11657142857142856, -0.07771428571428571, -0.038857142857142854, 0.0, 0.038857142857142854, 0.07771428571428571, 0.11657142857142858, 0.15542857142857142, 0.19428571428571428, 0.23314285714285715, 0.272, 0.3108571428571429, 0.34971428571428576, 0.38857142857142857], [0.0, 0.028609715648034235, 0.05612995431106244, 0.08077192139144868, 0.10300866784645096, 0.12352424621175077, 0.14325153651019273, 0.16283388206256655, 0.18241219761147554, 0.20199051316038394])), (V_beta = [0.013463171940758013 -0.0035348883389090527 … 0.002145067997574774 0.002145067997435365; -0.0035348883389090527 0.002138384923372728 … 0.000846839765880474 0.0008468397658254375; … ; 0.002145067997574774 0.000846839765880474 … 0.05389647984536958 0.053896479841866816; 0.002145067997435365 0.0008468397658254375 … 0.053896479841866816 0.07869002598152158], sigma2 = 1.0, converged = true, iterations = 26, reason = :converged_tol, laml_failures = 0, criterion = :working, laml = -136.01085750086148, stationarity = 8.790579231199303e-8, smoothing_advanced = true, smoothing_fixed = false, solver = :LAML, smoothing_state = nothing))

    Poisson fit — data_loss: 2425.5, EDF: 2.34

### Parametric bootstrap with Poisson likelihood

The parametric bootstrap now samples from $\text{Pois}(\hat\mu_t)$:

``` julia
bs_pois = bootstrap(sol_pois, prob_pois, LAML(maxiters=80, verbose=false);
    nboot=50, method=:parametric, rng=Random.Xoshiro(42), verbose=true)
```

    Bootstrap replicate 1 / 50
    Bootstrap replicate 2 / 50
    Bootstrap replicate 3 / 50
    Bootstrap replicate 50 / 50
    Bootstrap complete: 50 / 50 successful

    BootstrapResult([-3.5305458191875863 -3.572493592112995 … -3.956354863293022 -3.9563548632930514; -3.537691832258953 -3.583248855582636 … -3.871445098304204 -3.8714450983042443; … ; -3.5629512527569953 -3.5961905385829547 … -3.8297388084335946 -3.8297388078237566; -3.5150636420239225 -3.568781616146172 … -4.046905249540222 -4.046905249540252], [10.0; 15.866738066480979; … ; 5.568581578744227; 4.704883760891557;;; 10.0; 15.765459761139487; … ; 5.313295013528141; 4.4789702783978855;;; 10.0; 15.404853258460351; … ; 5.374871546986176; 4.513035065030482;;; … ;;; 10.0; 15.40755885332509; … ; 4.993503889578151; 4.189861890302603;;; 10.0; 15.58212887993149; … ; 5.038526068961057; 4.233870217263413;;; 10.0; 15.955204216970913; … ; 6.108833134731433; 5.182622246398576], Dict(:λ => [0.0 0.0 … 0.0 0.0; 0.0019967755734389137 0.0019790782382557125 … 0.0019427632051651107 0.0020152192654440404; … ; 0.15076543058275169 0.15558791104190134 … 0.15898585181399333 0.14445673297971368; 0.15210547375458885 0.15704549050164673 … 0.1605048420684002 0.14568176471256117]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 15.240847796884504; … ; 4.663794095192094; 3.906357796658809;;], upper = [10.0; 16.08399967077154; … ; 6.101501600939992; 5.17777411897035;;]), Dict(:λ => (lower = [0.0, 0.0018635520964848465, 0.0037324253502498475, 0.00560672638095305, 0.007481520167182546, 0.009354158512739708, 0.011227281003123486, 0.013095669220595048, 0.014957422209468, 0.016815673244445634  …  0.1327650833900195, 0.13381138161428432, 0.1348918509423368, 0.1360890078584908, 0.1372861178088463, 0.13848318862103642, 0.1396802281226942, 0.14087724414145275, 0.14207424450494513, 0.1432712370408044], upper = [0.0, 0.002045218884987447, 0.004078259408146014, 0.006098264216576475, 0.008104375957379638, 0.010100635075712204, 0.012154139976674635, 0.014171269951393785, 0.016100969914400853, 0.018003998214478626  …  0.15390056437552058, 0.1558244231577656, 0.15774912362249507, 0.15967456055942233, 0.16160062875826042, 0.16352722300872266, 0.16545423810052215, 0.1673815688233721, 0.16930910996698564, 0.17123675632107613])), 0.95, 50, Dict{Symbol, Matrix{Float64}}(), Dict(:λ => [50, 50, 50, 50, 50, 50, 50, 50, 50, 50  …  50, 50, 50, 50, 50, 50, 50, 50, 50, 50]))

``` julia
p1 = plot(data_times, I_true, label="True I(t)", lw=2, color=:black, ls=:dash,
     xlabel="Time (days)", ylabel="Infected (count)",
     title="Poisson bootstrap: trajectory CI")
scatter!(p1, data_times, I_pois, label="Data", ms=3, alpha=0.4, color=:purple)
plot!(p1, data_times, sol_pois.fitted_values[:, 1], label="PSM fit", lw=2, color=:purple)
plot!(p1, data_times, bs_pois.ci_fitted.lower[:, 1],
      fillrange=bs_pois.ci_fitted.upper[:, 1],
      fillalpha=0.2, color=:purple, label="95% CI", ls=:dot, lw=0)

p2 = plot(prev_grid, λ_true, label="True λ", lw=2, color=:black, ls=:dash,
     xlabel="Prevalence (I/N)", ylabel="λ(I/N)",
     title="Poisson bootstrap: unknown function CI")
plot!(p2, bs_pois.uf_grid[:λ], bs_pois.ci_uf[:λ].lower,
      fillrange=bs_pois.ci_uf[:λ].upper,
      fillalpha=0.2, color=:purple, label="95% CI", ls=:dot, lw=0)
λ_pois_est = [sol_pois.unknown_functions[:λ](p) for p in prev_grid]
plot!(p2, prev_grid, λ_pois_est, label="Estimated λ", lw=2, color=:purple)

plot(p1, p2, layout=(1, 2), size=(900, 400))
```

![](29_bootstrap_files/figure-commonmark/cell-26-output-1.svg)

Note how the Poisson CIs are **narrower near zero** (where counts are
small and Poisson variance is low) and **wider at the peak** (where
counts — and hence Poisson variance — are large). This
heteroscedasticity is automatically captured by the parametric
bootstrap.

## Section 7: Diagnostic Plots

Standard 4-panel diagnostics for the primary Gaussian fit help verify
that the residuals are well-behaved:

``` julia
diag = appraise(sol)

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

![](29_bootstrap_files/figure-commonmark/cell-27-output-1.svg)

    Durbin-Watson: 1.971

A Durbin-Watson statistic near 2 indicates no strong autocorrelation in
the residuals, supporting the validity of the bootstrap CIs (which
assume approximately independent errors).

> [!TIP]
>
> ### See Also
>
> - [Vignette 14: MCMC](../14_mcmc/14_mcmc.qmd) — Bayesian credible
>   intervals via NUTS sampling
> - [Vignette 24: Variational](../24_variational/24_variational.qmd) —
>   approximate Bayesian posterior intervals
> - [Vignette 27: Predator–Prey Functional
>   Response](../27_predator_prey/27_predator_prey.qmd) — bootstrap CIs
>   on a shape-constrained functional response
> - [Vignette 28: Fisheries](../28_fisheries/28_fisheries.qmd) — Poisson
>   parametric bootstrap on count data

## Practical Guidance

### Choosing a bootstrap method

| Scenario | Recommended method |
|----|----|
| Gaussian noise, well-specified model | `:parametric` — narrowest CIs |
| Count data (Poisson, NegBin) | `:parametric` — respects variance–mean relationship |
| Suspect non-Gaussian errors | `:nonparametric` — no distributional assumption |
| Quick exploratory analysis | `:parametric` with `nboot=50` |

> [!NOTE]
>
> The `:case` bootstrap (resampling entire rows) has been **removed**
> because it scrambles the temporal structure of ODE/DDE data; the
> `:nonparametric` method requires a Gaussian likelihood.

### Tips

- **Start with `nboot=50–100`** to check the method works, then increase
  to 200+ for publication-quality CIs.
- **Check `bs.n_success`**: if many replicates fail (\< 80% success),
  the model may be unstable. Try increasing `maxiters` or simplifying
  the approximator (fewer knots).
- **Set `rng=Random.Xoshiro(seed)`** for reproducibility.
- The parametric bootstrap is the default for good reason: it is fast,
  well-calibrated, and adapts to the likelihood family. Use the
  nonparametric bootstrap when you have reason to doubt the error model
  (Gaussian likelihood only).
