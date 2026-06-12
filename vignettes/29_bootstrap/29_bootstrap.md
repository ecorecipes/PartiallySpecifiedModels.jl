# Bootstrap Confidence Intervals
Simon Frost
2026-06-12

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

`PartiallySpecifiedModels.jl` implements three bootstrap methods:

| Method | Description | Assumptions |
|----|----|----|
| `:parametric` | Simulate new data from the fitted likelihood (e.g., $N(\hat\mu, \hat\sigma)$ or $\text{Pois}(\hat\mu)$) | Correct likelihood family |
| `:nonparametric` | Resample residuals with replacement per state | Exchangeable residuals |
| `:case` | Resample entire observation rows with replacement | Weakest assumptions |

This vignette demonstrates all three methods on an SIR epidemic model
with a nonparametric force of infection, compares the resulting
confidence intervals, and shows how the parametric bootstrap adapts to
non-Gaussian likelihoods.

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

    PSMSolution((λ = [-4.483799426796068, -4.361150650389338, -3.8752836828985573, -3.641851051670254, -3.766546994323222, -3.784712932765237, -3.7567625838321694, -3.7508670191317557, -3.75086701913179]), 814.9275892152435, 1425.2693530137478, 3.744382000877284, [634.777593727556], [10.0; 22.313123136926283; … ; 8.733981312569304; 8.170734964363328;;], [3.6696475510497084; 19.43055459411786; … ; 5.397424420239466; 10.498804562183398;;], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0  …  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0], Dict{Symbol, Any}(:λ => PartiallySpecifiedModels.var"#evaluator#build_constrained_bspline_evaluator##0"{Float64, Float64, Float64, Float64, Float64, Float64, Int64, Vector{Float64}, Vector{Float64}}(0.16195827366683127, 0.011469798166673924, 0.5977163845243693, 0.30766669494620713, 0.272, 0.0, 4, [-0.11657142857142856, -0.07771428571428571, -0.038857142857142854, 0.0, 0.038857142857142854, 0.07771428571428571, 0.11657142857142858, 0.15542857142857142, 0.19428571428571428, 0.23314285714285715, 0.272, 0.3108571428571429, 0.34971428571428576, 0.38857142857142857], [0.0, 0.01122717314506519, 0.023910096419782784, 0.04444623018137478, 0.07031258729159219, 0.09318090250065707, 0.11564213246402386, 0.13873272267818207, 0.1619582736668314, 0.18518382465547994])), (V_beta = [1.3075360751855491e-5 -4.8131569085654764e-5 … -1.883398391660158e-5 -1.883398135734801e-5; -4.8131569085654764e-5 0.00017980508472011932 … 5.8652276449762435e-5 5.8652268479805127e-5; … ; -1.883398391660158e-5 5.8652276449762435e-5 … 0.0025680774336690766 0.0025680770847061932; -1.883398135734801e-5 5.8652268479805127e-5 … 0.0025680770847061932 0.0041434314331940045], sigma2 = 38.25649471302045))

    Data loss (SS): 1425.3
    EDF:            3.74
    Smoothing λ:    [634.8]

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
    Bootstrap replicate 50 / 50
    Bootstrap complete: 50 / 50 successful

    BootstrapResult([-4.499105587355219 -4.478761570514281 … -3.6782799508661466 -3.6782799508661843; -4.47863840064737 -4.2769589367339504 … -3.739360345665995 -3.739459300760889; … ; -4.6748992258307975 -4.119358843841836 … -3.933876103836744 -3.933876103836775; -4.543971527882172 -4.259649594150126 … -3.7431460454200516 -3.74314604542008], [10.0; 21.57714176056252; … ; 8.346995001009143; 7.796023532076023;;; 10.0; 22.84039932937255; … ; 8.76101680769033; 8.200610871407177;;; 10.0; 22.399041436407433; … ; 8.696023208716722; 8.131940586455174;;; … ;;; 10.0; 29.233031517563997; … ; 10.406324910659563; 9.826569012470946;;; 10.0; 21.89280201042796; … ; 8.578896089447984; 7.985964668771135;;; 10.0; 22.269988679195958; … ; 8.520878994491007; 7.959312562428227], Dict(:λ => [0.011095239718836324 0.011702335332375208 … 0.010423407689810309 0.011150555427783369; 0.011886358236350246 0.012595212124170363 … 0.011338505141686018 0.012029192176077652; … ; 0.16209845492599528 0.16350370435473985 … 0.1548578535521724 0.16183855054340773; 0.16386272278560654 0.1651646434979662 … 0.15622810085934397 0.16349334595083917]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 21.529810569052366; … ; 8.320129986436672; 7.749535276370619;;], upper = [10.0; 24.37874104560581; … ; 9.743321994981994; 9.145106211182796;;]), Dict(:λ => (lower = [0.01036600785470323, 0.011246652433554296, 0.012155761160445564, 0.013079096106295252, 0.014028653455472702, 0.015006636737811252, 0.015997388709731886, 0.016928723071385376, 0.017882827478510825, 0.01887131108515927  …  0.1407047952656481, 0.14203662497888156, 0.1433681265473279, 0.14469934098908555, 0.14603030932225275, 0.14736107256492806, 0.1486916717352098, 0.1500221478511964, 0.15135254193098616, 0.15268289499267756], upper = [0.013819401211959358, 0.014570549334511771, 0.015288001876953703, 0.015981720540751126, 0.01681412293470088, 0.017796664211946343, 0.018788297650349846, 0.01978952305589592, 0.02081417781552198, 0.0218591925545426  …  0.15797072935513157, 0.1599953363480166, 0.16201829793326297, 0.16403981879788876, 0.1660601036289118, 0.16807935711335015, 0.17009778393822172, 0.1721155887905445, 0.17413297635733635, 0.17615015132561535])), 0.95, 50)

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

    BootstrapResult([-4.471608458522186 -4.241284231233176 … -3.699321323485501 -3.699321323485532; -4.488720445294768 -4.28770614281811 … -3.9628704713616996 -3.9628704713617355; … ; -4.572474205484575 -4.23803862949913 … -3.549867413353646 -3.5498674133536685; -4.568397231414501 -4.276441425078659 … -4.059749643098488 -4.0597496430985185], [10.0; 23.139925084760907; … ; 9.21217080552221; 8.628213301611883;;; 10.0; 22.6729830807737; … ; 8.624390564844889; 8.070577278663492;;; 10.0; 21.456260957066593; … ; 8.824645110671135; 8.217228570098593;;; … ;;; 10.0; 21.28934808713203; … ; 8.118626324134611; 7.557730421010524;;; 10.0; 22.115381414812003; … ; 8.612794656258693; 8.037874009893839;;; 10.0; 21.9023333342923; … ; 8.514507363009413; 7.944061901469779], Dict(:λ => [0.011851177918296804 0.011584123613961907 … 0.010954903548803397 0.01090033998117677; 0.012765409064968541 0.012467809977188375 … 0.011835305102060815 0.011761805895476112; … ; 0.16282364992319026 0.15413414654406943 … 0.16903260424725444 0.14941301116869865; 0.16455167660441158 0.1554655843208642 … 0.1710352524576462 0.15062254361101657]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 21.445053167875766; … ; 8.207699557881442; 7.644655004113857;;], upper = [10.0; 24.264728521878006; … ; 9.826122522664582; 9.240634227478509;;]), Dict(:λ => (lower = [0.010508971577086643, 0.01135470590586803, 0.012221152569055265, 0.013109534143864558, 0.014020998484484815, 0.014956651170807597, 0.015906265672464285, 0.016882344386637223, 0.0178863398716941, 0.018919704686002604  …  0.13828171663285324, 0.13949706324850825, 0.14071282464904028, 0.1419289489863397, 0.14314538441229685, 0.14436207907880216, 0.14557898113774592, 0.14679603874101857, 0.14801320004051044, 0.14923041318811198], upper = [0.013363446734418425, 0.014263421007172039, 0.015156710025329026, 0.016045306227849727, 0.0169312020536945, 0.017816389941823704, 0.018702862331197674, 0.01959261166077678, 0.020493177747209133, 0.021409546128926996  …  0.15562246729883572, 0.15763111549201994, 0.15963945244831418, 0.1616474988670894, 0.1636552754477163, 0.16567948661370827, 0.1677186415640779, 0.16975765263904244, 0.17179651924976225, 0.17383524080739796])), 0.95, 50)

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
> ### Case bootstrap is not recommended for ODE models
>
> The **case bootstrap** resamples entire observation rows with
> replacement. For time series data from ODE models, this scrambles the
> temporal structure — a resampled dataset might place the peak
> observation at an early time point. This produces unreliable CIs and
> high failure rates. Case resampling is designed for cross-sectional
> (i.i.d.) data, not time-ordered dynamical systems.
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
    Parametric      | 50/50     | 11.2             | 0.00737          | 58.0%
    Nonparametric   | 50/50     | 9.0              | 0.0072           | 60.0%

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

    PSMSolution((λ = [-4.661702023811262, -4.290758581275516, -3.807449767691886, -3.6115911139246695, -3.7179876643401815, -3.8221245903063883, -3.8464329771207155, -3.8474133566926376, -3.8474133566926705]), 1545.1103934007122, 3087.48949083035, 3.387215614654588, [6.318003593058379], [10.0; 20.94496985301474; … ; 8.009685028090173; 7.44313625854965;;], [6.0; 19.0; … ; 8.0; 7.0;;], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0  …  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0], Dict{Symbol, Any}(:λ => PartiallySpecifiedModels.var"#evaluator#build_constrained_bspline_evaluator##0"{Float64, Float64, Float64, Float64, Float64, Float64, Int64, Vector{Float64}, Vector{Float64}}(0.15949926989936425, 0.010105258892955618, 0.5432836116835497, 0.29605369937805714, 0.272, 0.0, 4, [-0.11657142857142856, -0.07771428571428571, -0.038857142857142854, 0.0, 0.038857142857142854, 0.07771428571428571, 0.11657142857142858, 0.15542857142857142, 0.19428571428571428, 0.23314285714285715, 0.272, 0.3108571428571429, 0.34971428571428576, 0.38857142857142857], [0.0, 0.00940598862794136, 0.02300759884596828, 0.04496939741623446, 0.07161993448022425, 0.09561257905155462, 0.11725788282693271, 0.13838882094344307, 0.1594992698993644, 0.180609718855285])), (V_beta = [0.026718118996109298 -0.024808996339621907 … -0.004340464722497425 -0.004340464719192632; -0.024808996339621907 0.03168626152018533 … -0.001572061184470409 -0.001572061183273455; … ; -0.004340464722497425 -0.001572061184470409 … 0.28490900134929004 0.2849090011323628; -0.004340464719192632 -0.001572061183273455 … 0.2849090011323628 0.44318684683773174], sigma2 = 1.0))

    Poisson fit — data_loss: 3087.5, EDF: 3.39

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

    BootstrapResult([-4.72779294619559 -4.240098110957625 … -3.914405658790872 -3.9144056587909075; -4.460808581297976 -4.276742763288162 … -3.6390289490690124 -3.639028949171977; … ; -4.462343953904841 -4.280922160042251 … -3.629689201096135 -3.629689201096164; -4.6169487481370135 -4.262828660785953 … -3.8222278583544074 -3.822227858354431], [10.0; 20.671319149099677; … ; 7.929156483132661; 7.353599546958788;;; 10.0; 23.003958836053457; … ; 8.224074550315706; 7.6945277167601995;;; 10.0; 21.07948380342998; … ; 7.893524599759278; 7.338462797057452;;; … ;;; 10.0; 21.78283362640422; … ; 8.45664048147991; 7.884090004107766;;; 10.0; 22.98742121947545; … ; 9.11630446330942; 8.537637810684311;;; 10.0; 21.528916999078547; … ; 8.125863003210767; 7.5683623838173295], Dict(:λ => [0.009723129762549626 0.011871038235870856 … 0.011846914680133132 0.010526019965001954; 0.010554105864778751 0.012770520470948527 … 0.01274377110188884 0.01137866460539861; … ; 0.15441192780549662 0.1729351357305859 … 0.16704912888968673 0.159406233969161; 0.1558088211862836 0.17476918521231907 … 0.168900150462952 0.1609365588660596]), Dict(:λ => [0.0, 0.0027474747474747476, 0.005494949494949495, 0.008242424242424242, 0.01098989898989899, 0.013737373737373737, 0.016484848484848484, 0.019232323232323233, 0.02197979797979798, 0.024727272727272726  …  0.24727272727272728, 0.25002020202020203, 0.25276767676767675, 0.25551515151515153, 0.25826262626262625, 0.261010101010101, 0.26375757575757575, 0.2665050505050505, 0.26925252525252524, 0.272]), (lower = [10.0; 20.326036687189582; … ; 7.4365813486146255; 6.897383375799695;;], upper = [10.0; 23.9805988465152; … ; 9.188086959660925; 8.601689289337076;;]), Dict(:λ => (lower = [0.009406758922221394, 0.010225569019970052, 0.011075784462753481, 0.011959400391718658, 0.012878411948012574, 0.013834814272782218, 0.014830602507174568, 0.015867771792336623, 0.01694831726941536, 0.018074234079557765  …  0.1389856147106291, 0.14002476576826894, 0.1410628673426444, 0.1421000506191636, 0.14313644678323445, 0.1441721870202651, 0.14520740251566355, 0.1462422244548379, 0.1472767840231961, 0.14831121240614625], upper = [0.012760291153672016, 0.013703015905349707, 0.014653889020685259, 0.015584837483651769, 0.016517928297951626, 0.01745781123554666, 0.018405567159091564, 0.01936227693124106, 0.020329021414649866, 0.021308577256033523  …  0.16081401623644612, 0.16294407207695338, 0.16504220188276658, 0.1670819652091145, 0.1691218628993633, 0.17118559179421025, 0.17325927192610063, 0.1753330801140571, 0.1774069739485814, 0.17948091102017544])), 0.95, 50)

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

    Durbin-Watson: 1.541

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
> - [Vignette 27: Blowfly DDE](../27_blowfly_dde/27_blowfly_dde.qmd) —
>   bootstrap CIs on a DDE model
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
> The `:case` bootstrap (resampling entire rows) is available but **not
> recommended** for ODE/DDE models because it scrambles the temporal
> structure of the data.

### Tips

- **Start with `nboot=50–100`** to check the method works, then increase
  to 200+ for publication-quality CIs.
- **Check `bs.n_success`**: if many replicates fail (\< 80% success),
  the model may be unstable. Try increasing `maxiters` or simplifying
  the approximator (fewer knots).
- **Set `rng=Random.Xoshiro(seed)`** for reproducibility.
- The parametric bootstrap is the default for good reason: it is fast,
  well-calibrated, and adapts to the likelihood family. Use
  nonparametric or case bootstrap when you have reason to doubt the
  error model.
