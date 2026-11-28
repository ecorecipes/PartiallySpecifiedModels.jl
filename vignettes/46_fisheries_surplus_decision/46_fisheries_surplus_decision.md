# Fisheries Surplus Production Under Functional-Form Uncertainty: DecisionAnalysis.jl Bridge
Simon Frost
2026-11-27

- [Overview](#overview)
- [Setup](#setup)
- [Biomass Survey Series and Nonparametric Surplus Production
  Fit](#biomass-survey-series-and-nonparametric-surplus-production-fit)
- [Harvest Decision Analysis & Value of Information
  (`DecisionAnalysis.jl`)](#harvest-decision-analysis--value-of-information-decisionanalysisjl)
- [Summary](#summary)

## Overview

Setting sustainable fisheries harvest quotas depends critically on the
shape of density-dependent surplus production $g(B)$ — whether biomass
productivity is symmetric Schaefer logistic, skewed Pella–Tomlinson, or
exhibits low-density depensation (Allee effects). Assuming the wrong
parametric curve risks overestimating maximum sustainable yield and
collapsing the stock.

Here we estimate the stock’s per-capita surplus production curve $r(B)$
nonparametrically with `PartiallySpecifiedModels.jl`, convert posterior
curve uncertainty into a `DecisionAnalysis.SampleBelief` via
`sample_belief`, wrap the stock model as a `ManagedPSM`, and evaluate
candidate harvest rates under **expected utility**, **minimax regret**,
**EVPI** (Expected Value of Perfect Information), and **EVPPI**
(Expected Value of Partial Perfect Information on low-biomass
productivity).

## Setup

``` julia
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve, build_evaluator
using DecisionAnalysis
using OrdinaryDiffEq
using Random
using Statistics
using Printf
```

## Biomass Survey Series and Nonparametric Surplus Production Fit

We model stock biomass $B(t)$ under harvest rate $h$:

$$\frac{dB}{dt} = \big(r(B) - h\big) B,$$

where the true dynamics include mild low-density depensation and
historical harvest rate $h_{\text{hist}} = 0.08\,\text{yr}^{-1}$:

``` julia
rng = Random.Xoshiro(4601)
ts = collect(0.0:1.0:18.0)
h_hist = 0.08
r_true(B) = 0.45 * (1.0 - (B / 100.0)^1.4)

sol_true = OrdinaryDiffEq.solve(
    ODEProblem((du, u, p, t) -> (du[1] = (r_true(u[1]) - h_hist) * u[1]),
               [18.0], (0.0, 18.0)),
    Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
B_true = Float64[sol_true[1, i] for i in eachindex(ts)]
# Multiplicative survey error -> LogNormalLikelihood
Y_survey = reshape(B_true .* exp.(0.06 .* randn(rng, length(ts))), :, 1)

function fishery_ode!(du, u, p, t)
    du[1] = (p.r(u[1]) - p.h) * u[1]
    nothing
end

prob = PSMProblem(fishery_ode!, [18.0], (0.0, 18.0),
    [BSplineApproximator(:r, (0.0, 120.0), 7; initial = B -> 0.4 * (1.0 - B / 100.0))];
    data_times = ts,
    data_values = Y_survey,
    obs_to_state = [1],
    known_params = (h=h_hist,),
    likelihood = LogNormalLikelihood(0.06))

sol = solve(prob, LAML(maxiters=60))
println("Converged: ", sol.convergence.converged, "  EDF: ", round(sol.edf, digits=2))
```

    Converged: true  EDF: 2.0

## Harvest Decision Analysis & Value of Information (`DecisionAnalysis.jl`)

We draw 60 posterior curves via `sample_belief`, summarizing low-biomass
productivity $r(20)$ and mid-biomass productivity $r(50)$ for EVPPI
diagnostics, and evaluate five candidate harvest policies
$h \in \{0.0, 0.10, 0.18, 0.25, 0.35\}$ over a 15-year management
horizon with a conservation penalty if terminal biomass drops below
$B_{\text{lim}} = 30$:

``` julia
belief = sample_belief(sol, prob; n=60, rng=Random.Xoshiro(4602),
    summary_fn = (β, p) -> begin
        rf = build_evaluator(p.approximators[1], β)
        (; r_low=rf(20.0), r_mid=rf(50.0))
    end)

# ManagedPSM maps a harvest action h to a 15-year forward projection starting from current stock B(18)
B_now = Float64(sol.fitted_values[end, 1])
t_proj = collect(0.0:1.0:15.0)
apply_harvest(p, h) = PSMProblem(
    p.dynamics!, [B_now], (0.0, 15.0), p.approximators;
    data_times = t_proj,
    data_values = zeros(length(t_proj), 1),
    obs_to_state = [1],
    known_params = (h=h,),
    likelihood = p.likelihood)

fishery_outcome(traj, p, h) = begin
    B_end = max(traj[end, 1], 0.0)
    cum_catch = h * sum(max.(traj[:, 1], 0.0))
    collapsed = B_end < 30.0
    (; cum_catch=cum_catch, B_end=B_end, collapsed=collapsed)
end

managed = ManagedPSM(prob, apply_harvest, fishery_outcome)
actions = [:moratorium => 0.0, :precautionary => 0.10,
           :moderate => 0.18, :msy_target => 0.25, :aggressive => 0.35]

ot = outcomes(managed, actions, belief; seed=4603)
util = FunctionUtility(o -> o.cum_catch + 0.8 * o.B_end - 250.0 * o.collapsed)
pt = payoff_table(ot, util)
dprob = DecisionProblem(pt, belief)

res_eu = decide(dprob, MaxExpectedUtility())
res_mr = decide(dprob, MinimaxRegret())
v_evpi = evpi(dprob, SampleAverage())
v_evppi_low = evppi(dprob, [:r_low], StrongOakley())

println("Optimal harvest (Max Expected Utility): ", first_choice(res_eu))
println("Optimal harvest (Minimax Regret):       ", first_choice(res_mr))
@printf("EVPI (resolving full curve uncertainty): %.2f utility units\n", v_evpi.value)
@printf("EVPPI (learning low-stock r(20) only):   %.2f utility units\n", v_evppi_low.value)
```

    Optimal harvest (Max Expected Utility): aggressive
    Optimal harvest (Minimax Regret):       aggressive
    EVPI (resolving full curve uncertainty): 0.10 utility units
    EVPPI (learning low-stock r(20) only):   0.00 utility units

## Summary

- `LogNormalLikelihood` matches multiplicative fisheries survey errors
  cleanly.
- `sample_belief` and `ManagedPSM` bridge `PartiallySpecifiedModels.jl`
  directly into `DecisionAnalysis.jl` for risk-sensitive harvest policy
  selection and EVPI/EVPPI value-of-information quantification.
