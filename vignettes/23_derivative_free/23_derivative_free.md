# Derivative-Free Optimization
Simon Frost
2026-03-23

- [Overview](#overview)
- [Exponential Decay with Unknown
  Rate](#exponential-decay-with-unknown-rate)
- [Nelder-Mead vs Adam Comparison](#nelder-mead-vs-adam-comparison)
  - [Recovered Function](#recovered-function)
  - [Trajectory Fit](#trajectory-fit)
- [Particle Swarm for Global Search](#particle-swarm-for-global-search)
- [When to Use DerivativeFree](#when-to-use-derivativefree)

## Overview

The `DerivativeFreeSolver` uses gradient-free optimization (Nelder-Mead
or Particle Swarm) to fit partially specified models. This is a **robust
fallback** when gradient-based methods (LAML, Adam) struggle with:

- Stiff or chaotic dynamics that break autodiff
- Non-smooth objectives or discontinuous model behavior
- Poor initial conditions where gradient methods get stuck

The tradeoff is slower convergence — derivative-free methods need more
function evaluations — but they can explore the objective landscape more
broadly.

``` julia
using PartiallySpecifiedModels
using OrdinaryDiffEq
using Plots
using Random
Random.seed!(42)
```

    TaskLocalRNG()

## Exponential Decay with Unknown Rate

A simple model: $du/dt = -f(u)$ where $f(u) = 0.5u$ is unknown.

``` julia
function decay!(du, u, p, t)
    du[1] = -p.f(u[1])
end

sol_true = solve(ODEProblem(decay!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
                 Tsit5(); saveat=0.5)
t_data = collect(sol_true.t)
data_vals = [sol_true.u[i][1] + 0.05 * randn() for i in 1:length(t_data)]
data_matrix = reshape(max.(data_vals, 0.01), :, 1)
```

    21×1 Matrix{Float64}:
     5.039417780080215
     3.8500110179259908
     2.9889635616951553
     2.325167455020758
     1.8042833200179555
     1.4288952108579625
     1.0699970727636061
     0.9004531983252085
     0.7486023017326826
     0.5092902184958961
     ⋮
     0.21926695332322416
     0.21534802822348154
     0.19860836555059358
     0.12404373603521487
     0.07684114721258986
     0.05261060443561179
     0.11401977466247498
     0.056103905895220345
     0.04373891053962077

## Nelder-Mead vs Adam Comparison

``` julia
uf = BSplineApproximator(:f, (0.01, 5.5), 8)

prob = PSMProblem(decay!, [5.0], (0.0, 10.0), [uf];
    data_times=t_data, data_values=Float64.(data_matrix),
    obs_to_state=[1], known_params=NamedTuple())

sol_nm = solve(prob, DerivativeFreeSolver(method=:nelder_mead, maxiters=5000, verbose=false))
sol_adam = solve(prob, AdamSolver(lr=0.01, maxiters=3000, verbose=false))
sol_laml = solve(prob, LAML(maxiters=50, verbose=false))
```

    PSMSolution((f = [-0.0010250074105533742, 0.3886883241739532, 0.7830112284768596, 1.1866458061621756, 1.5979871742704743, 2.0138735742847, 2.431440267714643, 2.8492195375941654]), 0.011718423221421978, 0.022702414082000813, 2.682793508833768, [0.010504747940562255], [5.0; 3.8662157144117018; … ; 0.05551420808835955; 0.045971001398971256;;], [5.039417780080215; 3.8500110179259908; … ; 0.056103905895220345; 0.04373891053962077;;], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5  …  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], Dict{Symbol, Any}(:f => DataInterpolations.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}([-0.0010250074105533742, 0.3886883241739532, 0.7830112284768596, 1.1866458061621756, 1.5979871742704743, 2.0138735742847, 2.431440267714643, 2.8492195375941654], [0.01, 0.7942857142857143, 1.5785714285714285, 2.362857142857143, 3.1471428571428572, 3.9314285714285715, 4.715714285714285, 5.5], Float64[], DataInterpolations.CubicSplineParameterCache{Vector{Float64}}(Float64[], Float64[]), [0.0, 0.7842857142857143, 0.7842857142857143, 0.7842857142857145, 0.7842857142857143, 0.7842857142857143, 0.7842857142857138, 0.7842857142857147], [0.0, 0.006766256472088999, 0.017898805658636236, 0.012468742904189725, 0.0074016993937904944, 0.002258731150223349, -4.6291373989625884e-5, 0.0], DataInterpolations.ExtrapolationType.Extension, DataInterpolations.ExtrapolationType.Extension, FindFirstFunctions.Guesser{Vector{Float64}}([0.01, 0.7942857142857143, 1.5785714285714285, 2.362857142857143, 3.1471428571428572, 3.9314285714285715, 4.715714285714285, 5.5], Base.RefValue{Int64}(1), true), false, false)), nothing)

### Recovered Function

``` julia
f_nm = sol_nm.unknown_functions[:f]
f_adam = sol_adam.unknown_functions[:f]
f_laml = sol_laml.unknown_functions[:f]
u_grid = range(0.01, 5.5, length=100)
f_true(u) = 0.5 * u

p1 = plot(u_grid, f_true.(u_grid), label="True f(u)", lw=2, color=:black,
          xlabel="u", ylabel="f(u)", title="Decay Rate Recovery")
plot!(p1, u_grid, [f_nm(x) for x in u_grid], label="NelderMead (loss=$(round(sol_nm.data_loss, digits=3)))", lw=2)
plot!(p1, u_grid, [f_adam(x) for x in u_grid], label="Adam (loss=$(round(sol_adam.data_loss, digits=3)))", lw=2, ls=:dash)
plot!(p1, u_grid, [f_laml(x) for x in u_grid], label="LAML (loss=$(round(sol_laml.data_loss, digits=3)))", lw=2, ls=:dot)
p1
```

![](23_derivative_free_files/figure-commonmark/cell-5-output-1.svg)

### Trajectory Fit

``` julia
p2 = plot(t_data, data_matrix[:, 1], seriestype=:scatter, label="Data",
          xlabel="Time", ylabel="u", title="Trajectory", ms=4, alpha=0.6)
plot!(p2, t_data, sol_nm.fitted_values[:, 1], label="NelderMead", lw=2)
plot!(p2, t_data, sol_adam.fitted_values[:, 1], label="Adam", lw=2, ls=:dash)
plot!(p2, t_data, sol_laml.fitted_values[:, 1], label="LAML", lw=2, ls=:dot)
p2
```

![](23_derivative_free_files/figure-commonmark/cell-6-output-1.svg)

## Particle Swarm for Global Search

Particle swarm optimization explores more broadly and can escape local
minima:

``` julia
sol_ps = solve(prob, DerivativeFreeSolver(method=:particle_swarm, maxiters=5000,
                                           n_particles=20, verbose=false))
f_ps = sol_ps.unknown_functions[:f]
println("NelderMead: loss=$(round(sol_nm.data_loss, digits=4)), f(3)=$(round(f_nm(3.0), digits=3))")
println("ParticleSwarm: loss=$(round(sol_ps.data_loss, digits=4)), f(3)=$(round(f_ps(3.0), digits=3))")
println("True: f(3)=$(round(f_true(3.0), digits=3))")
```

    NelderMead: loss=0.0249, f(3)=1.519
    ParticleSwarm: loss=43.4949, f(3)=0.52
    True: f(3)=1.5

## When to Use DerivativeFree

- **Robustness**: Works when autodiff fails (stiff systems,
  discontinuities)
- **Global search**: Particle swarm can find solutions that gradient
  methods miss
- **No gradients needed**: Useful for complex models where ForwardDiff
  is prohibitively expensive
- **Tradeoff**: Slower convergence — needs 5-10× more function
  evaluations than gradient methods
- Best for **low-dimensional** parameter spaces (\< 50 parameters)
