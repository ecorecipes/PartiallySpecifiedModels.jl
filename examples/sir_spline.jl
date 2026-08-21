# SIR epidemic model with unknown force of infection (spline approximator)
#
# This example fits an SIR model where the transmission rate β(t)
# is an unknown function of time, modeled with a penalized B-spline.
# The data are cumulative case counts (Poisson-noised daily incidence,
# accumulated) fitted with a Poisson likelihood (identity link, so the
# spline is fitted on the response scale).
#
# Model: dS/dt = -β(t)*S*I/N
#        dI/dt =  β(t)*S*I/N - γ*I
#        dR/dt =  γ*I
#
# Observed: cumulative cases C(t), whose daily increments
#           ≈ β(t)*S*I/N are Poisson distributed
#
# Reference: Based on the SIR PSM example from
# https://github.com/epirecipes/sir-julia/blob/master/markdown/psm/psm.md

using PartiallySpecifiedModels
using OrdinaryDiffEq
using Printf
using Random

Random.seed!(42)

# ─── Generate synthetic epidemic data ─────────────────────────────

# True SIR with time-varying β
N = 10000.0
γ = 0.1      # recovery rate (10-day infectious period)
tspan = (0.0, 80.0)
data_times = collect(1.0:1.0:80.0)

# True β(t): seasonal-like variation
true_β(t) = 0.3 + 0.15 * sin(2π * t / 40)

function sir_true!(du, u, p, t)
    S, I, R, C = u
    β = true_β(t)
    infection = β * S * I / N
    du[1] = -infection
    du[2] = infection - γ * I
    du[3] = γ * I
    du[4] = infection  # cumulative cases
end

u0_true = [N - 10.0, 10.0, 0.0, 0.0]
prob_true = ODEProblem(sir_true!, u0_true, tspan)
sol_true = OrdinaryDiffEq.solve(prob_true, Tsit5(); saveat=data_times)

# Generate daily new case counts from the true trajectory
cum_cases = [sol_true(t)[4] for t in data_times]
daily_cases = diff(vcat(0.0, cum_cases))
daily_cases = max.(daily_cases, 0.1)  # floor at 0.1

# Simple Poisson sampler (avoid extra dependency)
function poisson_rand(λ)
    if λ <= 0; return 0; end
    L = exp(-λ)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand()
    end
    return k - 1
end

# Poisson-noised daily counts, accumulated into observed cumulative counts.
# (A sum of independent Poisson increments is itself Poisson; successive
# cumulative observations are correlated, which we ignore for simplicity.)
observed_cases = Float64[poisson_rand(c) for c in daily_cases]
cum_obs = cumsum(observed_cases)

println("Synthetic SIR data: $(length(data_times)) days")
println("Total true cases: $(round(Int, sum(daily_cases)))")
println("Total observed:   $(round(Int, sum(observed_cases)))")

# ─── Define PSM dynamics ─────────────────────────────────────────

# We track S, I, R, and cumulative cases C
# The unknown function β(t) is the time-varying transmission rate
function sir_psm!(du, u, p, t)
    S, I, R, C = u
    β = p.β(t)         # unknown function of time
    γ_val = p.γ        # known recovery rate
    N_val = p.N        # known population size
    infection = max(β, 0.0) * S * I / N_val
    du[1] = -infection
    du[2] = infection - γ_val * I
    du[3] = γ_val * I
    du[4] = infection
end

# ─── Set up problem ──────────────────────────────────────────────

# β(t) over [0, 80] with 15 knots
approx_β = BSplineApproximator(:β, (0.0, 80.0), 15;
                                initial = x -> 0.3)

u0 = [N - 10.0, 10.0, 0.0, 0.0]

# Observed: Poisson-noised cumulative case counts (state 4).
# The package's Poisson likelihood uses the identity link, so the model
# prediction C(t) is compared with the counts on the response scale.
data_values = reshape(cum_obs, :, 1)

prob = PSMProblem(sir_psm!, u0, tspan, [approx_β];
    data_times = data_times,
    data_values = data_values,
    obs_to_state = [4],  # observe cumulative cases (state 4)
    known_params = (γ = γ, N = N),
    likelihood = Poisson(),  # counts → Poisson (identity link)
    solver = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6)

# ─── Solve ────────────────────────────────────────────────────────

println("\nFitting SIR PSM with spline β(t) via LAML...")
sol = solve(prob, LAML(maxiters=80, verbose=true))

println("\n" * "="^60)
println("Results:")
println("  Data loss:      ", @sprintf("%.4e", sol.data_loss))
println("  EDF:            ", round(sol.edf, digits=2))
println("  Smoothing λ:    ", [round(s, sigdigits=4) for s in sol.smoothing_params])

# ─── Compare fitted β(t) to truth ────────────────────────────────

if haskey(sol.unknown_functions, :β)
    β_eval = sol.unknown_functions[:β]
    println("\nβ(t) comparison (true vs fitted):")
    for t in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        @printf("  t=%2.0f: true=%.4f, fitted=%.4f\n",
                t, true_β(t), β_eval(t))
    end
end
