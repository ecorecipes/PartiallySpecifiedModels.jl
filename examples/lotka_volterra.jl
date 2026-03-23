# Lotka-Volterra model with unknown functional responses fit to hare-lynx data
#
# This example fits a partially specified Lotka-Volterra model to the
# classic Hudson Bay Company hare-lynx pelting data (1845-1935).
#
# The unknown functions are:
#   r(H) — per-capita hare growth rate as a function of hare density
#   δ(L) — per-capita lynx death rate as a function of lynx density
#
# The known parameter is:
#   α — predation rate (hare eaten per lynx per unit time)
#
# Both unknown functions are modeled with penalized cubic B-splines,
# and smoothing parameters are estimated automatically via LAML (≡ REML).

using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using DelimitedFiles
using Printf
using LinearAlgebra

# ─── Load data ────────────────────────────────────────────────────

data_path = joinpath(@__DIR__, "..", "data", "hare_lynx.csv")
raw = readdlm(data_path, ',', Any; header=true)
data = Float64.(raw[1])
years = data[:, 1]
hare = data[:, 2]
lynx = data[:, 3]

println("Hare-Lynx data: $(length(years)) observations, $(Int(years[1]))-$(Int(years[end]))")
println("Hare range: $(round(minimum(hare), digits=1)) - $(round(maximum(hare), digits=1))")
println("Lynx range: $(round(minimum(lynx), digits=1)) - $(round(maximum(lynx), digits=1))")

# ─── Define dynamics ──────────────────────────────────────────────

function lotka_volterra!(du, u, p, t)
    H, L = u
    r = p.r(H)         # unknown: per-capita hare growth rate
    δ = p.δ(L)         # unknown: per-capita lynx death rate
    α = p.α            # known: predation rate
    du[1] = r * H - α * H * L
    du[2] = α * H * L - δ * L
end

# ─── Define approximators ────────────────────────────────────────

# Growth rate r(H): expect positive values ~0.5, modeled over hare density range
approx_r = BSplineApproximator(:r, (0.0, 180.0), 10;
                                initial = x -> 0.5)

# Death rate δ(L): expect positive values ~0.5, modeled over lynx density range
approx_δ = BSplineApproximator(:δ, (0.0, 80.0), 10;
                                initial = x -> 0.3)

# ─── Set up problem ──────────────────────────────────────────────

# Initial conditions from first data point
u0 = [hare[1], lynx[1]]
tspan = (years[1], years[end])

# Data: both hare and lynx are observed
data_values = hcat(hare, lynx)

prob = PSMProblem(lotka_volterra!, u0, tspan,
    [approx_r, approx_δ];
    data_times = years,
    data_values = data_values,
    obs_to_state = [1, 2],
    known_params = (α = 0.01,),
    likelihood = Gaussian(),
    solver = BS3(),
    abstol = 1e-6,
    reltol = 1e-6,
    maxiters = 10000)  # allow ODE solver more steps

# ─── Solve ────────────────────────────────────────────────────────

println("\nFitting Lotka-Volterra PSM with LAML...")
sol = solve(prob, LAML(maxiters=100, verbose=true))

println("\n" * "="^60)
println("Results:")
println("  Data loss (SS): ", @sprintf("%.4e", sol.data_loss))
println("  Penalized obj:  ", @sprintf("%.4e", sol.objective))
println("  EDF:            ", round(sol.edf, digits=2))
println("  Smoothing λ:    ", [round(s, sigdigits=4) for s in sol.smoothing_params])

# ─── Evaluate fitted unknown functions ────────────────────────────

if haskey(sol.unknown_functions, :r)
    r_eval = sol.unknown_functions[:r]
    println("\nFitted r(H) at selected densities:")
    for H in [10.0, 50.0, 100.0, 150.0]
        println("  r($H) = ", round(r_eval(H), digits=4))
    end
end

if haskey(sol.unknown_functions, :δ)
    δ_eval = sol.unknown_functions[:δ]
    println("\nFitted δ(L) at selected densities:")
    for L in [5.0, 20.0, 40.0, 60.0]
        println("  δ($L) = ", round(δ_eval(L), digits=4))
    end
end

# ─── Predictions ──────────────────────────────────────────────────

pred = sol.fitted_values
println("\nPrediction quality (first 10 years):")
println("  Year  |  Hare_obs  Hare_fit  |  Lynx_obs  Lynx_fit")
for i in 1:min(10, length(years))
    @printf("  %4d  |  %7.2f   %7.2f   |  %7.2f   %7.2f\n",
            Int(years[i]), hare[i], pred[i,1], lynx[i], pred[i,2])
end
