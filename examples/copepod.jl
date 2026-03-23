# Copepod population dynamics with unknown vital rates
#
# Fits a partially specified 11-stage copepod (*Calanus finmarchicus*)
# population model where recruitment, naupliar mortality, and copepodite
# mortality are unknown functions of time, modeled with penalized B-splines.
#
# The model is from:
#   Wood (2001) "Partially specified ecological models", Ecological Monographs
#
# The model tracks 11 developmental stages (6 naupliar + 5 copepodite)
# with exponential stage durations. Three biological rates are treated as
# unknown functions of time:
#   R(t)   — egg recruitment rate
#   μⱼ(t)  — naupliar (juvenile) per-capita death rate
#   μₐ(t)  — copepodite (adult) per-capita death rate
#
# Known parameters are the mean stage durations from laboratory studies.

using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using DelimitedFiles
using Printf

# ─── Load data ────────────────────────────────────────────────────

data_path = joinpath(@__DIR__, "..", "data", "cop.dat")
raw = readdlm(data_path)
data_times = Float64.(raw[:, 1])
data_values = Float64.(raw[:, 2:12])  # 11 stage abundances

n_times = length(data_times)
n_stages = size(data_values, 2)
println("Copepod data: $n_times time points, $n_stages stages")
println("Time range: $(data_times[1]) - $(data_times[end]) days")

# ─── Known parameters ────────────────────────────────────────────
# Mean stage durations (days) from laboratory rearing experiments

stage_durations = (
    c1  = 0.75,   # Naupliar stage 1
    c2  = 1.4,    # Naupliar stage 2
    c3  = 4.55,   # Naupliar stage 3
    c4  = 2.8,    # Naupliar stage 4
    c5  = 2.5,    # Naupliar stage 5
    c6  = 1.7,    # Naupliar stage 6
    c7  = 3.5,    # Copepodite stage 1
    c8  = 3.1,    # Copepodite stage 2
    c9  = 3.2,    # Copepodite stage 3
    c10 = 3.7,    # Copepodite stage 4
    c11 = 4.7,    # Copepodite stage 5 (adult)
)
durations = collect(values(stage_durations))

# ─── ODE dynamics ─────────────────────────────────────────────────
#
# Exponential stage-duration model:
#   ds_i/dt = inflow_i - s_i/c_i - death_i × s_i
#
# where inflow_1 = R(t) × 1000 (recruitment), inflow_i = s_{i-1}/c_{i-1}
# death rate = μⱼ(t) for stages 1-6, μₐ(t) for stages 7-11

function copepod!(du, u, p, t)
    R_t = p.R(t) * 1000.0    # recruitment (scaled for numerical stability)
    μ_j = p.mu_j(t)          # naupliar death rate
    μ_a = p.mu_a(t)          # copepodite death rate

    inflow = R_t
    for i in 1:11
        death = i <= 6 ? μ_j : μ_a
        du[i] = inflow - u[i] / durations[i] - death * u[i]
        inflow = u[i] / durations[i]
    end
end

# ─── Initial conditions ──────────────────────────────────────────
# Steady-state abundances given initial vital rates

function compute_u0(p)
    R0 = p.R(0.0) * 1000.0
    μ_j0 = p.mu_j(0.0)
    μ_a0 = p.mu_a(0.0)

    u0 = zeros(Float64, 11)
    inflow = R0
    for i in 1:11
        death = i <= 6 ? μ_j0 : μ_a0
        u0[i] = inflow / (1.0 / durations[i] + death)
        inflow = u0[i] / durations[i]
    end
    u0
end

# ─── Define unknown function approximators ────────────────────────

# R(t): recruitment — Gaussian bump initial guess (peak at ~30 days)
approx_R = BSplineApproximator(:R, (0.0, 90.0), 15;
    initial = t -> begin
        μ, σ = 30.0, 20.0
        (0.01 + 0.39894228 / σ * exp(-(t - μ)^2 / σ^2)) * 400.0
    end)

# μⱼ(t): naupliar death — exponentially decaying initial guess
approx_mu_j = BSplineApproximator(:mu_j, (0.0, 90.0), 15;
    initial = t -> 0.1 * exp(-0.02 * t))

# μₐ(t): copepodite death — constant initial guess
approx_mu_a = BSplineApproximator(:mu_a, (0.0, 90.0), 15;
    initial = t -> 0.1)

# ─── Build PSM problem ───────────────────────────────────────────

prob = PSMProblem(copepod!, compute_u0, (0.0, 90.0),
    [approx_R, approx_mu_j, approx_mu_a];
    data_times = data_times,
    data_values = data_values,
    obs_to_state = collect(1:11),
    known_params = NamedTuple(),  # durations captured in closure
    likelihood = Gaussian(),
    solver = BS3(),
    abstol = 1e-6,
    reltol = 1e-6,
    maxiters = 10000)

# ─── Solve ────────────────────────────────────────────────────────

println("\nFitting copepod PSM with LAML (3 unknown functions × 15 knots = 45 params)...")
sol = solve(prob, LAML(maxiters=100, verbose=true))

println("\n" * "="^60)
println("Results:")
println("  Data loss (SS):  ", @sprintf("%.4e", sol.data_loss))
println("  Penalized obj:   ", @sprintf("%.4e", sol.objective))
println("  EDF:             ", round(sol.edf, digits=2))
println("  Smoothing λ:     ", [round(s, sigdigits=4) for s in sol.smoothing_params])

# ─── Evaluate fitted unknown functions ────────────────────────────

println("\nFitted R(t) (recruitment × 1000):")
if haskey(sol.unknown_functions, :R)
    R_eval = sol.unknown_functions[:R]
    for t in 0.0:10.0:90.0
        @printf("  t=%2.0f: R=%.4f (flux=%.1f)\n", t, R_eval(t), R_eval(t) * 1000.0)
    end
end

println("\nFitted μⱼ(t) (naupliar death rate):")
if haskey(sol.unknown_functions, :mu_j)
    mj_eval = sol.unknown_functions[:mu_j]
    for t in 0.0:10.0:90.0
        @printf("  t=%2.0f: μⱼ=%.6f\n", t, mj_eval(t))
    end
end

println("\nFitted μₐ(t) (copepodite death rate):")
if haskey(sol.unknown_functions, :mu_a)
    ma_eval = sol.unknown_functions[:mu_a]
    for t in 0.0:10.0:90.0
        @printf("  t=%2.0f: μₐ=%.6f\n", t, ma_eval(t))
    end
end

# ─── Prediction quality ──────────────────────────────────────────

pred = sol.fitted_values
println("\nPrediction quality (stage 1 and 7):")
println("  Time | Stage1_obs  Stage1_fit | Stage7_obs  Stage7_fit")
for i in 1:n_times
    @printf("  %4.0f | %10.1f  %10.1f | %10.1f  %10.1f\n",
            data_times[i], data_values[i,1], pred[i,1],
            data_values[i,7], pred[i,7])
end
