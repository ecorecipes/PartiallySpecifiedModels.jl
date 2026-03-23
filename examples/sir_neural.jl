# SIR epidemic model with neural network force of infection
#
# This example demonstrates the UDE (Universal Differential Equation)
# approach: the transmission rate β(I) is modeled as a neural network
# that depends on the number of infected individuals.
#
# This shows how PartiallySpecifiedModels.jl can use Lux.jl neural
# networks as approximators instead of B-splines.
#
# Note: Neural approximators don't have automatic smoothing penalties,
# so the LAML solver acts as an optimizer without penalty terms.
#
# Reference: Based on the SIR UDE example from
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ude/ude.md

using PartiallySpecifiedModels
using OrdinaryDiffEq
using Lux
using Random
using Printf

Random.seed!(123)

# ─── Generate synthetic epidemic data ─────────────────────────────

N = 1000.0
γ = 0.25     # recovery rate
tspan = (0.0, 40.0)
data_times = collect(0.5:0.5:40.0)

# True β depends on I (behavioral response to epidemic size)
true_β(I) = 0.5 * exp(-0.005 * I)  # people reduce contact as I grows

function sir_true!(du, u, p, t)
    S, I, R = u
    β = true_β(I)
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * I
    du[3] = γ * I
end

u0_true = [N - 1.0, 1.0, 0.0]
prob_true = ODEProblem(sir_true!, u0_true, tspan)
sol_true = OrdinaryDiffEq.solve(prob_true, Tsit5(); saveat=data_times)

# Extract observations: S and I with Gaussian noise
S_obs = Float64[sol_true(t)[1] + 5.0 * randn() for t in data_times]
I_obs = Float64[max(sol_true(t)[2] + 2.0 * randn(), 0.1) for t in data_times]

println("Synthetic SIR-UDE data: $(length(data_times)) observations")
println("Peak I: $(round(maximum([sol_true(t)[2] for t in data_times]), digits=1))")

# ─── Define PSM dynamics with neural β ───────────────────────────

function sir_neural!(du, u, p, t)
    S, I, R = u
    β = p.β(I)           # neural network: β depends on I
    γ_val = p.γ
    N_val = p.N
    foi = max(β, 0.001) * S * I / N_val
    du[1] = -foi
    du[2] = foi - γ_val * I
    du[3] = γ_val * I
end

# ─── Neural network approximator ─────────────────────────────────

# Small network: 1 input (I) → 8 hidden → 1 output
# softplus output to keep β > 0
nn_model = Lux.Chain(
    Lux.Dense(1, 8, Lux.relu),
    Lux.Dense(8, 1, Lux.softplus)
)

approx_β = NeuralApproximator(:β, nn_model)

println("Neural network parameters: $(nparams(approx_β))")

# ─── Set up problem ──────────────────────────────────────────────

u0 = [N - 1.0, 1.0, 0.0]

# Observe both S and I
data_values = hcat(S_obs, I_obs)

prob = PSMProblem(sir_neural!, u0, tspan, [approx_β];
    data_times = data_times,
    data_values = data_values,
    obs_to_state = [1, 2],  # observe S and I
    known_params = (γ = γ, N = N),
    likelihood = Gaussian(),
    solver = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6)

# ─── Solve ────────────────────────────────────────────────────────

println("\nFitting SIR with neural β(I) via LAML...")
sol = solve(prob, LAML(maxiters=50, verbose=true))

println("\n" * "="^60)
println("Results:")
println("  Data loss (SS): ", @sprintf("%.4e", sol.data_loss))
println("  EDF:            ", round(sol.edf, digits=2))

# ─── Compare fitted β(I) to truth ────────────────────────────────

println("\nβ(I) comparison (true vs fitted):")
for I_val in [1.0, 10.0, 50.0, 100.0, 200.0]
    true_val = true_β(I_val)
    @printf("  I=%5.1f: true=%.4f\n", I_val, true_val)
end
