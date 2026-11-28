# Copepod stage-structured population dynamics with unknown vital rates
#
# Fits Wood (2001)'s 11-stage copepod (*Calanus finmarchicus*) fixed-duration
# delay-differential equation (DDE) population model where egg recruitment,
# naupliar mortality, and copepodite mortality are strictly positive unknown
# functions of time modeled with shape-constrained penalized B-splines.
#
# Source & provenance:
#   Wood (2001) "Partially specified ecological models", Ecological Monographs 71(1):1–25.
#   The bundled dataset `data/cop.dat` is the simulated 11-stage benchmark dataset
#   distributed with Wood's `DDEfit` software (10 sampling days over [0, 90] days
#   for naupliar stages N1–N6 and pre-adult copepodite stages C1–C5).
#
# Fixed-duration stage-structured DDE (Nisbet & Gurney 1983; Wood 1994, 2001):
#   Let τ_i be the fixed duration of stage i (i = 1..11) and T_i = Σ_{k=1}^i τ_k
#   be cumulative age at exit of stage i. Tracking cumulative naupliar and
#   copepodite mortalities
#       dL_j/dt = μ_j(t),     dL_a/dt = μ_a(t),
#   an individual maturing out of stage i at time t entered stage 1 at t - T_i
#   and survived with probability exp(-cum_mort_i(t)), giving exact maturation
#   flux M_i(t) = R_1(t - T_i) exp(-cum_mort_i(t)) and stage dynamics:
#       ds_i/dt = R_i(t) - M_i(t) - μ_i(t) s_i(t),   R_1 = 1000 R(t), R_i = M_{i-1}.

using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using OrdinaryDiffEq
using DelimitedFiles
using Statistics
using Printf

# ─── Load data ────────────────────────────────────────────────────

data_path = joinpath(@__DIR__, "..", "data", "cop.dat")
raw = readdlm(data_path)
data_times = Float64.(raw[:, 1])
data_values = Float64.(raw[:, 2:12])  # 11 stage abundances: N1–N6, C1–C5

n_times = length(data_times)
n_stages = size(data_values, 2)
stage_names = ["N1", "N2", "N3", "N4", "N5", "N6", "C1", "C2", "C3", "C4", "C5"]
println("Copepod DDEfit simulated benchmark: $n_times time points, $n_stages stages (N1–N6, C1–C5)")
println("Time range: $(data_times[1]) - $(data_times[end]) days")

# ─── Stage-scale inverse-variance weights ─────────────────────────
# Raw counts span ~0 to 38,783 across stages; inverse-variance stage weights
# ensure late copepodite stages (C1–C5) contribute equitably alongside N1–N6.

col_stds = [max(std(data_values[:, j]), 1.0) for j in 1:n_stages]
w_stage = 1.0 ./ (col_stds .^ 2)
w_stage ./= mean(w_stage)   # normalize mean weight to 1.0
data_weights = repeat(reshape(w_stage, 1, n_stages), n_times, 1)

# ─── Known stage durations & fixed delays ─────────────────────────
# Mean stage durations (days) from laboratory rearing experiments (Wood 2001)

stage_durations = (
    c1  = 0.75,   # Naupliar stage N1
    c2  = 1.4,    # Naupliar stage N2
    c3  = 4.55,   # Naupliar stage N3
    c4  = 2.8,    # Naupliar stage N4
    c5  = 2.5,    # Naupliar stage N5
    c6  = 1.7,    # Naupliar stage N6
    c7  = 3.5,    # Copepodite stage C1
    c8  = 3.1,    # Copepodite stage C2
    c9  = 3.2,    # Copepodite stage C3
    c10 = 3.7,    # Copepodite stage C4
    c11 = 4.7,    # Copepodite stage C5 (final pre-adult stage)
)
const COPEPOD_DURATIONS = collect(values(stage_durations))
const COPEPOD_CUM_T = cumsum(COPEPOD_DURATIONS)
const COPEPOD_T_N6 = COPEPOD_CUM_T[6]
const COPEPOD_DELAYS = sort(unique(vcat(
    COPEPOD_CUM_T,
    [COPEPOD_CUM_T[i] - COPEPOD_T_N6 for i in 7:11]
)))

# Smooth helper for (1 - exp(-x)) / x safe at x -> 0 and under ForwardDiff Duals
@inline _expm1_ratio(x) = abs(x) < 1e-8 ? (1 - x / 2 + x^2 / 6) : (1 - exp(-x)) / x

# ─── Fixed-duration stage-structured DDE dynamics ─────────────────
# States u[1:11]: stage abundances (N1–N6, C1–C5)
# State  u[12]:   cumulative naupliar mortality L_j(t) = ∫_0^t μ_j(s) ds
# State  u[13]:   cumulative copepodite mortality L_a(t) = ∫_0^t μ_a(s) ds

function copepod_dde!(du, u, h, p, t)
    R_t = p.R(t) * 1000.0
    μ_j = p.mu_j(t)
    μ_a = p.mu_a(t)

    du[12] = μ_j
    du[13] = μ_a

    Lj_now = u[12]
    La_now = u[13]
    inflow = R_t

    @inbounds for i in 1:11
        t_rec = t - COPEPOD_CUM_T[i]
        R_rec = p.R(t_rec <= 0.0 ? 0.0 : t_rec) * 1000.0
        if i <= 6
            Lj_rec = h(p, t_rec; idxs=12)
            cum_mort = Lj_now - Lj_rec
            death = μ_j
        else
            t_trans = t - (COPEPOD_CUM_T[i] - COPEPOD_T_N6)
            Lj_rec = h(p, t_rec; idxs=12)
            Lj_trans = h(p, t_trans; idxs=12)
            La_trans = h(p, t_trans; idxs=13)
            cum_mort = (Lj_trans - Lj_rec) + (La_now - La_trans)
            death = μ_a
        end
        maturation = R_rec * exp(-cum_mort)
        du[i] = inflow - maturation - death * u[i]
        inflow = maturation
    end
end

# ─── Exact steady-state history & initial conditions (t ≤ 0) ──────

function copepod_history(p, t)
    R0 = p.R(0.0) * 1000.0
    μ_j0 = p.mu_j(0.0)
    μ_a0 = p.mu_a(0.0)
    T = promote_type(typeof(R0), typeof(μ_j0), typeof(μ_a0), typeof(t))
    u = zeros(T, 13)
    inflow = T(R0)
    @inbounds for i in 1:11
        death = i <= 6 ? T(μ_j0) : T(μ_a0)
        τ = COPEPOD_DURATIONS[i]
        u[i] = inflow * τ * _expm1_ratio(death * τ)
        inflow = inflow * exp(-death * τ)
    end
    u[12] = T(μ_j0) * t
    u[13] = T(μ_a0) * t
    u
end

copepod_u0(p) = copepod_history(p, 0.0)

# ─── Strictly positive unknown vital rate approximators ───────────

approx_R = ShapeConstrainedBSplineApproximator(:R, (0.0, 90.0), 12, :positive;
    initial = t -> begin
        μ, σ = 30.0, 20.0
        (0.01 + 0.39894228 / σ * exp(-(t - μ)^2 / σ^2)) * 400.0
    end)

approx_mu_j = ShapeConstrainedBSplineApproximator(:mu_j, (0.0, 90.0), 12, :positive;
    initial = t -> 0.1 * exp(-0.02 * t))

approx_mu_a = ShapeConstrainedBSplineApproximator(:mu_a, (0.0, 90.0), 12, :positive;
    initial = t -> 0.1)

# ─── Build DDE PSM problem ────────────────────────────────────────

prob = PSMProblem(copepod_dde!, copepod_u0, (0.0, 90.0),
    [approx_R, approx_mu_j, approx_mu_a];
    data_times = data_times,
    data_values = data_values,
    data_weights = data_weights,
    obs_to_state = collect(1:11),
    delays = COPEPOD_DELAYS,
    history = copepod_history,
    known_params = NamedTuple(),
    likelihood = GaussianLikelihood(),
    solver = BS3(),
    order_discontinuity_t0 = 2,
    abstol = 1e-5,
    reltol = 1e-5,
    maxiters = 10_000)

# ─── Solve with LAML ──────────────────────────────────────────────

println("\nFitting Wood (2001) fixed-duration DDE copepod PSM with LAML...")
sol = solve(prob, LAML(maxiters=40, verbose=true))

println("\n" * "="^60)
println("Results:")
println("  Converged:       ", sol.convergence.converged, " ($(sol.convergence.reason))")
println("  Weighted loss:   ", @sprintf("%.4e", sol.data_loss))
println("  Penalized obj:   ", @sprintf("%.4e", sol.objective))
println("  EDF:             ", round(sol.edf, digits=2))
println("  Smoothing λ:     ", [round(s, sigdigits=4) for s in sol.smoothing_params])

# ─── Evaluate fitted positive vital rates ─────────────────────────

R_eval = sol.unknown_functions[:R]
mj_eval = sol.unknown_functions[:mu_j]
ma_eval = sol.unknown_functions[:mu_a]

println("\nFitted strictly positive vital rates:")
println("  Day |  R(t)×1000  |   μ_j(t)  |   μ_a(t)")
for t in 0.0:10.0:90.0
    @printf("  %3.0f | %11.1f | %9.5f | %9.5f\n",
            t, R_eval(t) * 1000.0, mj_eval(t), ma_eval(t))
end

# ─── Prediction quality across early & late stages ────────────────

pred = sol.fitted_values
println("\nPrediction quality (N1, C1, and C5):")
println("  Day |   N1_obs    N1_fit |   C1_obs    C1_fit |   C5_obs    C5_fit")
for i in 1:n_times
    @printf("  %3.0f | %8.1f  %8.1f | %8.1f  %8.1f | %8.1f  %8.1f\n",
            data_times[i], data_values[i, 1], pred[i, 1],
            data_values[i, 7], pred[i, 7],
            data_values[i, 11], pred[i, 11])
end
