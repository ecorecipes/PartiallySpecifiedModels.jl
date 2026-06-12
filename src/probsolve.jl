# ─── Probabilistic ODE Solver (joint state space) ────────────────────
#
# Solves an ODE initial value problem with a Kalman filter/smoother on an
# integrated Brownian motion (IBM) prior.
#
# Unlike a per-variable block-diagonal filter, this implementation tracks
# the FULL joint state X = [x₁, x₁′, …, x₁^{(q-1)}, x₂, …] so that the ODE
# interrogation can use the complete Jacobian ∂f/∂x — including the
# off-diagonal coupling between state variables — as in the first-order
# (EKF1) linearization of Krämer et al. (2021)/Tronarp et al. (2019).
# The global diffusion is calibrated by quasi-maximum-likelihood from the
# ODE innovations (Bosch, Tronarp & Hennig 2021).
#
# Forward pass: Kalman filter with ODE interrogation at each step.
# Backward pass: RTS smoother (and, for Fenrir, data conditioning).
#
# Reference: Tronarp et al (2019, 2022), Krämer et al (2021),
#            Bosch et al (2021), Wu & Lysy (2024)

# ── joint state-space building blocks ────────────────────────────────

"""
    _joint_ibm(dt, q, sigma) -> (A, Q)

Block-diagonal joint transition `A` (D×D) and process-noise `Q` (D×D) for
`n_vars = length(sigma)` independent q-times integrated Wiener processes,
D = n_vars·q. Each block uses the per-variable scale `sigma[k]`.
"""
function _joint_ibm(dt::Float64, q::Int, sigma::Vector{Float64})
    n_vars = length(sigma)
    D = n_vars * q
    Q_base, R_base = ibm_state(dt, q, 1.0)
    A = zeros(D, D)
    Q = zeros(D, D)
    for k in 1:n_vars
        idx = ((k-1)*q+1):(k*q)
        A[idx, idx] .= Q_base
        Q[idx, idx] .= sigma[k]^2 .* R_base
    end
    A, Q
end

"""
    _joint_selectors(n_vars, q) -> (E0, E1)

Selection matrices (each n_vars × D): `E0` picks the value x_k^{(0)} and
`E1` picks the first derivative x_k^{(1)} of every state variable.
"""
function _joint_selectors(n_vars::Int, q::Int)
    D = n_vars * q
    E0 = zeros(n_vars, D)
    E1 = zeros(n_vars, D)
    for k in 1:n_vars
        E0[k, (k-1)*q + 1] = 1.0
        E1[k, (k-1)*q + 2] = 1.0
    end
    E0, E1
end

"""
    _joint_init(ode_fun!, u0, t0, p, q) -> X0

Initial joint state: value and ODE-derived first derivative per variable,
higher derivatives zero.
"""
function _joint_init(ode_fun!, u0::AbstractVector, t0::Float64, p, q::Int)
    n_vars = length(u0)
    du = zeros(n_vars)
    ode_fun!(du, Float64.(u0), p, t0)
    X0 = zeros(n_vars * q)
    for k in 1:n_vars
        X0[(k-1)*q + 1] = u0[k]
        X0[(k-1)*q + 2] = du[k]
    end
    X0
end

"""
    _joint_interrogate(ode_fun!, E0, E1, t, μ_pred, p, n_vars; method)

Linearize the ODE residual r(X) = E1·X − f(E0·X, t) around the predicted
mean and return the measurement model (H, b) so that the pseudo-observation
is `0 = H·X + b + noise`.

- `:kramer` (EKF1): full first-order Taylor, H = E1 − J·E0 with the COMPLETE
  Jacobian J = ∂f/∂u (off-diagonal coupling included), b = J·u − f(u).
- `:schober` (EKF0): H = E1, b = −f(u) (no Jacobian).

In both cases the innovation under z=0 is f(u) − E1·μ_pred, i.e. the ODE
defect, but the EKF1 measurement matrix propagates cross-variable
sensitivity into the covariance update.
"""
function _joint_interrogate(ode_fun!, E0::Matrix{Float64}, E1::Matrix{Float64},
                            t::Float64, μ_pred::Vector{Float64}, p, n_vars::Int;
                            method::Symbol=:kramer)
    u = E0 * μ_pred
    du = zeros(n_vars)
    ode_fun!(du, u, p, t)

    if method == :schober
        H = copy(E1)
        b = -du
        return H, b
    end

    # Full Jacobian J = ∂f/∂u via central finite differences.
    J = zeros(n_vars, n_vars)
    for j in 1:n_vars
        h = max(abs(u[j]), 1.0) * 1e-7
        up = copy(u); up[j] += h
        um = copy(u); um[j] -= h
        dup = zeros(n_vars); dum = zeros(n_vars)
        ode_fun!(dup, up, p, t)
        ode_fun!(dum, um, p, t)
        J[:, j] .= (dup .- dum) ./ (2h)
    end

    H = E1 - J * E0
    b = J * u - du
    H, b
end

"""
    probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                     interrogate=:kramer, calibrate=true)

Forward (filtering) pass of the joint probabilistic ODE solver with global
diffusion calibration. Returns a `Dict` holding the joint filter/predict
means and covariances, the transition matrix, the calibrated diffusion
`ssq`, and the time grid.
"""
function probsolve_filter(ode_fun!, p, u0::AbstractVector,
                          tspan::Tuple{Float64, Float64},
                          n_steps::Int, n_deriv::Int,
                          sigma::Vector{Float64};
                          interrogate::Symbol=:kramer,
                          calibrate::Bool=true)
    t_min, t_max = tspan
    dt = (t_max - t_min) / n_steps
    n_vars = length(u0)
    q = n_deriv
    D = n_vars * q

    A, Qmat = _joint_ibm(dt, q, sigma)
    E0, E1 = _joint_selectors(n_vars, q)
    X0 = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    V = Matrix(1e-10 * I, n_vars, n_vars)  # tiny ODE-measurement nugget

    μ_pred = Vector{Vector{Float64}}(undef, n_steps + 1)
    Σ_pred = Vector{Matrix{Float64}}(undef, n_steps + 1)
    μ_filt = Vector{Vector{Float64}}(undef, n_steps + 1)
    Σ_filt = Vector{Matrix{Float64}}(undef, n_steps + 1)

    μ_pred[1] = X0
    Σ_pred[1] = zeros(D, D)
    μ_filt[1] = X0
    Σ_filt[1] = zeros(D, D)

    calib_acc = 0.0  # Σ_n νᵀ S⁻¹ ν  (quasi-MLE diffusion statistic)

    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps

        μp = A * μ_filt[n]
        Σp = A * Σ_filt[n] * A' + Qmat
        Σp = 0.5 * (Σp + Σp')

        H, b = _joint_interrogate(ode_fun!, E0, E1, t_n, μp, p, n_vars;
                                  method=interrogate)
        ν = -(H * μp + b)                 # innovation at z = 0
        S = H * Σp * H' + V
        S = 0.5 * (S + S')
        Sf = cholesky(Symmetric(S), check=false)
        Sinv_ν = issuccess(Sf) ? (Sf \ ν) : (pinv(S) * ν)
        calib_acc += dot(ν, Sinv_ν)

        K = (Σp * H') * (issuccess(Sf) ? inv(Sf) : pinv(S))
        μf = μp + K * ν
        Σf = Σp - K * H * Σp
        Σf = 0.5 * (Σf + Σf')

        μ_pred[n+1] = μp; Σ_pred[n+1] = Σp
        μ_filt[n+1] = μf; Σ_filt[n+1] = Σf
    end

    # Global diffusion calibration: scale all covariances by the quasi-MLE
    # σ̂² = (1/(N·n_vars)) Σ_n νᵀ S⁻¹ ν.  With the tiny nugget the filter
    # gains (hence the means) are ~scale invariant, so a post-hoc rescale is
    # equivalent to re-running with the calibrated diffusion.
    ssq = calibrate ? max(calib_acc / max(n_steps * n_vars, 1), 1e-12) : 1.0
    if calibrate && ssq != 1.0
        for n in 1:(n_steps + 1)
            Σ_pred[n] .*= ssq
            Σ_filt[n] .*= ssq
        end
    end

    times = collect(range(t_min, t_max, length=n_steps + 1))
    Dict("μ_pred" => μ_pred, "Σ_pred" => Σ_pred,
         "μ_filt" => μ_filt, "Σ_filt" => Σ_filt,
         "times" => times, "A" => A, "ssq" => ssq,
         "n_vars" => n_vars, "q" => q)
end

"""
    probsolve_smooth(filt_out, n_vars)

Backward RTS smoother on the joint filter output. Returns posterior
mean/variance in the per-variable nested format
`(μ_smooth[n][k]::Vector, Σ_smooth[n][k]::Matrix)` expected by callers
(the per-variable q×q diagonal block of the joint covariance).
"""
function probsolve_smooth(filt_out::Dict, n_vars::Int)
    μ_filt = filt_out["μ_filt"]; Σ_filt = filt_out["Σ_filt"]
    μ_pred = filt_out["μ_pred"]; Σ_pred = filt_out["Σ_pred"]
    A = filt_out["A"]; q = filt_out["q"]
    n_steps = length(μ_filt) - 1
    D = n_vars * q

    μJ = Vector{Vector{Float64}}(undef, n_steps + 1)
    ΣJ = Vector{Matrix{Float64}}(undef, n_steps + 1)
    μJ[end] = μ_filt[end]
    ΣJ[end] = Σ_filt[end]

    for n in n_steps:-1:1
        Σpn = Symmetric(Σ_pred[n+1])
        Σpf = cholesky(Σpn + 1e-12 * I, check=false)
        G = issuccess(Σpf) ? (Σ_filt[n] * A') / Σpf : (Σ_filt[n] * A') * pinv(Σ_pred[n+1])
        μs = μ_filt[n] + G * (μJ[n+1] - μ_pred[n+1])
        Σs = Σ_filt[n] + G * (ΣJ[n+1] - Σ_pred[n+1]) * G'
        Σs = 0.5 * (Σs + Σs')
        μJ[n] = μs; ΣJ[n] = Σs
    end

    # Convert joint → per-variable nested format.
    μ_smooth = Vector{Vector{Vector{Float64}}}(undef, n_steps + 1)
    Σ_smooth = Vector{Vector{Matrix{Float64}}}(undef, n_steps + 1)
    for n in 1:(n_steps + 1)
        μ_smooth[n] = [μJ[n][((k-1)*q+1):(k*q)] for k in 1:n_vars]
        Σ_smooth[n] = [ΣJ[n][((k-1)*q+1):(k*q), ((k-1)*q+1):(k*q)] for k in 1:n_vars]
    end
    μ_smooth, Σ_smooth
end

"""
    probsolve(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
              interrogate=:kramer)

Probabilistic ODE solve. Returns `(μ_smooth, Σ_smooth, times)` in the
per-variable nested format described in [`probsolve_smooth`](@ref).
"""
function probsolve(ode_fun!, p, u0::AbstractVector,
                   tspan::Tuple{Float64, Float64},
                   n_steps::Int, n_deriv::Int,
                   sigma::Vector{Float64};
                   interrogate::Symbol=:kramer)
    n_vars = length(u0)
    filt_out = probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                interrogate=interrogate)
    μ_smooth, Σ_smooth = probsolve_smooth(filt_out, n_vars)
    μ_smooth, Σ_smooth, filt_out["times"]
end

"""
    basic_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                 obs_data, obs_times, obs_to_state, obs_var; interrogate=:kramer)

Plug-in data likelihood: solve the ODE probabilistically (with full EKF1
coupling) and evaluate the Gaussian data likelihood at the posterior mean.
"""
function basic_loglik(ode_fun!, p, u0::AbstractVector,
                      tspan::Tuple{Float64, Float64},
                      n_steps::Int, n_deriv::Int,
                      sigma::Vector{Float64},
                      obs_data::Matrix{Float64},
                      obs_times::Vector{Float64},
                      obs_to_state::Vector{Int},
                      obs_var::Float64;
                      interrogate::Symbol=:kramer)
    μ_smooth, _, times = probsolve(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                   interrogate=interrogate)
    n_obs = size(obs_data, 2); n_t = size(obs_data, 1)
    ll = 0.0
    for i in 1:n_t
        idx = clamp(searchsortedfirst(times, obs_times[i]), 1, length(times))
        for j in 1:n_obs
            sk = obs_to_state[j]
            pred = μ_smooth[idx][sk][1]
            ll += -0.5 * log(2π * obs_var) - 0.5 * (obs_data[i, j] - pred)^2 / obs_var
        end
    end
    ll
end

"""
    fenrir_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                  obs_data, obs_times, obs_to_state, obs_var; interrogate=:kramer)

Fenrir marginal data likelihood (Tronarp et al. 2022) on the joint state
space: forward filter conditioned on the ODE (with EKF1 coupling and
calibrated diffusion), then a backward Gauss–Markov pass that conditions on
the data and accumulates the data evidence `Σ_m log N(y_m; D_m b_m, …)`.
"""
function fenrir_loglik(ode_fun!, p, u0::AbstractVector,
                       tspan::Tuple{Float64, Float64},
                       n_steps::Int, n_deriv::Int,
                       sigma::Vector{Float64},
                       obs_data::Matrix{Float64},
                       obs_times::Vector{Float64},
                       obs_to_state::Vector{Int},
                       obs_var::Float64;
                       interrogate::Symbol=:kramer)
    n_vars = length(u0)
    q = n_deriv
    D = n_vars * q
    n_obs_vars = length(obs_to_state)

    filt_out = probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                interrogate=interrogate)
    μ_filt = filt_out["μ_filt"]; Σ_filt = filt_out["Σ_filt"]
    μ_pred = filt_out["μ_pred"]; Σ_pred = filt_out["Σ_pred"]
    A = filt_out["A"]; times = filt_out["times"]

    n_t_obs = size(obs_data, 1)
    obs_ind = clamp.([searchsortedfirst(times, obs_times[i]) for i in 1:n_t_obs],
                     1, n_steps + 1)

    # Data observation operators (one scalar per observed variable).
    Dmats = [reshape([(c == (obs_to_state[j]-1)*q + 1) ? 1.0 : 0.0 for c in 1:D], 1, D)
             for j in 1:n_obs_vars]
    Vobs = fill(obs_var, 1, 1)

    logdens = 0.0
    bμ = copy(μ_filt[end]); bΣ = copy(Σ_filt[end])
    obs_ptr = n_t_obs

    function condition!(ptr)
        for j in 1:n_obs_vars
            Dj = Dmats[j]
            μf, Σf = kalman_forecast(bμ, bΣ, zeros(1), Dj, Vobs)
            logdens += logpdf_mvn([obs_data[ptr, j]], μf, Σf)
            bμ, bΣ = kalman_update(bμ, bΣ, [obs_data[ptr, j]], zeros(1), Dj, Vobs)
        end
    end

    if obs_ptr >= 1 && obs_ind[obs_ptr] >= n_steps + 1
        condition!(obs_ptr); obs_ptr -= 1
    end

    for n in n_steps:-1:1
        Σpn = Symmetric(Σ_pred[n+1])
        Σpf = cholesky(Σpn + 1e-12 * I, check=false)
        G = issuccess(Σpf) ? (Σ_filt[n] * A') / Σpf : (Σ_filt[n] * A') * pinv(Σ_pred[n+1])
        bμ = μ_filt[n] + G * (bμ - μ_pred[n+1])
        bΣ = Σ_filt[n] + G * (bΣ - Σ_pred[n+1]) * G'
        bΣ = 0.5 * (bΣ + bΣ')
        if obs_ptr >= 1 && obs_ind[obs_ptr] == n
            condition!(obs_ptr); obs_ptr -= 1
        end
    end

    logdens
end
