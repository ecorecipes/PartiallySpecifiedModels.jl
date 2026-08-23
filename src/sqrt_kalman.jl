# ─── Square-Root (QR-based) Kalman Filtering ─────────────────────────
#
# Opt-in numerically stable formulation of the probabilistic-ODE Kalman
# recursions (`sqrt_filter=true` on `RodeoSolver`, `DaltonSolver`,
# `PseudoMarginalSolver`): instead of propagating a covariance P, every
# recursion propagates a RIGHT matrix square root R with P = RᵀR (an
# upper-triangular Cholesky/QR factor). Covariance updates become QR
# re-triangularizations of stacked pre-arrays, so each reconstructed
# covariance is a Gram matrix RᵀR — positive semidefinite BY CONSTRUCTION
# — and the linear algebra only ever handles condition number √cond(P).
# This matters at high IBM order `n_deriv` with small steps, where the
# standard recursion P⁺ = P − KHP loses positive-semidefiniteness.
#
# The formulations are the standard triangularization "array algorithms"
# (Kailath, Sayed & Hassibi 2000, "Linear Estimation", ch. 12), in the
# form used by modern probabilistic ODE solvers (Krämer & Hennig,
# "Stable implementation of probabilistic ODE solvers", JMLR 2024;
# ProbNumDiffEq.jl):
#
#  Predict  P⁺ = A P Aᵀ + Q, with Q = R_Qᵀ R_Q:
#      qr([R Aᵀ; R_Q]) → R⁺
#  Update   with measurement model (H, V = R_Vᵀ R_V), the joint pre-array
#      qr([R_V  0; R_pred Hᵀ  R_pred]) → [S_R  Y; 0  R_filt]
#      yields the innovation-covariance factor S = S_Rᵀ S_R, the gain
#      K = Yᵀ S_R⁻ᵀ, and the filtered factor R_filt in ONE orthogonal
#      transformation (Kailath et al. eq. 12.4.4; Krämer & Hennig §3).
#  Smoother P_s = G P_next Gᵀ + (I − GA) P_f (I − GA)ᵀ + G Q Gᵀ
#      — the Joseph-form rewrite of P_f + G (P_next − P_pred) Gᵀ, exact
#      because G P_pred = P_f Aᵀ —
#      qr([R_next Gᵀ; R_f (I − GA)ᵀ; R_Q Gᵀ]) → R_s
#
# Log-evidence contributions are read off the triangular innovation
# factor: log N(z; ẑ, S) = −½‖S_R⁻ᵀν‖² − Σᵢ log|S_R[i,i]| − m/2·log 2π,
# so log-determinants never require a Cholesky of a formed covariance.
#
# All functions here are Float64-only, matching the standard stack in
# `probsolve.jl` (no ForwardDiff Duals flow through the filter; the
# consuming solvers use Nelder–Mead and finite-difference gradients).

"""
    _sqrt_ibm_noise_factor(dt, q, sigma) -> R_Q

Right factor `R_Q` (upper triangular, D×D with D = n_vars·q) of the
block-diagonal joint IBM process noise, `R_Qᵀ R_Q = Q_joint`.

Computed from the dt-independent unit matrix
`Q̂[i,j] = 1/((2q−1−i−j)(q−1−i)!(q−1−j)!)` via the diagonal scaling
`Q(dt) = T Q̂ T` with `T = diag(dt^{q−1/2−i})` (the Krämer–Hennig
IBM preconditioner identity), so the Cholesky factorization never
touches the ill-conditioned small-`dt` matrix directly.
"""
function _sqrt_ibm_noise_factor(dt::Float64, q::Int, sigma::Vector{Float64})
    n_vars = length(sigma)
    D = n_vars * q
    Qhat = [1.0 / ((2q - 1 - i - j) * factorial(q - 1 - i) * factorial(q - 1 - j))
            for i in 0:q-1, j in 0:q-1]
    Rhat = Matrix(cholesky(Symmetric(Qhat)).U)   # Q̂ = Rhatᵀ Rhat
    T = Diagonal([dt^(q - 0.5 - i) for i in 0:q-1])
    Rblock = Rhat * T                            # right factor of T Q̂ T
    RQ = zeros(D, D)
    for k in 1:n_vars
        idx = ((k-1)*q+1):(k*q)
        RQ[idx, idx] .= sigma[k] .* Rblock
    end
    RQ
end

"""
    _sqrt_predict(R_f, A, R_Q) -> R_pred

Square-root predict step: right factor of `A P Aᵀ + Q` from the filtered
factor `R_f` (P = R_fᵀR_f) and process-noise factor `R_Q`, via QR of the
stacked pre-array `[R_f Aᵀ; R_Q]`.
"""
_sqrt_predict(R_f::AbstractMatrix, A::AbstractMatrix, R_Q::AbstractMatrix) =
    Matrix(qr(vcat(R_f * A', R_Q)).R)

"""
    _sqrt_update(R_pred, H, R_V) -> (S_R, K, R_filt)

Square-root measurement update via one QR of the joint pre-array

    [ R_V        0      ]        [ S_R   Y      ]
    [ R_pred Hᵀ  R_pred ]  →  QR [ 0     R_filt ]

giving the innovation-covariance right factor `S_R` (S = S_RᵀS_R =
H P Hᵀ + V), the Kalman gain `K = Yᵀ S_R⁻ᵀ = P Hᵀ S⁻¹` (triangular
solve), and the filtered right factor `R_filt` with
R_filtᵀR_filt = P − K S Kᵀ. The mean update is `μ + K ν` as usual.
"""
function _sqrt_update(R_pred::AbstractMatrix, H::AbstractMatrix,
                      R_V::AbstractMatrix)
    m = size(H, 1)
    D = size(R_pred, 1)
    pre = [R_V zeros(m, D);
           R_pred * H' R_pred]
    Rpost = Matrix(qr(pre).R)
    S_R = Rpost[1:m, 1:m]
    Y = Rpost[1:m, (m+1):(m+D)]
    R_filt = Rpost[(m+1):(m+D), (m+1):(m+D)]
    K = Matrix((UpperTriangular(S_R) \ Y)')      # K = Yᵀ S_R⁻ᵀ
    S_R, K, R_filt
end

"""
    _sqrt_gauss_logpdf(ν, S_R)

Gaussian log-density `log N(ν; 0, S)` of an innovation `ν` from the
triangular factor `S_R` (S = S_RᵀS_R):
`−½‖S_R⁻ᵀν‖² − Σᵢ log|S_R[i,i]| − m/2·log 2π`.
The log-determinant `log|S| = 2 Σ log|diag S_R|` comes straight from the
triangle (QR sign-indeterminacy handled by the absolute value).
"""
function _sqrt_gauss_logpdf(ν::AbstractVector, S_R::AbstractMatrix)
    m = length(ν)
    z = LowerTriangular(Matrix(S_R')) \ ν        # S_R⁻ᵀ ν
    -0.5 * sum(abs2, z) - sum(x -> log(abs(x)), diag(S_R)) - 0.5 * m * log(2π)
end

"""
    _sqrt_smoother_gain(R_f, A, R_pred) -> G

RTS smoother gain `G = P_f Aᵀ P_pred⁻¹` from right factors only:
`G = R_fᵀ ((R_f Aᵀ) R_pred⁻¹) R_pred⁻ᵀ` via two triangular solves —
no covariance is ever formed or jittered.
"""
function _sqrt_smoother_gain(R_f::AbstractMatrix, A::AbstractMatrix,
                             R_pred::AbstractMatrix)
    U = UpperTriangular(R_pred)
    X = (R_f * A') / U
    R_f' * (X / LowerTriangular(Matrix(R_pred')))
end

"""
    _sqrt_smooth_factor(R_next, G, R_f, A, R_Q) -> R_s

Square-root RTS smoother covariance update. Uses the Joseph-form
identity `P_f + G (P_next − P_pred) Gᵀ =
G P_next Gᵀ + (I − GA) P_f (I − GA)ᵀ + G Q Gᵀ` (exact because
`G P_pred = P_f Aᵀ`), each term PSD, triangularized in one QR of
`[R_next Gᵀ; R_f (I − GA)ᵀ; R_Q Gᵀ]`.
"""
function _sqrt_smooth_factor(R_next::AbstractMatrix, G::AbstractMatrix,
                             R_f::AbstractMatrix, A::AbstractMatrix,
                             R_Q::AbstractMatrix)
    IGA = I - G * A
    Matrix(qr(vcat(R_next * G', R_f * IGA', R_Q * G')).R)
end

"""
    _sqrt_ffbs_factor(G, R_f, A, R_Q) -> R_c

Right factor of the FFBS backward-sampling covariance
`P_f − G P_pred Gᵀ = (I − GA) P_f (I − GA)ᵀ + G Q Gᵀ` (same Joseph-form
identity as the smoother, with the `G P_next Gᵀ` term dropped). PSD by
construction, so backward sampling needs no jitter escalation.
"""
_sqrt_ffbs_factor(G::AbstractMatrix, R_f::AbstractMatrix,
                  A::AbstractMatrix, R_Q::AbstractMatrix) =
    Matrix(qr(vcat(R_f * (I - G * A)', R_Q * G')).R)

# ─── square-root probabilistic ODE filter/smoother ──────────────────

"""
    _sqrt_probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                           interrogate=:kramer, calibrate=true)

Square-root variant of [`probsolve_filter`](@ref) (dispatched to by its
`sqrt_filter=true` keyword). Identical model, interrogation points, and
diffusion calibration; the covariance recursion runs entirely on right
factors. The returned `Dict` carries the same keys (covariances are the
PSD-by-construction reconstructions `RᵀR`) plus the factors themselves
(`"R_pred"`, `"R_filt"`) and the calibrated process-noise factor
(`"R_Q"`) for the square-root smoother/FFBS backward passes.
"""
function _sqrt_probsolve_filter(ode_fun!, p, u0::AbstractVector,
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

    A, _ = _joint_ibm(dt, q, sigma)
    R_Q = _sqrt_ibm_noise_factor(dt, q, sigma)
    E0, E1 = _joint_selectors(n_vars, q)
    X0 = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    R_V = Matrix(sqrt(1e-10) * I, n_vars, n_vars)  # factor of the ODE nugget

    μ_pred = Vector{Vector{Float64}}(undef, n_steps + 1)
    R_pred = Vector{Matrix{Float64}}(undef, n_steps + 1)
    μ_filt = Vector{Vector{Float64}}(undef, n_steps + 1)
    R_filt = Vector{Matrix{Float64}}(undef, n_steps + 1)

    μ_pred[1] = X0
    R_pred[1] = zeros(D, D)
    μ_filt[1] = X0
    R_filt[1] = zeros(D, D)

    calib_acc = 0.0  # Σ_n νᵀ S⁻¹ ν = Σ_n ‖S_R⁻ᵀν‖²

    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps

        μp = A * μ_filt[n]
        Rp = _sqrt_predict(R_filt[n], A, R_Q)

        H, b = _joint_interrogate(ode_fun!, E0, E1, t_n, μp, p, n_vars;
                                  method=interrogate)
        ν = -(H * μp + b)                 # innovation at z = 0
        S_R, K, Rf = _sqrt_update(Rp, H, R_V)
        calib_acc += sum(abs2, LowerTriangular(Matrix(S_R')) \ ν)
        μf = μp + K * ν

        μ_pred[n+1] = μp; R_pred[n+1] = Rp
        μ_filt[n+1] = μf; R_filt[n+1] = Rf
    end

    # Same quasi-MLE global diffusion calibration as the standard path;
    # covariance factors (and the process-noise factor used by the
    # backward passes) scale by √σ̂².
    ssq = calibrate ? max(calib_acc / max(n_steps * n_vars, 1), 1e-12) : 1.0
    if calibrate && ssq != 1.0
        s = sqrt(ssq)
        for n in 1:(n_steps + 1)
            R_pred[n] .*= s
            R_filt[n] .*= s
        end
        R_Q = s .* R_Q
    end

    times = collect(range(t_min, t_max, length=n_steps + 1))
    Dict("μ_pred" => μ_pred, "Σ_pred" => [R' * R for R in R_pred],
         "μ_filt" => μ_filt, "Σ_filt" => [R' * R for R in R_filt],
         "R_pred" => R_pred, "R_filt" => R_filt, "R_Q" => R_Q,
         "times" => times, "A" => A, "ssq" => ssq,
         "n_vars" => n_vars, "q" => q)
end

"""
    _sqrt_probsolve_smooth(filt_out, n_vars)

Square-root RTS smoother on the factor output of
[`_sqrt_probsolve_filter`](@ref) (dispatched to by `probsolve_smooth`
when the factors are present). Same nested per-variable return format.
"""
function _sqrt_probsolve_smooth(filt_out::Dict, n_vars::Int)
    μ_filt = filt_out["μ_filt"]; R_filt = filt_out["R_filt"]
    μ_pred = filt_out["μ_pred"]; R_pred = filt_out["R_pred"]
    A = filt_out["A"]; R_Q = filt_out["R_Q"]; q = filt_out["q"]
    n_steps = length(μ_filt) - 1

    μJ = Vector{Vector{Float64}}(undef, n_steps + 1)
    RJ = Vector{Matrix{Float64}}(undef, n_steps + 1)
    μJ[end] = μ_filt[end]
    RJ[end] = R_filt[end]

    for n in n_steps:-1:1
        G = _sqrt_smoother_gain(R_filt[n], A, R_pred[n+1])
        μJ[n] = μ_filt[n] + G * (μJ[n+1] - μ_pred[n+1])
        RJ[n] = _sqrt_smooth_factor(RJ[n+1], G, R_filt[n], A, R_Q)
    end

    μ_smooth = Vector{Vector{Vector{Float64}}}(undef, n_steps + 1)
    Σ_smooth = Vector{Vector{Matrix{Float64}}}(undef, n_steps + 1)
    for n in 1:(n_steps + 1)
        ΣJ = RJ[n]' * RJ[n]
        μ_smooth[n] = [μJ[n][((k-1)*q+1):(k*q)] for k in 1:n_vars]
        Σ_smooth[n] = [ΣJ[((k-1)*q+1):(k*q), ((k-1)*q+1):(k*q)] for k in 1:n_vars]
    end
    μ_smooth, Σ_smooth
end

"""
    _sqrt_fenrir_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                        obs_data, obs_times, obs_to_state, obs_var;
                        interrogate=:kramer)

Square-root variant of [`fenrir_loglik`](@ref) (dispatched to by its
`sqrt_filter=true` keyword): forward square-root filter, then the
backward Gauss–Markov data-conditioning pass on right factors, with each
data-evidence contribution read off the triangular innovation factor.
"""
function _sqrt_fenrir_loglik(ode_fun!, p, u0::AbstractVector,
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

    filt_out = _sqrt_probsolve_filter(ode_fun!, p, u0, tspan, n_steps,
                                      n_deriv, sigma; interrogate=interrogate)
    μ_filt = filt_out["μ_filt"]; R_filt = filt_out["R_filt"]
    μ_pred = filt_out["μ_pred"]; R_pred = filt_out["R_pred"]
    A = filt_out["A"]; R_Q = filt_out["R_Q"]; times = filt_out["times"]

    n_t_obs = size(obs_data, 1)
    obs_ind = _nearest_grid_indices(times, obs_times)

    Dmats = [reshape([(c == (obs_to_state[j]-1)*q + 1) ? 1.0 : 0.0 for c in 1:D], 1, D)
             for j in 1:n_obs_vars]
    R_Vobs = fill(sqrt(obs_var), 1, 1)

    logdens = 0.0
    bμ = copy(μ_filt[end]); bR = copy(R_filt[end])
    obs_ptr = n_t_obs

    function condition!(ptr)
        for j in 1:n_obs_vars
            Dj = Dmats[j]
            ν = [obs_data[ptr, j]] - Dj * bμ
            S_R, K, bRnew = _sqrt_update(bR, Dj, R_Vobs)
            logdens += _sqrt_gauss_logpdf(ν, S_R)
            bμ = bμ + K * ν
            bR = bRnew
        end
    end

    while obs_ptr >= 1 && obs_ind[obs_ptr] >= n_steps + 1
        condition!(obs_ptr); obs_ptr -= 1
    end

    for n in n_steps:-1:1
        G = _sqrt_smoother_gain(R_filt[n], A, R_pred[n+1])
        bμ = μ_filt[n] + G * (bμ - μ_pred[n+1])
        bR = _sqrt_smooth_factor(bR, G, R_filt[n], A, R_Q)
        while obs_ptr >= 1 && obs_ind[obs_ptr] == n
            condition!(obs_ptr); obs_ptr -= 1
        end
    end

    logdens
end

# ─── square-root DALTON passes ───────────────────────────────────────

"""
    _sqrt_dalton_reference(ode_fun!, p, u0, tspan, n_steps, q, sigma;
                           interrogate=:kramer) -> (logZ, ssq, calib_acc)

Square-root variant of [`_dalton_reference`](@ref) (dispatched to by its
`sqrt_filter=true` keyword). Returns the same `(logZ, ssq, calib_acc)`
triple: the log-evidence comes from the triangular innovation factors
and the calibration statistic from `‖S_R⁻ᵀν‖²`, both mathematically
identical to the standard accumulators.
"""
function _sqrt_dalton_reference(ode_fun!, p, u0::AbstractVector,
                                tspan::Tuple{Float64,Float64},
                                n_steps::Int, q::Int, sigma::Vector{Float64};
                                interrogate::Symbol=:kramer)
    t_min, t_max = tspan
    n_vars = length(u0); D = n_vars * q
    A, _ = _joint_ibm((t_max - t_min) / n_steps, q, sigma)
    R_Q = _sqrt_ibm_noise_factor((t_max - t_min) / n_steps, q, sigma)
    E0, E1 = _joint_selectors(n_vars, q)
    R_V = Matrix(sqrt(1e-10) * I, n_vars, n_vars)

    μf = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    R_f = zeros(D, D)
    logZ = 0.0
    calib_acc = 0.0

    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps
        μp = A * μf
        Rp = _sqrt_predict(R_f, A, R_Q)
        H, b = _joint_interrogate(ode_fun!, E0, E1, t_n, μp, p, n_vars;
                                  method=interrogate)
        zmean = H * μp + b
        S_R, K, R_f = _sqrt_update(Rp, H, R_V)
        logZ += _sqrt_gauss_logpdf(-zmean, S_R)
        calib_acc += sum(abs2, LowerTriangular(Matrix(S_R')) \ zmean)
        μf = μp - K * zmean
    end
    ssq = max(calib_acc / max(n_steps * n_vars, 1), 1e-12)
    logZ, ssq, calib_acc
end

"""
    _sqrt_dalton_joint_evidence(ode_fun!, p, u0, tspan, n_steps, q, sigma,
                                obs_data, obs_times, obs_to_state, obs_var;
                                interrogate=:kramer)

Square-root variant of [`_dalton_joint_evidence`](@ref) (dispatched to
by its `sqrt_filter=true` keyword): the joint data-adaptive pass with
both the ODE pseudo-observations and the data assimilated through
square-root updates.
"""
function _sqrt_dalton_joint_evidence(ode_fun!, p, u0::AbstractVector,
                                     tspan::Tuple{Float64,Float64},
                                     n_steps::Int, q::Int, sigma::Vector{Float64},
                                     obs_data::Matrix{Float64}, obs_times::Vector{Float64},
                                     obs_to_state::Vector{Int}, obs_var::Float64;
                                     interrogate::Symbol=:kramer)
    t_min, t_max = tspan
    n_vars = length(u0); D = n_vars * q
    A, _ = _joint_ibm((t_max - t_min) / n_steps, q, sigma)
    R_Q = _sqrt_ibm_noise_factor((t_max - t_min) / n_steps, q, sigma)
    E0, E1 = _joint_selectors(n_vars, q)
    R_V = Matrix(sqrt(1e-10) * I, n_vars, n_vars)

    μf = _joint_init(ode_fun!, Float64.(u0), t_min, p, q)
    R_f = zeros(D, D)
    times = collect(range(t_min, t_max, length = n_steps + 1))
    n_t_obs = size(obs_data, 1); n_obs_vars = length(obs_to_state)
    obs_ind = _nearest_grid_indices(times, obs_times)
    Dmats = [reshape([(c == (obs_to_state[j]-1)*q + 1) ? 1.0 : 0.0 for c in 1:D], 1, D)
             for j in 1:n_obs_vars]
    R_Vobs = fill(sqrt(obs_var), 1, 1)
    logEv = 0.0

    function assimilate_data!(gi)
        for i in 1:n_t_obs
            obs_ind[i] == gi || continue
            for j in 1:n_obs_vars
                Dj = Dmats[j]
                ν = [obs_data[i, j]] - Dj * μf
                S_R, K, R_new = _sqrt_update(R_f, Dj, R_Vobs)
                logEv += _sqrt_gauss_logpdf(ν, S_R)
                μf = μf + K * ν
                R_f = R_new
            end
        end
    end

    assimilate_data!(1)
    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps
        μp = A * μf
        Rp = _sqrt_predict(R_f, A, R_Q)
        H, b = _joint_interrogate(ode_fun!, E0, E1, t_n, μp, p, n_vars;
                                  method=interrogate)
        zmean = H * μp + b
        S_R, K, R_f = _sqrt_update(Rp, H, R_V)
        logEv += _sqrt_gauss_logpdf(-zmean, S_R)
        μf = μp - K * zmean
        assimilate_data!(n + 1)
    end
    logEv
end

# ─── square-root FFBS backward sampling ──────────────────────────────

"""
    _sqrt_pm_sample_traj(filt_out, rng)

Square-root FFBS backward sampling (dispatched to by
`_pm_sample_traj` when the filter output carries factors): draws
`X = m + R_cᵀ z`, `z ∼ N(0, I)`, where `R_c` is the
PSD-by-construction backward covariance factor from
[`_sqrt_ffbs_factor`](@ref) — no jitter escalation is ever needed.
"""
function _sqrt_pm_sample_traj(filt_out::Dict, rng)
    μ_filt = filt_out["μ_filt"]; R_filt = filt_out["R_filt"]
    μ_pred = filt_out["μ_pred"]; R_pred = filt_out["R_pred"]
    A = filt_out["A"]; R_Q = filt_out["R_Q"]
    N = length(μ_filt) - 1
    D = length(μ_filt[1])
    X = Vector{Vector{Float64}}(undef, N + 1)
    X[N+1] = μ_filt[N+1] .+ R_filt[N+1]' * randn(rng, D)
    for n in N:-1:1
        G = _sqrt_smoother_gain(R_filt[n], A, R_pred[n+1])
        m = μ_filt[n] + G * (X[n+1] - μ_pred[n+1])
        R_c = _sqrt_ffbs_factor(G, R_filt[n], A, R_Q)
        X[n] = m .+ R_c' * randn(rng, D)
    end
    X
end
