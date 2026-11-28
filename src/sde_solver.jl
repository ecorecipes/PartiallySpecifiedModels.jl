# ─── Stochastic Differential Equation (SDE / Process-Noise) PSMs ─────
#
# Continuous-discrete Extended Kalman Filter (CD-EKF; Jazwinski 1970;
# Särkkä & Solin 2019) + Laplace Approximate Marginal Likelihood (LAML)
# smoothing for partially specified stochastic differential equations:
#
#   du = f(u, p, t) dt + G(u, p, t) dW_t       (process dynamics + noise)
#   y_i = H u(t_i) + ε_i,   ε_i ~ N(0, R_obs)  (observation error)
#
# Separates demographic/environmental process noise from measurement error
# while estimating unknown smooth drift functions via penalized splines/GPs.

"""
    SDEPSMProblem(drift!, diffusion, u0, tspan, approximators;
                  data_times, data_values, obs_to_state=...,
                  obs_noise=0.1, P0=nothing, known_params=NamedTuple(),
                  estimate_noise=true)

A partially specified stochastic differential equation (SDE) problem:

    du = f(u, p, t) dt + G(u, p, t) dW_t
    y_i = u[obs_to_state](t_i) + ε_i,   ε_i ~ N(0, diag(obs_noise.^2))

# Arguments
- `drift!`: in-place drift `f!(du, u, p, t)` (identical signature to ODE PSMs)
- `diffusion`: process-noise diffusion specification:
  - `Real` scalar `σ` → isotropic `Q = σ² I`
  - `AbstractVector` `σ` (length `n_vars`) → diagonal `Q = Diagonal(σ.^2)`
  - `AbstractMatrix` `G` (`n_vars × m`) → constant `Q = G * G'`
  - `Function` `g(u, p, t)` → state-dependent vector (diagonal SDs) or matrix `G`
- `u0`: initial state mean vector or function `u0(p)`
- `tspan`: `(t0, tf)`
- `approximators`: vector of `AbstractApproximator` for unknown drift functions
- `data_times`, `data_values`: observations (`NaN` entries are skipped cleanly in the filter)
- `obs_to_state`: maps observation columns to state indices
- `obs_noise`: scalar or vector of per-observed-state measurement noise SDs
  (initial guess when `estimate_noise=true`, fixed when `estimate_noise=false`)
- `P0`: initial state covariance matrix (default `1e-6 * I`)
- `estimate_noise`: if `true` (default), jointly estimate the process-noise
  scale multiplier and observation-noise SDs alongside the drift approximator
  parameters via EKF marginal likelihood.
"""
struct SDEPSMProblem{D, G, U}
    drift!::D
    diffusion::G
    u0::U
    P0::Matrix{Float64}
    tspan::Tuple{Float64, Float64}
    approximators::Vector{<:AbstractApproximator}
    data_times::Vector{Float64}
    data_values::Matrix{Float64}
    obs_to_state::Vector{Int}
    obs_noise::Vector{Float64}
    known_params::NamedTuple
    estimate_noise::Bool
end

function SDEPSMProblem(drift!, diffusion, u0, tspan,
                       approximators::Vector{<:AbstractApproximator};
                       data_times::AbstractVector,
                       data_values::AbstractMatrix,
                       obs_to_state::Vector{Int}=collect(1:size(data_values, 2)),
                       obs_noise::Union{Real, AbstractVector{<:Real}}=0.1,
                       P0::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
                       known_params::NamedTuple=NamedTuple(),
                       estimate_noise::Bool=true)
    n_times = length(data_times)
    n_obs = size(data_values, 2)
    size(data_values, 1) == n_times || throw(ArgumentError(
        "SDEPSMProblem: data_values rows ($(size(data_values, 1))) != data_times ($n_times)"))
    length(obs_to_state) == n_obs || throw(ArgumentError(
        "SDEPSMProblem: obs_to_state length ($(length(obs_to_state))) != data_values cols ($n_obs)"))
    issorted(data_times) || throw(ArgumentError("SDEPSMProblem: data_times must be sorted"))
    isempty(approximators) && throw(ArgumentError("SDEPSMProblem: approximators cannot be empty"))

    u0_len = u0 isa Function ? length(u0(build_param_struct_from_approxs(approximators, known_params))) : length(u0)
    P0_mat = P0 === nothing ? Matrix{Float64}(1e-6 * I, u0_len, u0_len) : Matrix{Float64}(P0)
    size(P0_mat) == (u0_len, u0_len) || throw(ArgumentError("SDEPSMProblem: P0 must be $(u0_len)×$(u0_len)"))

    obs_sd = obs_noise isa Real ? fill(Float64(obs_noise), n_obs) : Float64.(collect(obs_noise))
    length(obs_sd) == n_obs || throw(ArgumentError("SDEPSMProblem: obs_noise length must equal n_obs ($n_obs)"))
    all(>(0.0), obs_sd) || throw(ArgumentError("SDEPSMProblem: obs_noise must be positive"))

    SDEPSMProblem(drift!, diffusion, u0, P0_mat,
                  (Float64(tspan[1]), Float64(tspan[2])),
                  approximators, Float64.(collect(data_times)),
                  Float64.(Matrix(data_values)), obs_to_state,
                  obs_sd, known_params, estimate_noise)
end

function build_param_struct_from_approxs(approximators, known_params,
                                         beta::Union{Nothing, AbstractVector}=nothing)
    b = beta === nothing ? vcat([initial_params(a) for a in approximators]...) : beta
    offset = 0
    uf_entries = Pair{Symbol, Any}[]
    for approx in approximators
        np = nparams(approx)
        pk = @view b[offset+1:offset+np]
        offset += np
        push!(uf_entries, approx.name => build_evaluator(approx, pk))
    end
    merge(NamedTuple(uf_entries), known_params)
end

"""
    EKFSolver(; maxiters=150, n_substeps=8, lambda_smooth=nothing,
                estimate_laml=true, n_laml_outer=6, tol=1e-6, verbose=false)

Continuous-discrete Extended Kalman Filter (CD-EKF) solver for
[`SDEPSMProblem`](@ref), maximizing the penalized EKF marginal log-likelihood
with optional outer Laplace / Fellner–Schall smoothing-parameter estimation.
"""
struct EKFSolver
    maxiters::Int
    n_substeps::Int
    lambda_smooth::Union{Nothing, Float64, Vector{Float64}}
    estimate_laml::Bool
    n_laml_outer::Int
    tol::Float64
    verbose::Bool
end

function EKFSolver(; maxiters::Int=150, n_substeps::Int=8,
                   lambda_smooth::Union{Nothing, Real, AbstractVector{<:Real}}=nothing,
                   estimate_laml::Bool=true, n_laml_outer::Int=6,
                   tol::Float64=1e-6, verbose::Bool=false)
    ls = if lambda_smooth === nothing
        nothing
    elseif lambda_smooth isa Real
        Float64(lambda_smooth)
    else
        Float64.(collect(lambda_smooth))
    end
    EKFSolver(maxiters, max(n_substeps, 1), ls, estimate_laml,
              max(n_laml_outer, 1), Float64(tol), verbose)
end

function _sde_diffusion_Q(diff_spec, u::AbstractVector{T}, p, t::Float64,
                          proc_scale::T, n_vars::Int) where {T}
    Q = zeros(T, n_vars, n_vars)
    ps2 = proc_scale * proc_scale
    if diff_spec isa Real
        v = ps2 * T(diff_spec)^2
        @inbounds for i in 1:n_vars; Q[i, i] = v; end
    elseif diff_spec isa AbstractVector
        @inbounds for i in 1:n_vars; Q[i, i] = ps2 * T(diff_spec[i])^2; end
    elseif diff_spec isa AbstractMatrix
        G = T.(diff_spec)
        Q .= ps2 .* (G * G')
    elseif diff_spec isa Function
        val = diff_spec(u, p, t)
        if val isa Real
            v = ps2 * val^2
            @inbounds for i in 1:n_vars; Q[i, i] = v; end
        elseif val isa AbstractVector
            @inbounds for i in 1:n_vars; Q[i, i] = ps2 * val[i]^2; end
        else
            Q .= ps2 .* (val * val')
        end
    end
    Q
end

function _sde_drift_jac_fd(drift!, u::AbstractVector{T}, p, t::Float64,
                           du_p::Vector{T}, du_m::Vector{T}, u_tmp::Vector{T}) where {T}
    n = length(u)
    Fx = Matrix{T}(undef, n, n)
    h = 1e-6
    @inbounds for j in 1:n
        u_tmp .= u
        u_tmp[j] = u[j] + h
        drift!(du_p, u_tmp, p, t)
        u_tmp[j] = u[j] - h
        drift!(du_m, u_tmp, p, t)
        inv2h = 0.5 / h
        for i in 1:n
            Fx[i, j] = (du_p[i] - du_m[i]) * inv2h
        end
    end
    Fx
end

function _cd_ekf_pass(prob::SDEPSMProblem, beta::AbstractVector{T},
                      proc_scale::T, obs_sd::AbstractVector{T};
                      n_substeps::Int=8, store_states::Bool=false) where {T}
    p = build_param_struct_from_approxs(prob.approximators, prob.known_params, beta)
    u0_raw = prob.u0 isa Function ? prob.u0(p) : prob.u0
    n_vars = length(u0_raw)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)

    m = Vector{T}(u0_raw)
    P = Matrix{T}(prob.P0)
    t_cur = prob.tspan[1]

    k1 = Vector{T}(undef, n_vars); k2 = similar(k1); k3 = similar(k1); k4 = similar(k1)
    u_tmp = similar(k1); du_p = similar(k1); du_m = similar(k1)

    m_preds = store_states ? Vector{Vector{Float64}}(undef, n_times) : nothing
    P_preds = store_states ? Vector{Matrix{Float64}}(undef, n_times) : nothing
    m_filts = store_states ? Vector{Vector{Float64}}(undef, n_times) : nothing
    P_filts = store_states ? Vector{Matrix{Float64}}(undef, n_times) : nothing
    Phi_list = store_states ? Vector{Matrix{Float64}}(undef, n_times) : nothing

    ll = zero(T)

    for i in 1:n_times
        t_target = prob.data_times[i]
        dt_total = t_target - t_cur
        Phi_i = store_states ? Matrix{Float64}(I, n_vars, n_vars) : nothing

        if dt_total > 0.0
            h_step = dt_total / n_substeps
            for s in 1:n_substeps
                ts = t_cur + (s - 1) * h_step
                # RK4 mean propagation
                prob.drift!(k1, m, p, ts)
                @. u_tmp = m + 0.5 * h_step * k1
                prob.drift!(k2, u_tmp, p, ts + 0.5 * h_step)
                @. u_tmp = m + 0.5 * h_step * k2
                prob.drift!(k3, u_tmp, p, ts + 0.5 * h_step)
                @. u_tmp = m + h_step * k3
                prob.drift!(k4, u_tmp, p, ts + h_step)

                # Linearized transition Φ_s ≈ I + h F_x + ½ h² F_x² (guarantees PSD covariance!)
                Fx = _sde_drift_jac_fd(prob.drift!, m, p, ts, du_p, du_m, u_tmp)
                Q = _sde_diffusion_Q(prob.diffusion, m, p, ts, proc_scale, n_vars)
                F_step = Matrix{T}(I, n_vars, n_vars) .+ h_step .* Fx .+ (0.5 * h_step^2) .* (Fx * Fx)
                P = F_step * P * F_step' .+ h_step .* Q
                P = T(0.5) .* (P .+ P')
                if store_states
                    Phi_i = Float64.(ForwardDiff.value.(F_step)) * Phi_i
                end

                @. m = m + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            end
            t_cur = t_target
        end

        if store_states
            m_preds[i] = Float64.(ForwardDiff.value.(m))
            P_preds[i] = Float64.(ForwardDiff.value.(P))
            Phi_list[i] = Phi_i
        end

        # Measurement update on finite (non-NaN) observed columns
        vis = Int[j for j in 1:n_obs if isfinite(prob.data_values[i, j])]
        nv = length(vis)
        if nv > 0
            # Build observation selector H (nv × n_vars) and R (nv × nv)
            ν = Vector{T}(undef, nv)
            S = Matrix{T}(undef, nv, nv)
            PHt = Matrix{T}(undef, n_vars, nv)
            @inbounds for a in 1:nv
                ja = vis[a]
                sa = prob.obs_to_state[ja]
                ν[a] = T(prob.data_values[i, ja]) - m[sa]
                for r in 1:n_vars
                    PHt[r, a] = P[r, sa]
                end
                for b in 1:nv
                    sb = prob.obs_to_state[vis[b]]
                    S[a, b] = P[sa, sb]
                end
                S[a, a] += obs_sd[ja]^2 + T(1e-12)
            end
            S = T(0.5) .* (S .+ S')

            # Dual-safe Cholesky / logdet & solve (for small nv, direct explicit Cholesky)
            L_S = _dual_safe_chol(S)
            if L_S === nothing
                return T(-1e12), nothing
            end
            logdet_S = zero(T)
            @inbounds for a in 1:nv
                logdet_S += T(2.0) * log(L_S[a, a])
            end
            z_innov = LowerTriangular(L_S) \ ν
            quad = dot(z_innov, z_innov)
            ll -= T(0.5) * (logdet_S + quad + T(nv * log(2π)))

            # Kalman gain K = P H' S⁻¹ and Joseph-form covariance update
            K = (UpperTriangular(L_S') \ (LowerTriangular(L_S) \ PHt'))'
            m = m .+ K * ν
            # I - K H
            IKH = Matrix{T}(I, n_vars, n_vars)
            @inbounds for a in 1:nv
                sa = prob.obs_to_state[vis[a]]
                for r in 1:n_vars
                    IKH[r, sa] -= K[r, a]
                end
            end
            KRKt = zeros(T, n_vars, n_vars)
            @inbounds for a in 1:nv
                r_var = obs_sd[vis[a]]^2
                for r1 in 1:n_vars, r2 in 1:n_vars
                    KRKt[r1, r2] += r_var * K[r1, a] * K[r2, a]
                end
            end
            P = IKH * P * IKH' .+ KRKt
            P = T(0.5) .* (P .+ P')
        end

        if store_states
            m_filts[i] = Float64.(ForwardDiff.value.(m))
            P_filts[i] = Float64.(ForwardDiff.value.(P))
        end
    end

    if !store_states
        return ll, nothing
    end

    # Backward Rauch-Tung-Striebel (RTS) smoother
    m_smooth = Vector{Vector{Float64}}(undef, n_times)
    P_smooth = Vector{Matrix{Float64}}(undef, n_times)
    m_smooth[n_times] = copy(m_filts[n_times])
    P_smooth[n_times] = copy(P_filts[n_times])
    for i in (n_times - 1):-1:1
        Pp = Symmetric(P_preds[i + 1] + 1e-10 * I)
        F_pp = cholesky(Pp; check=false)
        Gain = issuccess(F_pp) ? (P_filts[i] * Phi_list[i + 1]') / F_pp :
                                 (P_filts[i] * Phi_list[i + 1]') * pinv(Matrix(Pp))
        m_smooth[i] = m_filts[i] + Gain * (m_smooth[i + 1] - m_preds[i + 1])
        Ps = P_filts[i] + Gain * (P_smooth[i + 1] - P_preds[i + 1]) * Gain'
        P_smooth[i] = 0.5 .* (Ps .+ Ps')
    end

    filt_mat = Matrix{Float64}(undef, n_times, n_vars)
    smooth_mat = Matrix{Float64}(undef, n_times, n_vars)
    smooth_sd = Matrix{Float64}(undef, n_times, n_vars)
    for i in 1:n_times, d in 1:n_vars
        filt_mat[i, d] = m_filts[i][d]
        smooth_mat[i, d] = m_smooth[i][d]
        smooth_sd[i, d] = sqrt(max(P_smooth[i][d, d], 0.0))
    end
    return ll, (filtered=filt_mat, smoothed=smooth_mat, smoothed_std=smooth_sd)
end

# Tiny Dual-safe lower-triangular Cholesky factorization (handles small n_obs cleanly)
function _dual_safe_chol(A::Matrix{T}) where {T}
    n = size(A, 1)
    L = zeros(T, n, n)
    @inbounds for j in 1:n
        s = A[j, j]
        for k in 1:(j - 1)
            s -= L[j, k] * L[j, k]
        end
        ForwardDiff.value(s) > 0.0 || return nothing
        L[j, j] = sqrt(s)
        inv_ljj = one(T) / L[j, j]
        for i in (j + 1):n
            s2 = A[i, j]
            for k in 1:(j - 1)
                s2 -= L[i, k] * L[j, k]
            end
            L[i, j] = s2 * inv_ljj
        end
    end
    L
end

"""
    ekf_loglik(prob::SDEPSMProblem, beta; process_scale=1.0,
               obs_noise=prob.obs_noise, n_substeps=8) -> Float64

Evaluate the continuous-discrete Extended Kalman Filter marginal log-likelihood
`log p(Y | β, σ_proc, σ_obs)` for an [`SDEPSMProblem`](@ref).
"""
function ekf_loglik(prob::SDEPSMProblem, beta::AbstractVector;
                    process_scale::Real=1.0,
                    obs_noise::AbstractVector=prob.obs_noise,
                    n_substeps::Int=8)
    T = promote_type(eltype(beta), typeof(process_scale), eltype(obs_noise))
    ll, _ = _cd_ekf_pass(prob, T.(beta), T(process_scale), T.(obs_noise);
                         n_substeps=n_substeps, store_states=false)
    ll
end

"""
    simulate_sde(prob::SDEPSMProblem, beta; rng=Random.default_rng(),
                 dt=0.01, process_scale=1.0, obs_noise=prob.obs_noise)
        -> (states, observations)

Simulate a sample path from an [`SDEPSMProblem`](@ref) via Euler–Maruyama,
returning `(states_at_data_times, noisy_observations)`.
"""
function simulate_sde(prob::SDEPSMProblem, beta::AbstractVector;
                      rng::AbstractRNG=Random.default_rng(),
                      dt::Float64=0.01,
                      process_scale::Float64=1.0,
                      obs_noise::Union{Nothing, AbstractVector{<:Real}}=prob.obs_noise)
    p = build_param_struct_from_approxs(prob.approximators, prob.known_params, beta)
    u = Float64.(collect(prob.u0 isa Function ? prob.u0(p) : prob.u0))
    n_vars = length(u)
    n_times = length(prob.data_times)
    n_obs = length(prob.obs_to_state)
    du = zeros(n_vars)
    states = Matrix{Float64}(undef, n_times, n_vars)
    t_cur = prob.tspan[1]

    for i in 1:n_times
        t_target = prob.data_times[i]
        dt_tot = t_target - t_cur
        if dt_tot > 0.0
            n_st = max(1, ceil(Int, dt_tot / dt))
            h = dt_tot / n_st
            sqh = sqrt(h)
            for s in 1:n_st
                ts = t_cur + (s - 1) * h
                prob.drift!(du, u, p, ts)
                Q = _sde_diffusion_Q(prob.diffusion, u, p, ts, process_scale, n_vars)
                F = cholesky(Symmetric(Q + 1e-14 * I); check=false)
                noise = issuccess(F) ? (F.L * randn(rng, n_vars)) :
                                       (sqrt.(max.(diag(Q), 0.0)) .* randn(rng, n_vars))
                u .= u .+ h .* du .+ sqh .* noise
            end
            t_cur = t_target
        end
        states[i, :] .= u
    end

    obs = Matrix{Float64}(undef, n_times, n_obs)
    for i in 1:n_times, j in 1:n_obs
        sd = obs_noise === nothing ? 0.0 : Float64(obs_noise[j])
        obs[i, j] = states[i, prob.obs_to_state[j]] + sd * randn(rng)
    end
    states, obs
end

"""
    solve(prob::SDEPSMProblem, alg::EKFSolver) -> PSMSolution

Fit a partially specified SDE via the continuous-discrete Extended Kalman Filter
(CD-EKF), separating process noise from observation error while estimating
unknown smooth drift functions with automatic LAML / Fellner–Schall smoothing.
"""
function SciMLBase.solve(prob::SDEPSMProblem, alg::EKFSolver)
    n_obs = size(prob.data_values, 2)
    beta = vcat([initial_params(a) for a in prob.approximators]...)
    n_beta = length(beta)

    # Penalty blocks on approximators
    S_list = Matrix{Float64}[]
    uf_offsets = Int[]
    uf_nk = Int[]
    let off = 0
        for approx in prob.approximators
            np = nparams(approx)
            for (Sb, rng_b) in penalty_blocks(approx)
                push!(S_list, Sb)
                push!(uf_offsets, off + first(rng_b) - 1)
                push!(uf_nk, length(rng_b))
            end
            off += np
        end
    end
    m = length(S_list)
    S_ranks = [count(>(1e-10), eigvals(Symmetric(S))) for S in S_list]

    theta = if alg.lambda_smooth === nothing
        Float64[tr(S) > 0 ? 1.0 / tr(S) : 1.0 for S in S_list]
    elseif alg.lambda_smooth isa Float64
        fill(alg.lambda_smooth, m)
    else
        length(alg.lambda_smooth) == m || throw(ArgumentError(
            "EKFSolver: lambda_smooth length must match penalty blocks ($m)"))
        copy(alg.lambda_smooth)
    end

    log_proc = 0.0
    log_obs = log.(prob.obs_noise)
    n_outer = (alg.estimate_laml && alg.lambda_smooth === nothing && m > 0) ? alg.n_laml_outer : 1

    conv_converged = false
    conv_reason = :maxiters
    total_iters = 0

    for outer in 1:n_outer
        B_mat = build_S_lambda(S_list, uf_offsets, uf_nk, log.(max.(theta, 1e-300)), n_beta)

        function _neg_pen_ll(x::AbstractVector{T}) where {T}
            b_v = @view x[1:n_beta]
            ps_v = prob.estimate_noise ? exp(clamp(x[n_beta + 1], T(-8.0), T(6.0))) : one(T)
            os_v = prob.estimate_noise ?
                   T[exp(clamp(x[n_beta + 1 + j], T(-8.0), T(6.0))) for j in 1:n_obs] :
                   T.(prob.obs_noise)
            ll_val, _ = _cd_ekf_pass(prob, Vector{T}(b_v), ps_v, os_v;
                                     n_substeps=alg.n_substeps, store_states=false)
            isfinite(ForwardDiff.value(ll_val)) || return T(1e12)
            pen = T(0.5) * dot(b_v, B_mat * b_v)
            -ll_val + pen
        end

        x0 = prob.estimate_noise ? vcat(beta, [log_proc], log_obs) : copy(beta)
        g_cfg = ForwardDiff.GradientConfig(_neg_pen_ll, x0)
        _neg_pen_ll_grad! = (g, x) -> ForwardDiff.gradient!(g, _neg_pen_ll, x, g_cfg)
        res = Optim.optimize(_neg_pen_ll, _neg_pen_ll_grad!, x0, Optim.LBFGS(),
                             Optim.Options(iterations=alg.maxiters,
                                           g_tol=alg.tol,
                                           show_trace=false))
        x_opt = Optim.minimizer(res)
        total_iters += Optim.iterations(res)
        conv_converged = Optim.converged(res)
        conv_reason = conv_converged ? :converged_tol : :maxiters

        beta .= x_opt[1:n_beta]
        if prob.estimate_noise
            log_proc = clamp(x_opt[n_beta + 1], -8.0, 6.0)
            log_obs .= clamp.(x_opt[(n_beta + 2):end], -8.0, 6.0)
        end

        if alg.verbose
            @printf("EKFSolver outer %d: neg_pen_ll=%.4f proc_scale=%.4g obs_sd=%s λ=%s\n",
                    outer, Optim.minimum(res), exp(log_proc),
                    string(round.(exp.(log_obs); sigdigits=3)),
                    string(round.(theta; sigdigits=3)))
        end

        # Outer Fellner–Schall update on λ using the Laplace Hessian of -ℓ_EKF(β)
        if outer < n_outer && m > 0
            ps_fix = prob.estimate_noise ? exp(log_proc) : 1.0
            os_fix = prob.estimate_noise ? exp.(log_obs) : prob.obs_noise
            H_data = try
                H_raw = ForwardDiff.hessian(
                    b -> -ekf_loglik(prob, b; process_scale=ps_fix, obs_noise=os_fix,
                                     n_substeps=alg.n_substeps),
                    beta)
                0.5 .* (H_raw .+ H_raw')
            catch
                break
            end
            # Project negative eigenvalues of H_data to ensure PSD Laplace curvature
            eig_hd = eigen(Symmetric(H_data))
            H_data_psd = eig_hd.vectors * Diagonal(max.(eig_hd.values, 0.0)) * eig_hd.vectors'
            H_tot = H_data_psd .+ B_mat
            Hi = _safe_inv(H_tot)
            max_rel_d = 0.0
            for l in 1:m
                rk = (uf_offsets[l] + 1):(uf_offsets[l] + uf_nk[l])
                bk = @view beta[rk]
                bSb = dot(bk, S_list[l] * bk)
                bSb <= 1e-30 && continue
                tr_term = theta[l] * tr(@view(Hi[rk, rk]) * S_list[l])
                num = max(Float64(S_ranks[l]) - tr_term, 1e-8)
                lam_new = clamp(num / bSb, exp(RHO_MIN), exp(RHO_MAX))
                max_rel_d = max(max_rel_d, abs(log(lam_new) - log(theta[l])))
                theta[l] = lam_new
            end
            max_rel_d < 1e-2 && break
        end
    end

    # Final EKF + RTS pass, Laplace covariance V_beta, and EDF
    proc_scale_hat = prob.estimate_noise ? exp(log_proc) : 1.0
    obs_sd_hat = prob.estimate_noise ? exp.(log_obs) : copy(prob.obs_noise)
    ll_final, state_info = _cd_ekf_pass(prob, beta, proc_scale_hat, obs_sd_hat;
                                        n_substeps=alg.n_substeps, store_states=true)

    B_final = build_S_lambda(S_list, uf_offsets, uf_nk, log.(max.(theta, 1e-300)), n_beta)
    H_data_final = try
        H_raw = ForwardDiff.hessian(
            b -> -ekf_loglik(prob, b; process_scale=proc_scale_hat, obs_noise=obs_sd_hat,
                             n_substeps=alg.n_substeps),
            beta)
        eig_hd = eigen(Symmetric(0.5 .* (H_raw .+ H_raw')))
        eig_hd.vectors * Diagonal(max.(eig_hd.values, 0.0)) * eig_hd.vectors'
    catch
        zeros(n_beta, n_beta)
    end
    H_final = H_data_final .+ B_final
    V_beta = try
        _safe_inv(H_final)
    catch
        nothing
    end
    edf = V_beta !== nothing ? clamp(tr(V_beta * H_data_final), 0.0, Float64(n_beta)) : Float64(n_beta)

    # Fitted observations from RTS smoothed states
    n_times = length(prob.data_times)
    pred = Matrix{Float64}(undef, n_times, n_obs)
    for i in 1:n_times, j in 1:n_obs
        pred[i, j] = state_info.smoothed[i, prob.obs_to_state[j]]
    end
    data_loss = 0.0
    for k in eachindex(prob.data_values)
        isfinite(prob.data_values[k]) || continue
        data_loss += (prob.data_values[k] - pred[k])^2
    end
    obj_val = -ll_final + 0.5 * dot(beta, B_final * beta)

    syms = Symbol[a.name for a in prob.approximators]
    vals = Vector{Float64}[]
    uf_evals = Dict{Symbol, Any}()
    off = 0
    for approx in prob.approximators
        np = nparams(approx)
        pk = Float64.(beta[off+1:off+np])
        push!(vals, pk)
        uf_evals[approx.name] = build_evaluator(approx, pk)
        off += np
    end
    params = ComponentArray(NamedTuple{Tuple(syms)}(Tuple(vals)))

    conv_info = (converged=conv_converged, iterations=total_iters,
                 reason=conv_reason, method=:sde_ekf,
                 loglik=ll_final, process_scale=proc_scale_hat,
                 obs_noise=obs_sd_hat,
                 filtered_states=state_info.filtered,
                 smoothed_states=state_info.smoothed,
                 smoothed_std=state_info.smoothed_std,
                 V_beta=V_beta, sigma2=1.0)

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                pred, copy(prob.data_values), copy(prob.data_times),
                uf_evals, conv_info)
end
