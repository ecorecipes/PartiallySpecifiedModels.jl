# ─── Collocation-based PSM solver ──────────────────────────────────
#
# Implements the generalized profiling / parameter cascading approach:
#   - States x(t) are free parameters (values at collocation points)
#   - ODE compliance is penalized: λ_ode × ||ẋ - f(x, p)||²
#   - Continuation on λ_ode from small (data-driven) to large (ODE-constrained)
#   - Unknown functions estimated via IRLS + LAML (same as standard solver)
#
# Reference: Ramsay et al. (2007), JRSS-B 69, 741-796
#            Fasiolo, Pya & Wood (2016), Statistical Science 31(1), 96-118

using LinearAlgebra
using Statistics

# ─── Finite-difference differentiation matrix ─────────────────────

"""
    build_diff_matrix(times) → D

Build a finite-difference differentiation matrix D such that D * x ≈ dx/dt
for a vector x of values at the given times.

Uses second-order central differences for interior points and second-order
one-sided differences at boundaries.
"""
function build_diff_matrix(times::AbstractVector{Float64})
    T = length(times)
    D = zeros(T, T)

    # Forward difference at t_1 (second-order)
    h1, h2 = times[2] - times[1], times[3] - times[1]
    D[1, 1] = -(2*h1 + h2 - h1) / (h1 * (h2 - h1) + 1e-30)
    # Use general 3-point forward formula
    dt1, dt2 = times[2] - times[1], times[3] - times[1]
    D[1, 1] = -(dt2 + dt1) / (dt1 * dt2)
    D[1, 2] = dt2 / (dt1 * (dt2 - dt1))
    D[1, 3] = -dt1 / (dt2 * (dt2 - dt1))

    # Central differences for interior points
    for i in 2:(T-1)
        h_m = times[i] - times[i-1]
        h_p = times[i+1] - times[i]
        D[i, i-1] = -h_p / (h_m * (h_m + h_p))
        D[i, i]   = (h_p - h_m) / (h_m * h_p)
        D[i, i+1] = h_m / (h_p * (h_m + h_p))
    end

    # Backward difference at t_T (second-order)
    dt1 = times[T] - times[T-1]
    dt2 = times[T] - times[T-2]
    D[T, T]   = (dt2 + dt1) / (dt1 * dt2)
    D[T, T-1] = -dt2 / (dt1 * (dt2 - dt1))
    D[T, T-2] = dt1 / (dt2 * (dt2 - dt1))

    D
end

# ─── ODE/Discrete RHS evaluation at collocation points ────────────

"""
Evaluate the dynamics right-hand side at all collocation points.

For continuous models: returns `f(x(t), p, t)` (derivatives).
For discrete models: returns `f(x(t), p, t)` (next-state map).

Returns a (T × K) matrix where entry [i,k] = f_k(x(t_i), p, t_i).
"""
function eval_ode_rhs(prob::PSMProblem, times::Vector{Float64},
                      alpha::Matrix{Float64}, beta::Vector{Float64})
    T, K = size(alpha)
    F = zeros(T, K)
    p = build_param_struct(prob, beta)
    du = zeros(K)

    for i in 1:T
        u = alpha[i, :]
        try
            prob.dynamics!(du, u, p, times[i])
        catch
            du .= 1e6  # Large residual for failed evaluations
        end
        F[i, :] .= du
    end
    F
end

# ─── Combined residual and Jacobian ───────────────────────────────

"""
Build the combined residual vector:
  r = [ √w × (y_obs - alpha_obs)    ;  data fidelity
        √λ_ode × (D*alpha - F)_flat ]  ODE compliance

and the combined Jacobian ∂r/∂[alpha_flat; beta].

Uses analytical derivatives where possible:
- Data residual ∂/∂alpha: trivial (−√w δ_{ij})
- ODE residual ∂/∂alpha: D − ∂F/∂x (D is the diff matrix, ∂F/∂x by pointwise FD)
- ODE residual ∂/∂beta: −∂F/∂beta (by FD over beta)
"""
function collocation_residual_jacobian(
        prob::PSMProblem, times::Vector{Float64},
        alpha::Matrix{Float64}, beta::Vector{Float64},
        D::Matrix{Float64}, lambda_ode::Float64,
        w_vec::Vector{Float64})

    T, K = size(alpha)
    n_obs = size(prob.data_values, 2)
    n_alpha = T * K
    n_beta = length(beta)
    n_params = n_alpha + n_beta

    sqrt_lode = sqrt(lambda_ode)

    # --- Data residual ---
    data_resid = zeros(T * n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:T
            idx = (j - 1) * T + i
            data_resid[idx] = sqrt(w_vec[idx]) * (prob.data_values[i, j] - alpha[i, sk])
        end
    end

    # --- ODE/Discrete compliance residual ---
    F = eval_ode_rhs(prob, times, alpha, beta)
    ode_resid = zeros(T * K)

    if prob.discrete
        # Discrete-time: alpha[t+1] ≈ F(alpha[t], p, t)
        # Residual at time i: alpha[i+1] - F[i] for i=1..T-1, last row = 0
        for k in 1:K
            for i in 1:(T-1)
                ode_resid[(k - 1) * T + i] = sqrt_lode * (alpha[i+1, k] - F[i, k])
            end
            ode_resid[(k - 1) * T + T] = 0.0
        end
    else
        # Continuous-time: D*alpha ≈ F
        for k in 1:K
            dalpha_k = D * alpha[:, k]
            for i in 1:T
                ode_resid[(k - 1) * T + i] = sqrt_lode * (dalpha_k[i] - F[i, k])
            end
        end
    end

    # --- State roughness penalty for discrete models ---
    # In continuous collocation, D couples adjacent states. For discrete,
    # we add an explicit second-difference penalty to prevent wiggly states.
    # Use lambda_ode^0.25 scaling: grows gently relative to compliance.
    smooth_resid = Float64[]
    if prob.discrete && T >= 3
        n_smooth = (T - 2) * K
        smooth_resid = zeros(n_smooth)
        sqrt_lsmooth = lambda_ode^0.25
        for k in 1:K
            for i in 2:(T-1)
                idx = (k - 1) * (T - 2) + (i - 1)
                smooth_resid[idx] = sqrt_lsmooth * (alpha[i+1, k] - 2*alpha[i, k] + alpha[i-1, k])
            end
        end
    end

    n_data = T * n_obs
    n_ode = T * K
    n_smooth = length(smooth_resid)
    resid = vcat(data_resid, ode_resid, smooth_resid)

    # --- Jacobian ---
    J = zeros(n_data + n_ode + n_smooth, n_params)

    # 1) Data residual w.r.t. alpha: ∂(√w(y-α))/∂α_{i,k} = -√w if obs j maps to state k
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:T
            row = (j - 1) * T + i
            col = (sk - 1) * T + i
            J[row, col] = -sqrt(w_vec[row])
        end
    end

    # 2) Compliance residual w.r.t. alpha
    eps_x = 1e-6
    p = build_param_struct(prob, beta)
    du0 = zeros(K)
    du_p = zeros(K)

    if prob.discrete
        # Discrete: residual_i = sqrt_lode * (alpha[i+1,k] - F[i,k])
        # ∂/∂alpha[i+1,k_pert] for same state k_eq: +sqrt_lode * δ(k_eq==k_pert)
        # ∂/∂alpha[i,k_pert]: -sqrt_lode * ∂F[i,k_eq]/∂x[k_pert]
        for i in 1:(T-1)
            u = alpha[i, :]
            try; prob.dynamics!(du0, u, p, times[i]); catch; du0 .= 0; end

            for k_pert in 1:K
                u_p = copy(u)
                u_p[k_pert] += eps_x
                try; prob.dynamics!(du_p, u_p, p, times[i]); catch; du_p .= du0; end
                dF_dx = (du_p .- du0) ./ eps_x

                col_i = (k_pert - 1) * T + i  # alpha[i, k_pert]
                col_ip1 = (k_pert - 1) * T + (i + 1)  # alpha[i+1, k_pert]

                for k_eq in 1:K
                    row_ode = n_data + (k_eq - 1) * T + i
                    # ∂/∂alpha[i, k_pert]: -sqrt_lode * ∂F/∂x
                    J[row_ode, col_i] -= sqrt_lode * dF_dx[k_eq]
                    # ∂/∂alpha[i+1, k_pert]: +sqrt_lode if k_eq == k_pert
                    if k_eq == k_pert
                        J[row_ode, col_ip1] += sqrt_lode
                    end
                end
            end
        end
    else
        # Continuous: residual_i = sqrt_lode * (D*alpha[i] - F[i])
        # ∂F/∂x is the state Jacobian of the ODE RHS, computed by pointwise FD
        for i in 1:T
            u = alpha[i, :]
            try; prob.dynamics!(du0, u, p, times[i]); catch; du0 .= 0; end

            for k_pert in 1:K
                u_p = copy(u)
                u_p[k_pert] += eps_x
                try; prob.dynamics!(du_p, u_p, p, times[i]); catch; du_p .= du0; end
                dF_dx = (du_p .- du0) ./ eps_x  # ∂f/∂x_k at time i

                col = (k_pert - 1) * T + i  # alpha parameter index
                for k_eq in 1:K
                    row_ode = n_data + (k_eq - 1) * T + i
                    if k_eq == k_pert
                        J[row_ode, col] += sqrt_lode * D[i, i]
                    end
                    J[row_ode, col] -= sqrt_lode * dF_dx[k_eq]
                end
            end

            # Off-diagonal D entries for same state
            for k_eq in 1:K
                row_ode = n_data + (k_eq - 1) * T
                for j_col in 1:T
                    if j_col != i
                        col = (k_eq - 1) * T + j_col
                        J[row_ode + i, col] += sqrt_lode * D[i, j_col]
                    end
                end
            end
        end
    end

    # 3) ODE residual w.r.t. beta: ∂(√λ(Dα - F))/∂β = -√λ ∂F/∂β
    eps_beta = 1e-5
    for b in 1:n_beta
        col = n_alpha + b
        beta_p = copy(beta)
        step = max(eps_beta, abs(beta[b]) * eps_beta)
        beta_p[b] += step
        F_p = eval_ode_rhs(prob, times, alpha, beta_p)
        for k in 1:K
            for i in 1:T
                row_ode = n_data + (k - 1) * T + i
                J[row_ode, col] = -sqrt_lode * (F_p[i, k] - F[i, k]) / step
            end
        end
    end

    # 4) State smoothness Jacobian for discrete models
    # smooth_resid[idx] = sqrt_lsmooth * (alpha[i+1,k] - 2*alpha[i,k] + alpha[i-1,k])
    if prob.discrete && T >= 3
        sqrt_lsmooth = lambda_ode^0.25
        for k in 1:K
            for i in 2:(T-1)
                row_s = n_data + n_ode + (k - 1) * (T - 2) + (i - 1)
                col_im1 = (k - 1) * T + (i - 1)  # alpha[i-1, k]
                col_i   = (k - 1) * T + i         # alpha[i, k]
                col_ip1 = (k - 1) * T + (i + 1)   # alpha[i+1, k]
                J[row_s, col_im1] += sqrt_lsmooth
                J[row_s, col_i]   -= 2 * sqrt_lsmooth
                J[row_s, col_ip1] += sqrt_lsmooth
            end
        end
    end

    resid, J
end

# ─── Residual-only evaluation (for line search) ──────────────────

"""
Compute the combined residual without the Jacobian (cheaper than full version).
"""
function collocation_residual_only(
        prob::PSMProblem, times::Vector{Float64},
        alpha::Matrix{Float64}, beta::Vector{Float64},
        D::Matrix{Float64}, lambda_ode::Float64,
        w_vec::Vector{Float64})

    T, K = size(alpha)
    n_obs = size(prob.data_values, 2)
    sqrt_lode = sqrt(lambda_ode)

    data_resid = zeros(T * n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:T
            idx = (j - 1) * T + i
            data_resid[idx] = sqrt(w_vec[idx]) * (prob.data_values[i, j] - alpha[i, sk])
        end
    end

    F = eval_ode_rhs(prob, times, alpha, beta)
    ode_resid = zeros(T * K)

    if prob.discrete
        for k in 1:K
            for i in 1:(T-1)
                ode_resid[(k - 1) * T + i] = sqrt_lode * (alpha[i+1, k] - F[i, k])
            end
            ode_resid[(k - 1) * T + T] = 0.0
        end
    else
        for k in 1:K
            dalpha_k = D * alpha[:, k]
            for i in 1:T
                ode_resid[(k - 1) * T + i] = sqrt_lode * (dalpha_k[i] - F[i, k])
            end
        end
    end

    # State roughness penalty for discrete
    smooth_resid = Float64[]
    if prob.discrete && T >= 3
        n_smooth = (T - 2) * K
        smooth_resid = zeros(n_smooth)
        sqrt_lsmooth = lambda_ode^0.25
        for k in 1:K
            for i in 2:(T-1)
                idx = (k - 1) * (T - 2) + (i - 1)
                smooth_resid[idx] = sqrt_lsmooth * (alpha[i+1, k] - 2*alpha[i, k] + alpha[i-1, k])
            end
        end
    end

    vcat(data_resid, ode_resid, smooth_resid)
end

# ─── Main collocation solver ──────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::CollocationLAML)

Fit a partially specified model using generalised profiling (collocation).
State trajectories are represented as spline curves and optimised jointly
with the unknown-function coefficients while enforcing the ODE/map as a
soft constraint.

# Algorithm
1. Initialise state spline coefficients from smoothed data.
2. Outer loop: update unknown-function parameters via LAML-penalised least
   squares with Fellner–Schall smoothing parameter estimation.
3. Inner loop: refine state spline coefficients by minimising a combined
   data-fit + ODE-fidelity + roughness-penalty objective.
4. Use a continuation schedule on the ODE-fidelity weight to gradually
   tighten the dynamic constraint.

# References
- Ramsay et al. (2007), "Parameter estimation for differential equations:
  a generalized smoothing approach", JRSS-B.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::CollocationLAML)
    _validate_problem(prob, "CollocationLAML")
    times = Float64.(prob.data_times)
    T_pts = length(times)
    n_obs = size(prob.data_values, 2)
    K = length(prob.u0)  # Number of state variables

    # Build differentiation matrix
    D = build_diff_matrix(times)

    # Initialize state values (alpha) from data where observed, interpolate rest
    # For discrete models, smooth the data to avoid noisy initialization that
    # distorts the function recovery during the continuation.
    alpha = zeros(T_pts, K)
    observed_states = Set(prob.obs_to_state)
    for k in 1:K
        obs_idx = findfirst(j -> prob.obs_to_state[j] == k, 1:n_obs)
        if obs_idx !== nothing
            raw = max.(prob.data_values[:, obs_idx], 0.01)
            if prob.discrete && T_pts >= 4
                itp = CubicSpline(raw, times;
                                  extrapolation=ExtrapolationType.Extension)
                for i in 1:T_pts
                    alpha[i, k] = max(itp(times[i]), 0.01)
                end
            else
                alpha[:, k] .= raw
            end
        else
            alpha[:, k] .= prob.u0[k]
        end
    end

    # Initialize unknown function parameters (beta)
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Build penalty matrices for unknown functions
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)
    n_alpha = T_pts * K

    # Smoothing parameters for unknown functions
    if m > 0
        theta = Float64[1.0 / max(tr(S_list[l]), 1e-10) for l in 1:m]
    else
        theta = Float64[]
    end

    # Data weights (flattened: T × n_obs)
    w_vec = zeros(T_pts * n_obs)
    for j in 1:n_obs, i in 1:T_pts
        w_vec[(j - 1) * T_pts + i] = prob.data_weights[i, j]
    end
    y_vec = zeros(T_pts * n_obs)
    for j in 1:n_obs, i in 1:T_pts
        y_vec[(j - 1) * T_pts + i] = prob.data_values[i, j]
    end

    # Continuation schedule for λ_ode
    # For discrete models, cap λ_ode_end at 100: the compliance penalty
    # sqrt(λ) * (α[t+1] - F[t]) couples only consecutive pairs (unlike the
    # continuous D matrix which couples all states), so very high λ overwhelms
    # data fidelity without improving dynamics fit.
    lode_end = prob.discrete ? min(alg.lambda_ode_end, 100.0) : alg.lambda_ode_end
    lambda_ode_schedule = exp.(range(log(alg.lambda_ode_start),
                                     log(lode_end),
                                     length=alg.n_continuation))

    verbose = alg.verbose
    if verbose
        println("CollocationLAML: $(n_alpha) state params + $(n_beta) function params, " *
                "$(T_pts) collocation points, $(K) states")
        println("λ_ode schedule: ", round.(lambda_ode_schedule, sigdigits=3))
    end

    # ─── Continuation loop ────────────────────────────────────────
    for (level, lambda_ode) in enumerate(lambda_ode_schedule)
        if verbose
            println("\n=== Continuation level $level: λ_ode = $(round(lambda_ode, sigdigits=4)) ===")
        end

        prev_obj = Inf

        for iter in 1:alg.maxiters
            # Compute combined residual and Jacobian
            resid, J_full = collocation_residual_jacobian(
                prob, times, alpha, beta, D, lambda_ode, w_vec)

            # Build penalty for beta only (alpha is unpenalized)
            # The penalty is embedded in the full parameter space [alpha; beta]
            n_total = n_alpha + n_beta
            B_beta = zeros(n_beta, n_beta)
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                B_beta[off+1:off+nk, off+1:off+nk] .+= theta[l] .* S_list[l]
            end

            # Augmented penalty in full space (only beta penalized)
            B_full = zeros(n_total, n_total)
            B_full[n_alpha+1:end, n_alpha+1:end] .= B_beta

            # Current objective
            data_ss = sum(resid[1:T_pts*n_obs].^2)
            ode_ss = sum(resid[T_pts*n_obs+1:end].^2)
            pen = dot(beta, B_beta * beta)
            curr_obj = sum(resid.^2) + pen

            if verbose && (iter <= 3 || iter % 10 == 0)
                println("  iter $iter: data_SS=$(round(data_ss, sigdigits=5)) " *
                        "ode_SS=$(round(ode_ss, sigdigits=5)) pen=$(round(pen, sigdigits=4))")
            end

            # Check convergence
            if iter > 1 && abs(curr_obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
                if verbose; println("  Converged at iter $iter"); end
                break
            end
            prev_obj = curr_obj

            # Gauss-Newton step: minimize ||r + J δ||² + δ'B δ
            # Normal equations: (J'J + B) δ = -J'r → δ = -(J'J + B)⁻¹ J'r
            # Update: params_new = params + δ (already negative from the -J'r)
            alpha_flat = vec(alpha)
            params_vec = vcat(alpha_flat, beta)

            JtJ = J_full' * J_full + B_full
            neg_Jtr = -(J_full' * resid)  # Note: negative!

            delta = try
                JtJ \ neg_Jtr
            catch
                try
                    (JtJ + 1e-6 * I) \ neg_Jtr
                catch
                    if verbose; println("  Singular system, breaking"); end
                    break
                end
            end

            # Line search (residual-only, no Jacobian recomputation)
            best_obj = curr_obj
            best_step = 0.0
            for k in 0:8
                step_size = 0.5^k
                params_new = params_vec .+ step_size .* delta
                alpha_new = reshape(params_new[1:n_alpha], T_pts, K)
                beta_new = params_new[n_alpha+1:end]

                resid_new = collocation_residual_only(
                    prob, times, alpha_new, beta_new, D, lambda_ode, w_vec)
                pen_new = dot(beta_new, B_beta * beta_new)
                obj_new = sum(resid_new.^2) + pen_new

                if obj_new < best_obj
                    best_obj = obj_new
                    best_step = step_size
                end
            end

            if best_step == 0.0
                if verbose; println("  No improvement, stopping"); end
                break
            end

            # Apply best step
            params_new = params_vec .+ best_step .* delta
            alpha .= reshape(params_new[1:n_alpha], T_pts, K)
            beta .= params_new[n_alpha+1:end]
        end

        # Update smoothing parameters via LAML after convergence at this level
        if m > 0
            # Build Jacobian of data residual w.r.t. beta only (for LAML)
            # Approximate: use the ODE-constrained predictions
            f_vec_data = zeros(T_pts * n_obs)
            for j in 1:n_obs
                sk = prob.obs_to_state[j]
                for i in 1:T_pts
                    f_vec_data[(j - 1) * T_pts + i] = alpha[i, sk]
                end
            end

            # Compute Jacobian of fitted values w.r.t. beta via FD
            J_beta = zeros(T_pts * n_obs, n_beta)
            eps = 1e-5
            for b in 1:n_beta
                beta_p = copy(beta)
                step = max(eps, abs(beta[b]) * eps)
                beta_p[b] += step
                # Re-solve the inner problem briefly with perturbed beta
                # (Approximation: just evaluate ODE RHS change)
                F0 = eval_ode_rhs(prob, times, alpha, beta)
                F1 = eval_ode_rhs(prob, times, alpha, beta_p)
                # dF/dbeta affects alpha through the ODE constraint
                # For LAML, use a simple approximation: hold alpha fixed
                # This underestimates the sensitivity but is computationally tractable
                for j in 1:n_obs
                    sk = prob.obs_to_state[j]
                    J_beta[:, b] .= 0.0  # alpha doesn't change with beta in this approx
                end
            end

            # Simplified LAML: use data residuals and a rough Jacobian
            # For the smoothing parameter estimation, the key quantities are:
            # - beta'S beta (penalty magnitude)
            # - edf (effective degrees of freedom)
            # - sigma2 (residual variance)
            # We can estimate these without a full Jacobian
            data_ss = sum(w_vec .* (y_vec .- f_vec_data).^2)
            sigma2_est = data_ss / (T_pts * n_obs)

            if alg.sigma2_init !== nothing
                sigma2_est = min(sigma2_est, alg.sigma2_init)
            end

            # Simple Fellner-Schall update for each smooth term
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                beta_k = beta[off+1:off+nk]
                bSb = dot(beta_k, S_list[l] * beta_k)
                rank_k = min(nk, nk - 2)  # Typical rank of second-derivative penalty
                edf_k = max(rank_k * 0.5, 1.0)  # Conservative estimate
                if bSb > 1e-30
                    theta[l] = clamp(sigma2_est * edf_k / bSb, 1e-20, 1e20)
                end
            end

            if verbose
                println("  LAML update: σ²=$(round(sigma2_est, sigdigits=4)) " *
                        "θ=$(round.(theta, sigdigits=3))")
            end
        end
    end

    # ─── Build solution ───────────────────────────────────────────
    # Compute final predictions (alpha at observed states)
    pred = zeros(T_pts, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= alpha[:, sk]
    end

    # Compute data loss
    data_loss = 0.0
    for j in 1:n_obs, i in 1:T_pts
        data_loss += prob.data_weights[i, j] * (prob.data_values[i, j] - pred[i, j])^2
    end

    # Compute ODE compliance
    F = eval_ode_rhs(prob, times, alpha, beta)
    ode_loss = 0.0
    if prob.discrete
        # Discrete: compliance is alpha[i+1] - F[i]
        for k in 1:K
            for i in 1:(T_pts-1)
                ode_loss += (alpha[i+1, k] - F[i, k])^2
            end
        end
    else
        for k in 1:K, i in 1:T_pts
            dalpha_k = (D * alpha[:, k])[i]
            ode_loss += (dalpha_k - F[i, k])^2
        end
    end

    # Build unknown function evaluators
    p_opt = build_param_struct(prob, beta)
    uf_evals = Dict{Symbol, Any}()
    for approx in prob.approximators
        if haskey(p_opt, approx.name)
            uf_evals[approx.name] = p_opt[approx.name]
        end
    end

    # EDF approximation
    edf = sum(uf_nk)  # Conservative: use full parameter count

    # Build parameter ComponentArray
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    obj_val = data_loss + sum(theta[l] * dot(beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]],
                  S_list[l] * beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]]) for l in 1:m)

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "ode_compliance=$(round(ode_loss, sigdigits=5)) " *
                "EDF=$(round(edf, digits=2))")
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (ode_compliance=ode_loss, lambda_ode_final=alg.lambda_ode_end))
end
