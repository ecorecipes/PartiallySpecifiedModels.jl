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
using ForwardDiff   # jac=:forwarddiff Jacobian path

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
    T >= 3 || error("build_diff_matrix: need at least 3 time points for " *
                    "second-order differences, got $T")
    for i in 1:T-1
        times[i+1] <= times[i] && error("build_diff_matrix: times must be strictly increasing (duplicate at index $i)")
    end
    D = zeros(T, T)

    # Forward difference at t_1: general 3-point second-order formula
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

# Failure-sentinel convention (note the deliberate asymmetry between the
# residual and the Jacobian):
#
#   RESIDUAL path — when the dynamics throw a genuinely numerical error
#   (domain error, NaN/Inf conversion, ...) at a collocation point, the RHS
#   is replaced by _COLLOC_FAIL_SENTINEL. That is a LARGE residual, which is
#   what we want: the optimizer is pushed away from infeasible points.
#
#   JACOBIAN path — the FD slope at a failed point is forced to ZERO, not
#   differenced against the sentinel. Differencing is only valid when the
#   base and perturbed evaluations live on the same branch: at a domain
#   boundary (e.g. `sqrt(u)`/`log(u)` with u dipping just below zero) the
#   base can fail while the perturbation succeeds, and
#   (f(u+ε) − sentinel)/ε ≈ −1e12 poisons J'J with ~1e24 entries and makes
#   the Gauss–Newton solve return garbage without erroring. A zero column
#   simply says "no reliable local linearization here"; the large residual
#   still supplies the pressure to move away.
#
# So: large residual, zero Jacobian. Programming errors (see
# `_is_program_error`) are always rethrown.
const _COLLOC_FAIL_SENTINEL = 1e6

"""
Evaluate the dynamics right-hand side at all collocation points, also
returning a per-point failure mask.

For continuous models: returns `f(x(t), p, t)` (derivatives).
For discrete models: returns `f(x(t), p, t)` (next-state map).

Returns `(F, failed)` where `F` is a (T × K) matrix with entry
[i,k] = f_k(x(t_i), p, t_i) and `failed[i]` is `true` when the dynamics
evaluation at point `i` raised a numerical error and `F[i, :]` therefore
holds the failure sentinel rather than a real derivative.
"""
function eval_ode_rhs_masked(prob::PSMProblem, times::Vector{Float64},
                             alpha::Matrix{Float64}, beta::Vector{Float64})
    T, K = size(alpha)
    F = zeros(T, K)
    failed = fill(false, T)
    p = build_param_struct(prob, beta)
    du = zeros(K)

    for i in 1:T
        u = alpha[i, :]
        try
            prob.dynamics!(du, u, p, times[i])
        catch e
            _is_program_error(e) && rethrow()
            du .= _COLLOC_FAIL_SENTINEL  # numerical failure → large residual
            failed[i] = true
        end
        F[i, :] .= du
    end
    F, failed
end

"""
Evaluate the dynamics right-hand side at all collocation points.

Returns a (T × K) matrix where entry [i,k] = f_k(x(t_i), p, t_i).
See [`eval_ode_rhs_masked`](@ref) for the variant that also reports which
points fell back to the failure sentinel.
"""
function eval_ode_rhs(prob::PSMProblem, times::Vector{Float64},
                      alpha::Matrix{Float64}, beta::Vector{Float64})
    first(eval_ode_rhs_masked(prob, times, alpha, beta))
end

# ─── Pointwise state Jacobian ∂F/∂x (FD / ForwardDiff) ────────────

"""
    _colloc_state_jac!(dFdx, prob, p, u, t, du0, du_p; jac=:fd)

Fill `dFdx` (K × K, entry `[k_eq, k_pert] = ∂f_{k_eq}/∂x_{k_pert}`) at one
collocation point, preserving the failure-sentinel convention above
(large residual, ZERO Jacobian — the sentinel is never differenced or
differentiated):

- `jac=:fd` (historical behavior, byte-identical): base evaluation plus one
  forward perturbation per state (step 1e-6). A failed base zeroes every
  column; a failed perturbation zeroes that column.
- `jac=:forwarddiff`: a single Dual-valued evaluation replaces the K+1
  Float64 evaluations. The same numerical exceptions the FD path converts
  to the sentinel zero the WHOLE block for the point (as a failed base does
  under `:fd`), and so do non-finite Dual partials; the sentinel value is
  never part of the differentiated computation. One deliberate nuance,
  strictly an improvement: when the base evaluation succeeds, forward mode
  returns the exact derivative at the point itself and never perturbs the
  state, so the FD path's defensive zero column for "perturbation crossed a
  domain boundary" cannot arise — that zero existed only to protect FD from
  differencing across a branch, a failure mode forward mode does not have.

`du0`/`du_p` are Float64 work buffers (used by the `:fd` path only).
Program errors are always rethrown.
"""
function _colloc_state_jac!(dFdx::Matrix{Float64}, prob::PSMProblem, p,
                            u::Vector{Float64}, t::Float64,
                            du0::Vector{Float64}, du_p::Vector{Float64};
                            jac::Symbol=:fd)
    K = length(u)
    if jac === :forwarddiff
        fill!(dFdx, 0.0)
        M = try
            ForwardDiff.jacobian(
                uu -> (duu = zero(uu);   # zero, not similar: dynamics that
                       # skip a du entry must read 0.0, not garbage memory
                       # (parity with _colloc_beta_jac_ad!'s fill!(dud, 0))
                       prob.dynamics!(duu, uu, p, t);
                       duu), u)
        catch e
            _is_program_error(e) && rethrow()
            nothing   # numerical failure → zero block (sentinel convention)
        end
        if M !== nothing && all(isfinite, M)
            dFdx .= M
        end
        return nothing
    end

    # :fd — byte-identical to the historical inline code.
    base_failed = false
    try
        prob.dynamics!(du0, u, p, t)
    catch e
        _is_program_error(e) && rethrow()
        du0 .= _COLLOC_FAIL_SENTINEL  # match the residual sentinel
        base_failed = true
    end

    eps_x = 1e-6
    for k_pert in 1:K
        u_p = copy(u)
        u_p[k_pert] += eps_x
        pert_failed = false
        try
            prob.dynamics!(du_p, u_p, p, t)
        catch e
            _is_program_error(e) && rethrow()
            du_p .= du0
            pert_failed = true
        end
        # Zero Jacobian at failed points — never difference against
        # the sentinel (see the sentinel convention above).
        if base_failed || pert_failed
            dFdx[:, k_pert] .= 0.0
        else
            dFdx[:, k_pert] .= (du_p .- du0) ./ eps_x
        end
    end
    nothing
end

# ─── ∂F/∂β via ForwardDiff (jac=:forwarddiff) ────────────────────

"""
    _colloc_beta_jac_ad!(J, prob, times, alpha, beta, F_failed, sqrt_lode,
                         n_data, n_alpha, cfg) -> Bool

Fill the `∂(√λ(Dα − F))/∂β = −√λ ∂F/∂β` block of the collocation Jacobian
by one chunked Dual sweep over β, preserving the failure-sentinel
convention EXACTLY:

- collocation points with `F_failed[i]` (their residual holds the sentinel)
  are skipped inside the Dual-valued map — the sentinel is never part of
  the differentiated computation — and their Jacobian rows are forced to
  zero, matching the FD path's zeroing (granularity differs in one
  practically-unreachable corner: FD's `Fp_failed` zeroes per
  (point, column), the AD path zeroes the point's whole row — the two
  coincide whenever the primal evaluation is deterministic);
- a point whose Dual evaluation raises a numerical error is likewise
  masked to zero (the FD path's `Fp_failed` case), and program errors are
  rethrown.

Returns `true` on success. Returns `false` — caller falls back to the FD
loop — when the sweep itself fails or produces non-finite entries.
"""
function _colloc_beta_jac_ad!(J::AbstractMatrix, prob::PSMProblem,
                              times::Vector{Float64},
                              alpha::Matrix{Float64},
                              beta::Vector{Float64},
                              F_failed::Vector{Bool}, sqrt_lode::Float64,
                              n_data::Int, n_alpha::Int, cfg)
    T, K = size(alpha)
    n_beta = length(beta)
    # Per-point failure mask for the Dual sweep. ForwardDiff may call the
    # map several times (one pass per chunk); the dynamics are
    # deterministic, so the same points fail on every pass and |= just
    # re-records them.
    ad_failed = fill(false, T)
    Fmap = function (bd)
        pd = build_param_struct(prob, bd)
        Fd = zeros(eltype(bd), T * K)
        dud = zeros(eltype(bd), K)
        for i in 1:T
            # Sentinel points: skipped entirely — rows stay identically
            # zero and the sentinel never touches a Dual.
            F_failed[i] && continue
            fill!(dud, zero(eltype(bd)))
            try
                prob.dynamics!(dud, alpha[i, :], pd, times[i])
            catch e
                _is_program_error(e) && rethrow()
                ad_failed[i] = true
                continue      # numerical failure → zero row (mask below)
            end
            for k in 1:K
                Fd[(k - 1) * T + i] = dud[k]
            end
        end
        Fd
    end
    JF = try
        if cfg === nothing
            ForwardDiff.jacobian(Fmap, beta)
        else
            ForwardDiff.jacobian(Fmap, beta, cfg, Val{false}())
        end
    catch e
        _is_program_error(e) && rethrow()
        return false
    end
    all(isfinite, JF) || return false
    for b in 1:n_beta
        col = n_alpha + b
        for k in 1:K, i in 1:T
            row_ode = n_data + (k - 1) * T + i
            # Zero Jacobian at failed points (see sentinel convention).
            J[row_ode, col] = (F_failed[i] || ad_failed[i]) ? 0.0 :
                -sqrt_lode * JF[(k - 1) * T + i, b]
        end
    end
    true
end

# ─── Combined residual and Jacobian ───────────────────────────────

"""
Build the combined residual vector:
  r = [ √w × (y_obs - alpha_obs)    ;  data fidelity
        √λ_ode × (D*alpha - F)_flat ]  ODE compliance

and the combined Jacobian ∂r/∂[alpha_flat; beta].

Uses analytical derivatives where possible:
- Data residual ∂/∂alpha: trivial (−√w δ_{ij})
- ODE residual ∂/∂alpha: D − ∂F/∂x (D is the diff matrix, ∂F/∂x pointwise
  via `_colloc_state_jac!` — FD by default, ForwardDiff under
  `jac=:forwarddiff`)
- ODE residual ∂/∂beta: −∂F/∂beta (FD over beta by default; one Dual sweep
  via `_colloc_beta_jac_ad!` under `jac=:forwarddiff`, with FD fallback)

`fd_cfg` is an optional per-solve `ForwardDiff.JacobianConfig` over β for
the `jac=:forwarddiff` path.
"""
function collocation_residual_jacobian(
        prob::PSMProblem, times::Vector{Float64},
        alpha::Matrix{Float64}, beta::Vector{Float64},
        D::Matrix{Float64}, lambda_ode::Float64,
        w_vec::Vector{Float64};
        jac::Symbol=:fd, fd_cfg=nothing)

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
            # `sqrt(0) * NaN = NaN`: a masked cell's ZERO weight does not
            # neutralize a NaN datum, so the residual must be zeroed
            # explicitly. Left unguarded, one masked cell makes the
            # Gauss-Newton step all-NaN, the line search rejects every
            # step, and the solver reports `:plateau` convergence at its
            # initialization. (The Jacobian row is already a clean 0.)
            data_resid[idx] = usable_cell(prob, i, j) ?
                sqrt(w_vec[idx]) * (prob.data_values[i, j] - alpha[i, sk]) : 0.0
        end
    end

    # --- ODE/Discrete compliance residual ---
    F, F_failed = eval_ode_rhs_masked(prob, times, alpha, beta)
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

    # 2) Compliance residual w.r.t. alpha — pointwise ∂F/∂x through
    #    _colloc_state_jac! (FD by default, ForwardDiff under
    #    jac=:forwarddiff; failure-sentinel convention preserved there)
    p = build_param_struct(prob, beta)
    du0 = zeros(K)
    du_p = zeros(K)
    dFdx = zeros(K, K)   # [k_eq, k_pert] = ∂f_{k_eq}/∂x_{k_pert}

    if prob.discrete
        # Discrete: residual_i = sqrt_lode * (alpha[i+1,k] - F[i,k])
        # ∂/∂alpha[i+1,k_pert] for same state k_eq: +sqrt_lode * δ(k_eq==k_pert)
        # ∂/∂alpha[i,k_pert]: -sqrt_lode * ∂F[i,k_eq]/∂x[k_pert]
        for i in 1:(T-1)
            u = alpha[i, :]
            _colloc_state_jac!(dFdx, prob, p, u, times[i], du0, du_p; jac=jac)

            for k_pert in 1:K
                col_i = (k_pert - 1) * T + i  # alpha[i, k_pert]
                col_ip1 = (k_pert - 1) * T + (i + 1)  # alpha[i+1, k_pert]

                for k_eq in 1:K
                    row_ode = n_data + (k_eq - 1) * T + i
                    # ∂/∂alpha[i, k_pert]: -sqrt_lode * ∂F/∂x
                    J[row_ode, col_i] -= sqrt_lode * dFdx[k_eq, k_pert]
                    # ∂/∂alpha[i+1, k_pert]: +sqrt_lode if k_eq == k_pert
                    if k_eq == k_pert
                        J[row_ode, col_ip1] += sqrt_lode
                    end
                end
            end
        end
    else
        # Continuous: residual_i = sqrt_lode * (D*alpha[i] - F[i])
        for i in 1:T
            u = alpha[i, :]
            _colloc_state_jac!(dFdx, prob, p, u, times[i], du0, du_p; jac=jac)

            for k_pert in 1:K
                col = (k_pert - 1) * T + i  # alpha parameter index
                for k_eq in 1:K
                    row_ode = n_data + (k_eq - 1) * T + i
                    if k_eq == k_pert
                        J[row_ode, col] += sqrt_lode * D[i, i]
                    end
                    J[row_ode, col] -= sqrt_lode * dFdx[k_eq, k_pert]
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
    beta_block_done = false
    if jac === :forwarddiff
        beta_block_done = _colloc_beta_jac_ad!(
            J, prob, times, alpha, beta, F_failed, sqrt_lode,
            n_data, n_alpha, fd_cfg)
        beta_block_done ||
            @debug "CollocationLAML: ForwardDiff ∂F/∂β sweep failed or " *
                   "returned non-finite entries; falling back to finite " *
                   "differences for this evaluation"
    end
    if !beta_block_done
        eps_beta = 1e-5
        for b in 1:n_beta
            col = n_alpha + b
            beta_p = copy(beta)
            step = max(eps_beta, abs(beta[b]) * eps_beta)
            beta_p[b] += step
            F_p, Fp_failed = eval_ode_rhs_masked(prob, times, alpha, beta_p)
            for k in 1:K
                for i in 1:T
                    row_ode = n_data + (k - 1) * T + i
                    # Zero Jacobian at failed points (see sentinel convention):
                    # differencing a real value against the sentinel — in either
                    # direction — fabricates a ~1e12 slope.
                    J[row_ode, col] = (F_failed[i] || Fp_failed[i]) ? 0.0 :
                        -sqrt_lode * (F_p[i, k] - F[i, k]) / step
                end
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
            # `sqrt(0) * NaN = NaN`: a masked cell's ZERO weight does not
            # neutralize a NaN datum, so the residual must be zeroed
            # explicitly. Left unguarded, one masked cell makes the
            # Gauss-Newton step all-NaN, the line search rejects every
            # step, and the solver reports `:plateau` convergence at its
            # initialization. (The Jacobian row is already a clean 0.)
            data_resid[idx] = usable_cell(prob, i, j) ?
                sqrt(w_vec[idx]) * (prob.data_values[i, j] - alpha[i, sk]) : 0.0
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
`sol.convergence` is a NamedTuple `(ode_compliance, lambda_ode_final,
converged, iterations, reason, iterations_total)` — see the
`CollocationLAML` docstring for the key taxonomy.
"""
function SciMLBase.solve(prob::PSMProblem, alg::CollocationLAML)
    _validate_problem(prob, "CollocationLAML")
    isempty(prob.delays) ||
        error("CollocationLAML does not support DDE problems: " *
              "the collocation residual evaluates the dynamics with the " *
              "4-argument ODE signature and cannot supply the delayed " *
              "history. Use LAML for DDEs.")
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
            # Clamp initial states away from zero only when the data are
            # themselves non-negative (a positivity-assuming model);
            # unconditionally forcing ≥ 0.01 corrupted legitimately negative
            # states (log-abundances, anomalies).
            # Seed from the USABLE rows only, interpolating across masked
            # gaps. Two distinct failures otherwise: `all(>=(0.0), col)` is
            # false whenever the column holds a NaN (NaN >= 0 is false), so
            # the positivity floor silently switches off for any partially
            # masked non-negative series; and the NaN itself lands in
            # `alpha`, making the whole state block NaN from step one.
            keep = usable_rows(prob, obs_idx)
            if isempty(keep)
                alpha[:, k] .= prob.u0[k]
            else
                col = Float64.(prob.data_values[keep, obs_idx])
                floor_k = all(>=(0.0), col) ? 0.01 : -Inf
                raw = max.(col, floor_k)
                if prob.discrete && T_pts >= 4 && length(keep) >= 4
                    # Unchanged discrete path: spline through the kept rows
                    # (all rows when nothing is masked).
                    itp = CubicSpline(raw, times[keep];
                                      extrapolation=ExtrapolationType.Extension)
                    for i in 1:T_pts
                        alpha[i, k] = max(itp(times[i]), floor_k)
                    end
                elseif length(keep) == T_pts
                    # Unchanged continuous path.
                    alpha[:, k] .= raw
                elseif length(keep) >= 4
                    # Continuous with masked rows: interpolate the kept rows
                    # onto the full grid rather than dropping the state.
                    itp = CubicSpline(raw, times[keep];
                                      extrapolation=ExtrapolationType.Extension)
                    for i in 1:T_pts
                        alpha[i, k] = max(itp(times[i]), floor_k)
                    end
                else
                    alpha[:, k] .= raw[1]
                end
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

    # jac=:forwarddiff — one JacobianConfig over β per solve (chunking
    # depends only on n_beta), shared by the ∂F/∂β residual block and the
    # Fellner–Schall EDF Jacobian below.
    fd_cfg = alg.jac === :forwarddiff ? _fd_jacobian_config(n_beta) : nothing

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

    # Data weights (flattened: T × n_obs), with masked cells forced to weight
    # 0 so the residual and Jacobian rows they generate are identically zero.
    w_vec = zeros(T_pts * n_obs)
    for j in 1:n_obs, i in 1:T_pts
        w_vec[(j - 1) * T_pts + i] = usable_cell(prob, i, j) ?
                                     prob.data_weights[i, j] : 0.0
    end
    # The package-wide count (`solver.jl`), not a re-derived one, so
    # this can never drift from the predicate the flatten applied.
    n_usable_cells = n_usable(prob)
    n_usable_cells == 0 && error("CollocationLAML: every observation is " *
        "masked (every weight is 0 or every value is non-finite); there " *
        "is nothing to fit.")

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
    # θ̂ REPORTED TO THE USER: the smoothing parameters β was actually fitted
    # under. Two θ vectors coexist in this loop and they are NOT
    # interchangeable (the same distinction `solver.jl` draws for LAML, which
    # carries three):
    #   theta     — the newest Fellner–Schall PROPOSAL. It is produced at the
    #               END of a continuation level and consumed only by the NEXT
    #               level's inner loop, so after the LAST level it is
    #               untested: no coefficient step was ever taken under it.
    #   theta_fit — the θ whose penalty B(θ) actually produced the current
    #               (α, β). This is the only one at which β is the penalized
    #               optimum, so it is what `smoothing_params` and `obj_val`
    #               are reported at. `edf`/σ̂² are ALREADY computed from the H
    #               built with this θ (the Fellner–Schall block below forms
    #               S_full before it overwrites `theta`), so collapsing onto
    #               theta_fit is also what makes edf and λ̂ describe one fit.
    # Measured, all on fits reporting converged == true: on the suite's
    # exponential-growth fixture the reported λ̂ was 5536.4 where β had been
    # fitted at 384.6 (14.4×); on a Poisson fixture 8.402 vs 0.1559 (53.9×),
    # inflating the reported penalty term to 0.653 from 0.0121 and leaving β̂
    # 1.3e5× further from stationarity of the penalized collocation objective
    # at the reported λ̂ than at the fitted one. The reported EDF was already
    # computed at theta_fit, so pre-fix a fit could print an EDF of 3.03
    # (its value at λ=0.156) beside a λ̂ of 8.402, whose own EDF is 2.08 —
    # 45% apart.
    #
    # Report rather than refit at the proposal. Unlike LAML — where the
    # accept logic had explicitly REJECTED the final θ's step — this
    # proposal was never rejected, so that argument is unavailable here.
    # What closes the door instead is that the Fellner–Schall sequence is
    # nowhere near a fixed point at exit: taking one further FS step after
    # refitting moves λ again by factors of 0.07× to 148×. Refitting would
    # therefore relocate the same inconsistency one step down a
    # non-convergent sequence, while changing the delivered fit by up to
    # 16% of ‖β‖ and 35% of data_loss IN BOTH DIRECTIONS — and collocation
    # computes no marginal likelihood that could say the new point is
    # better.
    theta_fit = copy(theta)
    edf_final = Float64(sum(uf_nk))   # updated by the Fellner–Schall step
    fs_skip_warned = false            # warn once if the FS step never runs
    # Honest convergence reporting: converged/reason describe the FINAL
    # continuation level's inner loop; iterations counts its inner iterations
    # (iterations_total accumulates across all levels).
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0
    conv_iters_total = 0
    for (level, lambda_ode) in enumerate(lambda_ode_schedule)
        if verbose
            println("\n=== Continuation level $level: λ_ode = $(round(lambda_ode, sigdigits=4)) ===")
        end

        prev_obj = Inf
        conv_converged = false
        conv_reason = :maxiters
        conv_iters = 0

        for iter in 1:alg.maxiters
            conv_iters = iter
            conv_iters_total += 1
            # Compute combined residual and Jacobian
            resid, J_full = collocation_residual_jacobian(
                prob, times, alpha, beta, D, lambda_ode, w_vec;
                jac=alg.jac, fd_cfg=fd_cfg)

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
                conv_converged = true
                conv_reason = :converged_tol
                break
            end
            prev_obj = curr_obj

            # Gauss-Newton step for the penalized objective
            #   f(p) = ||r(p)||² + p'B p,  ∇f = 2(J'r + B p),
            # so the normal equations are (J'J + B) δ = -(J'r + B p).
            # Omitting the B·p term makes δ nonzero at the true optimum
            # (where J'r = -B p) and stalls the line search short of it.
            alpha_flat = vec(alpha)
            params_vec = vcat(alpha_flat, beta)

            JtJ = J_full' * J_full + B_full
            neg_Jtr = -(J_full' * resid .+ B_full * params_vec)

            delta = try
                JtJ \ neg_Jtr
            catch
                try
                    (JtJ + 1e-6 * I) \ neg_Jtr
                catch
                    if verbose; println("  Singular system, breaking"); end
                    conv_reason = :early_break
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
                conv_converged = true
                conv_reason = :plateau
                break
            end

            # Apply best step
            params_new = params_vec .+ best_step .* delta
            alpha .= reshape(params_new[1:n_alpha], T_pts, K)
            beta .= params_new[n_alpha+1:end]
            # (α, β) were just stepped under the penalty B(theta). Record it
            # HERE — the only place β is accepted — and before the
            # Fellner–Schall block at the end of this level moves `theta` onto
            # its next proposal. If a level's inner loop accepts no step at
            # all (singular system, or the line search rejects every
            # contraction on its first iteration) theta_fit correctly stays on
            # the earlier θ that did produce the current β.
            theta_fit .= theta
        end

        # ── Fellner–Schall smoothing-parameter update ────────────────
        # Uses the genuine effective degrees of freedom
        #   edf_k = tr(H⁻¹ (J'WJ)_kk),   H = J'WJ + S^θ,
        # with J the Jacobian of the fitted values w.r.t. β (obtained by a
        # finite-difference model simulation), σ̂² = RSS/(n − edf), and the
        # update θ_k = σ̂² · edf_k / (β_kᵀ S_k β_k). This replaces the prior
        # fabricated `edf_k = 0.5·(nk−2)` placeholder.
        if m > 0
            ord(M) = Float64[M[i, j] for j in 1:n_obs for i in 1:T_pts]
            μ_pred = try
                simulate(prob, beta)
            catch e
                _is_program_error(e) && rethrow()
                nothing
            end
            if μ_pred === nothing
                if !fs_skip_warned
                    @warn "CollocationLAML: model simulation failed during " *
                          "the Fellner–Schall step; smoothing parameters " *
                          "remain at their initialization and the reported " *
                          "EDF is an upper bound from initialization."
                    fs_skip_warned = true
                end
            else
                y_vec2 = ord(prob.data_values)
                # Zero the weights of masked cells so they drop out of both
                # JWJ and the σ̂² numerator. Weight-aware is not enough on its
                # own: `w * NaN^2 = NaN` even when w = 0, and a NaN σ̂² then
                # survives `clamp` into theta, poisoning the penalty for every
                # subsequent continuation level.
                w_vec2 = ord(prob.data_weights)
                for k in eachindex(y_vec2)
                    _usable(y_vec2[k], w_vec2[k]) || (w_vec2[k] = 0.0;
                                                             y_vec2[k] = 0.0)
                end
                mu_base = ord(μ_pred)
                n_data = length(y_vec2)          # row count for Jb
                # σ̂²'s denominator is a per-observation residual dof and must
                # count USABLE cells; `n_data` counts the masked ones too,
                # biasing σ̂² low and hence the smoothing parameters. Equal to
                # `n_data` for complete data.
                n_eff_cells = count(>(0), w_vec2)
                Jb = zeros(n_data, n_beta)
                # jac=:forwarddiff — one Dual sweep via the shared
                # prediction-Jacobian backend (ord() and the compute_
                # jacobian! flattening are both obs-major, so the row
                # order matches); FD loop below on failure.
                jb_ad_ok = false
                if alg.jac === :forwarddiff
                    jb_ad_ok = _forwarddiff_jacobian!(Jb, prob, beta,
                                                      T_pts, n_obs, fd_cfg)
                    jb_ad_ok ||
                        @debug "CollocationLAML: ForwardDiff Fellner–Schall " *
                               "Jacobian failed or returned non-finite " *
                               "entries; falling back to finite differences"
                end
                if !jb_ad_ok
                    for b in 1:n_beta
                        step = max(1e-6, abs(beta[b]) * 1e-6)
                        bp = copy(beta); bp[b] += step
                        pp = try
                            simulate(prob, bp)
                        catch e
                            _is_program_error(e) && rethrow()
                            μ_pred  # zero FD column on numerical failure
                        end
                        Jb[:, b] .= (ord(pp) .- mu_base) ./ step
                    end
                end
                JWJ = Jb' * Diagonal(w_vec2) * Jb
                S_full = zeros(n_beta, n_beta)
                for l in 1:m
                    idx = (uf_offsets[l]+1):(uf_offsets[l]+uf_nk[l])
                    S_full[idx, idx] .+= theta[l] .* S_list[l]
                end
                H = JWJ + S_full
                maxd = maximum(abs.(diag(H)))
                for i in 1:n_beta; H[i, i] += 1e-10 * (maxd + 1); end
                H_inv = try; inv(cholesky(Symmetric(H))); catch; pinv(H); end
                edf_total = clamp(tr(H_inv * JWJ), 1.0, Float64(n_beta))
                sigma2_est = sum(w_vec2 .* (y_vec2 .- mu_base).^2) /
                             max(n_eff_cells - edf_total, 1.0)
                if alg.sigma2_init !== nothing
                    sigma2_est = min(sigma2_est, alg.sigma2_init)
                end
                edf_final = edf_total
                for l in 1:m
                    idx = (uf_offsets[l]+1):(uf_offsets[l]+uf_nk[l])
                    beta_k = beta[idx]
                    bSb = dot(beta_k, S_list[l] * beta_k)
                    # Fellner–Schall numerator (Wood & Fasiolo 2017):
                    #   rank(S_k) − θ_k·tr(H⁻¹S_k),
                    # matching estimate_smoothing_params in laml.jl. Using the
                    # hat-trace edf_k here instead would include the penalty
                    # null space (rank + nullity − θtr(H⁻¹S)) and bias θ up.
                    r_k = _rank_penalty(S_list[l])
                    trHS = tr(H_inv[idx, idx] * S_list[l])
                    fs_num = min(r_k - theta[l] * trHS, Float64(uf_nk[l]))
                    # Degenerate-update policy — the SAME policy as
                    # `estimate_smoothing_params`; the full rationale lives
                    # beside the sibling update in laml.jl. Summary: adopt
                    # mgcv's direction where the fixed point it names is
                    # FINITE (`fs_num ≤ 0` with β'S_lβ > 0 ⇒ θ* ≤ 0 ⇒ release
                    # by one bounded decade, which the old freeze blocked),
                    # and hold where it is not (β'S_lβ ≈ 0 ⇒ θ* = +∞: the
                    # penalty θ_l·β'S_lβ is zero for every θ_l, so escalating
                    # cannot improve the fit but does wreck the conditioning
                    # of H). Measured on the deterministic linear-map fixture
                    # in the F4 testset, escalating this branch made the
                    # collocation fit WORSE — data loss 1.04e-16 → 2.93e-13
                    # and g(1) error 1.4e-9 → 7.8e-8.
                    #
                    # Two structural facts make the conservative choice even
                    # safer here than in laml.jl: this update runs once per
                    # continuation level (`n_continuation`, default 8) with NO
                    # acceptance test of any kind — the next level's
                    # Gauss–Newton fit simply adopts whatever θ comes out,
                    # whereas the laml.jl proposal is still vetted by the
                    # accept block in solver.jl — and the criterion here
                    # carries an ODE-compliance term the laml.jl one does not,
                    # so a large jump is less well calibrated to begin with.
                    #
                    # Preserve the existing (1e-20, 1e20) cap as the outer
                    # bound, and leave the hold cases bit-identical (θ
                    # untouched, never re-clamped).
                    if bSb > 1e-30
                        if fs_num > 0
                            cand = sigma2_est * fs_num / bSb
                            # mgcv's `r[!is.finite(r)] <- 1e6` hole; we hold.
                            isfinite(cand) &&
                                (theta[l] = clamp(cand, 1e-20, 1e20))
                        else
                            theta[l] = max(theta[l] / 10.0, 1e-20)  # θ* ≤ 0
                        end
                    end                     # no finite θ*, or 0/0 ⇒ hold
                end
                if verbose
                    println("  FS update: σ²=$(round(sigma2_est, sigdigits=4)) " *
                            "edf=$(round(edf_total, sigdigits=3)) θ=$(round.(theta, sigdigits=3))")
                end
            end
        end
    end

    # Collapse onto the θ that β was actually fitted at (see the theta_fit
    # comment above). Both θ-dependent quantities below — `obj_val`'s penalty
    # term and `smoothing_params` itself — read `theta`, so this single
    # assignment is what makes the reported solution self-consistent: β̂ is the
    # penalized optimum at the λ̂ reported beside it, and that λ̂ is the one
    # `edf` was computed at.
    #
    # Reporting rather than refitting mirrors the same fix in `solver.jl`: the
    # final Fellner–Schall proposal exists only to seed the NEXT continuation
    # level and there is no next level, so no coefficient step was ever taken
    # under it. Refitting β to it would change the fit itself, which is a
    # behavior change rather than a reporting fix. A truncated fit
    # (`converged == false`) is still not stationary at ANY λ — no choice of
    # reported λ can fix that — but the λ/β PAIRING is now honest in all cases.
    theta .= theta_fit

    # ─── Build solution ───────────────────────────────────────────
    # Compute final predictions (alpha at observed states)
    pred = zeros(T_pts, n_obs)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        pred[:, j] .= alpha[:, sk]
    end

    # Compute data loss
    data_loss = weighted_data_loss(prob, pred)

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

    # Effective degrees of freedom from the Fellner–Schall step.
    edf = edf_final

    # Build parameter ComponentArray
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))

    obj_val = data_loss + sum((theta[l] * dot(beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]],
                  S_list[l] * beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]]) for l in 1:m);
                  init=0.0)   # m == 0 (all approximators unpenalized) is valid

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "ode_compliance=$(round(ode_loss, sigdigits=5)) " *
                "EDF=$(round(edf, digits=2))")
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (ode_compliance=ode_loss, lambda_ode_final=alg.lambda_ode_end,
                 converged=conv_converged, iterations=conv_iters,
                 reason=conv_reason, iterations_total=conv_iters_total))
end
