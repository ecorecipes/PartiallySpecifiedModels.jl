# ─── Gradient matching solver ─────────────────────────────────────
#
# Two-step approach inspired by NODEBNGM (Bonnaffé et al.):
#   Step 1: Smooth observed data with cubic splines to get ŷ(t) and dŷ/dt
#   Step 2: Fit unknown functions by matching ODE derivatives:
#           minimize ||dŷ/dt - f(ŷ, p, t)||²
#
# Key advantage: no ODE integration in the optimization loop, making it
# far more robust for neural network approximators.
#
# Reference: Bonnaffé & Coulson (2023), Methods in Ecology and Evolution

using LinearAlgebra
using Statistics

# ─── Step 1: Data smoothing ──────────────────────────────────────

"""
    _smoothing_spline(t, y; max_basis=15) -> (value, derivative)

Fit a penalized cubic regression spline (P-spline: cubic B-spline basis +
second-order difference penalty) to `(t, y)` with the smoothing parameter
chosen by Generalized Cross-Validation, and return callables for the fitted
value and its first derivative. The basis size is `clamp(n − 2, 4, max_basis)`
(`max_basis ≥ 4`; `TwoStageSolver` wires its `n_basis_smooth` field here).

This is a genuine SMOOTHER (it does not interpolate the noisy data), which
is what gradient matching requires: differentiating an interpolant amplifies
observation noise, whereas the penalized fit suppresses it (Wood 2001;
Varah 1982). Shared by the gradient-matching, two-stage, and BNG solvers.
"""
function _smoothing_spline(t::AbstractVector{Float64}, y::AbstractVector{Float64};
                           max_basis::Int=15)
    n = length(t)
    a, b = minimum(t), maximum(t)
    if b <= a || n < 4
        ȳ = sum(y) / max(n, 1)
        return (x -> ȳ), (x -> 0.0)
    end
    q = clamp(n - 2, 4, max(max_basis, 4))      # number of B-spline coefficients
    knots = _scam_knot_vector((a, b), q)
    B = zeros(n, q)
    for i in 1:n
        B[i, :] = _bspline_basis_vector(t[i], knots, 4)
    end
    # Second-order difference penalty D'D.
    D = zeros(q - 2, q)
    for i in 1:(q - 2)
        D[i, i] = 1.0; D[i, i+1] = -2.0; D[i, i+2] = 1.0
    end
    P = D' * D
    BtB = B' * B; Bty = B' * y
    best_gcv = Inf; β = BtB \ Bty
    for logλ in range(-6.0, 6.0, length=40)
        λ = 10.0^logλ
        F = cholesky(Symmetric(BtB + λ * P + 1e-10 * I), check=false)
        issuccess(F) || continue
        βλ = F \ Bty
        resid = y - B * βλ
        trH = tr(B * (F \ B'))
        denom = (n - trH)^2
        gcv = denom > 1e-8 ? n * sum(abs2, resid) / denom : Inf
        if gcv < best_gcv
            best_gcv = gcv; β = βλ
        end
    end
    h = (b - a) * 1e-6
    value = x -> dot(_bspline_basis_vector(clamp(Float64(x), a, b), knots, 4), β)
    deriv = x -> begin
        xc = clamp(Float64(x), a + h, b - h)
        (dot(_bspline_basis_vector(xc + h, knots, 4), β) -
         dot(_bspline_basis_vector(xc - h, knots, 4), β)) / (2h)
    end
    value, deriv
end

"""
    _smoothing_spline_masked(t, y, w; max_basis=15)

`_smoothing_spline` restricted to the USABLE rows of `(y, w)` — those with
positive weight and a non-NaN value.

Masked rows must be DROPPED from the fit, not merely down-weighted after
it. `_smoothing_spline` solves the normal equations `BtB \\ Bty` with
`Bty = B'y`; a single NaN in `y` makes every coefficient NaN, and because
`gcv < best_gcv` is false for a NaN gcv at every λ, the λ loop never
replaces that NaN β — so BOTH returned callables evaluate to NaN
everywhere, for every input. Every gradient-matching-family solver then
matches derivatives against an all-NaN target.

The returned callables clamp to the usable range, so evaluating them on
the full time grid extends the edge values across masked spans — the
right behavior for interior gaps and honest at the ends.

When every row is usable this delegates unchanged, so complete-data fits
are bit-for-bit identical to calling `_smoothing_spline` directly.
"""
function _smoothing_spline_masked(t::AbstractVector{Float64},
                                  y::AbstractVector{Float64},
                                  w::Union{Nothing,AbstractVector}=nothing;
                                  max_basis::Int=15)
    keep = if w === nothing
        [i for i in eachindex(y) if isfinite(y[i])]
    else
        [i for i in eachindex(y) if _usable(y[i], w[i])]
    end
    length(keep) == length(y) && return _smoothing_spline(t, y; max_basis=max_basis)
    isempty(keep) && error("_smoothing_spline: an observation column is " *
        "entirely masked (every weight is 0 or every value is NaN); there " *
        "is nothing to smooth.")
    _smoothing_spline(Float64.(t[keep]), Float64.(y[keep]); max_basis=max_basis)
end

"""
Smooth observed data with a penalized (GCV) smoothing spline and compute
time derivatives — see [`_smoothing_spline`](@ref).

`weights`, when supplied, is the `data_weights` matrix; masked cells are
dropped from each column's fit (see [`_smoothing_spline_masked`](@ref)).

Returns:
- `y_smooth`: smoothed state values (n_times × K)
- `dydt`: time derivatives from the smoother (n_times × K)
"""
function smooth_and_differentiate(times::Vector{Float64},
                                  data::Matrix{Float64},
                                  obs_to_state::Vector{Int},
                                  K::Int;
                                  weights::Union{Nothing,AbstractMatrix}=nothing)
    T = length(times)
    n_obs = size(data, 2)
    y_smooth = zeros(T, K)
    dydt = zeros(T, K)

    # KNOWN LIMITATION: when several observation columns map to the same
    # state, each later column overwrites the earlier one's smooth below —
    # only the LAST column mapped to a state is used (no averaging).
    if length(unique(@view obs_to_state[1:n_obs])) < n_obs
        @warn "smooth_and_differentiate: multiple observation columns map to " *
              "the same state; only the last column per state is used in the " *
              "smoothing path (earlier columns are ignored, not averaged)." maxlog=1
    end

    for j in 1:n_obs
        sk = obs_to_state[j]
        val, der = _smoothing_spline_masked(times, data[:, j],
                                            weights === nothing ? nothing :
                                            @view(weights[:, j]))
        for i in 1:T
            y_smooth[i, sk] = val(times[i])
            dydt[i, sk] = der(times[i])
        end
    end

    # For unobserved states, leave as zero (user must observe all states
    # for gradient matching to work)
    y_smooth, dydt
end

# ─── Step 2: Derivative matching ─────────────────────────────────

"""
Evaluate ODE RHS at all time points using smoothed state values.

Returns `(F, n_failed)` where `F` is the (T × K) matrix of f(ŷ(t), p, t)
and `n_failed` counts the time points whose evaluation raised and
therefore hold the `1e6` failure sentinel rather than a real derivative.
Same `(values, failure-count)` shape as `eval_ode_rhs_masked`
(collocation_solver.jl), and for the same reason: a caller that cannot
see how much of `F` is fictitious cannot report an honest convergence
verdict (see [`_dynamics_failure_verdict`](@ref)).
"""
function eval_rhs_at_smooth(prob::PSMProblem, times::Vector{Float64},
                            y_smooth::Matrix{Float64}, beta::Vector{Float64})
    T, K = size(y_smooth)
    F = zeros(T, K)
    p = build_param_struct(prob, beta)
    du = zeros(K)
    n_failed = 0

    for i in 1:T
        u = y_smooth[i, :]
        try
            prob.dynamics!(du, u, p, times[i])
        catch e
            _is_program_error(e) && rethrow()
            du .= 1e6
            n_failed += 1
        end
        F[i, :] .= du
    end
    F, n_failed
end

"""
Compute gradient-matching residual and Jacobian.

Residual: r[i,k] = √w × (dŷ_k/dt(t_i) - f_k(ŷ(t_i), p, t_i))
Jacobian: ∂r/∂β via finite differences
"""
function gm_residual_jacobian(prob::PSMProblem, times::Vector{Float64},
                              y_smooth::Matrix{Float64}, dydt::Matrix{Float64},
                              beta::Vector{Float64}, w::Vector{Float64})
    T, K = size(y_smooth)
    n_beta = length(beta)
    n_match = prob.discrete ? T - 1 : T

    F, _ = eval_rhs_at_smooth(prob, times, y_smooth, beta)

    # Residual: √w × (dydt - F)
    resid = zeros(n_match * K)
    for k in 1:K, i in 1:n_match
        idx = (k - 1) * n_match + i
        wi = idx <= length(w) ? sqrt(w[idx]) : 1.0
        resid[idx] = wi * (dydt[i, k] - F[i, k])
    end

    # Jacobian via FD
    J = zeros(n_match * K, n_beta)
    eps = 1e-5
    for b in 1:n_beta
        beta_p = copy(beta)
        step = max(eps, abs(beta[b]) * eps)
        beta_p[b] += step
        F_p, _ = eval_rhs_at_smooth(prob, times, y_smooth, beta_p)
        for k in 1:K, i in 1:n_match
            idx = (k - 1) * n_match + i
            wi = idx <= length(w) ? sqrt(w[idx]) : 1.0
            J[idx, b] = -wi * (F_p[i, k] - F[i, k]) / step
        end
    end

    resid, J
end

# ─── Main solver ─────────────────────────────────────────────────

"""
    solve(prob::PSMProblem, alg::GradientMatching)

Fit a partially specified model using the smooth-then-match approach.
Observed data are first smoothed to estimate state trajectories and their
derivatives; the unknown-function parameters are then found by matching
the smoothed derivatives to the model right-hand side.

# Algorithm
1. Smooth each observed state with cubic splines (continuous) or compute
   forward differences (discrete).
2. Build a gradient-matching objective: ∑ₜ ‖x′(t) − f(x(t), uf(t; β))‖².
3. Minimise with `Optim.NelderMead` (or user-specified method).
4. Reconstruct the unknown functions at the fitted parameters.

# References
- Varah (1982), "A Spline Least Squares Method for Numerical Parameter
  Estimation in Differential Equations", SIAM J. Sci. Stat. Comput.

# Returns
`PSMSolution` with fitted parameters, trajectory, and unknown functions.
"""
function SciMLBase.solve(prob::PSMProblem, alg::GradientMatching)
    _validate_problem(prob, "GradientMatching"; reject_delays=true)
    times = Float64.(prob.data_times)
    T_pts = length(times)
    K = length(prob.u0)
    n_obs = size(prob.data_values, 2)

    verbose = alg.verbose

    # Step 1: Smooth data and compute derivatives / forward differences
    if verbose; println("Step 1: Smoothing data and computing derivatives..."); end

    if prob.discrete
        # For discrete models, smooth data first (just like continuous) then
        # use smoothed next-state values as matching targets.
        # Without smoothing, noisy data→noisy targets produces poor recovery.
        # (An interpolating spline evaluated at its own knots returns the raw
        # data — a no-op; use the genuine penalized GCV smoother.)
        y_smooth = zeros(T_pts, K)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            sval, _ = _smoothing_spline_masked(times,
                                               Float64.(prob.data_values[:, j]),
                                               @view(prob.data_weights[:, j]))
            for i in 1:T_pts
                y_smooth[i, sk] = sval(times[i])
            end
        end
        # Target: smoothed next-state value u[t+1] = f(u[t], p, t)
        dydt = zeros(T_pts, K)
        for k in 1:K
            for i in 1:(T_pts-1)
                dydt[i, k] = y_smooth[i+1, k]
            end
            dydt[T_pts, k] = y_smooth[T_pts, k]  # unused; last point has no forward diff
        end
    else
        y_smooth, dydt = smooth_and_differentiate(times, Float64.(prob.data_values),
                                                   prob.obs_to_state, K;
                                                   weights=prob.data_weights)
    end

    # Initialize unknown function parameters
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)

    # Build penalty matrices
    S_list, uf_offsets, uf_nk = build_penalty_matrices(prob)
    m = length(S_list)

    # Smoothing parameters
    theta = if m > 0
        Float64[1.0 / max(tr(S_list[l]), 1e-10) for l in 1:m]
    else
        Float64[]
    end

    # Per-state weights for Gauss-Newton: normalize so each equation contributes
    # equally. Without this, states with large |dy/dt| dominate the penalty
    # update. For Adam (NNs), uniform weights work better since the NN
    # architecture already provides implicit regularization.
    n_match = prob.discrete ? T_pts - 1 : T_pts
    w = ones(n_match * K)
    if m > 0  # only weight for penalized (Gauss-Newton) branch
        for k in 1:K
            dydt_k = @view dydt[1:n_match, k]
            scale_k = max(std(dydt_k), 1e-10)
            for i in 1:n_match
                idx = (k - 1) * n_match + i
                w[idx] = 1.0 / scale_k^2
            end
        end
    end
    # Unobserved states carry fabricated targets (constant state, zero
    # derivative); including them would push the unknown functions to zero
    # the RHS along a fictitious trajectory. Zero their weights so both the
    # loss and the Gauss-Newton residuals ignore them.
    obs_set = Set(prob.obs_to_state)
    for k in 1:K
        k in obs_set && continue
        for i in 1:n_match
            w[(k - 1) * n_match + i] = 0.0
        end
    end
    # Masked observations carry fabricated targets for the same reason: the
    # smoother now excludes them (see `_smoothing_spline_masked`), so
    # `y_smooth`/`dydt` at a masked time are an EXTRAPOLATION, not data.
    # Zeroing their weight keeps them out of the loss, the Gauss-Newton
    # residuals, and — via `n_eff = count(!iszero, w)` below — out of the σ̂²
    # denominator that drives the smoothing-parameter update. A match point
    # counts if ANY observation column mapping to that state is usable there.
    # No-op when every cell is usable, so complete-data fits are unchanged.
    for k in 1:K
        k in obs_set || continue
        cols = [j for j in 1:n_obs if prob.obs_to_state[j] == k]
        for i in 1:n_match
            any(j -> usable_cell(prob, i, j), cols) && continue
            w[(k - 1) * n_match + i] = 0.0
        end
    end

    if verbose
        println("Step 2: Gradient matching — $(n_beta) params, $(n_match) match points, $(K) states")
        println("  Penalty terms: $m, optimizer: $(m > 0 ? "Gauss-Newton" : "Adam")")
    end

    # Loss function for derivative/map matching (with per-state weights)
    function gm_loss(β_eval)
        F, _ = eval_rhs_at_smooth(prob, times, y_smooth, β_eval)
        loss_val = 0.0
        for k in 1:K, i in 1:n_match
            idx = (k - 1) * n_match + i
            loss_val += w[idx] * (dydt[i, k] - F[i, k])^2
        end
        # Add penalty
        if m > 0
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                beta_k = β_eval[off+1:off+nk]
                loss_val += theta[l] * dot(beta_k, S_list[l] * beta_k)
            end
        end
        loss_val
    end

    # Honest convergence reporting, same taxonomy as the sibling
    # gradient-matching solvers: `converged` only when a criterion actually
    # fired; otherwise the loop exhausted its iteration budget.
    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0

    if m > 0
        # ─── Gauss-Newton for penalized approximators (B-spline, GP) ───
        prev_obj = Inf
        for iter in 1:alg.maxiters
            conv_iters = iter
            resid, J = gm_residual_jacobian(prob, times, y_smooth, dydt, beta, w)

            B = zeros(n_beta, n_beta)
            for l in 1:m
                off = uf_offsets[l]
                nk = uf_nk[l]
                B[off+1:off+nk, off+1:off+nk] .+= theta[l] .* S_list[l]
            end

            resid_ss = sum(resid.^2)
            pen_ss = dot(beta, B * beta)
            obj = resid_ss + pen_ss

            if verbose && (iter <= 3 || iter % 10 == 0)
                println("  iter $iter: deriv_SS=$(round(resid_ss, sigdigits=5)) pen=$(round(pen_ss, sigdigits=4))")
            end

            if iter > 1 && abs(obj - prev_obj) < alg.tol * max(abs(prev_obj), 1.0)
                if verbose; println("  Converged at iter $iter"); end
                conv_converged = true
                conv_reason = :objective_tol
                break
            end
            prev_obj = obj

            JtJ = J' * J + B
            neg_Jtr = -(J' * resid)
            delta = try
                JtJ \ neg_Jtr
            catch e
                _is_program_error(e) && rethrow()
                try
                    (JtJ + 1e-6 * I) \ neg_Jtr
                catch e2
                    _is_program_error(e2) && rethrow()
                    # Even the ridged normal equations are unsolvable — a
                    # numerical failure, not a converged fit.
                    conv_reason = :singular_system
                    break
                end
            end

            best_obj = obj; best_step = 0.0
            for k in 0:8
                ss = 0.5^k
                beta_new = beta .+ ss .* delta
                obj_new = gm_loss(beta_new)
                if obj_new < best_obj; best_obj = obj_new; best_step = ss; end
            end
            if best_step == 0.0
                if verbose; println("  No improvement, stopping"); end
                # No step of any length improved the objective. That is a
                # stalled line search, not a satisfied criterion.
                conv_reason = :line_search_failure
                break
            end
            beta .= beta .+ best_step .* delta

            # Update smoothing params
            if iter % 5 == 0
                # Divide by the number of residuals actually contributing:
                # unobserved-state rows are zero-weighted, and counting them
                # would bias sigma2 low (over-smoothing theta).
                n_eff = count(!iszero, w)
                sigma2 = resid_ss / max(n_eff, 1)
                if alg.sigma2_init !== nothing; sigma2 = min(sigma2, alg.sigma2_init); end
                # Fellner–Schall smoothing update (Wood & Fasiolo 2017), in
                # PARITY with `estimate_smoothing_params` (laml.jl) and the
                # collocation sibling (collocation_solver.jl). The numerator
                # is
                #     rank(S_l) − θ_l·tr(H⁻¹S_l)
                # with H = J'J + B the Gauss–Newton Hessian already formed
                # above.
                #
                # This replaces `max(nk - 2, 1)`, a hard-coded GUESS at "the
                # rank of a second-difference penalty on nk coefficients".
                # That guess is exactly right for a nullity-2 penalty
                # (B-spline, GP) and wrong for every other nullity: an SPDE
                # penalty is full rank, so on a 9-coefficient block the guess
                # said 7 where the rank is 9 and the reported θ̂ was off by
                # precisely 9/7 = 1.2857×; a convex-constrained B-spline block
                # of 8 has rank 5, not 6; a ridge block is full rank. Since
                # the update is a single multiplicative fixed point, the error
                # passes straight into the user-visible
                # `sol.smoothing_params`. `_rank_penalty` reads the rank off
                # the matrix instead of assuming it, which is what the other
                # four Fellner–Schall sites in this package already do —
                # GradientMatching was the only fabricated rank left in src/.
                #
                # The `−θ·tr(H⁻¹S)` term was absent entirely, making this the
                # τ = 0 limit of Fellner–Schall rather than the method itself.
                # `_safe_inv` (laml.jl) rather than an inline ridge, so the
                # parity with the LAML sibling is LITERAL and not merely
                # equivalent-looking: it is the same Cholesky-with-fallback
                # every other Fellner–Schall site inverts its Hessian with.
                H_inv = _safe_inv(JtJ)
                for l in 1:m
                    off = uf_offsets[l]; nk = uf_nk[l]
                    idx = (off + 1):(off + nk)
                    bSb = dot(beta[idx], S_list[l] * beta[idx])
                    fs_num = _rank_penalty(S_list[l]) -
                             theta[l] * tr(H_inv[idx, idx] * S_list[l])
                    # Degenerate-update policy, verbatim from the LAML and
                    # collocation siblings: adopt mgcv's direction where the
                    # fixed point it names is FINITE (fs_num ≤ 0 with
                    # β'S_lβ > 0 ⇒ θ* ≤ 0 ⇒ release by one bounded decade),
                    # and HOLD where it is not (β'S_lβ ≈ 0 ⇒ θ* = +∞).
                    if bSb > 1e-30
                        if fs_num > 0
                            cand = sigma2 * fs_num / bSb
                            # mgcv's `r[!is.finite(r)] <- 1e6` hole; we hold,
                            # as laml.jl and collocation_solver.jl do. A NaN
                            # θ would pass every subsequent comparison.
                            isfinite(cand) &&
                                (theta[l] = clamp(cand, 1e-20, 1e20))
                        else
                            theta[l] = max(theta[l] / 10.0, 1e-20)  # θ* ≤ 0
                        end
                    end                     # no finite θ*, or 0/0 ⇒ hold
                end
            end
        end
    else
        # ─── Adam optimizer for unpenalized approximators (NN) ─────────
        # Use ForwardDiff for exact gradients (avoids Float32 precision loss
        # from Lux evaluation that plagues finite-difference gradients)
        function gm_loss_ad(β_eval)
            p = build_autodiff_param_struct(prob, β_eval)
            T_pts_loc = size(y_smooth, 1)
            du = zeros(eltype(β_eval), K)
            loss_val = zero(eltype(β_eval))
            for i in 1:n_match
                u = eltype(β_eval).(y_smooth[i, :])
                try
                    prob.dynamics!(du, u, p, times[i])
                catch e
                    _is_program_error(e) && rethrow()
                    du .= eltype(β_eval)(1e6)
                end
                for k in 1:K
                    # w carries the observed-state mask (zeroed for
                    # unobserved states whose targets are fabricated)
                    wi = w[(k - 1) * n_match + i]
                    loss_val += wi * (dydt[i, k] - du[k])^2
                end
            end
            loss_val
        end

        # Adam state
        lr = alg.lr
        β1_adam, β2_adam, eps_adam = 0.9, 0.999, 1e-8
        m_adam = zeros(n_beta)
        v_adam = zeros(n_beta)
        best_beta = copy(beta)
        best_loss = Inf
        loss_window = fill(Inf, 20)

        for iter in 1:alg.maxiters
            conv_iters = iter
            grad = ForwardDiff.gradient(gm_loss_ad, beta)
            loss_val = gm_loss_ad(beta)

            if loss_val < best_loss
                best_loss = loss_val
                best_beta .= beta
            end
            loss_window[mod1(iter, 20)] = loss_val

            # Cosine learning rate annealing (floor at 10% of initial lr)
            lr_t = lr * (0.1 + 0.45 * (1 + cos(π * iter / alg.maxiters)))

            # Adam update
            m_adam .= β1_adam .* m_adam .+ (1 - β1_adam) .* grad
            v_adam .= β2_adam .* v_adam .+ (1 - β2_adam) .* grad.^2
            m_hat = m_adam ./ (1 - β1_adam^iter)
            v_hat = v_adam ./ (1 - β2_adam^iter)
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

            if verbose && (iter <= 5 || iter % 50 == 0 || iter == alg.maxiters)
                println("  Adam iter $iter: loss=$(round(loss_val, sigdigits=5)) lr=$(round(lr_t, sigdigits=3))")
            end

            # Convergence: loss plateau (relative change < tol over window).
            #
            # `best_loss < 1e9` is AdamSolver's failure-sentinel guard: the
            # dynamics fall back to `du .= 1e6` when the RHS throws, and a
            # run pinned at that sentinel is a stuck solver, not a converged
            # one. AdamSolver explicitly refuses to call that `:plateau`.
            #
            # `lr_t > 0.05 * lr` is the siblings' cosine-schedule guard,
            # written out for consistency even though it is DORMANT here:
            # this schedule has a 10% floor — `lr_t = lr*(0.1 + 0.45*(1 +
            # cos(...)))` never falls below `0.1*lr` — so the clause is
            # always true today. It is kept so the family reads the same
            # way and so lowering that floor cannot silently reintroduce
            # spurious end-of-schedule "convergence".
            if iter > 50 && best_loss < 1e9 && lr_t > 0.05 * lr
                recent_min = minimum(loss_window)
                recent_max = maximum(loss_window)
                if (recent_max - recent_min) / max(abs(recent_min), 1.0) < alg.tol
                    if verbose; println("  Converged at iter $iter (loss plateau)"); end
                    conv_converged = true
                    conv_reason = :plateau
                    break
                end
            end
        end
        beta .= best_beta
        if verbose; println("  Best loss: $(round(best_loss, sigdigits=5))"); end
    end

    # ─── Optional shooting refinement ────────────────────────────
    # GM fits derivatives/maps only; it can miss the true function at regions
    # where |dy/dt| is small.  A few Adam shooting steps through the actual
    # model corrects this by using trajectory-level information.
    if alg.refine_iters > 0
        if verbose; println("\nStep 3: Shooting refinement ($(alg.refine_iters) iters)..."); end

        # Reuse the proven adam_loss_mse (with data weights, correct ODE/discrete setup)
        refine_loss = β_eval -> adam_loss_mse(prob, β_eval)

        # Adam refinement — start with moderate lr, comparable to AdamSolver
        lr_refine = min(alg.lr, 0.01)
        m_r = zeros(n_beta)
        v_r = zeros(n_beta)
        best_r = copy(beta)
        best_rl = refine_loss(beta)

        for iter in 1:alg.refine_iters
            result = DiffResults.MutableDiffResult(0.0, (zeros(n_beta),))
            ForwardDiff.gradient!(result, refine_loss, beta)
            rl = DiffResults.value(result)
            grad_r = DiffResults.gradient(result)

            if rl < best_rl; best_rl = rl; best_r .= beta; end

            lr_t = lr_refine * (0.1 + 0.45 * (1 + cos(π * iter / alg.refine_iters)))
            m_r .= 0.9 .* m_r .+ 0.1 .* grad_r
            v_r .= 0.999 .* v_r .+ 0.001 .* grad_r.^2
            m_hat = m_r ./ (1 - 0.9^iter)
            v_hat = v_r ./ (1 - 0.999^iter)
            beta .-= lr_t .* m_hat ./ (sqrt.(v_hat) .+ 1e-8)

            if verbose && (iter <= 3 || iter % 20 == 0)
                println("  Refine iter $iter: loss=$(round(rl, sigdigits=5)) lr=$(round(lr_t, sigdigits=3))")
            end
        end
        beta .= best_r
        if verbose; println("  Best refine loss: $(round(best_rl, sigdigits=5))"); end
    end

    # Build solution
    # For discrete models, simulate forward to get actual trajectory predictions
    # (instead of using smoothed states which give misleading data_loss=0)
    if prob.discrete
        p_sim = build_param_struct(prob, beta)
        u_sim = Float64.(prob.u0)
        u_next_sim = similar(u_sim)
        sim_states = zeros(T_pts, K)
        sim_states[1, :] .= u_sim
        for step in 1:(T_pts-1)
            try
                prob.dynamics!(u_next_sim, u_sim, p_sim, times[step])
            catch e
                _is_program_error(e) && rethrow()
                u_next_sim .= 1e6
            end
            u_sim = copy(u_next_sim)
            sim_states[step+1, :] .= u_sim
        end
        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[:, j] .= sim_states[:, sk]
        end
    else
        # Continuous: use smoothed states as predictions
        pred = zeros(T_pts, n_obs)
        for j in 1:n_obs
            sk = prob.obs_to_state[j]
            pred[:, j] .= y_smooth[:, sk]
        end
    end

    # Data loss (against original data, not derivatives)
    data_loss = weighted_data_loss(prob, pred)

    # Derivative matching loss
    F_final, n_dyn_fail = eval_rhs_at_smooth(prob, times, y_smooth, beta)
    deriv_loss = sum((dydt .- F_final).^2)

    # Sentinel accounting at the FITTED parameters. `_dynamics_failure_verdict`
    # errors on total failure (nothing was fitted) and downgrades a partial
    # failure to converged=false, reason=:dynamics_failure. Without it, a
    # measured 31% sentinel substitution on a plain ODE was reported as
    # converged=true, reason=:objective_tol.
    conv_converged, conv_reason = _dynamics_failure_verdict(
        "GradientMatching", n_dyn_fail, T_pts, conv_converged, conv_reason)

    # Build evaluators
    p_opt = build_param_struct(prob, beta)
    uf_evals = Dict{Symbol, Any}()
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
    end

    # EDF and parameters
    edf = Float64(n_beta)  # conservative
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        push!(ca_entries, approx.name => beta[offset+1:offset+np])
        offset += np
    end
    params = ComponentArray(NamedTuple(ca_entries))
    pen_ss = m > 0 ? sum(theta[l] * dot(beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]],
                  S_list[l] * beta[uf_offsets[l]+1:uf_offsets[l]+uf_nk[l]]) for l in 1:m) : 0.0
    obj_val = data_loss + pen_ss

    if verbose
        println("\nFinal: data_SS=$(round(data_loss, sigdigits=5)) " *
                "deriv_SS=$(round(deriv_loss, sigdigits=5)) EDF=$(round(edf, digits=1))")
    end

    PSMSolution(params, obj_val, data_loss, edf, copy(theta),
                Float64.(pred), Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (deriv_loss=deriv_loss, method=:gradient_matching,
                 converged=conv_converged, reason=conv_reason,
                 iterations=conv_iters, n_dynamics_failures=n_dyn_fail))
end
