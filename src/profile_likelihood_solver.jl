# ─── Profile likelihood solver ─────────────────────────────────────
#
# For identifiability analysis and confidence interval construction.
# For each parameter βⱼ, sweeps it over a grid while optimising all
# other parameters at each grid point.  Returns likelihood-ratio CIs.
#
# Reference: Simpson & Maclaren (2023), PLOS Comp Biol
#            Raue et al. (2009), Bioinformatics

using LinearAlgebra: dot, norm

function _profile_laml_objective(prob::PSMProblem, β_full::AbstractVector;
                                 theta_init::Union{Nothing, Vector{Float64}}=nothing)
    n_times = length(prob.data_times)
    n_obs = size(prob.data_values, 2)
    n_data = n_times * n_obs
    n_p = length(β_full)

    pred = try
        simulate(prob, β_full)
    catch
        return (objective=Inf, theta=Float64[], H=nothing, sigma2=NaN)
    end

    y_vec = zeros(n_data)
    w_vec = zeros(n_data)
    f_vec = zeros(n_data)
    k = 1
    for oi in 1:n_obs, ti in 1:n_times
        y_vec[k] = prob.data_values[ti, oi]
        w_vec[k] = prob.data_weights[ti, oi]
        f_vec[k] = pred[ti, oi]
        k += 1
    end

    J = zeros(n_data, n_p)
    dam = fill(1e-8, n_p)
    compute_jacobian!(J, prob, β_full, f_vec, n_times, n_obs; dam=dam)

    penalties, offsets, _ = _build_penalty_info(prob)
    nknots_list = [size(S, 1) for S in penalties]
    w_irls = irls_weights(prob.likelihood, y_vec, f_vec, w_vec)

    theta = if isempty(penalties)
        Float64[]
    else
        rho_init = theta_init === nothing ? nothing : log.(max.(theta_init, 1e-20))
        theta_est, _ = estimate_smoothing_params(
            J, w_irls, w_vec, y_vec, f_vec, β_full,
            penalties, offsets, nknots_list, n_p;
            family=prob.likelihood, rho_init=rho_init, verbose=false)
        theta_est
    end
    rho = isempty(theta) ? Float64[] : log.(max.(theta, 1e-20))
    V, H, _, sigma2 = laml_objective(prob.likelihood, β_full, J, w_irls, w_vec,
                                     y_vec, f_vec, penalties, offsets, nknots_list,
                                     rho, n_p)
    (objective=-V, theta=theta, H=H, sigma2=sigma2)
end

"""
    solve(prob::PSMProblem, alg::ProfileLikelihoodSolver)

Compute profile likelihoods for unknown-function parameters.

For each profiled parameter index j, sweeps βⱼ over a grid centred on
the best-fit value while re-optimising β₋ⱼ at each grid point.  The
profile likelihood curve is used to compute likelihood-ratio confidence
intervals at the specified level.

First runs a full LAML fit to obtain the MLE, then profiles each
requested parameter.

# Returns
`PSMSolution` where `convergence` contains `:profiles` (a Dict mapping
parameter index → NamedTuple of grid values, profile likelihoods, and CI).
"""
function SciMLBase.solve(prob::PSMProblem, alg::ProfileLikelihoodSolver)
    _validate_problem(prob, "ProfileLikelihoodSolver")
    verbose = alg.verbose

    # ── Step 1: Full LAML fit for MLE ────────────────────────────
    if verbose; println("ProfileLikelihoodSolver: Running initial LAML fit..."); end
    base_sol = SciMLBase.solve(prob, LAML(verbose=false))
    beta_mle = Float64.(collect(base_sol.parameters))
    n_beta = length(beta_mle)
    base_profile = _profile_laml_objective(prob, beta_mle;
                                           theta_init=Float64.(base_sol.smoothing_params))
    mle_obj = base_profile.objective

    if verbose
        println("  MLE objective = $(round(mle_obj, sigdigits=6)), $n_beta parameters")
    end

    # Determine which parameters to profile
    indices = alg.param_indices
    if indices === nothing
        indices = collect(1:n_beta)
    end
    indices = [i for i in indices if 1 <= i <= n_beta]

    if verbose; println("  Profiling $(length(indices)) parameters..."); end

    # ── Step 2: Profile each parameter ───────────────────────────
    chi2_threshold = if alg.ci_level >= 0.99
        6.635  # χ²(1, 0.99)
    elseif alg.ci_level >= 0.95
        3.841  # χ²(1, 0.95)
    elseif alg.ci_level >= 0.90
        2.706  # χ²(1, 0.90)
    else
        3.841
    end

    profiles = Dict{Int, NamedTuple}()

    # ── Helper: compute objective for full beta vector ───────────
    function _profile_objective(prob, β_full; theta_init=nothing)
        _profile_laml_objective(prob, β_full; theta_init=theta_init).objective
    end

    # ── Estimate per-parameter scale via finite-difference Hessian diagonal ──
    mle_obj_val = _profile_objective(prob, beta_mle; theta_init=Float64.(base_sol.smoothing_params))
    hess_diag = zeros(n_beta)
    for j in 1:n_beta
        h = max(abs(beta_mle[j]) * 1e-4, 1e-5)
        bp = copy(beta_mle); bp[j] += h
        bm = copy(beta_mle); bm[j] -= h
        fp = _profile_objective(prob, bp; theta_init=Float64.(base_sol.smoothing_params))
        fm = _profile_objective(prob, bm; theta_init=Float64.(base_sol.smoothing_params))
        hess_diag[j] = (fp - 2*mle_obj_val + fm) / h^2
    end

    if verbose
        println("  Hessian diagonal: ", [round(h, sigdigits=3) for h in hess_diag])
    end

    for idx in indices
        if verbose; println("  Profiling parameter $idx (MLE=$(round(beta_mle[idx], sigdigits=4)))..."); end

        # Scale from Hessian: σ ≈ 1/√(H_jj), then grid = MLE ± 4σ
        if hess_diag[idx] > 1e-8
            sigma_j = 1.0 / sqrt(hess_diag[idx])
        else
            sigma_j = max(abs(beta_mle[idx]) * 0.3, 0.1)
        end
        half_width = 4.0 * sigma_j
        grid = collect(range(beta_mle[idx] - half_width, beta_mle[idx] + half_width,
                             length=alg.n_profile_points))

        profile_obj = fill(Inf, alg.n_profile_points)
        profile_beta = Vector{Vector{Float64}}(undef, alg.n_profile_points)

        other_idx = [i for i in 1:n_beta if i != idx]

        # Find the grid index closest to MLE (start point)
        mle_gi = argmin(abs.(grid .- beta_mle[idx]))

        # Profile loss for a given fixed value of parameter idx
        function make_profile_loss(beta_template, idx_fixed, other_idx_list, theta_start)
            function profile_loss(β_other_eval)
                β_full = copy(beta_template)
                for (oi, ov) in enumerate(other_idx_list)
                    β_full[ov] = β_other_eval[oi]
                end
                _profile_objective(prob, β_full; theta_init=theta_start)
            end
            profile_loss
        end

        # Evaluate at a single grid point, warm-started from beta_warm
        function _eval_grid_point!(gi, gval, beta_warm, theta_warm)
            beta_fixed = copy(beta_warm)
            beta_fixed[idx] = gval
            beta_other = beta_fixed[other_idx]

            if length(beta_other) > 0
                ploss = make_profile_loss(beta_fixed, idx, other_idx, theta_warm)
                opt_result = Optim.optimize(ploss, beta_other,
                                            Optim.NelderMead(),
                                            Optim.Options(iterations=500, show_trace=false))
                beta_other_opt = Optim.minimizer(opt_result)

                β_full = copy(beta_fixed)
                for (oi, ov) in enumerate(other_idx)
                    β_full[ov] = beta_other_opt[oi]
                end
                profile_eval = _profile_laml_objective(prob, β_full; theta_init=theta_warm)
                profile_obj[gi] = profile_eval.objective
                profile_beta[gi] = β_full
                return β_full, profile_eval.theta
            else
                profile_eval = _profile_laml_objective(prob, beta_fixed; theta_init=theta_warm)
                profile_obj[gi] = profile_eval.objective
                profile_beta[gi] = copy(beta_fixed)
                return beta_fixed, profile_eval.theta
            end
        end

        # Evaluate at MLE grid point first
        beta_warm_r, theta_warm_r = _eval_grid_point!(mle_gi, grid[mle_gi], beta_mle,
                                                      Float64.(base_sol.smoothing_params))

        # Sweep RIGHT from MLE
        beta_warm_right = copy(beta_warm_r)
        theta_warm_right = copy(theta_warm_r)
        for gi in (mle_gi+1):alg.n_profile_points
            beta_warm_right, theta_warm_right = _eval_grid_point!(gi, grid[gi], beta_warm_right, theta_warm_right)
        end

        # Sweep LEFT from MLE
        beta_warm_left = copy(beta_warm_r)
        theta_warm_left = copy(theta_warm_r)
        for gi in (mle_gi-1):-1:1
            beta_warm_left, theta_warm_left = _eval_grid_point!(gi, grid[gi], beta_warm_left, theta_warm_left)
        end

        # Compute profile likelihood ratio and CI
        min_obj = minimum(profile_obj)
        plr = 2.0 .* (profile_obj .- min_obj)  # profile likelihood ratio

        # Find CI: largest interval where PLR < threshold
        in_ci = plr .< chi2_threshold
        ci_lo = in_ci[1] ? grid[1] : grid[findfirst(in_ci)]
        ci_hi = in_ci[end] ? grid[end] : grid[findlast(in_ci)]

        profiles[idx] = (grid=grid, objective=profile_obj, plr=plr,
                         ci=(ci_lo, ci_hi), threshold=chi2_threshold)

        if verbose
            println("    CI: [$(round(ci_lo, sigdigits=4)), $(round(ci_hi, sigdigits=4))]")
        end
    end

    # Return solution with profiles in convergence field
    PSMSolution(base_sol.parameters, mle_obj, base_sol.data_loss,
                base_sol.edf, base_sol.smoothing_params,
                base_sol.fitted_values, base_sol.data_values,
                base_sol.data_times, base_sol.unknown_functions,
                (converged=true, method=:profile_likelihood,
                 profiles=profiles, mle_objective=mle_obj))
end
