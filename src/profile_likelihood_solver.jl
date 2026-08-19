# ─── Profile likelihood solver ─────────────────────────────────────
#
# For identifiability analysis and confidence interval construction.
# For each parameter βⱼ, sweeps it over a grid while optimising all
# other parameters at each grid point.  Returns likelihood-ratio CIs.
#
# Reference: Simpson & Maclaren (2023), PLOS Comp Biol
#            Raue et al. (2009), Bioinformatics

using LinearAlgebra: dot, norm

"""
    _normal_quantile(p) → Float64

Standard-normal inverse CDF Φ⁻¹(p) via Acklam's rational approximation
(relative error < 1.15e-9). Self-contained — avoids a Distributions
dependency for computing χ² thresholds.
"""
function _normal_quantile(p::Float64)
    (p <= 0.0) && return -Inf
    (p >= 1.0) && return Inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q*q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

"""
    _chisq1_quantile(level) → Float64

Quantile of the χ²₁ distribution at probability `level`, used as the
likelihood-ratio threshold for a pointwise profile confidence interval.
Uses the identity χ²₁ = Z²: q = Φ⁻¹((1+level)/2)².
"""
_chisq1_quantile(level::Float64) = _normal_quantile((1 + level) / 2)^2

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
    # The profile statistic ΔPenSS/σ̂² is the fixed-smoothing Gaussian
    # likelihood ratio; its χ²₁ calibration does not hold for other
    # families, so refuse rather than return wrong CIs.
    prob.likelihood isa Gaussian ||
        error("ProfileLikelihoodSolver supports Gaussian likelihoods only " *
              "(the penalized-SS profile statistic is χ²₁-calibrated for " *
              "Gaussian errors); got $(typeof(prob.likelihood)).")
    verbose = alg.verbose

    # ── Step 1: Full LAML fit for MLE ────────────────────────────
    if verbose; println("ProfileLikelihoodSolver: Running initial LAML fit..."); end
    base_sol = SciMLBase.solve(prob, LAML(verbose=false))
    beta_mle = Float64.(collect(base_sol.parameters))
    n_beta = length(beta_mle)
    mle_obj = base_sol.objective

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
    # Likelihood-ratio threshold = χ²₁ quantile at the requested level
    # (correct for any 0 < ci_level < 1, not just the three tabulated ones).
    chi2_threshold = _chisq1_quantile(alg.ci_level)

    # Number of scalar observations (for the Gaussian profile-likelihood ratio).
    n_obs = length(prob.data_times) * size(prob.data_values, 2)

    profiles = Dict{Int, NamedTuple}()

    # Fitted smoothing parameters and scale from the base LAML fit. The
    # profile is taken through the PENALIZED objective at fixed λ̂: a
    # penalized spline is not identified through its raw RSS (re-optimizing
    # the nuisance coefficients without the penalty can beat the MLE's own
    # RSS, giving negative "PLR"s and nonsense CIs), so the meaningful
    # profile statistic is ΔPenSS/σ̂² ~ χ²₁, conditional on λ̂ — the
    # profile analogue of fixed-smoothing (Bayesian) intervals.
    lam_hat = collect(base_sol.smoothing_params)
    sigma2_hat = max(base_sol.data_loss / max(n_obs - base_sol.edf, 1.0), 1e-12)

    # ── Helper: penalized objective for full beta vector ─────────
    # Route the trajectory through `simulate`, which already handles the
    # ODE, DDE, and discrete-map cases consistently with the LAML fit
    # (the previous hand-rolled version ignored prob.delays — turning DDE
    # profiles into constant 1e10 objectives and full-width CIs — and
    # iterated discrete maps once per observation instead of per step).
    S_list_prof, offs_prof, nks_prof = build_penalty_matrices(prob)
    function _profile_objective(prob, β_full; with_penalty::Bool=true)
        total_loss = 0.0
        try
            pred = simulate(prob, β_full)
            for i in 1:length(prob.data_times)
                for j in 1:size(prob.data_values, 2)
                    total_loss += (prob.data_values[i, j] - pred[i, j])^2
                end
            end
        catch e
            _is_program_error(e) && rethrow()
            return 1e10
        end
        # Penalty at the FITTED smoothing parameters λ̂, over exactly the
        # penalized blocks LAML used (build_penalty_matrices skips
        # unpenalized approximators, e.g. NeuralApproximator with
        # penalty_weight=0, so lam_hat aligns with S_list_prof — indexing
        # by raw approximator position mis-assigned λ̂ for mixed sets).
        if with_penalty
            for k in eachindex(S_list_prof)
                off = offs_prof[k]
                nk = nks_prof[k]
                pk = β_full[off+1:off+nk]
                lam_l = k <= length(lam_hat) ? lam_hat[k] : 1.0
                total_loss += lam_l * dot(pk, S_list_prof[k] * pk)
            end
        end
        total_loss
    end

    # ── Estimate per-parameter scale via finite-difference Hessian diagonal ──
    mle_obj_val = _profile_objective(prob, beta_mle)
    hess_diag = zeros(n_beta)
    for j in 1:n_beta
        h = max(abs(beta_mle[j]) * 1e-4, 1e-5)
        bp = copy(beta_mle); bp[j] += h
        bm = copy(beta_mle); bm[j] -= h
        fp = _profile_objective(prob, bp)
        fm = _profile_objective(prob, bm)
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
        # Insert the exact MLE value: with an even point count the center
        # falls between grid points, and a steep profile can then cross the
        # χ² threshold inside the first grid cell — quantizing the CI to a
        # single off-MLE point.
        if !any(g -> isapprox(g, beta_mle[idx]; atol=1e-12), grid)
            grid = sort(vcat(grid, beta_mle[idx]))
        end
        n_grid = length(grid)

        profile_obj = fill(Inf, n_grid)
        profile_beta = Vector{Vector{Float64}}(undef, n_grid)

        other_idx = [i for i in 1:n_beta if i != idx]

        # Find the grid index closest to MLE (start point)
        mle_gi = argmin(abs.(grid .- beta_mle[idx]))

        # Profile loss for a given fixed value of parameter idx
        function make_profile_loss(beta_template, idx_fixed, other_idx_list)
            function profile_loss(β_other_eval)
                β_full = copy(beta_template)
                for (oi, ov) in enumerate(other_idx_list)
                    β_full[ov] = β_other_eval[oi]
                end
                _profile_objective(prob, β_full)
            end
            profile_loss
        end

        # Evaluate at a single grid point, warm-started from beta_warm
        function _eval_grid_point!(gi, gval, beta_warm)
            beta_fixed = copy(beta_warm)
            beta_fixed[idx] = gval
            beta_other = beta_fixed[other_idx]

            if length(beta_other) > 0
                ploss = make_profile_loss(beta_fixed, idx, other_idx)
                # Long NelderMead over the nuisance coefficients. Gradient
                # methods stall immediately here (finite-difference gradients
                # through an adaptive ODE solve are noisy), and a short
                # NelderMead under-optimizes the nuisances — both inflate the
                # profile away from the MLE, producing anti-conservative
                # (sometimes single-point) CIs that can even exclude the
                # fitted parameter. Warm starts keep the long run affordable.
                opt_result = Optim.optimize(ploss, beta_other,
                                            Optim.NelderMead(),
                                            Optim.Options(iterations=5000,
                                                          f_abstol=1e-10,
                                                          show_trace=false))
                beta_other_opt = Optim.minimizer(opt_result)
                profile_obj[gi] = Optim.minimum(opt_result)

                β_full = copy(beta_fixed)
                for (oi, ov) in enumerate(other_idx)
                    β_full[ov] = beta_other_opt[oi]
                end
                profile_beta[gi] = β_full
                return β_full
            else
                profile_obj[gi] = _profile_objective(prob, beta_fixed)
                profile_beta[gi] = copy(beta_fixed)
                return beta_fixed
            end
        end

        # Evaluate at MLE grid point first
        beta_warm_r = _eval_grid_point!(mle_gi, grid[mle_gi], beta_mle)

        # Sweep RIGHT from MLE
        beta_warm_right = copy(beta_warm_r)
        for gi in (mle_gi+1):n_grid
            beta_warm_right = _eval_grid_point!(gi, grid[gi], beta_warm_right)
        end

        # Sweep LEFT from MLE
        beta_warm_left = copy(beta_warm_r)
        for gi in (mle_gi-1):-1:1
            beta_warm_left = _eval_grid_point!(gi, grid[gi], beta_warm_left)
        end

        # Penalized profile-likelihood-ratio statistic, conditional on λ̂:
        #   PLR(θ) = (PenSS_profile(θ) − PenSS_min) / σ̂²  ~  χ²₁,
        # the fixed-smoothing Gaussian LRT. (Raw-RSS ratios are NOT usable
        # here: the penalized MLE does not minimize raw RSS, so re-optimized
        # neighbors can beat it, giving negative ratios and nonsense CIs.)
        # Anchor the center at the MLE's own penalized objective so
        # PLR(center) = 0 exactly.
        pen_at_mle = _profile_objective(prob, beta_mle; with_penalty=true)
        if isfinite(pen_at_mle) && pen_at_mle < 1e10
            profile_obj[mle_gi] = min(profile_obj[mle_gi], pen_at_mle)
        end
        pen_min = minimum(filter(isfinite, profile_obj))
        plr = [isfinite(o) && o < 1e10 ? (o - pen_min) / sigma2_hat : Inf
               for o in profile_obj]

        # Interpolated threshold crossings: the χ² boundary generally falls
        # between grid points; snapping CI endpoints to grid values both
        # quantizes the interval and can exclude the MLE entirely when the
        # profile is steep relative to the grid spacing.
        function _cross(gL, gR, pL, pR)
            (isfinite(pL) && isfinite(pR) && pR != pL) ?
                gL + (chi2_threshold - pL) / (pR - pL) * (gR - gL) :
                gR
        end

        # Find CI: largest interval where PLR < threshold. An endpoint that
        # sits on the grid boundary means the profile never crossed the
        # threshold on that side — the interval is open (parameter weakly
        # identified or grid too narrow), which callers need to know.
        in_ci = plr .< chi2_threshold
        i1 = findfirst(in_ci)
        i2 = findlast(in_ci)
        open_left = in_ci[1]
        open_right = in_ci[end]
        ci_lo = open_left ? grid[1] :
                _cross(grid[i1-1], grid[i1], plr[i1-1], plr[i1])
        ci_hi = open_right ? grid[end] :
                _cross(grid[i2+1], grid[i2], plr[i2+1], plr[i2])

        profiles[idx] = (grid=grid, objective=profile_obj, plr=plr,
                         ci=(ci_lo, ci_hi), threshold=chi2_threshold,
                         open_left=open_left, open_right=open_right)

        if verbose
            lo_br = open_left ? "(" : "["
            hi_br = open_right ? ")" : "]"
            println("    CI: $(lo_br)$(round(ci_lo, sigdigits=4)), " *
                    "$(round(ci_hi, sigdigits=4))$(hi_br)" *
                    (open_left || open_right ?
                     "  (open endpoint: profile did not cross threshold)" : ""))
        end
    end

    # Return solution with profiles in convergence field
    PSMSolution(base_sol.parameters, base_sol.objective, base_sol.data_loss,
                base_sol.edf, base_sol.smoothing_params,
                base_sol.fitted_values, base_sol.data_values,
                base_sol.data_times, base_sol.unknown_functions,
                (converged=true, method=:profile_likelihood,
                 profiles=profiles, mle_objective=mle_obj))
end
