# ─── FGPGM solver (Fast GP-based gradient matching) ─────────────────
#
# Wenk, Gotovos, Bauer, Gorbach, Krause & Buhmann (2019), AISTATS —
# "Fast Gaussian process based gradient matching for parameter
# identification in systems of nonlinear ODEs".
#
# One product-of-experts density over the latent states AND the
# unknown-function parameters jointly, with the GP hyperparameters fixed
# BEFOREHAND (per-state marginal likelihood — reusing ODIN's
# `optimize_gp_hyperparams`), sampled by a single-chain adaptive
# Metropolis-within-Gibbs sampler:
#
#   p(x, θ | y) ∝ Π_k [ N(y_k | x_k, σ_{n,k}² I)      (data)
#                       · N(x_k | m_k, C_k)           (GP prior, centered)
#                       · N(f_k(x, θ) | D_k x̃_k, A_k + γI) ]   (ODE expert)
#                 · p(θ)                              (smoothing prior)
#
# with x̃_k = x_k − m_k, D_k = 'C_k C_k⁻¹ (GP conditional-mean derivative
# map) and A_k = ''C_k − 'C_k C_k⁻¹ 'C_kᵀ (conditional derivative
# covariance). This sits between AdaptiveGradientMatching (Dondelinger-
# style tempered population MCMC — expensive) and ODINSolver (Wenk's
# later pure-optimisation formulation — no posterior): FGPGM keeps
# genuine posterior samples but drops the temperature ladder and the
# γ/hyperparameter sampling, which is exactly the paper's speedup.
#
# The latent states live at the OBSERVATION times, as in the paper
# (MagiSolver is the sibling that discretizes on a finer grid).
#
# Reference: Wenk et al. (2019), AISTATS 89:1351-1360

using LinearAlgebra: dot, Symmetric, cholesky, I

"""
    _fgpgm_state_matrices(times, σ², ℓ, σn², γ_user) -> (Cinv, L, D, Λinv)

Fixed per-state matrices of the FGPGM density on the observation grid:
the GP prior precision `C⁻¹`, the Cholesky factor `L = chol(C).L` used
for GP-correlated state proposals (as in AGM's population MCMC), the
derivative map `D = 'C C⁻¹`, and the inverse ODE-mismatch covariance
`Λ⁻¹ = (A + γI)⁻¹` with `A = ''C − 'C C⁻¹ 'Cᵀ`.

The effective slack is `γ = γ_user + σn²/ℓ²`: the user's model-mismatch
variance (the paper's γ) plus the derivative-scale noise induced by
observation noise over one lengthscale — the same correction ODIN
applies, without which the ODE expert dwarfs the data expert and the
states abandon the data to satisfy `f = Dx̃` exactly.
"""
function _fgpgm_state_matrices(times::Vector{Float64}, σ²::Float64,
                               ℓ::Float64, σn²::Float64, γ_user::Float64)
    K, dK, d2K = rbf_kernel_with_derivs(times, σ², ℓ)
    # Noiseless RBF Grams are severely ill-conditioned; regularize with a
    # small jitter before factorizing (ODIN/AGM convention).
    jitter = 1e-6 * σ²
    C = cholesky(Symmetric(K + jitter * I))
    Cinv = Matrix(inv(C))
    L = Matrix(C.L)
    D = dK * Cinv
    γ = γ_user + σn² / ℓ^2 + 1e-10
    A = d2K - dK * (C \ dK') + γ * I
    Λinv = Matrix(inv(cholesky(Symmetric(0.5 * (A + A')))))
    Cinv, L, D, Λinv
end

"""
    solve(prob::PSMProblem, alg::FGPGMSolver)

Fit a partially specified model with Fast Gaussian process based gradient
matching (FGPGM; Wenk et al. 2019).

Stage 1 — per observed state, RBF-GP hyperparameters `(σ², ℓ, σ_n²)` are
estimated by maximising the GP marginal likelihood of the (centered)
data, or taken from the solver if both `gp_lengthscale` and `gp_variance`
are supplied. They are then FIXED — the paper's key simplification over
Dondelinger-style adaptive gradient matching. Unobserved states borrow
the mean hyperparameters of the observed states, are centered on their
initial condition, and carry no data term (ODIN precedent): they are
identified through the GP prior and the ODE experts alone.

Stage 2 — the states `X` (at the observation times, as in the paper) and
the unknown-function parameters `θ` are sampled JOINTLY from the single
product-of-experts density

    p(X, θ | y) ∝ Π_k [ N(y_k | x_k, σ_{n,k}² I) · N(x_k | m_k, C_k)
                        · N(f_k(X, θ) | D_k x̃_k, A_k + γI) ] · p(θ)

by one-chain Metropolis-within-Gibbs: one GP-correlated random-walk
block per state (proposal `x_k + s_k · L_k z`, `L_k = chol(C_k).L`) and
one random-walk block for θ. Per-block proposal scales adapt toward
`target_accept` during warmup ONLY and are frozen afterwards, so every
retained draw comes from a fixed Markov kernel. `p(θ)` is the package's
standard smoothing prior (penalty matrices via `_build_penalty_info`,
plus a broad ridge), as in `MCMCSolver`/`MagiSolver`.

Gaussian likelihoods only (the data expert is a Gaussian quadratic form);
non-Gaussian `prob.likelihood` errors at entry. Masked observations
(NaN value and/or zero weight) are dropped from the GP hyperparameter
fit, the state initialization and the data term, mirroring
`ODINSolver`/`AdaptiveGradientMatching`; per-cell `data_weights`
MAGNITUDES are not applied (FGPGM weights by state through σ_{n,k}²) —
only the zero/NaN mask is honored. Continuous-time problems only;
`NeuralApproximator` is rejected (as in AGM's population MCMC) — use
`AdamSolver` for neural unknowns.

# Returns
`PSMSolution` built from the posterior-MEAN θ̄ (`sol.parameters`,
`sol.unknown_functions`) and the posterior-mean state trajectory
(`sol.fitted_values`). `sol.objective` is the NEGATIVE log joint density
evaluated at the posterior means `(X̄, θ̄)` (lower is better, matching
`MCMCSolver`/`MagiSolver`); `sol.data_loss` is the usual mask-aware
`weighted_data_loss`. `sol.convergence` carries `chains`
(an `MCMCChains.Chains` of the θ draws), `beta_samples`, `state_mean`,
`gp_hyperparams`, per-block `accept_rates`, and the honest keys
`converged=false`/`reason=:maxiters`/`iterations` — a fixed-budget
sampler has no stopping criterion, so judge it by the R̂/ESS of
`convergence.chains` and the acceptance rates.
"""
function SciMLBase.solve(prob::PSMProblem, alg::FGPGMSolver)
    _validate_problem(prob, "FGPGMSolver"; require_continuous=true,
                      reject_delays=true)
    # The product-of-experts density assumes Gaussian observation noise
    # (the data expert is a Gaussian quadratic form), so refuse other
    # families rather than silently fitting Gaussian (MagiSolver style).
    prob.likelihood isa Gaussian ||
        error("FGPGMSolver supports Gaussian likelihoods only (the " *
              "product-of-experts density assumes Gaussian observation " *
              "noise); got $(typeof(prob.likelihood)). " *
              "Use MCMCSolver or LAML for other likelihood families.")
    any(a -> a isa NeuralApproximator, prob.approximators) &&
        error("FGPGMSolver does not support NeuralApproximator (random-walk " *
              "Metropolis over network weights does not mix); use " *
              "AdamSolver or MultipleShootingSolver.")
    verbose = alg.verbose

    times = Float64.(prob.data_times)
    n_times = length(times)
    u0_vec = Float64.(prob.u0 isa Function ? prob.u0(prob.known_params) : prob.u0)
    n_vars = length(u0_vec)
    n_obs = size(prob.data_values, 2)
    obs_of_state = Dict{Int, Vector{Int}}()   # state ↦ data columns
    for j in 1:n_obs
        push!(get!(obs_of_state, prob.obs_to_state[j], Int[]), j)
    end
    # rng_seed convention (AGM/BNG/MCMCSolver): the sampler owns its
    # stream; a fixed seed never touches the global RNG.
    rng = alg.rng_seed === nothing ? Random.Xoshiro(rand(UInt32)) :
          Random.Xoshiro(alg.rng_seed)

    if verbose
        println("FGPGMSolver: $n_obs observed states, $n_times time points")
    end

    # ── Stage 1: fixed GP hyperparameters and matrices per state ────
    m_center = zeros(n_vars)
    Cinv = Vector{Matrix{Float64}}(undef, n_vars)
    Lp = Vector{Matrix{Float64}}(undef, n_vars)
    D = Vector{Matrix{Float64}}(undef, n_vars)
    Λinv = Vector{Matrix{Float64}}(undef, n_vars)
    data_w = zeros(n_vars)                    # 1/σ_n² (0 for unobserved)
    hyper = Vector{NamedTuple}(undef, n_vars)
    x_init = zeros(n_times, n_vars)

    obs_states = sort(collect(keys(obs_of_state)))
    for sk in obs_states
        # Hyperparameters, center and initialization from the USABLE rows
        # only (ODIN/AGM convention): `mean`/`optimize_gp_hyperparams`
        # NaN-poison everything downstream otherwise. With several data
        # columns per state they come from the first column; replicate
        # columns enter only the data term.
        jc = obs_of_state[sk][1]
        keep_k = usable_rows(prob, jc)
        isempty(keep_k) && error("FGPGMSolver: observation column $jc is " *
            "entirely masked; state $sk has no usable data.")
        y_k = Float64.(prob.data_values[keep_k, jc])
        m_center[sk] = mean(y_k)
        yc = y_k .- m_center[sk]
        σ², ℓ, σn² = if alg.gp_lengthscale !== nothing && alg.gp_variance !== nothing
            # Fixed-hyperparameter path assumes 1% observation noise
            # (ODIN convention).
            (alg.gp_variance, alg.gp_lengthscale, 0.01 * alg.gp_variance)
        else
            optimize_gp_hyperparams(times[keep_k], yc, :rbf; verbose=verbose)
        end
        alg.obs_sigma !== nothing && (σn² = alg.obs_sigma^2)
        hyper[sk] = (σ²=σ², ℓ=ℓ, σn²=σn², observed=true)
        data_w[sk] = 1.0 / σn²
        Cinv[sk], Lp[sk], D[sk], Λinv[sk] =
            _fgpgm_state_matrices(times, σ², ℓ, σn², alg.gamma)
        # Initialise at the GP posterior mean, conditioned on the usable
        # rows and evaluated on the full grid via K(t, t_keep).
        K_full, _, _ = rbf_kernel_with_derivs(times, σ², ℓ)
        if length(keep_k) == n_times
            x_init[:, sk] = m_center[sk] .+
                            K_full * (cholesky(Symmetric(K_full + σn² * I)) \ yc)
        else
            K_cross = K_full[:, keep_k]
            K_kk = K_full[keep_k, keep_k]
            x_init[:, sk] = m_center[sk] .+
                            K_cross * (cholesky(Symmetric(K_kk + σn² * I)) \ yc)
        end
    end

    # Unobserved states: sampled blocks with a borrowed GP prior centered
    # at the initial condition and no data term; identified through the
    # ODE experts (ODIN precedent).
    if length(obs_states) < n_vars
        ℓ_bar = mean(hyper[sk].ℓ for sk in obs_states)
        σ²_bar = mean(hyper[sk].σ² for sk in obs_states)
        σn²_bar = mean(hyper[sk].σn² for sk in obs_states)
        for k in 1:n_vars
            haskey(obs_of_state, k) && continue
            m_center[k] = u0_vec[k]
            hyper[k] = (σ²=σ²_bar, ℓ=ℓ_bar, σn²=σn²_bar, observed=false)
            Cinv[k], Lp[k], D[k], Λinv[k] =
                _fgpgm_state_matrices(times, σ²_bar, ℓ_bar, σn²_bar, alg.gamma)
            x_init[:, k] .= u0_vec[k]
        end
    end

    # ── θ structure: initial values and smoothing prior ─────────────
    beta = Float64[]
    for approx in prob.approximators
        append!(beta, initial_params(approx))
    end
    n_beta = length(beta)
    penalties, offsets, _ = _build_penalty_info(prob)

    # Usable rows per observation column, precomputed for the data term.
    data_rows = Dict{Int, Vector{Int}}(j => usable_rows(prob, j)
                                       for j in 1:n_obs)
    any(!isempty, values(data_rows)) || error("FGPGMSolver: every " *
        "observation is masked; there is nothing to fit.")

    # ── The joint log-density log p(X, θ | y) (up to a constant) ────
    function fgpgm_logdens(X::Matrix{Float64}, β::Vector{Float64})
        p = build_param_struct(prob, β)
        du = zeros(n_vars)
        F = Matrix{Float64}(undef, n_times, n_vars)
        for i in 1:n_times
            try
                prob.dynamics!(du, X[i, :], p, times[i])
            catch e
                _is_program_error(e) && rethrow()
                return -Inf
            end
            F[i, :] .= du
        end
        lp = 0.0
        for k in 1:n_vars
            xc = X[:, k] .- m_center[k]
            lp -= 0.5 * dot(xc, Cinv[k] * xc)                # GP prior
            r = F[:, k] .- D[k] * xc                          # ODE expert
            lp -= 0.5 * dot(r, Λinv[k] * r)
            if haskey(obs_of_state, k)                        # data expert
                for j in obs_of_state[k]
                    rows = data_rows[j]
                    if length(rows) == n_times
                        lp -= 0.5 * data_w[k] *
                              sum(abs2, @view(prob.data_values[:, j]) .- @view(X[:, k]))
                    else
                        acc = 0.0
                        for i in rows
                            acc += (prob.data_values[i, j] - X[i, k])^2
                        end
                        lp -= 0.5 * data_w[k] * acc
                    end
                end
            end
        end
        # θ smoothing prior (MCMCSolver/MagiSolver convention): penalty
        # quadratic forms scaled by prior_scale, plus a broad ridge.
        for (S, off) in zip(penalties, offsets)
            np = size(S, 1)
            bk = @view β[off+1:off+np]
            lp -= 0.5 / alg.prior_scale * dot(bk, S * bk)
        end
        lp -= 0.5 / (100.0 * alg.prior_scale) * dot(β, β)
        isfinite(lp) ? lp : -Inf
    end

    X = copy(x_init)
    lp_cur = fgpgm_logdens(X, beta)
    isfinite(lp_cur) ||
        error("FGPGMSolver: the target density is not finite at the " *
              "initial state — the dynamics return non-finite values on " *
              "the GP-smoothed trajectory. Check the dynamics/approximator " *
              "domains.")

    # ── Stage 2: Metropolis-within-Gibbs over (X, θ) ────────────────
    n_total = alg.n_warmup + alg.n_samples
    sx = fill(0.1, n_vars)                    # per-state proposal scales
    st = 0.05                                 # θ proposal scale
    # Robbins–Monro-style multiplicative tuning toward target_accept:
    # E[Δ log s] = 0 exactly when the acceptance probability equals the
    # target. Adaptation runs during WARMUP ONLY — adapting after warmup
    # would make the chain non-Markovian and break detailed balance, so
    # the scales are frozen and every retained draw comes from a fixed
    # kernel. Acceptance rates are reported over the frozen phase.
    adapt_rate = 0.05
    acc_x = zeros(n_vars); tot_x = zeros(n_vars)
    acc_t = 0.0; tot_t = 0.0
    beta_samples = Matrix{Float64}(undef, alg.n_samples, n_beta)
    X_mean = zeros(n_times, n_vars)

    if verbose
        println("  Metropolis-within-Gibbs: $n_total sweeps " *
                "($(alg.n_warmup) warmup), $n_vars state blocks + " *
                "$n_beta θ parameters")
    end

    for sweep in 1:n_total
        adapting = sweep <= alg.n_warmup
        # State blocks: GP-correlated proposal per state (AGM convention)
        for k in 1:n_vars
            Xp = copy(X)
            Xp[:, k] .+= sx[k] .* (Lp[k] * randn(rng, n_times))
            lp_p = fgpgm_logdens(Xp, beta)
            if log(rand(rng)) < lp_p - lp_cur
                X = Xp; lp_cur = lp_p
                adapting ? (sx[k] *= exp(adapt_rate * (1 - alg.target_accept))) :
                           (acc_x[k] += 1)
            else
                adapting && (sx[k] *= exp(-adapt_rate * alg.target_accept))
            end
            adapting || (tot_x[k] += 1)
        end
        # θ block: joint random walk
        bp = beta .+ st .* randn(rng, n_beta)
        lp_p = fgpgm_logdens(X, bp)
        if log(rand(rng)) < lp_p - lp_cur
            beta = bp; lp_cur = lp_p
            adapting ? (st *= exp(adapt_rate * (1 - alg.target_accept))) :
                       (acc_t += 1)
        else
            adapting && (st *= exp(-adapt_rate * alg.target_accept))
        end
        adapting || (tot_t += 1)
        if !adapting
            s_idx = sweep - alg.n_warmup
            beta_samples[s_idx, :] = beta
            X_mean .+= X
        end
        if verbose && sweep % max(1, n_total ÷ 10) == 0
            println("  sweep $sweep/$n_total  log p = " *
                    "$(round(lp_cur, sigdigits=6))")
        end
    end
    X_mean ./= alg.n_samples
    beta_mean = vec(mean(beta_samples, dims=1))
    accept_rates = (x=acc_x ./ max.(tot_x, 1.0), theta=acc_t / max(tot_t, 1.0))

    if verbose
        println("  acceptance (post-warmup): X=$(round.(accept_rates.x, digits=2)) " *
                "θ=$(round(accept_rates.theta, digits=2))")
    end

    # ── Build solution from the posterior means ─────────────────────
    pred = zeros(n_times, n_obs)
    for j in 1:n_obs
        pred[:, j] .= X_mean[:, prob.obs_to_state[j]]
    end
    data_loss = weighted_data_loss(prob, pred)
    # The reported objective: negative log joint density (up to the
    # additive constants dropped during sampling — log-dets and 2π terms
    # of the fixed covariances; MagiSolver convention) at the posterior
    # means (X̄, θ̄) — the package-wide lower-is-better convention (MAGI
    # reports −mean log-posterior likewise).
    objective = -fgpgm_logdens(X_mean, beta_mean)

    uf_evals = Dict{Symbol, Any}()
    param_names = String[]
    ca_entries = Pair{Symbol, Any}[]
    offset = 0
    for approx in prob.approximators
        np = nparams(approx)
        params_k = beta_mean[offset+1:offset+np]
        offset += np
        uf_evals[approx.name] = build_evaluator(approx, params_k)
        push!(ca_entries, approx.name => params_k)
        for j in 1:np
            push!(param_names, "$(approx.name)[$j]")
        end
    end
    params = ComponentArray(NamedTuple(ca_entries))
    chains = MCMCChains.Chains(beta_samples, Symbol.(param_names))

    # A fixed-budget sampler has no stopping criterion: reporting
    # converged=true would be a fabrication (MagiSolver convention).
    # Judge the run by convergence.chains (R̂/ESS) and accept_rates.
    PSMSolution(params, objective, data_loss, Float64(n_beta), Float64[],
                pred, Float64.(prob.data_values),
                Float64.(prob.data_times), uf_evals,
                (method=:fgpgm, sampler=:metropolis_within_gibbs,
                 chains=chains, beta_samples=beta_samples,
                 state_mean=X_mean,
                 gp_hyperparams=[hyper[k] for k in 1:n_vars],
                 accept_rates=accept_rates,
                 converged=false, reason=:maxiters, iterations=n_total))
end
