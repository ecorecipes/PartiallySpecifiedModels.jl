# ─── Milestone 5: Flagship New Modeling Features Test Suite ──────────
#
# Verifies all M5 modeling additions:
#   1. Observation operators (`obs_fn` pointwise & `TrajectoryObservation`)
#      and latent initial condition estimation (`estimate_u0`)
#   2. Expanded likelihood families (`GammaLikelihood`, `LogNormalLikelihood`,
#      `BinomialLikelihood`, `StudentTLikelihood`) — score identity (★),
#      LAML recovery, deviance residuals, bootstrap, & DistributionsExt coercion
#   3. Multi-trajectory / hierarchical PSMs (`MultiTrajectoryPSMProblem` +
#      `RandomEffectApproximator`)
#   4. SDE / process-noise PSMs (`SDEPSMProblem` + `EKFSolver` + `simulate_sde`)
#   5. `compare_models` & `test_parametric_hypothesis` (exact, :constant, :linear)
#   6. Opt-in `parallel=true` deterministic parity across solvers

using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve, nparams, initial_params, penalty_matrix,
                                build_evaluator
using Distributions
using StatsAPI
using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using StableRNGs
using Statistics

const PSM = PartiallySpecifiedModels

@testset "v0.3 Flagship Modeling Features (M5)" begin

    # ── 1. Observation operators (obs_fn) & estimate_u0 ──────────────
    @testset "Observation operators (obs_fn & TrajectoryObservation) & estimate_u0" begin
        rng = StableRNG(501)
        # 2-state SIR-like system where only sum u1 + u2 and log(u2) are observed,
        # and initial conditions u0 are perturbed & jointly estimated via estimate_u0!
        function decay_sys!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
            du[2] = p.r(t) * u[1] - 0.3 * u[2]
        end
        r_true(t) = 0.4 + 0.2 * sin(0.5 * t)
        u0_true = [2.0, 0.5]
        ts = collect(range(0.0, 6.0, length=25))
        sol_t = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = -r_true(t) * u[1]; du[2] = r_true(t) * u[1] - 0.3 * u[2]),
                       u0_true, (0.0, 6.0)),
            Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        U_true = hcat(Float64[sol_t[1, i] for i in eachindex(ts)],
                      Float64[sol_t[2, i] for i in eachindex(ts)])

        # (a) Pointwise nonlinear obs_fn: observe [log(u1), u1 + u2] + estimate_u0=true
        obs_pw(u, p, t) = [log(max(u[1], 1e-6)), u[1] + u[2]]
        y_pw = hcat(log.(U_true[:, 1]), U_true[:, 1] .+ U_true[:, 2]) .+
               0.005 .* randn(rng, length(ts), 2)

        prob_pw = PSMProblem(decay_sys!, [1.6, 0.8], (0.0, 6.0),
                             [BSplineApproximator(:r, (0.0, 6.0), 7; initial=0.4)];
                             data_times=ts, data_values=y_pw,
                             obs_fn=obs_pw, estimate_u0=true)
        sol_pw = solve(prob_pw, LAML(maxiters=50))
        @test sol_pw.convergence.converged
        @test haskey(sol_pw.parameters, :u0)
        @test length(sol_pw.parameters.u0) == 2
        @test sol_pw.parameters.u0[1] ≈ u0_true[1] atol=0.08
        @test sol_pw.parameters.u0[2] ≈ u0_true[2] atol=0.08
        @test sol_pw.data_loss < 0.05
        # Covariance V_beta includes the 2 estimated u0 parameters
        @test size(StatsAPI.vcov(sol_pw)) == (9, 9)
        @test all(>(0.0), StatsAPI.stderror(sol_pw))

        # (b) TrajectoryObservation: interval cumulative incidence differences
        # State 3 = cumulative transitions ∫ r(t)*u1 dt; observe interval increments
        function inc_sys!(du, u, p, t)
            flow = p.r(t) * u[1]
            du[1] = -flow
            du[2] = flow - 0.3 * u[2]
            du[3] = flow
        end
        sol_inc = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (fl = r_true(t) * u[1]; du[1] = -fl; du[2] = fl - 0.3 * u[2]; du[3] = fl),
                       [2.0, 0.5, 0.0], (0.0, 6.0)),
            Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        C_true = Float64[sol_inc[3, i] for i in eachindex(ts)]
        inc_true = vcat([0.0], diff(C_true))
        y_inc = reshape(inc_true, :, 1)
        w_inc = ones(length(ts), 1); w_inc[1, 1] = 0.0   # t=0 has no prior interval

        traj_obs = TrajectoryObservation((U, times, p) -> begin
            T = eltype(U)
            out = zeros(T, size(U, 1), 1)
            for i in 2:size(U, 1)
                out[i, 1] = U[i, 3] - U[i - 1, 3]
            end
            out
        end)
        prob_inc = PSMProblem(inc_sys!, [2.0, 0.5, 0.0], (0.0, 6.0),
                              [BSplineApproximator(:r, (0.0, 6.0), 7; initial=0.4)];
                              data_times=ts, data_values=y_inc, data_weights=w_inc,
                              obs_fn=traj_obs)
        sol_inc_fit = solve(prob_inc, LAML(maxiters=40))
        @test sol_inc_fit.convergence.converged
        @test sol_inc_fit.data_loss < 1e-4
        r_hat = sol_inc_fit.unknown_functions[:r]
        @test sqrt(mean((r_hat(t) - r_true(t))^2 for t in range(0.5, 5.5, length=20))) < 0.05
    end

    # ── 2. Expanded Likelihood Families + DistributionsExt ───────────
    @testset "Expanded likelihood families: score identity, LAML & DistributionsExt" begin
        # Exact Fisher-scoring identity (★): W̃_i * (z - η)_i == w_i * ∂ℓ/∂μ_i
        fams_and_pts = [
            (GammaLikelihood(0.2),          [2.5, 5.1], [2.0, 4.8]),
            (LogNormalLikelihood(0.15),     [3.2, 8.7], [3.0, 9.1]),
            (BinomialLikelihood(20),        [4.0, 15.0], [0.25, 0.72]),
            (StudentTLikelihood(3.0, 0.5),  [1.2, 5.5], [1.0, 1.1]),   # includes strong outlier!
        ]
        w_test = [1.0, 2.5]
        for (fam, yv, muv) in fams_and_pts
            Wt = PSM.irls_weights(fam, yv, muv, w_test)
            wrk = PSM._working_residual(fam, yv, muv, w_test)
            @test all(>(0.0), Wt)
            for i in eachindex(yv)
                score_ad = ForwardDiff.derivative(m -> PSM.loglik_pointwise(fam, yv[i], m), muv[i])
                @test Wt[i] * wrk[i] ≈ w_test[i] * score_ad rtol=1e-10
            end
            # log_likelihood matches sum of w_i * loglik_pointwise
            @test PSM.log_likelihood(fam, yv, muv, w_test) ≈
                  sum(w_test[i] * PSM.loglik_pointwise(fam, yv[i], muv[i]) for i in eachindex(yv)) rtol=1e-12
            dr = deviance_residuals(fam, yv, muv)
            @test length(dr) == length(yv) && all(isfinite, dr)
        end

        # DistributionsExt coercion for all 4 new families
        @test PSM._coerce_likelihood(Distributions.Gamma(5.0, 2.0)) == GammaLikelihood(0.2)
        @test PSM._coerce_likelihood(Distributions.LogNormal(0.0, 0.25)) == LogNormalLikelihood(0.25)
        @test PSM._coerce_likelihood(Distributions.Binomial(30, 0.5)) == BinomialLikelihood(30)
        @test PSM._coerce_likelihood(Distributions.TDist(4.0)) == StudentTLikelihood(4.0, 1.0)
        @test PSM._coerce_likelihood(2.0 + 0.5 * Distributions.TDist(5.0)) == StudentTLikelihood(5.0, 0.5)

        # End-to-end LAML fit with Gamma, LogNormal, Binomial, and robust StudentT (with outlier!)
        rng = StableRNG(502)
        ts = collect(range(0.0, 5.0, length=26))
        # Positive logistic growth u(t) ∈ [0.5, 4.0]
        u_log = [4.0 / (1.0 + 7.0 * exp(-0.9 * t)) for t in ts]
        log_ode!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])

        # (a) LogNormal multiplicative data via Distributions.LogNormal
        y_ln = reshape([u_log[i] * exp(0.05 * randn(rng)) for i in eachindex(ts)], :, 1)
        prob_ln = PSMProblem(log_ode!, [0.5], (0.0, 5.0),
                             [BSplineApproximator(:r, (0.0, 4.5), 6; initial=0.5)];
                             data_times=ts, data_values=y_ln, obs_to_state=[1],
                             likelihood=Distributions.LogNormal(0.0, 0.05))
        sol_ln = solve(prob_ln, LAML(maxiters=40))
        @test isfinite(sol_ln.objective) && sol_ln.data_loss < 2.0
        @test isfinite(StatsAPI.aic(sol_ln, prob_ln))

        # (b) Gamma data via Distributions.Gamma(shape=100, scale=1) -> phi=0.01
        y_gam = reshape([u_log[i] * (1.0 + 0.05 * randn(rng)) for i in eachindex(ts)], :, 1)
        prob_gam = PSMProblem(log_ode!, [0.5], (0.0, 5.0),
                              [BSplineApproximator(:r, (0.0, 4.5), 6; initial=0.5)];
                              data_times=ts, data_values=y_gam, obs_to_state=[1],
                              likelihood=Distributions.Gamma(100.0, 1.0))
        sol_gam = solve(prob_gam, LAML(maxiters=40))
        @test isfinite(sol_gam.objective) && sol_gam.data_loss < 2.0

        # (c) Binomial proportion data (n=50) via Distributions.Binomial(50, 0.5)
        p_traj = [0.85 / (1.0 + 9.0 * exp(-0.8 * t)) for t in ts]
        y_bin = reshape(Float64[rand(rng, Distributions.Binomial(50, p_traj[i])) for i in eachindex(ts)], :, 1)
        prob_bin = PSMProblem((du, u, p, t) -> (du[1] = p.g(u[1]) * u[1] * (1.0 - u[1])),
                              [p_traj[1]], (0.0, 5.0),
                              [BSplineApproximator(:g, (0.0, 1.0), 5; initial=0.6)];
                              data_times=ts, data_values=y_bin, obs_to_state=[1],
                              likelihood=Distributions.Binomial(50, 0.5))
        sol_bin = solve(prob_bin, LAML(maxiters=40))
        @test isfinite(sol_bin.objective)
        @test maximum(abs.(sol_bin.fitted_values[:, 1] .- p_traj)) < 0.12

        # (d) StudentT robustness against a massive outlier vs Gaussian
        y_out = reshape(copy(u_log) .+ 0.03 .* randn(rng, length(ts)), :, 1)
        y_out[13, 1] += 4.5   # huge outlier!
        prob_t = PSMProblem(log_ode!, [0.5], (0.0, 5.0),
                            [BSplineApproximator(:r, (0.0, 4.5), 6; initial=0.5)];
                            data_times=ts, data_values=y_out, obs_to_state=[1],
                            likelihood=StudentTLikelihood(2.5, 0.05))
        prob_g = PSMProblem(log_ode!, [0.5], (0.0, 5.0),
                            [BSplineApproximator(:r, (0.0, 4.5), 6; initial=0.5)];
                            data_times=ts, data_values=y_out, obs_to_state=[1],
                            likelihood=GaussianLikelihood())
        sol_t = solve(prob_t, LAML(maxiters=50))
        sol_g = solve(prob_g, LAML(maxiters=50))
        err_t = maximum(abs.(sol_t.fitted_values[:, 1] .- u_log))
        err_g = maximum(abs.(sol_g.fitted_values[:, 1] .- u_log))
        @test err_t < 0.15
        @test err_t < err_g   # Student-t ignores the outlier much better than Gaussian!
    end

    # ── 3. MultiTrajectoryPSMProblem + RandomEffectApproximator ──────
    @testset "MultiTrajectoryPSMProblem & RandomEffectApproximator" begin
        rng = StableRNG(503)
        # 4 replicate microcosms sharing logistic per-capita growth r(u) = 1.0*(1 - u/5)
        # plus group-specific random effect b_g ∈ [-0.15, -0.05, +0.05, +0.15]
        b_true = [-0.15, -0.05, 0.05, 0.15]
        u0s = [[0.4], [0.8], [1.2], [0.6]]
        ts = collect(range(0.0, 6.0, length=19))

        function rep_ode!(du, u, p, t)
            du[1] = (p.r(u[1]) + p.re(p.group_id)) * u[1]
        end

        y_list = Matrix{Float64}[]
        for g in 1:4
            sol_g = OrdinaryDiffEq.solve(
                ODEProblem((du, u, p, t) -> (du[1] = (1.0 * (1 - u[1] / 5.0) + b_true[g]) * u[1]),
                           u0s[g], (0.0, 6.0)),
                Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
            yg = reshape(Float64[sol_g[1, i] for i in eachindex(ts)], :, 1) .+
                 0.02 .* randn(rng, length(ts), 1)
            push!(y_list, yg)
        end

        approxs = AbstractApproximator[
            BSplineApproximator(:r, (0.0, 6.0), 7; initial=0.5),
            RandomEffectApproximator(:re, 4; initial=0.0),
        ]
        mp = MultiTrajectoryPSMProblem(rep_ode!, u0s, (0.0, 6.0), approxs;
                                       data_times_list=fill(ts, 4),
                                       data_values_list=y_list)
        sol_mt = solve(mp, LAML(maxiters=50))
        @test sol_mt.convergence.converged
        @test sol_mt.convergence.n_trajectories == 4
        @test length(sol_mt.convergence.trajectory_fits) == 4
        @test length(sol_mt.smoothing_params) == 2   # [λ_r, λ_re]
        @test sol_mt.data_loss < 0.2
        # Random effects preserve ordering b_1 < b_2 < b_3 < b_4
        re_hat = sol_mt.parameters.re
        @test cor(re_hat, b_true) > 0.90
        # Shared growth curve matches true r(u) + mean(b) across interior
        r_hat = sol_mt.unknown_functions[:r]
        rmse_r = sqrt(mean((r_hat(u) - 1.0 * (1 - u / 5.0))^2 for u in range(0.8, 4.2, length=20)))
        @test rmse_r < 0.12
    end

    # ── 4. SDE / Process-Noise PSMs (SDEPSMProblem + EKFSolver) ──────
    @testset "SDEPSMProblem & EKFSolver (process vs observation noise)" begin
        rng = StableRNG(504)
        # Stochastic Ornstein-Uhlenbeck / logistic-like process:
        # du = (p.r(u) * u) dt + σ_proc dW_t, y_i = u(t_i) + σ_obs ε_i
        sde_drift!(du, u, p, t) = (du[1] = p.f(u[1]))
        f_true(u) = 0.8 * (3.0 - u)    # linear mean-reverting drift to K=3
        ts = collect(range(0.0, 12.0, length=49))

        # Generate exact SDE sample path via simulate_sde
        approx_f = BSplineApproximator(:f, (0.0, 6.0), 6; initial=u -> 0.5 * (2.5 - u))
        prob_sim = SDEPSMProblem(sde_drift!, 0.25, [0.5], (0.0, 12.0), [approx_f];
                                 data_times=ts, data_values=zeros(length(ts), 1),
                                 obs_noise=0.15)
        beta_true = Float64[f_true(x) for x in range(0.0, 6.0, length=6)]
        states_true, y_sde = simulate_sde(prob_sim, beta_true; rng=rng, dt=0.005,
                                          process_scale=1.0, obs_noise=[0.15])
        # Introduce one missing observation (NaN) to test EKF missing-data handling
        y_sde_masked = copy(y_sde)
        y_sde_masked[20, 1] = NaN

        prob_sde = SDEPSMProblem(sde_drift!, 0.25, [0.5], (0.0, 12.0), [approx_f];
                                 data_times=ts, data_values=y_sde_masked,
                                 obs_noise=0.2, estimate_noise=true)
        sol_sde = solve(prob_sde, EKFSolver(maxiters=80, n_substeps=6, n_laml_outer=4))
        @test isfinite(sol_sde.convergence.loglik)
        @test sol_sde.convergence.process_scale > 0.0
        @test sol_sde.convergence.obs_noise[1] > 0.0
        @test size(sol_sde.convergence.smoothed_states) == (length(ts), 1)
        @test all(isfinite, sol_sde.convergence.smoothed_states)
        @test all(>=(0.0), sol_sde.convergence.smoothed_std)
        # RTS smoothed state tracks latent truth closely (even across the NaN gap!)
        rmse_state = sqrt(mean((sol_sde.convergence.smoothed_states[:, 1] .- states_true[:, 1]).^2))
        @test rmse_state < 0.25
        # Drift recovery
        f_hat = sol_sde.unknown_functions[:f]
        rmse_f = sqrt(mean((f_hat(u) - f_true(u))^2 for u in range(1.5, 3.8, length=15)))
        @test rmse_f < 0.6
    end

    # ── 5. compare_models & test_parametric_hypothesis ───────────────
    @testset "compare_models & test_parametric_hypothesis" begin
        rng = StableRNG(515)
        ts = collect(range(0.0, 6.0, length=31))
        # Strongly nonlinear Allee-like per-capita growth rate r(u) = 0.2 + 0.8*u - 0.25*u^2
        r_nl(u) = 0.2 + 0.8 * u - 0.25 * u^2
        sol_nl = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = r_nl(u[1]) * u[1]), [0.4], (0.0, 6.0)),
            Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        y_nl = reshape(Float64[sol_nl[1, i] for i in eachindex(ts)], :, 1) .+
               0.015 .* randn(rng, length(ts), 1)

        ode_r!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        prob_flex = PSMProblem(ode_r!, [0.4], (0.0, 6.0),
                               [BSplineApproximator(:r, (0.0, 4.0), 8; initial=0.5)];
                               data_times=ts, data_values=y_nl, obs_to_state=[1])
        # Rigid linear model (heavy fixed lambda shrinks to linear null space)
        sol_flex = solve(prob_flex, LAML(maxiters=50))
        sol_lin  = solve(prob_flex, LAML(maxiters=50, fixed_lambda=1e8))

        # Model comparison table favors flexible nonlinear fit over linear model
        tbl = compare_models(sol_lin => prob_flex, sol_flex => prob_flex;
                             names=["Linear", "Flexible"])
        @test length(tbl) == 2
        @test tbl[1].name == "Flexible"
        @test tbl[1].delta_aicc ≈ 0.0
        @test tbl[1].aicc_weight > 0.99
        @test sum(r.aicc_weight for r in tbl) ≈ 1.0 atol=1e-12
        buf = IOBuffer(); show(buf, tbl)
        @test occursin("Flexible", String(take!(buf)))

        # Wald test rejects :constant and :linear null on the curved Allee growth rate,
        # and does NOT reject the true quadratic function r_nl!
        t_const = test_parametric_hypothesis(sol_flex, prob_flex, :r, :constant;
                                             grid=range(0.6, 3.0, length=25))
        t_lin   = test_parametric_hypothesis(sol_flex, prob_flex, :r, :linear;
                                             grid=range(0.6, 3.0, length=25))
        t_true  = test_parametric_hypothesis(sol_flex, prob_flex, :r, r_nl;
                                             grid=range(0.6, 3.0, length=25),
                                             unconditional=true)
        @test t_const.reject_05 && t_const.pvalue < 1e-4
        @test t_lin.reject_05 && t_lin.pvalue < 1e-4
        @test !t_true.reject_05 && t_true.pvalue > 0.05
    end

    # ── 6. Opt-in parallel=true API & deterministic parity ───────────
    @testset "Opt-in parallel=true API across solvers" begin
        rng = StableRNG(506)
        ts = collect(range(0.0, 4.0, length=15))
        y = reshape([1.0 / (1.0 + 3.0 * exp(-0.8 * t)) for t in ts], :, 1) .+
            0.01 .* randn(rng, length(ts), 1)
        prob_p = PSMProblem((du, u, p, t) -> (du[1] = p.r(u[1]) * u[1]),
                            [0.25], (0.0, 4.0),
                            [SPDEApproximator(:r, (0.0, 1.5), 6; initial=0.5)];
                            data_times=ts, data_values=y, obs_to_state=[1])

        res_seq = optimize_spde_range(prob_p, LAML(maxiters=20);
                                      range_multipliers=[0.5, 1.0, 2.0], parallel=false)
        res_par = optimize_spde_range(prob_p, LAML(maxiters=20);
                                      range_multipliers=[0.5, 1.0, 2.0], parallel=true)
        @test res_seq.gcv_scores ≈ res_par.gcv_scores rtol=1e-10

        bng_seq = solve(prob_p, BNGSolver(k_obs=2, k_proc=2, maxiters=80, rng_seed=42, parallel=false))
        bng_par = solve(prob_p, BNGSolver(k_obs=2, k_proc=2, maxiters=80, rng_seed=42, parallel=true))
        @test Vector{Float64}(bng_seq.parameters) ≈ Vector{Float64}(bng_par.parameters) atol=1e-12

        abc_seq = solve(prob_p, ABCSolver(n_particles=20, n_generations=2, rng_seed=42, parallel=false))
        abc_par = solve(prob_p, ABCSolver(n_particles=20, n_generations=2, rng_seed=42, parallel=true))
        @test isfinite(abc_seq.data_loss) && isfinite(abc_par.data_loss)
    end
end
