# ─── Milestone 4: Literature Faithfulness Upgrades Test Suite ────────
#
# Verifies all M4 algorithmic faithfulness upgrades:
#   1. CyclicBSplineApproximator (C² periodicity, rank n-1 circulant penalty,
#      constant null space, 0 B/call, Dual-safe, seasonal ODE recovery)
#   2. TensorBSplineApproximator(anisotropy=:estimate) (Wood 2006 t2 3-block
#      orthogonal decomposition, 3 smoothing params, scale invariance, 0 B/call)
#   3. Vector / partial per-block fixed_lambda in LAML
#   4. Exact Wood (2011) Theorem 1 implicit-derivative Hessian in Phase-2 LAML Newton
#   5. ProbSolve Krämer & Hennig (2024) q=4 stability + DALTON-Fenrir linear parity
#   6. MAGI exact grid inclusion (I_obs ⊆ I) & Yang et al. (2021) prior tempering
#   7. IntegralMatchingSolver Dattner & Klaassen (2015) mean-centering + Simpson
#   8. ODINSolver & FGPGMSolver Matérn-5/2 kernel + ODIN state standardization
#   9. BNGSolver scale-invariant derivative standardization

using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve, penalty_matrix, penalty_blocks,
                                nparams, initial_params, build_evaluator,
                                band_domain
using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using StableRNGs
using Statistics

const PSM = PartiallySpecifiedModels

@testset "v0.3 Literature Faithfulness Upgrades" begin

    # ── 1. CyclicBSplineApproximator ─────────────────────────────────
    @testset "CyclicBSplineApproximator: C² periodicity, penalty & seasonal fit" begin
        ca = CyclicBSplineApproximator(:season, (0.0, 1.0), 8)
        @test nparams(ca) == 8
        @test band_domain(ca) == (0.0, 1.0)

        # Circulant penalty matrix: symmetric PSD, rank n-1, constant null space
        S = penalty_matrix(ca)
        @test size(S) == (8, 8)
        @test S ≈ S' atol=1e-12
        evals = eigvals(Symmetric(S))
        @test count(>(1e-10), evals) == 7
        @test dot(ones(8), S * ones(8)) ≈ 0.0 atol=1e-12

        # C² periodicity at domain boundary and wrapping outside [lo, hi]
        rng = StableRNG(401)
        beta = randn(rng, 8)
        ev = build_evaluator(ca, beta)

        @test ev(0.0) ≈ ev(1.0) atol=1e-12
        @test ev(0.25) ≈ ev(1.25) atol=1e-12
        @test ev(0.25) ≈ ev(-0.75) atol=1e-12

        # Exact C¹ and C² continuity across seam t = 0 ≡ 1 via ForwardDiff
        d1_lo = ForwardDiff.derivative(ev, 1e-9)
        d1_hi = ForwardDiff.derivative(ev, 1.0 - 1e-9)
        @test d1_lo ≈ d1_hi atol=1e-5

        d2_lo = ForwardDiff.derivative(x -> ForwardDiff.derivative(ev, x), 1e-8)
        d2_hi = ForwardDiff.derivative(x -> ForwardDiff.derivative(ev, x), 1.0 - 1e-8)
        @test d2_lo ≈ d2_hi atol=1e-4

        # Zero allocations after warmup + Dual-safe in beta
        ev(0.37)
        @test (@allocated ev(0.37)) == 0
        g_ad = ForwardDiff.gradient(b -> build_evaluator(ca, b)(0.37), beta)
        h = 1e-6
        g_fd = [(build_evaluator(ca, beta .+ h .* (1:8 .== j))(0.37) -
                 build_evaluator(ca, beta .- h .* (1:8 .== j))(0.37)) / (2h) for j in 1:8]
        @test g_ad ≈ g_fd atol=1e-6

        # Seasonal ODE recovery: du/dt = u * (r_season(t) - 0.5 * u), period = 1.0
        r_true(t) = 1.2 + 0.6 * cos(2π * t)
        function seasonal_ode!(du, u, p, t)
            du[1] = u[1] * (p.r(t) - 0.5 * u[1])
        end
        ts = collect(range(0.0, 2.0, length=41))
        ode_true = ODEProblem((du, u, p, t) -> (du[1] = u[1] * (r_true(t) - 0.5 * u[1])),
                              [1.0], (0.0, 2.0))
        sol_true = OrdinaryDiffEq.solve(ode_true, Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        y_obs = reshape(Float64[sol_true[1, i] for i in eachindex(ts)], :, 1) .+
                0.01 .* randn(rng, length(ts), 1)

        prob_cyc = PSMProblem(seasonal_ode!, [1.0], (0.0, 2.0),
                              [CyclicBSplineApproximator(:r, (0.0, 1.0), 10; initial=1.0)];
                              data_times=ts, data_values=y_obs, obs_to_state=[1])
        sol_cyc = solve(prob_cyc, LAML(maxiters=50))
        @test sol_cyc.convergence.converged
        r_hat = sol_cyc.unknown_functions[:r]
        @test r_hat(0.0) ≈ r_hat(1.0) atol=1e-12
        grid_t = range(0.0, 1.0, length=50)
        rmse = sqrt(mean((r_hat(t) - r_true(t))^2 for t in grid_t))
        @test rmse < 0.08
    end

    # ── 2. TensorBSplineApproximator(anisotropy=:estimate) Wood t2 ───
    @testset "TensorBSplineApproximator Wood (2006) t2 multi-λ anisotropy" begin
        ta_iso = TensorBSplineApproximator(:f, (0.0, 2.0), (0.0, 2.0), 5, 5)
        @test ta_iso.anisotropy == 1.0
        @test length(penalty_blocks(ta_iso)) == 1

        ta_t2 = TensorBSplineApproximator(:f, (0.0, 2.0), (0.0, 2.0), 5, 5;
                                           anisotropy=:estimate)
        @test ta_t2.anisotropy === :estimate
        @test nparams(ta_t2) == 25
        blocks = penalty_blocks(ta_t2)
        @test length(blocks) == 3

        # Verify blocks are pairwise disjoint and strictly positive definite
        covered = Int[]
        for (Sb, rng_b) in blocks
            np_b = length(rng_b)
            @test size(Sb) == (np_b, np_b)
            @test Sb ≈ Sb' atol=1e-12
            @test minimum(eigvals(Symmetric(Sb))) > 1e-10
            append!(covered, collect(rng_b))
        end
        @test length(unique(covered)) == length(covered)
        # Total penalized dim = 25 - 4 (2×2 bilinear null space) = 21
        @test length(covered) == 21

        # Zero allocation & Dual-safe evaluation
        rng = StableRNG(402)
        beta = randn(rng, 25)
        ev = build_evaluator(ta_t2, beta)
        ev(0.7, 1.3)
        @test (@allocated ev(0.7, 1.3)) == 0
        g_ad = ForwardDiff.gradient(b -> build_evaluator(ta_t2, b)(0.7, 1.3), beta)
        h = 1e-6
        g_fd = [(build_evaluator(ta_t2, beta .+ h .* (1:25 .== j))(0.7, 1.3) -
                 build_evaluator(ta_t2, beta .- h .* (1:25 .== j))(0.7, 1.3)) / (2h) for j in 1:25]
        @test g_ad ≈ g_fd atol=1e-6

        # Fit an anisotropic 2D ODE (curved in x, nearly linear in y) and check
        # LAML estimates 3 separate smoothing params and achieves scale invariance
        # when y is rescaled by 100×.
        f2d_true(x, y) = 0.6 + 0.5 * sin(2.0 * x) - 0.1 * y
        function ode_2d!(du, u, p, t)
            du[1] = p.f(u[1], u[2]) - 0.3 * u[1]
            du[2] = 0.4 * (1.5 - u[2]) + 0.2 * cos(t)
        end
        ts2 = collect(range(0.0, 5.0, length=26))
        sol2_t = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = f2d_true(u[1], u[2]) - 0.3 * u[1];
                                         du[2] = 0.4 * (1.5 - u[2]) + 0.2 * cos(t)),
                       [0.4, 0.3], (0.0, 5.0)),
            Tsit5(); saveat=ts2, abstol=1e-10, reltol=1e-10)
        y2 = hcat(Float64[sol2_t[1, i] for i in eachindex(ts2)],
                  Float64[sol2_t[2, i] for i in eachindex(ts2)])

        prob_t2 = PSMProblem(ode_2d!, [0.4, 0.3], (0.0, 5.0),
                             [TensorBSplineApproximator(:f, (0.0, 2.5), (0.0, 2.0), 5, 5;
                                                        anisotropy=:estimate, initial=0.5)];
                             data_times=ts2, data_values=y2, obs_to_state=[1, 2])
        sol_t2 = solve(prob_t2, LAML(maxiters=40))
        @test length(sol_t2.smoothing_params) == 3
        @test all(isfinite, sol_t2.smoothing_params)
        @test sol_t2.data_loss < 1e-2

        # Rescale coordinate 2 by 100×: Wood's t2 should fit to comparable loss
        function ode_2d_scaled!(du, u, p, t)
            du[1] = p.f(u[1], u[2]) - 0.3 * u[1]
            du[2] = 0.4 * (150.0 - u[2]) + 20.0 * cos(t)
        end
        y2_scaled = hcat(y2[:, 1], 100.0 .* y2[:, 2])
        prob_t2_s = PSMProblem(ode_2d_scaled!, [0.4, 30.0], (0.0, 5.0),
                               [TensorBSplineApproximator(:f, (0.0, 2.5), (0.0, 200.0), 5, 5;
                                                          anisotropy=:estimate, initial=0.5)];
                               data_times=ts2, data_values=y2_scaled, obs_to_state=[1, 2],
                               data_weights=hcat(ones(length(ts2)), fill(1e-4, length(ts2))))
        sol_t2_s = solve(prob_t2_s, LAML(maxiters=40))
        @test length(sol_t2_s.smoothing_params) == 3
        @test sol_t2_s.data_loss < 1e-2
    end

    # ── 3. Per-block / partial vector fixed_lambda in LAML ───────────
    @testset "LAML per-block Vector fixed_lambda" begin
        rng = StableRNG(403)
        function two_uf_ode!(du, u, p, t)
            du[1] = p.g1(u[1]) - 0.2 * u[2]
            du[2] = p.g2(u[2]) + 0.1 * u[1]
        end
        ts = collect(range(0.0, 4.0, length=21))
        sol_true = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = 0.8 * u[1] * (1 - u[1]/3) - 0.2 * u[2];
                                         du[2] = 0.3 * u[2] * (1 - u[2]/2) + 0.1 * u[1]),
                       [0.5, 0.4], (0.0, 4.0)),
            Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        y = hcat(Float64[sol_true[1, i] for i in eachindex(ts)],
                 Float64[sol_true[2, i] for i in eachindex(ts)]) .+
            0.01 .* randn(rng, length(ts), 2)

        prob2 = PSMProblem(two_uf_ode!, [0.5, 0.4], (0.0, 4.0),
                           [BSplineApproximator(:g1, (0.0, 3.5), 7; initial=0.3),
                            BSplineApproximator(:g2, (0.0, 3.5), 7; initial=0.2)];
                           data_times=ts, data_values=y, obs_to_state=[1, 2])

        # Fix block 1 to 2.5, estimate block 2 via LAML
        sol_part = solve(prob2, LAML(maxiters=35, fixed_lambda=Union{Nothing,Float64}[2.5, nothing]))
        @test length(sol_part.smoothing_params) == 2
        @test sol_part.smoothing_params[1] ≈ 2.5 atol=1e-12
        @test isfinite(sol_part.smoothing_params[2]) && sol_part.smoothing_params[2] > 0.0
        @test sol_part.smoothing_params[2] != 2.5

        # Fix both blocks explicitly via vector
        sol_both = solve(prob2, LAML(maxiters=35, fixed_lambda=[1.5, 7.5]))
        @test sol_both.smoothing_params ≈ [1.5, 7.5] atol=1e-12
    end

    # ── 4. Exact Wood (2011) Theorem 1 Implicit Hessian in LAML ──────
    @testset "Wood (2011) _laml_newton_hessian exact FD parity" begin
        rng = StableRNG(404)
        n = 40; nk1 = 6; nk2 = 6; n_p = nk1 + nk2
        J = randn(rng, n, n_p)
        W = ones(n)
        beta_true = randn(rng, n_p)
        y = J * beta_true .+ 0.25 .* randn(rng, n)
        S1 = spline_penalty_matrix(collect(range(0.0, 1.0, length=nk1)))
        S2 = spline_penalty_matrix(collect(range(0.0, 1.0, length=nk2)))
        S_list = [S1, S2]
        offsets = [0, nk1]
        nknots_list = [nk1, nk2]
        JWJ = J' * (W .* J)
        JWz = J' * (W .* y)
        S_ranks = [count(>(1e-10), eigvals(Symmetric(S))) for S in S_list]
        M_p = n_p - sum(S_ranks)
        n_eff = Float64(n - M_p)

        # Profiled Gaussian LAML objective V_prof(rho) with re-solved β̂(ρ)
        function prof_laml(rho_v)
            H = Matrix{eltype(rho_v)}(JWJ)
            for k in 1:2
                rk = (offsets[k] + 1):(offsets[k] + nknots_list[k])
                H[rk, rk] .+= exp(rho_v[k]) .* S_list[k]
            end
            b_hat = H \ JWz
            r = y .- J * b_hat
            rss = dot(r, W .* r)
            pen = sum(exp(rho_v[k]) * dot(b_hat[(offsets[k]+1):(offsets[k]+nknots_list[k])],
                      S_list[k] * b_hat[(offsets[k]+1):(offsets[k]+nknots_list[k])]) for k in 1:2)
            s2 = (rss + pen) / n_eff
            # Matches laml_objective(GaussianLikelihood(), ...)
            ld_S = sum(S_ranks[k] * rho_v[k] for k in 1:2)
            -0.5 * (n_eff * log(s2) + logdet(H) - ld_S)
        end

        rho0 = [-0.5, 0.8]
        H0 = copy(JWJ)
        for k in 1:2
            rk = (offsets[k] + 1):(offsets[k] + nknots_list[k])
            H0[rk, rk] .+= exp(rho0[k]) .* S_list[k]
        end
        b0 = H0 \ JWz
        r0 = y .- J * b0
        pen0 = sum(exp(rho0[k]) * dot(b0[(offsets[k]+1):(offsets[k]+nknots_list[k])],
                   S_list[k] * b0[(offsets[k]+1):(offsets[k]+nknots_list[k])]) for k in 1:2)
        s2_0 = (dot(r0, W .* r0) + pen0) / n_eff

        H_exact = PSM._laml_newton_hessian(GaussianLikelihood(), b0, S_list,
                                           offsets, nknots_list, rho0, n_p,
                                           H0, s2_0, n_eff)
        # ForwardDiff exact Hessian of the profiled objective
        H_ad = ForwardDiff.hessian(prof_laml, rho0)
        @test H_exact ≈ H_ad rtol=1e-6 atol=1e-7

        # And verify non-Gaussian (unit scale) profiled implicit Hessian too
        function prof_laml_ng(rho_v)
            H = Matrix{eltype(rho_v)}(JWJ)
            for k in 1:2
                rk = (offsets[k] + 1):(offsets[k] + nknots_list[k])
                H[rk, rk] .+= exp(rho_v[k]) .* S_list[k]
            end
            b_hat = H \ JWz
            r = y .- J * b_hat
            rss = dot(r, W .* r)
            pen = sum(exp(rho_v[k]) * dot(b_hat[(offsets[k]+1):(offsets[k]+nknots_list[k])],
                      S_list[k] * b_hat[(offsets[k]+1):(offsets[k]+nknots_list[k])]) for k in 1:2)
            ld_S = sum(S_ranks[k] * rho_v[k] for k in 1:2)
            -0.5 * ((rss + pen) + logdet(H) - ld_S)
        end
        H_exact_ng = PSM._laml_newton_hessian(PoissonLikelihood(), b0, S_list,
                                              offsets, nknots_list, rho0, n_p,
                                              H0, 1.0, n_eff)
        H_ad_ng = ForwardDiff.hessian(prof_laml_ng, rho0)
        @test H_exact_ng ≈ H_ad_ng rtol=1e-6 atol=1e-7
    end

    # ── 5. ProbSolve q=4 stability + DALTON-Fenrir linear parity ─────
    @testset "ProbSolve q=4 Taylor init & DALTON-Fenrir linear parity" begin
        # Linear decay ODE: du/dt = -a * u
        a_true = 0.5
        lin_dyn!(du, u, p, t) = (du[1] = -p.a * u[1])
        u0 = [2.0]
        ts = collect(range(0.0, 4.0, length=21))
        y_lin = reshape([2.0 * exp(-a_true * t) for t in ts], :, 1)

        # q = 4 probsolve_filter stability (previously collapsed without diffuse init)
        filt_q4 = PSM.probsolve_filter(lin_dyn!, (a=a_true,), u0, (0.0, 4.0),
                                       80, 4, [1.0])
        means_q4 = [m[1] for m in filt_q4["μ_filt"]]
        times_q4 = filt_q4["times"]
        @test all(isfinite, means_q4)
        @test maximum(abs.(means_q4 .- [2.0 * exp(-a_true * t) for t in times_q4])) < 1e-3

        # DALTON == Fenrir marginal log-likelihood on linear ODE
        ll_fenrir = PSM.fenrir_loglik(lin_dyn!, (a=a_true,), u0, (0.0, 4.0),
                                      40, 2, [1.0], y_lin, ts, [1], 0.1^2)
        ll_dalton = PSM._dalton_loglik(lin_dyn!, (a=a_true,), u0, (0.0, 4.0),
                                       40, 2, [1.0], y_lin, ts, [1], 0.1^2)
        @test isfinite(ll_fenrir) && isfinite(ll_dalton)
        @test ll_dalton ≈ ll_fenrir rtol=1e-3
    end

    # ── 6. MAGI exact grid inclusion & prior tempering ───────────────
    @testset "MagiSolver exact grid inclusion (I_obs ⊆ I) & tempering" begin
        rng = StableRNG(406)
        # Deliberately irregular observation times that would be snapped by a uniform grid
        ts_irr = [0.0, 0.37, 0.91, 1.43, 2.18, 2.84, 3.55, 4.0]
        y_irr = reshape([1.0 / (1.0 + 3.0 * exp(-0.8 * t)) for t in ts_irr], :, 1) .+
                0.01 .* randn(rng, length(ts_irr), 1)
        prob_m = PSMProblem((du, u, p, t) -> (du[1] = p.g(u[1])),
                            [0.25], (0.0, 4.0),
                            [BSplineApproximator(:g, (0.0, 1.2), 5; initial=0.15)];
                            data_times=ts_irr, data_values=y_irr, obs_to_state=[1])

        sol_m = solve(prob_m, MagiSolver(n_samples=40, n_warmup=40, n_gridpoints=15,
                                         obs_var=0.01^2, rng_seed=406))
        gt = sol_m.convergence.grid_times
        oi = sol_m.convergence.obs_indices
        # Every observation timestamp appears EXACTLY on the discretization grid
        for (i, td) in enumerate(ts_irr)
            @test gt[oi[i]] == td
        end
        @test sol_m.convergence.temper ≈ length(gt) / length(ts_irr) atol=1e-12
        @test sol_m.convergence.temper >= 1.0
    end

    # ── 7. IntegralMatchingSolver mean-centering + Simpson quadrature ─
    @testset "IntegralMatchingSolver Dattner-Klaassen mean-centering + Simpson" begin
        rng = StableRNG(407)
        ts = collect(range(0.0, 5.0, length=26))
        u_true = [2.0 / (1.0 + 3.0 * exp(-0.8 * t)) for t in ts]
        y_im = reshape(copy(u_true), :, 1)
        # Perturb boundary point y[1] strongly — mean-centering resists boundary leverage
        y_im[1, 1] += 0.35
        prob_im = PSMProblem((du, u, p, t) -> (du[1] = p.g(u[1])),
                             [0.5], (0.0, 5.0),
                             [BSplineApproximator(:g, (0.0, 2.5), 6; initial=0.2)];
                             data_times=ts, data_values=y_im, obs_to_state=[1])
        sol_im = solve(prob_im, IntegralMatchingSolver(maxiters=400, lr=0.03, lambda_smooth=0.01))
        @test isfinite(sol_im.data_loss)
        # Check recovered g(u) ≈ 0.8 u (1 - u/2) on interior domain
        g_hat = sol_im.unknown_functions[:g]
        g_true(u) = 0.8 * u * (1.0 - u / 2.0)
        rmse = sqrt(mean((g_hat(u) - g_true(u))^2 for u in range(0.7, 1.8, length=20)))
        @test rmse < 0.15
    end

    # ── 8. ODINSolver & FGPGMSolver Matérn-5/2 & standardization ─────
    @testset "ODINSolver & FGPGMSolver Matérn-5/2 & BNG scaling" begin
        rng = StableRNG(408)
        ts = collect(range(0.0, 4.0, length=17))
        sol_lv = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = 0.8 * u[1] * (1 - u[1] / 100.0);
                                         du[2] = 0.4 * (0.5 - u[2])),
                       [10.0, 0.1], (0.0, 4.0)),
            Tsit5(); saveat=ts, abstol=1e-9, reltol=1e-9)
        # Two states spanning 3 orders of magnitude (10..100 vs 0.1..0.5)
        y_ms = hcat(Float64[sol_lv[1, i] for i in eachindex(ts)],
                    Float64[sol_lv[2, i] for i in eachindex(ts)])

        prob_ms = PSMProblem((du, u, p, t) -> (du[1] = p.r(u[1]) * u[1];
                                               du[2] = 0.4 * (0.5 - u[2])),
                             [10.0, 0.1], (0.0, 4.0),
                             [BSplineApproximator(:r, (0.0, 110.0), 6; initial=0.5)];
                             data_times=ts, data_values=y_ms, obs_to_state=[1, 2])

        # ODINSolver with Matérn-5/2 + standardization
        sol_odin = solve(prob_ms, ODINSolver(maxiters=25, kernel=:matern52, standardize=true))
        @test isfinite(sol_odin.data_loss)
        @test sol_odin.data_loss < 50.0

        # FGPGMSolver with Matérn-5/2
        sol_fg = solve(prob_ms, FGPGMSolver(n_samples=30, n_warmup=30, kernel=:matern52, rng_seed=408))
        @test isfinite(sol_fg.data_loss)

        # BNGSolver scale invariance on multi-scale problem
        sol_bng = solve(prob_ms, BNGSolver(k_obs=2, k_proc=2, maxiters=120, rng_seed=408))
        @test isfinite(sol_bng.data_loss)
    end
end
