using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: penalty_matrix, nparams, initial_params, build_evaluator
using PartiallySpecifiedModels: solve, build_evaluator, build_penalty_matrices,
    penalty_matrix, nparams, initial_params, _gcv_score, _coordinate_gcv,
    _parametric_resample!, _smoothing_spline, _log_det_pd, _safe_inv,
    estimate_smoothing_params, log_likelihood, irls_weights, optimize_spde_range,
    check_constraints, residual_acf, confidence_band
using OrdinaryDiffEq, LinearAlgebra, Random, Statistics
import Lux

@testset "v0.3 Bugfixes & Cross-Solver Parity (M1)" begin
    @testset "C1: NeuralApproximator rejects multi-input Dense(2=>...)" begin
        m = Lux.Chain(Lux.Dense(2 => 4, tanh), Lux.Dense(4 => 4, tanh), Lux.Dense(4 => 1))
        na = NeuralApproximator(:f, m; rng_seed=42)
        p = initial_params(na)
        @test_throws ArgumentError build_evaluator(na, p)
    end

    @testset "C2: Masked (NaN, 0.0) with positive-support CustomLikelihood" begin
        ts = collect(0.0:0.5:5.0)
        y = reshape(exp.(0.2 .* ts) .+ 0.01 .* randn(Xoshiro(1), length(ts)), :, 1)
        w = ones(size(y))
        y_m = copy(y); w_m = copy(w)
        y_m[5, 1] = NaN; w_m[5, 1] = 0.0
        pos_lik = CustomLikelihood((yi, mu) -> -(log(yi) - log(max(mu, 1e-6)))^2 / 0.02)
        dyn!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        prob_m = PSMProblem(dyn!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.0, 4.0), 6; initial=0.2)];
            data_times=ts, data_values=y_m, data_weights=w_m, likelihood=pos_lik)
        sol_m = solve(prob_m, LAML(maxiters=40, jac=:forwarddiff))
        @test sol_m.convergence.smoothing_advanced
        @test isfinite(sol_m.objective)
        # Direct estimate_smoothing_params with raw NaN/0-weight matches row-dropped exactly
        J = randn(Xoshiro(2), 20, 5)
        S = penalty_matrix(BSplineApproximator(:a, (0.0, 1.0), 5))
        beta_t = [0.5, 0.7, 0.4, 0.8, 0.6]
        y_clean = J * beta_t .+ 0.05 .* randn(Xoshiro(3), 20)
        y_raw = copy(y_clean); w_raw = ones(20)
        y_raw[3] = NaN; w_raw[3] = 0.0
        keep = [i for i in 1:20 if i != 3]
        lam_m, _ = estimate_smoothing_params(J, w_raw, w_raw, y_raw, J * beta_t,
                                             beta_t, [S], [0], [5], 5)
        lam_d, _ = estimate_smoothing_params(J[keep, :], ones(19), ones(19), y_clean[keep],
                                             (J * beta_t)[keep], beta_t, [S], [0], [5], 5)
        @test isapprox(lam_m, lam_d; rtol=1e-6)
    end

    @testset "C3: Multi-scale equilibrated _log_det_pd & _safe_inv" begin
        S = penalty_matrix(BSplineApproximator(:a, (0.0, 1.0), 6))
        J = randn(Xoshiro(11), 30, 12)
        JWJ = J' * J
        H = copy(JWJ)
        H[1:6, 1:6] .+= 1e12 .* S
        H[7:12, 7:12] .+= 1.0 .* S
        # BigFloat Cholesky reference
        H_bf = BigFloat.(H)
        C_bf = cholesky(Symmetric(H_bf))
        ld_exact = Float64(2 * sum(log, diag(C_bf.U)))
        Hinv_exact = Float64.(inv(C_bf))
        # At λ1 = 1e12 (cond(H) ~ 1e13), Float64 rounding is O(eps * cond) ~ 1e-3
        # (vs +55.4 log|H| error and 99.9% V2 error under the old isotropic ridge).
        @test abs(_log_det_pd(H) - ld_exact) < 1e-2
        V2_exact = diag(Hinv_exact)[7:12]
        V2_got = diag(_safe_inv(H))[7:12]
        @test maximum(abs.(V2_got .- V2_exact) ./ V2_exact) < 1e-3
    end

    @testset "H1: LAML, GCV, Collocation non-Gaussian sol.objective" begin
        ts = collect(0.0:0.5:5.0)
        y = reshape(round.(10.0 .* exp.(0.2 .* ts)), :, 1)
        dyn!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        uf = BSplineApproximator(:r, (0.0, 40.0), 5; initial=0.2)
        prob = PSMProblem(dyn!, [10.0], (0.0, 5.0), [uf];
                          data_times=ts, data_values=y, likelihood=PoissonLikelihood())
        for alg in (LAML(maxiters=20, jac=:forwarddiff),
                    GCVSolver(maxiters=20, jac=:forwarddiff))
            sol = solve(prob, alg)
            S = penalty_matrix(uf)
            β = collect(sol.parameters)
            pen = sol.smoothing_params[1] * dot(β, S * β)
            expected = -log_likelihood(PoissonLikelihood(), vec(y), vec(sol.fitted_values), ones(length(y))) + 0.5 * pen
            @test isapprox(sol.objective, expected; rtol=1e-6, atol=1e-6)
        end
        # CollocationLAML deliberately honors the documented squared-error reporting
        # convention `objective == data_loss + β'S^λβ` (docs/src/solvers.md).
        sol_c = solve(prob, CollocationLAML(maxiters=15, n_continuation=3))
        S_c = penalty_matrix(uf)
        β_c = collect(sol_c.parameters)
        pen_c = sol_c.smoothing_params[1] * dot(β_c, S_c * β_c)
        @test isapprox(sol_c.objective, sol_c.data_loss + pen_c; rtol=1e-6, atol=1e-6)
    end

    @testset "H2/M1/H3/M2/M3/M4: SCOP :dec_positive, COMONet, confidence_band, NegBin, diagnostics" begin
        a = ShapeConstrainedBSplineApproximator(:f, (0.0, 5.0), 8, :dec_positive)
        # Constant positive function f(x)=5 => γ_1..γ_{q-1} very negative (slopes≈0), γ_q large (level=5)
        γ_const = vcat(fill(-20.0, 7), [5.0])
        @test dot(γ_const, penalty_matrix(a) * γ_const) < 1e-10

        c0 = COMONetApproximator(:f, (0.0, 5.0), (4,), :increasing; penalty_weight=0.0, rng_seed=1)
        @test penalty_matrix(c0) === nothing

        ts = collect(0.0:0.5:5.0)
        y = reshape(exp.(0.2 .* ts) .+ 0.01 .* randn(Xoshiro(1), length(ts)), :, 1)
        dyn!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        for approx in (
            COMONetApproximator(:r, (0.0, 5.0), (4,), :increasing; rng_seed=1),
            NeuralApproximator(:r, Lux.Chain(Lux.Dense(1=>4, tanh), Lux.Dense(4=>1));
                               domain=(0.0, 5.0), penalty_weight=0.01, rng_seed=1)
        )
            prob = PSMProblem(dyn!, [1.0], (0.0, 5.0), [approx]; data_times=ts, data_values=y)
            sol = solve(prob, LAML(maxiters=5))
            cb = confidence_band(sol, prob)
            @test haskey(cb, :r) && all(isfinite, cb[:r].se)
        end

        # NegBin negative-mu parity
        nb = NegativeBinomialLikelihood(2.0)
        @test log_likelihood(nb, [1.0], [-0.1], [1.0]) == log_likelihood(nb, [1.0], [0.1], [1.0])

        # Diagnostics
        cc = check_constraints(x -> NaN, :positive, (0.0, 1.0))
        @test !cc.satisfied
        @test residual_acf(Float64[]) == Float64[]
    end

    @testset "GM1/GM2/CL1/GCV1/SPDE1: GradientMatching, Collocation, GCV, SPDE range" begin
        ts = collect(range(0.0, 10.0, length=25))
        y_clean = [10.0 / (1.0 + 9.0 * exp(-0.5 * t)) for t in ts]
        y_mat = reshape(y_clean .+ 0.05 .* randn(Xoshiro(42), length(ts)), :, 1)
        dyn!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        prob_gm = PSMProblem(dyn!, [1.0], (0.0, 10.0),
                             [BSplineApproximator(:r, (0.0, 12.0), 8; initial=0.0)];
                             data_times=ts, data_values=y_mat)
        sol_gm = solve(prob_gm, GradientMatching(maxiters=50))
        @test sol_gm.convergence.iterations > 2
        @test sol_gm.convergence.reason != :line_search_failure

        # _smoothing_spline on wide gap
        ts_gap = vcat(collect(range(0.0, 1.5, length=10)), collect(range(8.5, 10.0, length=10)))
        val_fn, der_fn = _smoothing_spline(ts_gap, sin.(ts_gap))
        @test isfinite(val_fn(5.0)) && isfinite(der_fn(5.0))

        # CollocationLAML discrete dt=2.0
        ts_d = collect(0.0:2.0:10.0)
        y_d = reshape(1.0 .+ 0.25 .* ts_d, :, 1)
        prob_d = PSMProblem((un, u, p, t) -> (un[1] = u[1] + p.r(u[1]); nothing),
                            [1.0], (0.0, 10.0), [BSplineApproximator(:r, (0.0, 5.0), 5; initial=0.25)];
                            data_times=ts_d, data_values=y_d, discrete=true)
        sol_c = solve(prob_d, CollocationLAML(maxiters=20, n_continuation=3))
        @test isapprox(sol_c.unknown_functions[:r](2.0), 0.25; atol=0.03)

        # GCV masked NaN + small scale
        g, bh, _, _ = _gcv_score([1.0 0.0; 0.0 1.0], [1.0, 0.0], [2.0, NaN], zeros(2, 2), 1, 1.4)
        @test all(isfinite, bh)
        S1 = penalty_matrix(BSplineApproximator(:a, (0.0, 1.0), 5))
        S2 = penalty_matrix(BSplineApproximator(:b, (0.0, 1.0), 5))
        J = randn(Xoshiro(5), 30, 10)
        z_small = 1e-7 .* (J * vcat(ones(5), [0.0, 1.0, -2.0, 1.0, 0.0]) .+ 0.01 .* randn(Xoshiro(6), 30))
        rho_s, _, _, _ = _coordinate_gcv(S_lam -> _gcv_score(J, ones(30), z_small, S_lam, 30, 1.4),
                                         [S1, S2], [0, 5], [5, 5], 10, [0.0, 0.0], 1e-3)
        @test rho_s != [0.0, 0.0]
    end

    @testset "PB1/PR1/PR3/BY1/UD1/UD2: Rodeo scale-invariance, ProfileLik, TruncNorm boot, MCMC, MS, DerivFree" begin
        # Rodeo scaling invariance under c=1 vs c=100
        lams = Float64[]
        for c in (1.0, 100.0)
            dyn_s!(du, u, p, t) = (du[1] = p.r(u[1]/p.K) * u[1])
            ts = collect(0.0:0.5:8.0)
            st = OrdinaryDiffEq.solve(ODEProblem((du,u,p,t)->(du[1]=0.5*(u[1]/c)/(0.3+u[1]/c)*u[1]),
                                                 [0.1*c], (0.0,8.0)), Tsit5(); saveat=ts)
            rng = Xoshiro(7)
            y = reshape([u[1] + 0.02*c*randn(rng) for u in st.u], :, 1)
            prob = PSMProblem(dyn_s!, [0.1*c], (0.0, 8.0),
                              [BSplineApproximator(:r, (0.0, 1.5), 6; initial=0.3)];
                              data_times=ts, data_values=y, known_params=(K=c,))
            s = solve(prob, RodeoSolver(n_steps=60, maxiters=20, obs_var=(0.02*c)^2))
            push!(lams, s.smoothing_params[1])
        end
        @test abs(log10(lams[2] / lams[1])) < 1.0

        # ProfileLikelihoodSolver large scale + σ=10 width
        ts = collect(0.0:0.5:5.0)
        dyn_lin!(du, u, p, t) = (du[1] = p.r(t); nothing)
        y_big = reshape(1e6 .* (1.0 .+ 0.5 .* ts) .+ 1e5 .* randn(Xoshiro(3), length(ts)), :, 1)
        prob_big = PSMProblem(dyn_lin!, [1e6], (0.0, 5.0),
                              [BSplineApproximator(:r, (0.0, 5.0), 4; initial=5e5)];
                              data_times=ts, data_values=y_big)
        sol_pl = solve(prob_big, ProfileLikelihoodSolver(n_profile_points=7, param_indices=[1]))
        @test isfinite(sol_pl.convergence.profiles[1].ci[1]) && isfinite(sol_pl.convergence.profiles[1].ci[2])

        # TruncatedNormal bootstrap Robert (1995) tail accuracy
        yb = zeros(2000, 1)
        _parametric_resample!(yb, TruncatedNormalLikelihood(0.0, 0.2), fill(-0.8, 2000, 1),
                              [0.2], ones(2000, 1), 2000, 1, Xoshiro(1))
        @test mean(yb .== 0.0) < 0.01
        @test isapprox(mean(yb), 0.0476; atol=0.01)

        # MultipleShooting u0==0 & 2-row mask
        ts_ms = collect(0.0:0.5:5.0)
        y_ms = reshape(2.0 .* (1.0 .- exp.(-0.5 .* ts_ms)), :, 1)
        prob0 = PSMProblem((du,u,p,t)->(du[1]=p.f(t)-0.5*u[1]), [0.0], (0.0,5.0),
                           [BSplineApproximator(:f,(0.0,5.0),5; initial=1.0)]; data_times=ts_ms, data_values=y_ms)
        s0 = solve(prob0, MultipleShootingSolver(n_intervals=4, maxiters_outer=4, maxiters_inner=30))
        @test s0.convergence.converged

        # DerivativeFreeSolver false converged on exploding map
        ts_e = collect(0.0:1.0:10.0)
        prob_e = PSMProblem((un,u,p,t)->(un[1]=1e6*u[1]^3 + p.r(t); nothing), [2.0], (0.0,10.0),
                            [BSplineApproximator(:r,(0.0,10.0),4; initial=0.1)];
                            data_times=ts_e, data_values=reshape(ones(length(ts_e)), :, 1), discrete=true)
        se = solve(prob_e, DerivativeFreeSolver(maxiters=50))
        @test !se.convergence.converged && se.convergence.reason == :dynamics_failure
    end

    @testset "P1/P2: u0::Function parity & strict Gaussian likelihood guards" begin
        ts = collect(0.0:1.0:5.0)
        y = reshape(exp.(0.2 .* ts) .+ 0.01 .* randn(Xoshiro(1), length(ts)), :, 1)
        dyn!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        uf = BSplineApproximator(:r, (0.0, 5.0), 5; initial=0.2)
        prob_fn = PSMProblem(dyn!, p -> [max(0.5, 1.0 + 0.05 * p.r(1.0))], (0.0, 5.0), [uf];
                             data_times=ts, data_values=y)
        for alg in (CollocationLAML(maxiters=5, n_continuation=2),
                    GradientMatching(maxiters=5),
                    TwoStageSolver(maxiters=5),
                    AdaptiveGradientMatching(maxiters=5),
                    BNGSolver(k_obs=1, k_proc=1, maxiters=5, rng_seed=1),
                    RodeoSolver(n_steps=20, maxiters=2),
                    DaltonSolver(n_steps=20, maxiters=2),
                    PseudoMarginalSolver(n_samples=5, n_warmup=5, n_steps=20, rng_seed=1),
                    FGPGMSolver(n_samples=5, n_warmup=5, rng_seed=1),
                    ODINSolver(maxiters=5),
                    RKHSSolver(maxiters=5),
                    IntegralMatchingSolver(maxiters=5),
                    MagiSolver(n_samples=5, n_warmup=5, n_gridpoints=10, rng_seed=1))
            sol = solve(prob_fn, alg)
            @test sol isa PSMSolution
        end

        # Strict ArgumentError on unsupported non-Gaussian likelihood
        prob_pois = PSMProblem(dyn!, [1.0], (0.0, 5.0), [uf];
                               data_times=ts, data_values=round.(10 .* y), likelihood=PoissonLikelihood())
        for alg in (GradientMatching(maxiters=5), TwoStageSolver(maxiters=5),
                    AdaptiveGradientMatching(maxiters=5), FGPGMSolver(n_samples=5, n_warmup=5),
                    ODINSolver(maxiters=5), RKHSSolver(maxiters=5), IntegralMatchingSolver(maxiters=5),
                    BNGSolver(k_obs=1, k_proc=1, maxiters=5), RodeoSolver(n_steps=20, maxiters=2),
                    DaltonSolver(n_steps=20, maxiters=2), PseudoMarginalSolver(n_samples=5, n_warmup=5, n_steps=20),
                    MagiSolver(n_samples=5, n_warmup=5, n_gridpoints=10), EnsembleKalmanSolver(n_ensemble=10, n_iterations=2))
            @test_throws ArgumentError solve(prob_pois, alg)
        end
    end
end
