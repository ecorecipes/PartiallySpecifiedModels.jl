using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using LinearAlgebra
using MCMCChains
using Random
using OrdinaryDiffEq

@testset "PartiallySpecifiedModels.jl" begin

    @testset "Spline penalty matrix" begin
        # Uniform knots
        x = collect(range(0.0, 1.0, length=10))
        S = spline_penalty_matrix(x)
        @test size(S) == (10, 10)
        @test maximum(abs.(S .- S')) < 1e-10  # numerically symmetric
        # S should be positive semi-definite with rank n-2 = 8
        evals = eigvals(Symmetric(S))
        @test count(e -> e > 1e-10, evals) == 8
        @test all(e -> e >= -1e-10, evals)
        # Penalty of linear function should be zero
        y_linear = collect(range(1.0, 5.0, length=10))
        @test dot(y_linear, S * y_linear) < 1e-10
    end

    @testset "BSpline approximator" begin
        a = BSplineApproximator(:f, (0.0, 1.0), 10)
        @test nparams(a) == 10
        @test a.name == :f
        @test a.domain == (0.0, 1.0)

        p0 = initial_params(a)
        @test length(p0) == 10
        @test all(p0 .== 0.0)

        S = penalty_matrix(a)
        @test size(S) == (10, 10)

        # With initial function
        a2 = BSplineApproximator(:g, (0.0, 1.0), 5; initial=x -> x^2)
        p2 = initial_params(a2)
        @test length(p2) == 5
        @test p2[end] ≈ 1.0  # x=1.0 → 1.0
    end

    @testset "Likelihood families" begin
        y = [1.0, 2.0, 3.0]
        mu = [1.1, 2.1, 2.9]
        w = [1.0, 1.0, 1.0]

        # Gaussian
        ll = PartiallySpecifiedModels.log_likelihood(Gaussian(), y, mu, w)
        @test ll < 0.0
        wt = PartiallySpecifiedModels.irls_weights(Gaussian(), y, mu, w)
        @test wt == w

        # Poisson
        y_p = [5.0, 10.0, 3.0]
        mu_p = [4.5, 11.0, 3.2]
        ll_p = PartiallySpecifiedModels.log_likelihood(Poisson(), y_p, mu_p, w)
        @test isfinite(ll_p)
        wt_p = PartiallySpecifiedModels.irls_weights(Poisson(), y_p, mu_p, w)
        @test all(wt_p .> 0)
        @test wt_p ≈ 1.0 ./ mu_p  # identity link: W = 1/μ

        # NegBin
        ll_nb = PartiallySpecifiedModels.log_likelihood(NegativeBinomial(5.0), y_p, mu_p, w)
        @test isfinite(ll_nb)
    end

    @testset "CustomLikelihood with ForwardDiff" begin
        # Custom Gaussian should match built-in
        custom_gauss = CustomLikelihood((y, μ) -> -0.5 * (y - μ)^2)
        y = [1.0, 2.0, 3.0]
        mu = [1.1, 2.1, 2.9]
        w = [1.0, 1.0, 1.0]

        ll_builtin = PartiallySpecifiedModels.log_likelihood(Gaussian(), y, mu, w)
        ll_custom = PartiallySpecifiedModels.log_likelihood(custom_gauss, y, mu, w)
        @test ll_builtin ≈ ll_custom

        wt = PartiallySpecifiedModels.irls_weights(custom_gauss, y, mu, w)
        @test all(wt .≈ 1.0)  # Gaussian has weight = 1
    end

    @testset "LAML helpers" begin
        # Use a matrix with known positive determinant
        S = Float64[2 1; 1 3]  # det = 5
        ld = PartiallySpecifiedModels._log_det_pd(S)
        @test isfinite(ld)
        @test ld ≈ log(5.0) atol=1e-8

        r = PartiallySpecifiedModels._rank_penalty(S)
        @test r == 2

        ldp = PartiallySpecifiedModels._log_det_plus(S)
        @test ldp ≈ ld atol=0.01

        # Singular matrix: rank should be less than size
        S_sing = [1.0 1.0; 1.0 1.0]
        @test PartiallySpecifiedModels._rank_penalty(S_sing) == 1
    end

    @testset "Simple ODE fit" begin
        # Exponential growth: du/dt = r*u, data = u0*exp(r*t)
        # Unknown function: r(u) ≈ constant
        true_r = 0.1
        u0_val = 1.0
        tspan = (0.0, 10.0)
        data_times = collect(0.0:0.5:10.0)
        true_data = u0_val .* exp.(true_r .* data_times)
        # Add small noise
        data_values = true_data .+ 0.01 .* randn(length(data_times))
        data_values = reshape(data_values, :, 1)

        function exp_growth!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        approx_r = BSplineApproximator(:r, (0.0, 5.0), 5;
                                        initial=x -> 0.05)

        prob = PSMProblem(exp_growth!, [u0_val], tspan, [approx_r];
                          data_times=data_times,
                          data_values=data_values,
                          obs_to_state=[1],
                          likelihood=Gaussian(),
                          solver=Tsit5())

        sol = solve(prob, LAML(maxiters=30, verbose=false))

        @test sol.edf > 1.0
        @test sol.data_loss < 1.0  # Should fit well
        @test haskey(sol.unknown_functions, :r)

        # Check that the fitted r function is approximately constant at true_r
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.05
    end

    @testset "CollocationLAML solver" begin
        # Simple exponential growth with unknown growth rate
        function growth!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, CollocationLAML(
            maxiters=20, verbose=false,
            lambda_ode_start=0.01, lambda_ode_end=100.0,
            n_continuation=4))

        @test sol.data_loss < 5.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
    end

    @testset "GPApproximator" begin
        # Kernel matrix is positive definite
        gp = GPApproximator(:f, (0.0, 1.0), 8; kernel=:matern52)
        @test nparams(gp) == 8
        @test gp.name == :f

        S = penalty_matrix(gp)
        @test size(S) == (8, 8)
        evals = eigvals(Symmetric(S))
        @test all(e -> e > -1e-8, evals)  # positive semi-definite

        # GP evaluator interpolates function values
        vals = Float64[sin(x) for x in gp.inducing_points]
        eval_gp = PartiallySpecifiedModels.build_gp_evaluator(gp, vals)
        for (xi, vi) in zip(gp.inducing_points, vals)
            @test abs(eval_gp(xi) - vi) < 0.05
        end

        # Different kernels
        for kern in [:sqexp, :matern32, :matern52]
            gp_k = GPApproximator(:g, (0.0, 5.0), 6; kernel=kern)
            @test nparams(gp_k) == 6
            @test penalty_matrix(gp_k) !== nothing
        end
    end

    @testset "GP solver integration" begin
        function growth_gp!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_gp!, [1.0], tspan,
            [GPApproximator(:r, (0.5, 5.0), 6; kernel=:matern52, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.1
    end

    @testset "SPDEApproximator — construction" begin
        a = SPDEApproximator(:f, (0.0, 1.0), 10)
        @test nparams(a) == 10
        @test a.name == :f
        @test a.domain == (0.0, 1.0)
        @test a.nu ≈ 1.5
        @test a.range_param ≈ 1.0 / 3.0
        @test a.kappa ≈ sqrt(8.0 * 1.5) / (1.0 / 3.0)
        @test length(a.mesh_points) == 10

        p0 = initial_params(a)
        @test length(p0) == 10
        @test all(p0 .== 0.0)

        # With initial function
        a2 = SPDEApproximator(:g, (0.0, 1.0), 5; initial=x -> x^2)
        p2 = initial_params(a2)
        @test length(p2) == 5
        @test p2[1] ≈ 0.0 atol=1e-10
        @test p2[end] ≈ 1.0 atol=1e-10

        # Custom range
        a3 = SPDEApproximator(:h, (0.0, 10.0), 8; nu=0.5, range_param=2.0)
        @test a3.nu ≈ 0.5
        @test a3.range_param ≈ 2.0
        @test a3.kappa ≈ sqrt(4.0) / 2.0

        # Validation
        @test_throws ErrorException SPDEApproximator(:f, (0.0, 1.0), 10; nu=0.7)
        @test_throws ErrorException SPDEApproximator(:f, (0.0, 1.0), 2)
    end

    @testset "SPDEApproximator — FEM matrices" begin
        mesh = collect(range(0.0, 1.0, length=5))
        C, G = PartiallySpecifiedModels.spde_fem_matrices(mesh)
        h = 0.25

        # Mass matrix: diagonal with h/2 at boundaries, h at interior
        @test C[1,1] ≈ h/2
        @test C[2,2] ≈ h
        @test C[5,5] ≈ h/2
        @test all(C[i,j] ≈ 0 for i in 1:5, j in 1:5 if i ≠ j)

        # Stiffness matrix: tridiagonal
        @test G[1,1] ≈ 1/h
        @test G[1,2] ≈ -1/h
        @test G[2,2] ≈ 2/h
        @test G[2,1] ≈ -1/h
        @test G[2,3] ≈ -1/h
        @test maximum(abs.(G .- G')) < 1e-10  # symmetric

        # Stiffness matrix annihilates constant functions: G * ones = 0
        @test norm(G * ones(5)) < 1e-10
    end

    @testset "SPDEApproximator — penalty matrix" begin
        for ν in [0.5, 1.5, 2.5]
            a = SPDEApproximator(:f, (0.0, 1.0), 10; nu=ν, range_param=0.3)
            S = penalty_matrix(a)
            @test size(S) == (10, 10)
            @test maximum(abs.(S .- S')) < 1e-8  # symmetric
            evals = eigvals(Symmetric(S))
            @test all(e -> e >= -1e-8, evals)  # positive semi-definite
        end

        # Penalty of constant function should be small (not exactly zero due to
        # mass matrix contribution, but much smaller than wiggly function)
        a = SPDEApproximator(:f, (0.0, 1.0), 10; nu=1.5)
        S = penalty_matrix(a)
        y_const = ones(10)
        y_wiggly = [sin(10π * x) for x in a.mesh_points]
        @test dot(y_const, S * y_const) < dot(y_wiggly, S * y_wiggly)
    end

    @testset "SPDEApproximator — evaluator" begin
        mesh = collect(range(0.0, 1.0, length=10))
        params = sin.(mesh)
        eval_spde = PartiallySpecifiedModels.build_spde_evaluator(mesh, params)

        # Should interpolate at mesh points
        for (xi, vi) in zip(mesh, params)
            @test abs(eval_spde(xi) - vi) < 1e-10
        end

        # Should interpolate smoothly between points
        x_mid = 0.55
        @test abs(eval_spde(x_mid) - sin(x_mid)) < 0.05
    end

    @testset "SPDEApproximator — LAML solver" begin
        function growth_spde!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_spde!, [1.0], tspan,
            [SPDEApproximator(:r, (0.5, 5.0), 8; nu=1.5, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.1
    end

    @testset "ShapeConstrainedSPDEApproximator — construction" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :increasing)
        @test a.name == :f
        @test a.domain == (0.0, 5.0)
        @test a.n_basis == 8
        @test a.nu == 1.5
        @test a.constraint == :increasing
        @test size(a.Sigma) == (8, 8)
        @test nparams(a) == 8
        @test length(initial_params(a)) == 8

        # With custom settings
        a2 = ShapeConstrainedSPDEApproximator(:g, (0.0, 1.0), 6, :dec_positive;
            nu=0.5, range_param=0.5, initial=x -> 1.0 - x)
        @test a2.nu == 0.5
        @test a2.range_param == 0.5
        @test nparams(a2) == 6

        # Zero-endpoint constraint: one fewer parameter
        a3 = ShapeConstrainedSPDEApproximator(:h, (0.0, 1.0), 8, :inc_zero_left)
        @test nparams(a3) == 7
        @test size(a3.Sigma) == (8, 7)

        # Errors
        @test_throws ArgumentError ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :bad_constraint)
        @test_throws ErrorException ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 3, :increasing)
    end

    @testset "ShapeConstrainedSPDEApproximator — evaluator" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :increasing;
            initial=x -> 0.1 * x)
        gamma = initial_params(a)
        eval_fn = PartiallySpecifiedModels.build_constrained_spde_evaluator(a, gamma)

        # Should be callable and increasing
        vals = [eval_fn(x) for x in range(0.0, 5.0, length=20)]
        @test all(diff(vals) .>= -1e-10)  # Approximately increasing

        # Positive constraint
        a_pos = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :positive;
            initial=x -> 1.0)
        gamma_pos = initial_params(a_pos)
        eval_pos = PartiallySpecifiedModels.build_constrained_spde_evaluator(a_pos, gamma_pos)
        mesh_vals = PartiallySpecifiedModels.gamma_to_mesh_values(a_pos, gamma_pos)
        @test all(mesh_vals .> 0)  # Positive at mesh nodes
    end

    @testset "ShapeConstrainedSPDEApproximator — penalty matrix" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :increasing; nu=1.5)
        P = penalty_matrix(a)
        np = nparams(a)
        @test size(P) == (np, np)
        @test issymmetric(P)
        @test all(eigvals(Symmetric(P)) .>= -1e-10)  # Positive semi-definite

        # Zero-endpoint has smaller penalty
        a_z = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :inc_zero_left; nu=1.5)
        P_z = penalty_matrix(a_z)
        @test size(P_z) == (7, 7)
    end

    @testset "ShapeConstrainedSPDEApproximator — LAML solver" begin
        function growth_scspde!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        # Use :positive constraint since r(u) = 0.3 > 0
        prob = PSMProblem(growth_scspde!, [1.0], tspan,
            [ShapeConstrainedSPDEApproximator(:r, (0.5, 5.0), 8, :positive;
                nu=1.5, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
        # Verify positivity at mesh nodes
        mesh_vals = PartiallySpecifiedModels.gamma_to_mesh_values(
            prob.approximators[1], collect(sol.parameters[:r]))
        @test all(mesh_vals .> 0)
    end

    @testset "GradientMatching solver" begin
        function growth_gm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_gm!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, GradientMatching(maxiters=50, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
    end

    @testset "Adam solver (B-spline)" begin
        function growth_adam!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_adam!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, AdamSolver(maxiters=150, lr=0.01, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
        @test sol.data_loss < 1.0  # should fit exponential growth well
    end

    @testset "Multiple shooting solver (B-spline)" begin
        function growth_ms!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_ms!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, MultipleShootingSolver(
            n_intervals=3, maxiters_inner=50, maxiters_outer=5,
            rho_init=1.0, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
    end

    @testset "Adaptive gradient matching (B-spline)" begin
        # Exponential growth: du/dt = r(x)*u, r(x) ≈ constant
        true_r = 0.3
        function exp_growth_agm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        u0_exp = [2.0]; tspan_exp = (0.0, 5.0)
        data_t_exp = collect(0.0:0.25:5.0)
        data_true_exp = 2.0 .* exp.(true_r .* data_t_exp)
        data_vals_exp = reshape(data_true_exp .+ 0.05 .* randn(length(data_t_exp)),
                                :, 1)

        bs_r = BSplineApproximator(:r, (1.5, 10.0), 6; initial=0.5)
        prob = PSMProblem(exp_growth_agm!, u0_exp, tspan_exp, [bs_r];
            data_times=data_t_exp, data_values=data_vals_exp,
            obs_to_state=[1], solver=Tsit5())

        sol = solve(prob, AdaptiveGradientMatching(maxiters=100, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(3.0) - true_r) < 0.15
    end

    @testset "Rodeo solver (B-spline)" begin
        # Exponential growth: du/dt = r(x)*u, r(x) ≈ constant
        true_r = 0.3
        function exp_growth_rodeo!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        u0_exp = [2.0]; tspan_exp = (0.0, 5.0)
        data_t_exp = collect(0.0:0.25:5.0)
        data_true_exp = 2.0 .* exp.(true_r .* data_t_exp)
        data_vals_exp = reshape(data_true_exp .+ 0.05 .* randn(length(data_t_exp)),
                                :, 1)

        bs_r = BSplineApproximator(:r, (1.5, 10.0), 6; initial=0.5)
        prob = PSMProblem(exp_growth_rodeo!, u0_exp, tspan_exp, [bs_r];
            data_times=data_t_exp, data_values=data_vals_exp,
            obs_to_state=[1], solver=Tsit5())

        sol = solve(prob, RodeoSolver(n_steps=100, n_deriv=3, maxiters=100,
                    obs_var=0.01, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(3.0) - true_r) < 0.15
    end

    @testset "MCMCSolver (B-spline)" begin
        # Exponential decay: du/dt = -r(t)*u, r(t) ≈ 0.3
        function exp_decay_mcmc!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_mcmc = collect(0.0:0.5:10.0)
        true_sol_mcmc = exp.(-0.3 .* times_mcmc)
        data_mcmc = reshape(true_sol_mcmc .+ 0.02 .* randn(length(times_mcmc)), :, 1)

        bs_mcmc = BSplineApproximator(:r, (0.0, 10.0), 8; initial=0.3)
        prob_mcmc = PSMProblem(
            ODEProblem(exp_decay_mcmc!, [1.0], (0.0, 10.0)),
            [bs_mcmc]; data_times=times_mcmc, data_values=data_mcmc,
            obs_to_state=[1], solver=Tsit5())

        sol_mcmc = solve(prob_mcmc, MCMCSolver(
            n_samples=50, n_warmup=25, verbose=false))

        @test sol_mcmc.convergence isa MCMCChains.Chains
        @test size(sol_mcmc.convergence, 1) == 50   # n_samples
        @test size(sol_mcmc.convergence, 2) == 9     # 8 params + log_σ
        @test haskey(sol_mcmc.unknown_functions, :r)
        r_map = sol_mcmc.unknown_functions[:r](5.0)
        @test abs(r_map - 0.3) < 0.2
    end

    @testset "MagiSolver (B-spline)" begin
        # Exponential decay: du/dt = -r(t)*u, r(t) ≈ 0.3
        function exp_decay_magi!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_magi = collect(0.0:1.0:10.0)
        true_sol_magi = exp.(-0.3 .* times_magi)
        data_magi = reshape(true_sol_magi, :, 1)

        bs_magi = BSplineApproximator(:r, (0.0, 10.0), 6; initial=0.3)
        prob_magi = PSMProblem(exp_decay_magi!, [1.0], (0.0, 10.0), [bs_magi];
            data_times=times_magi, data_values=data_magi,
            obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        sol_magi = solve(prob_magi, MagiSolver(
            n_samples=50, n_warmup=50, n_gridpoints=50,
            obs_var=0.01, verbose=false))

        # Returns chains in convergence field
        @test sol_magi.convergence isa NamedTuple
        @test sol_magi.convergence.chains isa MCMCChains.Chains
        @test size(sol_magi.convergence.chains, 1) == 50  # n_samples
        @test size(sol_magi.convergence.chains, 2) == 6   # 6 B-spline params
        @test haskey(sol_magi.unknown_functions, :r)
    end

    # ─── Discrete-time model tests ─────────────────────────────────

    @testset "Discrete-time: Ricker model with LAML" begin
        # Ricker model: N[t+1] = N[t] * exp(r * (1 - N[t]/K))
        # Unknown: density dependence f(N) where N[t+1] = N[t] * exp(f(N[t]))
        # True: f(N) = r * (1 - N/K)

        true_r = 0.8
        true_K = 100.0

        function ricker_true!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(true_r * (1.0 - N / true_K))
        end

        # Generate data
        N0 = [10.0]
        tspan = (0.0, 30.0)
        n_steps = Int(tspan[2] - tspan[1])
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:n_steps
            ricker_true!(u_next, u, nothing, Float64(i-1))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 2.0 .* randn(length(times))
        data = max.(data, 0.1)

        # PSM: unknown f(N) where N[t+1] = N[t] * exp(f(N[t]))
        function ricker_psm!(u_next, u, p, t)
            N = u[1]
            f_N = p.f(N)
            u_next[1] = N * exp(f_N)
        end

        uf = BSplineApproximator(:f, (0.0, 150.0), 8;
                                  initial=x -> 0.5 * (1.0 - x / 100.0))

        prob = PSMProblem(ricker_psm!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, LAML(maxiters=50, verbose=false))
        @test sol.data_loss < sum((data .- N_true).^2) * 5  # reasonable fit

        f_eval = sol.unknown_functions[:f]
        # At N=50 (half K), true f = 0.8*(1-50/100) = 0.4
        @test abs(f_eval(50.0) - 0.4) < 0.3
    end

    @testset "Discrete-time: Beverton-Holt with AdamSolver" begin
        # Beverton-Holt: N[t+1] = r * N[t] / (1 + (r-1)/K * N[t])
        # Unknown: f(N) where N[t+1] = f(N[t])
        true_r = 2.0
        true_K = 500.0

        function bh_true!(u_next, u, p, t)
            N = u[1]
            u_next[1] = true_r * N / (1.0 + (true_r - 1.0) / true_K * N)
        end

        N0 = [50.0]
        tspan = (0.0, 20.0)
        n_steps = Int(tspan[2])
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:n_steps
            bh_true!(u_next, u, nothing, Float64(i-1))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 5.0 .* randn(length(times))
        data = max.(data, 1.0)

        # PSM with directly unknown map f(N) = N[t+1]
        function bh_psm!(u_next, u, p, t)
            N = u[1]
            u_next[1] = p.f(N)
        end

        uf = BSplineApproximator(:f, (0.0, 600.0), 10;
                                  initial=x -> x)  # identity initial guess

        prob = PSMProblem(bh_psm!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, AdamSolver(maxiters=500, lr=0.005, verbose=false))
        @test haskey(sol.unknown_functions, :f)

        f_eval = sol.unknown_functions[:f]
        # At N=250, true f = 2*250/(1 + 1/500*250) = 500/1.5 = 333.3
        @test abs(f_eval(250.0) - 333.3) < 100
    end

    @testset "Discrete-time: GradientMatching" begin
        # Simple exponential growth: N[t+1] = r*N[t], unknown f(N) = r*N
        true_r = 1.05
        N0 = [10.0]
        tspan = (0.0, 15.0)
        times = collect(0.0:1.0:tspan[2])
        N_true = N0[1] .* true_r .^ (0:15)
        data = N_true .+ 0.5 .* randn(length(times))
        data = max.(data, 0.1)

        function exp_growth!(u_next, u, p, t)
            u_next[1] = p.f(u[1])
        end

        uf = BSplineApproximator(:f, (0.0, 25.0), 6; initial=x -> x)

        prob = PSMProblem(exp_growth!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, GradientMatching(maxiters=100, verbose=false))
        @test haskey(sol.unknown_functions, :f)

        f_eval = sol.unknown_functions[:f]
        # At N=10, true f = 10.5; at N=15, true f = 15.75
        @test abs(f_eval(10.0) - 10.5) < 2.0
    end

    @testset "Discrete-time: CollocationLAML" begin
        # Ricker model with collocation
        true_r = 0.5
        true_K = 80.0
        N0 = [20.0]
        tspan = (0.0, 25.0)
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:Int(tspan[2])
            u_next[1] = u[1] * exp(true_r * (1.0 - u[1] / true_K))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 1.5 .* randn(length(times))
        data = max.(data, 0.1)

        function ricker_coll!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(p.f(N))
        end

        uf = BSplineApproximator(:f, (0.0, 120.0), 8;
                                  initial=x -> 0.3 * (1.0 - x / 80.0))

        prob = PSMProblem(ricker_coll!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, CollocationLAML(maxiters=30, verbose=false))
        @test haskey(sol.unknown_functions, :f)
        @test sol.data_loss < Inf
    end

    # ─── SciML problem type constructors ───────────────────────────

    @testset "PSMProblem from ODEProblem" begin
        # Same exponential growth as "Simple ODE fit" but via ODEProblem
        true_r = 0.3
        function exp_ode!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        ode = ODEProblem(exp_ode!, [1.0], (0.0, 5.0))
        approx_r = BSplineApproximator(:r, (0.0, 10.0), 6)
        dt = collect(0:0.5:5)
        y_data = exp.(true_r .* dt) .+ [0.01*randn() for _ in dt]
        prob = PSMProblem(ode, [approx_r];
            data_times=dt, data_values=reshape(y_data, :, 1))

        @test prob.discrete == false
        @test prob.ode_solver isa typeof(Tsit5())

        sol = solve(prob, LAML(verbose=false))
        @test haskey(sol.unknown_functions, :r)
        @test abs(sol.unknown_functions[:r](1.0) - true_r) < 0.15
    end

    @testset "PSMProblem from DiscreteProblem" begin
        # Ricker model via DiscreteProblem
        true_r = 0.5; true_K = 100.0
        function ricker_disc!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(p.g(N))
            nothing
        end
        disc = DiscreteProblem(ricker_disc!, [20.0], (0.0, 30.0))
        uf_g = BSplineApproximator(:g, (0.0, 150.0), 8)

        g_true(N) = true_r * (1 - N/true_K)
        N_data = zeros(31); N_data[1] = 20.0
        for t in 1:30
            N_data[t+1] = max(N_data[t] * exp(g_true(N_data[t])) + 0.5*randn(), 1.0)
        end
        times = Float64.(0:30)

        prob = PSMProblem(disc, [uf_g];
            data_times=times, data_values=reshape(N_data, :, 1))

        @test prob.discrete == true
        @test prob.ode_solver === nothing

        sol = solve(prob, LAML(verbose=false))
        @test haskey(sol.unknown_functions, :g)
        @test abs(sol.unknown_functions[:g](50.0) - g_true(50.0)) < 0.2
    end

    @testset "PSMProblem from out-of-place ODEProblem" begin
        true_r = 0.3
        exp_oop(u, p, t) = [p.r(u[1]) * u[1]]
        ode = ODEProblem(exp_oop, [1.0], (0.0, 5.0))
        approx_r = BSplineApproximator(:r, (0.0, 10.0), 6)
        dt = collect(0:0.5:5)
        y_data = exp.(true_r .* dt) .+ [0.01*randn() for _ in dt]
        prob = PSMProblem(ode, [approx_r];
            data_times=dt, data_values=reshape(y_data, :, 1))

        @test prob.discrete == false
        sol = solve(prob, LAML(verbose=false))
        @test abs(sol.unknown_functions[:r](1.0) - true_r) < 0.15
    end

    @testset "PSMProblem from out-of-place DiscreteProblem" begin
        true_r = 0.5; true_K = 100.0
        ricker_oop(u, p, t) = [u[1] * exp(p.g(u[1]))]
        disc = DiscreteProblem(ricker_oop, [20.0], (0.0, 30.0))
        uf_g = BSplineApproximator(:g, (0.0, 150.0), 8)

        g_true(N) = true_r * (1 - N/true_K)
        N_data = zeros(31); N_data[1] = 20.0
        for t in 1:30
            N_data[t+1] = max(N_data[t] * exp(g_true(N_data[t])) + 0.5*randn(), 1.0)
        end
        times = Float64.(0:30)

        prob = PSMProblem(disc, [uf_g];
            data_times=times, data_values=reshape(N_data, :, 1))

        @test prob.discrete == true
        sol = solve(prob, LAML(verbose=false))
        @test abs(sol.unknown_functions[:g](50.0) - g_true(50.0)) < 0.2
    end

    # ─── Shape-constrained B-spline approximator ──────────────────
    @testset "ShapeConstrainedBSplineApproximator — construction" begin
        using PartiallySpecifiedModels: _build_sigma_matrix, _softplus,
            gamma_to_knot_values, build_constrained_bspline_evaluator

        # All constraint types should construct without error
        for c in SHAPE_CONSTRAINTS
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c; initial=0.5)
            np = nparams(a)
            ip = initial_params(a)
            S = penalty_matrix(a)
            @test length(ip) == np
            @test size(S) == (np, np)
        end

        # Zero-endpoint constraints have nknots-1 params
        for c in (:inc_zero_left, :dec_zero_right, :inc_zero_right, :dec_zero_left)
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c)
            @test nparams(a) == 7
            @test size(a.Sigma) == (8, 7)
        end

        # Square constraints have nknots params
        for c in (:increasing, :decreasing, :convex, :concave, :positive, :dec_positive)
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c)
            @test nparams(a) == 8
            @test size(a.Sigma) == (8, 8)
        end

        # Invalid constraint throws
        @test_throws ArgumentError ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, :invalid)
        @test_throws ArgumentError ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 3, :increasing)
    end

    @testset "ShapeConstrainedBSplineApproximator — shape enforcement" begin
        using PartiallySpecifiedModels: gamma_to_knot_values

        xs = range(0.0, 1.0, length=50)

        # :increasing — knot values should be monotonically increasing
        a_inc = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :increasing)
        gamma = randn(10)
        beta = gamma_to_knot_values(a_inc, gamma)
        @test all(diff(beta) .> 0)  # strictly increasing (softplus > 0)

        # :decreasing — knot values should be monotonically decreasing
        a_dec = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :decreasing)
        beta_d = gamma_to_knot_values(a_dec, gamma)
        @test all(diff(beta_d) .< 0)

        # :positive — all knot values positive
        a_pos = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :positive)
        beta_p = gamma_to_knot_values(a_pos, gamma)
        @test all(beta_p .> 0)

        # :dec_positive — decreasing AND positive
        a_dp = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :dec_positive)
        beta_dp = gamma_to_knot_values(a_dp, gamma)
        @test all(diff(beta_dp) .< 0)
        @test all(beta_dp .> 0)

        # :inc_zero_left — increasing, first knot = 0
        a_izl = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :inc_zero_left)
        gamma7 = randn(9)
        beta_izl = gamma_to_knot_values(a_izl, gamma7)
        @test abs(beta_izl[1]) < 1e-15  # first knot exactly 0
        @test all(diff(beta_izl) .> 0)

        # :dec_zero_right — decreasing, last knot = 0
        a_dzr = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :dec_zero_right)
        beta_dzr = gamma_to_knot_values(a_dzr, gamma7)
        @test abs(beta_dzr[end]) < 1e-15  # last knot exactly 0
        @test all(diff(beta_dzr) .< 0)

        # :convex — second differences positive
        a_cx = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :convex)
        beta_cx = gamma_to_knot_values(a_cx, gamma)
        d2 = diff(diff(beta_cx))
        @test all(d2 .> 0)

        # :concave — second differences negative
        a_cv = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :concave)
        beta_cv = gamma_to_knot_values(a_cv, gamma)
        d2_cv = diff(diff(beta_cv))
        @test all(d2_cv .< 0)

        # Evaluator works and respects domain
        eval_inc = build_constrained_bspline_evaluator(a_inc, gamma)
        @test eval_inc(0.0) < eval_inc(0.5) < eval_inc(1.0)
    end

    @testset "ShapeConstrainedBSplineApproximator — LAML solver (SIR)" begin
        # SIR model with decreasing β(I)
        β_true(I) = 0.5 * exp(-5.0 * I)

        function sir!(du, u, p, t)
            S, I = u
            β_val = p.β(I)
            du[1] = -β_val * S * I
            du[2] = β_val * S * I - 0.25 * I
        end

        u0 = [0.99, 0.01]
        tspan = (0.0, 40.0)
        ode = ODEProblem(sir!, u0, tspan)

        using OrdinaryDiffEq
        sol_true = OrdinaryDiffEq.solve(ode,
            Tsit5(); p=(β=β_true,), saveat=1.0)
        t_obs = sol_true.t[1:end]
        data = hcat([u[1] for u in sol_true.u], [u[2] for u in sol_true.u])
        data .+= 0.005 .* randn(size(data))

        # Fit with decreasing shape constraint
        uf = ShapeConstrainedBSplineApproximator(:β, (0.0, 0.15), 8, :decreasing;
            initial=0.4)

        prob = PSMProblem(ode, [uf];
            data_times=t_obs, data_values=data,
            known_params=(;))

        sol = solve(prob, LAML(verbose=false))
        @test sol.data_loss < 1.0  # good fit
        @test sol.edf < 8.0  # some smoothing applied

        # Verify decreasing shape is maintained
        I_vals = range(0.01, 0.12, length=20)
        β_fitted = [sol.unknown_functions[:β](I) for I in I_vals]
        @test all(diff(β_fitted) .< 0.01)  # approximately decreasing
    end

    # ─── COMONet tests ────────────────────────────────────────────

    @testset "COMONetApproximator — construction" begin
        a = COMONetApproximator(:f, (0.0, 1.0), (16, 16), :increasing)
        @test a.name == :f
        @test a.domain == (0.0, 1.0)
        @test a.hidden_sizes == (16, 16)
        @test a.constraint == :increasing

        # Parameter count: 1→16 (16+16) + 16→16 (256+16) + 16→1 (16+1) = 321
        @test nparams(a) == 1*16 + 16 + 16*16 + 16 + 16*1 + 1

        # Invalid constraint
        @test_throws ArgumentError COMONetApproximator(:f, (0.0, 1.0), (8,), :invalid)
    end

    @testset "COMONetApproximator — shape enforcement" begin
        import PartiallySpecifiedModels as PSM

        xs = range(0.0, 1.0, length=50)

        for c in (:increasing, :decreasing, :convex, :concave,
                  :inc_convex, :inc_concave, :dec_convex, :dec_concave, :positive)
            a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), c)
            p = initial_params(a)
            ev = PSM.build_comonet_evaluator(a, p)
            vals = [ev(x) for x in xs]
            diffs = diff(vals)

            if c in (:increasing, :inc_convex, :inc_concave)
                @test all(d -> d >= -1e-10, diffs)
            elseif c in (:decreasing, :dec_convex, :dec_concave)
                @test all(d -> d <= 1e-10, diffs)
            elseif c == :positive
                @test all(v -> v > 0, vals)
            end

            if c in (:convex, :inc_convex, :dec_convex)
                dd = diff(diffs)
                @test all(d -> d >= -1e-8, dd)
            elseif c in (:concave, :inc_concave, :dec_concave)
                dd = diff(diffs)
                @test all(d -> d <= 1e-8, dd)
            end
        end
    end

    @testset "COMONetApproximator — ForwardDiff" begin
        import PartiallySpecifiedModels as PSM
        using ForwardDiff

        a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), :increasing)
        p = initial_params(a)

        function comonet_loss(params)
            ev = PSM.build_comonet_evaluator(a, params)
            sum(ev(x)^2 for x in 0.0:0.2:1.0)
        end

        g = ForwardDiff.gradient(comonet_loss, p)
        @test length(g) == nparams(a)
        @test all(isfinite, g)
    end

    @testset "COMONetApproximator — penalty matrix" begin
        import PartiallySpecifiedModels as PSM

        a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), :increasing; penalty_weight=0.05)
        S = PSM.penalty_matrix(a)
        @test size(S) == (nparams(a), nparams(a))
        @test S ≈ 0.05 * I(nparams(a))  # L2 penalty
    end

    @testset "COMONetApproximator — AdamSolver (exponential decay)" begin
        import PartiallySpecifiedModels as PSM

        # Exponential decay: du/dt = -r(t)*u, true r(t) = 0.3
        function exp_decay!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end

        uf_r = COMONetApproximator(:r, (0.0, 10.0), (8,), :positive;
                                   penalty_weight=0.001)

        t_data = collect(0.0:1.0:10.0)
        u_true = exp.(-0.3 .* t_data)
        u_data = reshape(u_true, :, 1)

        prob = PSMProblem(exp_decay!, [1.0], (0.0, 10.0), [uf_r];
            data_times=t_data, data_values=u_data,
            obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PSM.Gaussian())

        sol = solve(prob, AdamSolver(lr=0.01, maxiters=100))

        # Check that the fitted rate is positive (COMONet constraint)
        r_fitted = sol.unknown_functions[:r]
        r_vals = [r_fitted(t) for t in 0.0:2.0:10.0]
        @test all(v -> v > 0, r_vals)  # positive constraint guaranteed
    end

    # ─── New solver tests ─────────────────────────────────────────

    @testset "BNGSolver — logistic growth" begin
        r_true(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_bng!(du, u, p, t)
            N = u[1]
            du[1] = p.r(N) * N
        end
        rng_bng = Random.Xoshiro(42)
        sol_true = OrdinaryDiffEq.solve(
            ODEProblem(logistic_bng!, [1.0], (0.0, 15.0), (; r=r_true)),
            Tsit5(); saveat=1.0)
        t_bng = collect(sol_true.t)
        data_bng = [sol_true.u[i][1] + 0.1*randn(rng_bng) for i in 1:length(t_bng)]
        data_bng = max.(data_bng, 0.01)

        uf_bng = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_bng = PSMProblem(logistic_bng!, [1.0], (0.0, 15.0), [uf_bng];
            data_times=t_bng, data_values=reshape(data_bng, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_bng = solve(prob_bng, BNGSolver(maxiters=500, verbose=false))

        @test sol_bng isa PSMSolution
        @test sol_bng.data_loss < 50.0
        @test haskey(sol_bng.unknown_functions, :r)
        r_fitted = sol_bng.unknown_functions[:r]
        @test r_fitted(5.0) isa Float64
    end

    @testset "DaltonSolver — exponential decay" begin
        function decay_dal!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_dal = Random.Xoshiro(42)
        sol_true_d = OrdinaryDiffEq.solve(
            ODEProblem(decay_dal!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_dal = collect(sol_true_d.t)
        data_dal = [sol_true_d.u[i][1] + 0.05*randn(rng_dal) for i in 1:length(t_dal)]

        uf_dal = BSplineApproximator(:f, (0.0, 6.0), 8)
        prob_dal = PSMProblem(decay_dal!, [5.0], (0.0, 10.0), [uf_dal];
            data_times=t_dal, data_values=reshape(max.(data_dal, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_dal = solve(prob_dal, DaltonSolver(n_steps=100, maxiters=50, verbose=false))

        @test sol_dal isa PSMSolution
        @test isfinite(sol_dal.objective)
        @test haskey(sol_dal.unknown_functions, :f)
    end

    @testset "PseudoMarginalSolver — logistic growth" begin
        r_true_pm(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_pm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_pm = Random.Xoshiro(42)
        sol_true_pm = OrdinaryDiffEq.solve(
            ODEProblem(logistic_pm!, [1.0], (0.0, 15.0), (; r=r_true_pm)),
            Tsit5(); saveat=1.0)
        t_pm = collect(sol_true_pm.t)
        data_pm = [sol_true_pm.u[i][1] + 0.1*randn(rng_pm) for i in 1:length(t_pm)]

        uf_pm = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_pm = PSMProblem(logistic_pm!, [1.0], (0.0, 15.0), [uf_pm];
            data_times=t_pm, data_values=reshape(max.(data_pm, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_pm = solve(prob_pm, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, verbose=false))

        @test sol_pm isa PSMSolution
        @test sol_pm.convergence isa MCMCChains.Chains
        @test size(sol_pm.convergence, 1) == 50  # n_samples
    end

    @testset "DDE support — delay exponential decay" begin
        using DelayDiffEq
        function dde_decay!(du, u, h, p, t)
            u_delayed = h(p, t - 1.0)
            du[1] = -p.f(u_delayed[1])
        end
        h_dde(p, t) = [1.0]

        # Generate data with known params
        function dde_true_decay!(du, u, h, p, t)
            u_delayed = h(p, t - 1.0)
            du[1] = -0.5 * u_delayed[1]
        end
        prob_true_dde = DDEProblem(dde_true_decay!, [1.0], h_dde, (0.0, 8.0);
            constant_lags=[1.0])
        sol_true_dde = OrdinaryDiffEq.solve(prob_true_dde, MethodOfSteps(Tsit5());
            saveat=0.5)
        rng_dde = Random.Xoshiro(42)
        t_dde = collect(sol_true_dde.t)
        data_dde = [sol_true_dde.u[i][1] + 0.02*randn(rng_dde) for i in 1:length(t_dde)]

        uf_dde = BSplineApproximator(:f, (0.0, 1.5), 6)
        prob_dde = PSMProblem(dde_decay!, [1.0], (0.0, 8.0), [uf_dde];
            data_times=t_dde, data_values=reshape(max.(data_dde, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(),
            delays=[1.0], history=h_dde)

        @test !isempty(prob_dde.delays)
        @test prob_dde.delays == [1.0]
        @test prob_dde.history !== nothing

        sol_dde = solve(prob_dde, LAML(maxiters=30, verbose=false))
        @test sol_dde isa PSMSolution
        @test isfinite(sol_dde.data_loss)
        @test haskey(sol_dde.unknown_functions, :f)
    end

    # ─── New solver tests ─────────────────────────────────────────────

    @testset "GCVSolver — logistic growth" begin
        r_gcv(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_gcv!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_gcv = Random.Xoshiro(123)
        sol_true_gcv = OrdinaryDiffEq.solve(
            ODEProblem(logistic_gcv!, [1.0], (0.0, 15.0), (; r=r_gcv)),
            Tsit5(); saveat=0.5)
        t_gcv = collect(sol_true_gcv.t)
        data_gcv = [sol_true_gcv.u[i][1] + 0.1*randn(rng_gcv) for i in 1:length(t_gcv)]

        uf_gcv = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_gcv = PSMProblem(logistic_gcv!, [1.0], (0.0, 15.0), [uf_gcv];
            data_times=t_gcv, data_values=reshape(max.(data_gcv, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_gcv = solve(prob_gcv, GCVSolver(maxiters=30, verbose=false))

        @test sol_gcv isa PSMSolution
        @test isfinite(sol_gcv.data_loss)
        @test haskey(sol_gcv.unknown_functions, :r)
        r_fitted_gcv = sol_gcv.unknown_functions[:r]
        @test r_fitted_gcv(5.0) isa Float64
    end

    @testset "TwoStageSolver — logistic growth" begin
        r_ts(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_ts!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_ts = Random.Xoshiro(123)
        sol_true_ts = OrdinaryDiffEq.solve(
            ODEProblem(logistic_ts!, [1.0], (0.0, 15.0), (; r=r_ts)),
            Tsit5(); saveat=0.5)
        t_ts = collect(sol_true_ts.t)
        data_ts = [sol_true_ts.u[i][1] + 0.1*randn(rng_ts) for i in 1:length(t_ts)]

        uf_ts = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_ts = PSMProblem(logistic_ts!, [1.0], (0.0, 15.0), [uf_ts];
            data_times=t_ts, data_values=reshape(max.(data_ts, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ts = solve(prob_ts, TwoStageSolver(maxiters=500, verbose=false))

        @test sol_ts isa PSMSolution
        @test isfinite(sol_ts.data_loss)
        @test haskey(sol_ts.unknown_functions, :r)
    end

    @testset "DerivativeFreeSolver — exponential decay" begin
        function decay_df!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_df = Random.Xoshiro(42)
        sol_true_df = OrdinaryDiffEq.solve(
            ODEProblem(decay_df!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_df = collect(sol_true_df.t)
        data_df = [sol_true_df.u[i][1] + 0.05*randn(rng_df) for i in 1:length(t_df)]

        uf_df = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_df = PSMProblem(decay_df!, [5.0], (0.0, 10.0), [uf_df];
            data_times=t_df, data_values=reshape(max.(data_df, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_df = solve(prob_df, DerivativeFreeSolver(maxiters=5000, verbose=false))

        @test sol_df isa PSMSolution
        @test isfinite(sol_df.objective)
        @test haskey(sol_df.unknown_functions, :f)
    end

    @testset "VariationalSolver — logistic growth" begin
        r_vi(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_vi!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_vi = Random.Xoshiro(42)
        sol_true_vi = OrdinaryDiffEq.solve(
            ODEProblem(logistic_vi!, [1.0], (0.0, 15.0), (; r=r_vi)),
            Tsit5(); saveat=1.0)
        t_vi = collect(sol_true_vi.t)
        data_vi = [sol_true_vi.u[i][1] + 0.1*randn(rng_vi) for i in 1:length(t_vi)]

        uf_vi = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_vi = PSMProblem(logistic_vi!, [1.0], (0.0, 15.0), [uf_vi];
            data_times=t_vi, data_values=reshape(max.(data_vi, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_vi = solve(prob_vi, VariationalSolver(maxiters=500, n_elbo_samples=5, verbose=false))

        @test sol_vi isa PSMSolution
        @test isfinite(sol_vi.objective)
        @test haskey(sol_vi.unknown_functions, :r)
        @test haskey(sol_vi.convergence, :posterior_std)
    end

    @testset "ABCSolver — exponential decay" begin
        function decay_abc!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_abc = Random.Xoshiro(42)
        sol_true_abc = OrdinaryDiffEq.solve(
            ODEProblem(decay_abc!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=1.0)
        t_abc = collect(sol_true_abc.t)
        data_abc = [sol_true_abc.u[i][1] + 0.05*randn(rng_abc) for i in 1:length(t_abc)]

        uf_abc = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_abc = PSMProblem(decay_abc!, [5.0], (0.0, 10.0), [uf_abc];
            data_times=t_abc, data_values=reshape(max.(data_abc, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_abc = solve(prob_abc, ABCSolver(n_particles=50, n_generations=3, verbose=false))

        @test sol_abc isa PSMSolution
        @test isfinite(sol_abc.objective)
        @test haskey(sol_abc.unknown_functions, :f)
    end

    # ─── Discrete-time tests for additional solvers ───────────────────

    @testset "Discrete-time Ricker — multiple solvers" begin
        # Ricker model: N_{t+1} = N_t * exp(r(N_t))
        # True: r(N) = 0.5*(1 - N/10)
        r_ricker(N) = 0.5 * (1.0 - N / 10.0)
        function ricker!(u_next, u, p, t)
            u_next[1] = u[1] * exp(p.r(u[1]))
        end
        rng_rick = Random.Xoshiro(123)
        N0 = 2.0
        T_end = 30
        t_rick = collect(0.0:1.0:T_end)
        N_true = zeros(length(t_rick))
        N_true[1] = N0
        for i in 1:(length(t_rick)-1)
            N_true[i+1] = N_true[i] * exp(r_ricker(N_true[i]))
        end
        data_rick = N_true .+ 0.1*randn(rng_rick, length(t_rick))

        uf_rick = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_rick = PSMProblem(ricker!, [N0], (0.0, Float64(T_end)), [uf_rick];
            data_times=t_rick, data_values=reshape(max.(data_rick, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), discrete=true)

        # Test with BNGSolver (discrete)
        sol_bng_d = solve(prob_rick, BNGSolver(maxiters=500, verbose=false))
        @test sol_bng_d isa PSMSolution
        @test isfinite(sol_bng_d.data_loss)

        # Test with TwoStageSolver (discrete)
        sol_ts_d = solve(prob_rick, TwoStageSolver(maxiters=500, verbose=false))
        @test sol_ts_d isa PSMSolution
        @test isfinite(sol_ts_d.data_loss)

        # Test with DerivativeFreeSolver (discrete)
        sol_df_d = solve(prob_rick, DerivativeFreeSolver(maxiters=2000, verbose=false))
        @test sol_df_d isa PSMSolution
        @test isfinite(sol_df_d.objective)

        # Test with GCVSolver (discrete)
        sol_gcv_d = solve(prob_rick, GCVSolver(maxiters=20, verbose=false))
        @test sol_gcv_d isa PSMSolution
        @test isfinite(sol_gcv_d.data_loss)
    end

    @testset "Kalman solvers reject discrete" begin
        function dummy_disc!(u_next, u, p, t)
            u_next[1] = p.f(u[1])
        end
        uf_dummy = BSplineApproximator(:f, (0.0, 5.0), 6)
        prob_disc = PSMProblem(dummy_disc!, [1.0], (0.0, 10.0), [uf_dummy];
            data_times=collect(0.0:1.0:10.0),
            data_values=reshape(ones(11), :, 1),
            obs_to_state=[1], discrete=true)

        @test_throws ErrorException solve(prob_disc, RodeoSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, MagiSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, DaltonSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, PseudoMarginalSolver(verbose=false))
    end

    @testset "with_range_param" begin
        a_spde = SPDEApproximator(:f, (0.0, 1.0), 8; nu=1.5, range_param=0.2)
        a2 = with_range_param(a_spde, 0.5)
        @test a2 isa SPDEApproximator
        @test a2.range_param ≈ 0.5
        @test a2.name == :f
        @test a2.n_basis == 8
        @test a2.nu ≈ 1.5

        a_sc = ShapeConstrainedSPDEApproximator(:g, (0.0, 2.0), 8, :increasing; nu=1.5)
        a_sc2 = with_range_param(a_sc, 1.0)
        @test a_sc2 isa ShapeConstrainedSPDEApproximator
        @test a_sc2.range_param ≈ 1.0
        @test a_sc2.constraint == :increasing

        # Non-SPDE returns unchanged
        a_bs = BSplineApproximator(:h, (0.0, 1.0), 8)
        @test with_range_param(a_bs, 0.5) === a_bs
    end

    @testset "optimize_spde_range" begin
        Random.seed!(42)
        r_true(u) = 0.5 * u
        function decay!(du, u, p, t)
            du[1] = -p.r(u[1]) * u[1]
        end
        u0 = [5.0]
        tspan = (0.0, 5.0)
        sol_true = solve(ODEProblem(decay!, u0, tspan, (; r=r_true)), Tsit5(); saveat=0.25)
        t_obs = collect(sol_true.t)
        data = reshape([sol_true.u[i][1] + 0.05 * randn() for i in 1:length(t_obs)], :, 1)

        uf = SPDEApproximator(:r, (0.01, 5.5), 8; nu=1.5, initial=x -> 0.5)
        prob = PSMProblem(decay!, u0, tspan, [uf];
            data_times=t_obs, data_values=Float64.(data),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        result = optimize_spde_range(prob, LAML(maxiters=50, verbose=false);
            range_multipliers=[0.5, 1.0, 2.0], verbose=false)

        @test result.solution isa PSMSolution
        @test result.range_param > 0
        @test length(result.gcv_scores) == 3
        @test length(result.range_values) == 3
        @test all(isfinite, result.gcv_scores)
        @test result.solution.data_loss < 5.0
    end

end
