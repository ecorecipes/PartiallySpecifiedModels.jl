module NumericalRegressions

using Test
using LinearAlgebra
using Random
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve

const PSM = PartiallySpecifiedModels
const FD = PSM.ForwardDiff

struct ConstantApproximator <: AbstractApproximator
    name::Symbol
    initial::Float64
end
PSM.nparams(::ConstantApproximator) = 1
PSM.initial_params(a::ConstantApproximator) = [a.initial]
PSM.penalty_matrix(::ConstantApproximator) = nothing
PSM.build_evaluator(::ConstantApproximator, beta) = x -> beta[1]

struct NullSpaceConstant <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::NullSpaceConstant) = 3
PSM.initial_params(::NullSpaceConstant) = [0.9, 0.0, 0.0]
PSM.penalty_matrix(::NullSpaceConstant) = Matrix(Diagonal([0.0, 1.0, 1.0]))
PSM.build_evaluator(::NullSpaceConstant, beta) = x -> beta[1]

function d5_smoothing_fixture(family, mu_target, noise)
    approx = BSplineApproximator(:f, (0.0, 5.0), 8)
    np = nparams(approx)
    S = Matrix(first(first(penalty_blocks(approx))))
    xs = collect(range(0.05, 4.9, length=25))
    J = zeros(25, np)
    for k in 1:np
        e = zeros(np)
        e[k] = 1.0
        f = build_evaluator(approx, e)
        J[:, k] .= f.(xs)
    end
    beta = J \ mu_target
    mu = J * beta
    y = noise(mu, Random.Xoshiro(555))
    W = PSM.irls_weights(family, y, mu, ones(25))
    PSM.estimate_smoothing_params(
        J, W, ones(25), y, mu, beta, [S], [0], [np], np; family=family)
end

@testset "Numerical regression fixes" begin
    @testset "Truncated-normal tails and dispersion" begin
        for fam in (TruncatedNormal(), TruncatedNormal(lower=3.0, sigma=0.5))
            a, sigma = fam.lower, fam.sigma
            for xi in (-1e6, -1000.0, -40.0, -10.0, -8.0, -6.0, -1.0,
                       -0.999, 0.0, 4.0)
                mu, y = a + sigma * xi, a + 0.01sigma
                # Independently differentiate the defining density at high
                # precision, not the moments or the stabilized scalar kernel.
                ref_ll = m -> -((big(y)-m)/big(sigma))^2 / 2 -
                    log(big(sigma)) - log(2big(pi)) / 2 -
                    PSM._normlogcdf((m-big(a))/big(sigma))
                score_big = FD.derivative(ref_ll, big(mu))
                info_big = -FD.derivative(m -> FD.derivative(ref_ll, m), big(mu))
                score, info = Float64(score_big), Float64(info_big)
                mean_ref = Float64(big(y) - big(sigma)^2 * score_big)
                variance_ref = Float64(big(sigma)^4 * info_big)
                weight = PSM.irls_weights(fam, [y], [mu], [1.0])[1]
                residual = PSM._working_residual_scalar(fam, y, mu)

                @test 0 < weight <= 1 / sigma^2
                # Measured relative weight/variance error <=7.55e-15 over
                # this sweep; 1e-8 leaves >1e6x headroom.
                @test isapprox(weight, max(info, 1e-10); rtol=1e-8)
                @test isapprox(PSM._variance_function(fam, mu), variance_ref; rtol=1e-8)
                # Measured score/mean errors <=1.34e-15; >70000x headroom.
                @test isapprox(weight * residual, score; atol=1e-10, rtol=1e-10)
                @test isapprox(PSM._family_mean(fam, mu), mean_ref; atol=1e-10, rtol=1e-10)
                # Measured absolute log-density error <=1.82e-12, including
                # xi=-1e6; the absolute gate alone leaves >50x headroom.
                @test isapprox(PSM.loglik_pointwise(fam, y, mu),
                               Float64(ref_ll(big(mu))); atol=1e-10, rtol=1e-12)
                @test PSM.log_likelihood(fam, [y], [mu], [1.0]) ==
                      PSM.loglik_pointwise(fam, y, mu)
            end
        end

        fam = TruncatedNormal()
        times = collect(1.0:20.0)
        data = reshape(repeat([0.02, 0.04, 0.06, 0.08], 5), :, 1)
        map!(du, u, p, t) = (du[1] = p.g(u[1]))
        prob = PSMProblem(map!, [0.0], (0.0, 20.0),
            [ConstantApproximator(:g, -10.0)];
            data_times=times, data_values=data, likelihood=fam,
            discrete=true, solver=nothing)
        ll = b -> PSM.log_likelihood(fam, vec(data), fill(b, 20), ones(20))
        for alg in (LAML(jac=:forwarddiff), GCVSolver(jac=:forwarddiff))
            sol = solve(prob, alg)
            b = only(sol.parameters)
            @test ll(b) >= ll(-20.0)
            # Measured score <=1.05e-10 vs 0.962 before; >9000x headroom.
            @test abs(FD.derivative(ll, b)) < 1e-6
        end

        # Translating y, the latent location and the truncation point by
        # the same amount must not change the Pearson-scaled FS update.
        x = collect(range(0.0, 1.0, length=6))
        J = hcat(ones(6), x, x.^2)
        S = Matrix(Diagonal([0.0, 0.0, 1.0]))
        lambdas = Float64[]
        for shift in (0.0, -5.0)
            family = TruncatedNormal(lower=shift)
            beta = [0.2 + shift, 0.05, 0.1]
            mu = J * beta
            y = mu + [0.1, 1.2, 2.5, 0.4, 3.3, 1.1]
            W = PSM.irls_weights(family, y, mu, ones(6))
            lambda, _ = PSM.estimate_smoothing_params(
                J, W, ones(6), y, mu, beta, [S], [0], [3], 3;
                family=family, rho_init=[0.0], maxiter=1)
            push!(lambdas, only(lambda))

            # One-step oracle: refit the working model and use the actual
            # truncated mean/variance in the Pearson statistic. A fixed-point
            # lambda window cannot test this wiring reliably.
            means = [m + FD.derivative(PSM._normlogcdf, m-shift) for m in mu]
            variances = [1 + FD.derivative(
                v -> FD.derivative(PSM._normlogcdf, v), m-shift) for m in mu]
            score = FD.gradient(v -> PSM.log_likelihood(family, y, v, ones(6)), mu)
            H = J' * Diagonal(W) * J + S
            beta_hat = H \ (J' * (score + W .* (J * beta)))
            phi = sum((y-means).^2 ./ variances) / (6-1)
            @test phi > 1
            reference = phi * (1-tr(H \ S)) / dot(beta_hat, S * beta_hat)
            # Same ridge scale as the measured 1.28e-9 translation drift;
            # 1e-6 leaves >780x headroom without pinning an optimizer.
            @test isapprox(only(lambda), reference; rtol=1e-6)
        end
        # Measured relative drift 1.28e-9 from the working-Hessian ridge;
        # 1e-6 allows >780x numerical headroom.
        @test isapprox(lambdas[1], lambdas[2]; rtol=1e-6)

        for (family, target, noise) in (
            (TruncatedNormal(sigma=0.15), collect(range(0.30, 0.02, length=25)),
             (m, rng) -> max.(m .+ 0.15 .* randn(rng, 25), 0.0)),
            (Gaussian(), collect(range(2.0, 0.5, length=25)),
             (m, rng) -> m .+ 0.10 .* randn(rng, 25)),
            (Poisson(), collect(range(9.0, 2.0, length=25)),
             (m, rng) -> max.(round.(m .+ 0.8 .* randn(rng, 25)), 0.0)),
            (NegativeBinomial(4.0), collect(range(9.0, 2.0, length=25)),
             (m, rng) -> max.(round.(m .+ 0.8 .* randn(rng, 25)), 0.0)))
            lambda, edf = d5_smoothing_fixture(family, target, noise)
            @test isfinite(only(lambda)) && only(lambda) > 0
            @test 0 < edf <= 8  # hat-trace bound for this eight-coefficient fixture
            if !(family isa TruncatedNormal)
                # Legacy D5 controls: measured EDF 1.90..1.99 across local/CI
                # resolutions; the bounds leave >=0.51 EDF headroom.
                @test 1.0 < edf < 2.5
            end
        end
    end

    @testset "Prediction times with extra saved states" begin
        times = collect(1.0:5.0)
        expected = hcat(10.0 .- 0.8times, 2.0 .+ 0.4times, 10.0 .- 0.8times)
        expected_jac = vec(hcat(-2times, times, -2times))
        function rhs!(du, u, p, t)
            du[1] = p.g(u[1])
            du[2] = -2p.g(u[2])
        end
        function delayed!(du, u, h, p, t)
            du[1] = p.g(h(p, t-1.0)[1])
            du[2] = -2p.g(h(p, t-1.0)[2])
        end
        for delayed in (false, true)
            for kwargs in ((; save_start=false), (; save_start=true),
                           (; save_everystep=true),
                           (; save_start=true, dense=false),
                           (; saveat=[0.0, 5.0]))
                prob = PSMProblem(delayed ? delayed! : rhs!, [2.0, 10.0], (0.0, 5.0),
                    [ConstantApproximator(:g, 0.4)];
                    data_times=times, data_values=expected, obs_to_state=[2, 1, 2],
                    delays=delayed ? [1.0] : Float64[], kwargs...)
                pred = simulate(prob, [0.4])
                J = FD.jacobian(b -> vec(simulate(prob, b)), [0.4])
                # Review affine prediction error <=5.78e-15; this 1e-10
                # gate leaves >17000x headroom, including a second state.
                @test maximum(abs, pred-expected) < 1e-10
                @test maximum(abs, vec(J)-expected_jac) < 1e-10
                sol = solve(prob, LAML(maxiters=20, jac=:forwarddiff))
                # Review slope error <1e-15 vs 0.1333 before; >1e7x headroom.
                @test abs(only(sol.parameters)-0.4) < 1e-8
            end

            # Extra saved callback states must not let an early termination
            # satisfy a row-count guard while later data times are missing.
            stop = PSM.SciMLBase.DiscreteCallback(
                (u, t, integrator) -> t >= 2.0,
                integrator -> PSM.SciMLBase.terminate!(integrator);
                save_positions=(true, true))
            prob = PSMProblem(delayed ? delayed! : rhs!, [2.0, 10.0], (0.0, 5.0),
                [ConstantApproximator(:g, 0.4)];
                data_times=[1.0, 3.0, 5.0], data_values=expected[[1, 3, 5], :],
                obs_to_state=[2, 1, 2], delays=delayed ? [1.0] : Float64[],
                callback=stop, tstops=[2.0], save_start=true)
            @test_throws ErrorException simulate(prob, [0.4])
        end

        # EKI forwards save options through its own ODE path.
        fits = PSMSolution[]
        for save_start in (false, true)
            prob = PSMProblem(rhs!, [2.0, 10.0], (0.0, 5.0),
                [ConstantApproximator(:g, 0.4)];
                data_times=times, data_values=expected, obs_to_state=[2, 1, 2],
                save_start=save_start)
            push!(fits, solve(prob, EnsembleKalmanSolver(
                n_ensemble=8, n_iterations=3, rng_seed=42)))
        end
        @test fits[1].parameters == fits[2].parameters
        @test fits[1].fitted_values == fits[2].fitted_values
    end

    @testset "Discrete observation grids and shooting" begin
        function advance!(du, u, p, t)
            du[1] = u[1] + p.g(u[1])
            du[2] = u[2] - 2p.g(u[2])
        end
        for origin in (0.0, 0.5, 0.1, -0.5)
            # Gaps, duplicate observations and off-grid rounding all retain
            # their meanings when the model's time origin moves.
            offsets = [0.0, 0.2, 1.0, 2.0, 2.0, 3.2, 5.0]
            steps = round.(Int, offsets)
            times = origin .+ offsets
            expected = hcat(10.0 .- 0.8steps, 2.0 .+ 0.4steps)
            prob = PSMProblem(advance!, [2.0, 10.0], (origin, origin+5.0),
                [ConstantApproximator(:g, 0.4)];
                data_times=times, data_values=expected, obs_to_state=[2, 1],
                discrete=true, solver=nothing)
            beta = [0.4]
            p = PSM.build_autodiff_param_struct(prob, beta)
            pred = simulate(prob, beta)
            pred_adam = PSM.adam_simulate_discrete(prob, p, Float64)
            # Review unit-step error <=4.45e-16; 1e-12 gives >2000x headroom.
            @test maximum(abs, pred-expected) < 1e-12
            @test pred_adam == pred
            J = FD.jacobian(b -> vec(PSM.adam_simulate_discrete(
                prob, PSM.build_autodiff_param_struct(prob, b), eltype(b))), beta)
            @test vec(J) == vec(hcat(-2steps, steps))
            sol = solve(prob, LAML(maxiters=10, jac=:forwarddiff))
            # Review RSS 5.92e-31 vs 56.8 before; >1e14x headroom.
            @test sol.data_loss < 1e-16

            boundaries, intervals = PSM.partition_intervals(
                times, 2; t0=origin, discrete=true)
            @test first(boundaries) == origin
            @test sort(vcat(intervals...)) == collect(eachindex(times))
            middle_steps = round.(Int, boundaries[2:end-1] .- origin)
            shooting = hcat(2.0 .+ 0.4middle_steps, 10.0 .- 0.8middle_steps)
            z = vcat(beta, vec(shooting))
            loss = PSM.ms_loss(prob, z, 1, 2, boundaries, intervals,
                               zeros(length(boundaries)-2, 2), 10.0)
            # Exact affine data/continuity: same roundoff RSS scale as above.
            @test loss < 1e-16
        end

        # Half-step ties must be rounded globally, not anew per segment.
        times = [0.5, 1.0, 2.0, 3.0, 3.5]
        boundaries, intervals = PSM.partition_intervals(
            times, 3; t0=0.5, discrete=true)
        @test boundaries == [0.5, 1.5, 2.5, 3.5]
        @test intervals == [[1, 2], Int[], [3, 4, 5]]

        for times in ([0.5, 7.5], [0.5, NaN])
            prob = PSMProblem(advance!, [2.0, 10.0], (0.5, 5.5),
                [ConstantApproximator(:g, 0.4)];
                data_times=times, data_values=zeros(2, 2), discrete=true,
                solver=nothing)
            @test_throws ArgumentError simulate(prob, [0.4])
            @test_throws ArgumentError solve(prob, LAML(maxiters=1))
        end
    end

    @testset "Discrete collocation terminal Jacobian" begin
        times = collect(0.0:2.0)
        data = reshape([0.5, 2.0, 3.0], :, 1)
        terminal_calls = Ref(0)
        function map!(du, u, p, t)
            t == last(times) && (terminal_calls[] += 1)
            du[1] = p.g(u[1])
        end
        prob = PSMProblem(map!, [0.5], (0.0, 2.0),
            [ConstantApproximator(:g, 1.0)];
            data_times=times, data_values=data, discrete=true, solver=nothing)
        D = PSM.build_diff_matrix(times)
        x = vcat(vec(data), 1.0)
        residual = v -> PSM.collocation_residual_only(
            prob, times, reshape(v[1:3], 3, 1), v[4:end], D, 10.0, ones(3))
        J_ref = zeros(length(residual(x)), length(x))
        h = 1e-5
        for j in eachindex(x)
            xp, xm = copy(x), copy(x)
            xp[j] += h
            xm[j] -= h
            J_ref[:, j] .= (residual(xp) - residual(xm)) / (2h)
        end
        optimum = x - J_ref \ residual(x)
        minimum_ss = sum(abs2, residual(optimum))
        for jac in (:fd, :forwarddiff)
            r, J = PSM.collocation_residual_jacobian(
                prob, times, copy(data), [1.0], D, 10.0, ones(3); jac=jac)
            @test r[6] == 0
            @test all(iszero, J[6, :])
            # Measured 2.81e-11 vs 3.162 before; >350x headroom.
            @test maximum(abs, J-J_ref) < 1e-8
            sol = solve(prob, CollocationLAML(
                maxiters=1, n_continuation=1, lambda_ode_start=10.0,
                lambda_ode_end=10.0, jac=jac))
            fitted = vcat(vec(sol.fitted_values), collect(sol.parameters))
            # Exactly quadratic: one GN step suffices. Measured objective
            # excess <=4.45e-16 vs 1.770 before; >2e7x headroom.
            @test abs(sum(abs2, residual(fitted)) - minimum_ss) < 1e-8
        end
        @test terminal_calls[] == 0

        # Every state's terminal compliance row must be zero, including
        # when observation columns are permuted.
        function coupled!(du, u, p, t)
            du[1] = p.a(u[1]) + 0.2u[2]
            du[2] = p.b(u[2]) + 0.1u[1]
        end
        alpha = [0.5 1.0; 0.8 1.1; 0.9 1.3]
        prob2 = PSMProblem(coupled!, [0.5, 1.0], (0.0, 2.0),
            [ConstantApproximator(:a, 0.2), ConstantApproximator(:b, 0.3)];
            data_times=times, data_values=alpha[:, [2, 1]], obs_to_state=[2, 1],
            discrete=true, solver=nothing)
        for jac in (:fd, :forwarddiff)
            r, J = PSM.collocation_residual_jacobian(
                prob2, times, alpha, [0.2, 0.3], D, 10.0, ones(6); jac=jac)
            @test all(iszero, r[[9, 12]])
            @test all(iszero, J[[9, 12], :])
        end
    end

    @testset "Collocation holds an exactly null penalty" begin
        # The penalized coefficients are structurally unused and stay
        # exactly zero. Noisy data keep continuation active, so theta_fit
        # cannot hide an escalating FS proposal behind an unchanged fit.
        approx = NullSpaceConstant(:g)
        map!(du, u, p, t) = (du[1] = p.g(1.0) * u[1])
        data = [1.0, 1.01, 0.99, 1.02, 0.97, 1.03, 1.0, 0.98, 1.04]
        prob = PSMProblem(map!, [1.0], (0.0, 8.0), [approx];
            data_times=collect(0.0:8.0), data_values=reshape(data, :, 1),
            discrete=true)
        sol = solve(prob, CollocationLAML(maxiters=30))
        @test all(iszero, sol.parameters.g[2:3])
        @test sol.data_loss > 0
        @test sol.smoothing_params == [1 / tr(penalty_matrix(approx))]
    end
end

end
