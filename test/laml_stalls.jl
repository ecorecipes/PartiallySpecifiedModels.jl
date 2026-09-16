module LAMLStallTests

using Test
include("../benchmarks/kan/multivariate.jl")

@testset "LAML smoothing-step priority" begin
    prefer = PSM._laml_prefer_old_step
    tol = 1e-6
    @test prefer(1.0, 0.9, 10.0, 9.0, tol)
    @test !prefer(1.0, 1.0-0.5tol, 10.0, 9.0, tol)
    @test prefer(1.0, 1.0-2tol, 10.0, 9.0, tol)
    @test !prefer(1000.0, 999.9995, 20.0, 19.0, tol)
    @test !prefer(1.0, 1.0, 10.0, 9.0, tol)
    @test !prefer(1.0, 1.1, 10.0, 9.0, tol)
    @test prefer(1.0, prevfloat(1.0), 10.0, 9.0, 0.0)
    @test prefer(Inf, 1.0, 10.0, 9.0, tol)
    @test !prefer(1.0, prevfloat(1.0), Inf, 9.0, tol)
    for candidate in (10.0, 11.0, Inf, -Inf, NaN)
        @test prefer(1.0, prevfloat(1.0), 10.0, candidate, tol)
    end

    @testset "A likelihood veto rejects a better smoothing criterion" begin
        x = collect(range(0.0, 1.0, length=9))
        J = hcat(ones(9), x, x.^2)
        y = 1 .+ 0.2x .+ [0.01*(sin(3.1i)+0.5cos(7.7i)) for i in 1:9]
        S = Matrix(Diagonal([0.0, 0.0, 1.0]))
        old_lambda, new_lambda = 1e-6, 100.0
        old_beta = (J'J+old_lambda*S) \ (J'y)
        new_beta = (J'J+new_lambda*S) \ (J'y)
        current = old_beta+[0.0, 0.0, 1e-4]
        objective(b, lambda) = 0.5*(sum(abs2, y-J*b)+lambda*dot(b, S*b))
        old_value, old_candidate = objective(current, old_lambda), objective(old_beta, old_lambda)
        new_value, new_candidate = objective(current, new_lambda), objective(new_beta, new_lambda)
        # Measured old gain 1.07e-8, well below the solver's declared 1e-6
        # precision, while the new-penalty gain is 0.00824.
        @test 0 < old_value-old_candidate < tol*max(abs(old_value), 1.0)
        @test new_candidate < new_value
        @test sum(abs2, y-J*new_beta) > sum(abs2, y-J*old_beta)
        old_V = first(PSM.laml_objective(Gaussian(), old_beta, J, ones(9), ones(9),
            y, J*old_beta, [S], [0], [3], [log(old_lambda)], 3))
        new_V = first(PSM.laml_objective(Gaussian(), new_beta, J, ones(9), ones(9),
            y, J*new_beta, [S], [0], [3], [log(new_lambda)], 3))
        @test new_V > old_V
        @test !prefer(old_value, old_candidate, new_value, new_candidate, tol)
    end

    @testset "Tensor smoothing advances before stopping" begin
        for case in MV_CASES
            ds = mv_dataset(case, 201; split="local")
            a = mv_make_approximator("tensor64", 10201, ds.train_values)
            prob = mv_problem(a, ds.train_ics, ds.train_times, ds.train_values, Ref(0))
            fit = solve(prob, LAML(maxiters=25, jac=:forwarddiff))
            @test fit.convergence.smoothing_advanced
            @test all(x -> isfinite(x) && x > 0, fit.smoothing_params)
            # Measured residuals 9.78e-6 / 1.18e-5; 1e-3 allows >84x
            # headroom, versus the previous 0.109 / 0.0985 stalls.
            @test fit.convergence.stationarity < 1e-3
            @test fit.fitted_values == simulate(prob, collect(fit.parameters))
            scores = [mv_score_trajectory(a, collect(fit.parameters), ds, case, id, MV_SUPPORT_RADIUS)
                      for id in eachindex(ds.test_ics)]
            @test all(s.test_status == "ok" for s in scores)
            # Measured trajectory NRMSE 0.00236 / 0.00306; >6x headroom.
            @test sqrt(mean(abs2, getproperty.(scores, :full_rmse))) < 0.02
        end
    end

    @testset "Stability remains distinct from fit quality" begin
        times = collect(0.0:0.25:2.0)
        flat!(du, u, p, t) = fill!(du, 0)
        prob = PSMProblem(flat!, [1.0], (0.0, 2.0),
            [BSplineApproximator(:unused, (0.0, 2.0), 4; initial=x -> 0.0)];
            data_times=times, data_values=fill(10.0, length(times), 1))
        fit = solve(prob, LAML(maxiters=20))
        @test fit.convergence.converged
        @test !fit.convergence.smoothing_advanced
        @test fit.fitted_values == ones(length(times), 1)
        @test fit.data_loss == sum(abs2, prob.data_values-fit.fitted_values)
        @test fit.data_loss == 81length(times)
    end
end

end
