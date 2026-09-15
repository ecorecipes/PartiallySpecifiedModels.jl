module AdamControlTests

using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve

const PSM = PartiallySpecifiedModels

@testset "Adam plateau controls" begin
    default = AdamSolver()
    @test default.plateau_tol == 1e-4
    @test default.plateau_window == 30
    @test default.early_stopping
    for old in (AdamSolver(300, 0.01, false, :auto, 0.0, true),
                AdamSolver(300, 0.01, false, :auto, 0.0, true, nothing))
        @test all(isequal(getfield(old, name), getfield(default, name)) for name in fieldnames(AdamSolver))
    end
    @test_throws ArgumentError AdamSolver(plateau_tol=-1.0)
    @test_throws ArgumentError AdamSolver(plateau_tol=NaN)
    @test_throws ArgumentError AdamSolver(plateau_tol=Inf)
    @test_throws ArgumentError AdamSolver(plateau_window=1)

    flat = fill(0.1, 30)
    spread = copy(flat)
    spread[end] += 5e-5
    @test PSM._adam_plateau(spread, 61, 0.1, 0.01, default)
    @test !PSM._adam_plateau(spread, 61, 0.1, 0.01, AdamSolver(plateau_tol=1e-6))
    @test !PSM._adam_plateau(flat, 61, 0.1, 0.01, AdamSolver(early_stopping=false))
    @test !PSM._adam_plateau(flat, 61, 0.1, 0.01, AdamSolver(plateau_tol=0.0))
    @test !PSM._adam_plateau(flat, 60, 0.1, 0.01, default)
    @test !PSM._adam_plateau(flat, 61, 1e10, 0.01, default)
    @test !PSM._adam_plateau(flat, 149, 0.1, 0.0001, default)
    @test !PSM._adam_plateau(fill(Inf,30), 61, 0.1, 0.01, default)
    @test !PSM._adam_plateau(fill(0.1,100), 100, 0.1, 0.01, AdamSolver(plateau_window=100))
    @test PSM._adam_plateau(fill(0.1,100), 101, 0.1, 0.01, AdamSolver(plateau_window=100))
    for window in (flat, spread, collect(range(0.1,0.2,length=30))),
        iter in (60,61,149), best in (0.1,1e10), rate in (0.01,0.0001)
        legacy = iter > 60 && best < 1e9 && rate > 0.05default.lr &&
            (maximum(window)-minimum(window))/max(abs(minimum(window)),1.0) < 1e-4
        @test PSM._adam_plateau(window,iter,best,rate,default) == legacy
    end

    @testset "Fixed-budget execution does not claim convergence" begin
        flat!(du,u,p,t) = fill!(du,0)
        times = [0.0,0.5,1.0]
        prob = PSMProblem(flat!,[1.0],(0.0,1.0),
            [BSplineApproximator(:r,(0.0,1.0),4;initial=x->0.0)];
            data_times=times,data_values=ones(3,1))
        early = solve(prob,AdamSolver(maxiters=150))
        full = solve(prob,AdamSolver(maxiters=150,early_stopping=false))
        long_window = solve(prob,AdamSolver(maxiters=150,plateau_window=100))
        # This objective is identically zero, so these counts exercise
        # declared control flow, not a nonlinear optimization output pin.
        @test early.convergence.iterations == 61
        @test early.convergence.reason == :plateau && early.convergence.converged
        @test full.convergence.iterations == 150
        @test full.convergence.reason == :maxiters && !full.convergence.converged
        @test long_window.convergence.iterations == 101
        @test early.parameters == full.parameters == long_window.parameters
        @test early.fitted_values == full.fitted_values == long_window.fitted_values
    end
end

end
