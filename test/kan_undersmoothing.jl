module UndersmoothingTests

using Test
include("../benchmarks/kan/undersmoothing.jl")

@testset "Selection then fixed-smoothing coverage" begin
    @test_throws ArgumentError UndersmoothingFit(fraction=0.0)
    @test_throws ArgumentError UndersmoothingFit(fraction=2.0)
    @test_throws ArgumentError UndersmoothingFit(fraction=NaN)
    @test_throws ArgumentError UndersmoothingFit(iterations=0)
    ds=uc_dataset("nonlinear",2901)
    queries=uc_queries("nonlinear",uc_reference("nonlinear"))
    spec=cf_model("kan12_free")
    prob=uc_problem(spec.fit,ds)
    beta0=initial_params(spec.fit)
    for fraction in (1.0,0.25)
        sol=solve(prob,UndersmoothingFit(fraction=fraction,iterations=40))
        p=sol.convergence.procedure
        @test p.fraction == fraction
        @test sol.smoothing_params == [fraction*p.selected_lambda]
        @test sol.convergence.smoothing_fixed && !sol.convergence.smoothing_advanced
        @test p.selection_smoothing_advanced
        @test initial_params(spec.fit) == beta0
        @test sol.convergence.converged
    end
    cfg=cf_options(["--seeds=2901","--nboot=3","--sensitivity-seeds="])
    run=uc_fit_dataset("undersmooth",ds,queries,cfg;approximator=spec.fit,
        reporting_approximator=spec.report,algorithm=UndersmoothingFit(fraction=0.25))
    @test run.bootstrap_row.n_success == 3
    @test run.record["procedure"]["fraction"] == 0.25
    for d in run.record["refit_diagnostics"]
        p=d["procedure"]
        @test p["final_lambda"] == 0.25p["selected_lambda"]
        @test p["selection_smoothing_advanced"]
    end
    @test all(r->r.available,run.intervals)
end

end
