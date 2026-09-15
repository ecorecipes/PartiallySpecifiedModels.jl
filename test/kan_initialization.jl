module KANInitializationTests

using Test
include("../benchmarks/kan/diagnose_uncertainty_initialization.jl")

@testset "KAN initialization and prior diagnostics" begin
    @testset "Initialization changes no grids or penalties" begin
        for ridge in (0.0,1e-6)
            a = ui_approximator(ridge)
            saved = initial_params(a)
            state = deepcopy(a.state)
            original,constant = ui_start(a,"original"),ui_start(a,"constant")
            @test initial_params(original) == saved
            @test initial_params(a) == saved && isequal(a.state,state)
            @test penalty_matrix(original) == penalty_matrix(constant) == penalty_matrix(a)
            @test penalty_blocks(original) == penalty_blocks(constant) == penalty_blocks(a)
            # Measured constant-response discrepancy <=5.56e-17;
            # 1e-12 allows >17000x headroom.
            @test maximum(abs,build_evaluator(constant,initial_params(constant)).(UC_POINTS) .- .5) < 1e-12
            @test_throws ArgumentError ui_start(a,"unknown")
            @test_throws ArgumentError ui_start(a,"constant";rate=Inf)
        end
        free,proper = ui_approximator(0.0),ui_approximator(1e-6)
        @test initial_params(free) == initial_params(proper)
        @test PSM._rank_penalty(penalty_matrix(proper)) -
              PSM._rank_penalty(penalty_matrix(free)) == 2
        @test_throws ArgumentError ui_approximator(1.0)
    end

    @testset "Selection uses training evidence within one prior" begin
        a = (status="ok",laml=3.0,nullspace_penalty=1e-6,data_loss=1.0)
        b = (status="ok",laml=4.0,nullspace_penalty=1e-6,data_loss=2.0)
        @test ui_select([a,b]) == b
        @test ui_select([a,merge(b,(;laml=NaN))]) == a
        @test ui_select([merge(a,(;status="failed"))]) === nothing
        @test_throws ArgumentError ui_select([a,merge(b,(;nullspace_penalty=0.0))])
    end

    @testset "Persisted two-by-two diagnosis" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                input,output = joinpath(dir,"calibration"),joinpath(dir,"diagnosis")
                uc_main(["--cases=nonlinear","--models=kan12","--seeds=17","--nboot=3",
                         "--sensitivity-seeds=","--iterations=6","--output=$input"])
                run = ui_main(["--inputs=$input","--output=$output"])
                @test length(run.trials) == 4
                @test length(run.selected) == 2
                @test length(run.intervals) == 16*2*3*2
                @test length(run.parity) == 1
                for winner in run.selected
                    group = filter(r->r.nullspace_penalty==winner.nullspace_penalty,run.trials)
                    @test winner.laml == maximum(r.laml for r in group if r.status=="ok")
                    @test winner.search_seconds == sum(r.fit_seconds for r in group)
                end
                meta = TOML.parsefile(joinpath(output,"metadata.toml"))
                @test !meta["bootstrap_refit"] && meta["exploratory"]
                @test all(rd_digest(joinpath(ROOT,p))==v for (p,v) in meta["inputs_sha256"])
                @test meta["design_sha256"] == rd_digest(joinpath(output,"design.toml"))
                @test_throws ErrorException ui_main(["--inputs=$input","--output=$output"])
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
