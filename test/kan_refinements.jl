module KANRefinementTests

using Test
include("../benchmarks/kan/refine_reaction_diffusion.jl")
include("../benchmarks/kan/transfer_reaction_diffusion.jl")

struct TransferOracle <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::TransferOracle) = 1
PSM.initial_params(::TransferOracle) = error("mesh transfer must not initialize a refit")
PSM.build_evaluator(::TransferOracle,beta) = x -> zero(x)

@testset "KAN spatial assessment refinements" begin
    @testset "Explicit controls and no-refit mesh estimands" begin
        default = rd_options(String[])
        @test default.plateau_tol == 1e-4
        @test default.plateau_window == 30 && default.early_stopping
        cfg = rd_options(["--plateau-tol=1e-6","--plateau-window=50","--early-stopping=false"])
        @test cfg.plateau_tol == 1e-6 && cfg.plateau_window == 50 && !cfg.early_stopping
        @test_throws ArgumentError rd_options(["--plateau-tol=-1"])
        @test_throws ArgumentError rd_refine_options(["--iterations=0"])
        @test_throws ArgumentError rd_transfer_options(["--cells=24,30"])
        @test_throws ArgumentError rd_transfer_options(["--cells=48,24"])
        @test length(RD_OPTIMIZER_VARIANTS) == 4
        @test only(filter(v->v.variant=="full",RD_OPTIMIZER_VARIANTS)).early_stopping == false
        times = collect(0.0:0.2:1.0)
        cells = [8,16,32]
        refs = Dict(n=>rd_diffusion_exact("fine_scale",rd_grid(n),times) for n in cells)
        beta = [0.0]
        rows = rd_transfer_profile(TransferOracle(:r),beta,"diffusion","fine_scale",cells,times,refs)
        @test beta == [0.0]
        @test all(r->r.status=="ok",rows)
        # The independent cosine-mode oracle measured <=7.59e-13 under
        # tighter reference tolerances; 1e-7 allows the fit's 1e-8 solve.
        @test all(r->r.field_rmse<1e-7,rows)
        @test first(rows).learned_mesh_shift_rmse == 0.0
        @test last(rows).learned_to_finest_mesh_rmse == last(rows).truth_mesh_rmse == 0.0
        coarse = rd_restrict(refs[32],32,8)
        @test first(rows).truth_mesh_rmse == rmse(refs[8],coarse)
        @test all(r->r.projected_to_finest_truth_rmse>=0,rows)
    end

    @testset "Persisted validation-only selection and immutable replay" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                input = joinpath(dir,"original")
                original = rd_main(["--cells=8","--cases=crowding","--seeds=17",
                    "--models=spline8,kan28","--iterations=3","--etas=0,0.1","--output=$input"])
                archive = rd_load_archive(input)
                output = joinpath(dir,"refined")
                run = rd_refine_main(["--input=$input","--output=$output","--iterations=3",
                    "--cases=crowding","--seeds=17","--models=spline8,kan28"])
                @test length(run.trials) == 8
                @test length(run.selected) == 2
                @test length(run.profiles) == 16
                @test length(run.parity) == 16
                @test length(run.condition_seeds) == 8
                for winner in run.selected
                    group = filter(r->r.model==winner.model,run.trials)
                    expected = group[argmin(getproperty.(group,:validation_rmse))]
                    @test winner.candidate_id == expected.candidate_id
                    old = only(filter(r->r.model==winner.model,original.selected))
                    @test all(r->r.eta==old.eta,group)
                    @test winner.tuning_seconds == sum(r.fit_seconds for r in group)
                    @test winner.total_search_seconds == winner.tuning_seconds+old.tuning_seconds
                    saved = run.coefficients["crowding__$(winner.model)__17"]
                    @test saved["parameters"] == run.candidates[winner.candidate_id]["parameters"]
                end
                @test all(r->r.absolute_difference<=r.atol+r.rtol*abs(r.archived),run.parity)
                transfer = rd_transfer_main(["--inputs=$input","--output=$(joinpath(dir,"transfer"))",
                                            "--cells=8,16"])
                @test length(transfer.rows) == 8
                @test length(transfer.seeds) == 4
                @test all(r->r.finite_fields==2,transfer.seeds)
                @test all(r->r.status=="ok",transfer.rows)
                meta = TOML.parsefile(joinpath(dir,"transfer","metadata.toml"))
                @test meta["optimizer_refit"] == false
                @test meta["evaluation_source_sha256"] == source_hash()
                @test only(meta["input_archives"])["fitting_source_sha256"] == archive.meta["source_sha256"]
                @test isnothing(rd_archive_unchanged(archive))
                @test_throws ErrorException rd_refine_main(["--input=$input","--output=$output",
                    "--cases=crowding","--seeds=17","--models=spline8"])
                @test_throws ErrorException rd_transfer_seeds([transfer.rows;first(transfer.rows)])
                incomplete = rd_transfer_seeds(transfer.rows[1:1])
                @test only(incomplete).field_rmse == Inf
                ds = rd_dataset("crowding",17,8)
                saved = archive.coefficients["crowding__spline8__17"]
                @test_throws ErrorException rd_replay_parity("spline8",rd_approximator("spline8",20017),
                    saved["parameters"] .+ 1.0,ds,archive)
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
