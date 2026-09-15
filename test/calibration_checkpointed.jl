module CalibrationCheckpointedTests

using Test
include("../benchmarks/calibration/run.jl")
include("../benchmarks/calibration/checkpointed.jl")

@testset "Checkpointed bootstrap and confirmation" begin
    points = collect(range(0.3,1.8;length=7))
    ds = ce_dataset("smoke","integral","baseline","iid",0.015,"poly_a",901,points)
    cfg = ce_options(["--stage=smoke","--models=spline8","--operators=integral",
        "--methods=conditional,bootstrap_t","--rhos=-8,-4","--iterations=4","--ngrid=7","--nsim=20"])
    model = ce_model("spline8")
    opts = (;iterations=cfg.iterations,tol=cfg.tol,rhos=cfg.rhos,nboot=3,noise_mode="known",pilot_fraction=0.25)
    fit = ce_fit(model,ds,ds.observed,opts).fit
    @test fit!==nothing
    context = ceb_context(model,ds,cfg,"known","plan","data")

    @testset "Atomic attempt recovery preserves the original bootstrap" begin
        reference = ce_bootstrap(model,ds,fit,opts,"fit")
        mktempdir() do directory
            calls = Ref(0)
            fitfun = function(args...)
                calls[] += 1
                ce_fit(args...)
            end
            crash = (path,payload)->endswith(path,"000001.toml") && error("simulated interruption")
            @test_throws ErrorException ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;
                on_commit=crash,fitfun)
            @test calls[]==1
            @test isfile(joinpath(directory,"attempts","000001.toml"))
            resumed = ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;fitfun)
            @test calls[]==3
            @test resumed.draws==reference.draws
            @test resumed.scales==reference.scales
            @test resumed.generating==reference.generating
            @test isequal(resumed.records,reference.records)
            repeat = ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;fitfun)
            @test calls[]==3
            @test repeat.draws==resumed.draws
            @test_throws ErrorException ceb_bootstrap(model,ds,fit,opts,"fit",directory,
                merge(context,Dict("plan_sha256"=>"different"));fitfun)
            extra = joinpath(directory,"attempts","000004.toml")
            cer_atomic(extra,"extra = true\n")
            @test_throws ErrorException ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;fitfun)
        end
    end

    @testset "Failed attempts stay failed and are not redrawn" begin
        mktempdir() do directory
            calls = Ref(0)
            fail = function(args...)
                calls[] += 1
                (;fit=nothing,attempts=NamedTuple[],status="failed",message="intentional numerical failure")
            end
            first = ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;fitfun=fail)
            @test calls[]==3
            @test all(isnan,first.draws)
            @test [r["attempt"] for r in first.records]==[1,2,3]
            @test all(r["status"]=="failed" for r in first.records)
            again = ceb_bootstrap(model,ds,fit,opts,"fit",directory,context;fitfun=fail)
            @test calls[]==3
            @test isequal(first.draws,again.draws)
            @test_throws DomainError ce_studentized(fit.estimates,fit.se,again.generating,
                again.draws,again.scales,length(points),0.95)
        end
    end

    @testset "Pilot checkpoint and component integrity" begin
        mktempdir() do directory
            pilot_calls = Ref(0)
            pilot = function(args...)
                pilot_calls[] += 1
                ce_fixed_fit(args...)
            end
            original = ceb_bootstrap(model,ds,fit,opts,"pilot",directory,context;pilotfun=pilot)
            repeat = ceb_bootstrap(model,ds,fit,opts,"pilot",directory,context;pilotfun=pilot)
            @test pilot_calls[]==1
            @test original.draws==repeat.draws
            @test original.generating_rho==fit.rho+log(opts.pilot_fraction)
            path = joinpath(directory,"attempts","000001.toml")
            component = TOML.parsefile(path)
            component["payload_sha256"]="invalid"
            cer_atomic(path,cer_text(component))
            @test_throws ErrorException ceb_bootstrap(model,ds,fit,opts,"pilot",directory,context)
        end
    end

    @testset "Original-fit restoration and full endpoint parity" begin
        direct = ce_evaluate_model(model,ds,cfg,"known",["conditional","bootstrap_t"])
        mktempdir() do directory
            before = ceb_evaluate(model,ds,cfg,"known",["conditional","bootstrap_t"],directory,context)
            after = ceb_evaluate(model,ds,cfg,"known",["conditional","bootstrap_t"],directory,context)
            @test isequal(before.intervals,direct.intervals)
            @test isequal(after.intervals,before.intervals)
            @test isequal(after.record,before.record)
            original = TOML.parsefile(joinpath(directory,"original.toml"))
            @test haskey(TOML.parse(original["payload_toml"]),"fit")
        end
    end

    @testset "Known and estimated-noise confirmation configuration" begin
        mktempdir() do directory
            defaults = ce_options(["--stage=develop","--models=spline8","--operators=integral",
                "--methods=bootstrap_t","--noise-modes=known,estimated","--iterations=4",
                "--rhos=-8,-4","--ngrid=7"])
            participants = [Dict("operator"=>"integral","family"=>"spline","model"=>"spline8",
                "method"=>"bootstrap_t","noise_mode"=>mode) for mode in ("known","estimated")]
            lock = Dict("schema"=>"calibration-lock-v1","source_stage"=>"develop",
                "code_sha256"=>ce_code_hash(),"confirmation_datasets"=>2,
                "confirmation_bootstrap"=>99,"confirmation_nsim"=>20,
                "options"=>defaults.canonical,"participants"=>participants)
            rd_write_toml(joinpath(directory,"confirmation-lock.toml"),lock)
            plan = Dict("stage"=>"confirm","options"=>defaults.canonical,"seeds"=>[1,2],
                "nboot"=>99,"nsim"=>20,"participants"=>participants,"code_sha256"=>ce_code_hash())
            recovered = ceb_configuration(directory,plan)
            @test recovered.nboot==99 && recovered.noise_modes==["known","estimated"]
            wrong = deepcopy(plan)
            wrong["participants"]=participants[1:1]
            @test_throws ErrorException ceb_configuration(directory,wrong)
            wrong = deepcopy(plan)
            wrong["nboot"]=3
            @test_throws ArgumentError ceb_configuration(directory,wrong)
            wrong = deepcopy(plan)
            wrong["seeds"]=[1,3]
            @test_throws ErrorException ceb_configuration(directory,wrong)
        end

        @testset "Campaign pause preserves the last committed attempt" begin
            mktempdir() do directory
                flag = joinpath(directory,"PAUSE")
                write(flag,"disk reserve reached")
                ran = Ref(false)
                withenv("CALIBRATION_PAUSE_FILE"=>flag) do
                    @test_throws ErrorException ceb_component(joinpath(directory,"attempt.toml"),Dict()) do
                        ran[] = true
                        Dict("computed"=>true)
                    end
                    @test !ran[]
                    @test !isfile(joinpath(directory,"attempt.toml"))
                end
            end
        end
    end
end

end
