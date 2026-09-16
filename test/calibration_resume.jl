module CalibrationResumeTests

using Test
include("../benchmarks/calibration/run.jl")
include("../benchmarks/calibration/resume.jl")

@testset "Interrupted calibration recovery" begin
    points = collect(range(0.3,1.8;length=11))
    ds = ce_dataset("smoke","integral","baseline","iid",0.015,"poly_a",1,points)
    cfg = ce_options(["--stage=smoke","--operators=integral","--models=spline8",
        "--methods=conditional,split_bias_bank,split_bias_ellipsoid",
        "--rhos=-8,-4","--ngrid=11","--iterations=8","--nsim=200"])
    model = ce_model("spline8")
    methods = cfg.methods
    evaluated = ce_evaluate_model(model,ds,cfg,"known",methods)
    @test evaluated.record["status"]=="ok"
    original = deepcopy(evaluated.record)
    replay = cer_replay(evaluated.record,model,ds,cfg,"known",methods)
    @test length(replay.rows)==length(evaluated.intervals)
    @test isequal(original,evaluated.record)
    for (a,b) in zip(replay.rows,evaluated.intervals)
        @test (a.method,a.interval,a.target_id,a.available,a.covered)==
            (b.method,b.interval,b.target_id,b.available,b.covered)
        if a.method=="conditional" && a.interval=="simultaneous"
            # The 669-fit replay probe measured max SE error 1.04e-6,
            # relative 1.44e-5; 2e-4 leaves >13x relative headroom.
            @test isapprox(a.lower,b.lower;rtol=2e-4,atol=1e-8)
            @test isapprox(a.upper,b.upper;rtol=2e-4,atol=1e-8)
        else
            @test isequal(a,b)
        end
    end
    @test_throws ArgumentError cer_replay(original,model,ds,cfg,"known",["bootstrap_t"])
    @test_throws ErrorException cer_replay(original,model,ds,cfg,"estimated",methods)
    damaged = deepcopy(original)
    damaged["se"] .*= 2
    @test_throws ErrorException cer_replay(damaged,model,ds,cfg,"known",methods)

    mktempdir() do directory
        fit_path = joinpath(directory,"fits","job.toml")
        checkpoint = joinpath(directory,"checkpoints","job.toml")
        completed = cer_bundle(checkpoint,fit_path,"dataset","protocol",methods;
            result=evaluated,origin="new_fit_exact_endpoints")
        fit_bytes = read(fit_path,String)
        @test isequal(completed.record,evaluated.record)
        @test isequal(completed.rows,evaluated.intervals)
        @test isfile(checkpoint)
        rm(fit_path)
        # Simulate interruption after checkpoint publication but before
        # the ordinary fit record is published.
        restored = cer_bundle(checkpoint,fit_path,"dataset","protocol",methods)
        @test read(fit_path,String)==fit_bytes
        @test isequal(restored.rows,evaluated.intervals)
        @test all(keys(r)==CER_INTERVAL_FIELDS for r in restored.rows)
        @test_throws ErrorException cer_bundle(checkpoint,fit_path,"changed","protocol",methods)
        @test_throws ErrorException cer_bundle(checkpoint,fit_path,"dataset","changed",methods)
        cer_atomic(fit_path,"changed = true\n")
        @test_throws ErrorException cer_bundle(checkpoint,fit_path,"dataset","protocol",methods)
    end
    mktempdir() do directory
        path = joinpath(directory,"data.toml")
        id = "integral__baseline__iid__0.015__poly_a__1"
        cer_dataset(path,ds,id)
        bytes = read(path,String)
        cer_dataset(path,ds,id)
        @test read(path,String)==bytes
        shifted = merge(ds,(observed=ds.observed.+0.1,))
        @test_throws ErrorException cer_dataset(path,shifted,id)
        @test_throws ErrorException cer_preserve(path,"changed = true\n")
        @test read(path,String)==bytes
    end
    @testset "Resuming missing work and finalizing complete outputs" begin
        mktempdir() do directory
            output = joinpath(directory,"results")
            ce_run(["--stage=smoke","--operators=integral","--models=spline8,kan12",
                "--methods=conditional,split_bias_bank,split_bias_ellipsoid",
                "--iterations=4","--rhos=-8,-4","--ngrid=7","--nsim=20","--output=$output"])
            previous = TOML.parsefile(joinpath(output,"metadata.toml"))
            names = sort(readdir(joinpath(output,"fits")))
            original_bytes = read(joinpath(output,"fits",names[1]),String)
            # Emulate an interruption with one completed fit and no
            # final CSV/manifest; restart must not repeat the completed fit.
            rm(joinpath(output,"fits",names[2]))
            for name in ("metadata.toml","fits.csv","intervals.csv","weak_pairs.csv")
                rm(joinpath(output,name))
            end
            @test cer_execute(directory;recover_only=true)===nothing
            @test length(readdir(joinpath(output,"recovery","fits")))==1
            @test !isfile(joinpath(output,"fits",names[2]))
            @test cer_execute(directory)==output
            @test read(joinpath(output,"fits",names[1]),String)==original_bytes
            completed = TOML.parsefile(joinpath(output,"metadata.toml"))
            @test completed["complete"]
            @test completed["fit_count"]==previous["fit_count"]==2
            @test completed["interval_count"]==previous["interval_count"]
            @test completed["recovery"]["original_saved_fits"]==1
            @test all(rd_digest(joinpath(output,name))==sha for (name,sha) in completed["outputs_sha256"])
            @test cer_execute(directory)==output
        end
    end
end

end
