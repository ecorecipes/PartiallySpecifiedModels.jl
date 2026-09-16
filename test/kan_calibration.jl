module KANCalibrationTests

using Test
include("../benchmarks/kan/calibrate_uncertainty.jl")

struct CalibrationFailure end
PSM.SciMLBase.solve(prob::PSMProblem,::CalibrationFailure) =
    throw(DomainError(0,"intentional base-fit failure"))

mutable struct SelectiveRefitter
    calls::Int
end
function PSM.SciMLBase.solve(prob::PSMProblem,alg::SelectiveRefitter)
    alg.calls += 1
    id = alg.calls
    id == 2 && throw(DomainError(id,"intentional refit failure"))
    beta = fill(Float64(id),PSM.n_total_params(prob))
    parameters = PSM.ComponentArray(r=beta)
    id == 4 && (parameters[1] = Inf)
    fitted = ones(size(prob.data_values))
    PSMSolution(parameters,0.0,sum(abs2,prob.data_values-fitted),0.0,Float64[],
        fitted,copy(prob.data_values),copy(prob.data_times),Dict{Symbol,Any}(:r=>x->Float64(id)),
        (V_beta=Matrix{Float64}(I,length(beta),length(beta)),sigma2=0.04))
end

@testset "KAN uncertainty calibration contracts" begin
    @testset "Independent trajectory and fixed query geometry" begin
        r = uc_reference("logistic")
        analytic = [2/(1+(2/initial-1)*exp(-0.9t)) for t in r.times,initial in UC_INITIAL]
        # Independent analytic comparison measured <=6.90e-12;
        # the 1e-8 allowance leaves >1400x headroom for solve error.
        @test maximum(abs,r.values-analytic) < 1e-8
        for case in UC_CASES
            reference = uc_reference(case)
            ds = uc_dataset(case,17,reference)
            same = uc_dataset(case,17,reference)
            other = uc_dataset(case,18,reference)
            @test ds.values == same.values
            @test ds.values != other.values
            @test ds.reference == other.reference
            @test size(ds.values) == (21,2)
            queries = uc_queries(case,reference)
            @test length(queries) == 16
            @test count(q->q.region=="in_range",queries) == 11
            @test count(q->q.region=="below_range",queries) == 2
            @test count(q->q.region=="above_range",queries) == 3
            @test all(q->UC_DOMAIN[1]<q.x<UC_DOMAIN[2],queries)
        end
        @test nparams(uc_approximator("kan12")) == 12
        @test nparams(uc_approximator("spline8")) == 8
        @test initial_params(uc_approximator("kan12")) == initial_params(uc_approximator("kan12"))
        @test_throws ArgumentError uc_approximator("unknown")
        @test_throws ArgumentError uc_options(["--seeds=1,1"])
        @test_throws ArgumentError uc_options(["--nboot=2"])
        @test_throws ArgumentError uc_options(["--levels=1.0"])
        @test uc_options(["--seeds=17","--sensitivity-seeds="]).sensitivity_seeds == Int[]
    end

    @testset "Prefix budgets count attempted, not successful, replicates" begin
        values = [1.0 3.0 5.0 6.0 7.0; NaN 3.0 Inf 6.0 7.0]
        attempts = [1,3,5,6,7]
        prefix = uc_percentile(values,attempts,5,0.9)
        @test prefix.n_success == 3
        @test prefix.usable == [3,1]
        @test prefix.lower[1] == quantile([1.0,3.0,5.0],(1-0.9)/2)
        @test prefix.upper[1] == quantile([1.0,3.0,5.0],1-(1-0.9)/2)
        @test isnan(prefix.lower[2]) && isnan(prefix.upper[2])
        short = uc_percentile(values,attempts,3,0.9)
        @test short.n_success == 2
        @test all(isnan,short.lower)
        @test_throws DimensionMismatch uc_percentile(values,attempts[1:4],5,0.9)
        @test_throws ArgumentError uc_percentile(values,reverse(attempts),5,0.9)

        ds = uc_dataset("logistic",17)
        p = uc_problem(uc_approximator("spline8"),ds)
        base = solve(p,SelectiveRefitter(0))
        recorder = UCRefitter(SelectiveRefitter(0))
        full = bootstrap(base,p,recorder;nboot=7,uf_points=Dict(:r=>[0.4,1.3]),
                         rng=StableRNG(41))
        @test recorder.attempted == 7
        @test recorder.finite_attempts == [1,3,5,6,7]
        @test getindex.(recorder.diagnostics,"attempt") == recorder.finite_attempts
        @test full.n_success == 5
        prefix = uc_percentile(full.uf_values[:r],recorder.finite_attempts,5,full.level)
        small = bootstrap(base,p,UCRefitter(SelectiveRefitter(0));nboot=5,
                          uf_points=Dict(:r=>[0.4,1.3]),rng=StableRNG(41))
        @test prefix.lower == small.ci_uf[:r].lower
        @test prefix.upper == small.ci_uf[:r].upper
    end

    @testset "Failures remain in the requested interval population" begin
        query = (point_id=1,x=0.5,truth=1.0,region="in_range",training_min=0.2,training_max=1.8)
        common = (; case="logistic",model="kan12",seed=17,method="bootstrap",level=0.95)
        absent = uc_interval(query,1.0,NaN,NaN;common...)
        @test !absent.available && !absent.covered && isnan(absent.width)
        failed = uc_interval(query,1.0,0.0,2.0;common...,status="fit_failed")
        @test !failed.available && !failed.covered
        reversed = uc_interval(query,1.0,2.0,0.0;common...)
        @test !reversed.available && !reversed.covered
        valid = uc_interval(query,NaN,0.0,2.0;common...)
        @test valid.available && valid.covered
        miss = uc_interval(query,1.0,2.0,3.0;common...)
        @test miss.available && !miss.covered
        cfg = uc_options(["--seeds=17","--nboot=3","--sensitivity-seeds="])
        ds = uc_dataset("logistic",17)
        queries = uc_queries("logistic",uc_reference("logistic"))
        run = uc_fit_dataset("kan12",ds,queries,cfg;algorithm=CalibrationFailure())
        @test run.fit_row.fit_status == "fit_failed"
        @test run.bootstrap_row.attempted == 0
        @test length(run.intervals) == 16*2*2
        @test all(r->!r.available && !r.covered,run.intervals)
        @test all(r->r.status=="fit_failed",run.intervals)
        @test occursin("numerical",uc_failure_message(DomainError(0,"numerical failure")))
        @test_throws ErrorException uc_failure_message(ErrorException("unexpected programming fault"))
        @test_throws ArgumentError uc_failure_message(ArgumentError("bad configuration"))
    end

    @testset "Persistent calibration records" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                output = joinpath(dir,"study")
                args = ["--cases=logistic","--models=spline8","--seeds=17","--nboot=3",
                        "--sensitivity-seeds=17","--sensitivity-boot=5","--iterations=3","--output=$output"]
                cfg = uc_main(args)
                rows = rd_read_csv(joinpath(output,"intervals.csv"))
                @test length(rows) == 16*2*3
                @test length(rd_read_csv(joinpath(output,"fits.csv"))) == 1
                saved = TOML.parsefile(joinpath(output,"replicates","logistic__spline8__17.toml"))
                @test saved["budgets"] == [3,5]
                @test saved["points"] == UC_POINTS
                @test saved["bootstrap_attempted"] == 5
                @test length(saved["bootstrap_values"]) == length(saved["finite_attempts"])
                meta = TOML.parsefile(joinpath(output,"metadata.toml"))
                @test meta["evaluation_source_sha256"] == source_hash()
                @test meta["design_sha256"] == rd_digest(joinpath(output,"design.toml"))
                @test_throws ErrorException uc_main(args)
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
