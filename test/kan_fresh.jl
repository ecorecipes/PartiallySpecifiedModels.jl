module KANFreshTests

using Test
include("../benchmarks/kan/calibrate_families.jl")

struct FreshFixtureSolver end
const CF_DUAL_CALLS = Ref(0)
function cf_probe_activation(x)
    x isa PSM.ForwardDiff.Dual && (CF_DUAL_CALLS[] += 1)
    tanh(x)
end

function PSM.SciMLBase.solve(prob::PSMProblem,::FreshFixtureSolver)
    a = only(prob.approximators)
    beta = initial_params(a)
    fitted = ones(size(prob.data_values))
    PSMSolution(PSM.ComponentArray(r=beta),0.0,sum(abs2,prob.data_values-fitted),0.0,Float64[],
        fitted,copy(prob.data_values),copy(prob.data_times),
        Dict{Symbol,Any}(:r=>build_evaluator(a,beta)),
        (V_beta=Matrix{Float64}(I,length(beta),length(beta)),sigma2=0.04))
end

@testset "Fresh family coverage contracts" begin
    @testset "Declared populations and explicit priors" begin
        c,s = cf_options(String[]),cf_options(["--stage=screen"])
        @test c.seeds == collect(2001:2100) && c.nboot == 99
        @test c.sensitivity_seeds == collect(2001:2020)
        @test s.seeds == collect(2001:2020) && s.nboot == 19
        @test isempty(s.sensitivity_seeds)
        @test isempty(intersect(c.seeds,1001:1100))
        @test_throws ArgumentError cf_options(["--stage=unknown"])
        @test_throws ArgumentError cf_options(["--seeds=1,1"])
        @test_throws ArgumentError cf_options(["--models=missing"])
        counts = [8,12,12,12,12,13,28]
        ranks = [[6],[10],[10],[12],[10],[13],[10,10]]
        for (model,count,rank) in zip([CF_CONFIRM;CF_SCREEN],counts,ranks)
            spec = cf_model(model)
            @test nparams(spec.fit) == nparams(spec.report) == count
            @test spec.metadata["penalty_ranks"] == rank
            @test initial_params(spec.fit) == cf_model(model).metadata["initial_parameters"]
            @test penalty_matrix(spec.fit) == penalty_matrix(spec.report)
            @test penalty_blocks(spec.fit) == penalty_blocks(spec.report)
            @test all(isfinite,spec.metadata["initial_query_values"])
            if model in ("spline8","spline12","spde12","mlp13","kan28_free")
                # Pilot maximum constant-response error 5.56e-17;
                # 1e-12 leaves >17000x headroom.
                @test maximum(abs,spec.metadata["initial_query_values"] .- .5) < 1e-12
            end
        end
        gp = cf_model("gp12_fixed").report
        @test !gp.adapt && gp.lengthscale == 0.4 && gp.kernel == :matern52
        b = sin.(collect(1:12))
        @test build_evaluator(cf_model("spline12").report,b).(UC_POINTS) ==
              build_evaluator(cf_model("spde12").report,b).(UC_POINTS)
    end

    @testset "Reporting uses the underlying KAN AD path" begin
        layer(i,o) = LuxKANLinear(i,o;grid_size=3,standalone_spline_scale=false,
                                  base_activation=cf_probe_activation)
        a = KANApproximator(:r,Lux.Chain(layer(1,2),layer(2,1));input_domains=(UC_DOMAIN,))
        wrapped = InitializedApprox(:r,a,initial_params(a),penalty_matrix(a))
        cfg = cf_options(["--seeds=1907","--nboot=3","--sensitivity-seeds="])
        ds = uc_dataset("nonlinear",1907)
        queries = uc_queries("nonlinear",uc_reference("nonlinear"))
        CF_DUAL_CALLS[] = 0
        run = uc_fit_dataset("fixture",ds,queries,cfg;algorithm=FreshFixtureSolver(),
                             approximator=wrapped,reporting_approximator=a)
        @test CF_DUAL_CALLS[] > 0
        @test all(isfinite,run.record["standard_errors"])
        @test run.bootstrap_row.n_success == 3
        @test_throws ArgumentError uc_fit_dataset("fixture",ds,queries,cfg;
            algorithm=FreshFixtureSolver(),approximator=wrapped,
            reporting_approximator=cf_model("kan12_free").report)
        old = uc_fit_dataset("spline8",ds,queries,cfg;algorithm=FreshFixtureSolver())
        explicit = uc_fit_dataset("spline8",ds,queries,cfg;algorithm=FreshFixtureSolver(),
                                  approximator=uc_approximator("spline8"))
        @test isequal(old.intervals,explicit.intervals)
        @test old.record["bootstrap_values"] == explicit.record["bootstrap_values"]
    end

    @testset "Durable family metadata and old defaults" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                output = joinpath(dir,"fresh")
                args = ["--cases=nonlinear","--models=spline8,kan12_free","--seeds=1907",
                        "--nboot=3","--sensitivity-seeds=","--iterations=3","--output=$output"]
                cf_main(args)
                design = TOML.parsefile(joinpath(output,"design.toml"))
                @test design["fresh_against_original_calibration"]
                @test design["study_stage"] == "confirm"
                @test design["family_specs"]["kan12_free"]["penalty_ranks"] == [10]
                @test length(rd_read_csv(joinpath(output,"intervals.csv"))) == 2*16*2*2
                @test_throws ErrorException cf_main(args)
                cfg = merge(uc_options(["--seeds=17","--nboot=3"]),(
                    output=joinpath(dir,"invalid"),))
                @test_throws ArgumentError uc_run(cfg;design_extra=Dict("seeds"=>[99]))
                @test !ispath(cfg.output)
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
