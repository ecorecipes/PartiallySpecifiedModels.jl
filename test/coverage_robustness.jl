module CoverageRobustnessTests

using Test
include("../benchmarks/kan/coverage_robustness.jl")

@testset "Coverage robustness scenario contracts" begin
    reference=cr_scenario("reference")
    sparse=cr_scenario("sparse")
    noisy=cr_scenario("noisy")
    localized=cr_scenario("localized")
    @test length(reference.reference.times)==21
    @test length(sparse.reference.times)==11
    @test noisy.reference.values==reference.reference.values
    @test sparse.reference.values==reference.reference.values[1:2:end,:]
    @test localized.reference.values!=reference.reference.values
    @test cr_settings("reference").sigma==cr_settings("sparse").sigma==0.015
    @test cr_settings("noisy").sigma==0.03
    @test cr_settings("localized").step==0.25
    # The independently specified bump has multiplier1.5 at its centre.
    # Float64 evaluation differs by <=1 ulp; 1e-14 leaves >40x headroom.
    @test isapprox(cr_rate("localized",0.9),0.7425;atol=1e-14,rtol=1e-14)
    @test_throws ArgumentError cr_settings("unknown")
    @test_throws ArgumentError cr_rate("unknown",0.5)

    data_seeds,bootstrap_seeds=Int[],Int[]
    for case in CR_CASES
        scenario=cr_scenario(case)
        ds=scenario.dataset(3907)
        same=scenario.dataset(3907)
        @test ds.values==same.values
        @test ds.values!=scenario.dataset(3908).values
        @test ds.sigma==scenario.metadata["noise_sigma"]
        noise=randn(StableRNG(ds.data_seed),size(ds.values))
        @test ds.values==scenario.reference.values+ds.sigma*noise
        @test all(q->q.truth==cr_rate(scenario.metadata["response"],q.x),scenario.queries)
        @test count(q->q.region=="in_range",scenario.queries)==11
        push!(data_seeds,ds.data_seed)
        push!(bootstrap_seeds,ds.bootstrap_seed)
    end
    @test allunique(data_seeds) && allunique(bootstrap_seeds)
    @test isempty(intersect(data_seeds,bootstrap_seeds))

    # Independent separable-ODE inverse-time quadrature. It does not call
    # cr_rate or the reference RHS; measured discrepancies <=1.99e-11,
    # leaving >5000x headroom at 1e-7.
    for j in 1:2
        lo,hi=UC_INITIAL[j],localized.reference.values[13,j]
        n=20_000
        h=(hi-lo)/n
        integrand(x)=1/(x*0.9*(1-x/2)*(1+0.5exp(-((x-0.9)/0.25)^2)))
        t=h/3*(integrand(lo)+integrand(hi)+4sum(integrand(lo+i*h) for i in 1:2:n-1)+
               2sum(integrand(lo+i*h) for i in 2:2:n-2))
        @test abs(t-3.0)<1e-7
    end

    threads=BLAS.get_num_threads()
    try
        mktempdir() do dir
            output=joinpath(dir,"robust")
            args=["--cases=sparse,noisy,localized","--models=spline8","--fractions=1,0.25",
                  "--seeds=3907","--nboot=3","--iterations=6","--output=$output"]
            cr_main(args)
            design=TOML.parsefile(joinpath(output,"design.toml"))
            @test design["noise_sigma"]==Dict("sparse"=>0.015,"noisy"=>0.03,"localized"=>0.015)
            @test length(rd_read_csv(joinpath(output,"intervals.csv")))==3*2*16*2*2
            for case in ("sparse","noisy","localized")
                data=TOML.parsefile(joinpath(output,"datasets","$(case)__3907.toml"))
                @test length(data["times"])==(case=="sparse" ? 11 : 21)
                for fraction in (1.0,0.25)
                    saved=TOML.parsefile(joinpath(output,"replicates",
                        "$(case)__spline8_fraction_$(fraction)__3907.toml"))
                    @test saved["bootstrap_seed"]==cr_scenario(case).dataset(3907).bootstrap_seed
                    @test saved["procedure"]["fraction"]==fraction
                end
            end
            @test_throws ErrorException cr_main(args)
        end
    finally
        BLAS.set_num_threads(threads)
    end
end

end
