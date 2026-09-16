module SimultaneousBandTests

using Test, Random, LinearAlgebra, Statistics
using PartiallySpecifiedModels, Lux, FluxKAN
const PSM=PartiallySpecifiedModels

function fixture(approximators)
    rhs!(du,u,p,t)=fill!(du,0)
    prob=PSMProblem(rhs!,[1.0],(0.0,1.0),approximators;
        data_times=[0.0,0.5,1.0],data_values=reshape([1.0,1.1,0.9],:,1))
    names=Tuple(a.name for a in approximators)
    blocks=Tuple(initial_params(a) for a in approximators)
    params=PSM.ComponentArray(NamedTuple{names}(blocks))
    functions=Dict{Symbol,Any}(a.name=>build_evaluator(a,initial_params(a)) for a in approximators)
    n=length(params)
    sol=PSMSolution(params,0.0,0.02,0.0,Float64[],ones(3,1),copy(prob.data_values),
        copy(prob.data_times),functions,(V_beta=Matrix{Float64}(I,n,n),sigma2=0.04))
    prob,sol
end

function boot_fixture(sol,name,values;grid=nothing,points=nothing)
    nboot=size(values,2)
    uf=Dict(name=>Matrix{Float64}(values))
    grids=grid===nothing ? Dict{Symbol,Vector{Float64}}() : Dict(name=>Float64.(grid))
    queries=points===nothing ? Dict{Symbol,Matrix{Float64}}() : Dict(name=>Matrix{Float64}(points))
    ci=Dict{Symbol,NamedTuple{(:lower,:upper),Tuple{Vector{Float64},Vector{Float64}}}}(
        name=>(lower=[quantile(filter(isfinite,values[i,:]),0.025) for i in axes(values,1)],
               upper=[quantile(filter(isfinite,values[i,:]),0.975) for i in axes(values,1)]))
    BootstrapResult(repeat(permutedims(collect(sol.parameters)),nboot,1),ones(3,1,nboot),
        uf,grids,(lower=zeros(3,1),upper=ones(3,1)),ci,0.95,nboot,queries,
        Dict(name=>[count(isfinite,values[i,:]) for i in axes(values,1)]))
end

@testset "Simultaneous function bands" begin
    @testset "Maximum statistics and complete curve draws" begin
        deviations=[0.0 1.0 -2.0 3.0; 2.0 -4.0 0.0 8.0]
        @test PSM._max_standardized_deviations(deviations,[1.0,2.0]) == [1.0,2.0,2.0,4.0]
        @test_throws DimensionMismatch PSM._max_standardized_deviations(deviations,[1.0])
        @test_throws DomainError PSM._max_standardized_deviations(deviations,[0.0,2.0])
        values=[1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0]
        fitted=[3.0,3.0]
        joint=PSM._bootstrap_grid_band(values,fitted;level=0.6)
        expected=quantile([2.0,1.0,0.0,1.0,2.0]./sqrt(2.5),0.6)
        @test joint.critical == expected
        permutation=[3,2,1,4,5]
        @test PSM._bootstrap_grid_band(values[:,permutation],fitted;level=0.6).critical == joint.critical
        scrambled=copy(values)
        scrambled[2,:] .= values[2,permutation]
        @test PSM._bootstrap_grid_band(scrambled,fitted;level=0.6).critical > joint.critical
        @test_throws DomainError PSM._bootstrap_grid_band([NaN 2.0 3.0],[2.0];level=0.95)
        @test_throws DomainError PSM._bootstrap_grid_band(ones(1,3),[0.0];level=0.95)
        constant=PSM._bootstrap_grid_band(ones(1,3),[1.0];level=0.95)
        @test constant.critical==0 && constant.lower==constant.upper==[1.0]
        @test_throws DomainError PSM._bootstrap_grid_band(fill(0.1,1,99),[0.2];level=0.95)
        @test_throws ArgumentError PSM._bootstrap_grid_band(ones(1,2),[1.0];level=0.95)
    end

    @testset "Gaussian correlation and covariance integrity" begin
        C=[1.0 0.4;0.4 2.0]
        before=copy(C)
        root=PSM._band_covariance_root(C)
        # Eigen-factor reconstruction measured <=6.67e-16;
        # 1e-11 leaves >14000x roundoff headroom.
        @test isapprox(root*root',C;atol=1e-11,rtol=1e-11)
        @test C==before
        @test_throws DomainError PSM._band_covariance_root([1.0 0.2;0.5 1.0])
        @test_throws DomainError PSM._band_covariance_root([1.0 0.0;0.0 -0.1])
        @test_throws DomainError PSM._band_covariance_root([Inf;;])
        @test_throws DimensionMismatch PSM._band_covariance_root(ones(2,1))
        single=PSM._gaussian_grid_band(ones(1,1),[4.0;;];level=0.95,nsim=20,rng=Random.Xoshiro(1))
        @test single.se==[2.0]
        @test single.critical==PSM._qnorm(0.975) && single.n_draws==0
        two=PSM._gaussian_grid_band(Matrix{Float64}(I,2,2),Matrix{Float64}(I,2,2);
            level=0.95,nsim=100_000,rng=Random.Xoshiro(91))
        exact=PSM._qnorm((1+sqrt(0.95))/2)
        # The independent two-normal maximum has this analytic quantile.
        # Measured Monte Carlo discrepancy 0.00340; 0.04 gives >11x headroom.
        @test abs(two.critical-exact)<0.04
        @test two.critical>single.critical
        @test two.n_draws==100_000
        zero=PSM._gaussian_grid_band(zeros(2,1),ones(1,1);level=0.95,nsim=20,rng=Random.Xoshiro(1))
        @test zero.critical==0 && zero.n_draws==0 && zero.se==zeros(2)
    end

    @testset "Covariance API preserves pointwise defaults and ownership" begin
        a=BSplineApproximator(:r,(0.0,1.0),4;initial=x->1+x)
        prob,sol=fixture([a])
        points=[0.1,0.4,0.8]
        plain=confidence_band(sol,prob;uf_points=Dict(:r=>points))[:r]
        explicit=confidence_band(sol,prob;uf_points=Dict(:r=>points),interval=:pointwise)[:r]
        @test plain==explicit
        @test keys(plain)==(:points,:fitted,:lower,:upper,:se)
        rng=Random.Xoshiro(12)
        expected_rng=copy(rng)
        confidence_band(sol,prob;uf_points=Dict(:r=>points),rng)
        @test rand(rng)==rand(expected_rng)
        band=confidence_band(sol,prob;uf_points=Dict(:r=>points),interval=:simultaneous,
                             nsim=2000,rng=Random.Xoshiro(7))[:r]
        again=confidence_band(sol,prob;uf_points=Dict(:r=>points),interval=:simultaneous,
                              nsim=2000,rng=Random.Xoshiro(7))[:r]
        @test band==again
        @test band.interval==:simultaneous && band.scope==:function_grid
        @test band.conditioning==:fixed_smoothing && band.method==:gaussian_delta
        @test band.critical>=PSM._qnorm(0.975)
        @test band.fitted==plain.fitted && band.n_draws==2000
        # Factor-space and quadratic-form SEs differ at rounding scale;
        # 1e-10 is >250x the measured linear covariance oracle discrepancy.
        @test isapprox(band.se,plain.se;atol=1e-10,rtol=1e-10)
        band.points[1,1]=-99
        @test points==[0.1,0.4,0.8]
        @test_throws ArgumentError confidence_band(sol,prob;interval=:unknown)
        @test_throws ArgumentError confidence_band(sol,prob;interval=:simultaneous,nsim=1)
        @test confidence_band(sol,prob;interval=:simultaneous,uf_ngrid=1)[:r].n_draws==0
    end

    @testset "Outer curves and mixed coefficient offsets" begin
        first=BSplineApproximator(:s,(0.0,1.0),4;initial=x->8.0)
        index=SingleIndexApproximator(:g,2,6;
            index_stats=(zeros(2),Matrix{Float64}(I,2,2)),initial=z->0.4+0.2z,
            initial_loadings=[1.0,0.4])
        prob,sol=fixture([first,index])
        grid=[-0.5,0.0,0.5]
        outer=[PSM._eval_approx_at(index,initial_params(index),z) for z in grid]
        values=outer .+ reshape([-0.2,-0.1,0.0,0.1,0.2],1,:)
        bs=boot_fixture(sol,:g,values;grid)
        band=confidence_band(bs,sol,prob;interval=:simultaneous)[:g]
        @test band.fitted==outer
        @test band.grid==grid
        points=[0.2 0.8;1.1 0.4;1.4 1.2]
        full=[sol.unknown_functions[:g](x...) for x in eachrow(points)]
        explicit=boot_fixture(sol,:g,full .+ reshape([-0.2,-0.1,0.0,0.1,0.2],1,:);points)
        @test confidence_band(explicit,sol,prob;interval=:simultaneous)[:g].fitted==full
    end

    @testset "Bootstrap API, scopes and unavailable extremes" begin
        a=BSplineApproximator(:r,(0.0,1.0),4;initial=x->3.0)
        prob,sol=fixture([a])
        values=[1.0 2.0 3.0 4.0 5.0; 2.0 3.0 4.0 5.0 6.0; 3.0 2.0 3.0 4.0 3.0]
        bs=boot_fixture(sol,:r,values;grid=[0.1,0.5,0.9])
        point=confidence_band(bs,sol,prob)[:r]
        @test point.method==:percentile && point.critical===nothing
        @test point.lower==[quantile(values[i,:],(1-bs.level)/2) for i in 1:3]
        band=confidence_band(bs,sol,prob;interval=:simultaneous,level=0.8)[:r]
        reference=PSM._bootstrap_grid_band(values,band.fitted;level=0.8)
        @test band.critical==reference.critical && band.lower==reference.lower
        @test band.conditioning==:refit_procedure && band.usable==fill(5,3)
        subset=confidence_band(bs,sol,prob;interval=:simultaneous,level=0.8,
                               uf_indices=Dict(:r=>[1,3]))[:r]
        @test subset.grid==[0.1,0.9]
        @test subset.critical==PSM._bootstrap_grid_band(values[[1,3],:],subset.fitted;level=0.8).critical
        subset.grid[1]=-2
        @test bs.uf_grid[:r]==[0.1,0.5,0.9]
        for ids in (Int[],[0],[4],[1,1],[true,false])
            @test_throws ArgumentError confidence_band(bs,sol,prob;uf_indices=Dict(:r=>ids))
        end
        @test_throws ArgumentError confidence_band(bs,sol,prob;uf_indices=Dict(:missing=>[1]))
        damaged=deepcopy(bs)
        damaged.uf_values[:r][1,1]=NaN
        @test_throws DomainError confidence_band(damaged,sol,prob;interval=:simultaneous)
        point=@test_logs (:warn,r"incomplete") confidence_band(damaged,sol,prob)[:r]
        @test point.usable==[4,5,5]
    end

    @testset "Multivariate physical points" begin
        a=KANApproximator(:g,LuxKANLinear(2,1;grid_size=3,standalone_spline_scale=false);
                          input_domains=((0.0,2.0),(0.0,2.0)))
        prob,sol=fixture([a])
        points=[0.2 0.3;0.8 1.4;1.6 0.5]
        conditional=confidence_band(sol,prob;uf_points=Dict(:g=>points),
            interval=:simultaneous,nsim=2000,rng=Random.Xoshiro(9))[:g]
        @test conditional.points==points && all(isfinite,conditional.se)
        fitted=Float64[sol.unknown_functions[:g](x...) for x in eachrow(points)]
        bs=boot_fixture(sol,:g,fitted .+ reshape([-0.2,-0.1,0.0,0.1,0.2],1,:);points=points)
        b=confidence_band(bs,sol,prob;interval=:simultaneous,uf_indices=Dict(:g=>[1,2]))[:g]
        @test b.points==points[1:2,:]
        @test b.scope==:function_grid && b.n_draws==5
        b.points[1,1]=-5
        @test bs.uf_points[:g]==points
    end
end

end
