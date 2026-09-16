module KANUncertaintyTests

using Test
using Random, LinearAlgebra, Statistics
using PartiallySpecifiedModels, Lux, FluxKAN
using PartiallySpecifiedModels: solve

const PSM = PartiallySpecifiedModels
layer(ni,no) = LuxKANLinear(ni,no;grid_size=3,standalone_spline_scale=false)

function synthetic_solution(prob, beta; covariance=Matrix{Float64}(I,length(beta),length(beta)), scale=0.04)
    names, blocks = Symbol[], Vector{Float64}[]
    functions = Dict{Symbol,Any}()
    offset = 0
    for a in prob.approximators
        block = Float64.(beta[offset+1:offset+nparams(a)])
        push!(names,a.name)
        push!(blocks,block)
        functions[a.name] = build_evaluator(a,block)
        offset += nparams(a)
    end
    parameters = PSM.ComponentArray(NamedTuple{Tuple(names)}(Tuple(blocks)))
    fitted = ones(size(prob.data_values))
    PSMSolution(parameters,0.0,sum(abs2,prob.data_values-fitted),0.0,Float64[],
        fitted,copy(prob.data_values),copy(prob.data_times),functions,
        (V_beta=covariance,sigma2=scale))
end

struct QueryRefitter end
function PSM.SciMLBase.solve(prob::PSMProblem, ::QueryRefitter)
    beta = reduce(vcat,initial_params.(prob.approximators))
    beta[1] = mean(prob.data_values)
    synthetic_solution(prob,beta)
end

struct IncompleteRefitter
    calls::Threads.Atomic{Int}
    failed_fits::Bool
end
IncompleteRefitter(; failed_fits=false) = IncompleteRefitter(Threads.Atomic{Int}(0),failed_fits)
function PSM.SciMLBase.solve(prob::PSMProblem, alg::IncompleteRefitter)
    id = Threads.atomic_add!(alg.calls,1)+1
    alg.failed_fits && id == 1 && throw(DomainError(id,"intentional numerical fit failure"))
    sol = synthetic_solution(prob,reduce(vcat,initial_params.(prob.approximators)))
    if alg.failed_fits
        id == 2 && (sol.fitted_values[1,1] = Inf)
        id == 3 && (sol.parameters[1] = Inf)
    elseif id == 1
        delete!(sol.unknown_functions,:r)
    else
        sol.unknown_functions[:r] = x -> begin
            x == 0.5 && return NaN
            x == 0.2 && id == 2 && return Inf
            x == 0.2 && id == 3 && throw(DomainError(x,"intentional query failure"))
            Float64(id)
        end
    end
    sol
end

function lux_reference(a,beta,point)
    input = [(x-d[1])/(d[2]-d[1])*2-1 for (x,d) in zip(point,a.input_domains)]
    only(first(Lux.apply(a.model,reshape(input,:,1),
                        PSM.ComponentArray(beta,a.param_axes),a.state)))
end

function query_problem(approximators)
    rhs!(du,u,p,t) = fill!(du,0)
    times = collect(0.0:0.2:1.0)
    values = reshape(1 .+ 0.1sin.(collect(1:6)),:,1)
    PSMProblem(rhs!,[1.0],(0.0,1.0),approximators;data_times=times,data_values=values)
end

@testset "KAN pointwise uncertainty" begin
    @testset "Linear edge covariance and mixed parameter offsets" begin
        spline = BSplineApproximator(:s,(0.0,1.0),4)
        a = KANApproximator(:g,layer(2,1);input_domains=((0.0,2.0),(10.0,20.0)))
        prob = query_problem([spline,a])
        beta = [initial_params(spline); 0.2cos.(collect(1:nparams(a)))]
        covariance = Matrix(Diagonal(collect(range(0.02,0.2,length=length(beta)))))
        sol = synthetic_solution(prob,beta;covariance,scale=0.25)
        points = [0.2 11.0; 1.0 15.0; 1.7 19.0]
        bands = confidence_band(sol,prob;uf_ngrid=5,uf_points=Dict(:g=>points))
        @test length(bands[:s].grid) == 5
        @test bands[:g].points == points
        @test length(bands[:g].fitted) == 3
        @test !haskey(bands[:g],:grid)
        grid = only(a.layers).knots
        idx = 5:length(beta)
        for (i,point) in enumerate(eachrow(points))
            x = [point[1]-1, (point[2]-10)/5-1]
            base = FluxKAN.SiLU.(x)
            spline_basis = reduce(vcat,[vec(FluxKAN.bspline_basis(reshape([x[j]],1,1),vec(grid[j,:]),3))
                                       for j in 1:2])
            design = [base; spline_basis]
            expected = sqrt(dot(design,0.25covariance[idx,idx]*design))
            # Independent raw-basis calculations are Float64 algebra;
            # the edge oracle measured <=5.56e-17, with >1e4x headroom.
            @test isapprox(bands[:g].se[i],expected;atol=1e-12,rtol=1e-12)
            @test isapprox(bands[:g].fitted[i],dot(design,beta[idx]);atol=1e-12,rtol=1e-12)
        end
        before = copy(points)
        bands[:g].points[1,1] = -100
        @test points == before
        @test_throws ArgumentError confidence_band(sol,prob)
        @test_throws ArgumentError confidence_band(sol,prob;level=1.0)
        @test_throws ArgumentError confidence_band(sol,prob;uf_points=Dict(:absent=>points))
        @test_throws DimensionMismatch confidence_band(sol,prob;uf_points=Dict(:g=>zeros(3,1)))
        @test_throws ArgumentError confidence_band(sol,prob;uf_points=Dict(:g=>zeros(0,2)))
        @test_throws DomainError confidence_band(sol,prob;uf_points=Dict(:g=>[NaN 1.0]))
        bad = synthetic_solution(prob,beta;covariance=-covariance,scale=0.25)
        @test_throws DomainError confidence_band(bad,prob;uf_points=Dict(:g=>points))
        @test_throws DimensionMismatch confidence_band(
            synthetic_solution(prob,beta;covariance=zeros(1,1)),prob;uf_points=Dict(:g=>points))
        for scale in (-1.0,Inf,NaN)
            @test_throws DomainError confidence_band(
                synthetic_solution(prob,beta;scale),prob;uf_points=Dict(:g=>points))
        end
        @test_throws ArgumentError confidence_band(sol,prob;uf_points=points)
        @test_throws ArgumentError confidence_band(sol,prob;uf_points=Dict("g"=>points))
        @test_throws ArgumentError confidence_band(sol,prob;uf_points=Dict(:g=>["a" "b"]))
        @test_throws DomainError PSM._band_standard_error([Inf],ones(1,1))
        @test PSM._band_standard_error([1.0,-1.0],ones(2,2)) == 0
    end

    @testset "Composed KAN and unary defaults" begin
        a = KANApproximator(:g,Lux.Chain(layer(2,2),layer(2,1));
                           input_domains=((0.0,2.0),(0.0,2.0)))
        prob = query_problem([a])
        beta = 0.1sin.(collect(1:nparams(a)))
        covariance = Matrix{Float64}(I,length(beta),length(beta)) + 0.02ones(length(beta),length(beta))
        sol = synthetic_solution(prob,beta;covariance)
        points = [0.4 0.8; 1.0 1.2; 1.7 0.3]
        band = confidence_band(sol,prob;uf_points=Dict(:g=>points))[:g]
        @test all(isfinite,band.se) && all(>(0),band.se)
        @test band.fitted == [build_evaluator(a,beta)(x...) for x in eachrow(points)]
        for (i,point) in enumerate(eachrow(points))
            jac = map(eachindex(beta)) do j
                plus,minus = copy(beta),copy(beta)
                plus[j] += 1e-5
                minus[j] -= 1e-5
                (lux_reference(a,plus,point)-lux_reference(a,minus,point))/2e-5
            end
            expected = sqrt(dot(jac,0.04covariance*jac))
            # Independent Lux finite differences measured a maximum
            # sensitivity error 1.92e-12; 1e-9 permits >500x that scale.
            @test isapprox(band.se[i],expected;atol=1e-9,rtol=1e-9)
        end
        unary = KANApproximator(:r,layer(1,1);input_domains=((0.0,1.0),))
        p = query_problem([unary])
        s = synthetic_solution(p,initial_params(unary))
        b = confidence_band(s,p;uf_ngrid=7)[:r]
        @test b.grid == collect(range(0.0,1.0,length=7))
        @test all(b.lower .<= b.fitted .<= b.upper)
        q = confidence_band(s,p;uf_points=Dict(:r=>[0.2,0.6]))[:r]
        @test q.points == reshape([0.2,0.6],:,1)
        @test confidence_band(s,p;uf_ngrid=1)[:r].grid == [0.5]
        @test length(confidence_band(s,p;uf_points=Dict(:r=>[0.2]))[:r].se) == 1
        @test_throws ArgumentError confidence_band(s,p;uf_ngrid=0)
        raw = KANApproximator(:r,layer(1,1))
        raw_prob = query_problem([raw])
        raw_sol = synthetic_solution(raw_prob,initial_params(raw))
        @test_throws ArgumentError confidence_band(raw_sol,raw_prob)
        @test length(confidence_band(raw_sol,raw_prob;uf_points=Dict(:r=>[-0.5,0.5]))[:r].se) == 2
    end

    @testset "Tensor and full versus outer composed queries" begin
        tensor = TensorBSplineApproximator(:g,(0.0,2.0),(0.0,2.0),4,4)
        tp = query_problem([tensor])
        beta = 0.2cos.(1:nparams(tensor))
        ts = synthetic_solution(tp,beta)
        points = [0.3 1.4; 1.7 0.2]
        band = confidence_band(ts,tp;uf_points=Dict(:g=>points))[:g]
        unit = Matrix{Float64}(I,length(beta),length(beta))
        for (i,point) in enumerate(eachrow(points))
            basis = [build_evaluator(tensor,unit[:,j])(point...) for j in eachindex(beta)]
            # Finite differences versus unit-vector evaluation measured
            # <=3.80e-13; 1e-9 leaves >2600x headroom.
            @test isapprox(band.se[i],0.2norm(basis);atol=1e-9,rtol=1e-9)
        end
        si = SingleIndexApproximator(:g,2,6;index_stats=(zeros(2),Matrix{Float64}(I,2,2)),
            initial=z->0.4+0.2z,initial_loadings=[1.0,0.4])
        times = collect(0.0:0.2:2.0)
        tc = TransformedCovariateApproximator(:r,times,sin.(times);trans=:expsm,nknots=6,
            initial=z->0.4+0.2z)
        for (a,queries) in ((si,points),(tc,reshape([0.4,1.6],:,1)))
            p = query_problem([a])
            b = initial_params(a)
            cov = zeros(length(b),length(b))
            cov[1,1] = 0.09
            s = synthetic_solution(p,b;covariance=cov)
            outer = confidence_band(s,p;uf_ngrid=5)[a.name]
            full = confidence_band(s,p;uf_points=Dict(a.name=>queries))[a.name]
            @test all(iszero,outer.se)
            @test all(>(0),full.se)
            @test full.fitted == [build_evaluator(a,b)(x...) for x in eachrow(queries)]
            if a isa SingleIndexApproximator
                expected = [0.06abs(0.2*(x[2]-b[1]*x[1])/(1+b[1]^2)^1.5)
                            for x in eachrow(queries)]
                # The affine outer curve permits an analytic loading
                # derivative. Measured SE error <=1.01e-11, >900x headroom.
                @test isapprox(full.se,expected;atol=1e-8,rtol=1e-8)
            end
            boot = bootstrap(s,p,QueryRefitter();nboot=3,uf_points=Dict(a.name=>queries),
                             rng=Random.Xoshiro(31))
            for k in 1:3
                @test boot.uf_values[a.name][:,k] ==
                    [build_evaluator(a,boot.coefs[k,:])(x...) for x in eachrow(queries)]
            end
        end
    end

    @testset "Explicit bootstrap points in both execution paths" begin
        a = KANApproximator(:g,layer(2,1);input_domains=((0.0,2.0),(0.0,2.0)))
        prob = query_problem([a])
        sol = synthetic_solution(prob,initial_params(a))
        points = [0.2 0.3; 0.8 1.2; 1.5 0.6; 1.8 1.7]
        for parallel in (false,true)
            b = bootstrap(sol,prob,QueryRefitter();nboot=5,uf_ngrid=11,
                          uf_points=Dict(:g=>points),parallel,rng=Random.Xoshiro(17))
            @test b.n_success == 5
            @test !haskey(b.uf_grid,:g)
            @test b.uf_points[:g] == points
            @test size(b.uf_values[:g]) == (4,5)
            @test b.uf_success[:g] == fill(5,4)
            for replicate in 1:5
                expected = [build_evaluator(a,b.coefs[replicate,:])(x...) for x in eachrow(points)]
                @test b.uf_values[:g][:,replicate] == expected
            end
            @test b.ci_uf[:g].lower == [quantile(vec(b.uf_values[:g][i,:]),(1-b.level)/2) for i in 1:4]
            saved = copy(points)
            b.uf_points[:g][1,1] = -99
            @test points == saved
        end
        @test_throws DimensionMismatch bootstrap(sol,prob,QueryRefitter();nboot=3,uf_points=Dict(:g=>[1.0,2.0]))
        @test_throws ArgumentError bootstrap(sol,prob,QueryRefitter();nboot=2,uf_points=Dict(:g=>points))
    end

    @testset "Point counts, unavailable evaluations and legacy results" begin
        a = KANApproximator(:g,layer(2,1);input_domains=((0.0,2.0),(0.0,2.0)))
        unary = KANApproximator(:r,layer(1,1);input_domains=((0.0,1.0),))
        mixed = query_problem([a,unary])
        sol = synthetic_solution(mixed,[initial_params(a);initial_params(unary)])
        p = query_problem([unary])
        s = synthetic_solution(p,initial_params(unary))
        for parallel in (false,true)
            b = bootstrap(sol,mixed,QueryRefitter();nboot=3,uf_ngrid=1,
                uf_points=Dict(:g=>[0.2 0.3; 1.4 1.7]),parallel,rng=Random.Xoshiro(2))
            @test b.uf_grid[:r] == [0.5]
            @test size(b.uf_values[:r]) == (1,3)
            @test size(b.uf_values[:g]) == (2,3)
            @test b.uf_success[:r] == [3]
            legacy = BootstrapResult(b.coefs,b.fitted_values,b.uf_values,b.uf_grid,
                                     b.ci_fitted,b.ci_uf,b.level,b.n_success)
            @test isempty(legacy.uf_points)
            @test legacy.uf_success == b.uf_success
            incomplete = @test_logs (:warn,r"fewer than three") bootstrap(
                s,p,IncompleteRefitter();nboot=5,uf_points=Dict(:r=>[0.2,0.5,0.8]),
                parallel,rng=Random.Xoshiro(3))
            @test incomplete.n_success == 5
            @test incomplete.uf_success[:r] == [2,0,4]
            @test all(isnan,incomplete.ci_uf[:r].lower[1:2])
            @test all(isnan,incomplete.ci_uf[:r].upper[1:2])
            @test count(isfinite,incomplete.uf_values[:r]) == 6
            finite = filter(isfinite,vec(incomplete.uf_values[:r][3,:]))
            @test incomplete.ci_uf[:r].lower[3] == quantile(finite,(1-incomplete.level)/2)
            @test incomplete.ci_uf[:r].upper[3] == quantile(finite,1-(1-incomplete.level)/2)
            failed = bootstrap(s,p,IncompleteRefitter(failed_fits=true);
                nboot=6,uf_ngrid=1,parallel,rng=Random.Xoshiro(4))
            @test failed.n_success == 3
            @test failed.uf_success[:r] == [3]
            @test size(failed.coefs,1) == 3
            @test all(isfinite,failed.coefs)
            @test all(isfinite,failed.ci_uf[:r].lower)
        end
        @test_throws ArgumentError bootstrap(s,p,QueryRefitter();uf_ngrid=0)
        for value in (NaN,Inf,-Inf)
            @test isnan(PSM._bootstrap_function_value(x->value,0.0))
        end
        @test isnan(PSM._bootstrap_function_value(x->throw(DomainError(x)),0.0))
        for error in (ArgumentError("bad query"),ErrorException("bug"),BoundsError(),
                      InterruptException())
            @test_throws typeof(error) PSM._bootstrap_function_value(x->throw(error),0.0)
        end
    end
end

end
