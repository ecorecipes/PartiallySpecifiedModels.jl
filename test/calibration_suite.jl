module CalibrationSuiteTests

using Test
include("../benchmarks/calibration/run.jl")

@testset "Cross-approximator calibration experiments" begin
    @testset "Families, designs and reserved truths" begin
        for id in CE_MODELS
            model = ce_model(id)
            @test model.frame.rank+model.frame.nullity == nparams(model.approx)
            @test all(isfinite,model.initial)
            @test model.family in ("spline","gp","spde","mlp","kan")
        end
        @test ce_model("spde12").frame.nullity == 0
        @test ce_model("mlp13").frame.nullity == 0
        @test ce_model("spline8").frame.nullity == 2
        @test ce_model("gp12_short").metadata != ce_model("gp12_long").metadata ||
            ce_model("gp12_short").approx.lengthscale != ce_model("gp12_long").approx.lengthscale
        for operator in ("growth","integral"), design in ("baseline","diverse","replicated")
            d = ce_design(operator,design)
            @test length(d.u0)*length(d.times) == 40
            for noise in ("iid","hetero","ar1")
                n = ce_noise(d,noise)
                # Existing QR/Cholesky controls measured <=7.11e-15;
                # 1e-10 leaves >14000x algebraic headroom.
                @test isapprox(n.root*n.root',n.shape;atol=1e-10,rtol=1e-10)
            end
        end
        for operator in ("growth","integral")
            @test all(in(ce_cases("confirm",operator)),ce_cases("develop",operator))
            @test !("narrow" in ce_cases("develop",operator))
            @test "narrow" in ce_cases("confirm",operator)
        end
        @test_throws ArgumentError ce_options(["--stage=confirm"])
        @test_throws ArgumentError ce_options(["--plan-only=maybe"])
        @test_throws ArgumentError ce_options(["--models=unknown"])
        @test_throws ArgumentError ce_options(["--methods=split_bias_ellipsoid","--operators=growth"])
        @test_throws ArgumentError ce_options(["--shard=2/1"])
        @test !ce_applicable("split_bias_bank","growth","estimated")
        @test !ce_applicable("split_bias_ellipsoid","growth","known")
        @test ce_applicable("split_bias_ellipsoid","integral","known")
    end

    @testset "Reference integration and target functionals" begin
        b = ce_poly_coefficients("poly_holdout")
        @test norm(CE_REFERENCE_WEIGHTS.*(b-CE_REFERENCE_CENTER)) < CE_REFERENCE_RADIUS
        for t in (0.2,0.7,1.4), speed in (0.55,1.0,1.25)
            derivative = PSM.ForwardDiff.derivative(s->dot(ce_integrated_legendre(s,speed),b),t)
            # Polynomial antiderivative identities share the <=7.11e-15
            # matrix-control scale; 1e-10 allows >14000x headroom.
            @test isapprox(derivative,ce_truth("poly_holdout")(speed*t);atol=1e-10,rtol=1e-10)
        end
        points = [0.3,0.6,0.9,1.2,1.8]
        map = ce_output_map(points)
        @test dot(map[end,:],points) == -0.6
        @test isapprox(dot(map[end-1,:],points),(first(points)+last(points))/2;atol=1e-10,rtol=1e-10)
        @test all(map[end-1,:].>=0)
    end

    @testset "Stage independence and weak-identification pairs" begin
        points = collect(range(0.3,1.8;length=11))
        design = ce_design("growth","baseline")
        cache = ce_bank_cache(design,points)
        first = ce_dataset("develop","growth","baseline","iid",0.03,"weak_a",1,points,cache)
        same = ce_dataset("develop","growth","baseline","iid",0.03,"weak_a",1,points,cache)
        other = ce_dataset("confirm","growth","baseline","iid",0.03,"weak_a",1,points,cache)
        @test first.observed == same.observed
        @test first.observed != other.observed
        @test first.reference == other.reference
        @test first.actual == other.actual
        @test first.truth_in_bank
        pair = ce_weak_pair(cache,ce_noise(design,"iid"),0.03)
        @test pair.chosen.separation >= 0.02 # Prespecified minimum target separation.
        @test pair.chosen.kl == pair.chosen.distance^2/2
        @test length(pair.rows) == div(length(cache.ids)*(length(cache.ids)-1),2)
        @test ce_seed("data","develop",1) != ce_seed("data","confirm",1)
    end

    @testset "Studentization uses each replicate scale" begin
        draws = [1.0 2.0 4.0 5.0 8.0;0.0 1.0 2.0 3.0 4.0]
        scales = [1.0 2.0 1.0 0.5 2.0;2.0 1.0 1.0 2.0 1.0]
        center,se,generating = [3.0,2.0],[0.5,0.2],[4.0,1.0]
        band = ce_studentized(center,se,generating,draws,scales,2,0.8)
        t = (draws.-generating)./scales
        alpha = (1-0.8)/2
        @test band.pointwise.lower == [center[i]-quantile(t[i,:],1-alpha)*se[i] for i in 1:2]
        @test band.pointwise.upper == [center[i]-quantile(t[i,:],alpha)*se[i] for i in 1:2]
        @test band.simultaneous.critical == quantile(vec(maximum(abs.(t);dims=1)),0.8)
        damaged = copy(scales)
        damaged[1,1] = 0
        @test_throws DomainError ce_studentized(center,se,generating,draws,damaged,2,0.8)
        @test_throws DomainError ce_studentized(center,se,generating,fill(NaN,2,5),scales,2,0.8)
        @test_throws ArgumentError ce_studentized(center,se,generating,draws[:,1:2],scales[:,1:2],2,0.8)
        @test occursin("Unstable",ce_error(ErrorException("ODE solve failed: Unstable")))
        try
            error("program failure")
        catch e
            @test_throws ErrorException ce_error(e)
        end
    end

    @testset "Finite-class and continuous bias bounds" begin
        M = [1.0 0.2;0.5 -0.1]
        means = [0.0 1.0 -1.0;1.0 0.0 2.0]
        targets = [0.2 1.1 -0.2;0.1 0.5 -0.4]
        offset = [0.1,-0.1]
        B = ce_bias_bound(offset,M,means,targets)
        errors = offset .+ M*means-targets
        @test B == vec(maximum(abs.(errors);dims=2))
        se = [0.3,0.4]
        intervals = ce_bias_intervals(zeros(2),se,B,2,0.95)
        for i in 1:2, j in 1:3
            radius = intervals.pointwise.upper[i]
            normal = Distributions.Normal(errors[i,j],se[i])
            coverage = Distributions.cdf(normal,radius)-Distributions.cdf(normal,-radius)
            # Normal CDF/quantile rounding is at the few-ulp scale;
            # 1e-12 gives >1000x machine-epsilon headroom.
            @test coverage >= 0.95-1e-12
        end
        A = [1.0 0.4;-0.1 0.7]
        G = [0.4 1.0;1.0 0.0]
        center,weights,radius = [0.5,-0.2],[1.0,2.0],0.8
        bound = ce_ellipsoid_bias(offset,M,A,G,center,weights,radius)
        D = M*A-G
        for i in 1:2
            direction = D[i,:]./weights
            direction ./= norm(direction)
            extreme = [abs(offset[i]+dot(D[i,:],center+sign*radius.*direction./weights)) for sign in (-1,1)]
            @test isapprox(bound[i],maximum(extreme);atol=1e-10,rtol=1e-10)
        end
    end

    @testset "Correlated innovations and no inference-data leakage" begin
        points = collect(range(0.3,1.8;length=11))
        ds = ce_dataset("smoke","integral","baseline","ar1",0.015,"poly_a",1,points)
        model = ce_model("spline8")
        opts = (;iterations=12,tol=1e-7,rhos=[-8.0,-4.0],noise_mode="known",nboot=3,pilot_fraction=0.25)
        split = ce_split_inference(model,ds,opts)
        @test split.value !== nothing
        s = split.value
        innovation = ce_innovation(ds.noise.shape,s.train,s.infer)
        @test norm(ds.noise.shape[s.infer,s.train]-innovation.regression*ds.noise.shape[s.train,s.train]) < 1e-10
        changed = copy(ds.observed)
        changed[s.infer] .+= 100.0
        again = ce_split_inference(model,merge(ds,(observed=changed,)),opts)
        @test split.pilot.fit.beta == again.pilot.fit.beta
        @test s.M == again.value.M
        @test s.estimates != again.value.estimates
        @test ce_linear_set(ds,0.95).intervals !== nothing
        bad = copy(ds.observed)
        bad[1] = 1e6
        rejected = merge(ds,(observed=bad,))
        @test ce_linear_set(rejected,0.95).intervals === nothing
        @test ce_bank_set(rejected,0.95).intervals === nothing
    end

    @testset "Confirmation policy cannot change after freezing" begin
        cfg = ce_options(["--stage=develop","--plan-only=true"])
        mktempdir() do directory
            path = joinpath(directory,"lock.toml")
            rd_write_toml(path,Dict("schema"=>"calibration-lock-v1","source_stage"=>"develop",
                "code_sha256"=>ce_code_hash(),"confirmation_datasets"=>500,
                "confirmation_bootstrap"=>999,"confirmation_nsim"=>10000,"options"=>cfg.canonical,
                "participants"=>[Dict("operator"=>"growth","model"=>"spline8",
                    "method"=>"conditional","noise_mode"=>"known")]))
            fixed = ce_options(["--stage=confirm","--lock=$path","--n-datasets=2"])
            @test fixed.nboot==999 && fixed.nsim==10000
            @test fixed.seeds == [1,2]
            for option in ("--level=0.9","--rhos=0,1","--nboot=3","--nsim=2","--seed-start=501")
                @test_throws ArgumentError ce_options(["--stage=confirm","--lock=$path",option])
            end
        end
    end
end

end
