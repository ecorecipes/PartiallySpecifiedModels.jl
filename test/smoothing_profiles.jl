module SmoothingProfileTests

using Test
include("../benchmarks/kan/assess_smoothing_profiles.jl")

struct ProfilePolynomial <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::ProfilePolynomial) = 4
PSM.initial_params(::ProfilePolynomial) = zeros(4)
PSM.penalty_matrix(::ProfilePolynomial) = Matrix(Diagonal([0.0,0.0,2.0,7.0]))
PSM.penalty_blocks(a::ProfilePolynomial) = [(penalty_matrix(a),1:4)]
PSM.build_evaluator(::ProfilePolynomial,b) = x -> b[1]+b[2]*x+b[3]*x^2+b[4]*x^3

@testset "Stable smoothing profile diagnostics" begin
    @testset "Fixed-rank frame and determinant-one shear" begin
        a = ProfilePolynomial(:r)
        S = penalty_matrix(a)
        x = collect(range(-1.0,1.0,length=21))
        X = hcat(ones(length(x)),x,x.^2,x.^3)
        original = copy(S)
        frame = sp_center_frame(sp_penalty_frame(S),X)
        @test frame.rank == 2 && frame.nullity == 2
        @test S == original
        @test frame.eigenvalues == [2.0,7.0]
        beta = [0.5,0.2,-0.3,0.1]
        for rho in (-18.0,0.0,20.0,40.0)
            coordinates = sp_coordinates(a,frame,rho,beta)
            # BigFloat/QR probes measured <=7.11e-15; 1e-10 leaves
            # >14000x roundoff headroom for these matrix identities.
            @test isapprox(coordinates.transform*coordinates.initial,beta;atol=1e-10,rtol=1e-10)
            @test isapprox(sum(abs2,coordinates.initial[3:end]),exp(rho)*dot(beta,S*beta);
                           atol=1e-10,rtol=1e-10)
            unshifted = hcat(frame.U0,frame.Up ./ reshape(sqrt.(frame.eigenvalues),1,:))
            shifted = hcat(frame.U0,frame.penalized)
            @test isapprox(abs(det(shifted)/det(unshifted)),1.0;atol=1e-10,rtol=1e-10)
        end
        limit = sp_coordinates(a,frame,Inf,beta)
        @test nparams(limit) == 2 && penalty_matrix(limit) === nothing
        @test isempty(penalty_blocks(limit))
        @test norm((X*frame.U0)'*(X*frame.penalized)) < 1e-10
        @test_throws DimensionMismatch sp_penalty_frame(ones(3,2))
        @test_throws DomainError sp_penalty_frame([1.0 2.0;0.0 1.0])
        @test_throws DomainError sp_penalty_frame(Matrix(Diagonal([0.0,-1.0,2.0])))
        @test_throws ArgumentError sp_penalty_frame(Matrix{Float64}(I,3,3))
        @test_throws ArgumentError sp_penalty_frame(zeros(3,3))
        @test_throws DomainError sp_center_frame(sp_penalty_frame(S),zeros(10,4))
        @test_throws ArgumentError sp_coordinates(a,frame,-Inf,beta)
        @test_throws DomainError sp_coordinates(a,frame,2000.0,beta)
        @test_throws DimensionMismatch sp_coordinates(a,frame,0.0,[1.0])
    end

    @testset "Independent BigFloat objective, scale and affine limit" begin
        x = collect(range(-1.0,1.0,length=21))
        X = hcat(ones(length(x)),x,x.^2,x.^3)
        y = 0.5 .+ 0.3x .+ 0.2x.^2 .+ 0.1sin.(collect(1:length(x)))
        S = penalty_matrix(ProfilePolynomial(:r))
        frame = sp_center_frame(sp_penalty_frame(S),X)
        fits = Dict{Float64,Any}()
        for rho in (-18.0,0.0,20.0,40.0,Inf)
            a = sp_coordinates(ProfilePolynomial(:r),frame,rho,zeros(4))
            T = a.transform
            C = rho == Inf ? zeros(0,2) : hcat(zeros(2,2),Matrix{Float64}(I,2,2))
            theta = vcat(X*T,C) \ vcat(y,zeros(size(C,1)))
            f = sp_working_profile(X*T,ones(length(y)),theta,X*T*theta-y,2;sigma2=0.04)
            fits[rho] = f
            if isfinite(rho)
                xb,yb,sb,lambda = BigFloat.(X),BigFloat.(y),BigFloat.(S),exp(BigFloat(rho))
                H = xb'*xb + lambda*sb
                beta = H \ (xb'*yb)
                Q = sum(abs2,xb*beta-yb) + lambda*dot(beta,sb*beta)
                determinant = (2BigFloat(rho)+log(BigFloat(14))-logdet(H))/2
                expected = -(length(y)-2)/2*log(Q/(length(y)-2)) + determinant
                known = -Q/(2BigFloat(0.04)) - (length(y)-2)/2*log(BigFloat(0.04)) + determinant
                edf = tr(H \ (xb'*xb))
                # Independently factored BigFloat native-coordinate oracles
                # measured <=7.11e-15; 1e-10 gives >14000x headroom.
                @test isapprox(f.criterion,Float64(expected);atol=1e-10,rtol=1e-10)
                @test isapprox(f.known_criterion,Float64(known);atol=1e-10,rtol=1e-10)
                @test isapprox(f.edf,Float64(edf);atol=1e-10,rtol=1e-10)
                @test isapprox(X*T*theta,Float64.(xb*beta);atol=1e-10,rtol=1e-10)
            end
            @test f.n_eff == length(y)-2
            @test f.decrement < 1e-10
        end
        @test isapprox(fits[40.0].criterion,fits[Inf].criterion;atol=1e-10,rtol=1e-10)
        @test isapprox(fits[40.0].edf,2.0;atol=1e-10,rtol=1e-10)
        @test isnan(fits[Inf].local_hessian)
        function objective(rho)
            B = exp(rho)*S
            H = X'*X+B
            beta = H \ (X'*y)
            Q = sum(abs2,X*beta-y)+dot(beta,B*beta)
            -(length(y)-2)/2*log(Q/(length(y)-2)) +
                (2rho+log(14.0)-logdet(H))/2
        end
        g = PSM.ForwardDiff.derivative(objective,0.0)
        h = PSM.ForwardDiff.derivative(r -> PSM.ForwardDiff.derivative(objective,r),0.0)
        @test isapprox(fits[0.0].local_gradient,g;atol=1e-10,rtol=1e-10)
        @test isapprox(fits[0.0].local_hessian,-h;atol=1e-10,rtol=1e-10)
        @test_throws DomainError sp_working_profile([0.0 1.0;0.0 1.0],ones(2),zeros(2),ones(2),1)
        @test_throws DomainError sp_working_profile(ones(2,2),[1.0,-1.0],zeros(2),ones(2),1)
        @test_throws DomainError sp_working_profile(ones(2,2),[1.0,0.0],zeros(2),ones(2),1)
        @test_throws ArgumentError sp_working_profile(ones(2,2),ones(2),zeros(2),ones(2),1;sigma2=0.0)
    end

    @testset "KAN shear prevents data-informed PCLS truncation" begin
        a = cf_model("kan12_free").report
        x = collect(range(UC_DOMAIN...;length=41))
        X = sp_linear_design(a,x)
        frame = sp_center_frame(sp_penalty_frame(penalty_matrix(a)),X)
        coordinates = sp_coordinates(a,frame,-12.0,initial_params(a))
        J = X*coordinates.transform
        y = 0.8 .+ 0.3sin.(collect(range(0.0,4.0,length=length(x))))
        B = penalty_matrix(coordinates)
        pcls = PSM._pcls_augmented_solve(J,y,B,ones(length(y)))
        C = hcat(zeros(frame.rank,frame.nullity),Matrix{Float64}(I,frame.rank,frame.rank))
        oracle = vcat(J,C) \ vcat(y,zeros(frame.rank))
        # Independent QR agreement measured 2.49e-15 in predictions and
        # 5.99e-16 in penalized coefficients; 1e-9 leaves >400000x margin.
        # Plain whitening instead gives 1.41e-5 / 8.19e-5 discrepancies.
        @test norm(J*(pcls-oracle))/norm(y) < 1e-9
        @test norm(C*(pcls-oracle)) < 1e-9
        @test norm((X*frame.U0)'*(X*frame.penalized)) < 1e-9
    end

    @testset "Multistart refitting, failed starts, masks and ownership" begin
        a = ProfilePolynomial(:r)
        times = collect(0.0:0.1:1.0)
        y = 0.5times + 0.1times.^2 + 0.01sin.(collect(1:length(times)))
        y[3] = NaN
        weights = ones(length(y),1)
        weights[5,1] = 0.0
        rhs! = function(du,u,p,t)
            p.r(0.0) <= 5 || throw(DomainError(p.r(0.0),"intentional invalid starting region"))
            du[1] = p.r(t)
            nothing
        end
        prob = PSMProblem(rhs!,[0.0],(0.0,1.0),[a]; data_times=times,
            data_values=reshape(y,:,1),data_weights=weights,abstol=1e-11,reltol=1e-11)
        reference = hcat(ones(length(times)),times,times.^2,times.^3)
        before = (copy(prob.data_values),copy(prob.data_weights),initial_params(a))
        profile = sp_profile(prob,[-2.0,2.0]; beta_selected=[10.0,0.0,0.0,0.0],
            reference_design=reference,iterations=40)
        @test length(profile.candidates) == 8
        @test count(r->r.status=="failed",profile.candidates) == 3
        @test Set(keys(profile.best)) == Set([-2.0,2.0,Inf])
        for (rho,fit) in profile.best
            @test fit.Q == minimum(r.Q for r in profile.candidates if r.rho==rho && r.status=="ok")
            score = sp_score(fit,profile.frame;sigma2=0.01)
            @test score.n_used == length(y)-2 && score.n_eff == length(y)-4
            @test isfinite(score.criterion) && all(isfinite,score.step)
            uncertainty = sp_function_uncertainty(score,reference;sigma2=0.01)
            @test all(isfinite,uncertainty.se) && all(isfinite,uncertainty.known_se)
            @test uncertainty.fitted == reference*fit.beta
        end
        @test isequal(before,(prob.data_values,prob.data_weights,initial_params(a)))
        @test_throws ArgumentError sp_profile(prob,[-Inf];beta_selected=zeros(4),reference_design=reference)
        @test_throws ArgumentError sp_profile(prob,[0.0];beta_selected=fill(NaN,4),reference_design=reference)
        overflow = PSMProblem((du,u,p,t)->fill!(du,0.0),[1e200],(0.0,1.0),[a];
            data_times=[0.0,0.5,1.0],data_values=ones(3,1))
        @test_throws DomainError sp_refit(overflow,profile.frame,0.0,zeros(4);iterations=4)
        try
            sin("bad")
        catch e
            @test_throws MethodError sp_candidate_error(e)
        end
    end

    @testset "Representation oracle is separate from fitting" begin
        a = BSplineApproximator(:r,(0.0,2.4),8;initial=x->0.5)
        initial = initial_params(a)
        oracle = sp_representation_oracle(a,x->0.7-0.2x,(0.3,1.8);ngrid=101,neval=201)
        # Affine reproduction is a linear-algebra identity. The matrix
        # probes above measured <=7.11e-15; 1e-10 leaves >14000x headroom.
        @test oracle.max_error < 1e-10
        @test oracle.rmse < 1e-10
        @test initial_params(a) == initial
        @test_throws DomainError sp_representation_oracle(a,x->NaN,(0.3,1.8))
        @test_throws ArgumentError sp_representation_oracle(a,identity,(1.8,0.3))
        @test_throws ArgumentError sp_linear_design(a,zeros(2,2))
    end

    @testset "Profile archive schema and complete population" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                output = joinpath(dir,"profiles")
                args = ["--data=generated","--cases=reference","--models=spline8",
                    "--seeds=7","--rhos=-10,-8,0","--iterations=20","--output=$output"]
                sp_profile_main(args)
                design = TOML.parsefile(joinpath(output,"design.toml"))
                metadata = TOML.parsefile(joinpath(output,"metadata.toml"))
                profiles = rd_read_csv(joinpath(output,"profiles.csv"))
                points = rd_read_csv(joinpath(output,"dataset_points.csv"))
                @test design["data_source"] == "generated"
                @test metadata["native_fits"] == 1 && metadata["new_bootstrap_refits"] == 0
                @test length(points) == 76length(SP_METHODS)
                @test Set(r.method for r in points) == Set(SP_METHODS)
                @test length(rd_read_csv(joinpath(output,"curvature.csv"))) == 2
                @test length(rd_read_csv(joinpath(output,"selections.csv"))) == 5
                @test metadata["coefficient_refit_attempts"] == 3(length(profiles)-1)+2
                @test count(r->r.rho=="Inf",profiles) == 1
                @test isfile(joinpath(output,"datasets","reference__7.toml"))
                @test_throws ErrorException sp_profile_main(args)
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
