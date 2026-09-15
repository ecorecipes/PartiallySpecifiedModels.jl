module FixedSmoothingTests

using Test, LinearAlgebra
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
const PSM=PartiallySpecifiedModels

struct RidgeRate <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::RidgeRate)=1
PSM.initial_params(::RidgeRate)=[0.1]
PSM.penalty_matrix(::RidgeRate)=ones(1,1)
PSM.penalty_blocks(::RidgeRate)=[(ones(1,1),1:1)]
PSM.build_evaluator(::RidgeRate,b)=x->b[1]

@testset "Explicit fixed LAML smoothing" begin
    @test LAML().fixed_lambda === nothing
    @test LAML(100,1e-6,false,nothing,3,nothing,:working).fixed_lambda === nothing
    @test LAML(100,1e-6,false,nothing,3,nothing,:working,:forwarddiff).fixed_lambda === nothing
    for value in (0.0,-1.0,NaN,Inf)
        @test_throws ArgumentError LAML(fixed_lambda=value)
    end
    @test_throws ArgumentError LAML(fixed_lambda=1.0,initial_lambda=1.0)

    times=collect(0.0:0.2:1.0)
    y=2times + 0.01sin.(collect(1:length(times)))
    rhs!(du,u,p,t)=(du[1]=p.r(t);nothing)
    prob=PSMProblem(rhs!,[0.0],(0.0,1.0),[RidgeRate(:r)];
        data_times=times,data_values=reshape(y,:,1),abstol=1e-11,reltol=1e-11)
    lambda=0.4
    expected=dot(times,y)/(dot(times,times)+lambda)
    for jac in (:fd,:forwarddiff)
        sol=@test_logs solve(prob,LAML(maxiters=40,warmup=100,fixed_lambda=lambda,jac=jac))
        @test sol.smoothing_params == [lambda]
        @test sol.convergence.smoothing_fixed && !sol.convergence.smoothing_advanced
        @test sol.convergence.converged
        @test sol.convergence.laml_failures == 0
        # Independent ridge-regression oracle: the FD coefficient/EDF/
        # covariance discrepancy measured <=3.32e-10 (AD <=1.34e-15
        # for the coefficient). 1e-7 permits >300x headroom.
        @test isapprox(only(sol.parameters),expected;atol=1e-7,rtol=1e-7)
        @test isapprox(sol.edf,dot(times,times)/(dot(times,times)+lambda);atol=1e-7,rtol=1e-7)
        @test isapprox(only(sol.convergence.V_beta),1/(dot(times,times)+lambda);atol=1e-7,rtol=1e-7)
    end

    count_prob=PSMProblem(rhs!,[1.0],(0.0,1.0),[RidgeRate(:r)];
        data_times=times,data_values=reshape([1.,2.,2.,3.,3.,4.],:,1),likelihood=Poisson())
    count_fit=@test_logs solve(count_prob,LAML(maxiters=30,fixed_lambda=lambda,jac=:forwarddiff))
    @test count_fit.smoothing_params == [lambda]
    @test count_fit.convergence.smoothing_fixed && !count_fit.convergence.smoothing_advanced
    @test all(isfinite,count_fit.parameters)

    multi!(du,u,p,t)=(du[1]=p.a(t);du[2]=p.b(t);nothing)
    mp=PSMProblem(multi!,[0.0,0.0],(0.0,1.0),[RidgeRate(:a),RidgeRate(:b)];
        data_times=times,data_values=hcat(y,2y))
    ms=solve(mp,LAML(maxiters=30,fixed_lambda=lambda,jac=:forwarddiff))
    @test ms.smoothing_params == [lambda,lambda]
    # Same closed-form ridge oracle; the single-column discrepancy above
    # was <=3.32e-10, with >300x headroom.
    @test isapprox(collect(ms.parameters),[expected,2expected];atol=1e-7,rtol=1e-7)

    gp=GPApproximator(:r,(0.0,2.0),4;variance=1.234,initial=x->0.2)
    snapshot=(gp.lengthscale,gp.variance,copy(gp.K),copy(gp.K_inv))
    growth!(du,u,p,t)=(du[1]=p.r(u[1])*u[1];nothing)
    gp_prob=PSMProblem(growth!,[1.0],(0.0,1.0),[gp];
        data_times=times,data_values=reshape(exp.(0.2times),:,1))
    gs=solve(gp_prob,LAML(maxiters=10,fixed_lambda=0.1,jac=:forwarddiff))
    @test (gp.lengthscale,gp.variance,gp.K,gp.K_inv) == snapshot
    @test gs.smoothing_params == [0.1]

    unpen=PSMProblem((du,u,p,t)->fill!(du,0),[1.0],(0.0,1.0),
        [NeuralApproximator(:r,PSM.Lux.Dense(1,1);rng_seed=42)];
        data_times=times,data_values=ones(length(times),1))
    @test_throws ArgumentError solve(unpen,LAML(fixed_lambda=0.1))
end

end
