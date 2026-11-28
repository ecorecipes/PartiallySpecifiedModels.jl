using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve, nparams, initial_params, penalty_matrix,
    penalty_blocks, build_evaluator, band_domain
using Distributions
using StatsAPI
using GAM
using DecisionAnalysis
using DynamicalResilience
using ResilienceCore
using LinearAlgebra
using Random
using StableRNGs

@testset "M3 — v0.3.0 Clean API, Distributions, StatsAPI & ecorecipes Integration" begin

    @testset "Zero export collisions across ecorecipes + Distributions + StatsAPI" begin
        # All of these names must resolve UNAMBIGUOUSLY when all packages are loaded together!
        @test GaussianLikelihood <: AbstractLikelihood
        @test PoissonLikelihood <: AbstractLikelihood
        @test NegativeBinomialLikelihood <: AbstractLikelihood
        @test TruncatedNormalLikelihood <: AbstractLikelihood
        @test CustomLikelihood <: AbstractLikelihood

        # Distributions.jl names resolve cleanly to Distributions
        @test Normal === Distributions.Normal
        @test Poisson === Distributions.Poisson
        @test NegativeBinomial === Distributions.NegativeBinomial
        @test Truncated === Distributions.Truncated

        # StatsAPI.predict is shared across PSM and GAM without ambiguity
        @test predict === StatsAPI.predict

        # DecisionAnalysis verbs resolve cleanly to DecisionAnalysis
        @test simulate === DecisionAnalysis.simulate
        @test parameter_draws === DecisionAnalysis.parameter_draws
        @test simulate_psm === PartiallySpecifiedModels.simulate_psm
        @test posterior_draws === PartiallySpecifiedModels.posterior_draws

        # GAM verbs resolve cleanly to GAM
        @test appraise === GAM.appraise
        @test psm_appraise === PartiallySpecifiedModels.psm_appraise
        @test GAM.penalty_matrix !== PartiallySpecifiedModels.penalty_matrix
        @test GAM.nparams !== PartiallySpecifiedModels.nparams
        @test !(Symbol("penalty_matrix") in names(PartiallySpecifiedModels))
        @test !(Symbol("penalty_blocks") in names(PartiallySpecifiedModels))
        @test !(Symbol("nparams") in names(PartiallySpecifiedModels))
        @test !(Symbol("initial_params") in names(PartiallySpecifiedModels))
        @test !(Symbol("build_evaluator") in names(PartiallySpecifiedModels))
        @test !(Symbol("band_domain") in names(PartiallySpecifiedModels))
    end

    @testset "PSMProblem accepts Distributions.jl distributions directly" begin
        f!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        ts = collect(0.0:1.0:5.0)
        Y = reshape(exp.(0.2 .* ts), :, 1)
        a = BSplineApproximator(:r, (0.5, 4.0), 5; initial=0.2)

        p_norm = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=Y,
                            likelihood=Normal())
        @test p_norm.likelihood isa GaussianLikelihood

        p_norm_t = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=Y,
                              likelihood=Normal)
        @test p_norm_t.likelihood isa GaussianLikelihood

        p_pois = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=round.(10 .* Y),
                            likelihood=Poisson())
        @test p_pois.likelihood isa PoissonLikelihood

        p_pois_t = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=round.(10 .* Y),
                              likelihood=Poisson)
        @test p_pois_t.likelihood isa PoissonLikelihood

        p_nb = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=round.(10 .* Y),
                          likelihood=NegativeBinomial(7.5, 0.5))
        @test p_nb.likelihood isa NegativeBinomialLikelihood
        @test p_nb.likelihood.theta == 7.5

        p_tn = PSMProblem(f!, [1.0], (0.0, 5.0), [a]; data_times=ts, data_values=Y,
                          likelihood=truncated(Normal(0.0, 0.4), 0.0, Inf))
        @test p_tn.likelihood isa TruncatedNormalLikelihood
        @test p_tn.likelihood.lower == 0.0
        @test p_tn.likelihood.sigma == 0.4

        # Two-sided truncated normal is rejected cleanly
        @test_throws ArgumentError PSMProblem(f!, [1.0], (0.0, 5.0), [a];
            data_times=ts, data_values=Y, likelihood=truncated(Normal(0.0, 1.0), 0.0, 10.0))

        # Fit end-to-end with Distributions.Poisson()
        sol_p = solve(p_pois, LAML(maxiters=30))
        @test sol_p.convergence.converged
    end

    @testset "StatsAPI verbs on PSMSolution & PSMProblem" begin
        logistic!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        ts = collect(0.0:1.0:12.0)
        rng = StableRNG(77)
        truth = 10.0 ./ (1.0 .+ 9.0 .* exp.(-0.6 .* ts))
        Y = reshape(truth .+ 0.05 .* randn(rng, length(ts)), :, 1)
        prob = PSMProblem(logistic!, [1.0], (0.0, 12.0),
                          [BSplineApproximator(:r, (0.0, 12.0), 6; initial=0.3)];
                          data_times=ts, data_values=Y, likelihood=Normal())
        sol = solve(prob, LAML(maxiters=40))

        @test StatsAPI.predict(sol) == sol.fitted_values
        @test StatsAPI.predict(sol, prob) == sol.fitted_values
        @test StatsAPI.fitted(sol) == sol.fitted_values
        @test StatsAPI.residuals(sol) == prob.data_values .- sol.fitted_values
        @test StatsAPI.coef(sol) == Vector{Float64}(sol.parameters)
        @test StatsAPI.dof(sol) == sol.edf
        @test StatsAPI.nobs(sol) == length(ts)
        @test StatsAPI.nobs(prob) == length(ts)
        @test !StatsAPI.islinear(sol)
        V = StatsAPI.vcov(sol)
        @test size(V) == (6, 6)
        @test isapprox(V, sol.convergence.sigma2 .* sol.convergence.V_beta; rtol=1e-12)
        @test isapprox(StatsAPI.stderror(sol), sqrt.(max.(diag(V), 0.0)); rtol=1e-12)
        ll = StatsAPI.loglikelihood(sol, prob)
        @test isfinite(ll)
        @test StatsAPI.aic(sol, prob) ≈ -2ll + 2 * (sol.edf + 1)
        @test isfinite(StatsAPI.aicc(sol, prob))
        @test StatsAPI.bic(sol, prob) ≈ -2ll + log(length(ts)) * (sol.edf + 1)

        # psm_appraise works cleanly
        ap = psm_appraise(sol)
        @test haskey(ap, :durbin_watson)
    end

    @testset "Deepened DynamicalResilience / ResilienceCore bridge" begin
        logistic!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        ts = collect(0.0:1.0:15.0)
        rng = StableRNG(1234)
        truth = 10.0 ./ (1.0 .+ 9.0 .* exp.(-0.7 .* ts))
        Y = reshape(truth .+ 0.08 .* randn(rng, length(ts)), :, 1)
        prob = PSMProblem(logistic!, [1.0], (0.0, 15.0),
                          [BSplineApproximator(:r, (0.0, 12.0), 7; initial=N -> 0.7*(1-N/10))];
                          data_times=ts, data_values=Y)
        sol = solve(prob, LAML(maxiters=40))

        # Direct steady_state and linearise on (prob, sol, ...)
        ss = DynamicalResilience.steady_state(prob, sol, [9.0])
        @test ss.converged
        @test isapprox(ss.u[1], 10.0; atol=0.5)
        J = ResilienceCore.linearise(prob, sol, ss.u)
        @test ResilienceCore.isstable(J)
        @test ResilienceCore.resilience(J) > 0

        # Broadened posterior_draws on LAML (conditional), VI, EKI
        draws_c = posterior_draws(sol, prob; n=25, rng=StableRNG(5))
        @test length(draws_c) == 25 && all(length(d) == 7 for d in draws_c)
        # Unconditional smoothing-corrected draws on a genuinely curved (non-null-space) function
        curved!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        ts_c = collect(0.0:0.5:10.0)
        ode_c = ODEProblem((du, u, p, t) -> (du[1] = (0.8 / (1 + 0.3 * u[1]) - 0.2) * u[1]; nothing),
                           [0.5], (0.0, 10.0))
        using OrdinaryDiffEq: solve as ode_solve
        sol_c_true = ode_solve(ode_c, Tsit5(); saveat=ts_c, abstol=1e-9, reltol=1e-9)
        Y_c = reshape([u[1] for u in sol_c_true.u] .+ 0.03 .* randn(StableRNG(11), length(ts_c)), :, 1)
        prob_c = PSMProblem(curved!, [0.5], (0.0, 10.0),
                            [BSplineApproximator(:r, (0.0, 12.0), 7; initial=0.2)];
                            data_times=ts_c, data_values=Y_c)
        sol_c = solve(prob_c, LAML(maxiters=50))
        draws_u = posterior_draws(sol_c, prob_c; n=25, unconditional=true, rng=StableRNG(5))
        @test length(draws_u) == 25 && all(length(d) == 7 for d in draws_u)
        mo = metrics_over(prob, sol; u0=[9.0], n=15, rng=StableRNG(5))
        @test mo.n == 15

        vi = solve(prob, VariationalSolver(maxiters=20, n_elbo_samples=2, rng_seed=3))
        @test length(posterior_draws(vi, prob; n=10, rng=StableRNG(6))) == 10
        @test size(StatsAPI.vcov(vi)) == (7, 7)

        # to_discrete_problem
        ricker!(un, u, p, t) = (un[1] = u[1] * exp(p.g(u[1])); nothing)
        Ns = [2.0]; for _ in 1:12; push!(Ns, Ns[end] * exp(1.0 * (1 - Ns[end] / 20))); end
        dprob = PSMProblem(ricker!, [2.0], (0.0, 12.0),
                           [BSplineApproximator(:g, (0.0, 30.0), 6; initial=N -> 1.0*(1-N/20))];
                           data_times=collect(0.0:12.0), data_values=reshape(Ns, :, 1),
                           discrete=true)
        dp = to_discrete_problem(dprob, initial_params(dprob.approximators[1]))
        @test dp isa DiscreteProblem
        @test_throws ArgumentError to_discrete_problem(prob, sol)
    end

    @testset "GAMSmoothApproximator extension (GAM.jl: :cc, :tp, t2)" begin
        # 1. Cyclic cubic spline (:cc) for seasonal forcing in an ODE
        a_cc = GAMSmoothApproximator(:beta, GAM.s(:t; bs=:cc, k=8), (0.0, 12.0); initial=0.4)
        @test nparams(a_cc) >= 6
        @test band_domain(a_cc) == (0.0, 12.0)
        p0 = initial_params(a_cc)
        ev_cc = build_evaluator(a_cc, p0)
        # Periodic wrapping: f(0.5) == f(12.5)
        @test isapprox(ev_cc(0.5), ev_cc(12.5); atol=1e-12)
        @test isapprox(ev_cc(0.0), ev_cc(12.0); atol=1e-12)

        # Fit seasonal growth ODE du/dt = beta(t) * u with GAMSmoothApproximator(:cc)
        seasonal!(du, u, p, t) = (du[1] = p.beta(t) * u[1]; nothing)
        ts = collect(0.0:1.0:24.0)
        # Exact solution for beta(t) = 0.05 + 0.08 * sin(2π t / 12)
        u_true = [exp(0.05 * t + 0.08 * (12 / (2π)) * (1 - cos(2π * t / 12))) for t in ts]
        prob_cc = PSMProblem(seasonal!, [1.0], (0.0, 24.0), [a_cc];
                             data_times=ts, data_values=reshape(u_true, :, 1))
        sol_cc = solve(prob_cc, LAML(maxiters=40))
        @test sol_cc.convergence.converged
        @test sol_cc.data_loss < 0.05
        f_fit = sol_cc.unknown_functions[:beta]
        @test isapprox(f_fit(2.0), f_fit(14.0); atol=1e-12)

        # 2. Thin-plate regression spline (:tp) + linearise (Dual x safety)
        a_tp = GAMSmoothApproximator(:r, GAM.s(:x; bs=:tp, k=7), (0.0, 10.0); initial=0.2)
        ev_tp = build_evaluator(a_tp, initial_params(a_tp))
        @test isfinite(ev_tp(5.0))
        # Dual in x works (needed for stiff ODEs & DynamicalResilience.linearise)
        using ForwardDiff
        @test isfinite(ForwardDiff.derivative(ev_tp, 5.0))

        # 3. 2D t2 tensor product smooth
        a_t2 = GAMSmoothApproximator(:g, GAM.t2(:x, :y; k=[4, 4]), ((0.0, 5.0), (0.0, 5.0));
                                     initial=0.1)
        @test band_domain(a_t2) === nothing
        @test !isempty(penalty_blocks(a_t2))
        ev_t2 = build_evaluator(a_t2, initial_params(a_t2))
        @test isapprox(ev_t2(2.5, 2.5), 0.1; atol=1e-6)
    end

    @testset "DecisionAnalysis.jl extension (ManagedPSM, sample_belief, EVPI/EVPPI)" begin
        # Harvested logistic PSM: du/dt = (r(N) - h) * N
        function harvest_ode!(du, u, p, t)
            du[1] = (p.r(u[1]) - p.h) * u[1]
            nothing
        end
        ts = collect(0.0:1.0:12.0)
        rng = StableRNG(2026)
        truth = 10.0 ./ (1.0 .+ 9.0 .* exp.(-0.6 .* ts))
        Y = reshape(truth .+ 0.08 .* randn(rng, length(ts)), :, 1)
        prob = PSMProblem(harvest_ode!, [1.0], (0.0, 12.0),
                          [BSplineApproximator(:r, (0.0, 12.0), 6; initial=N -> 0.6*(1-N/10))];
                          data_times=ts, data_values=Y, known_params=(h=0.0,))
        sol = solve(prob, LAML(maxiters=40))

        belief = sample_belief(sol, prob; n=30, rng=StableRNG(99),
                               summary_fn=(β, p) -> begin
                                   rf = build_evaluator(p.approximators[1], β)
                                   (; r_low=rf(1.0), r_mid=rf(5.0))
                               end)
        @test belief isa SampleBelief
        @test length(belief) == 30
        @test haskey(parameter_draws(belief), :r_low)
        @test haskey(parameter_draws(belief), :r_mid)

        # ManagedPSM adapter
        apply_h(p, h) = PSMProblem(p.dynamics!, p.u0, p.tspan, p.approximators;
                                   data_times=p.data_times, data_values=p.data_values,
                                   known_params=(h=h,))
        yield_out(traj, p, h) = (; total_yield=h * sum(traj), final_stock=traj[end, 1])
        managed = ManagedPSM(prob, apply_h, yield_out)

        actions = [:none => 0.0, :moderate => 0.15, :heavy => 0.45]
        ot = outcomes(managed, actions, belief; seed=42)
        @test ot isa OutcomeTable
        @test run_counts(ot).failed == 0

        pt = payoff_table(ot, FunctionUtility(o -> o.total_yield + 0.5 * o.final_stock))
        dprob_da = DecisionProblem(pt, belief)
        res = decide(dprob_da, MaxExpectedUtility())
        @test first_choice(res) in (:none, :moderate, :heavy)

        v_evpi = evpi(dprob_da, SampleAverage())
        @test isfinite(v_evpi.value) && v_evpi.value >= -1e-10
        v_evppi = evppi(dprob_da, [:r_mid], StrongOakley())
        @test isfinite(v_evppi.value)
    end
end
