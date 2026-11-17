module ResilienceTests

# Fitted PSMs as SciML problems (`to_ode_problem`), parameter draws from the bootstrap and MCMC
# (`parameter_draws`), and resilience metrics over them (`metrics_over`, DynamicalResilience.jl).

using Test, LinearAlgebra
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using DynamicalResilience: DynamicalResilience, steady_state, linearise, resilience, isstable
using OrdinaryDiffEq
using MCMCChains
using SciMLBase: ODEProblem, DiscreteProblem
using StableRNGs

# logistic growth with an unknown per-capita rate r(N)
logistic!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
# Ricker map with an unknown per-capita log growth g(N): N' = N exp(g(N))
ricker!(un, u, p, t) = (un[1] = u[1] * exp(p.g(u[1])); nothing)
# delayed logistic, for the error path
delayed!(du, u, h, p, t) = (du[1] = p.r(h(p, t - 1.0)[1]) * u[1]; nothing)

# the starting coefficient vector β of a problem, as the solvers assemble it
beta0(prob) = reduce(vcat, initial_params.(prob.approximators))

const TIMES = collect(0.0:1.0:20.0)
const TRUTH = 10 ./ (1 .+ 9 .* exp.(-0.8 .* TIMES))     # logistic, r = 0.8, K = 10

logistic_problem(data) = PSMProblem(logistic!, [1.0], (0.0, 20.0),
                                    [BSplineApproximator(:r, (0.0, 12.0), 8; initial = N -> 0.8 * (1 - N / 10))];
                                    data_times = TIMES, data_values = reshape(data, :, 1), obs_to_state = [1],
                                    likelihood = Gaussian(), solver = Tsit5())

@testset "Resilience of fitted PSMs" begin
    rng = StableRNG(20261117)
    data = TRUTH .+ 0.1 .* randn(rng, length(TIMES))
    prob = logistic_problem(data)
    β = beta0(prob)

    @testset "to_ode_problem reproduces simulate" begin
        ode = to_ode_problem(prob, β)
        @test ode isa ODEProblem
        @test SciMLBase.isinplace(ode)
        @test ode.tspan == prob.tspan
        @test haskey(ode.p, :r)
        # `simulate` integrates exactly this problem with these options, so the predictions
        # are identical, not merely close
        sol = OrdinaryDiffEq.solve(ode, Tsit5(); saveat = TIMES, abstol = 1e-8, reltol = 1e-8,
                                   maxiters = 1_000_000, verbose = false)
        @test simulate(prob, β) == reshape([u[1] for u in sol.u], :, 1)
    end

    @testset "equilibrium and linearisation of the fitted model" begin
        ode = to_ode_problem(prob, β)
        ss = steady_state(ode, [8.0])
        @test ss.converged
        J = linearise(ode, ss.u)
        # d/dN [r(N) N] at N* by central differences; the Jacobian is by ForwardDiff through the
        # spline evaluator. Agreement measured at ~1e-9; 1e-6 leaves wide headroom for the
        # finite-difference step.
        r = ode.p.r
        h = 1e-5
        slope = ((r(ss.u[1] + h) * (ss.u[1] + h)) - (r(ss.u[1] - h) * (ss.u[1] - h))) / (2h)
        @test only(J.J) ≈ slope atol = 1e-6
        @test resilience(J) ≈ -slope atol = 1e-6
    end

    @testset "discrete-time PSMs give a DiscreteProblem" begin
        Ns = [2.0]
        for _ in 1:15
            push!(Ns, Ns[end] * exp(1.2 * (1 - Ns[end] / 50)))
        end
        dprob = PSMProblem(ricker!, [2.0], (0.0, 15.0),
                           [BSplineApproximator(:g, (0.0, 80.0), 8; initial = N -> 1.2 * (1 - N / 50))];
                           data_times = collect(0.0:15.0), data_values = reshape(Ns, :, 1),
                           obs_to_state = [1], discrete = true)
        βd = beta0(dprob)
        dp = to_ode_problem(dprob, βd)
        @test dp isa DiscreteProblem
        # iterating the DiscreteProblem's map is what simulate does, step for step
        u = copy(dp.u0)
        path = [u[1]]
        for t in 0.0:14.0
            un = similar(u)
            dp.f(un, u, dp.p, t)
            u = un
            push!(path, u[1])
        end
        @test simulate(dprob, βd) == reshape(path, :, 1)
        # its fixed point is where g(N) = 0
        ss = steady_state(dp, [40.0])
        @test ss.converged
        @test abs(dp.p.g(ss.u[1])) < 1e-8             # residual tolerance of steady_state is 1e-10
    end

    @testset "delay PSMs are rejected" begin
        ddeprob = PSMProblem(delayed!, [1.0], (0.0, 20.0),
                             [BSplineApproximator(:r, (0.0, 12.0), 6; initial = 0.5)];
                             data_times = TIMES, data_values = reshape(TRUTH, :, 1), obs_to_state = [1],
                             delays = [1.0], history = (p, t) -> [1.0])
        @test_throws ArgumentError to_ode_problem(ddeprob, beta0(ddeprob))
    end

    @testset "draws from a fit, the bootstrap and MCMC" begin
        fit = solve(prob, LAML(maxiters = 30, verbose = false))
        @test to_ode_problem(prob, fit).p.r(5.0) == to_ode_problem(prob, collect(fit.parameters)).p.r(5.0)
        # a LAML solution carries no posterior draws
        @test_throws ArgumentError parameter_draws(fit, prob)
        @test_throws ArgumentError DynamicalResilience.metrics_over(prob, fit; u0 = [8.0])

        bs = bootstrap(fit, prob, LAML(maxiters = 30, verbose = false); nboot = 4, rng = StableRNG(1))
        draws = parameter_draws(bs)
        @test length(draws) == bs.n_success
        @test all(length(d) == length(β) for d in draws)
        @test draws[1] == bs.coefs[1, :]
        mo = metrics_over(prob, bs; u0 = [8.0])
        @test mo.n == bs.n_success
        @test length(mo.metrics.resilience) == bs.n_success
        @test mo.n_converged + mo.n_unconverged == mo.n
        @test mo.n_stable + mo.n_unstable == mo.n_converged
        # the same as building each problem by hand
        hand = [resilience(linearise(to_ode_problem(prob, d), steady_state(to_ode_problem(prob, d), [8.0]).u))
                for d in draws]
        @test all(isfinite, hand)
        # warm starts can change the Newton path but not the converged equilibrium: agreement
        # measured at ~1e-12, allowed 1e-8
        @test mo.metrics.resilience ≈ hand atol = 1e-8

        mcmc = solve(prob, MCMCSolver(n_samples = 20, n_warmup = 20, rng_seed = 7, verbose = false))
        @test mcmc.convergence isa MCMCChains.Chains
        pd = parameter_draws(mcmc, prob)
        @test length(pd) == size(mcmc.convergence, 1) * size(mcmc.convergence, 3)
        @test all(length(d) == length(β) for d in pd)     # log_σ dropped
        @test length(parameter_draws(mcmc, prob; thin = 4)) == cld(length(pd), 4)
        @test_throws ArgumentError parameter_draws(mcmc, prob; thin = 0)
        mm = metrics_over(prob, mcmc; u0 = [8.0], thin = 4)
        @test mm.n == cld(length(pd), 4)
        @test mm.n_converged + mm.n_unconverged == mm.n
    end
end

end
