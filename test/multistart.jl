module MultiStartTests

using Test
using Lux
using FluxKAN
using OrdinaryDiffEq
using PartiallySpecifiedModels
using PartiallySpecifiedModels: nparams, initial_params
using PartiallySpecifiedModels: solve

const PSM = PartiallySpecifiedModels

# The fixture of "LAML stationarity diagnostics (F2)": an unpenalized
# growth-rate network whose LAML fit depends on its initialization.
# rng_seed = 42 alone ends at a local minimum (data loss 5.08) reported as
# converged; most other seeds reach the noise floor (≈ 7.7e-4).
growth!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
const t_ms = collect(0.0:0.5:10.0)
const y_ms = exp.(0.1 .* t_ms) .+ [0.01 * (sin(3.1i) + 0.5cos(7.7i)) for i in 1:21]

growth_net() = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))
# The second term, of time, gives mixed problems a deterministic partner.
mixed!(du, u, p, t) = (du[1] = (p.r(u[1]) + p.s(t)) * u[1])
# The integrator cap bounds the runtime when a restart's trial steps send the
# network where the ODE is expensive (measured: one seed spent > 20 min on 3
# LAML iterations uncapped, 10.7 s capped). Converged trajectories need ~50
# steps. The cap does change the path of some restarts, because a capped
# trial solve fails, but the escape test was measured with it in place.
function growth_problem(approximators; dynamics=growth!, cap=(maxiters=10_000,))
    PSMProblem(dynamics, [1.0], (0.0, 10.0), approximators;
               data_times=t_ms, data_values=reshape(y_ms, :, 1),
               obs_to_state=[1], likelihood=GaussianLikelihood(), solver=Tsit5(),
               cap...)
end
neural_problem(seed) = growth_problem(
    [NeuralApproximator(:r, growth_net(); domain=(0.0, 5.0), rng_seed=seed)])

strip_record(c) = Base.structdiff(c, NamedTuple{(:multistart,)})

@testset "MultiStart" begin
    alg = LAML(maxiters=100)

    @testset "constructor" begin
        ms = MultiStart(alg)
        @test ms.n_starts == 8 && ms.rng_seed == 1 && ms.alg === alg
        @test ms.ode_maxiters == 10_000
        @test MultiStart(alg; n_starts=3, rng_seed=UInt8(7)).rng_seed === 7
        @test MultiStart(alg; ode_maxiters=nothing).ode_maxiters === nothing
        @test MultiStart(alg; ode_maxiters=Int32(500)).ode_maxiters === 500
        @test_throws ArgumentError MultiStart(alg; n_starts=0)
        @test_throws ArgumentError MultiStart(alg; ode_maxiters=0)
        @test_throws MethodError MultiStart(GCVSolver())
    end

    @testset "integrator cap" begin
        short_cap = LAML(maxiters=10)
        net42() = [NeuralApproximator(:r, growth_net(); domain=(0.0, 5.0),
                                      rng_seed=42)]
        # A problem that sets its own maxiters keeps it.
        prob = growth_problem(net42())
        @test prob.ode_kwargs[:maxiters] == 10_000
        r = solve(prob, MultiStart(short_cap; n_starts=1, ode_maxiters=50))
        @test r.convergence.multistart.ode_maxiters == 10_000
        @test r.parameters == solve(prob, short_cap).parameters
        # Without one, the default cap applies: start 1 is then the plain fit
        # of the same problem capped at 10_000 steps, and `bare` is untouched.
        bare = growth_problem(net42(); cap=(;))
        @test !haskey(bare.ode_kwargs, :maxiters)
        capped = solve(bare, MultiStart(short_cap; n_starts=1))
        @test capped.convergence.multistart.ode_maxiters == 10_000
        @test capped.parameters == r.parameters
        @test capped.objective == r.objective
        @test !haskey(bare.ode_kwargs, :maxiters)
        # `nothing` turns the cap off: start 1 is the plain uncapped fit. Two
        # LAML iterations keep the uncapped solves cheap.
        tiny = LAML(maxiters=2)
        off = solve(bare, MultiStart(tiny; n_starts=1, ode_maxiters=nothing))
        @test off.convergence.multistart.ode_maxiters === nothing
        @test off.parameters == solve(bare, tiny).parameters
    end

    @testset "reseeding rebuilds only random initializations" begin
        nn = NeuralApproximator(:r, growth_net(); domain=(0.0, 5.0), rng_seed=42)
        nn2 = PSM._reseeded(nn, 7)
        @test nn2.rng_seed == 7 && nn2.domain == nn.domain && nn2.name == nn.name
        @test initial_params(nn2) == initial_params(
            NeuralApproximator(:r, growth_net(); domain=(0.0, 5.0), rng_seed=7))
        @test initial_params(nn2) != initial_params(nn)

        co = COMONetApproximator(:f, (0.0, 1.0), (4,), :increasing; rng_seed=3)
        co2 = PSM._reseeded(co, 9)
        @test co2.rng_seed == 9 && co2.constraint === :increasing
        @test initial_params(co2) != initial_params(co)

        kan_layer(ni, no) = LuxKANLinear(ni, no; grid_size=3, spline_order=3,
                                         standalone_spline_scale=false)
        kan = KANApproximator(:g, Lux.Chain(kan_layer(1, 2), kan_layer(2, 1));
                              input_domains=((0.0, 2.0),),
                              penalty=:edge_curvature, rng_seed=42)
        kan2 = PSM._reseeded(kan, 5)
        @test kan2.rng_seed == 5 && kan2.input_domains == kan.input_domains
        @test kan2.penalty === :edge_curvature
        # The cached initialization is regenerated; the grids are not.
        @test initial_params(kan2) != initial_params(kan)
        @test nparams(kan2) == nparams(kan)
        @test all(l2.knots == l1.knots for (l1, l2) in zip(kan.layers, kan2.layers))

        @test PSM._reseeded(BSplineApproximator(:r, (0.0, 5.0), 6), 1) === nothing
    end

    @testset "deterministic approximators only" begin
        prob = growth_problem([BSplineApproximator(:r, (0.0, 5.0), 6)])
        @test_throws ArgumentError solve(prob, MultiStart(alg; n_starts=2))
    end

    prob42 = neural_problem(42)
    # A short budget keeps the bookkeeping checks cheap; they do not need a
    # converged fit.
    short = LAML(maxiters=10)

    @testset "n_starts = 1 is the plain fit" begin
        plain = solve(prob42, short)
        one = solve(prob42, MultiStart(short; n_starts=1))
        @test one.parameters == plain.parameters
        @test one.objective == plain.objective
        @test one.data_loss == plain.data_loss
        @test one.fitted_values == plain.fitted_values
        @test one.smoothing_params == plain.smoothing_params
        @test isequal(strip_record(one.convergence), plain.convergence)
        r = one.convergence.multistart
        @test r.chosen == 1 && r.n_starts == 1 && r.seeds == [[42]]
        @test r.criterion === :objective   # unpenalized: laml is NaN
        @test isequal(r.values, [plain.objective])
        @test r.reasons == [plain.convergence.reason]
        @test r.converged == [plain.convergence.converged]
    end

    @testset "determinism and seed bookkeeping" begin
        ms = MultiStart(short; n_starts=3, rng_seed=5)
        sol = solve(prob42, ms)
        r = sol.convergence.multistart
        @test length(r.seeds) == length(r.values) == length(r.reasons) ==
              length(r.converged) == 3
        @test r.seeds[1] == [42]
        @test all(s -> length(s) == 1 && s[1] isa Int, r.seeds[2:end])
        @test allunique(first.(r.seeds))
        ok = findall(isfinite, r.values)
        @test r.chosen in ok && r.values[r.chosen] == minimum(r.values[ok])
        @test r.values[r.chosen] == sol.objective
        @test r.converged[r.chosen] == sol.convergence.converged
        @test r.reasons[r.chosen] === sol.convergence.reason

        again = solve(prob42, ms)
        @test again.parameters == sol.parameters
        @test isequal(again.convergence.multistart, r)
        # Fewer starts reuse the same leading seeds and give the same fits.
        two = solve(prob42, MultiStart(short; n_starts=2, rng_seed=5))
        @test two.convergence.multistart.seeds == r.seeds[1:2]
        @test isequal(two.convergence.multistart.values, r.values[1:2])
        # A different rng_seed gives different restarts.
        other = solve(prob42, MultiStart(short; n_starts=2, rng_seed=6))
        @test other.convergence.multistart.seeds[2] != r.seeds[2]
        # The returned start is reproducible from its recorded seed.
        refit = solve(neural_problem(r.seeds[r.chosen][1]), short)
        @test refit.parameters == sol.parameters
    end

    @testset "escapes the poor local minimum" begin
        # Measured before relying on it, with this fixture's integrator cap:
        # rng_seed = 42 alone stops at data loss 5.08 (objective 2.54) and
        # reports :converged_tol. With n_starts = 8, every ms.rng_seed in
        # 1:10 returned a fit below data loss 0.01 (4 to 7 of the 8 starts
        # got there); the default rng_seed = 1 had 6 of 8.
        # How start 1 ends is NOT asserted: as in the F2 testset, seed 42's
        # ending is chaotic in the exact arithmetic. On an ubuntu x64 runner
        # (Julia 1, 2026-11-22) start 1 escaped by itself to objective 1.4e-3
        # and hit :maxiters. What holds everywhere is that the returned fit
        # is good and is the best start.
        sol = solve(prob42, MultiStart(LAML(maxiters=100); n_starts=8))
        r = sol.convergence.multistart
        @test sol.data_loss < 0.01
        ok = findall(isfinite, r.values)
        @test r.values[r.chosen] == minimum(r.values[ok])
        @test r.values[r.chosen] <= r.values[1]
        if r.values[1] > 1   # start 1 stalled (macOS): it must not be returned
            @test r.chosen != 1
        end
    end

    mixed_problem(term) = growth_problem(
        [NeuralApproximator(:r, growth_net(); domain=(0.0, 5.0), rng_seed=42), term];
        dynamics=mixed!)

    @testset "penalized term: selection by the LAML criterion" begin
        # A penalized spline alongside the network makes laml finite, so
        # selection switches to the maximized criterion. Only the network is
        # re-seeded. (Measured laml at maxiters = 4: 120.9 for start 1.)
        prob = mixed_problem(BSplineApproximator(:s, (0.0, 10.0), 5))
        sol = solve(prob, MultiStart(LAML(maxiters=4); n_starts=3))
        r = sol.convergence.multistart
        @test length(r.seeds[1]) == 1
        @test r.criterion === :laml
        @test all(isfinite, r.values)
        @test r.values[r.chosen] == maximum(r.values)
        @test r.values[r.chosen] == sol.convergence.laml
    end

    @testset "the user's problem is not mutated" begin
        # A plain LAML fit adapts a GP kernel in place after warmup
        # (measured: this GP's variance moves at maxiters = 4). Every start,
        # including the first, fits a copy.
        gp = GPApproximator(:s, (0.0, 10.0), 4)
        before = (gp.lengthscale, gp.variance, copy(gp.K))
        sol = solve(mixed_problem(gp), MultiStart(LAML(maxiters=4); n_starts=1))
        @test (gp.lengthscale, gp.variance, gp.K) == before
        @test isfinite(sol.objective)
    end
end

end # module
