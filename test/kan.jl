module KANTests

using Test
using Random
using LinearAlgebra
using Lux
using FluxKAN
using SciMLSensitivity
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve

const PSM = PartiallySpecifiedModels
const FD = PSM.ForwardDiff

layer(ni, no) = LuxKANLinear(ni, no;
    grid_size=3, spline_order=3, standalone_spline_scale=false)
network(ni=1) = Lux.Chain(layer(ni, 2), layer(2, 1))

function reference(a, beta, xs...)
    input = collect(xs)
    if a.input_domains !== nothing
        lo, hi = a.logical_domain
        input = [(x-d[1]) / (d[2]-d[1]) * (hi-lo) + lo
                 for (x, d) in zip(input, a.input_domains)]
    end
    only(first(Lux.apply(a.model, reshape(input, :, 1),
                        PSM.ComponentArray(beta, a.param_axes), a.state)))
end

function finite_gradient(f, x)
    h = 1e-5
    grad = similar(x)
    for j in eachindex(x)
        xp, xm = copy(x), copy(x)
        xp[j] += h
        xm[j] -= h
        grad[j] = (f(xp)-f(xm)) / (2h)
    end
    grad
end

function evaluation_allocations(f, x)
    f(x)
    FD.derivative(f, x)
    plain = @allocated f(x)
    differentiated = @allocated FD.derivative(f, x)
    plain, differentiated
end

@testset "Fixed-grid KAN approximators" begin
    @testset "Optional backend and initialization" begin
        @test Base.get_extension(PSM, :PartiallySpecifiedModelsFluxKANExt) !== nothing
        probe = "using PartiallySpecifiedModels; " *
                "@assert isdefined(PartiallySpecifiedModels, :KANApproximator); " *
                "@assert isdefined(PartiallySpecifiedModels, :kan_edge_curves); " *
                "@assert isdefined(PartiallySpecifiedModels, :kan_activation_diagnostics); " *
                "@assert Base.get_extension(PartiallySpecifiedModels, " *
                ":PartiallySpecifiedModelsFluxKANExt) === nothing"
        @test success(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) -e $probe`)

        domains = [[0.0, 2.0], [10.0, 20.0]]
        peek = rand(copy(Random.default_rng()))
        a = KANApproximator("g", network(2); input_domains=domains, rng_seed=42)
        @test rand(copy(Random.default_rng())) == peek
        @test a.name === :g
        @test a.input_dim == 2
        @test nparams(a) == (2*2 + 2*1) * (3+3+1)
        @test eltype(initial_params(a)) === Float64
        @test all(l -> eltype(l.knots) === Float64, a.layers)
        @test initial_params(a) == initial_params(KANApproximator(:g, network(2); rng_seed=42))
        beta = initial_params(a)
        beta[1] += 1
        @test initial_params(a) != beta
        domains[1][1] = -10
        @test a.input_domains == ((0.0, 2.0), (10.0, 20.0))
        @test band_domain(a) === nothing
        @test band_domain(KANApproximator(:g, layer(1, 1))) === nothing
        @test band_domain(KANApproximator(:g, layer(1, 1);
            input_domains=((0.0, 2.0),))) == (0.0, 2.0)
        fresh = KANApproximator(:g, layer(1, 1); rng_seed=nothing)
        @test initial_params(fresh) == initial_params(fresh)

        named = Lux.Chain(first=layer(2, 2), last=layer(2, 1))
        an = KANApproximator(:g, named)
        ps_named, _ = Lux.f64(Lux.setup(Random.Xoshiro(42), named))
        @test keys(PSM.ComponentArray(initial_params(an), an.param_axes)) ==
              keys(PSM.ComponentArray(ps_named))
    end

    @testset "Validation" begin
        @test_throws ArgumentError KANApproximator(:g, Lux.Dense(1, 1))
        @test_throws ArgumentError KANApproximator(:g, Lux.Chain(layer(1, 1), Lux.Dense(1, 1)))
        @test_throws ArgumentError KANApproximator(:g, LuxKANLinear(1, 1))
        @test_throws ArgumentError KANApproximator(:g,
            LuxKANLinear(1, 1; spline_order=1, standalone_spline_scale=false))
        @test_throws ArgumentError KANApproximator(:g, layer(1, 2))
        @test_throws DimensionMismatch KANApproximator(:g, Lux.Chain(layer(1, 2), layer(3, 1)))
        @test_throws ArgumentError KANApproximator(:g, layer(1, 1); penalty=:unknown)
        @test_throws ArgumentError KANApproximator(:g, layer(1, 1); nullspace_penalty=1.0)
        @test_throws ArgumentError KANApproximator(:g, layer(1, 1);
            penalty=:edge_curvature, nullspace_penalty=-1.0)
        @test_throws ArgumentError KANApproximator(:g, layer(1, 1);
            penalty=:edge_curvature, nullspace_penalty=Inf)
        @test_throws ArgumentError KANApproximator(:g,
            LuxKANLinear(1, 1; standalone_spline_scale=false, base_activation=sin);
            penalty=:edge_curvature)
        @test_throws DimensionMismatch KANApproximator(:g, layer(2, 1); input_domains=((0, 1),))
        for domains in (((1.0, 1.0),), ((1.0, 0.0),), ((0.0, Inf),),
                        ((-1e308, 1e308),), ((NaN, 1.0),), (1.0,))
            @test_throws ArgumentError KANApproximator(:g, layer(1, 1); input_domains=domains)
        end
        a = KANApproximator(:g, network(2))
        @test_throws DimensionMismatch build_evaluator(a, zeros(nparams(a)-1))
        for bad in (NaN, Inf, -Inf)
            beta = initial_params(a)
            beta[end] = bad
            @test_throws DomainError build_evaluator(a, beta)
        end
        f = build_evaluator(a, initial_params(a))
        @test_throws DimensionMismatch f(1.0)
        @test_throws DimensionMismatch f(1.0, 2.0, 3.0)
    end

    @testset "Backend parity and pure evaluation" begin
        for a in (KANApproximator(:g, layer(1, 1)),
                  KANApproximator(:g, network()),
                  KANApproximator(:g, network(); input_domains=((0.0, 2.0),)))
            beta = initial_params(a)
            padded = vcat(99.0, beta, -99.0)
            f = build_evaluator(a, view(padded, 2:(length(beta)+1)))
            state_before = deepcopy(a.state)
            knots_before = [copy(l.knots) for l in a.layers]
            for x in (-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 2.0, 3.0)
                # Pilot Float64 backend difference was exactly zero;
                # 1e-11 allows >40000 machine eps of roundoff headroom.
                @test isapprox(f(x), reference(a, beta, x); atol=1e-11, rtol=1e-11)
                @test f([x]) == f(x)
            end
            if a.input_domains === nothing
                for knot in a.layers[1].knots[1, :]
                    for x in (prevfloat(knot), knot, nextfloat(knot))
                        @test isapprox(f(x), reference(a, beta, x); atol=1e-11, rtol=1e-11)
                    end
                end
            end
            first_value = f(0.2)
            f(0.9)
            @test f(0.2) == first_value
            @test a.state == state_before
            @test all(a.layers[i].knots == knots_before[i] for i in eachindex(a.layers))
        end

        a = KANApproximator(:g, network(2); input_domains=((0.0, 2.0), (10.0, 20.0)))
        beta = initial_params(a)
        f = build_evaluator(a, beta)
        for xs in ((0.0, 10.0), (1.0, 15.0), (2.0, 20.0), (-1.0, 25.0))
            @test isapprox(f(xs...), reference(a, beta, xs...); atol=1e-11, rtol=1e-11)
        end
        ac = deepcopy(a)
        ac.layers[1].knots[1, 1] -= 1
        @test ac.layers[1].knots != a.layers[1].knots
    end

    @testset "Local cubic support agrees with dense recursion" begin
        function local_basis(knots, x)
            span, values = PSM._kan_cubic_support(reshape(knots, 1, :), 1, x)
            result = zeros(typeof(values[1]), length(knots)-4)
            for j in max(span-3, 1):min(span, length(result))
                result[j] = values[j-span+4]
            end
            result
        end
        for knots in (collect(range(-3.0, 3.0, length=10)),
                      [-4.0, -3.0, -2.1, -1.4, -0.4, -0.1, 0.7, 1.1, 2.0, 2.9, 4.3, 5.0])
            points = vcat(first(knots)-1, last(knots)+1,
                          knots, prevfloat.(knots), nextfloat.(knots),
                          (knots[1:end-1] + knots[2:end])/2)
            dense = x -> PSM._bspline_basis_vector(x, knots, 4)
            sparse = x -> local_basis(knots, x)
            for x in points
                # Frozen-weight pilot value differences <=2.23e-16;
                # 1e-12 allows >4000x headroom for boundary roundoff.
                @test isapprox(sparse(x), dense(x); atol=1e-12, rtol=1e-12)
                d_dense = FD.jacobian(y -> dense(y[1]), [x])
                d_sparse = FD.jacobian(y -> sparse(y[1]), [x])
                @test isapprox(d_sparse, d_dense; atol=1e-12, rtol=1e-12)
            end
        end
    end

    @testset "Scalar evaluation allocation budgets" begin
        shallow = KANApproximator(:g, LuxKANLinear(1, 1;
            grid_size=24, spline_order=3, standalone_spline_scale=false);
            input_domains=((0.0, 2.0),))
        deep = KANApproximator(:g, network(); input_domains=((0.0, 2.0),))
        fs = build_evaluator(shallow, initial_params(shallow))
        fd = build_evaluator(deep, initial_params(deep))
        evaluation_allocations(fs, 0.2)
        evaluation_allocations(fd, 0.2)
        plain_s, dual_s = evaluation_allocations(fs, 0.2)
        plain_d, dual_d = evaluation_allocations(fd, 0.2)
        # Measured 0/0 bytes versus 2688/4416 before; allow 128 bytes of
        # platform/compiler bookkeeping without admitting full-basis arrays.
        @test plain_s <= 128
        @test dual_s <= 128
        # Measured 80/96 bytes versus 3840/5200 before; >2.6x headroom.
        @test plain_d <= 256
        @test dual_d <= 256
    end

    @testset "Parameter, input and nested derivatives" begin
        for ni in (1, 2)
            a = KANApproximator(:g, network(ni))
            beta = initial_params(a)
            xs = ni == 1 ? [0.2] : [0.13, 0.27]
            fp = b -> build_evaluator(a, b)(xs...)
            fx = x -> build_evaluator(a, beta)(x...)
            gp, gx = FD.gradient(fp, beta), FD.gradient(fx, xs)
            # Pilot parameter/input FD errors <=2.24e-12; >4000x headroom.
            @test maximum(abs, gp-finite_gradient(fp, beta)) < 1e-8
            @test maximum(abs, gx-finite_gradient(fx, xs)) < 1e-8
            @test norm(gp) > 0
            @test norm(gx) > 0
            H = FD.hessian(fx, xs)
            @test all(isfinite, H)
            @test norm(H) > 0
        end
        a = KANApproximator(:g, network())
        beta = initial_params(a)
        for x in (-1000.0, 1000.0)
            @test isfinite(build_evaluator(a, beta)(x))
            @test all(isfinite, FD.gradient(b -> build_evaluator(a, b)(x), beta))
            @test isfinite(FD.derivative(build_evaluator(a, beta), x))
        end

        # Loading FluxKAN must not change the existing Dense-only evaluator.
        mlp = NeuralApproximator(:old,
            Lux.Chain(Lux.Dense(1, 2, tanh), Lux.Dense(2, 1)); rng_seed=42)
        beta_mlp = initial_params(mlp)
        f_mlp = b -> build_evaluator(mlp, b)(0.2)
        # Same central-difference gate as the KAN pilot (2.24e-12 error,
        # >4000x headroom); this control is a smooth two-layer MLP.
        @test maximum(abs, FD.gradient(f_mlp, beta_mlp) -
                           finite_gradient(f_mlp, beta_mlp)) < 1e-8
    end

    @testset "Ridge and mixed parameter blocks" begin
        a = KANApproximator(:g, layer(1, 1); penalty=:ridge)
        @test penalty_matrix(KANApproximator(:g, layer(1, 1))) === nothing
        @test penalty_matrix(a) == Matrix{Float64}(I, nparams(a), nparams(a))
        @test length(penalty_blocks(a)) == 1
        @test last(only(penalty_blocks(a))) == 1:nparams(a)
        @test !(a isa PSM._BUILTIN_APPROX_TYPES)
        spline = BSplineApproximator(:s, (0.0, 1.0), 4; initial=x -> 0.1x)
        rhs!(du, u, p, t) = (du[1] = p.s(u[1]) + p.g(u[1]))
        prob = PSMProblem(rhs!, [0.2], (0.0, 1.0), [spline, a];
            data_times=[0.0, 1.0], data_values=reshape([0.2, 0.3], :, 1))
        beta = PSM.build_initial_params(prob)
        p = PSM.build_param_struct(prob, beta)
        @test p.g(0.2) == build_evaluator(a, initial_params(a))(0.2)
        S, offsets, sizes = PSM.build_penalty_matrices(prob)
        @test offsets == [0, 4]
        @test sizes == [4, nparams(a)]
        @test S[2] == penalty_matrix(a)
        @test PSM.adam_penalty(prob, beta, 1.0) ==
              dot(beta[1:4], S[1]*beta[1:4]) + sum(abs2, beta[5:end])
        @test_throws ErrorException solve(prob, FGPGMSolver(n_samples=1, n_warmup=0))
        @test_throws ErrorException solve(prob, AdaptiveGradientMatching(n_samples=1))
    end

    @testset "Edge curvature in raw coefficient coordinates" begin
        a = KANApproximator(:g, network(2); penalty=:edge_curvature)
        beta = initial_params(a)
        blocks = penalty_blocks(a)
        @test length(blocks) == 2
        @test [r for (_, r) in blocks] ==
              [first(l.base_range):last(l.spline_range) for l in a.layers]
        @test last(blocks[1][2]) < first(blocks[2][2])
        S = penalty_matrix(a)
        @test issymmetric(S)
        @test size(S) == (nparams(a), nparams(a))
        @test S[blocks[1][2], blocks[2][2]] == zeros(length(blocks[1][2]), length(blocks[2][2]))
        cached = copy(S)
        blocks[1][1][1, 1] += 1
        @test penalty_matrix(a) == cached
        @test all(l -> isfinite(sum(l.knots)), a.layers)

        # Independent composite Simpson integration of AD derivatives of
        # the actual FluxKAN edge functions (not the penalty design matrix).
        ps = PSM.ComponentArray(beta, a.param_axes)
        reference_penalty = 0.0
        for key in keys(a.model.layers)
            model_layer = getproperty(a.model.layers, key)
            lp, st = getproperty(ps, key), getproperty(a.state, key)
            lo, hi = Float64(model_layer.grid_min), Float64(model_layer.grid_max)
            nb = model_layer.grid_size + model_layer.spline_order
            for input in 1:model_layer.in_dim, output in 1:model_layer.out_dim
                knots = collect(st.grid[input, :])
                coeff = lp.spline_weight[output, (input-1)*nb+1:input*nb]
                edge = x -> lp.base_weight[output, input] * model_layer.base_activation(x) +
                    dot(coeff, vec(FluxKAN.bspline_basis(reshape([x], 1, 1), knots, 3)))
                second = x -> FD.derivative(z -> FD.derivative(edge, z), x)
                points = sort!(unique(vcat(lo, hi, filter(x -> lo < x < hi, knots))))
                for k in 1:length(points)-1
                    xs = collect(range(points[k], points[k+1], length=129))
                    y = second.(xs).^2
                    h = (last(xs)-first(xs)) / 128
                    reference_penalty += (hi-lo)^3 * h/3 *
                        (first(y)+last(y)+4sum(y[2:2:end-1])+2sum(y[3:2:end-2]))
                end
            end
        end
        actual_penalty = dot(beta, S*beta)
        # Measured Simpson/AD discrepancy 3.75e-11; the absolute gate alone
        # leaves >26x headroom, with a relative allowance for other scales.
        @test isapprox(actual_penalty, reference_penalty; rtol=1e-8, atol=1e-9)
        # Single-edge minimum eigenvalue/scale measured 5.21e-17; allow
        # 1e-10 relative roundoff for the assembled multilayer matrix.
        @test minimum(eigvals(Symmetric(S))) >= -1e-10 * opnorm(S)
        # Symmetric quadratic gradient: agreement is at Float64 roundoff;
        # the 3.75e-11 independent integral error above is already smaller.
        @test isapprox(FD.gradient(b -> dot(b, S*b), beta), 2S*beta; rtol=1e-10, atol=1e-10)

        # A cubic B-spline representation of x^2 has an exact penalty of
        # 4*span^4 after normalization to z in [0,1].
        quadratic = KANApproximator(:g, layer(1, 1); penalty=:edge_curvature)
        spec = only(quadratic.layers)
        knots = vec(spec.knots)
        nb = length(knots)-4
        coeff = [(knots[j+1]*knots[j+2] + knots[j+1]*knots[j+3] +
                  knots[j+2]*knots[j+3])/3 for j in 1:nb]
        bq = vcat(0.0, coeff)
        expected = 4 * (spec.logical_domain[2]-spec.logical_domain[1])^4
        # Measured absolute error 5.69e-14 at expected=64; >1e5x headroom.
        @test isapprox(dot(bq, penalty_matrix(quadratic)*bq), expected; rtol=1e-10)
        affine = vcat(0.0, [sum(knots[j+1:j+3])/3 for j in 1:nb])
        # Measured null energy 2.18e-14; >45000x headroom.
        @test abs(dot(affine, penalty_matrix(quadratic)*affine)) < 1e-9
        shrunk = KANApproximator(:g, layer(1, 1);
            penalty=:edge_curvature, nullspace_penalty=0.01)
        # Measured absolute ridge error 7.43e-15; >1000x headroom.
        @test isapprox(dot(affine, (penalty_matrix(shrunk)-penalty_matrix(quadratic))*affine),
                       0.01sum(abs2, affine); rtol=1e-9)
        identity_layer = LuxKANLinear(1, 1; grid_size=3,
            standalone_spline_scale=false, base_activation=identity)
        ai = KANApproximator(:g, identity_layer; penalty=:edge_curvature)
        air = KANApproximator(:g, identity_layer;
            penalty=:edge_curvature, nullspace_penalty=0.01)
        @test all(iszero, penalty_matrix(ai)[1, :])
        @test penalty_matrix(air)[1, 1] == 0.01

        # Dense solvers see one correctly offset block per KAN layer.
        bs = BSplineApproximator(:s, (0.0, 1.0), 4)
        rhs!(du, u, p, t) = (du[1] = p.g(u[1], t) + p.s(t))
        prob = PSMProblem(rhs!, [0.2], (0.0, 1.0), [bs, a];
            data_times=[0.0, 1.0], data_values=reshape([0.2, 0.3], :, 1))
        _, offsets, sizes = PSM.build_penalty_matrices(prob)
        @test offsets == [0, 4, 4+length(first(a.curvature_blocks)[2])]
        @test sizes == [4, length(a.curvature_blocks[1][2]), length(a.curvature_blocks[2][2])]
        @test isapprox(PSM.adam_penalty(prob, vcat(zeros(4), beta), 1.0),
                       actual_penalty; rtol=1e-12)
    end

    @testset "Layer-wise smoothing through LAML and GCV" begin
        a = KANApproximator(:r, Lux.Chain(layer(1, 1), layer(1, 1));
            input_domains=((0.0, 1.0),), penalty=:edge_curvature,
            nullspace_penalty=1e-6)
        ts = collect(0.0:0.05:1.0)
        rhs!(du, u, p, t) = (du[1] = p.r(t))
        prob = PSMProblem(rhs!, [0.2], (0.0, 1.0), [a]; data_times=ts,
            data_values=reshape(0.2 .+ 0.25ts .+ 0.05ts.^2, :, 1))
        for alg in (LAML(maxiters=25, jac=:forwarddiff),
                    GCVSolver(maxiters=20, jac=:forwarddiff))
            sol = solve(prob, alg)
            @test length(sol.smoothing_params) == 2
            @test all(x -> isfinite(x) && x > 0, sol.smoothing_params)
            # Measured RSS <=1.71e-11 from initial 0.608; >5e7x headroom.
            @test sol.data_loss < 1e-3
            @test sol.fitted_values == simulate(prob, collect(sol.parameters))
        end
    end

    @testset "ODE, stiff Jacobian and adjoint integration" begin
        a = KANApproximator(:f, network())
        beta = initial_params(a)
        rhs!(du, u, p, t) = (du[1] = p.f(u[1]))
        times = collect(0.0:0.2:1.0)
        prob = PSMProblem(rhs!, [0.2], (0.0, 1.0), [a];
            data_times=times, data_values=reshape(0.2 .+ 0.1times, :, 1))
        objective = b -> PSM.adam_loss_mse(prob, b)
        g_forward = FD.gradient(objective, beta)
        value_adjoint, g_adjoint = PSM._adam_adjoint_loss_grad(prob, beta, :mse, 0.0, :auto)
        # Pilot objective mismatch 2.8e-17 and gradient mismatch 7.64e-11;
        # gates leave >1e6x and >100x headroom respectively.
        @test isapprox(value_adjoint, objective(beta); atol=1e-10, rtol=1e-10)
        @test maximum(abs, g_forward-g_adjoint) < 1e-8
        @test norm(g_adjoint) > 0

        stiff = PSMProblem(rhs!, [0.2], (0.0, 1.0), [a];
            data_times=times, data_values=prob.data_values, solver=TRBDF2(),
            abstol=1e-10, reltol=1e-10)
        J = FD.jacobian(b -> vec(simulate(stiff, b)), beta)
        @test all(isfinite, J)
        @test norm(J) > 0

        dde!(du, u, h, p, t) = (du[1] = p.f(h(p, t-0.2)[1]))
        delayed = PSMProblem(dde!, [0.2], (0.0, 1.0), [a];
            data_times=times, data_values=prob.data_values, delays=[0.2])
        @test all(isfinite, FD.jacobian(b -> vec(simulate(delayed, b)), beta))

        map!(du, u, p, t) = (du[1] = u[1] + 0.1p.f(u[1]))
        discrete = PSMProblem(map!, [0.2], (0.5, 5.5), [a];
            data_times=collect(0.5:5.5), data_values=prob.data_values, discrete=true)
        @test all(isfinite, FD.jacobian(b -> vec(simulate(discrete, b)), beta))
    end

    @testset "Fitting and univariate bootstrap" begin
        a = KANApproximator(:r, layer(1, 1); input_domains=((0.0, 1.0),))
        times = collect(0.0:0.05:1.0)
        data = reshape(0.2 .+ 0.25times .+ 0.05times.^2, :, 1)
        rhs!(du, u, p, t) = (du[1] = p.r(t))
        prob = PSMProblem(rhs!, [0.2], (0.0, 1.0), [a];
            data_times=times, data_values=data, abstol=1e-10, reltol=1e-10)
        initial_loss = PSM.weighted_data_loss(prob, simulate(prob, initial_params(a)))
        for alg in (LAML(maxiters=20, jac=:forwarddiff),
                    GCVSolver(maxiters=20, jac=:forwarddiff),
                    AdamSolver(maxiters=200, lr=0.03))
            sol = solve(prob, alg)
            @test sol.data_loss < initial_loss
            @test sol.parameters.r != initial_params(a)
            @test sol.fitted_values == simulate(prob, collect(sol.parameters))
            if alg isa AdamSolver
                # Measured 2.08e-5 after 200 steps; >48x headroom.
                @test sol.data_loss < 1e-3
            else
                # This one-layer, time-driven model is linear in its weights.
                # Measured RSS <=1.98e-24; >5e11x headroom.
                @test sol.data_loss < 1e-12
            end
        end
        sol = solve(prob, LAML(maxiters=20, jac=:forwarddiff))
        bands = confidence_band(sol, prob; uf_ngrid=5)
        @test all(isfinite, bands[:r].se)
        before = deepcopy(a.state)
        bs = bootstrap(sol, prob, LAML(maxiters=20, jac=:forwarddiff);
                       nboot=3, uf_ngrid=5, rng=Random.Xoshiro(17))
        @test bs.n_success == 3
        @test size(bs.uf_values[:r]) == (5, 3)
        @test bs.uf_grid[:r] == collect(range(0.0, 1.0, length=5))
        @test a.state == before
    end
end

end
