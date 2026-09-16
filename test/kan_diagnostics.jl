module KANDiagnosticTests

using Test
using Random
using LinearAlgebra
using Lux, FluxKAN
using PartiallySpecifiedModels

const PSM = PartiallySpecifiedModels

layer(ni, no, grid=2; activation=identity) =
    LuxKANLinear(ni, no; grid_size=grid, standalone_spline_scale=false,
                 base_activation=activation)

@testset "KAN representation diagnostics" begin
    @testset "Complete edges agree with independent backend bases" begin
        model = Lux.Chain(first=layer(2, 3, 3; activation=FluxKAN.SiLU),
                          middle=layer(3, 2, 2; activation=tanh),
                          last=layer(2, 1, 4))
        a = KANApproximator(:g, model; input_domains=((0.0, 2.0), (10.0, 20.0)))
        beta = 0.2cos.(collect(1:nparams(a)))
        ps = PSM.ComponentArray(beta, a.param_axes)
        for extent in (:grid, :support)
            curves = kan_edge_curves(a, beta; npoints=17, extent)
            @test length(curves) == 14
            @test [(c.layer, c.input, c.output) for c in curves] ==
                  [(l, i, o) for (l, spec) in enumerate(a.layers)
                   for i in 1:spec.input_dim for o in 1:spec.output_dim]
            for c in curves
                key = keys(a.model.layers)[c.layer]
                model_layer = getproperty(a.model.layers, key)
                lp, st = getproperty(ps, key), getproperty(a.state, key)
                nb = model_layer.grid_size + 3
                coeff = lp.spline_weight[c.output, (c.input-1)*nb+1:c.input*nb]
                reference_base = lp.base_weight[c.output, c.input] .* model_layer.base_activation.(c.x)
                reference_spline = [dot(coeff, vec(FluxKAN.bspline_basis(
                    reshape([x], 1, 1), vec(st.grid[c.input, :]), 3))) for x in c.x]
                # Measured component/recomposition discrepancies <=5.56e-17
                # on this three-layer fixture; 1e-12 gives >17000x headroom.
                @test isapprox(c.base, reference_base; atol=1e-12, rtol=1e-12)
                @test isapprox(c.spline, reference_spline; atol=1e-12, rtol=1e-12)
                @test isapprox(c.total, reference_base+reference_spline; atol=1e-12, rtol=1e-12)
                @test isapprox(c.total, c.base+c.spline; atol=1e-12, rtol=1e-12)
                @test length(c.x) == length(c.total) == 17
                domain = extent == :grid ? c.grid_domain : c.support_domain
                @test (first(c.x), last(c.x)) == domain
                @test c.support_domain[1] < c.grid_domain[1] < c.grid_domain[2] < c.support_domain[2]
                if c.layer == 1
                    lo, hi = a.logical_domain
                    d = a.input_domains[c.input]
                    @test c.physical_x == [(x-lo)*((d[2]-d[1])/(hi-lo))+d[1] for x in c.x]
                else
                    @test c.physical_x === nothing
                end
                if extent == :support
                    @test first(c.spline) == last(c.spline) == 0
                end
            end
        end

        samples = [0.0 10.0; 0.4 11.0; 1.0 15.0; 2.0 20.0; -3.0 25.0; 5.0 -10.0]
        traced = kan_activation_diagnostics(a, beta, samples)
        f = build_evaluator(a, beta)
        @test traced.nsamples == size(samples, 1)
        @test length(traced.layers) == 3
        # The trace uses the same scalar operations; the initial two-layer
        # probe agreed exactly. Allow 1e-12 for storage/association differences.
        @test isapprox(traced.predictions, [f(row...) for row in eachrow(samples)];
                       atol=1e-12, rtol=1e-12)
        backend_input = permutedims(traced.layers[1].inputs)
        for (l, key) in enumerate(keys(a.model.layers))
            backend_output, _ = Lux.apply(getproperty(a.model.layers, key), backend_input,
                                           getproperty(ps, key), getproperty(a.state, key))
            # Frozen 80-fit backend errors measured <=3.34e-16;
            # 1e-11 leaves >29000x headroom for layer-wise comparisons.
            @test isapprox(traced.layers[l].outputs, permutedims(backend_output);
                           atol=1e-11, rtol=1e-11)
            backend_input = backend_output
        end
    end

    @testset "Hidden coordinates are neither normalized nor clamped" begin
        a = KANApproximator(:g, Lux.Chain(layer(2, 2), layer(2, 1));
                           input_domains=((10.0, 14.0), (20.0, 24.0)))
        beta = zeros(nparams(a))
        beta[a.layers[1].base_range] .= [2.0, -2.0, 0.0, 0.0]
        beta[a.layers[2].base_range] .= [1.0, -0.5]
        samples = [6 22; 10 22; 12 22; 14 22; 18 22]
        d = kan_activation_diagnostics(a, beta, samples)
        logical = [-3.0, -1.0, 0.0, 1.0, 3.0]
        @test d.layers[1].inputs == hcat(logical, zeros(5))
        @test d.layers[1].outputs == hcat(2logical, -2logical)
        @test d.layers[2].inputs == d.layers[1].outputs
        @test d.predictions == 3logical
        @test d.layers[1].coverage[1].interval_counts == [1, 2]
        @test d.layers[1].coverage[2].interval_counts == [0, 5]
        @test d.layers[1].coverage[1].outside_grid_fraction == 2/5
        @test d.layers[1].coverage[1].no_spline_basis == 0
        @test d.layers[1].coverage[2].interval_coverage == 1/2
        for c in d.layers[2].coverage
            @test c.observed_range == (-6.0, 6.0)
            @test c.outside_grid_fraction == 4/5
            @test c.no_spline_basis == 2
            @test c.no_spline_basis_fraction == 2/5
        end
        @test d.layers[2].outputs[:, 1] == d.predictions

        curves = kan_edge_curves(a, beta; npoints=5)
        @test curves[1].physical_x == collect(10.0:14.0)
        @test curves[3].physical_x == collect(20.0:24.0)
        @test all(c.physical_x === nothing for c in curves if c.layer == 2)
        @test all(all(iszero, c.spline) for c in curves)
        @test curves[1].total == 2curves[1].x
    end

    @testset "Grid occupancy is distinct from spline support" begin
        a = KANApproximator(:g, layer(1, 1))
        beta = ones(nparams(a))
        beta[1] = 2.0
        knots = vec(only(a.layers).knots)
        samples = [first(knots)-1, first(knots), (knots[1]+knots[2])/2,
                   -1.0, 0.0, 1.0, (knots[end-1]+knots[end])/2, last(knots), last(knots)+1]
        d = kan_activation_diagnostics(a, beta, samples)
        c = only(only(d.layers).coverage)
        @test c.grid_knots == [-1.0, 0.0, 1.0]
        @test c.support_domain == (-4.0, 4.0)
        @test c.below_grid == c.above_grid == 3
        @test c.outside_grid_fraction == 6/9
        @test c.no_spline_basis == 4
        @test c.no_spline_basis_fraction == 4/9
        @test c.interval_counts == [1, 2]
        @test sum(c.interval_counts)+c.below_grid+c.above_grid == length(samples)
        @test c.interval_coverage == 1.0
        @test d.predictions[[1, 2, 8, 9]] == 2samples[[1, 2, 8, 9]]
        curve = only(kan_edge_curves(a, beta; npoints=17, extent=:support))
        @test curve.spline[1] == curve.spline[end] == 0
        @test curve.spline[2] > 0 && curve.spline[end-1] > 0
        @test curve.physical_x === nothing
        @test d.predictions == kan_activation_diagnostics(a, beta, reshape(samples, :, 1)).predictions
        @test eltype(kan_activation_diagnostics(a, beta, Float32.(samples)).predictions) === Float64

        # Read actual fixed knots rather than reconstructing a uniform grid
        # from declared limits. Only this private fixture copy is shifted.
        shifted = deepcopy(a)
        shifted.layers[1].knots .+= 0.25
        cshift = only(only(kan_activation_diagnostics(shifted, beta, [0.5, 1.25]).layers).coverage)
        @test shifted.logical_domain == (-1.0, 1.0)
        @test cshift.grid_domain == (-0.75, 1.25)
        @test cshift.interval_counts == [0, 2]
        @test cshift.outside_grid_fraction == 0
    end

    @testset "Validation and independent snapshots" begin
        a = KANApproximator(:g, Lux.Chain(layer(2, 2), layer(2, 1));
                           input_domains=((10.0, 14.0), (20.0, 24.0)))
        beta = initial_params(a)
        samples = [10.0 20.0; 12.0 22.0; 14.0 24.0]
        @test_throws ArgumentError kan_edge_curves(a, beta; npoints=1)
        @test_throws ArgumentError kan_edge_curves(a, beta; extent=:unknown)
        @test_throws DimensionMismatch kan_edge_curves(a, beta[2:end])
        @test_throws DimensionMismatch kan_activation_diagnostics(a, beta[2:end], samples)
        @test_throws DimensionMismatch kan_activation_diagnostics(a, beta, zeros(3, 1))
        @test_throws DimensionMismatch kan_activation_diagnostics(a, beta, [1.0, 2.0])
        @test_throws ArgumentError kan_activation_diagnostics(a, beta, zeros(0, 2))
        for invalid in (NaN, Inf, -Inf)
            bad = copy(beta)
            bad[end] = invalid
            @test_throws DomainError kan_edge_curves(a, bad)
            @test_throws DomainError kan_activation_diagnostics(a, bad, samples)
            xs = copy(samples)
            xs[1, 2] = invalid
            @test_throws DomainError kan_activation_diagnostics(a, beta, xs)
        end
        overflow = KANApproximator(:g, Lux.Chain(layer(1, 1), layer(1, 1)))
        large = zeros(nparams(overflow))
        large[only(overflow.layers[1].base_range)] = 2.0
        large[only(overflow.layers[2].base_range)] = 1e308
        @test_throws DomainError kan_activation_diagnostics(overflow, large, [1.0])
        @test_throws r"non-finite outputs in layer 2" kan_activation_diagnostics(overflow, large, [1.0])
        @test_throws DomainError kan_edge_curves(overflow, large; extent=:support)

        initial = initial_params(a)
        state = deepcopy(a.state)
        knots = [copy(l.knots) for l in a.layers]
        saved_samples, saved_beta = copy(samples), copy(beta)
        rng_next = rand(copy(Random.default_rng()))
        curves = kan_edge_curves(a, beta)
        d = kan_activation_diagnostics(a, beta, samples)
        @test rand(copy(Random.default_rng())) == rng_next
        @test samples == saved_samples && beta == saved_beta
        @test a.state == state && initial_params(a) == initial
        @test all(a.layers[l].knots == knots[l] for l in eachindex(a.layers))
        next_inputs = copy(d.layers[2].inputs)
        predictions = copy(d.predictions)
        other_x = copy(curves[2].x)
        d.layers[1].inputs .= -100
        d.layers[1].outputs .= -100
        d.layers[end].outputs .= -100
        d.layers[1].coverage[1].grid_knots .= -100
        curves[1].x .= -100
        curves[1].physical_x .= -100
        curves[1].base .= -100
        @test d.layers[2].inputs == next_inputs
        @test d.predictions == predictions
        @test curves[2].x == other_x
        @test samples == saved_samples && beta == saved_beta
        @test a.state == state && initial_params(a) == initial
        @test all(a.layers[l].knots == knots[l] for l in eachindex(a.layers))
    end
end

end
