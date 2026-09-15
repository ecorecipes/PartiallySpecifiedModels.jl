function _kan_diagnostic_parameters(a::KANApproximator, params::AbstractVector{<:Real})
    beta = Vector{Float64}(params)
    build_evaluator(a, beta)
    beta
end

_kan_grid_knots(layer, input) =
    collect(@view layer.knots[input, layer.degree+1:end-layer.degree])

"""
    kan_edge_curves(approx::KANApproximator, params; npoints=101, extent=:grid)

Sample the effective univariate edge functions of a fixed-grid KAN.
`params` is this approximator's parameter block, not a whole mixed-model
parameter vector. Returns a vector of named tuples, ordered by layer,
input, then output, with:

- `layer`, `input`, `output`: one-based edge indices.
- `x`: the layer's logical input coordinates.
- `physical_x`: corresponding physical coordinates for first-layer edges
  when `input_domains` was supplied; `nothing` for hidden/raw-input layers.
- `base`, `spline`, `total`: residual/base, spline, and complete edge values.
- `grid_domain`, `support_domain`: the central fixed-knot interval and the
  full padded knot interval.

`extent=:grid` samples the central knot interval; `:support` includes its
padding. The spline vanishes at the outer support endpoints, but the base
branch need not. Complete values use the evaluator's accumulation order;
`base + spline` can differ from `total` by floating-point roundoff.

These are owned CPU Float64 data snapshots, not differentiable evaluators.
Inputs, weights and model grids are not modified. Hidden coordinates are
activations, not physical covariates. Edge decompositions are not uniquely
identified scientific components, and these curves are not confidence bands.
"""
function kan_edge_curves(a::KANApproximator, params::AbstractVector{<:Real};
                         npoints::Int=101, extent::Symbol=:grid)
    npoints >= 2 || throw(ArgumentError("kan_edge_curves: npoints must be at least 2"))
    extent in (:grid, :support) ||
        throw(ArgumentError("kan_edge_curves: extent must be :grid or :support"))
    beta = _kan_diagnostic_parameters(a, params)
    curves = NamedTuple[]
    for (index, layer) in enumerate(a.layers), input in 1:layer.input_dim
        grid = _kan_grid_knots(layer, input)
        grid_domain = (first(grid), last(grid))
        support_domain = (layer.knots[input, 1], layer.knots[input, end])
        domain = extent === :grid ? grid_domain : support_domain
        for output in 1:layer.output_dim
            x = collect(range(domain[1], domain[2]; length=npoints))
            physical_x = if index == 1 && a.input_domains !== nothing
                d = a.input_domains[input]
                lo, hi = a.logical_domain
                [(v-lo) * ((d[2]-d[1])/(hi-lo)) + d[1] for v in x]
            else
                nothing
            end
            physical_x === nothing || all(isfinite, physical_x) ||
                throw(DomainError(physical_x, "kan_edge_curves: physical coordinates overflow in layer $index input $input"))
            base, spline, total = similar(x), similar(x), similar(x)
            for k in eachindex(x)
                span, basis = _kan_cubic_support(layer.knots, input, x[k])
                activation = layer.activation(x[k])
                # A zero span isolates the base; zero activation isolates
                # the spline without duplicating coefficient-layout logic.
                base[k] = _kan_edge_value(layer, beta, input, output, activation, 0, basis)
                spline[k] = _kan_edge_value(layer, beta, input, output, zero(activation), span, basis)
                total[k] = _kan_edge_value(layer, beta, input, output, activation, span, basis)
            end
            all(isfinite, base) && all(isfinite, spline) && all(isfinite, total) ||
                throw(DomainError(total, "kan_edge_curves: non-finite edge in layer $index input $input output $output"))
            push!(curves, (; layer=index, input, output, x, physical_x,
                           base, spline, total, grid_domain, support_domain))
        end
    end
    curves
end

function _kan_input_coverage(layer, input, values)
    grid = _kan_grid_knots(layer, input)
    counts = zeros(Int, length(grid)-1)
    below = above = no_spline = 0
    for x in values
        if x < first(grid)
            below += 1
        elseif x > last(grid)
            above += 1
        else
            # Internal knots belong to the interval on their right; the
            # central grid's upper endpoint belongs to its final interval.
            bin = min(searchsortedlast(grid, x), length(counts))
            counts[bin] += 1
        end
        _, basis = _kan_cubic_support(layer.knots, input, x)
        all(iszero, basis) && (no_spline += 1)
    end
    n = length(values)
    (; input, grid_knots=grid, grid_domain=(first(grid), last(grid)),
       support_domain=(layer.knots[input, 1], layer.knots[input, end]),
       observed_range=extrema(values), below_grid=below, above_grid=above,
       outside_grid_fraction=(below+above)/n,
       no_spline_basis=no_spline, no_spline_basis_fraction=no_spline/n,
       interval_counts=counts, interval_coverage=count(!iszero, counts)/length(counts))
end

"""
    kan_activation_diagnostics(approx::KANApproximator, params, samples)

Trace a fixed-grid KAN on an `n_samples`-by-`n_inputs` real matrix. For a
univariate KAN, a vector of samples is also accepted. Samples and this
approximator's parameter block are copied to CPU Float64; empty or
non-finite samples and non-finite intermediate outputs are rejected.

Returns `(nsamples, layers, predictions)`. Each layer contains `layer`,
`logical_domain`, `inputs`, `outputs`, and a `coverage` entry per input.
Only the first layer applies `input_domains` normalization. Hidden-layer
inputs are the preceding outputs, with no clamping or rescaling.

Each coverage entry reports:

- Actual central `grid_knots`, `grid_domain`, padded `support_domain`,
  and the sample `observed_range`.
- `below_grid`, `above_grid`, and `outside_grid_fraction`.
- `no_spline_basis` and its fraction: samples at which every cubic basis
  is zero. The residual/base branch may still contribute there.
- `interval_counts` and `interval_coverage`: occupancy of central knot
  intervals. Interior boundaries go right; the upper endpoint goes left.

Actual knot limits can differ slightly from the declared `logical_domain`
after upstream grid rounding. Occupancy is measured against the actual
knots. All returned arrays are owned snapshots; no model/optimizer state,
parameters or grids are changed.

Coverage is marginal exposure in each layer coordinate, not joint
trajectory support, fit quality, identifiability, or a pruning guarantee.
Unoccupied intervals do not imply unused coefficients because spline
bases overlap intervals. This diagnostic does not adapt grids or provide
uncertainty intervals.
"""
function kan_activation_diagnostics(a::KANApproximator, params::AbstractVector{<:Real},
                                    samples::AbstractMatrix{<:Real})
    size(samples, 2) == a.input_dim ||
        throw(DimensionMismatch("kan_activation_diagnostics: expected $(a.input_dim) input columns, got $(size(samples, 2))"))
    size(samples, 1) > 0 ||
        throw(ArgumentError("kan_activation_diagnostics: at least one sample is required"))
    beta = _kan_diagnostic_parameters(a, params)
    current = Matrix{Float64}(samples)
    all(isfinite, current) ||
        throw(DomainError(samples, "kan_activation_diagnostics: samples must be finite Float64 values"))
    for row in axes(current, 1)
        normalized = _kan_normalize_inputs(a, Tuple(@view current[row, :]))
        current[row, :] .= normalized
    end
    all(isfinite, current) ||
        throw(DomainError(current, "kan_activation_diagnostics: non-finite normalized inputs"))
    traces = NamedTuple[]
    for (index, layer) in enumerate(a.layers)
        inputs = copy(current)
        outputs = Matrix{Float64}(undef, size(inputs, 1), layer.output_dim)
        for row in axes(inputs, 1)
            outputs[row, :] .= _kan_layer_output(layer, beta, Tuple(@view inputs[row, :]))
        end
        all(isfinite, outputs) ||
            throw(DomainError(outputs, "kan_activation_diagnostics: non-finite outputs in layer $index"))
        coverage = [_kan_input_coverage(layer, input, @view(inputs[:, input]))
                    for input in 1:layer.input_dim]
        push!(traces, (; layer=index, logical_domain=layer.logical_domain,
                       inputs, outputs, coverage))
        current = outputs
    end
    (; nsamples=size(samples, 1), layers=traces, predictions=vec(copy(current)))
end

function kan_activation_diagnostics(a::KANApproximator, params::AbstractVector{<:Real},
                                    samples::AbstractVector{<:Real})
    a.input_dim == 1 ||
        throw(DimensionMismatch("kan_activation_diagnostics: multivariate samples require a matrix with one sample per row"))
    kan_activation_diagnostics(a, params, reshape(samples, :, 1))
end
