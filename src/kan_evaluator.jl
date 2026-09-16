struct _KANSplineLayer{F}
    input_dim::Int
    output_dim::Int
    degree::Int
    knots::Matrix{Float64}
    logical_domain::Tuple{Float64,Float64}
    base_range::UnitRange{Int}
    spline_range::UnitRange{Int}
    activation::F
end

function _kan_layer_info(layer)
    throw(ArgumentError(
        "KANApproximator: unsupported layer $(typeof(layer)). Load FluxKAN " *
        "and use cubic LuxKANLinear layers with standalone_spline_scale=false."))
end

function _kan_layer_spec(layer, ps, st, offset)
    throw(ArgumentError("KANApproximator: no parameter adapter for $(typeof(layer))"))
end

function _kan_input_domains(domains, input_dim)
    domains === nothing && return nothing
    (domains isa Tuple || domains isa AbstractVector) ||
        throw(ArgumentError("KANApproximator: input_domains must contain one (lo, hi) pair per input"))
    length(domains) == input_dim ||
        throw(DimensionMismatch("KANApproximator: expected $input_dim input domains, got $(length(domains))"))
    parsed = map(enumerate(domains)) do (i, domain)
        (domain isa Tuple || domain isa AbstractVector) && length(domain) == 2 &&
            all(x -> x isa Real, domain) ||
            throw(ArgumentError("KANApproximator: input domain $i must be a pair of real numbers"))
        d = (Float64(domain[1]), Float64(domain[2]))
        all(isfinite, d) && isfinite(d[2]-d[1]) ||
            throw(ArgumentError("KANApproximator: input domain $i must be finite"))
        _validate_domain("KANApproximator input $i", d)
        d
    end
    Tuple(parsed)
end

function KANApproximator(name::Union{Symbol,String}, model;
                         input_domains=nothing, penalty::Symbol=:none,
                         nullspace_penalty::Real=0.0,
                         rng_seed::Union{Nothing,Int}=42)
    penalty in (:none, :ridge, :edge_curvature) ||
        throw(ArgumentError("KANApproximator: penalty must be :none, :ridge, or :edge_curvature"))
    ridge = Float64(nullspace_penalty)
    isfinite(ridge) && ridge >= 0 ||
        throw(ArgumentError("KANApproximator: nullspace_penalty must be finite and nonnegative"))
    penalty === :edge_curvature || iszero(ridge) ||
        throw(ArgumentError("KANApproximator: nullspace_penalty requires penalty=:edge_curvature"))
    layer_models = model isa Lux.Chain ? Tuple(values(model.layers)) : (model,)
    isempty(layer_models) &&
        throw(ArgumentError("KANApproximator: the network must contain at least one layer"))
    info = map(_kan_layer_info, layer_models)
    for i in 2:length(info)
        info[i-1].output_dim == info[i].input_dim ||
            throw(DimensionMismatch("KANApproximator: dimensions do not match between layers $(i-1) and $i"))
    end
    last(info).output_dim == 1 ||
        throw(ArgumentError("KANApproximator: the final layer must have one scalar output"))
    domains = _kan_input_domains(input_domains, first(info).input_dim)

    owned_model = deepcopy(model)
    rng = rng_seed === nothing ? Random.Xoshiro(rand(UInt64)) : Random.Xoshiro(rng_seed)
    ps, st = Lux.f64(Lux.setup(rng, owned_model))
    packed = ComponentArray(ps)
    beta = Float64.(collect(packed))
    all(isfinite, beta) ||
        throw(ArgumentError("KANApproximator: backend initialization produced non-finite parameters"))
    axes = getaxes(packed)
    offset = 0
    if owned_model isa Lux.Chain
        specs = map(keys(owned_model.layers)) do key
            spec = _kan_layer_spec(getproperty(owned_model.layers, key),
                                   getproperty(ps, key), getproperty(st, key), offset)
            offset = last(spec.spline_range)
            spec
        end
    else
        spec = _kan_layer_spec(owned_model, ps, st, offset)
        specs = (spec,)
        offset = last(spec.spline_range)
    end
    offset == length(beta) ||
        throw(DimensionMismatch("KANApproximator: the adapter did not account for every network parameter"))
    layers = Tuple(specs)
    blocks = penalty === :edge_curvature ? _kan_curvature_blocks(layers, ridge) :
             Tuple{Matrix{Float64},UnitRange{Int}}[]
    KANApproximator(Symbol(name), owned_model, st, axes, layers, domains,
                    first(info).logical_domain, first(info).input_dim, beta,
                    penalty, ridge, blocks, rng_seed)
end

nparams(a::KANApproximator) = length(a.initial_parameters)
initial_params(a::KANApproximator) = copy(a.initial_parameters)
function penalty_matrix(a::KANApproximator)
    a.penalty === :none && return nothing
    a.penalty === :ridge && return Matrix{Float64}(I, nparams(a), nparams(a))
    S = zeros(nparams(a), nparams(a))
    for (block, indices) in a.curvature_blocks
        S[indices, indices] .= block
    end
    S
end

function penalty_blocks(a::KANApproximator)
    a.penalty === :none && return Tuple{Matrix{Float64},UnitRange{Int}}[]
    a.penalty === :ridge && return [(penalty_matrix(a), 1:nparams(a))]
    [(copy(S), indices) for (S, indices) in a.curvature_blocks]
end
band_domain(a::KANApproximator) =
    a.input_dim == 1 && a.input_domains !== nothing ? only(a.input_domains) : nothing

function _kan_silu(x)
    if x >= 0
        x / (one(x) + exp(-x))
    else
        e = exp(x)
        x * e / (one(x) + e)
    end
end

struct _KANEvaluator{A,P}
    approx::A
    params::P
end

@inline function _kan_knot_span(knots, feature, x)
    nk = size(knots, 2)
    (x < knots[feature, 1] || !(x < knots[feature, nk])) && return 0
    lo, hi = 1, nk-1
    while lo < hi
        mid = lo + div(hi-lo+1, 2)
        if x < knots[feature, mid]
            hi = mid-1
        else
            lo = mid
        end
    end
    lo
end

@inline function _kan_local_basis_step(knots, feature, j, degree, x, left, right, z)
    1 <= j <= size(knots, 2)-degree-1 || return z
    a = (x-knots[feature, j]) /
        (knots[feature, j+degree]-knots[feature, j])
    b = (knots[feature, j+degree+1]-x) /
        (knots[feature, j+degree+1]-knots[feature, j+1])
    a*left + b*right
end

@inline function _kan_cubic_support(knots, feature, x)
    span = _kan_knot_span(knots, feature, x)
    z = zero(x-knots[feature, 1])
    span == 0 && return (span, (z, z, z, z))
    u = one(z)
    # Only four cubic bases can be nonzero. The index guards also handle
    # the padded boundary intervals, where fewer than four bases exist.
    a = _kan_local_basis_step(knots, feature, span-1, 1, x, z, u, z)
    b = _kan_local_basis_step(knots, feature, span,   1, x, u, z, z)
    c = _kan_local_basis_step(knots, feature, span-2, 2, x, z, a, z)
    d = _kan_local_basis_step(knots, feature, span-1, 2, x, a, b, z)
    e = _kan_local_basis_step(knots, feature, span,   2, x, b, z, z)
    values = (
        _kan_local_basis_step(knots, feature, span-3, 3, x, z, c, z),
        _kan_local_basis_step(knots, feature, span-2, 3, x, c, d, z),
        _kan_local_basis_step(knots, feature, span-1, 3, x, d, e, z),
        _kan_local_basis_step(knots, feature, span,   3, x, e, z, z))
    span, values
end

@inline function _kan_edge_value(layer, beta, feature, output, base, span, basis)
    no = layer.output_dim
    nb = size(layer.knots, 2)-4
    value = beta[first(layer.base_range)+(feature-1)*no+output-1] * base
    first_basis = max(span-3, 1)
    last_basis = min(span, nb)
    offset = first(layer.spline_range) + (feature-1)*nb*no + output-1
    for j in first_basis:last_basis
        value += beta[offset+(j-1)*no] * basis[j-span+4]
    end
    value
end

function _kan_layer_output(layer, beta, input)
    first_span, first_basis = _kan_cubic_support(layer.knots, 1, input[1])
    first_base = layer.activation(input[1])
    output = [_kan_edge_value(layer, beta, 1, o, first_base, first_span, first_basis)
              for o in 1:layer.output_dim]
    for i in 2:layer.input_dim
        span, basis = _kan_cubic_support(layer.knots, i, input[i])
        base = layer.activation(input[i])
        for o in eachindex(output)
            output[o] += _kan_edge_value(layer, beta, i, o, base, span, basis)
        end
    end
    output
end

@inline function _kan_evaluate_layers(layers::Tuple{L}, beta, input) where {L<:_KANSplineLayer}
    layer = first(layers)
    span, basis = _kan_cubic_support(layer.knots, 1, input[1])
    value = _kan_edge_value(layer, beta, 1, 1, layer.activation(input[1]), span, basis)
    for i in 2:layer.input_dim
        span, basis = _kan_cubic_support(layer.knots, i, input[i])
        value += _kan_edge_value(layer, beta, i, 1, layer.activation(input[i]), span, basis)
    end
    value
end

@inline function _kan_evaluate_layers(layers::Tuple, beta, input)
    hidden = _kan_layer_output(first(layers), beta, input)
    _kan_evaluate_layers(Base.tail(layers), beta, hidden)
end

@inline function _kan_normalize_inputs(a::KANApproximator, input::Tuple)
    if a.input_domains !== nothing
        lo, hi = a.logical_domain
        input = map((x, d) -> (x-d[1]) * ((hi-lo)/(d[2]-d[1])) + lo,
                    input, a.input_domains)
    end
    input
end

function (f::_KANEvaluator)(args::Vararg{Real})
    a, beta = f.approx, f.params
    length(args) == a.input_dim ||
        throw(DimensionMismatch("KANApproximator :$(a.name) expects $(a.input_dim) inputs, got $(length(args))"))
    input = _kan_normalize_inputs(a, promote(args...))
    _kan_evaluate_layers(a.layers, beta, input)
end

(f::_KANEvaluator)(input::AbstractVector{<:Real}) = f(input...)

function build_evaluator(a::KANApproximator, params_k)
    length(params_k) == nparams(a) ||
        throw(DimensionMismatch("KANApproximator :$(a.name) expects $(nparams(a)) parameters, got $(length(params_k))"))
    all(isfinite, params_k) ||
        throw(DomainError(params_k, "KANApproximator :$(a.name) requires finite parameters"))
    _KANEvaluator(a, params_k)
end
