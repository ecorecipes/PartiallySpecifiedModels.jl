function _kan_gauss_rule(n::Int)
    offdiag = [k / sqrt(4.0k^2 - 1) for k in 1:n-1]
    eig = eigen(SymTridiagonal(zeros(n), offdiag))
    eig.values, 2 .* eig.vectors[1, :].^2
end

const _KAN_GL16 = _kan_gauss_rule(16)
const _KAN_GL32 = _kan_gauss_rule(32)

function _kan_basis_derivative_map(knots, degree)
    n = length(knots) - degree - 1
    D = zeros(n + 1, n)
    for j in 1:n
        D[j, j] = degree / (knots[j+degree] - knots[j])
        D[j+1, j] = -degree / (knots[j+degree+1] - knots[j+1])
    end
    D
end

function _kan_gram_quadrature(design, lo, hi, rule)
    nodes, weights = rule
    halfspan = (hi-lo) / 2
    midpoint = lo + halfspan
    first_value = design(midpoint + halfspan * first(nodes))
    S = zeros(length(first_value), length(first_value))
    for (node, weight) in zip(nodes, weights)
        v = design(midpoint + halfspan * node)
        all(isfinite, v) ||
            throw(ArgumentError("KANApproximator: non-finite edge curvature on [$lo, $hi]"))
        S .+= (halfspan * weight) .* (v * v')
    end
    S
end

function _kan_adaptive_gram(design, lo, hi, atol, depth=0)
    coarse = _kan_gram_quadrature(design, lo, hi, _KAN_GL16)
    fine = _kan_gram_quadrature(design, lo, hi, _KAN_GL32)
    # Entrywise control prevents large spline terms from hiding errors in
    # the base-activation contribution or its cross terms.
    if all(abs.(fine-coarse) .<= atol .+ 1e-10 .* max.(abs.(fine), abs.(coarse)))
        return fine
    end
    depth < 16 ||
        error("KANApproximator: edge-curvature quadrature did not converge on [$lo, $hi]")
    mid = lo + (hi-lo) / 2
    lo < mid < hi ||
        error("KANApproximator: edge-curvature quadrature reached floating-point resolution")
    _kan_adaptive_gram(design, lo, mid, atol/2, depth+1) +
        _kan_adaptive_gram(design, mid, hi, atol/2, depth+1)
end

function _kan_edge_gram(layer::_KANSplineLayer, feature::Int, ridge::Float64)
    activation = layer.activation
    activation in (_kan_silu, tanh, identity) ||
        throw(ArgumentError("KANApproximator: edge_curvature supports SiLU, tanh, or identity " *
                            "base activations; use :ridge for other activations"))
    knots = view(layer.knots, feature, :)
    degree = layer.degree
    nb = length(knots) - degree - 1
    D1 = _kan_basis_derivative_map(knots, degree)
    D2 = _kan_basis_derivative_map(knots, degree-1) * D1
    design = x -> vcat(
        ForwardDiff.derivative(t -> ForwardDiff.derivative(activation, t), x),
        D2' * _bspline_basis_vector(x, knots, degree-1))
    lo, hi = layer.logical_domain
    span = hi-lo
    # Split at knots and at the characteristic scale of the supported
    # nonlinear activations. This also resolves their localized curvature
    # on unusually wide logical grids.
    breaks = sort!(unique(vcat(lo, hi,
        [x for x in knots if lo < x < hi],
        [x for x in (-32.0, -16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0, 32.0)
         if lo < x < hi])))
    S = zeros(nb+1, nb+1)
    for i in 1:length(breaks)-1
        a, b = breaks[i], breaks[i+1]
        S .+= _kan_adaptive_gram(design, a, b, 1e-12 * ((b-a)/span))
    end
    # z=(x-lo)/span: integral_0^1 (d^2 phi(lo+span*z)/dz^2)^2 dz.
    S .*= span^3

    if ridge > 0
        greville = [sum(knots[j+1:j+degree]) / degree for j in 1:nb]
        affine = hcat(ones(nb), (greville .- (lo+span/2)) ./ span)
        Q = Matrix(qr(affine).Q)[:, 1:2]
        S[2:end, 2:end] .+= ridge .* (Q * Q')
        # An identity base adds a third raw-coefficient null direction.
        activation === identity && (S[1, 1] += ridge)
    end
    Matrix(Symmetric((S + S') / 2))
end

function _kan_curvature_blocks(layers, ridge)
    blocks = Tuple{Matrix{Float64},UnitRange{Int}}[]
    for layer in layers
        ni, no = layer.input_dim, layer.output_dim
        nb = size(layer.knots, 2) - layer.degree - 1
        nbase = ni * no
        indices = first(layer.base_range):last(layer.spline_range)
        S = zeros(length(indices), length(indices))
        for feature in 1:ni
            edge = _kan_edge_gram(layer, feature, ridge)
            for output in 1:no
                # FluxKAN stores C as [output, basis, input]; edge
                # coefficients are strided, not contiguous in vec(C).
                coeff = nbase .+ output .+
                        no .* ((feature-1)*nb .+ (0:nb-1))
                edge_indices = vcat(output + (feature-1)*no, coeff)
                S[edge_indices, edge_indices] .+= edge
            end
        end
        push!(blocks, (S, indices))
    end
    blocks
end
