module PartiallySpecifiedModelsFluxKANExt

using PartiallySpecifiedModels
using FluxKAN

const PSM = PartiallySpecifiedModels

function PSM._kan_layer_info(layer::FluxKAN.LuxKANLinear)
    layer.in_dim > 0 && layer.out_dim > 0 && layer.grid_size > 0 ||
        throw(ArgumentError("KANApproximator: layer dimensions and grid_size must be positive"))
    layer.spline_order == 3 ||
        throw(ArgumentError("KANApproximator: only cubic LuxKANLinear layers (spline_order=3) are supported"))
    !layer.standalone_spline_scale ||
        throw(ArgumentError("KANApproximator: construct LuxKANLinear with standalone_spline_scale=false"))
    domain = (Float64(layer.grid_min), Float64(layer.grid_max))
    all(isfinite, domain) ||
        throw(ArgumentError("KANApproximator: backend grid limits must be finite"))
    PSM._validate_domain("KANApproximator backend grid", domain)
    (input_dim=layer.in_dim, output_dim=layer.out_dim, logical_domain=domain)
end

function PSM._kan_layer_spec(layer::FluxKAN.LuxKANLinear, ps, st, offset)
    keys(ps) == (:base_weight, :spline_weight) ||
        throw(ArgumentError("KANApproximator: unsupported LuxKANLinear parameter layout"))
    ni, no = layer.in_dim, layer.out_dim
    nb = layer.grid_size + layer.spline_order
    size(ps.base_weight) == (no, ni) &&
        size(ps.spline_weight) == (no, ni * nb) ||
        throw(DimensionMismatch("KANApproximator: unexpected LuxKANLinear weight dimensions"))
    knots = Matrix{Float64}(st.grid)
    size(knots) == (ni, layer.grid_size + 2layer.spline_order + 1) ||
        throw(DimensionMismatch("KANApproximator: unexpected LuxKANLinear grid dimensions"))
    all(isfinite, knots) && all(diff(knots; dims=2) .> 0) ||
        throw(ArgumentError("KANApproximator: spline knots must be finite and strictly increasing"))
    nbase = no * ni
    nspline = no * ni * nb
    activation = layer.base_activation === FluxKAN.SiLU ?
                 PSM._kan_silu : layer.base_activation
    PSM._KANSplineLayer(ni, no, layer.spline_order, knots,
        (Float64(layer.grid_min), Float64(layer.grid_max)),
        (offset+1):(offset+nbase), (offset+nbase+1):(offset+nbase+nspline),
        activation)
end

end
