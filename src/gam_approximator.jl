# ─── GAM.jl smooth approximator bridge (weak-dependency interface) ───
#
# Defines `GAMSmoothApproximator`, wrapping any `GAM.SmoothSpec` (e.g.,
# `GAM.s(:x, bs=:cc, k=10)`, `GAM.t2(:x, :y, k=(6,6))`, `GAM.s(:x, bs=:tp)`,
# `GAM.s(:x, bs=:ad)`, `GAM.s(:g, bs=:re)`) into PartiallySpecifiedModels'
# 4-function approximator protocol via `PartiallySpecifiedModelsGAMExt`.

"""
    GAMSmoothApproximator

Approximator backed by a `GAM.jl` constructed smooth (`GAM.ConstructedSmooth`),
unlocked when `GAM` is loaded (`using GAM`).

Supports any smooth basis `GAM.jl` constructs (`:cr`, `:cc`, `:tp`, `:ts`,
`:ps`, `:cps`, `:ad`, `:re`, `:mrf`, `te`, `t2`, `ti`, …), exposing its
multi-penalty structure via `penalty_blocks` (for disjoint or single-block
penalties) and `penalty_matrix`, and evaluating via `GAM.predict_matrix`.

# Construction (requires `using GAM`)
```julia
using GAM, PartiallySpecifiedModels
a = GAMSmoothApproximator(:r, s(:x; bs=:cc, k=10), (0.0, 12.0))
a2 = GAMSmoothApproximator(:g, t2(:x, :y; k=(6, 6)), ((0.0, 5.0), (0.0, 5.0)))
```
"""
struct GAMSmoothApproximator{CS, D} <: AbstractApproximator
    name::Symbol
    smooth::CS                 # GAM.ConstructedSmooth
    var_names::Vector{Symbol}  # covariate names expected by GAM.predict_matrix
    domain::D                  # Tuple{Float64,Float64} (1D) or Tuple of Tuples (kD)
    n_basis::Int
    penalties::Vector{Matrix{Float64}}
    initial_vec::Vector{Float64}
    centered::Bool             # whether sum-to-zero constraint was absorbed
end

# Hook implemented by PartiallySpecifiedModelsGAMExt
function _build_gam_smooth_approximator(name::Symbol, spec, domain; kwargs...)
    throw(ArgumentError(
        "GAMSmoothApproximator requires GAM.jl to be loaded (`using GAM`). " *
        "Load GAM before constructing GAMSmoothApproximator(:$name, ...)."))
end

function _gam_predict_row(smooth, var_names::Vector{Symbol}, x_tuple)
    throw(ArgumentError(
        "GAMSmoothApproximator evaluation requires GAM.jl to be loaded (`using GAM`)."))
end

function GAMSmoothApproximator(name::Union{Symbol,String}, spec, domain; kwargs...)
    _build_gam_smooth_approximator(Symbol(name), spec, domain; kwargs...)
end

nparams(a::GAMSmoothApproximator) = a.n_basis
initial_params(a::GAMSmoothApproximator) = copy(a.initial_vec)

function band_domain(a::GAMSmoothApproximator)
    a.domain isa Tuple{Float64,Float64} ? a.domain : nothing
end

function penalty_matrix(a::GAMSmoothApproximator)
    isempty(a.penalties) && return nothing
    S = zeros(Float64, a.n_basis, a.n_basis)
    for Sk in a.penalties
        S .+= Sk
    end
    0.5 .* (S .+ S')
end

function build_evaluator(a::GAMSmoothApproximator, params_k::AbstractVector)
    β = collect(params_k)
    sm = a.smooth
    vnames = a.var_names
    d = length(vnames)
    nm = a.name
    function gam_eval(xs...)
        length(xs) == d || throw(ArgumentError(
            "GAMSmoothApproximator(:$nm) expects $d argument(s), got $(length(xs))"))
        row = _gam_predict_row(sm, vnames, xs)
        acc = zero(promote_type(eltype(β), eltype(row)))
        @inbounds for j in 1:length(β)
            acc += row[j] * β[j]
        end
        acc
    end
    gam_eval
end
