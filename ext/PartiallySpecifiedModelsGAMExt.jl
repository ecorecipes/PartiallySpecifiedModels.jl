module PartiallySpecifiedModelsGAMExt

using PartiallySpecifiedModels
const PSM = PartiallySpecifiedModels
using GAM
using LinearAlgebra
using ForwardDiff

# ─── Helper: normalize domain to Vector{Tuple{Float64,Float64}} ──────
function _normalize_gam_domain(domain, d::Int, name::Symbol)
    if d == 1
        if domain isa Tuple{Real,Real}
            lo, hi = Float64(domain[1]), Float64(domain[2])
            (isfinite(lo) && isfinite(hi) && lo < hi) ||
                throw(ArgumentError("GAMSmoothApproximator(:$name): invalid domain $domain"))
            return [(lo, hi)], (lo, hi)
        elseif domain isa Tuple && length(domain) == 1 && domain[1] isa Tuple{Real,Real}
            lo, hi = Float64(domain[1][1]), Float64(domain[1][2])
            (isfinite(lo) && isfinite(hi) && lo < hi) ||
                throw(ArgumentError("GAMSmoothApproximator(:$name): invalid domain $domain"))
            return [(lo, hi)], (lo, hi)
        end
        throw(ArgumentError(
            "GAMSmoothApproximator(:$name): 1D smooth requires domain::Tuple{Real,Real}, got $(typeof(domain))"))
    else
        if domain isa Tuple && length(domain) == d && all(t -> t isa Tuple{Real,Real}, domain)
            vec_d = Tuple{Float64,Float64}[(Float64(t[1]), Float64(t[2])) for t in domain]
            for (lo, hi) in vec_d
                (isfinite(lo) && isfinite(hi) && lo < hi) ||
                    throw(ArgumentError("GAMSmoothApproximator(:$name): invalid domain $domain"))
            end
            return vec_d, Tuple(vec_d)
        end
        throw(ArgumentError(
            "GAMSmoothApproximator(:$name): $(d)D smooth requires domain as a $d-tuple of (lo, hi) tuples, got $(typeof(domain))"))
    end
end

function _build_grid_table(vnames::Vector{Symbol}, doms::Vector{Tuple{Float64,Float64}},
                           n_grid::Int)
    d = length(vnames)
    if d == 1
        lo, hi = doms[1]
        xs = collect(range(lo, hi; length=n_grid))
        return NamedTuple{(vnames[1],)}((xs,))
    else
        # Tensor grid with total points ≈ n_grid^d (capped per axis for memory safety)
        n_ax = max(12, min(n_grid, round(Int, 400^(1 / d))))
        axes_pts = [collect(range(lo, hi; length=n_ax)) for (lo, hi) in doms]
        n_tot = n_ax^d
        cols = Vector{Vector{Float64}}(undef, d)
        for j in 1:d
            inner = n_ax^(j - 1)
            outer = n_ax^(d - j)
            cols[j] = repeat(axes_pts[j]; inner=inner, outer=outer)
        end
        return NamedTuple{Tuple(vnames)}(Tuple(cols))
    end
end

# Wrap cyclic covariates into [lo, hi] so multi-period ODEs/maps work seamlessly.
_is_cyclic_basis(::GAM.CyclicCubic) = true
_is_cyclic_basis(::GAM.CyclicPSpline) = true
_is_cyclic_basis(_) = false

@inline function _wrap_period(x::Float64, lo::Float64, hi::Float64)
    p = hi - lo
    lo + mod(x - lo, p)
end

function PSM._build_gam_smooth_approximator(name::Symbol, spec::GAM.SmoothSpec, domain;
                                            include_intercept::Bool=true,
                                            n_grid::Int=100,
                                            initial=nothing,
                                            knots=nothing)
    vnames = copy(spec.term_vars)
    d = length(vnames)
    d >= 1 || throw(ArgumentError("GAMSmoothApproximator(:$name): SmoothSpec has no term_vars"))
    doms, dom_stored = _normalize_gam_domain(domain, d, name)
    grid_tbl = _build_grid_table(vnames, doms, max(n_grid, max(spec.k * 3, 40)))

    sm = GAM.smooth_construct(spec, grid_tbl, knots)
    k_sm = size(sm.X, 2)
    pens_raw = GAM.penalty_matrix(sm)
    pens = Matrix{Float64}[0.5 .* (Matrix{Float64}(S) .+ Matrix{Float64}(S)') for S in pens_raw]

    # Optionally prepend an unpenalized intercept so the smooth can represent
    # arbitrary constant levels (GAM.smooth_construct absorbs sum-to-zero).
    n_basis = include_intercept ? k_sm + 1 : k_sm
    pens_full = if include_intercept
        out = Vector{Matrix{Float64}}(undef, length(pens))
        for (i, S) in enumerate(pens)
            Sf = zeros(Float64, n_basis, n_basis)
            Sf[2:end, 2:end] .= S
            out[i] = Sf
        end
        out
    else
        pens
    end

    # Fit initial vector to `initial` if supplied (constant or callable).
    init_vec = zeros(Float64, n_basis)
    if initial !== nothing
        if initial isa Real
            if include_intercept
                init_vec[1] = Float64(initial)
            else
                X = sm.X
                init_vec .= (X' * X + 1e-8 * I) \ (X' * fill(Float64(initial), size(X, 1)))
            end
        elseif initial isa Function
            X_design = include_intercept ? hcat(ones(size(sm.X, 1)), sm.X) : sm.X
            cols = [getproperty(grid_tbl, v) for v in vnames]
            y_init = Float64[Float64(initial((c[i] for c in cols)...)) for i in 1:size(sm.X, 1)]
            init_vec .= (X_design' * X_design + 1e-8 * I) \ (X_design' * y_init)
        else
            throw(ArgumentError("GAMSmoothApproximator(:$name): initial must be nothing, Real, or Function"))
        end
    end

    PSM.GAMSmoothApproximator(name, sm, vnames, dom_stored, n_basis, pens_full,
                              init_vec, include_intercept)
end

# Extract disjoint contiguous penalty blocks when GAM.jl provides them (e.g.
# factor-by or orthogonalized t2), otherwise fall back to a single combined PSD
# block on the smooth coefficients (leaving any prepended intercept unpenalized).
function PSM.penalty_blocks(a::PSM.GAMSmoothApproximator{<:GAM.ConstructedSmooth})
    isempty(a.penalties) && return Tuple{Matrix{Float64}, UnitRange{Int}}[]
    start_idx = a.centered ? 2 : 1
    k_sm = a.n_basis - (start_idx - 1)
    k_sm < 1 && return Tuple{Matrix{Float64}, UnitRange{Int}}[]

    # Check if penalties have pairwise disjoint contiguous supports
    blocks = Tuple{Matrix{Float64}, UnitRange{Int}}[]
    used = falses(a.n_basis)
    disjoint_ok = true
    for S in a.penalties
        nz = [i for i in start_idx:a.n_basis if maximum(abs, @view S[i, :]) > 1e-15]
        if isempty(nz)
            continue
        end
        r = first(nz):last(nz)
        if length(r) != length(nz) || any(i -> used[i], r)
            disjoint_ok = false
            break
        end
        for i in r
            used[i] = true
        end
        Sb = Matrix{Float64}(S[r, r])
        push!(blocks, (0.5 .* (Sb .+ Sb'), r))
    end
    if disjoint_ok && !isempty(blocks)
        return blocks
    end

    # Overlapping multi-penalty smooth (e.g., te()): combine into single block
    # on start_idx:a.n_basis (intercept remains unpenalized).
    S_comb = zeros(Float64, k_sm, k_sm)
    r_sm = start_idx:a.n_basis
    for S in a.penalties
        S_comb .+= @view S[r_sm, r_sm]
    end
    S_comb = 0.5 .* (S_comb .+ S_comb')
    [(S_comb, r_sm)]
end

# Evaluate a single row of the GAM basis at `xs` (Float64), handling cyclic
# wrapping and prepending the intercept column when `a.centered == true`.
function _gam_eval_float_row(sm::GAM.ConstructedSmooth,
                             vnames::Vector{Symbol},
                             doms::Vector{Tuple{Float64,Float64}},
                             xs_f::NTuple{D, Float64},
                             centered::Bool) where {D}
    cyc = _is_cyclic_basis(sm.spec.basis)
    vals = ntuple(D) do j
        xj = xs_f[j]
        if cyc
            [_wrap_period(xj, doms[j][1], doms[j][2])]
        else
            [xj]
        end
    end
    tbl = NamedTuple{Tuple(vnames)}(vals)
    Xrow = GAM.predict_matrix(sm, tbl)
    k_sm = size(Xrow, 2)
    if centered
        out = Vector{Float64}(undef, k_sm + 1)
        out[1] = 1.0
        @inbounds for j in 1:k_sm
            out[j + 1] = Xrow[1, j]
        end
        return out
    else
        return vec(Xrow)
    end
end

function PSM._gam_predict_row(sm::GAM.ConstructedSmooth,
                              vnames::Vector{Symbol},
                              xs)
    # Kept for fallback compatibility; build_evaluator below uses the
    # Dual-aware evaluator with domain/intercept metadata directly.
    tbl = NamedTuple{Tuple(vnames)}(Tuple([Float64(ForwardDiff.value(x))] for x in xs))
    vec(GAM.predict_matrix(sm, tbl))
end

@inline _strip_dual_float(x::ForwardDiff.Dual) = _strip_dual_float(ForwardDiff.value(x))
@inline _strip_dual_float(x::Real) = Float64(x)

# Override build_evaluator for GAM-backed approximators to support intercept,
# cyclic domain wrapping, and ForwardDiff.Dual propagation in BOTH β and x
# (including when β and x share the same ForwardDiff tag during ODE sensitivity)!
function PSM.build_evaluator(a::PSM.GAMSmoothApproximator{<:GAM.ConstructedSmooth},
                             params_k::AbstractVector)
    β = collect(params_k)
    sm = a.smooth
    vnames = a.var_names
    d = length(vnames)
    doms = a.domain isa Tuple{Float64,Float64} ? [a.domain] : collect(a.domain)
    centered = a.centered
    nm = a.name

    function gam_eval(xs...)
        length(xs) == d || throw(ArgumentError(
            "GAMSmoothApproximator(:$nm) expects $d argument(s), got $(length(xs))"))
        has_dual_x = any(x -> x isa ForwardDiff.Dual, xs)
        if !has_dual_x
            xs_f = ntuple(j -> Float64(xs[j]), d)
            row = _gam_eval_float_row(sm, vnames, doms, xs_f, centered)
            return dot(row, β)
        else
            # Propagate ForwardDiff.Dual through input x via central differences
            # on the Float64 GAM basis row (enables ODE ForwardDiff, stiff Jacobians & linearise!).
            xs_f = ntuple(j -> _strip_dual_float(xs[j]), d)
            row0 = _gam_eval_float_row(sm, vnames, doms, xs_f, centered)
            val0 = dot(row0, β)
            proto = xs[findfirst(x -> x isa ForwardDiff.Dual, xs)]
            Tag = ForwardDiff.tagtype(typeof(proto))
            N = ForwardDiff.npartials(typeof(proto))
            same_tag = (val0 isa ForwardDiff.Dual) && (ForwardDiff.tagtype(typeof(val0)) === Tag)
            if same_tag
                # Chain rule within the SAME ForwardDiff pass (e.g. LAML jac=:forwarddiff):
                # d/dβ [b(x(β))' β] = b(x)' + (db/dx · β) * dx/dβ.
                v0_primal = ForwardDiff.value(val0)
                acc_partials = ntuple(p -> ForwardDiff.partials(val0)[p], N)
                for j in 1:d
                    xj = xs[j]
                    xj isa ForwardDiff.Dual || continue
                    h = max(1e-6 * (doms[j][2] - doms[j][1]), 1e-7)
                    xs_p = ntuple(k -> k == j ? xs_f[k] + h : xs_f[k], d)
                    xs_m = ntuple(k -> k == j ? xs_f[k] - h : xs_f[k], d)
                    df_dxj = ForwardDiff.value(
                        (dot(_gam_eval_float_row(sm, vnames, doms, xs_p, centered), β) -
                         dot(_gam_eval_float_row(sm, vnames, doms, xs_m, centered), β)) / (2.0 * h))
                    px = ForwardDiff.partials(xj)
                    acc_partials = ntuple(p -> acc_partials[p] + df_dxj * px[p], N)
                end
                return ForwardDiff.Dual{Tag}(v0_primal, ForwardDiff.Partials(acc_partials))
            else
                # Distinct tag (e.g. linearise w.r.t. state x, or nested stiff ODE Jacobian).
                acc_partials = ntuple(_ -> zero(val0), N)
                for j in 1:d
                    xj = xs[j]
                    xj isa ForwardDiff.Dual || continue
                    h = max(1e-6 * (doms[j][2] - doms[j][1]), 1e-7)
                    xs_p = ntuple(k -> k == j ? xs_f[k] + h : xs_f[k], d)
                    xs_m = ntuple(k -> k == j ? xs_f[k] - h : xs_f[k], d)
                    df_dxj = (dot(_gam_eval_float_row(sm, vnames, doms, xs_p, centered), β) -
                              dot(_gam_eval_float_row(sm, vnames, doms, xs_m, centered), β)) / (2.0 * h)
                    px = ForwardDiff.partials(xj)
                    acc_partials = ntuple(p -> acc_partials[p] + df_dxj * px[p], N)
                end
                return ForwardDiff.Dual{Tag}(val0, ForwardDiff.Partials(acc_partials))
            end
        end
    end
    gam_eval
end

end # module
