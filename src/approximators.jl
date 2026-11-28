# B-spline approximator: evaluation and penalty matrix
#
# Unconstrained BSplineApproximator uses DataInterpolations.CubicSpline for
# evaluation and the standard natural cubic spline penalty S = H'B⁻¹H.
#
# Shape-constrained approximators use proper B-spline basis evaluation via
# de Boor recursion. The SCOP-spline reparameterization (Pya & Wood 2015)
# constrains B-spline coefficients, and the convex hull property of B-splines
# guarantees shape constraints hold everywhere on the domain.

"""
    evaluate_bspline(knots_x, knots_y, x)

Evaluate a cubic spline interpolant at point `x`.
"""
function evaluate_bspline(knots_x::AbstractVector, knots_y::AbstractVector, x::Real)
    # Linear extrapolation: continuing the boundary CUBIC outside the domain
    # grows without bound and can destabilize the ODE solve whenever a state
    # wanders past the fitted range mid-integration.
    itp = CubicSpline(knots_y, knots_x; extrapolation=ExtrapolationType.Linear)
    return itp(x)
end

"""
    build_bspline_evaluator(knots_x, knots_y)

Build a callable cubic spline evaluator (caches the interpolation object).
"""
function build_bspline_evaluator(knots_x::AbstractVector, knots_y::AbstractVector)
    # Linear (slope-preserving) extrapolation outside the fitted domain —
    # see evaluate_bspline.
    CubicSpline(knots_y, knots_x; extrapolation=ExtrapolationType.Linear)
end

"""
    spline_penalty_matrix(knots_x)

Compute the natural cubic spline smoothing penalty matrix S such that
the wiggliness penalty is `y'Sy = ∫(f'')² dx`.

This is the standard result: S = H'B⁻¹H where H contains second divided
differences and B is the tridiagonal mass matrix from cubic spline theory.
The matrix has rank `nknots - 2`.

Reference: Green & Silverman (1994), Chapter 2.
"""
function spline_penalty_matrix(knots_x::AbstractVector)
    n = length(knots_x)
    if n < 3
        return zeros(n, n)
    end

    h = diff(knots_x)
    any(h .<= 0) && error("spline_penalty_matrix: knots must be strictly increasing")
    m = n - 2  # number of interior knots

    # H: m × n matrix of second divided differences
    H = zeros(m, n)
    for i in 1:m
        H[i, i]   =  1.0 / h[i]
        H[i, i+1] = -(1.0 / h[i] + 1.0 / h[i+1])
        H[i, i+2] =  1.0 / h[i+1]
    end

    # B: m × m tridiagonal mass matrix
    B = zeros(m, m)
    for i in 1:m
        B[i, i] = (h[i] + h[i+1]) / 3.0
    end
    for i in 1:m-1
        B[i, i+1] = h[i+1] / 6.0
        B[i+1, i] = h[i+1] / 6.0
    end

    # S = H'B⁻¹H
    return H' * (B \ H)
end

"""
    penalty_matrix(a::BSplineApproximator)

Return the smoothing penalty matrix for a B-spline approximator.
Computed on the unit interval for scale-invariant smoothing parameters.
"""
function penalty_matrix(a::BSplineApproximator)
    knots_unit = collect(range(0.0, 1.0, length=a.nknots))
    spline_penalty_matrix(knots_unit)
end

# Precompute the (n × n) linear map from knot values `v` to natural cubic
# spline second derivatives `z` on a uniform grid with spacing `h` (z₁ = zₙ = 0).
function _natural_cubic_z_matrix(n::Int, h::Float64)
    M = zeros(Float64, n, n)
    n <= 2 && return M
    m = n - 2
    T_mat = SymTridiagonal(fill(2.0 / 3.0, m), fill(1.0 / 6.0, m - 1))
    R_mat = zeros(Float64, m, n)
    inv_h2 = 1.0 / (h * h)
    @inbounds for i in 1:m
        R_mat[i, i]     =  inv_h2
        R_mat[i, i + 1] = -2.0 * inv_h2
        R_mat[i, i + 2] =  inv_h2
    end
    M[2:(n - 1), :] .= T_mat \ R_mat
    M
end

@inline function _eval_1d_natural_col(V::Matrix{T}, Z::Matrix{T}, j::Int,
                                      x, x0::Float64, x1::Float64,
                                      hx::Float64, inv_hx::Float64, nx::Int) where {T}
    @inbounds if x < x0
        slope = (V[2, j] - V[1, j]) * inv_hx - (hx / 6.0) * Z[2, j]
        return V[1, j] + slope * (x - x0)
    elseif x > x1
        slope = (V[nx, j] - V[nx - 1, j]) * inv_hx + (hx / 6.0) * Z[nx - 1, j]
        return V[nx, j] + slope * (x - x1)
    else
        xv = ForwardDiff.value(x)
        i = clamp(floor(Int, (xv - x0) * inv_hx) + 1, 1, nx - 1)
        b = (x - (x0 + (i - 1) * hx)) * inv_hx
        a = one(b) - b
        c = (hx * hx) / 6.0
        return a * V[i, j] + b * V[i + 1, j] +
               c * ((a * a * a - a) * Z[i, j] + (b * b * b - b) * Z[i + 1, j])
    end
end

struct _TensorNaturalSplineEval{T}
    C::Matrix{T}
    Z_x::Matrix{T}
    Z_y::Matrix{T}
    Z_xy::Matrix{T}
    x0::Float64
    x1::Float64
    hx::Float64
    inv_hx::Float64
    nx::Int
    y0::Float64
    y1::Float64
    hy::Float64
    inv_hy::Float64
    ny::Int
end

function (e::_TensorNaturalSplineEval)(x, y)
    C, Zx, Zy, Zxy = e.C, e.Z_x, e.Z_y, e.Z_xy
    x0, x1, hx, inv_hx, nx = e.x0, e.x1, e.hx, e.inv_hx, e.nx
    y0, y1, hy, inv_hy, ny = e.y0, e.y1, e.hy, e.inv_hy, e.ny
    if y < y0
        v1 = _eval_1d_natural_col(C, Zx, 1, x, x0, x1, hx, inv_hx, nx)
        v2 = _eval_1d_natural_col(C, Zx, 2, x, x0, x1, hx, inv_hx, nx)
        z2 = _eval_1d_natural_col(Zy, Zxy, 2, x, x0, x1, hx, inv_hx, nx)
        slope_y = (v2 - v1) * inv_hy - (hy / 6.0) * z2
        return v1 + slope_y * (y - y0)
    elseif y > y1
        vn1 = _eval_1d_natural_col(C, Zx, ny - 1, x, x0, x1, hx, inv_hx, nx)
        vn  = _eval_1d_natural_col(C, Zx, ny, x, x0, x1, hx, inv_hx, nx)
        zn1 = _eval_1d_natural_col(Zy, Zxy, ny - 1, x, x0, x1, hx, inv_hx, nx)
        slope_y = (vn - vn1) * inv_hy + (hy / 6.0) * zn1
        return vn + slope_y * (y - y1)
    else
        yv = ForwardDiff.value(y)
        j = clamp(floor(Int, (yv - y0) * inv_hy) + 1, 1, ny - 1)
        by = (y - (y0 + (j - 1) * hy)) * inv_hy
        ay = one(by) - by
        cy = (hy * hy) / 6.0
        vj  = _eval_1d_natural_col(C, Zx, j, x, x0, x1, hx, inv_hx, nx)
        vj1 = _eval_1d_natural_col(C, Zx, j + 1, x, x0, x1, hx, inv_hx, nx)
        zj  = _eval_1d_natural_col(Zy, Zxy, j, x, x0, x1, hx, inv_hx, nx)
        zj1 = _eval_1d_natural_col(Zy, Zxy, j + 1, x, x0, x1, hx, inv_hx, nx)
        return ay * vj + by * vj1 +
               cy * ((ay * ay * ay - ay) * zj + (by * by * by - by) * zj1)
    end
end

"""
    build_tensor_bspline_evaluator(a::TensorBSplineApproximator, params_k)

Build the two-argument callable `f(x, y)` for a tensor-product spline from
the column-major coefficient grid `params_k = vec(C)`, `C[i, j] = f(x_i, y_j)`.

Uses the exact separable bicubic natural-spline representation precomputed
once at build time (`Z_x = M_x C`, `Z_y = C M_y'`, `Z_xy = Z_x M_y'`), so each
`f(x, y)` call runs in O(1) time with zero heap allocations while agreeing
to machine precision with nested univariate `CubicSpline` interpolation and
linear extrapolation along both margins. Eltype-generic in both `params_k`
and `(x, y)`, so ForwardDiff Duals propagate through either.
"""
# ─── Periodic (cyclic) cubic spline evaluator and penalty ───────────

# Circulant second-derivative map M_cyc (n × n) for a periodic cubic spline
# on n uniform knots spaced by h: (1/6)z_{i-1} + (2/3)z_i + (1/6)z_{i+1}
# = (v_{i+1} - 2v_i + v_{i-1}) / h² with wrap-around indices mod n.
function _cyclic_cubic_z_matrix(n::Int, h::Float64)
    B = zeros(Float64, n, n)
    H = zeros(Float64, n, n)
    inv_h2 = 1.0 / (h * h)
    @inbounds for i in 1:n
        im1 = i == 1 ? n : i - 1
        ip1 = i == n ? 1 : i + 1
        B[i, im1] = 1.0 / 6.0
        B[i, i]   = 2.0 / 3.0
        B[i, ip1] = 1.0 / 6.0
        H[i, im1] =  inv_h2
        H[i, i]   = -2.0 * inv_h2
        H[i, ip1] =  inv_h2
    end
    B \ H
end

"""
    cyclic_spline_penalty_matrix(n::Int) -> Matrix{Float64}

Exact periodic cubic spline roughness penalty `S_cyc = ∫₀¹ (f''(u))² du` on `n`
uniformly spaced periodic knots over the unit period `[0, 1]`.

Symmetric PSD circulant matrix of rank `n - 1` with 1D constant null space
`ones(n)`: as `λ → ∞`, the periodic function shrinks smoothly to a constant.
"""
function cyclic_spline_penalty_matrix(n::Int)
    n >= 4 || throw(ArgumentError("cyclic_spline_penalty_matrix requires n ≥ 4, got $n"))
    h = 1.0 / n
    B = zeros(Float64, n, n)
    H = zeros(Float64, n, n)
    inv_h = 1.0 / h
    @inbounds for i in 1:n
        im1 = i == 1 ? n : i - 1
        ip1 = i == n ? 1 : i + 1
        B[i, im1] = h / 6.0
        B[i, i]   = 2.0 * h / 3.0
        B[i, ip1] = h / 6.0
        H[i, im1] =  inv_h
        H[i, i]   = -2.0 * inv_h
        H[i, ip1] =  inv_h
    end
    S = H' * (B \ H)
    0.5 .* (S .+ S')
end

penalty_matrix(a::CyclicBSplineApproximator) = cyclic_spline_penalty_matrix(a.nknots)

struct _CyclicCubicSplineEval{T}
    v::Vector{T}
    z::Vector{T}
    lo::Float64
    period::Float64
    h::Float64
    inv_h::Float64
    n::Int
end

function (e::_CyclicCubicSplineEval)(x)
    # Wrap x periodically into [0, period) while preserving Dual derivatives
    xv = ForwardDiff.value(x)
    shift = floor((xv - e.lo) / e.period) * e.period
    xw = x - e.lo - shift
    xwv = ForwardDiff.value(xw)
    k = clamp(floor(Int, xwv * e.inv_h) + 1, 1, e.n)
    kn = k == e.n ? 1 : k + 1
    b = (xw - (k - 1) * e.h) * e.inv_h
    a = one(b) - b
    c = (e.h * e.h) / 6.0
    @inbounds a * e.v[k] + b * e.v[kn] +
              c * ((a * a * a - a) * e.z[k] + (b * b * b - b) * e.z[kn])
end

function build_cyclic_bspline_evaluator(a::CyclicBSplineApproximator,
                                        params_k::AbstractVector)
    n = a.nknots
    lo, hi = Float64(a.domain[1]), Float64(a.domain[2])
    period = hi - lo
    h = period / n
    M = _cyclic_cubic_z_matrix(n, h)
    v = Vector(params_k)
    z = M * v
    _CyclicCubicSplineEval(v, z, lo, period, h, 1.0 / h, n)
end

# ─── Wood (2006) t2 orthogonal tensor-product decomposition ─────────
#
# For `TensorBSplineApproximator(...; anisotropy=:estimate)`, rotates the
# (nx × ny) grid coefficients via the orthogonal matrix Q = [U_y0 ⊗ U_x0,
# U_y0 ⊗ U_x+, U_y+ ⊗ U_x0, U_y+ ⊗ U_x+] into four mutually orthogonal,
# contiguous subspaces:
#   1. Bilinear null space (size 4): unpenalized
#   2. Smooth-x × affine-y (size 2*(nx-2)): own λ_x
#   3. Affine-x × smooth-y (size 2*(ny-2)): own λ_y
#   4. Pure interaction smooth (size (nx-2)*(ny-2)): own λ_xy
# Each penalized block is pairwise disjoint so LAML/GCV/CollocationLAML
# estimates all three smoothing parameters jointly with exact log|S_λ|₊!
function _tensor_t2_decomp(nx::Int, ny::Int)
    Sx = spline_penalty_matrix(collect(range(0.0, 1.0, length=nx)))
    Sy = spline_penalty_matrix(collect(range(0.0, 1.0, length=ny)))
    ex = eigen(Symmetric(Sx))
    ey = eigen(Symmetric(Sy))
    # Ascending eigenvalues: first 2 are null space, 3:n are strictly positive range space
    Ux0 = ex.vectors[:, 1:2]
    Uxp = ex.vectors[:, 3:nx]
    dxp = max.(ex.values[3:nx], 1e-12)
    Uy0 = ey.vectors[:, 1:2]
    Uyp = ey.vectors[:, 3:ny]
    dyp = max.(ey.values[3:ny], 1e-12)

    Q00 = kron(Uy0, Ux0)   # (nx*ny) × 4
    Q0x = kron(Uy0, Uxp)   # (nx*ny) × 2*(nx-2)
    Qy0 = kron(Uyp, Ux0)   # (nx*ny) × 2*(ny-2)
    Qxy = kron(Uyp, Uxp)   # (nx*ny) × (nx-2)*(ny-2)
    Q = hcat(Q00, Q0x, Qy0, Qxy)

    n0 = 4
    n1 = 2 * (nx - 2)
    n2 = 2 * (ny - 2)
    n3 = (nx - 2) * (ny - 2)

    S1 = Matrix{Float64}(Diagonal(repeat(dxp, 2)))
    S2 = Matrix{Float64}(Diagonal(repeat(dyp; inner=2)))
    # Interaction penalty: additive Kronecker sum of normalized positive eigenvalues
    dx_n = dxp ./ maximum(dxp)
    dy_n = dyp ./ maximum(dyp)
    d_int = Float64[dx_n[ix] + dy_n[iy] for iy in 1:(ny - 2) for ix in 1:(nx - 2)]
    S3 = Matrix{Float64}(Diagonal(d_int))

    r1 = (n0 + 1):(n0 + n1)
    r2 = (n0 + n1 + 1):(n0 + n1 + n2)
    r3 = (n0 + n1 + n2 + 1):(n0 + n1 + n2 + n3)
    blocks = Tuple{Matrix{Float64}, UnitRange{Int}}[(S1, r1), (S2, r2), (S3, r3)]
    Q, blocks
end

function build_tensor_bspline_evaluator(a::TensorBSplineApproximator,
                                        params_k::AbstractVector)
    nx, ny = a.nknots_x, a.nknots_y
    x0, x1 = Float64(a.domain_x[1]), Float64(a.domain_x[2])
    y0, y1 = Float64(a.domain_y[1]), Float64(a.domain_y[2])
    hx = (x1 - x0) / (nx - 1)
    hy = (y1 - y0) / (ny - 1)
    Mx = _natural_cubic_z_matrix(nx, hx)
    My = _natural_cubic_z_matrix(ny, hy)
    c_vec = if a.anisotropy === :estimate
        Q, _ = _tensor_t2_decomp(nx, ny)
        Q * params_k
    else
        params_k
    end
    C = Matrix(reshape(c_vec, nx, ny))
    Z_x = Mx * C
    Z_y = C * My'
    Z_xy = Z_x * My'
    _TensorNaturalSplineEval(C, Z_x, Z_y, Z_xy,
                             x0, x1, hx, 1.0 / hx, nx,
                             y0, y1, hy, 1.0 / hy, ny)
end

function penalty_blocks(a::TensorBSplineApproximator)
    if a.anisotropy === :estimate
        _, blocks = _tensor_t2_decomp(a.nknots_x, a.nknots_y)
        return blocks
    end
    S = penalty_matrix(a)
    S === nothing ? Tuple{Matrix{Float64}, UnitRange{Int}}[] : [(S, 1:nparams(a))]
end

function penalty_matrix(a::TensorBSplineApproximator)
    nx, ny = a.nknots_x, a.nknots_y
    if a.anisotropy === :estimate
        _, blocks = _tensor_t2_decomp(nx, ny)
        S = zeros(Float64, nx * ny, nx * ny)
        for (Sb, rb) in blocks
            S[rb, rb] .= Sb
        end
        return 0.5 .* (S .+ S')
    end
    Sx = spline_penalty_matrix(collect(range(0.0, 1.0, length=nx)))
    Sy = spline_penalty_matrix(collect(range(0.0, 1.0, length=ny)))
    S = kron(Matrix{Float64}(I, ny, ny), Sx) .+
        a.anisotropy .* kron(Sy, Matrix{Float64}(I, nx, nx))
    (S + S') ./ 2
end

"""
    penalty_matrix(a::NeuralApproximator)

Returns scaled identity matrix for L2 regularization when `penalty_weight > 0`,
enabling LAML smoothing parameter estimation. Returns `nothing` otherwise.
"""
function penalty_matrix(a::NeuralApproximator)
    if a.penalty_weight > 0.0
        n = Lux.parameterlength(a.model)
        return a.penalty_weight * Matrix{Float64}(I, n, n)
    end
    return nothing
end

# ─── Gaussian Process kernels and penalty ─────────────────────────

"""Squared exponential kernel: k(r) = σ² exp(-r²/(2ℓ²))"""
_kernel_sqexp(r, ℓ, σ²) = σ² * exp(-r^2 / (2 * ℓ^2))

"""Matérn 3/2 kernel: k(r) = σ²(1 + √3|r|/ℓ) exp(-√3|r|/ℓ)"""
function _kernel_matern32(r, ℓ, σ²)
    s = sqrt(3) * abs(r) / ℓ
    σ² * (1 + s) * exp(-s)
end

"""Matérn 5/2 kernel: k(r) = σ²(1 + √5|r|/ℓ + 5r²/(3ℓ²)) exp(-√5|r|/ℓ)"""
function _kernel_matern52(r, ℓ, σ²)
    s = sqrt(5) * abs(r) / ℓ
    σ² * (1 + s + s^2 / 3) * exp(-s)
end

"""Return a kernel function closure for the given type."""
function _kernel_func(kernel::Symbol, ℓ::Float64, σ²::Float64)
    if kernel == :sqexp
        (x1, x2) -> _kernel_sqexp(x1 - x2, ℓ, σ²)
    elseif kernel == :matern32
        (x1, x2) -> _kernel_matern32(x1 - x2, ℓ, σ²)
    elseif kernel == :matern52
        (x1, x2) -> _kernel_matern52(x1 - x2, ℓ, σ²)
    else
        error("Unknown kernel: $kernel. Use :sqexp, :matern32, or :matern52")
    end
end

"""Build kernel matrix K[i,j] = k(x_i, x_j)."""
function _build_kernel_matrix(kfunc, points::Vector{Float64})
    n = length(points)
    K = Matrix{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        K[i, j] = kfunc(points[i], points[j])
    end
    K
end

"""
    penalty_matrix(a::GPApproximator)

Returns a second-derivative (spline-style) penalty matrix on the inducing points.

While the theoretical GP penalty is `f'K⁻¹f` (negative log-prior), the K⁻¹
matrix has a narrow eigenvalue spectrum that makes LAML smoothing parameter
estimation unreliable — the penalty lacks the dynamic range to distinguish
smooth from wiggly solutions. Using a spline penalty `∫(f'')²dx` instead
gives LAML a well-conditioned penalty with wide eigenvalue range, while the
GP kernel is still used for evaluation (interpolation between inducing points).
"""
function penalty_matrix(a::GPApproximator)
    knots_unit = collect(range(0.0, 1.0, length=a.n_inducing))
    spline_penalty_matrix(knots_unit)
end

@inline _gp_kernel_val(::Val{:sqexp}, r, ℓ::Float64, σ²::Float64) = _kernel_sqexp(r, ℓ, σ²)
@inline _gp_kernel_val(::Val{:matern32}, r, ℓ::Float64, σ²::Float64) = _kernel_matern32(r, ℓ, σ²)
@inline _gp_kernel_val(::Val{:matern52}, r, ℓ::Float64, σ²::Float64) = _kernel_matern52(r, ℓ, σ²)

struct _GPMeanEval{K, W}
    x_ind::Vector{Float64}
    weights::Vector{W}
    ℓ::Float64
    σ²::Float64
    lo::Float64
    hi::Float64
    f_lo::W
    s_lo::W
    f_hi::W
    s_hi::W
    offset::W
end

@inline function _gp_raw_eval(::Val{K}, x_ind::Vector{Float64}, weights::Vector{W},
                              ℓ::Float64, σ²::Float64, xv) where {K, W}
    T = promote_type(typeof(xv), W)
    s = zero(T)
    @inbounds for j in eachindex(x_ind)
        s += _gp_kernel_val(Val(K), xv - x_ind[j], ℓ, σ²) * weights[j]
    end
    s
end

function (e::_GPMeanEval{K})(x) where {K}
    xv = x isa AbstractArray ? x[1] : x
    if xv < e.lo
        return (e.f_lo - e.offset) + e.s_lo * (xv - e.lo)
    elseif xv > e.hi
        return (e.f_hi - e.offset) + e.s_hi * (xv - e.hi)
    else
        return _gp_raw_eval(Val(K), e.x_ind, e.weights, e.ℓ, e.σ², xv) - e.offset
    end
end

"""
    _gp_mean_interpolant(a, values; offset=nothing)

Kernel predictive-mean interpolant through `values` at `a.inducing_points`:
  f(x) = k(x, X)' K⁻¹ values
Shared by `GPApproximator` and `ShapeConstrainedGPApproximator`. Concrete
callable struct with Val-dispatched kernel evaluation — zero allocations per
call and eltype-generic for ForwardDiff Duals.
"""
function _gp_mean_interpolant(a, values::AbstractVector; zero_at::Symbol=:none)
    a.kernel in (:sqexp, :matern32, :matern52) ||
        error("Unknown kernel: $(a.kernel). Use :sqexp, :matern32, or :matern52")
    weights = Vector(a.K_inv * values)
    x_ind = a.inducing_points
    ℓ = Float64(a.lengthscale)
    σ² = Float64(a.variance)
    lo, hi = extrema(x_ind)
    h = (hi - lo) * 1e-6
    kv = Val(a.kernel)
    f_lo = _gp_raw_eval(kv, x_ind, weights, ℓ, σ², lo)
    s_lo = (_gp_raw_eval(kv, x_ind, weights, ℓ, σ², lo + h) - f_lo) / h
    f_hi = _gp_raw_eval(kv, x_ind, weights, ℓ, σ², hi)
    s_hi = (f_hi - _gp_raw_eval(kv, x_ind, weights, ℓ, σ², hi - h)) / h
    off = if zero_at === :left
        _gp_raw_eval(kv, x_ind, weights, ℓ, σ², Float64(a.domain[1]))
    elseif zero_at === :right
        _gp_raw_eval(kv, x_ind, weights, ℓ, σ², Float64(a.domain[2]))
    else
        zero(f_lo)
    end
    _GPMeanEval{a.kernel, eltype(weights)}(
        x_ind, weights, ℓ, σ², lo, hi, f_lo, s_lo, f_hi, s_hi, off)
end

"""
    build_gp_evaluator(a::GPApproximator, params)

Build a callable that evaluates the GP predictive mean at any input x:
  f(x) = k(x, X)' K⁻¹ f_X
where f_X are the function values at inducing points (= params).
"""
build_gp_evaluator(a::GPApproximator, params::AbstractVector) =
    _gp_mean_interpolant(a, params)

"""
    _adapt_gp_hyperparams!(a, beta_k) -> Bool

Empirical-Bayes update of the kernel hyperparameters DURING a fit: choose
(ℓ, σ²) maximizing the GP log marginal likelihood of the current inducing
values `beta_k` (small fixed nugget), then rebuild `K`/`K_inv`. Called by
the LAML/GCV loops when `a.adapt` (no user-supplied lengthscale). Returns
whether the hyperparameters changed materially.

For `ShapeConstrainedGPApproximator` the caller passes the IMPLIED inducing
values `β = Σ·d(γ)` (see `gamma_to_inducing_values`), not the unconstrained
γ — the marginal likelihood is over function values, and a kernel change
does not move the values the interpolant passes through.
"""
function _adapt_gp_hyperparams!(a::Union{GPApproximator,
                                         ShapeConstrainedGPApproximator},
                                beta_k::AbstractVector)
    a.adapt || return false
    n = a.n_inducing
    v = var(beta_k)
    v > 1e-12 || return false          # flat function: nothing to adapt to
    x = a.inducing_points
    span = a.domain[2] - a.domain[1]
    nug = 1e-6 * v

    best_ll = -Inf
    best_ℓ = a.lengthscale
    best_σ² = a.variance
    # Score the INCUMBENT (current ℓ, σ²) first: with the strict `>` below,
    # "no change" wins ties and adaptation is monotone in the marginal
    # likelihood — a grid that never scored the incumbent could replace a
    # better off-grid setting with a worse grid point.
    cands = [(a.lengthscale, a.variance)]
    for frac in (0.08, 0.15, 0.25, 0.4, 0.6, 1.0), σm in (0.5, 1.0, 2.0)
        push!(cands, (frac * span, σm * v))
    end
    for (ℓ_try, σ²_try) in cands
        (ℓ_try > 0 && σ²_try > 0) || continue
        kf = _kernel_func(a.kernel, ℓ_try, σ²_try)
        K = _build_kernel_matrix(kf, x)
        F = cholesky(Symmetric(K + nug * I), check=false)
        issuccess(F) || continue
        α = F \ beta_k
        ll = -0.5 * dot(beta_k, α) - sum(log, diag(F.U))
        if ll > best_ll
            best_ll = ll; best_ℓ = ℓ_try; best_σ² = σ²_try
        end
    end
    isfinite(best_ll) || return false
    changed = abs(log(best_ℓ / a.lengthscale)) > 0.05 ||
              abs(log(best_σ² / max(a.variance, 1e-12))) > 0.05
    changed || return false

    a.lengthscale = best_ℓ
    a.variance = best_σ²
    a.K, a.K_inv = _gp_kernel_matrices(a.kernel, best_ℓ, best_σ², x)
    true
end

# ─── Shape-constrained GP: evaluator and penalty ──────────────────

"""
    gamma_to_inducing_values(a::ShapeConstrainedGPApproximator, gamma)

Transform unconstrained parameters γ to inducing-point values β = Σ * d,
where `d` passes the free (linear) components through unchanged and applies
`softplus` to the rest — the same `_apply_constraint_transform` used by the
B-spline and SPDE paths.
"""
gamma_to_inducing_values(a::ShapeConstrainedGPApproximator,
                         gamma::AbstractVector) =
    a.Sigma * _apply_constraint_transform(a.constraint, gamma)

"""
    build_constrained_gp_evaluator(a::ShapeConstrainedGPApproximator, gamma)

Build a callable evaluator from unconstrained parameters γ.

Applies the SCOP-spline reparameterization: inducing-point values are
computed as `β = Σ · _apply_constraint_transform(γ)` (free components
linear, the rest softplus'd), then interpolated with the GP predictive-mean
formula (`_gp_mean_interpolant`, shared with `GPApproximator`), including
its linear extrapolation outside the inducing range. Shape constraints are
enforced at the inducing values ONLY; the kernel interpolant between them
can violate the constraint materially (measured dip −0.505 against a max
of 5.6 on an all-positive-inducing-value `:positive` fixture — see the
`ShapeConstrainedGPApproximator` docstring and `check_constraints`).

For zero-at-endpoint constraints the interpolant is centered by subtracting
its value at the pinned endpoint — a constant shift that preserves
monotonicity exactly and pins f(endpoint) = 0 despite the jittered K⁻¹
(mirroring `build_constrained_bspline_evaluator`).
"""
function build_constrained_gp_evaluator(a::ShapeConstrainedGPApproximator,
                                        gamma::AbstractVector)
    beta = gamma_to_inducing_values(a, gamma)
    za = if a.constraint in (:inc_zero_left, :dec_zero_left)
        :left
    elseif a.constraint in (:inc_zero_right, :dec_zero_right)
        :right
    else
        :none
    end
    _gp_mean_interpolant(a, beta; zero_at=za)
end

"""
    penalty_matrix(a::ShapeConstrainedGPApproximator)

Pya & Wood (2015) SCOP first-difference penalty on the unconstrained γ,
exactly as `penalty_matrix(::ShapeConstrainedBSplineApproximator)`: the
free level (and the slope-like component for curvature constraints) lies in
the penalty null space, so λ→∞ shrinks toward a maximally smooth member of
the constraint family without biasing the level. The GP kernel is used for
evaluation only — using the prior precision K⁻¹ as the penalty would suffer
the same narrow-eigenvalue-spectrum LAML problem documented at
`penalty_matrix(::GPApproximator)`.
"""
penalty_matrix(a::ShapeConstrainedGPApproximator) =
    _scop_difference_penalty(a.constraint, nparams(a))

# ─── SPDE (Matérn) FEM matrices and evaluation ────────────────────

"""
    spde_fem_matrices(mesh_points)

Compute the 1D finite element matrices for the Matérn SPDE on a mesh.

Returns `(C, G)` where:
- `C` is the lumped (diagonal) mass matrix: `C[i,i] = ∫ φᵢ dx`
- `G` is the stiffness matrix: `G[i,j] = ∫ φᵢ' φⱼ' dx`

For piecewise linear hat functions on a mesh with spacings `h`.
"""
function spde_fem_matrices(mesh::AbstractVector)
    n = length(mesh)
    h = diff(mesh)
    any(h .<= 0) && error("spde_fem_matrices: mesh must be strictly increasing")

    # Lumped mass matrix (diagonal)
    C = zeros(n, n)
    C[1, 1] = h[1] / 2.0
    for i in 2:n-1
        C[i, i] = (h[i-1] + h[i]) / 2.0
    end
    C[n, n] = h[n-1] / 2.0

    # Stiffness matrix (tridiagonal)
    G = zeros(n, n)
    G[1, 1] = 1.0 / h[1]
    G[1, 2] = -1.0 / h[1]
    for i in 2:n-1
        G[i, i-1] = -1.0 / h[i-1]
        G[i, i]   = 1.0 / h[i-1] + 1.0 / h[i]
        G[i, i+1] = -1.0 / h[i]
    end
    G[n, n-1] = -1.0 / h[n-1]
    G[n, n]   = 1.0 / h[n-1]

    return C, G
end

"""
    spde_penalty_matrix(a::SPDEApproximator)

Compute the Matérn SPDE penalty matrix.

For ν = 0.5 (α = 1): `P = κ² C + G`
For ν = 1.5 (α = 2): `P = κ⁴ C + 2κ² G + G₂` where `G₂ = G C⁻¹ G`
For ν = 2.5 (α = 3): `P = κ⁶ C + 3κ⁴ G + 3κ² G₂ + G₃`
  where `G₂ = G C⁻¹ G` and `G₃ = G C⁻¹ G₂`
"""
function spde_penalty_matrix(a::SPDEApproximator)
    C, G = spde_fem_matrices(a.mesh_points)
    κ = a.kappa

    # C is diagonal (lumped mass), so C⁻¹ is just element-wise reciprocal
    C_inv_diag = 1.0 ./ diag(C)
    C_inv = Diagonal(C_inv_diag)

    if a.nu ≈ 0.5
        # α = 1: P = κ² C + G
        return κ^2 * C + G
    elseif a.nu ≈ 1.5
        # α = 2: P = κ⁴ C + 2κ² G + G₂
        G2 = G * C_inv * G
        return κ^4 * C + 2.0 * κ^2 * G + G2
    elseif a.nu ≈ 2.5
        # α = 3: P = κ⁶ C + 3κ⁴ G + 3κ² G₂ + G₃
        G2 = G * C_inv * G
        G3 = G * C_inv * G2
        return κ^6 * C + 3.0 * κ^4 * G + 3.0 * κ^2 * G2 + G3
    else
        error("Unsupported nu=$(a.nu), must be 0.5, 1.5, or 2.5")
    end
end

"""
    penalty_matrix(a::SPDEApproximator)

Return the Matérn SPDE penalty matrix.
"""
function penalty_matrix(a::SPDEApproximator)
    spde_penalty_matrix(a)
end

"""
    build_spde_evaluator(mesh_x, params)

Build a callable cubic spline evaluator for the SPDE mesh node values.
Uses cubic spline interpolation for smooth ODE-compatible evaluation.
"""
function build_spde_evaluator(mesh_x::AbstractVector, params::AbstractVector)
    CubicSpline(params, mesh_x; extrapolation=ExtrapolationType.Linear)
end

# ─── Shape-constrained SPDE: evaluator and penalty ────────────────

"""
    build_constrained_spde_evaluator(a::ShapeConstrainedSPDEApproximator, gamma)

Build a callable evaluator from unconstrained parameters γ.

Applies the SCOP-spline reparameterization: mesh node values are computed
as `β = Σ · _apply_constraint_transform(γ)` (free components linear, the
rest softplus'd), then interpolated with a cubic spline.
Shape constraints are enforced at mesh nodes ONLY; the cubic interpolant
between them can violate the constraint (measured dip −0.121 on an
all-positive-node `:positive` fixture — see the
`ShapeConstrainedSPDEApproximator` docstring and `check_constraints`).
"""
function build_constrained_spde_evaluator(a::ShapeConstrainedSPDEApproximator,
                                          gamma::AbstractVector)
    mesh_values = gamma_to_mesh_values(a, gamma)
    # Linear extrapolation also preserves the boundary monotonicity of the
    # constrained fit, which the boundary cubic could invert just outside
    # the domain.
    CubicSpline(mesh_values, a.mesh_points; extrapolation=ExtrapolationType.Linear)
end

"""
    gamma_to_mesh_values(a::ShapeConstrainedSPDEApproximator, gamma)

Transform unconstrained parameters γ to mesh node values β = Σ * d, where
`d` passes the free (linear) components through unchanged and applies
`softplus` to the rest — the same `_apply_constraint_transform` used by the
B-spline path. (The old softplus-everything version silently destroyed the
free intercept/slope of `:convex`/`:concave`, degenerating them to
increasing-convex-positive.)
"""
function gamma_to_mesh_values(a::ShapeConstrainedSPDEApproximator,
                              gamma::AbstractVector)
    a.Sigma * _apply_constraint_transform(a.constraint, gamma)
end

"""
    penalty_matrix(a::ShapeConstrainedSPDEApproximator)

Penalty matrix in the unconstrained parameter space (γ). Two modes,
selected by the approximator's `penalty` field:

- `:gamma_matern` (default): the Matérn FEM precision `P_β` is pushed
  through the LINEAR map Σ as `P_γ = Σᵀ P_β Σ`, ignoring the nonlinear
  softplus in β = Σ·d(γ). Since `P_β` is SPD and Σ has full column rank,
  `P_γ` is SPD — it has NO null space. Consequently λ→∞ drives γ → 0
  entirely, whose image under the constraint map is a fixed ramp with
  per-node increment `softplus(0) = log 2` (for monotone constraints) —
  an arbitrary function, not a maximally smooth member of the constraint
  family — and the free level γ₁ is shrunk along with everything else.
  Kept as the default for backward compatibility.

- `:difference`: the Pya & Wood (2015) SCOP first-difference penalty
  `DᵀD` built directly on γ, mirroring the
  `penalty_matrix(::ShapeConstrainedBSplineApproximator)` construction: the
  chain of first differences runs over the curvature-carrying components
  only, skipping the free level (and the slope-like component for
  curvature constraints) per `_penalty_skip_indices`. The null space then
  contains the free level/slope shifts and the constant-increment vector,
  so the λ→∞ limit is a maximally smooth member of the constraint family
  (e.g. the best-fitting straight line for `:increasing`), with the level
  unshrunk. The SCSPDE mesh is always uniform (built with `range`), so
  the plain unweighted difference matrix is the natural choice — no mesh
  weighting is needed. Note this mode does not use the Matérn κ/ρ
  structure; smoothness is controlled by λ alone.

Prefer `:difference` when you want the classical penalized-spline λ→∞
semantics (shrinkage toward the smooth null family); keep `:gamma_matern`
if you want Matérn-correlated shrinkage of γ toward zero and accept its
fixed-ramp limit.
"""
function penalty_matrix(a::ShapeConstrainedSPDEApproximator)
    if a.penalty == :difference
        # P&W SCOP difference penalty on γ, exactly as in the SCBSpline path.
        return _scop_difference_penalty(a.constraint, nparams(a))
    end

    # :gamma_matern — build the SPDE penalty in mesh-value space
    C, G = spde_fem_matrices(a.mesh_points)
    κ = a.kappa
    C_inv = Diagonal(1.0 ./ diag(C))

    P_beta = if a.nu ≈ 0.5
        κ^2 * C + G
    elseif a.nu ≈ 1.5
        G2 = G * C_inv * G
        κ^4 * C + 2.0 * κ^2 * G + G2
    elseif a.nu ≈ 2.5
        G2 = G * C_inv * G
        G3 = G * C_inv * G2
        κ^6 * C + 3.0 * κ^4 * G + 3.0 * κ^2 * G2 + G3
    else
        error("Unsupported nu=$(a.nu)")
    end

    # Transform penalty to unconstrained parameter space: Σᵀ P Σ
    Sig = a.Sigma
    P_gamma = Matrix(Sig' * P_beta * Sig)
    # Symmetrize to eliminate floating-point asymmetry
    (P_gamma + P_gamma') / 2
end

# ─── Shape-constrained B-spline: Sigma matrices and evaluation ────

"""Softplus function: log(1 + exp(x)), numerically stable."""
# Type-generic and Dual-safe: for large x, softplus(x) = x + log1p(exp(-x))
# is exact and avoids the old Float64(x) cast that broke ForwardDiff.
_softplus(x::Real) = x > 20.0 ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    _build_sigma_matrix(constraint, nknots) -> Matrix{Float64}

Build the Σ constraint matrix mapping the transformed coefficient vector
`d = _apply_constraint_transform(constraint, γ)` — free (linear) components
pass through, the rest are softplus'd nonnegative — to B-spline coefficients
β = Σ * d satisfying the given shape constraint. Monotone and combined
constraints carry a free level d₁ = γ₁ per Pya & Wood (2015) Table 1.

For most constraints, Σ is q × q (square). For zero-at-endpoint constraints,
Σ is q × (q-1) since one knot value is fixed at 0.

Following Pya & Wood (2015) SCOP-spline reparameterization.
"""
function _build_sigma_matrix(constraint::Symbol, q::Int)
    if constraint == :increasing
        # Lower triangular of 1's: β_j = ν₁ + ν₂ + ... + νⱼ (cumulative sum)
        Sig = zeros(q, q)
        for i in 1:q, j in 1:i
            Sig[i, j] = 1.0
        end
    elseif constraint == :decreasing
        # β_j = ν₁ - ν₂ - ... - νⱼ (cumulative sum, negated from col 2)
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
            for j in 2:i
                Sig[i, j] = -1.0
            end
        end
    elseif constraint == :convex
        # Convex (NOT necessarily monotone): free intercept β₁ and free
        # initial slope, with NONNEGATIVE second differences Δ²β_j ≥ 0.
        # β_i = d₁ + (i-1)·d₂ + Σ_{k=3}^{i} (i-k+1)·d_k, where d₁=γ₁ and
        # d₂=γ₂ are unconstrained (see `_linear_param_indices`) and
        # d_k=softplus(γ_k) ≥ 0 for k≥3 are the second differences.
        # This admits U-shaped (decreasing-then-increasing) functions, which
        # the previous parameterization (identical to :inc_convex) could not.
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0          # intercept (free)
            Sig[i, 2] = Float64(i - 1)  # linear ramp (free slope)
        end
        for k in 3:q, i in k:q
            Sig[i, k] = Float64(i - k + 1)
        end
    elseif constraint == :concave
        # Concave (NOT necessarily monotone): free intercept and slope with
        # NONPOSITIVE second differences. Same construction as :convex with
        # the curvature columns negated; admits ∩-shaped functions.
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
            Sig[i, 2] = Float64(i - 1)
        end
        for k in 3:q, i in k:q
            Sig[i, k] = -Float64(i - k + 1)
        end
    elseif constraint == :inc_convex
        # Monotone increasing + convex, Pya & Wood (2015) Table 1: free level
        # β₁ = γ₁ (first column of ones), increments cumulate the nonnegative
        # ν's so Δβᵢ = Σ_{k≤i+1} νₖ grows: βᵢ = γ₁ + Σ_{k=2}^{i} (i-k+1)νₖ.
        # (The old ramp first column forced β ≥ 0 and tied level to slope,
        # making e.g. f ≈ 5 + 0.01x unrepresentable.)
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
        end
        for k in 2:q, i in k:q
            Sig[i, k] = Float64(i - k + 1)
        end
    elseif constraint == :inc_concave
        # Monotone increasing + concave: free level β₁ = γ₁; increments
        # δᵢ = Σ_{k≥i+1} νₖ are nonnegative and decreasing:
        # βᵢ = γ₁ + Σ_{k=2}^{q} min(k-1, i-1)·νₖ.
        # NOTE: this is Pya & Wood Table 1's matrix with the ν-indices
        # reversed (they use min(i-1, q-j+1)); the spanned cone is identical
        # and Σ remains invertible, but the slope-like parameter is ν_q here
        # rather than ν₂ (see _penalty_skip_indices).
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
        end
        for k in 2:q, i in 1:q
            Sig[i, k] = Float64(min(k - 1, i - 1))
        end
    elseif constraint == :dec_convex
        # Monotone decreasing + convex: free level, increments
        # δᵢ = -Σ_{k≥i+1} νₖ ≤ 0 rising toward 0 (convex).
        # (The old all-negative Σ forced β ≤ 0 everywhere: a positive
        # declining rate like 1/x was unrepresentable.)
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
        end
        for k in 2:q, i in 1:q
            Sig[i, k] = -Float64(min(k - 1, i - 1))
        end
    elseif constraint == :dec_concave
        # Monotone decreasing + concave: free level, increasingly negative
        # increments: βᵢ = γ₁ - Σ_{k=2}^{i} (i-k+1)νₖ.
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = 1.0
        end
        for k in 2:q, i in k:q
            Sig[i, k] = -Float64(i - k + 1)
        end
    elseif constraint == :positive
        # Identity: β_j = softplus(γ_j) directly — all knot values positive
        Sig = Matrix{Float64}(I, q, q)
    elseif constraint == :dec_positive
        # Upper triangular of 1's: β_j = ν_j + ν_{j+1} + ... + ν_q
        # Decreasing (since later terms have fewer summed positives) and
        # always positive (β_q = ν_q > 0, β_j ≥ ν_q > 0)
        Sig = zeros(q, q)
        for i in 1:q, j in i:q
            Sig[i, j] = 1.0
        end

    # ── Zero-at-endpoint constraints (Σ is q × (q-1)) ──────────
    elseif constraint == :inc_zero_left
        # Increasing, f(x_min) = 0: β₁ = 0, β_j = Σ_{k=1}^{j-1} νₖ
        np = q - 1
        Sig = zeros(q, np)
        for i in 2:q, j in 1:(i-1)
            Sig[i, j] = 1.0
        end
    elseif constraint == :dec_zero_right
        # Decreasing, f(x_max) = 0: β_j = Σ_{k=j}^{q-1} νₖ, β_q = 0
        np = q - 1
        Sig = zeros(q, np)
        for i in 1:(q-1), j in i:np
            Sig[i, j] = 1.0
        end
    elseif constraint == :inc_zero_right
        # Increasing, f(x_max) = 0: β_j ≤ 0 increasing to 0
        # β_q = 0, β_j = -Σ_{k=j}^{q-1} νₖ
        np = q - 1
        Sig = zeros(q, np)
        for i in 1:(q-1), j in i:np
            Sig[i, j] = -1.0
        end
    elseif constraint == :dec_zero_left
        # Decreasing, f(x_min) = 0: β₁ = 0, β_j ≤ 0 decreasing
        # β_j = -Σ_{k=1}^{j-1} νₖ
        np = q - 1
        Sig = zeros(q, np)
        for i in 2:q, j in 1:(i-1)
            Sig[i, j] = -1.0
        end
    else
        error("Unknown constraint: $constraint")
    end
    return Sig
end

"""
    _linear_param_indices(constraint) -> Tuple

Indices of the γ components that enter the reparameterization *linearly*
(unconstrained), as opposed to through `softplus` (nonnegative). For most
constraints this is empty (every component is nonnegative). For `:convex`
and `:concave` the first two components are the free intercept and slope,
so only the remaining components (the second differences) are nonnegative.
"""
_linear_param_indices(constraint::Symbol) =
    constraint in (:convex, :concave)               ? (1, 2) :
    constraint in (:increasing, :decreasing,
                   :inc_convex, :inc_concave,
                   :dec_convex, :dec_concave)       ? (1,)   : ()

"""
    _apply_constraint_transform(constraint, gamma) -> Vector

Map γ to the nonnegative/linear coefficient vector `d` that `Σ` multiplies:
`d[i] = γ[i]` for the linear indices, `softplus(γ[i])` otherwise.
"""
function _apply_constraint_transform(constraint::Symbol, gamma::AbstractVector)
    lin = _linear_param_indices(constraint)
    [i in lin ? gamma[i] : _softplus(gamma[i]) for i in eachindex(gamma)]
end

"""
    gamma_to_knot_values(a::ShapeConstrainedBSplineApproximator, gamma)

Transform unconstrained parameters γ to B-spline coefficients β = Σ * d,
where `d` applies `softplus` to the nonnegative components and the identity
to the free (linear) components — see [`_apply_constraint_transform`](@ref).
These are de Boor control point values, not interpolation knot values.
"""
function gamma_to_knot_values(a::ShapeConstrainedBSplineApproximator,
                              gamma::AbstractVector)
    a.Sigma * _apply_constraint_transform(a.constraint, gamma)
end

"""
    _bspline_basis_vector(x, knots, order)

Evaluate all B-spline basis functions at a single point `x` using the de Boor
recursion. Returns a vector of length `length(knots) - order`.

This is the standard Cox-de Boor recursion for B-splines of given `order`
(order 4 = cubic). The `knots` vector includes boundary padding.
"""
function _bspline_basis_vector(x::Real, knots::AbstractVector, order::Int)
    nk = length(knots)
    n_basis = nk - order
    # Buffers take the evaluation point's type so ForwardDiff Duals propagate
    # (stiff ODE solvers with autodiff Jacobians evaluate splines at Dual x).
    T = float(typeof(x))

    # Order 1: piecewise constant
    b = zeros(T, nk - 1)
    for j in 1:(nk - 1)
        if j == nk - 1
            b[j] = (knots[j] <= x <= knots[j + 1]) ? one(T) : zero(T)
        else
            b[j] = (knots[j] <= x < knots[j + 1]) ? one(T) : zero(T)
        end
    end

    # Recursion for higher orders
    for p in 2:order
        b_new = zeros(T, nk - p)
        for j in 1:(nk - p)
            d1 = knots[j + p - 1] - knots[j]
            d2 = knots[j + p] - knots[j + 1]
            t1 = d1 > 0 ? (x - knots[j]) / d1 * b[j] : zero(T)
            t2 = d2 > 0 ? (knots[j + p] - x) / d2 * b[j + 1] : zero(T)
            b_new[j] = t1 + t2
        end
        b = b_new
    end
    return b[1:n_basis]
end

"""
    _scam_knot_vector(domain, q; m=2)

Build a B-spline knot vector for `q` basis functions with penalty order `m`
(default 2 = cubic spline, order m+2=4). Includes boundary padding following
the scam convention.
"""
function _scam_knot_vector(domain::Tuple{Float64, Float64}, q::Int; m::Int=2)
    nk = q + m + 2   # total knots
    lo, hi = domain
    n_interior = q - m
    interior = collect(range(lo, hi; length=n_interior))
    dx = interior[2] - interior[1]

    xk = zeros(nk)
    xk[(m + 2):(q + 1)] .= interior
    for i in 1:(m + 1)
        xk[i] = xk[m + 2] - (m + 2 - i) * dx
    end
    for i in (q + 2):(q + m + 2)
        xk[i] = xk[q + 1] + (i - q - 1) * dx
    end
    return xk
end

struct _SCBSplineEval{T}
    beta::Vector{T}
    ll::Float64
    ul::Float64
    dx::Float64
    inv_dx::Float64
    q::Int
    f_ll::T
    f_ul::T
    slope_lo::T
    slope_hi::T
    offset::T
end

@inline function _eval_uniform_cubic_bspline(x, ll::Float64, dx::Float64,
                                             inv_dx::Float64, q::Int,
                                             beta::Vector{T}) where {T}
    xv = ForwardDiff.value(x)
    k = clamp(floor(Int, (xv - ll) * inv_dx) + 1, 1, q - 3)
    u = (x - (ll + (k - 1) * dx)) * inv_dx
    om = one(u) - u
    u2 = u * u
    u3 = u2 * u
    b0 = (om * om * om) / 6
    b1 = (3 * u3 - 6 * u2 + 4) / 6
    b2 = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6
    b3 = u3 / 6
    @inbounds b0 * beta[k] + b1 * beta[k + 1] + b2 * beta[k + 2] + b3 * beta[k + 3]
end

function (e::_SCBSplineEval)(x::Real)
    if x < e.ll
        return (e.f_ll - e.offset) + e.slope_lo * (x - e.ll)
    elseif x > e.ul
        return (e.f_ul - e.offset) + e.slope_hi * (x - e.ul)
    else
        return _eval_uniform_cubic_bspline(x, e.ll, e.dx, e.inv_dx, e.q, e.beta) - e.offset
    end
end

"""
    build_constrained_bspline_evaluator(a, gamma)

Build a zero-allocation callable evaluator from unconstrained parameters γ
using exact local uniform cubic B-spline evaluation (4 active basis terms).

Uses the SCOP-spline approach (Pya & Wood 2015): the constrained coefficients
β = Σ · _apply_constraint_transform(γ) are B-spline basis coefficients, and
the function is evaluated as f(x) = Σⱼ βⱼ Bⱼ(x). The convex hull property of
B-splines guarantees that shape constraints (monotonicity, convexity,
positivity) hold everywhere on the domain, not just at knot points. For
zero-at-endpoint constraints the spline is centered by subtracting its
boundary value, pinning f(endpoint) = 0 exactly.
"""
function build_constrained_bspline_evaluator(a::ShapeConstrainedBSplineApproximator,
                                             gamma::AbstractVector)
    beta = Vector(gamma_to_knot_values(a, gamma))
    q = a.nknots
    ll = Float64(a.domain[1])
    ul = Float64(a.domain[2])
    dx = (ul - ll) / (q - 3)
    inv_dx = 1.0 / dx

    # Exact cubic B-spline boundary values and derivatives at uniform knots
    f_ll = (beta[1] + 4 * beta[2] + beta[3]) / 6
    f_ul = (beta[q - 2] + 4 * beta[q - 1] + beta[q]) / 6
    slope_lo = 0.5 * (beta[3] - beta[1]) * inv_dx
    slope_hi = 0.5 * (beta[q] - beta[q - 2]) * inv_dx

    offset = if a.constraint in (:inc_zero_left, :dec_zero_left)
        f_ll
    elseif a.constraint in (:inc_zero_right, :dec_zero_right)
        f_ul
    else
        zero(f_ll)
    end

    _SCBSplineEval(beta, ll, ul, dx, inv_dx, q, f_ll, f_ul, slope_lo, slope_hi, offset)
end

"""
    _penalty_skip_indices(constraint, np) -> Tuple

Indices excluded from the first-difference penalty chain, following
Pya & Wood (2015). The free level (γ₁) is always excluded; constraints
involving curvature also exclude their slope-like parameter, so the λ→∞
limit is an arbitrary member of the constraint's polynomial null family
(free level + free slope + constant curvature) rather than a family with
slope tied to curvature. For the reversed parameterizations
(`:inc_concave`/`:dec_convex`, where increments cumulate from the right)
the slope-like parameter is the LAST component, not the second.
"""
_penalty_skip_indices(constraint::Symbol, np::Int) =
    constraint in (:convex, :concave, :inc_convex, :dec_concave) ? (1, 2)  :
    constraint in (:inc_concave, :dec_convex)                    ? (1, np) :
    constraint in (:increasing, :decreasing)                     ? (1,)    :
    constraint === :dec_positive                                 ? (np,)   : ()

"""
    _scop_difference_penalty(constraint, np) -> Matrix{Float64}

Pya & Wood (2015) SCOP first-difference penalty `DᵀD` on the unconstrained
γ: the chain of first differences runs over the curvature-carrying
components only, per `_penalty_skip_indices` — their D starts the
differences from β̃₂ for monotone smooths and from β̃₃ for
curvature-constrained smooths. Free level/slope components live in the
penalty null space — shrinking them with λ would bias the function's level,
not its wiggliness. Shared by the shape-constrained B-spline, SPDE
(`:difference` mode), and GP penalty matrices.
"""
function _scop_difference_penalty(constraint::Symbol, np::Int)
    skip = _penalty_skip_indices(constraint, np)
    idxs = [i for i in 1:np if !(i in skip)]
    n_c = length(idxs)
    D = zeros(max(n_c - 1, 0), np)
    for r in 1:(n_c - 1)
        D[r, idxs[r]]   = -1.0
        D[r, idxs[r+1]] =  1.0
    end
    D' * D
end

penalty_matrix(a::ShapeConstrainedBSplineApproximator) =
    _scop_difference_penalty(a.constraint, nparams(a))


# ═══════════════════════════════════════════════════════════════════════
# COMONet: shape-constrained neural network evaluator
# ═══════════════════════════════════════════════════════════════════════
#
# COMONet guarantees shape constraints architecturally. The architecture
# is chosen PER CONSTRAINT CLASS so that each advertised constraint is the
# network's actual function class (positive weights composed with a convex
# nondecreasing activation are always convex — so a ReLU network cannot be
# "monotone but not convex", and a single positive-weight branch cannot be
# "convex but not monotone"):
#   - Pure monotone (:increasing/:decreasing): exp(W̃)/fanin weights + tanh
#     hidden units. tanh is nondecreasing but saturating (neither convex
#     nor concave), so the class is all monotone shapes — including
#     sigmoids and Holling type-II/III responses.
#   - Curvature-and-monotone (:inc_convex, :inc_concave, :dec_convex,
#     :dec_concave): positive weights + ReLU/softplus (or their negated
#     concave forms); decreasing variants negate the input.
#   - Curvature only (:convex/:concave): two-branch input-convex form
#     g₁(x) + g₂(−x) with both branches positive-weight convex-increasing
#     nets — the sum is convex and may be non-monotone (U-shapes);
#     :concave negates the sum.
#   - :positive: an UNCONSTRAINED tanh MLP with exp output — positivity
#     comes from exp alone, so humps are representable.
#
# Positive weights are exp(W̃)/fanin: the fan-in scaling makes W̃ = 0 a
# unit-gain network, so the L2 penalty on W̃ shrinks toward a tame function
# (raw exp(W̃) weights ≈ 1 amplify the signal by ∏ layer widths, and
# penalizing W̃ → 0 pushed toward that amplifier — backwards for smoothing).

"""
    _comonet_unpack_single(hidden_sizes, theta)

Unpack a flat parameter vector into (weights, biases) pairs per layer for
one network (input 1 → hidden_sizes… → 1).
"""
function _comonet_unpack_single(hidden_sizes, theta::AbstractVector{T}) where T
    layers = Tuple{Matrix{T}, Vector{T}}[]
    idx = 1
    prev = 1  # input dimension
    for h in hidden_sizes
        n_w = prev * h
        W = reshape(theta[idx:idx+n_w-1], h, prev)
        idx += n_w
        b = theta[idx:idx+h-1]
        idx += h
        push!(layers, (W, b))
        prev = h
    end
    # Output layer: prev → 1
    n_w = prev
    W = reshape(theta[idx:idx+n_w-1], 1, prev)
    idx += n_w
    b = theta[idx:idx]
    push!(layers, (W, b))
    return layers
end

"""
    _comonet_unpack(a::COMONetApproximator, theta)

Unpack the parameter vector: one network for most constraints, a pair of
branch networks (θ split in half) for the two-branch `:convex`/`:concave`.
"""
function _comonet_unpack(a::COMONetApproximator, theta::AbstractVector)
    if a.constraint in (:convex, :concave)
        nh = length(theta) ÷ 2
        return (_comonet_unpack_single(a.hidden_sizes, theta[1:nh]),
                _comonet_unpack_single(a.hidden_sizes, theta[nh+1:end]))
    end
    _comonet_unpack_single(a.hidden_sizes, theta)
end

# Precomputed COMONet branch with positive weights materialized once at build
# time and atomic ping-pong scratch buffers for zero-allocation Float64 evaluation.
struct _COMONetBranch{T}
    Ws::Vector{Matrix{T}}
    bs::Vector{Vector{T}}
    width::Int
end

mutable struct _COMONetScratch
    a::Vector{Float64}
    b::Vector{Float64}
    @atomic busy::Bool
end
_COMONetScratch(w::Int) = _COMONetScratch(zeros(w), zeros(w), false)

function _build_comonet_branch(layers::Vector{Tuple{Matrix{T}, Vector{T}}},
                               positive_weights::Bool) where {T}
    n_layers = length(layers)
    Ws = Vector{Matrix{T}}(undef, n_layers)
    bs = Vector{Vector{T}}(undef, n_layers)
    wmax = 1
    for i in 1:n_layers
        W_raw, b_raw = layers[i]
        Ws[i] = positive_weights ? (exp.(W_raw) ./ size(W_raw, 2)) : W_raw
        bs[i] = b_raw
        wmax = max(wmax, size(W_raw, 1), size(W_raw, 2))
    end
    _COMONetBranch{T}(Ws, bs, wmax)
end

@inline _como_relu(z) = max(zero(z), z)
@inline _como_concave_relu(z) = -max(zero(z), -z)
@inline _como_softplus(z) = _softplus(z)
@inline _como_concave_softplus(z) = -_softplus(-z)
@inline _como_tanh(z) = tanh(z)

function _comonet_branch_forward!(a::AbstractVector, b::AbstractVector,
                                  br::_COMONetBranch, x, act::F) where {F}
    n_layers = length(br.Ws)
    @inbounds a[1] = x
    for i in 1:n_layers
        W = br.Ws[i]
        bias = br.bs[i]
        n_out, n_in = size(W)
        is_hidden = i < n_layers
        @inbounds for r in 1:n_out
            s = W[r, 1] * a[1]
            for c in 2:n_in
                s += W[r, c] * a[c]
            end
            val = s + bias[r]
            b[r] = is_hidden ? act(val) : val
        end
        a, b = b, a
    end
    @inbounds a[1]
end

@inline function _comonet_eval_branch(br::_COMONetBranch{T}, x, act::F,
                                      sc::_COMONetScratch) where {T, F}
    if T === Float64 && x isa Float64 &&
       (@atomicreplace sc.busy false => true).success
        try
            return _comonet_branch_forward!(sc.a, sc.b, br, x, act)
        finally
            @atomic sc.busy = false
        end
    end
    R = promote_type(T, typeof(x))
    w = br.width
    _comonet_branch_forward!(Vector{R}(undef, w), Vector{R}(undef, w), br, R(x), act)
end

struct _COMONetSingleEval{T, F}
    br::_COMONetBranch{T}
    sc::_COMONetScratch
    act::F
    lo::Float64
    inv_span::Float64
    negate_in::Bool
    exp_out::Bool
end

function (e::_COMONetSingleEval)(x)
    x_norm = (x - e.lo) * e.inv_span
    xin = e.negate_in ? -x_norm : x_norm
    y = _comonet_eval_branch(e.br, xin, e.act, e.sc)
    e.exp_out ? exp(y) : y
end

struct _COMONetTwoBranchEval{T, F}
    br1::_COMONetBranch{T}
    br2::_COMONetBranch{T}
    sc::_COMONetScratch
    act::F
    lo::Float64
    inv_span::Float64
    negate_out::Bool
end

function (e::_COMONetTwoBranchEval)(x)
    x_norm = (x - e.lo) * e.inv_span
    s = _comonet_eval_branch(e.br1, x_norm, e.act, e.sc) +
        _comonet_eval_branch(e.br2, -x_norm, e.act, e.sc)
    e.negate_out ? -s : s
end

"""
    build_comonet_evaluator(a::COMONetApproximator, theta)

Build a zero-allocation evaluator function `x → f(x)` for a COMONet
approximator with the given parameters. Positive-weight exponentials are
materialized once at build time and forward passes reuse atomic scratch
buffers for Float64 inputs (falling back cleanly for ForwardDiff Duals).
"""
function build_comonet_evaluator(a::COMONetApproximator, theta::AbstractVector)
    lo, hi = Float64(a.domain[1]), Float64(a.domain[2])
    inv_span = 1.0 / (hi - lo)
    c = a.constraint

    if c in (:convex, :concave)
        l1, l2 = _comonet_unpack(a, theta)
        br1 = _build_comonet_branch(l1, true)
        br2 = _build_comonet_branch(l2, true)
        sc = _COMONetScratch(max(br1.width, br2.width))
        act = a.activation == :softplus ? _como_softplus : _como_relu
        return _COMONetTwoBranchEval(br1, br2, sc, act, lo, inv_span, c == :concave)
    end

    layers = _comonet_unpack(a, theta)
    if c == :positive
        br = _build_comonet_branch(layers, false)
        return _COMONetSingleEval(br, _COMONetScratch(br.width), _como_tanh,
                                  lo, inv_span, false, true)
    end

    br = _build_comonet_branch(layers, true)
    negate_in = c in (:decreasing, :dec_convex, :dec_concave)
    act = if c in (:increasing, :decreasing)
        _como_tanh
    elseif c in (:inc_concave, :dec_concave)
        a.activation == :softplus ? _como_concave_softplus : _como_concave_relu
    else
        a.activation == :softplus ? _como_softplus : _como_relu
    end
    _COMONetSingleEval(br, _COMONetScratch(br.width), act,
                       lo, inv_span, negate_in, false)
end

"""
    penalty_matrix(a::COMONetApproximator)

L2 (Tikhonov) penalty matrix for COMONet: `penalty_weight * I`.
This provides gentle regularization on the unconstrained weights.
"""
function penalty_matrix(a::COMONetApproximator)
    a.penalty_weight <= 0.0 && return nothing
    np = nparams(a)
    return a.penalty_weight * I(np) |> Matrix{Float64}
end

# ═══════════════════════════════════════════════════════════════════════
# Single index: inner loadings composed with a univariate outer smooth
# ═══════════════════════════════════════════════════════════════════════
#
# f(u₁, …, u_p) = s(z), z = (aᵀu − aᵀμ̂)/√(aᵀΣ̂a). The coefficient block is
# [free loadings; outer smooth coefficients], and the outer smooth is a
# REAL BSplineApproximator (or ShapeConstrainedBSplineApproximator) over
# [−xi, xi] held in the `outer` field — so every piece of the univariate
# machinery, including the whole SCOP-spline construction, is reused by
# composition rather than duplicated, and no existing type's behavior can
# be perturbed by this one.

"""
    _si_loadings(a::SingleIndexApproximator, params_k) -> Vector

The full length-`p` loading vector implied by a coefficient block. In
anchored mode the stored parameters are the `p − 1` free loadings and
`a[anchor]` is a hard 1; in free mode they are the loadings themselves.
Eltype-generic, so `ForwardDiff.Dual` parameters propagate.
"""
function _si_loadings(a::SingleIndexApproximator, params_k::AbstractVector)
    ni = _si_n_inner(a)
    inner = params_k[1:ni]
    a.anchor === nothing && return inner
    k = a.anchor
    T = eltype(inner)
    [i == k ? one(T) : inner[i < k ? i : i - 1] for i in 1:a.p]
end

"""
    index_loadings(a::SingleIndexApproximator, params) -> Vector{Float64}

Report the fitted index direction from a full coefficient vector (e.g.
`sol.parameters` restricted to this approximator's block).

In anchored mode the loadings are returned on the anchor's own scale
(`a[anchor] = 1`), which is what makes them directly comparable across
fits. In free mode — where `z` is invariant under `a → c·a`, so the
returned scale would otherwise be arbitrary — they are normalized to
`‖a‖ = 1` with the sign fixed so the first non-negligible loading is
positive.
"""
function index_loadings(a::SingleIndexApproximator, params::AbstractVector)
    # `params` is this approximator's block, not the whole coefficient
    # vector: without the check, a multi-approximator fit silently reports
    # loadings read from whichever approximator happens to come first.
    length(params) == nparams(a) || throw(ArgumentError(
        "index_loadings(:$(a.name)): expected this approximator's own " *
        "coefficient block of length $(nparams(a)), got $(length(params)). " *
        "For a multi-approximator problem, slice the fitted vector to " *
        "this approximator's range first."))
    load = Float64.(_si_loadings(a, params))
    a.anchor === nothing || return load
    nrm = norm(load)
    nrm > 0 || return load
    load = load ./ nrm
    k = findfirst(v -> abs(v) > 1e-8, load)
    (k !== nothing && load[k] < 0) && (load = -load)
    load
end

"""
    build_single_index_evaluator(a::SingleIndexApproximator, params_k)

Build the `p`-ARGUMENT callable `f(u₁, …, u_p) = s(z)` with
`z = (aᵀu − aᵀμ̂)/√(aᵀΣ̂a)`.

`(μ̂, Σ̂)` are the approximator's FIXED reference statistics; the evaluator
never recomputes them, so the geometry of `z` is identical at every
iteration of every solver no matter where the trajectory goes. `aᵀΣ̂a` is
floored at `1e-8 · tr(Σ̂)/p · ‖a‖²`, a guard proportional to `‖a‖²` so it
scales exactly like the numerator and therefore preserves the exact
invariance of `z` under `a → c·a` (`c > 0`) even when it binds.

Dual-safe in BOTH the parameters and the inputs: the loadings, the
standardization, and the outer evaluator are all eltype-generic, which
autodiff solvers and stiff ODE integrators with autodiff Jacobians
require (states arrive as `Dual`s).
"""
struct _SingleIndexEval{VL<:AbstractVector, F, T}
    load::VL
    centre::T
    inv_denom::T
    outer_eval::F
    p::Int
    name::Symbol
end

function (e::_SingleIndexEval)(u::Vararg{Any, N}) where {N}
    N == e.p || throw(ArgumentError(
        "SingleIndexApproximator(:$(e.name)) is a function of $(e.p) arguments " *
        "but was called with $N; the dynamics must pass " *
        "exactly $(e.p) states, e.g. p.$(e.name)(u[1], …, u[$(e.p)])"))
    @inbounds begin
        s = e.load[1] * u[1]
        for i in 2:N
            s += e.load[i] * u[i]
        end
    end
    z = (s - e.centre) * e.inv_denom
    e.outer_eval(z)
end

function build_single_index_evaluator(a::SingleIndexApproximator,
                                      params_k::AbstractVector)
    (a.mu === nothing || a.Sigma === nothing) && error(
        "SingleIndexApproximator(:$(a.name)): the reference statistics " *
        "(μ̂, Σ̂) are unresolved. They are filled in by PSMProblem " *
        "construction from the data; to build an evaluator from a bare " *
        "approximator, construct it with index_stats=(mu, Sigma).")

    ni = _si_n_inner(a)
    load = _si_loadings(a, params_k)
    outer_eval = build_evaluator(a.outer, params_k[(ni + 1):end])

    μ̂, Σ̂, p = a.mu, a.Sigma, a.p
    q = dot(load, Σ̂ * load)
    floor_rel = 1e-8 * tr(Σ̂) / p
    denom = sqrt(max(q, floor_rel * dot(load, load)))
    centre = dot(load, μ̂)
    T = promote_type(typeof(centre), typeof(denom))

    _SingleIndexEval(load, T(centre), T(one(T) / denom), outer_eval, p, a.name)
end

"""
    penalty_matrix(a::SingleIndexApproximator)

Block-diagonal merge of the two penalties with FIXED relative weights:
`inner_ridge · I` on the free loadings, and the outer smooth's own
roughness penalty (`∫(s'')²` on unit knots, or the Pya & Wood SCOP
first-difference penalty when the outer smooth is shape-constrained) on
the outer coefficients.

This is the single-λ view read by the gradient-matching and
probabilistic-numerics penalty sites and by the MCMC/VI/ABC prior
builder, which apply ONE weight per approximator. The
penalized-likelihood solvers instead read [`penalty_blocks`](@ref) and
give the inner and outer penalties SEPARATE smoothing parameters, in
which case `inner_ridge` is irrelevant.
"""
function penalty_matrix(a::SingleIndexApproximator)
    ni = _si_n_inner(a)
    no = nparams(a.outer)
    S = zeros(ni + no, ni + no)
    for i in 1:ni
        S[i, i] = a.inner_ridge
    end
    So = penalty_matrix(a.outer)
    So === nothing || (S[(ni + 1):end, (ni + 1):end] .= So)
    (S .+ S') ./ 2
end

"""
    penalty_blocks(a::SingleIndexApproximator)

Two disjoint blocks — the nested-effects structure of Fasiolo et al.
(arXiv:2511.19234), where the inner transformation and the outer smooth
carry independent smoothing parameters estimated jointly under LAML:

1. `(I, 1:n_inner)` — a ridge on the free loadings. In anchored mode this
   is a genuine, well-posed shrinkage of the index direction toward the
   anchor variable, and its λ is identified by the marginal likelihood.
   It also keeps the penalized Hessian non-singular when a loading is
   weakly identified.
2. `(S_outer, n_inner+1:end)` — the outer smooth's roughness penalty,
   exactly as the corresponding univariate approximator supplies it.

In free (`anchor = nothing`) mode the DATA term is exactly flat along
`a → c·a`, so block 1 is minimized by `‖a‖ → 0`: use anchored mode with
`LAML`/`GCVSolver` and keep free mode for the flat-objective solvers.
"""
function penalty_blocks(a::SingleIndexApproximator)
    ni = _si_n_inner(a)
    no = nparams(a.outer)
    blocks = Tuple{Matrix{Float64}, UnitRange{Int}}[
        (Matrix{Float64}(I, ni, ni), 1:ni)]
    So = penalty_matrix(a.outer)
    So === nothing || push!(blocks, (Matrix{Float64}(So), (ni + 1):(ni + no)))
    blocks
end

# ═══════════════════════════════════════════════════════════════════════
# Transformed covariate: learned inner transform of an EXOGENOUS series
# composed with a univariate outer smooth
# ═══════════════════════════════════════════════════════════════════════
#
# f(t) = s(z(t)), z the standardized transformed covariate. The coefficient
# block is [free inner parameters; outer smooth coefficients], and — exactly
# as for SingleIndexApproximator — the outer smooth is a REAL
# BSplineApproximator (or ShapeConstrainedBSplineApproximator) over
# [−xi, xi] held in the `outer` field, so the whole univariate machinery
# including the SCOP-spline construction is reused by composition.
#
# What differs from the single index is the standardization. There the
# "covariate" is the fitted trajectory, which MOVES, so N1 had to freeze
# reference statistics at problem construction. Here the covariate is fixed
# data: mean and sd of the transformed series are ordinary smooth functions
# of the inner parameters, recomputed exactly once per evaluator build, and
# none of N1's resolve-at-construction machinery is needed.

"""
    lag_weights(a::TransformedCovariateApproximator, params) -> Vector{Float64}

The fitted distributed-lag profile `[a₀, a₁, …]` from a `:lagindex`
approximator's own coefficient block, on the anchor's scale
(`a[anchor] = 1`), which is what makes profiles comparable across fits.
Entry `ℓ + 1` is the weight on `x(t − ℓΔ)`.
"""
function lag_weights(a::TransformedCovariateApproximator,
                     params::AbstractVector)
    a.trans === :lagindex || throw(ArgumentError(
        "lag_weights(:$(a.name)): defined only for trans=:lagindex, but " *
        "this approximator uses trans=:$(a.trans). For :expsm see " *
        "`smoothing_inertia`."))
    _tc_check_block(a, params, "lag_weights")
    Float64.(_tc_weights(a, params[1:_tc_n_inner(a)]))
end

"""
    smoothing_inertia(a::TransformedCovariateApproximator, params) -> Vector{Float64}

The fitted exponential-smoothing inertia `ωᵢ = logistic(w̃ᵢᵀa)` at each
covariate time, from an `:expsm` approximator's own coefficient block.
With the default single covariate column the inertia is constant and every
entry is the same number; auxiliary columns make it vary.
"""
function smoothing_inertia(a::TransformedCovariateApproximator,
                           params::AbstractVector)
    a.trans === :expsm || throw(ArgumentError(
        "smoothing_inertia(:$(a.name)): defined only for trans=:expsm, but " *
        "this approximator uses trans=:$(a.trans). For :lagindex see " *
        "`lag_weights`."))
    _tc_check_block(a, params, "smoothing_inertia")
    inner = params[1:_tc_n_inner(a)]
    Float64[_tc_logistic(dot(view(a.inner_design, i, :), inner))
            for i in 1:size(a.inner_design, 1)]
end

"""
    transformed_covariate(a::TransformedCovariateApproximator, params) -> Vector{Float64}

The standardized transformed covariate `z` at each of `a.times` — the
inner statistic the outer smooth actually sees. Mean ≈ 0 and variance ≈ 1
by construction (up to the additive variance floor).
"""
function transformed_covariate(a::TransformedCovariateApproximator,
                               params::AbstractVector)
    _tc_check_block(a, params, "transformed_covariate")
    Float64.(_tc_standardize(a, _tc_inner_series(a, params[1:_tc_n_inner(a)])))
end

# `params` must be THIS approximator's block. Without the check, a
# multi-approximator fit silently reports a transform decoded from whichever
# approximator happens to come first — the failure N1 hit with
# `index_loadings`.
function _tc_check_block(a::TransformedCovariateApproximator,
                         params::AbstractVector, fname::AbstractString)
    length(params) == nparams(a) || throw(ArgumentError(
        "$fname(:$(a.name)): expected this approximator's own coefficient " *
        "block of length $(nparams(a)), got $(length(params)). For a " *
        "multi-approximator problem, slice the fitted vector to this " *
        "approximator's range first."))
    nothing
end

"""
    _tc_weights(a, inner) -> Vector

The full length-`lags` weight vector implied by the free `:lagindex`
parameters: `a[anchor]` is a hard 1 and the stored parameters fill the
rest, in order. Eltype-generic, so `ForwardDiff.Dual` parameters propagate.
"""
function _tc_weights(a::TransformedCovariateApproximator,
                     inner::AbstractVector)
    L, k = a.lags, a.anchor
    T = eltype(inner)
    [i == k ? one(T) : inner[i < k ? i : i - 1] for i in 1:L]
end

"""
    _tc_inner_series(a, inner) -> Vector

The UNSTANDARDIZED transformed covariate `s̃` at every covariate time.

`:lagindex` is exactly the linear map `inner_design * weights` — the lag
matrix is fixed data built in the constructor, so nothing is recomputed
here beyond one matrix–vector product.

`:expsm` is the sequential scan `s̃ᵢ = ωᵢ s̃ᵢ₋₁ + (1 − ωᵢ) xᵢ`, run ONCE per
`build_evaluator` call in O(n) — not once per evaluation, which would make
every right-hand-side call O(n). It is branch-free in the parameters
(`_tc_logistic` is `tanh`-based), so it is Dual-safe.
"""
function _tc_inner_series(a::TransformedCovariateApproximator,
                          inner::AbstractVector)
    W = a.inner_design
    if a.trans === :lagindex
        return W * _tc_weights(a, inner)
    end
    n = size(W, 1)
    T = promote_type(eltype(inner), Float64)
    s = Vector{T}(undef, n)
    s[1] = a.X[1, 1]
    @inbounds for i in 2:n
        η = zero(T)
        for j in axes(W, 2)
            η += W[i, j] * inner[j]
        end
        ω = _tc_logistic(η)
        s[i] = ω * s[i - 1] + (one(T) - ω) * a.X[i, 1]
    end
    s
end

"""
    _tc_standardize(a, stil) -> Vector

Centre and scale the transformed series over the covariate SAMPLE — the
paper's recipe, applicable verbatim because the covariate is fixed data.
The scale is `√(v + κ)` with the constructor's additive floor `κ`, which
keeps `z` smooth (no `max` kink) even where `s̃` degenerates to a constant.
"""
function _tc_standardize(a::TransformedCovariateApproximator,
                         stil::AbstractVector)
    n = length(stil)
    m = sum(stil) / n
    c = stil .- m
    v = sum(abs2, c) / (n - 1)
    c ./ sqrt(v + a.var_floor)
end

"""
    build_transformed_covariate_evaluator(a, params_k)

Build the ONE-ARGUMENT callable `f(t) = s(z(t))`, the same convention every
other time-varying unknown function in the package uses (`p.beta(t)`).

The transformed, standardized series is computed once here — including the
`:expsm` scan — and the returned closure only interpolates it linearly in
time and evaluates the outer smooth, so a right-hand-side call is O(log n).
Linear interpolation keeps `f` CONTINUOUS in `t`, which adaptive ODE
stepping requires; it is only C⁰, hence the `jac=:forwarddiff` advice in
the type's docstring.

Dual-safe in the parameters (the scan, the standardization and the outer
evaluator are all eltype-generic) and in `t`.
"""
function build_transformed_covariate_evaluator(a::TransformedCovariateApproximator,
                                               params_k::AbstractVector)
    ni = _tc_n_inner(a)
    z = _tc_standardize(a, _tc_inner_series(a, params_k[1:ni]))
    outer_eval = build_evaluator(a.outer, params_k[(ni + 1):end])
    ts = a.times
    nm = a.name

    function transformed_covariate_eval(u...)
        length(u) == 1 || throw(ArgumentError(
            "TransformedCovariateApproximator(:$nm) is a function of TIME " *
            "alone but was called with $(length(u)) arguments; the dynamics " *
            "must pass the time, e.g. p.$nm(t)"))
        outer_eval(_tc_linterp(ts, z, u[1]))
    end
    transformed_covariate_eval
end

"""
    _tc_inner_penalty(a) -> Union{Nothing, Matrix{Float64}}

The inner penalty on the free transformation parameters, or `nothing` when
there is nothing to penalize.

`:expsm` — a ridge `I` on the inertia parameters, shrinking `ω` toward
1/2 and keeping the penalized Hessian non-singular when the inertia is
weakly identified (which, per the type's docstring, it often is).

`:lagindex` — the FIRST-DIFFERENCE penalty `DᵀD` of the paper's smooth-lag
prior, so neighbouring lags are pulled together and a large λ flattens the
lag profile. It acts on the FREE weights only: the anchored weight is not
a parameter, so the step from it to the first free weight would contribute
an affine (not quadratic) term that the `λ βᵀSβ` penalty interface cannot
represent. The null space is therefore "all free weights equal", and the
`λ → ∞` limit of the profile is `[1, c, c, …]` rather than a flat line.
With fewer than two free weights there are no differences to take and this
returns `nothing`.
"""
function _tc_inner_penalty(a::TransformedCovariateApproximator)
    ni = _tc_n_inner(a)
    ni >= 1 || return nothing
    a.trans === :expsm && return Matrix{Float64}(I, ni, ni)
    ni >= 2 || return nothing
    D = zeros(ni - 1, ni)
    for i in 1:(ni - 1)
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    end
    S = D' * D
    (S .+ S') ./ 2
end

"""
    penalty_matrix(a::TransformedCovariateApproximator)

Block-diagonal merge of the two penalties with FIXED relative weights:
`inner_ridge ·` the inner penalty (a ridge for `:expsm`, the lag
first-difference penalty for `:lagindex`), and the outer smooth's own
roughness penalty on the outer coefficients.

This is the single-λ view read by the gradient-matching and
probabilistic-numerics penalty sites and by the MCMC/VI/ABC prior builder,
which apply ONE weight per approximator. The penalized-likelihood solvers
instead read [`penalty_blocks`](@ref) and give the inner and outer
penalties SEPARATE smoothing parameters, in which case `inner_ridge` is
irrelevant.
"""
function penalty_matrix(a::TransformedCovariateApproximator)
    ni = _tc_n_inner(a)
    no = nparams(a.outer)
    S = zeros(ni + no, ni + no)
    Si = _tc_inner_penalty(a)
    Si === nothing || (S[1:ni, 1:ni] .= a.inner_ridge .* Si)
    So = penalty_matrix(a.outer)
    So === nothing || (S[(ni + 1):end, (ni + 1):end] .= So)
    (S .+ S') ./ 2
end

"""
    penalty_blocks(a::TransformedCovariateApproximator)

Two disjoint blocks — the nested-effects structure of Fasiolo et al.
(arXiv:2511.19234), where the inner transformation and the outer smooth
carry independent smoothing parameters estimated jointly under LAML:

1. `(S_inner, 1:n_inner)` — the ridge on the `:expsm` inertia parameters,
   or the first-difference smooth-lag penalty on the free `:lagindex`
   weights (see `_tc_inner_penalty`). Omitted when there is nothing to
   penalize (a `:lagindex` window with fewer than two free weights).
2. `(S_outer, n_inner+1:end)` — the outer smooth's roughness penalty,
   exactly as the corresponding univariate approximator supplies it.
"""
function penalty_blocks(a::TransformedCovariateApproximator)
    ni = _tc_n_inner(a)
    no = nparams(a.outer)
    blocks = Tuple{Matrix{Float64}, UnitRange{Int}}[]
    Si = _tc_inner_penalty(a)
    Si === nothing || push!(blocks, (Si, 1:ni))
    So = penalty_matrix(a.outer)
    So === nothing || push!(blocks, (Matrix{Float64}(So), (ni + 1):(ni + no)))
    blocks
end
