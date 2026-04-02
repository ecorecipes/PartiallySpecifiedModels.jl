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
    itp = CubicSpline(knots_y, knots_x; extrapolation=ExtrapolationType.Extension)
    return itp(x)
end

"""
    build_bspline_evaluator(knots_x, knots_y)

Build a callable cubic spline evaluator (caches the interpolation object).
"""
function build_bspline_evaluator(knots_x::AbstractVector, knots_y::AbstractVector)
    CubicSpline(knots_y, knots_x; extrapolation=ExtrapolationType.Extension)
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

"""
    build_gp_evaluator(a::GPApproximator, params)

Build a callable that evaluates the GP predictive mean at any input x:
  f(x) = k(x, X)' K⁻¹ f_X
where f_X are the function values at inducing points (= params).
"""
function build_gp_evaluator(a::GPApproximator, params::AbstractVector)
    kfunc = _kernel_func(a.kernel, a.lengthscale, a.variance)
    weights = a.K_inv * params  # precompute α = K⁻¹ f
    x_ind = a.inducing_points
    x -> begin
        xv = Float64(x isa AbstractArray ? x[1] : x)
        sum(kfunc(xv, x_ind[j]) * weights[j] for j in eachindex(x_ind))
    end
end

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
    CubicSpline(params, mesh_x; extrapolation=ExtrapolationType.Extension)
end

# ─── Shape-constrained SPDE: evaluator and penalty ────────────────

"""
    build_constrained_spde_evaluator(a::ShapeConstrainedSPDEApproximator, gamma)

Build a callable evaluator from unconstrained parameters γ.

Applies the SCOP-spline reparameterization: mesh node values are computed
as `β = Σ * softplus(γ)`, then interpolated with a cubic spline.
Shape constraints are enforced at mesh nodes; the cubic spline interpolation
between nodes may slightly overshoot.
"""
function build_constrained_spde_evaluator(a::ShapeConstrainedSPDEApproximator,
                                          gamma::AbstractVector)
    ν = [_softplus(g) for g in gamma]
    mesh_values = a.Sigma * ν
    CubicSpline(mesh_values, a.mesh_points; extrapolation=ExtrapolationType.Extension)
end

"""
    gamma_to_mesh_values(a::ShapeConstrainedSPDEApproximator, gamma)

Transform unconstrained parameters γ to mesh node values β = Σ * softplus(γ).
"""
function gamma_to_mesh_values(a::ShapeConstrainedSPDEApproximator,
                              gamma::AbstractVector)
    ν = [_softplus(g) for g in gamma]
    a.Sigma * ν
end

"""
    penalty_matrix(a::ShapeConstrainedSPDEApproximator)

Compute the Matérn SPDE penalty matrix for the constrained approximator.
The penalty operates in the unconstrained parameter space (γ), so it is
transformed as P_γ = Σᵀ P_β Σ where P_β is the SPDE FEM penalty.
"""
function penalty_matrix(a::ShapeConstrainedSPDEApproximator)
    # Build the SPDE penalty in mesh-value space
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
_softplus(x::Real) = x > 20.0 ? Float64(x) : log1p(exp(x))

"""
    _build_sigma_matrix(constraint, nknots) -> Matrix{Float64}

Build the Σ constraint matrix that maps positive coefficients ν = softplus(γ)
to knot values β = Σ * ν satisfying the given shape constraint.

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
        # Second differences increasing: β_j = j*ν₁ + Σ_{k=2}^{j} (j-k+1)*νₖ
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = Float64(i)
        end
        for j in 2:q, i in j:q
            Sig[i, j] = Float64(i - j + 1)
        end
    elseif constraint == :concave
        # Second differences decreasing: negate off-diagonal of convex
        Sig = zeros(q, q)
        for i in 1:q
            Sig[i, 1] = Float64(i)
        end
        for j in 2:q, i in j:q
            Sig[i, j] = -Float64(i - j + 1)
        end
    elseif constraint == :inc_convex
        # Monotone increasing + convex: cumsum of cumsums
        Sig = zeros(q, q)
        for j in 1:q, i in j:q
            Sig[i, j] = Float64(i - j + 1)
        end
    elseif constraint == :inc_concave
        # Monotone increasing + concave
        Sig = zeros(q, q)
        for j in 1:q, i in 1:q
            Sig[i, j] = Float64(min(i, q - j + 1))
        end
    elseif constraint == :dec_convex
        # Monotone decreasing + convex: negate inc_concave
        Sig = zeros(q, q)
        for j in 1:q, i in 1:q
            Sig[i, j] = -Float64(min(i, q - j + 1))
        end
    elseif constraint == :dec_concave
        # Monotone decreasing + concave: negate inc_convex
        Sig = zeros(q, q)
        for j in 1:q, i in j:q
            Sig[i, j] = -Float64(i - j + 1)
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
    gamma_to_knot_values(a::ShapeConstrainedBSplineApproximator, gamma)

Transform unconstrained parameters γ to B-spline coefficients β = Σ * softplus(γ).
These are de Boor control point values, not interpolation knot values.
"""
function gamma_to_knot_values(a::ShapeConstrainedBSplineApproximator,
                              gamma::AbstractVector)
    ν = [_softplus(g) for g in gamma]
    a.Sigma * ν
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

    # Order 1: piecewise constant
    b = zeros(nk - 1)
    for j in 1:(nk - 1)
        if j == nk - 1
            b[j] = (knots[j] <= x <= knots[j + 1]) ? 1.0 : 0.0
        else
            b[j] = (knots[j] <= x < knots[j + 1]) ? 1.0 : 0.0
        end
    end

    # Recursion for higher orders
    for p in 2:order
        b_new = zeros(nk - p)
        for j in 1:(nk - p)
            d1 = knots[j + p - 1] - knots[j]
            d2 = knots[j + p] - knots[j + 1]
            t1 = d1 > 0 ? (x - knots[j]) / d1 * b[j] : 0.0
            t2 = d2 > 0 ? (knots[j + p] - x) / d2 * b[j + 1] : 0.0
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

"""
    build_constrained_bspline_evaluator(a, gamma)

Build a callable evaluator from unconstrained parameters γ using proper
B-spline basis evaluation.

Uses the SCOP-spline approach (Pya & Wood 2015): the constrained coefficients
β = Σ * softplus(γ) are B-spline basis coefficients, and the function is
evaluated as f(x) = Σⱼ βⱼ Bⱼ(x). The convex hull property of B-splines
guarantees that shape constraints (monotonicity, convexity, positivity)
hold everywhere on the domain, not just at knot points.
"""
function build_constrained_bspline_evaluator(a::ShapeConstrainedBSplineApproximator,
                                             gamma::AbstractVector)
    beta = gamma_to_knot_values(a, gamma)  # B-spline coefficients
    xk = _scam_knot_vector(a.domain, a.nknots)
    spline_order = 4  # cubic B-spline (m=2, order=m+2)

    # Inner knot range for extrapolation
    m = 2
    ll = xk[m + 2]      # lower boundary of inner range
    ul = xk[end - m - 1] # upper boundary of inner range

    # Precompute linear extrapolation slopes at boundaries
    h = (ul - ll) * 1e-7
    B_ll = _bspline_basis_vector(ll, xk, spline_order)
    B_ll_p = _bspline_basis_vector(ll + h, xk, spline_order)
    slope_lo = ((B_ll_p .- B_ll) ./ h)' * beta  # derivative at lower boundary

    B_ul = _bspline_basis_vector(ul, xk, spline_order)
    B_ul_m = _bspline_basis_vector(ul - h, xk, spline_order)
    slope_hi = ((B_ul .- B_ul_m) ./ h)' * beta  # derivative at upper boundary

    f_ll = dot(B_ll, beta)
    f_ul = dot(B_ul, beta)

    # Return a callable that evaluates the B-spline at any point
    function evaluator(x::Real)
        if x < ll
            # Linear extrapolation below domain
            return f_ll + slope_lo * (x - ll)
        elseif x > ul
            # Linear extrapolation above domain
            return f_ul + slope_hi * (x - ul)
        else
            B = _bspline_basis_vector(x, xk, spline_order)
            return dot(B, beta)
        end
    end
    return evaluator
end

"""
    penalty_matrix(a::ShapeConstrainedBSplineApproximator)

First-order difference penalty on unconstrained parameters (size np × np).
This penalizes roughness while the Sigma reparameterization enforces shape.
"""
function penalty_matrix(a::ShapeConstrainedBSplineApproximator)
    np = nparams(a)
    # First-order difference penalty: D'D where D is (np-1) × np
    D = zeros(np - 1, np)
    for i in 1:(np - 1)
        D[i, i]   = -1.0
        D[i, i+1] =  1.0
    end
    D' * D
end


# ═══════════════════════════════════════════════════════════════════════
# COMONet: shape-constrained neural network evaluator
# ═══════════════════════════════════════════════════════════════════════
#
# COMONet guarantees shape constraints architecturally:
#   - Monotone increasing: all weights W > 0 (via exp(W̃)) + non-decreasing activation
#   - Convex: positive weights + ReLU (convex, non-decreasing) composition
#   - Concave: negate the convex network: f(x) = -g(x) where g is convex
#   - Decreasing: negate input: f(x) = g(-x) where g is increasing
#
# Parameters are stored unconstrained (W̃, b); at evaluation time we apply
# exp(W̃) to get positive weights.

"""
    _comonet_unpack(a::COMONetApproximator, theta)

Unpack flat parameter vector into (weights, biases) pairs per layer.
Returns vector of (W_matrix, b_vector) tuples.
"""
function _comonet_unpack(a::COMONetApproximator, theta::AbstractVector{T}) where T
    layers = Tuple{Matrix{T}, Vector{T}}[]
    idx = 1
    prev = 1  # input dimension
    for h in a.hidden_sizes
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
    _comonet_forward(layers, x_norm, constraint, activation)

Forward pass through COMONet. `x_norm` is normalized to [0,1].
All hidden layers use exp(W̃) for positive weights + activation function.
Output layer uses exp(W̃) + identity (no activation).

Activations:
- `:relu`: ReLU / min(0,·) — piecewise linear (C⁰), exact theoretical match
- `:softplus`: softplus / -softplus(-·) — smooth (C∞), same guarantees

Constraint-specific input/output transforms:
- `:decreasing` / `:dec_*`: negate input
- `:concave` / `:inc_concave` / `:dec_concave`: concave-branch activation
- `:positive`: apply exp to output
"""
function _comonet_forward(layers, x_norm, constraint::Symbol, activation::Symbol=:relu)
    # Input transform for decreasing constraints
    x = if constraint in (:decreasing, :dec_convex, :dec_concave)
        -x_norm
    else
        x_norm
    end

    # Use concave-branch activation?
    use_concave = constraint in (:concave, :inc_concave, :dec_concave)

    # Forward pass through hidden layers — use promote_type for ForwardDiff compatibility
    h = [x]
    n_layers = length(layers)
    for (i, (W_tilde, b)) in enumerate(layers)
        W_pos = exp.(W_tilde)  # guaranteed positive weights
        z = W_pos * h .+ b
        if i < n_layers
            # Hidden layer activation
            if activation == :softplus
                if use_concave
                    # -softplus(-z): smooth, concave, non-decreasing, ≤ 0
                    h = [-_softplus(-zi) for zi in z]
                else
                    # softplus(z): smooth, convex, non-decreasing, ≥ 0
                    h = [_softplus(zi) for zi in z]
                end
            else  # :relu (default)
                if use_concave
                    h = [-max(zero(eltype(z)), -zi) for zi in z]
                else
                    h = [max(zero(eltype(z)), zi) for zi in z]
                end
            end
        else
            # Output layer: linear (no activation)
            h = z
        end
    end

    out = h[1]  # scalar output

    # Output transform
    if constraint == :positive
        return exp(out)
    else
        return out
    end
end

"""
    build_comonet_evaluator(a::COMONetApproximator, theta)

Build an evaluator function `x → f(x)` for a COMONet approximator with
the given parameters. The function maps from the original domain to ℝ.
"""
function build_comonet_evaluator(a::COMONetApproximator, theta::AbstractVector)
    layers = _comonet_unpack(a, theta)
    lo, hi = a.domain
    span = hi - lo
    act = a.activation

    function evaluator(x)
        x_norm = (x - lo) / span  # normalize to ~[0, 1]
        return _comonet_forward(layers, x_norm, a.constraint, act)
    end
    return evaluator
end

"""
    penalty_matrix(a::COMONetApproximator)

L2 (Tikhonov) penalty matrix for COMONet: `penalty_weight * I`.
This provides gentle regularization on the unconstrained weights.
"""
function penalty_matrix(a::COMONetApproximator)
    np = nparams(a)
    return a.penalty_weight * I(np) |> Matrix{Float64}
end
