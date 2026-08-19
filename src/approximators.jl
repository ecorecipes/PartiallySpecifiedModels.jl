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
    raw = xv -> sum(kfunc(xv, x_ind[j]) * weights[j] for j in eachindex(x_ind))
    # Outside the inducing-point range the kernel decays and the predictive
    # mean reverts to 0 REGARDLESS of the fitted level — an ODE excursion
    # past the domain saw f ≈ 0. Extrapolate linearly from the boundary
    # value and slope instead (consistent with the spline evaluators).
    lo, hi = extrema(x_ind)
    h = (hi - lo) * 1e-6
    f_lo = raw(lo);  s_lo = (raw(lo + h) - f_lo) / h
    f_hi = raw(hi);  s_hi = (f_hi - raw(hi - h)) / h
    x -> begin
        xv = x isa AbstractArray ? x[1] : x   # keep Duals intact
        if xv < lo
            f_lo + s_lo * (xv - lo)
        elseif xv > hi
            f_hi + s_hi * (xv - hi)
        else
            raw(xv)
        end
    end
end

"""
    _adapt_gp_hyperparams!(a::GPApproximator, beta_k) -> Bool

Empirical-Bayes update of the kernel hyperparameters DURING a fit: choose
(ℓ, σ²) maximizing the GP log marginal likelihood of the current inducing
values `beta_k` (small fixed nugget), then rebuild `K`/`K_inv`. Called by
the LAML/GCV loops when `a.adapt` (no user-supplied lengthscale). Returns
whether the hyperparameters changed materially.
"""
function _adapt_gp_hyperparams!(a::GPApproximator, beta_k::AbstractVector)
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
    for frac in (0.08, 0.15, 0.25, 0.4, 0.6, 1.0), σm in (0.5, 1.0, 2.0)
        ℓ_try = frac * span
        σ²_try = σm * v
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
    kf = _kernel_func(a.kernel, best_ℓ, best_σ²)
    K = _build_kernel_matrix(kf, x)
    scale = max(maximum(abs, K), 1.0)
    K_inv = nothing
    for jit in (1e-8, 1e-6, 1e-4)
        F = cholesky(Symmetric(K + jit * scale * I), check=false)
        if issuccess(F)
            K_inv = Matrix(inv(F)); break
        end
    end
    K_inv === nothing && (K_inv = pinv(K + 1e-4 * scale * I))
    a.K = K
    a.K_inv = 0.5 * (K_inv + K_inv')
    true
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
    CubicSpline(params, mesh_x; extrapolation=ExtrapolationType.Linear)
end

# ─── Shape-constrained SPDE: evaluator and penalty ────────────────

"""
    build_constrained_spde_evaluator(a::ShapeConstrainedSPDEApproximator, gamma)

Build a callable evaluator from unconstrained parameters γ.

Applies the SCOP-spline reparameterization: mesh node values are computed
as `β = Σ · _apply_constraint_transform(γ)` (free components linear, the
rest softplus'd), then interpolated with a cubic spline.
Shape constraints are enforced at mesh nodes; the cubic spline interpolation
between nodes may slightly overshoot.
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

"""
    build_constrained_bspline_evaluator(a, gamma)

Build a callable evaluator from unconstrained parameters γ using proper
B-spline basis evaluation.

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

    # Zero-at-endpoint constraints: zeroing one B-spline COEFFICIENT does not
    # zero the function VALUE at the boundary (three basis functions are
    # active there, weights 1/6, 4/6, 1/6). Center the spline by subtracting
    # its boundary value — a constant shift that is linear in β, preserves
    # monotonicity/curvature exactly, and pins f(endpoint) = 0.
    offset = if a.constraint in (:inc_zero_left, :dec_zero_left)
        f_ll
    elseif a.constraint in (:inc_zero_right, :dec_zero_right)
        f_ul
    else
        zero(f_ll)
    end

    # Return a callable that evaluates the (centered) B-spline at any point
    function evaluator(x::Real)
        if x < ll
            # Linear extrapolation below domain
            return f_ll - offset + slope_lo * (x - ll)
        elseif x > ul
            # Linear extrapolation above domain
            return f_ul - offset + slope_hi * (x - ul)
        else
            B = _bspline_basis_vector(x, xk, spline_order)
            return dot(B, beta) - offset
        end
    end
    return evaluator
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
    constraint in (:increasing, :decreasing)                     ? (1,)    : ()

function penalty_matrix(a::ShapeConstrainedBSplineApproximator)
    np = nparams(a)
    # First-order difference penalty over the curvature-carrying components
    # only, per Pya & Wood (2015): their D starts the differences from β̃₂
    # for monotone smooths and from β̃₃ for curvature-constrained smooths.
    # Free level/slope components live in the penalty null space — shrinking
    # them with λ would bias the function's level, not its wiggliness.
    skip = _penalty_skip_indices(a.constraint, np)
    idxs = [i for i in 1:np if !(i in skip)]
    n_c = length(idxs)
    D = zeros(max(n_c - 1, 0), np)
    for r in 1:(n_c - 1)
        D[r, idxs[r]]   = -1.0
        D[r, idxs[r+1]] =  1.0
    end
    D' * D
end


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

"""
Positive-weight branch: exp(W̃)/fanin weights, ReLU/softplus (or negated
concave) hidden activations, linear output. Convex-nondecreasing (or
concave-nondecreasing) by construction.
"""
function _comonet_branch(layers, x, use_concave::Bool, activation::Symbol)
    h = [x]
    n_layers = length(layers)
    for (i, (W_tilde, b)) in enumerate(layers)
        W_pos = exp.(W_tilde) ./ size(W_tilde, 2)   # positive, fan-in scaled
        z = W_pos * h .+ b
        if i < n_layers
            if activation == :softplus
                h = use_concave ? [-_softplus(-zi) for zi in z] :
                                  [_softplus(zi) for zi in z]
            else  # :relu (default)
                h = use_concave ? [-max(zero(eltype(z)), -zi) for zi in z] :
                                  [max(zero(eltype(z)), zi) for zi in z]
            end
        else
            h = z   # linear output
        end
    end
    h[1]
end

"""
Positive-weight monotone net: exp(W̃)/fanin weights with tanh hidden units.
Monotone nondecreasing (positive weights + nondecreasing activation) but
NOT constrained in curvature — tanh saturates, so sigmoids and other
monotone-nonconvex shapes are representable.
"""
function _comonet_monotone(layers, x)
    h = [x]
    n_layers = length(layers)
    for (i, (W_tilde, b)) in enumerate(layers)
        W_pos = exp.(W_tilde) ./ size(W_tilde, 2)
        z = W_pos * h .+ b
        h = i < n_layers ? [tanh(zi) for zi in z] : z
    end
    h[1]
end

"""
Unconstrained tanh MLP (raw weights): used inside `exp(·)` for `:positive`,
where positivity comes from the output transform alone.
"""
function _comonet_raw_mlp(layers, x)
    h = [x]
    n_layers = length(layers)
    for (i, (W, b)) in enumerate(layers)
        z = W * h .+ b
        h = i < n_layers ? [tanh(zi) for zi in z] : z
    end
    h[1]
end

function _comonet_forward(layers, x_norm, constraint::Symbol, activation::Symbol=:relu)
    if constraint in (:convex, :concave)
        # Two-branch input-convex construction: g₁(x) + g₂(−x), both convex
        # nondecreasing ⇒ sum convex, possibly non-monotone (U-shapes).
        l1, l2 = layers
        s = _comonet_branch(l1, x_norm, false, activation) +
            _comonet_branch(l2, -x_norm, false, activation)
        return constraint == :convex ? s : -s
    elseif constraint == :positive
        return exp(_comonet_raw_mlp(layers, x_norm))
    end

    # Input transform for decreasing constraints
    x = constraint in (:decreasing, :dec_convex, :dec_concave) ? -x_norm : x_norm

    if constraint in (:increasing, :decreasing)
        return _comonet_monotone(layers, x)
    end

    use_concave = constraint in (:inc_concave, :dec_concave)
    _comonet_branch(layers, x, use_concave, activation)
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
