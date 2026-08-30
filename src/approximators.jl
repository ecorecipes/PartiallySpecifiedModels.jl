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
    build_tensor_bspline_evaluator(a::TensorBSplineApproximator, params_k)

Build the two-argument callable `f(x, y)` for a tensor-product spline from
the column-major coefficient grid `params_k = vec(C)`, `C[i, j] = f(x_i, y_j)`.

The construction is the tensor product of the SAME univariate machinery
`BSplineApproximator` uses (`build_bspline_evaluator`: natural cubic
interpolation THROUGH the values, linear extrapolation outside the knots):
each x-column `C[:, j]` is interpolated in x once at build time; per call,
the `nknots_y` line values `sⱼ(x)` are interpolated in y. Because both
stages interpolate through values, the surface agrees exactly with the
univariate evaluator on every grid line —
`f(x, y_j) = CubicSpline(C[:, j], knots_x)(x)` and
`f(x_i, y) = CubicSpline(C[i, :], knots_y)(y)` — and extrapolates linearly
per margin, matching the univariate behavior. Eltype-generic in both
`params_k` and `(x, y)`, so ForwardDiff Duals propagate through either.
"""
function build_tensor_bspline_evaluator(a::TensorBSplineApproximator,
                                        params_k::AbstractVector)
    nx, ny = a.nknots_x, a.nknots_y
    knots_x = collect(range(a.domain_x[1], a.domain_x[2], length=nx))
    knots_y = collect(range(a.domain_y[1], a.domain_y[2], length=ny))
    C = reshape(params_k, nx, ny)
    col_splines = [build_bspline_evaluator(knots_x, C[:, j]) for j in 1:ny]
    function tensor_eval(x, y)
        line_vals = [s(x) for s in col_splines]
        build_bspline_evaluator(knots_y, line_vals)(y)
    end
    tensor_eval
end

"""
    penalty_matrix(a::TensorBSplineApproximator)

Kronecker-sum roughness penalty on the column-major coefficient grid:

    S = I_ny ⊗ S_x + anisotropy · (S_y ⊗ I_nx)

where `S_x`, `S_y` are the univariate natural-cubic `∫(f'')²` penalties
(`spline_penalty_matrix`) on unit-interval knot grids, exactly as
`penalty_matrix(::BSplineApproximator)` builds them. With `β = vec(C)` and
the x index fastest, `β' (I_ny ⊗ S_x) β = Σⱼ C[:, j]' S_x C[:, j]`
penalizes roughness along x only, and `β' (S_y ⊗ I_nx) β =
Σᵢ C[i, :]' S_y C[i, :]` penalizes roughness along y only; `anisotropy` is
their fixed relative weight under the single per-approximator λ.

The null space is the intersection of "every column affine in x" with
"every row affine in y": the bilinear surfaces
`C[i, j] = a + b·x_i + c·y_j + d·x_i·y_j` (rank `nx·ny − 4`).
"""
function penalty_matrix(a::TensorBSplineApproximator)
    nx, ny = a.nknots_x, a.nknots_y
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

"""
    _gp_mean_interpolant(a, values)

Kernel predictive-mean interpolant through `values` at `a.inducing_points`:
  f(x) = k(x, X)' K⁻¹ values
Shared by `GPApproximator` (whose parameters ARE the inducing values) and
`ShapeConstrainedGPApproximator` (whose constrained values β = Σ·d(γ) are
interpolated the same way). `a` needs the fields `kernel`, `lengthscale`,
`variance`, `K_inv`, and `inducing_points`.
"""
function _gp_mean_interpolant(a, values::AbstractVector)
    kfunc = _kernel_func(a.kernel, a.lengthscale, a.variance)
    weights = a.K_inv * values  # precompute α = K⁻¹ f
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
    core = _gp_mean_interpolant(a, beta)
    offset = if a.constraint in (:inc_zero_left, :dec_zero_left)
        core(a.domain[1])
    elseif a.constraint in (:inc_zero_right, :dec_zero_right)
        core(a.domain[2])
    else
        nothing
    end
    offset === nothing ? core : (x -> core(x) - offset)
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
    nm = a.name

    function single_index_eval(u...)
        length(u) == p || throw(ArgumentError(
            "SingleIndexApproximator(:$nm) is a function of $p arguments " *
            "but was called with $(length(u)); the dynamics must pass " *
            "exactly $p states, e.g. p.$nm(u[1], …, u[$p])"))
        z = (sum(load[i] * u[i] for i in 1:p) - centre) / denom
        outer_eval(z)
    end
    single_index_eval
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
