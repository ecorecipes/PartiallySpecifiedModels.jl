# Shared penalized least-squares (PCLS) step machinery for the IRLS
# solvers (LAML in solver.jl, GCV in gcv_solver.jl).
#
# Both solvers advance the working linear model by solving the augmented
# penalized system and then backtracking along the resulting direction.
# The numerically delicate parts — the truncated-SVD solve and the
# explosive-step rescue — live here so the two paths cannot drift apart.

"""
    _pcls_augmented_solve(J_mat, z_pseudo, B, w_irls)

Solve the penalized weighted least-squares step via the augmented system

    [W^½ J; C] β = [W^½ z; 0]

where `C'C = B` (see [`penalty_sqrt_matrix`](@ref)) and `W` holds the IRLS
weights (floored at 1e-15).

Uses truncated-SVD least squares. The FD Jacobian is trajectory-local: at a
poor initialization (e.g. the x -> 0 default, where the trajectory sits at
u0) most basis columns are numerically null, and a plain QR solve returns
O(1e9) coefficients along those directions — a step no contraction can
rescue. Zeroing components with σ < 1e-7·σ_max keeps the step inside the
identified subspace; for well-conditioned systems the result matches
backslash to 1e-7.

Returns the step coefficients `β`.
"""
function _pcls_augmented_solve(J_mat::AbstractMatrix, z_pseudo::AbstractVector,
                               B::AbstractMatrix, w_irls::AbstractVector)
    C = penalty_sqrt_matrix(B)
    n_pen = size(C, 1)
    W_sqrt = sqrt.(max.(w_irls, 1e-15))
    F_aug = vcat(Diagonal(W_sqrt) * J_mat, C)
    z_aug = vcat(W_sqrt .* z_pseudo, zeros(n_pen))
    F = svd(F_aug)
    σmax = F.S[1]
    d = [σ > 1e-7 * σmax ? 1.0 / σ : 0.0 for σ in F.S]
    F.V * (d .* (F.U' * z_aug))
end

"""
    _pcls_step_contract(objective, a_old, a_new, B)

Step contraction: backtracking line search with exponential step sizes along
the PCLS direction `a_new - a_old`, scoring candidates with
`objective(a, B)` (the solver's penalized objective closure; must return
`Inf` on simulation failure).

Phase 1 tries α = 1, 0.5, ..., 2^(-15) and keeps the best (unchanged legacy
behavior for sane steps). Phase 2 rescues EXPLOSIVE steps: when the
trajectory-localized Jacobian is numerically rank-deficient (e.g. every
coefficient at the x -> 0 default initialization), the PCLS solution can be
O(1e9) along near-null directions and even 2^(-15) of it still blows up the
ODE; continue halving with a first-improvement exit so the fit can escape
instead of silently rejecting every step.

Returns `(a_best, f_best)`.
"""
function _pcls_step_contract(objective, a_old::AbstractVector,
                             a_new::AbstractVector, B::AbstractMatrix)
    f_old = objective(a_old, B)
    direction = a_new .- a_old

    best_f = f_old
    best_a = copy(a_old)

    for k in 0:15
        α = 2.0^(-k)
        a_try = a_old .+ α .* direction
        f_try = objective(a_try, B)
        if f_try < best_f
            best_f = f_try
            best_a = copy(a_try)
        end
    end
    if best_f >= f_old
        for k in 16:50
            α = 2.0^(-k)
            a_try = a_old .+ α .* direction
            f_try = objective(a_try, B)
            if f_try < f_old
                return a_try, f_try
            end
        end
    end
    best_a, best_f
end
