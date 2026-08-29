# Shared penalized least-squares (PCLS) step machinery for the IRLS
# solvers (LAML in solver.jl, GCV in gcv_solver.jl).
#
# Both solvers advance the working linear model by solving the augmented
# penalized system and then backtracking along the resulting direction.
# The numerically delicate parts — the truncated-SVD solve, the explicit
# trust region on the step, and the explosive-step rescue — live here so
# the two paths cannot drift apart.

"""
    _PCLSFactorization

One SVD of the augmented penalized design `F = [W^½J; C]`, reused for every
step this working model can propose: the truncated least-squares step
([`_pcls_truncated_step`](@ref)) and the whole Levenberg-Marquardt damping
ladder ([`_pcls_damped_step`](@ref)).

`sigma_data` is `σ_max(W^½J)` — the DATA block's scale, deliberately not the
augmented matrix's. Everything scaled against it is invariant to the
smoothing parameters; see [`_pcls_truncated_step`](@ref).
"""
struct _PCLSFactorization{TF,TV}
    F::TF           # svd([W^½J; C])
    Utz::TV         # U'·[W^½z; 0] — all any step or the model needs of the RHS
    sigma_data::Float64
end

"""
    _pcls_factorize(J_mat, z_pseudo, B, w_irls)

Factorize the working penalized least-squares problem

    [W^½ J; C] β ≈ [W^½ z; 0]

where `C'C = B` (see [`penalty_sqrt_matrix`](@ref)) and `W` holds the IRLS
weights (floored at 1e-15).
"""
function _pcls_factorize(J_mat::AbstractMatrix, z_pseudo::AbstractVector,
                         B::AbstractMatrix, w_irls::AbstractVector)
    C = penalty_sqrt_matrix(B)
    n_pen = size(C, 1)
    W_sqrt = sqrt.(max.(w_irls, 1e-15))
    A_data = Diagonal(W_sqrt) * J_mat
    F = svd(vcat(A_data, C))
    z_aug = vcat(W_sqrt .* z_pseudo, zeros(n_pen))
    # Reference scale for the rank test.  Fall back to the augmented scale
    # only when the data block is identically zero: there is then no data
    # scale to reference, and β is zero regardless (z_aug's penalty block is
    # zero, and every u_i with σ_i > 0 has zero data part).
    σ_data = isempty(A_data) ? 0.0 : maximum(svdvals(A_data))
    if σ_data <= 0
        σ_data = isempty(F.S) ? 1.0 : F.S[1]
    end
    _PCLSFactorization(F, F.U' * z_aug, σ_data)
end

"""
    _pcls_truncated_step(fac)

Truncated-SVD least-squares solution of the factorized system: components
with `σ ≤ 1e-7·σ_max(W^½J)` are zeroed.

## Why truncate at all

The FD Jacobian is trajectory-local: at a poor initialization (e.g. the
`x -> 0` default, where the trajectory sits at u0) most basis columns are
numerically null, and a plain QR solve returns O(1e9) coefficients along
those directions — a step no contraction can rescue. Measured on the
exponential-growth fixture with `x -> 0` init and u0 = 1.1 placed OFF a knot
(so the dead columns are ~1e-10 rather than exactly 0), the eight data-block
singular values are 25.9 then 6.4e-11 … 1.0e-9: plain QR gives ‖β‖ = 2.27e9
(λ = 0) and 2.39e9 (λ = 1/tr(S)), this truncation 0.153 and 0.282.

## Why the reference is the DATA block

`σ_max` of the AUGMENTED matrix grows like √λ‖C‖ while `W^½J` stays O(1), so
referencing it makes the effective tolerance on data-informed directions
inflate with λ, and past a fixture-dependent λ it zeroes directions the data
actually determines. Those live in null(S) — exactly where a fit with
β'S_kβ = 0 sits. Measured on the null(S) exponential-growth fixture at
β = 0.10·1, ‖β̂ − β_true‖ under the augmented reference is 0.00164 at
λ ≤ 1e10, 0.113 at λ = 1e12 and 0.283 at every λ ≥ 1e13; under this data
reference it is 0.00164 at all nine decades (λ = 1e8 … 1e17).

Rank deficiency here is a statement about the data block's conditioning, and
the resulting retention bound is λ-free: for a retained direction the RHS
penalty block is zero, so |v_i'β| = |u_i'z_aug|/σ_i ≤ ‖W^½z‖/σ_i <
1e7·‖W^½z‖/σ_max(W^½J), the same bound the augmented reference gives at λ = 0.

The augmented reference was ALSO acting as an undocumented subspace trust
region at high λ, and simply dropping it regresses the nonlinear iteration.
That job now has its own explicit mechanism — see
[`_pcls_step_contract`](@ref).
"""
function _pcls_truncated_step(fac::_PCLSFactorization)
    F = fac.F
    isempty(F.S) && return zeros(size(F.V, 1))
    tol = 1e-7 * fac.sigma_data
    F.V * ([σ > tol ? 1.0 / σ : 0.0 for σ in F.S] .* fac.Utz)
end

"""
    _pcls_damped_step(fac, a_old, μ)

Levenberg-Marquardt step of radius control `μ`, taken on the INCREMENT:
minimize `‖W^½(z − Jβ)‖² + β'Bβ + μ²‖β − a_old‖²`, i.e.

    [W^½J; C; μI] (β − a_old) = [W^½(z − J·a_old); −C·a_old; 0].

From the factorization this is one filtered back-substitution,

    β = a_old + V·diag(σ/(σ²+μ²))·(U'z_aug − S·V'a_old),

so the whole ladder costs no additional decomposition. `μ = 0` reproduces
the untruncated least-squares solution exactly.

The damping is on the STEP, never on `β` itself, so a converged fit is a
fixed point for every `μ`; and `μ` is referenced to `σ_max(W^½J)` by the
caller, so it introduces no new λ-dependence.
"""
function _pcls_damped_step(fac::_PCLSFactorization, a_old::AbstractVector, μ::Real)
    F = fac.F
    isempty(F.S) && return copy(a_old)
    r = fac.Utz .- F.S .* (F.V' * a_old)
    a_old .+ F.V * ((F.S ./ (F.S .^ 2 .+ μ^2)) .* r)
end

"""
    _pcls_augmented_solve(J_mat, z_pseudo, B, w_irls)

Convenience wrapper: factorize and return the truncated step. Inside an IRLS
loop prefer keeping the [`_PCLSFactorization`](@ref) and handing it to
[`_pcls_step_contract`](@ref), so the trust region can reuse it.
"""
_pcls_augmented_solve(J_mat::AbstractMatrix, z_pseudo::AbstractVector,
                      B::AbstractMatrix, w_irls::AbstractVector) =
    _pcls_truncated_step(_pcls_factorize(J_mat, z_pseudo, B, w_irls))

# Trust-region controls.
#
# `_PCLS_STEP_REJECT` is when the region engages: the damping ladder is only
# built if the UNDAMPED full step multiplies the penalized objective by more
# than this — i.e. the Gauss-Newton direction is not merely imperfect but
# catastrophic, which is the only case backtracking cannot handle on its own.
# Measured f(α=1)/f(a_old) over six suite fixtures, across every PCLS step
# whose full step failed to improve at all:
#   Poisson count-data LAML fit           1.00 – 28.6   (65 of 146 steps)
#   kinked discrete-map LAML fit          1.04 – 1.87   (97 of 115)
#   unconstrained-GP LAML fit             1.00 – 1.07   ( 8 of  47)
#   two-λ GCV fit (:reuse equivalence)    1.04 – 2.73   ( 7 of  15)
#   Lotka-Volterra GCV fit, seed 1        52.4 – 2.5e4  (28 of  30)
#   LAML mixed spline+NN fit              96.7 – Inf    (29 of  29)
# The two fixtures that NEED damping sit above 52; the three Gaussian ones
# that must not move sit below 2.8.  (The Poisson fit reaches 28.6 but is
# indifferent: max spline error 0.06921 with the trust region against 0.06922
# without.)  10 is the geometric midpoint of the 2.73 … 52.4 gap.  Raising it
# to 30 breaks the Lotka-Volterra ensemble again (one seed back to data_loss
# 26.9); 5 also works.  That gap is a property of these six fixtures, NOT a
# universal separation: independent review probes on out-of-calibration
# fixtures land inside and above it (NegBin LAML 2.97–5.71, one benign
# firing; TruncatedNormal LAML up to 12.39, firing twice with the damped
# candidate losing both times, so the result was bit-identical to
# backtracking; SCOP-spline SIR 17.3/40.4/227, firing 6 times and moving
# λ̂ 9.65e-6 → 6.05e-7 while β̂(0.05) stays 12x inside its 0.08 suite gate).
# The constant's real defense is therefore not the separation but that the
# rule is FAIL-SAFE by construction: not firing returns exactly the
# backtracking result, and firing accepts a damped candidate only when it
# strictly lowers the penalized objective (then re-contracts it against
# f_old), so a threshold set too high only costs the fix, never
# correctness, and a firing on a healthy fit can only improve the
# objective it is scored on.
#
# `_PCLS_TRUST_TAUS` is a coarse decade ladder of damping strengths
# μ = τ·σ_max(W^½J).  It BRACKETS the damping rather than tuning it: the
# penalized objective picks among the candidates, so what matters is that the
# rungs span the useful range, not where exactly they sit.  Measured on the
# two fixtures that need it (12-seed Lotka-Volterra GCV ensemble, data_loss
# range; and the LAML mixed spline+NN fit, data_loss / max spline error):
#   (1e-1, 1e-2, 1e-3)                        0.187–2.259   0.902 / 0.0073
#   (1e-1, 10^-1.5, 1e-2, 10^-2.5, 1e-3)      0.187–2.259   1.05  / 0.0032
#   (1e-1, 1e-3)         — middle rung gone   0.187–2.259   83.4  / 0.616 ✗
#   (3e-2,)              — single rung        0.187–2.195   1.12  / 0.0037
# Halving the rung spacing changes nothing material; leaving a two-decade hole
# in the middle costs the mixed fit its 0.15 spline-error gate.  Three rungs
# is the cheapest spacing that keeps every gate.
const _PCLS_STEP_REJECT = 10.0
const _PCLS_TRUST_TAUS = (1e-1, 1e-2, 1e-3)

# Backtracking along one direction.  Returns (a_best, f_best, f_full), where
# `f_full` is the objective at the α = 1 step — the quantity the trust-region
# trigger tests.  It is deliberately NOT `f_best`: a contracted step is short
# enough that the linearization describes it well, so scoring the trigger
# there is blind to a useless direction (measured on the 3 stuck
# Lotka-Volterra seeds below: a trigger scored at the contracted step never
# fires on 2 of them, leaving data_loss at 29.0 and 27.4, and only partly
# rescues the third — 3.04 against the 1.64 the α = 1 trigger reaches).
function _pcls_contract_along(objective, a_old::AbstractVector,
                              direction::AbstractVector, B::AbstractMatrix,
                              f_old::Real)
    best_f = f_old
    best_a = copy(a_old)
    f_full = Inf

    for k in 0:15
        α = 2.0^(-k)
        a_try = a_old .+ α .* direction
        f_try = objective(a_try, B)
        k == 0 && (f_full = f_try)
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
                return a_try, f_try, f_full
            end
        end
    end
    best_a, best_f, f_full
end

"""
    _pcls_step_contract(objective, a_old, a_new, B)
    _pcls_step_contract(objective, a_old, a_new, B, fac)

Advance from `a_old` towards the PCLS solution `a_new`, scoring candidates
with `objective(a, B)` (the solver's penalized objective closure; must
return `Inf` on simulation failure). Returns `(a_best, f_best)`.

## Step contraction (both methods)

Backtracking with exponential step sizes along `a_new - a_old`. Phase 1
tries α = 1, 0.5, ..., 2^(-15) and keeps the best. Phase 2 rescues EXPLOSIVE
steps: when the trajectory-localized Jacobian is numerically rank-deficient
(e.g. every coefficient at the `x -> 0` default initialization), the PCLS
solution can be O(1e9) along near-null directions and even 2^(-15) of it
still blows up the ODE; continue halving with a first-improvement exit so
the fit can escape instead of silently rejecting every step.

## Trust region (the 5-argument method)

Given the [`_PCLSFactorization`](@ref), the step also gets an EXPLICIT
Levenberg-Marquardt trust region, because backtracking alone cannot fix it:
backtracking can only SHORTEN the Gauss-Newton direction, and at high λ that
direction is wrong, not merely long.

Measured on a 7-point/2-state Lotka-Volterra working model whose `δ` truth is
constant (hence in null(S), so GCV drives λ₂ to ~5e9), at the second PCLS
step, with the penalized objective at 14.904 before the step: the undamped
step has ‖s‖ = 3.88 and scores 350.6 at α = 1 — 23.5× worse than not
stepping — while backtracking recovers only 14.573. Damping the SOLVE instead scores 2.012
(τ = 0.1), 3.360 (τ = 0.01), 8.170 (τ = 0.001) at α = 1. Rescaling the
undamped DIRECTION down to the length the trust region actually takes scores
14.745, worse than what backtracking already found: the direction is wrong,
not merely long, which is why this cannot live in the line search. End to
end over 12 noise seeds of that fixture the undamped step leaves 3 seeds
stuck at data_loss 26.9–29.0 against 0.72–1.47 for the other 9; with this
trust region every seed lands in 0.19–2.26.

The region ENGAGES only on a catastrophic full step — `f(α=1)` more than
`_PCLS_STEP_REJECT` times `f(a_old)`, or non-finite. Merely-imperfect steps
are left to backtracking, which handles them; see the constant's comment for
the measured separation. Ordinary fits therefore pay nothing and, more
importantly, do not move: the damped candidates are a DISCRETE choice, and
letting them compete on near-ties turns a 1e-9 perturbation into a different
fit. Measured on the two-λ fixture that pins the GCVSolver `:reuse` ≡ direct
equivalence, engaging the region on every non-improving step breaks it —
|Δ log λ̂| 0 → 0.019 and max |Δ fitted| 0 → 1.0e-5, against gates of 1e-4 and
1e-6; with the `_PCLS_STEP_REJECT` trigger the two paths stay bit-identical
(both differences exactly 0), because the region never engages there (its
largest f(α=1)/f(a_old) over that testset's fits is 2.73; the equivalence
fit itself peaks at 1.07).

"""
function _pcls_step_contract(objective, a_old::AbstractVector,
                             a_new::AbstractVector, B::AbstractMatrix)
    f_old = objective(a_old, B)
    a_best, f_best, _ = _pcls_contract_along(objective, a_old, a_new .- a_old,
                                             B, f_old)
    a_best, f_best
end

function _pcls_step_contract(objective, a_old::AbstractVector,
                             a_new::AbstractVector, B::AbstractMatrix,
                             fac::_PCLSFactorization)
    f_old = objective(a_old, B)
    a_best, f_best, f_full = _pcls_contract_along(objective, a_old,
                                                  a_new .- a_old, B, f_old)

    # Written as a negated `<=` so a NaN or Inf `f_full` engages the region.
    # `f_old > 0` keeps the ratio meaningful: the Gaussian penalized objective
    # is ½RSS + ½β'Bβ ≥ 0, and for a family whose −log-likelihood can go
    # negative the conservative answer is to leave the step alone.
    (f_old > 0 && !(f_full <= _PCLS_STEP_REJECT * f_old)) &&
        return _pcls_damped_retry(objective, a_old, B, fac, a_best, f_best, f_old)
    a_best, f_best
end

# The linearization has failed outright along the Gauss-Newton direction.
# Damp the step and let the penalized objective pick the damping.
function _pcls_damped_retry(objective, a_old::AbstractVector, B::AbstractMatrix,
                            fac::_PCLSFactorization, a_best::AbstractVector,
                            f_best::Real, f_old::Real)
    cand_a = nothing
    cand_f = f_best
    for τ in _PCLS_TRUST_TAUS
        a_try = _pcls_damped_step(fac, a_old, τ * fac.sigma_data)
        f_try = objective(a_try, B)
        if f_try < cand_f
            cand_a = a_try
            cand_f = f_try
        end
    end
    cand_a === nothing && return a_best, f_best

    a_damp, f_damp, _ = _pcls_contract_along(objective, a_old, cand_a .- a_old,
                                             B, f_old)
    f_damp < f_best ? (a_damp, f_damp) : (a_best, f_best)
end
