# ─── SciMLSensitivity extension: adjoint gradients for AdamSolver ────
#
# Implements PartiallySpecifiedModels._adam_adjoint_loss_grad, the opt-in
# `sensealg` gradient backend of AdamSolver. Loaded automatically when the
# user runs `using SciMLSensitivity` next to PartiallySpecifiedModels
# (Julia package extension; trigger = SciMLSensitivity only — the internal
# vector-Jacobian products use ReverseDiff, a hard dependency of
# SciMLSensitivity, so no separate `using Zygote` is required).
#
# Design: instead of reverse-differentiating a loss closure with Zygote
# (which would require a non-mutating rewrite of the whole accumulation),
# the AdamSolver loss families (:mse, :poisson) have CLOSED-FORM ∂g/∂u at
# each observation time, so we call `adjoint_sensitivities` directly with
# analytic `dgdu_discrete` callbacks. The masking (usable_cell), the
# penalty term (adam_penalty, with its analytic gradient), the loss=:auto
# dispatch, and the 1e10 failure sentinel all mirror the ForwardDiff path
# in src/adam_solver.jl exactly; equivalence is asserted by the parity
# tests in test/runtests.jl.
#
# The ODEProblem parameter object is the flat coefficient vector `beta`
# (adjoints require an array parameter); the dynamics wrapper rebuilds the
# callable-parameter NamedTuple via build_autodiff_param_struct inside the
# RHS, and the sensealg's autojacvec (default ReverseDiffVJP(false):
# re-taped every call, so state-dependent branches in spline evaluators
# stay correct) differentiates through evaluator construction + dynamics.

module PartiallySpecifiedModelsSciMLSensitivityExt

using PartiallySpecifiedModels
using SciMLSensitivity
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP,
    adjoint_sensitivities

const PSM = PartiallySpecifiedModels
const SciMLBase = PSM.SciMLBase
const OrdinaryDiffEq = PSM.OrdinaryDiffEq
const ForwardDiff = PSM.ForwardDiff

# ─── sensealg resolution ─────────────────────────────────────────

"""
    _resolve_sensealg(sensealg)

Map the user-facing `AdamSolver.sensealg` value to a concrete
SciMLSensitivity adjoint algorithm. `:auto` selects
`InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))`: interpolating
adjoints are the robust default for the small, moderately stiff systems
PSMs typically produce (BacksolveAdjoint is unstable on them), and the
non-compiled ReverseDiff tape is safe for the state-dependent branches
inside spline evaluators (knot-interval lookup), which a compiled tape
(`ReverseDiffVJP(true)`) would freeze.
"""
function _resolve_sensealg(sensealg::Symbol)
    sensealg === :auto &&
        return InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
    error("AdamSolver: unknown sensealg symbol $(repr(sensealg)); use " *
          ":auto or a concrete SciMLSensitivity adjoint algorithm such as " *
          "InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)). " *
          "(GaussAdjoint/QuadratureAdjoint were observed to mis-" *
          "differentiate spline-parameterized dynamics — validate against " *
          "ForwardDiff before trusting them; see the AdamSolver docs.)")
end
_resolve_sensealg(sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm) =
    sensealg
_resolve_sensealg(sensealg::SciMLBase.AbstractSensitivityAlgorithm) =
    error("AdamSolver: sensealg=$(typeof(sensealg)) is not an adjoint " *
          "method. The adjoint gradient backend needs a continuous " *
          "adjoint algorithm (InterpolatingAdjoint, GaussAdjoint, " *
          "QuadratureAdjoint, BacksolveAdjoint); forward sensitivity " *
          "algorithms are pointless here — for forward-mode gradients " *
          "use the default sensealg=nothing (ForwardDiff) path.")
_resolve_sensealg(sensealg) =
    error("AdamSolver: sensealg=$(repr(sensealg)) is not a valid adjoint " *
          "specification; use :auto or a SciMLSensitivity adjoint " *
          "algorithm object.")

# ─── Analytic ∂g/∂u callbacks (mirror adam_loss_* accumulation) ──

# G = Σᵢ gᵢ(u(tᵢ)) with gᵢ(u) = Σⱼ wᵢⱼ (u[skⱼ] − yᵢⱼ)² over usable cells.
function _dgdu_mse(prob::PSM.PSMProblem)
    (out, u, p, t, i) -> begin
        fill!(out, 0.0)
        for j in 1:size(prob.data_values, 2)
            PSM.usable_cell(prob, i, j) || continue
            sk = prob.obs_to_state[j]
            out[sk] += 2.0 * prob.data_weights[i, j] *
                       (u[sk] - prob.data_values[i, j])
        end
        nothing
    end
end

# gᵢ(u) = −Σⱼ wᵢⱼ (yᵢⱼ log μ − μ), μ = max(u[skⱼ], 1e-10). In the clamped
# region (u ≤ 1e-10) the derivative is 0, matching ForwardDiff through
# `max(pred, T(1e-10))`.
function _dgdu_poisson(prob::PSM.PSMProblem)
    (out, u, p, t, i) -> begin
        fill!(out, 0.0)
        for j in 1:size(prob.data_values, 2)
            PSM.usable_cell(prob, i, j) || continue
            sk = prob.obs_to_state[j]
            mu = u[sk]
            mu > 1e-10 || continue
            out[sk] -= prob.data_weights[i, j] *
                       (prob.data_values[i, j] / mu - 1.0)
        end
        nothing
    end
end

# ─── Loss value (identical accumulation order to the FD path) ────

function _adjoint_data_loss(prob::PSM.PSMProblem, us, loss_sym::Symbol)
    loss = 0.0
    n_obs = size(prob.data_values, 2)
    n_t = length(prob.data_times)
    for j in 1:n_obs
        sk = prob.obs_to_state[j]
        for i in 1:n_t
            PSM.usable_cell(prob, i, j) || continue
            pred = us[i][sk]
            y = prob.data_values[i, j]
            if loss_sym === :poisson
                mu = max(pred, 1e-10)
                loss -= prob.data_weights[i, j] * (y * log(mu) - mu)
            else
                loss += prob.data_weights[i, j] * (pred - y)^2
            end
        end
    end
    loss
end

# ─── Analytic penalty gradient (matches adam_penalty's value) ────

function _adjoint_penalty_grad!(grad::Vector{Float64}, prob::PSM.PSMProblem,
                                beta::AbstractVector, w::Float64)
    w == 0.0 && return grad
    off = 0
    for approx in prob.approximators
        np = PSM.nparams(approx)
        S = PSM.penalty_matrix(approx)
        if S !== nothing
            bk = view(beta, off+1:off+np)
            # ∇ᵦ w·βᵀSβ = w·(S + Sᵀ)β (exact also for non-symmetric S).
            grad[off+1:off+np] .+= w .* (S * bk .+ S' * bk)
        end
        off += np
    end
    grad
end

# ─── The backend ─────────────────────────────────────────────────

function PSM._adam_adjoint_loss_grad(prob::PSM.PSMProblem,
                                     beta::AbstractVector{Float64},
                                     loss_sym::Symbol, penalty_w::Float64,
                                     sensealg::Union{Symbol,
                                         SciMLBase.AbstractSensitivityAlgorithm})
    salg = _resolve_sensealg(sensealg)
    n_beta = length(beta)
    fail = () -> (1e10, zeros(n_beta))

    p_vec = collect(beta)
    p_struct = PSM.build_autodiff_param_struct(prob, p_vec)
    u0 = prob.u0 isa Function ? Float64.(collect(prob.u0(p_struct))) :
         Float64.(prob.u0)

    abstol = get(prob.ode_kwargs, :abstol, 1e-7)
    reltol = get(prob.ode_kwargs, :reltol, 1e-7)
    maxiters = get(prob.ode_kwargs, :maxiters, 10000)

    # Flat-vector parameters: the param NamedTuple (callable evaluators) is
    # rebuilt inside the RHS so the adjoint's vjp differentiates through it.
    ode_fn = SciMLBase.ODEFunction{true, SciMLBase.FullSpecialize}(
        (du, u, p, t) -> prob.dynamics!(
            du, u, PSM.build_autodiff_param_struct(prob, p), t))
    ode_prob = SciMLBase.ODEProblem(ode_fn, u0, prob.tspan, p_vec)

    # Dense forward solve (the continuous adjoint interpolates through it).
    sol = try
        OrdinaryDiffEq.solve(ode_prob, prob.ode_solver;
                             abstol=abstol, reltol=reltol, maxiters=maxiters)
    catch e
        PSM._is_program_error(e) && rethrow()
        return fail()
    end
    # Same success predicate as the ForwardDiff loss paths (Success only).
    (sol.retcode == SciMLBase.ReturnCode.Success || sol.retcode == :Success) ||
        return fail()

    us = sol(prob.data_times).u
    loss = _adjoint_data_loss(prob, us, loss_sym) +
           PSM.adam_penalty(prob, beta, penalty_w)
    isfinite(loss) || return fail()

    dgdu = loss_sym === :poisson ? _dgdu_poisson(prob) : _dgdu_mse(prob)

    du0, dp = try
        adjoint_sensitivities(sol, prob.ode_solver;
                              t=prob.data_times, dgdu_discrete=dgdu,
                              sensealg=salg,
                              abstol=abstol, reltol=reltol)
    catch e
        PSM._is_program_error(e) && rethrow()
        # Failed backward pass: same large-but-finite sentinel (with a flat
        # gradient) the forward failure produces, so Adam's moments can
        # carry the iterate out of the failing region.
        return fail()
    end

    grad = vec(collect(dp'))
    length(grad) == n_beta ||
        error("adjoint_sensitivities returned a gradient of length " *
              "$(length(grad)) for $(n_beta) parameters")
    # SciMLSensitivity can also fail SOFTLY: on a blow-up-adjacent iterate
    # the backward pass warns (dt = NaN) and returns without throwing. Any
    # non-finite entry must hit the sentinel, not Adam's moments (a single
    # NaN would poison them permanently). Known residual limitation: a soft
    # failure that returns all-ZERO gradients is indistinguishable from a
    # genuine stationary point and passes this guard — the retcode of the
    # internal backward solution is not exposed to check.
    all(isfinite, grad) || return fail()
    _adjoint_penalty_grad!(grad, prob, beta, penalty_w)

    # u0 = u0(p): chain dG/du0 back to beta. Cheap forward-mode Jacobian of
    # the initial-condition map only (NOT through the ODE solve).
    if prob.u0 isa Function
        J = ForwardDiff.jacobian(
            b -> collect(prob.u0(PSM.build_autodiff_param_struct(prob, b))),
            p_vec)
        grad .+= J' * vec(collect(du0))
    end

    (loss, grad)
end

end # module
