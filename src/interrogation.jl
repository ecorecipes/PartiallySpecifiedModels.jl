# ─── ODE Interrogation Methods ───────────────────────────────────────
#
# At each time step of the probabilistic solver, we "interrogate" the ODE
# by evaluating its RHS and constructing a pseudo-observation:
#   Z_n = W X_n - f(X_n, t_n) ≈ 0
#
# Different methods linearize/approximate this differently:
# - Schober: evaluate f at predicted mean (deterministic)
# - Kramer: first-order Taylor expansion (better numerical stability)
#
# All return (wgt_meas, mean_meas, var_meas) defining the linearized
# measurement equation: Z_n = wgt_meas * X_n + mean_meas + V^{1/2} η
#
# Reference: Schober et al (2019), Kramer et al (2021)

"""
    interrogate_schober(ode_fun!, W, t, μ_pred, Σ_pred, p, n_vars)

Schober et al (2019) interrogation: evaluate ODE at predicted mean.

The measurement is Z_n = W·X_n - f(μ_pred, t) with zero variance.
This is the simplest method — no additional measurement weight or variance.
"""
function interrogate_schober(ode_fun!, W_list, t::Float64,
                             μ_pred::Vector{Vector{Float64}},
                             Σ_pred::Vector{Matrix{Float64}},
                             p, n_vars::Int)
    n_deriv = length(μ_pred[1])
    n_meas = size(W_list[1], 1)

    # Reconstruct state values for ODE evaluation
    u = [μ_pred[k][1] for k in 1:n_vars]
    du = zeros(n_vars)
    ode_fun!(du, u, p, t)

    wgt_meas = [zeros(n_meas, n_deriv) for _ in 1:n_vars]
    mean_meas = [zeros(n_meas) for _ in 1:n_vars]
    var_meas = [zeros(n_meas, n_meas) for _ in 1:n_vars]

    for k in 1:n_vars
        mean_meas[k] .= -reshape([du[k]], :)
    end

    wgt_meas, mean_meas, var_meas
end

"""
    interrogate_kramer(ode_fun!, W_list, t, μ_pred, Σ_pred, p, n_vars)

Kramer et al (2021) interrogation: first-order Taylor linearization.

Linearizes f(X, t) around the predicted mean via Jacobian:
  f(X, t) ≈ f(μ, t) + J(μ, t)(X - μ)

The measurement equation becomes:
  Z = (W - J)X + (Jμ - f(μ, t)) ≈ 0
so wgt_meas = -J, mean_meas = Jμ - f(μ, t)

This is more numerically stable than Schober for stiff/nonlinear systems.
"""
function interrogate_kramer(ode_fun!, W_list, t::Float64,
                            μ_pred::Vector{Vector{Float64}},
                            Σ_pred::Vector{Matrix{Float64}},
                            p, n_vars::Int)
    n_deriv = length(μ_pred[1])
    n_meas = size(W_list[1], 1)

    # Evaluate ODE at predicted mean
    u = [μ_pred[k][1] for k in 1:n_vars]
    du = zeros(n_vars)
    ode_fun!(du, u, p, t)

    # Compute Jacobian ∂f/∂u via finite differences
    ε = 1e-7
    J = zeros(n_vars, n_vars)
    for j in 1:n_vars
        u_pert = copy(u)
        u_pert[j] += ε
        du_pert = zeros(n_vars)
        ode_fun!(du_pert, u_pert, p, t)
        J[:, j] .= (du_pert .- du) ./ ε
    end

    # Build per-variable measurement parameters.
    #
    # The Kalman backend propagates one IBM block per state variable, so each
    # update can only act directly on x_k and its derivatives. We therefore:
    #   1. keep the within-block Jacobian term -J[k,k] in the measurement weight
    #   2. retain the full Jacobian row in the measurement offset J[k,:]u - f_k(u,t)
    #   3. fold off-diagonal state uncertainty into additional measurement noise
    #      using the predicted marginal variances of the coupled states
    wgt_meas = [zeros(n_meas, n_deriv) for _ in 1:n_vars]
    mean_meas = [zeros(n_meas) for _ in 1:n_vars]
    var_meas = [zeros(n_meas, n_meas) for _ in 1:n_vars]

    for k in 1:n_vars
        wgt_meas[k][1, 1] = -J[k, k]
        mean_meas[k][1] = dot(J[k, :], u) - du[k]
        offdiag_var = 0.0
        for j in 1:n_vars
            j == k && continue
            offdiag_var += J[k, j]^2 * max(Σ_pred[j][1, 1], 0.0)
        end
        var_meas[k][1, 1] = offdiag_var
    end

    wgt_meas, mean_meas, var_meas
end
