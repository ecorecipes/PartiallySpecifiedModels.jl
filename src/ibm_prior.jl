# ─── Integrated Brownian Motion (IBM) Prior ──────────────────────────
#
# Provides the Gaussian process prior for the probabilistic ODE solver.
# Each state variable x(t) is modeled as q-times integrated Brownian motion:
#   x^(q)(t) = σ B(t)
# giving a (q+1)-dimensional state vector [x(t), x'(t), ..., x^(q-1)(t)].
#
# The IBM prior has a linear Gaussian state-space representation:
#   X_n = Q X_{n-1} + R^{1/2} ε_n
# with known Q (transition) and R (variance) matrices.
#
# Reference: Schober et al (2019), Wu & Lysy (2024)

"""
    ibm_state(dt, q, σ)

Compute the state transition matrix Q and variance matrix R for
q-times integrated Brownian motion with step size `dt` and scale `σ`.

State dimension is `q` (derivatives 0 through q-1).

Returns `(Q, R)` where:
- `Q[i,j] = dt^(j-i)/(j-i)!` for j ≥ i, 0 otherwise
- `R[i,j] = σ² dt^(2q-1-i-j) / ((2q-1-i-j) (q-1-i)! (q-1-j)!)`

Uses 0-based indexing internally to match the mathematical formulas,
but returns 1-based Julia matrices.
"""
function ibm_state(dt::Float64, q::Int, σ::Float64)
    n = q  # state dimension = q (derivatives 0..q-1)
    Q = zeros(n, n)
    R = zeros(n, n)

    for i in 0:n-1, j in 0:n-1
        # Transition matrix: Q[i,j] = dt^(j-i)/(j-i)! for j >= i
        if j >= i
            Q[i+1, j+1] = dt^(j - i) / factorial(j - i)
        end

        # Variance matrix: R[i,j] = σ² dt^(2q-1-i-j) / ((2q-1-i-j)(q-1-i)!(q-1-j)!)
        exp_val = 2 * q - 1 - i - j
        if exp_val > 0
            R[i+1, j+1] = σ^2 * dt^exp_val /
                (exp_val * factorial(q - 1 - i) * factorial(q - 1 - j))
        end
    end

    Q, R
end

"""
    ibm_init(dt, n_deriv, sigma)

Initialize the IBM prior parameters for n_vars independent state variables.

Args:
- `dt`: step size
- `n_deriv`: number of derivatives to track (q). State dimension per variable.
- `sigma`: vector of scale parameters, one per state variable (length n_vars)

Returns `(wgt_state, var_state)` where:
- `wgt_state[k]` is the q×q transition matrix for variable k
- `var_state[k]` is the q×q variance matrix for variable k
"""
function ibm_init(dt::Float64, n_deriv::Int, sigma::Vector{Float64})
    n_vars = length(sigma)
    Q_base, R_base = ibm_state(dt, n_deriv, 1.0)

    wgt_state = [copy(Q_base) for _ in 1:n_vars]
    var_state = [sigma[k]^2 .* R_base for k in 1:n_vars]

    wgt_state, var_state
end

"""
    first_order_init(ode_fun!, u0, t0, p, n_deriv)

Initialize the state for a first-order ODE system in rodeo format.

For a first-order ODE `dx/dt = f(x, t)`, the state for each variable is
`[x(t), f(x,t), 0, 0, ...]` — the value, its first derivative from the ODE,
and zeros for higher derivatives.

Args:
- `ode_fun!`: in-place ODE function `(du, u, p, t)`
- `u0`: initial values (length n_vars)
- `t0`: initial time
- `p`: parameter struct
- `n_deriv`: number of derivatives

Returns `X0` — vector of vectors, `X0[k]` is the q-dimensional initial state for variable k.
"""
function first_order_init(ode_fun!, u0::AbstractVector, t0::Float64, p, n_deriv::Int)
    n_vars = length(u0)
    du = zeros(n_vars)
    ode_fun!(du, u0, p, t0)

    X0 = [zeros(n_deriv) for _ in 1:n_vars]
    for k in 1:n_vars
        X0[k][1] = u0[k]       # x^(0) = initial value
        X0[k][2] = du[k]       # x^(1) = f(x, t)
        # higher derivatives zero-padded
    end

    X0
end

"""
    first_order_weight(n_vars, n_deriv)

Construct the W matrix that picks out the first derivative for a first-order ODE.

For WX = f(X, t), W selects x^(1) from the state vector [x^(0), x^(1), ..., x^(q-1)].
W is a 1×q matrix: [0, 1, 0, 0, ...] for each variable.

Returns vector of W matrices, one per variable.
"""
function first_order_weight(n_vars::Int, n_deriv::Int)
    W = zeros(1, n_deriv)
    W[1, 2] = 1.0  # select x^(1) (the first derivative)
    [copy(W) for _ in 1:n_vars]
end
