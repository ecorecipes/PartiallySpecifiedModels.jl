# ─── Integrated Brownian Motion (IBM) Prior ──────────────────────────
#
# Provides the Gaussian process prior for the probabilistic ODE solver.
# Each state variable x(t) is modeled as q-times integrated Brownian motion:
#   x^(q)(t) = σ B(t)
# giving a q-dimensional state block [x(t), x'(t), ..., x^(q-1)(t)].
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



