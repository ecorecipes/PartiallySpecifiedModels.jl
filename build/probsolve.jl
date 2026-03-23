# ─── Probabilistic ODE Solver ────────────────────────────────────────
#
# Solves an ODE initial value problem using a Kalman filter/smoother
# approach with an integrated Brownian motion prior.
#
# Forward pass: Kalman filter with interrogation at each time step
# Backward pass: RTS smoother for posterior mean and variance
#
# The ODE residual W·X - f(X,t) = 0 is treated as a pseudo-observation
# at each time step, conditioning the Gaussian process prior on the ODE.
#
# Reference: Tronarp et al (2018), Schober et al (2019), Wu & Lysy (2024)

"""
    probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                     interrogate=:kramer)

Forward (filtering) pass of the probabilistic ODE solver.

Returns a dictionary with:
- `μ_pred`: predicted means [n_steps+1][n_vars][n_deriv]
- `Σ_pred`: predicted variances [n_steps+1][n_vars][n_deriv × n_deriv]
- `μ_filt`: filtered means
- `Σ_filt`: filtered variances
- `times`: time grid
"""
function probsolve_filter(ode_fun!, p, u0::AbstractVector,
                          tspan::Tuple{Float64, Float64},
                          n_steps::Int, n_deriv::Int,
                          sigma::Vector{Float64};
                          interrogate::Symbol=:kramer)
    t_min, t_max = tspan
    dt = (t_max - t_min) / n_steps
    n_vars = length(u0)

    # IBM prior
    wgt_state, var_state = ibm_init(dt, n_deriv, sigma)
    W_list = first_order_weight(n_vars, n_deriv)

    # Initial state
    X0 = first_order_init(ode_fun!, Float64.(u0), t_min, p, n_deriv)

    # Allocate storage
    μ_pred = Vector{Vector{Vector{Float64}}}(undef, n_steps + 1)
    Σ_pred = Vector{Vector{Matrix{Float64}}}(undef, n_steps + 1)
    μ_filt = Vector{Vector{Vector{Float64}}}(undef, n_steps + 1)
    Σ_filt = Vector{Vector{Matrix{Float64}}}(undef, n_steps + 1)

    # Initial: known exactly (zero variance)
    μ_pred[1] = X0
    Σ_pred[1] = [zeros(n_deriv, n_deriv) for _ in 1:n_vars]
    μ_filt[1] = X0
    Σ_filt[1] = [zeros(n_deriv, n_deriv) for _ in 1:n_vars]

    # Select interrogation method
    interrogate_fn = if interrogate == :kramer
        interrogate_kramer
    else
        interrogate_schober
    end

    # Forward pass
    z_meas = zeros(1)  # pseudo-observation is always 0 (ODE residual = 0)

    for n in 1:n_steps
        t_n = t_min + (t_max - t_min) * n / n_steps

        # Predict (per variable)
        μ_p = Vector{Vector{Float64}}(undef, n_vars)
        Σ_p = Vector{Matrix{Float64}}(undef, n_vars)
        for k in 1:n_vars
            μ_p[k], Σ_p[k] = kalman_predict(μ_filt[n][k], Σ_filt[n][k],
                                              wgt_state[k], var_state[k])
        end

        # Interrogate ODE
        wgt_m, mean_m, var_m = interrogate_fn(ode_fun!, W_list, t_n, μ_p, Σ_p, p, n_vars)

        # Update (per variable)
        μ_f = Vector{Vector{Float64}}(undef, n_vars)
        Σ_f = Vector{Matrix{Float64}}(undef, n_vars)
        for k in 1:n_vars
            # Total measurement weight: W + wgt_meas
            W_total = W_list[k] + wgt_m[k]
            μ_f[k], Σ_f[k] = kalman_update(μ_p[k], Σ_p[k],
                                             z_meas, mean_m[k],
                                             W_total, var_m[k])
        end

        μ_pred[n+1] = μ_p
        Σ_pred[n+1] = Σ_p
        μ_filt[n+1] = μ_f
        Σ_filt[n+1] = Σ_f
    end

    times = collect(range(t_min, t_max, length=n_steps + 1))

    Dict("μ_pred" => μ_pred, "Σ_pred" => Σ_pred,
         "μ_filt" => μ_filt, "Σ_filt" => Σ_filt,
         "times" => times, "wgt_state" => wgt_state, "var_state" => var_state)
end

"""
    probsolve_smooth(filt_out, n_vars)

Backward (smoothing) pass: RTS smoother on the filter output.

Returns `(μ_smooth, Σ_smooth)` — posterior mean and variance at all time points.
"""
function probsolve_smooth(filt_out::Dict, n_vars::Int)
    μ_filt = filt_out["μ_filt"]
    Σ_filt = filt_out["Σ_filt"]
    μ_pred = filt_out["μ_pred"]
    Σ_pred = filt_out["Σ_pred"]
    wgt_state = filt_out["wgt_state"]

    n_steps = length(μ_filt) - 1

    # Initialize smooth = filt at terminal time
    μ_smooth = Vector{Vector{Vector{Float64}}}(undef, n_steps + 1)
    Σ_smooth = Vector{Vector{Matrix{Float64}}}(undef, n_steps + 1)
    μ_smooth[n_steps+1] = μ_filt[n_steps+1]
    Σ_smooth[n_steps+1] = Σ_filt[n_steps+1]

    # Backward pass
    for n in n_steps:-1:1
        μ_s = Vector{Vector{Float64}}(undef, n_vars)
        Σ_s = Vector{Matrix{Float64}}(undef, n_vars)
        for k in 1:n_vars
            μ_s[k], Σ_s[k] = kalman_smooth_mv(
                μ_smooth[n+1][k], Σ_smooth[n+1][k],
                μ_filt[n][k], Σ_filt[n][k],
                μ_pred[n+1][k], Σ_pred[n+1][k],
                wgt_state[k]
            )
        end
        μ_smooth[n] = μ_s
        Σ_smooth[n] = Σ_s
    end

    μ_smooth, Σ_smooth
end

"""
    probsolve(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
              interrogate=:kramer)

Probabilistic ODE solver: returns posterior mean and variance of the solution.

Args:
- `ode_fun!`: in-place ODE function (du, u, p, t)
- `p`: parameter struct
- `u0`: initial values
- `tspan`: (t_min, t_max)
- `n_steps`: number of solver steps
- `n_deriv`: number of derivatives in IBM prior (q)
- `sigma`: IBM scale parameters (one per variable)
- `interrogate`: `:kramer` or `:schober`

Returns `(μ_smooth, Σ_smooth, times)` where:
- `μ_smooth[n][k]`: posterior mean of variable k at time n (vector of length n_deriv)
- `Σ_smooth[n][k]`: posterior covariance of variable k at time n (n_deriv × n_deriv)
- `times`: time grid
"""
function probsolve(ode_fun!, p, u0::AbstractVector,
                   tspan::Tuple{Float64, Float64},
                   n_steps::Int, n_deriv::Int,
                   sigma::Vector{Float64};
                   interrogate::Symbol=:kramer)
    n_vars = length(u0)

    filt_out = probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                 interrogate=interrogate)
    μ_smooth, Σ_smooth = probsolve_smooth(filt_out, n_vars)

    μ_smooth, Σ_smooth, filt_out["times"]
end

"""
    basic_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                 obs_data, obs_times, obs_to_state, obs_var;
                 interrogate=:kramer)

Basic likelihood approximation: solve ODE probabilistically, then
evaluate data likelihood at posterior mean.

p(Y | θ) ≈ Π_i N(y_i; μ_smooth(t_i), obs_var)

Returns the log-likelihood value.
"""
function basic_loglik(ode_fun!, p, u0::AbstractVector,
                      tspan::Tuple{Float64, Float64},
                      n_steps::Int, n_deriv::Int,
                      sigma::Vector{Float64},
                      obs_data::Matrix{Float64},
                      obs_times::Vector{Float64},
                      obs_to_state::Vector{Int},
                      obs_var::Float64;
                      interrogate::Symbol=:kramer)
    μ_smooth, _, times = probsolve(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                    interrogate=interrogate)

    n_obs = size(obs_data, 2)
    n_t = size(obs_data, 1)
    ll = 0.0

    for i in 1:n_t
        # Find closest solver time point
        idx = searchsortedfirst(times, obs_times[i])
        idx = clamp(idx, 1, length(times))

        for j in 1:n_obs
            sk = obs_to_state[j]
            pred = μ_smooth[idx][sk][1]  # zeroth derivative = function value
            ll += -0.5 * log(2π * obs_var) - 0.5 * (obs_data[i, j] - pred)^2 / obs_var
        end
    end

    ll
end

"""
    fenrir_loglik(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma,
                  obs_data, obs_times, obs_to_state, obs_weight, obs_var_mat;
                  interrogate=:kramer)

Fenrir likelihood approximation (Tronarp et al 2022):
Forward pass with ODE interrogation, then backward pass conditioning on data.

The observation model is: Y_m = D_m X_m + Ω^{1/2} η_m
where D_m picks out the zeroth derivative of observed state variables.

Returns the approximate marginal log-likelihood.
"""
function fenrir_loglik(ode_fun!, p, u0::AbstractVector,
                       tspan::Tuple{Float64, Float64},
                       n_steps::Int, n_deriv::Int,
                       sigma::Vector{Float64},
                       obs_data::Matrix{Float64},
                       obs_times::Vector{Float64},
                       obs_to_state::Vector{Int},
                       obs_var::Float64;
                       interrogate::Symbol=:kramer)
    n_vars = length(u0)
    n_obs_vars = length(obs_to_state)

    # Forward pass
    filt_out = probsolve_filter(ode_fun!, p, u0, tspan, n_steps, n_deriv, sigma;
                                 interrogate=interrogate)
    μ_filt = filt_out["μ_filt"]
    Σ_filt = filt_out["Σ_filt"]
    μ_pred = filt_out["μ_pred"]
    Σ_pred = filt_out["Σ_pred"]
    wgt_state = filt_out["wgt_state"]
    var_state = filt_out["var_state"]
    times = filt_out["times"]

    n_t_obs = size(obs_data, 1)

    # Map observation times to solver grid indices
    obs_ind = [searchsortedfirst(times, obs_times[i]) for i in 1:n_t_obs]
    obs_ind = clamp.(obs_ind, 1, n_steps + 1)

    # Build observation matrices per observed variable
    # D picks out the zeroth derivative of each observed state
    D = zeros(1, n_deriv)
    D[1, 1] = 1.0
    obs_var_mat = fill(obs_var, 1, 1)

    # Backward pass with data conditioning
    # Start from terminal filter state
    logdens = 0.0
    obs_ptr = n_t_obs  # pointer into observation array (backwards)

    # Initialize backward state with terminal filter
    bμ = [copy(μ_filt[n_steps+1][k]) for k in 1:n_vars]
    bΣ = [copy(Σ_filt[n_steps+1][k]) for k in 1:n_vars]

    # Check terminal observation
    if obs_ptr >= 1 && obs_ind[obs_ptr] >= n_steps + 1
        for j in 1:n_obs_vars
            k = obs_to_state[j]
            μ_f, Σ_f = kalman_forecast(bμ[k], bΣ[k], zeros(1), D, obs_var_mat)
            logdens += logpdf_mvn([obs_data[obs_ptr, j]], μ_f, Σ_f)
            bμ[k], bΣ[k] = kalman_update(bμ[k], bΣ[k],
                                           [obs_data[obs_ptr, j]], zeros(1),
                                           D, obs_var_mat)
        end
        obs_ptr -= 1
    end

    # Backward sweep
    for n in n_steps:-1:1
        # Compute backward Markov parameters (smooth conditional)
        for k in 1:n_vars
            G = Σ_filt[n][k] * wgt_state[k]' / Σ_pred[n+1][k]
            bμ_pred = G * bμ[k] + (μ_filt[n][k] - G * μ_pred[n+1][k])
            bΣ_pred = G * bΣ[k] * G' + (Σ_filt[n][k] - G * Σ_filt[n][k] * wgt_state[k]' / Σ_pred[n+1][k] * wgt_state[k] * Σ_filt[n][k])
            bΣ_pred = 0.5 * (bΣ_pred + bΣ_pred')
            bμ[k] = bμ_pred
            bΣ[k] = bΣ_pred
        end

        # Check if this time has an observation
        if obs_ptr >= 1 && obs_ind[obs_ptr] == n
            for j in 1:n_obs_vars
                k = obs_to_state[j]
                μ_f, Σ_f = kalman_forecast(bμ[k], bΣ[k], zeros(1), D, obs_var_mat)
                logdens += logpdf_mvn([obs_data[obs_ptr, j]], μ_f, Σ_f)
                bμ[k], bΣ[k] = kalman_update(bμ[k], bΣ[k],
                                               [obs_data[obs_ptr, j]], zeros(1),
                                               D, obs_var_mat)
            end
            obs_ptr -= 1
        end
    end

    logdens
end
