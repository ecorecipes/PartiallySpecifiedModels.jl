function _laml_smoothing_state(prob, beta, lambda, S_list, offsets, sizes,
                               J, W, y, fitted, information, rss, ridge; jac)
    ranks = _rank_penalty.(S_list)
    residual_dof = _n_usable(y, W) -
        _restricted_dof_Mp(J, W, offsets, sizes, length(beta), sum(ranks))
    B = build_S_lambda(S_list, offsets, sizes, log.(lambda), length(beta))
    (; beta=copy(beta), lambda=copy(lambda), penalties=copy.(S_list),
       offsets=copy(offsets), sizes=copy(sizes), ranks,
       information=copy(information), rss, residual_dof, covariance_ridge=ridge,
       coefficient_score=J' * (W .* (fitted - y)) + B * beta,
       names=Tuple(a.name for a in prob.approximators),
       data_weights=copy(prob.data_weights), jac)
end

function _smoothing_cholesky(H)
    all(isfinite, H) ||
        throw(DomainError(H, "smoothing covariance: non-finite coefficient information"))
    allowance = 64size(H, 1) * eps(Float64) * maximum(abs, H)
    maximum(abs, H - H') <= allowance ||
        throw(DomainError(H, "smoothing covariance: coefficient information is not symmetric"))
    try
        cholesky(Symmetric(H))
    catch e
        e isa PosDefException || rethrow()
        throw(DomainError(H, "smoothing covariance: coefficient information is not positive " *
            "definite; a numerical ridge must not identify an otherwise unidentified model"))
    end
end

function _smoothing_derivatives(beta, H, penalties; covariance_ridge=0.0)
    F = _smoothing_cholesky(H)
    sensitivity = -(F \ hcat((B * beta for B in penalties)...))
    Hc = copy(H)
    Hc[diagind(Hc)] .+= covariance_ridge
    Fc = _smoothing_cholesky(Hc)
    T = inv(Fc.U)
    derivatives = Matrix{Float64}[]
    for B in penalties
        # Hc = R'R, T = inv(R), C = sigma2*T*T'. Differentiate this
        # inverse-precision root, not a Cholesky factor of the penalty.
        E = Matrix(UpperTriangular(T' * B * T))
        E[diagind(E)] ./= 2
        push!(derivatives, -T * E)
    end
    (; sensitivity, root_derivatives=derivatives, conditional_inverse=Matrix(inv(Fc)))
end

function _smoothing_profile_curvature(beta, H, penalties, ranks, sensitivity,
                                      rss, n_eff; determinant_ridge=0.0)
    n_eff > 0 || throw(DomainError(n_eff,
        "smoothing covariance: positive restricted residual degrees of freedom are required"))
    u = [B * beta for B in penalties]
    q = [dot(beta, v) for v in u]
    Q = rss + sum(q)
    profile_sigma2 = Q / n_eff
    isfinite(profile_sigma2) && profile_sigma2 > 1e-30 ||
        throw(DomainError(profile_sigma2,
            "smoothing covariance: the profiled REML scale is at its numerical floor or invalid"))
    Hd = copy(H)
    Hd[diagind(Hd)] .+= determinant_ridge
    Fd = _smoothing_cholesky(Hd)
    P = [Fd \ B for B in penalties]
    traces = tr.(P)
    m = length(penalties)
    hessian = zeros(m, m)
    for j in 1:m, k in j:m
        # Differentiate the profiled coefficient optimum AND the REML
        # scale. laml_hessian's expected/partial Hessian omits these terms.
        qjk = (j == k ? q[j] : 0.0) + 2dot(u[j], view(sensitivity, :, k))
        tjk = (j == k ? traces[j] : 0.0) - tr(P[j] * P[k])
        hessian[j, k] = (n_eff * (qjk / Q - (q[j] / Q) * (q[k] / Q)) + tjk) / 2
        hessian[k, j] = hessian[j, k]
    end
    gradient = (ranks - traces - n_eff .* (q ./ Q)) / 2
    (; hessian, gradient, profile_sigma2)
end

function _smoothing_rho_covariance(hessian, regularization)
    isfinite(regularization) && regularization >= 0 ||
        throw(ArgumentError("smoothing covariance: rho_regularization must be finite and nonnegative"))
    all(isfinite, hessian) ||
        throw(DomainError(hessian, "smoothing covariance: non-finite log-smoothing Hessian"))
    eig = eigen(Symmetric(hessian))
    adjusted = eig.values .+ regularization
    allowance = 64length(adjusted) * eps(Float64) * max(1.0, maximum(abs, adjusted))
    minimum(adjusted) > allowance ||
        throw(DomainError(eig.values, "smoothing covariance: log-smoothing curvature is not " *
            "numerically positive definite; use an explicit rho_regularization or a supplied " *
            "rho_covariance if scientifically justified (no conditional-covariance fallback)"))
    root = eig.vectors .* reshape(1 ./ sqrt.(adjusted), 1, :)
    (; covariance=root * root', eigenvalues=eig.values)
end

function _smoothing_delta_covariance(sensitivity, derivatives, sigma2, rho_covariance)
    root = _band_covariance_root(rho_covariance; context="smoothing covariance")
    mean_factor = sensitivity * root
    mean_correction = mean_factor * mean_factor'
    root_correction = zeros(size(mean_correction))
    for l in axes(root, 2)
        factor = zeros(size(root_correction))
        for k in eachindex(derivatives)
            factor .+= root[k, l] .* derivatives[k]
        end
        root_correction .+= sigma2 .* (factor * factor')
    end
    (; mean_correction, root_correction)
end

"""
    smoothing_covariance_correction(sol, prob;
                                    rho_covariance=nothing, rho_regularization=0.0)

Compute the Wood--Pya--Saefken (2016, equation 7) smoothing-uncertainty
correction for a **Gaussian LAML local working model**:

    C_corrected = C_conditional + D * V_rho * D' + C_root

Here `rho = log(lambda)`, `D = d(beta_hat)/d(rho)`, and `C_root` contracts
derivatives of the inverse Cholesky factor of the coefficient precision
with `V_rho`. Both terms are included; this is not the mean-only
Kass--Steffey approximation.

Returns a NamedTuple containing the already SCALED `covariance`,
`conditional_covariance`, `mean_correction`, `root_correction`,
`rho_covariance`, `rho_hessian` (negative profiled log-REML curvature),
`rho_gradient` (log-REML gradient), `coefficient_sensitivity`,
unscaled `root_derivatives`, and provenance/regularity diagnostics.
Do not multiply `covariance` by `sigma2` again.

By default, invert analytic profiled-REML curvature, including the response
of the coefficient optimum and of the profiled scale. The final prediction
Jacobian/weights are frozen: this is exact algebra for the local linear
Gaussian model, NOT the full higher derivatives of a nonlinear ODE fit.
It uses final working information saved by current `LAML` fits, without
re-simulating or requiring a different Jacobian backend. Other solvers,
non-Gaussian likelihoods, old fits without that information, fixed or
unadvanced smoothing, and smoothing at an optimization bound are rejected.

`rho_regularization` is an explicit, isotropic, covariance-only precision
added to the log-smoothing Hessian; it does not refit the smoothing mean.
The default is zero: flat/indefinite curvature raises a `DomainError`,
never a silent conditional-covariance fallback or eigenvalue clipping.
This follows equation 7 with ONE `V_rho` for both additions, not mgcv's
separate, implicitly regularized covariance for the root term.

Alternatively, supply a finite PSD `rho_covariance`, in the same penalty
block order as `sol.smoothing_params`. This also supports fixed/external
smoothing: no smoothing covariance is inferred from a fixed-lambda fit.
For a predeclared multiplicative undersmoothing factor, propagate the
selection-stage log-lambda covariance explicitly and evaluate this helper
at the final refit. A supplied covariance cannot be combined with
`rho_regularization`; its estimation and ordering are the caller's responsibility.

The coefficient dispersion `sol.convergence.sigma2` is held fixed when
differentiating the covariance root; uncertainty in dispersion, GP kernel
parameters, architecture, grids and model selection is NOT integrated.
The REML-profile scale is used only for log-smoothing curvature, not to
rescale the reported conditional covariance. The fit's numerical precision
and log-determinant ridges are retained but frozen when differentiated.
The coefficient optimum must be identified WITHOUT a numerical ridge.

Read `coefficient_score`, `coefficient_step`, `rho_gradient`, the original
`stationarity` and convergence flags: no universal stationarity threshold
is imposed. Local normality and a sufficiently stationary coefficient and
smoothing fit are assumptions, not consequences of obtaining a matrix.
This is neither a smoothing-bias correction nor a coverage guarantee.
`prob` must describe the original fit with unchanged approximators/data.
"""
function smoothing_covariance_correction(sol::PSMSolution, prob::PSMProblem;
                                          rho_covariance=nothing,
                                          rho_regularization::Real=0.0)
    prob.likelihood isa Gaussian ||
        throw(ArgumentError("smoothing covariance: only Gaussian LAML working models are supported"))
    c = sol.convergence
    c !== nothing && get(c, :solver, nothing) === :LAML &&
        get(c, :smoothing_state, nothing) !== nothing ||
        throw(ArgumentError("smoothing covariance: a current Gaussian LAML fit with saved " *
            "working information and at least one penalty block is required"))
    s = c.smoothing_state
    n = length(sol.parameters)
    c.V_beta isa AbstractMatrix ||
        throw(ArgumentError("smoothing covariance: a coefficient covariance is required"))
    n == n_total_params(prob) && size(c.V_beta) == (n, n) ||
        throw(DimensionMismatch("smoothing covariance: parameter/covariance dimensions differ"))
    isequal(collect(sol.parameters), s.beta) && isequal(sol.smoothing_params, s.lambda) ||
        throw(ArgumentError("smoothing covariance: fitted coefficients or smoothing parameters have changed"))
    isequal(sol.data_values, prob.data_values) && isequal(sol.data_times, prob.data_times) &&
        isequal(s.data_weights, prob.data_weights) &&
        s.names == Tuple(a.name for a in prob.approximators) ||
        throw(ArgumentError("smoothing covariance: the problem does not match the fitted data/parameter layout"))
    S_list, offsets, sizes = build_penalty_matrices(prob)
    isequal((S_list, offsets, sizes), (s.penalties, s.offsets, s.sizes)) ||
        throw(ArgumentError("smoothing covariance: the problem's penalty definitions have changed"))
    isfinite(rho_regularization) && rho_regularization >= 0 ||
        throw(ArgumentError("smoothing covariance: rho_regularization must be finite and nonnegative"))
    rho_covariance === nothing || iszero(rho_regularization) ||
        throw(ArgumentError("smoothing covariance: use rho_covariance or rho_regularization, not both"))
    all(isfinite, s.beta) && all(x -> isfinite(x) && x > 0, s.lambda) &&
        isfinite(c.sigma2) && c.sigma2 >= 0 ||
        throw(DomainError(c.sigma2, "smoothing covariance: invalid coefficients, smoothing or dispersion"))
    m = length(s.lambda)
    if rho_covariance === nothing
        !c.smoothing_fixed && c.smoothing_advanced ||
            throw(ArgumentError("smoothing covariance: fixed or unadvanced smoothing requires an " *
                "explicit externally estimated rho_covariance"))
        all(x -> RHO_MIN < log(x) < RHO_MAX, s.lambda) ||
            throw(DomainError(s.lambda, "smoothing covariance: a smoothing estimate is at an " *
                "optimization bound; an interior Gaussian approximation is not available"))
    else
        rho_covariance isa AbstractMatrix{<:Real} ||
            throw(ArgumentError("smoothing covariance: rho_covariance must be a real matrix"))
        size(rho_covariance) == (m, m) ||
            throw(DimensionMismatch("smoothing covariance: rho_covariance must match the penalty block count"))
    end

    penalties = Matrix{Float64}[]
    H = copy(s.information)
    for k in 1:m
        B = zeros(n, n)
        ids = (s.offsets[k] + 1):(s.offsets[k] + s.sizes[k])
        B[ids, ids] .= s.lambda[k] .* s.penalties[k]
        push!(penalties, B)
        H .+= B
    end
    derivatives = _smoothing_derivatives(s.beta, H, penalties;
                                         covariance_ridge=s.covariance_ridge)
    # Keep the original conditional covariance and scale, but refuse a
    # modified/stale covariance instead of correcting a different fit.
    expected = derivatives.conditional_inverse
    allowance = 64n * eps(Float64) * maximum(abs, expected)
    all(isfinite, c.V_beta) && maximum(abs, c.V_beta - expected) <= allowance ||
        throw(DomainError(c.V_beta, "smoothing covariance: stored covariance differs from the final working model"))
    determinant_ridge = 1e-10 * maximum(abs, diag(H)) + 1e-15
    profile = nothing
    eigenvalues = nothing
    Vrho = if rho_covariance === nothing
        profile = _smoothing_profile_curvature(s.beta, H, penalties, s.ranks,
            derivatives.sensitivity, s.rss, s.residual_dof; determinant_ridge)
        estimated = _smoothing_rho_covariance(profile.hessian, Float64(rho_regularization))
        eigenvalues = estimated.eigenvalues
        estimated.covariance
    else
        Matrix{Float64}(rho_covariance)
    end
    terms = _smoothing_delta_covariance(derivatives.sensitivity, derivatives.root_derivatives,
                                       c.sigma2, Vrho)
    conditional = c.sigma2 .* Matrix(c.V_beta)
    covariance = conditional + terms.mean_correction + terms.root_correction
    all(isfinite, covariance) ||
        throw(DomainError(covariance, "smoothing covariance: the corrected covariance is non-finite"))
    (; covariance, conditional_covariance=conditional, terms..., rho_covariance=Vrho,
       rho_hessian=profile === nothing ? nothing : profile.hessian,
       rho_gradient=profile === nothing ? nothing : profile.gradient,
       coefficient_sensitivity=derivatives.sensitivity,
       root_derivatives=derivatives.root_derivatives, eigenvalues,
       method=:wps_local_gaussian, rho_source=profile === nothing ? :supplied : :profile_reml,
       rho_regularization=Float64(rho_regularization), sigma2=c.sigma2,
       profile_sigma2=profile === nothing ? nothing : profile.profile_sigma2,
       n_eff=s.residual_dof, coefficient_score=copy(s.coefficient_score),
       coefficient_step=-(_smoothing_cholesky(H) \ s.coefficient_score),
       jac=s.jac, stationarity=c.stationarity, converged=c.converged,
       smoothing_fixed=c.smoothing_fixed, smoothing_advanced=c.smoothing_advanced,
       covariance_ridge=s.covariance_ridge, determinant_ridge)
end
