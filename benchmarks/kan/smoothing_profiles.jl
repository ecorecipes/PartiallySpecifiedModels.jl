include("coverage_robustness.jl")

struct ProfileCoordinates{A} <: AbstractApproximator
    name::Symbol
    base::A
    transform::Matrix{Float64}
    initial::Vector{Float64}
    nullity::Int
end
PSM.nparams(a::ProfileCoordinates) = size(a.transform, 2)
PSM.initial_params(a::ProfileCoordinates) = copy(a.initial)
PSM.build_evaluator(a::ProfileCoordinates, theta) = build_evaluator(a.base, a.transform * theta)
PSM.band_domain(a::ProfileCoordinates) = band_domain(a.base)
function PSM.penalty_matrix(a::ProfileCoordinates)
    n = nparams(a)
    n == a.nullity ? nothing : Matrix(Diagonal([zeros(a.nullity); ones(n-a.nullity)]))
end
function PSM.penalty_blocks(a::ProfileCoordinates)
    S = penalty_matrix(a)
    S === nothing ? Tuple{Matrix{Float64},UnitRange{Int}}[] : [(S, 1:nparams(a))]
end

function sp_penalty_frame(S::AbstractMatrix; allow_proper=false)
    n, m = size(S)
    n == m && n > 0 || throw(DimensionMismatch("profile penalty must be nonempty and square"))
    all(isfinite, S) || throw(DomainError(S, "non-finite profile penalty"))
    maximum(abs, S-S') <= 64n*eps(Float64)*maximum(abs, S) ||
        throw(DomainError(S, "profile penalty must be symmetric"))
    eig = eigen(Symmetric(Matrix{Float64}(S)))
    # The existing fixed-rank LAML convention, not a new affine prior or
    # a covariance repair. Retain the discarded eigenvalues in the record.
    tolerance = max(1e-10maximum(abs, eig.values), 1e-14)
    minimum(eig.values) >= -tolerance || throw(DomainError(eig.values, "indefinite profile penalty"))
    positive = eig.values .> tolerance
    (0 < count(positive) < n || (allow_proper && count(positive)==n)) ||
        throw(ArgumentError("this diagnostic requires both penalized and null-space directions"))
    U0, Up = eig.vectors[:,.!positive], eig.vectors[:,positive]
    d = eig.values[positive]
    effective = Up * Diagonal(d) * Up'
    (; U0, Up, eigenvalues=d, discarded=eig.values[.!positive], tolerance,
       penalized=Up ./ reshape(sqrt.(d),1,:), shear=zeros(size(U0,2),length(d)),
       rank=length(d), nullity=size(U0,2), effective_penalty=effective,
       reconstruction_relative=norm(effective-S)/norm(S))
end

function sp_center_frame(frame, design)
    size(design,2) == size(frame.U0,1) ||
        throw(DimensionMismatch("reference design does not match the penalty frame"))
    all(isfinite,design) || throw(DomainError(design,"non-finite reference design"))
    frame.nullity == 0 && return frame
    X0 = design*frame.U0
    rank(X0) == frame.nullity ||
        throw(DomainError(X0,"reference design does not identify the penalty null space"))
    # A unit-determinant shear changes coordinates, not the prior. Removing
    # affine function components avoids inflating the PCLS data-rank cutoff
    # when a weak-curvature KAN direction contains a large affine component.
    shear = qr(X0) \ (design*frame.penalized)
    merge(frame,(;shear,penalized=frame.penalized-frame.U0*shear))
end

function sp_coordinates(a, frame, rho::Real, beta)
    (isfinite(rho) || rho == Inf) || throw(ArgumentError("log lambda must be finite or +Inf"))
    length(beta) == size(frame.U0,1) == nparams(a) ||
        throw(DimensionMismatch("profile coefficient layout differs"))
    all(isfinite, beta) || throw(DomainError(beta, "non-finite profile start"))
    penalized = sqrt.(frame.eigenvalues) .* (frame.Up' * beta)
    null = frame.U0' * beta + frame.shear * penalized
    T, theta = if rho == Inf
        copy(frame.U0), null
    else
        scale = exp(-Float64(rho)/2)
        isfinite(scale) && scale > 0 ||
            throw(DomainError(rho, "log lambda exceeds representable profile coordinates"))
        hcat(frame.U0, frame.penalized .* scale), vcat(null, penalized ./ scale)
    end
    all(isfinite,T) || throw(DomainError(rho,"profile transformation overflows"))
    all(isfinite, theta) || throw(DomainError(theta, "profile start overflows transformed coordinates"))
    ProfileCoordinates(a.name, a, T, theta, frame.nullity)
end

function sp_problem(prob, a)
    length(prob.approximators) == 1 && prob.likelihood isa Gaussian ||
        throw(ArgumentError("smoothing profiles require one Gaussian approximator"))
    PSMProblem(prob.dynamics!, prob.u0, prob.tspan, [a];
        data_times=prob.data_times, data_values=prob.data_values,
        data_weights=prob.data_weights, obs_to_state=prob.obs_to_state,
        known_params=prob.known_params, likelihood=prob.likelihood,
        solver=prob.ode_solver, discrete=prob.discrete, delays=prob.delays,
        history=prob.history, prob.ode_kwargs...)
end

function sp_refit(prob, frame, rho, beta; iterations=80, tol=1e-9, jac=:forwarddiff)
    iterations > 0 && isfinite(tol) && tol > 0 ||
        throw(ArgumentError("profile fitting budget and tolerance must be positive"))
    a = sp_coordinates(only(prob.approximators), frame, rho, beta)
    transformed = sp_problem(prob, a)
    alg = rho == Inf ? LAML(maxiters=iterations, tol=tol, jac=jac) :
        LAML(maxiters=iterations, tol=tol, jac=jac, fixed_lambda=1.0)
    sol = solve(transformed, alg)
    all(isfinite, sol.parameters) && all(isfinite, sol.fitted_values) ||
        throw(DomainError(rho, "non-finite profile coefficient fit"))
    theta = collect(sol.parameters)
    penalty = rho == Inf ? 0.0 : sum(abs2, view(theta, frame.nullity+1:length(theta)))
    Q = sol.data_loss+penalty
    isfinite(Q) && Q >= 0 ||
        throw(DomainError(Q,"non-finite profile penalized objective"))
    (; rho, beta=a.transform*theta, theta, penalty, Q,
       rss=sol.data_loss, coordinates=a, problem=transformed, solution=sol, jac)
end

function sp_working_profile(J, weights, theta, residual, nullity; sigma2=nothing)
    n, p = size(J)
    length(weights) == length(residual) == n && length(theta) == p ||
        throw(DimensionMismatch("profile working arrays do not align"))
    0 <= nullity <= p || throw(ArgumentError("invalid profile nullity"))
    all(isfinite, J) && all(isfinite, theta) && all(isfinite, residual) &&
        all(x->isfinite(x)&&x>=0, weights) ||
        throw(DomainError(weights, "profile working arrays must be finite with nonnegative weights"))
    n_used = count(>(0), weights)
    n_eff = n_used - nullity
    n_eff > 0 || throw(DomainError(n_eff, "profile has no restricted residual degrees of freedom"))
    sigma2 === nothing || (isfinite(sigma2) && sigma2 > 0) ||
        throw(ArgumentError("known profile dispersion must be finite and positive"))
    r = p-nullity
    C = hcat(zeros(r,nullity), Matrix{Float64}(I,r,r))
    Z = sqrt.(weights) .* J
    augmented = vcat(Z,C)
    # Column equilibration plus QR avoids forming J'WJ, and never adds a
    # ridge to the null space. The column scales are undone in R.
    scales = [norm(view(augmented,:,k)) for k in 1:p]
    all(>(0), scales) || throw(DomainError(scales, "unidentified profile coefficient"))
    equilibrated = augmented ./ reshape(scales,1,:)
    singular = svdvals(equilibrated)
    minimum(singular) > max(size(equilibrated)...)*eps(Float64)*maximum(singular) ||
        throw(DomainError(singular, "rank-deficient profile information"))
    R = UpperTriangular(Matrix(qr(equilibrated).R) * Diagonal(scales))
    logdet = 2sum(log, abs.(diag(R)))
    rss = sum(weights .* residual.^2)
    penalty = sum(abs2, view(theta,nullity+1:p))
    Q = rss + penalty
    Q > 0 && isfinite(Q) || throw(DomainError(Q, "invalid profile residual-plus-penalty sum"))
    edf = sum(abs2, Z / R)
    dispersion = rss / max(n_used-edf,1.0)
    profile_dispersion = Q/n_eff
    # Penalty whitening and exp(-rho/2) scaling cancel r*rho + log|S|_+
    # against the change in log|H|. The affine shear has determinant one.
    criterion = -n_eff/2*log(profile_dispersion) - logdet/2
    known_criterion = sigma2 === nothing ? NaN :
        -Q/(2sigma2) - n_eff/2*log(sigma2) - logdet/2
    score = J'*(weights .* residual) + C'*(C*theta)
    step = -(R \ (R' \ score))
    decrement = norm(R' \ score)/sqrt(Q)
    all(isfinite,(logdet,edf,dispersion,profile_dispersion,criterion,decrement)) &&
        all(isfinite,step) ||
        throw(DomainError(Q,"non-finite profile information or stationarity diagnostic"))
    local_gradient, local_hessian = NaN, NaN
    if r > 0
        B = C'*C
        sensitivity = -(R \ (R' \ (B*theta)))
        q = penalty
        P = R \ (R' \ B)
        trace = tr(P)
        q2 = q + 2dot(B*theta, sensitivity)
        local_gradient = (r-trace-n_eff*q/Q)/2
        local_hessian = (n_eff*(q2/Q-(q/Q)^2) + trace - tr(P*P))/2
    end
    (; criterion, known_criterion, rss, penalty, Q, edf, dispersion, profile_dispersion,
       n_used, n_eff, R, score, step, decrement, local_gradient, local_hessian,
       equilibrated_condition=maximum(singular)/minimum(singular))
end

function sp_score(fit, frame; sigma2=nothing, predictions=fit.solution.fitted_values)
    prob, theta = fit.problem, fit.theta
    pred = predictions
    nt, no = size(pred)
    y, w = vec(prob.data_values), vec(prob.data_weights)
    residual, weights = zeros(length(y)), zeros(length(y))
    for k in eachindex(y)
        PSM._usable(y[k],w[k]) || continue
        residual[k] = pred[k]-y[k]
        weights[k] = w[k]
    end
    J = zeros(length(y), length(theta))
    PSM.compute_jacobian!(J, prob, theta, vec(pred), nt, no;
        dam=fill(1e-8,length(theta)), jac=fit.jac)
    working = sp_working_profile(J, weights, theta, residual, frame.nullity; sigma2)
    (; fit..., working..., J, predictions=pred)
end

function sp_evaluate(prob,frame,rho,beta; sigma2=nothing,jac=:forwarddiff)
    a = sp_coordinates(only(prob.approximators),frame,rho,beta)
    transformed = sp_problem(prob,a)
    theta = a.initial
    fit = (;rho,beta=a.transform*theta,theta,coordinates=a,problem=transformed,jac)
    sp_score(fit,frame;sigma2,predictions=simulate(transformed,theta))
end

function sp_native_terms(scored,frame,S)
    isfinite(scored.rho) || return (;native_criterion=NaN,native_penalty=NaN,
        native_profile_scale=NaN,native_scale_floored=false,native_logdet_shift=NaN,native_ridge=NaN)
    P = Diagonal(sqrt.(frame.eigenvalues))*frame.Up'
    inverse_transform = vcat(frame.U0' + frame.shear*P, exp(scored.rho/2).*P)
    native_J = scored.J*inverse_transform
    prob = scored.problem
    y, w = vec(prob.data_values), vec(prob.data_weights)
    weights = [PSM._usable(y[k],w[k]) ? w[k] : 0.0 for k in eachindex(y)]
    value,H,B,_ = PSM.laml_objective(Gaussian(),scored.beta,native_J,weights,w,y,
        vec(scored.predictions),[S],[0],[length(scored.beta)],[scored.rho],length(scored.beta))
    penalty = dot(scored.beta,B*scored.beta)
    scale = (scored.rss+penalty)/scored.n_eff
    stable_logdet = 2sum(log,abs.(diag(scored.R))) +
        frame.rank*scored.rho + sum(log,frame.eigenvalues)
    (;native_criterion=value,native_penalty=penalty,native_profile_scale=scale,
      native_scale_floored=scale<=1e-30,
      native_logdet_shift=PSM._log_det_pd(H)-stable_logdet,
      native_ridge=1e-10maximum(abs,diag(H))+1e-15)
end

function sp_function_uncertainty(scored, native_design; sigma2=nothing)
    size(native_design,2) == length(scored.beta) ||
        throw(DimensionMismatch("function design differs from the profile coefficients"))
    projected = (native_design*scored.coordinates.transform) / scored.R
    unscaled = [norm(view(projected,i,:)) for i in axes(projected,1)]
    fitted = native_design*scored.beta
    all(isfinite,fitted) && all(isfinite,unscaled) ||
        throw(DomainError(fitted,"non-finite profile function or uncertainty"))
    (;fitted,se=sqrt(scored.dispersion).*unscaled,
      known_se=sigma2 === nothing ? fill(NaN,length(fitted)) : sqrt(sigma2).*unscaled)
end

function sp_linear_design(a, points)
    a isa Union{BSplineApproximator,GPApproximator,SPDEApproximator} ||
        (a isa KANApproximator && length(a.layers)==1 && a.input_dim==1) ||
        throw(ArgumentError("representation oracle requires a unary spline or single-layer KAN"))
    points isa AbstractVector{<:Real} && all(isfinite, points) && !isempty(points) ||
        throw(ArgumentError("representation points must be finite and nonempty"))
    n = nparams(a)
    identity = Matrix{Float64}(I,n,n)
    hcat([Float64.(build_evaluator(a,identity[:,k]).(points)) for k in 1:n]...)
end

function sp_representation_oracle(a, truth, domain; ngrid=801, neval=1601)
    ngrid >= 2 && neval >= 2 && all(isfinite,domain) && domain[1] < domain[2] ||
        throw(ArgumentError("invalid representation-oracle grid"))
    x = collect(range(domain...;length=ngrid))
    weights = ones(ngrid)
    weights[[1,end]] .= 0.5
    design = sp_linear_design(a,x)
    target = truth.(x)
    all(isfinite,target) || throw(DomainError(target,"non-finite representation target"))
    F = svd(sqrt.(weights) .* design)
    tolerance = max(size(design)...)*eps(Float64)*maximum(F.S)
    inverse = [s>tolerance ? inv(s) : 0.0 for s in F.S]
    beta = F.V * (inverse .* (F.U' * (sqrt.(weights).*target)))
    query = collect(range(domain...;length=neval))
    errors = build_evaluator(a,beta).(query) - truth.(query)
    (; beta, rank=count(>(tolerance),F.S), singular_values=F.S,
       domain=collect(domain), ngrid, neval, rmse=sqrt(mean(abs2,errors)),
       max_error=maximum(abs,errors))
end

function sp_candidate_error(e)
    numerical = e isa Union{DomainError,OverflowError,DivideError,PosDefException,
        SingularException,ZeroPivotException,LAPACKException,RankDeficientException} ||
        (e isa InexactError && !PSM._is_program_error(e))
    numerical || rethrow(e)
    sprint(showerror,e)
end

function sp_profile(prob, rhos; beta_selected, iterations=80, tol=1e-9, jac=:forwarddiff,
                    reference_design=nothing)
    S, offsets, sizes = PSM.build_penalty_matrices(prob)
    length(S)==1 && only(offsets)==0 && only(sizes)==length(beta_selected) ||
        throw(ArgumentError("profile requires one penalty block covering the approximator"))
    all(isfinite,beta_selected) || throw(ArgumentError("profile needs a finite selected coefficient vector"))
    grid = sort!(unique(Float64.(rhos)))
    !isempty(grid) && all(isfinite,grid) ||
        throw(ArgumentError("finite profile grid must be nonempty"))
    if reference_design === nothing
        a = only(prob.approximators)
        base = a isa InitializedApprox ? a.approx : a
        dom = band_domain(base)
        dom === nothing && throw(ArgumentError("profile requires a reference design or a unary domain"))
        reference_design = sp_linear_design(base,collect(range(dom...;length=401)))
    end
    frame = sp_center_frame(sp_penalty_frame(only(S)),reference_design)
    starts = (initial=PSM.build_initial_params(prob), selected=collect(beta_selected))
    candidates = NamedTuple[]
    best = Dict{Float64,Any}()
    function attempt(rho, label, beta)
        try
            fit = sp_refit(prob,frame,rho,beta;iterations,tol,jac)
            push!(candidates,(;rho,start=label,status="ok",message="",Q=fit.Q,rss=fit.rss,
                converged=fit.solution.convergence.converged,
                iterations=fit.solution.convergence.iterations,beta=copy(fit.beta)))
            if !haskey(best,rho) || fit.Q < best[rho].Q
                best[rho] = fit
            end
            fit.beta
        catch e
            message = sp_candidate_error(e)
            push!(candidates,(;rho,start=label,status="failed",message,Q=NaN,rss=NaN,
                converged=false,iterations=0,beta=Float64[]))
            nothing
        end
    end
    for (name,beta) in pairs(starts)
        attempt(Inf,"null_"*string(name),beta)
    end
    for rho in grid
        attempt(rho,"selected",starts.selected)
    end
    current = copy(starts.initial)
    for rho in grid
        value = attempt(rho,"ascending",current)
        value === nothing || (current = value)
    end
    current = haskey(best,Inf) ? best[Inf].beta : copy(starts.selected)
    for rho in reverse(grid)
        value = attempt(rho,"descending",current)
        value === nothing || (current = value)
    end
    (; frame, grid, candidates, best)
end
