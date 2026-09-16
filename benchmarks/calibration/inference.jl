include("models.jl")

function ce_error(e)
    if e isa ErrorException && startswith(e.msg,"ODE solve failed: ")
        return sprint(showerror,e)
    end
    sp_candidate_error(e)
end

function ce_jacobian(prob,beta,pred)
    nt,no = size(pred)
    J = zeros(length(pred),length(beta))
    PSM.compute_jacobian!(J,prob,beta,vec(pred),nt,no;
        dam=fill(1e-8,length(beta)),jac=:forwarddiff)
    all(isfinite,J) || throw(DomainError(J,"non-finite prediction Jacobian"))
    J
end

function ce_output_jacobian(model,beta,points)
    map = ce_output_map(points)
    model.approx isa NeuralApproximator ?
        PSM.ForwardDiff.jacobian(b->map*build_evaluator(model.approx,b).(points),beta) :
        map*sp_linear_design(model.approx,points)
end

function ce_fixed_fit(model,ds,values,rho,initial,indices,opts)
    a = sp_coordinates(model.approx,model.frame,rho,initial)
    prob = sp_problem(ce_problem(model,ds,values),a)
    L = cholesky(Symmetric(ds.noise.shape[indices,indices])).L
    theta = copy(a.initial)
    B = penalty_matrix(a)
    q = model.frame.nullity
    y = vec(values)[indices]
    function objective(b,unused)
        pred = try
            simulate(prob,b)
        catch e
            ce_error(e)
            return Inf
        end
        all(isfinite,pred) || return Inf
        residual = L \ (vec(pred)[indices]-y)
        (sum(abs2,residual)+sum(abs2,view(b,q+1:length(b))))/2
    end
    previous = objective(theta,B)
    isfinite(previous) || throw(DomainError(previous,"non-finite initial penalized objective"))
    converged,iterations = false,0
    for iteration in 1:opts.iterations
        iterations = iteration
        pred = simulate(prob,theta)
        J = ce_jacobian(prob,theta,pred)
        residual = L \ (vec(pred)[indices]-y)
        JW = L \ J[indices,:]
        pseudo = JW*theta-residual
        factor = PSM._pcls_factorize(JW,pseudo,B,ones(length(indices)))
        proposal = PSM._pcls_truncated_step(factor)
        next,value = PSM._pcls_step_contract(objective,theta,proposal,B,factor)
        isfinite(value) || throw(DomainError(value,"non-finite coefficient objective"))
        stable = abs(previous-value) <= opts.tol*max(1.0,abs(previous))
        theta = next
        previous = value
        if iteration >= 3 && stable
            converged = true
            break
        end
    end
    pred = simulate(prob,theta)
    J = ce_jacobian(prob,theta,pred)
    residual = L \ (vec(pred)[indices]-y)
    score = sp_working_profile(L\J[indices,:],ones(length(indices)),theta,residual,q;
        sigma2=opts.noise_mode=="known" ? ds.sigma^2 : nothing)
    beta = a.transform*theta
    all(isfinite,beta) && all(isfinite,pred) ||
        throw(DomainError(beta,"non-finite fitted coefficients or trajectory"))
    sigma2 = opts.noise_mode=="known" ? ds.sigma^2 : score.dispersion
    G = ce_output_jacobian(model,beta,ds.points)*a.transform
    root = sqrt(sigma2).*(G/score.R)
    estimates = ce_output_map(ds.points)*build_evaluator(model.approx,beta).(ds.points)
    se = [norm(view(root,i,:)) for i in axes(root,1)]
    all(isfinite,estimates) && all(isfinite,se) ||
        throw(DomainError(se,"non-finite function uncertainty"))
    (;rho,beta,theta,transform=a.transform,prob,pred,J,G,root,estimates,se,sigma2,
      score,converged,iterations,indices)
end

function ce_fit(model,ds,values,opts; indices=collect(1:length(values)))
    !isempty(indices) && allunique(indices) &&
        all(i->1<=i<=length(values),indices) || throw(ArgumentError("invalid fitting subset"))
    grid = sort!(unique(Float64.(opts.rhos)))
    !isempty(grid) && all(isfinite,grid) || throw(ArgumentError("invalid smoothing grid"))
    best = Dict{Float64,Any}()
    attempts = NamedTuple[]
    for direction in ("ascending","descending")
        current = copy(model.initial)
        for rho in (direction=="ascending" ? grid : reverse(grid))
            try
                fit = ce_fixed_fit(model,ds,values,rho,current,indices,opts)
                push!(attempts,(;rho,direction,status="ok",message="",Q=fit.score.Q,
                    converged=fit.converged,iterations=fit.iterations,decrement=fit.score.decrement))
                if !haskey(best,rho) || fit.score.Q<best[rho].score.Q
                    best[rho] = fit
                end
                current = fit.beta
            catch e
                message = ce_error(e)
                push!(attempts,(;rho,direction,status="failed",message,Q=NaN,
                    converged=false,iterations=0,decrement=NaN))
            end
        end
    end
    isempty(best) && return (;fit=nothing,attempts,status="failed",message="all smoothing candidates failed")
    rhos = sort!(collect(keys(best)))
    criterion = opts.noise_mode=="known" ? :known_criterion : :criterion
    chosen = rhos[argmax([getproperty(best[r].score,criterion) for r in rhos])]
    fit = merge(best[chosen],(;grid_boundary=chosen==first(grid)||chosen==last(grid)))
    (;fit,attempts,status="ok",message="")
end

function ce_normal_intervals(center,se,level; bias=zeros(length(center)))
    0<level<1 && length(center)==length(se)==length(bias) ||
        throw(ArgumentError("invalid interval level or dimensions"))
    all(isfinite,center) && all(x->isfinite(x)&&x>=0,se) &&
        all(x->isfinite(x)&&x>=0,bias) || throw(DomainError(se,"invalid interval inputs"))
    z = Distributions.quantile(Distributions.Normal(),(1+level)/2)
    half = z.*se+bias
    (;lower=center-half,upper=center+half,critical=z)
end

function ce_conditional(fit,npoints,level,nsim,rng)
    pointwise = ce_normal_intervals(fit.estimates,fit.se,level)
    joint = PSM._gaussian_grid_band(fit.root[1:npoints,:],
        Matrix{Float64}(I,size(fit.root,2),size(fit.root,2));level,nsim,rng)
    simultaneous = (;lower=fit.estimates[1:npoints]-joint.critical.*joint.se,
        upper=fit.estimates[1:npoints]+joint.critical.*joint.se,critical=joint.critical)
    (;pointwise,simultaneous)
end

function ce_studentized(center,se,generating,draws,draw_se,npoints,level)
    size(draws)==size(draw_se) && size(draws,1)==length(center)==length(se)==length(generating) ||
        throw(DimensionMismatch("studentized bootstrap arrays do not align"))
    size(draws,2)>=3 && 0<level<1 || throw(ArgumentError("insufficient bootstrap budget or invalid level"))
    all(isfinite,center) && all(isfinite,generating) && all(x->isfinite(x)&&x>=0,se) &&
        1<=npoints<=length(center) || throw(DomainError(se,"invalid original studentization"))
    all(isfinite,draws) && all(x->isfinite(x)&&x>=0,draw_se) ||
        throw(DomainError(draws,"studentized bands require every attempted refit to yield a complete finite curve and scale"))
    t = similar(draws)
    for i in axes(draws,1), b in axes(draws,2)
        difference = draws[i,b]-generating[i]
        if iszero(draw_se[i,b])
            iszero(difference) || throw(DomainError(i,"zero bootstrap scale with a nonzero deviation"))
            t[i,b] = 0.0
        else
            t[i,b] = difference/draw_se[i,b]
        end
    end
    all(isfinite,t) || throw(DomainError(t,"non-finite studentized deviations"))
    alpha = (1-level)/2
    pointwise = (;lower=[center[i]-quantile(t[i,:],1-alpha)*se[i] for i in eachindex(center)],
        upper=[center[i]-quantile(t[i,:],alpha)*se[i] for i in eachindex(center)],critical=NaN)
    maxima = PSM._max_standardized_deviations(t[1:npoints,:],ones(npoints))
    critical = quantile(maxima,level)
    simultaneous = (;lower=center[1:npoints]-critical.*se[1:npoints],
        upper=center[1:npoints]+critical.*se[1:npoints],critical)
    (;pointwise,simultaneous)
end

function ce_bootstrap(model,ds,fit,opts,generator)
    generator in ("fit","pilot") || throw(ArgumentError("unknown bootstrap generator"))
    source = generator=="fit" ? fit :
        ce_fixed_fit(model,ds,ds.observed,fit.rho+log(opts.pilot_fraction),fit.beta,
            collect(1:length(ds.observed)),opts)
    nout = length(fit.estimates)
    draws,scales = fill(NaN,nout,opts.nboot),fill(NaN,nout,opts.nboot)
    records = Dict{String,Any}[]
    for b in 1:opts.nboot
        seed = ce_seed("bootstrap",ds.stage,ds.operator,ds.design_id,ds.noise_id,
            ds.sigma,ds.case,ds.seed,generator,b)
        noise = sqrt(source.sigma2).*(ds.noise.root*randn(StableRNG(seed),length(ds.observed)))
        observed = source.pred+reshape(noise,size(source.pred))
        result = ce_fit(model,ds,observed,opts)
        row = Dict{String,Any}("attempt"=>b,"rng_seed"=>string(seed),"status"=>result.status,
            "message"=>result.message,"candidate_count"=>length(result.attempts),
            "candidate_failures"=>[Dict(string(k)=>v for (k,v) in pairs(r)) for r in result.attempts if r.status!="ok"])
        if result.fit !== nothing
            draws[:,b],scales[:,b] = result.fit.estimates,result.fit.se
            row["rho"],row["sigma2"],row["parameters"] = result.fit.rho,result.fit.sigma2,result.fit.beta
            row["converged"],row["grid_boundary"] = result.fit.converged,result.fit.grid_boundary
        end
        push!(records,row)
    end
    (;draws,scales,records,generating=source.estimates,generating_rho=source.rho,
      generating_sigma2=source.sigma2)
end

function ce_innovation(shape,train,infer)
    isempty(intersect(train,infer)) || throw(ArgumentError("training and inference observations overlap"))
    S = shape[train,train]
    regression = shape[infer,train] / cholesky(Symmetric(S))
    covariance = shape[infer,infer]-regression*shape[train,infer]
    L = Matrix(cholesky(Symmetric((covariance+covariance')/2)).L)
    (;regression,L)
end

function ce_bias_bound(offset,M,means,targets)
    size(M,2)==size(means,1) && size(M,1)==size(targets,1)==length(offset) &&
        size(means,2)==size(targets,2)>0 ||
        throw(DimensionMismatch("finite-class bias arrays do not align"))
    errors = offset .+ M*means-targets
    all(isfinite,errors) || throw(DomainError(errors,"non-finite finite-class bias"))
    vec(maximum(abs.(errors);dims=2))
end

function ce_ellipsoid_bias(offset,M,A,G,center,weights,radius)
    length(center)==length(weights)==size(A,2)==size(G,2) &&
        size(M,2)==size(A,1) && size(M,1)==size(G,1)==length(offset) ||
        throw(DimensionMismatch("ellipsoid bias arrays do not align"))
    radius>=0 && isfinite(radius) && all(x->isfinite(x)&&x>0,weights) ||
        throw(ArgumentError("invalid reference ellipsoid"))
    difference = M*A-G
    abs.(offset+difference*center) +
        radius.*[norm(view(difference,i,:)./weights) for i in axes(difference,1)]
end

function ce_reference_design(ds)
    ds.operator=="integral" || throw(ArgumentError("polynomial reference requires the integral operator"))
    A = reduce(vcat,[permutedims(ce_integrated_legendre(t,v))
        for v in ds.design.speeds for t in ds.design.times])
    G = ce_output_map(ds.points)*reduce(vcat,[permutedims(ce_legendre(x)) for x in ds.points])
    (;A,G)
end

function ce_split_inference(model,ds,opts)
    opts.noise_mode=="known" || throw(ArgumentError("bias bounds currently require known Gaussian covariance"))
    nt,np = size(ds.observed)
    train = [i+(j-1)*nt for j in 1:np for i in 1:2:nt]
    infer = setdiff(collect(1:length(ds.observed)),train)
    pilot = ce_fit(model,ds,ds.observed,opts;indices=train)
    pilot.fit === nothing && return (;value=nothing,pilot)
    f = pilot.fit
    innovation = ce_innovation(ds.noise.shape,train,infer)
    regression,L = innovation.regression,innovation.L
    J = f.J[infer,:]-regression*f.J[train,:]
    JW = L \ J
    p,q = length(f.theta),model.frame.nullity
    C = hcat(zeros(p-q,q),Matrix{Float64}(I,p-q,p-q))
    augmented = vcat(JW,C)
    column_scales = [norm(view(augmented,:,i)) for i in 1:p]
    all(>(0),column_scales) || throw(DomainError(column_scales,"unidentified one-step update"))
    scaled = augmented ./ reshape(column_scales,1,:)
    rank(scaled)==p || throw(DomainError(scaled,"rank-deficient one-step information"))
    R = UpperTriangular(Matrix(qr(scaled).R)*Diagonal(column_scales))
    M = (f.G/R)/R' * JW' / L
    y = vec(ds.observed)
    mean0 = vec(f.pred)[infer]-regression*vec(f.pred)[train]
    offset = f.estimates-M*mean0
    estimates = offset+M*(y[infer]-regression*y[train])
    projected = M*L
    se = ds.sigma.*[norm(view(projected,i,:)) for i in axes(M,1)]
    bank_means = ds.cache.predictions[infer,:]-regression*ds.cache.predictions[train,:]
    bank_bias = ce_bias_bound(offset,M,bank_means,ds.cache.outputs)
    ellipsoid_bias = nothing
    if ds.operator=="integral"
        reference = ce_reference_design(ds)
        A = reference.A[infer,:]-regression*reference.A[train,:]
        ellipsoid_bias = ce_ellipsoid_bias(offset,M,A,reference.G,CE_REFERENCE_CENTER,
            CE_REFERENCE_WEIGHTS,CE_REFERENCE_RADIUS)
    end
    (;value=(;estimates,se,bank_bias,ellipsoid_bias,offset,M,train,infer,
        pilot_rho=f.rho,innovation_regression=regression),pilot)
end

function ce_bias_intervals(center,se,bound,npoints,level)
    pointwise = ce_normal_intervals(center,se,level;bias=bound)
    critical = Distributions.quantile(Distributions.Normal(),1-(1-level)/(2npoints))
    half = critical.*se[1:npoints]+bound[1:npoints]
    simultaneous = (;lower=center[1:npoints]-half,upper=center[1:npoints]+half,critical)
    (;pointwise,simultaneous)
end

function ce_bank_set(ds,level)
    residuals = ds.noise.root \ (vec(ds.observed) .- ds.cache.predictions)
    losses = vec(sum(abs2,residuals;dims=1))./ds.sigma^2
    critical = Distributions.quantile(Distributions.Chisq(length(ds.observed)),level)
    accepted = findall(<=(critical),losses)
    estimates = ds.cache.outputs[:,argmin(losses)]
    isempty(accepted) && return (;estimates,accepted,losses,critical,intervals=nothing)
    lower,upper = vec(minimum(ds.cache.outputs[:,accepted];dims=2)),vec(maximum(ds.cache.outputs[:,accepted];dims=2))
    n = length(ds.points)
    intervals = (pointwise=(;lower,upper,critical),
        simultaneous=(;lower=lower[1:n],upper=upper[1:n],critical))
    (;estimates,accepted,losses,critical,intervals)
end

function ce_linear_set(ds,level)
    reference = ce_reference_design(ds)
    L = ds.sigma.*ds.noise.root
    A,y = L \ reference.A,L \ vec(ds.observed)
    rank(A)==size(A,2) || throw(DomainError(A,"unidentified polynomial reference"))
    factor = qr(A)
    beta = factor \ y
    loss = sum(abs2,A*beta-y)
    critical = Distributions.quantile(Distributions.Chisq(length(y)),level)
    estimates = reference.G*beta
    loss>critical && return (;estimates,loss,critical,intervals=nothing)
    root = reference.G/UpperTriangular(Matrix(factor.R))
    half = sqrt(critical-loss).*[norm(view(root,i,:)) for i in axes(root,1)]
    lower,upper = estimates-half,estimates+half
    n = length(ds.points)
    (;estimates,loss,critical,intervals=(pointwise=(;lower,upper,critical),
        simultaneous=(;lower=lower[1:n],upper=upper[1:n],critical)))
end
