function _band_interval(interval)
    interval in (:pointwise,:simultaneous) ||
        throw(ArgumentError("confidence_band: interval must be :pointwise or :simultaneous"))
    interval
end

function _band_covariance_root(covariance::AbstractMatrix; context="confidence_band")
    n,m=size(covariance)
    n==m || throw(DimensionMismatch("$context: covariance must be square"))
    all(isfinite,covariance) || throw(DomainError(covariance,"$context: non-finite covariance"))
    n==0 && return zeros(0,0)
    C=Matrix{Float64}(covariance)
    scale=maximum(abs,C)
    allowance=64n*eps(Float64)*scale
    maximum(abs,C-C') <= allowance ||
        throw(DomainError(C,"$context: covariance is not symmetric"))
    eig=eigen(Symmetric(C/2+C'/2))
    all(isfinite,eig.values) ||
        throw(DomainError(eig.values,"$context: covariance factorization is non-finite"))
    allowance=64n*eps(Float64)*max(maximum(abs,eig.values),scale)
    minimum(eig.values) >= -allowance ||
        throw(DomainError(minimum(eig.values),"$context: covariance is not positive semidefinite"))
    # Clamp only eigensolver-scale negative roundoff, not a negative
    # covariance direction large enough to alter the statistical model.
    eig.vectors .* reshape(sqrt.(max.(eig.values,0.0)),1,:)
end

function _max_standardized_deviations(deviations,se)
    size(deviations,1)==length(se) ||
        throw(DimensionMismatch("confidence_band: deviations and standard errors do not align"))
    all(isfinite,deviations) && all(x->isfinite(x)&&x>=0,se) ||
        throw(DomainError(se,"confidence_band: deviations and scales must be finite"))
    maxima=zeros(size(deviations,2))
    for i in eachindex(se)
        if iszero(se[i])
            all(iszero,@view deviations[i,:]) ||
                throw(DomainError(i,"confidence_band: zero uncertainty scale has nonzero deviations"))
        else
            for b in eachindex(maxima)
                maxima[b]=max(maxima[b],abs(deviations[i,b]/se[i]))
            end
        end
    end
    all(isfinite,maxima) ||
        throw(DomainError(maxima,"confidence_band: standardized deviations overflowed"))
    maxima
end

function _gaussian_grid_band(jacobian,covariance;level,nsim,rng)
    0<level<1 && nsim>=2 || throw(ArgumentError("invalid simultaneous-band level or simulation count"))
    size(jacobian,1)>0 && size(jacobian,2)==size(covariance,1) ||
        throw(DimensionMismatch("confidence_band: Jacobian and covariance do not align"))
    all(isfinite,jacobian) ||
        throw(DomainError(jacobian,"confidence_band: non-finite parameter sensitivity"))
    projected=jacobian*_band_covariance_root(covariance)
    # Factor-space norms avoid cancellation in j'Cj. They agree with the
    # pointwise quadratic form up to floating-point factorization error.
    se=[norm(@view projected[i,:]) for i in axes(projected,1)]
    all(isfinite,se) || throw(DomainError(se,"confidence_band: non-finite projected scale"))
    active=count(>(0),se)
    active==0 && return (;se,critical=0.0,n_draws=0)
    normal=_qnorm(1-(1-level)/2)
    active==1 && return (;se,critical=normal,n_draws=0)
    deviations=projected*randn(rng,size(projected,2),nsim)
    maxima=_max_standardized_deviations(deviations,se)
    # Monte Carlo noise must not make a Gaussian simultaneous critical
    # value smaller than its known marginal normal critical value.
    (;se,critical=max(normal,quantile(maxima,level)),n_draws=nsim)
end

function _bootstrap_grid_band(values,fitted;level)
    size(values,1)==length(fitted) && size(values,1)>0 ||
        throw(DimensionMismatch("confidence_band: bootstrap values and fitted curve do not align"))
    size(values,2)>=3 || throw(ArgumentError("confidence_band: at least three refits are required"))
    0<level<1 || throw(ArgumentError("confidence_band: level must be in (0,1)"))
    all(isfinite,fitted) || throw(DomainError(fitted,"confidence_band: original fitted curve is non-finite"))
    all(isfinite,values) || throw(DomainError(values,
        "confidence_band: simultaneous bootstrap bands require complete finite curve draws; " *
        "incomplete draws must not silently lose their most extreme query points"))
    se=vec(std(values;dims=2,corrected=true))
    for i in axes(values,1)
        all(==(values[i,1]),@view values[i,:]) && (se[i]=0.0)
    end
    deviations=values .- fitted
    maxima=_max_standardized_deviations(deviations,se)
    critical=quantile(maxima,level)
    (;se,critical,n_draws=size(values,2),lower=fitted-critical*se,upper=fitted+critical*se)
end

function _bootstrap_band_indices(bs,requested)
    requested === nothing && return Dict{Symbol,Vector{Int}}()
    requested isa AbstractDict ||
        throw(ArgumentError("confidence_band: uf_indices must be a dictionary"))
    result=Dict{Symbol,Vector{Int}}()
    for (name,indices) in requested
        name isa Symbol && haskey(bs.uf_values,name) ||
            throw(ArgumentError("confidence_band: unknown bootstrap function $name"))
        indices isa AbstractVector{<:Integer} && !(eltype(indices)<:Bool) ||
            throw(ArgumentError("confidence_band: uf_indices values must be integer index vectors"))
        ids=Int.(collect(indices))
        !isempty(ids) && allunique(ids) &&
            all(i->1<=i<=size(bs.uf_values[name],1),ids) ||
                throw(ArgumentError("confidence_band: query indices must be nonempty, unique and in range"))
        result[name]=ids
    end
    result
end

"""
    confidence_band(bs::BootstrapResult, sol::PSMSolution, prob::PSMProblem;
                    level=bs.level, interval=:pointwise, uf_indices=nothing)

Construct function intervals from an existing bootstrap of `sol`/`prob`,
without refitting. The bootstrap must correspond to this original fit.
Default `:pointwise` returns percentile intervals, preserving the original
grid/explicit-coordinate interpretation. `se` is the empirical bootstrap
standard deviation, not the conditional parameter-covariance standard error.

With `interval=:simultaneous`, use the level-quantile of
`max_x abs((f_boot(x)-fitted(x))/bootstrap_sd(x))`. The result is the symmetric
band `fitted +/- critical*bootstrap_sd`, simultaneous over each function's
selected finite query set separately. This is not a simultaneous percentile
interval, not bootstrap-t with replicate-specific studentization, and not
joint coverage across different functions.

`uf_indices=Dict(:r => indices)` restricts a stored query grid before the
maximum is calibrated; it cannot invent new query coordinates. Returned
coordinates are owned copies. Every retained refit must evaluate finitely
at every selected point for simultaneous bands: incomplete curves raise an
error rather than silently discarding extreme points. A zero-variance point
must equal the original fitted value in every draw.

The added metadata records `interval`, `method`, `critical`, `level`, `scope`,
`conditioning`, `n_draws` and `usable`. Pointwise percentile bands have no
single critical value (`critical=nothing`). Bootstrap uncertainty reflects
whatever the supplied refitting algorithm actually re-estimated; this method
cannot turn a fixed-penalty bootstrap into smoothing-selection uncertainty,
or produce an analytic smoothing-uncertainty-corrected covariance.

Both modes use Julia's default linearly interpolated empirical quantile.
Coverage over a finite grid is not a continuum or identifiability guarantee.
"""
function confidence_band(bs::BootstrapResult,sol::PSMSolution,prob::PSMProblem;
                         level::Float64=bs.level,interval::Symbol=:pointwise,
                         uf_indices=nothing)
    _band_interval(interval)
    0<level<1 || throw(ArgumentError("confidence_band: level must be in (0,1)"))
    bs.n_success>=3 || throw(ArgumentError("confidence_band: at least three successful refits are required"))
    size(bs.coefs)==(bs.n_success,length(sol.parameters)) &&
        length(sol.parameters)==n_total_params(prob) ||
            throw(DimensionMismatch("confidence_band: bootstrap coefficient dimensions differ from the fit"))
    indices=_bootstrap_band_indices(bs,uf_indices)
    points=_function_query_points(prob,bs.uf_points)
    known=Set(a.name for a in prob.approximators)
    all(in(known),keys(bs.uf_values)) ||
        throw(ArgumentError("confidence_band: bootstrap contains an unknown function"))
    result=Dict{Symbol,NamedTuple}()
    offset=0
    for a in prob.approximators
        block=(offset+1):(offset+nparams(a))
        offset+=nparams(a)
        haskey(bs.uf_values,a.name) || continue
        values=bs.uf_values[a.name]
        n=size(values,1)
        size(values,2)==bs.n_success && n>0 ||
            throw(DimensionMismatch("confidence_band: invalid bootstrap function sample dimensions"))
        explicit=haskey(points,a.name)
        explicit == haskey(bs.uf_grid,a.name) &&
            throw(ArgumentError("confidence_band: each bootstrap function needs exactly one coordinate source"))
        ids=get(indices,a.name,collect(1:n))
        coordinates=explicit ? points[a.name] : bs.uf_grid[a.name]
        size(coordinates,1)==n ||
            throw(DimensionMismatch("confidence_band: bootstrap coordinates do not match the samples"))
        all(isfinite,coordinates) ||
            throw(DomainError(coordinates,"confidence_band: non-finite bootstrap coordinates"))
        selected=explicit ? copy(coordinates[ids,:]) : copy(coordinates[ids])
        fitted=if explicit
            f=sol.unknown_functions[a.name]
            Float64[_bootstrap_function_value(f,x...) for x in eachrow(selected)]
        elseif a isa SingleIndexApproximator || a isa TransformedCovariateApproximator
            beta=collect(Float64.(sol.parameters[block]))
            Float64[_bootstrap_function_value(x->_eval_approx_at(a,beta,x),x) for x in selected]
        else
            Float64[_bootstrap_function_value(sol.unknown_functions[a.name],x) for x in selected]
        end
        samples=values[ids,:]
        usable=[count(isfinite,@view samples[i,:]) for i in axes(samples,1)]
        critical=nothing
        if interval === :simultaneous
            calibrated=_bootstrap_grid_band(samples,fitted;level)
            se,critical=calibrated.se,calibrated.critical
            lower,upper=calibrated.lower,calibrated.upper
        else
            se,lower,upper=fill(NaN,length(ids)),fill(NaN,length(ids)),fill(NaN,length(ids))
            alpha=(1-level)/2
            for i in eachindex(ids)
                finite=filter(isfinite,@view samples[i,:])
                if length(finite)>=3
                    se[i]=std(finite)
                    lower[i],upper[i]=quantile(finite,alpha),quantile(finite,1-alpha)
                end
            end
            any(<(bs.n_success),usable) &&
                @warn "confidence_band: pointwise bootstrap intervals use incomplete function samples" function_name=a.name usable
        end
        all(x->isfinite(x),lower) && all(x->isfinite(x),upper) || interval === :pointwise ||
            throw(DomainError((lower,upper),"confidence_band: non-finite simultaneous endpoints"))
        band=explicit ? (;points=selected,fitted,lower,upper,se) : (;grid=selected,fitted,lower,upper,se)
        result[a.name]=merge(band,(;interval,
            method=interval === :pointwise ? :percentile : :bootstrap_maxdev,
            critical,level,scope=interval === :pointwise ? :pointwise : :function_grid,
            conditioning=:refit_procedure,n_draws=bs.n_success,usable))
    end
    result
end
