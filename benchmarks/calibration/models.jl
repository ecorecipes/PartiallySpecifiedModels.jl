include("../kan/smoothing_profiles.jl")
import Distributions

const CE_BASE_MODELS = ["spline8","gp12","spde12","mlp13","kan12"]
const CE_MODELS = ["spline8","spline12","spline20","gp12","gp20","gp12_short","gp12_long",
    "spde12","spde20","spde12_short","spde12_long","mlp13","mlp25","kan12","kan16"]
const CE_METHODS = ["conditional","bootstrap_percentile","bootstrap_t","bootstrap_t_pilot",
    "split_bias_bank","split_bias_ellipsoid"]
const CE_DOMAIN = (0.0,2.4)
const CE_REFERENCE_CENTER = [0.6,-0.2,0.0,0.0,0.0,0.0]
const CE_REFERENCE_WEIGHTS = Float64.((1:6).^2)
const CE_REFERENCE_RADIUS = 1.0
const CE_CONFIRM_EXTRA_DESIGNS = ["diverse","replicated"]
const CE_CONFIRM_EXTRA_NOISES = ["hetero","ar1"]

function ce_seed(parts...)
    text = join(string.(parts),"|")
    parse(UInt64,bytes2hex(SHA.sha256(text))[1:16];base=16) & 0x7fffffffffffffff
end

function ce_model(id)
    id in CE_MODELS || throw(ArgumentError("unknown calibration model $id"))
    initial = x->0.5
    a, family, description = if startswith(id,"spline")
        n = parse(Int,id[7:end])
        BSplineApproximator(:r,CE_DOMAIN,n;initial),"spline","intrinsic cubic curvature"
    elseif startswith(id,"gp")
        n = startswith(id,"gp20") ? 20 : 12
        lengthscale = endswith(id,"short") ? 0.2 : endswith(id,"long") ? 0.8 : 0.4
        GPApproximator(:r,CE_DOMAIN,n;kernel=:matern52,lengthscale,variance=1.0,initial),
            "gp","finite Matern-5/2 interpolant, nodes=$n, lengthscale=$lengthscale, variance=1; nodal curvature; no residual process variance"
    elseif startswith(id,"spde")
        n = startswith(id,"spde20") ? 20 : 12
        range_param = endswith(id,"short") ? 0.4 : endswith(id,"long") ? 1.6 : 0.8
        SPDEApproximator(:r,CE_DOMAIN,n;nu=1.5,range_param,initial),
            "spde","full-rank Matern precision, nodes=$n, nu=1.5, range=$range_param"
    elseif startswith(id,"mlp")
        width = id=="mlp13" ? 4 : 8
        NeuralApproximator(:r,Lux.Chain(Lux.Dense(1,width,tanh),Lux.Dense(width,1));
            domain=CE_DOMAIN,penalty_weight=1.0,rng_seed=42),
            "mlp","full-rank weight/bias ridge; width=$width; hidden-feature seed=42"
    else
        grid = id=="kan12" ? 8 : 12
        KANApproximator(:r,LuxKANLinear(1,1;grid_size=grid,standalone_spline_scale=false);
            input_domains=(CE_DOMAIN,),penalty=:edge_curvature,nullspace_penalty=0.0,rng_seed=42),
            "kan","single-edge complete curvature; free affine part; fixed grid"
    end
    beta = initial_params(a)
    if a isa NeuralApproximator
        width = id=="mlp13" ? 4 : 8
        beta[end-width:end-1] .= 0.0
        beta[end] = 0.5
    elseif a isa KANApproximator
        beta[only(a.layers).base_range] .= 0.0
        beta[only(a.layers).spline_range] .= 0.5
    end
    S = penalty_matrix(a)
    frame = sp_penalty_frame(S;allow_proper=true)
    if frame.nullity > 0
        frame = sp_center_frame(frame,sp_linear_design(a,collect(range(CE_DOMAIN...;length=401))))
    end
    metadata = Dict{String,Any}("family"=>family,"parameters"=>nparams(a),"description"=>description,
        "penalty_rank"=>frame.rank,"nullity"=>frame.nullity,"initial_parameters"=>beta,
        "initial_values"=>build_evaluator(a,beta).([0.3,0.6,1.2,1.8]),
        "initialization"=>"response 0.5 where representable; GP inducing values 0.5")
    (;id,family,approx=a,initial=beta,frame,metadata)
end

function ce_legendre(x; degree=5)
    z = 2x/CE_DOMAIN[2]-1
    values = [one(z),z]
    for k in 2:degree
        push!(values,((2k-1)*z*values[end]-(k-1)*values[end-1])/k)
    end
    values[1:degree+1]
end

function ce_integrated_legendre(t,speed=1.0)
    zvalues = ce_legendre(speed*t;degree=6)
    [t;[CE_DOMAIN[2]/(2speed*(2k+1))*(zvalues[k+2]-zvalues[k]) for k in 1:5]]
end

function ce_poly_coefficients(id)
    direction = id=="poly_a" ? [0.0,0.0,1.0,0.0,0.0,0.0] :
        id=="poly_b" ? [0.0,0.0,0.0,0.6,0.8,0.0] :
        id=="poly_holdout" ? [0.1,-0.2,0.3,-0.4,0.5,-0.6] :
        throw(ArgumentError("unknown polynomial truth $id"))
    CE_REFERENCE_CENTER + 0.8CE_REFERENCE_RADIUS .* direction ./ norm(direction) ./ CE_REFERENCE_WEIGHTS
end

function ce_bank()
    bank = Dict{String,Function}("affine"=>x->0.9*(1-x/2))
    for (suffix,amplitude) in (("minus",-0.35),("plus",0.35))
        for frequency in (2,4)
            bank["sine$(frequency)_$suffix"] = let a=amplitude,w=frequency
                x->0.9*(1-x/2)*exp(a*sin(w*x))
            end
        end
        for (i,center) in enumerate((0.6,1.0,1.4))
            bank["bump$(i)_$suffix"] = let a=amplitude,c=center
                x->0.9*(1-x/2)*exp(a*exp(-((x-c)/0.25)^2))
            end
        end
        for (i,center) in enumerate((0.7,1.1))
            bank["wave$(i)_$suffix"] = let a=amplitude,c=center
                x->0.9*(1-x/2)*exp(a*sin(60*(x-c))*exp(-((x-c)/0.12)^2))
            end
        end
    end
    bank
end

function ce_truth(id)
    bank = ce_bank()
    haskey(bank,id) && return bank[id]
    startswith(id,"poly") && return let b=ce_poly_coefficients(id)
        x->dot(ce_legendre(x),b)
    end
    id=="narrow" && return x->0.9*(1-x/2)*exp(0.5exp(-((x-0.9)/0.08)^2))
    id=="shoulder" && return x->0.9*(1-x/2)*(0.65+0.7/(1+exp(-12*(x-0.85))))
    id=="double" && return x->0.9*(1-x/2)*exp(0.4exp(-((x-0.65)/0.16)^2)-0.3exp(-((x-1.3)/0.13)^2))
    id=="chirp" && return x->0.9*(1-x/2)*exp(0.3sin(2x+4x^2))
    throw(ArgumentError("unknown calibration truth $id"))
end

function ce_cases(stage,operator)
    stage in ("smoke","develop","confirm") || throw(ArgumentError("unknown experiment stage"))
    operator in ("growth","integral") || throw(ArgumentError("unknown observation operator"))
    stage=="smoke" && return operator=="growth" ? ["sine2_plus"] : ["poly_a"]
    operator=="growth" && return stage=="develop" ?
        ["affine","sine2_plus","bump2_minus","weak_a","weak_b"] :
        ["affine","sine2_plus","bump2_minus","weak_a","weak_b","narrow","shoulder","double","chirp"]
    stage=="develop" ? ["poly_a","poly_b","sine2_plus"] :
        ["poly_a","poly_b","poly_holdout","sine2_plus","narrow","chirp"]
end

function ce_design(operator,id)
    operator in ("growth","integral") && id in ("baseline","diverse","replicated") ||
        throw(ArgumentError("unknown operator or observation design"))
    ntraj,ntimes = id=="baseline" ? (2,20) : (4,10)
    speeds = operator=="integral" && id=="diverse" ? [0.55,0.8,1.0,1.25] : ones(ntraj)
    u0 = operator=="integral" ? zeros(ntraj) : id=="baseline" ? [0.2,0.7] :
        id=="diverse" ? [0.2,0.55,1.0,1.45] : [0.2,0.7,0.2,0.7]
    horizon = operator=="growth" ? 8.0 : CE_DOMAIN[2]/maximum(speeds)
    (;operator,id,u0,speeds,times=collect(range(0.0,horizon;length=ntimes)),horizon)
end

function ce_noise(design,id)
    id in ("iid","hetero","ar1") || throw(ArgumentError("unknown noise design"))
    nt,np = length(design.times),length(design.u0)
    scale = id=="hetero" ? 0.7 .+ 0.6 .* design.times ./ design.horizon : ones(nt)
    correlation = id=="ar1" ? [0.5^abs(i-j) for i in 1:nt,j in 1:nt] : Matrix{Float64}(I,nt,nt)
    shape = kron(Matrix{Float64}(I,np,np),Diagonal(scale)*correlation*Diagonal(scale))
    (;id,shape,root=Matrix(cholesky(Symmetric(shape)).L))
end

function ce_output_map(points)
    n = length(points)
    n >= 3 && issorted(points) && allunique(points) || throw(ArgumentError("invalid target grid"))
    gaps = diff(points)
    weights = [first(gaps)/2; (gaps[1:end-1]+gaps[2:end])/2; last(gaps)/2] ./ (last(points)-first(points))
    contrast = zeros(n)
    contrast[argmin(abs.(points.-0.6))] = 1
    contrast[argmin(abs.(points.-1.2))] -= 1
    vcat(Matrix{Float64}(I,n,n),permutedims(weights),permutedims(contrast))
end

function ce_predictions(f,design)
    rhs! = design.operator=="growth" ?
        ((du,u,p,t)->(du .= f.(u).*u)) :
        ((du,u,p,t)->(du .= f.(design.speeds.*t)))
    solution = PSM.OrdinaryDiffEq.solve(ODEProblem(rhs!,design.u0,(0.0,design.horizon)),Tsit5();
        saveat=design.times,abstol=1e-11,reltol=1e-11)
    PSM.SciMLBase.successful_retcode(solution) || throw(DomainError(design.id,"reference integration failed"))
    predictions = permutedims(reduce(hcat,[solution(t) for t in design.times]))
    all(isfinite,predictions) || throw(DomainError(predictions,"non-finite reference"))
    predictions
end

function ce_bank_cache(design,points)
    bank = ce_bank()
    if design.operator=="integral"
        for id in ("poly_a","poly_b")
            bank[id] = ce_truth(id)
        end
    end
    ids = sort!(collect(keys(bank)))
    predictions = hcat([vec(startswith(id,"poly") ?
        [dot(ce_integrated_legendre(t,v),ce_poly_coefficients(id)) for t in design.times,v in design.speeds] :
        ce_predictions(bank[id],design)) for id in ids]...)
    outputs = hcat([ce_output_map(points)*bank[id].(points) for id in ids]...)
    (;ids,predictions,outputs)
end

function ce_weak_pair(cache,noise,sigma)
    sigma>0 || throw(ArgumentError("positive noise scale required"))
    rows = NamedTuple[]
    for i in 1:length(cache.ids), j in i+1:length(cache.ids)
        distance = norm(noise.root \ (cache.predictions[:,i]-cache.predictions[:,j]))/sigma
        separation = maximum(abs,cache.outputs[:,i]-cache.outputs[:,j])
        push!(rows,(;first=cache.ids[i],second=cache.ids[j],distance,
            kl=distance^2/2,separation,ratio=distance/max(separation,eps(Float64))))
    end
    eligible = filter(r->r.separation>=0.02,rows)
    isempty(eligible) && throw(DomainError(cache.ids,"no separated candidate functions"))
    chosen = eligible[argmin(getproperty.(eligible,:ratio))]
    (;chosen,rows)
end

function ce_dataset(stage,operator,design_id,noise_id,sigma,case,seed,points,cache=nothing)
    design = ce_design(operator,design_id)
    noise = ce_noise(design,noise_id)
    cache === nothing && (cache=ce_bank_cache(design,points))
    actual = case
    if case in ("weak_a","weak_b")
        pair = ce_weak_pair(cache,noise,sigma).chosen
        actual = case=="weak_a" ? pair.first : pair.second
    end
    bank_index = findfirst(==(actual),cache.ids)
    polynomial = startswith(actual,"poly")
    polynomial && operator!="integral" && throw(ArgumentError("polynomial controls require the integral operator"))
    f = ce_truth(actual)
    reference = if polynomial
        b = ce_poly_coefficients(actual)
        [dot(ce_integrated_legendre(t,v),b) for t in design.times,v in design.speeds]
    elseif bank_index !== nothing
        reshape(copy(cache.predictions[:,bank_index]),length(design.times),:)
    else
        ce_predictions(f,design)
    end
    truth = ce_output_map(points)*f.(points)
    rng_seed = ce_seed("data",stage,operator,design_id,noise_id,sigma,case,seed)
    observed = reference + reshape(sigma.*(noise.root*randn(StableRNG(rng_seed),length(reference))),size(reference))
    support = operator=="growth" ? extrema(reference) : (0.0,maximum(design.speeds)*design.horizon)
    (;stage,operator,design_id,noise_id,sigma,case,seed,actual,design,noise,cache,
      reference,observed,truth,points,support,rng_seed,truth_in_bank=bank_index!==nothing,
      truth_in_ellipsoid=polynomial)
end

function ce_problem(model,ds,values=ds.observed)
    design = ds.design
    rhs! = design.operator=="growth" ?
        ((du,u,p,t)->(du .= p.r.(u).*u)) :
        ((du,u,p,t)->(du .= p.r.(design.speeds.*t)))
    PSMProblem(rhs!,design.u0,(0.0,design.horizon),
        [InitializedApprox(:r,model.approx,model.initial,penalty_matrix(model.approx))];
        data_times=design.times,data_values=values,abstol=1e-8,reltol=1e-8,maxiters=10_000)
end
