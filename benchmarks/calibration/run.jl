include("inference.jl")

ce_dictionary(x::NamedTuple) = Dict{String,Any}(string(k)=>v for (k,v) in pairs(x))
ce_matrix(x) = collect.(eachrow(x))

function ce_scripts()
    [joinpath(@__DIR__,f) for f in ("models.jl","inference.jl","run.jl")] |>
        files -> vcat(files,[joinpath(@__DIR__,"..","kan",f) for f in
            ("smoothing_profiles.jl","coverage_robustness.jl","undersmoothing.jl",
             "calibrate_families.jl","calibrate_uncertainty.jl","rd_archive.jl","reaction_diffusion.jl")])
end

function ce_code_hash()
    context = SHA.SHA2_256_CTX()
    SHA.update!(context,codeunits(source_hash()))
    for file in ce_scripts()
        SHA.update!(context,codeunits(basename(file)*"\0"))
        SHA.update!(context,read(file))
    end
    bytes2hex(SHA.digest!(context))
end

function ce_options(args)
    options = Dict("stage"=>"smoke","models"=>join(CE_BASE_MODELS,","),
        "methods"=>"conditional,bootstrap_t,split_bias_bank,split_bias_ellipsoid",
        "operators"=>"growth,integral","designs"=>"baseline","noises"=>"iid",
        "noise-modes"=>"known","sigmas"=>"0.015","n-datasets"=>"","seed-start"=>"1",
        "iterations"=>"","tol"=>"1e-8","rhos"=>"","nboot"=>"","nsim"=>"",
        "ngrid"=>"31","level"=>"0.95","pilot-fraction"=>"0.25","output"=>"","lock"=>"",
        "plan-only"=>"false","shard"=>"1/1")
    supplied = Set{String}()
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(options,key) || throw(ArgumentError("unknown calibration option $key"))
        options[key] = value
        push!(supplied,key)
    end
    stage = options["stage"]
    stage in ("smoke","develop","confirm") || throw(ArgumentError("stage must be smoke, develop or confirm"))
    lock = nothing
    options["plan-only"] in ("true","false") || throw(ArgumentError("plan-only must be true or false"))
    if stage=="confirm"
        isempty(options["lock"]) && throw(ArgumentError("confirmation requires a frozen development --lock"))
        lock = TOML.parsefile(options["lock"])
        lock["schema"]=="calibration-lock-v1" && lock["source_stage"]=="develop" ||
            throw(ArgumentError("invalid development lock"))
        lock["code_sha256"]==ce_code_hash() || throw(ArgumentError("numerical code differs from the development lock"))
        for key in ("models","methods","operators","designs","noises","noise-modes","sigmas",
                    "iterations","tol","rhos","ngrid","level","pilot-fraction")
            value = lock["options"][key]
            key in supplied && options[key]!=value &&
                throw(ArgumentError("confirmation cannot override locked $key"))
            options[key] = value
        end
    end
    models = String.(split(options["models"],','))
    methods = String.(split(options["methods"],','))
    operators = String.(split(options["operators"],','))
    designs = String.(split(options["designs"],','))
    noises = String.(split(options["noises"],','))
    noise_modes = String.(split(options["noise-modes"],','))
    sigmas = parse.(Float64,split(options["sigmas"],','))
    for (values,allowed) in ((models,CE_MODELS),(methods,CE_METHODS),(operators,["growth","integral"]),
                            (designs,["baseline","diverse","replicated"]),(noises,["iid","hetero","ar1"]),
                            (noise_modes,["known","estimated"]))
        !isempty(values) && allunique(values) && all(in(allowed),values) ||
            throw(ArgumentError("invalid or duplicated experiment identifiers"))
    end
    for operator in operators, mode in noise_modes
        any(ce_applicable(m,operator,mode) for m in methods) ||
            throw(ArgumentError("no requested inference method applies to $operator / $mode"))
    end
    default_count = stage=="smoke" ? 1 : stage=="develop" ? 20 : lock["confirmation_datasets"]
    count = isempty(options["n-datasets"]) ? default_count : parse(Int,options["n-datasets"])
    start = parse(Int,options["seed-start"])
    count>0 && start>0 || throw(ArgumentError("positive dataset counts/indices required"))
    seeds = collect(start:start+count-1)
    shard = parse.(Int,split(options["shard"],'/'))
    length(shard)==2 && 1<=shard[1]<=shard[2] || throw(ArgumentError("shard must be i/n with 1 <= i <= n"))
    stage=="confirm" && last(seeds)>lock["confirmation_datasets"] &&
        throw(ArgumentError("confirmation shard exceeds the locked population"))
    iterations = isempty(options["iterations"]) ? (stage=="smoke" ? 16 : 40) : parse(Int,options["iterations"])
    rhos = isempty(options["rhos"]) ? (stage=="smoke" ? [-12.0,-8.0,-4.0,0.0] :
        [-16.0,-12.0,-10.0,-8.0,-6.0,-4.0,-2.0,2.0,8.0]) : parse.(Float64,split(options["rhos"],','))
    nboot = isempty(options["nboot"]) ? (stage=="smoke" ? 3 : stage=="develop" ? 99 : lock["confirmation_bootstrap"]) :
        parse(Int,options["nboot"])
    nsim = isempty(options["nsim"]) ? (stage=="smoke" ? 1000 : stage=="confirm" ? lock["confirmation_nsim"] : 10000) :
        parse(Int,options["nsim"])
    stage=="confirm" && nboot!=lock["confirmation_bootstrap"] &&
        throw(ArgumentError("confirmation bootstrap budget is locked"))
    stage=="confirm" && nsim!=lock["confirmation_nsim"] &&
        throw(ArgumentError("confirmation Gaussian simulation budget is locked"))
    tol,level,pilot_fraction = parse.(Float64,[options["tol"],options["level"],options["pilot-fraction"]])
    ngrid = parse(Int,options["ngrid"])
    all(x->isfinite(x)&&x>0,sigmas) && allunique(sigmas) && all(isfinite,rhos) && allunique(rhos) &&
        !isempty(rhos) && iterations>0 && nboot>=3 && nsim>=2 && ngrid>=7 &&
        isfinite(tol) && tol>0 && 0<level<1 && 0<pilot_fraction<=1 ||
        throw(ArgumentError("invalid numerical experiment controls"))
    output = isempty(options["output"]) ? joinpath(@__DIR__,"results",stage) : abspath(options["output"])
    canonical = Dict("models"=>join(models,","),"methods"=>join(methods,","),"operators"=>join(operators,","),
        "designs"=>join(designs,","),"noises"=>join(noises,","),"noise-modes"=>join(noise_modes,","),
        "sigmas"=>join(sigmas,","),"iterations"=>string(iterations),"tol"=>string(tol),
        "rhos"=>join(rhos,","),"ngrid"=>string(ngrid),"level"=>string(level),"pilot-fraction"=>string(pilot_fraction))
    (;stage,models,methods,operators,designs,noises,noise_modes,sigmas,seeds,iterations,tol,rhos,
      nboot,nsim,ngrid,level,pilot_fraction,output,canonical,lock,
      lock_sha256=lock===nothing ? "" : rd_digest(options["lock"]),
      shard=(shard[1],shard[2]),plan_only=options["plan-only"]=="true")
end

function ce_applicable(method,operator,noise_mode)
    !startswith(method,"split_bias") || (noise_mode=="known" &&
        (method!="split_bias_ellipsoid" || operator=="integral"))
end

function ce_selected(cfg,model,method,operator,noise_mode)
    cfg.lock===nothing && return true
    any(r->r["model"]==model && r["method"]==method &&
        r["operator"]==operator && r["noise_mode"]==noise_mode,cfg.lock["participants"])
end

function ce_interval_rows(ds,model,family,noise_mode,method,estimates,intervals;
                          status="ok",message="",guarantee="none",assumption=false,reference=false)
    n = length(ds.points)
    common = (;stage=ds.stage,operator=ds.operator,case=ds.case,actual_truth=ds.actual,
        design=ds.design_id,noise=ds.noise_id,sigma=ds.sigma,seed=ds.seed,model,family,noise_mode,method,
        guarantee,assumption_satisfied=assumption,reference_only=reference,status,message)
    rows = NamedTuple[]
    for interval in ("pointwise","simultaneous")
        method=="bootstrap_percentile" && interval=="simultaneous" && continue
        ids = interval=="pointwise" ? eachindex(ds.truth) : 1:n
        band = intervals===nothing ? nothing : getproperty(intervals,Symbol(interval))
        for i in ids
            lower = band===nothing ? NaN : band.lower[i]
            upper = band===nothing ? NaN : band.upper[i]
            available = status=="ok" && isfinite(lower) && isfinite(upper) && lower<=upper
            row_status = status=="ok" && !available ? "invalid_interval" : status
            row_message = row_status=="invalid_interval" ? "non-finite or reversed interval endpoints" : message
            scope = i<=n ? "density" : i==n+1 ? "grid_mean" : "contrast"
            region = i>n ? "functional" : ds.points[i]<ds.support[1] || ds.points[i]>ds.support[2] ?
                "outside_support" : "in_support"
            push!(rows,(;common...,status=row_status,message=row_message,interval,scope,region,target_id=i,x=i<=n ? ds.points[i] : NaN,
                truth=ds.truth[i],estimate=estimates===nothing ? NaN : estimates[i],lower,upper,
                available,covered=available && lower<=ds.truth[i]<=upper,
                width=available ? upper-lower : NaN,critical=band===nothing ? NaN : band.critical))
        end
    end
    rows
end

function ce_evaluate_model(model,ds,cfg,noise_mode,methods)
    opts = (;iterations=cfg.iterations,tol=cfg.tol,rhos=cfg.rhos,nboot=cfg.nboot,
        noise_mode,pilot_fraction=cfg.pilot_fraction)
    result = ce_fit(model,ds,ds.observed,opts)
    fit = result.fit
    record = Dict{String,Any}("model"=>model.id,"family"=>model.family,"operator"=>ds.operator,
        "case"=>ds.case,"seed"=>ds.seed,"noise_mode"=>noise_mode,"status"=>result.status,"message"=>result.message,
        "candidates"=>ce_dictionary.(result.attempts),"bootstrap"=>Dict{String,Any}())
    intervals = NamedTuple[]
    if fit !== nothing
        merge!(record,Dict("parameters"=>fit.beta,"rho"=>fit.rho,"sigma2"=>fit.sigma2,
            "edf"=>fit.score.edf,"decrement"=>fit.score.decrement,"converged"=>fit.converged,
            "grid_boundary"=>fit.grid_boundary,"estimates"=>fit.estimates,"se"=>fit.se))
    end
    boot = Dict{String,Any}()
    split = nothing
    split_error = ""
    for method in methods
        bands,status,message = nothing,result.status,result.message
        estimates = fit===nothing ? nothing : fit.estimates
        guarantee,assumption = "none",false
        if fit !== nothing || startswith(method,"split_bias")
            try
                if method=="conditional"
                    rng = StableRNG(ce_seed("gaussian",ds.stage,ds.operator,ds.design_id,ds.noise_id,ds.sigma,ds.case,ds.seed))
                    bands = ce_conditional(fit,length(ds.points),cfg.level,cfg.nsim,rng)
                elseif startswith(method,"bootstrap")
                    generator = method=="bootstrap_t_pilot" ? "pilot" : "fit"
                    if !haskey(boot,generator)
                        b = ce_bootstrap(model,ds,fit,opts,generator)
                        boot[generator] = b
                        record["bootstrap"][generator] = Dict("values"=>ce_matrix(b.draws),"se"=>ce_matrix(b.scales),
                            "generating"=>b.generating,"generating_rho"=>b.generating_rho,
                            "generating_sigma2"=>b.generating_sigma2,"attempts"=>b.records)
                    end
                    b = boot[generator]
                    if method=="bootstrap_percentile"
                        all(isfinite,b.draws) || throw(DomainError(b.draws,"percentiles require complete attempted refits"))
                        alpha = (1-cfg.level)/2
                        bands = (pointwise=(lower=[quantile(row,alpha) for row in eachrow(b.draws)],
                            upper=[quantile(row,1-alpha) for row in eachrow(b.draws)],critical=NaN),simultaneous=nothing)
                    else
                        bands = ce_studentized(fit.estimates,fit.se,b.generating,b.draws,b.scales,
                            length(ds.points),cfg.level)
                    end
                else
                    if split===nothing && isempty(split_error)
                        split = ce_split_inference(model,ds,opts)
                        record["split_pilot"] = Dict("status"=>split.pilot.status,
                            "candidates"=>ce_dictionary.(split.pilot.attempts))
                        split.value===nothing && (split_error=split.pilot.message)
                        if split.value !== nothing
                            s = split.value
                            record["split"] = Dict("estimates"=>s.estimates,"se"=>s.se,"bank_bias"=>s.bank_bias,
                                "train"=>s.train,"infer"=>s.infer,"pilot_rho"=>s.pilot_rho,
                                "M"=>ce_matrix(s.M),"offset"=>s.offset,
                                "innovation_regression"=>ce_matrix(s.innovation_regression))
                            s.ellipsoid_bias===nothing || (record["split"]["ellipsoid_bias"]=s.ellipsoid_bias)
                        end
                    end
                    isempty(split_error) || throw(DomainError(split_error,"split pilot failed"))
                    s = split.value
                    estimates = s.estimates
                    bound = method=="split_bias_bank" ? s.bank_bias : s.ellipsoid_bias
                    bands = ce_bias_intervals(estimates,s.se,bound,length(ds.points),cfg.level)
                    guarantee = method=="split_bias_bank" ? "known_gaussian_finite_class" : "known_gaussian_polynomial_ball"
                    assumption = method=="split_bias_bank" ? ds.truth_in_bank : ds.truth_in_ellipsoid
                end
                status,message = "ok",""
            catch e
                status,message = "unavailable",ce_error(e)
            end
        end
        append!(intervals,ce_interval_rows(ds,model.id,model.family,noise_mode,method,estimates,bands;
            status,message,guarantee,assumption))
    end
    (;record,intervals)
end

function ce_run(args)
    cfg = ce_options(args)
    all_jobs = [(;operator,case,design,noise,sigma,seed) for operator in cfg.operators
        for case in ce_cases(cfg.stage,operator) for design in cfg.designs for noise in cfg.noises
        for sigma in cfg.sigmas for seed in cfg.seeds]
    jobs = [job for (i,job) in enumerate(all_jobs) if mod(i-1,cfg.shard[2])==cfg.shard[1]-1]
    isempty(jobs) && throw(ArgumentError("requested shard has no datasets"))
    fit_jobs = sum(any(ce_applicable(m,j.operator,nm) && ce_selected(cfg,model,m,j.operator,nm)
        for m in cfg.methods) for j in jobs for model in cfg.models for nm in cfg.noise_modes)
    refits = 0
    for j in jobs, model in cfg.models, nm in cfg.noise_modes
        methods = [m for m in cfg.methods if ce_applicable(m,j.operator,nm) && ce_selected(cfg,model,m,j.operator,nm)]
        isempty(methods) && continue
        generators = Set(m=="bootstrap_t_pilot" ? "pilot" : "fit" for m in methods if startswith(m,"bootstrap"))
        refits += 1 + any(startswith(m,"split_bias") for m in methods) + cfg.nboot*length(generators)
    end
    println("Plan: $(length(jobs)) datasets, $fit_jobs model/noise-arm fits; " *
        "up to $refits profile selections including bootstraps/pilots; $(2length(cfg.rhos)) candidates per selection. " *
        "Shard $(cfg.shard[1])/$(cfg.shard[2]).")
    cfg.plan_only && return cfg
    ce_execute(cfg,jobs,fit_jobs)
end

Base.@nospecializeinfer function ce_execute(@nospecialize(cfg),@nospecialize(jobs),fit_jobs)
    ispath(cfg.output) && error("experiment output already exists; choose a fresh directory")
    BLAS.set_num_threads(1)
    scripts = ce_scripts()
    initial_source = source_hash()
    code_hash = ce_code_hash()
    hashes = rd_followon_snapshot(cfg.output,scripts)
    mkpath(joinpath(cfg.output,"datasets"))
    mkpath(joinpath(cfg.output,"fits"))
    design = Dict{String,Any}("schema"=>"calibration-suite-v1","stage"=>cfg.stage,
        "code_sha256"=>code_hash,"options"=>cfg.canonical,"models"=>cfg.models,"methods"=>cfg.methods,
        "operators"=>cfg.operators,"designs"=>cfg.designs,"noises"=>cfg.noises,"noise_modes"=>cfg.noise_modes,
        "sigmas"=>cfg.sigmas,"seeds"=>cfg.seeds,"nboot"=>cfg.nboot,"nsim"=>cfg.nsim,
        "ngrid"=>cfg.ngrid,"level"=>cfg.level,"lock_sha256"=>cfg.lock_sha256,
        "expected_datasets"=>length(jobs),"expected_fit_jobs"=>fit_jobs,
        "dataset_jobs"=>ce_dictionary.(jobs),"shard"=>collect(cfg.shard),
        "confirmation_extra_designs"=>CE_CONFIRM_EXTRA_DESIGNS,
        "confirmation_extra_noises"=>CE_CONFIRM_EXTRA_NOISES,
        "confirmation_total"=>cfg.lock===nothing ? 0 : cfg.lock["confirmation_datasets"],
        "cases"=>Dict(op=>ce_cases(cfg.stage,op) for op in cfg.operators),
        "model_specs"=>Dict(id=>ce_model(id).metadata for id in cfg.models),
        "known_covariance"=>"Gaussian covariance shape is known; scale is known or estimated as labelled",
        "reference_scope"=>"finite function bank; polynomial ellipsoid/span only under the integral operator",
        "reference_center"=>CE_REFERENCE_CENTER,"reference_weights"=>CE_REFERENCE_WEIGHTS,
        "reference_radius"=>CE_REFERENCE_RADIUS,
        "functional_scope"=>"trapezoidal grid-weighted mean and nearest-grid 0.6-minus-1.2 contrast",
        "bootstrap_policy"=>"reselect smoothing and refit coefficients from declared starts on every attempt; no dropping failed curves",
        "confirmation_case_policy"=>"fresh noise for development truths plus held-out functional forms",
        "participants"=>cfg.lock===nothing ? Dict{String,Any}[] : cfg.lock["participants"])
    rd_write_toml(joinpath(cfg.output,"design.toml"),design)
    interval_rows,fit_rows,weak_rows = NamedTuple[],NamedTuple[],NamedTuple[]
    caches = Dict{Tuple{String,String},Any}()
    paired = Set{Tuple{String,String,String,Float64}}()
    points = collect(range(0.3,1.8;length=cfg.ngrid))
    for job in jobs
        key = (job.operator,job.design)
        if !haskey(caches,key)
            caches[key] = ce_bank_cache(ce_design(key...),points)
        end
        ds = ce_dataset(cfg.stage,job.operator,job.design,job.noise,job.sigma,job.case,job.seed,points,caches[key])
        id = join((job.operator,job.design,job.noise,job.sigma,job.case,job.seed),"__")
        dataset_path = joinpath(cfg.output,"datasets",id*".toml")
        rd_write_toml(dataset_path,Dict("id"=>id,"rng_seed"=>string(ds.rng_seed),"actual_truth"=>ds.actual,
            "observed"=>collect.(eachcol(ds.observed)),"reference"=>collect.(eachcol(ds.reference)),
            "points"=>points,"targets"=>ds.truth,"times"=>ds.design.times,"initial"=>ds.design.u0,
            "speeds"=>ds.design.speeds,"noise_shape"=>ce_matrix(ds.noise.shape),
            "truth_in_bank"=>ds.truth_in_bank,"truth_in_ellipsoid"=>ds.truth_in_ellipsoid,
            "bank_ids"=>ds.cache.ids))
        pair_key = (job.operator,job.design,job.noise,job.sigma)
        if !(pair_key in paired)
            pair = ce_weak_pair(ds.cache,ds.noise,ds.sigma)
            append!(weak_rows,[(;operator=job.operator,design=job.design,noise=job.noise,sigma=job.sigma,
                selected=r.first==pair.chosen.first && r.second==pair.chosen.second,r...) for r in pair.rows])
            push!(paired,pair_key)
        end
        for noise_mode in cfg.noise_modes
            if noise_mode=="known"
                for method in (job.operator=="integral" ? ("bank_set","linear_set") : ("bank_set",))
                    result = method=="bank_set" ? ce_bank_set(ds,cfg.level) : ce_linear_set(ds,cfg.level)
                    status = result.intervals===nothing ? "empty_set" : "ok"
                    append!(interval_rows,ce_interval_rows(ds,"reference","reference",noise_mode,method,
                        result.estimates,result.intervals;status,message=status=="ok" ? "" : "reference class rejected",
                        guarantee=method=="bank_set" ? "known_gaussian_finite_class" : "known_gaussian_polynomial_span",
                        assumption=method=="bank_set" ? ds.truth_in_bank : ds.truth_in_ellipsoid,reference=true))
                end
            end
            for model_id in cfg.models
                methods = [m for m in cfg.methods if ce_applicable(m,job.operator,noise_mode) &&
                    ce_selected(cfg,model_id,m,job.operator,noise_mode)]
                isempty(methods) && continue
                model = ce_model(model_id)
                output = ce_evaluate_model(model,ds,cfg,noise_mode,methods)
                append!(interval_rows,output.intervals)
                r = output.record
                push!(fit_rows,(;operator=job.operator,case=job.case,design=job.design,noise=job.noise,
                    sigma=job.sigma,seed=job.seed,model=model_id,family=model.family,noise_mode,
                    status=r["status"],message=r["message"],rho=get(r,"rho",NaN),sigma2=get(r,"sigma2",NaN),
                    edf=get(r,"edf",NaN),decrement=get(r,"decrement",NaN),
                    converged=get(r,"converged",false),grid_boundary=get(r,"grid_boundary",false)))
                rd_write_toml(joinpath(cfg.output,"fits",id*"__"*model_id*"__"*noise_mode*".toml"),r)
                println("$id $model_id $noise_mode $(r["status"])")
                flush(stdout)
            end
        end
    end
    length(fit_rows)==fit_jobs || error("missing fit jobs")
    CSV.write(joinpath(cfg.output,"fits.csv"),fit_rows)
    CSV.write(joinpath(cfg.output,"intervals.csv"),interval_rows)
    CSV.write(joinpath(cfg.output,"weak_pairs.csv"),weak_rows)
    files = ["design.toml","fits.csv","intervals.csv","weak_pairs.csv"]
    append!(files,["datasets/"*name for name in readdir(joinpath(cfg.output,"datasets"))])
    append!(files,["fits/"*name for name in readdir(joinpath(cfg.output,"fits"))])
    ce_code_hash()==code_hash || error("experiment code changed during execution")
    metadata = rd_followon_metadata(cfg.output,scripts,hashes,initial_source,[])
    merge!(metadata,Dict("schema"=>"calibration-results-v1","complete"=>true,"code_sha256"=>code_hash,
        "dataset_count"=>length(jobs),"fit_count"=>length(fit_rows),"interval_count"=>length(interval_rows),
        "outputs_sha256"=>Dict(name=>rd_digest(joinpath(cfg.output,name)) for name in files)))
    rd_write_toml(joinpath(cfg.output,"metadata.toml"),metadata)
    cfg.output
end

if abspath(PROGRAM_FILE)==@__FILE__
    ce_run(ARGS)
end
