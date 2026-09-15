if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Usage: checkpointed.jl RUN_DIRECTORY [--plan-only] [--job-limit=N]")
    include(joinpath(abspath(ARGS[1]),"source","benchmarks","calibration","run.jl"))
end
include("resume.jl")

ceb_matrix(rows) = permutedims(hcat((Float64.(row) for row in rows)...))

function ceb_component(build,path,context;on_commit=(path,payload)->nothing)
    cer_pause_check()
    if isfile(path)
        saved = TOML.parsefile(path)
        saved["schema"]=="calibration-component-v1" && isequal(saved["context"],context) ||
            error("checkpoint component context differs: $path")
        cer_digest(saved["payload_toml"])==saved["payload_sha256"] ||
            error("checkpoint component fingerprint differs: $path")
        return TOML.parse(saved["payload_toml"])
    end
    payload = build()
    text = cer_text(payload)
    cer_atomic(path,cer_text(Dict("schema"=>"calibration-component-v1","context"=>context,
        "payload_toml"=>text,"payload_sha256"=>cer_digest(text))))
    on_commit(path,payload)
    payload
end

function ceb_fit_state(f)
    Dict{String,Any}("beta"=>f.beta,"rho"=>f.rho,"sigma2"=>f.sigma2,"pred"=>ce_matrix(f.pred),
        "root"=>ce_matrix(f.root),"estimates"=>f.estimates,"se"=>f.se,
        "edf"=>f.score.edf,"decrement"=>f.score.decrement,"converged"=>f.converged,
        "grid_boundary"=>hasproperty(f,:grid_boundary) ? f.grid_boundary : false)
end

function ceb_fit_value(s)
    (;beta=Float64.(s["beta"]),rho=Float64(s["rho"]),sigma2=Float64(s["sigma2"]),
      pred=ceb_matrix(s["pred"]),root=ceb_matrix(s["root"]),
      estimates=Float64.(s["estimates"]),se=Float64.(s["se"]),
      score=(edf=Float64(s["edf"]),decrement=Float64(s["decrement"])),
      converged=s["converged"],grid_boundary=s["grid_boundary"])
end

function ceb_context(model,ds,cfg,mode,plan_sha,dataset_sha)
    Dict{String,Any}("plan_sha256"=>plan_sha,"dataset_sha256"=>dataset_sha,
        "code_sha256"=>ce_code_hash(),"model"=>model.id,"noise_mode"=>mode,
        "stage"=>ds.stage,"operator"=>ds.operator,"case"=>ds.case,"seed"=>ds.seed,
        "bootstrap_attempts"=>cfg.nboot,"pilot_fraction"=>cfg.pilot_fraction)
end

function ceb_original(model,ds,cfg,mode,path,context;parent=nothing,on_commit=(p,v)->nothing,fitfun=ce_fit)
    ceb_component(path,context;on_commit) do
        if parent!==nothing
            text = read(parent,String)
            record = TOML.parse(text)
            (record["model"],record["operator"],record["case"],record["seed"],record["noise_mode"]) ==
                (model.id,ds.operator,ds.case,ds.seed,mode) || error("parent fit identity differs")
            payload = Dict{String,Any}("record"=>record,"parent_sha256"=>cer_digest(text),
                "origin"=>"screen_coefficients_without_refitting")
            if record["status"]=="ok"
                f,diagnostics = cer_working_fit(record,model,ds)
                payload["fit"],payload["replay"] = ceb_fit_state(f),diagnostics
            end
            return payload
        end
        opts = (;iterations=cfg.iterations,tol=cfg.tol,rhos=cfg.rhos,nboot=cfg.nboot,
            noise_mode=mode,pilot_fraction=cfg.pilot_fraction)
        result = fitfun(model,ds,ds.observed,opts)
        record = Dict{String,Any}("model"=>model.id,"family"=>model.family,
            "operator"=>ds.operator,"case"=>ds.case,"seed"=>ds.seed,"noise_mode"=>mode,
            "status"=>result.status,"message"=>result.message,"bootstrap"=>Dict{String,Any}(),
            "candidates"=>ce_dictionary.(result.attempts))
        payload = Dict{String,Any}("record"=>record,"origin"=>"new_fit")
        if result.fit!==nothing
            f = result.fit
            merge!(record,Dict("parameters"=>f.beta,"rho"=>f.rho,"sigma2"=>f.sigma2,
                "edf"=>f.score.edf,"decrement"=>f.score.decrement,"converged"=>f.converged,
                "grid_boundary"=>f.grid_boundary,"estimates"=>f.estimates,"se"=>f.se))
            payload["fit"] = ceb_fit_state(f)
        end
        payload
    end
end

function ceb_bootstrap(model,ds,fit,opts,generator,directory,context;
                       on_commit=(p,v)->nothing,fitfun=ce_fit,pilotfun=ce_fixed_fit)
    generator in ("fit","pilot") || throw(ArgumentError("unknown bootstrap generator"))
    generator_path = joinpath(directory,"generator.toml")
    state = ceb_component(generator_path,merge(context,Dict("component"=>"generator","generator"=>generator));
                          on_commit) do
        try
            source = generator=="fit" ? fit :
                pilotfun(model,ds,ds.observed,fit.rho+log(opts.pilot_fraction),fit.beta,
                    collect(1:length(ds.observed)),opts)
            Dict{String,Any}("status"=>"ok","message"=>"","fit"=>ceb_fit_state(source))
        catch e
            Dict{String,Any}("status"=>"failed","message"=>ce_error(e))
        end
    end
    state["status"]=="ok" || throw(DomainError(state["message"],"bootstrap generating fit unavailable"))
    source = ceb_fit_value(state["fit"])
    source_sha = rd_digest(generator_path)
    attempt_dir = joinpath(directory,"attempts")
    expected = Set(@sprintf("%06d.toml",b) for b in 1:opts.nboot)
    if isdir(attempt_dir)
        present = Set(filter(n->endswith(n,".toml"),readdir(attempt_dir)))
        issubset(present,expected) || error("unexpected bootstrap attempt files")
    end
    draws,scales = fill(NaN,length(fit.estimates),opts.nboot),fill(NaN,length(fit.estimates),opts.nboot)
    records = Dict{String,Any}[]
    for b in 1:opts.nboot
        seed = ce_seed("bootstrap",ds.stage,ds.operator,ds.design_id,ds.noise_id,
            ds.sigma,ds.case,ds.seed,generator,b)
        payload = ceb_component(joinpath(attempt_dir,@sprintf("%06d.toml",b)),
            merge(context,Dict("component"=>"attempt","generator"=>generator,
                "generator_sha256"=>source_sha,"attempt"=>b,"rng_seed"=>string(seed)));on_commit) do
            noise = sqrt(source.sigma2).*(ds.noise.root*randn(StableRNG(seed),length(ds.observed)))
            observed = source.pred+reshape(noise,size(source.pred))
            result = fitfun(model,ds,observed,opts)
            row = Dict{String,Any}("attempt"=>b,"rng_seed"=>string(seed),"status"=>result.status,
                "message"=>result.message,"candidate_count"=>length(result.attempts),
                "candidate_failures"=>[ce_dictionary(r) for r in result.attempts if r.status!="ok"])
            result_data = Dict{String,Any}("record"=>row)
            if result.fit!==nothing
                f = result.fit
                row["rho"],row["sigma2"],row["parameters"] = f.rho,f.sigma2,f.beta
                row["converged"],row["grid_boundary"] = f.converged,f.grid_boundary
                result_data["values"],result_data["se"] = f.estimates,f.se
            end
            result_data
        end
        row = payload["record"]
        row["attempt"]==b && row["rng_seed"]==string(seed) || error("bootstrap attempt identity differs")
        if row["status"]=="ok"
            length(payload["values"])==length(payload["se"])==length(fit.estimates) ||
                error("bootstrap checkpoint dimensions differ")
            draws[:,b],scales[:,b] = payload["values"],payload["se"]
        else
            haskey(payload,"values") && error("failed bootstrap attempt contains successful values")
        end
        push!(records,row)
    end
    (;draws,scales,records,generating=source.estimates,
      generating_rho=source.rho,generating_sigma2=source.sigma2)
end

function ceb_split(model,ds,opts,path,context;parent_record=nothing,on_commit=(p,v)->nothing)
    ceb_component(path,merge(context,Dict("component"=>"split"));on_commit) do
        if parent_record!==nothing && haskey(parent_record,"split")
            return Dict{String,Any}("status"=>"ok","message"=>"",
                "split"=>parent_record["split"],"pilot"=>parent_record["split_pilot"])
        end
        try
            s = ce_split_inference(model,ds,opts)
            pilot = Dict("status"=>s.pilot.status,"candidates"=>ce_dictionary.(s.pilot.attempts))
            data = Dict{String,Any}("status"=>s.value===nothing ? "failed" : "ok",
                "message"=>s.value===nothing ? s.pilot.message : "","pilot"=>pilot)
            if s.value!==nothing
                v = s.value
                data["split"] = Dict{String,Any}("estimates"=>v.estimates,"se"=>v.se,
                    "bank_bias"=>v.bank_bias,"train"=>v.train,"infer"=>v.infer,"pilot_rho"=>v.pilot_rho,
                    "M"=>ce_matrix(v.M),"offset"=>v.offset,
                    "innovation_regression"=>ce_matrix(v.innovation_regression))
                v.ellipsoid_bias===nothing || (data["split"]["ellipsoid_bias"]=v.ellipsoid_bias)
            end
            data
        catch e
            Dict{String,Any}("status"=>"failed","message"=>ce_error(e))
        end
    end
end

function ceb_evaluate(model,ds,cfg,mode,methods,directory,context;parent=nothing,on_commit=(p,v)->nothing)
    original = ceb_original(model,ds,cfg,mode,joinpath(directory,"original.toml"),context;parent,on_commit)
    record = deepcopy(original["record"])
    record["bootstrap"] = Dict{String,Any}()
    f = haskey(original,"fit") ? ceb_fit_value(original["fit"]) : nothing
    opts = (;iterations=cfg.iterations,tol=cfg.tol,rhos=cfg.rhos,nboot=cfg.nboot,
        noise_mode=mode,pilot_fraction=cfg.pilot_fraction)
    intervals = NamedTuple[]
    boot = Dict{String,Any}()
    for method in methods
        bands,estimates = nothing,f===nothing ? nothing : f.estimates
        status,message = record["status"],record["message"]
        guarantee,assumption = "none",false
        if f!==nothing || startswith(method,"split_bias")
            try
                if method=="conditional"
                    rng = StableRNG(ce_seed("gaussian",ds.stage,ds.operator,ds.design_id,
                        ds.noise_id,ds.sigma,ds.case,ds.seed))
                    bands = ce_conditional(f,length(ds.points),cfg.level,cfg.nsim,rng)
                elseif startswith(method,"bootstrap")
                    generator = method=="bootstrap_t_pilot" ? "pilot" : "fit"
                    if !haskey(boot,generator)
                        b = ceb_bootstrap(model,ds,f,opts,generator,joinpath(directory,"bootstrap-"*generator),
                            context;on_commit)
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
                        bands = ce_studentized(f.estimates,f.se,b.generating,b.draws,b.scales,length(ds.points),cfg.level)
                    end
                else
                    s = ceb_split(model,ds,opts,joinpath(directory,"split.toml"),context;
                        parent_record=record,on_commit)
                    haskey(s,"pilot") && (record["split_pilot"]=s["pilot"])
                    s["status"]=="ok" || throw(DomainError(s["message"],"split pilot unavailable"))
                    record["split"] = s["split"]
                    v = s["split"]
                    estimates = v["estimates"]
                    bound = v[method=="split_bias_bank" ? "bank_bias" : "ellipsoid_bias"]
                    bands = ce_bias_intervals(estimates,v["se"],bound,length(ds.points),cfg.level)
                    guarantee = method=="split_bias_bank" ? "known_gaussian_finite_class" : "known_gaussian_polynomial_ball"
                    assumption = method=="split_bias_bank" ? ds.truth_in_bank : ds.truth_in_ellipsoid
                end
                status,message = "ok",""
            catch e
                status,message = "unavailable",ce_error(e)
            end
        end
        append!(intervals,ce_interval_rows(ds,model.id,model.family,mode,method,estimates,bands;
            status,message,guarantee,assumption))
    end
    (;record,intervals)
end

function ceb_configuration(run_dir,plan)
    options = plan["options"]
    args = ["--"*k*"="*v for (k,v) in options]
    stage = plan["stage"]
    append!(args,["--stage="*stage,"--seed-start="*string(first(plan["seeds"])),
        "--n-datasets="*string(length(plan["seeds"])),"--nboot="*string(plan["nboot"]),
        "--nsim="*string(plan["nsim"]),"--output="*joinpath(run_dir,"results")])
    stage=="confirm" && push!(args,"--lock="*joinpath(run_dir,"confirmation-lock.toml"))
    cfg = ce_options(args)
    if stage!="confirm"
        cfg = merge(cfg,(lock=Dict("participants"=>plan["participants"]),))
    else
        isequal(plan["participants"],cfg.lock["participants"]) ||
            error("confirmation participants differ from the selection lock")
    end
    cfg.canonical==options || error("execution plan options do not match the scientific runner")
    cfg.seeds==plan["seeds"] || error("execution plan dataset indices are not the declared contiguous population")
    ce_code_hash()==plan["code_sha256"] || error("execution plan scientific code differs")
    participants = plan["participants"]
    !isempty(participants) || error("execution plan has no estimator participants")
    identities = [(p["operator"],p["model"],p["noise_mode"],p["method"]) for p in participants]
    allunique(identities) || error("execution plan duplicates an estimator participant")
    all(p->p["operator"] in cfg.operators && p["model"] in cfg.models &&
        p["noise_mode"] in cfg.noise_modes && p["method"] in cfg.methods &&
        ce_applicable(p["method"],p["operator"],p["noise_mode"]),participants) ||
        error("execution plan has an unknown or inapplicable participant")
    cfg
end

function ceb_initialize(run_dir,plan,cfg,jobs)
    output = cfg.output
    path = joinpath(output,"design.toml")
    plan_sha = rd_digest(joinpath(run_dir,"plan.toml"))
    if isfile(path)
        design = TOML.parsefile(path)
        design["execution_plan_sha256"]==plan_sha || error("saved plan differs from continuation")
        return design
    end
    if !isdir(output)
        rd_followon_snapshot(output,ce_scripts())
    else
        # An interrupted initialization may have copied only a prefix of
        # the immutable scripts. Verify existing files before finishing it.
        for file in ce_scripts()
            destination = joinpath(output,basename(file))
            cer_preserve(destination,read(file,String))
        end
        cer_preserve(joinpath(output,"environment-manifest.toml"),
            read(joinpath(dirname(Base.active_project()),"Manifest.toml"),String))
        cer_preserve(joinpath(output,"environment-project.toml"),read(Base.active_project(),String))
    end
    mkpath(joinpath(output,"fits")); mkpath(joinpath(output,"datasets"))
    jobs_count = sum(any(ce_applicable(m,j.operator,nm) && ce_selected(cfg,model,m,j.operator,nm)
        for m in cfg.methods) for j in jobs for model in cfg.models for nm in cfg.noise_modes)
    design = Dict{String,Any}("schema"=>"calibration-suite-v1","stage"=>cfg.stage,"code_sha256"=>ce_code_hash(),
        "options"=>cfg.canonical,"models"=>cfg.models,"methods"=>cfg.methods,"operators"=>cfg.operators,
        "designs"=>cfg.designs,"noises"=>cfg.noises,"noise_modes"=>cfg.noise_modes,"sigmas"=>cfg.sigmas,
        "seeds"=>cfg.seeds,"nboot"=>cfg.nboot,"nsim"=>cfg.nsim,"ngrid"=>cfg.ngrid,"level"=>cfg.level,
        "lock_sha256"=>cfg.lock_sha256,"execution_plan_sha256"=>plan_sha,"phase"=>plan["phase"],
        "expected_datasets"=>length(jobs),"expected_fit_jobs"=>jobs_count,
        "dataset_jobs"=>ce_dictionary.(jobs),"shard"=>plan["shard"],
        "confirmation_total"=>cfg.stage=="confirm" ? cfg.lock["confirmation_datasets"] : 0,
        "cases"=>Dict(op=>ce_cases(cfg.stage,op) for op in cfg.operators),"participants"=>plan["participants"],
        "confirmation_extra_designs"=>CE_CONFIRM_EXTRA_DESIGNS,"confirmation_extra_noises"=>CE_CONFIRM_EXTRA_NOISES,
        "model_specs"=>Dict(id=>ce_model(id).metadata for id in cfg.models),
        "lineage"=>get(plan,"lineage",Dict{String,Any}()),
        "checkpoint_policy"=>"original fit, generating fit and every attempted bootstrap refit are atomic components")
    cer_atomic(path,cer_text(design))
    design
end

function ceb_execute(run_dir;plan_only=false,job_limit=typemax(Int))
    plan_path = joinpath(run_dir,"plan.toml")
    plan = TOML.parsefile(plan_path)
    plan["schema"]=="calibration-execution-plan-v1" || error("unknown execution plan")
    cfg = ceb_configuration(run_dir,plan)
    all_jobs = [(;operator,case,design,noise,sigma,seed) for operator in cfg.operators
        for case in ce_cases(cfg.stage,operator) for design in cfg.designs for noise in cfg.noises
        for sigma in cfg.sigmas for seed in cfg.seeds]
    shard = plan["shard"]
    length(shard)==2 && 1<=shard[1]<=shard[2] || error("invalid plan shard")
    jobs = [j for (i,j) in enumerate(all_jobs) if mod(i-1,shard[2])==shard[1]-1]
    isempty(jobs) && error("execution plan has an empty shard")
    fits,replicates = 0,0
    for j in jobs, model in cfg.models, mode in cfg.noise_modes
        methods = [m for m in cfg.methods if ce_applicable(m,j.operator,mode) && ce_selected(cfg,model,m,j.operator,mode)]
        isempty(methods) && continue
        fits += 1
        generators = Set(m=="bootstrap_t_pilot" ? "pilot" : "fit" for m in methods if startswith(m,"bootstrap"))
        replicates += cfg.nboot*length(generators)
    end
    println("Checkpointed $(plan["phase"]): $(length(jobs)) datasets, $fits fit jobs, $replicates bootstrap attempts.")
    flush(stdout)
    if plan_only
        cer_atomic(joinpath(run_dir,"preview.toml"),cer_text(Dict(
            "code_sha256"=>ce_code_hash(),"plan_sha256"=>rd_digest(plan_path),
            "total_datasets"=>length(all_jobs),"shard_datasets"=>length(jobs),
            "fit_jobs"=>fits,"bootstrap_attempts"=>replicates)))
        return nothing
    end
    design = ceb_initialize(run_dir,plan,cfg,jobs)
    parent = get(plan,"parent_results","")
    parent_meta = nothing
    if !isempty(parent)
        parent_meta = TOML.parsefile(joinpath(parent,"metadata.toml"))
        rd_digest(joinpath(parent,"metadata.toml"))==plan["parent_metadata_sha256"] ||
            error("parent results manifest differs")
        parent_meta["complete"] && parent_meta["code_sha256"]==ce_code_hash() ||
            error("parent results use a different or incomplete scientific protocol")
        parent_design = TOML.parsefile(joinpath(parent,"design.toml"))
        parent_design["stage"]==cfg.stage || error("parent and refinement data stages differ")
        for (key,value) in cfg.canonical
            key in ("models","methods") && continue
            value==parent_design["options"][key] ||
                error("parent estimator differs in $key; it cannot seed a matched bootstrap")
        end
        for job in jobs
            name = "datasets/"*cer_job_id(job)*".toml"
            text = read(joinpath(parent,name),String)
            cer_digest(text)==parent_meta["outputs_sha256"][name] || error("parent dataset changed")
            cer_preserve(joinpath(cfg.output,name),text)
        end
    end
    plan_sha = rd_digest(plan_path)
    adapter_hash = rd_digest(@__FILE__)
    evaluate = function(model,ds,cfg,mode,methods)
        job = (;operator=ds.operator,case=ds.case,design=ds.design_id,noise=ds.noise_id,sigma=ds.sigma,seed=ds.seed)
        id = cer_job_id(job)*"__"*model.id*"__"*mode
        dataset_sha = rd_digest(joinpath(cfg.output,"datasets",cer_job_id(job)*".toml"))
        context = ceb_context(model,ds,cfg,mode,plan_sha,dataset_sha)
        parent_path = isempty(parent) ? nothing : joinpath(parent,"fits",id*".toml")
        if parent_path!==nothing
            rd_digest(parent_path)==parent_meta["outputs_sha256"]["fits/"*id*".toml"] ||
                error("parent fit fingerprint differs")
        end
        on_commit = function(path,payload)
            row = get(payload,"record",payload)
            cer_atomic(joinpath(run_dir,"component-progress.toml"),cer_text(Dict(
                "updated_unix"=>time(),"job"=>id,"component"=>relpath(path,cfg.output),
                "status"=>get(row,"status","checkpointed"),"attempt"=>get(row,"attempt",0),
                "planned_bootstrap_attempts"=>cfg.nboot,"plan_sha256"=>plan_sha)))
        end
        result = ceb_evaluate(model,ds,cfg,mode,methods,joinpath(cfg.output,"components",id),context;
            parent=parent_path,on_commit)
        rd_digest(plan_path)==plan_sha && rd_digest(@__FILE__)==adapter_hash ||
            error("plan or checkpoint adapter changed during the fit")
        result
    end
    cer_execute(run_dir;config=cfg,evaluate,limit=job_limit)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_dir = abspath(ARGS[1])
    flags = ARGS[2:end]
    all(x->x=="--plan-only" || startswith(x,"--job-limit="),flags) || error("unknown checkpointed-run option")
    values = filter(x->startswith(x,"--job-limit="),flags)
    limit = isempty(values) ? typemax(Int) : parse(Int,split(only(values),'=')[2])
    limit>0 || error("job limit must be positive")
    realpath(pathof(PSM))==realpath(joinpath(run_dir,"source","src","PartiallySpecifiedModels.jl")) ||
        error("checkpointed execution loaded the live package")
    ceb_execute(run_dir;plan_only="--plan-only" in flags,job_limit=limit)
end
