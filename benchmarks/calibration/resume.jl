if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Usage: resume.jl RUN_DIRECTORY [--recover-only] [--limit=N]")
    include(joinpath(abspath(ARGS[1]),"source","benchmarks","calibration","run.jl"))
end

const CER_INTERVAL_FIELDS = (:stage,:operator,:case,:actual_truth,:design,:noise,:sigma,:seed,
    :model,:family,:noise_mode,:method,:guarantee,:assumption_satisfied,:reference_only,
    :status,:message,:interval,:scope,:region,:target_id,:x,:truth,:estimate,:lower,:upper,
    :available,:covered,:width,:critical)

struct CERCoordinates{A} <: AbstractApproximator
    name::Symbol
    base::A
    transform::Matrix{Float64}
    beta::Vector{Float64}
    theta::Vector{Float64}
end
PSM.nparams(a::CERCoordinates) = length(a.theta)
PSM.build_evaluator(a::CERCoordinates,t) = build_evaluator(a.base,a.beta+a.transform*(t-a.theta))

cer_text(x) = sprint(io->TOML.print(io,x))
cer_digest(text::AbstractString) = bytes2hex(SHA.sha256(codeunits(text)))
cer_job_id(job) = join((job.operator,job.design,job.noise,job.sigma,job.case,job.seed),"__")
cer_row(x) = NamedTuple{CER_INTERVAL_FIELDS}(Tuple(x[string(k)] for k in CER_INTERVAL_FIELDS))

function cer_pause_check()
    path = get(ENV,"CALIBRATION_PAUSE_FILE","")
    isempty(path) || !isfile(path) || error("Calibration campaign paused: "*read(path,String))
end

function cer_atomic(path,text)
    cer_pause_check()
    mkpath(dirname(path))
    temporary,io = mktemp(dirname(path);cleanup=false)
    try
        write(io,text)
        flush(io)
        if Sys.isunix()
            ccall(:fsync,Cint,(Cint,),fd(io)) == 0 || error("checkpoint fsync failed: $path")
        end
        close(io)
        Base.Filesystem.rename(temporary,path)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    path
end

function cer_preserve(path,text)
    if isfile(path)
        read(path,String)==text || error("refusing to overwrite a different saved record: $path")
    else
        cer_atomic(path,text)
    end
end

function cer_compare(actual,expected,label;rtol=2e-4,atol=1e-8)
    size(actual)==size(expected) || error("$label replay dimensions differ")
    all(isfinite,actual) && all(isfinite,expected) || error("$label replay is non-finite")
    error_abs = maximum(abs,actual-expected)
    allowance = atol .+ rtol.*abs.(expected)
    all(abs.(actual-expected).<=allowance) ||
        error("$label replay differs: maximum absolute difference $error_abs")
    error_abs
end

function cer_working_fit(record,model,ds)
    beta = Float64.(record["parameters"])
    coordinates = sp_coordinates(model.approx,model.frame,record["rho"],beta)
    theta = coordinates.initial
    # Anchor the forward solve at the stored beta exactly, rather than
    # perturbing it through a transform/inverse-transform round trip.
    anchored = CERCoordinates(model.approx.name,model.approx,coordinates.transform,beta,theta)
    prob = sp_problem(ce_problem(model,ds,ds.observed),anchored)
    pred = simulate(prob,theta)
    J = ce_jacobian(prob,theta,pred)
    L = cholesky(Symmetric(ds.noise.shape)).L
    residual = L \ vec(pred-ds.observed)
    score = sp_working_profile(L\J,ones(length(residual)),theta,residual,model.frame.nullity)
    G = ce_output_jacobian(model,beta,ds.points)*coordinates.transform
    root = sqrt(record["sigma2"]).*(G/score.R)
    recomputed_se = [norm(view(root,i,:)) for i in axes(root,1)]
    estimates = ce_output_map(ds.points)*build_evaluator(model.approx,beta).(ds.points)
    # Replaying 669 saved fits measured max SE discrepancy 1.04e-6
    # (relative 1.44e-5). 2e-4 permits >13x integration/AD headroom.
    # Check every remaining record too; never silently accept a large drift.
    discrepancies = Dict("estimate_max_abs"=>cer_compare(estimates,record["estimates"],"estimate";rtol=1e-10,atol=1e-12),
        "se_max_abs"=>cer_compare(recomputed_se,record["se"],"standard error"),
        "edf_abs"=>cer_compare([score.edf],[record["edf"]],"EDF"),
        "rtol"=>2e-4,"atol"=>1e-8,
        "adapter_sha256"=>rd_digest(@__FILE__),
        "coefficient_policy"=>"exact saved beta anchor; no optimization")
    fit = (;root,estimates=Float64.(record["estimates"]),se=Float64.(record["se"]),
        beta,rho=Float64(record["rho"]),sigma2=Float64(record["sigma2"]),pred,
        converged=record["converged"],grid_boundary=record["grid_boundary"],score)
    fit,discrepancies
end

function cer_conditional(record,model,ds,cfg)
    fit,discrepancies = cer_working_fit(record,model,ds)
    rng = StableRNG(ce_seed("gaussian",ds.stage,ds.operator,ds.design_id,ds.noise_id,
        ds.sigma,ds.case,ds.seed))
    ce_conditional(fit,length(ds.points),cfg.level,cfg.nsim,rng),discrepancies
end

function cer_replay(record,model,ds,cfg,mode,methods)
    all(in(("conditional","split_bias_bank","split_bias_ellipsoid")),methods) ||
        throw(ArgumentError("recovery currently supports the non-bootstrap screen only"))
    (record["model"],record["family"],record["operator"],record["case"],record["seed"],
        record["noise_mode"]) == (model.id,model.family,ds.operator,ds.case,ds.seed,mode) ||
        error("saved fit identity differs from the requested job")
    rows = NamedTuple[]
    replay = Dict{String,Any}()
    for method in methods
        bands,estimates = nothing,get(record,"estimates",nothing)
        status,message = record["status"],record["message"]
        guarantee,assumption = "none",false
        if method=="conditional"
            if record["status"]=="ok"
                bands,replay = cer_conditional(record,model,ds,cfg)
            end
        elseif haskey(record,"split")
            s = record["split"]
            estimates = s["estimates"]
            field = method=="split_bias_bank" ? "bank_bias" : "ellipsoid_bias"
            haskey(s,field) || error("saved split result lacks $field")
            bands = ce_bias_intervals(estimates,s["se"],s[field],length(ds.points),cfg.level)
            status,message = "ok",""
            guarantee = method=="split_bias_bank" ? "known_gaussian_finite_class" : "known_gaussian_polynomial_ball"
            assumption = method=="split_bias_bank" ? ds.truth_in_bank : ds.truth_in_ellipsoid
        else
            error("saved split inference lacks endpoints/status; refusing to invent its failure outcome")
        end
        append!(rows,ce_interval_rows(ds,model.id,model.family,mode,method,estimates,bands;
            status,message,guarantee,assumption))
    end
    (;rows,replay)
end

function cer_dataset_record(ds,id)
    Dict("id"=>id,"rng_seed"=>string(ds.rng_seed),"actual_truth"=>ds.actual,
        "observed"=>collect.(eachcol(ds.observed)),"reference"=>collect.(eachcol(ds.reference)),
        "points"=>ds.points,"targets"=>ds.truth,"times"=>ds.design.times,"initial"=>ds.design.u0,
        "speeds"=>ds.design.speeds,"noise_shape"=>ce_matrix(ds.noise.shape),
        "truth_in_bank"=>ds.truth_in_bank,"truth_in_ellipsoid"=>ds.truth_in_ellipsoid,
        "bank_ids"=>ds.cache.ids)
end

function cer_dataset(path,ds,id)
    fresh = cer_dataset_record(ds,id)
    if isfile(path)
        saved = TOML.parsefile(path)
        keys(saved)==keys(fresh) || error("saved dataset fields differ: $id")
        for (k,v) in fresh
            isequal(saved[k],v) || error("saved dataset differs in $k: $id")
        end
    else
        cer_preserve(path,cer_text(fresh))
    end
    rd_digest(path)
end

function cer_fit_row(record,job,model,mode)
    (;operator=job.operator,case=job.case,design=job.design,noise=job.noise,sigma=job.sigma,
      seed=job.seed,model=model.id,family=model.family,noise_mode=mode,
      status=record["status"],message=record["message"],rho=get(record,"rho",NaN),
      sigma2=get(record,"sigma2",NaN),edf=get(record,"edf",NaN),
      decrement=get(record,"decrement",NaN),converged=get(record,"converged",false),
      grid_boundary=get(record,"grid_boundary",false))
end

function cer_bundle(path,fit_path,dataset_sha,code_hash,methods;result=nothing,origin="",replay=Dict())
    if isfile(path)
        bundle = TOML.parsefile(path)
        bundle["schema"]=="calibration-checkpoint-v1" &&
            bundle["code_sha256"]==code_hash && bundle["dataset_sha256"]==dataset_sha &&
            bundle["methods"]==methods || error("checkpoint does not match the experiment: $path")
        text = bundle["fit_toml"]
        cer_digest(text)==bundle["fit_sha256"] || error("checkpoint fit fingerprint differs: $path")
        cer_preserve(fit_path,text)
        rows = cer_row.(bundle["intervals"])
        length(rows)==bundle["interval_count"] || error("checkpoint interval count differs: $path")
        return (;record=TOML.parse(text),rows,origin=bundle["origin"])
    end
    result===nothing && return nothing
    text = isfile(fit_path) ? read(fit_path,String) : cer_text(result.record)
    isequal(TOML.parse(text),result.record) || error("saved fit record changed during checkpointing")
    bundle = Dict("schema"=>"calibration-checkpoint-v1","code_sha256"=>code_hash,
        "dataset_sha256"=>dataset_sha,"fit_sha256"=>cer_digest(text),"fit_toml"=>text,
        "methods"=>methods,"origin"=>origin,"replay"=>replay,
        "interval_count"=>length(result.intervals),"intervals"=>ce_dictionary.(result.intervals))
    # Commit the fit and endpoints together first. If interrupted before
    # the ordinary fit file is published, it is restored from this bundle.
    cer_atomic(path,cer_text(bundle))
    cer_preserve(fit_path,text)
    (;record=result.record,rows=result.intervals,origin)
end

function cer_config(design,output)
    options = design["options"]
    args = ["--"*k*"="*v for (k,v) in options]
    append!(args,["--stage="*design["stage"],"--n-datasets="*string(length(design["seeds"])),
        "--seed-start="*string(first(design["seeds"])),"--nboot="*string(design["nboot"]),
        "--nsim="*string(design["nsim"]),"--output="*output,
        "--shard="*join(design["shard"],"/")])
    cfg = ce_options(args)
    cfg.canonical==options || error("recovered fitting options differ from the frozen design")
    all(in(("conditional","split_bias_bank","split_bias_ellipsoid")),cfg.methods) ||
        throw(ArgumentError("only non-bootstrap screens can be resumed by this adapter"))
    cfg
end

function cer_origin(output,design)
    path = joinpath(output,"recovery","origin.toml")
    if !isfile(path)
        inputs = Dict{String,String}("design.toml"=>rd_digest(joinpath(output,"design.toml")))
        for subdir in ("fits","datasets"), file in readdir(joinpath(output,subdir))
            endswith(file,".toml") || error("unexpected original result file: $file")
            relative = joinpath(subdir,file)
            TOML.parsefile(joinpath(output,relative))
            inputs[relative] = rd_digest(joinpath(output,relative))
        end
        cer_atomic(path,cer_text(Dict("schema"=>"calibration-recovery-v1",
            "started_unix"=>time(),"code_sha256"=>design["code_sha256"],
            "original_files_sha256"=>inputs,
            "recovery_script_sha256"=>rd_digest(@__FILE__),
            "replay_policy"=>"stored coefficients unchanged; reconstruct final working root; replay original RNG seed")))
    end
    origin = TOML.parsefile(path)
    origin["code_sha256"]==design["code_sha256"] || error("recovery origin protocol differs")
    for (relative,sha) in origin["original_files_sha256"]
        rd_digest(joinpath(output,relative))==sha || error("original saved result changed: $relative")
    end
    origin
end

function cer_reference(ds,cfg,mode)
    rows = NamedTuple[]
    mode=="known" || return rows
    for method in (ds.operator=="integral" ? ("bank_set","linear_set") : ("bank_set",))
        result = method=="bank_set" ? ce_bank_set(ds,cfg.level) : ce_linear_set(ds,cfg.level)
        status = result.intervals===nothing ? "empty_set" : "ok"
        append!(rows,ce_interval_rows(ds,"reference","reference",mode,method,result.estimates,
            result.intervals;status,message=status=="ok" ? "" : "reference class rejected",
            guarantee=method=="bank_set" ? "known_gaussian_finite_class" : "known_gaussian_polynomial_span",
            assumption=method=="bank_set" ? ds.truth_in_bank : ds.truth_in_ellipsoid,reference=true))
    end
    rows
end

function cer_append_csv(path,rows)
    isempty(rows) && return
    exists = isfile(path)
    CSV.write(path,rows;append=exists,header=!exists)
end

function cer_execute(run_dir;recover_only=false,limit=typemax(Int),
                     config=nothing,evaluate=ce_evaluate_model,replay=cer_replay)
    output = joinpath(run_dir,"results")
    if isfile(joinpath(output,"metadata.toml"))
        metadata = TOML.parsefile(joinpath(output,"metadata.toml"))
        metadata["schema"]=="calibration-results-v1" && metadata["complete"] ||
            error("existing completion manifest is invalid")
        all(rd_digest(joinpath(output,name))==sha for (name,sha) in metadata["outputs_sha256"]) ||
            error("a completed result differs from its manifest")
        return output
    end
    design = TOML.parsefile(joinpath(output,"design.toml"))
    ce_code_hash()==design["code_sha256"] || error("loaded code differs from the interrupted protocol")
    driver_hash = rd_digest(@__FILE__)
    cfg = config===nothing ? cer_config(design,output) : config
    cfg.canonical==design["options"] && cfg.stage==design["stage"] &&
        cfg.nboot==design["nboot"] && cfg.nsim==design["nsim"] &&
        cfg.seeds==design["seeds"] || error("continuation configuration differs from the saved design")
    BLAS.set_num_threads(1)
    origin = cer_origin(output,design)
    originals = count(startswith(k,"fits/") for k in keys(origin["original_files_sha256"]))
    jobs = [(;operator=j["operator"],case=j["case"],design=j["design"],noise=j["noise"],
        sigma=Float64(j["sigma"]),seed=Int(j["seed"])) for j in design["dataset_jobs"]]
    println("Resuming $(length(jobs)) datasets / $(design["expected_fit_jobs"]) fits; $originals original fit records.")
    println("Protocol hash: ",ce_code_hash())
    flush(stdout)
    caches = Dict{Tuple{String,String},Any}()
    points = collect(range(0.3,1.8;length=cfg.ngrid))
    completed,replayed,computed,skipped,new_checkpoints = 0,0,0,0,0
    entries,reference_rows,weak_rows = NamedTuple[],NamedTuple[],NamedTuple[]
    pairs = Set{Tuple{String,String,String,Float64}}()
    expected_names = Set{String}()
    for job in jobs
        key = (job.operator,job.design)
        haskey(caches,key) || (caches[key]=ce_bank_cache(ce_design(key...),points))
        ds = ce_dataset(cfg.stage,job.operator,job.design,job.noise,job.sigma,job.case,job.seed,points,caches[key])
        id = cer_job_id(job)
        dataset_path = joinpath(output,"datasets",id*".toml")
        if recover_only && !isfile(dataset_path)
            continue
        end
        dataset_sha = cer_dataset(dataset_path,ds,id)
        pair_key = (job.operator,job.design,job.noise,job.sigma)
        if !(pair_key in pairs)
            pair = ce_weak_pair(ds.cache,ds.noise,ds.sigma)
            append!(weak_rows,[(;operator=job.operator,design=job.design,noise=job.noise,
                sigma=job.sigma,selected=r.first==pair.chosen.first && r.second==pair.chosen.second,r...)
                for r in pair.rows])
            push!(pairs,pair_key)
        end
        for mode in cfg.noise_modes
            append!(reference_rows,cer_reference(ds,cfg,mode))
            for model_id in cfg.models
                methods = [m for m in cfg.methods if ce_applicable(m,job.operator,mode) &&
                    ce_selected(cfg,model_id,m,job.operator,mode)]
                isempty(methods) && continue
                name = id*"__"*model_id*"__"*mode*".toml"
                push!(expected_names,name)
                fit_path = joinpath(output,"fits",name)
                checkpoint = joinpath(output,"recovery","fits",name)
                result = cer_bundle(checkpoint,fit_path,dataset_sha,design["code_sha256"],methods)
                if result===nothing
                    if new_checkpoints>=limit || (recover_only && !isfile(fit_path))
                        skipped += 1
                        continue
                    end
                    model = ce_model(model_id)
                    if isfile(fit_path)
                        record = TOML.parsefile(fit_path)
                        restored = replay(record,model,ds,cfg,mode,methods)
                        evaluated = (;record,intervals=restored.rows)
                        result = cer_bundle(checkpoint,fit_path,dataset_sha,design["code_sha256"],methods;
                            result=evaluated,origin="replayed_saved_fit",replay=restored.replay)
                        replayed += 1
                    else
                        evaluated = evaluate(model,ds,cfg,mode,methods)
                        result = cer_bundle(checkpoint,fit_path,dataset_sha,design["code_sha256"],methods;
                            result=evaluated,origin="new_fit_exact_endpoints")
                        computed += 1
                    end
                    new_checkpoints += 1
                end
                completed += 1
                model = (;id=model_id,family=result.record["family"])
                push!(entries,(;checkpoint,fit_row=cer_fit_row(result.record,job,model,mode)))
                if completed%30==0 || result.origin=="new_fit_exact_endpoints"
                    println("$completed/$(design["expected_fit_jobs"]) checkpointed; $id $model_id $mode; " *
                        "replayed=$replayed new=$computed")
                    flush(stdout)
                end
                if new_checkpoints>0 && new_checkpoints%30==0
                    cer_atomic(joinpath(run_dir,"progress.toml"),cer_text(Dict(
                        "updated_unix"=>time(),"complete"=>false,"checkpointed"=>completed,
                        "expected"=>design["expected_fit_jobs"],"replayed_this_run"=>replayed,
                        "fitted_this_run"=>computed,"last_job"=>name)))
                end
            end
        end
    end
    if completed!=design["expected_fit_jobs"]
        remaining = design["expected_fit_jobs"]-completed
        println("Recovery pass finished: $completed checkpointed; $remaining remaining ($skipped in inspected datasets).")
        flush(stdout)
        return nothing
    end
    Set(readdir(joinpath(output,"fits")))==expected_names || error("unexpected saved fit files")
    cer_origin(output,design)
    for filename in ("intervals.csv","fits.csv","weak_pairs.csv")
        temporary = joinpath(output,filename*".building")
        isfile(temporary) && rm(temporary)
    end
    interval_path = joinpath(output,"intervals.csv.building")
    cer_append_csv(interval_path,reference_rows)
    interval_count = length(reference_rows)
    for entry in entries
        bundle = TOML.parsefile(entry.checkpoint)
        rows = cer_row.(bundle["intervals"])
        cer_append_csv(interval_path,rows)
        interval_count += length(rows)
    end
    CSV.write(joinpath(output,"fits.csv.building"),[e.fit_row for e in entries])
    CSV.write(joinpath(output,"weak_pairs.csv.building"),weak_rows)
    for filename in ("intervals.csv","fits.csv","weak_pairs.csv")
        Base.Filesystem.rename(joinpath(output,filename*".building"),joinpath(output,filename))
    end
    ce_code_hash()==design["code_sha256"] && rd_digest(@__FILE__)==driver_hash ||
        error("execution/recovery code changed during continuation")
    scripts = ce_scripts()
    hashes = Dict(basename(file)=>rd_digest(file) for file in scripts)
    all(rd_digest(joinpath(output,name))==sha for (name,sha) in hashes) ||
        error("saved execution scripts differ from the loaded frozen sources")
    files = ["design.toml","fits.csv","intervals.csv","weak_pairs.csv","recovery/origin.toml"]
    for subdir in ("fits","datasets","recovery/fits","recovery/adapters","components")
        isdir(joinpath(output,subdir)) || continue
        for (directory,_,names) in walkdir(joinpath(output,subdir))
            append!(files,[relpath(joinpath(directory,name),output) for name in names
                if endswith(name,".toml") || endswith(name,".jl")])
        end
    end
    metadata = rd_followon_metadata(output,scripts,hashes,source_hash(),[])
    merge!(metadata,Dict("schema"=>"calibration-results-v1","complete"=>true,
        "code_sha256"=>design["code_sha256"],"dataset_count"=>length(jobs),
        "fit_count"=>completed,"interval_count"=>interval_count,
        "recovery"=>Dict("original_saved_fits"=>originals,"recovery_script_sha256"=>driver_hash,
            "conditional_replay"=>"pointwise estimates/SEs retained; simultaneous root regenerated within recorded replay tolerances, without optimization",
            "new_results"=>"exact fit and interval output atomically checkpointed"),
        "outputs_sha256"=>Dict(name=>rd_digest(joinpath(output,name)) for name in files)))
    cer_atomic(joinpath(output,"metadata.toml"),cer_text(metadata))
    cer_atomic(joinpath(run_dir,"progress.toml"),cer_text(Dict(
        "updated_unix"=>time(),"complete"=>true,"checkpointed"=>completed,
        "expected"=>design["expected_fit_jobs"],"intervals"=>interval_count)))
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Usage: resume.jl RUN_DIRECTORY [--recover-only] [--limit=N]")
    run_dir = abspath(ARGS[1])
    flags = ARGS[2:end]
    all(x->x=="--recover-only" || startswith(x,"--limit="),flags) || error("unknown recovery option")
    limit_options = filter(x->startswith(x,"--limit="),flags)
    limit = isempty(limit_options) ? typemax(Int) : parse(Int,split(only(limit_options),'=')[2])
    limit>0 || error("checkpoint limit must be positive")
    source_dir = joinpath(run_dir,"source")
    realpath(pathof(PSM))==realpath(joinpath(source_dir,"src","PartiallySpecifiedModels.jl")) ||
        error("recovery loaded the live package instead of the frozen source")
    cer_execute(run_dir;recover_only="--recover-only" in flags,limit)
end
