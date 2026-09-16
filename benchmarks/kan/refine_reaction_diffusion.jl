include("rd_archive.jl")

const RD_OPTIMIZER_VARIANTS = [
    (variant="default",lr=0.03,plateau_tol=1e-4,plateau_window=30,early_stopping=true),
    (variant="tight",lr=0.03,plateau_tol=1e-6,plateau_window=30,early_stopping=true),
    (variant="full",lr=0.03,plateau_tol=1e-4,plateau_window=30,early_stopping=false),
    (variant="slow_tight",lr=0.01,plateau_tol=1e-6,plateau_window=30,early_stopping=true)]

function rd_refine_options(args)
    opts = Dict("input"=>joinpath(@__DIR__,"results","reaction-diffusion-adam"),
        "output"=>joinpath(@__DIR__,"results","reaction-diffusion-optimizer"),
        "iterations"=>"400","cases"=>"crowding","seeds"=>join(301:310,","),
        "models"=>join(RD_MODELS,","))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown optimizer refinement option $key"))
        opts[key] = value
    end
    iterations = parse(Int,opts["iterations"])
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    cases,models = String.(split(opts["cases"],',')),String.(split(opts["models"],','))
    seeds = parse.(Int,split(opts["seeds"],','))
    all(in(RD_CASES),cases) && all(in(RD_MODELS),models) &&
        allunique(cases) && allunique(models) && allunique(seeds) ||
            throw(ArgumentError("unknown or duplicated cases, models or seeds"))
    (; input=abspath(opts["input"]),output=abspath(opts["output"]),iterations,cases,models,seeds)
end

function rd_refine_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    opts = rd_refine_options(args)
    archive = rd_load_archive(opts.input)
    archive.meta["track"] == "adam" || error("optimizer refinement requires the shared-Adam archive")
    all(haskey(archive.selected,(c,m,s)) for c in opts.cases,m in opts.models,s in opts.seeds) ||
        error("requested fitting cases are absent from the archive")
    BLAS.set_num_threads(1)
    cfg = merge(opts,(; track="adam",cells=archive.meta["cells"]))
    scripts = [@__FILE__,joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    initial_source = source_hash()
    hashes = rd_followon_snapshot(opts.output,scripts)
    # Persist the declared conditions before looking at any new test errors.
    rd_write_toml(joinpath(opts.output,"design.toml"),Dict(
        "iterations"=>cfg.iterations,"variants"=>[Dict(string(k)=>v for (k,v) in pairs(x))
                                                for x in RD_OPTIMIZER_VARIANTS],
        "penalty_policy"=>"freeze each original validation-selected eta",
        "initialization"=>"restart every condition from the original model_seed",
        "selection"=>"minimum new validation RMSE; test profiles never select settings"))
    trials, selected, profiles, parity = NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    candidates, coefficients = Dict{String,Any}(),Dict{String,Any}()
    warmed = Set{String}()
    for case in cfg.cases, seed in cfg.seeds
        ds = rd_dataset(case,seed,cfg.cells)
        for model in cfg.models
            old = archive.selected[(case,model,seed)]
            saved = archive.coefficients["$(case)__$(model)__$(seed)"]
            created = @timed rd_approximator(model,saved["model_seed"])
            a = created.value
            append!(parity,rd_replay_parity(model,a,saved["parameters"],ds,archive))
            eta = parse(Float64,old.eta)
            if !(model in warmed)
                warm = merge(cfg,first(RD_OPTIMIZER_VARIANTS),(; iterations=2))
                _,row = rd_train(a,ds,eta,warm)
                row.status == "ok" || error("warm-up failed: $model")
                push!(warmed,model)
            end
            common = (; case,model,seed,track=cfg.track,cells=cfg.cells,nparams=nparams(a),
                setup_seconds=created.time,baseline_candidate_id=old.candidate_id,budget=cfg.iterations)
            fitted = Union{Nothing,PSMSolution}[]
            group = NamedTuple[]
            for variant in RD_OPTIMIZER_VARIANTS
                fit,row = rd_train(a,ds,eta,merge(cfg,variant))
                candidate_id = "$(case)__$(model)__$(seed)__$(variant.variant)"
                result = merge(common,variant,(; candidate_id),row)
                push!(trials,result)
                push!(group,result)
                push!(fitted,fit)
                entry = Dict{String,Any}("status"=>row.status,"model_seed"=>saved["model_seed"],
                    "eta"=>eta,"variant"=>variant.variant)
                if fit !== nothing && all(isfinite,fit.parameters)
                    entry["parameters"] = collect(fit.parameters)
                end
                candidates[candidate_id] = entry
            end
            valid = findall(r -> r.status == "ok" && isfinite(r.validation_rmse),group)
            tuning_seconds = sum(r.fit_seconds for r in group if isfinite(r.fit_seconds))
            prior_tuning_seconds = parse(Float64,old.tuning_seconds)
            cost = (; tuning_seconds,prior_tuning_seconds,
                      total_search_seconds=tuning_seconds+prior_tuning_seconds)
            if isempty(valid)
                push!(selected,merge(last(group),cost,(; selection_status="failed")))
            else
                best = valid[argmin([group[i].validation_rmse for i in valid])]
                winner = group[best]
                push!(selected,merge(winner,cost,(; selection_status="ok")))
                coefficients["$(case)__$(model)__$(seed)"] = Dict(
                    "candidate_id"=>winner.candidate_id,
                    "parameters"=>collect(fitted[best].parameters),"model_seed"=>saved["model_seed"])
            end
            for (row,fit) in zip(group,fitted)
                row.status == "ok" || continue
                for id in eachindex(RD_TEST_PROFILES)
                    push!(profiles,merge(common,(; variant=row.variant,candidate_id=row.candidate_id),
                        rd_score(a,collect(fit.parameters),ds,id)))
                end
            end
            for (file,rows) in (("trials.csv",trials),("selected.csv",selected),
                               ("test_profiles.csv",profiles),("replay_parity.csv",parity))
                isempty(rows) || write_csv(joinpath(opts.output,file),rows)
            end
            rd_write_toml(joinpath(opts.output,"candidate_coefficients.toml"),candidates)
            rd_write_toml(joinpath(opts.output,"coefficients.toml"),coefficients)
            println("optimizer $case seed=$seed model=$model selected=$(last(selected).variant) tuning_s=$tuning_seconds")
            flush(stdout)
        end
    end
    condition_seeds = reduce(vcat,[
        [merge(row,(; variant=v.variant)) for row in rd_seed_metrics(
            filter(r -> r.variant == v.variant,profiles),cfg)] for v in RD_OPTIMIZER_VARIANTS])
    winners = Set(r.candidate_id for r in selected if r.selection_status == "ok")
    selected_seeds = rd_seed_metrics(filter(r -> r.candidate_id in winners,profiles),cfg)
    write_csv(joinpath(opts.output,"condition_seed_metrics.csv"),condition_seeds)
    write_csv(joinpath(opts.output,"selected_seed_metrics.csv"),selected_seeds)
    meta = rd_followon_metadata(opts.output,scripts,hashes,initial_source,[archive])
    merge!(meta,Dict("optimizer_refit"=>true,"iterations"=>cfg.iterations,
        "cases"=>cfg.cases,"models"=>cfg.models,"seeds"=>cfg.seeds,"cells"=>cfg.cells,
        "design_sha256"=>rd_digest(joinpath(opts.output,"design.toml"))))
    rd_write_toml(joinpath(opts.output,"metadata.toml"),meta)
    (; cfg,trials,selected,profiles,parity,condition_seeds,selected_seeds,candidates,coefficients)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rd_refine_main(ARGS)
end
