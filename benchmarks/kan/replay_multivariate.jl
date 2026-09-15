isdefined(@__MODULE__, :mv_main) || include("multivariate.jl")

function mv_replay_options(args)
    opts = Dict("input" => "", "output" => "")
    for arg in args
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown replay option $key"))
        opts[key] = value
    end
    all(!isempty, values(opts)) || throw(ArgumentError("replay requires --input and --output"))
    (; input=abspath(opts["input"]), output=abspath(opts["output"]))
end

function mv_unavailable_score(geometry, message)
    (; trajectory_id=geometry.trajectory_id, initial_N=geometry.initial_N,
       initial_P=geometry.initial_P, test_status="no_valid_candidate", test_message=message,
       full_rmse=Inf, near_rmse=Inf, response_rmse=Inf, near_response_rmse=Inf,
       function_status="no_valid_candidate", function_message=message,
       negative_fraction=NaN, support_fraction=geometry.support_fraction,
       median_support_distance=geometry.median_support_distance, test_rhs_calls=0)
end

function mv_replay_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    opts = mv_replay_options(args)
    BLAS.set_num_threads(1)
    metadata_path = joinpath(opts.input, "metadata.toml")
    metadata_text = read(metadata_path, String)
    meta = TOML.parse(metadata_text)
    meta["track"] in ("laml", "gcv") && get(meta, "starts", "") == "index-multistart" ||
        throw(ArgumentError("constant-start replay requires a native index-multistart archive"))
    length(meta["etas"]) == 1 && isnan(only(meta["etas"])) ||
        throw(ArgumentError("native replay expects one internally selected smoothing fit per start"))
    source_hash() == meta["source_sha256"] || error("package/initialization-wrapper source differs from archive")
    harness = joinpath(@__DIR__, "multivariate.jl")
    bytes2hex(SHA.sha256(read(harness))) == meta["assessment_script_sha256"] ||
        error("assessment harness differs from archive; use the matching source revision")
    bytes2hex(SHA.sha256(read(joinpath(opts.input, "assessment-script.jl")))) == meta["assessment_script_sha256"] ||
        error("archived assessment snapshot does not match its metadata")

    cfg = (; cases=meta["cases"], models=meta["models"], seeds=meta["seeds"],
             track=meta["track"], split=meta["split"])
    geometry = mv_split(cfg.split)
    collect.(geometry.train_ics) == meta["training_ics"] &&
        collect.(geometry.validation_ics) == meta["validation_ics"] &&
        collect.(geometry.test_ics) == meta["test_ics"] && geometry.test_end == meta["test_end"] ||
        error("archive split does not match the replay geometry")
    candidates_path = joinpath(opts.input, "candidate_coefficients.toml")
    candidates_text = read(candidates_path, String)
    candidates = TOML.parse(candidates_text)
    files = ("test_trajectories.csv", "seed_metrics.csv", "metadata.toml", "replay-script.jl")
    any(isfile(joinpath(opts.output, name)) for name in files) &&
        error("replay output already exists; choose a new --output")
    mkpath(opts.output)
    script = read(@__FILE__)
    write(joinpath(opts.output, "replay-script.jl"), script)
    trajectories = NamedTuple[]
    for case in cfg.cases, seed in cfg.seeds
        ds = mv_dataset(case, seed; split=cfg.split)
        support = filter(r -> r.kind == "test", mv_geometry(ds, meta["support_radius"]))
        for model in cfg.models
            candidate_id = "$(case)__$(model)__$(seed)__constant__1"
            saved = candidates[candidate_id]
            saved["initialization"] == "constant" && saved["model_seed"] == 10_000+seed ||
                error("constant candidate provenance is inconsistent for $candidate_id")
            a = mv_make_approximator(model, saved["model_seed"], ds.train_values)
            common = (; case, model, seed, track=cfg.track, split=cfg.split,
                       initialization="constant", candidate_id, nparams=nparams(a),
                       candidate_status=saved["status"])
            for id in eachindex(ds.test_ics)
                score = if saved["status"] == "ok"
                    beta = saved["parameters"]
                    length(beta) == nparams(a) && all(isfinite, beta) ||
                        error("invalid constant candidate coefficients for $candidate_id")
                    mv_score_trajectory(a, beta, ds, case, id, meta["support_radius"])
                else
                    message = "constant candidate was not validation-eligible: " *
                              saved["status"] * ": " * saved["message"]
                    mv_unavailable_score(support[id], message)
                end
                push!(trajectories, merge(common, score))
            end
        end
    end
    source_hash() == meta["source_sha256"] &&
        bytes2hex(SHA.sha256(read(harness))) == meta["assessment_script_sha256"] &&
        read(@__FILE__) == script || error("source changed during replay")
    read(metadata_path, String) == metadata_text && read(candidates_path, String) == candidates_text ||
        error("input archive changed during replay")
    seed_rows = mv_seed_metrics(trajectories, cfg)
    write_csv(joinpath(opts.output, "test_trajectories.csv"), trajectories)
    write_csv(joinpath(opts.output, "seed_metrics.csv"), seed_rows)
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    replay_meta = Dict(
        "optimizer_refit" => false, "candidate_subset" => "constant",
        "source_sha256" => source_hash(),
        "assessment_script_sha256" => meta["assessment_script_sha256"],
        "replay_script_sha256" => bytes2hex(SHA.sha256(script)),
        "archive_metadata_sha256" => bytes2hex(SHA.sha256(metadata_text)),
        "candidate_coefficients_sha256" => bytes2hex(SHA.sha256(candidates_text)),
        "julia_version" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "check_bounds" => true, "blas_threads" => BLAS.get_num_threads(),
        "track" => cfg.track, "split" => cfg.split,
        "package_versions" => Dict(name => get(first(entries), "version", "unversioned")
                                   for (name, entries) in manifest["deps"]))
    open(joinpath(opts.output, "metadata.toml"), "w") do io
        TOML.print(io, replay_meta)
    end
    (; trajectories, seed_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mv_replay_main(ARGS)
end
