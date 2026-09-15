isdefined(@__MODULE__, :rd_main) || include("reaction_diffusion.jl")
using CSV

rd_digest(path) = bytes2hex(SHA.sha256(read(path)))
rd_read_csv(path) = collect(CSV.File(path; types=String, stringtype=String,
                                   missingstring=nothing, ntasks=1))
rd_key(row) = (row.case, row.model, parse(Int,row.seed))

function rd_load_archive(path)
    input = abspath(path)
    files = ("metadata.toml","assessment-script.jl","trials.csv","selected.csv",
             "coefficients.toml","candidate_coefficients.toml","test_profiles.csv")
    fingerprints = Dict(name=>rd_digest(joinpath(input,name)) for name in files)
    meta = TOML.parsefile(joinpath(input,"metadata.toml"))
    fingerprints["assessment-script.jl"] == meta["assessment_script_sha256"] ||
        error("archived spatial harness does not match its metadata")
    meta["training_profiles"] == RD_TRAIN_PROFILES &&
        meta["validation_profiles"] == RD_VALIDATION_PROFILES &&
        meta["test_profiles"] == RD_TEST_PROFILES &&
        meta["density_domain"] == collect(RD_DENSITY_DOMAIN) &&
        meta["initial_rate"] == RD_INITIAL_RATE && meta["length"] == 6.0 &&
        meta["diffusion"] == 0.1 && meta["observation_sigma"] == 0.01 &&
        meta["train_end"] == 2.5 && meta["test_end"] == 4.0 &&
        meta["loss_weights"] == "one over cells times training profiles" &&
        meta["check_bounds"] || error("archive does not match the declared spatial fixture")
    meta["track"] in ("adam","laml") || error("unsupported archived fitting track")
    selected_rows = rd_read_csv(joinpath(input,"selected.csv"))
    selected = Dict(rd_key(row)=>row for row in selected_rows)
    trials = rd_read_csv(joinpath(input,"trials.csv"))
    coefficients = TOML.parsefile(joinpath(input,"coefficients.toml"))
    candidates = TOML.parsefile(joinpath(input,"candidate_coefficients.toml"))
    profiles = rd_read_csv(joinpath(input,"test_profiles.csv"))
    expected = Set((case,model,seed) for case in meta["cases"],
                   model in meta["models"], seed in meta["seeds"])
    length(selected_rows) == length(expected) && Set(keys(selected)) == expected ||
        error("archive has duplicate or missing selections")
    Set(keys(coefficients)) == Set(join(key,"__") for key in expected) ||
        error("archive has missing or unexpected selected coefficients")
    for key in expected
        row = selected[key]
        row.selection_status == "ok" && row.track == meta["track"] &&
            parse(Int,row.cells) == meta["cells"] || error("archive selection is not usable: $key")
        eligible = filter(r -> rd_key(r) == key && r.status == "ok" &&
            isfinite(parse(Float64,r.validation_rmse)), trials)
        isempty(eligible) && error("archive selection has no eligible candidate: $key")
        winner = eligible[argmin([parse(Float64,r.validation_rmse) for r in eligible])]
        all(row[name] == winner[name] for name in propertynames(winner)) ||
            error("archive selection is not the validation winner: $key")
        saved = coefficients[join(key,"__")]
        candidate = candidates[row.candidate_id]
        saved["candidate_id"] == row.candidate_id &&
            saved["model_seed"] == candidate["model_seed"] == 20_000+key[3] &&
            saved["parameters"] == candidate["parameters"] &&
            isequal(parse(Float64,row.eta),candidate["eta"]) &&
            all(isfinite,saved["parameters"]) &&
            length(saved["parameters"]) == parse(Int,row.nparams) ||
                error("coefficient provenance mismatch: $key")
        scored = filter(r -> rd_key(r) == key,profiles)
        length(scored) == length(RD_TEST_PROFILES) &&
            Set(parse(Int,r.profile_id) for r in scored) == Set(eachindex(RD_TEST_PROFILES)) &&
            all(r -> r.candidate_id == row.candidate_id,scored) ||
                error("archive has missing or inconsistent held-out profiles: $key")
    end
    (; input, meta, fingerprints, selected, coefficients, profiles)
end

function rd_archive_unchanged(archive)
    all(rd_digest(joinpath(archive.input,file)) == hash for (file,hash) in archive.fingerprints) ||
        error("input archive changed during the run")
    nothing
end

function rd_replay_parity(model, a, beta, ds, archive)
    key = (ds.case, model, ds.seed)
    selected = archive.selected[key]
    rows = NamedTuple[]
    function compare(metric, actual, recorded; profile_id=0)
        old = parse(Float64,recorded)
        # Fixed-coefficient replays should differ only at numerical-solve
        # scale. Both the allowance and observed discrepancies are retained.
        atol, rtol = 1e-9, 1e-6
        isfinite(actual) && isfinite(old) && isapprox(actual,old;atol,rtol) ||
            error("same-mesh replay mismatch for $key, profile $profile_id, $metric: $actual versus $old")
        push!(rows,(; case=ds.case,model=key[2],seed=ds.seed,track=archive.meta["track"],
                      profile_id,metric,archived=old,replayed=actual,
                      absolute_difference=abs(actual-old),atol,rtol))
    end
    for (profiles,times,values,metric) in (
        (RD_TRAIN_PROFILES,ds.train_times,ds.train_values,"training_rmse"),
        (RD_VALIDATION_PROFILES,ds.validation_times,ds.validation_values,"validation_rmse"))
        p = rd_problem(a,profiles,ds.grid,times,values,Ref(0))
        compare(metric,rmse(simulate(p,beta),values),selected[Symbol(metric)])
    end
    for id in eachindex(RD_TEST_PROFILES)
        old = only(filter(r -> rd_key(r) == key && parse(Int,r.profile_id) == id,archive.profiles))
        score = rd_score(a,beta,ds,id)
        score.test_status == score.response_status == "ok" || error("same-mesh replay failed: $key")
        for metric in (:field_rmse,:coefficient_rmse,:reaction_rmse)
            compare(string(metric),getproperty(score,metric),old[metric];profile_id=id)
        end
    end
    rows
end

function rd_followon_snapshot(output, scripts)
    ispath(output) && error("output already exists; choose a new directory")
    mkpath(output)
    hashes = Dict{String,String}()
    for file in scripts
        hashes[basename(file)] = rd_digest(file)
        cp(file,joinpath(output,basename(file)))
    end
    manifest = joinpath(dirname(Base.active_project()),"Manifest.toml")
    cp(manifest,joinpath(output,"environment-manifest.toml"))
    cp(Base.active_project(),joinpath(output,"environment-project.toml"))
    hashes
end

function rd_followon_metadata(output, scripts, hashes, initial_source, archives)
    source_hash() == initial_source &&
        all(rd_digest(file) == hashes[basename(file)] for file in scripts) ||
            error("source changed during the run")
    foreach(rd_archive_unchanged,archives)
    Dict{String,Any}(
        "evaluation_source_sha256"=>initial_source, "script_sha256"=>hashes,
        "input_archives"=>[Dict("path"=>relpath(a.input,ROOT),
            "fitting_source_sha256"=>a.meta["source_sha256"], "files_sha256"=>a.fingerprints)
            for a in archives],
        "julia_version"=>string(VERSION), "platform"=>string(Sys.KERNEL,"-",Sys.ARCH),
        "check_bounds"=>true, "blas_threads"=>BLAS.get_num_threads(),
        "manifest_sha256"=>rd_digest(joinpath(output,"environment-manifest.toml")),
        "environment_project_sha256"=>rd_digest(joinpath(output,"environment-project.toml")))
end

function rd_write_toml(path, data)
    open(path,"w") do io
        TOML.print(io,data)
    end
end
