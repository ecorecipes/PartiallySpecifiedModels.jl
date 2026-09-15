include("calibrate_uncertainty.jl")

function ui_approximator(nullspace_penalty)
    base = uc_approximator("kan12")
    nullspace_penalty == 1e-6 && return base
    nullspace_penalty == 0.0 || throw(ArgumentError("diagnostic prior must be 0 or 1e-6"))
    KANApproximator(:r,base.model;input_domains=base.input_domains,
        penalty=:edge_curvature,nullspace_penalty=0.0,rng_seed=42)
end

function ui_start(a,kind;rate=0.5)
    beta = initial_params(a)
    if kind == "constant"
        isfinite(rate) || throw(ArgumentError("constant rate must be finite"))
        layer = only(a.layers)
        layer.input_dim == layer.output_dim == 1 || throw(ArgumentError("diagnostic expects a unary single edge"))
        beta[layer.base_range] .= 0.0
        beta[layer.spline_range] .= rate
    elseif kind != "original"
        throw(ArgumentError("unknown initialization $kind"))
    end
    InitializedApprox(a.name,a,beta,penalty_matrix(a))
end

function ui_select(rows)
    length(unique(r.nullspace_penalty for r in rows)) <= 1 ||
        throw(ArgumentError("LAML starts must be compared within the same prior"))
    valid = filter(r->r.status=="ok" && isfinite(r.laml),rows)
    isempty(valid) && return nothing
    valid[argmax(getproperty.(valid,:laml))]
end

function ui_fit(a,ds,queries,kind,iterations)
    fit = nothing
    status,message = "ok",""
    started = time_ns()
    try
        fit = solve(uc_problem(ui_start(a,kind),ds),LAML(maxiters=iterations,jac=:forwarddiff))
        all(isfinite,fit.parameters) && all(isfinite,fit.fitted_values) &&
            isfinite(fit.convergence.laml) ||
                (status="fit_failed";message="non-finite fit or LAML criterion")
    catch e
        status,message = "fit_failed",uc_failure_message(e)
    end
    elapsed = (time_ns()-started)/1e9
    c = fit === nothing ? NamedTuple() : fit.convergence
    row = (; case=ds.case,seed=ds.seed,nullspace_penalty=a.nullspace_penalty,
        penalty_rank=PSM._rank_penalty(penalty_matrix(a)),
        initialization=kind,status,message,fit_seconds=elapsed,
        laml=get(c,:laml,NaN),edf=fit === nothing ? NaN : fit.edf,
        data_loss=fit === nothing ? NaN : fit.data_loss,sigma2=get(c,:sigma2,NaN),
        stationarity=get(c,:stationarity,NaN),smoothing_advanced=get(c,:smoothing_advanced,false),
        reason=string(get(c,:reason,:not_started)))
    estimates,se = fill(NaN,length(queries)),fill(NaN,length(queries))
    interval_status,interval_message = status,message
    if status == "ok"
        try
            # Report using the underlying KAN so sensitivities use the same
            # parameter-AD path as the original calibration, not wrapper FD.
            band = confidence_band(fit,uc_problem(a,ds);uf_points=Dict(:r=>getproperty.(queries,:x)))[:r]
            estimates,se = band.fitted,band.se
        catch e
            interval_status,interval_message = "covariance_failed",uc_failure_message(e)
        end
    end
    (; fit,row,estimates,se,interval_status,interval_message)
end

function ui_replay_parity(run,saved)
    run.row.status == "ok" || error("original initialization replay failed")
    # The original fixed-coefficient replays differed by <=1.22e-17.
    # Allow numerical-solve scale, but not a change in fitted response.
    maximum(abs,run.estimates-saved["estimates"]) <= 1e-8 ||
        error("original fitted-response replay differs from calibration")
    maximum(abs,run.se-saved["standard_errors"]) <= 1e-8 ||
        error("original covariance replay differs from calibration")
    (; maximum_response_difference=maximum(abs,run.estimates-saved["estimates"]),
       maximum_se_difference=maximum(abs,run.se-saved["standard_errors"]))
end

function ui_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    opts = Dict("inputs"=>join([joinpath(@__DIR__,"results","uncertainty-calibration-$c-kan12")
                               for c in UC_CASES],","),
        "output"=>joinpath(@__DIR__,"results","uncertainty-initialization"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown initialization option $key"))
        opts[key] = value
    end
    inputs,output = abspath.(split(opts["inputs"],',')),abspath(opts["output"])
    BLAS.set_num_threads(1)
    scripts = [@__FILE__,joinpath(@__DIR__,"calibrate_uncertainty.jl"),
               joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    initial_source = source_hash()
    hashes = rd_followon_snapshot(output,scripts)
    rd_write_toml(joinpath(output,"design.toml"),Dict(
        "initializations"=>["original","constant"],"constant_rate"=>0.5,
        "nullspace_penalties"=>[1e-6,0.0],
        "selection"=>"maximum finite reported LAML criterion within each prior on training observations",
        "between_prior_selection"=>"none; intrinsic and proper-prior criterion values are not ranked together",
        "scope"=>"post-hoc initialization/prior diagnosis, not independently validated recalibration",
        "bootstrap_refit"=>false,"original_bootstraps_reused_for_changed_fits"=>false))
    trials,selected,intervals,parity = NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    records,inputs_sha = Dict{String,Any}(),Dict{String,String}()
    seen = Set{Tuple{String,Int}}()
    for input in inputs
        design = TOML.parsefile(joinpath(input,"design.toml"))
        design["models"] == ["kan12"] || error("diagnosis requires the original KAN calibration cohort")
        meta = TOML.parsefile(joinpath(input,"metadata.toml"))
        rd_digest(joinpath(input,"design.toml")) == meta["design_sha256"] ||
            error("calibration design fingerprint mismatch")
        for name in ("design.toml","metadata.toml","queries.csv")
            path = joinpath(input,name)
            inputs_sha[relpath(path,ROOT)] = rd_digest(path)
        end
        for case in design["cases"],seed in design["seeds"]
            (case,seed) in seen && error("duplicate calibration dataset")
            push!(seen,(case,seed))
            reference = uc_reference(case)
            ds = uc_dataset(case,seed,reference)
            data_path = joinpath(input,"datasets","$(case)__$(seed).toml")
            saved_data = TOML.parsefile(data_path)
            hcat(saved_data["values"]...) == ds.values && saved_data["times"] == ds.times ||
                error("calibration observations differ from the replay")
            record_path = joinpath(input,"replicates","$(case)__kan12__$(seed).toml")
            saved = TOML.parsefile(record_path)
            for path in (data_path,record_path)
                inputs_sha[relpath(path,ROOT)] = rd_digest(path)
            end
            queries = uc_queries(case,reference)
            for nullspace in (1e-6,0.0)
                a = ui_approximator(nullspace)
                model = nullspace == 0 ? "kan12_free_affine" : "kan12_affine_penalty"
                runs = [ui_fit(a,ds,queries,kind,design["iterations"]) for kind in ("original","constant")]
                if nullspace == 1e-6
                    difference = ui_replay_parity(first(runs),saved)
                    push!(parity,(;case,seed,difference...))
                end
                append!(trials,getproperty.(runs,:row))
                winner = ui_select(getproperty.(runs,:row))
                winner === nothing && error("neither initialization gave an eligible LAML fit")
                push!(selected,merge(winner,(; search_seconds=sum(r.row.fit_seconds for r in runs))))
                for run in runs
                    key = "$(case)__$(seed)__$(model)__$(run.row.initialization)"
                    entry = Dict{String,Any}("estimates"=>run.estimates,"se"=>run.se,
                        "status"=>run.row.status,"selected"=>run.row.initialization==winner.initialization)
                    if run.fit !== nothing
                        entry["parameters"] = collect(run.fit.parameters)
                        entry["smoothing_params"] = run.fit.smoothing_params
                    end
                    records[key] = entry
                    for level in design["levels"]
                        z = PSM._qnorm(1-(1-level)/2)
                        for (i,q) in enumerate(queries)
                            row = uc_interval(q,run.estimates[i],run.estimates[i]-z*run.se[i],
                                run.estimates[i]+z*run.se[i];case,model,seed,
                                method=run.row.initialization,level,status=run.interval_status,
                                message=run.interval_message)
                            push!(intervals,row)
                            if run.row.initialization == winner.initialization
                                push!(intervals,merge(row,(; method="laml_selected")))
                            end
                        end
                    end
                end
                println("initialization $case seed=$seed nullspace=$nullspace selected=$(winner.initialization) ",
                        "original_laml=$(runs[1].row.laml) constant_laml=$(runs[2].row.laml)")
            end
            for (name,rows) in (("trials.csv",trials),("selected.csv",selected),
                                ("intervals.csv",intervals),("replay_parity.csv",parity))
                write_csv(joinpath(output,name),rows)
            end
            rd_write_toml(joinpath(output,"coefficients.toml"),records)
            flush(stdout)
        end
    end
    all(rd_digest(joinpath(ROOT,p))==hash for (p,hash) in inputs_sha) ||
        error("calibration archive changed during diagnosis")
    metadata = rd_followon_metadata(output,scripts,hashes,initial_source,[])
    merge!(metadata,Dict("inputs_sha256"=>inputs_sha,"bootstrap_refit"=>false,
        "fitting_source"=>initial_source,"exploratory"=>true,
        "normal_quantiles"=>[Dict("level"=>level,"z"=>PSM._qnorm(1-(1-level)/2))
                            for level in unique(getproperty.(intervals,:level))],
        "design_sha256"=>rd_digest(joinpath(output,"design.toml"))))
    rd_write_toml(joinpath(output,"metadata.toml"),metadata)
    (; trials,selected,intervals,parity,records)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ui_main(ARGS)
end
