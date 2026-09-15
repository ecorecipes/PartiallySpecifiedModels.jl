include("coverage_robustness.jl")

function sb_score(fitted,lower,upper,truth)
    length(fitted)==length(lower)==length(upper)==length(truth) && !isempty(truth) ||
        throw(DimensionMismatch("band score vectors must align"))
    available=isfinite.(lower) .& isfinite.(upper) .& (lower .<= upper)
    covered=available .& (lower .<= truth) .& (truth .<= upper)
    usable=count(available)
    (;points=length(truth),point_availability=usable/length(truth),
       point_coverage_yield=count(covered)/length(truth),
       band_available=all(available),joint_covered=all(covered),
       mean_width=usable==0 ? NaN : mean((upper-lower)[available]),
       max_width=usable==0 ? NaN : maximum((upper-lower)[available]))
end

function sb_solution(a,prob,saved,fit_row)
    beta=Float64.(saved["parameters"])
    covariance=permutedims(hcat(saved["V_beta"]...))
    f=build_evaluator(a,beta)
    prediction=simulate(prob,beta)
    all(isfinite,prediction) || error("archived original trajectory no longer evaluates")
    PSMSolution(PSM.ComponentArray(r=beta),NaN,parse(Float64,fit_row.data_loss),
        parse(Float64,fit_row.edf),Float64.(saved["smoothing_params"]),
        prediction,copy(prob.data_values),copy(prob.data_times),
        Dict{Symbol,Any}(:r=>f),(V_beta=covariance,sigma2=saved["sigma2"]))
end

function sb_bootstrap(prob,a,saved,points,budget)
    attempts=saved["finite_attempts"]
    columns=findall(<=(budget),attempts)
    length(columns)>=3 || error("archived bootstrap has insufficient successful refits")
    betas=permutedims(hcat(saved["bootstrap_parameters"][columns]...))
    n=length(columns)
    values=Matrix{Float64}(undef,length(points),n)
    trajectories=Array{Float64}(undef,length(prob.data_times),size(prob.data_values,2),n)
    original_values=permutedims(hcat(saved["bootstrap_values"][columns]...))
    replay_error=0.0
    for b in 1:n
        f=build_evaluator(a,@view betas[b,:])
        values[:,b] .= f.(points)
        trajectories[:,:,b] .= simulate(prob,@view betas[b,:])
        all(isfinite,@view trajectories[:,:,b]) || error("archived bootstrap trajectory no longer evaluates")
        replay_error=max(replay_error,maximum(abs,f.(saved["points"])-original_values[b,:]))
    end
    # These are exact coefficient replays, not refits. The original
    # cross-source numerical replay discrepancy was <=1.22e-17.
    replay_error<=1e-8 || error("bootstrap function replay changed the archived fit")
    lower=dropdims(mapslices(x->quantile(x,0.025),trajectories;dims=3);dims=3)
    upper=dropdims(mapslices(x->quantile(x,0.975),trajectories;dims=3);dims=3)
    lo=[quantile(values[i,:],0.025) for i in axes(values,1)]
    hi=[quantile(values[i,:],0.975) for i in axes(values,1)]
    bs=BootstrapResult(betas,trajectories,Dict(:r=>values),Dict(:r=>copy(points)),
        (lower=lower,upper=upper),Dict{Symbol,NamedTuple{(:lower,:upper),Tuple{Vector{Float64},Vector{Float64}}}}(
            :r=>(lower=lo,upper=hi)),0.95,n)
    (;bs,replay_error)
end

function sb_main(args)
    opts=Dict("inputs"=>join([joinpath(@__DIR__,"results","robustness-$m-$f")
            for m in ("spline8","kan12_free") for f in ("1","quarter")],","),
        "output"=>joinpath(@__DIR__,"results","simultaneous-band-assessment"),
        "nsim"=>"10000","dense"=>"101","seeds"=>"")
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value=split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown simultaneous assessment option $key"))
        opts[key]=value
    end
    nsim,dense=parse.(Int,[opts["nsim"],opts["dense"]])
    nsim>=2 && dense>=2 || throw(ArgumentError("simulation and grid counts must be at least two"))
    output=abspath(opts["output"])
    selected_seeds=isempty(opts["seeds"]) ? nothing : parse.(Int,split(opts["seeds"],','))
    selected_seeds===nothing || (allunique(selected_seeds) && all(>(0),selected_seeds)) ||
        throw(ArgumentError("selected seeds must be positive and unique"))
    BLAS.set_num_threads(1)
    scripts=[@__FILE__,joinpath(@__DIR__,"coverage_robustness.jl"),
        joinpath(@__DIR__,"undersmoothing.jl"),joinpath(@__DIR__,"calibrate_families.jl"),
        joinpath(@__DIR__,"calibrate_uncertainty.jl"),joinpath(@__DIR__,"rd_archive.jl"),
        joinpath(@__DIR__,"reaction_diffusion.jl")]
    initial_source=source_hash()
    hashes=rd_followon_snapshot(output,scripts)
    rd_write_toml(joinpath(output,"design.toml"),Dict(
        "nsim"=>nsim,"dense_base_points"=>dense,"levels"=>[0.9,0.95],
        "grids"=>["archived","dense_with_archived_points"],
        "scopes"=>["in_range","all_queries"],"bootstrap_budget"=>99,
        "covariance"=>"conditional Gaussian/delta; no analytic smoothing-uncertainty correction",
        "bootstrap"=>"original full-refit coefficient draws; original-centred maximum standardized deviations",
        "target"=>"per-function finite-grid joint coverage; not continuum coverage",
        "optimizer_refit"=>false,"quantiles"=>"Julia default linear interpolation (R type7)",
        "seed_selection"=>selected_seeds===nothing ? "all archived datasets" : selected_seeds))
    mkpath(joinpath(output,"bands"))
    fingerprints=Dict{String,String}()
    seen=Set{Tuple{String,String,Int}}()
    for input in abspath.(split(opts["inputs"],','))
        design=TOML.parsefile(joinpath(input,"design.toml"))
        origin=TOML.parsefile(joinpath(input,"metadata.toml"))
        rd_digest(joinpath(input,"design.toml"))==origin["design_sha256"] ||
            error("input design fingerprint mismatch")
        design["study_stage"]=="robustness" && design["nboot"]>=99 ||
            error("assessment requires the retained robustness study")
        for name in ("design.toml","metadata.toml","fits.csv","queries.csv")
            path=joinpath(input,name); fingerprints[relpath(path,ROOT)]=rd_digest(path)
        end
        fits=rd_read_csv(joinpath(input,"fits.csv"))
        for row in fits
            case,model,seed=row.case,row.model,parse(Int,row.seed)
            selected_seeds===nothing || seed in selected_seeds || continue
            key=(case,model,seed)
            key in seen && error("duplicate input fit")
            push!(seen,key)
            saved_path=joinpath(input,"replicates","$(case)__$(model)__$(seed).toml")
            data_path=joinpath(input,"datasets","$(case)__$(seed).toml")
            for path in (saved_path,data_path)
                fingerprints[relpath(path,ROOT)]=rd_digest(path)
            end
            saved=TOML.parsefile(saved_path)
            observed=TOML.parsefile(data_path)
            row.fit_status=="ok" && saved["bootstrap_status"]=="ok" ||
                error("this replay requires original successful fits; it cannot fabricate missing coefficient draws")
            spec=cf_model(design["family_specs"][model]["base_model"])
            a=spec.report
            ds=(times=Float64.(observed["times"]),values=hcat(observed["values"]...))
            prob=uc_problem(a,ds)
            sol=sb_solution(a,prob,saved,row)
            sparse=Float64.(saved["points"])
            replay=confidence_band(sol,prob;uf_points=Dict(:r=>sparse))[:r]
            maximum(abs,replay.fitted-saved["estimates"])<=1e-8 &&
                maximum(abs,replay.se-saved["standard_errors"])<=1e-8 ||
                    error("original pointwise covariance replay differs from archive")
            grids=(archived=sparse,dense_with_archived_points=sort!(unique(
                [collect(range(first(sparse),last(sparse),length=dense));sparse])))
            reference=cr_scenario(case)
            in_points=[q.x for q in reference.queries if q.region=="in_range"]
            lo,hi=extrema(in_points)
            scores=NamedTuple[]
            bands=Dict{String,Any}()
            max_replay_error=0.0
            for (grid_name,points) in pairs(grids)
                b=sb_bootstrap(prob,a,saved,points,99)
                max_replay_error=max(max_replay_error,b.replay_error)
                for scope in ("in_range","all_queries")
                    ids=scope=="in_range" ? findall(x->lo<=x<=hi,points) : collect(eachindex(points))
                    coordinates=points[ids]
                    truth=Float64[cr_rate(reference.metadata["response"],x) for x in coordinates]
                    seed_mc=6_000_000+100_000findfirst(==(case),CR_CASES)+seed
                    for level in (0.9,0.95),source in ("covariance","bootstrap"),interval in (:pointwise,:simultaneous)
                        name="$(grid_name)__$(scope)__$(source)__$(interval)__$(level)"
                        band=nothing
                        status,message="ok",""
                        try
                            band=source=="covariance" ?
                                confidence_band(sol,prob;level,interval,nsim,
                                    uf_points=Dict(:r=>coordinates),rng=StableRNG(seed_mc))[:r] :
                                confidence_band(b.bs,sol,prob;level,interval,
                                    uf_indices=Dict(:r=>ids))[:r]
                        catch e
                            status,message="unavailable",uc_failure_message(e)
                        end
                        fitted=sol.unknown_functions[:r].(coordinates)
                        lower=band===nothing ? fill(NaN,length(ids)) : band.lower
                        upper=band===nothing ? fill(NaN,length(ids)) : band.upper
                        score=sb_score(fitted,lower,upper,truth)
                        critical=band===nothing ? NaN : interval===:simultaneous ? band.critical :
                            source=="covariance" ? PSM._qnorm(1-(1-level)/2) : NaN
                        push!(scores,(;case,model,seed,grid=string(grid_name),scope,source,
                            interval=string(interval),level,status,message,critical,
                            gaussian_seed=seed_mc,gaussian_draws=source=="covariance" && interval===:simultaneous ?
                                (band===nothing ? 0 : band.n_draws) : 0,
                            bootstrap_refits=b.bs.n_success,score...))
                        bands[name]=Dict("points"=>coordinates,"truth"=>truth,"fitted"=>fitted,
                            "lower"=>lower,"upper"=>upper,"se"=>band===nothing ? fill(NaN,length(ids)) : band.se,
                            "critical"=>critical,"status"=>status,"message"=>message)
                    end
                end
            end
            path=joinpath(output,"dataset_bands.csv")
            CSV.write(path,scores;append=isfile(path),header=!isfile(path))
            rd_write_toml(joinpath(output,"bands","$(case)__$(model)__$(seed).toml"),bands)
            println("simultaneous $case $model seed=$seed bootstrap_replay_error=$max_replay_error")
            flush(stdout)
        end
    end
    all(rd_digest(joinpath(ROOT,path))==hash for (path,hash) in fingerprints) ||
        error("input archive changed during band assessment")
    isempty(seen) && error("no archived fits match the requested selection")
    meta=rd_followon_metadata(output,scripts,hashes,initial_source,[])
    merge!(meta,Dict("input_sha256"=>fingerprints,"optimizer_refit"=>false,
        "model_evaluation"=>true,"design_sha256"=>rd_digest(joinpath(output,"design.toml")),
        "fits"=>length(seen)))
    rd_write_toml(joinpath(output,"metadata.toml"),meta)
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    sb_main(ARGS)
end
