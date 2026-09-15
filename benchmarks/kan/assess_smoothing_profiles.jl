include("smoothing_profiles.jl")

const SP_METHODS = ["native","native_stable","fixed_selected","profile_grid","known_scale_grid","null_limit"]

function sp_row(common,rho,score,status,message,native)
    (;common...,rho,lambda=exp(rho),status,message,
      criterion=score === nothing ? NaN : score.criterion,
      known_criterion=score === nothing ? NaN : score.known_criterion,
      rss=score === nothing ? NaN : score.rss,penalty=score === nothing ? NaN : score.penalty,
      Q=score === nothing ? NaN : score.Q,edf=score === nothing ? NaN : score.edf,
      dispersion=score === nothing ? NaN : score.dispersion,
      profile_dispersion=score === nothing ? NaN : score.profile_dispersion,
      decrement=score === nothing ? NaN : score.decrement,
      local_gradient=score === nothing ? NaN : score.local_gradient,
      local_hessian=score === nothing ? NaN : score.local_hessian,
      equilibrated_condition=score === nothing ? NaN : score.equilibrated_condition,native...)
end

function sp_point_rows(common,method,uncertainty,points,truth,oracle;status="ok",message="")
    n = length(points)
    fitted = uncertainty === nothing ? fill(NaN,n) : uncertainty.fitted
    se = uncertainty === nothing ? fill(NaN,n) : uncertainty.se
    known_se = uncertainty === nothing ? fill(NaN,n) : uncertainty.known_se
    z = PSM._qnorm(0.975)
    [(;common...,method,point_id=i,x=points[i],truth=truth[i],oracle=oracle[i],
        fitted=fitted[i],se=se[i],known_se=known_se[i],status,message,
        covered=status=="ok" && abs(fitted[i]-truth[i])<=z*se[i],
        known_scale_covered=status=="ok" && abs(fitted[i]-truth[i])<=z*known_se[i],
        width=2z*se[i]) for i in 1:n]
end

function sp_best_score(scores,key)
    isempty(scores) && return nothing
    # This is the best SAMPLED profile point, not a global-optimality claim.
    candidates = sort!(collect(keys(scores)))
    scores[candidates[argmax([getproperty(scores[r],key) for r in candidates])]]
end

function sp_profile_main(args)
    opts = Dict("cases"=>"noisy,localized","models"=>"spline8,kan12_free","data"=>"archived",
        "seeds"=>join(4001:4030,","),"iterations"=>"80","tol"=>"1e-10","ode-tol"=>"1e-10",
        "rhos"=>join([-20.0,-18.0,-16.0,collect(-14.0:0.5:-2.0)...,0.0,2.0,5.0,10.0,20.0,30.0,40.0],","),
        "output"=>joinpath(@__DIR__,"results","smoothing-profile-diagnostics"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown profile option $key"))
        opts[key] = value
    end
    cases,models = String.(split(opts["cases"],',')),String.(split(opts["models"],','))
    seeds = parse.(Int,split(opts["seeds"],','))
    rhos = parse.(Float64,split(opts["rhos"],','))
    iterations = parse(Int,opts["iterations"])
    tol,ode_tol = parse.(Float64,[opts["tol"],opts["ode-tol"]])
    opts["data"] in ("archived","generated") && all(in(CR_CASES),cases) &&
        all(in(CF_CONFIRM),models) && all(>(0),seeds) &&
        (opts["data"]=="generated" || all(in(4001:4030),seeds)) &&
        all(allunique,(cases,models,seeds,rhos)) && all(isfinite,rhos) &&
        iterations>0 && all(x->isfinite(x)&&x>0,(tol,ode_tol)) ||
        throw(ArgumentError("invalid profile cases, models, archived seeds, grid or budget"))
    output = abspath(opts["output"])
    scripts = [@__FILE__,joinpath(@__DIR__,"smoothing_profiles.jl"),
        joinpath(@__DIR__,"coverage_robustness.jl"),joinpath(@__DIR__,"undersmoothing.jl"),
        joinpath(@__DIR__,"calibrate_families.jl"),joinpath(@__DIR__,"calibrate_uncertainty.jl"),
        joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    BLAS.set_num_threads(1)
    initial_source = source_hash()
    hashes = rd_followon_snapshot(output,scripts)
    design = Dict{String,Any}("cases"=>cases,"models"=>models,"seeds"=>seeds,"data_source"=>opts["data"],
        "rho_grid"=>sort(rhos),"local_steps"=>[0.1,0.2],"iterations"=>iterations,
        "tol"=>tol,"ode_tolerance"=>ode_tol,"native_iterations"=>40,
        "native_ode_tolerance"=>1e-8,"methods"=>SP_METHODS,
        "conditioning"=>"conditional working covariance only; these are bias/scale diagnostics",
        "starts"=>["selected","ascending from cached initialization","descending from fitted null limit"],
        "null_starts"=>["cached initialization","native selected fit"],
        "coefficient_selection"=>"smallest residual-plus-penalty sum; retain every attempted start",
        "smoothing_selection"=>"largest sampled within-model working-LAML; no posterior weights or LR cutoff",
        "prior"=>"same fixed-rank intrinsic penalty; deterministic unit-determinant affine shear",
        "noise_control"=>"known-scale scores and SEs use generating sigma only as an oracle diagnostic",
        "oracle"=>"unpenalized function projection on 801 density points, assessed on 1601; never a refit start",
        "query_grid"=>"76 refined in-range points, matching the previous analytic-covariance assessment",
        "data_policy"=>"all requested datasets, no truth-based tuning or low-EDF exclusions")
    rd_write_toml(joinpath(output,"design.toml"),design)
    mkpath(joinpath(output,"profiles"))
    mkpath(joinpath(output,"datasets"))
    fingerprints = Dict{String,String}()
    native_rows,profile_rows,candidate_rows,point_rows = NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    curvature_rows,selection_rows = NamedTuple[],NamedTuple[]
    oracles = Dict{String,Any}()
    query_counts = Dict{String,Int}()
    fits_by_model = Dict{String,Any}()
    for model in models
        opts["data"]=="archived" || continue
        input = joinpath(@__DIR__,"results","robustness-$model-1")
        metadata = TOML.parsefile(joinpath(input,"metadata.toml"))
        rd_digest(joinpath(input,"design.toml")) == metadata["design_sha256"] ||
            error("archived robustness design fingerprint differs")
        for file in ("design.toml","metadata.toml","fits.csv")
            path = joinpath(input,file)
            fingerprints[relpath(path,ROOT)] = rd_digest(path)
        end
        fits_by_model[model] = Dict((r.case,parse(Int,r.seed))=>r for r in rd_read_csv(joinpath(input,"fits.csv")))
    end
    for case in cases
        scenario = cr_scenario(case)
        bounds = extrema(q.x for q in scenario.queries if q.region=="in_range")
        points = filter(x->bounds[1]<=x<=bounds[2],sort!(unique(
            [collect(range(first(UC_POINTS),last(UC_POINTS),length=101));UC_POINTS])))
        query_counts[case] = length(points)
        truth_function = x->cr_rate(scenario.metadata["response"],x)
        truth = truth_function.(points)
        sigma2 = cr_settings(case).sigma^2
        for seed in seeds
            observation_hash, paired_data = nothing, nothing
            for model in models
                ds = if opts["data"]=="archived"
                    input = joinpath(@__DIR__,"results","robustness-$model-1")
                    path = joinpath(input,"datasets","$(case)__$(seed).toml")
                    digest = rd_digest(path)
                    fingerprints[relpath(path,ROOT)] = digest
                    observation_hash === nothing || observation_hash == digest || error("unmatched archived data")
                    observation_hash = digest
                    observed = TOML.parsefile(path)
                    (;times=Float64.(observed["times"]),values=hcat(observed["values"]...))
                else
                    generated = scenario.dataset(seed)
                    (;times=generated.times,values=generated.values)
                end
                if paired_data === nothing
                    paired_data = deepcopy(ds)
                    rd_write_toml(joinpath(output,"datasets","$(case)__$(seed).toml"),
                        Dict("times"=>ds.times,"values"=>collect.(eachcol(ds.values)),
                            "source"=>opts["data"],"sigma"=>sqrt(sigma2)))
                else
                    isequal(ds,paired_data) || error("models received different observations")
                end
                spec = cf_model(model)
                prob = uc_problem(spec.fit,ds)
                native = solve(prob,LAML(maxiters=40,jac=:forwarddiff))
                all(isfinite,native.parameters) && all(isfinite,native.fitted_values) ||
                    error("non-finite native fit; diagnostic stopped rather than dropping the dataset")
                lambda = only(native.smoothing_params)
                rho = log(lambda)
                archived_edf = opts["data"]=="archived" ?
                    parse(Float64,fits_by_model[model][case,seed].edf) : NaN
                common = (;case,model,seed)
                push!(native_rows,(;common...,lambda,rho,edf=native.edf,
                    data_loss=native.data_loss,sigma2=native.convergence.sigma2,
                    stationarity=native.convergence.stationarity,
                    smoothing_advanced=native.convergence.smoothing_advanced,
                    archived_control_edf=archived_edf))
                profile_prob = PSMProblem(prob.dynamics!,prob.u0,prob.tspan,prob.approximators;
                    data_times=prob.data_times,data_values=prob.data_values,data_weights=prob.data_weights,
                    abstol=ode_tol,reltol=ode_tol,maxiters=10_000)
                grid = sort!(unique([rhos;rho;rho-0.1;rho+0.1;rho-0.2;rho+0.2]))
                profile = sp_profile(profile_prob,grid;beta_selected=collect(native.parameters),iterations,tol)
                frame = profile.frame
                oracle_key = case*"__"*model
                oracle = if haskey(oracles,oracle_key)
                    oracles[oracle_key]
                else
                    projection = sp_representation_oracle(spec.report,truth_function,bounds)
                    data = Dict{String,Any}(string(k)=>v for (k,v) in pairs(projection))
                    data["query_values"] = build_evaluator(spec.report,projection.beta).(points)
                    data["points"],data["truth"] = points,truth
                    oracles[oracle_key] = data
                    data
                end
                native_design = sp_linear_design(spec.report,points)
                native_band = confidence_band(native,uc_problem(spec.report,ds);uf_points=Dict(:r=>points))[:r]
                native_known_se = [sqrt(sigma2)*PSM._band_standard_error(collect(row),native.convergence.V_beta)
                                   for row in eachrow(native_design)]
                append!(point_rows,sp_point_rows(common,"native",
                    (;fitted=native_band.fitted,se=native_band.se,known_se=native_known_se),
                    points,truth,oracle["query_values"]))
                local_candidates = [merge(common,r) for r in profile.candidates]
                append!(candidate_rows,[(; (k=>v for (k,v) in pairs(r) if k != :beta)...) for r in local_candidates])
                scores = Dict{Float64,Any}()
                records = Dict{String,Any}[]
                for rr in [grid;Inf]
                    score,status,message = nothing,"ok",""
                    if !haskey(profile.best,rr)
                        status,message = "fit_failed","all coefficient starts failed"
                    else
                        try
                            score = sp_score(profile.best[rr],frame;sigma2)
                            scores[rr] = score
                        catch e
                            status,message = "score_failed",sp_candidate_error(e)
                        end
                    end
                    default_native = (;native_criterion=NaN,native_penalty=NaN,native_profile_scale=NaN,
                        native_scale_floored=false,native_logdet_shift=NaN,native_ridge=NaN)
                    old_status,old_message = isfinite(rr) ? ("unavailable","") : ("not_applicable","")
                    old = default_native
                    if score !== nothing && isfinite(rr)
                        try
                            old = sp_native_terms(score,frame,penalty_matrix(spec.report))
                            old_status = "ok"
                        catch e
                            old_status,old_message = "failed",sp_candidate_error(e)
                        end
                    end
                    row = sp_row(common,rr,score,status,message,
                        (;old...,native_status=old_status,native_message=old_message))
                    push!(profile_rows,row)
                    record = Dict{String,Any}(string(k)=>v for (k,v) in pairs(row))
                    if score !== nothing
                        record["parameters"] = score.beta
                        record["coordinates"] = score.theta
                        record["R"] = collect.(eachrow(Matrix(score.R)))
                    end
                    push!(records,record)
                end
                native_stable = nothing
                stable_status,stable_message = "ok",""
                try
                    native_stable = sp_evaluate(profile_prob,frame,rho,collect(native.parameters);sigma2)
                catch e
                    stable_status,stable_message = "unavailable",sp_candidate_error(e)
                end
                best = sp_best_score(scores,:criterion)
                known = sp_best_score(scores,:known_criterion)
                methods = (native_stable=native_stable,fixed_selected=get(scores,rho,nothing),
                    profile_grid=best,known_scale_grid=known,null_limit=get(scores,Inf,nothing))
                for (method,score) in pairs(methods)
                    status = score === nothing ? "unavailable" : "ok"
                    message = score === nothing ? (method == :native_stable ? stable_message :
                        "required profile point unavailable") : ""
                    uncertainty = score === nothing ? nothing :
                        sp_function_uncertainty(score,native_design;sigma2)
                    append!(point_rows,sp_point_rows(common,string(method),uncertainty,points,truth,
                        oracle["query_values"];status,message))
                    push!(selection_rows,(;common...,method=string(method),status,message,
                        rho=score === nothing ? NaN : score.rho,
                        criterion=score === nothing ? NaN : score.criterion,
                        known_criterion=score === nothing ? NaN : score.known_criterion,
                        edf=score === nothing ? NaN : score.edf,
                        decrement=score === nothing ? NaN : score.decrement,
                        profile_scale=score === nothing ? NaN : score.profile_dispersion,
                        reported_scale=score === nothing ? NaN : score.dispersion))
                end
                for h in (0.1,0.2)
                    usable = all(haskey(scores,r) for r in (rho-h,rho,rho+h))
                    grad = usable ? (scores[rho+h].criterion-scores[rho-h].criterion)/(2h) : NaN
                    hessian = usable ? -((scores[rho+h].criterion-scores[rho].criterion)+
                        (scores[rho-h].criterion-scores[rho].criterion))/h^2 : NaN
                    push!(curvature_rows,(;common...,rho,step=h,available=usable,
                        refitted_gradient=grad,refitted_hessian=hessian,
                        local_gradient=usable ? scores[rho].local_gradient : NaN,
                        local_hessian=usable ? scores[rho].local_hessian : NaN))
                end
                rd_write_toml(joinpath(output,"profiles","$(case)__$(model)__$(seed).toml"),
                    Dict("native_parameters"=>collect(native.parameters),"native_lambda"=>lambda,
                        "nullity"=>frame.nullity,"penalty_rank"=>frame.rank,
                        "positive_eigenvalues"=>frame.eigenvalues,"discarded_eigenvalues"=>frame.discarded,
                        "rank_tolerance"=>frame.tolerance,"penalty_reconstruction_relative"=>frame.reconstruction_relative,
                        "U0"=>collect.(eachrow(frame.U0)),"penalized_frame"=>collect.(eachrow(frame.penalized)),
                        "shear"=>collect.(eachrow(frame.shear)),
                        "candidates"=>[Dict(string(k)=>v for (k,v) in pairs(r)) for r in local_candidates],
                        "profile"=>records))
                println("profile $case $model seed=$seed native_rho=$rho best_rho=$(best === nothing ? NaN : best.rho) " *
                    "known_rho=$(known === nothing ? NaN : known.rho) available=$(length(scores))/$(length(grid)+1)")
                flush(stdout)
            end
        end
    end
    expected = length(cases)*length(models)*length(seeds)
    length(native_rows)==expected && length(selection_rows)==5expected &&
        length(curvature_rows)==2expected &&
        length(point_rows)==sum(values(query_counts))*length(models)*length(seeds)*length(SP_METHODS) ||
        error("incomplete profile population")
    for (name,rows) in (("native_fits.csv",native_rows),("profiles.csv",profile_rows),
        ("candidates.csv",candidate_rows),("dataset_points.csv",point_rows),
        ("selections.csv",selection_rows),("curvature.csv",curvature_rows))
        CSV.write(joinpath(output,name),rows)
    end
    rd_write_toml(joinpath(output,"representation_oracles.toml"),oracles)
    all(rd_digest(joinpath(ROOT,path))==hash for (path,hash) in fingerprints) ||
        error("an input archive changed during profiling")
    metadata = rd_followon_metadata(output,scripts,hashes,initial_source,[])
    output_files = ["native_fits.csv","profiles.csv","candidates.csv","dataset_points.csv",
        "selections.csv","curvature.csv","representation_oracles.toml"]
    append!(output_files,["profiles/"*name for name in readdir(joinpath(output,"profiles"))])
    append!(output_files,["datasets/"*name for name in readdir(joinpath(output,"datasets"))])
    merge!(metadata,Dict("input_sha256"=>fingerprints,"native_fits"=>expected,"query_counts"=>query_counts,
        "coefficient_refit_attempts"=>length(candidate_rows),"new_bootstrap_refits"=>0,
        "design_sha256"=>rd_digest(joinpath(output,"design.toml")),
        "outputs_sha256"=>Dict(name=>rd_digest(joinpath(output,name)) for name in output_files)))
    rd_write_toml(joinpath(output,"metadata.toml"),metadata)
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    sp_profile_main(ARGS)
end
