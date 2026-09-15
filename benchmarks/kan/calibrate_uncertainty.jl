include("rd_archive.jl")

const UC_CASES = ["logistic","nonlinear"]
const UC_MODELS = ["spline8","kan12"]
const UC_DOMAIN = (0.0,2.4)
const UC_INITIAL = [0.2,0.7]
const UC_POINTS = [0.05,0.1,collect(0.3:0.15:1.8)...,2.1,2.25,2.35]

function uc_failure_message(e)
    numerical = e isa Union{DomainError,OverflowError,DivideError,PosDefException,
                            SingularException,ZeroPivotException,LAPACKException,RankDeficientException} ||
                (e isa InexactError && !PSM._is_program_error(e))
    unavailable = e isa ErrorException &&
        (startswith(e.msg,"bootstrap: only ") ||
         startswith(e.msg,"confidence_band: posterior covariance "))
    numerical || unavailable || throw(e)
    sprint(showerror,e)
end

function uc_rate(case,x)
    case == "logistic" && return 0.9*(1-x/2)
    case == "nonlinear" && return 0.9*(1-x/2)*(1+0.35sin(2x))
    throw(ArgumentError("unknown uncertainty calibration case $case"))
end

function uc_reference(case)
    times = collect(0.0:0.25:5.0)
    rhs! = (du,u,p,t) -> (du .= uc_rate.(Ref(case),u).*u)
    ode = ODEProblem(rhs!,copy(UC_INITIAL),(0.0,5.0))
    sol = PSM.OrdinaryDiffEq.solve(ode,Tsit5();saveat=times,abstol=1e-11,reltol=1e-11)
    PSM.SciMLBase.successful_retcode(sol) || error("calibration reference solve failed")
    values = permutedims(reduce(hcat,[sol(t) for t in times]))
    (; times,values)
end

function uc_queries(case,reference)
    lo,hi = extrema(reference.values)
    [(; point_id=i,x,truth=Float64(uc_rate(case,x)),
        region=x < lo ? "below_range" : x > hi ? "above_range" : "in_range",
        training_min=lo,training_max=hi) for (i,x) in enumerate(UC_POINTS)]
end

function uc_dataset(case,seed,reference=uc_reference(case))
    data_seed = 100_000findfirst(==(case),UC_CASES)+seed
    sigma = 0.015
    values = reference.values + sigma*randn(StableRNG(data_seed),size(reference.values))
    (; case,seed,data_seed,sigma,times=reference.times,values,reference=reference.values)
end

function uc_approximator(model)
    model == "spline8" && return BSplineApproximator(:r,UC_DOMAIN,8;initial=x->0.5)
    model == "kan12" && return KANApproximator(:r,
        LuxKANLinear(1,1;grid_size=8,spline_order=3,standalone_spline_scale=false);
        input_domains=(UC_DOMAIN,),penalty=:edge_curvature,nullspace_penalty=1e-6,rng_seed=42)
    throw(ArgumentError("unknown uncertainty calibration model $model"))
end

function uc_problem(a,ds)
    rhs! = (du,u,p,t) -> (du .= p.r.(u).*u)
    PSMProblem(rhs!,copy(UC_INITIAL),(0.0,5.0),[a];
        data_times=ds.times,data_values=ds.values,abstol=1e-8,reltol=1e-8,maxiters=10_000)
end

mutable struct UCRefitter{A}
    algorithm::A
    attempted::Int
    finite_attempts::Vector{Int}
    diagnostics::Vector{Dict{String,Any}}
end
UCRefitter(algorithm) = UCRefitter(algorithm,0,Int[],Dict{String,Any}[])
function PSM.SciMLBase.solve(prob::PSMProblem,alg::UCRefitter)
    alg.attempted += 1
    sol = solve(prob,alg.algorithm)
    if all(isfinite,sol.parameters) && all(isfinite,sol.fitted_values)
        push!(alg.finite_attempts,alg.attempted)
        c = sol.convergence === nothing ? NamedTuple() : sol.convergence
        push!(alg.diagnostics,Dict("attempt"=>alg.attempted,
            "reason"=>string(get(c,:reason,:not_reported)),
            "converged"=>get(c,:converged,"not_reported"),
            "smoothing_advanced"=>get(c,:smoothing_advanced,"not_reported"),
            "stationarity"=>get(c,:stationarity,NaN),
            "edf"=>sol.edf,"data_loss"=>sol.data_loss))
        hasproperty(c,:procedure) &&
            (alg.diagnostics[end]["procedure"] = Dict(string(k)=>v for (k,v) in pairs(c.procedure)))
    end
    sol
end

function uc_percentile(values,attempts,budget,level)
    size(values,2) == length(attempts) || throw(DimensionMismatch("bootstrap columns and attempt IDs differ"))
    issorted(attempts) && allunique(attempts) && all(>(0),attempts) ||
        throw(ArgumentError("bootstrap attempt IDs must be positive and strictly increasing"))
    budget >= 3 && 0 < level < 1 || throw(ArgumentError("invalid bootstrap budget or interval level"))
    columns = findall(<=(budget),attempts)
    alpha = (1-level)/2
    lower,upper = fill(NaN,size(values,1)),fill(NaN,size(values,1))
    usable = zeros(Int,size(values,1))
    for i in axes(values,1)
        finite = filter(isfinite,values[i,columns])
        usable[i] = length(finite)
        if length(columns) >= 3 && usable[i] >= 3
            lower[i],upper[i] = quantile(finite,alpha),quantile(finite,1-alpha)
        end
    end
    (; lower,upper,usable,n_success=length(columns))
end

function uc_interval(query,mean,lower,upper;case,model,seed,method,level,
                     budget=0,n_success=0,usable=0,status="ok",message="")
    available = status == "ok" && all(isfinite,(lower,upper)) && lower <= upper
    interval_status = available ? "ok" : status == "ok" ? "unavailable_interval" : status
    (; case,model,seed,method,level,query...,estimate=mean,lower,upper,
       available,covered=available && lower <= query.truth <= upper,
       width=available ? upper-lower : NaN,budget,n_success,usable,
       status=interval_status,message)
end

function uc_fit_dataset(model,ds,queries,cfg;algorithm=LAML(maxiters=cfg.iterations,jac=:forwarddiff),
                        approximator=uc_approximator(model),reporting_approximator=nothing)
    a = approximator
    reporting_approximator === nothing ||
        (a isa InitializedApprox && a.approx === reporting_approximator) ||
        throw(ArgumentError("reporting approximator must be the fitted wrapper's underlying model"))
    prob = uc_problem(a,ds)
    report_prob = reporting_approximator === nothing ? prob : uc_problem(reporting_approximator,ds)
    PSM.n_total_params(report_prob) == PSM.n_total_params(prob) ||
        throw(DimensionMismatch("fitting and reporting coefficient layouts must agree"))
    alg = algorithm
    bootstrap_seed = hasproperty(ds,:bootstrap_seed) ? ds.bootstrap_seed :
        800_000findfirst(==(ds.case),UC_CASES)+ds.seed
    budgets = ds.seed in cfg.sensitivity_seeds ? [cfg.nboot,cfg.sensitivity_boot] : [cfg.nboot]
    common = (; case=ds.case,model,seed=ds.seed)
    fit = nothing
    fit_status,fit_message = "ok",""
    started = time_ns()
    try
        fit = solve(prob,alg)
        if !all(isfinite,fit.parameters) || !all(isfinite,fit.fitted_values) ||
                !isfinite(fit.edf) || !isfinite(fit.data_loss)
            fit_status,fit_message = "fit_failed","non-finite parameters, fitted values, EDF or loss"
        end
    catch e
        fit_status,fit_message = "fit_failed",uc_failure_message(e)
    end
    fit_seconds = (time_ns()-started)/1e9
    c = fit === nothing ? NamedTuple() : fit.convergence
    fit_row = (; common...,nparams=nparams(a),fit_status,fit_message,fit_seconds,
        iterations=get(c,:iterations,0),reason=string(get(c,:reason,:not_started)),
        converged=get(c,:converged,missing),stationarity=get(c,:stationarity,missing),
        smoothing_advanced=get(c,:smoothing_advanced,missing),
        edf=fit === nothing ? NaN : fit.edf,sigma2=get(c,:sigma2,NaN),
        data_loss=fit === nothing ? NaN : fit.data_loss)
    record = Dict{String,Any}("case"=>ds.case,"model"=>model,"seed"=>ds.seed,
        "fit_status"=>fit_status,"fit_message"=>fit_message,"bootstrap_seed"=>bootstrap_seed,
        "budgets"=>budgets,"levels"=>cfg.levels,"points"=>getproperty.(queries,:x))
    hasproperty(c,:procedure) &&
        (record["procedure"] = Dict(string(k)=>v for (k,v) in pairs(c.procedure)))
    intervals = NamedTuple[]
    estimates = fill(NaN,length(queries))
    if fit_status == "ok"
        record["parameters"] = collect(fit.parameters)
        record["smoothing_params"] = fit.smoothing_params
        record["sigma2"] = get(c,:sigma2,NaN)
        if hasproperty(c,:V_beta) && c.V_beta !== nothing
            record["V_beta"] = [collect(row) for row in eachrow(c.V_beta)]
        end
        for i in eachindex(queries)
            estimates[i] = PSM._bootstrap_function_value(fit.unknown_functions[:r],queries[i].x)
        end
    end
    record["estimates"] = estimates
    for level in cfg.levels
        band = nothing
        status,message = fit_status,fit_message
        if fit_status == "ok"
            try
                band = confidence_band(fit,report_prob;level,uf_points=Dict(:r=>getproperty.(queries,:x)))[:r]
            catch e
                status,message = "covariance_failed",uc_failure_message(e)
            end
        end
        band === nothing || (record["standard_errors"] = band.se)
        for (i,query) in enumerate(queries)
            lo,hi = band === nothing ? (NaN,NaN) : (band.lower[i],band.upper[i])
            push!(intervals,uc_interval(query,estimates[i],lo,hi;
                common...,method="covariance",level,status,message))
        end
    end

    refitter = UCRefitter(alg)
    boot = nothing
    boot_status,boot_message = fit_status,fit_message
    started = time_ns()
    if fit_status == "ok"
        try
            boot = bootstrap(fit,prob,refitter;nboot=maximum(budgets),level=maximum(cfg.levels),
                uf_points=Dict(:r=>getproperty.(queries,:x)),rng=StableRNG(bootstrap_seed),parallel=false)
        catch e
            boot_status,boot_message = "bootstrap_failed",uc_failure_message(e)
        end
    end
    boot_seconds = (time_ns()-started)/1e9
    record["bootstrap_status"] = boot_status
    record["bootstrap_message"] = boot_message
    record["bootstrap_attempted"] = refitter.attempted
    record["finite_attempts"] = refitter.finite_attempts
    record["refit_diagnostics"] = refitter.diagnostics
    values = fill(NaN,length(queries),length(refitter.finite_attempts))
    if boot !== nothing
        boot.n_success == length(refitter.finite_attempts) ||
            error("bootstrap replicate identity accounting differs from the public API")
        values = boot.uf_values[:r]
        record["bootstrap_parameters"] = [collect(row) for row in eachrow(boot.coefs)]
        record["bootstrap_values"] = [collect(column) for column in eachcol(values)]
        check = uc_percentile(values,refitter.finite_attempts,maximum(budgets),maximum(cfg.levels))
        isequal(check.lower,boot.ci_uf[:r].lower) && isequal(check.upper,boot.ci_uf[:r].upper) ||
            error("calibration percentile calculation differs from the public API")
    end
    for budget in budgets,level in cfg.levels
        band = uc_percentile(values,refitter.finite_attempts,budget,level)
        status = boot_status == "ok" && band.n_success < 3 ? "insufficient_refits" : boot_status
        for (i,query) in enumerate(queries)
            push!(intervals,uc_interval(query,estimates[i],band.lower[i],band.upper[i];
                common...,method="bootstrap",level,budget,n_success=band.n_success,
                usable=band.usable[i],status,message=boot_message))
        end
    end
    bootstrap_row = (; common...,boot_status,boot_message,bootstrap_seed,
        requested=maximum(budgets),attempted=refitter.attempted,
        n_success=length(refitter.finite_attempts),boot_seconds)
    (; fit_row,bootstrap_row,intervals,record)
end

function uc_options(args)
    opts = Dict("cases"=>join(UC_CASES,","),"models"=>join(UC_MODELS,","),
        "seeds"=>join(1001:1100,","),"iterations"=>"40","nboot"=>"99",
        "levels"=>"0.90,0.95","sensitivity-seeds"=>join(1001:1020,","),
        "sensitivity-boot"=>"199","output"=>joinpath(@__DIR__,"results","uncertainty-calibration"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown calibration option $key"))
        opts[key] = value
    end
    cases,models = String.(split(opts["cases"],',')),String.(split(opts["models"],','))
    seeds = parse.(Int,split(opts["seeds"],','))
    sensitivity_seeds = isempty(opts["sensitivity-seeds"]) ? Int[] :
        parse.(Int,split(opts["sensitivity-seeds"],','))
    levels = parse.(Float64,split(opts["levels"],','))
    iterations,nboot,sensitivity_boot = parse.(Int,[opts["iterations"],opts["nboot"],opts["sensitivity-boot"]])
    all(in(UC_CASES),cases) && all(in(UC_MODELS),models) && all(x->0<x<1,levels) ||
        throw(ArgumentError("unknown cases, models or invalid interval levels"))
    all(allunique,(cases,models,seeds,levels,sensitivity_seeds)) &&
        all(>(0),seeds) && all(>(0),sensitivity_seeds) ||
            throw(ArgumentError("identifiers/levels must be unique and seeds positive"))
    iterations > 0 && nboot >= 3 && sensitivity_boot > nboot ||
        throw(ArgumentError("positive iterations and 3 <= nboot < sensitivity-boot are required"))
    (; cases,models,seeds,levels,iterations,nboot,sensitivity_boot,
       sensitivity_seeds=intersect(seeds,sensitivity_seeds),output=abspath(opts["output"]))
end

function uc_run(cfg;model_factory=nothing,scenario_factory=nothing,design_extra=Dict{String,Any}(),
                scripts=[@__FILE__,joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")])
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    BLAS.set_num_threads(1)
    initial_source = source_hash()
    scenarios=scenario_factory === nothing ? nothing :
        Dict(case=>scenario_factory(case) for case in cfg.cases)
    if scenarios !== nothing
        all(getproperty.(s.queries,:x)==UC_POINTS for s in values(scenarios)) ||
            throw(ArgumentError("custom scenarios must retain the declared physical query grid"))
        all(isfinite(s.metadata["noise_sigma"]) && s.metadata["noise_sigma"]>=0
            for s in values(scenarios)) || throw(ArgumentError("scenario noise scales must be finite and nonnegative"))
    end
    design = Dict{String,Any}(string(k)=>v for (k,v) in pairs(cfg))
    delete!(design,"output")
    noise=scenarios === nothing ? 0.015 :
        Dict(case=>scenarios[case].metadata["noise_sigma"] for case in cfg.cases)
    merge!(design,Dict("domain"=>collect(UC_DOMAIN),"initial_states"=>UC_INITIAL,"noise_sigma"=>noise,
        "query_points"=>UC_POINTS,
        "model_seed"=>42,"jacobian"=>"forwarddiff","bootstrap_method"=>"parametric",
        "bootstrap_execution"=>"serial; prefixes selected by attempted replicate identity",
        "selection"=>"none; architectures, initialization, queries and budgets are fixed",
        "replication_unit"=>"independent noisy dataset; query points are correlated within dataset"))
    isempty(intersect(keys(design),keys(design_extra))) ||
        throw(ArgumentError("extra design metadata must not override the declared experiment"))
    merge!(design,design_extra)
    scenarios === nothing ||
        (design["scenarios"]=Dict(case=>scenarios[case].metadata for case in cfg.cases))
    hashes = rd_followon_snapshot(cfg.output,scripts)
    rd_write_toml(joinpath(cfg.output,"design.toml"),design)
    mkpath(joinpath(cfg.output,"replicates"))
    mkpath(joinpath(cfg.output,"datasets"))
    queries_all = NamedTuple[]
    for case in cfg.cases
        scenario=scenarios === nothing ? nothing : scenarios[case]
        reference = scenario === nothing ? uc_reference(case) : scenario.reference
        queries = scenario === nothing ? uc_queries(case,reference) : scenario.queries
        append!(queries_all,[merge((;case),row) for row in queries])
        write_csv(joinpath(cfg.output,"queries.csv"),queries_all)
        for seed in cfg.seeds
            ds = scenario === nothing ? uc_dataset(case,seed,reference) : scenario.dataset(seed)
            if scenario !== nothing
                ds.case==case && ds.seed==seed && ds.times==reference.times &&
                    size(ds.values)==size(reference.values) && ds.sigma==noise[case] ||
                    throw(ArgumentError("scenario dataset does not match its declared condition"))
            end
            rd_write_toml(joinpath(cfg.output,"datasets","$(case)__$(seed).toml"),
                Dict("data_seed"=>ds.data_seed,"times"=>ds.times,
                     "values"=>[collect(column) for column in eachcol(ds.values)]))
            for model in cfg.models
                run = if model_factory === nothing
                    uc_fit_dataset(model,ds,queries,cfg)
                else
                    spec = model_factory(model)
                    uc_fit_dataset(model,ds,queries,cfg;
                        approximator=spec.fit,reporting_approximator=spec.report,
                        algorithm=get(spec,:algorithm,LAML(maxiters=cfg.iterations,jac=:forwarddiff)))
                end
                for (name,rows) in (("fits.csv",[run.fit_row]),
                    ("bootstrap.csv",[run.bootstrap_row]),("intervals.csv",run.intervals))
                    path = joinpath(cfg.output,name)
                    exists = isfile(path)
                    CSV.write(path,rows;append=exists,header=!exists)
                end
                rd_write_toml(joinpath(cfg.output,"replicates","$(case)__$(model)__$(seed).toml"),run.record)
                println("calibration $case seed=$seed model=$model fit=$(run.fit_row.fit_status) ",
                    "bootstrap=$(run.bootstrap_row.n_success)/$(run.bootstrap_row.attempted) ",
                    "seconds=$(run.fit_row.fit_seconds+run.bootstrap_row.boot_seconds)")
                flush(stdout)
            end
        end
    end
    metadata = rd_followon_metadata(cfg.output,scripts,hashes,initial_source,[])
    metadata["julia_threads"] = Threads.nthreads()
    metadata["design_sha256"] = rd_digest(joinpath(cfg.output,"design.toml"))
    metadata["normal_quantiles"] = [Dict("level"=>level,
        "z"=>PSM._qnorm(1-(1-level)/2)) for level in cfg.levels]
    rd_write_toml(joinpath(cfg.output,"metadata.toml"),metadata)
    cfg
end

uc_main(args) = uc_run(uc_options(args))

if abspath(PROGRAM_FILE) == @__FILE__
    uc_main(ARGS)
end
