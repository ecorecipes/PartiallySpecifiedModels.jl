include("undersmoothing.jl")

const CR_CASES=["reference","sparse","noisy","localized"]

function cr_settings(case)
    case in CR_CASES || throw(ArgumentError("unknown robustness condition $case"))
    (; step=case=="sparse" ? 0.5 : 0.25,
       sigma=case=="noisy" ? 0.03 : 0.015,
       response=case=="localized" ? "localized" : "nonlinear")
end

function cr_rate(response,x)
    response=="nonlinear" && return uc_rate("nonlinear",x)
    response=="localized" && return 0.9*(1-x/2)*(1+0.5exp(-((x-0.9)/0.25)^2))
    throw(ArgumentError("unknown robustness response $response"))
end

function cr_scenario(case)
    settings=cr_settings(case)
    times=collect(0.0:settings.step:5.0)
    rhs! = (du,u,p,t)->(du .= cr_rate.(Ref(settings.response),u).*u)
    ode=ODEProblem(rhs!,copy(UC_INITIAL),(0.0,5.0))
    sol=PSM.OrdinaryDiffEq.solve(ode,Tsit5();saveat=times,abstol=1e-11,reltol=1e-11)
    PSM.SciMLBase.successful_retcode(sol) || error("robustness reference solve failed")
    values=permutedims(reduce(hcat,[sol(t) for t in times]))
    reference=(;times,values)
    lo,hi=extrema(values)
    queries=[(;point_id=i,x,truth=Float64(cr_rate(settings.response,x)),
        region=x<lo ? "below_range" : x>hi ? "above_range" : "in_range",
        training_min=lo,training_max=hi) for (i,x) in enumerate(UC_POINTS)]
    id=findfirst(==(case),CR_CASES)
    dataset=seed->begin
        data_seed=4_000_000+10_000id+seed
        observed=values+settings.sigma*randn(StableRNG(data_seed),size(values))
        (;case,seed,data_seed,sigma=settings.sigma,times,values=observed,reference=values,
          bootstrap_seed=8_000_000+10_000id+seed)
    end
    metadata=Dict{String,Any}("noise_sigma"=>settings.sigma,"observation_step"=>settings.step,
        "observation_times"=>times,"response"=>settings.response,
        "truth"=>"0.9*(1-x/2) times the declared response multiplier",
        "localized_centre"=>0.9,"localized_width"=>0.25,"localized_amplitude"=>0.5,
        "condition_rng_id"=>id,"data_seed_base"=>4_000_000+10_000id,
        "bootstrap_seed_base"=>8_000_000+10_000id)
    (;reference,queries,dataset,metadata)
end

function cr_main(args)
    opts=Dict("cases"=>join(CR_CASES,","),"models"=>"spline8,kan12_free",
        "fractions"=>"1,0.25","seeds"=>join(4001:4030,","),
        "nboot"=>"99","iterations"=>"40","output"=>joinpath(@__DIR__,"results","coverage-robustness"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value=split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown robustness option $key"))
        opts[key]=value
    end
    cases=String.(split(opts["cases"],','))
    bases=String.(split(opts["models"],','))
    fractions=parse.(Float64,split(opts["fractions"],','))
    seeds=parse.(Int,split(opts["seeds"],','))
    nboot,iterations=parse.(Int,[opts["nboot"],opts["iterations"]])
    all(in(CR_CASES),cases) && all(in(CF_CONFIRM),bases) &&
        all(allunique,(cases,bases,fractions,seeds)) ||
            throw(ArgumentError("unknown or repeated conditions/models/settings"))
    all(>(0),seeds) && all(x->isfinite(x)&&0<x<=1,fractions) && nboot>=3 && iterations>0 ||
        throw(ArgumentError("invalid seed, smoothing fraction or fitting budget"))
    settings=Dict("$(model)_fraction_$(fraction)"=>(;model,fraction)
                  for model in bases for fraction in fractions)
    models=["$(model)_fraction_$(fraction)" for model in bases for fraction in fractions]
    cfg=(;cases,models,seeds,levels=[0.9,0.95],iterations,nboot,
          sensitivity_boot=nboot+1,sensitivity_seeds=Int[],output=abspath(opts["output"]))
    factory=id->merge(cf_model(settings[id].model),(;algorithm=UndersmoothingFit(
        fraction=settings[id].fraction,iterations=iterations)))
    specs=Dict(id=>merge(cf_model(settings[id].model).metadata,Dict(
        "base_model"=>settings[id].model,"smoothing_fraction"=>settings[id].fraction)) for id in models)
    extra=Dict{String,Any}("study_stage"=>"robustness","family_specs"=>specs,
        "procedure"=>"LAML selection followed by fixed-fraction coefficient fitting, repeated in every bootstrap",
        "fractions"=>fractions,"factor_selection"=>"none; same declared factors as the previous experiment",
        "paired_data"=>"same observations across models/fractions within each condition",
        "across_conditions"=>"independent random streams; summarize conditions separately")
    scripts=[@__FILE__,joinpath(@__DIR__,"undersmoothing.jl"),
        joinpath(@__DIR__,"calibrate_families.jl"),joinpath(@__DIR__,"calibrate_uncertainty.jl"),
        joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    uc_run(cfg;model_factory=factory,scenario_factory=cr_scenario,design_extra=extra,scripts)
end

if abspath(PROGRAM_FILE)==@__FILE__
    cr_main(ARGS)
end
