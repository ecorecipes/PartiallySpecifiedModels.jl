include("calibrate_families.jl")

struct UndersmoothingFit
    fraction::Float64
    iterations::Int
end
function UndersmoothingFit(;fraction=0.25,iterations=40)
    isfinite(fraction) && 0 < fraction <= 1 ||
        throw(ArgumentError("smoothing fraction must be in (0,1]"))
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    UndersmoothingFit(Float64(fraction),iterations)
end

function PSM.SciMLBase.solve(prob::PSMProblem,alg::UndersmoothingFit)
    prob.likelihood isa Gaussian && length(prob.approximators)==1 ||
        throw(ArgumentError("this experiment supports one Gaussian unknown function"))
    selected=solve(prob,LAML(maxiters=alg.iterations,jac=:forwarddiff))
    length(selected.smoothing_params)==1 ||
        throw(ArgumentError("this experiment requires one smoothing block"))
    lambda=only(selected.smoothing_params)
    all(isfinite,selected.parameters) && isfinite(lambda) && lambda>0 ||
        throw(DomainError(lambda,"smoothing-selection fit is not usable"))
    target=alg.fraction*lambda
    a=only(prob.approximators)
    underlying=a isa InitializedApprox ? a.approx : a
    warm=InitializedApprox(a.name,underlying,collect(selected.parameters),penalty_matrix(underlying))
    conditional=PSMProblem(prob.dynamics!,prob.u0,prob.tspan,[warm];
        data_times=prob.data_times,data_values=prob.data_values,data_weights=prob.data_weights,
        obs_to_state=prob.obs_to_state,known_params=prob.known_params,
        likelihood=prob.likelihood,solver=prob.ode_solver,discrete=prob.discrete,
        delays=prob.delays,history=prob.history,prob.ode_kwargs...)
    final=solve(conditional,LAML(maxiters=alg.iterations,jac=:forwarddiff,fixed_lambda=target))
    only(final.smoothing_params)==target || error("fixed-smoothing refit changed the requested lambda")
    c=selected.convergence
    procedure=(name="select_then_fixed_refit",fraction=alg.fraction,
        selected_lambda=lambda,final_lambda=target,
        selection_converged=c.converged,selection_reason=string(c.reason),
        selection_iterations=c.iterations,selection_stationarity=c.stationarity,
        selection_smoothing_advanced=c.smoothing_advanced)
    convergence=merge(final.convergence,(;procedure))
    PSMSolution(final.parameters,final.objective,final.data_loss,final.edf,final.smoothing_params,
        final.fitted_values,final.data_values,final.data_times,final.unknown_functions,convergence)
end

function us_main(args)
    opts=Dict("fractions"=>"1,0.25","models"=>"spline8,kan12_free",
        "cases"=>"logistic,nonlinear","seeds"=>join(3001:3050,","),
        "nboot"=>"99","iterations"=>"40","output"=>joinpath(@__DIR__,"results","coverage-undersmoothing"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value=split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown undersmoothing option $key"))
        opts[key]=value
    end
    fractions=parse.(Float64,split(opts["fractions"],','))
    allunique(fractions) && all(x->isfinite(x) && 0<x<=1,fractions) ||
        throw(ArgumentError("fractions must be unique values in (0,1]"))
    bases=String.(split(opts["models"],','))
    all(in(CF_CONFIRM),bases) && allunique(bases) ||
        throw(ArgumentError("only spline8 and kan12_free are supported"))
    cfg0=cf_options(["--stage=confirm","--cases=$(opts["cases"])","--models=$(opts["models"])",
        "--seeds=$(opts["seeds"])","--nboot=$(opts["nboot"])","--sensitivity-seeds=",
        "--sensitivity-boot=$(parse(Int,opts["nboot"])+1)","--iterations=$(opts["iterations"])",
        "--output=$(opts["output"])"])
    settings=Dict("$(model)_fraction_$(fraction)"=>(;model,fraction)
                  for model in bases for fraction in fractions)
    models=["$(model)_fraction_$(fraction)" for model in bases for fraction in fractions]
    cfg=merge((; (k=>v for (k,v) in pairs(cfg0) if k != :stage)...),(;models))
    factory=id->begin
        setting=settings[id]
        merge(cf_model(setting.model),(;algorithm=UndersmoothingFit(
            fraction=setting.fraction,iterations=cfg.iterations)))
    end
    specs=Dict(id=>merge(cf_model(settings[id].model).metadata,Dict(
        "base_model"=>settings[id].model,"smoothing_fraction"=>settings[id].fraction)) for id in models)
    extra=Dict{String,Any}("study_stage"=>"undersmoothing","family_specs"=>specs,
        "procedure"=>"select lambda using LAML, then warm-start a fixed-lambda coefficient fit",
        "selection_repeated_in_bootstrap"=>true,"fractions"=>fractions,
        "fixed_stage_smoothing_advanced"=>"false by design; inspect recorded selection-stage diagnostics",
        "data_policy"=>"independent datasets; no fraction chosen from measured coverage")
    scripts=[@__FILE__,joinpath(@__DIR__,"calibrate_families.jl"),
        joinpath(@__DIR__,"calibrate_uncertainty.jl"),joinpath(@__DIR__,"rd_archive.jl"),
        joinpath(@__DIR__,"reaction_diffusion.jl")]
    uc_run(cfg;model_factory=factory,design_extra=extra,scripts)
end

if abspath(PROGRAM_FILE)==@__FILE__
    us_main(ARGS)
end
