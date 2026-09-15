include("calibrate_uncertainty.jl")

const CF_CONFIRM = ["spline8","kan12_free"]
const CF_SCREEN = ["spline12","spde12","gp12_fixed","mlp13","kan28_free"]

function cf_model(model)
    initial = x -> 0.5
    prior,initialization = "","constant response 0.5"
    a = if model in ("spline8","spline12")
        prior = "natural cubic curvature; unpenalized affine component"
        BSplineApproximator(:r,UC_DOMAIN,model=="spline8" ? 8 : 12;initial)
    elseif model == "spde12"
        prior = "native Matern SPDE precision; nu=1.5; fixed range=0.8; full-rank prior"
        SPDEApproximator(:r,UC_DOMAIN,12;nu=1.5,range_param=0.8,initial)
    elseif model == "gp12_fixed"
        prior = "nodal spline curvature; fixed Matern-5/2 interpolant, lengthscale=0.4, variance=1; not full GP inference"
        initialization = "inducing values 0.5; between-node initial values follow the kernel interpolant"
        GPApproximator(:r,UC_DOMAIN,12;kernel=:matern52,lengthscale=0.4,variance=1.0,initial)
    elseif model == "mlp13"
        prior = "native identity penalty on all weights and biases; full-rank prior"
        initialization = "seed42 hidden features; zero output weights and output bias 0.5"
        NeuralApproximator(:r,Lux.Chain(Lux.Dense(1,4,tanh),Lux.Dense(4,1));
                          domain=UC_DOMAIN,penalty_weight=1.0,rng_seed=42)
    elseif model in ("kan12_free","kan28_free")
        layer(ni,no,grid) = LuxKANLinear(ni,no;grid_size=grid,spline_order=3,
                                        standalone_spline_scale=false)
        network = model=="kan12_free" ? layer(1,1,8) : Lux.Chain(layer(1,2,3),layer(2,1,3))
        prior = "complete-edge curvature; zero affine penalty; one smoothing block per layer"
        initialization = model=="kan12_free" ? "original cached seed42 initialization" :
            "seed42 hidden features; constant final response 0.5"
        KANApproximator(:r,network;input_domains=(UC_DOMAIN,),penalty=:edge_curvature,
                       nullspace_penalty=0.0,rng_seed=42)
    else
        throw(ArgumentError("unknown fresh coverage model $model"))
    end
    beta = initial_params(a)
    if model == "mlp13"
        beta[end-4:end-1] .= 0.0
        beta[end] = 0.5
    elseif model == "kan28_free"
        layer = last(a.layers)
        beta[layer.base_range] .= 0.0
        beta[layer.spline_range] .= 0.5/layer.input_dim
    end
    fit = InitializedApprox(:r,a,beta,penalty_matrix(a))
    blocks = penalty_blocks(a)
    metadata = Dict{String,Any}("parameters"=>nparams(a),"prior"=>prior,
        "initialization"=>initialization,"initial_parameters"=>beta,
        "initial_query_values"=>Float64.(build_evaluator(fit,beta).(UC_POINTS)),
        "penalty_ranks"=>[PSM._rank_penalty(S) for (S,_) in blocks],
        "penalty_ranges"=>[collect(idx) for (_,idx) in blocks],
        "covariance_gradient"=>a isa KANApproximator ? "ForwardDiff" : "central finite differences")
    (; fit,report=a,metadata)
end

function cf_options(args)
    opts = Dict("stage"=>"confirm","cases"=>"","models"=>"","seeds"=>"",
        "iterations"=>"40","nboot"=>"99","levels"=>"0.9,0.95",
        "sensitivity-seeds"=>"","sensitivity-boot"=>"199","output"=>"")
    supplied = Set{String}()
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown fresh coverage option $key"))
        opts[key] = value
        push!(supplied,key)
    end
    stage = opts["stage"]
    stage in ("confirm","screen") || throw(ArgumentError("stage must be confirm or screen"))
    "cases" in supplied || (opts["cases"]=stage=="confirm" ? "logistic,nonlinear" : "nonlinear")
    "models" in supplied || (opts["models"]=join(stage=="confirm" ? CF_CONFIRM : CF_SCREEN,","))
    "seeds" in supplied || (opts["seeds"]=join(stage=="confirm" ? (2001:2100) : (2001:2020),","))
    "nboot" in supplied || (opts["nboot"]=stage=="confirm" ? "99" : "19")
    "sensitivity-seeds" in supplied || (opts["sensitivity-seeds"]=stage=="confirm" ? join(2001:2020,",") : "")
    "output" in supplied || (opts["output"]=joinpath(@__DIR__,"results","coverage-fresh-$stage"))
    cases,models = String.(split(opts["cases"],',')),String.(split(opts["models"],','))
    seeds = parse.(Int,split(opts["seeds"],','))
    levels = parse.(Float64,split(opts["levels"],','))
    sensitivity = isempty(opts["sensitivity-seeds"]) ? Int[] : parse.(Int,split(opts["sensitivity-seeds"],','))
    iterations,nboot,sensitivity_boot = parse.(Int,[opts["iterations"],opts["nboot"],opts["sensitivity-boot"]])
    all(in(UC_CASES),cases) && all(in([CF_CONFIRM;CF_SCREEN]),models) ||
        throw(ArgumentError("unknown cases or models"))
    all(allunique,(cases,models,seeds,levels,sensitivity)) &&
        all(>(0),seeds) && all(>(0),sensitivity) && all(x->0<x<1,levels) ||
            throw(ArgumentError("seeds/levels must be valid and identifiers unique"))
    iterations>0 && nboot>=3 && sensitivity_boot>nboot ||
        throw(ArgumentError("require positive iterations and 3 <= nboot < sensitivity-boot"))
    # Development runs may override seeds, but the retained default cohorts
    # cannot overlap the earlier 1001:1100 calibration.
    (; stage,cases,models,seeds,levels,iterations,nboot,sensitivity_boot,
       sensitivity_seeds=intersect(seeds,sensitivity),output=abspath(opts["output"]))
end

function cf_main(args)
    options = cf_options(args)
    cfg = (; (k=>v for (k,v) in pairs(options) if k != :stage)...)
    specs = Dict(model=>cf_model(model).metadata for model in cfg.models)
    extra = Dict{String,Any}("study_stage"=>options.stage,"family_specs"=>specs,
        "fresh_against_original_calibration"=>isempty(intersect(cfg.seeds,1001:1100)),
        "source_data_policy"=>"new independent noise draws; no tuning on earlier coverage",
        "confirmation_scope"=>"spline8 and kan12_free, 100 datasets per case",
        "screen_scope"=>"20 matched nonlinear datasets; explicit different priors, not a universal family ranking",
        "gp_scope"=>"finite kernel interpolant and coefficient uncertainty only; no conditional GP process variance")
    scripts = [@__FILE__,joinpath(@__DIR__,"calibrate_uncertainty.jl"),
               joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    uc_run(cfg;model_factory=cf_model,design_extra=extra,scripts)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cf_main(ARGS)
end
