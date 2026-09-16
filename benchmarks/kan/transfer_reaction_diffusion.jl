isdefined(@__MODULE__, :rd_load_archive) || include("rd_archive.jl")

const RD_TRANSFER_METRICS = (:field_rmse,:projected_model_rmse,:projected_to_finest_truth_rmse,
    :truth_mesh_rmse,:learned_mesh_shift_rmse,:learned_to_finest_mesh_rmse)

function rd_transfer_options(args)
    opts = Dict("inputs"=>join([joinpath(@__DIR__,"results","reaction-diffusion-$t")
                                for t in ("adam","laml")],","),
        "output"=>joinpath(@__DIR__,"results","reaction-diffusion-transfer"),"cells"=>"24,48,96")
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown mesh-transfer option $key"))
        opts[key] = value
    end
    cells = parse.(Int,split(opts["cells"],','))
    length(cells) >= 2 && issorted(cells) && allunique(cells) &&
        first(cells) >= 2 && all(n -> n % first(cells) == 0,cells) ||
            throw(ArgumentError("cells must be increasing integer multiples of the fitting mesh"))
    inputs = abspath.(split(opts["inputs"],','))
    allunique(inputs) || throw(ArgumentError("duplicate input archives"))
    (; inputs,output=abspath(opts["output"]),cells)
end

function rd_transfer_profile(a, beta, case, profile, cells, times, references)
    base, finest = first(cells),last(cells)
    predictions = Dict{Int,Union{Nothing,Matrix{Float64}}}()
    messages = Dict{Int,String}()
    calls = Dict{Int,Int}()
    for n in cells
        counter = Ref(0)
        p = rd_problem(a,[profile],rd_grid(n),times,references[n],counter)
        prediction, message = nothing,""
        try
            values = simulate(p,beta)
            if all(isfinite,values)
                prediction = Matrix{Float64}(values)
            else
                message = "non-finite transferred field"
            end
        catch e
            message = numerical_exception(e)
        end
        predictions[n],messages[n],calls[n] = prediction,message,counter[]
    end
    projected = Dict(n=>v === nothing ? nothing : rd_restrict(v,n,base)
                     for (n,v) in predictions)
    truth = Dict(n=>rd_restrict(references[n],n,base) for n in cells)
    distance(a,b) = a === nothing || b === nothing ? Inf : rmse(a,b)
    [(; case,profile,cells=n,fit_cells=base,finest_cells=finest,
        status=predictions[n] === nothing ? "test_failed" : "ok",
        message=messages[n],rhs_calls=calls[n],
        field_rmse=distance(predictions[n],references[n]),
        projected_model_rmse=distance(projected[n],truth[n]),
        projected_to_finest_truth_rmse=distance(projected[n],truth[finest]),
        truth_mesh_rmse=rmse(truth[n],truth[finest]),
        learned_mesh_shift_rmse=distance(projected[n],projected[base]),
        learned_to_finest_mesh_rmse=distance(projected[n],projected[finest])) for n in cells]
end

function rd_transfer_seeds(rows)
    keys = sort!(unique((r.track,r.case,r.model,r.seed,r.cells) for r in rows))
    result = NamedTuple[]
    for (track,case,model,seed,cells) in keys
        group = filter(r -> (r.track,r.case,r.model,r.seed,r.cells) == (track,case,model,seed,cells),rows)
        names = getproperty.(group,:profile)
        allunique(names) && all(in(RD_TEST_PROFILES),names) ||
            error("duplicate or unknown profiles in mesh-transfer results")
        complete = length(group) == length(RD_TEST_PROFILES)
        aggregate(metric) = complete && all(r -> isfinite(getproperty(r,metric)),group) ?
            sqrt(mean(abs2,getproperty.(group,metric))) : Inf
        metrics = NamedTuple{RD_TRANSFER_METRICS}(Tuple(aggregate(m) for m in RD_TRANSFER_METRICS))
        push!(result,merge((; track,case,model,seed,cells,
            finite_fields=count(r -> r.status == "ok" && isfinite(r.field_rmse),group)),metrics))
    end
    result
end

function rd_transfer_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    opts = rd_transfer_options(args)
    archives = rd_load_archive.(opts.inputs)
    all(a -> a.meta["cells"] == first(opts.cells),archives) ||
        error("the first transfer mesh must equal every archive's fitting mesh")
    allunique(a.meta["track"] for a in archives) || error("only one archive per fitting track is allowed")
    BLAS.set_num_threads(1)
    scripts = [@__FILE__,joinpath(@__DIR__,"rd_archive.jl"),joinpath(@__DIR__,"reaction_diffusion.jl")]
    initial_source = source_hash()
    hashes = rd_followon_snapshot(opts.output,scripts)
    cases = unique(vcat([a.meta["cases"] for a in archives]...))
    times = collect(0.0:0.1:4.0)
    truth = Dict((case,profile)=>Dict(n=>rd_reference(case,[profile],rd_grid(n),times)
                                     for n in opts.cells)
                 for case in cases, profile in RD_TEST_PROFILES)
    rows,parity = NamedTuple[],NamedTuple[]
    for archive in archives, case in archive.meta["cases"], seed in archive.meta["seeds"]
        ds = rd_dataset(case,seed,first(opts.cells))
        for model in archive.meta["models"]
            saved = archive.coefficients["$(case)__$(model)__$(seed)"]
            a = rd_approximator(model,saved["model_seed"])
            beta = saved["parameters"]
            before = copy(beta)
            append!(parity,rd_replay_parity(model,a,beta,ds,archive))
            common = (; track=archive.meta["track"],model,seed,candidate_id=saved["candidate_id"])
            for profile in RD_TEST_PROFILES
                append!(rows,[merge(common,row) for row in rd_transfer_profile(
                    a,beta,case,profile,opts.cells,times,truth[(case,profile)])])
            end
            beta == before || error("mesh transfer mutated fitted parameters")
            write_csv(joinpath(opts.output,"test_profiles.csv"),rows)
            write_csv(joinpath(opts.output,"replay_parity.csv"),parity)
            println("mesh transfer $(common.track) $case seed=$seed model=$model cells=$(opts.cells)")
            flush(stdout)
        end
    end
    seeds = rd_transfer_seeds(rows)
    write_csv(joinpath(opts.output,"seed_metrics.csv"),seeds)
    meta = rd_followon_metadata(opts.output,scripts,hashes,initial_source,archives)
    merge!(meta,Dict("optimizer_refit"=>false,"cells"=>opts.cells,"cases"=>cases,
        "reference"=>"truth solved separately on each mesh; finest is discrete, not continuum",
        "projection"=>"conservative cell averages onto the original fitting mesh",
        "coefficient_policy"=>"unchanged original validation-selected coefficients and model_seed"))
    rd_write_toml(joinpath(opts.output,"metadata.toml"),meta)
    (; opts,rows,seeds,parity)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rd_transfer_main(ARGS)
end
