include("compare.jl")

const FD = PSM.ForwardDiff

function evaluation_loop(f, xs, repetitions, differentiated)
    total = 0.0
    for i in 1:repetitions
        x = xs[mod1(i, length(xs))]
        total += differentiated ? FD.derivative(f, x) : f(x)
    end
    total
end

function timed_evaluations(f, xs, repetitions, differentiated)
    evaluation_loop(f, xs, 10, differentiated)
    seconds, bytes = Float64[], Float64[]
    for _ in 1:5
        GC.gc()
        measured = @timed evaluation_loop(f, xs, repetitions, differentiated)
        isfinite(measured.value) || error("non-finite scalar evaluation checksum")
        push!(seconds, measured.time / repetitions)
        push!(bytes, measured.bytes / repetitions)
    end
    median(seconds), median(bytes)
end

function profile_options(args)
    opts = Dict(
        "weights" => joinpath(@__DIR__, "results", "pilot-1000", "coefficients.toml"),
        "output" => joinpath(@__DIR__, "results", "profile"),
        "models" => "kan_shallow28,kan28,mlp28",
        "repetitions" => "10000", "reference" => "")
    for arg in args
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown profile option $key"))
        opts[key] = value
    end
    repetitions = parse(Int, opts["repetitions"])
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    models = split(opts["models"], ',')
    all(in(MODEL_IDS), models) || throw(ArgumentError("unknown profile model"))
    (; weights=abspath(opts["weights"]), output=abspath(opts["output"]),
       models, repetitions, reference=opts["reference"])
end

function profile_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    cfg = profile_options(args)
    BLAS.set_num_threads(1)
    saved = TOML.parsefile(cfg.weights)
    references = isempty(cfg.reference) ? nothing : TOML.parsefile(cfg.reference)
    any(isfile(joinpath(cfg.output, file)) for file in ("profile.csv", "oracle.toml", "metadata.toml")) &&
        error("profile output exists; use a new --output directory")
    mkpath(cfg.output)
    rows = NamedTuple[]
    oracles = Dict{String,Any}()
    for key in sort(collect(keys(saved)))
        case, model, seed_text = split(key, "__")
        model in cfg.models || continue
        seed = parse(Int, seed_text)
        a = make_approximator(model, saved[key]["model_seed"])
        beta = Float64.(saved[key]["parameters"])
        f = build_evaluator(a, beta)
        ds = dataset(case, seed)
        xs = ds.train_truth
        values = f.(xs)
        dx = [FD.derivative(f, x) for x in xs]
        jp = FD.jacobian(b -> [build_evaluator(a, b)(x) for x in xs], beta)
        p = problem(a, 0.5, ds.test_times, ds.test_truth, Ref(0))
        simulate(p, beta)
        GC.gc()
        trajectory = @timed simulate(p, beta)
        pred = vec(trajectory.value)
        scalar_seconds, scalar_bytes = timed_evaluations(f, xs, cfg.repetitions, false)
        derivative_seconds, derivative_bytes = timed_evaluations(f, xs, cfg.repetitions, true)
        value_error = input_gradient_error = parameter_gradient_error = trajectory_error = 0.0
        if references !== nothing
            reference = references[key]
            reference["parameters"] == beta && reference["inputs"] == xs ||
                error("profile reference uses different weights or inputs for $key")
            value_error = maximum(abs, values-reference["values"])
            input_gradient_error = maximum(abs, dx-reference["input_gradients"])
            parameter_gradient_error = maximum(abs, vec(jp)-reference["parameter_jacobian"])
            trajectory_error = maximum(abs, pred-reference["trajectory"])
        end
        oracles[key] = Dict(
            "parameters" => beta, "inputs" => xs, "values" => values,
            "input_gradients" => dx, "parameter_jacobian" => vec(jp),
            "trajectory" => pred)
        row = (; case=String(case), model=String(model), seed, nparams=nparams(a),
               scalar_seconds, scalar_bytes, derivative_seconds, derivative_bytes,
               trajectory_seconds=trajectory.time, trajectory_bytes=Float64(trajectory.bytes),
               value_error, input_gradient_error, parameter_gradient_error, trajectory_error)
        push!(rows, row)
        write_csv(joinpath(cfg.output, "profile.csv"), rows)
        println(key, " scalar_bytes=", scalar_bytes, " derivative_bytes=", derivative_bytes,
                " ns=", 1e9scalar_seconds, " value_error=", value_error,
                " trajectory_error=", trajectory_error)
        flush(stdout)
    end
    isempty(rows) && error("no selected coefficients match the profile models")
    open(joinpath(cfg.output, "oracle.toml"), "w") do io
        TOML.print(io, oracles)
    end
    metadata = Dict(
        "source_sha256" => source_hash(),
        "profile_script_sha256" => bytes2hex(SHA.sha256(read(@__FILE__))),
        "weights_sha256" => bytes2hex(SHA.sha256(read(cfg.weights))),
        "julia_version" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "blas_threads" => BLAS.get_num_threads(), "repetitions" => cfg.repetitions,
        "models" => cfg.models, "reference" => cfg.reference)
    open(joinpath(cfg.output, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    profile_main(ARGS)
end
