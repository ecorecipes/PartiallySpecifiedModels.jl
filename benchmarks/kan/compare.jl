using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using Lux, FluxKAN, StableRNGs
using LinearAlgebra, Random, Statistics, Printf, TOML, SHA

const PSM = PartiallySpecifiedModels
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MODEL_IDS = ["spline8", "spline28", "spde28", "gp28", "mlp28", "kan_shallow28", "kan28"]
const CASE_IDS = ["logistic", "nonlinear"]
const DOMAIN = (0.0, 2.0)
const INITIAL_RATE = 0.4

# All methods receive an explicit constant-response warm start. This wrapper
# also prevents Adam's NeuralApproximator-specific initialization from
# replacing the intended last-layer initialization.
struct InitializedApprox{A,S} <: AbstractApproximator
    name::Symbol
    approx::A
    beta::Vector{Float64}
    S::S
end
PSM.nparams(a::InitializedApprox) = length(a.beta)
PSM.initial_params(a::InitializedApprox) = copy(a.beta)
PSM.penalty_matrix(a::InitializedApprox) = a.S
PSM.penalty_blocks(a::InitializedApprox) = penalty_blocks(a.approx)
PSM.build_evaluator(a::InitializedApprox, beta) = build_evaluator(a.approx, beta)
PSM.band_domain(a::InitializedApprox) = band_domain(a.approx)

function make_approximator(id, seed; domain=DOMAIN, initial_rate=INITIAL_RATE)
    rate = Float64(initial_rate)
    isfinite(rate) || throw(ArgumentError("initial_rate must be finite"))
    initial = x -> rate
    a = if id == "spline8"
        BSplineApproximator(:r, domain, 8; initial=initial)
    elseif id == "spline28"
        BSplineApproximator(:r, domain, 28; initial=initial)
    elseif id == "spde28"
        SPDEApproximator(:r, domain, 28; initial=initial, range_param=0.4)
    elseif id == "gp28"
        GPApproximator(:r, domain, 28; initial=initial,
                       kernel=:matern52, lengthscale=0.4)
    elseif id == "mlp28"
        NeuralApproximator(:r, Lux.Chain(Lux.Dense(1, 9, tanh), Lux.Dense(9, 1));
                           domain=domain, penalty_weight=1.0, rng_seed=seed)
    elseif id in ("kan_shallow28", "kan28")
        layer(ni, no, grid) = LuxKANLinear(ni, no;
            grid_size=grid, spline_order=3, standalone_spline_scale=false)
        model = id == "kan28" ? Lux.Chain(layer(1, 2, 3), layer(2, 1, 3)) :
                                layer(1, 1, 24)
        KANApproximator(:r, model; input_domains=(domain,),
                        penalty=:edge_curvature, nullspace_penalty=1e-6,
                        rng_seed=seed)
    else
        throw(ArgumentError("unknown model $id"))
    end
    beta = initial_params(a)
    if id == "mlp28"
        beta[end-9:end-1] .= 0.0
        beta[end] = rate
    elseif a isa KANApproximator
        last_layer = last(a.layers)
        beta[last_layer.base_range] .= 0.0
        beta[last_layer.spline_range] .= rate / last_layer.input_dim
    end
    InitializedApprox(:r, a, beta, penalty_matrix(a))
end

true_rate(case, N) = case == "logistic" ? 0.8 * (1-N/2) :
    0.8 * (1-N/2) * (1 + 0.5sin(3pi*N/2))

function truth(case, u0, times)
    rhs!(du, u, p, t) = (du[1] = true_rate(case, u[1]) * u[1])
    ode = ODEProblem(rhs!, [u0], (0.0, maximum(times)))
    sol = PSM.OrdinaryDiffEq.solve(ode, Tsit5();
        saveat=times, abstol=1e-12, reltol=1e-12)
    PSM.SciMLBase.successful_retcode(sol) || error("reference ODE failed")
    Float64[sol(t)[1] for t in times]
end

function dataset(case, seed)
    train_times = collect(0.0:0.2:6.0)
    validation_times = collect(0.1:0.2:5.9)
    test_times = collect(0.0:0.1:8.0)
    rng = StableRNG(10_000seed + (case == "logistic" ? 1 : 2))
    train_truth = truth(case, 0.2, train_times)
    validation_truth = truth(case, 0.2, validation_times)
    test_truth = truth(case, 0.5, test_times)
    sigma = 0.02
    (; train_times, validation_times, test_times, train_truth, test_truth, sigma,
       train_values=train_truth + sigma * randn(rng, length(train_times)),
       validation_values=validation_truth + sigma * randn(rng, length(validation_times)))
end

function problem(a, u0, times, data, calls)
    rhs! = function (du, u, p, t)
        calls[] += 1
        du[1] = p.r(u[1]) * u[1]
    end
    PSMProblem(rhs!, [u0], (0.0, maximum(times)), [a];
        data_times=times, data_values=reshape(data, :, 1),
        abstol=1e-8, reltol=1e-8, maxiters=10_000)
end

rmse(a, b) = sqrt(mean(abs2, a-b))

function numerical_exception(e)
    (PSM._is_program_error(e) || e isa ArgumentError) && throw(e)
    sprint(showerror, e)
end

function train_candidate(a, ds, eta, iterations, lr)
    calls = Ref(0)
    prob = problem(a, 0.2, ds.train_times, ds.train_values, calls)
    scale = a.S === nothing ? 1.0 : max(tr(a.S), eps(Float64))
    penalty_weight = eta / scale
    alg = AdamSolver(maxiters=iterations, lr=lr, penalty_weight=penalty_weight)
    fit = nothing
    status, message = "ok", ""
    elapsed = allocated = gc_seconds = NaN
    training_rmse = validation_rmse = Inf
    used_iters = 0
    reason = "not_started"
    started = time_ns()
    try
        timed = @timed solve(prob, alg)
        fit = timed.value
        elapsed, allocated, gc_seconds = timed.time, Float64(timed.bytes), timed.gctime
        used_iters = fit.convergence.iterations
        reason = string(fit.convergence.reason)
        if !all(isfinite, fit.parameters) || !isfinite(fit.data_loss)
            status, message = "nonfinite_fit", "non-finite returned fit"
        else
            training_rmse = sqrt(fit.data_loss / length(ds.train_times))
        end
    catch e
        elapsed = (time_ns()-started) / 1e9
        status, message = "fit_failed", numerical_exception(e)
    end
    train_calls = calls[]
    if status == "ok"
        try
            val_prob = problem(a, 0.2, ds.validation_times, ds.validation_values, Ref(0))
            pred = vec(simulate(val_prob, collect(fit.parameters)))
            validation_rmse = rmse(pred, ds.validation_values)
            if !isfinite(validation_rmse)
                status, message = "validation_failed", "non-finite validation predictions"
            end
        catch e
            status, message = "validation_failed", numerical_exception(e)
        end
    end
    row = (; eta, penalty_weight, status, message, training_rmse, validation_rmse,
            fit_seconds=elapsed, allocated_bytes=allocated, gc_seconds,
            rhs_calls=train_calls, iterations=used_iters, reason)
    fit, row
end

function score_selected(a, fit, ds, case)
    status, message = "ok", ""
    function_status, function_message = "ok", ""
    support_status, support_message = "ok", ""
    trajectory_rmse = response_rmse = supported_trajectory_rmse = Inf
    lo, hi = extrema(ds.train_truth)
    keep = findall(x -> lo <= x <= hi, ds.test_truth)
    isempty(keep) && error("no common response support in the fixture")
    support_fraction = length(keep) / length(ds.test_truth)
    try
        f = fit.unknown_functions[:r]
        response_rmse = rmse(f.(ds.test_truth[keep]),
                             true_rate.(Ref(case), ds.test_truth[keep]))
        isfinite(response_rmse) ||
            (function_status="nonfinite_response"; function_message="non-finite response values")
    catch e
        function_status, function_message = "response_failed", numerical_exception(e)
    end
    eval_calls = Ref(0)
    pred = nothing
    try
        prob = problem(a, 0.5, ds.test_times, ds.test_truth, eval_calls)
        pred = vec(simulate(prob, collect(fit.parameters)))
        trajectory_rmse = rmse(pred, ds.test_truth)
        if !isfinite(trajectory_rmse)
            status, message = "test_failed", "non-finite test predictions"
        end
    catch e
        status, message = "test_failed", numerical_exception(e)
    end
    try
        if status == "ok"
            supported_trajectory_rmse = rmse(pred[keep], ds.test_truth[keep])
        else
            # A late forecast failure must not erase an otherwise valid
            # comparison over the state range actually informed by training.
            prob = problem(a, 0.5, ds.test_times[keep], ds.test_truth[keep], eval_calls)
            supported = vec(simulate(prob, collect(fit.parameters)))
            supported_trajectory_rmse = rmse(supported, ds.test_truth[keep])
        end
        isfinite(supported_trajectory_rmse) ||
            (support_status="nonfinite_support"; support_message="non-finite in-support trajectory")
    catch e
        support_status, support_message = "support_failed", numerical_exception(e)
    end
    (; test_status=status, test_message=message, trajectory_rmse,
       function_status, function_message, response_rmse,
       support_status, support_message, supported_trajectory_rmse,
       support_fraction, test_rhs_calls=eval_calls[])
end

csv_cell(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
function write_csv(path, rows)
    isempty(rows) && error("no benchmark rows to write")
    names = keys(first(rows))
    open(path, "w") do io
        println(io, join(string.(names), ","))
        for row in rows
            println(io, join(csv_cell.(values(row)), ","))
        end
    end
end

function source_hash()
    files = sort(vcat([joinpath(ROOT, "Project.toml"), @__FILE__,
                       joinpath(@__DIR__, "Project.toml")],
        [joinpath(ROOT, d, file) for d in ("src", "ext")
         for file in readdir(joinpath(ROOT, d)) if endswith(file, ".jl")]))
    digest = SHA.SHA2_256_CTX()
    for file in files
        SHA.update!(digest, codeunits(relpath(file, ROOT) * "\0"))
        SHA.update!(digest, read(file))
        SHA.update!(digest, UInt8[0])
    end
    bytes2hex(SHA.digest!(digest))
end

function options(args)
    opts = Dict(
        "seeds" => "1,2,3", "iterations" => "250", "lr" => "0.03",
        "etas" => "0,0.1,10,1000", "cases" => join(CASE_IDS, ","),
        "models" => join(MODEL_IDS, ","),
        "output" => joinpath(@__DIR__, "results", "pilot"))
    for arg in args
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown option $key"))
        opts[key] = value
    end
    seeds = parse.(Int, split(opts["seeds"], ','))
    iterations, lr = parse(Int, opts["iterations"]), parse(Float64, opts["lr"])
    etas = parse.(Float64, split(opts["etas"], ','))
    cases, models = split(opts["cases"], ','), split(opts["models"], ',')
    !isempty(seeds) && iterations > 0 && isfinite(lr) && lr > 0 ||
        throw(ArgumentError("provide seeds, positive iterations and a finite positive learning rate"))
    all(x -> isfinite(x) && x >= 0, etas) ||
        throw(ArgumentError("etas must be finite and nonnegative"))
    all(in(CASE_IDS), cases) && all(in(MODEL_IDS), models) ||
        throw(ArgumentError("unknown case or model"))
    (; seeds, iterations, lr, etas, cases, models, output=abspath(opts["output"]))
end

function summarize(path, selected, cfg)
    open(path, "w") do io
        println(io, "# KAN ODE comparison: pilot results\n")
        println(io, "Configuration: $(length(cfg.seeds)) seeds, $(cfg.iterations) maximum Adam iterations ",
            "per candidate, learning rate $(cfg.lr), eta grid $(cfg.etas). ",
            "Selection uses independent noisy validation observations; test trajectories ",
            "start at a different initial condition and extend to time 8.\n")
        println(io, "| Case | Model | Params | Full-horizon success | Median full RMSE (successful) | Median in-support trajectory RMSE | Median response RMSE | Median selected-fit seconds | Median tuning seconds |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for case in cfg.cases, model in cfg.models
            rows = filter(r -> r.case == case && r.model == model, selected)
            good = filter(r -> r.test_status == "ok", rows)
            fmt(x) = @sprintf("%.5g", x)
            function stat(field, population=rows)
                values = filter(isfinite, getproperty.(population, field))
                isempty(values) ? "NA" : fmt(median(values))
            end
            println(io, "| $case | $model | $(first(rows).nparams) | $(length(good))/$(length(rows)) | ",
                stat(:trajectory_rmse, good), " | ", stat(:supported_trajectory_rmse),
                " | ", stat(:response_rmse), " | ",
                stat(:fit_seconds), " | ", stat(:tuning_seconds), " |")
        end
        println(io, "\nThese are configuration-specific, CPU Float64 results, not a universal ",
            "KAN ranking. Response error is evaluated only on the held-out trajectory ",
            "where its states overlap the training trajectory's state range, independently ",
            "of full-horizon simulation success. In-support trajectory errors are also ",
            "retained when a later forecast fails. Failed ",
            "candidates remain in trials.csv and failed selections remain in selected.csv. ",
            "Timings exclude per-model compilation warm-up but include fit allocations; ",
            "they are not peak-memory measurements. See metadata.toml and the benchmark README.")
    end
end

function main(args)
    Base.JLOptions().check_bounds == 1 ||
        error("run this benchmark with --check-bounds=yes")
    cfg = options(args)
    BLAS.set_num_threads(1)
    any(isfile(joinpath(cfg.output, file)) for file in
        ("trials.csv", "selected.csv", "metadata.toml", "coefficients.toml", "summary.md")) &&
        error("output already contains benchmark results; choose a new --output directory")
    mkpath(cfg.output)
    trials, selected = NamedTuple[], NamedTuple[]
    coefficients = Dict{String,Any}()
    warmed = Set{String}()
    for case in cfg.cases, seed in cfg.seeds
        ds = dataset(case, seed)
        for model in cfg.models
            setup = @timed make_approximator(model, 100+seed)
            a = setup.value
            if !(model in warmed)
                warm = problem(a, 0.2, ds.train_times, ds.train_values, Ref(0))
                scale = a.S === nothing ? 1.0 : max(tr(a.S), eps(Float64))
                solve(warm, AdamSolver(maxiters=2, lr=cfg.lr, penalty_weight=1/scale))
                push!(warmed, model)
            end
            init = build_evaluator(a, initial_params(a))
            initial_error = rmse(init.(ds.train_truth), fill(INITIAL_RATE, length(ds.train_truth)))
            common = (; case=String(case), model=String(model), seed, nparams=nparams(a),
                      setup_seconds=setup.time, initial_response_rmse=initial_error)
            best_fit, best_row = nothing, nothing
            tuning_seconds = 0.0
            for eta in cfg.etas
                fit, row = train_candidate(a, ds, eta, cfg.iterations, cfg.lr)
                push!(trials, merge(common, row))
                isfinite(row.fit_seconds) && (tuning_seconds += row.fit_seconds)
                if row.status == "ok" &&
                   (best_row === nothing || row.validation_rmse < best_row.validation_rmse)
                    best_fit, best_row = fit, row
                end
                write_csv(joinpath(cfg.output, "trials.csv"), trials)
            end
            if best_row === nothing
                row = last(trials)
                score = (; test_status="no_valid_candidate", test_message=row.message,
                          trajectory_rmse=Inf, function_status="no_valid_candidate",
                          function_message=row.message, response_rmse=Inf,
                          support_status="no_valid_candidate", support_message=row.message,
                          supported_trajectory_rmse=Inf, support_fraction=0.0,
                          test_rhs_calls=0)
                push!(selected, merge(row, (; tuning_seconds), score))
            else
                score = score_selected(a, best_fit, ds, case)
                push!(selected, merge(common, best_row, (; tuning_seconds), score))
                coefficients["$(case)__$(model)__$(seed)"] = Dict(
                    "model_seed" => 100+seed, "eta" => best_row.eta,
                    "parameters" => collect(best_fit.parameters))
            end
            write_csv(joinpath(cfg.output, "selected.csv"), selected)
            open(joinpath(cfg.output, "coefficients.toml"), "w") do io
                TOML.print(io, coefficients)
            end
            last_row = last(selected)
            println(case, " seed=", seed, " model=", model, " eta=", last_row.eta,
                    " test_rmse=", last_row.trajectory_rmse,
                    " response_rmse=", last_row.response_rmse, " status=", last_row.test_status)
            flush(stdout)
        end
    end
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    versions = Dict(name => get(first(entries), "version", "unversioned")
                    for (name, entries) in manifest["deps"])
    metadata = Dict(
        "julia_version" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "blas_threads" => BLAS.get_num_threads(), "check_bounds" => true,
        "source_sha256" => source_hash(),
        "seeds" => cfg.seeds, "iterations" => cfg.iterations, "learning_rate" => cfg.lr,
        "etas" => cfg.etas, "cases" => cfg.cases, "models" => cfg.models,
        "training_initial_state" => 0.2, "test_initial_state" => 0.5,
        "observation_sigma" => 0.02,
        "package_versions" => versions)
    open(joinpath(cfg.output, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
    summarize(joinpath(cfg.output, "summary.md"), selected, cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
