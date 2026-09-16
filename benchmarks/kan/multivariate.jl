include("compare.jl")

const MV_MODELS = ["tensor64", "single_index9", "mlp65", "kan_additive64", "kan63"]
const MV_CASES = ["single_index", "interaction"]
const MV_DOMAINS = ((0.0, 3.0), (0.0, 2.5))
const MV_SCALES = [3.0, 2.5]
const MV_TRAIN_ICS = [(0.4, 0.3), (1.5, 0.6), (2.6, 1.8)]
const MV_VALIDATION_ICS = [(1.1, 0.9)]
const MV_TEST_ICS = [(0.8, 1.5), (2.7, 0.25)]
const MV_INITIAL_ATTACK = 0.3
const MV_SUPPORT_RADIUS = 0.05
const MV_CONSTANT_START = (name="constant", loading=1.0, slope=0.0)
const MV_INDEX_STARTS = [MV_CONSTANT_START;
    [(name="index_$(loading)_$(slope)", loading=loading, slope=slope)
     for loading in (-2.0, -1.0, 0.0, 1.0, 2.0) for slope in (-0.04, 0.04)]]

function mv_split(name)
    name in ("forecast", "local") || throw(ArgumentError("split must be forecast or local"))
    (; train_ics=copy(MV_TRAIN_ICS),
       validation_ics=name == "forecast" ? copy(MV_VALIDATION_ICS) : [(1.45, 0.65)],
       test_ics=name == "forecast" ? copy(MV_TEST_ICS) : [(0.45, 0.33), (2.5, 1.7)],
       test_end=name == "forecast" ? 8.0 : 6.0)
end

function mv_initializations(model, strategy)
    model in MV_MODELS || throw(ArgumentError("unknown multivariate model $model"))
    strategy in ("constant", "index-multistart") ||
        throw(ArgumentError("starts must be constant or index-multistart"))
    strategy == "constant" && return [MV_CONSTANT_START]
    model == "tensor64" && return [MV_CONSTANT_START]
    model == "single_index9" && return copy(MV_INDEX_STARTS)
    throw(ArgumentError("index-multistart supports only tensor64 and single_index9"))
end

function true_attack(case, N, P)
    case == "single_index" && return 0.2 + 0.6 / (1 + exp(2 * (N + 0.8P - 1.8)))
    case == "interaction" && return 0.9 / ((1 + 0.6N) * (1 + 0.8P))
    throw(ArgumentError("unknown multivariate case $case"))
end

function predator_prey!(du, u, attack)
    for j in 1:2:length(u)
        N, P = u[j], u[j+1]
        consumption = attack(N, P) * N * P
        du[j] = N * (1-N/3) - consumption
        du[j+1] = 0.6consumption - 0.25P
    end
    nothing
end

mv_u0(ics) = reduce(vcat, collect.(ics))

function mv_truth(case, ics, times)
    attack = (N, P) -> true_attack(case, N, P)
    rhs! = (du, u, p, t) -> predator_prey!(du, u, attack)
    ode = ODEProblem(rhs!, mv_u0(ics), (0.0, maximum(times)))
    sol = PSM.OrdinaryDiffEq.solve(ode, Tsit5();
        saveat=times, abstol=1e-12, reltol=1e-12)
    PSM.SciMLBase.successful_retcode(sol) || error("multivariate reference ODE failed")
    permutedims(reduce(hcat, [sol(t) for t in times]))
end

function state_points(values)
    reduce(vcat, [values[:, j:j+1] for j in 1:2:size(values, 2)])
end

function mv_dataset(case, seed; split="forecast")
    geometry = mv_split(split)
    train_ics, validation_ics, test_ics = geometry.train_ics, geometry.validation_ics, geometry.test_ics
    train_times = collect(0.0:0.3:6.0)
    validation_times = collect(0.0:0.3:6.0)
    test_times = collect(0.0:0.1:geometry.test_end)
    train_truth = mv_truth(case, train_ics, train_times)
    validation_truth = mv_truth(case, validation_ics, validation_times)
    test_truth = [mv_truth(case, [ic], test_times) for ic in test_ics]
    rng = StableRNG(50_000seed + (case == "single_index" ? 1 : 2))
    sigma_relative = 0.01
    noise(values) = randn(rng, size(values)) .*
        reshape(repeat(MV_SCALES, div(size(values, 2), 2)), 1, :) .* sigma_relative
    train_values = train_truth + noise(train_truth)
    validation_values = validation_truth + noise(validation_truth)
    (; case, seed, split, train_ics, validation_ics, test_ics,
       train_times, validation_times, test_times,
       train_truth, validation_truth, test_truth, train_values, validation_values,
       training_points=state_points(train_truth), sigma_relative)
end

function mv_index_stats(train_values)
    points = state_points(train_values)
    vec(mean(points; dims=1)), cov(points; dims=1, corrected=false)
end

# A genuine two-input Dense MLP, not the unary NeuralApproximator path
# that extracts the first element of an array argument.
struct BenchmarkMLP2 <: AbstractApproximator
    name::Symbol
    model::Any
    axes::Any
    beta::Vector{Float64}
    hidden::Int
end
PSM.nparams(a::BenchmarkMLP2) = length(a.beta)
PSM.initial_params(a::BenchmarkMLP2) = copy(a.beta)
PSM.penalty_matrix(a::BenchmarkMLP2) = Matrix{Float64}(I, nparams(a), nparams(a))
PSM.band_domain(::BenchmarkMLP2) = nothing
function PSM.build_evaluator(a::BenchmarkMLP2, beta)
    h = a.hidden
    W1 = reshape(view(beta, 1:2h), h, 2)
    b1 = view(beta, 2h+1:3h)
    W2 = view(beta, 3h+1:4h)
    bias = beta[4h+1]
    (N, P) -> begin
        input = [2N/3-1, 2P/2.5-1]
        dot(W2, tanh.(W1*input + b1)) + bias
    end
end

function mv_make_approximator(id, seed, train_values; start=MV_CONSTANT_START)
    start in MV_INDEX_STARTS || throw(ArgumentError("unknown multivariate initialization"))
    id == "single_index9" || start == MV_CONSTANT_START ||
        throw(ArgumentError("nonconstant starts are only supported for single_index9"))
    constant = (args...) -> MV_INITIAL_ATTACK
    a = if id == "tensor64"
        TensorBSplineApproximator(:g, MV_DOMAINS[1], MV_DOMAINS[2], 8, 8; initial=constant)
    elseif id == "single_index9"
        initial = iszero(start.slope) ? constant : z -> MV_INITIAL_ATTACK + start.slope*z
        SingleIndexApproximator(:g, 2, 8; index_stats=mv_index_stats(train_values),
            initial=initial, initial_loadings=[1.0, start.loading], xi=3.0, anchor=1)
    elseif id == "mlp65"
        model = Lux.Chain(Lux.Dense(2, 16, tanh), Lux.Dense(16, 1))
        ps, _ = Lux.f64(Lux.setup(Random.Xoshiro(seed), model))
        packed = PSM.ComponentArray(ps)
        beta = collect(packed)
        beta[49:64] .= 0
        beta[65] = MV_INITIAL_ATTACK
        BenchmarkMLP2(:g, model, PSM.getaxes(packed), beta, 16)
    elseif id in ("kan_additive64", "kan63")
        layer(ni, no, grid) = LuxKANLinear(ni, no;
            grid_size=grid, spline_order=3, standalone_spline_scale=false)
        model = id == "kan63" ? Lux.Chain(layer(2, 3, 3), layer(3, 1, 3)) :
                                layer(2, 1, 28)
        KANApproximator(:g, model; input_domains=MV_DOMAINS,
            penalty=:edge_curvature, nullspace_penalty=1e-6, rng_seed=seed)
    else
        throw(ArgumentError("unknown multivariate model $id"))
    end
    beta = initial_params(a)
    if a isa KANApproximator
        output = last(a.layers)
        beta[output.base_range] .= 0
        beta[output.spline_range] .= MV_INITIAL_ATTACK / output.input_dim
    end
    InitializedApprox(:g, a, beta, penalty_matrix(a))
end

function mv_problem(a, ics, times, values, calls)
    rhs! = function (du, u, p, t)
        calls[] += 1
        predator_prey!(du, u, p.g)
    end
    weights = repeat(reshape(1 ./ repeat(MV_SCALES, length(ics)).^2, 1, :),
                     length(times), 1)
    PSMProblem(rhs!, mv_u0(ics), (0.0, maximum(times)), [a];
        data_times=times, data_values=values, data_weights=weights,
        abstol=1e-8, reltol=1e-8, maxiters=10_000)
end

function mv_scaled_rmse(pred, reference)
    size(pred) == size(reference) || throw(DimensionMismatch("trajectory shapes differ"))
    scales = reshape(repeat(MV_SCALES, div(size(reference, 2), 2)), 1, :)
    sqrt(mean(abs2, (pred-reference) ./ scales))
end

function support_distances(points, training_points)
    [minimum(sqrt(sum(((points[i, k]-training_points[j, k])/MV_SCALES[k])^2
                      for k in 1:2)) for j in axes(training_points, 1))
     for i in axes(points, 1)]
end

function mv_geometry(ds, radius)
    rows = NamedTuple[]
    validation_truth = [ds.validation_truth[:, j:j+1] for j in 1:2:size(ds.validation_truth, 2)]
    for (kind, ics, truths) in (("validation", ds.validation_ics, validation_truth),
                                ("test", ds.test_ics, ds.test_truth))
        for (id, (ic, points)) in enumerate(zip(ics, truths))
            distances = support_distances(points, ds.training_points)
            supported = count(<=(radius), distances)
            push!(rows, (; case=ds.case, split=ds.split, kind, trajectory_id=id,
                         initial_N=ic[1], initial_P=ic[2], samples=length(distances),
                         supported_samples=supported, support_fraction=supported/length(distances),
                         median_support_distance=median(distances),
                         maximum_support_distance=maximum(distances), support_radius=radius))
        end
    end
    rows
end

function mv_algorithm(track, iterations, lr, penalty_weight)
    track == "adam" && return AdamSolver(maxiters=iterations, lr=lr, penalty_weight=penalty_weight)
    track == "laml" && return LAML(maxiters=iterations, jac=:forwarddiff)
    track == "gcv" && return GCVSolver(maxiters=iterations, jac=:forwarddiff, search=:reuse)
    throw(ArgumentError("unknown solver track $track"))
end

function mv_diagnostics(fit)
    c = fit === nothing ? NamedTuple() : fit.convergence
    (; converged=get(c, :converged, missing),
       stationarity=get(c, :stationarity, missing),
       smoothing_advanced=get(c, :smoothing_advanced, missing),
       laml_failures=get(c, :laml_failures, missing),
       criterion=get(c, :criterion, missing),
       criterion_value=get(c, :laml, get(c, :gcv, missing)),
       edf=fit === nothing ? missing : fit.edf)
end

mv_better_candidate(row, best) =
    row.status == "ok" && isfinite(row.validation_rmse) &&
    (best === nothing || row.validation_rmse < best.validation_rmse)

function mv_train_candidate(a, ds, eta, cfg)
    calls = Ref(0)
    prob = mv_problem(a, ds.train_ics, ds.train_times, ds.train_values, calls)
    penalty_weight = cfg.track == "adam" ? eta/max(tr(a.S), eps(Float64)) : 0.0
    alg = mv_algorithm(cfg.track, cfg.iterations, cfg.lr, penalty_weight)
    fit = nothing
    status, message, reason = "ok", "", "not_started"
    seconds = bytes = gc_seconds = NaN
    train_rmse = val_rmse = Inf
    iterations = 0
    started = time_ns()
    try
        measured = @timed solve(prob, alg)
        fit = measured.value
        seconds, bytes, gc_seconds = measured.time, Float64(measured.bytes), measured.gctime
        iterations = fit.convergence.iterations
        reason = string(fit.convergence.reason)
        train_rmse = sqrt(fit.data_loss/length(ds.train_values))
        if !isfinite(train_rmse) || !all(isfinite, fit.parameters)
            status, message = "fit_failed", "non-finite fitted values or parameters"
        end
    catch e
        seconds = (time_ns()-started)/1e9
        status, message = "fit_failed", numerical_exception(e)
    end
    training_calls = calls[]
    if status == "ok"
        try
            prob_v = mv_problem(a, ds.validation_ics, ds.validation_times,
                                ds.validation_values, Ref(0))
            pred = simulate(prob_v, collect(fit.parameters))
            val_rmse = mv_scaled_rmse(pred, ds.validation_values)
            isfinite(val_rmse) ||
                (status="validation_failed"; message="non-finite validation predictions")
        catch e
            status, message = "validation_failed", numerical_exception(e)
        end
    end
    row = (; eta, penalty_weight, status, message, training_rmse=train_rmse,
            validation_rmse=val_rmse, fit_seconds=seconds, allocated_bytes=bytes,
            gc_seconds, rhs_calls=training_calls, iterations, reason)
    fit, merge(row, mv_diagnostics(fit))
end

function mv_score_trajectory(a, beta, ds, case, trajectory_id, radius)
    reference = ds.test_truth[trajectory_id]
    points = reference
    distances = support_distances(points, ds.training_points)
    keep = findall(<=(radius), distances)
    f = build_evaluator(a, beta)
    full_rmse = near_rmse = response_rmse = near_response_rmse = Inf
    negative_fraction = NaN
    function_status, function_message = "ok", ""
    try
        predicted = f.(points[:, 1], points[:, 2])
        actual = true_attack.(Ref(case), points[:, 1], points[:, 2])
        response_rmse = rmse(predicted, actual)
        near_response_rmse = isempty(keep) ? NaN : rmse(predicted[keep], actual[keep])
        negative_fraction = count(<(0), predicted) / length(predicted)
        all(isfinite, predicted) ||
            (function_status="response_failed"; function_message="non-finite response predictions")
    catch e
        function_status, function_message = "response_failed", numerical_exception(e)
    end
    calls = Ref(0)
    test_status, test_message = "ok", ""
    try
        p = mv_problem(a, [ds.test_ics[trajectory_id]], ds.test_times, reference, calls)
        predicted = simulate(p, beta)
        full_rmse = mv_scaled_rmse(predicted, reference)
        near_rmse = isempty(keep) ? NaN : mv_scaled_rmse(predicted[keep, :], reference[keep, :])
        isfinite(full_rmse) ||
            (test_status="test_failed"; test_message="non-finite test predictions")
    catch e
        test_status, test_message = "test_failed", numerical_exception(e)
    end
    (; trajectory_id, initial_N=ds.test_ics[trajectory_id][1],
       initial_P=ds.test_ics[trajectory_id][2], test_status, test_message,
       full_rmse, near_rmse, response_rmse, near_response_rmse,
       function_status, function_message, negative_fraction,
       support_fraction=length(keep)/length(distances),
       median_support_distance=median(distances), test_rhs_calls=calls[])
end

function mv_seed_metrics(trajectories, cfg)
    rows = NamedTuple[]
    ntest = length(mv_split(cfg.split).test_ics)
    for case in cfg.cases, model in cfg.models, seed in cfg.seeds
        scored = filter(r -> r.case == case && r.model == model && r.seed == seed, trajectories)
        ids = getproperty.(scored, :trajectory_id)
        allunique(ids) && all(in(1:ntest), ids) ||
            throw(ArgumentError("duplicate or unknown test trajectory for $case/$model/$seed"))
        nt = count(r -> r.test_status == "ok" && isfinite(r.full_rmse), scored)
        nr = count(r -> r.function_status == "ok" && isfinite(r.response_rmse), scored)
        full_rmse = nt == ntest ? sqrt(mean(abs2, getproperty.(scored, :full_rmse))) : Inf
        response_rmse = nr == ntest ? sqrt(mean(abs2, getproperty.(scored, :response_rmse))) : Inf
        supported = filter(r -> r.support_fraction > 0, scored)
        # All trajectories have the same number of times. Coverage weights
        # therefore pool squared errors by the number of supported points.
        near_response_rmse = isempty(supported) ? NaN :
            nr == ntest ? sqrt(sum(r.support_fraction * r.near_response_rmse^2 for r in supported) /
                              sum(r.support_fraction for r in supported)) : Inf
        support_fraction = length(scored) == ntest ? mean(getproperty.(scored, :support_fraction)) : NaN
        negative_fraction = nr == ntest ? mean(getproperty.(scored, :negative_fraction)) : NaN
        push!(rows, (; case, model, seed, track=cfg.track, split=cfg.split,
                     test_trajectories=nt, response_trajectories=nr,
                     full_rmse, response_rmse, near_response_rmse,
                     support_fraction, negative_fraction))
    end
    rows
end

function mv_options(args)
    opts = Dict(
        "seeds" => join(101:110, ","), "iterations" => "250", "lr" => "0.03",
        "etas" => "0,0.1,10,1000", "cases" => join(MV_CASES, ","),
        "models" => join(MV_MODELS, ","), "track" => "adam",
        "split" => "forecast", "starts" => "constant",
        "support-radius" => string(MV_SUPPORT_RADIUS),
        "output" => joinpath(@__DIR__, "results", "multivariate"))
    provided = Set{String}()
    for arg in args
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown multivariate option $key"))
        opts[key] = value
        push!(provided, key)
    end
    track = opts["track"]
    track in ("adam", "laml", "gcv") || throw(ArgumentError("track must be adam, laml, or gcv"))
    split_name = opts["split"]
    mv_split(split_name)
    split_name == "local" && !("seeds" in provided) && (opts["seeds"] = join(201:210, ","))
    if track != "adam"
        "etas" in provided && throw(ArgumentError("native LAML/GCV selects smoothing internally; do not supply --etas"))
        "iterations" in provided || (opts["iterations"] = "25")
    end
    seeds = parse.(Int, split(opts["seeds"], ','))
    iterations, lr = parse(Int, opts["iterations"]), parse(Float64, opts["lr"])
    etas = track == "adam" ? parse.(Float64, split(opts["etas"], ',')) : [NaN]
    radius = parse(Float64, opts["support-radius"])
    models, cases = String.(split(opts["models"], ',')), String.(split(opts["cases"], ','))
    all(in(MV_MODELS), models) && all(in(MV_CASES), cases) ||
        throw(ArgumentError("unknown multivariate model or case"))
    allunique(seeds) && allunique(models) && allunique(cases) ||
        throw(ArgumentError("seeds, models and cases must not repeat"))
    iterations > 0 && isfinite(lr) && lr > 0 && isfinite(radius) && radius > 0 ||
        throw(ArgumentError("iterations, learning rate and support radius must be positive"))
    track != "adam" || all(x -> isfinite(x) && x >= 0, etas) ||
        throw(ArgumentError("etas must be finite and nonnegative"))
    starts = opts["starts"]
    track == "adam" && starts != "constant" &&
        throw(ArgumentError("index-multistart is a separate native LAML/GCV track, not a shared-Adam option"))
    for model in models
        mv_initializations(model, starts)
    end
    (; seeds, iterations, lr, etas, cases, models, track, split=split_name, starts,
       radius, output=abspath(opts["output"]))
end

function mv_summary(path, selections, trajectories, seed_rows, cfg)
    geometry = mv_split(cfg.split)
    ntest = length(geometry.test_ics)
    finite_median(rows, key) = begin
        values = filter(isfinite, getproperty.(rows, key))
        isempty(values) ? "NA" : @sprintf("%.5g", median(values))
    end
    open(path, "w") do io
        println(io, "# Multivariate KAN assessment: $(cfg.track), $(cfg.split) split\n")
        println(io, "Seeds: $(cfg.seeds). Solver ceiling: $(cfg.iterations) iterations. ",
            "Initialization strategy: $(cfg.starts). Test horizon: $(geometry.test_end). ",
            "Three training initial conditions, one separate validation initial condition, ",
            "and two held-out test initial conditions. Support radius: $(cfg.radius) in ",
            "domain-scaled Euclidean state coordinates.\n")
        println(io, "| Case | Model | Parameters | Valid selections | Finite test trajectories | Complete test seeds | Complete response seeds | Median trajectory NRMSE | Median response RMSE | Near-support response RMSE | Median fit seconds | Median tuning seconds |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case in cfg.cases, model in cfg.models
            selected = filter(r -> r.case == case && r.model == model, selections)
            scored = filter(r -> r.case == case && r.model == model, trajectories)
            nselected = count(r -> r.selection_status == "ok", selected)
            good = filter(r -> r.test_status == "ok", scored)
            seeds = filter(r -> r.case == case && r.model == model, seed_rows)
            nt = count(r -> r.test_trajectories == ntest, seeds)
            nr = count(r -> r.response_trajectories == ntest, seeds)
            nparams_ = first(selected).nparams
            println(io, "| $case | $model | $nparams_ | $nselected/$(length(selected)) | ",
                "$(length(good))/$(ntest*length(selected)) | ",
                "$nt/$(length(seeds)) | $nr/$(length(seeds)) | ",
                finite_median(seeds, :full_rmse), " | ", finite_median(seeds, :response_rmse), " | ",
                finite_median(seeds, :near_response_rmse), " | ",
                finite_median(selected, :fit_seconds), " | ", finite_median(selected, :tuning_seconds), " |")
        end
        println(io, "\nError medians are over complete seeds, pooling squared errors across each ",
            "seed's two test trajectories before taking the square root. Near-support errors ",
            "are weighted by supported point counts; NA means no finite value is available, ",
            "including when there are no supported points. ",
            "Read medians with success counts; two trajectories are not independent seed replicates. ",
            "Trajectory errors are normalized by the fixed state-domain spans. ",
            "Response errors are measured on actual held-out trajectories, not a rectangle. ",
            "The single-layer KAN is additive; the composed KAN can express interactions. ",
            "This track is $(cfg.track); do not conflate shared-Adam results with native ",
            "LAML/GCV results or claim globally optimal model-family performance. ",
            "Selection status only describes finite completion. Converged is an iteration-stability ",
            "flag, not fit quality; trials/selected CSVs retain stationarity and smoothing_advanced ",
            "where the solver supplies them (missing otherwise).")
        cfg.starts == "index-multistart" && println(io,
            "\nThe native single-index track selects initialization by validation error; ",
            "tensor fits retain their constant start. Tuning time includes every attempted ",
            "start, including failed candidates. This is not a matched-compute comparison.")
    end
end

function mv_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    cfg = mv_options(args)
    BLAS.set_num_threads(1)
    names = ("trials.csv", "selected.csv", "test_trajectories.csv", "coefficients.toml",
             "candidate_coefficients.toml", "seed_metrics.csv", "geometry.csv",
             "metadata.toml", "summary.md", "assessment-script.jl")
    any(isfile(joinpath(cfg.output, name)) for name in names) &&
        error("output contains previous results; choose a new --output")
    mkpath(cfg.output)
    initial_source_hash = source_hash()
    script = read(@__FILE__)
    write(joinpath(cfg.output, "assessment-script.jl"), script)
    trials, selections, trajectories = NamedTuple[], NamedTuple[], NamedTuple[]
    coefficients = Dict{String,Any}()
    candidate_coefficients = Dict{String,Any}()
    warmed = Set{String}()
    geometry_rows = reduce(vcat, [mv_geometry(mv_dataset(case, first(cfg.seeds); split=cfg.split), cfg.radius)
                                 for case in cfg.cases])
    write_csv(joinpath(cfg.output, "geometry.csv"), geometry_rows)
    for case in cfg.cases, seed in cfg.seeds
        ds = mv_dataset(case, seed; split=cfg.split)
        for model in cfg.models
            best = nothing
            total_seconds = 0.0
            for start in mv_initializations(model, cfg.starts)
                created = @timed mv_make_approximator(model, 10_000+seed, ds.train_values; start)
                a = created.value
                if !(model in warmed)
                    p = mv_problem(a, ds.train_ics, ds.train_times, ds.train_values, Ref(0))
                    solve(p, mv_algorithm(cfg.track, 2, cfg.lr, 1/max(tr(a.S), eps(Float64))))
                    push!(warmed, model)
                end
                f0 = build_evaluator(a, initial_params(a))
                points = ds.training_points
                initialization_error = rmse(f0.(points[:, 1], points[:, 2]),
                                             fill(MV_INITIAL_ATTACK, size(points, 1)))
                common = (; case, model, seed, track=cfg.track, split=cfg.split, nparams=nparams(a),
                           initialization=start.name,
                           initial_loading=model == "single_index9" ? start.loading : missing,
                           initial_slope=model == "single_index9" ? start.slope : missing,
                           setup_seconds=created.time,
                           initial_response_rmse=initialization_error)
                for (eta_id, eta) in enumerate(cfg.etas)
                    fit, row = mv_train_candidate(a, ds, eta, cfg)
                    candidate_id = "$(case)__$(model)__$(seed)__$(start.name)__$(eta_id)"
                    provenance = merge(common, (; candidate_id))
                    candidate = merge(provenance, row)
                    push!(trials, candidate)
                    saved = Dict{String,Any}("model_seed" => 10_000+seed,
                        "initialization" => start.name, "eta" => eta,
                        "status" => row.status, "message" => row.message)
                    if fit !== nothing && all(isfinite, fit.parameters)
                        saved["parameters"] = collect(fit.parameters)
                        saved["smoothing_params"] = fit.smoothing_params
                    end
                    candidate_coefficients[candidate_id] = saved
                    isfinite(row.fit_seconds) && (total_seconds += row.fit_seconds)
                    if mv_better_candidate(row, best === nothing ? nothing : best.row)
                        best = (; approx=a, fit, row=candidate, common=provenance)
                    end
                    write_csv(joinpath(cfg.output, "trials.csv"), trials)
                end
            end
            if best === nothing
                push!(selections, merge(last(trials), (; selection_status="failed", tuning_seconds=total_seconds)))
            else
                push!(selections, merge(best.row, (; selection_status="ok", tuning_seconds=total_seconds)))
                beta = collect(best.fit.parameters)
                key = "$(case)__$(model)__$(seed)"
                coefficients[key] = Dict("parameters" => beta, "model_seed" => 10_000+seed,
                                         "initialization" => best.row.initialization,
                                         "candidate_id" => best.row.candidate_id,
                                         "smoothing_params" => best.fit.smoothing_params)
                for trajectory_id in eachindex(ds.test_ics)
                    score = mv_score_trajectory(best.approx, beta, ds, case, trajectory_id, cfg.radius)
                    push!(trajectories, merge(best.common, score))
                end
            end
            write_csv(joinpath(cfg.output, "selected.csv"), selections)
            isempty(trajectories) || write_csv(joinpath(cfg.output, "test_trajectories.csv"), trajectories)
            open(joinpath(cfg.output, "coefficients.toml"), "w") do io
                TOML.print(io, coefficients)
            end
            open(joinpath(cfg.output, "candidate_coefficients.toml"), "w") do io
                TOML.print(io, candidate_coefficients)
            end
            row = last(selections)
            println(case, " seed=", seed, " model=", model, " track=", cfg.track,
                    " split=", cfg.split, " initialization=", row.initialization,
                    " selection=", row.selection_status, " validation=", row.validation_rmse,
                    " fitting_s=", row.fit_seconds)
            flush(stdout)
        end
    end
    source_hash() == initial_source_hash && read(@__FILE__) == script ||
        error("source changed during the benchmark; do not use the partial results")
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    geometry = mv_split(cfg.split)
    metadata = Dict(
        "source_sha256" => initial_source_hash,
        "assessment_script_sha256" => bytes2hex(SHA.sha256(script)),
        "julia_version" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "check_bounds" => true,
        "seeds" => cfg.seeds, "models" => cfg.models, "cases" => cfg.cases,
        "track" => cfg.track, "split" => cfg.split, "starts" => cfg.starts,
        "iterations" => cfg.iterations, "learning_rate" => cfg.lr,
        "etas" => cfg.etas, "support_radius" => cfg.radius,
        "training_ics" => collect.(geometry.train_ics), "validation_ics" => collect.(geometry.validation_ics),
        "test_ics" => collect.(geometry.test_ics), "test_end" => geometry.test_end,
        "state_domain_spans" => MV_SCALES,
        "initializations" => Dict(model => [model == "single_index9" ?
                                           Dict(string(k) => v for (k, v) in pairs(start)) :
                                           Dict("name" => start.name)
                                           for start in mv_initializations(model, cfg.starts)]
                                 for model in cfg.models),
        "relative_observation_sigma" => 0.01, "blas_threads" => BLAS.get_num_threads(),
        "package_versions" => Dict(name => get(first(entries), "version", "unversioned")
                                   for (name, entries) in manifest["deps"]))
    open(joinpath(cfg.output, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
    seed_rows = mv_seed_metrics(trajectories, cfg)
    write_csv(joinpath(cfg.output, "seed_metrics.csv"), seed_rows)
    mv_summary(joinpath(cfg.output, "summary.md"), selections, trajectories, seed_rows, cfg)
    (; cfg, trials, selections, trajectories, coefficients, candidate_coefficients,
       geometry_rows, seed_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mv_main(ARGS)
end
