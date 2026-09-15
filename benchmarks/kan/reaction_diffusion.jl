include("compare.jl")

const RD_CASES = ["logistic", "crowding"]
const RD_MODELS = ["spline8", "spline28", "mlp28", "kan_shallow28", "kan28"]
const RD_TRAIN_PROFILES = ["front", "patches"]
const RD_VALIDATION_PROFILES = ["mixed"]
const RD_TEST_PROFILES = ["reverse_front", "fine_scale"]
const RD_INITIAL_RATE = 0.8
const RD_DENSITY_DOMAIN = (0.0, 1.0)

function rd_grid(cells::Int; length=6.0, diffusion=0.1)
    cells >= 2 || throw(ArgumentError("at least two spatial cells are required"))
    isfinite(length) && length > 0 && isfinite(diffusion) && diffusion >= 0 ||
        throw(ArgumentError("length must be positive and diffusion nonnegative and finite"))
    dx = Float64(length)/cells
    (; cells, length=Float64(length), diffusion=Float64(diffusion), dx,
       centres=[(i-0.5)*dx for i in 1:cells])
end

function rd_profile(name)
    name == "front" && return (mean=0.4, modes=((1, 0.35),))
    name == "patches" && return (mean=0.4, modes=((2, 0.22), (4, -0.12)))
    name == "mixed" && return (mean=0.45, modes=((1, -0.2), (3, 0.12)))
    name == "reverse_front" && return (mean=0.5, modes=((1, -0.4),))
    name == "fine_scale" && return (mean=0.35, modes=((5, 0.25),))
    throw(ArgumentError("unknown spatial profile $name"))
end

function rd_initial(name, grid)
    profile = rd_profile(name)
    # Exact cell averages of the cosine profiles, not centre samples.
    [profile.mean + sum(amplitude*sinc(mode/(2grid.cells))*cospi(mode*(i-0.5)/grid.cells)
                        for (mode, amplitude) in profile.modes)
     for i in 1:grid.cells]
end

rd_u0(profiles, grid) = reduce(vcat, [rd_initial(name, grid) for name in profiles])

function rd_rate(case, u)
    case == "logistic" && return one(u)
    case == "crowding" && return 0.4 + 1.2/(1+9u^2)
    case == "diffusion" && return zero(u)
    throw(ArgumentError("unknown reaction case $case"))
end

function rd_rhs!(du, u, rate, grid)
    length(du) == length(u) && length(u) > 0 && length(u) % grid.cells == 0 ||
        throw(DimensionMismatch("spatial states must contain complete, equally sized profiles"))
    fill!(du, 0)
    scale = grid.diffusion/grid.dx^2
    for offset in 0:grid.cells:length(u)-1
        # Every interior face transfers equal and opposite mass. Boundary
        # fluxes are zero, and no face connects separate initial profiles.
        for j in 1:grid.cells-1
            i = offset+j
            flux = scale*(u[i+1]-u[i])
            du[i] += flux
            du[i+1] -= flux
        end
        for j in 1:grid.cells
            i = offset+j
            du[i] += u[i]*(1-u[i])*rate(u[i])
        end
    end
    nothing
end

function rd_reference(case, profiles, grid, times)
    rhs! = (du, u, p, t) -> rd_rhs!(du, u, x -> rd_rate(case, x), grid)
    ode = ODEProblem(rhs!, rd_u0(profiles, grid), (0.0, maximum(times)))
    sol = PSM.OrdinaryDiffEq.solve(ode, Tsit5(); saveat=times, abstol=1e-11, reltol=1e-11)
    PSM.SciMLBase.successful_retcode(sol) || error("reaction-diffusion reference solve failed")
    permutedims(reduce(hcat, [sol(t) for t in times]))
end

function rd_diffusion_exact(profile_name, grid, times; continuum=false)
    profile = rd_profile(profile_name)
    [profile.mean + sum(amplitude*sinc(mode/(2grid.cells))*cospi(mode*(i-0.5)/grid.cells) *
        exp(-grid.diffusion*(continuum ? (mode*pi/grid.length)^2 :
             4sinpi(mode/(2grid.cells))^2/grid.dx^2)*t)
        for (mode, amplitude) in profile.modes)
     for t in times, i in 1:grid.cells]
end

function rd_restrict(values, fine_cells, coarse_cells)
    fine_cells >= coarse_cells >= 2 && fine_cells % coarse_cells == 0 ||
        throw(ArgumentError("restriction requires an integer spatial refinement factor"))
    size(values, 2) % fine_cells == 0 ||
        throw(DimensionMismatch("fine fields do not contain complete profiles"))
    profiles, factor = div(size(values, 2), fine_cells), div(fine_cells, coarse_cells)
    result = Matrix{Float64}(undef, size(values, 1), profiles*coarse_cells)
    for p in 1:profiles, i in 1:coarse_cells
        columns = (p-1)*fine_cells+(i-1)*factor+1:(p-1)*fine_cells+i*factor
        result[:, (p-1)*coarse_cells+i] .= vec(mean(@view(values[:, columns]); dims=2))
    end
    result
end

function rd_dataset(case, seed, cells)
    grid = rd_grid(cells)
    train_times = collect(0.0:0.25:2.5)
    validation_times = copy(train_times)
    test_times = collect(0.0:0.1:4.0)
    train_truth = rd_reference(case, RD_TRAIN_PROFILES, grid, train_times)
    validation_truth = rd_reference(case, RD_VALIDATION_PROFILES, grid, validation_times)
    test_truth = [rd_reference(case, [name], grid, test_times) for name in RD_TEST_PROFILES]
    rng = StableRNG(70_000seed + (case == "logistic" ? 1 : 2))
    sigma = 0.01
    train_values = train_truth + sigma*randn(rng, size(train_truth))
    validation_values = validation_truth + sigma*randn(rng, size(validation_truth))
    (; case, seed, grid, train_times, validation_times, test_times, train_truth,
       validation_truth, test_truth, train_values, validation_values, sigma)
end

rd_approximator(model, seed) =
    make_approximator(model, seed; domain=RD_DENSITY_DOMAIN, initial_rate=RD_INITIAL_RATE)

function rd_problem(a, profiles, grid, times, values, calls)
    size(values) == (length(times), grid.cells*length(profiles)) ||
        throw(DimensionMismatch("observations must contain every cell of each spatial profile"))
    rhs! = (du, u, p, t) -> begin
        calls[] += 1
        rd_rhs!(du, u, p.r, grid)
    end
    weights = fill(1/(grid.cells*length(profiles)), size(values))
    PSMProblem(rhs!, rd_u0(profiles, grid), (0.0, maximum(times)), [a];
        data_times=times, data_values=values, data_weights=weights,
        abstol=1e-8, reltol=1e-8, maxiters=10_000)
end

function rd_train(a, ds, eta, cfg)
    calls = Ref(0)
    prob = rd_problem(a, RD_TRAIN_PROFILES, ds.grid, ds.train_times, ds.train_values, calls)
    weight = cfg.track == "adam" ? eta/max(tr(a.S), eps(Float64)) : 0.0
    alg = cfg.track == "adam" ? AdamSolver(maxiters=cfg.iterations, lr=cfg.lr, penalty_weight=weight,
                               plateau_tol=get(cfg,:plateau_tol,1e-4),
                               plateau_window=get(cfg,:plateau_window,30),
                               early_stopping=get(cfg,:early_stopping,true)) :
                               LAML(maxiters=cfg.iterations, jac=:forwarddiff)
    fit = nothing
    status, message, reason = "ok", "", "not_started"
    elapsed = allocated = gc_seconds = NaN
    training_rmse = validation_rmse = Inf
    used_iters = 0
    started = time_ns()
    try
        measured = @timed solve(prob, alg)
        fit = measured.value
        elapsed, allocated, gc_seconds = measured.time, Float64(measured.bytes), measured.gctime
        used_iters, reason = fit.convergence.iterations, string(fit.convergence.reason)
        training_rmse = sqrt(fit.data_loss/length(ds.train_times))
        if !all(isfinite, fit.parameters) || !isfinite(training_rmse)
            status, message = "fit_failed", "non-finite fitted values or parameters"
        end
    catch e
        elapsed = (time_ns()-started)/1e9
        status, message = "fit_failed", numerical_exception(e)
    end
    training_calls = calls[]
    if status == "ok"
        try
            p = rd_problem(a, RD_VALIDATION_PROFILES, ds.grid, ds.validation_times,
                           ds.validation_values, Ref(0))
            validation_rmse = rmse(simulate(p, collect(fit.parameters)), ds.validation_values)
            isfinite(validation_rmse) ||
                (status="validation_failed"; message="non-finite validation predictions")
        catch e
            status, message = "validation_failed", numerical_exception(e)
        end
    end
    c = fit === nothing ? NamedTuple() : fit.convergence
    row = (; eta, penalty_weight=weight, status, message, training_rmse, validation_rmse,
        fit_seconds=elapsed, allocated_bytes=allocated, gc_seconds, rhs_calls=training_calls,
        iterations=used_iters, reason, converged=get(c, :converged, missing),
        stationarity=get(c, :stationarity, missing),
        smoothing_advanced=get(c, :smoothing_advanced, missing),
        laml=get(c, :laml, missing), edf=fit === nothing ? missing : fit.edf,
        plateau_tol=cfg.track == "adam" ? alg.plateau_tol : missing,
        plateau_window=cfg.track == "adam" ? alg.plateau_window : missing,
        early_stopping=cfg.track == "adam" ? alg.early_stopping : missing)
    fit, row
end

function rd_score(a, beta, ds, profile_id)
    reference = ds.test_truth[profile_id]
    density = vec(reference)
    lo, hi = extrema(ds.train_truth)
    keep = findall(x -> lo <= x <= hi, density)
    f = build_evaluator(a, beta)
    field_rmse = coefficient_rmse = reaction_rmse = supported_reaction_rmse = Inf
    negative_rate_fraction = NaN
    response_status, response_message = "ok", ""
    try
        estimated = f.(density)
        actual = rd_rate.(Ref(ds.case), density)
        factor = density .* (1 .- density)
        coefficient_rmse = rmse(estimated, actual)
        reaction_rmse = rmse(factor .* estimated, factor .* actual)
        supported_reaction_rmse = isempty(keep) ? NaN :
            rmse(factor[keep] .* estimated[keep], factor[keep] .* actual[keep])
        negative_rate_fraction = count(<(0), estimated)/length(estimated)
        all(isfinite, estimated) ||
            (response_status="response_failed"; response_message="non-finite reaction coefficient")
    catch e
        response_status, response_message = "response_failed", numerical_exception(e)
    end
    test_status, test_message = "ok", ""
    calls = Ref(0)
    try
        p = rd_problem(a, [RD_TEST_PROFILES[profile_id]], ds.grid, ds.test_times, reference, calls)
        field_rmse = rmse(simulate(p, beta), reference)
        isfinite(field_rmse) ||
            (test_status="test_failed"; test_message="non-finite spatial predictions")
    catch e
        test_status, test_message = "test_failed", numerical_exception(e)
    end
    (; profile_id, profile=RD_TEST_PROFILES[profile_id], test_status, test_message,
       response_status, response_message, field_rmse, coefficient_rmse, reaction_rmse,
       supported_reaction_rmse, support_fraction=length(keep)/length(density),
       negative_rate_fraction, test_rhs_calls=calls[])
end

function rd_mesh_errors(case, cells)
    times = collect(0.0:0.1:4.0)
    grid, fine, finer = rd_grid(cells), rd_grid(2cells), rd_grid(4cells)
    rows = NamedTuple[]
    for profile in RD_TEST_PROFILES
        coarse_values = rd_reference(case, [profile], grid, times)
        fine_values = rd_restrict(rd_reference(case, [profile], fine, times), fine.cells, cells)
        finer_values = rd_restrict(rd_reference(case, [profile], finer, times), finer.cells, cells)
        coarse_error, fine_error = rmse(coarse_values, fine_values), rmse(fine_values, finer_values)
        push!(rows, (; case, profile, cells, fine_cells=fine.cells, finer_cells=finer.cells,
                     coarse_fine_rmse=coarse_error, fine_finer_rmse=fine_error,
                     refinement_ratio=coarse_error/fine_error))
    end
    rows
end

function rd_options(args)
    opts = Dict("cells"=>"24", "seeds"=>join(301:310, ","), "cases"=>join(RD_CASES, ","),
        "models"=>join(RD_MODELS, ","), "track"=>"adam", "iterations"=>"150",
        "lr"=>"0.03", "etas"=>"0,0.1,10,1000", "plateau-tol"=>"1e-4",
        "plateau-window"=>"30", "early-stopping"=>"true",
        "output"=>joinpath(@__DIR__, "results", "reaction-diffusion-adam"))
    supplied = Set{String}()
    for arg in args
        startswith(arg, "--") && occursin('=', arg) || throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown reaction-diffusion option $key"))
        opts[key] = value
        push!(supplied, key)
    end
    track = opts["track"]
    track in ("adam", "laml") || throw(ArgumentError("track must be adam or laml"))
    if track == "laml"
        "etas" in supplied && throw(ArgumentError("LAML selects smoothing internally"))
        any(in(supplied),("plateau-tol","plateau-window","early-stopping")) &&
            throw(ArgumentError("plateau controls apply only to the Adam track"))
        "iterations" in supplied || (opts["iterations"]="25")
        "models" in supplied || (opts["models"]="spline8,spline28")
    end
    cells, iterations, lr = parse(Int, opts["cells"]), parse(Int, opts["iterations"]), parse(Float64, opts["lr"])
    rd_grid(cells)
    iterations > 0 && isfinite(lr) && lr > 0 || throw(ArgumentError("iterations and learning rate must be positive"))
    plateau_tol = parse(Float64,opts["plateau-tol"])
    plateau_window = parse(Int,opts["plateau-window"])
    early_stopping = parse(Bool,opts["early-stopping"])
    AdamSolver(; plateau_tol, plateau_window, early_stopping)
    seeds = parse.(Int, split(opts["seeds"], ','))
    cases, models = String.(split(opts["cases"], ',')), String.(split(opts["models"], ','))
    all(in(RD_CASES), cases) && all(in(RD_MODELS), models) ||
        throw(ArgumentError("unknown reaction case or approximator"))
    allunique(seeds) && allunique(cases) && allunique(models) || throw(ArgumentError("repeated seeds, cases or models"))
    track == "adam" || all(in(("spline8","spline28")), models) ||
        throw(ArgumentError("the native baseline track supports spline8 and spline28"))
    etas = track == "adam" ? parse.(Float64, split(opts["etas"], ',')) : [NaN]
    track != "adam" || all(x -> isfinite(x) && x >= 0, etas) ||
        throw(ArgumentError("etas must be finite and nonnegative"))
    (; cells, seeds, cases, models, track, iterations, lr, etas, plateau_tol,
       plateau_window, early_stopping, output=abspath(opts["output"]))
end

function rd_seed_metrics(trajectories, cfg)
    result = NamedTuple[]
    for case in cfg.cases, model in cfg.models, seed in cfg.seeds
        rows = filter(r -> r.case == case && r.model == model && r.seed == seed, trajectories)
        ids = getproperty.(rows, :profile_id)
        allunique(ids) && all(in(eachindex(RD_TEST_PROFILES)), ids) ||
            throw(ArgumentError("duplicate or unknown test profiles"))
        nf = count(r -> r.test_status == "ok" && isfinite(r.field_rmse), rows)
        nr = count(r -> r.response_status == "ok" && isfinite(r.reaction_rmse), rows)
        complete = length(RD_TEST_PROFILES)
        field_rmse = nf == complete ? sqrt(mean(abs2, getproperty.(rows, :field_rmse))) : Inf
        reaction_rmse = nr == complete ? sqrt(mean(abs2, getproperty.(rows, :reaction_rmse))) : Inf
        coefficient_rmse = nr == complete ? sqrt(mean(abs2, getproperty.(rows, :coefficient_rmse))) : Inf
        supported = filter(r -> r.support_fraction > 0, rows)
        supported_reaction_rmse = isempty(supported) ? NaN : nr == complete ?
            sqrt(sum(r.support_fraction*r.supported_reaction_rmse^2 for r in supported) /
                 sum(r.support_fraction for r in supported)) : Inf
        push!(result, (; case, model, seed, track=cfg.track, finite_fields=nf, finite_responses=nr,
            field_rmse, reaction_rmse, coefficient_rmse, supported_reaction_rmse))
    end
    result
end

function rd_summary(path, selections, seed_rows, cfg)
    med(rows, field) = begin
        xs = filter(isfinite, getproperty.(rows, field))
        isempty(xs) ? "NA" : @sprintf("%.5g", median(xs))
    end
    open(path, "w") do io
        println(io, "# Reaction-diffusion assessment: $(cfg.track)\n")
        println(io, "$(cfg.cells) cells, seeds $(cfg.seeds), $(cfg.iterations)-iteration ceiling. ",
            "Same-mesh fitting comparison; mesh refinement is reported separately.\n")
        println(io, "| Reaction | Model | Valid selections | Complete field seeds | Complete response seeds | Median field RMSE | Median reaction RMSE | Median coefficient RMSE | Median fit seconds | Median total tuning seconds |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case in cfg.cases, model in cfg.models
            selected = filter(r -> r.case == case && r.model == model, selections)
            seeds = filter(r -> r.case == case && r.model == model, seed_rows)
            println(io, "| $case | $model | ", count(r -> r.selection_status=="ok", selected), "/", length(selected),
                " | ", count(r -> r.finite_fields==length(RD_TEST_PROFILES), seeds), "/", length(seeds),
                " | ", count(r -> r.finite_responses==length(RD_TEST_PROFILES), seeds), "/", length(seeds),
                " | ", med(seeds,:field_rmse), " | ", med(seeds,:reaction_rmse), " | ",
                med(seeds,:coefficient_rmse), " | ", med(selected,:fit_seconds), " | ", med(selected,:tuning_seconds), " |")
        end
        println(io, "\nErrors pool squared errors across the two held-out profiles within seed. ",
            "Finite medians require complete profile sets; no cells are treated as independent seed replicates. ",
            "Reaction error is for u(1-u)r(u); coefficient error is for r(u), which is weakly ",
            "informed near the known reaction zeros. Allocation volume is cumulative, not peak memory. ",
            "This is one optimizer/initialization protocol, not a best-attainable model ranking.")
    end
end

function rd_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    cfg = rd_options(args)
    BLAS.set_num_threads(1)
    files = ("trials.csv","selected.csv","test_profiles.csv","seed_metrics.csv","mesh_refinement.csv",
             "coefficients.toml","candidate_coefficients.toml","metadata.toml","summary.md","assessment-script.jl")
    any(isfile(joinpath(cfg.output, f)) for f in files) && error("output contains previous results; choose a new --output")
    mkpath(cfg.output)
    initial_source = source_hash()
    script = read(@__FILE__)
    write(joinpath(cfg.output, "assessment-script.jl"), script)
    mesh_rows = reduce(vcat, [rd_mesh_errors(case, cfg.cells) for case in cfg.cases])
    write_csv(joinpath(cfg.output,"mesh_refinement.csv"), mesh_rows)
    trials, selected, trajectories = NamedTuple[], NamedTuple[], NamedTuple[]
    coefficients, candidates = Dict{String,Any}(), Dict{String,Any}()
    warmed = Set{String}()
    for case in cfg.cases, seed in cfg.seeds
        ds = rd_dataset(case, seed, cfg.cells)
        for model in cfg.models
            created = @timed rd_approximator(model, 20_000+seed)
            a = created.value
            if !(model in warmed)
                warm = merge(cfg, (; iterations=2))
                fit, row = rd_train(a, ds, first(cfg.etas), warm)
                row.status == "ok" || error("warm-up failed for $model: $(row.message)")
                push!(warmed, model)
            end
            common = (; case, model, seed, track=cfg.track, cells=cfg.cells, nparams=nparams(a),
                       setup_seconds=created.time)
            best_fit = best_row = nothing
            tuning_seconds = 0.0
            for (eta_id, eta) in enumerate(cfg.etas)
                fit, row = rd_train(a, ds, eta, cfg)
                candidate_id = "$(case)__$(model)__$(seed)__$(eta_id)"
                candidate = merge(common, (; candidate_id), row)
                push!(trials, candidate)
                entry = Dict{String,Any}("status"=>row.status, "message"=>row.message,
                    "model_seed"=>20_000+seed, "eta"=>eta)
                if fit !== nothing && all(isfinite, fit.parameters)
                    entry["parameters"] = collect(fit.parameters)
                    entry["smoothing_params"] = fit.smoothing_params
                end
                candidates[candidate_id] = entry
                isfinite(row.fit_seconds) && (tuning_seconds += row.fit_seconds)
                if row.status == "ok" && isfinite(row.validation_rmse) &&
                    (best_row === nothing || row.validation_rmse < best_row.validation_rmse)
                    best_fit, best_row = fit, candidate
                end
                write_csv(joinpath(cfg.output,"trials.csv"), trials)
            end
            if best_fit === nothing
                push!(selected, merge(last(trials), (; selection_status="failed", tuning_seconds)))
            else
                push!(selected, merge(best_row, (; selection_status="ok", tuning_seconds)))
                beta = collect(best_fit.parameters)
                key = "$(case)__$(model)__$(seed)"
                coefficients[key] = Dict("parameters"=>beta, "model_seed"=>20_000+seed,
                    "candidate_id"=>best_row.candidate_id, "smoothing_params"=>best_fit.smoothing_params)
                for id in eachindex(RD_TEST_PROFILES)
                    push!(trajectories, merge(common, (; candidate_id=best_row.candidate_id), rd_score(a,beta,ds,id)))
                end
            end
            write_csv(joinpath(cfg.output,"selected.csv"), selected)
            isempty(trajectories) || write_csv(joinpath(cfg.output,"test_profiles.csv"), trajectories)
            for (file, data) in (("coefficients.toml",coefficients), ("candidate_coefficients.toml",candidates))
                open(joinpath(cfg.output,file),"w") do io
                    TOML.print(io,data)
                end
            end
            row = last(selected)
            println(case, " seed=",seed," model=",model," track=",cfg.track,
                " selection=",row.selection_status," validation=",row.validation_rmse,
                " fitting_s=",row.fit_seconds)
            flush(stdout)
        end
    end
    source_hash() == initial_source && read(@__FILE__) == script || error("source changed during assessment")
    seed_rows = rd_seed_metrics(trajectories,cfg)
    write_csv(joinpath(cfg.output,"seed_metrics.csv"),seed_rows)
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()),"Manifest.toml"))
    metadata = Dict("source_sha256"=>initial_source,
        "assessment_script_sha256"=>bytes2hex(SHA.sha256(script)),
        "julia_version"=>string(VERSION), "platform"=>string(Sys.KERNEL,"-",Sys.ARCH),
        "check_bounds"=>true, "blas_threads"=>BLAS.get_num_threads(),
        "cells"=>cfg.cells, "length"=>6.0, "diffusion"=>0.1, "density_domain"=>collect(RD_DENSITY_DOMAIN),
        "initial_rate"=>RD_INITIAL_RATE, "observation_sigma"=>0.01,
        "training_profiles"=>RD_TRAIN_PROFILES, "validation_profiles"=>RD_VALIDATION_PROFILES,
        "test_profiles"=>RD_TEST_PROFILES, "train_end"=>2.5, "test_end"=>4.0,
        "seeds"=>cfg.seeds, "cases"=>cfg.cases, "models"=>cfg.models, "track"=>cfg.track,
        "iterations"=>cfg.iterations, "learning_rate"=>cfg.lr, "etas"=>cfg.etas,
        "plateau_tol"=>cfg.plateau_tol, "plateau_window"=>cfg.plateau_window,
        "early_stopping"=>cfg.early_stopping,
        "spatial_model"=>"cell-centred finite volume; zero boundary flux; exact initial cell averages",
        "loss_weights"=>"one over cells times training profiles",
        "package_versions"=>Dict(name=>get(first(entries),"version","unversioned")
                                 for (name,entries) in manifest["deps"]))
    open(joinpath(cfg.output,"metadata.toml"),"w") do io
        TOML.print(io,metadata)
    end
    rd_summary(joinpath(cfg.output,"summary.md"),selected,seed_rows,cfg)
    (; cfg, trials, selected, trajectories, seed_rows, mesh_rows, coefficients, candidates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rd_main(ARGS)
end
