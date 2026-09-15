include("assess_simultaneous_bands.jl")

function sc_correction(sol, prob; kwargs...)
    try
        (; correction=smoothing_covariance_correction(sol, prob; kwargs...), status="ok", message="")
    catch e
        e isa DomainError || (e isa ArgumentError && startswith(e.msg, "smoothing covariance:")) || rethrow()
        (; correction=nothing, status="unavailable", message=sprint(showerror, e))
    end
end

function sc_record(sol, corrected)
    record = Dict{String,Any}(
        "parameters"=>collect(sol.parameters), "smoothing_params"=>sol.smoothing_params,
        "V_beta"=>collect.(eachrow(sol.convergence.V_beta)),
        "sigma2"=>sol.convergence.sigma2, "edf"=>sol.edf,
        "data_loss"=>sol.data_loss, "converged"=>sol.convergence.converged,
        "stationarity"=>sol.convergence.stationarity,
        "smoothing_fixed"=>sol.convergence.smoothing_fixed,
        "smoothing_advanced"=>sol.convergence.smoothing_advanced,
        "correction_status"=>corrected.status, "correction_message"=>corrected.message)
    corrected.correction === nothing && return record
    c = corrected.correction
    for key in (:covariance, :conditional_covariance, :mean_correction, :root_correction,
                :rho_covariance, :rho_hessian, :coefficient_sensitivity)
        value = getproperty(c, key)
        value === nothing || (record[string(key)] = collect.(eachrow(value)))
    end
    for key in (:sigma2, :profile_sigma2, :n_eff, :eigenvalues, :rho_gradient,
                :coefficient_score, :coefficient_step, :rho_regularization,
                :covariance_ridge, :determinant_ridge)
        value = getproperty(c, key)
        value === nothing || (record[string(key)] = value)
    end
    record["rho_source"] = string(c.rho_source)
    record["method"] = string(c.method)
    record
end

function sc_main(args)
    opts = Dict("cases"=>join(CR_CASES, ","), "models"=>"spline8,kan12_free",
        "seeds"=>join(4001:4010, ","), "iterations"=>"40", "nsim"=>"10000",
        "rho-regularization"=>"0.0",
        "output"=>joinpath(@__DIR__, "results", "analytic-smoothing-covariance"))
    for arg in args
        startswith(arg, "--") && occursin('=', arg) || throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown smoothing assessment option $key"))
        opts[key] = value
    end
    cases, models = String.(split(opts["cases"], ',')), String.(split(opts["models"], ','))
    seeds = parse.(Int, split(opts["seeds"], ','))
    iterations, nsim = parse.(Int, [opts["iterations"], opts["nsim"]])
    reg = parse(Float64, opts["rho-regularization"])
    all(in(CR_CASES), cases) && all(in(CF_CONFIRM), models) &&
        all(in(4001:4030), seeds) && all(allunique, (cases, models, seeds)) &&
        iterations > 0 && nsim >= 2 && isfinite(reg) && reg >= 0 ||
        throw(ArgumentError("invalid conditions, models, archived seeds, budget or regularization"))
    output = abspath(opts["output"])
    BLAS.set_num_threads(1)
    scripts = [@__FILE__, joinpath(@__DIR__, "assess_simultaneous_bands.jl"),
        joinpath(@__DIR__, "coverage_robustness.jl"), joinpath(@__DIR__, "undersmoothing.jl"),
        joinpath(@__DIR__, "calibrate_families.jl"), joinpath(@__DIR__, "calibrate_uncertainty.jl"),
        joinpath(@__DIR__, "rd_archive.jl"), joinpath(@__DIR__, "reaction_diffusion.jl")]
    initial_source = source_hash()
    hashes = rd_followon_snapshot(output, scripts)
    design = Dict{String,Any}("cases"=>cases, "models"=>models, "seeds"=>seeds,
        "iterations"=>iterations, "nsim"=>nsim, "level"=>0.95,
        "rho_regularization"=>reg, "stages"=>["automatic", "quarter"],
        "sources"=>["conditional", "analytic"], "intervals"=>["pointwise", "simultaneous"],
        "data"=>"unchanged retained robustness observations; refitted with current source",
        "coverage_scope"=>"exploratory reused datasets, NOT fresh calibration or a coverage guarantee",
        "quarter"=>"fixed 0.25 times selected lambda; propagate selection-stage log-lambda covariance",
        "conditioning"=>"local Gaussian; coefficient dispersion and data Jacobian/weights held fixed",
        "grid"=>"101-point base grid united with archived queries; selected in-range points only",
        "selection"=>"no tuning or curvature-dependent dataset exclusion",
        "bootstrap_refits"=>0)
    rd_write_toml(joinpath(output, "design.toml"), design)
    mkpath(joinpath(output, "fits"))
    scores, fits = NamedTuple[], NamedTuple[]
    fingerprints = Dict{String,String}()
    for case in cases, seed in seeds
        reference = cr_scenario(case)
        in_points = [q.x for q in reference.queries if q.region == "in_range"]
        lo, hi = extrema(in_points)
        points = filter(x->lo<=x<=hi, sort!(unique(
            [collect(range(first(UC_POINTS), last(UC_POINTS), length=101)); UC_POINTS])))
        truth = Float64[cr_rate(reference.metadata["response"], x) for x in points]
        observed_hash = nothing
        for model in models
            input = joinpath(@__DIR__, "results", "robustness-$model-1")
            data_path = joinpath(input, "datasets", "$(case)__$(seed).toml")
            for path in (data_path, joinpath(input,"design.toml"), joinpath(input,"metadata.toml"))
                fingerprints[relpath(path, ROOT)] = rd_digest(path)
            end
            # Matched models must really receive the same retained observations.
            digest = rd_digest(data_path)
            observed_hash === nothing || digest == observed_hash || error("unmatched archived data")
            observed_hash = digest
            ds0 = TOML.parsefile(data_path)
            ds = (; times=Float64.(ds0["times"]), values=hcat(ds0["values"]...))
            spec = cf_model(model)
            prob, report_prob = uc_problem(spec.fit, ds), uc_problem(spec.report, ds)
            selected = solve(prob, LAML(maxiters=iterations, jac=:forwarddiff))
            all(isfinite, selected.parameters) && all(isfinite, selected.fitted_values) ||
                error("non-finite original fit; assessment stopped rather than dropping a dataset")
            selected_correction = sc_correction(selected, report_prob; rho_regularization=reg)
            lambda = only(selected.smoothing_params)
            warm = InitializedApprox(:r, spec.report, collect(selected.parameters), penalty_matrix(spec.report))
            quarter = solve(uc_problem(warm, ds),
                LAML(maxiters=iterations, jac=:forwarddiff, fixed_lambda=0.25lambda))
            quarter_correction = selected_correction.correction === nothing ?
                (; correction=nothing, status="unavailable",
                   message="selection-stage covariance unavailable: " * selected_correction.message) :
                sc_correction(quarter, report_prob;
                    rho_covariance=selected_correction.correction.rho_covariance)
            for (stage, sol, corrected) in (("automatic", selected, selected_correction),
                                           ("quarter", quarter, quarter_correction))
                c = corrected.correction
                all(isfinite, sol.parameters) && all(isfinite, sol.fitted_values) ||
                    error("non-finite refit; assessment stopped rather than dropping a dataset")
                push!(fits, (; case, model, seed, stage, selected_lambda=lambda,
                    lambda=only(sol.smoothing_params), edf=sol.edf,
                    converged=sol.convergence.converged, stationarity=sol.convergence.stationarity,
                    smoothing_advanced=sol.convergence.smoothing_advanced,
                    status=corrected.status, message=corrected.message,
                    rho_variance=c === nothing ? NaN : only(c.rho_covariance),
                    mean_trace=c === nothing ? NaN : tr(c.mean_correction),
                    root_trace=c === nothing ? NaN : tr(c.root_correction),
                    conditional_trace=tr(sol.convergence.sigma2*sol.convergence.V_beta),
                    coefficient_step=c === nothing ? NaN : norm(c.coefficient_step)))
                rd_write_toml(joinpath(output,"fits","$(case)__$(model)__$(seed)__$(stage).toml"),
                    sc_record(sol, corrected))
                for source in ("conditional", "analytic"), interval in (:pointwise, :simultaneous)
                    band = nothing
                    status, message = source == "analytic" ? (corrected.status, corrected.message) : ("ok", "")
                    if status == "ok"
                        band = confidence_band(sol, report_prob; uf_points=Dict(:r=>points), interval,
                            nsim, rng=StableRNG(9_000_000 + 10_000findfirst(==(case),CR_CASES) + seed),
                            unconditional=source=="analytic",
                            rho_regularization=source=="analytic" && stage=="automatic" ? reg : 0.0,
                            rho_covariance=source=="analytic" && stage=="quarter" ? c.rho_covariance : nothing)[:r]
                    end
                    fitted = sol.unknown_functions[:r].(points)
                    lower = band === nothing ? fill(NaN,length(points)) : band.lower
                    upper = band === nothing ? fill(NaN,length(points)) : band.upper
                    score = sb_score(fitted, lower, upper, truth)
                    push!(scores, (; case, model, seed, stage, source, interval=string(interval),
                        level=0.95, status, message,
                        critical=band === nothing ? NaN : hasproperty(band,:critical) ?
                            band.critical : PSM._qnorm(0.975), score...))
                end
                println("smoothing covariance $case $model seed=$seed $stage $(corrected.status)")
                flush(stdout)
            end
        end
    end
    expected_fits = 2length(cases)*length(models)*length(seeds)
    length(fits) == expected_fits && length(scores) == 4expected_fits ||
        error("incomplete assessment population")
    CSV.write(joinpath(output,"fits.csv"), fits)
    CSV.write(joinpath(output,"dataset_bands.csv"), scores)
    summary = NamedTuple[]
    for case in cases, model in models, stage in ("automatic","quarter"),
        source in ("conditional","analytic"), interval in ("pointwise","simultaneous")
        rows = filter(r -> (r.case,r.model,r.stage,r.source,r.interval) ==
                           (case,model,stage,source,interval), scores)
        length(rows) == length(seeds) || error("missing group records")
        widths = [r.mean_width for r in rows if r.band_available]
        push!(summary, (; case, model, stage, source, interval, datasets=length(rows),
            available=count(r->r.band_available,rows),
            point_coverage_yield=mean(r.point_coverage_yield for r in rows),
            joint_coverage_yield=mean(r.joint_covered for r in rows),
            mean_width_available=isempty(widths) ? NaN : mean(widths)))
    end
    CSV.write(joinpath(output,"summary.csv"), summary)
    all(rd_digest(joinpath(ROOT,path)) == hash for (path,hash) in fingerprints) ||
        error("an input archive changed during the assessment")
    meta = rd_followon_metadata(output, scripts, hashes, initial_source, [])
    merge!(meta, Dict("input_sha256"=>fingerprints, "optimizer_refit"=>true,
        "new_bootstrap_refits"=>0, "fitting_calls"=>expected_fits,
        "design_sha256"=>rd_digest(joinpath(output,"design.toml")),
        "fits_sha256"=>rd_digest(joinpath(output,"fits.csv")),
        "bands_sha256"=>rd_digest(joinpath(output,"dataset_bands.csv")),
        "summary_sha256"=>rd_digest(joinpath(output,"summary.csv"))))
    rd_write_toml(joinpath(output,"metadata.toml"), meta)
    output
end

if abspath(PROGRAM_FILE) == @__FILE__
    sc_main(ARGS)
end
