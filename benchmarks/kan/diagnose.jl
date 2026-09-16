include("multivariate.jl")

function diagnostic_options(args)
    opts = Dict("reference" => joinpath(@__DIR__, "results", "kan-diagnostic-reference"),
                "output" => joinpath(@__DIR__, "results", "kan-diagnostics"),
                "npoints" => "101")
    for arg in args
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("options must be --name=value"))
        key, value = split(arg[3:end], '='; limit=2)
        haskey(opts, key) || throw(ArgumentError("unknown diagnostic option $key"))
        opts[key] = value
    end
    npoints = parse(Int, opts["npoints"])
    npoints >= 2 || throw(ArgumentError("npoints must be at least 2"))
    (; reference=abspath(opts["reference"]), output=abspath(opts["output"]), npoints)
end

function diagnostic_main(args)
    Base.JLOptions().check_bounds == 1 || error("run with --check-bounds=yes")
    cfg = diagnostic_options(args)
    BLAS.set_num_threads(1)
    files = ("coverage.csv", "layers.csv", "intervals.csv", "edge_curves.csv",
             "reference_comparison.csv", "metadata.toml", "diagnostic-script.jl")
    any(isfile(joinpath(cfg.output, name)) for name in files) &&
        error("diagnostic output already exists; choose a new --output")
    source = source_hash()
    script = read(@__FILE__)
    metadata_text = read(joinpath(cfg.reference, "metadata.toml"), String)
    reference_meta = TOML.parse(metadata_text)
    oracle_text = read(joinpath(cfg.reference, "oracle.toml"), String)
    bytes2hex(SHA.sha256(oracle_text)) == reference_meta["oracle_sha256"] ||
        error("frozen prediction oracle does not match its fingerprint")
    bytes2hex(SHA.sha256(read(joinpath(cfg.reference, "capture-script.jl")))) ==
        reference_meta["capture_script_sha256"] || error("reference capture script differs")
    bytes2hex(SHA.sha256(read(joinpath(@__DIR__, "multivariate.jl")))) ==
        reference_meta["capture_harness_sha256"] || error("model/data constructor differs from reference capture")
    archive = normpath(joinpath(cfg.reference, reference_meta["source_archive"]))
    bytes2hex(SHA.sha256(read(archive))) == reference_meta["source_archive_sha256"] ||
        error("archived fitting source does not match its fingerprint")
    oracle = TOML.parse(oracle_text)
    length(oracle) == reference_meta["models"] || error("incomplete model oracle")
    archives, weights = Dict{String,Any}(), Dict{String,Any}()
    for (study, recorded) in reference_meta["archives"]
        directory = joinpath(@__DIR__, "results", study)
        meta_path, weights_path = joinpath(directory, "metadata.toml"), joinpath(directory, "coefficients.toml")
        bytes2hex(SHA.sha256(read(meta_path))) == recorded["metadata_sha256"] ||
            error("fit metadata changed for $study")
        bytes2hex(SHA.sha256(read(weights_path))) == recorded["coefficients_sha256"] ||
            error("fit coefficients changed for $study")
        archives[study] = TOML.parsefile(meta_path)
        archives[study]["source_sha256"] == reference_meta["source_sha256"] ||
            error("fitting source differs for $study")
        weights[study] = TOML.parsefile(weights_path)
    end
    coverage_rows, layer_rows, interval_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    curve_rows, comparisons = NamedTuple[], NamedTuple[]
    datasets = Dict{Tuple{String,String,Int},Any}()
    for key in sort!(collect(keys(oracle)))
        record = oracle[key]
        study, case, seed, model = record["study"], record["case"], record["seed"], record["model"]
        model in ("kan_additive64", "kan63") || error("unsupported diagnostic model $model")
        saved = weights[study]["$(case)__$(model)__$(seed)"]
        saved["parameters"] == record["parameters"] && saved["model_seed"] == record["model_seed"] ||
            error("oracle and fit weights differ for $key")
        ds = get!(datasets, (study, case, seed)) do
            mv_dataset(case, seed; split=get(archives[study], "split", "forecast"))
        end
        a = mv_make_approximator(model, record["model_seed"], ds.train_values).approx
        beta = record["parameters"]
        parts = record["parts"]
        combined_test = Dict("inputs" => [parts["test1"]["inputs"]; parts["test2"]["inputs"]],
                             "values" => [parts["test1"]["values"]; parts["test2"]["values"]])
        for (partition, part) in (("training", parts["training"]),
                                  ("validation", parts["validation"]), ("test", combined_test))
            inputs = permutedims(reduce(hcat, part["inputs"]))
            traced = kan_activation_diagnostics(a, beta, inputs)
            discrepancy = maximum(abs, traced.predictions-part["values"])
            # The pre-change scalar/Lux discrepancy was <=3.34e-16.
            # This replay gate leaves >29000x headroom at unit output scale.
            discrepancy <= 1e-11 * max(1.0, maximum(abs, part["values"])) ||
                error("diagnostic replay changed predictions for $key/$partition: $discrepancy")
            common = (; study, case, seed, model, partition, samples=traced.nsamples)
            push!(comparisons, merge(common, (; maximum_prediction_error=discrepancy)))
            for layer in traced.layers
                for c in layer.coverage
                    row = merge(common, (; layer=layer.layer, input=c.input,
                        observed_lo=c.observed_range[1], observed_hi=c.observed_range[2],
                        grid_lo=c.grid_domain[1], grid_hi=c.grid_domain[2],
                        support_lo=c.support_domain[1], support_hi=c.support_domain[2],
                        below_grid=c.below_grid, above_grid=c.above_grid,
                        outside_grid_fraction=c.outside_grid_fraction,
                        no_spline_basis=c.no_spline_basis,
                        no_spline_basis_fraction=c.no_spline_basis_fraction,
                        intervals=length(c.interval_counts),
                        occupied_intervals=count(!iszero, c.interval_counts),
                        interval_coverage=c.interval_coverage))
                    push!(coverage_rows, row)
                    for interval in eachindex(c.interval_counts)
                        push!(interval_rows, merge(common, (; layer=layer.layer, input=c.input,
                            interval, lower=c.grid_knots[interval], upper=c.grid_knots[interval+1],
                            count=c.interval_counts[interval],
                            fraction=c.interval_counts[interval]/traced.nsamples)))
                    end
                end
                push!(layer_rows, merge(common, (; layer=layer.layer,
                    inputs=length(layer.coverage),
                    outside_grid_fraction=mean(c.outside_grid_fraction for c in layer.coverage),
                    maximum_outside_grid_fraction=maximum(c.outside_grid_fraction for c in layer.coverage),
                    no_spline_basis_fraction=mean(c.no_spline_basis_fraction for c in layer.coverage),
                    minimum_interval_coverage=minimum(c.interval_coverage for c in layer.coverage),
                    median_interval_coverage=median([c.interval_coverage for c in layer.coverage]),
                    maximum_absolute_input=maximum(abs, layer.inputs))))
            end
        end
        # Representative edge plots use the first declared seed, not the
        # best fit or smallest test error. Coverage above uses every seed.
        if seed == first(archives[study]["seeds"])
            for c in kan_edge_curves(a, beta; npoints=cfg.npoints, extent=:support), k in eachindex(c.x)
                push!(curve_rows, (; study, case, seed, model, layer=c.layer,
                    input=c.input, output=c.output, logical_x=c.x[k],
                    physical_x=c.physical_x === nothing ? missing : c.physical_x[k],
                    base=c.base[k], spline=c.spline[k], total=c.total[k],
                    grid_lo=c.grid_domain[1], grid_hi=c.grid_domain[2]))
            end
        end
    end
    sum(r.samples for r in comparisons) == reference_meta["samples"] || error("incomplete sample replay")
    source_hash() == source && read(@__FILE__) == script || error("source changed during diagnostics")
    read(joinpath(cfg.reference, "oracle.toml"), String) == oracle_text &&
        read(joinpath(cfg.reference, "metadata.toml"), String) == metadata_text ||
        error("reference data changed during diagnostics")
    mkpath(cfg.output)
    for (name, rows) in (("coverage.csv", coverage_rows), ("layers.csv", layer_rows),
                         ("intervals.csv", interval_rows), ("edge_curves.csv", curve_rows),
                         ("reference_comparison.csv", comparisons))
        write_csv(joinpath(cfg.output, name), rows)
    end
    write(joinpath(cfg.output, "diagnostic-script.jl"), script)
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    metadata = Dict(
        "source_sha256" => source, "fitting_source_sha256" => reference_meta["source_sha256"],
        "diagnostic_script_sha256" => bytes2hex(SHA.sha256(script)),
        "reference_oracle_sha256" => bytes2hex(SHA.sha256(oracle_text)),
        "reference_metadata_sha256" => bytes2hex(SHA.sha256(metadata_text)),
        "model_evaluation" => true, "optimizer_refit" => false,
        "models" => length(oracle), "samples" => reference_meta["samples"],
        "maximum_prediction_error" => maximum(r.maximum_prediction_error for r in comparisons),
        "edge_selection" => "first declared seed of each study; all cases and KAN models",
        "edge_extent" => "support", "edge_npoints" => cfg.npoints,
        "julia_version" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "check_bounds" => true, "blas_threads" => BLAS.get_num_threads(),
        "package_versions" => Dict(name => get(first(entries), "version", "unversioned")
                                   for (name, entries) in manifest["deps"]))
    open(joinpath(cfg.output, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
    println("Diagnosed ", length(oracle), " frozen KAN fits; maximum prediction change ",
            metadata["maximum_prediction_error"])
    (; coverage_rows, layer_rows, interval_rows, curve_rows, comparisons, metadata)
end

if abspath(PROGRAM_FILE) == @__FILE__
    diagnostic_main(ARGS)
end
