using CSV, TOML, SHA, Statistics, Plots

profile_number(row,key) = parse(Float64,getproperty(row,key))
profile_rows(path) = collect(CSV.File(path;types=String,stringtype=String,missingstring=nothing,ntasks=1))
profile_digest(path) = bytes2hex(SHA.sha256(read(path)))

function plot_profile_main(args)
    opts = Dict("input"=>joinpath(@__DIR__,"results","smoothing-profile-diagnostics"),
        "analysis"=>joinpath(@__DIR__,"results","smoothing-profile-analysis"),
        "seeds"=>"4001,4003",
        "output"=>joinpath(@__DIR__,"results","smoothing-profile-analysis","figures"))
    for arg in args
        startswith(arg,"--") && occursin('=',arg) || throw(ArgumentError("options must be --name=value"))
        key,value = split(arg[3:end],'=';limit=2)
        haskey(opts,key) || throw(ArgumentError("unknown plotting option $key"))
        opts[key] = value
    end
    source,analysis,output = abspath.([opts["input"],opts["analysis"],opts["output"]])
    seeds = parse.(Int,split(opts["seeds"],','))
    design = TOML.parsefile(joinpath(source,"design.toml"))
    all(in(design["seeds"]),seeds) && allunique(seeds) ||
        throw(ArgumentError("plot seeds must be unique members of the retained cohort"))
    ispath(output) && error("plot output already exists; choose a fresh directory")
    files = [joinpath(source,"profiles.csv"),joinpath(source,"native_fits.csv"),
        joinpath(analysis,"decomposition.csv"),joinpath(source,"design.toml")]
    fingerprints = Dict(path=>profile_digest(path) for path in files)
    profiles,native,decomposition = profile_rows.(files[1:3])
    mkpath(output)
    default(size=(1400,1000),legendfontsize=7,guidefontsize=9,titlefontsize=10,
            linewidth=2,legend_column=2,margin=6*Plots.PlotMeasures.mm,
            foreground_color_legend=nothing,background_color_legend=:transparent)
    for seed in seeds
        panels = []
        for case in design["cases"], model in design["models"]
            rows = filter(r->r.case==case && r.model==model && parse(Int,r.seed)==seed && r.status=="ok",profiles)
            limit = only(filter(r->isinf(profile_number(r,:rho)),rows))
            finite = sort!(filter(r->isfinite(profile_number(r,:rho)),rows);by=r->profile_number(r,:rho))
            base = only(filter(r->r.case==case && r.model==model && parse(Int,r.seed)==seed,native))
            rho = profile_number.(finite,Ref(:rho))
            estimated_max = maximum(profile_number(r,:criterion) for r in rows)
            known_max = maximum(profile_number(r,:known_criterion) for r in rows)
            native_rho = profile_number(base,:rho)
            p = plot(rho,profile_number.(finite,Ref(:criterion)).-estimated_max;
                label="Profiled noise scale",color=:royalblue,
                xlabel="log(lambda)",ylabel="Centered working criterion",
                title="$case / $model (native rho=$(round(native_rho;digits=2)))",
                ylims=(-12.0,0.5),legend=:outerbottom)
            plot!(p,rho,profile_number.(finite,Ref(:known_criterion)).-known_max;
                label="Known noise scale (oracle)",color=:darkorange)
            hline!(p,[profile_number(limit,:criterion)-estimated_max];
                label="Affine limit, profiled scale",color=:royalblue,linestyle=:dash)
            hline!(p,[profile_number(limit,:known_criterion)-known_max];
                label="Affine limit, known scale",color=:darkorange,linestyle=:dash)
            vline!(p,[native_rho];label="Native selected smoothing",color=:black,linestyle=:dot)
            push!(panels,p)
        end
        figure = plot(panels...;layout=(length(design["cases"]),length(design["models"])),
            plot_title="Fully refitted working profiles, dataset $seed")
        savefig(figure,joinpath(output,"profiles-$seed.svg"))
    end
    panels = []
    for case in design["cases"], model in design["models"]
        rows = filter(r->r.case==case && r.model==model,decomposition)
        base = sort!(filter(r->r.method=="native",rows);by=r->profile_number(r,:x))
        x = profile_number.(base,Ref(:x))
        sd = profile_number.(base,Ref(:empirical_sd))
        p = plot(x,zeros(length(x));ribbon=sd,fillalpha=0.15,linealpha=0,
            color=:gray,label="Native +/- sampling SD",xlabel="Density N",ylabel="Response error",
            title="$case / $model",legend=:outerbottom)
        for (method,label,color) in (("native","Native bias",:royalblue),
            ("profile_grid","Profile-grid bias",:purple),
            ("known_scale_grid","Known-scale grid bias (oracle)",:darkorange))
            values = sort!(filter(r->r.method==method,rows);by=r->profile_number(r,:x))
            plot!(p,profile_number.(values,Ref(:x)),profile_number.(values,Ref(:bias));label,color)
        end
        plot!(p,x,profile_number.(base,Ref(:representation_bias));label="Representation bias",
            color=:black,linestyle=:dot)
        push!(panels,p)
    end
    savefig(plot(panels...;layout=(length(design["cases"]),length(design["models"])),
        plot_title="Bias versus sampling variability; truth used for diagnosis only"),
        joinpath(output,"bias-decomposition.svg"))
    all(profile_digest(path)==hash for (path,hash) in fingerprints) || error("plot input changed")
    open(joinpath(output,"metadata.toml"),"w") do io
        TOML.print(io,Dict("input_sha256"=>fingerprints,"plot_script_sha256"=>profile_digest(@__FILE__),
            "profile_seeds"=>seeds,"profile_normalization"=>"each scale criterion centered at its own maximum",
            "profile_ylim"=>[-12.0,0.5],"bias_ribbon"=>"one empirical sampling SD, not a confidence band",
            "generated_figures"=>sort(filter(n->endswith(n,".svg"),readdir(output)))))
    end
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    plot_profile_main(ARGS)
end
