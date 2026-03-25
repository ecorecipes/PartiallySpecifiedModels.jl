using Documenter
using PartiallySpecifiedModels

makedocs(
    sitename = "PartiallySpecifiedModels.jl",
    authors = "Simon Frost",
    modules = [PartiallySpecifiedModels],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://ecorecipes.github.io/PartiallySpecifiedModels.jl/stable/",
        repolink = "https://github.com/ecorecipes/PartiallySpecifiedModels.jl",
    ),
    repo = "https://github.com/ecorecipes/PartiallySpecifiedModels.jl/blob/{commit}{path}#{line}",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Approximators" => "approximators.md",
        "Solvers" => "solvers.md",
        "Vignettes" => "vignettes.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/ecorecipes/PartiallySpecifiedModels.jl.git",
    devbranch = "main",
    push_preview = true,
)
