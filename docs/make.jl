using Documenter
using PartiallySpecifiedModels

const REPO = "https://github.com/ecorecipes/PartiallySpecifiedModels.jl"

# Quarto's GitHub-flavoured Markdown, made readable by Documenter:
# - the author/date lines and Quarto's table of contents are dropped (Documenter shows its own);
# - the `<div id="fig-...">` wrappers around figures are dropped, since Documenter escapes raw HTML;
# - links to sibling vignettes (`../NN_name/NN_name.qmd`, `.html` or `.md`) point at their tutorial page;
# - links out of `vignettes/` point at the API page or at the file on GitHub.
# Code fences are left alone.
function documenter_markdown(text::AbstractString)
    text = replace(text, r"\A(# [^\n]*\n)(?:[^\n#\-\s][^\n]*\n){0,3}\s*(?:[ \t]*- \[[^\n]*\n(?:[ \t]+[^\n]*\n)*)+"s => s"\1\n")
    text = replace(text, r"\]\(\.\./(\d+_\w+)/\1\.(?:qmd|html|md)(?:#[^)]*)?\)" => s"](\1.md)")
    text = replace(text, r"\]\(\.\./\.\./docs/src/([^)]+\.md)\)" => s"](../\1)")
    text = replace(text, r"\]\(\.\./\.\./([^)]+)\)" => SubstitutionString("]($REPO/blob/main/\\1)"))
    out = IOBuffer()
    fence = ""
    for line in eachline(IOBuffer(text); keep = true)
        m = match(r"^[ \t]*(`{3,}|~{3,})", line)
        if !isempty(fence)
            m !== nothing && strip(line) == m.captures[1] && first(m.captures[1]) == first(fence) && (fence = "")
        elseif m !== nothing
            fence = m.captures[1]
        elseif occursin(r"^\s*(<div\b[^>]*>|</div>)\s*$", line)
            continue
        end
        print(out, line)
    end
    return String(take!(out))
end

# Copy the rendered vignettes (vignettes/NN_name/NN_name.md and the figures in NN_name_files/)
# into docs/src/tutorials/ at build time, so the docs serve them without a second copy in git.
function sync_tutorials()
    vig = joinpath(@__DIR__, "..", "vignettes")
    out = joinpath(@__DIR__, "src", "tutorials")
    rm(out; recursive = true, force = true)
    mkpath(out)
    pages = Pair{String,String}[]
    for d in sort(filter(d -> occursin(r"^\d+_", d) && isfile(joinpath(vig, d, d * ".md")), readdir(vig)))
        text = documenter_markdown(read(joinpath(vig, d, d * ".md"), String))
        write(joinpath(out, d * ".md"), text)
        files = joinpath(vig, d, d * "_files")
        if isdir(files)
            for (root, _, fs) in walkdir(files), f in fs
                rel = relpath(joinpath(root, f), joinpath(vig, d))
                startswith(rel, joinpath(d * "_files", "libs")) && continue  # Quarto's HTML assets
                mkpath(dirname(joinpath(out, rel)))
                cp(joinpath(root, f), joinpath(out, rel))
            end
        end
        m = match(r"^# (.+)$"m, text)
        push!(pages, (m === nothing ? d : strip(m.captures[1])) => "tutorials/$d.md")
    end
    return pages
end

tutorials = sync_tutorials()

makedocs(
    sitename = "PartiallySpecifiedModels.jl",
    authors = "Simon Frost",
    modules = [PartiallySpecifiedModels],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://ecorecipes.github.io/PartiallySpecifiedModels.jl/stable/",
        repolink = REPO,
        size_threshold = 1024 * 1024,
        size_threshold_warn = 512 * 1024,
        search_size_threshold_warn = 2 * 1024 * 1024,
    ),
    repo = "$REPO/blob/{commit}{path}#{line}",
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Migration to v0.3" => "migration_v03.md",
        "Approximators" => "approximators.md",
        "Custom Approximators" => "extending.md",
        "Solvers" => "solvers.md",
        "Vignettes" => ["Overview" => "vignettes.md"; tutorials],
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/ecorecipes/PartiallySpecifiedModels.jl.git",
    devbranch = "main",
    push_preview = true,
)
