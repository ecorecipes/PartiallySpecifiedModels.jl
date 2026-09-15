function _legacy_testset_name(expr)
    expr isa Expr && expr.head === :macrocall && expr.args[1] === Symbol("@testset") ||
        return nothing
    for arg in expr.args[2:end]
        arg isa String && return arg
    end
    error("legacy testset selection requires a literal testset name")
end

function _run_selected_legacy_testsets(path, pattern::Regex)
    started = false
    found_root = false
    selected = String[]
    function select(expr)
        # Skip the command-line dispatch prefix when including this same
        # runner again. All shared imports and fixture definitions follow it.
        if !started
            expr == :(using Test) || return nothing
            started = true
        end
        expr isa Expr && expr.head === :call && expr.args[1] === :include &&
            return nothing  # Focused file groups have their own selectors.
        name = _legacy_testset_name(expr)
        name === nothing && return expr
        name == "PartiallySpecifiedModels.jl" ||
            error("unexpected top-level legacy testset $name")
        found_root = true
        body = expr.args[end]
        body isa Expr && body.head === :block ||
            error("unexpected legacy root testset structure")
        kept = Any[]
        for child in body.args
            child_name = _legacy_testset_name(child)
            if child_name === nothing
                push!(kept, child)
            elseif occursin(pattern, child_name)
                push!(selected, child_name)
                push!(kept, child)
            end
        end
        isempty(selected) && error("no legacy testsets match $pattern")
        println("Selected legacy testsets: ", join(selected, "; "))
        flush(stdout)
        Expr(:macrocall, expr.args[1:end-1]..., Expr(:block, kept...))
    end
    Base.include(select, @__MODULE__, path)
    started && found_root || error("legacy testset root was not found")
    selected
end
