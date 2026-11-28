using Test
using Aqua
using PartiallySpecifiedModels

@testset "Aqua.jl package quality checks" begin
    # Test all core Aqua checks (skipping deps_compat for unversioned GitHub URL
    # sources and ambiguity checks on foreign SciML/OrdinaryDiffEq methods).
    Aqua.test_ambiguities(PartiallySpecifiedModels)
    Aqua.test_unbound_args(PartiallySpecifiedModels)
    Aqua.test_undefined_exports(PartiallySpecifiedModels)
    Aqua.test_project_extras(PartiallySpecifiedModels)
    Aqua.test_stale_deps(PartiallySpecifiedModels; ignore=[:OrdinaryDiffEqLowOrderRK, :OrdinaryDiffEqSDIRK])
    Aqua.test_piracies(PartiallySpecifiedModels;
                       treat_as_own=[PartiallySpecifiedModels.PSMProblem,
                                     PartiallySpecifiedModels.PSMSolution,
                                     PartiallySpecifiedModels.BootstrapResult])
end
