using Test
using PartiallySpecifiedModels

# Helper: extract all ```{julia} ... ``` code chunks from a Quarto .qmd file
# (stripping Quarto `#| ...` cell option comments) and execute them sequentially
# inside a fresh anonymous Module so every vignette runs cleanly and isolated.
function _extract_quarto_julia_chunks(qmd_path::AbstractString)
    lines = readlines(qmd_path)
    chunks = String[]
    in_chunk = false
    buf = String[]
    for line in lines
        s = strip(line)
        if !in_chunk
            if startswith(s, "```{julia")
                in_chunk = true
                empty!(buf)
            end
        else
            if s == "```"
                in_chunk = false
                push!(chunks, join(buf, "\n"))
            elseif startswith(s, "#|")
                # Quarto cell option — skip
            else
                push!(buf, line)
            end
        end
    end
    return chunks
end

function _run_vignette_qmd(qmd_path::AbstractString, modname::Symbol)
    isfile(qmd_path) || error("Missing vignette file: $qmd_path")
    chunks = _extract_quarto_julia_chunks(qmd_path)
    @assert !isempty(chunks) "No {julia} code chunks found in $qmd_path"
    mod = Module(modname)
    for (idx, code) in enumerate(chunks)
        isempty(strip(code)) && continue
        try
            Base.include_string(mod, code, "$(basename(qmd_path))#chunk$(idx)")
        catch err
            @error "Vignette execution failed" vignette=basename(qmd_path) chunk=idx exception=(err, catch_backtrace())
            rethrow()
        end
    end
    return mod
end

@testset "v0.3 Flagship Biological Vignettes (43-48)" begin
    vig_dir = joinpath(@__DIR__, "..", "vignettes")

    @testset "Vignette 43: Gause Paramecium Competition (MultiTrajectory + RandomEffect + Wald)" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "43_gause_competition", "43_gause_competition.qmd"),
            :Vig43Gause)
        @test mod.sol_mt isa PSMSolution
        @test mod.sol_mt.convergence.converged
        @test isfinite(mod.sol_mt.data_loss)
        @test mod.rmse_r1 < 0.05
        @test mod.wald_lin.pvalue < 1e-6
    end

    @testset "Vignette 44: Tribolium LPA Beetle Chaos (Discrete + Shape + DynamicalResilience)" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "44_tribolium_lpa", "44_tribolium_lpa.qmd"),
            :Vig44Tribolium)
        @test mod.sol isa PSMSolution
        @test isfinite(mod.sol.data_loss)
        @test mod.rmse_sP < 0.08
        @test mod.dp_fit isa DiscreteProblem
        @test all(isfinite, mod.eq.u)
        @test mod.res_uq.n == 40
    end

    @testset "Vignette 45: Measles Seasonal Forcing (CyclicBSpline + TrajectoryObservation + Distributions)" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "45_measles_seasonal", "45_measles_seasonal.qmd"),
            :Vig45Measles)
        @test mod.sol_seasonal isa PSMSolution
        @test mod.sol_const isa PSMSolution
        @test mod.tbl isa ModelComparisonTable
        @test isfinite(mod.sol_seasonal.data_loss)
        @test mod.rel_rmse < 0.20
        @test isapprox(mod.β_hat(0.0), mod.β_hat(1.0); atol=1e-10)
    end

    @testset "Vignette 46: Fisheries Surplus Production + DecisionAnalysis (ManagedPSM + EVPI)" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "46_fisheries_surplus_decision", "46_fisheries_surplus_decision.qmd"),
            :Vig46Fisheries)
        @test mod.sol isa PSMSolution
        @test mod.sol.convergence.converged
        @test mod.managed isa ManagedPSM
        @test isfinite(mod.v_evpi.value) && mod.v_evpi.value >= 0.0
        @test isfinite(mod.v_evppi_low.value) && mod.v_evppi_low.value >= 0.0
    end

    @testset "Vignette 47: GAM.jl + DynamicalResilience Ecosystem Bridge" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "47_gam_ecosystem_bridge", "47_gam_ecosystem_bridge.qmd"),
            :Vig47GAMBridge)
        @test mod.sol_gam isa PSMSolution
        @test isfinite(mod.sol_gam.data_loss)
        @test all(isfinite, mod.eq_pp.u)
        @test isfinite(mod.dom_re)
        @test mod.res_pp.n == 40
        @test length(PartiallySpecifiedModels.penalty_blocks(mod.approx_bd)) >= 1
    end

    @testset "Vignette 48: Stochastic Predator-Prey with Process Noise (SDEPSMProblem + EKFSolver)" begin
        mod = _run_vignette_qmd(
            joinpath(vig_dir, "48_sde_process_noise", "48_sde_process_noise.qmd"),
            :Vig48SDE)
        @test mod.sol_sde isa PSMSolution
        @test mod.sol_ode isa PSMSolution
        @test mod.tbl isa ModelComparisonTable
        @test isfinite(mod.sol_sde.convergence.loglik)
        @test mod.rmse_sde_state < mod.rmse_ode_state
    end
end
