module KANReactionDiffusionTests

using Test
include("../benchmarks/kan/reaction_diffusion.jl")

struct ReactionProbe{F} <: AbstractApproximator
    name::Symbol
    f::F
end
PSM.nparams(::ReactionProbe) = 1
PSM.initial_params(::ReactionProbe) = [0.8]
PSM.penalty_matrix(::ReactionProbe) = nothing
PSM.build_evaluator(a::ReactionProbe, beta) = a.f

@testset "KAN reaction-diffusion contracts" begin
    @testset "Conservative no-flux spatial operator" begin
        @test_throws ArgumentError rd_grid(1)
        @test_throws ArgumentError rd_grid(8; diffusion=-1.0)
        @test_throws ArgumentError rd_grid(8; length=Inf)
        grid = rd_grid(24)
        profiles = [RD_TRAIN_PROFILES; RD_VALIDATION_PROFILES; RD_TEST_PROFILES]
        for name in profiles
            initial = rd_initial(name, grid)
            @test all(x -> 0 < x < 1, initial)
            fine = rd_initial(name, rd_grid(48))
            restricted = vec(rd_restrict(reshape(fine, 1, :), 48, 24))
            # Measured exact-average restriction discrepancies <=4.45e-16;
            # 1e-12 leaves >2200x headroom for trigonometric roundoff.
            @test isapprox(initial, restricted; atol=1e-12, rtol=1e-12)
        end
        u = rd_u0(RD_TRAIN_PROFILES, grid)
        saved = copy(u)
        du = similar(u)
        rd_rhs!(du, u, x -> zero(x), grid)
        @test u == saved
        # Measured mass-rate error is zero; 1e-12 permits accumulation
        # roundoff while rejecting a leaking boundary or cross-profile face.
        @test abs(grid.dx*sum(du)) < 1e-12
        first_profile = zeros(grid.cells)
        rd_rhs!(first_profile, u[1:grid.cells], x -> zero(x), grid)
        @test du[1:grid.cells] == first_profile
        perturbed = copy(u)
        perturbed[grid.cells+1:end] .+= 0.25
        other = similar(u)
        rd_rhs!(other, perturbed, x -> zero(x), grid)
        @test other[1:grid.cells] == du[1:grid.cells]
        for constant in (0.0, 1.0)
            state = fill(constant, 2grid.cells)
            out = similar(state)
            rd_rhs!(out, state, x -> 0.7, grid)
            @test all(iszero, out)
        end
        @test_throws DimensionMismatch rd_rhs!(zeros(5), zeros(5), identity, grid)
        @test_throws ArgumentError rd_restrict(zeros(1, 12), 12, 5)
        @test_throws DimensionMismatch rd_restrict(zeros(1, 13), 12, 6)
        @test_throws ArgumentError rd_profile("unknown")
        @test_throws ArgumentError rd_rate("unknown", 0.5)

        continuum_errors = Float64[]
        for cells in (12, 24, 48)
            g = rd_grid(cells)
            times = collect(0.0:0.2:2.0)
            computed = rd_reference("diffusion", ["fine_scale"], g, times)
            exact = rd_diffusion_exact("fine_scale", g, times)
            # Measured modal integration error <=7.59e-13; >1300x headroom.
            @test maximum(abs, computed-exact) < 1e-9
            mass = vec(sum(computed; dims=2))
            # Measured mass drift <=1.07e-14, before multiplying by dx;
            # 1e-11 leaves >900x headroom.
            @test maximum(abs, mass .- sum(rd_initial("fine_scale", g))) < 1e-11
            push!(continuum_errors, rmse(computed, rd_diffusion_exact("fine_scale", g, times; continuum=true)))
        end
        # Measured refinement ratios 3.80 and 3.95; >20% margin to 3.
        @test all(continuum_errors[1:2] ./ continuum_errors[2:3] .> 3.0)
    end

    @testset "Independent data splits and correct estimands" begin
        @test isempty(intersect(RD_TRAIN_PROFILES, RD_VALIDATION_PROFILES))
        @test isempty(intersect(RD_TRAIN_PROFILES, RD_TEST_PROFILES))
        @test isempty(intersect(RD_VALIDATION_PROFILES, RD_TEST_PROFILES))
        @test rd_options(String[]).seeds == collect(301:310)
        @test rd_options(["--track=laml"]).models == ["spline8", "spline28"]
        @test_throws ArgumentError rd_options(["--track=laml", "--etas=0"])
        @test_throws ArgumentError rd_options(["--track=laml", "--models=kan28"])
        for case in RD_CASES
            ds = rd_dataset(case, 17, 24)
            same = rd_dataset(case, 17, 24)
            different = rd_dataset(case, 18, 24)
            @test ds.train_values == same.train_values
            @test ds.validation_values == same.validation_values
            @test ds.train_values != different.train_values
            @test ds.train_truth == different.train_truth
            @test size(ds.train_values) == (11, 48)
            @test size(ds.validation_values) == (11, 24)
            @test all(size(values) == (41, 24) for values in ds.test_truth)
            truth = ReactionProbe(:r, x -> rd_rate(case, x))
            p = rd_problem(truth, RD_TRAIN_PROFILES, ds.grid, ds.train_times, ds.train_values, Ref(0))
            @test all(==(1/48), p.data_weights)
            # Measured relative difference <=1.79e-16; >5000x headroom.
            @test isapprox(sqrt(PSM.weighted_data_loss(p, ds.train_truth)/length(ds.train_times)),
                           rmse(ds.train_truth, ds.train_values); rtol=1e-12)
            @test_throws DimensionMismatch rd_problem(truth, RD_TRAIN_PROFILES, ds.grid,
                ds.train_times, ds.train_values[:, 1:24], Ref(0))
            for id in eachindex(RD_TEST_PROFILES)
                score = rd_score(truth, [0.8], ds, id)
                @test score.test_status == score.response_status == "ok"
                @test score.coefficient_rmse == score.reaction_rmse == 0
                # Measured true-RHS field error <=1.004e-9; >90x headroom.
                @test score.field_rmse < 1e-7
            end
            for row in rd_mesh_errors(case, 24)
                # Measured nonlinear refinement ratios 4.005--4.035;
                # (3,5) allows about 25% margin around second order.
                @test 3.0 < row.refinement_ratio < 5.0
            end
        end
    end

    @testset "Shared constructors and differentiable spatial fitting" begin
        ds = rd_dataset("crowding", 17, 8)
        for (model, count) in zip(RD_MODELS, (8, 28, 28, 28, 28))
            old = make_approximator(model, 20017)
            explicit = make_approximator(model, 20017; domain=DOMAIN, initial_rate=INITIAL_RATE)
            @test initial_params(old) == initial_params(explicit)
            @test penalty_matrix(old) == penalty_matrix(explicit)
            a = rd_approximator(model, 20017)
            beta = initial_params(a)
            @test nparams(a) == count
            @test band_domain(a) == RD_DENSITY_DOMAIN
            # Measured initial coefficient discrepancies <=1.12e-16;
            # 1e-12 leaves >8000x headroom.
            @test maximum(abs, build_evaluator(a, beta).([0.1, 0.5, 0.9]) .- RD_INITIAL_RATE) < 1e-12
            times = ds.train_times[1:3]
            p = rd_problem(a, RD_TRAIN_PROFILES, ds.grid, times, ds.train_values[1:3, :], Ref(0))
            objective = b -> PSM.adam_loss_mse(p, b)
            gradient = PSM.ForwardDiff.gradient(objective, beta)
            direction = cos.(collect(1:length(beta)))
            direction ./= norm(direction)
            h = 1e-5
            numerical = (objective(beta+h*direction)-objective(beta-h*direction))/(2h)
            @test all(isfinite, gradient) && norm(gradient) > 0
            # Measured worst directional discrepancy 8.69e-10;
            # the absolute allowance gives >110x headroom.
            @test isapprox(dot(gradient, direction), numerical; atol=1e-7, rtol=1e-5)
        end
        @test_throws ArgumentError make_approximator("spline8", 1; initial_rate=Inf)
    end

    @testset "Independent response failures and complete-seed accounting" begin
        ds = rd_dataset("crowding", 17, 8)
        points = Set(vec(ds.test_truth[1]))
        fail_off_reference = ReactionProbe(:r, x -> x in points ? 0.8 :
            throw(DomainError(x, "intentional spatial prediction failure")))
        score = rd_score(fail_off_reference, [0.8], ds, 1)
        @test score.test_status == "test_failed"
        @test score.response_status == "ok"
        @test isfinite(score.reaction_rmse)
        cfg = (; cases=["case"], models=["model"], seeds=[1,2], track="adam")
        common = (; case="case", model="model", seed=1, test_status="ok", response_status="ok",
                    coefficient_rmse=4.0, supported_reaction_rmse=2.0, support_fraction=0.5)
        rows = [merge(common, (; profile_id=1, field_rmse=3.0, reaction_rmse=6.0)),
                merge(common, (; profile_id=2, field_rmse=4.0, reaction_rmse=8.0,
                                supported_reaction_rmse=NaN, support_fraction=0.0))]
        complete, absent = rd_seed_metrics(rows, cfg)
        @test complete.field_rmse == sqrt(12.5)
        @test complete.reaction_rmse == sqrt(50.0)
        @test complete.supported_reaction_rmse == 2.0
        @test absent.field_rmse == absent.reaction_rmse == Inf
        @test absent.finite_fields == absent.finite_responses == 0
        @test only(rd_seed_metrics(rows[1:1], merge(cfg, (; seeds=[1])))).field_rmse == Inf
        @test_throws ArgumentError rd_seed_metrics([rows; rows[1]], cfg)
    end

    @testset "Persisted spatial model selection" begin
        threads = BLAS.get_num_threads()
        try
            mktempdir() do dir
                args = ["--cells=8", "--cases=crowding", "--seeds=17",
                        "--models=spline8,kan28", "--iterations=3", "--etas=0,0.1", "--output=$dir"]
                run = rd_main(args)
                @test length(run.trials) == 4
                @test length(run.selected) == 2
                @test length(run.trajectories) == 4
                for selected in run.selected
                    trials = filter(r -> r.model == selected.model, run.trials)
                    valid = filter(r -> r.status == "ok" && isfinite(r.validation_rmse), trials)
                    @test !isempty(valid)
                    winner = valid[argmin(getproperty.(valid, :validation_rmse))]
                    @test all(isequal(getproperty(selected,k), v) for (k,v) in pairs(winner))
                    @test selected.tuning_seconds == sum(r.fit_seconds for r in trials)
                    saved = run.coefficients["crowding__$(selected.model)__17"]
                    @test saved["parameters"] == run.candidates[selected.candidate_id]["parameters"]
                    @test saved["candidate_id"] == selected.candidate_id
                end
                meta = TOML.parsefile(joinpath(dir, "metadata.toml"))
                @test meta["cells"] == 8
                @test meta["diffusion"] == 0.1
                @test meta["density_domain"] == [0.0, 1.0]
                @test meta["assessment_script_sha256"] ==
                    bytes2hex(SHA.sha256(read(joinpath(dir,"assessment-script.jl"))))
                @test isfile(joinpath(dir, "mesh_refinement.csv"))
                @test_throws ErrorException rd_main(args)
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
