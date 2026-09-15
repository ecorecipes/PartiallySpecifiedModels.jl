module KANMultivariateTests

using Test
include("../benchmarks/kan/replay_multivariate.jl")

const FD = PSM.ForwardDiff

struct AssessmentProbe{F} <: AbstractApproximator
    name::Symbol
    f::F
end
PSM.nparams(::AssessmentProbe) = 1
PSM.initial_params(::AssessmentProbe) = [0.3]
PSM.penalty_matrix(::AssessmentProbe) = nothing
PSM.build_evaluator(a::AssessmentProbe, beta) = a.f

@testset "Multivariate KAN assessment contracts" begin
    @test isempty(intersect(Set(MV_TRAIN_ICS), Set(MV_VALIDATION_ICS)))
    @test isempty(intersect(Set(MV_TRAIN_ICS), Set(MV_TEST_ICS)))
    @test isempty(intersect(Set(MV_VALIDATION_ICS), Set(MV_TEST_ICS)))
    @test all(>(3), mv_options(String[]).seeds)
    @test_throws ArgumentError mv_options(["--track=laml", "--etas=0.1"])

    for case in MV_CASES
        ds = mv_dataset(case, 101)
        repeated = mv_dataset(case, 101)
        other = mv_dataset(case, 102)
        @test ds.train_values == repeated.train_values
        @test ds.validation_values == repeated.validation_values
        @test ds.train_values != other.train_values
        @test ds.train_truth == other.train_truth
        @test size(ds.train_values) == (21, 6)
        @test size(ds.validation_values) == (21, 2)
        @test length(ds.test_truth) == 2
        @test all(>(0), ds.train_truth)
        @test all(all(>(0), t) for t in ds.test_truth)
        separate = reduce(hcat, [mv_truth(case, [ic], ds.train_times) for ic in MV_TRAIN_ICS])
        # Measured stacked/separate differences <=1.64e-12 across both
        # fixtures; 1e-8 gives >6000x headroom for adaptive-step variation.
        @test maximum(abs, ds.train_truth-separate) < 1e-8
        for (model, count) in zip(MV_MODELS, (64, 9, 65, 64, 63))
            a = mv_make_approximator(model, 10101, ds.train_values)
            @test nparams(a) == count
            @test length(initial_params(a)) == count
            @test sum(length(r) for (_, r) in penalty_blocks(a)) == count
            f = build_evaluator(a, initial_params(a))
            # Measured warm-start error <=6e-17 at (1,1); >1000x headroom.
            @test isapprox(f(1.0, 1.0), MV_INITIAL_ATTACK; atol=1e-13)
            calls = Ref(0)
            prob = mv_problem(a, MV_TRAIN_ICS, ds.train_times, ds.train_values, calls)
            @test prob.data_weights == repeat(reshape(repeat(1 ./ MV_SCALES.^2, 3), 1, :), 21, 1)
            # The two weighted-RSS reductions agree exactly in both fixtures;
            # 1e-12 allows thousands of Float64 ulps for accumulation order.
            @test isapprox(sqrt(PSM.weighted_data_loss(prob, ds.train_truth)/length(ds.train_truth)),
                           mv_scaled_rmse(ds.train_truth, ds.train_values); rtol=1e-12)
        end

        stats = mv_index_stats(ds.train_values)
        a = mv_make_approximator("single_index9", 10101, ds.train_values)
        @test a.approx.mu == stats[1]
        @test a.approx.Sigma == stats[2]
        @test length(penalty_blocks(a)) == 2
    end

    # A bounding rectangle is not joint trajectory support.
    cloud = [0.0 0.0; 3.0 2.5]
    points = [0.0 0.0; 0.0 2.5]
    distances = support_distances(points, cloud)
    @test distances == [0.0, 1.0]

    for point in ([0.8, 0.7], [1.8, 1.1])
        g = FD.gradient(x -> true_attack("single_index", x[1], x[2]), point)
        # Measured error in the closed-form ratio is zero at both points;
        # 1e-13 allows hundreds of Float64 ulps.
        @test isapprox(g[2], 0.8g[1]; atol=1e-13)
    end
    ratios = map(([0.8, 0.7], [1.8, 1.1])) do point
        g = FD.gradient(x -> true_attack("interaction", x[1], x[2]), point)
        g[2]/g[1]
    end
    @test ratios[1] != ratios[2]

    ds = mv_dataset("interaction", 101)
    wrapped = mv_make_approximator("mlp65", 10101, ds.train_values)
    a = wrapped.approx
    beta = copy(a.beta)
    beta[49:64] .= collect(range(-0.2, 0.3, length=16))
    f = build_evaluator(a, beta)
    x = [1.2, 0.7]
    normalized = [2x[1]/3-1, 2x[2]/2.5-1]
    ps = PSM.ComponentArray(beta, a.axes)
    _, st = Lux.setup(Random.Xoshiro(10101), a.model)
    reference = only(first(Lux.apply(a.model, reshape(normalized, 2, 1), ps, st)))
    # Measured Dense/Lux discrepancy is zero; 1e-12 allows thousands of
    # Float64 ulps rather than tying the comparison to accumulation order.
    @test isapprox(f(x...), reference; atol=1e-12, rtol=1e-12)
    grad = FD.gradient(v -> f(v...), x)
    @test all(!iszero, grad)
    h = 1e-5
    numerical = [(f((x+h.*(1:2 .== j))...) - f((x-h.*(1:2 .== j))...))/(2h) for j in 1:2]
    # Measured two-input finite-difference error is 1.21e-11; >800x headroom.
    @test isapprox(grad, numerical; atol=1e-8, rtol=1e-8)
    @test norm(FD.gradient(b -> build_evaluator(a, b)(x...), beta)) > 0

    additive = mv_make_approximator("kan_additive64", 10101, ds.train_values)
    composed = mv_make_approximator("kan63", 10101, ds.train_values)
    ba = 0.1sin.(collect(1:nparams(additive)))
    bc = 0.1sin.(collect(1:nparams(composed)))
    fa, fc = build_evaluator(additive, ba), build_evaluator(composed, bc)
    @test FD.hessian(v -> fa(v...), x)[1, 2] == 0
    @test FD.hessian(v -> fc(v...), x)[1, 2] != 0

    @testset "Selection and diagnostic semantics" begin
        valid = (; status="ok", validation_rmse=2.0)
        @test mv_better_candidate(valid, nothing)
        @test mv_better_candidate(merge(valid, (; validation_rmse=1.0)), valid)
        @test !mv_better_candidate(valid, valid)
        @test !mv_better_candidate(merge(valid, (; validation_rmse=3.0)), valid)
        @test !mv_better_candidate((; status="fit_failed", validation_rmse=0.0), valid)
        @test !mv_better_candidate(merge(valid, (; validation_rmse=NaN)), nothing)
        @test !mv_better_candidate(merge(valid, (; validation_rmse=Inf)), nothing)
        @test all(ismissing, values(mv_diagnostics(nothing)))
        fit = (; edf=4.0, convergence=(; converged=true, stationarity=1e26,
            smoothing_advanced=false, laml_failures=3, criterion=:working, laml=-100.0))
        diag = mv_diagnostics(fit)
        @test diag.converged && diag.stationarity == 1e26 && !diag.smoothing_advanced
        @test diag.laml_failures == 3 && diag.criterion_value == -100.0 && diag.edf == 4.0
        @test ismissing(mv_diagnostics((; edf=4.0, convergence=(; converged=false))).stationarity)
    end

    @testset "Independent failures and seed aggregation" begin
        fail_at_start = AssessmentProbe(:g, (N, P) ->
            (N, P) == first(MV_TEST_ICS) ? throw(DomainError(N, "intentional failed trajectory")) : 0.3)
        failed = mv_score_trajectory(fail_at_start, [0.3], ds, ds.case, 1, MV_SUPPORT_RADIUS)
        other = mv_score_trajectory(fail_at_start, [0.3], ds, ds.case, 2, MV_SUPPORT_RADIUS)
        @test failed.test_status == "test_failed"
        @test failed.function_status == "response_failed"
        @test occursin("intentional failed trajectory", failed.test_message)
        @test other.test_status == other.function_status == "ok"
        @test isfinite(other.full_rmse) && isfinite(other.response_rmse)

        points = Set(Tuple(row) for row in eachrow(ds.test_truth[1]))
        fail_off_reference = AssessmentProbe(:g, (N, P) ->
            (N, P) in points ? 0.3 : throw(DomainError(N, "intentional off-reference failure")))
        scored = mv_score_trajectory(fail_off_reference, [0.3], ds, ds.case, 1, MV_SUPPORT_RADIUS)
        @test scored.test_status == "test_failed"
        @test scored.function_status == "ok" && isfinite(scored.response_rmse)
        @test isnan(scored.near_response_rmse) && scored.support_fraction == 0

        cfg = (; cases=["case"], models=["model"], seeds=[101, 102], track="adam", split="forecast")
        common = (; case="case", model="model", seed=101, test_status="ok",
                    function_status="ok", near_response_rmse=2.0,
                    negative_fraction=0.0, support_fraction=0.5)
        trajectories = [merge(common, (; trajectory_id=1, full_rmse=3.0, response_rmse=6.0)),
                        merge(common, (; trajectory_id=2, full_rmse=4.0, response_rmse=8.0,
                                        near_response_rmse=NaN, support_fraction=0.0))]
        complete, absent = mv_seed_metrics(trajectories, cfg)
        @test complete.full_rmse == sqrt(12.5)
        @test complete.response_rmse == sqrt(50.0)
        @test complete.near_response_rmse == 2.0
        @test complete.support_fraction == 0.25
        @test complete.test_trajectories == complete.response_trajectories == 2
        @test absent.full_rmse == absent.response_rmse == Inf
        @test absent.test_trajectories == absent.response_trajectories == 0
        partial = mv_seed_metrics(trajectories[1:1], cfg)[1]
        @test partial.full_rmse == partial.response_rmse == Inf
        @test_throws ArgumentError mv_seed_metrics([trajectories; trajectories[1]], cfg)
    end

    @testset "Separate local split and scoring" begin
        @test mv_options(String[]).split == "forecast"
        @test mv_options(String[]).starts == "constant"
        @test mv_options(String[]).seeds == collect(101:110)
        @test mv_options(["--split=local"]).seeds == collect(201:210)
        @test mv_options(["--split=local", "--seeds=17"]).seeds == [17]
        @test_throws ArgumentError mv_options(["--split=unknown"])
        geometry = mv_split("forecast")
        geometry.train_ics[1] = (0.0, 0.0)
        @test first(mv_split("forecast").train_ics) == first(MV_TRAIN_ICS)

        for case in MV_CASES
            forecast = mv_dataset(case, 17)
            local_ds = mv_dataset(case, 17; split="local")
            @test forecast.train_ics == MV_TRAIN_ICS
            @test forecast.validation_ics == MV_VALIDATION_ICS
            @test forecast.test_ics == MV_TEST_ICS
            @test last(forecast.test_times) == 8.0
            @test local_ds.train_truth == forecast.train_truth
            @test local_ds.train_values == forecast.train_values
            @test local_ds.validation_ics == [(1.45, 0.65)]
            @test local_ds.test_ics == [(0.45, 0.33), (2.5, 1.7)]
            @test local_ds.validation_truth != forecast.validation_truth
            @test local_ds.validation_values != forecast.validation_values
            @test local_ds.test_times == collect(0.0:0.1:6.0)
            @test all(size(t) == (61, 2) for t in local_ds.test_truth)
            @test isempty(intersect(Set(local_ds.train_ics), Set(local_ds.validation_ics)))
            @test isempty(intersect(Set(local_ds.train_ics), Set(local_ds.test_ics)))
            @test isempty(intersect(Set(local_ds.validation_ics), Set(local_ds.test_ics)))
            geometry = mv_geometry(local_ds, MV_SUPPORT_RADIUS)
            @test length(geometry) == 3
            @test all(r.split == "local" for r in geometry)
            # Geometry-only measurement: minimum coverage 59/61 = 96.72%;
            # the 95% design gate allows one additional unsupported sample.
            @test all(r.support_fraction >= 0.95 for r in geometry)
            @test all(r.supported_samples/r.samples == r.support_fraction for r in geometry)

            truth = AssessmentProbe(:g, (N, P) -> true_attack(case, N, P))
            for id in eachindex(local_ds.test_ics)
                score = mv_score_trajectory(truth, [0.3], local_ds, case, id, MV_SUPPORT_RADIUS)
                @test (score.initial_N, score.initial_P) == local_ds.test_ics[id]
                @test score.test_status == score.function_status == "ok"
                @test score.response_rmse == 0
                # Measured worst true-RHS trajectory NRMSE is 7.35e-10;
                # 1e-7 leaves >130x headroom for adaptive integration.
                @test score.full_rmse < 1e-7
            end
        end
    end

    @testset "Native slope/loading initialization grid" begin
        starts = mv_initializations("single_index9", "index-multistart")
        @test length(starts) == 11
        @test first(starts) == MV_CONSTANT_START
        @test allunique(getproperty.(starts, :name))
        @test Set((s.loading, s.slope) for s in starts[2:end]) ==
              Set((a, s) for a in (-2.0, -1.0, 0.0, 1.0, 2.0) for s in (-0.04, 0.04))
        @test mv_initializations("tensor64", "index-multistart") == [MV_CONSTANT_START]
        @test all(mv_initializations(model, "constant") == [MV_CONSTANT_START] for model in MV_MODELS)
        @test_throws ArgumentError mv_options(["--starts=index-multistart"])
        @test_throws ArgumentError mv_options(["--track=laml", "--models=kan63", "--starts=index-multistart"])
        @test_throws ArgumentError mv_options(["--track=gcv", "--models=single_index9", "--starts=unknown"])
        @test_throws ArgumentError mv_make_approximator("mlp65", 10017, ds.train_values; start=starts[2])

        for case in MV_CASES
            local_ds = mv_dataset(case, 17; split="local")
            default = mv_make_approximator("single_index9", 10017, local_ds.train_values)
            points = local_ds.training_points
            for start in starts
                a = mv_make_approximator("single_index9", 10017, local_ds.train_values; start)
                beta = initial_params(a)
                f = build_evaluator(a, beta)
                loadings = [1.0, start.loading]
                denominator = sqrt(dot(loadings, a.approx.Sigma*loadings))
                expected = [MV_INITIAL_ATTACK + start.slope*dot(loadings, collect(x)-a.approx.mu)/denominator
                            for x in eachrow(points)]
                @test nparams(a) == 9
                @test index_loadings(a.approx, beta) == loadings
                @test a.approx.mu == default.approx.mu && a.approx.Sigma == default.approx.Sigma
                @test penalty_blocks(a) == penalty_blocks(default)
                # Measured affine-spline discrepancy <=1.12e-16 across
                # both fixtures and all eleven starts; >8000x headroom.
                @test maximum(abs, f.(points[:, 1], points[:, 2])-expected) < 1e-12
                if start == MV_CONSTANT_START
                    @test beta == initial_params(default)
                else
                    J = FD.jacobian(b -> build_evaluator(a, b).(points[:, 1], points[:, 2]), beta)
                    # Measured minimum loading-column norm 0.0395; the
                    # 0.001 gate leaves >39x headroom and rejects a flat start.
                    @test norm(J[:, 1]) > 1e-3
                end
                # Changing only initialization must not change the evaluator's
                # geometry when replaying the same fitted coefficients.
                replay_beta = 0.1sin.(collect(1:9))
                @test build_evaluator(a, replay_beta)(1.2, 0.7) ==
                      build_evaluator(default, replay_beta)(1.2, 0.7)
            end
        end
    end

    @testset "Persisted native multistart selection" begin
        threads = BLAS.get_num_threads()
        try
            for track in ("laml", "gcv")
                mktempdir() do dir
                    args = ["--split=local", "--seeds=17", "--cases=single_index",
                            "--models=single_index9", "--track=$track",
                            "--starts=index-multistart", "--iterations=1", "--output=$dir"]
                    run = mv_main(args)
                    @test length(run.trials) == 11
                    @test all(r.split == "local" && r.track == track for r in run.trials)
                    @test Set(r.initialization for r in run.trials) == Set(s.name for s in MV_INDEX_STARTS)
                    valid = filter(r -> r.status == "ok" && isfinite(r.validation_rmse), run.trials)
                    @test !isempty(valid)
                    winner = valid[argmin(getproperty.(valid, :validation_rmse))]
                    selected = only(run.selections)
                    @test selected.selection_status == "ok"
                    @test all(isequal(getproperty(selected, k), v) for (k, v) in pairs(winner))
                    @test allunique(r.candidate_id for r in run.trials)
                    tuning = 0.0
                    for row in run.trials
                        isfinite(row.fit_seconds) && (tuning += row.fit_seconds)
                    end
                    @test selected.tuning_seconds == tuning
                    constant = only(filter(r -> r.initialization == "constant", run.trials))
                    @test selected.validation_rmse <= constant.validation_rmse
                    @test length(run.trajectories) == 2
                    @test all(r.initialization == selected.initialization for r in run.trajectories)
                    @test all(r.initial_slope == selected.initial_slope for r in run.trajectories)
                    saved = TOML.parsefile(joinpath(dir, "coefficients.toml"))
                    weights = saved["single_index__single_index9__17"]
                    @test weights["initialization"] == selected.initialization
                    @test weights["candidate_id"] == selected.candidate_id
                    @test length(weights["parameters"]) == 9
                    candidates = TOML.parsefile(joinpath(dir, "candidate_coefficients.toml"))
                    @test Set(keys(candidates)) == Set(r.candidate_id for r in run.trials)
                    @test all(candidates[r.candidate_id]["status"] == r.status for r in run.trials)
                    @test candidates[selected.candidate_id]["parameters"] == weights["parameters"]
                    meta = TOML.parsefile(joinpath(dir, "metadata.toml"))
                    @test meta["split"] == "local" && meta["starts"] == "index-multistart"
                    @test meta["test_ics"] == [[0.45, 0.33], [2.5, 1.7]]
                    @test meta["test_end"] == 6.0
                    @test length(meta["initializations"]["single_index9"]) == 11
                    @test meta["assessment_script_sha256"] ==
                          bytes2hex(SHA.sha256(read(joinpath(dir, "assessment-script.jl"))))
                    @test isfile(joinpath(dir, "geometry.csv"))
                    @test isfile(joinpath(dir, "seed_metrics.csv"))
                    replay = mv_replay_main(["--input=$dir", "--output=$(joinpath(dir, "constant"))"])
                    @test length(replay.trajectories) == 2
                    @test all(r.candidate_id == constant.candidate_id for r in replay.trajectories)
                    @test all(r.initialization == "constant" for r in replay.trajectories)
                    if constant.status == "ok"
                        replay_ds = mv_dataset("single_index", 17; split="local")
                        replay_a = mv_make_approximator("single_index9", 10017, replay_ds.train_values)
                        for id in eachindex(replay_ds.test_ics)
                            expected = mv_score_trajectory(replay_a, candidates[constant.candidate_id]["parameters"],
                                                           replay_ds, "single_index", id, MV_SUPPORT_RADIUS)
                            @test isequal(replay.trajectories[id].full_rmse, expected.full_rmse)
                            @test isequal(replay.trajectories[id].response_rmse, expected.response_rmse)
                        end
                    else
                        @test all(r.test_status == "no_valid_candidate" for r in replay.trajectories)
                    end
                    replay_meta = TOML.parsefile(joinpath(dir, "constant", "metadata.toml"))
                    @test !replay_meta["optimizer_refit"]
                    bad = candidates[constant.candidate_id]
                    bad["status"], bad["message"] = "fit_failed", "intentional failed baseline"
                    delete!(bad, "parameters")
                    delete!(bad, "smoothing_params")
                    open(joinpath(dir, "candidate_coefficients.toml"), "w") do io
                        TOML.print(io, candidates)
                    end
                    failed_replay = mv_replay_main(["--input=$dir", "--output=$(joinpath(dir, "failed-constant"))"])
                    @test all(r.test_status == r.function_status == "no_valid_candidate" for r in failed_replay.trajectories)
                    @test all(occursin("intentional failed baseline", r.test_message) for r in failed_replay.trajectories)
                    @test only(failed_replay.seed_rows).test_trajectories == 0
                    @test only(failed_replay.seed_rows).full_rmse == Inf
                    @test_throws ErrorException mv_main(args)
                end
            end
        finally
            BLAS.set_num_threads(threads)
        end
    end
end

end
