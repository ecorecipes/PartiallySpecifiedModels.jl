import argparse
import csv
import hashlib
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(root):
    files = [root / "Project.toml", root / "benchmarks/kan/compare.jl",
             root / "benchmarks/kan/Project.toml"]
    files += list((root / "src").glob("*.jl")) + list((root / "ext").glob("*.jl"))
    result = hashlib.sha256()
    for path in sorted(files):
        result.update((str(path.relative_to(root)) + "\0").encode())
        result.update(path.read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def key(row):
    return row["case"], row["model"], int(row["seed"])


def rms(values):
    return math.sqrt(sum(x*x for x in values) / len(values))


def number(row, name):
    value = row.get(name, "missing")
    return math.nan if value == "missing" else float(value)


def finite_median(values):
    finite = [x for x in values if math.isfinite(x)]
    return statistics.median(finite) if finite else math.nan


def same_metric(x, y):
    if x == y or math.isnan(x) and math.isnan(y):
        return True
    # The original archived replay agreed exactly. Allow four ulps for
    # association differences between Julia and Python reductions.
    return math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 4*max(math.ulp(x), math.ulp(y))


def distribution(values):
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return dict(n=0, minimum=math.nan, q25=math.nan, median=math.nan,
                    q75=math.nan, maximum=math.nan)
    if len(finite) == 1:
        q25 = q75 = finite[0]
    else:
        q25, _, q75 = statistics.quantiles(finite, n=4, method="inclusive")
    return dict(n=len(finite), minimum=min(finite), q25=q25,
                median=statistics.median(finite), q75=q75, maximum=max(finite))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, prefix, track, source):
    archive = root / f"benchmarks/kan/results/{prefix}-{track}"
    inputs = {}
    names = ["metadata.toml", "coefficients.toml", "assessment-script.jl",
             "trials.csv", "selected.csv"]
    for name in names:
        inputs[str((archive / name).relative_to(root))] = digest(archive / name)
    meta = tomllib.loads((archive / "metadata.toml").read_text())
    weights = tomllib.loads((archive / "coefficients.toml").read_text())
    assert meta["source_sha256"] == source, "package/benchmark source changed"
    assert meta["assessment_script_sha256"] == digest(archive / "assessment-script.jl")
    trials, selected = [read_csv(archive / name) for name in names[-2:]]
    if (archive / "test_trajectories.csv").exists():
        scored = read_csv(archive / "test_trajectories.csv")
        inputs[str((archive / "test_trajectories.csv").relative_to(root))] = digest(archive / "test_trajectories.csv")
    else:
        assert all(r["selection_status"] == "failed" for r in selected), "missing scored trajectories"
        scored = []
    assert meta["track"] == track and len(meta["test_ics"]) == 2
    groups, scores = defaultdict(list), defaultdict(list)
    for row in trials:
        assert row["track"] == track
        assert row.get("split", "forecast") == meta.get("split", "forecast")
        groups[key(row)].append(row)
    for row in scored:
        assert row["track"] == track
        scores[key(row)].append(row)
    expected = {(case, model, seed) for case in meta["cases"]
                for model in meta["models"] for seed in meta["seeds"]}
    assert set(groups) == {key(row) for row in selected} == expected
    assert len(selected) == len(expected)
    assert set(scores) <= expected
    eta_key = lambda value: "native" if math.isnan(float(value)) else float(value)
    for (_, model, _), rows in groups.items():
        starts = meta.get("initializations", {}).get(model, [{"name": "constant"}])
        declared = Counter((start["name"], eta_key(eta)) for start in starts for eta in meta["etas"])
        actual = Counter((row.get("initialization", "constant"), eta_key(row["eta"])) for row in rows)
        assert declared == actual, (model, declared, actual)
        if "initializations" in meta:
            by_name = {s["name"]: s for s in starts}
            for row in rows:
                start = by_name[row["initialization"]]
                for field in ("loading", "slope"):
                    if field in start:
                        assert float(row["initial_" + field]) == start[field]
                    else:
                        assert row["initial_" + field] == "missing"
    candidate_weights = None
    if (archive / "candidate_coefficients.toml").exists():
        candidate_weights = tomllib.loads((archive / "candidate_coefficients.toml").read_text())
        inputs[str((archive / "candidate_coefficients.toml").relative_to(root))] = digest(archive / "candidate_coefficients.toml")
        assert len({r["candidate_id"] for r in trials}) == len(trials)
        assert set(candidate_weights) == {r["candidate_id"] for r in trials}
        for row in trials:
            saved = candidate_weights[row["candidate_id"]]
            assert saved["status"] == row["status"] and saved["message"] == row["message"]
            assert saved["initialization"] == row["initialization"]
            assert saved["model_seed"] == 10000 + int(row["seed"])
            assert eta_key(saved["eta"]) == eta_key(row["eta"])
            if row["status"] == "ok":
                assert "parameters" in saved
            if "parameters" in saved:
                assert len(saved["parameters"]) == int(row["nparams"])
                assert all(map(math.isfinite, saved["parameters"]))
    geometry = None
    if (archive / "geometry.csv").exists():
        rows = read_csv(archive / "geometry.csv")
        inputs[str((archive / "geometry.csv").relative_to(root))] = digest(archive / "geometry.csv")
        geometry = {(r["case"], int(r["trajectory_id"])): r for r in rows if r["kind"] == "test"}
    valid_weight_keys = set()
    seed_rows = []
    for row in selected:
        group = groups[key(row)]
        candidates = [r for r in group if r["status"] == "ok"
                      and math.isfinite(float(r["validation_rmse"]))]
        tuning_seconds = 0.0
        for trial in group:
            seconds = float(trial["fit_seconds"])
            if math.isfinite(seconds):
                tuning_seconds += seconds
        assert float(row["tuning_seconds"]) == tuning_seconds, key(row)
        if candidates:
            winner = min(candidates, key=lambda r: float(r["validation_rmse"]))
            assert row["selection_status"] == "ok"
            assert all(row[name] == value for name, value in winner.items())
            weight_key = "__".join(map(str, key(row)))
            valid_weight_keys.add(weight_key)
            saved = weights[weight_key]
            assert len(saved["parameters"]) == int(row["nparams"])
            assert saved["model_seed"] == 10000 + int(row["seed"])
            assert all(map(math.isfinite, saved["parameters"]))
            if candidate_weights is not None:
                assert saved["candidate_id"] == row["candidate_id"]
                assert saved["initialization"] == row["initialization"]
                assert candidate_weights[row["candidate_id"]]["parameters"] == saved["parameters"]
                assert candidate_weights[row["candidate_id"]]["smoothing_params"] == saved["smoothing_params"]
        else:
            assert row["selection_status"] == "failed"
        trajectories = sorted(scores[key(row)], key=lambda r: int(r["trajectory_id"]))
        ids = [int(r["trajectory_id"]) for r in trajectories]
        assert len(set(ids)) == len(ids) and set(ids) <= {1, 2}
        for trajectory in trajectories:
            ic = meta["test_ics"][int(trajectory["trajectory_id"])-1]
            assert [float(trajectory["initial_N"]), float(trajectory["initial_P"])] == ic
            if "candidate_id" in row:
                assert trajectory["candidate_id"] == row["candidate_id"]
                assert trajectory["initialization"] == row["initialization"]
            if geometry is not None:
                support = geometry[row["case"], int(trajectory["trajectory_id"])]
                assert trajectory["support_fraction"] == support["support_fraction"]
        nt = sum(r["test_status"] == "ok" and math.isfinite(float(r["full_rmse"]))
                 for r in trajectories)
        nr = sum(r["function_status"] == "ok" and math.isfinite(float(r["response_rmse"]))
                 for r in trajectories)
        supported = [r for r in trajectories if float(r["support_fraction"]) > 0]
        near = math.nan
        if supported:
            near = (math.sqrt(sum(float(r["support_fraction"])*float(r["near_response_rmse"])**2
                                  for r in supported) /
                              sum(float(r["support_fraction"]) for r in supported))
                    if nr == 2 else math.inf)
        seed_rows.append(dict(track=track, case=row["case"], model=row["model"],
                              seed=int(row["seed"]), test_trajectories=nt,
                              response_trajectories=nr,
                              full_rmse=rms([float(r["full_rmse"]) for r in trajectories])
                              if nt == 2 else math.inf,
                              response_rmse=rms([float(r["response_rmse"]) for r in trajectories])
                              if nr == 2 else math.inf,
                              near_response_rmse=near,
                              support_fraction=sum(float(r["support_fraction"]) for r in trajectories)/2
                              if len(trajectories) == 2 else math.nan,
                              negative_fraction=sum(float(r["negative_fraction"]) for r in trajectories)/2
                              if nr == 2 else math.nan))
    assert set(weights) == valid_weight_keys
    if (archive / "seed_metrics.csv").exists():
        published = {key(r): r for r in read_csv(archive / "seed_metrics.csv")}
        inputs[str((archive / "seed_metrics.csv").relative_to(root))] = digest(archive / "seed_metrics.csv")
        assert set(published) == expected
        for row in seed_rows:
            for name in ("full_rmse", "response_rmse", "near_response_rmse",
                         "support_fraction", "negative_fraction"):
                x, y = row[name], float(published[key(row)][name])
                assert same_metric(x, y), (key(row), name, x, y)
    print(track, "candidate statuses:", Counter(r["status"] for r in trials),
          "selection statuses:", Counter(r["selection_status"] for r in selected),
          "test/response statuses:", Counter((r["test_status"], r["function_status"]) for r in scored))
    print("support fractions:", sorted({(r["case"], r["trajectory_id"], r["support_fraction"])
                                        for r in scored}))
    print("negative response trajectories:", sum(float(r["negative_fraction"]) > 0 for r in scored))
    diagnostics = selected
    if (archive / "diagnostics.csv").exists():
        diagnostics = read_csv(archive / "diagnostics.csv")
        inputs[str((archive / "diagnostics.csv").relative_to(root))] = digest(archive / "diagnostics.csv")
    summaries = []
    for case in meta["cases"]:
        for model in meta["models"]:
            seeds = [r for r in seed_rows if r["case"] == case and r["model"] == model]
            fits = [r for r in selected if r["case"] == case and r["model"] == model]
            diag = [r for r in diagnostics if r["case"] == case and r["model"] == model]
            summary = dict(track=track, case=case, model=model, seeds=len(seeds),
                           complete_test_seeds=sum(r["test_trajectories"] == 2 for r in seeds),
                           complete_response_seeds=sum(r["response_trajectories"] == 2 for r in seeds))
            for metric in ("full_rmse", "response_rmse", "near_response_rmse"):
                summary.update({metric + "_" + name: value
                                for name, value in distribution([r[metric] for r in seeds]).items()})
            for metric in ("fit_seconds", "tuning_seconds", "allocated_bytes", "training_rmse", "validation_rmse"):
                summary[metric + "_median"] = finite_median(number(r, metric) for r in fits)
            summaries.append(summary)
            print("\n", track, case, model)
            print("trajectory", distribution([r["full_rmse"] for r in seeds]))
            print("response", distribution([r["response_rmse"] for r in seeds]))
            print("near response", distribution([r["near_response_rmse"] for r in seeds]))
            print("fit/tuning seconds", summary["fit_seconds_median"], summary["tuning_seconds_median"],
                  "allocated MiB", summary["allocated_bytes_median"]/2**20)
            print("iterations", Counter(r["iterations"] for r in fits), "reasons", Counter(r["reason"] for r in fits))
            if track == "adam":
                print("selected eta", Counter(r["eta"] for r in fits))
            if meta.get("starts") == "index-multistart":
                print("selected initialization", Counter(r["initialization"] for r in fits))
            if track == "laml":
                print("stationarity", distribution([number(r, "stationarity") for r in diag]),
                      "smoothing advanced", Counter(r["smoothing_advanced"] for r in diag),
                      "LAML failures", Counter(r["laml_failures"] for r in diag))
    return seed_rows, summaries, meta, inputs


def compare_constants(root, prefix, selected_seeds, metadata):
    selected_by_key = {(r["track"], *key(r)): r for r in selected_seeds}
    records, summaries, inputs = [], [], {}
    for track in ("laml", "gcv"):
        parent = root / f"benchmarks/kan/results/{prefix}-{track}"
        replay = root / f"benchmarks/kan/results/{prefix}-{track}-constant"
        for name in ("metadata.toml", "seed_metrics.csv", "test_trajectories.csv", "replay-script.jl"):
            inputs[str((replay / name).relative_to(root))] = digest(replay / name)
        replay_meta = tomllib.loads((replay / "metadata.toml").read_text())
        assert replay_meta["optimizer_refit"] is False
        assert replay_meta["candidate_subset"] == "constant"
        assert replay_meta["source_sha256"] == metadata[track]["source_sha256"]
        assert replay_meta["assessment_script_sha256"] == metadata[track]["assessment_script_sha256"]
        assert replay_meta["replay_script_sha256"] == digest(replay / "replay-script.jl")
        assert replay_meta["archive_metadata_sha256"] == digest(parent / "metadata.toml")
        assert replay_meta["candidate_coefficients_sha256"] == digest(parent / "candidate_coefficients.toml")
        assert replay_meta["package_versions"] == metadata[track]["package_versions"]
        base = {key(r): r for r in read_csv(replay / "seed_metrics.csv")}
        fits = {key(r): r for r in read_csv(parent / "selected.csv")}
        constant = {key(r): r for r in read_csv(parent / "trials.csv")
                    if r["initialization"] == "constant"}
        assert set(base) == set(fits) == set(constant)
        scores = defaultdict(list)
        for row in read_csv(replay / "test_trajectories.csv"):
            assert row["initialization"] == "constant"
            assert row["candidate_id"] == constant[key(row)]["candidate_id"]
            scores[key(row)].append(row)
        for group, baseline in base.items():
            trial, selected = constant[group], fits[group]
            scored = scores[group]
            assert len(scored) == 2 and {int(r["trajectory_id"]) for r in scored} == {1, 2}
            if trial["status"] == "ok":
                assert selected["selection_status"] == "ok"
                assert float(selected["validation_rmse"]) <= float(trial["validation_rmse"])
            else:
                assert all(r["test_status"] == r["function_status"] == "no_valid_candidate" for r in scored)
            for metric, status in (("full_rmse", "test_status"), ("response_rmse", "function_status")):
                finite = all(r[status] == "ok" and math.isfinite(float(r[metric])) for r in scored)
                expected = rms([float(r[metric]) for r in scored]) if finite else math.inf
                assert same_metric(float(baseline[metric]), expected)
                records.append(dict(
                    track=track, case=group[0], model=group[1], seed=group[2], metric=metric,
                    constant_status=trial["status"], selected_status=selected["selection_status"],
                    selected_initialization=selected["initialization"],
                    constant_error=float(baseline[metric]),
                    selected_error=selected_by_key[track, *group][metric],
                    constant_validation_rmse=float(trial["validation_rmse"]),
                    selected_validation_rmse=float(selected["validation_rmse"]),
                    constant_fit_seconds=float(trial["fit_seconds"]),
                    selected_tuning_seconds=float(selected["tuning_seconds"])))
        for case in metadata[track]["cases"]:
            for model in metadata[track]["models"]:
                for metric in ("full_rmse", "response_rmse"):
                    rows = [r for r in records if r["track"] == track and r["case"] == case
                            and r["model"] == model and r["metric"] == metric]
                    finite = [r for r in rows if math.isfinite(r["constant_error"])
                              and math.isfinite(r["selected_error"])]
                    summary = dict(
                        track=track, case=case, model=model, metric=metric,
                        seeds=len(rows), finite_pairs=len(finite),
                        selected_wins=sum(r["selected_error"] < r["constant_error"] and
                                          not same_metric(r["selected_error"], r["constant_error"]) for r in finite),
                        ties=sum(same_metric(r["selected_error"], r["constant_error"]) for r in finite),
                        median_constant_error=finite_median(r["constant_error"] for r in rows),
                        median_selected_error=finite_median(r["selected_error"] for r in rows),
                        median_selected_to_constant_ratio=finite_median(
                            r["selected_error"]/r["constant_error"] for r in finite if r["constant_error"] > 0),
                        median_constant_fit_seconds=finite_median(r["constant_fit_seconds"] for r in rows),
                        median_selected_tuning_seconds=finite_median(r["selected_tuning_seconds"] for r in rows))
                    summaries.append(summary)
                    print("CONSTANT", summary)
    return records, summaries, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="multivariate",
                        help="Read results/PREFIX-adam, PREFIX-laml and PREFIX-gcv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--constant-baselines", action="store_true",
                        help="Compare saved native constant-start replays with selected starts")
    parser.add_argument("--require-current-source", action="store_true",
                        help="Also require the working tree to match the archived fitting source")
    args = parser.parse_args()
    root = Path.cwd()
    worktree_source = source_hash(root)
    origin = root / f"benchmarks/kan/results/{args.prefix}-adam/metadata.toml"
    source = tomllib.loads(origin.read_text())["source_sha256"]
    if args.require_current_source and source != worktree_source:
        raise RuntimeError("working tree differs from the archived fitting source")
    if source != worktree_source:
        print("Summarizing saved scores from source", source,
              "; current source differs. No model evaluation or refitting is performed.")
    seed_rows, summaries, metadata, inputs = [], [], {}, {}
    for track in ("adam", "laml", "gcv"):
        seeds, summary, meta, files = analyze(root, args.prefix, track, source)
        seed_rows.extend(seeds)
        summaries.extend(summary)
        metadata[track] = meta
        inputs.update(files)
    assert all(meta["package_versions"] == metadata["adam"]["package_versions"] for meta in metadata.values())
    for field in ("cases", "seeds", "training_ics", "validation_ics", "test_ics",
                  "state_domain_spans", "relative_observation_sigma", "support_radius",
                  "split", "test_end"):
        assert all(meta.get(field) == metadata["adam"].get(field) for meta in metadata.values()), field
    paired = []
    adam = {(r["case"], r["model"], r["seed"]): r for r in seed_rows if r["track"] == "adam"}
    for case in metadata["adam"]["cases"]:
        for model in metadata["adam"]["models"]:
            if model == "kan63":
                continue
            for metric in ("full_rmse", "response_rmse"):
                pairs = [(adam[case, "kan63", seed][metric], adam[case, model, seed][metric])
                         for seed in metadata["adam"]["seeds"]]
                finite = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
                ratios = [a/b for a, b in finite if b > 0]
                row = dict(case=case, model=model, metric=metric, finite_pairs=len(finite),
                           kan_wins=sum(a < b for a, b in finite),
                           ties=sum(a == b for a, b in finite),
                           median_kan_to_baseline_ratio=finite_median(ratios),
                           median_kan_minus_baseline=finite_median(a-b for a, b in finite))
                paired.append(row)
                print("PAIRED", row)
    constants = None
    if args.constant_baselines:
        comparisons, constant_summary, files = compare_constants(root, args.prefix, seed_rows, metadata)
        constants = comparisons, constant_summary
        inputs.update(files)
    output = args.output or Path(f"benchmarks/kan/results/{args.prefix}-analysis")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "seed_metrics.csv", seed_rows)
    write_csv(output / "model_summary.csv", summaries)
    write_csv(output / "paired_comparisons.csv", paired)
    if constants is not None:
        write_csv(output / "constant_comparisons.csv", constants[0])
        write_csv(output / "constant_summary.csv", constants[1])
    script = Path(__file__).read_bytes()
    (output / "analysis-script.py").write_bytes(script)
    (output / "metadata.json").write_text(json.dumps(
        dict(prefix=args.prefix, constant_baselines=args.constant_baselines,
             source_sha256=source, input_sha256=inputs,
             analysis_worktree_source_sha256=worktree_source,
             require_current_source=args.require_current_source,
             model_evaluation=False, optimizer_refit=False,
             script_sha256=hashlib.sha256(script).hexdigest(),
             aggregation="RMS of two trajectory errors within seed; median and inclusive IQR across seeds",
             near_support="squared errors weighted by supported point counts",
             paired="same seed; no independence assumed between trajectories and no significance tests"),
        indent=2) + "\n")


if __name__ == "__main__":
    main()
