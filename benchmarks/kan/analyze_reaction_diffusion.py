import argparse
import json
import math
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from analyze_multivariate import (
    digest, distribution, finite_median, key, number, read_csv, rms, same_metric, write_csv,
)


def analyze_track(root, track):
    archive = root / f"benchmarks/kan/results/reaction-diffusion-{track}"
    files = ("metadata.toml", "assessment-script.jl", "trials.csv", "selected.csv",
             "test_profiles.csv", "seed_metrics.csv", "coefficients.toml",
             "candidate_coefficients.toml", "mesh_refinement.csv")
    fingerprints = {str((archive/name).relative_to(root)): digest(archive/name) for name in files}
    meta = tomllib.loads((archive/"metadata.toml").read_text())
    assert meta["track"] == track
    assert meta["assessment_script_sha256"] == digest(archive/"assessment-script.jl")
    trials, selected, profiles = [read_csv(archive/name) for name in files[2:5]]
    published = {key(row): row for row in read_csv(archive/"seed_metrics.csv")}
    weights = tomllib.loads((archive/"coefficients.toml").read_text())
    candidates = tomllib.loads((archive/"candidate_coefficients.toml").read_text())
    groups, scored = defaultdict(list), defaultdict(list)
    for row in trials:
        assert row["track"] == track and int(row["cells"]) == meta["cells"]
        groups[key(row)].append(row)
    for row in profiles:
        scored[key(row)].append(row)
    expected = {(case, model, seed) for case in meta["cases"]
                for model in meta["models"] for seed in meta["seeds"]}
    assert set(groups) == set(published) == {key(row) for row in selected} == expected
    assert len(selected) == len(expected)
    assert set(scored) <= expected
    assert len({r["candidate_id"] for r in trials}) == len(trials)
    assert set(candidates) == {r["candidate_id"] for r in trials}
    eta_key = lambda value: "native" if math.isnan(float(value)) else float(value)
    declared = Counter(eta_key(eta) for eta in meta["etas"])
    for rows in groups.values():
        assert Counter(eta_key(r["eta"]) for r in rows) == declared
    for row in trials:
        saved = candidates[row["candidate_id"]]
        assert saved["status"] == row["status"] and saved["message"] == row["message"]
        assert saved["model_seed"] == 20000+int(row["seed"])
        if row["status"] == "ok":
            assert len(saved["parameters"]) == int(row["nparams"])
            assert all(map(math.isfinite, saved["parameters"]))
    seeds, valid_keys = [], set()
    nprofiles = len(meta["test_profiles"])
    for row in selected:
        trial_rows = groups[key(row)]
        valid = [r for r in trial_rows if r["status"] == "ok" and math.isfinite(float(r["validation_rmse"]))]
        total = 0.0
        for candidate in trial_rows:
            if math.isfinite(float(candidate["fit_seconds"])):
                total += float(candidate["fit_seconds"])
        assert float(row["tuning_seconds"]) == total
        if valid:
            winner = min(valid, key=lambda r: float(r["validation_rmse"]))
            assert row["selection_status"] == "ok"
            assert all(row[k] == v for k, v in winner.items())
            name = "__".join(map(str, key(row)))
            valid_keys.add(name)
            saved = weights[name]
            assert saved["candidate_id"] == row["candidate_id"]
            assert saved["parameters"] == candidates[row["candidate_id"]]["parameters"]
        else:
            assert row["selection_status"] == "failed"
        rows = scored[key(row)]
        ids = [int(r["profile_id"]) for r in rows]
        assert len(set(ids)) == len(ids) and set(ids) <= set(range(1, nprofiles+1))
        for r in rows:
            assert r["candidate_id"] == row["candidate_id"]
            assert r["profile"] == meta["test_profiles"][int(r["profile_id"])-1]
            assert 0 <= float(r["support_fraction"]) <= 1
        nf = sum(r["test_status"] == "ok" and math.isfinite(float(r["field_rmse"])) for r in rows)
        nr = sum(r["response_status"] == "ok" and math.isfinite(float(r["reaction_rmse"])) for r in rows)
        seed = dict(track=track, case=row["case"], model=row["model"], seed=int(row["seed"]),
                    finite_fields=nf, finite_responses=nr)
        for metric in ("field_rmse", "reaction_rmse", "coefficient_rmse"):
            complete = nf == nprofiles if metric == "field_rmse" else nr == nprofiles
            seed[metric] = rms([float(r[metric]) for r in rows]) if complete else math.inf
        supported = [r for r in rows if float(r["support_fraction"]) > 0]
        seed["supported_reaction_rmse"] = math.nan if not supported else (
            math.sqrt(sum(float(r["support_fraction"])*float(r["supported_reaction_rmse"])**2 for r in supported) /
                      sum(float(r["support_fraction"]) for r in supported)) if nr == nprofiles else math.inf)
        for metric in ("field_rmse", "reaction_rmse", "coefficient_rmse", "supported_reaction_rmse"):
            assert same_metric(seed[metric], float(published[key(row)][metric])), (key(row), metric)
        seeds.append(seed)
    assert set(weights) == valid_keys
    print(track, "candidates", Counter(r["status"] for r in trials),
          "selected", Counter(r["selection_status"] for r in selected),
          "fields", Counter(r["test_status"] for r in profiles),
          "responses", Counter(r["response_status"] for r in profiles))
    print("negative-coefficient profiles", sum(float(r["negative_rate_fraction"]) > 0 for r in profiles))
    summaries = []
    for case in meta["cases"]:
        for model in meta["models"]:
            rows = [r for r in seeds if r["case"] == case and r["model"] == model]
            fits = [r for r in selected if r["case"] == case and r["model"] == model]
            summary = dict(track=track, case=case, model=model, seeds=len(rows),
                complete_fields=sum(r["finite_fields"] == nprofiles for r in rows),
                complete_responses=sum(r["finite_responses"] == nprofiles for r in rows))
            for metric in ("field_rmse", "reaction_rmse", "coefficient_rmse", "supported_reaction_rmse"):
                summary.update({metric+"_"+k: v for k, v in distribution([r[metric] for r in rows]).items()})
            for metric in ("fit_seconds", "tuning_seconds", "allocated_bytes", "stationarity"):
                summary[metric+"_median"] = finite_median(number(r, metric) for r in fits)
            summaries.append(summary)
            print("RESULT", summary)
            print("settings", Counter(r["eta"] for r in fits), "reasons", Counter(r["reason"] for r in fits),
                  "smoothing", Counter(r["smoothing_advanced"] for r in fits))
    return seeds, summaries, meta, fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/kan/results/reaction-diffusion-analysis"))
    args = parser.parse_args()
    root = Path.cwd()
    seeds, summaries, metadata, fingerprints = [], [], {}, {}
    for track in ("adam", "laml"):
        rows, summary, meta, inputs = analyze_track(root, track)
        seeds.extend(rows)
        summaries.extend(summary)
        metadata[track] = meta
        fingerprints.update(inputs)
    for field in ("source_sha256", "assessment_script_sha256", "cells", "length", "diffusion",
                  "density_domain", "initial_rate", "observation_sigma", "training_profiles",
                  "validation_profiles", "test_profiles", "train_end", "test_end", "seeds",
                  "cases", "spatial_model", "loss_weights", "package_versions"):
        assert metadata["adam"][field] == metadata["laml"][field], field
    lookup = {(r["case"], r["model"], r["seed"]): r for r in seeds if r["track"] == "adam"}
    paired = []
    for case in metadata["adam"]["cases"]:
        for model in metadata["adam"]["models"]:
            if model == "kan28":
                continue
            for metric in ("field_rmse", "reaction_rmse", "coefficient_rmse"):
                pairs = [(lookup[case, "kan28", seed][metric], lookup[case, model, seed][metric])
                         for seed in metadata["adam"]["seeds"]]
                finite = [(a,b) for a,b in pairs if math.isfinite(a) and math.isfinite(b)]
                row = dict(case=case, baseline=model, metric=metric, finite_pairs=len(finite),
                    kan_wins=sum(a < b for a,b in finite), ties=sum(a == b for a,b in finite),
                    median_kan_to_baseline_ratio=finite_median(a/b for a,b in finite if b > 0))
                paired.append(row)
                print("PAIRED", row)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/"seed_metrics.csv", seeds)
    write_csv(args.output/"model_summary.csv", summaries)
    write_csv(args.output/"paired_comparisons.csv", paired)
    script = Path(__file__)
    helpers = script.with_name("analyze_multivariate.py")
    (args.output/"analysis-script.py").write_bytes(script.read_bytes())
    (args.output/"analyze_multivariate.py").write_bytes(helpers.read_bytes())
    (args.output/"metadata.json").write_text(json.dumps(dict(
        source_sha256=metadata["adam"]["source_sha256"], input_sha256=fingerprints,
        script_sha256=digest(script), helper_sha256=digest(helpers),
        model_evaluation=False, optimizer_refit=False,
        aggregation="Pool squared space-time errors across held-out profiles within each seed; summarize seeds",
        spatial_reference="Same-mesh fitting truth; mesh-refinement errors are separate"), indent=2)+"\n")


if __name__ == "__main__":
    main()
