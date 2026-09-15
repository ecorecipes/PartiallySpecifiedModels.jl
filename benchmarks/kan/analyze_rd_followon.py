import argparse
import json
import math
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from analyze_multivariate import digest, distribution, finite_median, key, read_csv, rms, same_metric, write_csv


def load_run(path, root):
    meta = tomllib.loads((path / "metadata.toml").read_text())
    assert meta["check_bounds"] and meta["blas_threads"] == 1
    for name, fingerprint in meta["script_sha256"].items():
        assert digest(path / name) == fingerprint
    assert digest(path / "environment-manifest.toml") == meta["manifest_sha256"]
    for archive in meta["input_archives"]:
        for name, fingerprint in archive["files_sha256"].items():
            assert digest(root / archive["path"] / name) == fingerprint
    parity = read_csv(path / "replay_parity.csv")
    assert parity
    parity_groups = defaultdict(list)
    for row in parity:
        parity_groups[row["track"], key(row)].append(row)
        actual, original = float(row["replayed"]), float(row["archived"])
        assert math.isfinite(actual) and math.isfinite(original)
        assert abs(actual-original) <= float(row["atol"]) + float(row["rtol"])*abs(original)
        assert float(row["absolute_difference"]) == abs(actual-original)
    expected_metrics = {(0, "training_rmse"), (0, "validation_rmse")} | {
        (i, m) for i in (1, 2) for m in ("field_rmse", "reaction_rmse", "coefficient_rmse")}
    for rows in parity_groups.values():
        assert len(rows) == 8
        assert {(int(r["profile_id"]), r["metric"]) for r in rows} == expected_metrics
    return meta, parity


def profile_metrics(rows, metrics, profile_key="profile_id"):
    ids = [r[profile_key] for r in rows]
    assert len(set(ids)) == len(ids)
    expected = {"1", "2"} if profile_key == "profile_id" else {"reverse_front", "fine_scale"}
    assert set(ids) <= expected
    def available(row, metric):
        if metric == "field_rmse" and row.get("test_status", "ok") != "ok":
            return False
        if metric in ("reaction_rmse", "coefficient_rmse") and row.get("response_status", "ok") != "ok":
            return False
        return math.isfinite(float(row[metric]))
    return {m: rms([float(r[m]) for r in rows]) if len(rows) == 2 and
            all(available(r, m) for r in rows) else math.inf for m in metrics}


def summarize(rows, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in group_fields)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        row = dict(zip(group_fields, group), seeds=len(values))
        for metric in metrics:
            row.update({metric+"_"+k: v for k, v in distribution(
                [float(r[metric]) for r in values]).items()})
        result.append(row)
    return result


def optimizer(path, root):
    meta, parity = load_run(path, root)
    assert meta["optimizer_refit"]
    design = tomllib.loads((path / "design.toml").read_text())
    assert digest(path / "design.toml") == meta["design_sha256"]
    variants = {v["variant"]: v for v in design["variants"]}
    trials, selected, profiles = [read_csv(path / name) for name in
                                  ("trials.csv", "selected.csv", "test_profiles.csv")]
    old_path = root / meta["input_archives"][0]["path"]
    old_meta = tomllib.loads((old_path / "metadata.toml").read_text())
    original_label = f"original{old_meta['iterations']}"
    old = {key(r): r for r in read_csv(old_path / "selected.csv")}
    candidates = tomllib.loads((path / "candidate_coefficients.toml").read_text())
    coefficients = tomllib.loads((path / "coefficients.toml").read_text())
    expected = {(c, m, s) for c in meta["cases"] for m in meta["models"] for s in meta["seeds"]}
    assert {key(r) for r in selected} == expected and len(selected) == len(expected)
    assert len(trials) == len(expected)*len(variants)
    assert len(parity) == 8*len(expected) and {key(r) for r in parity} == expected
    assert {(key(r), r["variant"]) for r in trials} == {(k, v) for k in expected for v in variants}
    assert set(candidates) == {r["candidate_id"] for r in trials}
    groups, test_groups = defaultdict(list), defaultdict(list)
    for row in trials:
        k = key(row)
        groups[k].append(row)
        assert row["baseline_candidate_id"] == old[k]["candidate_id"]
        assert float(row["eta"]) == float(old[k]["eta"])
        assert float(row["penalty_weight"]) == float(old[k]["penalty_weight"])
        assert int(row["budget"]) == design["iterations"] == meta["iterations"]
        for field, value in variants[row["variant"]].items():
            actual = row[field]
            if isinstance(value, bool):
                assert actual == str(value).lower()
            elif isinstance(value, (float, int)):
                assert float(actual) == value
            else:
                assert actual == value
        saved = candidates[row["candidate_id"]]
        assert saved["status"] == row["status"] and saved["model_seed"] == 20000+k[2]
        if row["status"] == "ok":
            assert all(map(math.isfinite, saved["parameters"]))
            assert len(saved["parameters"]) == int(row["nparams"])
    by_id = {r["candidate_id"]: r for r in trials}
    for row in profiles:
        candidate = by_id[row["candidate_id"]]
        assert key(row) == key(candidate) and row["variant"] == candidate["variant"]
        assert row["profile"] == old_meta["test_profiles"][int(row["profile_id"])-1]
        test_groups[key(row), row["variant"]].append(row)
    metrics = ("field_rmse", "reaction_rmse", "coefficient_rmse")
    seeds = read_csv(path / "condition_seed_metrics.csv")
    assert len(seeds) == len(trials)
    assert {(key(r), r["variant"]) for r in seeds} == {(key(r), r["variant"]) for r in trials}
    lookup = {}
    for row in seeds:
        k = key(row), row["variant"]
        measured = profile_metrics(test_groups[k], metrics)
        assert all(same_metric(float(row[m]), measured[m]) for m in metrics)
        lookup[k] = row
    picked = {key(r): r for r in read_csv(path / "selected_seed_metrics.csv")}
    result = [dict(r) for r in seeds]
    cost_rows = []
    for row in selected:
        k = key(row)
        valid = [r for r in groups[k] if r["status"] == "ok" and math.isfinite(float(r["validation_rmse"]))]
        cost = sum(float(r["fit_seconds"]) for r in groups[k] if math.isfinite(float(r["fit_seconds"])))
        assert same_metric(float(row["tuning_seconds"]), cost)
        assert float(row["prior_tuning_seconds"]) == float(old[k]["tuning_seconds"])
        assert same_metric(float(row["total_search_seconds"]), cost+float(old[k]["tuning_seconds"]))
        if valid:
            winner = min(valid, key=lambda r: float(r["validation_rmse"]))
            assert row["selection_status"] == "ok" and row["candidate_id"] == winner["candidate_id"]
            assert all(row[f] == v for f, v in winner.items())
            saved = coefficients["__".join(map(str, k))]
            assert saved["parameters"] == candidates[row["candidate_id"]]["parameters"]
            assert all(picked[k][m] == lookup[k, row["variant"]][m] for m in metrics)
        else:
            assert row["selection_status"] == "failed"
        result.append(dict(picked[k], variant="validation-selected"))
        cost_rows.append(dict(case=k[0], model=k[1], seed=k[2],
            variant=row["variant"], fit_seconds=float(row["fit_seconds"]),
            tuning_seconds=cost, total_search_seconds=float(row["total_search_seconds"])))
    old_profiles = defaultdict(list)
    for row in read_csv(old_path / "test_profiles.csv"):
        old_profiles[key(row)].append(row)
    for k in sorted(expected):
        result.append(dict(case=k[0], model=k[1], seed=k[2], variant=original_label,
                           **profile_metrics(old_profiles[k], metrics)))
    summary = summarize(result, ("case", "model", "variant"), metrics)
    timing = summarize(cost_rows, ("case", "model"),
                       ("fit_seconds", "tuning_seconds", "total_search_seconds"))
    condition_cost = summarize(trials, ("case", "model", "variant"), ("fit_seconds", "iterations"))
    paired = []
    lookup = {(key(r), r["variant"]): r for r in result}
    for case in meta["cases"]:
        for variant in [*variants, original_label, "validation-selected"]:
            for baseline in meta["models"]:
                if baseline == "kan28" or "kan28" not in meta["models"]:
                    continue
                for metric in metrics:
                    pairs = [(float(lookup[(case, "kan28", s), variant][metric]),
                              float(lookup[(case, baseline, s), variant][metric])) for s in meta["seeds"]]
                    finite = [(a,b) for a,b in pairs if math.isfinite(a) and math.isfinite(b)]
                    paired.append(dict(case=case, variant=variant, baseline=baseline, metric=metric,
                        finite_pairs=len(finite), kan_wins=sum(a<b for a,b in finite),
                        ties=sum(a==b for a,b in finite),
                        median_kan_to_baseline_ratio=finite_median(a/b for a,b in finite if b>0)))
    print("Optimizer candidates:", Counter(r["status"] for r in trials),
          "selections:", Counter(r["selection_status"] for r in selected))
    print("Optimizer same-mesh maximum discrepancy:", max(float(r["absolute_difference"]) for r in parity))
    return dict(optimizer_summary=summary, optimizer_cost=timing,
                optimizer_condition_cost=condition_cost, optimizer_pairs=paired,
                optimizer_seeds=result, optimizer_selected_cost=cost_rows)


def transfer(path, root):
    meta, parity = load_run(path, root)
    assert not meta["optimizer_refit"]
    rows, seeds = [read_csv(path / name) for name in ("test_profiles.csv", "seed_metrics.csv")]
    metrics = ("field_rmse", "projected_model_rmse", "projected_to_finest_truth_rmse",
               "truth_mesh_rmse", "learned_mesh_shift_rmse", "learned_to_finest_mesh_rmse")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["track"], *key(row), int(row["cells"]))].append(row)
    expected = set()
    selected_ids = {}
    for archive in meta["input_archives"]:
        original = tomllib.loads((root / archive["path"] / "metadata.toml").read_text())
        expected.update((original["track"], c, m, s, n) for c in original["cases"]
                        for m in original["models"] for s in original["seeds"] for n in meta["cells"])
        selected_ids.update({(original["track"], *key(r)): r["candidate_id"]
                             for r in read_csv(root / archive["path"] / "selected.csv")})
    assert set(groups) == expected and len(rows) == 2*len(expected)
    assert len(parity) == 8*len(selected_ids)
    assert {(r["track"], *key(r)) for r in parity} == set(selected_ids)
    for row in rows:
        assert row["candidate_id"] == selected_ids[row["track"], *key(row)]
    assert len(seeds) == len(expected)
    assert {(r["track"], *key(r), int(r["cells"])) for r in seeds} == expected
    for row in seeds:
        group = groups[(row["track"], *key(row), int(row["cells"]))]
        measured = profile_metrics(group, metrics, profile_key="profile")
        assert all(same_metric(float(row[m]), measured[m]) for m in metrics)
    for row in rows:
        if row["status"] == "ok":
            if int(row["cells"]) == min(meta["cells"]):
                assert float(row["learned_mesh_shift_rmse"]) == 0.0
            if int(row["cells"]) == max(meta["cells"]):
                assert float(row["learned_to_finest_mesh_rmse"]) == float(row["truth_mesh_rmse"]) == 0.0
    print("Transferred profiles:", Counter(r["status"] for r in rows))
    print("Transfer same-mesh maximum discrepancy:", max(float(r["absolute_difference"]) for r in parity))
    return dict(mesh_summary=summarize(seeds, ("track", "case", "model", "cells"), metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=Path, default=Path("benchmarks/kan/results/reaction-diffusion-optimizer"))
    parser.add_argument("--transfer", type=Path, default=Path("benchmarks/kan/results/reaction-diffusion-transfer"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/kan/results/reaction-diffusion-followon-analysis"))
    args = parser.parse_args()
    root = Path.cwd()
    metadata = [tomllib.loads((p / "metadata.toml").read_text()) for p in (args.optimizer, args.transfer)]
    assert metadata[0]["evaluation_source_sha256"] == metadata[1]["evaluation_source_sha256"]
    assert metadata[0]["manifest_sha256"] == metadata[1]["manifest_sha256"]
    tables = optimizer(args.optimizer, root) | transfer(args.transfer, root)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        if rows:
            write_csv(args.output / f"{name}.csv", rows)
    script, helper = Path(__file__), Path(__file__).with_name("analyze_multivariate.py")
    for file in (script, helper):
        (args.output / file.name).write_bytes(file.read_bytes())
    fingerprints = {str(p.relative_to(root)): digest(p) for directory in (args.optimizer, args.transfer)
                    for p in directory.resolve().iterdir() if p.is_file()}
    (args.output / "metadata.json").write_text(json.dumps(dict(
        model_evaluation=False, optimizer_refit=False, inputs_sha256=fingerprints,
        analysis_sha256=digest(script), helper_sha256=digest(helper),
        aggregation="Pool squared errors across both profiles within seed, then summarize independent seeds"),
        indent=2)+"\n")


if __name__ == "__main__":
    main()
