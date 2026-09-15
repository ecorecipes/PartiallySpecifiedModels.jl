import argparse
import json
import math
from pathlib import Path

from analyze_multivariate import analyze, digest, finite_median, key, number, read_csv, same_metric, write_csv
import tomllib


def metadata(root, prefix):
    path = root / f"benchmarks/kan/results/{prefix}-laml/metadata.toml"
    return tomllib.loads(path.read_text())


def diagnostic_rows(root, prefix):
    directory = root / f"benchmarks/kan/results/{prefix}-laml"
    selections = {key(row): row for row in read_csv(directory/"selected.csv")}
    if (directory/"diagnostics.csv").exists():
        for row in read_csv(directory/"diagnostics.csv"):
            selections[key(row)].update(row)
    return selections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/kan/results/laml-correction-analysis"))
    args = parser.parse_args()
    root = Path.cwd()
    records, summaries, fingerprints, sources = [], [], {}, {}
    for split_name, old_prefix, new_prefix in (
        ("forecast", "multivariate", "corrected-forecast"),
        ("local", "local", "corrected-local"),
    ):
        old_meta, new_meta = metadata(root, old_prefix), metadata(root, new_prefix)
        for field in ("seeds", "models", "cases", "iterations", "training_ics",
                      "validation_ics", "test_ics", "state_domain_spans",
                      "relative_observation_sigma", "support_radius", "package_versions"):
            assert old_meta[field] == new_meta[field], (split_name, field)
        assert old_meta.get("split", "forecast") == new_meta["split"] == split_name
        assert old_meta.get("starts", "constant") == new_meta["starts"]
        assert old_meta.get("test_end", 8.0) == new_meta["test_end"]
        if "initializations" in old_meta:
            assert old_meta["initializations"] == new_meta["initializations"]
        old_seeds, _, _, old_files = analyze(root, old_prefix, "laml", old_meta["source_sha256"])
        new_seeds, _, _, new_files = analyze(root, new_prefix, "laml", new_meta["source_sha256"])
        fingerprints.update(old_files)
        fingerprints.update(new_files)
        sources[split_name] = dict(before=old_meta["source_sha256"], after=new_meta["source_sha256"])
        old_diagnostics, new_diagnostics = diagnostic_rows(root, old_prefix), diagnostic_rows(root, new_prefix)
        old_by_key, new_by_key = {key(r): r for r in old_seeds}, {key(r): r for r in new_seeds}
        assert set(old_by_key) == set(new_by_key) == set(old_diagnostics) == set(new_diagnostics)
        for group in old_by_key:
            before, after = old_by_key[group], new_by_key[group]
            bd, ad = old_diagnostics[group], new_diagnostics[group]
            record = dict(
                split=split_name, case=group[0], model=group[1], seed=group[2],
                before_status=bd["selection_status"], after_status=ad["selection_status"],
                before_trajectory=before["full_rmse"], after_trajectory=after["full_rmse"],
                before_response=before["response_rmse"], after_response=after["response_rmse"],
                before_stationarity=number(bd, "stationarity"), after_stationarity=number(ad, "stationarity"),
                before_advanced=bd["smoothing_advanced"] == "true",
                after_advanced=ad["smoothing_advanced"] == "true",
                before_advanced_known=bd["smoothing_advanced"] != "missing",
                after_advanced_known=ad["smoothing_advanced"] != "missing",
                before_fit_seconds=float(bd["fit_seconds"]), after_fit_seconds=float(ad["fit_seconds"]),
                before_tuning_seconds=float(bd["tuning_seconds"]), after_tuning_seconds=float(ad["tuning_seconds"]),
                before_initialization=bd.get("initialization", "constant"),
                after_initialization=ad["initialization"])
            records.append(record)
        for case in new_meta["cases"]:
            for model in new_meta["models"]:
                rows = [r for r in records if r["split"] == split_name and r["case"] == case and r["model"] == model]
                summary = dict(
                    split=split_name, case=case, model=model, seeds=len(rows),
                    before_valid=sum(r["before_status"] == "ok" for r in rows),
                    after_valid=sum(r["after_status"] == "ok" for r in rows),
                    before_advanced=sum(r["before_advanced"] for r in rows),
                    after_advanced=sum(r["after_advanced"] for r in rows),
                    before_advanced_known=sum(r["before_advanced_known"] for r in rows),
                    after_advanced_known=sum(r["after_advanced_known"] for r in rows))
                for metric in ("trajectory", "response", "stationarity", "fit_seconds", "tuning_seconds"):
                    summary["before_"+metric] = finite_median(r["before_"+metric] for r in rows)
                    summary["after_"+metric] = finite_median(r["after_"+metric] for r in rows)
                for metric in ("trajectory", "response"):
                    pairs = [(r["before_"+metric], r["after_"+metric]) for r in rows]
                    finite = [(b, a) for b, a in pairs if math.isfinite(b) and math.isfinite(a)]
                    summary[metric+"_finite_pairs"] = len(finite)
                    summary[metric+"_wins"] = sum(a < b and not same_metric(a, b) for b, a in finite)
                    summary[metric+"_ties"] = sum(same_metric(a, b) for b, a in finite)
                summaries.append(summary)
                print("CORRECTION", summary)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/"paired.csv", records)
    write_csv(args.output/"summary.csv", summaries)
    script = Path(__file__)
    helpers = script.with_name("analyze_multivariate.py")
    (args.output/"compare-script.py").write_bytes(script.read_bytes())
    (args.output/"analyze_multivariate.py").write_bytes(helpers.read_bytes())
    (args.output/"metadata.json").write_text(json.dumps(dict(
        comparison="Same data, solver budgets, initialization grids and package versions; different recorded LAML source",
        sources=sources, input_sha256=fingerprints, script_sha256=digest(script),
        helper_sha256=digest(helpers), model_evaluation=False, optimizer_refit=False,
        aggregation="Pool the two held-out trajectories within seed; compare paired seeds"), indent=2)+"\n")


if __name__ == "__main__":
    main()
