import argparse
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from analyze_calibration import close, summaries
from analyze_multivariate import digest, finite_median, read_csv, write_csv


def group_key(row):
    return row["case"],int(row["seed"]),float(row["nullspace_penalty"])


def analyze(path,output):
    meta = tomllib.loads((path/"metadata.toml").read_text())
    assert meta["exploratory"] and not meta["bootstrap_refit"]
    assert digest(path/"design.toml") == meta["design_sha256"]
    for name,value in meta["script_sha256"].items():
        assert digest(path/name) == value
    for name,value in meta["inputs_sha256"].items():
        assert digest(Path.cwd()/name) == value
    zvalues = {r["level"]:r["z"] for r in meta["normal_quantiles"]}
    trials,selected,raw,parity = [read_csv(path/name) for name in
        ("trials.csv","selected.csv","intervals.csv","replay_parity.csv")]
    coefficients = tomllib.loads((path/"coefficients.toml").read_text())
    groups = defaultdict(list)
    for row in trials:
        groups[group_key(row)].append(row)
    expected_datasets = {(r["case"],int(r["seed"])) for r in trials}
    assert len(parity) == len(expected_datasets)
    assert {(r["case"],int(r["seed"])) for r in parity} == expected_datasets
    assert all(float(r["maximum_response_difference"]) <= 1e-8 and
               float(r["maximum_se_difference"]) <= 1e-8 for r in parity)
    assert len(groups) == 2*len(expected_datasets)
    winners = {}
    for row in selected:
        key = group_key(row)
        assert key not in winners
        rows = groups[key]
        assert len(rows) == 2 and {r["initialization"] for r in rows} == {"original","constant"}
        eligible = [r for r in rows if r["status"] == "ok" and math.isfinite(float(r["laml"]))]
        winner = max(eligible,key=lambda r:float(r["laml"]))
        assert all(row[k] == value for k,value in winner.items())
        assert close(float(row["search_seconds"]),sum(float(r["fit_seconds"]) for r in rows))
        winners[key] = row["initialization"]
    assert set(winners) == set(groups)
    labels = {0.0:"kan12_free_affine",1e-6:"kan12_affine_penalty"}
    priors = {value:key for key,value in labels.items()}
    intervals,seen = [],set()
    for row in raw:
        case,model,seed,method = row["case"],row["model"],int(row["seed"]),row["method"]
        prior = priors[model]
        selected_kind = winners[case,seed,prior]
        kind = selected_kind if method == "laml_selected" else method
        saved = coefficients[f"{case}__{seed}__{model}__{kind}"]
        point,level = int(row["point_id"]),float(row["level"])
        identity = (case,seed,model,method,point,level)
        assert identity not in seen
        seen.add(identity)
        estimate,se = saved["estimates"][point-1],saved["se"][point-1]
        lower,upper,truth = float(row["lower"]),float(row["upper"]),float(row["truth"])
        assert close(float(row["estimate"]),estimate)
        assert close(lower,estimate-zvalues[level]*se) and close(upper,estimate+zvalues[level]*se)
        available = row["status"] == "ok" and all(map(math.isfinite,(lower,upper))) and lower <= upper
        covered = available and lower <= truth <= upper
        assert (row["available"]=="true") == available and (row["covered"]=="true") == covered
        intervals.append(dict(case=case,model=model,seed=seed,method=method,budget=0,level=level,
            point_id=point,x=float(row["x"]),truth=truth,region=row["region"],estimate=estimate,
            lower=lower,upper=upper,available=available,covered=covered,
            width=upper-lower if available else math.nan))
    expected = {(case,seed,model,method,point,level) for case,seed in expected_datasets
                for model in labels.values() for method in ("original","constant","laml_selected")
                for point in range(1,17) for level in zvalues}
    assert seen == expected
    points,regions,seeds = summaries(intervals)
    comparisons = []
    for key,rows in sorted(groups.items()):
        by_start = {r["initialization"]:r for r in rows}
        before,after = by_start["original"],by_start["constant"]
        comparisons.append(dict(case=key[0],seed=key[1],nullspace_penalty=key[2],
            original_laml=float(before["laml"]),constant_laml=float(after["laml"]),
            delta_laml=float(after["laml"])-float(before["laml"]),
            original_edf=float(before["edf"]),constant_edf=float(after["edf"]),
            selected=winners[key],search_seconds=sum(float(r["fit_seconds"]) for r in rows)))
    diagnostics = []
    for case in sorted({r["case"] for r in trials}):
        for prior in labels:
            for kind in ("original","constant"):
                rows = [r for r in trials if r["case"]==case and
                        float(r["nullspace_penalty"])==prior and r["initialization"]==kind]
                diagnostics.append(dict(case=case,nullspace_penalty=prior,initialization=kind,
                    fits=len(rows),finite=sum(r["status"]=="ok" for r in rows),
                    penalty_rank=int(rows[0]["penalty_rank"]),
                    edf_lt_3=sum(float(r["edf"])<3 for r in rows),
                    edf_median=finite_median(float(r["edf"]) for r in rows),
                    stationarity_median=finite_median(float(r["stationarity"]) for r in rows)))
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("pointwise.csv",points),("regions.csv",regions),("dataset_regions.csv",seeds),
                      ("within_prior_pairs.csv",comparisons),("fit_diagnostics.csv",diagnostics)):
        write_csv(output/name,rows)
    scripts = [Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
               Path(__file__).with_name("analyze_multivariate.py")]
    for script in scripts:
        (output/script.name).write_bytes(script.read_bytes())
    fingerprints = {str(p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p):digest(p)
                    for p in path.iterdir() if p.is_file()}
    (output/"metadata.json").write_text(json.dumps(dict(
        model_evaluation=False,optimizer_refit=False,exploratory=True,inputs_sha256=fingerprints,
        scripts_sha256={p.name:digest(p) for p in scripts},
        scope="Covariance intervals for changed starts/priors; no new bootstrap coverage",
        selection="LAML comparison strictly within each prior; no cross-prior evidence ranking"),
        indent=2)+"\n")
    print("Analyzed",len(trials),"diagnostic fits and",len(intervals),"covariance intervals.")
    print("Fit statuses:",dict(Counter(r["status"] for r in trials)))
    print("Maximum original-response replay difference:",max(float(r["maximum_response_difference"]) for r in parity))
    print("Maximum original-SE replay difference:",max(float(r["maximum_se_difference"]) for r in parity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("benchmarks/kan/results/uncertainty-initialization"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/uncertainty-initialization-analysis"))
    args = parser.parse_args()
    analyze(args.input.resolve(),args.output)


if __name__ == "__main__":
    main()
