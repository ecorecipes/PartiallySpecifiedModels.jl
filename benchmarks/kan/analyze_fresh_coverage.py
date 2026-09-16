import argparse
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from analyze_calibration import close, load_cohort, mcse, quantile, summaries
from analyze_multivariate import digest, write_csv
from compare_uncertainty_methods import bootstrap_intervals, extend_intervals


def prefix_intervals(path, original, seeds, budget):
    records = {}
    rows = []
    # One template per point/level. Other archived bootstrap budgets must
    # not duplicate or change the smaller attempted-prefix population.
    templates = [r for r in original if r["case"] == "nonlinear" and r["seed"] in seeds
                 and r["method"] == "covariance"]
    for row in templates:
        rows.append(dict(row))
        key = (row["case"],row["model"],row["seed"])
        if key not in records:
            records[key] = tomllib.loads((path/"replicates"/("__".join(map(str,key))+".toml")).read_text())
        saved = records[key]
        attempts = saved["finite_attempts"]
        columns = saved.get("bootstrap_values",[])
        keep = [i for i,a in enumerate(attempts) if a <= budget]
        values = [columns[i][row["point_id"]-1] for i in keep
                  if math.isfinite(columns[i][row["point_id"]-1])] if columns else []
        alpha = (1-row["level"])/2
        valid = saved["bootstrap_status"] == "ok" and len(keep)>=3 and len(values)>=3
        lo,hi = (quantile(values,alpha),quantile(values,1-alpha)) if valid else (math.nan,math.nan)
        available = valid and all(map(math.isfinite,(lo,hi))) and lo <= hi
        basic = dict(row,method="bootstrap",budget=budget,lower=lo,upper=hi,available=available,
            covered=available and lo <= row["truth"] <= hi,width=hi-lo if available else math.nan)
        rows.append(basic)
        for interval in bootstrap_intervals(row["estimate"],values,row["level"]):
            lo,hi = interval["lower"],interval["upper"]
            ok = available and interval["status"]=="ok" and lo<=hi
            rows.append(dict(basic,method=interval["method"],lower=lo,upper=hi,
                available=ok,covered=ok and lo<=row["truth"]<=hi,width=hi-lo if ok else math.nan))
    return rows


def compare_models(seed_rows,reference_model):
    lookup = {(r["case"],r["model"],r["method"],r["budget"],r["level"],r["region"],r["seed"]):r
              for r in seed_rows}
    groups = defaultdict(list)
    for key,row in lookup.items():
        case,model,method,budget,level,region,seed = key
        if model == reference_model:
            continue
        ref = lookup.get((case,reference_model,method,budget,level,region,seed))
        if ref is None:
            raise ValueError("paired comparison requires the identical reference dataset and budget")
        groups[(case,model,method,budget,level,region)].append(
            row["available_and_covers"]-ref["available_and_covers"])
    names = ("case","model","method","budget","level","region")
    return [dict(zip(names,key),reference_model=reference_model,paired_datasets=len(values),
                 coverage_yield_difference=statistics.mean(values),mcse=mcse(values))
            for key,values in sorted(groups.items())]


def write_scope(path,intervals,reference):
    points,regions,seeds = summaries(intervals)
    path.mkdir()
    for name,rows in (("intervals.csv",intervals),("pointwise.csv",points),("regions.csv",regions),
                      ("dataset_regions.csv",seeds),("paired_models.csv",compare_models(seeds,reference))):
        write_csv(path/name,rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",default=",".join(f"benchmarks/kan/results/{name}" for name in
        ("coverage-fresh-spline8","coverage-fresh-kan12-free","coverage-screen-kan28","coverage-screen-baselines")))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/coverage-fresh-analysis"))
    args = parser.parse_args()
    inputs = [Path(p).resolve() for p in args.inputs.split(",")]
    cohorts,seen,data_hashes,fingerprints = [],set(),{},{}
    first_meta = None
    for path in inputs:
        design,meta,fits,boots,rows,diagnostics = load_cohort(path)
        assert design["fresh_against_original_calibration"]
        assert not set(design["seeds"]) & set(range(1001,1101))
        if first_meta is None:
            first_meta = meta
        else:
            for field in ("evaluation_source_sha256","manifest_sha256","script_sha256"):
                assert meta[field] == first_meta[field]
        for r in fits:
            key = (r["case"],r["model"],int(r["seed"]))
            assert key not in seen
            seen.add(key)
            assert int(r["nparams"]) == design["family_specs"][r["model"]]["parameters"]
        for file in path.glob("datasets/*.toml"):
            assert file.name not in data_hashes or data_hashes[file.name] == digest(file)
            data_hashes[file.name] = digest(file)
        for file in path.rglob("*"):
            if file.is_file():
                name = file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
                fingerprints[str(name)] = digest(file)
        cohorts.append((path,design,fits,boots,rows,diagnostics))
    confirmed = [c for c in cohorts if c[1]["study_stage"] == "confirm"]
    screened = [c for c in cohorts if c[1]["study_stage"] == "screen"]
    assert confirmed and screened
    screen_sets = [set(c[1]["seeds"]) for c in screened]
    assert all(s == screen_sets[0] for s in screen_sets)
    screen_seeds = screen_sets[0]
    screen_budgets = {c[1]["nboot"] for c in screened}
    assert len(screen_budgets) == 1
    budget = next(iter(screen_budgets))
    confirmation = []
    for path,design,_,_,rows,_ in confirmed:
        assert screen_seeds <= set(design["seeds"])
        assert budget <= design["nboot"]
        confirmation.extend(extend_intervals(path,rows))
    matched = []
    for path,design,_,_,rows,_ in cohorts:
        assert "nonlinear" in design["cases"]
        assert budget <= design["nboot"]
        matched.extend(prefix_intervals(path,rows,screen_seeds,budget))
    args.output.mkdir(parents=True,exist_ok=False)
    write_scope(args.output/"confirmation",confirmation,"spline8")
    write_scope(args.output/"matched_screen",matched,"spline8")
    status = []
    for _,design,fits,boots,_,diagnostics in cohorts:
        for model in design["models"]:
            for case in design["cases"]:
                f = [r for r in fits if r["model"]==model and r["case"]==case]
                b = [r for r in boots if r["model"]==model and r["case"]==case]
                d = [r for r in diagnostics if r["model"]==model and r["case"]==case]
                status.append(dict(stage=design["study_stage"],case=case,model=model,datasets=len(f),
                    base_status=dict(Counter(r["fit_status"] for r in f)),
                    base_stopping=dict(Counter(r["reason"] for r in f)),
                    base_smoothing_advanced=dict(Counter(r["smoothing_advanced"] for r in f)),
                    bootstrap_status=dict(Counter(r["boot_status"] for r in b)),
                    attempted=sum(int(r["attempted"]) for r in b),
                    finite=sum(int(r["n_success"]) for r in b),
                    refit_stopping=dict(Counter(r["reason"] for r in d)),
                    refit_smoothing_advanced=dict(Counter(str(r["smoothing_advanced"]) for r in d))))
    (args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    files = [Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
             Path(__file__).with_name("compare_uncertainty_methods.py"),
             Path(__file__).with_name("analyze_multivariate.py")]
    for file in files:
        (args.output/file.name).write_bytes(file.read_bytes())
    (args.output/"metadata.json").write_text(json.dumps(dict(
        optimizer_refit=False,model_evaluation=False,inputs_sha256=fingerprints,
        script_sha256={p.name:digest(p) for p in files},
        matched_screen_seeds=sorted(screen_seeds),matched_screen_bootstrap_budget=budget,
        confirmation="Full fresh confirmation cohorts only",
        screening="All models restricted to the same nonlinear datasets and attempted-prefix budget",
        scope="Explicit different priors/configurations; not a best-attainable family ranking"),
        indent=2)+"\n")
    print("Confirmed",len(confirmation),"intervals; paired screen",len(matched),"intervals.")
    print(json.dumps(status,indent=2))


if __name__ == "__main__":
    main()
