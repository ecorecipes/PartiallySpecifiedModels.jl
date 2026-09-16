import argparse
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from analyze_calibration import close, fit_key, load_cohort, mcse, summaries
from analyze_multivariate import digest, write_csv
from compare_uncertainty_methods import extend_intervals


def procedure_ok(procedure,fraction):
    return (procedure["name"]=="select_then_fixed_refit" and
            procedure["fraction"]==fraction and
            math.isfinite(procedure["selected_lambda"]) and procedure["selected_lambda"]>0 and
            procedure["final_lambda"]==fraction*procedure["selected_lambda"])


def paired(seed_rows,specs):
    lookup={(r["case"],r["model"],r["method"],r["budget"],r["level"],r["region"],r["seed"]):r
            for r in seed_rows}
    controls={spec["base_model"]:model for model,spec in specs.items() if spec["smoothing_fraction"]==1.0}
    groups=defaultdict(list)
    for key,row in lookup.items():
        case,model,method,budget,level,region,seed=key
        spec=specs[model]
        if spec["smoothing_fraction"]==1:
            continue
        ref_model=controls[spec["base_model"]]
        ref=lookup.get((case,ref_model,method,budget,level,region,seed))
        if ref is None:
            raise ValueError("undersmoothing comparison lacks the paired same-strength control")
        groups[(case,spec["base_model"],spec["smoothing_fraction"],method,budget,level,region)].append(
            row["available_and_covers"]-ref["available_and_covers"])
    names=("case","base_model","fraction","method","budget","level","region")
    return [dict(zip(names,key),paired_datasets=len(values),
                 coverage_yield_difference=statistics.mean(values),mcse=mcse(values))
            for key,values in sorted(groups.items())]


def analyze(inputs,output,truth_function=None,extra_scripts=()):
    intervals,seen,fingerprints,specs,status,base_lambdas=[],set(),{}, {},[],{}
    observations={}
    first_meta=first_design=None
    for name in inputs:
        path=Path(name).resolve()
        design,meta,fits,boots,rows,diags=load_cohort(path,truth_function=truth_function)
        if first_meta is None:
            first_meta,first_design=meta,design
        else:
            for field in ("evaluation_source_sha256","manifest_sha256","script_sha256"):
                assert meta[field]==first_meta[field]
            for field in ("seeds","cases","nboot","iterations","query_points","levels"):
                assert design[field]==first_design[field]
        assert not set(design["seeds"]) & (set(range(1001,1101))|set(range(2001,2101)))
        for file in path.glob("datasets/*.toml"):
            fingerprint=digest(file)
            if file.name in observations:
                assert observations[file.name]==fingerprint, "paired observations differ"
            else:
                observations[file.name]=fingerprint
        for model,spec in design["family_specs"].items():
            assert model not in specs
            specs[model]=spec
        for fit in fits:
            key=fit_key(fit)
            assert key not in seen
            seen.add(key)
            spec=specs[key[1]]
            record=tomllib.loads((path/"replicates"/("__".join(map(str,key))+".toml")).read_text())
            if fit["fit_status"]=="ok":
                assert procedure_ok(record["procedure"],spec["smoothing_fraction"])
                assert record["smoothing_params"]==[record["procedure"]["final_lambda"]]
                pair=key[0],spec["base_model"],key[2]
                if pair in base_lambdas:
                    assert close(base_lambdas[pair],record["procedure"]["selected_lambda"])
                else:
                    base_lambdas[pair]=record["procedure"]["selected_lambda"]
        for row in diags:
            assert procedure_ok(row["procedure"],specs[row["model"]]["smoothing_fraction"])
            assert not row["smoothing_advanced"]
        intervals.extend(extend_intervals(path,rows))
        for case in design["cases"]:
            for model in design["models"]:
                f=[r for r in fits if r["case"]==case and r["model"]==model]
                d=[r for r in diags if r["case"]==case and r["model"]==model]
                b=[r for r in boots if r["case"]==case and r["model"]==model]
                status.append(dict(case=case,model=model,datasets=len(f),
                    base_status=dict(Counter(r["fit_status"] for r in f)),
                    final_stopping=dict(Counter(r["reason"] for r in f)),
                    attempted=sum(int(r["attempted"]) for r in b),
                    finite=sum(int(r["n_success"]) for r in b),
                    selection_smoothing=dict(Counter(str(r["procedure"]["selection_smoothing_advanced"]) for r in d)),
                    selection_stopping=dict(Counter(r["procedure"]["selection_reason"] for r in d)),
                    final_refit_stopping=dict(Counter(r["reason"] for r in d))))
        for file in path.rglob("*"):
            if file.is_file():
                key=file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
                fingerprints[str(key)]=digest(file)
    points,regions,seeds=summaries(intervals)
    comparisons=paired(seeds,specs)
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("intervals.csv",intervals),("pointwise.csv",points),("regions.csv",regions),
                      ("dataset_regions.csv",seeds),("paired_fractions.csv",comparisons)):
        write_csv(output/name,rows)
    (output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    scripts=[Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
             Path(__file__).with_name("compare_uncertainty_methods.py"),
             Path(__file__).with_name("analyze_multivariate.py"),*extra_scripts]
    for script in scripts:
        (output/script.name).write_bytes(script.read_bytes())
    (output/"metadata.json").write_text(json.dumps(dict(
        model_evaluation=False,optimizer_refit=False,inputs_sha256=fingerprints,
        scripts_sha256={p.name:digest(p) for p in scripts},family_specs=specs,
        scope="Independent-data confirmation of two-stage fixed fractions; no truth-selected fraction",
        bootstrap="Both selection and fixed-lambda coefficient fitting repeated for every refit",
        diagnostics="Final smoothing is intentionally fixed; selection-stage diagnostics stored separately"),
        indent=2)+"\n")
    print("Compared",len(intervals),"intervals from",len(seen),"two-stage original fits.")
    print(json.dumps(status,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",default=",".join(
        f"benchmarks/kan/results/undersmoothing-{model}-{fraction}"
        for model in ("spline8","kan12_free") for fraction in ("1","quarter")))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/undersmoothing-analysis"))
    args=parser.parse_args()
    analyze(args.inputs.split(","),args.output)


if __name__=="__main__":
    main()
