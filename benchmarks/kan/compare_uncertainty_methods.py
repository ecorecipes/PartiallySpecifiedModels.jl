import argparse
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

from analyze_calibration import close, fit_key, load_cohort, mcse, quantile, summaries
from analyze_multivariate import digest, write_csv


NEW_METHODS = ("bootstrap_basic", "bootstrap_normal", "bootstrap_normal_bc")


def bootstrap_intervals(estimate, samples, level):
    if not 0 < level < 1:
        raise ValueError("interval level must be in (0,1)")
    values = [x for x in samples if math.isfinite(x)]
    status = "ok" if len(values) >= 3 and math.isfinite(estimate) else "unavailable"
    bias = sd = math.nan
    endpoints = {name: (math.nan, math.nan) for name in NEW_METHODS}
    if status == "ok":
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        bias = mean-estimate
        z = statistics.NormalDist().inv_cdf((1+level)/2)
        alpha = (1-level)/2
        endpoints["bootstrap_basic"] = (2*estimate-quantile(values,1-alpha),
                                        2*estimate-quantile(values,alpha))
        endpoints["bootstrap_normal"] = (estimate-z*sd,estimate+z*sd)
        centre = estimate-bias
        endpoints["bootstrap_normal_bc"] = (centre-z*sd,centre+z*sd)
    return [dict(method=name,lower=lo,upper=hi,bootstrap_bias=bias,bootstrap_sd=sd,
                 usable=len(values),status=status if all(map(math.isfinite,(lo,hi))) else "unavailable")
            for name,(lo,hi) in endpoints.items()]


def extend_intervals(path, original):
    records = {}
    result = []
    for row in original:
        item = dict(row,bootstrap_bias=math.nan,bootstrap_sd=math.nan,usable=0,
                    status="ok" if row["available"] else "unavailable")
        if row["method"] != "bootstrap":
            result.append(item)
            continue
        key = fit_key(row)
        if key not in records:
            file = path/"replicates"/("__".join(map(str,key))+".toml")
            records[key] = tomllib.loads(file.read_text())
        saved = records[key]
        attempts = saved["finite_attempts"]
        columns = saved.get("bootstrap_values",[])
        keep = [i for i,attempt in enumerate(attempts) if attempt <= row["budget"]]
        samples = [columns[i][row["point_id"]-1] for i in keep] if columns else []
        extra = bootstrap_intervals(row["estimate"],samples,row["level"])
        item.update(bootstrap_bias=extra[0]["bootstrap_bias"],
                    bootstrap_sd=extra[0]["bootstrap_sd"],usable=extra[0]["usable"])
        result.append(item)
        for interval in extra:
            lower,upper = interval["lower"],interval["upper"]
            available = row["available"] and interval["status"] == "ok" and lower <= upper
            derived = dict(item,**interval)
            derived.update(available=available,
                covered=available and lower <= row["truth"] <= upper,
                width=upper-lower if available else math.nan,
                status="ok" if available else "unavailable")
            result.append(derived)
    return result


def method_pairs(seed_rows, primary_budget):
    lookup = {(r["case"],r["model"],r["method"],r["budget"],r["level"],r["region"],r["seed"]):r
              for r in seed_rows}
    groups = defaultdict(list)
    for key,row in lookup.items():
        case,model,method,budget,level,region,seed = key
        if method == "bootstrap":
            continue
        reference_budget = primary_budget if method == "covariance" else budget
        other = lookup[(case,model,"bootstrap",reference_budget,level,region,seed)]
        groups[(case,model,method,reference_budget,level,region)].append(
            row["available_and_covers"]-other["available_and_covers"])
    names = ("case","model","method","bootstrap_budget","level","region")
    return [dict(zip(names,key),datasets=len(values),
                 mean_coverage_yield_difference=statistics.mean(values),mcse=mcse(values))
            for key,values in sorted(groups.items())]


def compare(inputs, output):
    intervals, inputs_sha, seen = [],{},set()
    first_design = first_meta = None
    for path in inputs:
        design,meta,fits,boots,rows,diagnostics = load_cohort(path)
        if first_design is None:
            first_design,first_meta = design,meta
        else:
            for field in ("nboot","sensitivity_boot","levels","query_points","iterations"):
                assert design[field] == first_design[field]
            assert meta["evaluation_source_sha256"] == first_meta["evaluation_source_sha256"]
        for row in fits:
            assert fit_key(row) not in seen
            seen.add(fit_key(row))
        extended = extend_intervals(path,rows)
        # The existing covariance/percentile procedures are retained verbatim.
        kept = [r for r in extended if r["method"] in ("covariance","bootstrap")]
        assert len(kept) == len(rows)
        assert all(all(close(a[k],b[k]) for k in ("lower","upper","estimate")) and
                   a["available"] == b["available"] and a["covered"] == b["covered"]
                   for a,b in zip(kept,rows))
        intervals.extend(extended)
        for file in path.rglob("*"):
            if file.is_file():
                name = file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
                inputs_sha[str(name)] = digest(file)
    points,regions,seeds = summaries(intervals)
    pairs = method_pairs(seeds,first_design["nboot"])
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("intervals.csv",intervals),("pointwise.csv",points),("regions.csv",regions),
                      ("dataset_regions.csv",seeds),("paired_methods.csv",pairs)):
        write_csv(output/name,rows)
    scripts = [Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
               Path(__file__).with_name("analyze_multivariate.py")]
    for script in scripts:
        (output/script.name).write_bytes(script.read_bytes())
    metadata = dict(model_evaluation=False,optimizer_refit=False,posthoc_method_comparison=True,
        execution_source_sha256=first_meta["evaluation_source_sha256"],inputs_sha256=inputs_sha,
        scripts_sha256={p.name:digest(p) for p in scripts},
        formulas=dict(bootstrap="empirical percentile",
            bootstrap_basic="[2*estimate - q_upper, 2*estimate - q_lower]",
            bootstrap_normal="estimate +/- normal_quantile * bootstrap_sample_sd",
            bootstrap_normal_bc="(2*estimate - bootstrap_mean) +/- normal_quantile * bootstrap_sample_sd"),
        scope="Pointwise intervals; identical retained draws and attempted-prefix budgets; no truth-based tuning",
        excluded="No BCa acceleration or bootstrap-t studentization; these need additional fitted quantities",
        replication_unit="Independent dataset, with query-point clustering")
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Compared",len(intervals),"intervals from",len(seen),"original fits without refitting.")
    return points,regions,seeds,pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",default=",".join(
        f"benchmarks/kan/results/uncertainty-calibration-{case}-{model}"
        for case in ("logistic","nonlinear") for model in ("spline8","kan12")))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/uncertainty-method-comparison"))
    args = parser.parse_args()
    compare([Path(p).resolve() for p in args.inputs.split(",")],args.output)


if __name__ == "__main__":
    main()
