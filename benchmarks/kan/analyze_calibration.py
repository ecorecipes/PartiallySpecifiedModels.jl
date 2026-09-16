import argparse
import html
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

from analyze_multivariate import digest, finite_median, read_csv, write_csv


def quantile(values, probability):
    ordered = sorted(x for x in values if math.isfinite(x))
    if not ordered:
        return math.nan
    h = (len(ordered)-1)*probability
    i = math.floor(h)
    return ordered[i] if i == len(ordered)-1 else ordered[i]+(h-i)*(ordered[i+1]-ordered[i])


def close(a, b):
    # Pilot percentile reconstruction differs at Float64 association scale.
    return a == b or math.isnan(a) and math.isnan(b) or (
        math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 2e-12*max(1.0, abs(a), abs(b)))


def wilson(covered, count, level=0.95):
    if not 0 <= covered <= count or count < 0 or not 0 < level < 1:
        raise ValueError("invalid binomial counts or confidence level")
    if count == 0:
        return math.nan, math.nan
    z = statistics.NormalDist().inv_cdf((1+level)/2)
    p = covered/count
    denominator = 1+z*z/count
    centre = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0.0, centre-radius), min(1.0, centre+radius)


def mcse(values):
    return statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else math.nan


def clustered_summary(covered, available):
    if len(covered) != len(available) or not covered:
        raise ValueError("dataset coverage and availability scores must align")
    if not all(0 <= c <= a <= 1 for c, a in zip(covered, available)):
        raise ValueError("invalid dataset coverage/availability score")
    n = len(covered)
    availability = statistics.mean(available)
    yield_ = statistics.mean(covered)
    conditional = yield_/availability if availability else math.nan
    influence = [(c-conditional*a)/availability for c, a in zip(covered, available)] if availability else []
    return dict(datasets=n, availability=availability, available_and_covers=yield_,
                coverage_given_available=conditional, mcse_available_and_covers=mcse(covered),
                mcse_given_available=mcse(influence), mcse_availability=mcse(available))


def fit_key(row):
    return row["case"], row["model"], int(row["seed"])


def load_cohort(path, truth_function=None):
    meta = tomllib.loads((path/"metadata.toml").read_text())
    design = tomllib.loads((path/"design.toml").read_text())
    assert meta["check_bounds"] and meta["blas_threads"] == 1
    assert digest(path/"design.toml") == meta["design_sha256"]
    assert digest(path/"environment-manifest.toml") == meta["manifest_sha256"]
    assert digest(path/"environment-project.toml") == meta["environment_project_sha256"]
    for name, value in meta["script_sha256"].items():
        assert digest(path/name) == value
    zvalues = {r["level"]: r["z"] for r in meta["normal_quantiles"]}
    for level, z in zvalues.items():
        # The library's normal approximation differs by <=1.82e-9 at
        # the declared levels; 1e-7 gives >50x numerical headroom.
        assert abs(z-statistics.NormalDist().inv_cdf((1+level)/2)) < 1e-7
    fits, boots, raw = [read_csv(path/name) for name in ("fits.csv", "bootstrap.csv", "intervals.csv")]
    expected = {(c,m,s) for c in design["cases"] for m in design["models"] for s in design["seeds"]}
    assert {fit_key(r) for r in fits} == expected and len(fits) == len(expected)
    assert {fit_key(r) for r in boots} == expected and len(boots) == len(expected)
    records = {}
    diagnostics = []
    for key in expected:
        records[key] = tomllib.loads((path/"replicates"/("__".join(map(str,key))+".toml")).read_text())
        record = records[key]
        assert [r["attempt"] for r in record["refit_diagnostics"]] == record["finite_attempts"]
        diagnostics.extend(dict(case=key[0],model=key[1],seed=key[2],**r)
                           for r in record["refit_diagnostics"])
    query_rows = read_csv(path/"queries.csv")
    queries = {(r["case"],int(r["point_id"])): r for r in query_rows}
    assert len(queries) == len(query_rows)
    point_ids = set(range(1,len(design["query_points"])+1))
    assert all({int(r["point_id"]) for r in query_rows if r["case"] == c} == point_ids
               for c in design["cases"])
    assert all(float(r["x"]) == design["query_points"][int(r["point_id"])-1] for r in query_rows)
    expected_intervals = set()
    for c,m,s in expected:
        budgets = [design["nboot"]] + ([design["sensitivity_boot"]] if s in design["sensitivity_seeds"] else [])
        for level in design["levels"]:
            for point in point_ids:
                expected_intervals.add((c,m,s,"covariance",0,level,point))
                expected_intervals.update((c,m,s,"bootstrap",b,level,point) for b in budgets)
    intervals = []
    observed = set()
    for row in raw:
        c,m,s = fit_key(row)
        record = records[c,m,s]
        method, budget, level, point = row["method"], int(row["budget"]), float(row["level"]), int(row["point_id"])
        identity = (c,m,s,method,budget,level,point)
        assert identity not in observed
        observed.add(identity)
        query = queries[c,point]
        x, truth = float(row["x"]), float(row["truth"])
        assert x == float(query["x"]) and truth == float(query["truth"])
        if truth_function is None:
            assert c in ("logistic","nonlinear"), "custom conditions require an explicit truth oracle"
            expected_truth = 0.9*(1-x/2)*(1+0.35*math.sin(2*x) if c == "nonlinear" else 1)
        else:
            expected_truth = truth_function(c,x)
        assert close(truth,expected_truth)
        lo,hi = float(query["training_min"]),float(query["training_max"])
        region = "below_range" if x < lo else "above_range" if x > hi else "in_range"
        assert row["region"] == query["region"] == region
        estimate, lower, upper = [float(row[k]) for k in ("estimate","lower","upper")]
        assert close(estimate,record["estimates"][point-1])
        available = all(map(math.isfinite,(lower,upper))) and lower <= upper and row["status"] == "ok"
        covered = available and lower <= truth <= upper
        assert (row["available"] == "true") == available
        assert (row["covered"] == "true") == covered
        assert close(float(row["width"]),upper-lower if available else math.nan)
        if method == "covariance" and "standard_errors" in record and row["status"] != "covariance_failed":
            se = record["standard_errors"][point-1]
            assert close(lower,estimate-zvalues[level]*se)
            assert close(upper,estimate+zvalues[level]*se)
        if method == "bootstrap":
            attempts = record["finite_attempts"]
            assert attempts == sorted(set(attempts)) and all(i>0 for i in attempts)
            keep = [i for i,a in enumerate(attempts) if a <= budget]
            assert int(row["n_success"]) == len(keep)
            columns = record.get("bootstrap_values",[])
            values = [columns[i][point-1] for i in keep if math.isfinite(columns[i][point-1])] if columns else []
            assert int(row["usable"]) == len(values)
            if columns:
                assert len(columns) == len(attempts)
                assert len(record["bootstrap_parameters"]) == len(attempts)
            if len(keep) >= 3 and len(values) >= 3 and record["bootstrap_status"] == "ok":
                assert close(lower,quantile(values,(1-level)/2))
                assert close(upper,quantile(values,1-(1-level)/2))
            else:
                assert not available
        intervals.append(dict(case=c,model=m,seed=s,method=method,budget=budget,level=level,
            point_id=point,x=x,truth=truth,region=region,estimate=estimate,
            lower=lower,upper=upper,available=available,covered=covered,
            width=upper-lower if available else math.nan))
    assert observed == expected_intervals
    boot_lookup = {fit_key(r):r for r in boots}
    for row in fits:
        key = fit_key(row)
        record, boot = records[key],boot_lookup[key]
        assert record["fit_status"] == row["fit_status"]
        assert int(boot["attempted"]) == record["bootstrap_attempted"]
        assert int(boot["n_success"]) == len(record["finite_attempts"])
        assert int(boot["attempted"]) <= int(boot["requested"])
        if row["fit_status"] == "ok":
            assert len(record["parameters"]) == int(row["nparams"])
            assert all(map(math.isfinite,record["parameters"]))
        else:
            assert int(boot["attempted"]) == 0
    return design,meta,fits,boots,intervals,diagnostics


def summaries(intervals):
    points,regions = defaultdict(list),defaultdict(list)
    for r in intervals:
        prefix = (r["case"],r["model"],r["method"],r["budget"],r["level"])
        points[prefix+(r["point_id"],)].append(r)
        regions[prefix+(r["region"],)].append(r)
    point_rows,region_rows,seed_rows = [],[],[]
    prefix_names = ("case","model","method","budget","level")
    for key,rows in sorted(points.items()):
        assert len({r["seed"] for r in rows}) == len(rows)
        available = [r for r in rows if r["available"]]
        k,n,total = sum(r["covered"] for r in rows),len(available),len(rows)
        clo,chi = wilson(k,n)
        ylo,yhi = wilson(k,total)
        errors = [r["estimate"]-r["truth"] for r in rows if math.isfinite(r["estimate"])]
        point_rows.append(dict(zip(prefix_names+("point_id",),key),
            x=rows[0]["x"],region=rows[0]["region"],datasets=total,available=n,covered=k,
            availability=n/total,coverage_given_available=k/n if n else math.nan,
            coverage_mc_lower=clo,coverage_mc_upper=chi,available_and_covers=k/total,
            yield_mc_lower=ylo,yield_mc_upper=yhi,
            mean_width=statistics.mean(r["width"] for r in available) if n else math.nan,
            median_width=finite_median(r["width"] for r in available),
            bias=statistics.mean(errors) if errors else math.nan,
            rmse=math.sqrt(statistics.mean(e*e for e in errors)) if errors else math.nan))
    for key,rows in sorted(regions.items()):
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r)
        expected_points = {r["point_id"] for r in rows}
        cover,avail = [],[]
        for seed,group in sorted(by_seed.items()):
            assert len(group) == len(expected_points) and {r["point_id"] for r in group} == expected_points
            c,a = statistics.mean(r["covered"] for r in group),statistics.mean(r["available"] for r in group)
            cover.append(c)
            avail.append(a)
            seed_rows.append(dict(zip(prefix_names+("region",),key),seed=seed,
                                  available_and_covers=c,availability=a))
        usable = [r for r in rows if r["available"]]
        errors = [r["estimate"]-r["truth"] for r in rows if math.isfinite(r["estimate"])]
        region_rows.append(dict(zip(prefix_names+("region",),key),
            **clustered_summary(cover,avail),points_per_dataset=len(expected_points),
            mean_width=statistics.mean(r["width"] for r in usable) if usable else math.nan,
            rmse=math.sqrt(statistics.mean(e*e for e in errors)) if errors else math.nan))
    return point_rows,region_rows,seed_rows


def paired_summaries(seed_rows,primary_budget,sensitivity_budget):
    lookup = {(r["case"],r["model"],r["method"],r["budget"],r["level"],r["region"],r["seed"]):r
              for r in seed_rows}
    pairs = defaultdict(list)
    for key,r in lookup.items():
        case,model,method,budget,level,region,seed = key
        other = None
        if model == "kan12":
            other = lookup.get((case,"spline8",method,budget,level,region,seed))
            label = "KAN-minus-spline"
        if other is not None:
            pairs[(label,case,model,method,budget,level,region)].append(
                r["available_and_covers"]-other["available_and_covers"])
        if method == "bootstrap" and budget == sensitivity_budget:
            other = lookup[(case,model,method,primary_budget,level,region,seed)]
            pairs[("larger-minus-primary-bootstrap",case,model,method,budget,level,region)].append(
                r["available_and_covers"]-other["available_and_covers"])
    names = ("comparison","case","model","method","budget","level","region")
    return [dict(zip(names,key),paired_datasets=len(values),
                 mean_coverage_yield_difference=statistics.mean(values),mcse=mcse(values))
            for key,values in sorted(pairs.items())]


def coverage_svg(points,queries,primary_budget):
    panels = sorted({(r["case"],r["method"]) for r in points
                     if r["level"] == 0.95 and r["budget"] in (0,primary_budget)})
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="710" viewBox="0 0 1000 710">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#252525">',
             '<text x="25" y="25" font-size="18">Empirical coverage of nominal 95% response intervals</text>',
             '<text x="25" y="47">Bars: 95% Wilson Monte Carlo limits across datasets, not response confidence bands.</text>',
             '<text x="25" y="66">Green: noiseless training-density range, not a guarantee of identifiability.</text>']
    colors = {"spline8":"#2563eb","kan12":"#c2410c"}
    for i,(model,color) in enumerate(colors.items()):
        parts.append(f'<text x="{25+150*i}" y="89" fill="{color}">{html.escape(model)}</text>')
    for i,(case,method) in enumerate(panels):
        left,top,pw,ph = 62+(i%2)*495,135+(i//2)*290,405,220
        px = lambda x:left+x/2.4*pw
        py = lambda y:top+ph-y*ph
        lo,hi = queries[case]
        parts.extend([f'<text x="{left}" y="{top-13}" font-weight="bold">{html.escape(case)} / {method}</text>',
            f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#aaa"/>',
            f'<rect x="{px(lo)}" y="{top}" width="{px(hi)-px(lo)}" height="{ph}" fill="#e6f3e9"/>',
            f'<line x1="{left}" x2="{left+pw}" y1="{py(.95)}" y2="{py(.95)}" stroke="#555" stroke-dasharray="4 3"/>'])
        for y in (0,.25,.5,.75,1):
            parts.append(f'<text x="{left-8}" y="{py(y)+4}" text-anchor="end">{100*y:g}%</text>')
        for x in (0,.5,1,1.5,2,2.4):
            parts.append(f'<text x="{px(x)}" y="{top+ph+20}" text-anchor="middle">{x:g}</text>')
        parts.append(f'<text x="{left+pw/2}" y="{top+ph+41}" text-anchor="middle">density N</text>')
        for r in points:
            if (r["case"],r["method"],r["level"]) != (case,method,.95) or r["budget"] not in (0,primary_budget):
                continue
            if not math.isfinite(r["coverage_given_available"]):
                continue
            color = colors[r["model"]]
            x = px(r["x"]+(-.008 if r["model"]=="spline8" else .008))
            a,b = py(r["coverage_mc_lower"]),py(r["coverage_mc_upper"])
            parts.extend([f'<path d="M{x},{a} V{b} M{x-3},{a} H{x+3} M{x-3},{b} H{x+3}" stroke="{color}" fill="none"/>',
                f'<circle cx="{x}" cy="{py(r["coverage_given_available"])}" r="3" fill="{color}"/>'])
    parts.append("</g></svg>")
    svg = "\n".join(parts)+"\n"
    ElementTree.fromstring(svg)
    return svg


def main():
    parser = argparse.ArgumentParser()
    default = ",".join(f"benchmarks/kan/results/uncertainty-calibration-{case}-{model}"
                       for case in ("logistic","nonlinear") for model in ("spline8","kan12"))
    parser.add_argument("--inputs",default=default)
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/uncertainty-calibration-analysis"))
    args = parser.parse_args()
    paths = [Path(p).resolve() for p in args.inputs.split(",")]
    fits,boots,intervals,metadata,diagnostics,fingerprints,data_hashes = [],[],[],[],[],{},{}
    seen = set()
    first_design = None
    query_ranges = {}
    for path in paths:
        design,meta,f,b,rows,diag = load_cohort(path)
        if first_design is None:
            first_design = design
        else:
            for field in ("iterations","nboot","sensitivity_boot","levels","domain","initial_states",
                          "noise_sigma","model_seed","jacobian","bootstrap_method","selection","query_points"):
                assert first_design[field] == design[field]
            for field in ("evaluation_source_sha256","manifest_sha256","script_sha256"):
                assert metadata[0][field] == meta[field]
        for row in f:
            assert fit_key(row) not in seen
            seen.add(fit_key(row))
        for file in path.glob("datasets/*.toml"):
            fingerprint = digest(file)
            assert file.name not in data_hashes or data_hashes[file.name] == fingerprint
            data_hashes[file.name] = fingerprint
        for query in read_csv(path/"queries.csv"):
            pair = float(query["training_min"]),float(query["training_max"])
            assert query["case"] not in query_ranges or query_ranges[query["case"]] == pair
            query_ranges[query["case"]] = pair
        for file in path.rglob("*"):
            if file.is_file():
                name = file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
                fingerprints[str(name)] = digest(file)
        metadata.append(meta)
        fits.extend(f)
        boots.extend(b)
        intervals.extend(rows)
        diagnostics.extend(diag)
    points,regions,seeds = summaries(intervals)
    paired = paired_summaries(seeds,first_design["nboot"],first_design["sensitivity_boot"])
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("pointwise.csv",points),("regions.csv",regions),("dataset_regions.csv",seeds),
                      ("paired.csv",paired)):
        write_csv(args.output/name,rows)
    (args.output/"coverage.svg").write_text(coverage_svg(points,query_ranges,first_design["nboot"]))
    status = dict(base_fits=dict(Counter(r["fit_status"] for r in fits)),
        bootstrap=dict(Counter(r["boot_status"] for r in boots)),
        requested_refits=sum(int(r["requested"]) for r in boots),
        attempted_refits=sum(int(r["attempted"]) for r in boots),
        finite_refits=sum(int(r["n_success"]) for r in boots),
        base_stopping=dict(Counter(r["reason"] for r in fits)),
        smoothing_advanced=dict(Counter(r["smoothing_advanced"] for r in fits)),
        refit_stopping=dict(Counter(r["reason"] for r in diagnostics)),
        refit_smoothing_advanced=dict(Counter(str(r["smoothing_advanced"]) for r in diagnostics)))
    (args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    script,helper = Path(__file__),Path(__file__).with_name("analyze_multivariate.py")
    for file in (script,helper):
        (args.output/file.name).write_bytes(file.read_bytes())
    (args.output/"metadata.json").write_text(json.dumps(dict(
        model_evaluation=False,optimizer_refit=False,inputs_sha256=fingerprints,
        analysis_sha256=digest(script),helper_sha256=digest(helper),
        execution_source_sha256=metadata[0]["evaluation_source_sha256"],
        path_base="repository working directory for relative input paths",
        replication_unit="independent dataset; query points clustered within dataset",
        pointwise_mc="95% Wilson intervals conditional on availability; yields retain all datasets",
        regional_mc="dataset-clustered ratio delta-method MCSE; zero estimated MCSE is not certainty"),
        indent=2)+"\n")
    print(json.dumps(status,indent=2))
    print("Analyzed",len(intervals),"intervals from",len(fits),"base fits.")


if __name__ == "__main__":
    main()
