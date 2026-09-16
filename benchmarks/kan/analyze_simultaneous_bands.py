import argparse
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

from analyze_calibration import close, clustered_summary, mcse, wilson
from analyze_coverage_robustness import truth as true_response
from analyze_multivariate import digest, finite_median, read_csv, write_csv


def band_score(lower,upper,truth):
    if not truth or len({len(x) for x in (lower,upper,truth)}) != 1:
        raise ValueError("band vectors must be nonempty and aligned")
    available=[math.isfinite(a) and math.isfinite(b) and a<=b for a,b in zip(lower,upper)]
    covered=[ok and a<=t<=b for ok,a,b,t in zip(available,lower,upper,truth)]
    widths=[b-a for ok,a,b in zip(available,lower,upper) if ok]
    return dict(point_availability=statistics.mean(available),
        point_coverage_yield=statistics.mean(covered),band_available=all(available),
        joint_covered=all(covered),mean_width=statistics.mean(widths) if widths else math.nan,
        max_width=max(widths) if widths else math.nan)


def summarize(rows):
    names=("case","model","grid","scope","source","interval","level")
    grouped=defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in names)].append(row)
    result=[]
    for key,group in sorted(grouped.items()):
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("duplicate dataset in simultaneous-band population")
        point=clustered_summary([r["point_coverage_yield"] for r in group],
                                [r["point_availability"] for r in group])
        n=sum(r["band_available"] for r in group)
        covered=sum(r["joint_covered"] for r in group)
        lower,upper=wilson(covered,n)
        ylower,yupper=wilson(covered,len(group))
        result.append(dict(zip(names,key),datasets=len(group),points=group[0]["points"],
            bands_available=n,joint_covered=covered,joint_coverage=covered/n if n else math.nan,
            joint_yield=covered/len(group),joint_mc_lower=lower,joint_mc_upper=upper,
            joint_yield_mc_lower=ylower,joint_yield_mc_upper=yupper,
            pointwise_coverage=point["coverage_given_available"],pointwise_mcse=point["mcse_given_available"],
            mean_width=statistics.mean(r["mean_width"] for r in group if math.isfinite(r["mean_width"]))
                       if any(math.isfinite(r["mean_width"]) for r in group) else math.nan,
            median_critical=finite_median(r["critical"] for r in group)))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("benchmarks/kan/results/simultaneous-band-assessment"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/simultaneous-band-analysis"))
    args=parser.parse_args()
    path=args.input.resolve()
    meta=tomllib.loads((path/"metadata.toml").read_text())
    design=tomllib.loads((path/"design.toml").read_text())
    assert not meta["optimizer_refit"] and meta["model_evaluation"]
    assert digest(path/"design.toml")==meta["design_sha256"]
    for name,value in meta["script_sha256"].items():
        assert digest(path/name)==value
    for name,value in meta["input_sha256"].items():
        assert digest(Path.cwd()/name)==value
    rows=read_csv(path/"dataset_bands.csv")
    records={}
    checked=[]
    identities=set()
    for row in rows:
        key=row["case"],row["model"],int(row["seed"])
        if key not in records:
            records[key]=tomllib.loads((path/"bands"/("__".join(map(str,key))+".toml")).read_text())
        identity=key+(row["grid"],row["scope"],row["source"],row["interval"],float(row["level"]))
        assert identity not in identities
        identities.add(identity)
        name="__".join((row["grid"],row["scope"],row["source"],row["interval"],row["level"]))
        band=records[key][name]
        actual=[true_response(key[0],x) for x in band["points"]]
        assert all(close(a,b) for a,b in zip(actual,band["truth"]))
        assert len(actual)==int(row["points"])
        score=band_score(band["lower"],band["upper"],actual)
        for field in ("point_availability","point_coverage_yield","mean_width","max_width"):
            assert close(score[field],float(row[field]))
        for field in ("band_available","joint_covered"):
            assert score[field]==(row[field]=="true")
        critical=float(row["critical"])
        if row["interval"]=="simultaneous" and row["status"]=="ok":
            assert math.isfinite(critical) and critical>=0
            assert all(close(a,m-critical*s) for a,m,s in zip(band["lower"],band["fitted"],band["se"]))
            assert all(close(a,m+critical*s) for a,m,s in zip(band["upper"],band["fitted"],band["se"]))
        checked.append(dict(case=key[0],model=key[1],seed=key[2],grid=row["grid"],scope=row["scope"],
            source=row["source"],interval=row["interval"],level=float(row["level"]),
            points=len(actual),critical=critical,**score))
    expected={key+(grid,scope,source,interval,level) for key in records
              for grid in design["grids"] for scope in design["scopes"]
              for source in ("covariance","bootstrap") for interval in ("pointwise","simultaneous")
              for level in design["levels"]}
    assert identities==expected and len(records)==meta["fits"]
    lookup={(r["case"],r["model"],r["seed"],r["grid"],r["scope"],r["source"],r["interval"],r["level"]):r
            for r in checked}
    # Common Gaussian draws and unchanged bootstrap columns make the
    # simulated grid scopes nested; allow only numerical roundoff here.
    for r in checked:
        if r["interval"]!="simultaneous" or not r["band_available"]:
            continue
        if r["grid"]=="archived":
            other=lookup[r["case"],r["model"],r["seed"],"dense_with_archived_points",
                         r["scope"],r["source"],r["interval"],r["level"]]
            if other["band_available"]:
                assert other["critical"]+1e-7>=r["critical"]
        if r["scope"]=="in_range":
            other=lookup[r["case"],r["model"],r["seed"],r["grid"],"all_queries",
                         r["source"],r["interval"],r["level"]]
            if other["band_available"]:
                assert other["critical"]+1e-7>=r["critical"]
    args.output.mkdir(parents=True,exist_ok=False)
    summary=summarize(checked)
    write_csv(args.output/"summary.csv",summary)
    write_csv(args.output/"dataset_scores.csv",checked)
    files=[Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
           Path(__file__).with_name("analyze_coverage_robustness.py"),
           Path(__file__).with_name("analyze_undersmoothing.py"),
           Path(__file__).with_name("analyze_multivariate.py"),
           Path(__file__).with_name("compare_uncertainty_methods.py")]
    for file in files:
        (args.output/file.name).write_bytes(file.read_bytes())
    fingerprints={}
    for file in path.rglob("*"):
        if file.is_file():
            name=file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
            fingerprints[str(name)]=digest(file)
    (args.output/"metadata.json").write_text(json.dumps(dict(
        model_evaluation=False,optimizer_refit=False,inputs_sha256=fingerprints,
        scripts_sha256={p.name:digest(p) for p in files},
        target="Joint coverage is all query points inside per dataset; pointwise averages are separate",
        scope="Finite per-function grids; no continuum or cross-function simultaneous guarantee",
        covariance="Conditional Gaussian/delta, not analytic unconditional covariance"),
        indent=2)+"\n")
    print("Audited",len(rows),"dataset/band combinations from",len(records),"original fits.")


if __name__=="__main__":
    main()
