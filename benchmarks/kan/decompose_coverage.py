import argparse
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

from analyze_calibration import load_cohort, mcse, quantile, summaries
from analyze_multivariate import digest, write_csv


def sampling_decomposition(errors, covariance_se):
    if len(errors) != len(covariance_se) or len(errors) < 2:
        raise ValueError("sampling decomposition requires at least two aligned datasets")
    if not all(math.isfinite(x) for values in (errors, covariance_se) for x in values):
        raise ValueError("sampling decomposition requires finite values")
    if any(s < 0 for s in covariance_se):
        raise ValueError("uncertainty scales must be nonnegative")
    bias = statistics.mean(errors)
    sampling_sd = statistics.stdev(errors)
    cov_rms = math.sqrt(statistics.mean(s*s for s in covariance_se))
    return dict(datasets=len(errors), bias=bias, bias_mcse=mcse(errors),
        empirical_sd=sampling_sd, rmse=math.sqrt(statistics.mean(e*e for e in errors)),
        covariance_rms_se=cov_rms,
        covariance_scale_ratio=cov_rms/sampling_sd if sampling_sd else math.nan,
        bias_to_sd=bias/sampling_sd if sampling_sd else math.nan)


def decomposition(errors, covariance_se, bootstrap_bias, bootstrap_sd):
    if not errors or len({len(x) for x in (errors,covariance_se,bootstrap_bias,bootstrap_sd)}) != 1:
        raise ValueError("decomposition arrays must be nonempty and aligned")
    if len(errors)<2 or not all(math.isfinite(x) for values in
            (errors,covariance_se,bootstrap_bias,bootstrap_sd) for x in values):
        raise ValueError("decomposition requires at least two complete finite datasets")
    if any(s<0 for values in (covariance_se,bootstrap_sd) for s in values):
        raise ValueError("uncertainty scales must be nonnegative")
    n=len(errors)
    sampling=sampling_decomposition(errors,covariance_se)
    corrected=[e-b for e,b in zip(errors,bootstrap_bias)]
    # This uses truth from OTHER datasets and is explicitly an oracle
    # diagnostic, never a correction available to the fitted procedure.
    oracle=[e-(sum(errors)-e)/(n-1) for e in errors]
    boot_rms=math.sqrt(statistics.mean(s*s for s in bootstrap_sd))
    corrected_sd=statistics.stdev(corrected)
    ratio=lambda a,b:a/b if b else math.nan
    result=dict(datasets=n,bias=sampling["bias"],bias_mcse=sampling["bias_mcse"],
        empirical_sd=sampling["empirical_sd"],rmse=sampling["rmse"],
        covariance_rms_se=sampling["covariance_rms_se"],bootstrap_rms_sd=boot_rms,
        covariance_scale_ratio=sampling["covariance_scale_ratio"],
        bootstrap_scale_ratio=ratio(boot_rms,sampling["empirical_sd"]),
        bias_to_sd=sampling["bias_to_sd"],
        mean_bootstrap_bias=statistics.mean(bootstrap_bias),
        corrected_bias=statistics.mean(corrected),corrected_empirical_sd=corrected_sd,
        corrected_bootstrap_scale_ratio=ratio(boot_rms,corrected_sd))
    return result,oracle


def counterfactual_rows(row,oracle_error):
    level=row["level"]
    z=statistics.NormalDist().inv_cdf((1+level)/2)
    result=[]
    for label,error,scale in (
        ("covariance_observed",row["error"],row["covariance_se"]),
        ("covariance_oracle_loo_centered",oracle_error,row["covariance_se"]),
        ("bootstrap_normal_observed",row["error"],row["bootstrap_sd"]),
        ("bootstrap_normal_oracle_loo_centered",oracle_error,row["bootstrap_sd"]),
        ("bootstrap_normal_bc_observed",row["error"]-row["bootstrap_bias"],row["bootstrap_sd"])):
        available=math.isfinite(error) and math.isfinite(scale) and scale>=0
        result.append(dict(case=row["case"],model=row["model"],seed=row["seed"],method=label,
            budget=99,level=level,point_id=row["point_id"],x=row["x"],truth=0.0,
            region=row["region"],estimate=error,lower=error-z*scale,upper=error+z*scale,
            available=available,covered=available and abs(error)<=z*scale,
            width=2*z*scale if available else math.nan))
    return result


def run(inputs,output):
    grouped=defaultdict(list)
    rawrows,diagnostic_intervals,fingerprints=[],[],{}
    seen=set()
    for path in inputs:
        design,meta,fits,boots,intervals,diags=load_cohort(path)
        assert design["study_stage"]=="confirm" and design["nboot"]==99
        query={(r["case"],r["seed"],r["point_id"]):r for r in intervals
               if r["method"]=="covariance" and r["level"]==max(design["levels"])}
        for fit in fits:
            key=fit["case"],fit["model"],int(fit["seed"])
            if key in seen:
                raise ValueError("duplicate dataset/model in decomposition inputs")
            seen.add(key)
            record=tomllib.loads((path/"replicates"/("__".join(map(str,key))+".toml")).read_text())
            for i,x in enumerate(record["points"]):
                q=query[key[0],key[2],i+1]
                values=[v[i] for attempt,v in zip(record["finite_attempts"],record.get("bootstrap_values",[]))
                        if attempt<=99 and math.isfinite(v[i])]
                estimate=record["estimates"][i]
                se=record.get("standard_errors",[math.nan]*len(record["points"]))[i]
                complete=math.isfinite(estimate) and math.isfinite(se) and len(values)>=3
                b=statistics.mean(values)-estimate if complete else math.nan
                sd=statistics.stdev(values) if complete else math.nan
                error=estimate-q["truth"]
                row=dict(case=key[0],model=key[1],seed=key[2],point_id=i+1,x=x,
                    region=q["region"],truth=q["truth"],estimate=estimate,error=error,
                    covariance_se=se,bootstrap_bias=b,bootstrap_sd=sd,usable=len(values),
                    complete=complete,edf=float(fit["edf"]),level=max(design["levels"]))
                rawrows.append(row)
                grouped[(key[0],key[1],i+1)].append(row)
        for file in path.rglob("*"):
            if file.is_file():
                name=file.relative_to(Path.cwd()) if file.is_relative_to(Path.cwd()) else file
                fingerprints[str(name)]=digest(file)
    points=[]
    for (case,model,point),rows in sorted(grouped.items()):
        good=[r for r in rows if r["complete"]]
        if len(good)<2:
            raise ValueError("insufficient complete datasets for the requested diagnostic")
        result,oracle=decomposition([r["error"] for r in good],[r["covariance_se"] for r in good],
            [r["bootstrap_bias"] for r in good],[r["bootstrap_sd"] for r in good])
        points.append(dict(case=case,model=model,point_id=point,x=rows[0]["x"],region=rows[0]["region"],
                           requested_datasets=len(rows),complete_datasets=len(good),**result))
        for row,error in zip(good,oracle):
            diagnostic_intervals.extend(counterfactual_rows(row,error))
    point_coverage,regions,seeds=summaries(diagnostic_intervals)
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (("dataset_points.csv",rawrows),("decomposition.csv",points),
                      ("oracle_pointwise.csv",point_coverage),("oracle_regions.csv",regions),
                      ("oracle_dataset_regions.csv",seeds)):
        write_csv(output/name,rows)
    scripts=[Path(__file__),Path(__file__).with_name("analyze_calibration.py"),
             Path(__file__).with_name("analyze_multivariate.py")]
    for script in scripts:
        (output/script.name).write_bytes(script.read_bytes())
    (output/"metadata.json").write_text(json.dumps(dict(
        model_evaluation=False,optimizer_refit=False,inputs_sha256=fingerprints,
        scripts_sha256={p.name:digest(p) for p in scripts},
        scope="Diagnostic decomposition of retained fresh confirmation fits; not a deployable calibration",
        oracle="LOO centering uses known truth and other datasets; no oracle adjustment is used by any fitted method",
        missingness="Requested and complete counts are separate; oracle tables condition on complete diagnostics"),
        indent=2)+"\n")
    print("Decomposed",len(rawrows),"dataset/query records;",sum(r["complete"] for r in rawrows),"complete.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",default="benchmarks/kan/results/coverage-fresh-spline8,benchmarks/kan/results/coverage-fresh-kan12-free")
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/coverage-bias-variance"))
    args=parser.parse_args()
    run([Path(p).resolve() for p in args.inputs.split(",")],args.output)


if __name__=="__main__":
    main()
