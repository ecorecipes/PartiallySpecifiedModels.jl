import argparse
import csv
import hashlib
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    require(bool(rows), f"no rows for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth_value(value):
    require(value in ("true", "false"), "invalid Boolean in experiment output")
    return value == "true"


def dataset_key(row):
    return tuple(row[k] for k in ("stage", "operator", "case", "design", "noise", "sigma", "seed"))


def active(method, operator, noise_mode):
    return not method.startswith("split_bias") or (
        noise_mode == "known" and (method != "split_bias_ellipsoid" or operator == "integral"))


def selected(design, model, method, operator, noise_mode):
    return not design["participants"] or any(
        (r["model"], r["method"], r["operator"], r["noise_mode"]) ==
        (model, method, operator, noise_mode) for r in design["participants"])


def expected_intervals(design):
    n = design["ngrid"]
    result = set()
    jobs = design.get("dataset_jobs", [
        dict(operator=op, case=case, design=obs, noise=noise, sigma=sigma, seed=seed)
        for op in design["operators"] for case in design["cases"][op] for obs in design["designs"]
        for noise in design["noises"] for sigma in design["sigmas"] for seed in design["seeds"]])
    for job in jobs:
        op = job["operator"]
        key = (design["stage"], op, job["case"], job["design"], job["noise"], float(job["sigma"]), int(job["seed"]))
        for mode in design["noise_modes"]:
            participants = [(m, method) for m in design["models"] for method in design["methods"]
                            if active(method, op, mode) and selected(design, m, method, op, mode)]
            if mode == "known":
                participants += [("reference", "bank_set")]
                if op == "integral":
                    participants += [("reference", "linear_set")]
            for model, method in participants:
                for interval in ("pointwise", "simultaneous"):
                    if method == "bootstrap_percentile" and interval == "simultaneous":
                        continue
                    for target in range(1, (n+2 if interval == "pointwise" else n)+1):
                        result.add((key, mode, model, method, interval, target))
    return result


def load(source):
    meta = tomllib.loads((source/"metadata.toml").read_text())
    require(meta["schema"] == "calibration-results-v1" and meta["complete"], "experiment is incomplete")
    for name, sha in meta["outputs_sha256"].items():
        file = (source/name).resolve()
        require(file.is_relative_to(source.resolve()), "output path escapes archive")
        require(digest(file) == sha, f"output fingerprint differs: {name}")
    for name, sha in meta["script_sha256"].items():
        require(digest(source/name) == sha, f"execution script changed: {name}")
    design = tomllib.loads((source/"design.toml").read_text())
    require(design["code_sha256"] == meta["code_sha256"], "code fingerprints differ")
    rows = csv_rows(source/"intervals.csv")
    seen = set()
    for row in rows:
        for field in ("sigma", "x", "truth", "estimate", "lower", "upper", "width", "critical"):
            row[field] = float(row[field])
        row["seed"], row["target_id"] = int(row["seed"]), int(row["target_id"])
        for field in ("available", "covered", "assumption_satisfied", "reference_only"):
            row[field] = truth_value(row[field])
        key = (dataset_key(row), row["noise_mode"], row["model"], row["method"], row["interval"], row["target_id"])
        require(key not in seen, "duplicate interval")
        seen.add(key)
        valid = row["status"] == "ok" and math.isfinite(row["lower"]) and math.isfinite(row["upper"]) and row["lower"] <= row["upper"]
        require(row["available"] == valid, "availability does not match endpoints")
        require(row["covered"] == (valid and row["lower"] <= row["truth"] <= row["upper"]),
                "coverage does not match endpoints")
    require(seen == expected_intervals(design), "missing or unexpected requested intervals")
    require(len(rows) == meta["interval_count"], "interval count differs")
    fits = csv_rows(source/"fits.csv")
    require(len(fits) == meta["fit_count"] == design["expected_fit_jobs"], "fit count differs")
    audit_bootstrap(source, design, rows)
    return design, rows


def audit_bootstrap(source, design, intervals):
    if not any(m.startswith("bootstrap") for m in design["methods"]):
        return
    method_availability = defaultdict(bool)
    for row in intervals:
        key = (row["operator"], row["design"], row["noise"], row["sigma"], row["case"],
               row["seed"], row["model"], row["noise_mode"], row["method"])
        method_availability[key] |= row["available"]
    for path in (source/"fits").glob("*.toml"):
        fields = path.stem.split("__")
        require(len(fields) == 8, "unexpected bootstrap fit filename")
        op, obs, noise, sigma, case, seed, model, mode = fields
        key = (op, obs, noise, float(sigma), case, int(seed), model, mode)
        record = tomllib.loads(path.read_text())
        methods = [m for m in design["methods"] if m.startswith("bootstrap")
                   and selected(design, model, m, op, mode)]
        for generator in {"pilot" if m == "bootstrap_t_pilot" else "fit" for m in methods}:
            related = [m for m in methods if (m == "bootstrap_t_pilot") == (generator == "pilot")]
            boot = record.get("bootstrap", {}).get(generator)
            if boot is None:
                require(not any(method_availability[key+(m,)] for m in related),
                        "available bootstrap interval lacks saved draws")
                continue
            attempts = boot["attempts"]
            require([r["attempt"] for r in attempts] == list(range(1, design["nboot"]+1)),
                    "missing, duplicated or reordered bootstrap attempts")
            nout = design["ngrid"]+2
            require(len(boot["values"]) == len(boot["se"]) == len(boot["generating"]) == nout,
                    "bootstrap output dimensions differ")
            require(all(len(row) == design["nboot"] for row in boot["values"]+boot["se"]),
                    "bootstrap attempt columns differ")
            for b, attempt in enumerate(attempts):
                # Filenames retain Julia's exact Float64 spelling of the
                # noise scale, including scientific notation.
                payload = "|".join(map(str, ("bootstrap", design["stage"], op, obs, noise,
                                           sigma, case, int(seed), generator, b+1)))
                expected_seed = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) & 0x7fffffffffffffff
                require(attempt["rng_seed"] == str(expected_seed), "bootstrap RNG lineage differs")
                if attempt["status"] == "ok":
                    require(all(math.isfinite(row[b]) for row in boot["values"]) and
                            all(math.isfinite(row[b]) and row[b] >= 0 for row in boot["se"]),
                            "successful bootstrap attempt is incomplete")
                else:
                    require(all(math.isnan(row[b]) for row in boot["values"]+boot["se"]),
                            "failed bootstrap attempt has successful values")
                    require(not any(method_availability[key+(m,)] for m in related),
                            "failed bootstrap curve was silently discarded")


def wilson(successes, n):
    z = statistics.NormalDist().inv_cdf(0.975)
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return center-half, center+half


def score_intervals(rows, level):
    available = all(r["available"] for r in rows)
    joint = rows[0]["interval"] == "simultaneous"
    coverage = float(all(r["covered"] for r in rows)) if joint else statistics.mean(r["covered"] for r in rows)
    widths = [r["width"] for r in rows if r["available"]]
    alpha = 1-level
    scores = [(r["upper"]-r["lower"]) +
              2/alpha*max(r["lower"]-r["truth"], 0) +
              2/alpha*max(r["truth"]-r["upper"], 0)
              for r in rows if r["available"]]
    return dict(coverage=coverage, available=available, available_points=len(widths), points=len(rows),
                mean_width=statistics.mean(widths) if widths else math.nan,
                interval_score=statistics.mean(scores) if scores else math.nan)


def summarize(sources, output):
    all_rows, designs, fingerprints = [], [], {}
    seen = {}
    data_fingerprints = {}
    for source in sources:
        design, rows = load(source)
        designs.append(design)
        fingerprints[str(source/"metadata.toml")] = digest(source/"metadata.toml")
        run_meta = tomllib.loads((source/"metadata.toml").read_text())
        for name, sha in run_meta["outputs_sha256"].items():
            if name.startswith("datasets/"):
                require(name not in data_fingerprints or data_fingerprints[name] == sha,
                        "paired datasets differ between experiment shards")
                data_fingerprints[name] = sha
        if len(designs) > 1:
            for field in ("stage", "code_sha256", "operators", "designs", "noises",
                          "noise_modes", "sigmas", "nboot", "nsim", "level", "ngrid", "lock_sha256"):
                require(design[field] == designs[0][field], f"incompatible cohort field {field}")
            for key, value in design["options"].items():
                if key not in ("models", "methods"):
                    require(value == designs[0]["options"][key], f"incompatible fitting option {key}")
        for row in rows:
            key = (dataset_key(row), row["noise_mode"], row["model"], row["method"], row["interval"], row["target_id"])
            if key in seen:
                # Repeated reference controls in model-sharded runs are the
                # same experiment, never extra independent observations.
                require(row["reference_only"], "overlapping model/dataset cohorts")
                old = seen[key]
                for field in ("truth", "lower", "upper", "available", "covered"):
                    a, b = row[field], old[field]
                    require(a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)),
                            "reference controls differ between shards")
                continue
            seen[key] = row
            all_rows.append(row)
    design = designs[0]
    grouped = defaultdict(list)
    pointwise = defaultdict(list)
    for row in all_rows:
        target = "density_"+row["interval"] if row["scope"] == "density" else row["scope"]
        key = (row["operator"], row["case"], row["design"], row["noise"], row["sigma"],
               row["model"], row["family"], row["noise_mode"], row["method"], target, row["seed"])
        grouped[key].append(row)
        if row["interval"] == "pointwise":
            pointwise[key[:-1]+(row["target_id"],)].append(row)
    fields = ("operator", "case", "design", "noise", "sigma", "model", "family", "noise_mode", "method", "target", "seed")
    dataset_scores = []
    for key, rows in sorted(grouped.items()):
        dataset_scores.append(dict(zip(fields, key)) | dict(
            reference_only=rows[0]["reference_only"],
            assumption_satisfied=all(r["assumption_satisfied"] for r in rows),
            **score_intervals(rows, design["level"])))
    by_stratum = defaultdict(list)
    for row in dataset_scores:
        by_stratum[tuple(row[k] for k in fields[:-1])].append(row)
    summary = []
    for key, rows in sorted(by_stratum.items()):
        n = len(rows)
        p = statistics.mean(r["coverage"] for r in rows)
        se = statistics.stdev(r["coverage"] for r in rows)/math.sqrt(n) if n > 1 else math.nan
        binary = key[-1] != "density_pointwise"
        lower, upper = wilson(sum(r["coverage"] for r in rows), n) if binary else (
            max(0.0, p-1.96*se) if n > 1 else math.nan,
            min(1.0, p+1.96*se) if n > 1 else math.nan)
        complete = [r for r in rows if r["available"]]
        summary.append(dict(zip(fields[:-1], key)) | dict(datasets=n, coverage=p, mcse=se,
            lower_mc=lower, upper_mc=upper, availability=len(complete)/n,
            mean_width_available=statistics.mean(r["mean_width"] for r in complete) if complete else math.nan,
            interval_score_available=statistics.mean(r["interval_score"] for r in complete) if complete else math.nan,
            reference_only=rows[0]["reference_only"],
            assumption_satisfied_fraction=statistics.mean(r["assumption_satisfied"] for r in rows)))
    point_rows = []
    for key, rows in sorted(pointwise.items()):
        n = len(rows)
        covered = sum(r["covered"] for r in rows)
        low, high = wilson(covered, n)
        point_rows.append(dict(zip(fields[:-1], key[:-1])) | dict(target_id=key[-1], x=rows[0]["x"],
            datasets=n, coverage=covered/n, lower_mc=low, upper_mc=high,
            availability=sum(r["available"] for r in rows)/n))
    ranks = defaultdict(list)
    for row in summary:
        if not row["reference_only"]:
            ranks[row["operator"], row["family"], row["noise_mode"], row["target"],
                  row["model"], row["method"]].append(row)
    ranking = []
    for key, rows in sorted(ranks.items()):
        # Rankings are exploratory development criteria, not a declaration
        # that an estimated coverage value establishes nominal calibration.
        widths = [r["mean_width_available"] for r in rows if math.isfinite(r["mean_width_available"])]
        minimum_coverage = min(r["coverage"] for r in rows)
        lower = min(r["lower_mc"] for r in rows) if all(math.isfinite(r["lower_mc"]) for r in rows) else math.nan
        op, family, mode, target, model, method = key
        expected_seeds = set()
        for d in designs:
            if model in d["models"] and method in d["methods"] and active(method, op, mode) and selected(d, model, method, op, mode):
                expected_seeds.update(d["seeds"])
        expected_count = design["confirmation_total"] if design["stage"] == "confirm" else len(expected_seeds)
        expected_strata = len(design["cases"][op])*len(design["designs"])*len(design["noises"])*len(design["sigmas"])
        complete = len(rows) == expected_strata and all(r["datasets"] == expected_count for r in rows)
        ranking.append(dict(zip(("operator", "family", "noise_mode", "target", "model", "method"), key)) |
            dict(strata=len(rows), min_datasets=min(r["datasets"] for r in rows),
                 expected_strata=expected_strata, expected_datasets=expected_count, complete_population=complete,
                 worst_coverage=minimum_coverage, worst_lower_mc=lower,
                 coverage_deficit=max(0.0, design["level"]-minimum_coverage),
                 min_availability=min(r["availability"] for r in rows),
                 mean_width=statistics.mean(widths) if widths else math.inf))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("dataset_scores.csv", dataset_scores), ("summary.csv", summary),
                       ("pointwise.csv", point_rows), ("rankings.csv", ranking)):
        write_csv(output/name, rows)
    union_options = dict(design["options"])
    union_options["methods"] = ",".join(sorted({m for d in designs for m in d["methods"]}))
    refinement_expected = {(r["operator"], r["family"], r["noise_mode"], r["model"], r["method"])
                           for d in designs if d.get("phase") == "bootstrap-refinement"
                           for r in d["participants"]}
    complete_procedures = {(r["operator"], r["family"], r["noise_mode"], r["model"], r["method"])
                          for r in ranking if r["target"] == "density_pointwise" and r["complete_population"]}
    metadata = dict(schema="calibration-analysis-v1", stage=design["stage"],
        code_sha256=design["code_sha256"], options=union_options,
        nboot=design["nboot"], nsim=design["nsim"], level=design["level"],
        input_sha256=fingerprints, analysis_sha256=digest(Path(__file__)),
        sources=[str(s.resolve()) for s in sources], models=sorted({r["model"] for r in ranking}),
        methods=union_options["methods"].split(","), seeds=sorted({r["seed"] for r in all_rows}),
        confirmation_extra_designs=design.get("confirmation_extra_designs", []),
        confirmation_extra_noises=design.get("confirmation_extra_noises", []),
        ranking_scope="procedures, not approximator-only effects; separate operator/noise/target strata",
        bootstrap_scope="all attempted curves required; unavailable intervals remain in coverage yield",
        mc_scope="independent datasets, not density points",
        confirmation_complete=design["stage"] == "confirm" and all(r["complete_population"] for r in ranking),
        refinement_planned=bool(refinement_expected),
        refinement_complete=bool(refinement_expected) and refinement_expected <= complete_procedures,
        outputs_sha256={p.name: digest(p) for p in output.glob("*.csv")})
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Summarized {len(all_rows)} intervals into {len(summary)} strata; stage={design['stage']}.")


def toml_string(value):
    return json.dumps(str(value))


def freeze(analysis, output, target, top, confirmation_datasets, confirmation_bootstrap, require_refinement=False):
    meta = json.loads((analysis/"metadata.json").read_text())
    require(meta["stage"] == "develop", "only development experiments may produce a lock")
    require(top >= 1 and confirmation_datasets >= 1 and confirmation_bootstrap >= 99, "invalid confirmation budget")
    for name, sha in meta["outputs_sha256"].items():
        require(digest(analysis/name) == sha, "analysis output fingerprint differs")
    require(digest(Path(__file__)) == meta["analysis_sha256"], "analysis code changed before freezing")
    require(meta["nboot"] >= 99, "development bootstrap budget must be at least 99 before freezing")
    if require_refinement:
        require(meta.get("refinement_complete", False), "complete matched bootstrap refinement is required")
    rows = csv_rows(analysis/"rankings.csv")
    groups = defaultdict(list)
    for row in rows:
        if row["target"] == target:
            require(int(row["min_datasets"]) >= 20, "development needs at least 20 datasets per stratum")
            require(row["complete_population"] == "True",
                    "incomplete development shards cannot produce a lock")
            groups[row["operator"], row["family"], row["noise_mode"]].append(row)
    require(groups, "no eligible ranking target")
    # Every candidate in a ranking group must have exactly the same
    # development datasets; this catches uneven model/seed shards.
    scores = csv_rows(analysis/"dataset_scores.csv")
    for rows0 in groups.values():
        cohort = None
        for candidate in rows0:
            keys = {(r["case"], r["design"], r["noise"], r["sigma"], r["seed"]) for r in scores
                    if all(r[k] == candidate[k] for k in ("operator", "model", "noise_mode", "method", "target"))}
            require(cohort is None or cohort == keys, "unmatched development cohorts cannot be ranked")
            cohort = keys
    chosen = []
    for rows0 in groups.values():
        rows0.sort(key=lambda r: (float(r["coverage_deficit"]), 1-float(r["min_availability"]),
                                 float(r["mean_width"]), r["model"], r["method"]))
        chosen.extend(rows0[:top])
    options = dict(meta["options"])
    options["models"] = ",".join(sorted({r["model"] for r in chosen}))
    options["methods"] = ",".join(sorted({r["method"] for r in chosen}))
    options["designs"] = ",".join(sorted(set(options.get("designs", "baseline").split(",")) |
                                       set(meta.get("confirmation_extra_designs", []))))
    options["noises"] = ",".join(sorted(set(options.get("noises", "iid").split(",")) |
                                      set(meta.get("confirmation_extra_noises", []))))
    lines = ['schema = "calibration-lock-v1"', 'source_stage = "develop"',
             f'code_sha256 = {toml_string(meta["code_sha256"])}',
             f'analysis_sha256 = {toml_string(digest(analysis/"metadata.json"))}',
             f'target = {toml_string(target)}', "exploratory_selection = true",
             f"confirmation_datasets = {confirmation_datasets}",
             f"confirmation_bootstrap = {confirmation_bootstrap}",
             "confirmation_nsim = 10000",
             f"refinement_required = {str(require_refinement).lower()}",
             f"refinement_complete = {str(meta.get('refinement_complete', False)).lower()}", "", "[options]"]
    lines += [f"{toml_string(k)} = {toml_string(v)}" for k, v in sorted(options.items())]
    for row in chosen:
        lines += ["", "[[participants]]"] + [
            f"{k} = {toml_string(row[k])}" for k in ("operator", "family", "noise_mode", "model", "method", "target")]
        lines += [f'development_worst_coverage = {float(row["worst_coverage"])}',
                  f'development_availability = {float(row["min_availability"])}']
    with output.open("x") as handle:
        handle.write("\n".join(lines)+"\n")
    print(f"Frozen {len(chosen)} procedures for independent confirmation; this is not a calibration certificate.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("summarize")
    analyze.add_argument("--inputs", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("--analysis", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--target", default="density_simultaneous",
                      choices=["density_pointwise", "density_simultaneous", "grid_mean", "contrast"])
    lock.add_argument("--top", type=int, default=1)
    lock.add_argument("--confirmation-datasets", type=int, default=500)
    lock.add_argument("--confirmation-bootstrap", type=int, default=999)
    lock.add_argument("--require-refinement", action="store_true")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize([Path(p).resolve() for p in args.inputs.split(",")], args.output)
    else:
        freeze(args.analysis, args.output, args.target, args.top,
               args.confirmation_datasets, args.confirmation_bootstrap, args.require_refinement)


if __name__ == "__main__":
    main()
