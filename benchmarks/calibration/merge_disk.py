"""Disk-backed aggregation using the ordinary analyzer's per-source audits."""

import argparse
import csv
from collections import defaultdict
from itertools import groupby
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile
import tomllib

import analyze

GROUP = ("operator", "case", "design", "noise", "sigma", "model", "family", "noise_mode", "method", "target")
PRIMARY = tuple(k for k in GROUP if k != "family")+("seed", "target_id")
COLUMNS = GROUP+("seed", "target_id", "x", "truth", "lower", "upper", "available", "covered",
                "width", "score", "reference_only", "assumption_satisfied")

def pause_check():
    name = os.environ.get("CALIBRATION_PAUSE_FILE")
    if name and Path(name).is_file():
        raise RuntimeError("Calibration campaign paused: "+Path(name).read_text())


def quoted(names):
    return ",".join('"'+name+'"' for name in names)


def sqlite_value(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def compatible(design, first):
    for field in ("stage", "code_sha256", "operators", "designs", "noises",
                  "noise_modes", "sigmas", "nboot", "nsim", "level", "ngrid", "lock_sha256"):
        analyze.require(design[field] == first[field], f"incompatible cohort field {field}")
    for key, value in design["options"].items():
        if key not in ("models", "methods"):
            analyze.require(value == first["options"][key], f"incompatible fitting option {key}")


def add_rows(db, rows, level):
    insert = (f"INSERT INTO intervals ({quoted(COLUMNS)}) VALUES ({','.join('?' for _ in COLUMNS)}) "
              f"ON CONFLICT ({quoted(PRIMARY)}) DO NOTHING")
    where = " AND ".join('"'+key+'"=?' for key in PRIMARY)
    added = 0
    for row in rows:
        r = dict(row)
        r["target"] = "density_"+r["interval"] if r["scope"] == "density" else r["scope"]
        r["score"] = ((r["upper"]-r["lower"])+2/(1-level)*max(r["lower"]-r["truth"], 0)+
                      2/(1-level)*max(r["truth"]-r["upper"], 0)) if r["available"] else None
        values = tuple(sqlite_value(r[key]) for key in COLUMNS)
        inserted = db.execute(insert, values).rowcount
        if not inserted:
            analyze.require(r["reference_only"], "overlapping model/dataset cohorts")
            previous = db.execute(f"SELECT * FROM intervals WHERE {where}",
                                  tuple(r[key] for key in PRIMARY)).fetchone()
            for field in ("truth", "lower", "upper", "available", "covered"):
                analyze.require(previous[field] == sqlite_value(r[field]), "reference controls differ between shards")
        added += inserted
    return added


def write_query(path, db, sql, boolean=()):
    cursor = db.execute(sql)
    names = [r[0] for r in cursor.description]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for index, row in enumerate(cursor):
            if index % 10000 == 0:
                pause_check()
            r = {k: math.nan if row[k] is None else row[k] for k in names}
            for key in boolean:
                r[key] = bool(r[key])
            writer.writerow(r)


def rankings(summary, designs):
    grouped = defaultdict(list)
    for row in summary:
        if not row["reference_only"]:
            grouped[row["operator"], row["family"], row["noise_mode"], row["target"],
                    row["model"], row["method"]].append(row)
    result = []
    first = designs[0]
    for key, rows in sorted(grouped.items()):
        op, family, mode, target, model, method = key
        seeds = set()
        for d in designs:
            if model in d["models"] and method in d["methods"] and analyze.active(method, op, mode) and analyze.selected(d, model, method, op, mode):
                seeds.update(d["seeds"])
        count = first["confirmation_total"] if first["stage"] == "confirm" else len(seeds)
        strata = len(first["cases"][op])*len(first["designs"])*len(first["noises"])*len(first["sigmas"])
        widths = [r["mean_width_available"] for r in rows if math.isfinite(r["mean_width_available"])]
        minimum = min(r["coverage"] for r in rows)
        result.append(dict(zip(("operator", "family", "noise_mode", "target", "model", "method"), key)) |
            dict(strata=len(rows), min_datasets=min(r["datasets"] for r in rows),
                 expected_strata=strata, expected_datasets=count,
                 complete_population=len(rows) == strata and all(r["datasets"] == count for r in rows),
                 worst_coverage=minimum,
                 worst_lower_mc=min(r["lower_mc"] for r in rows) if all(math.isfinite(r["lower_mc"]) for r in rows) else math.nan,
                 coverage_deficit=max(0, first["level"]-minimum),
                 min_availability=min(r["availability"] for r in rows),
                 mean_width=statistics.mean(widths) if widths else math.inf))
    return result


def summarize(sources, output):
    pause_check()
    analyze.require(sources and not output.exists(), "supply sources and a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    designs, fingerprints = [], {}
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=output.parent) as temporary:
        db = sqlite3.connect(str(Path(temporary)/"intervals.sqlite"))
        db.row_factory = sqlite3.Row
        pause = os.environ.get("CALIBRATION_PAUSE_FILE")
        if pause:
            db.set_progress_handler(lambda: int(Path(pause).is_file()), 100000)
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        numeric = {"sigma", "x", "truth", "lower", "upper", "width", "score"}
        integer = {"seed", "target_id", "available", "covered", "reference_only", "assumption_satisfied"}
        declarations = [f'"{key}" '+("REAL" if key in numeric else "INTEGER" if key in integer else "TEXT") for key in COLUMNS]
        db.execute(f"CREATE TABLE intervals ({','.join(declarations)}, PRIMARY KEY ({quoted(PRIMARY)}))")
        db.execute("CREATE TABLE dataset_hashes (name TEXT PRIMARY KEY, sha TEXT NOT NULL)")
        total = 0
        for i, source in enumerate(sources, 1):
            pause_check()
            source = source.resolve()
            design, rows = analyze.load(source)
            if designs:
                compatible(design, designs[0])
            designs.append(design)
            metadata_path = source/"metadata.toml"
            fingerprints[str(metadata_path)] = analyze.digest(metadata_path)
            meta = tomllib.loads(metadata_path.read_text())
            for name, sha in meta["outputs_sha256"].items():
                if name.startswith("datasets/"):
                    old = db.execute("SELECT sha FROM dataset_hashes WHERE name=?", (name,)).fetchone()
                    analyze.require(old is None or old["sha"] == sha, "paired datasets differ between shards")
                    db.execute("INSERT OR IGNORE INTO dataset_hashes VALUES (?,?)", (name, sha))
            total += add_rows(db, rows, design["level"])
            db.commit()
            del rows, meta
            print(f"Audited source {i}/{len(sources)}; {total} unique interval records.", flush=True)
        first = designs[0]
        db.execute(f"""
            CREATE TABLE scores AS SELECT {quoted(GROUP)}, seed,
                MIN(reference_only) AS reference_only, MIN(assumption_satisfied) AS assumption_satisfied,
                CASE WHEN target='density_simultaneous' THEN MIN(covered) ELSE AVG(covered) END AS coverage,
                MIN(available) AS available, SUM(available) AS available_points, COUNT(*) AS points,
                AVG(width) AS mean_width, AVG(score) AS interval_score
            FROM intervals GROUP BY {quoted(GROUP)}, seed
        """)
        output.mkdir()
        write_query(output/"dataset_scores.csv", db, f"SELECT * FROM scores ORDER BY {quoted(GROUP)},seed",
                    ("reference_only", "assumption_satisfied", "available"))
        summary = []
        cursor = db.execute(f"SELECT * FROM scores ORDER BY {quoted(GROUP)},seed")
        for key, group in groupby(cursor, lambda r: tuple(r[k] for k in GROUP)):
            rows = list(group)  # At most the independent replication count, not all intervals.
            n = len(rows)
            coverage = statistics.mean(r["coverage"] for r in rows)
            se = statistics.stdev(r["coverage"] for r in rows)/math.sqrt(n) if n > 1 else math.nan
            low, high = analyze.wilson(sum(r["coverage"] for r in rows), n) if key[-1] != "density_pointwise" else (
                max(0, coverage-1.96*se) if n > 1 else math.nan,
                min(1, coverage+1.96*se) if n > 1 else math.nan)
            good = [r for r in rows if r["available"]]
            summary.append(dict(zip(GROUP, key)) | dict(datasets=n, coverage=coverage, mcse=se,
                lower_mc=low, upper_mc=high, availability=len(good)/n,
                mean_width_available=statistics.mean(r["mean_width"] for r in good) if good else math.nan,
                interval_score_available=statistics.mean(r["interval_score"] for r in good) if good else math.nan,
                reference_only=bool(rows[0]["reference_only"]),
                assumption_satisfied_fraction=statistics.mean(r["assumption_satisfied"] for r in rows)))
        pointwise = []
        query = (f"SELECT {quoted(GROUP)},target_id,MIN(x) AS x,COUNT(*) AS n,SUM(covered) AS covered,"
                 f"SUM(available) AS available FROM intervals WHERE target!='density_simultaneous' "
                 f"GROUP BY {quoted(GROUP)},target_id ORDER BY {quoted(GROUP)},target_id")
        for r in db.execute(query):
            low, high = analyze.wilson(r["covered"], r["n"])
            pointwise.append({k: r[k] for k in GROUP} | dict(target_id=r["target_id"],
                x=r["x"] if r["x"] is not None else math.nan, datasets=r["n"],
                coverage=r["covered"]/r["n"], lower_mc=low, upper_mc=high,
                availability=r["available"]/r["n"]))
        ranks = rankings(summary, designs)
        for name, rows in (("summary.csv", summary), ("pointwise.csv", pointwise), ("rankings.csv", ranks)):
            analyze.write_csv(output/name, rows)
        methods = sorted({m for d in designs for m in d["methods"]})
        options = dict(first["options"], methods=",".join(methods))
        expected_refinement = {(r["operator"], r["family"], r["noise_mode"], r["model"], r["method"])
                               for d in designs if d.get("phase") == "bootstrap-refinement" for r in d["participants"]}
        completed = {(r["operator"], r["family"], r["noise_mode"], r["model"], r["method"])
                     for r in ranks if r["target"] == "density_pointwise" and r["complete_population"]}
        expected_confirmation = {(r["operator"], r["family"], r["noise_mode"], r["model"], r["method"])
                                 for d in designs if d["stage"] == "confirm" for r in d["participants"]}
        metadata = dict(schema="calibration-analysis-v1", stage=first["stage"],
            code_sha256=first["code_sha256"], options=options, nboot=first["nboot"],
            nsim=first["nsim"], level=first["level"], input_sha256=fingerprints,
            analysis_sha256=analyze.digest(Path(analyze.__file__)),
            aggregation_sha256=analyze.digest(Path(__file__)),
            aggregation="SQLite-backed interval aggregation; original per-source audits and statistical definitions",
            sources=[str(s.resolve()) for s in sources], models=sorted({r["model"] for r in ranks}),
            methods=methods, seeds=sorted({s for d in designs for s in d["seeds"]}),
            confirmation_extra_designs=first.get("confirmation_extra_designs", []),
            confirmation_extra_noises=first.get("confirmation_extra_noises", []),
            ranking_scope="procedures, separate operator/noise/target strata",
            bootstrap_scope="all attempted curves required", mc_scope="independent datasets",
            refinement_planned=bool(expected_refinement),
            refinement_complete=bool(expected_refinement) and expected_refinement <= completed,
            confirmation_complete=bool(expected_confirmation) and expected_confirmation <= completed
                and all(r["complete_population"] for r in ranks),
            unique_interval_records=total,
            outputs_sha256={p.name: analyze.digest(p) for p in output.glob("*.csv")})
        for source in (Path(__file__), Path(analyze.__file__)):
            shutil.copyfile(source, output/source.name)
        (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        db.close()
    print(f"Published {len(summary)} strata using disk-backed aggregation.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize([Path(p) for p in json.loads(args.sources_file.read_text())], args.output)
