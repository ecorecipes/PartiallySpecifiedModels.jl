import argparse
import csv
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

from analyze_multivariate import digest, write_csv
from decompose_coverage import sampling_decomposition

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_file(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "archive path escapes its root")
    return path


def read_rows(path, floats=(), ints=(), bools=()):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in floats:
            row[key] = float(row[key])
        for key in ints:
            row[key] = int(row[key])
        for key in bools:
            require(row[key] in ("true", "false"), f"invalid Boolean {key}")
            row[key] = row[key] == "true"
    return rows


def audit_population(design, native, profiles, candidates, points, selections, curvature):
    expected = {(c, m, s) for c in design["cases"] for m in design["models"] for s in design["seeds"]}
    key = lambda r: (r["case"], r["model"], r["seed"])
    require(len(native) == len(expected) and {key(r) for r in native} == expected,
            "missing or duplicate native fits")
    originals = {key(r): r for r in native}
    grouped_profiles, grouped_candidates = defaultdict(list), defaultdict(list)
    for row in profiles:
        require(key(row) in expected, "unexpected profile dataset")
        grouped_profiles[key(row)].append(row)
    for row in candidates:
        require(key(row) in expected, "unexpected candidate dataset")
        if row["status"] == "ok":
            require(math.isfinite(row["Q"]) and row["Q"] >= 0, "non-finite successful coefficient objective")
        grouped_candidates[key(row)].append(row)
    for identity in expected:
        rho = originals[identity]["rho"]
        grid = set(design["rho_grid"]) | {rho}
        for h in design["local_steps"]:
            grid.update((rho-h, rho+h))
        rows = grouped_profiles[identity]
        require(len(rows) == len(grid)+1 and {r["rho"] for r in rows} == grid | {math.inf},
                "incomplete or duplicated smoothing grid")
        attempts = grouped_candidates[identity]
        wanted = {(r, start) for r in grid for start in ("selected", "ascending", "descending")}
        wanted |= {(math.inf, "null_initial"), (math.inf, "null_selected")}
        require(len(attempts) == len(wanted) and {(r["rho"], r["start"]) for r in attempts} == wanted,
                "missing or duplicated coefficient starts")
        for row in rows:
            good = [r for r in attempts if r["rho"] == row["rho"] and r["status"] == "ok"]
            if row["status"] == "ok":
                require(good, "successful profile has no finite coefficient candidate")
                # The independently accumulated sums agree to roundoff;
                # the scalar probe measured exact equality. This allowance
                # accommodates different reduction orders, not optimizer changes.
                require(math.isclose(row["Q"], min(r["Q"] for r in good), rel_tol=1e-10, abs_tol=1e-12),
                        "profile did not retain the minimum penalized objective")
    point_keys = [(key(r), r["method"], r["point_id"]) for r in points]
    require(len(point_keys) == len(set(point_keys)), "duplicate function-query record")
    query_sets = defaultdict(set)
    for row in points:
        require(key(row) in expected and row["method"] in design["methods"], "unexpected function-query record")
        query_sets[key(row), row["method"]].add((row["point_id"], row["x"], row["truth"], row["oracle"]))
        if row["status"] == "ok":
            require(all(math.isfinite(row[k]) for k in ("fitted", "se", "known_se")) and
                    row["se"] >= 0 and row["known_se"] >= 0, "invalid available uncertainty")
        else:
            require(not row["covered"] and not row["known_scale_covered"], "unavailable interval counted as covered")
    for identity in expected:
        reference = query_sets[identity, "native"]
        require(reference, "missing native query grid")
        for method in design["methods"]:
            require(query_sets[identity, method] == reference, "unmatched method queries")
    for case in design["cases"]:
        for model in design["models"]:
            grids = [query_sets[(case, model, seed), "native"] for seed in design["seeds"]]
            require(all(g == grids[0] for g in grids), "query grid or truth changes across datasets")
    selected_keys = [(key(r), r["method"]) for r in selections]
    require(len(selected_keys) == 5*len(expected) and set(selected_keys) ==
            {(identity, method) for identity in expected for method in design["methods"] if method != "native"},
            "missing or duplicated diagnostic estimator")
    curvature_keys = [(key(r), r["step"]) for r in curvature]
    require(len(curvature_keys) == len(expected)*len(design["local_steps"]) and set(curvature_keys) ==
            {(identity, h) for identity in expected for h in design["local_steps"]},
            "missing or duplicated profile-curvature controls")


def point_decomposition(rows, requested):
    require(len(rows) == requested and len({r["seed"] for r in rows}) == requested,
            "point decomposition has missing or duplicated datasets")
    good = [r for r in rows if r["status"] == "ok"]
    reference = rows[0]
    representation_bias = reference["oracle"] - reference["truth"]
    result = dict(requested_datasets=requested, available_datasets=len(good),
                  representation_bias=representation_bias,
                  coverage_yield=sum(r["covered"] for r in rows)/requested,
                  known_scale_coverage_yield=sum(r["known_scale_covered"] for r in rows)/requested,
                  decomposition_available=len(good) >= 2)
    fields = ("bias", "bias_mcse", "empirical_sd", "rmse", "covariance_rms_se",
              "covariance_scale_ratio", "bias_to_sd", "known_rms_se", "excess_bias")
    result.update({name: math.nan for name in fields})
    if len(good) >= 2:
        errors = [r["fitted"]-r["truth"] for r in good]
        sampling = sampling_decomposition(errors, [r["se"] for r in good])
        result.update({name: sampling[name] for name in fields if name in sampling})
        result["known_rms_se"] = math.sqrt(statistics.mean(r["known_se"]**2 for r in good))
        result["excess_bias"] = sampling["bias"]-representation_bias
    return result


def dataset_summary(rows):
    good = [r for r in rows if r["status"] == "ok"]
    errors = [r["fitted"]-r["truth"] for r in good]
    return dict(points=len(rows), available_points=len(good), complete=len(good) == len(rows),
                point_coverage_yield=statistics.mean(r["covered"] for r in rows),
                known_scale_coverage_yield=statistics.mean(r["known_scale_covered"] for r in rows),
                mean_width_available=statistics.mean(r["width"] for r in good) if good else math.nan,
                response_rmse_available=math.sqrt(statistics.mean(e*e for e in errors)) if good else math.nan)


def run(source, output):
    source = source.resolve()
    metadata = tomllib.loads((source/"metadata.toml").read_text())
    require(digest(source/"design.toml") == metadata["design_sha256"], "profile design fingerprint differs")
    fingerprints = {str(source/"metadata.toml"): digest(source/"metadata.toml"),
                    str(source/"design.toml"): digest(source/"design.toml")}
    for name, sha in metadata["outputs_sha256"].items():
        file = bounded_file(source, name)
        require(digest(file) == sha, f"output fingerprint differs: {name}")
        fingerprints[str(file)] = sha
    for name, sha in metadata["input_sha256"].items():
        require(digest(bounded_file(ROOT, name)) == sha, f"original input changed: {name}")
    for name, sha in metadata["script_sha256"].items():
        require(digest(bounded_file(source, name)) == sha, f"archived script changed: {name}")
    design = tomllib.loads((source/"design.toml").read_text())
    native = read_rows(source/"native_fits.csv", floats=("rho", "edf", "archived_control_edf"), ints=("seed",))
    profiles = read_rows(source/"profiles.csv",
        floats=("rho", "criterion", "known_criterion", "Q", "edf", "decrement", "native_criterion",
                "native_penalty", "native_logdet_shift", "local_hessian"),
        ints=("seed",), bools=("native_scale_floored",))
    candidates = read_rows(source/"candidates.csv", floats=("rho", "Q"), ints=("seed",))
    points = read_rows(source/"dataset_points.csv",
        floats=("x", "truth", "oracle", "fitted", "se", "known_se", "width"),
        ints=("seed", "point_id"), bools=("covered", "known_scale_covered"))
    selections = read_rows(source/"selections.csv", floats=("rho", "edf"), ints=("seed",))
    curvature = read_rows(source/"curvature.csv", floats=("step",), ints=("seed",))
    audit_population(design, native, profiles, candidates, points, selections, curvature)
    by_point, by_dataset = defaultdict(list), defaultdict(list)
    for row in points:
        by_point[row["case"], row["model"], row["method"], row["point_id"]].append(row)
        by_dataset[row["case"], row["model"], row["method"], row["seed"]].append(row)
    decomposition = []
    for (case, model, method, point), rows in sorted(by_point.items()):
        decomposition.append(dict(case=case, model=model, method=method, point_id=point,
            x=rows[0]["x"], truth=rows[0]["truth"], **point_decomposition(rows, len(design["seeds"]))))
    datasets = [dict(case=case, model=model, method=method, seed=seed, **dataset_summary(rows))
                for (case, model, method, seed), rows in sorted(by_dataset.items())]
    summaries = []
    for case in design["cases"]:
        for model in design["models"]:
            for method in design["methods"]:
                ds = [r for r in datasets if (r["case"], r["model"], r["method"]) == (case, model, method)]
                ps = [r for r in decomposition if (r["case"], r["model"], r["method"]) == (case, model, method)]
                complete = [r for r in ds if r["complete"]]
                rms = lambda name: math.sqrt(statistics.mean(r[name]**2 for r in ps))
                summaries.append(dict(case=case, model=model, method=method, datasets=len(ds),
                    complete_datasets=len(complete),
                    point_coverage_yield=statistics.mean(r["point_coverage_yield"] for r in ds),
                    known_scale_coverage_yield=statistics.mean(r["known_scale_coverage_yield"] for r in ds),
                    coverage_mcse=statistics.stdev(r["point_coverage_yield"] for r in ds)/math.sqrt(len(ds))
                        if len(ds) > 1 else math.nan,
                    mean_width_complete=statistics.mean(r["mean_width_available"] for r in complete)
                        if complete else math.nan,
                    rms_bias=rms("bias"), rms_sampling_sd=rms("empirical_sd"),
                    rms_reported_se=rms("covariance_rms_se"), rms_known_se=rms("known_rms_se"),
                    rms_representation_bias=rms("representation_bias"), rms_excess_bias=rms("excess_bias")))
    diagnostics = []
    for base in native:
        identity = base["case"], base["model"], base["seed"]
        group = [r for r in profiles if (r["case"], r["model"], r["seed"]) == identity and r["status"] == "ok"]
        at_native = next((r for r in group if r["rho"] == base["rho"]), None)
        limit = next((r for r in group if r["rho"] == math.inf), None)
        finite = [r for r in group if math.isfinite(r["rho"])]
        selected = {r["method"]: r for r in selections if (r["case"], r["model"], r["seed"]) == identity}
        diagnostics.append(dict(case=identity[0], model=identity[1], seed=identity[2],
            native_rho=base["rho"], native_edf=base["edf"],
            archived_control_edf=base["archived_control_edf"],
            best_profile_rho=selected["profile_grid"]["rho"], best_profile_edf=selected["profile_grid"]["edf"],
            known_profile_rho=selected["known_scale_grid"]["rho"], known_profile_edf=selected["known_scale_grid"]["edf"],
            native_to_profile_gain=max(r["criterion"] for r in group)-at_native["criterion"]
                if at_native and group else math.nan,
            best_finite_minus_null=max(r["criterion"] for r in finite)-limit["criterion"]
                if finite and limit else math.nan,
            native_criterion_shift=at_native["native_criterion"]-at_native["criterion"]
                if at_native else math.nan,
            native_logdet_shift=at_native["native_logdet_shift"] if at_native else math.nan,
            max_coefficient_decrement=max(r["decrement"] for r in group) if group else math.nan,
            raw_negative_penalties=sum(r["native_penalty"] < 0 for r in finite),
            raw_scale_floors=sum(r["native_scale_floored"] for r in finite)))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("decomposition.csv", decomposition), ("dataset_summary.csv", datasets),
                       ("summary.csv", summaries), ("profile_diagnostics.csv", diagnostics)):
        write_csv(output/name, rows)
    scripts = [Path(__file__), Path(__file__).with_name("decompose_coverage.py"),
               Path(__file__).with_name("analyze_calibration.py"), Path(__file__).with_name("analyze_multivariate.py")]
    for script in scripts:
        (output/script.name).write_bytes(script.read_bytes())
    require(all(digest(Path(path)) == sha for path, sha in fingerprints.items()), "input changed during analysis")
    (output/"metadata.json").write_text(json.dumps(dict(
        input_sha256=fingerprints, scripts_sha256={p.name: digest(p) for p in scripts},
        native_fits=len(native), coefficient_attempts=len(candidates),
        function_records=len(points), failed_starts=sum(r["status"] != "ok" for r in candidates),
        unavailable_profile_scores=sum(r["status"] != "ok" for r in profiles),
        oracle_scope="Known response is used only for representation/bias diagnosis, not interval construction.",
        uncertainty_scope="Pointwise conditional working intervals; not simultaneous or a new calibrated procedure.",
        missingness="Coverage yield retains all requested datasets; decomposition conditions on available records."
    ), indent=2)+"\n")
    print(f"Analyzed {len(native)} native fits, {len(candidates)} coefficient attempts and {len(points)} function records.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("benchmarks/kan/results/smoothing-profile-diagnostics"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/kan/results/smoothing-profile-analysis"))
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
