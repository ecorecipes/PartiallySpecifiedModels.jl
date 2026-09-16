import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

from analyze import csv_rows, digest, require, toml_string

HERE = Path(__file__).resolve().parent
BOOTSTRAP_METHODS = ["bootstrap_percentile", "bootstrap_t", "bootstrap_t_pilot"]
SCIENTIFIC_FILES = [
    "Project.toml", "benchmarks/kan/Project.toml", "benchmarks/kan/Manifest.toml",
    "benchmarks/kan/compare.jl", "benchmarks/kan/smoothing_profiles.jl",
    "benchmarks/kan/coverage_robustness.jl", "benchmarks/kan/undersmoothing.jl",
    "benchmarks/kan/calibrate_families.jl", "benchmarks/kan/calibrate_uncertainty.jl",
    "benchmarks/kan/rd_archive.jl", "benchmarks/kan/reaction_diffusion.jl",
    "benchmarks/calibration/models.jl", "benchmarks/calibration/inference.jl",
    "benchmarks/calibration/run.jl", "benchmarks/calibration/analyze.py"]


def write_plan(path, plan):
    scalars = {k: v for k, v in plan.items() if not isinstance(v, dict) and k != "participants"}
    encode = lambda value: json.dumps(value, ensure_ascii=True, allow_nan=False)
    lines = [f"{toml_string(k)} = {encode(v)}" for k, v in scalars.items()]
    for section, values in plan.items():
        if isinstance(values, dict):
            lines += ["", f"[{toml_string(section)}]"]
            lines += [f"{toml_string(k)} = {encode(v)}" for k, v in sorted(values.items())]
    for row in plan.get("participants", []):
        lines += ["", "[[participants]]"]
        lines += [f"{toml_string(k)} = {encode(v)}" for k, v in row.items()]
    with path.open("x") as handle:
        handle.write("\n".join(lines)+"\n")


def audit_analysis(directory):
    metadata = json.loads((directory/"metadata.json").read_text())
    require(metadata["schema"] == "calibration-analysis-v1" and metadata["stage"] == "develop",
            "shortlisting requires a development analysis")
    for name, sha in metadata["outputs_sha256"].items():
        require(digest(directory/name) == sha, f"development analysis changed: {name}")
    for name, sha in metadata["input_sha256"].items():
        require(digest(Path(name)) == sha, "development input manifest changed")
    return metadata


def refinement_plan(screen, output, target, nboot, top):
    screen = screen.resolve()
    analysis = screen/"analysis"
    metadata = audit_analysis(analysis)
    require(top >= 1 and nboot >= 99, "refinement requires positive shortlist size and at least 99 attempts")
    rankings = csv_rows(analysis/"rankings.csv")
    other = "density_pointwise" if target == "density_simultaneous" else "density_simultaneous"
    secondary = {(r["operator"], r["family"], r["noise_mode"], r["model"]): r
                 for r in rankings if r["method"] == "conditional" and r["target"] == other}
    groups = defaultdict(list)
    for row in rankings:
        if row["method"] == "conditional" and row["target"] == target:
            require(row["complete_population"] == "True" and int(row["min_datasets"]) >= 20,
                    "incomplete or undersized screen cannot define a shortlist")
            groups[row["operator"], row["family"], row["noise_mode"]].append(row)
    require(groups, "no conditional-estimator candidates")
    chosen, participants = [], []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (
            float(r["coverage_deficit"]),
            float(secondary[key+(r["model"],)]["coverage_deficit"]),
            1-float(r["min_availability"]), float(r["mean_width"]), r["model"]))
        for row in rows[:top]:
            chosen.append(row)
            participants += [dict(operator=row["operator"], family=row["family"], noise_mode=row["noise_mode"],
                                 model=row["model"], method=method, target=target)
                             for method in BOOTSTRAP_METHODS]
    options = dict(metadata["options"])
    options["models"] = ",".join(sorted({row["model"] for row in chosen}))
    options["methods"] = ",".join(BOOTSTRAP_METHODS)
    parent = screen/"results"
    parent_meta = tomllib.loads((parent/"metadata.toml").read_text())
    require(parent_meta["complete"] and parent_meta["code_sha256"] == metadata["code_sha256"],
            "screen results are incomplete or use different code")
    design = tomllib.loads((parent/"design.toml").read_text())
    for source in metadata["sources"]:
        require(Path(source).resolve() == parent, "refinement currently requires one complete screen archive")
    plan = dict(schema="calibration-execution-plan-v1", phase="bootstrap-refinement", stage="develop",
        code_sha256=metadata["code_sha256"], source_root=str(screen/"source"),
        parent_results=str(parent), parent_metadata_sha256=digest(parent/"metadata.toml"),
        nboot=nboot, nsim=metadata["nsim"], seeds=metadata["seeds"], shard=[1, 1],
        target=target, selection_secondary_target=other, shortlist_size_per_family=top,
        selection_rule="conditional estimator: primary worst-stratum deficit, secondary deficit, availability, width, model ID",
        options=options, participants=participants,
        lineage=dict(screen_analysis=str(analysis), screen_analysis_sha256=digest(analysis/"metadata.json"),
                     screen_design_sha256=digest(parent/"design.toml"),
                     selection_stage="development only; not a confirmed calibration claim",
                     screen_methods="conditional,split_bias_bank,split_bias_ellipsoid"),
        confirmation_not_started=True, cases=design["cases"])
    write_plan(output, plan)
    fit_jobs = sum(len(design["cases"][op])*len(design["designs"])*len(design["noises"])*
                   len(design["sigmas"])*len(plan["seeds"])
                   for op, family, mode in groups for _ in range(min(top, len(groups[op, family, mode]))))
    print(f"Frozen {len(chosen)} conditional-estimator choices; {fit_jobs} parent fits; "
          f"{2*nboot*fit_jobs} bootstrap attempts for two generators.")


def audit_source(source):
    required = ["Project.toml", "src/PartiallySpecifiedModels.jl", "benchmarks/kan/Project.toml",
                "benchmarks/kan/Manifest.toml", "benchmarks/calibration/run.jl"]
    require(all((source/name).is_file() for name in required), "source root lacks the frozen package/environment")


def pilot_plan(refinement, output):
    original = tomllib.loads(refinement.read_text())
    require(original["schema"] == "calibration-execution-plan-v1" and original["phase"] == "bootstrap-refinement",
            "pilot requires a refinement plan")
    options = dict(original["options"])
    options.update(iterations="8", rhos="-8.0,-4.0", sigmas="0.015", ngrid="11")
    plan = dict(schema="calibration-execution-plan-v1", phase="implementation-pilot", stage="smoke",
        code_sha256=original["code_sha256"], source_root=original["source_root"],
        nboot=3, nsim=200, seeds=[1], shard=[1, 1],
        options=options, participants=original["participants"],
        lineage=dict(refinement_plan_sha256=digest(refinement),
                     scope="implementation qualification only; cannot produce a confirmation lock"))
    write_plan(output, plan)
    print("Prepared a small implementation pilot; not a development or confirmation cohort.")


def prepare(plan_file, output, shard, shared_source=False):
    require(not output.exists(), "execution directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    prepare_at(plan_file, staging/"complete", shard, shared_source)
    (staging/"complete").rename(output)
    staging.rmdir()
    print(f"Prepared {output}; no fitting started.")


def prepare_at(plan_file, output, shard, shared_source=False):
    plan = tomllib.loads(plan_file.read_text())
    require(plan["schema"] == "calibration-execution-plan-v1", "unknown execution-plan schema")
    i, n = map(int, shard.split("/"))
    require(1 <= i <= n, "shard must satisfy 1 <= i <= n")
    require(not output.exists(), "execution directory already exists")
    source = Path(plan["source_root"]).resolve()
    audit_source(source)
    output.mkdir(parents=True)
    if shared_source:
        (output/"source").symlink_to(source, target_is_directory=True)
    else:
        # Copy only the scientific inputs, never earlier result directories.
        for directory in ("src", "ext"):
            shutil.copytree(source/directory, output/"source"/directory)
        for name in SCIENTIFIC_FILES:
            destination = output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, destination)
    (output/"adapters").mkdir()
    for name in ("checkpointed.jl", "resume.jl", "stages.py", "analyze.py"):
        shutil.copyfile(HERE/name, output/"adapters"/name)
    if plan["stage"] == "confirm":
        lock = Path(plan["confirmation_lock"])
        require(digest(lock) == plan["confirmation_lock_sha256"], "confirmation lock changed")
        shutil.copyfile(lock, output/"confirmation-lock.toml")
    plan["shard"] = [i, n]
    plan["parent_plan_sha256"] = digest(plan_file)
    write_plan(output/"plan.toml", plan)
    source_files = [source/name for name in SCIENTIFIC_FILES]
    source_files += [file for directory in ("src", "ext") for file in (source/directory).rglob("*") if file.is_file()]
    inventory = {"source/"+str(file.relative_to(source)): digest(file) for file in source_files}
    inventory.update({str(file.relative_to(output)): digest(file) for file in (output/"adapters").iterdir()})
    inventory["plan.toml"] = digest(output/"plan.toml")
    if (output/"confirmation-lock.toml").exists():
        inventory["confirmation-lock.toml"] = digest(output/"confirmation-lock.toml")
    (output/"execution.json").write_text(json.dumps(dict(schema="checkpointed-calibration-v1",
        files_sha256=inventory, plan_sha256=digest(output/"plan.toml"),
        shared_source_root=str(source) if shared_source else "",
        adapter_policy="scientific kernels unchanged; checkpoints are separate orchestration",
        initial_plan=str(plan_file.resolve())), indent=2)+"\n")
    if not shared_source:
        for path in (output/"source").rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~0o222)
    for path in (output/"adapters").iterdir():
        path.chmod(path.stat().st_mode & ~0o222)


def run(directory, plan_only, job_limit):
    directory = directory.resolve()
    inventory = json.loads((directory/"execution.json").read_text())
    require(inventory["schema"] == "checkpointed-calibration-v1", "unknown execution inventory")
    shared = Path(inventory["shared_source_root"]).resolve() if inventory.get("shared_source_root") else None
    if shared:
        require((directory/"source").resolve() == shared, "shared source link changed")
    for name, sha in inventory["files_sha256"].items():
        path = (directory/name).resolve()
        allowed = shared if shared and name.startswith("source/") else directory
        require(path.is_relative_to(allowed) and digest(path) == sha, f"frozen execution input changed: {name}")
    environment = dict(os.environ, JULIA_NUM_THREADS="1", JULIA_NUM_PRECOMPILE_TASKS="1", OPENBLAS_NUM_THREADS="1")
    project_source = shared if shared else directory/"source"
    command = [os.environ.get("CALIBRATION_JULIA", "julia"),
               f"--project={project_source/'benchmarks/kan'}", "--check-bounds=yes",
               str(directory/"adapters/checkpointed.jl"), str(directory)]
    if plan_only:
        command.append("--plan-only")
    if job_limit is not None:
        require(job_limit > 0, "job limit must be positive")
        command.append(f"--job-limit={job_limit}")
    with (directory/"execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("a continuation is already active")
        subprocess.run(command, check=True, env=environment)
        if not plan_only and (directory/"results/metadata.toml").is_file():
            destination = directory/"analysis"
            if destination.exists():
                metadata = json.loads((destination/"metadata.json").read_text())
                require(metadata["schema"] == "calibration-analysis-v1", "unexpected existing analysis")
                require(all(digest(destination/name) == sha for name, sha in metadata["outputs_sha256"].items()),
                        "existing analysis output changed")
            else:
                staging = Path(tempfile.mkdtemp(prefix=".analysis-", dir=directory))
                subprocess.run(["python3", "-B", str(directory/"adapters/analyze.py"),
                    "summarize", f"--inputs={directory/'results'}", f"--output={staging/'complete'}"], check=True)
                (staging/"complete").rename(destination)
                staging.rmdir()


def confirmation_plan(lock_path, source_root, output):
    lock = tomllib.loads(lock_path.read_text())
    require(lock["schema"] == "calibration-lock-v1" and lock["source_stage"] == "develop",
            "confirmation requires a development-only lock")
    require(not lock.get("refinement_required", False) or lock.get("refinement_complete", False),
            "confirmation lock requires unfinished bootstrap refinement")
    source_root = source_root.resolve()
    audit_source(source_root)
    plan = dict(schema="calibration-execution-plan-v1", phase="held-out-confirmation", stage="confirm",
        source_root=str(source_root), code_sha256=lock["code_sha256"],
        confirmation_lock=str(lock_path.resolve()), confirmation_lock_sha256=digest(lock_path),
        nboot=lock["confirmation_bootstrap"], nsim=lock["confirmation_nsim"],
        seeds=list(range(1, lock["confirmation_datasets"]+1)), shard=[1, 1],
        options=lock["options"], participants=lock["participants"],
        lineage=dict(development_analysis_sha256=lock["analysis_sha256"], selection_frozen=True))
    write_plan(output, plan)
    print("Prepared a locked confirmation plan; no confirmation observations generated.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    shortlist = commands.add_parser("shortlist")
    shortlist.add_argument("--screen", type=Path, required=True)
    shortlist.add_argument("--output", type=Path, required=True)
    shortlist.add_argument("--target", choices=["density_pointwise", "density_simultaneous"], default="density_simultaneous")
    shortlist.add_argument("--nboot", type=int, default=99)
    shortlist.add_argument("--top", type=int, default=1)
    setup = commands.add_parser("prepare")
    setup.add_argument("--plan", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--shard", default="1/1")
    setup.add_argument("--shared-source", action="store_true")
    pilot = commands.add_parser("pilot-plan")
    pilot.add_argument("--refinement", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--directory", type=Path, required=True)
    execute.add_argument("--plan-only", action="store_true")
    execute.add_argument("--job-limit", type=int)
    confirm = commands.add_parser("confirmation-plan")
    confirm.add_argument("--lock", type=Path, required=True)
    confirm.add_argument("--source-root", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shortlist":
        refinement_plan(args.screen, args.output, args.target, args.nboot, args.top)
    elif args.command == "prepare":
        prepare(args.plan, args.output, args.shard, args.shared_source)
    elif args.command == "run":
        run(args.directory, args.plan_only, args.job_limit)
    elif args.command == "pilot-plan":
        pilot_plan(args.refinement, args.output)
    else:
        confirmation_plan(args.lock, args.source_root, args.output)


if __name__ == "__main__":
    main()
