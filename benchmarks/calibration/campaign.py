"""Restartable refinement -> locked selection -> held-out confirmation campaign."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib

import analyze
import stages

HERE = Path(__file__).resolve().parent
CONTROLLERS = ("campaign.py", "merge_disk.py", "analyze.py", "stages.py", "checkpointed.jl", "resume.jl")


def atomic_json(path, data):
    fd, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create(directory, refinement, screen, existing_first, datasets, min_free_gib):
    analyze.require(not directory.exists(), "campaign directory already exists")
    plan = tomllib.loads(refinement.read_text())
    analyze.require(plan["phase"] == "bootstrap-refinement" and plan["stage"] == "develop",
                    "campaign must begin with the frozen development refinement")
    analyze.require(datasets >= 1 and min_free_gib > 0, "positive confirmation count and disk reserve required")
    source = Path(plan["source_root"]).resolve()
    stages.audit_source(source)
    meta = tomllib.loads((screen/"results/metadata.toml").read_text())
    analyze.require(meta["complete"] and meta["code_sha256"] == plan["code_sha256"], "screen protocol differs")
    directory.mkdir(parents=True)
    (directory/"controllers").mkdir()
    for name in CONTROLLERS:
        shutil.copyfile(HERE/name, directory/"controllers"/name)
    shutil.copyfile(refinement, directory/"refinement-plan.toml")
    executable = subprocess.check_output(
        ["julia", "-e", "print(joinpath(Sys.BINDIR,Base.julia_exename()))"], text=True).strip()
    version = subprocess.check_output([executable, "--version"], text=True).strip()
    analyze.require(version == "julia version "+meta["julia_version"],
                    "campaign Julia version differs from the completed development screen")
    config = dict(schema="calibration-campaign-v1", scope="refinement-first",
        refinement_plan_sha256=analyze.digest(directory/"refinement-plan.toml"),
        scientific_protocol=plan["code_sha256"], scientific_source=str(source),
        screen=str(screen.resolve()), screen_metadata_sha256=analyze.digest(screen/"results/metadata.toml"),
        existing_first_refinement=str(existing_first.resolve()), refinement_shards=meta["dataset_count"],
        confirmation_datasets=datasets, confirmation_bootstrap=999, confirmation_nsim=10000,
        confirmation_max_datasets_per_shard=100, selection_target=plan["target"],
        selection_top_per_family=1, minimum_free_bytes=int(min_free_gib*1024**3),
        workers=1, julia_executable=executable, julia_version=version,
        controller_sha256={name: analyze.digest(directory/"controllers"/name) for name in CONTROLLERS})
    atomic_json(directory/"campaign.json", config)
    atomic_json(directory/"state.json", dict(phase="prepared", complete=False,
        confirmation_data_started=False, completed_refinement_shards=0, completed_confirmation_shards=0))
    for path in (directory/"controllers").iterdir():
        path.chmod(path.stat().st_mode & ~0o222)
    print(f"Prepared refinement-first campaign with {datasets} confirmation datasets per stratum.")


def update(directory, **changes):
    path = directory/"state.json"
    state = json.loads(path.read_text())
    state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(path, state)


def disk_available(directory, config):
    free = shutil.disk_usage(directory).free
    analyze.require(free >= config["minimum_free_bytes"],
                    f"disk reserve reached: {free/1024**3:.1f} GiB free; campaign paused without changing its protocol")


def child(directory, config, command, log, **state):
    disk_available(directory, config)
    update(directory, **state, active_log=str(log), status="running", last_error="")
    pause = directory/"PAUSE"
    if pause.exists():
        pause.unlink()
    environment = dict(os.environ, CALIBRATION_PAUSE_FILE=str(pause),
                       CALIBRATION_JULIA=config["julia_executable"],
                       JULIA_NUM_THREADS="1", JULIA_NUM_PRECOMPILE_TASKS="1", OPENBLAS_NUM_THREADS="1")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{state.get('phase', 'working')}: {state.get('active_run', log.name)}", flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                free = shutil.disk_usage(directory).free
                if free < config["minimum_free_bytes"] and not pause.exists():
                    pause.write_text(f"disk reserve reached ({free/1024**3:.1f} GiB free)\n")
                    update(directory, status="pausing", last_error=pause.read_text().strip())
        if code:
            reason = pause.read_text().strip() if pause.exists() else f"worker exited with code {code}; inspect {log}"
            raise RuntimeError(reason)
    disk_available(directory, config)


def completed_run(run, protocol, shard, parent_plan_sha=None):
    if not (run/"results/metadata.toml").exists() or not (run/"analysis/metadata.json").exists():
        return False
    plan = tomllib.loads((run/"plan.toml").read_text())
    metadata = tomllib.loads((run/"results/metadata.toml").read_text())
    analysis = json.loads((run/"analysis/metadata.json").read_text())
    analyze.require(plan["shard"] == list(shard) and metadata["code_sha256"] == protocol,
                    "completed shard uses a different plan or scientific protocol")
    analyze.require(parent_plan_sha is None or plan["parent_plan_sha256"] == parent_plan_sha,
                    "completed shard belongs to a different campaign plan")
    analyze.require(metadata["complete"] and analysis["schema"] == "calibration-analysis-v1",
                    "invalid completed shard")
    analyze.require(all(analyze.digest(run/"analysis"/name) == sha
                        for name, sha in analysis["outputs_sha256"].items()), "completed shard analysis changed")
    analyze.require(all(analyze.digest(Path(name)) == sha for name, sha in analysis["input_sha256"].items()),
                    "completed shard results manifest changed")
    return True


def publish_analysis(directory, config, sources, destination, phase):
    if (destination/"metadata.json").exists():
        metadata = json.loads((destination/"metadata.json").read_text())
        analyze.require(all(analyze.digest(destination/name) == sha for name, sha in metadata["outputs_sha256"].items()),
                        "campaign aggregate analysis changed")
        analyze.require(all(analyze.digest(Path(name)) == sha for name, sha in metadata["input_sha256"].items()),
                        "campaign aggregate inputs changed")
        return metadata
    analyze.require(not destination.exists(), "incomplete aggregate output exists")
    source_list = directory/(phase+"-sources.json")
    atomic_json(source_list, [str(p.resolve()) for p in sources])
    staging = Path(tempfile.mkdtemp(prefix=".aggregate-", dir=directory))
    child(directory, config, [sys.executable, "-B", str(directory/"controllers/merge_disk.py"),
        f"--sources-file={source_list}", f"--output={staging/'complete'}"],
        directory/(phase+"-analysis.log"), phase=phase, active_run="disk-backed aggregate analysis")
    (staging/"complete").rename(destination)
    staging.rmdir()
    return json.loads((destination/"metadata.json").read_text())


def execute(directory):
    directory = directory.resolve()
    config = json.loads((directory/"campaign.json").read_text())
    analyze.require(config["schema"] == "calibration-campaign-v1" and config["scope"] == "refinement-first",
                    "unknown campaign policy")
    for name, sha in config["controller_sha256"].items():
        analyze.require(analyze.digest(directory/"controllers"/name) == sha, f"frozen controller changed: {name}")
    frozen_entry = directory/"controllers/campaign.py"
    if Path(__file__).resolve() != frozen_entry:
        subprocess.run([sys.executable, "-B", str(frozen_entry), "run", f"--directory={directory}"], check=True)
        return
    plan = directory/"refinement-plan.toml"
    analyze.require(analyze.digest(plan) == config["refinement_plan_sha256"], "refinement plan changed")
    screen = Path(config["screen"])
    analyze.require(analyze.digest(screen/"results/metadata.toml") == config["screen_metadata_sha256"],
                    "development screen changed")
    with (directory/"campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("this campaign is already active")
        if json.loads((directory/"state.json").read_text()).get("complete"):
            print("Campaign already complete.")
            return
        try:
            disk_available(directory, config)
            controller = directory/"controllers/stages.py"
            refinement = []
            for i in range(1, config["refinement_shards"]+1):
                run = Path(config["existing_first_refinement"]) if i == 1 else directory/"refinement"/f"{i:03d}"
                shard = (i, config["refinement_shards"])
                if not run.exists():
                    stages.prepare(plan, run, f"{i}/{shard[1]}", shared_source=True)
                if not completed_run(run, config["scientific_protocol"], shard, config["refinement_plan_sha256"]):
                    child(directory, config, [sys.executable, "-B", str(controller), "run", f"--directory={run}"],
                          run/"run.log", phase="refinement", active_run=str(run),
                          active_refinement_shard=i, completed_refinement_shards=i-1,
                          confirmation_data_started=False)
                analyze.require(completed_run(run, config["scientific_protocol"], shard, config["refinement_plan_sha256"]),
                                "refinement shard did not complete")
                refinement.append(run/"results")
                update(directory, completed_refinement_shards=i)
            development = directory/"development-analysis"
            metadata = publish_analysis(directory, config, [screen/"results", *refinement], development, "selection")
            analyze.require(metadata["refinement_complete"], "matched refinement is incomplete; confirmation is blocked")
            lock_path = directory/"confirmation-lock.toml"
            if not lock_path.exists():
                update(directory, phase="selection", active_run="freezing procedures", confirmation_data_started=False)
                analyze.freeze(development, lock_path, config["selection_target"], config["selection_top_per_family"],
                               config["confirmation_datasets"], config["confirmation_bootstrap"], True)
            selection = tomllib.loads(lock_path.read_text())
            analyze.require(selection["refinement_complete"] and
                            selection["confirmation_datasets"] == config["confirmation_datasets"] and
                            selection["analysis_sha256"] == analyze.digest(development/"metadata.json"),
                            "confirmation lock differs from the approved campaign")
            confirmation_plan = directory/"confirmation-plan.toml"
            if not confirmation_plan.exists():
                stages.confirmation_plan(lock_path, Path(config["scientific_source"]), confirmation_plan)
            confirmation_plan_sha = analyze.digest(confirmation_plan)
            preview = directory/"confirmation-preview"
            if not preview.exists():
                stages.prepare(confirmation_plan, preview, "1/1", shared_source=True)
            if not (preview/"preview.toml").exists():
                child(directory, config, [sys.executable, "-B", str(controller), "run",
                    f"--directory={preview}", "--plan-only"], preview/"preview.log",
                    phase="confirmation-preflight", active_run="locked confirmation workload",
                    confirmation_data_started=False)
            workload = tomllib.loads((preview/"preview.toml").read_text())
            total = workload["total_datasets"]
            shards = math.ceil(total/config["confirmation_max_datasets_per_shard"])
            update(directory, confirmation_dataset_count=total, confirmation_fit_jobs=workload["fit_jobs"],
                   confirmation_bootstrap_attempts=workload["bootstrap_attempts"], confirmation_shards=shards)
            confirmations = []
            for i in range(1, shards+1):
                run = directory/"confirmation"/f"{i:05d}"
                if not run.exists():
                    stages.prepare(confirmation_plan, run, f"{i}/{shards}", shared_source=True)
                if not completed_run(run, config["scientific_protocol"], (i, shards), confirmation_plan_sha):
                    child(directory, config, [sys.executable, "-B", str(controller), "run", f"--directory={run}"],
                          run/"run.log", phase="confirmation", active_run=str(run), active_confirmation_shard=i,
                          completed_confirmation_shards=i-1, confirmation_data_started=True)
                analyze.require(completed_run(run, config["scientific_protocol"], (i, shards), confirmation_plan_sha),
                                "confirmation shard did not complete")
                confirmations.append(run/"results")
                update(directory, completed_confirmation_shards=i)
            final = publish_analysis(directory, config, confirmations, directory/"confirmation-analysis", "report")
            analyze.require(final["confirmation_complete"], "confirmation population is incomplete")
            update(directory, phase="complete", complete=True, status="complete", active_run="",
                   report=str(directory/"confirmation-analysis"), last_error="")
            print("Complete confirmation published.", flush=True)
        except Exception as error:
            update(directory, status="blocked", complete=False, last_error=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("create")
    setup.add_argument("--directory", type=Path, required=True)
    setup.add_argument("--refinement", type=Path, required=True)
    setup.add_argument("--screen", type=Path, required=True)
    setup.add_argument("--existing-first", type=Path, required=True)
    setup.add_argument("--datasets", type=int, default=500)
    setup.add_argument("--min-free-gib", type=float, default=50)
    run = commands.add_parser("run")
    run.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create(args.directory, args.refinement, args.screen, args.existing_first, args.datasets, args.min_free_gib)
    else:
        execute(args.directory)
