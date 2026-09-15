import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks/calibration"))
import analyze
import campaign
import merge_disk
import stages


class DiskAggregationTests(unittest.TestCase):
    def fixture(self, root):
        source = root/"input"
        source.mkdir()
        (source/"metadata.toml").write_text("[outputs_sha256]\n")
        design = dict(stage="confirm", code_sha256="fixture", operators=["growth", "integral"],
            designs=["baseline"], noises=["iid"], noise_modes=["known"], sigmas=[0.015],
            nboot=99, nsim=100, level=0.95, ngrid=2, lock_sha256="lock", seeds=[1, 2],
            confirmation_total=2, models=["spline8"], methods=["conditional"], options={},
            cases={"growth": ["a"], "integral": ["b"]},
            participants=[dict(operator=op, family="spline", model="spline8",
                               noise_mode="known", method="conditional") for op in ("growth", "integral")])
        rows = []
        for seed in (1, 2):
            for interval in ("pointwise", "simultaneous"):
                for point in (1, 2):
                    covered = not (seed == 2 and point == 2)
                    rows.append(dict(operator="growth", case="a", design="baseline", noise="iid",
                        sigma=0.015, model="spline8", family="spline", noise_mode="known",
                        method="conditional", interval=interval, scope="density", seed=seed, target_id=point,
                        x=float(point), truth=0.5, lower=0.0 if covered else 2.0, upper=1.0 if covered else 3.0,
                        available=True, covered=covered, width=1.0, reference_only=False,
                        assumption_satisfied=False))
        return source, design, rows

    def test_clustered_and_joint_coverage_match_the_original_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, design, rows = self.fixture(root)
            with patch.object(analyze, "load", return_value=(design, rows)):
                merge_disk.summarize([source], root/"output")
            records = {r["target"]: r for r in analyze.csv_rows(root/"output/summary.csv")}
            self.assertEqual(float(records["density_pointwise"]["coverage"]), 0.75)
            self.assertEqual(float(records["density_pointwise"]["mcse"]), 0.25)
            self.assertEqual(float(records["density_simultaneous"]["coverage"]), 0.5)
            meta = json.loads((root/"output/metadata.json").read_text())
            # Missing all integral shards must not become "complete" just
            # because the observed growth strata are complete.
            self.assertFalse(meta["confirmation_complete"])

    def test_duplicate_model_datasets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, design, rows = self.fixture(root)
            with patch.object(analyze, "load", return_value=(design, rows)):
                with self.assertRaisesRegex(ValueError, "overlapping model"):
                    merge_disk.summarize([source, source], root/"output")

    def test_disk_pause_prevents_starting_an_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flag = root/"PAUSE"
            flag.write_text("disk reserve")
            with patch.dict(merge_disk.os.environ, {"CALIBRATION_PAUSE_FILE": str(flag)}):
                with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                    merge_disk.summarize([root/"input"], root/"output")
            self.assertFalse((root/"output").exists())


class CampaignStateTests(unittest.TestCase):
    def configure(self, root):
        (root/"controllers").mkdir()
        for name in campaign.CONTROLLERS:
            (root/"controllers"/name).write_text("# orchestration fixture\n")
        (root/"refinement-plan.toml").write_text('phase = "bootstrap-refinement"\n')
        screen = root/"screen"
        (screen/"results").mkdir(parents=True)
        (screen/"results/metadata.toml").write_text("complete = true\n")
        first = root/"existing-first"
        first.mkdir()
        config = dict(schema="calibration-campaign-v1", scope="refinement-first",
            controller_sha256={name: analyze.digest(root/"controllers"/name) for name in campaign.CONTROLLERS},
            refinement_plan_sha256=analyze.digest(root/"refinement-plan.toml"),
            screen=str(screen), screen_metadata_sha256=analyze.digest(screen/"results/metadata.toml"),
            existing_first_refinement=str(first), refinement_shards=2, scientific_protocol="fixture",
            scientific_source=str(root/"source"), minimum_free_bytes=1,
            confirmation_datasets=500, confirmation_bootstrap=999, confirmation_max_datasets_per_shard=1,
            selection_target="density_simultaneous", selection_top_per_family=1)
        (root/"campaign.json").write_text(json.dumps(config))
        (root/"state.json").write_text(json.dumps(dict(phase="prepared", complete=False)))
        return first

    def exercise(self, root, incomplete=False, interrupt=False):
        root = root.resolve()
        first = self.configure(root)
        done = {first}
        events = []
        interrupted = False

        def prepare(plan, output, shard, shared_source=False):
            self.assertTrue(shared_source)
            output.mkdir(parents=True)

        def complete(run, *unused):
            return run in done

        def child(directory, config, command, log, **state):
            nonlocal interrupted
            phase = state["phase"]
            events.append(phase)
            if phase == "confirmation-preflight":
                self.assertFalse(state["confirmation_data_started"])
                (root/"confirmation-preview/preview.toml").write_text(
                    "total_datasets = 2\nfit_jobs = 20\nbootstrap_attempts = 0\n")
            else:
                if phase == "confirmation":
                    self.assertTrue((root/"confirmation-lock.toml").exists())
                    self.assertTrue(root/"refinement/002" in done)
                done.add(Path(state["active_run"]))
                if phase == "confirmation" and interrupt and not interrupted:
                    interrupted = True
                    raise RuntimeError("simulated reboot after committed shard")

        def aggregate(directory, config, sources, output, phase):
            output.mkdir(exist_ok=True)
            value = dict(refinement_complete=not incomplete, confirmation_complete=True)
            (output/"metadata.json").write_text(json.dumps(value))
            events.append(phase+"-aggregate")
            return value

        def freeze(analysis, output, target, top, count, budget, required):
            self.assertTrue(required)
            self.assertEqual((count, budget), (500, 999))
            events.append("lock")
            output.write_text(f'refinement_complete = true\nconfirmation_datasets = 500\n'
                              f'analysis_sha256 = "{analyze.digest(analysis/"metadata.json")}"\n')

        def confirm(lock, source, output):
            events.append("confirmation-plan")
            output.write_text('phase = "held-out-confirmation"\n')

        with patch.object(campaign, "__file__", str(root/"controllers/campaign.py")), \
             patch.object(campaign, "disk_available"), patch.object(campaign, "completed_run", side_effect=complete), \
             patch.object(campaign, "child", side_effect=child), \
             patch.object(campaign, "publish_analysis", side_effect=aggregate), \
             patch.object(stages, "prepare", side_effect=prepare), \
             patch.object(analyze, "freeze", side_effect=freeze), \
             patch.object(stages, "confirmation_plan", side_effect=confirm):
            if incomplete:
                with self.assertRaisesRegex(ValueError, "refinement is incomplete"):
                    campaign.execute(root)
                self.assertNotIn("lock", events)
                self.assertNotIn("confirmation", events)
            else:
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "simulated reboot"):
                        campaign.execute(root)
                campaign.execute(root)
                self.assertEqual(events.count("refinement"), 1)
                self.assertEqual(events.count("lock"), 1)
                self.assertEqual(events.count("confirmation"), 2)
                self.assertLess(events.index("selection-aggregate"), events.index("lock"))
                self.assertLess(events.index("lock"), events.index("confirmation"))
                self.assertTrue(json.loads((root/"state.json").read_text())["complete"])

    def test_confirmation_follows_complete_refinement_and_frozen_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp))

    def test_incomplete_refinement_blocks_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), incomplete=True)

    def test_reboot_does_not_repeat_committed_shards_or_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), interrupt=True)

    def test_disk_reserve_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            free = type("Usage", (), {"free": 10})()
            with patch.object(campaign.shutil, "disk_usage", return_value=free):
                with self.assertRaisesRegex(ValueError, "disk reserve"):
                    campaign.disk_available(Path(tmp), {"minimum_free_bytes": 20})


if __name__ == "__main__":
    unittest.main()
