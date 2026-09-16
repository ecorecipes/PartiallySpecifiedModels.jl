import json
import hashlib
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks/calibration"))
import analyze
import stages


class StagedCalibrationTests(unittest.TestCase):
    def screen(self, root, stage="develop"):
        result, analysis = root/"results", root/"analysis"
        result.mkdir()
        analysis.mkdir()
        rows = []
        for model, point, width in (("spline8", 0.7, 0.05), ("spline12", 0.95, 0.10)):
            for target, coverage in (("density_pointwise", point), ("density_simultaneous", 0.8)):
                rows.append(dict(operator="growth", family="spline", noise_mode="known",
                    model=model, method="conditional", target=target,
                    coverage_deficit=max(0, 0.95-coverage), worst_coverage=coverage,
                    mean_width=width, min_availability=1.0, complete_population=True, min_datasets=20))
        analyze.write_csv(analysis/"rankings.csv", rows)
        (result/"metadata.toml").write_text('complete = true\ncode_sha256 = "fixture"\n')
        (result/"design.toml").write_text('designs = ["baseline"]\nnoises = ["iid"]\nsigmas = [0.015]\n[cases]\ngrowth = ["affine"]\n')
        meta = dict(schema="calibration-analysis-v1", stage=stage, code_sha256="fixture",
                    seeds=list(range(1, 21)), nsim=10000,
                    options={"models": "spline8,spline12", "methods": "conditional",
                             "operators": "growth", "noise-modes": "known", "level": "0.95"},
                    sources=[str(result.resolve())],
                    input_sha256={str(result/"metadata.toml"): analyze.digest(result/"metadata.toml")},
                    outputs_sha256={"rankings.csv": analyze.digest(analysis/"rankings.csv")})
        (analysis/"metadata.json").write_text(json.dumps(meta))

    def test_shortlist_breaks_simultaneous_ties_with_pointwise_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.screen(root)
            plan = root/"refinement.toml"
            stages.refinement_plan(root, plan, "density_simultaneous", 99, 1)
            actual = tomllib.loads(plan.read_text())
            self.assertEqual({r["model"] for r in actual["participants"]}, {"spline12"})
            self.assertEqual({r["method"] for r in actual["participants"]}, set(stages.BOOTSTRAP_METHODS))
            self.assertEqual(actual["stage"], "develop")
            self.assertEqual(actual["nboot"], 99)
            self.assertTrue(actual["confirmation_not_started"])
            self.assertEqual(actual["parent_metadata_sha256"], analyze.digest(root/"results/metadata.toml"))

    def test_smoke_and_changed_screen_cannot_create_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.screen(root, stage="smoke")
            with self.assertRaisesRegex(ValueError, "development"):
                stages.refinement_plan(root, root/"refinement.toml", "density_simultaneous", 99, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.screen(root)
            (root/"analysis/rankings.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "analysis changed"):
                stages.refinement_plan(root, root/"refinement.toml", "density_simultaneous", 99, 1)

    def test_pilot_cannot_inherit_development_observations_or_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.screen(root)
            stages.refinement_plan(root, root/"refinement.toml", "density_simultaneous", 99, 1)
            stages.pilot_plan(root/"refinement.toml", root/"pilot.toml")
            plan = tomllib.loads((root/"pilot.toml").read_text())
            self.assertEqual(plan["stage"], "smoke")
            self.assertNotIn("parent_results", plan)
            self.assertEqual(plan["nboot"], 3)
            self.assertEqual(plan["phase"], "implementation-pilot")

    def test_refinement_required_lock_rejects_unfinished_refinement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = dict(stage="develop", nboot=99, outputs_sha256={},
                        analysis_sha256=analyze.digest(Path(analyze.__file__)),
                        refinement_complete=False)
            (root/"metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "complete matched"):
                analyze.freeze(root, root/"lock.toml", "density_simultaneous", 1, 500, 999, True)

    def test_toml_plans_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"plan.toml"
            original = dict(schema="test", seeds=[1, 2], shard=[2, 3], nboot=99,
                            options={"noise-modes": "known,estimated"},
                            participants=[dict(model="spline8", method="bootstrap_t")])
            stages.write_plan(path, original)
            self.assertEqual(tomllib.loads(path.read_text()), original)
            with self.assertRaises(FileExistsError):
                stages.write_plan(path, original)

    def test_bootstrap_audit_rejects_attempt_and_rng_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"fits").mkdir()
            path = root/"fits/integral__baseline__iid__0.015__poly_a__1__spline8__known.toml"
            seeds = []
            for b in range(1, 4):
                payload = f"bootstrap|smoke|integral|baseline|iid|0.015|poly_a|1|fit|{b}"
                seeds.append(str(int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) & 0x7fffffffffffffff))
            text = ('[bootstrap.fit]\nvalues = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]\n'
                    'se = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]\n'
                    'generating = [2.0, 2.0, 2.0]\n')
            for b, seed in enumerate(seeds, 1):
                text += f'[[bootstrap.fit.attempts]]\nattempt = {b}\nrng_seed = "{seed}"\nstatus = "ok"\n'
            path.write_text(text)
            design = dict(methods=["bootstrap_t"], participants=[], stage="smoke", ngrid=1, nboot=3)
            interval = dict(operator="integral", design="baseline", noise="iid", sigma=0.015,
                            case="poly_a", seed=1, model="spline8", noise_mode="known",
                            method="bootstrap_t", available=True)
            analyze.audit_bootstrap(root, design, [interval])
            path.write_text(text.replace("attempt = 2", "attempt = 1"))
            with self.assertRaisesRegex(ValueError, "missing, duplicated"):
                analyze.audit_bootstrap(root, design, [interval])
            path.write_text(text.replace(seeds[0], "wrong"))
            with self.assertRaisesRegex(ValueError, "RNG lineage"):
                analyze.audit_bootstrap(root, design, [interval])
            path.write_text(text.replace('status = "ok"', 'status = "failed"', 1))
            with self.assertRaisesRegex(ValueError, "failed bootstrap"):
                analyze.audit_bootstrap(root, design, [interval])

    def test_prepared_snapshot_is_published_complete_and_tamper_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source-input"
            (source/"src").mkdir(parents=True)
            (source/"ext").mkdir()
            for name in stages.SCIENTIFIC_FILES+["src/PartiallySpecifiedModels.jl"]:
                path = source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# file-copy fixture\n")
            plan = dict(schema="calibration-execution-plan-v1", phase="implementation-pilot",
                        stage="smoke", source_root=str(source), options={}, participants=[])
            stages.write_plan(root/"plan.toml", plan)
            destination = root/"run"
            stages.prepare(root/"plan.toml", destination, "1/2")
            inventory = json.loads((destination/"execution.json").read_text())
            self.assertTrue(all(analyze.digest(destination/name) == sha
                                for name, sha in inventory["files_sha256"].items()))
            self.assertEqual(tomllib.loads((destination/"plan.toml").read_text())["shard"], [1, 2])
            self.assertEqual(list(root.glob(".prepare-*")), [])
            with self.assertRaisesRegex(ValueError, "already exists"):
                stages.prepare(root/"plan.toml", destination, "1/2")
            (destination/"plan.toml").write_text("changed = true\n")
            with self.assertRaisesRegex(ValueError, "frozen execution input"):
                stages.run(destination, True, None)

    def test_shared_scientific_source_is_explicit_and_not_made_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"frozen"
            (source/"src").mkdir(parents=True)
            (source/"ext").mkdir()
            for name in stages.SCIENTIFIC_FILES+["src/PartiallySpecifiedModels.jl"]:
                path = source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# shared-source fixture\n")
            plan = dict(schema="calibration-execution-plan-v1", phase="implementation-pilot",
                        stage="smoke", source_root=str(source), options={}, participants=[])
            stages.write_plan(root/"plan.toml", plan)
            destination = root/"run"
            mode = (source/"Project.toml").stat().st_mode
            stages.prepare(root/"plan.toml", destination, "1/1", shared_source=True)
            self.assertTrue((destination/"source").is_symlink())
            self.assertEqual((source/"Project.toml").stat().st_mode, mode)
            with patch.object(stages.subprocess, "run") as run:
                stages.run(destination, True, None)
                self.assertEqual(run.call_args.args[0][1], f"--project={source.resolve()/'benchmarks/kan'}")
            (destination/"source").unlink()
            (destination/"source").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "shared source link"):
                stages.run(destination, True, None)


if __name__ == "__main__":
    unittest.main()
