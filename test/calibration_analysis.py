import csv
import importlib.util
import json
import math
import tempfile
import tomllib
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]/"benchmarks/calibration/analyze.py"
spec = importlib.util.spec_from_file_location("calibration_suite_analysis", MODULE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class CalibrationAnalysisTests(unittest.TestCase):
    def test_joint_coverage_is_not_pointwise_average(self):
        rows = [dict(interval="simultaneous", available=True, covered=c,
                     lower=0.0, upper=1.0, truth=0.5 if c else 2.0, width=1.0)
                for c in (True, False)]
        self.assertEqual(analysis.score_intervals(rows, 0.95)["coverage"], 0)
        for row in rows:
            row["interval"] = "pointwise"
        self.assertEqual(analysis.score_intervals(rows, 0.95)["coverage"], 0.5)
        rows[1].update(available=False, width=math.nan)
        score = analysis.score_intervals(rows, 0.95)
        self.assertFalse(score["available"])
        self.assertEqual(score["coverage"], 0.5)

    def test_applicability_does_not_depend_on_true_membership(self):
        self.assertTrue(analysis.active("split_bias_bank", "growth", "known"))
        self.assertFalse(analysis.active("split_bias_bank", "growth", "estimated"))
        self.assertFalse(analysis.active("split_bias_ellipsoid", "growth", "known"))
        self.assertTrue(analysis.active("split_bias_ellipsoid", "integral", "known"))
        design = dict(stage="smoke", operators=["growth", "integral"],
                      cases={"growth": ["a"], "integral": ["b"]}, designs=["baseline"],
                      noises=["iid"], sigmas=[0.015], seeds=[1, 2], ngrid=7,
                      noise_modes=["known", "estimated"], models=["m"], participants=[],
                      methods=["conditional", "bootstrap_percentile", "split_bias_bank", "split_bias_ellipsoid"])
        self.assertEqual(len(analysis.expected_intervals(design)), 392)

    def test_smoke_cannot_select_a_confirmation_procedure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"metadata.json").write_text(json.dumps(dict(stage="smoke")))
            with self.assertRaisesRegex(ValueError, "only development"):
                analysis.freeze(root, root/"lock.toml", "density_simultaneous", 1, 500, 999)

    def make_development(self, root, omit=False):
        rankings = []
        scores = []
        for model, coverage, width in (("spline8", 0.8, 0.05), ("spline12", 0.95, 0.1)):
            rankings.append(dict(operator="growth", family="spline", noise_mode="known",
                target="density_simultaneous", model=model, method="conditional",
                min_datasets=20, coverage_deficit=max(0, 0.95-coverage),
                worst_coverage=coverage, min_availability=1.0, mean_width=width,
                complete_population=True))
            for seed in range(1, 21):
                if omit and model == "spline12" and seed == 20:
                    continue
                scores.append(dict(operator="growth", model=model, noise_mode="known",
                    method="conditional", target="density_simultaneous", case="a",
                    design="baseline", noise="iid", sigma=0.015, seed=seed))
        analysis.write_csv(root/"rankings.csv", rankings)
        analysis.write_csv(root/"dataset_scores.csv", scores)
        meta = dict(stage="develop", nboot=99, code_sha256="fixture",
                    analysis_sha256=analysis.digest(MODULE), options=dict(level="0.95"),
                    outputs_sha256={p.name: analysis.digest(p) for p in root.glob("*.csv")})
        (root/"metadata.json").write_text(json.dumps(meta))

    def test_development_ranking_is_frozen_without_confirmation_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_development(root)
            output = root/"lock.toml"
            analysis.freeze(root, output, "density_simultaneous", 1, 500, 999)
            locked = tomllib.loads(output.read_text())
            self.assertEqual(locked["source_stage"], "develop")
            self.assertEqual(locked["participants"][0]["model"], "spline12")
            self.assertEqual(locked["confirmation_bootstrap"], 999)
            self.assertTrue(locked["exploratory_selection"])
            with self.assertRaises(FileExistsError):
                analysis.freeze(root, output, "density_simultaneous", 1, 500, 999)

    def test_unmatched_development_cohorts_cannot_be_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_development(root, omit=True)
            with self.assertRaisesRegex(ValueError, "unmatched"):
                analysis.freeze(root, root/"lock.toml", "density_simultaneous", 1, 500, 999)

    def test_wilson_limits_keep_small_samples_uncertain(self):
        low, high = analysis.wilson(10, 10)
        self.assertLess(low, 0.8)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, 0.99)
        low, high = analysis.wilson(0, 10)
        self.assertLess(low, 0.01)
        self.assertGreater(high, 0.2)


if __name__ == "__main__":
    unittest.main()
