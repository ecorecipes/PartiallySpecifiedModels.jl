import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_smoothing_profiles import audit_population, dataset_summary, point_decomposition
from decompose_coverage import sampling_decomposition


def point(seed, value, status="ok"):
    return dict(case="noisy", model="spline8", method="native", seed=seed,
                point_id=1, x=0.6, truth=1.0, oracle=1.25, fitted=value,
                se=1.0, known_se=2.0, status=status, width=4.0,
                covered=status == "ok" and abs(value-1) <= 1.9,
                known_scale_covered=status == "ok" and abs(value-1) <= 3.9)


class ProfileAnalysisTests(unittest.TestCase):
    def test_bias_variance_and_representation_are_separate(self):
        rows = [point(1, 3.0), point(2, 4.0), point(3, 5.0)]
        result = point_decomposition(rows, 3)
        self.assertEqual(result["bias"], 3.0)
        self.assertEqual(result["empirical_sd"], 1.0)
        self.assertEqual(result["representation_bias"], 0.25)
        self.assertEqual(result["excess_bias"], 2.75)
        self.assertEqual(result["known_rms_se"], 2.0)
        self.assertEqual(result["coverage_yield"], 0.0)
        self.assertEqual(result["known_scale_coverage_yield"], 2/3)

    def test_failed_records_stay_in_coverage_denominator(self):
        rows = [point(1, 1.0), point(2, 1.0), point(3, math.nan, "unavailable")]
        result = point_decomposition(rows, 3)
        self.assertEqual(result["requested_datasets"], 3)
        self.assertEqual(result["available_datasets"], 2)
        self.assertEqual(result["coverage_yield"], 2/3)
        self.assertTrue(result["decomposition_available"])
        summary = dataset_summary(rows)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["available_points"], 2)
        self.assertEqual(summary["point_coverage_yield"], 2/3)

    def test_one_replicate_does_not_estimate_sampling_variance(self):
        result = point_decomposition([point(1, 2.0)], 1)
        self.assertFalse(result["decomposition_available"])
        self.assertTrue(math.isnan(result["empirical_sd"]))

    def test_sampling_variance_is_not_mean_squared_error(self):
        result = sampling_decomposition([2.0, 3.0, 4.0], [1.0, 2.0, 2.0])
        self.assertEqual(result["empirical_sd"], 1.0)
        self.assertEqual(result["covariance_rms_se"], math.sqrt(3))
        # Exact variance decomposition differs by <=2 ulps on this fixture;
        # 1e-12 permits >200x arithmetic headroom.
        self.assertAlmostEqual(result["rmse"]**2, result["bias"]**2+2/3, delta=1e-12)

    def test_malformed_replicates_are_rejected(self):
        for errors, se in (([], []), ([1], [1]), ([1, 2], [1]),
                           ([1, math.nan], [1, 1]), ([1, 2], [-1, 1])):
            with self.assertRaises(ValueError):
                sampling_decomposition(errors, se)
        with self.assertRaises(ValueError):
            point_decomposition([point(1, 1.0), point(1, 2.0)], 2)

    def test_profile_accounting_includes_all_starts(self):
        design = dict(cases=["noisy"], models=["spline8"], seeds=[1],
                      rho_grid=[0.0], local_steps=[0.1, 0.2],
                      methods=["native", "native_stable", "fixed_selected",
                               "profile_grid", "known_scale_grid", "null_limit"])
        common = dict(case="noisy", model="spline8", seed=1)
        grid = [-0.2, -0.1, 0.0, 0.1, 0.2]
        native = [dict(common, rho=0.0)]
        profiles = [dict(common, rho=r, status="ok", Q=1.0) for r in grid+[math.inf]]
        candidates = [dict(common, rho=r, start=s, status="ok", Q=1.0)
                      for r in grid for s in ("selected", "ascending", "descending")]
        candidates += [dict(common, rho=math.inf, start=s, status="ok", Q=1.0)
                       for s in ("null_initial", "null_selected")]
        points = [dict(point(1, 1.0), method=m) for m in design["methods"]]
        selections = [dict(common, method=m) for m in design["methods"] if m != "native"]
        curvature = [dict(common, step=h) for h in design["local_steps"]]
        audit_population(design, native, profiles, candidates, points, selections, curvature)
        for changed in (candidates[:-1], candidates+[candidates[0]]):
            with self.assertRaises(ValueError):
                audit_population(design, native, profiles, changed, points, selections, curvature)
        overflow = deepcopy(candidates)
        overflow[0]["Q"] = math.inf
        with self.assertRaises(ValueError):
            audit_population(design, native, profiles, overflow, points, selections, curvature)
        wrong = deepcopy(profiles)
        wrong[0]["Q"] = 2.0
        with self.assertRaises(ValueError):
            audit_population(design, native, wrong, candidates, points, selections, curvature)
        with self.assertRaises(ValueError):
            audit_population(design, native, profiles, candidates, points[:-1], selections, curvature)


if __name__ == "__main__":
    unittest.main()
