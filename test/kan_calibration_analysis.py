import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_calibration import clustered_summary, summaries, quantile, wilson


class CalibrationAnalysisTests(unittest.TestCase):
    def test_independent_wilson_reference_and_boundaries(self):
        lo,hi = wilson(95,100)
        # Independent binomial-score values: 0.88824953077, 0.97845632085.
        # Rounded-reference differences <5e-12; allow twice that.
        self.assertAlmostEqual(lo,0.88824953077,delta=1e-11)
        self.assertAlmostEqual(hi,0.97845632085,delta=1e-11)
        lo,hi = wilson(100,100)
        self.assertLess(lo,1.0)
        self.assertAlmostEqual(hi,1.0,delta=1e-15)  # Boundary differs by <=1 ulp; >4x headroom.
        self.assertTrue(all(math.isnan(x) for x in wilson(0,0)))
        with self.assertRaises(ValueError):
            wilson(2,1)

    def test_dataset_not_point_replication(self):
        result = clustered_summary([0.0,1.0,0.0,1.0],[1.0]*4)
        self.assertEqual(result["coverage_given_available"],0.5)
        # Bernoulli scores give sample variance 1/3 and SE sqrt(1/12);
        # the algebra differs by <=1 ulp, with >4x headroom.
        self.assertAlmostEqual(result["mcse_given_available"],math.sqrt(1/12),delta=1e-15)
        self.assertEqual(result["datasets"],4)
        def rows(copies):
            return [dict(case="c",model="m",seed=seed,method="covariance",budget=0,
                         level=0.95,point_id=point,x=float(point),truth=0.0,
                         region="in_range",estimate=0.0 if seed%2 else 2.0,
                         lower=-1.0 if seed%2 else 1.0,upper=1.0 if seed%2 else 3.0,
                         available=True,covered=bool(seed%2),width=2.0)
                    for seed in range(4) for point in range(copies)]
        _,one,_ = summaries(rows(1))
        _,many,_ = summaries(rows(10))
        self.assertEqual(one[0]["mcse_given_available"],many[0]["mcse_given_available"])
        self.assertEqual(many[0]["datasets"],4)

    def test_failed_intervals_do_not_become_coverage(self):
        result = clustered_summary([1.0,0.0,0.0,0.0],[1.0,0.0,1.0,0.0])
        self.assertEqual(result["coverage_given_available"],0.5)
        self.assertEqual(result["available_and_covers"],0.25)
        self.assertEqual(result["availability"],0.5)
        self.assertTrue(math.isnan(clustered_summary([0.0]*4,[0.0]*4)["coverage_given_available"]))
        with self.assertRaises(ValueError):
            clustered_summary([1.0],[0.0])

    def test_percentile_interpolation_ignores_nonfinite_samples(self):
        self.assertEqual(quantile([1.0,3.0,5.0,math.inf,math.nan],0.25),2.0)
        self.assertEqual(quantile([1.0,3.0,5.0],0.75),4.0)
        self.assertTrue(math.isnan(quantile([math.nan],0.5)))


if __name__ == "__main__":
    unittest.main()
