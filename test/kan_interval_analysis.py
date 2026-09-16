import math
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from compare_uncertainty_methods import bootstrap_intervals, method_pairs


class IntervalMethodTests(unittest.TestCase):
    def test_basic_reflects_the_correct_tails(self):
        intervals = {r["method"]:r for r in bootstrap_intervals(10.0,[6.0,7.0,8.0,9.0,10.0],0.5)}
        basic = intervals["bootstrap_basic"]
        self.assertEqual((basic["lower"],basic["upper"]),(11.0,13.0))
        self.assertEqual(basic["bootstrap_bias"],-2.0)
        self.assertEqual(basic["bootstrap_sd"],math.sqrt(2.5))

    def test_normal_and_bias_corrected_centres(self):
        intervals = {r["method"]:r for r in bootstrap_intervals(10.0,[6.0,7.0,8.0,9.0,10.0],0.95)}
        normal,corrected = intervals["bootstrap_normal"],intervals["bootstrap_normal_bc"]
        self.assertEqual((normal["lower"]+normal["upper"])/2,10.0)
        self.assertEqual((corrected["lower"]+corrected["upper"])/2,12.0)
        # Both use precisely the same scale; subtraction differs by at
        # most one ulp at this magnitude, with >50x headroom.
        self.assertAlmostEqual(normal["upper"]-normal["lower"],
                               corrected["upper"]-corrected["lower"],delta=1e-13)

    def test_unavailable_inputs_and_finite_sample_counts(self):
        for samples in ([],[1.0,2.0],[math.nan,1.0,2.0,math.inf]):
            rows = bootstrap_intervals(1.0,samples,0.95)
            self.assertTrue(all(r["status"]=="unavailable" for r in rows))
            self.assertTrue(all(math.isnan(r["lower"]) for r in rows))
        self.assertTrue(all(r["status"]=="unavailable" for r in bootstrap_intervals(math.nan,[1.,2.,3.],.95)))
        self.assertTrue(all(r["usable"]==3 for r in bootstrap_intervals(2.,[1.,2.,3.,math.inf],.95)))
        with self.assertRaises(ValueError):
            bootstrap_intervals(1.,[1.,2.,3.],1.)

    def test_pairing_is_with_the_same_dataset(self):
        rows = []
        for seed,a,b in ((1,1.0,0.0),(2,0.0,1.0),(3,1.0,1.0)):
            for method,value in (("bootstrap",a),("bootstrap_basic",b)):
                rows.append(dict(case="c",model="m",method=method,budget=99,level=.95,
                                 region="in_range",seed=seed,available_and_covers=value))
        result = method_pairs(rows,99)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["mean_coverage_yield_difference"],0.0)
        self.assertEqual(result[0]["datasets"],3)
        self.assertEqual(result[0]["mcse"],statistics.stdev([-1.,1.,0.])/math.sqrt(3))


if __name__ == "__main__":
    unittest.main()
