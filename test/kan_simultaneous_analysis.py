import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_simultaneous_bands import band_score,summarize


class JointCoverageTests(unittest.TestCase):
    def test_joint_is_not_mean_pointwise_coverage(self):
        score=band_score([-1.0,1.0],[1.0,2.0],[0.0,0.0])
        self.assertEqual(score["point_coverage_yield"],0.5)
        self.assertFalse(score["joint_covered"])
        self.assertTrue(score["band_available"])

    def test_missing_point_invalidates_joint_band(self):
        score=band_score([-1.0,math.nan],[1.0,math.nan],[0.0,0.0])
        self.assertFalse(score["band_available"])
        self.assertFalse(score["joint_covered"])
        self.assertEqual(score["point_availability"],0.5)

    def test_dataset_denominator_and_wilson_boundaries(self):
        common=dict(case="c",model="m",grid="g",scope="in_range",source="covariance",
                    interval="simultaneous",level=.95,points=2,critical=2.5)
        rows=[dict(common,seed=1,**band_score([-1.,-1.],[1.,1.],[0.,0.])),
              dict(common,seed=2,**band_score([-1.,1.],[1.,2.],[0.,0.])),
              dict(common,seed=3,**band_score([math.nan,math.nan],[math.nan,math.nan],[0.,0.]))]
        result=summarize(rows)[0]
        self.assertEqual(result["datasets"],3)
        self.assertEqual(result["bands_available"],2)
        self.assertEqual(result["joint_covered"],1)
        self.assertEqual(result["joint_coverage"],0.5)
        self.assertEqual(result["joint_yield"],1/3)
        self.assertLess(result["joint_mc_lower"],result["joint_coverage"])
        self.assertGreater(result["joint_mc_upper"],result["joint_coverage"])
        with self.assertRaises(ValueError):
            summarize(rows+[rows[0]])


if __name__=="__main__":
    unittest.main()
