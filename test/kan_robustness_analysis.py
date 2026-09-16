import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_coverage_robustness import CONDITIONS,truth


class RobustnessAnalysisTests(unittest.TestCase):
    def test_one_factor_changes(self):
        reference=CONDITIONS["reference"]
        self.assertEqual(CONDITIONS["sparse"],dict(reference,step=0.5))
        self.assertEqual(CONDITIONS["noisy"],dict(reference,sigma=0.03))
        self.assertEqual(CONDITIONS["localized"],dict(reference,response="localized"))
        for x in (0.05,0.6,1.2,2.35):
            self.assertEqual(truth("reference",x),truth("sparse",x))
            self.assertEqual(truth("reference",x),truth("noisy",x))

    def test_localized_response_and_equilibria(self):
        # At x=.9 the bump multiplier is exactly1.5; decimal-rounding
        # discrepancy is <=1 ulp, with >40x headroom.
        self.assertAlmostEqual(truth("localized",.9),.7425,delta=1e-14)
        self.assertEqual(truth("localized",2.),0.0)
        self.assertGreater(truth("localized",.9),.9*(1-.9/2))
        with self.assertRaises(ValueError):
            truth("unknown",.9)


if __name__=="__main__":
    unittest.main()
