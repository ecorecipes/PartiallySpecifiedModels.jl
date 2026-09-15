import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from decompose_coverage import decomposition


class BiasVarianceTests(unittest.TestCase):
    def test_bias_is_not_counted_as_sampling_variance(self):
        result,oracle=decomposition([2.0,3.0,4.0],[1.0]*3,[0.5]*3,[1.0]*3)
        self.assertEqual(result["bias"],3.0)
        self.assertEqual(result["empirical_sd"],1.0)
        self.assertEqual(result["covariance_scale_ratio"],1.0)
        self.assertEqual(result["corrected_bias"],2.5)
        self.assertEqual(oracle,[-1.5,0.0,1.5])

    def test_bias_estimation_noise_changes_corrected_variability(self):
        result,_=decomposition([-1.0,0.0,1.0],[1.0]*3,[1.0,0.0,-1.0],[1.0]*3)
        self.assertEqual(result["corrected_bias"],0.0)
        self.assertEqual(result["corrected_empirical_sd"],2.0)
        self.assertEqual(result["corrected_bootstrap_scale_ratio"],0.5)

    def test_rms_scale_uses_variances(self):
        result,_=decomposition([0.0,1.0,2.0],[1.0,2.0,2.0],[0.0]*3,[1.0]*3)
        self.assertEqual(result["covariance_rms_se"],math.sqrt(3.0))

    def test_invalid_and_incomplete_inputs_are_not_silent(self):
        for errors,se,bias,sd in (([],[],[],[]),([1.],[1.],[0.],[1.]),
            ([1.,2.],[1.],[0.,0.],[1.,1.]),([1.,math.nan],[1.,1.],[0.,0.],[1.,1.]),
            ([1.,2.],[-1.,1.],[0.,0.],[1.,1.])):
            with self.assertRaises(ValueError):
                decomposition(errors,se,bias,sd)


if __name__=="__main__":
    unittest.main()
