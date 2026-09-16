import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_undersmoothing import paired,procedure_ok


class UndersmoothingAnalysisTests(unittest.TestCase):
    def test_fraction_applies_to_selected_not_initial_smoothing(self):
        p=dict(name="select_then_fixed_refit",fraction=.25,selected_lambda=4.,final_lambda=1.)
        self.assertTrue(procedure_ok(p,.25))
        self.assertFalse(procedure_ok(dict(p,final_lambda=.25),.25))
        self.assertFalse(procedure_ok(p,1.))

    def test_same_strength_control_is_paired_by_dataset(self):
        specs={"control":dict(base_model="kan",smoothing_fraction=1.),
               "quarter":dict(base_model="kan",smoothing_fraction=.25)}
        common=dict(case="nonlinear",method="covariance",budget=0,level=.95,region="in_range",seed=1)
        rows=[dict(common,model="control",available_and_covers=.2),
              dict(common,model="quarter",available_and_covers=.7)]
        result=paired(rows,specs)
        self.assertEqual(result[0]["paired_datasets"],1)
        # Binary floating subtraction of .7-.2 is within one ulp of .5.
        self.assertAlmostEqual(result[0]["coverage_yield_difference"],.5,delta=1e-15)
        with self.assertRaises(ValueError):
            paired(rows[1:],specs)


if __name__=="__main__":
    unittest.main()
