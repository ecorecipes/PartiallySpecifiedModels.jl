import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/kan"))
from analyze_fresh_coverage import compare_models, prefix_intervals


class FreshAnalysisTests(unittest.TestCase):
    def test_matched_prefix_uses_attempt_ids_and_requested_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"replicates").mkdir()
            (root/"replicates/nonlinear__m__1.toml").write_text(
                'finite_attempts=[1,3,5,6,7]\nbootstrap_status="ok"\n'
                'bootstrap_values=[[1.0],[3.0],[5.0],[60.0],[70.0]]\n')
            template=dict(case="nonlinear",model="m",seed=1,method="covariance",budget=0,
                level=0.5,point_id=1,x=0.5,truth=3.0,estimate=3.0,
                region="in_range",lower=math.nan,upper=math.nan,available=False,covered=False,width=math.nan)
            rows=prefix_intervals(root,[template,dict(template,seed=2)],{1},5)
            self.assertEqual(len(rows),5)
            self.assertTrue(all(r["seed"]==1 for r in rows))
            percentile=next(r for r in rows if r["method"]=="bootstrap")
            self.assertEqual((percentile["lower"],percentile["upper"]),(2.0,4.0))
            self.assertTrue(percentile["available"])
            self.assertFalse(next(r for r in rows if r["method"]=="covariance")["available"])

    def test_pairing_refuses_missing_reference_datasets(self):
        row=dict(case="nonlinear",model="m",method="bootstrap",budget=19,level=.95,
                 region="in_range",seed=1,available_and_covers=1.0)
        with self.assertRaises(ValueError):
            compare_models([row],"spline8")
        result=compare_models([row,dict(row,model="spline8",available_and_covers=0.0)],"spline8")
        self.assertEqual(result[0]["paired_datasets"],1)
        self.assertEqual(result[0]["coverage_yield_difference"],1.0)


if __name__=="__main__":
    unittest.main()
