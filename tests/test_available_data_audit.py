"""Phase 15 maturity-safe walk-forward tests."""

import unittest
import numpy as np
import pandas as pd

from src.available_data_audit import (TARGET, fixed_threshold_metrics,
    maturity_windows, walk_forward)


def fixture():
    rows=[]
    for month in pd.period_range("2025-07","2026-04",freq="M"):
        for project in range(40):
            row={"project_code":str(project),"identity_key":f"P:{project}","prediction_month":str(month),TARGET:int((project+month.month)%4==0)}
            for feature in __import__('src.available_data_audit',fromlist=['FEATURES']).FEATURES: row[feature]="A" if feature in {"sector","state","ministry","agency"} else float(project+month.month)
            rows.append(row)
    return pd.DataFrame(rows)


class AvailableDataAuditTest(unittest.TestCase):
    def test_windows_are_chronological_and_label_mature(self):
        windows=maturity_windows(fixture()); self.assertEqual(len(windows),6); self.assertTrue((windows.training_period_end<windows.evaluation_period).all()); self.assertTrue((windows.training_label_endpoint<windows.evaluation_period).all())

    def test_may_to_july_are_not_evaluation_folds(self):
        self.assertEqual(maturity_windows(fixture()).evaluation_period.max(),"2026-04")

    def test_fixed_threshold_calculation(self):
        score=fixed_threshold_metrics(pd.Series([0,0,1,1]),np.array([.2,.6,.4,.8]),.5); self.assertEqual((score['tn'],score['fp'],score['fn'],score['tp']),(1,1,1,1))

    def test_walk_forward_is_deterministic(self):
        sets={"STABLE":["progress_current"]}; a,_,_=walk_forward(fixture(),sets); b,_,_=walk_forward(fixture(),sets); pd.testing.assert_frame_equal(a,b)

    def test_thresholds_are_fixed_not_selected(self):
        out,_,_=walk_forward(fixture(),{"STABLE":["progress_current"]}); self.assertIn("f1_at_40",out); self.assertIn("f1_at_50",out); self.assertNotIn("selected_threshold",out)


if __name__=="__main__": unittest.main()
