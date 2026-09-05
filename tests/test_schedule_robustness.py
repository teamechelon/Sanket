"""Robustness-analysis invariants."""

import unittest
import numpy as np
import pandas as pd

from src.schedule_robustness import (TARGET, alternative_split,
    calibration_bins, monthly_distribution, project_disjoint_split,
    threshold_table)


def sample():
    rows=[]
    months=["2025-07","2025-08","2025-09","2025-10","2025-11","2025-12","2026-01","2026-02","2026-03","2026-04"]
    for m in months:
        for p in range(100): rows.append({"identity_key":f"P:{p}","prediction_month":m,TARGET:(p+int(m[-2:]))%3==0})
    return pd.DataFrame(rows)


class RobustnessTest(unittest.TestCase):
    def test_monthly_distribution(self):
        out=monthly_distribution(sample()); self.assertEqual(len(out),10); self.assertTrue((out.eligible_rows==100).all()); self.assertTrue((out.positive_rows==out.target_event_count).all())

    def test_no_random_temporal_shuffling(self):
        s=alternative_split(sample()); self.assertLess(s["train"].prediction_month.max(),s["validation"].prediction_month.min()); self.assertLess(s["validation"].prediction_month.max(),s["test"].prediction_month.min())

    def test_project_disjoint_logic_and_determinism(self):
        a=project_disjoint_split(sample()); b=project_disjoint_split(sample()); self.assertEqual([len(x) for x in a.values()],[len(x) for x in b.values()]); ids=[set(x.identity_key) for x in a.values()]; self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])

    def test_threshold_analysis_is_validation_input_only(self):
        y=pd.Series([0,0,1,1]); p=np.array([.1,.4,.6,.9]); out=threshold_table(y,p); self.assertEqual(out.threshold.tolist(),[.2,.3,.4,.5,.6,.7]); self.assertAlmostEqual(out.loc[out.threshold.eq(.5),"recall"].iloc[0],1)

    def test_calibration_calculation(self):
        out=calibration_bins(pd.Series([0,0,1,1]),np.array([.1,.2,.8,.9])); self.assertEqual(out.rows.sum(),4); self.assertAlmostEqual(out.observed_rate.dropna().iloc[-1],1)


if __name__=="__main__": unittest.main()
