"""Methodological safeguards for deterministic temporal baselines."""

import unittest
import numpy as np
import pandas as pd

from src.baseline_models import CORE, FEATURE_SETS, make_pipeline, temporal_split


def sample() -> pd.DataFrame:
    rows=[]
    for month in range(7,13):
        for project in range(1,7):
            rows.append({"identity_key":f"P:{project}","prediction_month":f"2025-{month:02d}","sector":"A" if project%2 else "B","progress_current":month+project,"future_schedule_later_3m":int((month+project)%3==0)})
    for year_month in ("2026-01","2026-02","2026-03","2026-04"):
        for project in range(1,7): rows.append({"identity_key":f"P:{project}","prediction_month":year_month,"sector":"A" if project%2 else "B","progress_current":10+project,"future_schedule_later_3m":project%2})
    return pd.DataFrame(rows)


class BaselineSafeguardsTest(unittest.TestCase):
    def test_split_is_deterministic_and_forward(self):
        a=temporal_split(sample(),"future_schedule_later_3m","schedule"); b=temporal_split(sample(),"future_schedule_later_3m","schedule")
        self.assertEqual([len(a[x]) for x in a],[len(b[x]) for x in b]); self.assertLess(a["train"].prediction_month.max(),a["test"].prediction_month.min())

    def test_target_integrity_and_no_missing_target(self):
        d=sample(); self.assertEqual(set(d.future_schedule_later_3m),{0,1}); self.assertFalse(d.future_schedule_later_3m.isna().any())

    def test_feature_count_consistency(self):
        self.assertEqual(len(CORE),21); self.assertEqual(len(FEATURE_SETS["CORE_PLUS_CONDITIONAL"]),29); self.assertEqual(len(set(FEATURE_SETS["CORE_PLUS_CONDITIONAL"])),29)

    def test_deterministic_preprocessing_and_model(self):
        d=sample(); features=["sector","progress_current"]
        p1=make_pipeline(features,"Logistic Regression"); p2=make_pipeline(features,"Logistic Regression")
        p1.fit(d[features],d.future_schedule_later_3m); p2.fit(d[features],d.future_schedule_later_3m)
        np.testing.assert_allclose(p1.predict_proba(d[features]),p2.predict_proba(d[features]))

    def test_preprocessing_not_fit_on_test(self):
        d=sample(); s=temporal_split(d,"future_schedule_later_3m","schedule"); features=["sector","progress_current"]
        pipe=make_pipeline(features,"Logistic Regression"); pipe.fit(s["train"][features],s["train"].future_schedule_later_3m)
        median=pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_[0]
        self.assertEqual(median,s["train"].progress_current.median()); self.assertNotEqual(median,s["test"].progress_current.median())


if __name__=="__main__": unittest.main()
