"""Proofs of point-in-time feature and target alignment."""

import hashlib
import unittest

import pandas as pd

from src.feature_engineering import build_features, feature_inventory, modeling_tables
from src.label_feasibility import load_data


def synthetic() -> pd.DataFrame:
    rows=[]
    for month, progress, expenditure, revised_cost, revised_doc in [
        (1,20,20,None,None),(2,30,30,None,None),(3,45,45,None,None),
        (4,50,55,120,"09/2026"),(5,60,65,120,"09/2026"),(6,70,75,120,"09/2026"),(7,80,85,120,"09/2026")]:
        rows.append(dict(report_month=f"2026-{month:02d}", project_code="1", legacy_ocms_code=pd.NA,
            identity_key="P:1", project_name="A", agency="Agency", ministry="Ministry", sector="Sector", state="State",
            date_of_approval="01/2025", start_date="01/2025", original_doc="06/2026", revised_doc=revised_doc,
            original_cost=100, revised_cost=revised_cost, expenditure=expenditure, physical_progress=progress,
            source_file="x.pdf", source_page=1, source_section="Table 6", quality_flags="NONE"))
    d=pd.DataFrame(rows); d["month_number"]=pd.PeriodIndex(d.report_month,freq="M").year*12+pd.PeriodIndex(d.report_month,freq="M").month
    d["original_end"]=pd.to_datetime(d.original_doc,format="%m/%Y").dt.to_period("M"); d["revised_end"]=pd.to_datetime(d.revised_doc,format="%m/%Y",errors="coerce").dt.to_period("M")
    d["effective_end"]=d.revised_end.fillna(d.original_end); d["effective_cost"]=d.revised_cost.fillna(d.original_cost); d["traceable"]=True
    return d


class PointInTimeFeatureTest(unittest.TestCase):
    def setUp(self): self.data=synthetic(); self.features=build_features(self.data).set_index("prediction_month")

    def test_no_feature_uses_future_months(self):
        before=build_features(self.data.iloc[:3]).set_index("prediction_month").loc["2026-03"]; after=self.features.loc["2026-03"]
        pd.testing.assert_series_equal(before,after,check_names=False)

    def test_rolling_features_are_backward_looking(self):
        self.assertEqual(self.features.loc["2026-03","progress_acceleration_3m"],5)

    def test_shifted_features_align_exactly(self):
        self.assertEqual(self.features.loc["2026-04","progress_change_3m"],30)

    def test_future_revised_cost_not_backfilled(self):
        self.assertTrue(pd.isna(self.features.loc["2026-03","revised_cost_current"]))

    def test_future_revised_completion_not_backfilled(self):
        self.assertEqual(self.features.loc["2026-03","effective_target_months_from_cutoff"],3)

    def test_full_history_aggregates_are_rejected(self):
        inv=feature_inventory().set_index("feature_name"); self.assertEqual(inv.loc["full_history_revision_count","status"],"UNSAFE"); self.assertNotIn("full_history_revision_count",self.features)

    def test_exact_target_horizons(self):
        _,schedule,cost=modeling_tables(self.data); self.assertEqual(len(schedule),4); self.assertEqual(len(cost),1)

    def test_unknown_is_not_negative(self):
        _,schedule,_=modeling_tables(self.data); self.assertNotIn("2026-05",set(schedule.prediction_month))

    def test_project_month_uniqueness(self):
        f=build_features(self.data); self.assertFalse(f.duplicated(["identity_key","prediction_month"]).any())

    def test_generation_is_deterministic(self):
        a=build_features(self.data).to_csv(index=False).encode(); b=build_features(self.data).to_csv(index=False).encode(); self.assertEqual(hashlib.sha256(a).digest(),hashlib.sha256(b).digest())


if __name__ == "__main__": unittest.main()
