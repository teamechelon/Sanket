"""Phase 9 regression tests for the approved proxy boundary."""

from pathlib import Path
import unittest

import pandas as pd

from src.label_feasibility import load_data
from src.target_validation import _first_later_change_month, march_later_events, schedule_candidates


DATA = Path("data/processed/project_monthly.csv")


class TargetValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_data(DATA)

    def test_march_spike_is_exactly_reproduced(self) -> None:
        events = march_later_events(self.df)
        self.assertEqual(len(events), 447)
        self.assertEqual(events["prediction_month"].value_counts().to_dict(), {"2026-02": 441, "2026-01": 6})

    def test_primary_candidate_counts_and_endpoint_accounting(self) -> None:
        row = schedule_candidates(self.df).set_index("target_name").loc["future_schedule_later_3m"]
        self.assertEqual((row.eligible_rows, row.positive_rows, row.negative_rows), (11111, 4092, 7019))
        self.assertEqual(row.eligible_rows + row.unknown_rows, len(self.df))

    def test_future_values_are_not_returned_as_features(self) -> None:
        labels = _first_later_change_month(self.df, 3)
        self.assertEqual(set(labels.columns), {"identity_key", "prediction_month", "label", "first_change_month"})
        forbidden = {"future_target", "future_progress", "future_expenditure", "future_revised_cost"}
        self.assertTrue(forbidden.isdisjoint(labels.columns))


if __name__ == "__main__":
    unittest.main()
