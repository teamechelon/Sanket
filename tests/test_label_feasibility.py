"""Regression tests for temporal label construction rules."""

from __future__ import annotations

import unittest

import pandas as pd

from src.label_feasibility import CANDIDATES, _candidate_labels


def _candidate(name: str):
    return next(candidate for candidate in CANDIDATES if candidate.target_name == name)


def _rows() -> pd.DataFrame:
    values = []
    for month, revised_end, revised_cost in (
        (1, "2026-04", 100.0),
        (2, "2026-04", 100.0),
        (3, "2026-06", 100.0),
        (4, "2026-06", 106.0),
        (7, "2026-06", 106.0),
    ):
        values.append({
            "identity_key": "P:1",
            "report_month": f"2026-{month:02d}",
            "month_number": 2026 * 12 + month,
            "traceable": True,
            "effective_end": pd.Period(revised_end, freq="M"),
            "effective_cost": revised_cost,
            "original_cost": 100.0,
        })
    return pd.DataFrame(values)


class TemporalLabelRulesTest(unittest.TestCase):
    def test_schedule_label_uses_only_future_window(self) -> None:
        labels, _ = _candidate_labels(_rows(), _candidate("future_schedule_later_3m"))
        january = labels.set_index("prediction_month").loc["2026-01"]
        self.assertEqual(january["label"], 1)

    def test_negative_requires_exact_horizon_observation(self) -> None:
        labels, unknown = _candidate_labels(_rows(), _candidate("future_cost_increase_3m"))
        self.assertNotIn("2026-02", set(labels["prediction_month"]))
        self.assertGreater(unknown, 0)

    def test_material_cost_threshold(self) -> None:
        labels, _ = _candidate_labels(_rows(), _candidate("future_cost_increase_5pct_6m"))
        january = labels.set_index("prediction_month").loc["2026-01"]
        self.assertEqual(january["label"], 1)


if __name__ == "__main__":
    unittest.main()
