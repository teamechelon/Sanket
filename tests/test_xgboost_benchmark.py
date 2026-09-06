"""Safeguards for the single predeclared Phase 16 XGBoost benchmark."""

import unittest

import numpy as np
import pandas as pd

from src.available_data_audit import maturity_windows
from src.xgboost_benchmark import (
    FEATURES,
    NEGLIGIBLE_DELTA,
    XGB_CONFIG,
    compare_temporal,
    decision,
    make_xgb_pipeline,
)


class XGBoostBenchmarkTest(unittest.TestCase):
    def test_configuration_is_exact_and_untuned(self):
        self.assertEqual(XGB_CONFIG["objective"], "binary:logistic")
        self.assertEqual(XGB_CONFIG["eval_metric"], "aucpr")
        self.assertEqual(XGB_CONFIG["n_estimators"], 300)
        self.assertEqual(XGB_CONFIG["random_state"], 26103)
        self.assertEqual(XGB_CONFIG["n_jobs"], 1)
        self.assertNotIn("early_stopping_rounds", XGB_CONFIG)

    def test_uses_exact_phase15_folds(self):
        rows = []
        for month in pd.period_range("2025-07", "2026-04", freq="M"):
            rows.append({
                "prediction_month": str(month),
                "identity_key": str(month),
                "future_schedule_later_3m": int(month.month % 2),
            })
        windows = maturity_windows(pd.DataFrame(rows))
        self.assertEqual(
            windows.evaluation_period.tolist(),
            ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"],
        )

    def test_primary_direction_uses_predeclared_pr_tolerance(self):
        baseline = pd.DataFrame({
            "fold": [1], "evaluation_period": ["2025-11"],
            "roc_auc": [.70], "pr_auc": [.40],
            "precision_at_40": [.3], "recall_at_40": [.2], "f1_at_40": [.24],
            "precision_at_50": [.4], "recall_at_50": [.1], "f1_at_50": [.16],
        })
        xgb = baseline.copy()
        xgb["pr_auc"] += NEGLIGIBLE_DELTA / 2
        paired = compare_temporal(baseline, xgb)
        self.assertEqual(paired.loc[0, "pr_direction"], "NEGLIGIBLE")

    def test_clear_gate_changes_final_model_only_when_satisfied(self):
        paired = pd.DataFrame({
            "delta_pr_auc": [.02] * 5 + [0.0],
            "delta_roc_auc": [.01] * 6,
            "pr_direction": ["XGBOOST"] * 5 + ["NEGLIGIBLE"],
        })
        project = pd.DataFrame({
            "model": ["BASELINE_RANDOM_FOREST", "XGBOOST"],
            "roc_auc": [.80, .81], "pr_auc": [.75, .76],
        })
        self.assertEqual(decision(paired, project), ("XGBOOST CLEARLY BETTER", "XGBOOST"))

    def test_pipeline_is_deterministic_on_synthetic_data(self):
        categorical = {"sector", "state", "ministry", "agency"}
        x = pd.DataFrame({
            feature: (["A", "B"] * 20 if feature in categorical
                      else np.arange(40, dtype=float))
            for feature in FEATURES
        })
        y = pd.Series([0, 1] * 20)
        first = make_xgb_pipeline().fit(x, y).predict_proba(x)[:, 1]
        second = make_xgb_pipeline().fit(x, y).predict_proba(x)[:, 1]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
