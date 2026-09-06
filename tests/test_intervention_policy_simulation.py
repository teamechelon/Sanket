"""Phase 18 frozen intervention-policy simulation safeguards."""

import unittest

import numpy as np
import pandas as pd

from src.intervention_policy_simulation import (
    MODEL_NAMES,
    assert_labeled_scope,
    cost_scenarios,
    fold_summary,
    intervention_metrics,
    operational_classification,
    validate_threshold,
)
from src.schedule_robustness import TARGET


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "identity_key": ["P:1", "P:2", "P:3", "P:4"],
        "prediction_month": ["2026-04"] * 4,
        TARGET: [0, 0, 1, 1],
        "probability": [.2, .6, .4, .8],
    })


class InterventionPolicySimulationTest(unittest.TestCase):
    def test_only_frozen_thresholds_are_allowed(self):
        validate_threshold(.4)
        validate_threshold(.5)
        with self.assertRaises(ValueError):
            validate_threshold(.45)

    def test_confusion_precision_recall_and_f1(self):
        result = intervention_metrics(fixture(), .5)
        self.assertEqual(
            (result["true_negatives"], result["false_interventions"],
             result["error_observations_missed"], result["true_errors_captured"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(result["intervention_precision"], .5)
        self.assertEqual(result["intervention_recall"], .5)
        self.assertEqual(result["f1"], .5)

    def test_intervention_burden_and_workload_ratio(self):
        result = intervention_metrics(fixture(), .5)
        self.assertEqual(result["intervention_burden"], .5)
        self.assertEqual(result["review_workload_ratio"], 1.0)
        self.assertEqual(result["false_interventions_per_true_error_captured"], 1.0)

    def test_cost_sensitivity_formula(self):
        overall = pd.DataFrame([
            {"model": model, "threshold": threshold,
             "error_observations_missed": 2 if model == MODEL_NAMES[0] else 1,
             "false_interventions": 1 if model == MODEL_NAMES[0] else 2}
            for threshold in (.4, .5) for model in MODEL_NAMES
        ])
        result = cost_scenarios(overall)
        scenario = result[(result.scenario == "B_FN_DOUBLE") & result.threshold.eq(.4)].iloc[0]
        self.assertEqual(scenario.baseline_total_simulated_cost, 5)
        self.assertEqual(scenario.xgb_total_simulated_cost, 4)

    def test_fold_aggregation_uses_population_sd(self):
        folds = pd.DataFrame([
            {"model": MODEL_NAMES[0], "threshold": .4,
             "flagged_review_actions": n, "flagged_percent": n / 10,
             "true_errors_captured": n, "false_interventions": 0,
             "error_observations_missed": 2, "intervention_precision": 1.0,
             "intervention_recall": .5, "f1": 2 / 3,
             "intervention_burden": n / 10, "review_workload_ratio": n / 2}
            for n in (2, 4)
        ])
        result = fold_summary(folds).iloc[0]
        self.assertEqual(result.mean_flagged_review_actions, 3)
        self.assertEqual(result.sd_flagged_review_actions, 1)

    def test_unknown_and_future_rows_are_rejected(self):
        bad = fixture()
        bad.loc[0, "prediction_month"] = "2026-05"
        with self.assertRaises(ValueError):
            assert_labeled_scope(bad)
        bad = fixture()
        bad.loc[0, TARGET] = np.nan
        with self.assertRaises(ValueError):
            assert_labeled_scope(bad)

    def test_pooled_project_flagging_is_not_invented(self):
        repeated = pd.concat([fixture(), fixture().assign(prediction_month="2026-03")])
        result = intervention_metrics(repeated, .5)
        self.assertTrue(np.isnan(result["unique_projects_flagged"]))
        self.assertEqual(result["project_flagging_scope"], "NOT_AGGREGATED_ACROSS_MONTHS")

    def test_operational_classification_does_not_select_threshold(self):
        comparison = pd.DataFrame({
            "delta_xgb_minus_baseline_true_errors_captured": [10, 20],
            "delta_xgb_minus_baseline_flagged_percent": [.12, .20],
            "delta_xgb_minus_baseline_false_interventions": [5, 10],
        })
        self.assertEqual(operational_classification(comparison), "OPERATIONALLY MIXED")


if __name__ == "__main__":
    unittest.main()
