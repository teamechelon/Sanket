"""Phase 20 raw-calibration metric and safeguard tests."""

import unittest

import numpy as np
import pandas as pd

from src.available_data_audit import EVALUATION_MONTHS
from src.raw_calibration_audit import (
    BIN_COUNT,
    _assert_identical,
    _assert_project_disjoint,
    audit_contract,
    brier_score,
    brier_skill,
    calibration_intercept_slope,
    calibration_metrics,
    expected_calibration_error,
    filter_label_eligible_predictions,
    fold_calibration,
    fold_metric_summary,
    prevalence_brier,
    reliability_bins,
    score_distribution,
    slice_calibration,
)
from src.schedule_robustness import FEATURES, TARGET
from src.xgboost_benchmark import XGB_CONFIG


def prediction_fixture() -> pd.DataFrame:
    rows = []
    for fold, month in enumerate(EVALUATION_MONTHS, start=1):
        for index, (actual, score) in enumerate(
            [(0, 0.1), (1, 0.2), (0, 0.7), (1, 0.8)]
        ):
            rows.append({
                "fold": fold,
                "prediction_month": month,
                "identity_key": f"P:{fold}:{index}",
                TARGET: actual,
                "probability": score,
                "sector": "Education",
                "ministry": "Ministry of Power",
                "progress_current": 50,
                "project_age_months": 60,
                "expenditure_to_original_cost": 0.5,
            })
    return pd.DataFrame(rows)


class RawCalibrationAuditTest(unittest.TestCase):
    def test_brier_score(self):
        self.assertAlmostEqual(brier_score([0, 1], [0.2, 0.8]), 0.04)

    def test_prevalence_brier(self):
        self.assertAlmostEqual(prevalence_brier([0, 0, 1, 1]), 0.25)

    def test_brier_skill(self):
        self.assertAlmostEqual(brier_skill([0, 1], [0.2, 0.8]), 0.84)

    def test_ece(self):
        self.assertAlmostEqual(
            expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            0.15,
        )

    def test_calibration_intercept_and_slope(self):
        scores = np.repeat([0.2, 0.8], 100)
        actual = np.concatenate([
            np.array([1] * 20 + [0] * 80),
            np.array([1] * 80 + [0] * 20),
        ])
        intercept, slope, status = calibration_intercept_slope(actual, scores)
        self.assertEqual(status, "DEFINED")
        self.assertAlmostEqual(intercept, 0.0, places=8)
        self.assertAlmostEqual(slope, 1.0, places=8)

    def test_deterministic_binning_and_empty_bins(self):
        first = reliability_bins([0, 1], [0.05, 0.95])
        second = reliability_bins([0, 1], [0.05, 0.95])
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        self.assertEqual(len(first), BIN_COUNT)
        self.assertEqual(int(first.observations.sum()), 2)
        self.assertEqual(int(first.bin_status.eq("EMPTY").sum()), 8)

    def test_fold_calculation(self):
        result = fold_calibration(prediction_fixture())
        self.assertEqual(result.fold.tolist(), list(range(1, 7)))
        self.assertTrue((result.observations == 4).all())
        self.assertTrue((result.event_rate == 0.5).all())

    def test_fold_aggregation_uses_population_standard_deviation(self):
        folds = fold_calibration(prediction_fixture())
        summary = fold_metric_summary(folds).set_index("metric")
        self.assertAlmostEqual(summary.loc["event_rate", "mean"], 0.5)
        self.assertAlmostEqual(summary.loc["event_rate", "standard_deviation"], 0.0)

    def test_unknown_observations_are_excluded_not_recast(self):
        fixture = prediction_fixture()
        future = fixture.iloc[[0]].assign(prediction_month="2026-05", **{TARGET: np.nan})
        filtered = filter_label_eligible_predictions(pd.concat([fixture, future]))
        self.assertEqual(len(filtered), len(fixture))
        self.assertNotIn("2026-05", set(filtered.prediction_month))

    def test_project_disjoint_contract_and_overlap_handling(self):
        contract = audit_contract()
        self.assertEqual(contract["evaluation_months"], tuple(EVALUATION_MONTHS))
        self.assertEqual(contract["configuration"], XGB_CONFIG)
        self.assertEqual(contract["feature_count"], 29)
        split = {
            "train": pd.DataFrame({"identity_key": ["P:1"]}),
            "validation": pd.DataFrame({"identity_key": ["P:2"]}),
            "test": pd.DataFrame({"identity_key": ["P:3"]}),
        }
        _assert_project_disjoint(split)
        split["test"] = pd.DataFrame({"identity_key": ["P:1"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            _assert_project_disjoint(split)

    def test_slice_support_rule_is_reused(self):
        base = prediction_fixture().iloc[:1]
        rows = []
        for index in range(50):
            row = base.iloc[0].to_dict()
            row.update({
                "identity_key": f"P:{index}",
                "prediction_month": EVALUATION_MONTHS[index % 6],
                "fold": index % 6 + 1,
                TARGET: 1 if index < 20 else 0,
                "probability": 0.7 if index < 20 else 0.2,
                "sector": "Education",
            })
            rows.append(row)
        result = slice_calibration(pd.DataFrame(rows)).set_index("important_slice")
        self.assertEqual(result.loc["Education", "sample_status"], "ADEQUATE")
        self.assertEqual(result.loc["Railways", "sample_status"], "INSUFFICIENT_SAMPLE")
        self.assertTrue(pd.isna(result.loc["Railways", "brier_score"]))

    def test_deterministic_output_comparison(self):
        overall = calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        frame = pd.DataFrame([score_distribution([0, 1], [0.2, 0.8])])
        first = {
            "overall": overall,
            "bins": frame,
            "folds": frame,
            "fold_summary": frame,
            "project_disjoint": frame,
            "slices": frame,
        }
        second = {
            key: value.copy() if isinstance(value, pd.DataFrame) else value.copy()
            for key, value in first.items()
        }
        _assert_identical(first, second)

    def test_audit_contract_prohibits_fitting_tuning_and_modification(self):
        contract = audit_contract()
        raw_scores = np.array([0.2, 0.8])
        untouched = raw_scores.copy()
        calibration_metrics([0, 1], raw_scores)
        np.testing.assert_array_equal(raw_scores, untouched)
        self.assertFalse(contract["calibration_fitted"])
        self.assertFalse(contract["scores_transformed"])
        self.assertFalse(contract["thresholds_selected"])
        self.assertFalse(contract["model_tuned"])
        self.assertFalse(contract["features_modified"])
        self.assertEqual(contract["features"], tuple(FEATURES))


if __name__ == "__main__":
    unittest.main()
