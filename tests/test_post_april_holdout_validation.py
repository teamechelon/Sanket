"""Phase 21 post-April maturity-gate regression tests."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.post_april_holdout_validation import (
    DEVELOPMENT_END,
    InvalidHoldoutError,
    _assert_identical,
    audit_once,
    maturity_availability,
    model_contract,
    post_april_rows,
    require_valid_holdout,
    required_endpoint,
    run,
    validate_temporal_separation,
)
from src.label_feasibility import load_data


DATA_PATH = Path("data/processed/project_monthly.csv")


class PostAprilHoldoutValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(DATA_PATH)

    def test_correct_post_april_date_filtering(self):
        filtered = post_april_rows(self.data)
        self.assertEqual(sorted(filtered.report_month.unique().tolist()), [
            "2026-05", "2026-06", "2026-07",
        ])
        self.assertTrue((filtered.report_month > DEVELOPMENT_END).all())

    def test_exact_maturity_endpoints(self):
        self.assertEqual(required_endpoint("2026-05"), "2026-08")
        self.assertEqual(required_endpoint("2026-06"), "2026-09")
        self.assertEqual(required_endpoint("2026-07"), "2026-10")

    def test_maturity_filter_excludes_missing_endpoints(self):
        availability = maturity_availability(self.data)
        self.assertEqual(int(availability.mature_label_rows.sum()), 0)
        self.assertFalse(availability.endpoint_month_available.any())
        self.assertTrue(availability.maturity_status.str.startswith("IMMATURE").all())

    def test_temporal_separation_and_observation_nonoverlap(self):
        development = pd.DataFrame({
            "identity_key": ["P:1"], "report_month": ["2026-04"],
        })
        holdout = pd.DataFrame({
            "identity_key": ["P:1"], "report_month": ["2026-05"],
        })
        validate_temporal_separation(development, holdout)
        overlapping = holdout.assign(report_month="2026-04")
        with self.assertRaises(ValueError):
            validate_temporal_separation(development, overlapping)

    def test_stable_local_post_april_counts(self):
        availability = maturity_availability(self.data).set_index("cutoff_month")
        self.assertEqual(availability.cutoff_observations.to_dict(), {
            "2026-05": 1987, "2026-06": 1847, "2026-07": 1775,
        })
        self.assertEqual(int(availability.cutoff_observations.sum()), 5609)

    def test_invalid_holdout_prevents_threshold_or_metric_evaluation(self):
        with self.assertRaisesRegex(InvalidHoldoutError, "exact t\+3"):
            require_valid_holdout(maturity_availability(self.data))

    def test_model_contract_is_frozen_and_deterministic(self):
        first = model_contract()
        second = model_contract()
        self.assertEqual(first, second)
        self.assertEqual(first["xgboost_version"], "2.1.4")
        self.assertEqual(first["feature_count"], 29)
        self.assertEqual(first["thresholds"], [0.40, 0.50])
        self.assertEqual(first["calibration"], "NONE")

    def test_deterministic_maturity_audit(self):
        first = audit_once(self.data)
        second = audit_once(self.data)
        _assert_identical(first, second)
        self.assertEqual(first["status"], "INVALID HOLDOUT")
        self.assertEqual(first["mature_observations"], 0)

    def test_invalid_artifact_generation_is_fail_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(DATA_PATH, output)
            names = sorted(path.name for path in output.iterdir())
            self.assertEqual(names, [
                "phase21_post_april_forward_holdout_report.txt",
                "phase21_post_april_holdout_availability.csv",
            ])
            report = (output / names[0]).read_text()
            self.assertIn("PHASE 21 STATUS: INVALID HOLDOUT", report)
            self.assertNotIn("ROC-AUC=", report)


if __name__ == "__main__":
    unittest.main()
