"""Phase 22 August-source, ingestion, identity, and readiness tests."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.august_2026_ingestion_validation import (
    AUGUST_MONTH,
    EXPECTED_MODEL_CONTRACT_SHA256,
    PROJECT_SCHEMA,
    _assert_identical,
    audit_once,
    foundation_summary,
    may_august_matching,
    official_source_url,
    project_id_quality,
    run,
    unavailable_readiness,
    validate_august_data,
    validate_reporting_period,
    validate_schema,
    verify_frozen_contract,
    verify_no_future_feature_leakage,
)
from src.label_feasibility import load_data
from src.schedule_robustness import FEATURES


DATA_PATH = Path("data/processed/project_monthly.csv")
PDF_DIR = Path("data/raw/pdf")


def synthetic_history(duplicate_august: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for identity, may_end, later_end in (
        ("P:1", "2026-12", "2027-01"),
        ("P:2", "2026-12", "2026-12"),
    ):
        for month, number, endpoint in (
            ("2026-05", 2026 * 12 + 5, may_end),
            ("2026-06", 2026 * 12 + 6, may_end),
            ("2026-07", 2026 * 12 + 7, later_end),
        ):
            rows.append({
                "identity_key": identity, "report_month": month,
                "month_number": number, "effective_end": pd.Period(endpoint),
                "traceable": True,
            })
    existing = pd.DataFrame(rows)
    august = pd.DataFrame([
        {
            "identity_key": "P:1", "report_month": AUGUST_MONTH,
            "month_number": 2026 * 12 + 8, "effective_end": pd.Period("2027-01"),
            "traceable": True,
        },
        {
            "identity_key": "P:2", "report_month": AUGUST_MONTH,
            "month_number": 2026 * 12 + 8, "effective_end": pd.Period("2026-12"),
            "traceable": True,
        },
    ])
    if duplicate_august:
        august = pd.concat([august, august.iloc[[0]]], ignore_index=True)
    return existing, august


class August2026IngestionValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing = load_data(DATA_PATH)

    def test_august_source_must_be_official(self):
        self.assertTrue(official_source_url(
            "https://paimana-proj.mospi.gov.in/reports/august.pdf"
        ))
        self.assertTrue(official_source_url(
            "https://mospi.gov.in/sites/default/files/august.pdf"
        ))
        self.assertFalse(official_source_url("https://example.com/august.pdf"))
        self.assertFalse(official_source_url("http://mospi.gov.in/august.pdf"))

    def test_august_schema_validation(self):
        valid = pd.DataFrame(columns=PROJECT_SCHEMA)
        self.assertTrue(validate_schema(valid)["schema_valid"])
        result = validate_schema(valid.drop(columns=["identity_key"]))
        self.assertFalse(result["schema_valid"])
        self.assertIn("identity_key", result["missing_columns"])

    def test_reporting_period_validation(self):
        self.assertTrue(validate_reporting_period(pd.DataFrame({
            "report_month": [AUGUST_MONTH, AUGUST_MONTH],
        })))
        self.assertFalse(validate_reporting_period(pd.DataFrame({
            "report_month": ["2026-07"],
        })))

    def test_project_id_and_duplicate_detection(self):
        result = project_id_quality(pd.DataFrame({
            "identity_key": ["P:1", "P:1", "bad id", pd.NA],
        }))
        self.assertEqual(result["duplicate_project_rows"], 2)
        self.assertEqual(result["duplicate_project_ids"], 1)
        self.assertEqual(result["malformed_ids"], 1)
        self.assertEqual(result["missing_ids"], 1)

    def test_may_to_august_exact_project_matching(self):
        existing, august = synthetic_history()
        result = may_august_matching(existing, august)
        self.assertEqual(result["valid_august_endpoint_projects"], 2)
        self.assertEqual(result["mature_may_observations"], 2)
        self.assertEqual(result["mature_may_projects"], 2)
        self.assertEqual(result["mature_may_events"], 1)
        self.assertEqual(result["maturity_coverage"], 1.0)

    def test_missing_endpoint_safe_failure(self):
        result = unavailable_readiness(self.existing).iloc[0]
        self.assertFalse(result.may_holdout_ready)
        self.assertEqual(result.mature_may_observations, 0)
        self.assertEqual(result.valid_august_endpoint_projects, 0)

    def test_ambiguous_matches_are_excluded(self):
        existing, august = synthetic_history(duplicate_august=True)
        result = may_august_matching(existing, august)
        self.assertEqual(result["ambiguous_matches"], 1)
        self.assertEqual(result["valid_august_endpoint_projects"], 1)
        self.assertEqual(result["mature_may_projects"], 1)

    def test_data_quality_rejects_invalid_values(self):
        frame = pd.DataFrame([{column: pd.NA for column in PROJECT_SCHEMA}])
        frame.loc[0, "report_month"] = AUGUST_MONTH
        frame.loc[0, "identity_key"] = "P:1"
        frame.loc[0, "physical_progress"] = 101
        frame.loc[0, "original_cost"] = -1
        result = validate_august_data(frame)
        self.assertEqual(result["validation_status"], "INVALID")
        self.assertEqual(result["impossible_progress_values"], 1)
        self.assertEqual(result["invalid_numeric_values"], 1)

    def test_leakage_prevention_and_feature_order(self):
        verify_no_future_feature_leakage(list(FEATURES))
        changed = list(FEATURES)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(ValueError, "ordering"):
            verify_no_future_feature_leakage(changed)

    def test_frozen_model_contract_preserved(self):
        contract = verify_frozen_contract()
        self.assertEqual(contract["contract_sha256"], EXPECTED_MODEL_CONTRACT_SHA256)
        self.assertEqual(contract["feature_count"], 29)
        self.assertEqual(contract["calibration"], "NONE")

    def test_17_pdf_foundation_is_intact(self):
        first = foundation_summary(PDF_DIR)
        second = foundation_summary(PDF_DIR)
        self.assertEqual(first, second)
        self.assertTrue(first["intact"])
        self.assertEqual(first["pdf_count"], 17)

    def test_unavailable_run_is_reproducible_and_creates_required_artifacts(self):
        first = audit_once(self.existing, PDF_DIR)
        second = audit_once(self.existing, PDF_DIR)
        _assert_identical(first, second)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(DATA_PATH, PDF_DIR, output)
            self.assertEqual(sorted(path.name for path in output.iterdir()), [
                "phase22_august_data_quality.csv",
                "phase22_august_ingestion_report.txt",
                "phase22_may_holdout_readiness.csv",
            ])
            report = (output / "phase22_august_ingestion_report.txt").read_text()
            self.assertIn("PHASE 22 STATUS: NOT READY", report)
            self.assertIn("MAY HOLDOUT READY: FALSE", report)


if __name__ == "__main__":
    unittest.main()
