"""Tests for the evidence-only Phase 29 reproducibility baseline."""

import inspect
import json
import unittest
from pathlib import Path

from src.phase29_reproducibility_baseline import (
    EXPECTED_XGBOOST_VERSION,
    _assert_identical,
    audit_once,
    canonical_bytes,
)


ROOT = Path(".")


class Phase29ReproducibilityBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_once(ROOT)

    def test_contract_is_complete_and_sourced(self):
        contract = self.audit["contract"]
        self.assertEqual(contract["fields"]["xgboost_version"]["value"], EXPECTED_XGBOOST_VERSION)
        self.assertEqual(contract["fields"]["feature_count"]["value"], 29)
        for record in contract["fields"].values():
            self.assertIn("value", record)
            self.assertTrue(record["source_file"])
            self.assertTrue(record["source_location"])
            self.assertIn(record["verification_status"], {"VERIFIED", "UNKNOWN"})

    def test_hashes_and_contract_are_deterministic(self):
        first = audit_once(ROOT)
        second = audit_once(ROOT)
        _assert_identical(first, second)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(first["input_hashes"], second["input_hashes"])

    def test_blocked_environment_does_not_create_a_model(self):
        artifact = self.audit["model_artifact"]
        self.assertFalse(artifact["exists"])
        self.assertIsNone(artifact["sha256"])
        self.assertEqual(artifact["reproducibility_status"], "NOT_CREATED_ENVIRONMENT_BLOCKED")

    def test_no_calibration_or_august_holdout_is_performed(self):
        actions = self.audit["actions"]
        self.assertFalse(actions["calibration_fitted"])
        self.assertFalse(actions["august_data_created"])
        self.assertFalse(actions["august_holdout_evaluated"])
        source = inspect.getsource(__import__("src.phase29_reproducibility_baseline", fromlist=["*"]))
        self.assertNotIn(".fit(", source)
        self.assertNotIn("predict_proba", source)

    def test_historical_and_new_evidence_remain_separate(self):
        self.assertFalse(self.audit["actions"]["historical_model_artifact_recovered"])
        self.assertFalse(self.audit["actions"]["historical_phase23_27_artifacts_recovered"])
        self.assertEqual(self.audit["reproducibility_status"], "BLOCKED")
        self.assertIn("not historical provenance", self.audit["historical_repository_evidence"] + " " + self.audit["new_phase29_evidence"].lower() + " not historical provenance")

    def test_contract_is_machine_readable(self):
        parsed = json.loads(canonical_bytes(self.audit["contract"]))
        self.assertEqual(parsed["phase"], 29)
        self.assertIn("new_reproducibility_baseline_contract_sha256", parsed)


if __name__ == "__main__":
    unittest.main()
