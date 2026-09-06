"""Tests for the Phase 27 final model decision and freeze audit."""

import json
import unittest
from pathlib import Path

from src.final_model_freeze_audit import (
    EXPECTED_CONTRACT_SHA256,
    FINAL_CANDIDATE_NOT_APPROVED,
    REQUIRED_EVIDENCE,
    _assert_identical,
    build_audit,
    manifest_bytes,
)
from src.final_model_freeze_audit import FEATURES


ROOT = Path(".")


class FinalModelFreezeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = build_audit(ROOT)

    def test_current_missing_phase_evidence_blocks_approval(self):
        self.assertEqual(self.audit["final_candidate_status"], FINAL_CANDIDATE_NOT_APPROVED)
        self.assertFalse(self.audit["final_freeze_created"])
        self.assertTrue(self.audit["decision_blockers"])
        for phase in (
            "phase23_readiness_monitoring",
            "phase24_product_integration",
            "phase25_generalization_error_analysis",
            "phase26_calibration_protocol",
        ):
            self.assertFalse(self.audit["required_evidence"][phase]["exists"])

    def test_logical_model_hash_and_contract_integrity(self):
        contract = self.audit["frozen_model_contract"]
        self.assertEqual(contract["logical_model_contract_sha256"], EXPECTED_CONTRACT_SHA256)
        self.assertEqual(contract["model_artifact"]["status"], "not_serialized")
        self.assertIsNone(contract["model_artifact"]["sha256"])

    def test_feature_ordering_and_threshold_integrity(self):
        contract = self.audit["frozen_model_contract"]
        self.assertEqual(contract["feature_list_ordered"], list(FEATURES))
        self.assertEqual(contract["thresholds"], [0.40, 0.50])

    def test_calibration_and_score_semantics_are_non_probabilistic(self):
        contract = self.audit["frozen_model_contract"]
        self.assertEqual(contract["calibration_status"], "not_calibrated")
        self.assertEqual(contract["score_semantics"], "relative_risk_ranking")

    def test_august_readiness_gate_is_not_ready(self):
        readiness = self.audit["august_readiness"]
        self.assertFalse(readiness["august_data_available"])
        self.assertEqual(readiness["mature_may_observations"], 0)
        self.assertEqual(readiness["mature_may_projects"], 0)
        self.assertFalse(readiness["may_to_august_holdout_evaluated"])

    def test_manifest_generation_is_deterministic(self):
        first = build_audit(ROOT)
        second = build_audit(ROOT)
        _assert_identical(first, second)
        self.assertEqual(manifest_bytes(first), manifest_bytes(second))

    def test_manifest_contains_all_required_evidence_records(self):
        manifest = json.loads(manifest_bytes(self.audit))
        self.assertEqual(manifest["final_candidate_status"], FINAL_CANDIDATE_NOT_APPROVED)
        self.assertEqual(set(manifest["required_evidence"]), set(REQUIRED_EVIDENCE))


if __name__ == "__main__":
    unittest.main()
