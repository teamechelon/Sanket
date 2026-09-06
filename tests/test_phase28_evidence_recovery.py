"""Phase 28 evidence-recovery tests; no ML library or fitting is required."""

import inspect
import json
import unittest
from pathlib import Path

from src.phase28_evidence_recovery import (
    EXPECTED_ARTIFACTS,
    EXPECTED_CONTRACT_SHA256,
    _assert_identical,
    audit_once,
    canonical_bytes,
    logical_contract_verification,
)


ROOT = Path(".")


class Phase28EvidenceRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_once(ROOT)

    def test_expected_inventory_is_deterministic(self):
        first = audit_once(ROOT)
        second = audit_once(ROOT)
        _assert_identical(first, second)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(len(first["artifact_inventory"]), len(EXPECTED_ARTIFACTS))

    def test_missing_phase23_to_26_artifacts_are_missing(self):
        for record in self.audit["artifact_inventory"]:
            if record["phase"] in {23, 24, 25, 26}:
                self.assertEqual(record["evidence_type"], "missing")
                self.assertFalse(record["found_in_worktree"])
                self.assertFalse(record["found_in_git_history"])

    def test_recovered_artifacts_require_a_source_commit(self):
        for record in self.audit["artifact_inventory"]:
            if record["recovered"]:
                self.assertTrue(record["found_in_git_history"])
                self.assertTrue(record["source_commit"])

    def test_untracked_material_cannot_be_mislabeled_historical(self):
        for record in self.audit["artifact_inventory"]:
            if (record["found_in_worktree"] and not record["found_in_git_history"]
                    and not record["found_in_unreachable_git_objects"]):
                self.assertEqual(record["evidence_type"], "newly_generated_audit_artifact")
                self.assertNotEqual(record["evidence_type"], "historical_repository_artifact")

    def test_dangling_git_objects_are_never_mistaken_for_commit_history(self):
        object_records = [
            record for record in self.audit["artifact_inventory"]
            if record["found_in_unreachable_git_objects"]
        ]
        self.assertTrue(object_records)
        for record in object_records:
            self.assertFalse(record["found_in_git_history"])
            self.assertIsNone(record["source_commit"])
            self.assertTrue(record["source_git_objects"])
            self.assertIn("without_commit_provenance", record["verification_status"])

    def test_model_and_contract_verification_are_deterministic(self):
        first = self.audit
        contract = logical_contract_verification(ROOT)
        self.assertEqual(contract["expected_sha256"], EXPECTED_CONTRACT_SHA256)
        self.assertTrue(contract["matches_expected"])
        self.assertEqual(contract["frozen_artifact_verification"], "BLOCKED_NO_SERIALIZED_MODEL_ARTIFACT")
        self.assertEqual(first["model_artifact_investigation"]["serialized_model_files_in_worktree"], [])
        self.assertEqual(first["model_artifact_investigation"]["serialized_model_paths_in_reachable_git_history"], [])
        self.assertEqual(first["model_artifact_investigation"]["serialized_model_paths_in_unreachable_git_objects"], [])

    def test_phase28_cannot_evaluate_august_or_fit_ml(self):
        actions = self.audit["phase28_actions"]
        self.assertFalse(actions["august_holdout_evaluated"])
        self.assertFalse(actions["ml_fitting_performed"])
        source = inspect.getsource(__import__("src.phase28_evidence_recovery", fromlist=["*"]))
        self.assertNotIn(".fit(", source)
        self.assertNotIn("predict_proba", source)

    def test_canonical_inventory_is_machine_readable_json(self):
        parsed = json.loads(canonical_bytes(self.audit))
        self.assertEqual(parsed["phase"], 28)
        self.assertEqual(parsed["final_candidate_status"], "NOT APPROVED")


if __name__ == "__main__":
    unittest.main()
