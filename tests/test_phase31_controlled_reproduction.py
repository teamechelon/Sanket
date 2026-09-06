"""Tests for the Phase 31 pre-execution serialization gate."""

import inspect
import json
import unittest
from pathlib import Path

from src.phase31_controlled_reproduction import (
    DATASET_PATH,
    SERIALIZATION_TOKENS,
    canonical_json,
    feature_contract,
    pretraining_manifest,
    serialization_evidence,
    sha256,
)


ROOT = Path(".")
PHASE30_PYTHON = Path(r"C:\Users\ronny\AppData\Local\Temp\sih26103-phase30-py313-recovery\Scripts\python.exe")


class Phase31ControlledReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = pretraining_manifest(ROOT, PHASE30_PYTHON)

    def test_training_configuration_and_dataset_hash_integrity(self):
        self.assertEqual(self.audit["dataset"]["sha256"], sha256(ROOT / DATASET_PATH))
        self.assertEqual(self.audit["source_hashes"]["requirements.txt"], sha256(ROOT / "requirements.txt"))
        self.assertEqual(self.audit["xgboost_configuration"]["random_seed"], 26103)

    def test_feature_contract_is_ordered_and_hashed(self):
        contract = feature_contract(ROOT)
        self.assertEqual(contract["feature_count"], 29)
        self.assertEqual(contract["ordered_features"], self.audit["feature_contract"]["ordered_features"])
        self.assertEqual(len(contract["feature_contract_sha256"]), 64)

    def test_missing_artifact_gate_is_explicit(self):
        evidence = serialization_evidence(ROOT)
        self.assertEqual(evidence["serialization_sources"], [])
        self.assertEqual(evidence["loading_sources"], [])
        self.assertEqual(self.audit["execution_gate"]["result"], "BLOCKED")
        self.assertFalse(self.audit["execution_gate"]["existing_entry_point_has_serializer"])
        self.assertFalse(self.audit["execution_gate"]["existing_prediction_path_has_loader"])

    def test_manifest_is_machine_readable_and_deterministic_for_a_fixed_audit(self):
        self.assertEqual(canonical_json(self.audit), canonical_json(self.audit))
        self.assertEqual(json.loads(canonical_json(self.audit))["phase"], 31)

    def test_no_training_calibration_or_august_evaluation_is_performed(self):
        import src.phase31_controlled_reproduction as phase31

        source = inspect.getsource(phase31)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("calibrat", source.lower().replace("calibration", ""))
        self.assertNotIn("august_2026", source)
        self.assertTrue(SERIALIZATION_TOKENS)

    def test_historical_and_new_evidence_are_separated(self):
        report = (ROOT / "reports/phase31_controlled_reproduction_report.txt").read_text(encoding="utf-8")
        self.assertIn("HISTORICAL DOCUMENTED RESULT", report)
        self.assertIn("PHASE 31 NEW REPRODUCED RESULT: none", report)
        self.assertIn("not historical artifact recovery", report)


if __name__ == "__main__":
    unittest.main()
