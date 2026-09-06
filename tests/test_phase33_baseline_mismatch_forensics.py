"""Read-only checks for Phase 33's baseline mismatch forensic record."""

import json
import unittest
from pathlib import Path

from src.phase33_baseline_mismatch_forensics import (
    DECISION, REPRODUCTION_STATUS, ROOT_CAUSE, audit, manifest_bytes,
    report_text,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase33BaselineMismatchForensicsTest(unittest.TestCase):
    def setUp(self):
        self.result = audit(ROOT)

    def test_forensic_result_is_deterministic(self):
        self.assertEqual(manifest_bytes(self.result), manifest_bytes(audit(ROOT)))

    def test_historical_reference_is_not_the_phase15_assertion_target(self):
        self.assertEqual(self.result["historical_metric_evidence"]["reference"], {"roc_auc": 0.855136, "pr_auc": 0.804918})
        self.assertEqual(len(self.result["phase15_assertion_evidence"]["stored_fold_metrics"]), 6)
        self.assertIn("not the 0.855136/0.804918", self.result["phase15_assertion_evidence"]["assertion_target"])

    def test_contract_and_final_status_are_preserved(self):
        state = self.result["current_state"]
        self.assertEqual(state["target"], "future_schedule_later_3m")
        self.assertEqual(state["feature_contract"]["count"], 29)
        self.assertEqual(state["feature_contract"]["sha256"], state["feature_contract"]["expected_phase32_sha256"])
        self.assertEqual(self.result["classification"]["root_cause"], ROOT_CAUSE)
        self.assertEqual(self.result["classification"]["decision"], DECISION)
        self.assertEqual(self.result["classification"]["reproduction_status"], REPRODUCTION_STATUS)

    def test_phase32_cannot_precede_the_failed_assertion(self):
        phase32 = self.result["phase32_modification_audit"]
        self.assertFalse(phase32["caused_phase15_mismatch"])
        self.assertIn("before the artifact-path branch", phase32["ordering_evidence"])
        self.assertEqual(phase32["fit_final_temporal_fold"].split(" — ")[0], "METHODOLOGY CHANGE")

    def test_outputs_are_valid_and_repeatable(self):
        report = ROOT / "reports" / "phase33_baseline_mismatch_forensics_report.txt"
        manifest = ROOT / "reports" / "phase33_baseline_mismatch_forensics_manifest.json"
        self.assertEqual(report.read_text(encoding="utf-8"), report_text(self.result))
        self.assertEqual(manifest.read_bytes(), manifest_bytes(self.result))
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["phase"], 33)
        self.assertIn("ROOT CAUSE PARTIALLY IDENTIFIED", report_text(self.result))


if __name__ == "__main__":
    unittest.main()
