"""Non-training verification of the committed Phase 34 diagnostic artifacts."""

import json
import unittest
from pathlib import Path

from src.phase34_controlled_reproduction import report_text


ROOT = Path(__file__).resolve().parents[1]


class Phase34ControlledReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "reports/phase34_controlled_reproduction_manifest.json").read_text(encoding="utf-8"))
        cls.report = (ROOT / "reports/phase34_controlled_reproduction_report.txt").read_text(encoding="utf-8")

    def test_historical_reference_and_counts_are_recorded(self):
        self.assertEqual(len(self.manifest["historical_reference"]), 6)
        self.assertEqual(self.manifest["historical_reference"][0], {"fold": 1, "pr_auc": 0.309756, "roc_auc": 0.778113})
        self.assertTrue(self.manifest["integrity"]["expected_counts_match"])

    def test_random_forest_contract_is_frozen(self):
        contract = self.manifest["estimator_contract"]
        self.assertEqual(contract["class"], "RandomForestClassifier")
        self.assertEqual(contract["n_estimators"], 200)
        self.assertEqual(contract["min_samples_leaf"], 5)
        self.assertEqual(contract["max_features"], "sqrt")
        self.assertEqual(contract["random_state"], 26103)
        self.assertEqual(contract["n_jobs"], -1)

    def test_two_executions_and_phase32_ordering_are_recorded(self):
        self.assertEqual(self.manifest["repeated_execution"]["classification"], "NUMERICALLY IDENTICAL / BYTE-DIFFERENT")
        self.assertTrue(self.manifest["repeated_execution"]["numerically_identical_at_1e12"])
        self.assertEqual(len(self.manifest["repeated_execution"]["fold_comparison"]), 6)
        self.assertTrue(self.manifest["integrity"]["phase15_assertion_unchanged"])
        self.assertIn("before the artifact-path branch", self.manifest["integrity"]["phase32_impact"])

    def test_report_has_the_required_decision(self):
        self.assertIn(self.manifest["decision"]["reproduction_status"], self.report)
        self.assertEqual(self.manifest["decision"]["root_cause_code"], "J")
        self.assertIn("ROOT CAUSE:", self.report)
        self.assertIn("NEXT ACTION:", self.report)
        self.assertIn("FOLD COMPARISON", self.report)

    def test_report_renderer_is_non_training(self):
        self.assertIn("FINAL DECISION", report_text(self.manifest))


if __name__ == "__main__":
    unittest.main()
