"""Tests for the non-training Phase 30 environment recovery audit."""

import inspect
import json
import unittest
from pathlib import Path

from src.phase30_environment_recovery import (
    canonical_json,
    dependency_compatibility,
    dependency_files,
    manifest,
    parse_requirements,
    satisfies_specifier,
)


ROOT = Path(".")


class Phase30EnvironmentRecoveryTest(unittest.TestCase):
    def test_dependency_declarations_are_parsed_correctly(self):
        records = parse_requirements(ROOT / "requirements.txt")
        self.assertEqual(
            [(record["normalized_name"], record["specifier"]) for record in records],
            [("pandas", ">=2.0"), ("numpy", ">=1.24"), ("pdfplumber", ">=0.10"),
             ("scikit-learn", ">=1.6,<2"), ("xgboost", "==2.1.4")],
        )
        files = dependency_files(ROOT)
        self.assertIn("requirements.txt", files["found"])
        self.assertIn("pyproject.toml", files["absent_root_level_declarations"])

    def test_python_and_package_compatibility_rules_are_checked(self):
        self.assertTrue(satisfies_specifier("1.9.0", ">=1.6,<2"))
        self.assertFalse(satisfies_specifier("2.0.0", ">=1.6,<2"))
        records = parse_requirements(ROOT / "requirements.txt")
        compatibility = dependency_compatibility(records, {
            "pandas": "3.0.5", "numpy": "2.5.3", "pdfplumber": "0.11.10",
            "scikit-learn": "1.9.0", "xgboost": "2.1.4",
        })
        self.assertTrue(all(record["satisfies_declared_constraint"] for record in compatibility))

    def test_package_versions_and_manifest_are_deterministic(self):
        audit = {
            "repository_commit": "abc", "environment_status": "READY",
            "python_compatibility": {"selected_python": {
                "os": "Windows", "architecture": "AMD64", "python_version": "3.13.5",
            }},
            "environment": {
                "pip_version": "pip 25.1.1", "packages": {"xgboost": "2.1.4", "numpy": "2.5.3"},
                "dependency_source": "requirements.txt only", "environment_creation_timestamp": "2026-09-06T00:00:00+00:00",
            },
        }
        self.assertEqual(canonical_json(manifest(audit)), canonical_json(manifest(audit)))
        self.assertEqual(json.loads(canonical_json(manifest(audit)))["package_versions"]["xgboost"], "2.1.4")

    def test_phase_does_not_train_calibrate_or_evaluate_august(self):
        import src.phase30_environment_recovery as phase30

        source = inspect.getsource(phase30)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("august_2026", source)

    def test_required_project_imports_are_explicit_and_stable(self):
        import src.phase30_environment_recovery as phase30

        self.assertEqual(phase30.REQUIRED_RUNTIME_IMPORTS[:5], (
            "numpy", "pandas", "scipy", "sklearn", "xgboost",
        ))
        self.assertIn("src.xgboost_benchmark", phase30.TRAINING_IMPORTS)


if __name__ == "__main__":
    unittest.main()
