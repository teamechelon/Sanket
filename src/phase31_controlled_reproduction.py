"""Phase 31 controlled-reproduction preflight.

This audit intentionally does not train.  It records the frozen training
contract and stops before execution when the surviving entry point cannot
serialize or load the required model artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PHASE = 31
TRAINING_ENTRY_POINT = "src/xgboost_benchmark.py"
DATASET_PATH = "data/features/schedule_modeling.csv"
SOURCE_FILES = (
    "src/xgboost_benchmark.py",
    "src/baseline_models.py",
    "src/schedule_robustness.py",
    "src/label_feasibility.py",
    "src/available_data_audit.py",
    "requirements.txt",
)
SERIALIZATION_TOKENS = (
    "joblib.dump", "pickle.dump", "cloudpickle.dump", ".save_model(",
)
LOADING_TOKENS = (
    "joblib.load", "pickle.load", "cloudpickle.load", ".load_model(",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )


def git(root: Path, *args: str) -> str:
    result = run_command("git", *args, cwd=root)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} is not a literal assignment in {path}")


def feature_contract(root: Path) -> dict[str, Any]:
    source = root / "src/baseline_models.py"
    features = literal_assignment(source, "CORE") + literal_assignment(source, "CONDITIONAL_INCLUDED")
    return {
        "source": "src/baseline_models.py: CORE + CONDITIONAL_INCLUDED",
        "ordered_features": features,
        "feature_count": len(features),
        "feature_contract_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_sha256": sha256(source),
    }


def xgboost_configuration(root: Path) -> dict[str, Any]:
    source = (root / TRAINING_ENTRY_POINT).read_text(encoding="utf-8")
    block = source[source.index("XGB_CONFIG = {"):source.index("}\nNEGLIGIBLE_DELTA") + 1]
    seed = literal_assignment(root / "src/baseline_models.py", "SEED")
    return {
        "source": "src/xgboost_benchmark.py: XGB_CONFIG",
        "source_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
        "random_seed": seed,
        "configuration_text": block,
    }


def serialization_evidence(root: Path) -> dict[str, list[str]]:
    """Search executable ML source, excluding evidence-only phase audits."""
    candidates = [path for path in (root / "src").glob("*.py") if not path.name.startswith("phase")]
    saved, loaded = [], []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(root)).replace("\\", "/")
        if any(token in text for token in SERIALIZATION_TOKENS):
            saved.append(relative)
        if any(token in text for token in LOADING_TOKENS):
            loaded.append(relative)
    return {"serialization_sources": saved, "loading_sources": loaded}


def environment_versions(python: Path, root: Path) -> dict[str, str]:
    result = run_command(str(python), "-m", "pip", "list", "--format=json", cwd=root)
    if result.returncode:
        raise RuntimeError(f"unable to inspect Phase 30 environment: {result.stderr.strip()}")
    packages = json.loads(result.stdout)
    wanted = {"numpy", "pandas", "scipy", "scikit-learn", "xgboost"}
    return {
        package["name"].lower(): package["version"]
        for package in packages if package["name"].lower() in wanted
    }


def baseline_working_tree_status(root: Path) -> str:
    """Keep Phase 31 audit files out of the state captured before training."""
    status = git(root, "status", "--short")
    phase31_paths = {
        "src/phase31_controlled_reproduction.py",
        "tests/test_phase31_controlled_reproduction.py",
        "reports/phase31_pretraining_manifest.json",
        "reports/phase31_controlled_reproduction_report.txt",
    }
    return "\n".join(
        line for line in status.splitlines()
        if not any(path in line.replace("\\", "/") for path in phase31_paths)
    )


def pretraining_manifest(root: Path, python: Path) -> dict[str, Any]:
    serialization = serialization_evidence(root)
    files = {path: sha256(root / path) for path in SOURCE_FILES}
    data = root / DATASET_PATH
    return {
        "schema_version": 1,
        "phase": PHASE,
        "source_commit": git(root, "rev-parse", "HEAD"),
        "working_tree_status_before_training": baseline_working_tree_status(root),
        "working_tree_diff_before_training": git(root, "diff", "--no-ext-diff"),
        "training_entry_point": TRAINING_ENTRY_POINT,
        "documented_command_not_executed": (
            "python -m src.xgboost_benchmark --data data/features/schedule_modeling.csv --report-dir reports"
        ),
        "dataset": {"path": DATASET_PATH, "sha256": sha256(data), "size_bytes": data.stat().st_size},
        "source_hashes": files,
        "feature_contract": feature_contract(root),
        "label_contract": {
            "target": "future_schedule_later_3m",
            "definition_source": "src/label_feasibility.py: maturity-safe next-three-endpoint label construction",
            "source_sha256": files["src/label_feasibility.py"],
        },
        "preprocessing": {
            "source": "src/baseline_models.py: training-fitted imputation, indicators, scaling, and one-hot encoding",
            "source_sha256": files["src/baseline_models.py"],
        },
        "split_logic": {
            "source": "src/available_data_audit.py and src/schedule_robustness.py",
            "source_sha256": files["src/available_data_audit.py"],
        },
        "xgboost_configuration": xgboost_configuration(root),
        "environment": {"python_executable": str(python), "package_versions": environment_versions(python, root)},
        "serialization_evidence": serialization,
        "execution_gate": {
            "existing_entry_point_has_serializer": bool(serialization["serialization_sources"]),
            "existing_prediction_path_has_loader": bool(serialization["loading_sources"]),
            "result": "BLOCKED" if not all(serialization.values()) else "READY",
            "reason": (
                "The surviving XGBoost training entry point has no serializer and the repository has no persisted-model loader. "
                "Phase 31 requires a real artifact and normal load-back validation; altering source would violate the methodology freeze."
            ),
        },
        "test_execution": {
            "phase31_tests": "Run after preflight manifest generation.",
            "full_suite": "NOT RUN: its estimator-fitting tests are outside the failed pre-execution gate.",
        },
    }


def report_text(manifest: dict[str, Any]) -> str:
    gate = manifest["execution_gate"]
    versions = manifest["environment"]["package_versions"]
    lines = [
        "PHASE 31 — CONTROLLED MODEL REPRODUCTION & ARTIFACT CREATION",
        "=" * 68,
        "", "1. OBJECTIVE", "Establish a new Phase 31 reproducibility-baseline artifact only if the unchanged surviving pipeline can serialize and load it.",
        "", "2. REPOSITORY STATE", f"Source commit: {manifest['source_commit']}",
        f"Working tree before training: {'clean' if not manifest['working_tree_status_before_training'] else manifest['working_tree_status_before_training']}",
        "", "3. TRAINING SOURCE", f"Entry point: {manifest['training_entry_point']}.",
        f"Documented command: {manifest['documented_command_not_executed']}.",
        "", "4. ENVIRONMENT", ", ".join(f"{name}={version}" for name, version in sorted(versions.items())),
        "", "5. DATASET EVIDENCE", f"{manifest['dataset']['path']} SHA-256: {manifest['dataset']['sha256']}",
        "", "6. FEATURE/LABEL CONTRACT",
        f"Feature count: {manifest['feature_contract']['feature_count']}; feature contract SHA-256: {manifest['feature_contract']['feature_contract_sha256']}",
        f"Target: {manifest['label_contract']['target']}.",
        f"Random seed: {manifest['xgboost_configuration']['random_seed']}; XGBoost configuration SHA-256: {manifest['xgboost_configuration']['source_sha256']}",
        "", "7. EXECUTION GATE", gate["reason"],
        f"Serializer sources: {manifest['serialization_evidence']['serialization_sources'] or 'NONE'}.",
        f"Loader sources: {manifest['serialization_evidence']['loading_sources'] or 'NONE'}.",
        "", "8. FIRST EXECUTION RESULT", "NOT RUN. The gate failed before training; no model was fitted.",
        "", "9. MODEL ARTIFACT RESULT", "NOT CREATED. No existing serializer or normal inference loader exists.",
        "", "10. LOAD-BACK VALIDATION", "NOT POSSIBLE without changing source, which Phase 31 forbids.",
        "", "11. SECOND EXECUTION RESULT", "NOT RUN because the first controlled execution was blocked.",
        "", "12. REPRODUCIBILITY CLASSIFICATION", "BLOCKED.",
        "", "13. HISTORICAL COMPARISON", "HISTORICAL DOCUMENTED RESULT: ROC-AUC approximately 0.855 and PR-AUC approximately 0.805. PHASE 31 NEW REPRODUCED RESULT: none; no comparison is possible.",
        "", "14. CALIBRATION STATUS", "NOT FITTED. Scores remain relative-risk rankings, not probabilities.",
        "", "15. AUGUST HOLDOUT STATUS", "Not created or evaluated.",
        "", "16. LIMITATIONS", "This is not historical artifact recovery. The missing serialization and loading capability is an execution blocker, not a reason to change methodology.",
        "", "17. EXACT NEXT STEP", "Obtain an existing approved serialization/inference implementation, or separately authorize a source change in a later phase; do not modify the frozen training pipeline in Phase 31.",
        "", "18. TEST EXECUTION", "Phase 31 tests validate the integrity and block conditions. The full suite is not executed because it fits estimators after the serialization gate has failed.",
        "", "REQUIRED STATUS FIELDS",
        "MODEL CHANGED: NO", "MODEL RETRAINED: NO", "NEW MODEL ARTIFACT CREATED: NO",
        "HISTORICAL MODEL RECOVERED: NO", "FEATURES CHANGED: NO", "THRESHOLDS CHANGED: NO",
        "CALIBRATION FITTED: NO", "AUGUST DATA CREATED: NO", "AUGUST HOLDOUT EVALUATED: NO",
        "METHODOLOGY CHANGED: NO", "REPRODUCIBILITY STATUS: BLOCKED",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(manifest: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase31_pretraining_manifest.json").write_bytes(canonical_json(manifest))
    (report_dir / "phase31_controlled_reproduction_report.txt").write_text(report_text(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    audit = pretraining_manifest(args.root.resolve(), args.python.resolve())
    write_outputs(audit, args.report_dir)


if __name__ == "__main__":
    main()
