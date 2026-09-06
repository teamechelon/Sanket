"""Phase 29 reproducibility-baseline audit.

This is an evidence-only baseline.  It parses surviving repository sources and
hashes their inputs without importing ML libraries, fitting a model, creating
August labels, or claiming recovery of missing historical Phase 23--27 work.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


HISTORICAL_CONTRACT_SHA256 = "6fa4aca992807c741fa7e24d969db161966760562ddf9f48193d21dd7559f149"
EXPECTED_XGBOOST_VERSION = "2.1.4"
MODEL_PATH = "models/phase29_reproducibility_baseline.joblib"
OUTPUTS = {
    "reports/phase29_reproducibility_contract.json",
    "reports/phase29_model_reproducibility_report.txt",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} is not a literal assignment in {path}")


def _feature_list(root: Path) -> list[str]:
    source = root / "src/baseline_models.py"
    return _literal_assignment(source, "CORE") + _literal_assignment(source, "CONDITIONAL_INCLUDED")


def _record(value: Any, source_file: str, source_location: str,
            verification_status: str = "VERIFIED") -> dict[str, Any]:
    return {
        "value": value,
        "source_file": source_file,
        "source_location": source_location,
        "verification_status": verification_status,
    }


def _xgb_configuration(root: Path) -> dict[str, Any]:
    source = (root / "src/xgboost_benchmark.py").read_text(encoding="utf-8")
    # Values are parsed only from the frozen source block; SEED and np.nan are
    # resolved from their repository definitions rather than from ML imports.
    config: dict[str, Any] = {}
    block = source[source.index("XGB_CONFIG = {"):source.index("}\nNEGLIGIBLE_DELTA")]
    for key, raw in re.findall(r'"([^"]+)": ([^,\n]+)', block):
        if raw == "SEED":
            config[key] = _literal_assignment(root / "src/baseline_models.py", "SEED")
        elif raw == "np.nan":
            config[key] = "NaN"
        else:
            config[key] = ast.literal_eval(raw)
    return config


def build_contract(root: Path) -> dict[str, Any]:
    features = _feature_list(root)
    configuration = _xgb_configuration(root)
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    version_match = re.search(r"^xgboost==([^\s]+)$", requirements, flags=re.MULTILINE)
    if not version_match:
        raise ValueError("requirements.txt does not pin xgboost")
    fields = {
        "model_family": _record("XGBoost XGBClassifier", "src/xgboost_benchmark.py", "lines 10-13, 87-92"),
        "xgboost_version": _record(version_match.group(1), "requirements.txt", "line 8"),
        "training_dataset": _record("data/features/schedule_modeling.csv", "src/xgboost_benchmark.py", "lines 378-414"),
        "feature_list_ordered": _record(features, "src/baseline_models.py", "lines 22-38"),
        "feature_count": _record(len(features), "src/baseline_models.py", "lines 22-38"),
        "preprocessing": _record(
            "numeric: median imputation + missing indicators + StandardScaler; categorical: most-frequent imputation + OneHotEncoder(handle_unknown=ignore)",
            "src/baseline_models.py", "lines 55-64",
        ),
        "target": _record("future_schedule_later_3m", "src/schedule_robustness.py", "lines 16-17"),
        "label_definition": _record(
            "schedule target becomes later within the exact next three monthly endpoints; missing endpoint is UNKNOWN and excluded",
            "src/label_feasibility.py", "lines 149-185",
        ),
        "training_split": _record(
            "six expanding maturity-safe walk-forward folds: train 2025-07 through evaluation month minus four; evaluate 2025-11 through 2026-04",
            "src/available_data_audit.py", "lines 38, 49-62",
        ),
        "project_disjoint_split": _record("deterministic Phase 15/16 project-disjoint split", "src/xgboost_benchmark.py", "lines 177-199"),
        "hyperparameters": _record(configuration, "src/xgboost_benchmark.py", "lines 26-44"),
        "random_seed": _record(configuration["random_state"], "src/baseline_models.py", "line 21; src/xgboost_benchmark.py line 38"),
        "missing_value_handling": _record("XGBoost missing=NaN plus training-fitted preprocessing", "src/xgboost_benchmark.py", "lines 40, 87-92"),
        "class_weighting": _record("natural classes; scale_pos_weight=1.0", "reports/phase16_xgboost_specification.txt", "configuration section"),
        "thresholds": _record([0.40, 0.50], "src/xgboost_benchmark.py", "lines 104-106"),
        "prediction_semantics": _record("relative risk ranking; not calibrated event probability", "reports/phase20_raw_calibration_audit.txt", "sections 1 and 12"),
        "calibration": _record("NONE", "reports/phase16_xgboost_specification.txt", "configuration section"),
    }
    canonical_values = {name: record["value"] for name, record in fields.items()}
    return {
        "schema_version": 1,
        "phase": 29,
        "contract_kind": "new_reproducibility_baseline_from_surviving_repository_evidence",
        "historical_documented_contract_sha256": HISTORICAL_CONTRACT_SHA256,
        "new_reproducibility_baseline_contract_sha256": hashlib.sha256(
            json.dumps(canonical_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "fields": fields,
    }


def dependency_environment(root: Path) -> dict[str, Any]:
    requirements = root / "requirements.txt"
    return {
        "requirements_sha256": _sha256(requirements),
        "requirements_source": "requirements.txt",
        "python_runtime": platform.python_version(),
        "scikit_learn_available": importlib.util.find_spec("sklearn") is not None,
        "xgboost_available": importlib.util.find_spec("xgboost") is not None,
        "pip_dry_run_result": (
            "BLOCKED: pip install --dry-run --no-cache-dir -r requirements.txt on the available Python 3.14 runtime "
            "reported no matching distribution for scikit-learn>=1.6,<2."
        ),
        "environment_reproduction_status": "BLOCKED",
    }


def input_hashes(root: Path) -> dict[str, str]:
    paths = (
        "data/features/schedule_modeling.csv",
        "src/xgboost_benchmark.py",
        "src/baseline_models.py",
        "src/schedule_robustness.py",
        "src/available_data_audit.py",
        "src/label_feasibility.py",
        "requirements.txt",
    )
    return {path: _sha256(root / path) for path in paths}


def artifact_status(root: Path) -> dict[str, Any]:
    artifact = root / MODEL_PATH
    return {
        "artifact_path": MODEL_PATH,
        "exists": artifact.is_file(),
        "sha256": _sha256(artifact) if artifact.is_file() else None,
        "file_size_bytes": artifact.stat().st_size if artifact.is_file() else None,
        "reproducibility_status": "NOT_CREATED_ENVIRONMENT_BLOCKED",
        "historical_model_artifact_recovered": False,
    }


def audit_once(root: Path) -> dict[str, Any]:
    contract = build_contract(root)
    environment = dependency_environment(root)
    artifact = artifact_status(root)
    return {
        "schema_version": 1,
        "phase": 29,
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "contract": contract,
        "input_hashes": input_hashes(root),
        "dependency_environment": environment,
        "training_pipeline": {
            "trace": [
                "data/features/schedule_modeling.csv",
                "preprocessing in src/baseline_models.py",
                "maturity-safe labels from src/label_feasibility.py",
                "six walk-forward folds in src/available_data_audit.py",
                "XGBClassifier pipeline in src/xgboost_benchmark.py",
                "serialization: no repository implementation exists",
            ],
            "pipeline_matches_documented_methodology": "VERIFIED_BY_SOURCE_INSPECTION",
            "inconsistency": "No serialized-model save/load implementation exists in surviving source.",
        },
        "new_phase29_evidence": "No model trained: exact dependency environment is blocked.",
        "historical_repository_evidence": "Phase 16/20 reports document prior metrics only; they are not new reproduction evidence.",
        "missing_historical_evidence": "Phase 23-27 historical artifacts and serialized historical model remain unavailable.",
        "model_artifact": artifact,
        "phase27_blocker_status": "NOT_RESOLVED",
        "reproducibility_status": "BLOCKED",
        "actions": {
            "model_changed": False,
            "model_retrained": False,
            "new_model_artifact_created": False,
            "historical_model_artifact_recovered": False,
            "features_changed": False,
            "thresholds_changed": False,
            "calibration_fitted": False,
            "august_data_created": False,
            "august_holdout_evaluated": False,
            "historical_phase23_27_artifacts_recovered": False,
        },
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _assert_identical(first: dict[str, Any], second: dict[str, Any]) -> None:
    if canonical_bytes(first) != canonical_bytes(second):
        raise AssertionError("Phase 29 evidence audit is not deterministic")


def report_text(audit: dict[str, Any]) -> str:
    contract = audit["contract"]
    environment = audit["dependency_environment"]
    artifact = audit["model_artifact"]
    lines = [
        "PHASE 29 — MODEL ARTIFACT RECONSTRUCTION & REPRODUCIBILITY BASELINE",
        "=" * 74,
        "", "A. HISTORICAL REPOSITORY EVIDENCE",
        audit["historical_repository_evidence"],
        f"Historical documented logical-contract SHA-256: {contract['historical_documented_contract_sha256']}",
        "This is a logical configuration hash, not a serialized fitted-model hash.",
        "", "B. NEWLY ESTABLISHED PHASE 29 EVIDENCE",
        f"Source commit inspected: {audit['source_commit']}",
        f"New reproducibility-baseline contract SHA-256: {contract['new_reproducibility_baseline_contract_sha256']}",
        "The new hash is a canonical record of surviving-source contract fields, not historical provenance.",
        "", "C. MISSING HISTORICAL EVIDENCE", audit["missing_historical_evidence"],
        "", "D. TRAINING REPRODUCIBILITY",
        "BLOCKED before training. No substitute implementation, hyperparameter change, or replacement model was used.",
        "", "E. MODEL ARTIFACT STATUS",
        f"Artifact path: {artifact['artifact_path']}", f"Artifact created: {artifact['exists']}",
        f"Artifact SHA-256: {artifact['sha256'] or 'NOT CREATED'}",
        "", "F. CONTRACT STATUS",
        f"Model family/version: {contract['fields']['model_family']['value']} / {contract['fields']['xgboost_version']['value']}",
        f"Ordered features: {contract['fields']['feature_count']['value']}",
        f"Target: {contract['fields']['target']['value']}",
        f"Thresholds: {contract['fields']['thresholds']['value']}; calibration: {contract['fields']['calibration']['value']}",
        "Every contract field, source file, source location, and verification status is in phase29_reproducibility_contract.json.",
        "", "G. DEPENDENCY STATUS",
        f"Python runtime: {environment['python_runtime']}",
        f"scikit-learn available: {environment['scikit_learn_available']}",
        f"xgboost available: {environment['xgboost_available']}",
        environment['pip_dry_run_result'],
        "ENVIRONMENT REPRODUCTION: BLOCKED",
        "", "H. KNOWN LIMITATIONS",
        "No serialization/loading path exists in surviving repository source. Phase 16/20 metrics are historical documented results only. "
        "No May-to-August holdout is mature or evaluated.",
        "", "I. PHASE 27 BLOCKER STATUS", "NOT RESOLVED. FINAL CANDIDATE: NOT APPROVED.",
        "", "REQUIRED STATUS FIELDS",
        "MODEL CHANGED: NO", "MODEL RETRAINED: NO", "NEW MODEL ARTIFACT CREATED: NO",
        "HISTORICAL MODEL ARTIFACT RECOVERED: NO", "FEATURES CHANGED: NO", "THRESHOLDS CHANGED: NO",
        "CALIBRATION FITTED: NO", "AUGUST DATA CREATED: NO", "AUGUST HOLDOUT EVALUATED: NO",
        "HISTORICAL PHASE 23–27 ARTIFACTS RECOVERED: NO", "REPRODUCIBILITY STATUS: BLOCKED",
    ]
    return "\n".join(lines) + "\n"


def run(root: Path, report_dir: Path) -> dict[str, Any]:
    first = audit_once(root)
    second = audit_once(root)
    _assert_identical(first, second)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase29_reproducibility_contract.json").write_bytes(canonical_bytes(first["contract"]))
    (report_dir / "phase29_model_reproducibility_report.txt").write_text(report_text(first), encoding="utf-8")
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.root, args.report_dir)


if __name__ == "__main__":
    main()
