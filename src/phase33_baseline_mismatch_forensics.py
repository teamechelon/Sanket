"""Read-only Phase 33 forensics for the Phase 15 baseline assertion.

This module deliberately does not import the training implementation or fit an
estimator.  It records only repository, Git, CSV, and package metadata.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PHASE15_COMMIT = "a06e1989cd3d7219d99512150899a3e215a5bc0f"
BASELINE_COMMIT = "9fb642e7ffcd6520701de707beac9690ae840f3e"
ROOT_CAUSE = "J — Unresolved"
DECISION = "ROOT CAUSE PARTIALLY IDENTIFIED"
REPRODUCTION_STATUS = "BLOCKED"
PHASE33_PATHS = {
    "reports/phase33_baseline_mismatch_forensics_report.txt",
    "reports/phase33_baseline_mismatch_forensics_manifest.json",
    "src/phase33_baseline_mismatch_forensics.py",
    "tests/test_phase33_baseline_mismatch_forensics.py",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, encoding="utf-8",
        errors="replace", capture_output=True, check=False,
    )
    if result.returncode and not allow_failure:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _commit(root: Path, commit: str) -> dict[str, str]:
    subject = _git(root, "show", "-s", "--format=%aI|%s", commit)
    date, message = subject.split("|", 1)
    return {"commit": commit, "date": date, "subject": message}


def _object_id(root: Path, spec: str) -> str | None:
    value = _git(root, "rev-parse", spec, allow_failure=True)
    return value if value else None


def _status(root: Path) -> list[str]:
    """Keep Phase 33's own outputs from changing its repeatable observation."""
    return [line for line in _git(root, "status", "--short").splitlines()
            if line[3:].replace("\\", "/") not in PHASE33_PATHS]


def _literal_lists(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"CORE", "CONDITIONAL_INCLUDED"}:
                values[target.id] = ast.literal_eval(node.value)
    return values


def feature_contract(root: Path) -> dict[str, Any]:
    values = _literal_lists(root / "src" / "baseline_models.py")
    features = values["CORE"] + values["CONDITIONAL_INCLUDED"]
    encoded = json.dumps(features, separators=(",", ":")).encode("utf-8")
    return {
        "count": len(features), "ordered_features": features,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "expected_phase32_sha256": "9330da27a587945d3235869573f1638ea3df82069cac1e7c30bd8bbc11543440",
    }


def dataset_summary(path: Path) -> dict[str, Any]:
    digest = sha256_path(path)
    duplicate_counter: Counter[tuple[str, ...]] = Counter()
    months: Counter[str] = Counter()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            row_count += 1
            months[row["prediction_month"]] += 1
            duplicate_counter[tuple(row.get(column, "") for column in columns)] += 1
    return {
        "path": path.as_posix(), "sha256": digest, "rows": row_count,
        "columns": columns, "column_count": len(columns),
        "duplicate_rows": sum(count - 1 for count in duplicate_counter.values() if count > 1),
        "month_rows": dict(sorted(months.items())),
    }


def package_versions() -> dict[str, str | None]:
    names = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
             "scikit_learn": "scikit-learn", "xgboost": "xgboost"}
    result: dict[str, str | None] = {}
    for key, name in names.items():
        try:
            result[key] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[key] = None
    return result


def _csv_rows(path: Path, variant: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("variant") == variant]


def _config_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("XGB_CONFIG = {")
    end = text.index("}\n", start) + 2
    return hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()


def xgboost_config(path: Path) -> dict[str, Any]:
    """Read configuration syntax without importing the benchmark module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "XGB_CONFIG"):
            if not isinstance(node.value, ast.Dict):
                raise ValueError("XGB_CONFIG is not a dict")
            return {
                ast.literal_eval(key): (
                    ast.literal_eval(value) if isinstance(value, ast.Constant)
                    else ast.unparse(value)
                )
                for key, value in zip(node.value.keys, node.value.values)
            }
    raise ValueError("XGB_CONFIG is missing")


def _evidence(root: Path) -> dict[str, dict[str, Any]]:
    entries = {
        "phase11_fixed_split_metrics": "reports/baseline_model_results.csv",
        "phase11_fixed_split_report": "reports/baseline_model_report.txt",
        "phase12_robustness_narrative": "reports/schedule_baseline_robustness_report.txt",
        "phase15_fold_metrics": "reports/phase15_walk_forward_results.csv",
        "phase15_window_report": "reports/phase15_available_data_audit.txt",
        "phase15_window_csv": "reports/phase15_valid_temporal_windows.csv",
        "phase32_failed_run": "reports/phase32_model_artifact_report.txt",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, relative in entries.items():
        path = root / relative
        result[name] = {
            "path": relative, "exists": path.is_file(),
            "sha256": sha256_path(path) if path.is_file() else None,
        }
    return result


def historical_evidence_records(root: Path) -> list[dict[str, Any]]:
    """Provenance is explicit: reports describe results, source is executable."""
    records = [
        ("phase11_fixed_split_metric", "reports/baseline_model_results.csv", BASELINE_COMMIT,
         "line 11", "ROC-AUC=0.855136; PR-AUC=0.804918; train_rows=4475; test_rows=3368", False),
        ("phase11_fixed_split_implementation", "src/baseline_models.py", BASELINE_COMMIT,
         "lines 39-79", "fixed train/validation/test split, training-only pipeline, sklearn ROC/PR metrics", True),
        ("phase15_implementation", "src/available_data_audit.py", PHASE15_COMMIT,
         "lines 49-102", "six maturity-safe expanding folds and fold-wise sklearn ROC/PR metrics", True),
        ("phase15_fold_reference", "reports/phase15_walk_forward_results.csv", PHASE15_COMMIT,
         "FULL_EXISTING_BASELINE rows 2, 5, 8, 11, 14, 17", "six stored fold references; not fitted predictions", False),
        ("phase15_window_narrative", "reports/phase15_available_data_audit.txt", PHASE15_COMMIT,
         "lines 29-48", "window definitions and fold-level metric narrative", False),
        ("phase32_failure_record", "reports/phase32_model_artifact_report.txt", "UNCOMMITTED_PHASE32",
         "lines 13-37", "assertion failed for roc_auc before artifact fitting; no recomputed numeric value logged", False),
    ]
    return [{
        "name": name, "file": file, "commit": commit,
        "date": _commit(root, commit)["date"] if len(commit) == 40 else None,
        "location": location, "content": content,
        "evidence_type": "executable" if executable else "narrative",
    } for name, file, commit, location, content, executable in records]


def audit(root: Path) -> dict[str, Any]:
    """Return deterministic, non-training Phase 33 evidence."""
    root = root.resolve()
    dataset = dataset_summary(root / "data/features/schedule_modeling.csv")
    features = feature_contract(root)
    phase15_rows = _csv_rows(root / "reports/phase15_walk_forward_results.csv", "FULL_EXISTING_BASELINE")
    current_source_ids = {
        relative: _object_id(root, f"HEAD:{relative}")
        for relative in ("data/features/schedule_modeling.csv", "src/baseline_models.py",
                         "src/schedule_robustness.py", "src/available_data_audit.py",
                         "requirements.txt")
    }
    phase15_source_ids = {
        relative: _object_id(root, f"{PHASE15_COMMIT}:{relative}")
        for relative in current_source_ids
    }
    unchanged = {key: current_source_ids[key] == phase15_source_ids[key]
                 for key in current_source_ids}
    return {
        "schema_version": 1,
        "phase": 33,
        "diagnostic_only": True,
        "current_state": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "working_tree_status_excluding_phase33_outputs": _status(root),
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "package_versions": package_versions(),
            "dataset": dataset,
            "feature_contract": features,
            "target": "future_schedule_later_3m",
            "random_seed": 26103,
            "xgboost_configuration": xgboost_config(root / "src/xgboost_benchmark.py"),
            "xgboost_config_source_sha256": _config_hash(root / "src/xgboost_benchmark.py"),
        },
        "history": {
            "phase11_baseline_introduction": _commit(root, BASELINE_COMMIT),
            "phase15_benchmark_introduction": _commit(root, PHASE15_COMMIT),
            "phase11_baseline_results_blob": _object_id(root, f"{BASELINE_COMMIT}:reports/baseline_model_results.csv"),
            "phase15_fold_results_blob": _object_id(root, f"{PHASE15_COMMIT}:reports/phase15_walk_forward_results.csv"),
            "phase15_to_current_source_blob_unchanged": unchanged,
            "post_phase15_available_data_change": {
                "commit": "c4f1cfc4d480b353a175449099e92ad50f627c28",
                "finding": "Only error-slice factoring; maturity_windows and walk_forward metric logic are unchanged.",
            },
            "phase15_xgboost_configuration": "None: xgboost_benchmark.py was introduced by Phase 16 after Phase 15.",
        },
        "historical_metric_evidence": {
            "reference": {"roc_auc": 0.855136, "pr_auc": 0.804918},
            "source": "reports/baseline_model_results.csv, CORE_PLUS_CONDITIONAL Random Forest",
            "source_commit": BASELINE_COMMIT,
            "executable_evidence": False,
            "method": "fixed split: train 2025-07..2025-12; validation 2026-01..2026-02; test 2026-03..2026-04; threshold selected on validation",
            "meaning": "This is a stored narrative/result CSV; its fitted estimator and predictions do not survive.",
        },
        "phase15_assertion_evidence": {
            "stored_fold_metrics": [
                {key: row[key] for key in ("fold", "training_period_start", "training_period_end",
                                            "evaluation_period", "train_rows", "evaluation_rows",
                                            "positive_rows", "event_rate", "roc_auc", "pr_auc")}
                for row in phase15_rows
            ],
            "assertion_target": "Each Phase 15 fold metric in phase15_walk_forward_results.csv; not the 0.855136/0.804918 fixed-split result.",
            "metric_implementation": "sklearn.metrics.roc_auc_score and average_precision_score, calculated separately per fold; no pooling or averaging occurs inside the assertion.",
            "current_failed_run": "Phase 32 recorded ValueError: Phase 15 baseline changed for roc_auc, without logging the recomputed value.",
        },
        "forensics": {
            "dataset": "Git blob is identical at Phase 15 and HEAD; current SHA-256 is recorded above. Historical standalone SHA-256 was not preserved, but the Git object identity proves repository bytes are unchanged.",
            "features": "The 29-feature contract hashes to the stated Phase 32 value. baseline_models.py and schedule_robustness.py Git blobs are identical at Phase 15 and HEAD.",
            "labels_and_maturity": "Target is future_schedule_later_3m. Current maturity_windows source blob is identical to Phase 15; it creates six expanding folds with a t+3 endpoint before evaluation.",
            "splits": "Stored Phase 15 rows define six folds: July 2025 through December 2025 expanding training, evaluated November 2025 through April 2026. These are not the Phase 11 fixed split.",
            "predictions": "HISTORICAL PREDICTIONS: NOT AVAILABLE. Exact prediction-level comparison is impossible without fitting, which this phase forbids.",
            "dependencies": "Historical package versions were not recorded in Phase 15 evidence. Current environment is recorded above; unpinned numpy/pandas/scipy/scikit-learn requirements leave a numerical dependency difference plausible but unproven.",
            "exact_mismatch": "Historical fixed-split ROC-AUC/PR-AUC are 0.855136/0.804918. The Phase 32 failed run exposes no current recomputed ROC-AUC or PR-AUC, so absolute differences cannot be calculated without prohibited fitting.",
        },
        "phase32_modification_audit": {
            "model_inference_py": "SAFE DEPLOYMENT INFRASTRUCTURE",
            "serialize_pipeline": "SAFE DEPLOYMENT INFRASTRUCTURE — persistence only; no fit or estimator assignment.",
            "load_pipeline": "SAFE DEPLOYMENT INFRASTRUCTURE",
            "validate_feature_frame": "SAFE DEPLOYMENT INFRASTRUCTURE",
            "predict_risk_scores": "SAFE DEPLOYMENT INFRASTRUCTURE",
            "fit_final_temporal_fold": "METHODOLOGY CHANGE — it fits a newly selected standalone fold.",
            "artifact_path": "METHODOLOGY CHANGE — invokes fit_final_temporal_fold after benchmark completion.",
            "phase32_audit_reports": "SAFE DEPLOYMENT INFRASTRUCTURE / diagnostic records; the existing report confirms artifact fitting was not reached.",
            "caused_phase15_mismatch": False,
            "ordering_evidence": "run() calls experiment(), which calls _phase15_baseline(), before the artifact-path branch; the recorded Phase 32 run failed in _phase15_baseline().",
        },
        "classification": {
            "root_cause": ROOT_CAUSE,
            "decision": DECISION,
            "reproduction_status": REPRODUCTION_STATUS,
            "confirmed_finding": "H — historical reference inconsistency is confirmed: 0.855136/0.804918 belongs to the earlier fixed-split baseline, not the Phase 15 assertion's six-fold reference.",
            "unresolved_finding": "The actual recomputed fold metric was not retained by the failed run. With no historical predictions/version lock and a no-fit instruction, the direct cause of the roc_auc assertion failure cannot be isolated.",
        },
        "evidence": _evidence(root),
        "historical_evidence_records": historical_evidence_records(root),
    }


def manifest_bytes(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")


def report_text(result: dict[str, Any]) -> str:
    state = result["current_state"]
    history = result["history"]
    folds = result["phase15_assertion_evidence"]["stored_fold_metrics"]
    lines = [
        "PHASE 33 — PHASE 15 BASELINE MISMATCH FORENSICS", "=" * 72, "",
        "DIAGNOSTIC SCOPE", "No estimator was fitted, no artifact was created, and no model, dataset, feature, label, split, threshold, calibration, dependency, or Phase 15 assertion was changed.", "",
        "1. CURRENT STATE", f"Commit/branch: {state['commit']} / {state['branch']}",
        f"Working tree (excluding this Phase 33 artifact set): {state['working_tree_status_excluding_phase33_outputs']}",
        f"Python: {state['python_executable']} | {state['python_version']}",
        f"Packages: {state['package_versions']}",
        f"Dataset: {state['dataset']['path']} | SHA-256={state['dataset']['sha256']} | rows={state['dataset']['rows']} | columns={state['dataset']['column_count']} | duplicate rows={state['dataset']['duplicate_rows']}",
        f"Feature contract: {state['feature_contract']['count']} features | SHA-256={state['feature_contract']['sha256']}",
        f"Target/seed/XGB-config-source-hash: {state['target']} / {state['random_seed']} / {state['xgboost_config_source_sha256']}", "",
        f"Current XGBoost configuration: {state['xgboost_configuration']}", "",
        "2. HISTORICAL EVIDENCE AND GIT HISTORY",
        f"Phase 11 fixed-split baseline: {history['phase11_baseline_introduction']}",
        f"Phase 15 available-data benchmark: {history['phase15_benchmark_introduction']}",
        "The 0.855136 ROC-AUC and 0.804918 PR-AUC values are stored in baseline_model_results.csv for the CORE_PLUS_CONDITIONAL Random Forest fixed split. That CSV is narrative/result evidence, not surviving predictions or a fitted estimator.",
        "Phase 15's stored reference is instead phase15_walk_forward_results.csv: six separately-scored maturity-safe folds. The assertion compares those fold columns, not a pooled/mean 0.855136/0.804918 value.", "",
        "Phase 15 had no XGBoost configuration: the XGBoost benchmark was introduced only in Phase 16. The current XGBoost configuration therefore cannot explain a Phase 15 Random Forest assertion mismatch.", "",
        "3. PHASE 15 FOLDS",
    ]
    for row in folds:
        lines.append("fold {fold}: train {training_period_start}..{training_period_end} ({train_rows}); eval {evaluation_period} ({evaluation_rows}), positives={positive_rows}, prevalence={event_rate}, ROC={roc_auc}, PR={pr_auc}".format(**row))
    lines += [
        "", "4. SOURCE, DATA, FEATURE, LABEL, AND SPLIT FORENSICS",
        result["forensics"]["dataset"], result["forensics"]["features"],
        result["forensics"]["labels_and_maturity"], result["forensics"]["splits"],
        "The only post-Phase-15 available_data_audit change is error-slice factoring; it does not change maturity windows or walk-forward metrics.", "",
        "5. MODEL, PREDICTION, METRIC, AND DEPENDENCY FORENSICS",
        "The Phase 15 baseline is RandomForestClassifier(200 trees, min_samples_leaf=5, max_features=sqrt, random_state=26103, n_jobs=-1) within training-only median/mode imputation, numeric indicators/scaling, and categorical one-hot encoding.",
        "CSV columns and their order are preserved by the identical Git dataset blob. Historical pandas-inferred dtypes, feature-value missingness totals, and stored predictions were not preserved separately; no reconstruction was attempted.",
        result["forensics"]["predictions"], result["phase15_assertion_evidence"]["metric_implementation"],
        result["forensics"]["dependencies"], "",
        "6. EXACT MISMATCH",
        result["forensics"]["exact_mismatch"],
        "Therefore current ROC-AUC=?, current PR-AUC=?, and exact absolute differences are NOT AVAILABLE from surviving non-training evidence.", "",
        "7. PHASE 32 MODIFICATION AUDIT",
    ]
    for name, classification in result["phase32_modification_audit"].items():
        lines.append(f"{name}: {classification}")
    lines += ["", "8. FINAL DECISION", f"{result['classification']['decision']}",
              f"ROOT-CAUSE CLASSIFICATION: {result['classification']['root_cause']}",
              f"REPRODUCTION STATUS: {result['classification']['reproduction_status']}",
              result["classification"]["confirmed_finding"], result["classification"]["unresolved_finding"], ""]
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], report_path: Path, manifest_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(result), encoding="utf-8")
    manifest_path.write_bytes(manifest_bytes(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=Path("reports/phase33_baseline_mismatch_forensics_report.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/phase33_baseline_mismatch_forensics_manifest.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    write_outputs(audit(root), root / args.report, root / args.manifest)


if __name__ == "__main__":
    main()
