"""Controlled diagnostic reproduction of the six Phase 15 Random Forest folds.

This deliberately invokes only the surviving Phase 15 walk-forward computation.
It never imports or invokes Phase 32 artifact fitting or serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.available_data_audit import maturity_windows, walk_forward
from src.baseline_models import SEED, make_pipeline
from src.schedule_robustness import FEATURES, TARGET


REFERENCE = "reports/phase15_walk_forward_results.csv"
DATASET = "data/features/schedule_modeling.csv"
PHASE34_PATHS = {
    "reports/phase34_controlled_reproduction_report.txt",
    "reports/phase34_controlled_reproduction_manifest.json",
    "src/phase34_controlled_reproduction.py",
    "tests/test_phase34_controlled_reproduction.py",
}
METRICS = ("roc_auc", "pr_auc")
TOLERANCE = 5e-7


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, encoding="utf-8",
                            errors="replace", capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def feature_hash() -> str:
    return hashlib.sha256(json.dumps(FEATURES, separators=(",", ":")).encode("utf-8")).hexdigest()


def historical_reference(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / REFERENCE)
    return frame[frame.variant.eq("FULL_EXISTING_BASELINE")].sort_values("fold").reset_index(drop=True)


def estimator_contract() -> dict[str, Any]:
    pipeline = make_pipeline(FEATURES, "Random Forest")
    estimator = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    return {
        "class": type(estimator).__name__, "n_estimators": estimator.n_estimators,
        "min_samples_leaf": estimator.min_samples_leaf, "max_features": estimator.max_features,
        "random_state": estimator.random_state, "n_jobs": estimator.n_jobs,
        "class_weight": estimator.class_weight, "criterion": estimator.criterion,
        "pipeline_steps": list(pipeline.named_steps),
        "preprocessing": str(preprocess),
    }


def _prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, month in enumerate(["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"], start=1):
        probability = predictions.loc[predictions.prediction_month.eq(month), "probability"].to_numpy(dtype="<f8")
        rows.append({
            "fold": fold, "prediction_length": len(probability),
            "prediction_min": float(probability.min()), "prediction_max": float(probability.max()),
            "prediction_mean": float(probability.mean()),
            "prediction_sha256": hashlib.sha256(probability.tobytes()).hexdigest(),
        })
    return pd.DataFrame(rows)


def controlled_run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the exact Phase 15 implementation once; no artifact is persisted."""
    df = pd.read_csv(root / DATASET, dtype={"project_code": "string", "identity_key": "string"})
    results, predictions, _ = walk_forward(df, {"FULL_EXISTING_BASELINE": FEATURES})
    results = results.sort_values("fold").reset_index(drop=True)
    return results, predictions, _prediction_summary(predictions)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    historical = historical_reference(root)
    first, first_raw_predictions, first_predictions = controlled_run(root)
    second, second_raw_predictions, second_predictions = controlled_run(root)
    current = first.merge(first_predictions, on="fold", validate="one_to_one")
    repeat_metrics_identical = first.equals(second)
    repeat_predictions_identical = first_predictions.equals(second_predictions)
    repeated = first[["fold", *METRICS]].merge(second[["fold", *METRICS]], on="fold", suffixes=("_first", "_second"), validate="one_to_one")
    for metric in METRICS:
        repeated[f"{metric}_delta"] = repeated[f"{metric}_second"] - repeated[f"{metric}_first"]
        repeated[f"{metric}_absolute_delta"] = repeated[f"{metric}_delta"].abs()
    repeat_rows = []
    for fold, month in enumerate(["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"], start=1):
        left = first_raw_predictions.loc[first_raw_predictions.prediction_month.eq(month), "probability"].to_numpy(dtype=float)
        right = second_raw_predictions.loc[second_raw_predictions.prediction_month.eq(month), "probability"].to_numpy(dtype=float)
        repeat_rows.append({
            "fold": fold,
            "first_prediction_sha256": hashlib.sha256(left.astype("<f8").tobytes()).hexdigest(),
            "second_prediction_sha256": hashlib.sha256(right.astype("<f8").tobytes()).hexdigest(),
            "prediction_arrays_exactly_equal": bool(np.array_equal(left, right)),
            "prediction_arrays_allclose_at_1e12": bool(np.allclose(left, right, rtol=0, atol=1e-12)),
            "prediction_max_absolute_difference": float(np.max(np.abs(left - right))),
        })
    repeated = repeated.merge(pd.DataFrame(repeat_rows), on="fold", validate="one_to_one")
    comparison = historical[["fold", "training_period_start", "training_period_end", "evaluation_period",
                             "train_rows", "evaluation_rows", "positive_rows", "event_rate", *METRICS]].merge(
        current[["fold", *METRICS, "prediction_length", "prediction_min", "prediction_max",
                 "prediction_mean", "prediction_sha256"]], on="fold", suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    for metric in METRICS:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_current"] - comparison[f"{metric}_historical"]
        comparison[f"{metric}_absolute_delta"] = comparison[f"{metric}_delta"].abs()
    expected_counts_match = bool(
        (comparison.train_rows == [439, 1061, 1765, 2501, 3232, 4475]).all()
        and (comparison.evaluation_rows == [731, 1243, 1453, 1815, 1701, 1667]).all()
        and (comparison.positive_rows == [106, 565, 660, 925, 717, 665]).all()
    )
    within_tolerance = bool(all(
        comparison[f"{metric}_absolute_delta"].le(TOLERANCE).all() for metric in METRICS
    ))
    first_divergent = comparison.loc[
        comparison[[f"{metric}_absolute_delta" for metric in METRICS]].max(axis=1).gt(TOLERANCE)
    ].head(1)
    first_divergent_fold = None if first_divergent.empty else int(first_divergent.fold.iloc[0])
    numerical_repeatability = bool(
        repeated[[f"{metric}_absolute_delta" for metric in METRICS]].le(1e-12).all().all()
        and repeated.prediction_arrays_allclose_at_1e12.all()
    )
    repeat_classification = (
        "BYTE-IDENTICAL" if repeat_metrics_identical and repeat_predictions_identical else
        "NUMERICALLY IDENTICAL / BYTE-DIFFERENT" if numerical_repeatability else
        "NON-DETERMINISTIC"
    )
    classification = "J" if not within_tolerance else "H"
    root_cause = (
        "J — Multiple reproducible metric differences remain after matching data, feature, label, and split counts; historical dependency lock and prediction artifacts are unavailable, so no individual cause is proven."
        if not within_tolerance else
        "H — The Phase 32 failure is inconsistent with this controlled reproduction; investigate its execution/report-directory inputs without changing the assertion."
    )
    assertion_source = (root / "src/xgboost_benchmark.py").read_text(encoding="utf-8")
    return {
        "schema_version": 1, "phase": 34, "diagnostic_reproduction_only": True,
        "current_state": {
            "commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current"),
            "working_tree_status_excluding_phase34": [
                line for line in _git(root, "status", "--short").splitlines()
                if line[3:].replace("\\", "/") not in PHASE34_PATHS
            ],
            "python_executable": sys.executable, "python_version": sys.version.replace("\n", " "),
            "package_versions": package_versions(), "dataset_path": DATASET,
            "dataset_sha256": sha256_path(root / DATASET), "target": TARGET,
            "feature_count": len(FEATURES), "feature_sha256": feature_hash(), "random_seed": SEED,
        },
        "historical_reference": _records(historical[["fold", *METRICS]]),
        "fold_windows": _records(maturity_windows(pd.read_csv(root / DATASET, dtype={"project_code": "string", "identity_key": "string"}))),
        "estimator_contract": estimator_contract(),
        "comparison": _records(comparison),
        "repeated_execution": {
            "metrics_byte_identical": repeat_metrics_identical,
            "predictions_byte_identical": repeat_predictions_identical,
            "classification": repeat_classification,
            "numerically_identical_at_1e12": numerical_repeatability,
            "fold_comparison": _records(repeated),
        },
        "integrity": {
            "expected_counts_match": expected_counts_match,
            "historical_dependency_lock": "NOT AVAILABLE",
            "historical_predictions": "NOT AVAILABLE",
            "phase15_assertion_unchanged": "if not np.allclose(recomputed[metric], stored[metric], atol=5e-7, rtol=0):" in assertion_source,
            "phase32_impact": "No impact: the Phase 15 assertion executes in experiment() before the artifact-path branch and before fit_final_temporal_fold().",
        },
        "decision": {
            "reproduction_status": "PHASE 15 REPRODUCTION: PASS" if within_tolerance else "PHASE 15 REPRODUCTION: FAIL",
            "root_cause": root_cause, "root_cause_code": classification,
            "first_divergent_fold": first_divergent_fold,
            "largest_roc_absolute_difference": float(comparison.roc_auc_absolute_delta.max()),
            "largest_pr_absolute_difference": float(comparison.pr_auc_absolute_delta.max()),
            "all_folds_differ": bool(comparison[["roc_auc_absolute_delta", "pr_auc_absolute_delta"]].max(axis=1).gt(TOLERANCE).all()),
            "next_action": (
                "Preserve the assertion and historical values. Obtain the historical dependency lock or prediction artifacts before attributing the remaining difference."
                if not within_tolerance else
                "Inspect the exact Phase 32 invocation and copied reference file; do not alter the Phase 15 assertion."
            ),
        },
    }


def manifest_bytes(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")


def report_text(result: dict[str, Any]) -> str:
    state, decision = result["current_state"], result["decision"]
    lines = [
        "PHASE 34 — CONTROLLED PHASE 15 WALK-FORWARD REPRODUCTION", "=" * 72, "",
        "SCOPE", "The existing Phase 15 Random Forest walk-forward computation was run twice. No XGBoost, artifact serialization, deployment model, calibration, tuning, August data, or Phase 15 assertion modification was used.", "",
        "ENVIRONMENT", f"Commit/branch: {state['commit']} / {state['branch']}",
        f"Python: {state['python_executable']} | {state['python_version']}",
        f"Packages: {state['package_versions']}", f"Dataset SHA-256: {state['dataset_sha256']}",
        f"Target/features/hash/seed: {state['target']} / {state['feature_count']} / {state['feature_sha256']} / {state['random_seed']}", "",
        "HISTORICAL DEPENDENCY LOCK: NOT AVAILABLE", "HISTORICAL PREDICTIONS: NOT AVAILABLE", "",
        "RANDOM FOREST CONTRACT", json.dumps(result["estimator_contract"], sort_keys=True), "",
        "FOLD COMPARISON", "fold | train/eval/positives | historical ROC/PR | current ROC/PR | ROC delta | PR delta | prediction SHA-256",
    ]
    for row in result["comparison"]:
        lines.append(
            f"{row['fold']} | {row['train_rows']}/{row['evaluation_rows']}/{row['positive_rows']} | "
            f"{row['roc_auc_historical']:.15f}/{row['pr_auc_historical']:.15f} | "
            f"{row['roc_auc_current']:.15f}/{row['pr_auc_current']:.15f} | "
            f"{row['roc_auc_delta']:+.15f} | {row['pr_auc_delta']:+.15f} | {row['prediction_sha256']}"
        )
    lines += [
        "", "INTEGRITY", f"Expected data/split/label counts match: {result['integrity']['expected_counts_match']}",
        f"Phase 15 assertion unchanged: {result['integrity']['phase15_assertion_unchanged']}",
        result["integrity"]["phase32_impact"], "",
        "REPEATED EXECUTION", json.dumps(result["repeated_execution"], sort_keys=True), "",
        "FINAL DECISION", decision["reproduction_status"], f"ROOT CAUSE: {decision['root_cause_code']}",
        decision["root_cause"], f"First divergent fold: {decision['first_divergent_fold']}",
        f"Largest ROC/PR absolute difference: {decision['largest_roc_absolute_difference']:.15f} / {decision['largest_pr_absolute_difference']:.15f}",
        f"All folds differ: {decision['all_folds_differ']}", f"NEXT ACTION: {decision['next_action']}", "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], report: Path, manifest: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(result), encoding="utf-8")
    manifest.write_bytes(manifest_bytes(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=Path("reports/phase34_controlled_reproduction_report.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/phase34_controlled_reproduction_manifest.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    write_outputs(audit(root), root / args.report, root / args.manifest)


if __name__ == "__main__":
    main()
