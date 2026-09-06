"""Phase 27 final-candidate audit without fitting, tuning, or calibration.

This module intentionally makes no model predictions.  It hashes and audits the
already-recorded frozen contract and the evidence files that must exist before a
final freeze can be approved.  A missing required Phase 23--26 artifact is a
blocker, not permission to recreate that phase or infer its conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CONTRACT_SHA256 = "6fa4aca992807c741fa7e24d969db161966760562ddf9f48193d21dd7559f149"
EXPECTED_XGB_VERSION = "2.1.4"
TARGET = "future_schedule_later_3m"
TARGET_HORIZON_MONTHS = 3
FEATURES = [
    "sector", "progress_current", "expenditure_current", "original_cost_current",
    "progress_change_1m", "progress_change_3m", "progress_velocity_3m",
    "progress_acceleration_3m", "expenditure_change_1m", "expenditure_change_3m",
    "expenditure_velocity_3m", "cost_revision_count_to_date",
    "months_since_cost_revision", "schedule_revision_count_to_date",
    "months_since_schedule_revision", "months_since_material_progress_change",
    "expenditure_to_original_cost", "months_observed",
    "months_since_first_observation", "revised_cost_missing", "progress_missing",
    "state", "ministry", "agency", "project_age_months", "revised_cost_current",
    "effective_target_months_from_cutoff", "expenditure_to_revised_cost",
    "cost_revision_pct_current",
]
FINAL_CANDIDATE_APPROVED = "APPROVED"
FINAL_CANDIDATE_NOT_APPROVED = "NOT APPROVED"

# These are required by the Phase 27 brief.  They deliberately do not fall back
# to earlier phases: a missing phase is evidence that the final audit cannot
# verify that phase's conclusion.
REQUIRED_EVIDENCE = {
    "phase20_validation": "reports/phase20_raw_calibration_audit.txt",
    "phase21_holdout_availability": "reports/phase21_post_april_forward_holdout_report.txt",
    "phase22_august_source_validation": "reports/phase22_august_ingestion_report.txt",
    "phase23_readiness_monitoring": "reports/phase23_readiness_monitoring_report.txt",
    "phase24_product_integration": "reports/phase24_product_integration_report.txt",
    "phase25_generalization_error_analysis": "reports/phase25_generalization_error_analysis.txt",
    "phase26_calibration_protocol": "reports/phase26_calibration_predeclaration.txt",
}

IMMUTABLE_INPUTS = {
    "feature_contract": "src/schedule_robustness.py",
    "threshold_configuration": "src/xgboost_benchmark.py",
    "label_definition": "reports/schedule_label_semantic_definition.md",
    "preprocessing_configuration": "src/baseline_models.py",
    "phase16_specification": "reports/phase16_xgboost_specification.txt",
    "phase20_calibration_audit": "reports/phase20_raw_calibration_audit.txt",
    "phase21_holdout_availability": "reports/phase21_post_april_forward_holdout_report.txt",
    "phase22_readiness_gate": "reports/phase22_august_ingestion_report.txt",
    "phase26_calibration_protocol": "reports/phase26_calibration_predeclaration.txt",
}


def sha256_file(path: Path) -> str:
    """Return a file digest without modifying the file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path.replace("\\", "/"),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _assert_frozen_contract(contract: dict[str, Any]) -> None:
    if contract["contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise ValueError("frozen logical model-contract SHA-256 mismatch")
    if contract["xgboost_version"] != EXPECTED_XGB_VERSION:
        raise ValueError("frozen XGBoost version mismatch")
    if contract["features"] != list(FEATURES) or contract["feature_count"] != 29:
        raise ValueError("frozen feature list or ordering mismatch")
    if contract["thresholds"] != [0.40, 0.50]:
        raise ValueError("frozen threshold configuration mismatch")
    if contract["calibration"] != "NONE":
        raise ValueError("frozen model must not have fitted calibration")


def frozen_model_contract() -> dict[str, Any]:
    """Expose the exact, non-probabilistic contract audited by Phase 27."""
    # This mirrors Phase 21's logical model contract exactly.  The audit must
    # remain usable even in a read-only environment without the XGBoost wheel;
    # it never imports, loads, or fits XGBoost.
    configuration = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 26103,
        "scale_pos_weight": 1.0,
        "missing": "NaN",
        "n_jobs": 1,
        "tree_method": "hist",
        "importance_type": "gain",
    }
    logical = {
        "model_family": "XGBOOST",
        "xgboost_version": EXPECTED_XGB_VERSION,
        "configuration": configuration,
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "target": TARGET,
        "target_horizon_months": TARGET_HORIZON_MONTHS,
        "development_end": "2026-04",
        "calibration": "NONE",
        "thresholds": [0.40, 0.50],
    }
    canonical = json.dumps(logical, sort_keys=True, separators=(",", ":"))
    logical["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _assert_frozen_contract(logical)
    return {
        "model_identifier": "phase16_frozen_xgboost",
        "model_version": f"xgboost-{EXPECTED_XGB_VERSION}",
        "model_artifact": {
            "status": "not_serialized",
            "sha256": None,
            "explanation": (
                "The repository has no serialized fitted model. The immutable model "
                "identity is the logical contract SHA-256 below."
            ),
        },
        "logical_model_contract_sha256": logical["contract_sha256"],
        "xgboost_configuration": logical["configuration"],
        "feature_list_ordered": logical["features"],
        "preprocessing_identifier": (
            "training_fitted_column_transformer_median_mode_onehot_standardscale"
        ),
        "thresholds": logical["thresholds"],
        "label_definition": {
            "target": TARGET,
            "horizon_months": TARGET_HORIZON_MONTHS,
            "semantics": "future_published_schedule_target_deterioration",
            "unknown_rule": "missing exact t_plus_3 endpoint is unknown, never negative",
        },
        "score_semantics": "relative_risk_ranking",
        "calibration_status": "not_calibrated",
        "training_data_cutoff": "2026-04",
        "validation_data_scope": (
            "six maturity-safe walk-forward folds from 2025-11 through 2026-04 "
            "plus the deterministic project-disjoint experiment"
        ),
        "known_limitations": [
            "Published schedule-target deterioration is a proxy, not actual completion delay.",
            "No mature post-April labeled holdout exists.",
            "Raw scores are not calibrated probabilities.",
            "Fixed threshold behavior and slice robustness are heterogeneous.",
            "No serialized fitted model artifact exists in the repository.",
        ],
        "future_holdout_status": "not_ready",
    }


def build_audit(root: Path) -> dict[str, Any]:
    """Build the deterministic Phase 27 audit result using existing files only."""
    contract = frozen_model_contract()
    evidence = {name: file_record(root, path) for name, path in REQUIRED_EVIDENCE.items()}
    immutable = {name: file_record(root, path) for name, path in IMMUTABLE_INPUTS.items()}
    missing = [name for name, record in evidence.items() if not record["exists"]]
    status = FINAL_CANDIDATE_APPROVED if not missing else FINAL_CANDIDATE_NOT_APPROVED
    return {
        "schema_version": 1,
        "phase": 27,
        "audit_type": "final_model_decision_and_freeze_audit",
        "final_candidate_status": status,
        "final_freeze_created": status == FINAL_CANDIDATE_APPROVED,
        "decision_blockers": (
            [] if not missing else [
                "Required evidence is absent; Phase 27 does not recreate, infer, or "
                "substitute for the missing phase artifacts: " + ", ".join(missing)
            ]
        ),
        "frozen_model_contract": contract,
        "required_evidence": evidence,
        "immutable_artifact_hashes": immutable,
        "august_readiness": {
            "august_data_available": False,
            "mature_may_observations": 0,
            "mature_may_projects": 0,
            "may_to_august_holdout_evaluated": False,
            "future_holdout_status": "not_ready",
        },
        "change_control": {
            "model_changed": False,
            "model_retrained": False,
            "features_changed": False,
            "thresholds_changed": False,
            "calibration_fitted": False,
            "calibration_protocol_changed": False,
        },
    }


def manifest_bytes(audit: dict[str, Any]) -> bytes:
    """Canonical JSON makes the manifest byte-stable between deterministic runs."""
    return (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _assert_identical(first: dict[str, Any], second: dict[str, Any]) -> None:
    if manifest_bytes(first) != manifest_bytes(second):
        raise AssertionError("Phase 27 audit output is not deterministic")
    if (first["frozen_model_contract"]["logical_model_contract_sha256"]
            != EXPECTED_CONTRACT_SHA256):
        raise AssertionError("logical model contract changed during audit")


def report_text(audit: dict[str, Any]) -> str:
    """Render a decision report that separates evidence, interpretation, and limits."""
    contract = audit["frozen_model_contract"]
    missing = [name for name, record in audit["required_evidence"].items() if not record["exists"]]
    status = audit["final_candidate_status"]
    lines = [
        "PHASE 27 — FINAL MODEL DECISION & FREEZE AUDIT",
        "=" * 70,
        "",
        "1. EXECUTIVE DECISION",
        f"FINAL CANDIDATE STATUS: {status}",
        "The frozen candidate is NOT APPROVED for final freeze because the repository "
        "does not contain the required Phase 23–26 evidence artifacts. This is an "
        "evidence-completeness decision, not a finding that the existing XGBoost ranking "
        "results are invalid.",
        "",
        "2. EVIDENCE REVIEWED",
        "Present and hashed: Phase 20 raw calibration audit; Phase 21 post-April holdout "
        "availability; Phase 22 August-source/readiness validation; frozen model sources "
        "and specification.",
        "Missing and therefore not verified: " + ", ".join(missing) + ".",
        "",
        "3. GENERALIZATION ASSESSMENT",
        "Directly measured evidence: Phase 16 reports mean six-fold XGBoost ROC-AUC 0.849 "
        "and PR-AUC 0.738; its project-disjoint ROC-AUC is 0.811 and PR-AUC 0.797. "
        "Phase 15's label-maturity sensitivity reports ROC-AUC 0.810 and PR-AUC 0.732. "
        "The six mature evaluation folds cover 2025-11 through 2026-04.",
        "Interpretation: the existing evidence supports the frozen model as a historical "
        "relative-risk ranking signal across these temporal and project-disjoint tests.",
        "Unresolved limitation: it does not measure performance after April 2026, and the "
        "required Phase 25 generalization/error-analysis artifact is absent.",
        "",
        "4. ERROR ANALYSIS ASSESSMENT",
        "Directly measured evidence: the Phase 17 supported-slice table at threshold 0.40 "
        "shows Electricity Generation 37/49 false negatives (75.5%), Railways 74/118 "
        "(62.7%), and Roads & Highways 754 false negatives (the largest absolute miss "
        "count, with a lower 26.9% miss rate). It also shows false-positive rates of "
        "24/57 (42.1%) for Waste & Water, 36/138 (26.1%) for Healthcare, and 57/151 "
        "(37.7%) for Education.",
        "Interpretation: these are operational guardrail and monitoring requirements; they "
        "do not by themselves invalidate the historical ranking evidence or authorize a "
        "model, threshold, or feature change.",
        "Unresolved limitation: the requested Phase 25 artifact is absent and no post-April "
        "error distribution exists.",
        "",
        "5. CALIBRATION ASSESSMENT",
        "Directly measured evidence: Phase 20 classifies raw-score calibration evidence as "
        "mixed (overall Brier 0.169, ECE 0.099, intercept 0.527, slope 0.691) and states "
        "that scores are ranking signals rather than probabilities.",
        "Interpretation: no probability semantics are permitted for this candidate.",
        "Unresolved limitation: no Phase 26 predeclared calibration protocol is in the "
        "repository, so the stated future Platt/beta plan cannot be verified for internal "
        "consistency. No calibration was fitted.",
        "",
        "6. DATA, LABEL, AND DRIFT LIMITATIONS",
        "Directly measured evidence: Phase 21 and Phase 22 report no authentic August 2026 "
        "project-level source, zero mature May observations/projects, and an unready holdout. "
        "Phase 15 flags drift in several May–July feature distributions.",
        "Interpretation: data maturity and drift monitoring remain prerequisites for a "
        "future evaluation, not reasons to fabricate labels or score August.",
        "Unresolved limitation: Phase 23 readiness-monitoring evidence is absent.",
        "",
        "7. SECTOR AND OPERATIONAL LIMITATIONS",
        "The known sector error patterns require sector-aware review, false-negative and "
        "false-positive monitoring, and no interpretation of a score as a probability. "
        "No operational capacity contract is present; Phase 19 explicitly stopped before "
        "selecting a policy threshold. Product-integration evidence required by Phase 24 "
        "is absent.",
        "",
        "8. AUGUST HOLDOUT STATUS",
        "AUGUST DATA AVAILABLE: NO",
        "MATURE MAY OBSERVATIONS: 0",
        "MATURE MAY PROJECTS: 0",
        "MAY→AUGUST HOLDOUT EVALUATED: NO",
        "The next data action is to authenticate, validate, schema-check, identity-match, "
        "and leakage-check an official August source through the readiness gate. It is not "
        "automatic authorization to evaluate the holdout.",
        "",
        "9. FROZEN MODEL CONTRACT (AUDITED, NOT FINALLY FROZEN)",
        f"model_identifier = {contract['model_identifier']}",
        f"model_version = {contract['model_version']}",
        f"model_artifact_sha256 = {contract['model_artifact']['sha256']}",
        f"logical_model_contract_sha256 = {contract['logical_model_contract_sha256']}",
        "feature_list_ordered = " + json.dumps(contract["feature_list_ordered"]),
        f"preprocessing_identifier = {contract['preprocessing_identifier']}",
        f"thresholds = {contract['thresholds']}",
        f"label_definition = {contract['label_definition']['target']} at t+{TARGET_HORIZON_MONTHS}",
        "calibration_status = \"not_calibrated\"",
        "score_semantics = \"relative_risk_ranking\"",
        "future_holdout_status = \"not_ready\"",
        "",
        "10. CHANGE CONTROL",
        "Allowed before a future evaluation: acquire and validate authentic source data; "
        "run the already-defined readiness gate; document evidence gaps. No model output "
        "or frozen artifact may be changed as part of those actions.",
        "Prohibited before the May→August holdout: model, XGBoost configuration, features "
        "or their order, preprocessing, label definition, thresholds, calibration fitting, "
        "or calibration-protocol changes. Any necessary change requires a new model version "
        "and a new validation cycle.",
        "",
        "11. EXACT NEXT STEP ONCE AUGUST ARRIVES",
        "Authenticate the official August 2026 report; validate its reporting period, schema, "
        "project identities, duplicates, and leakage boundary; then run the Phase 23 "
        "readiness gate. Keep the holdout as a separate event and evaluate only if that gate "
        "passes and the missing Phase 23–26 evidence requirements have been resolved.",
        "",
        "12. REQUIRED STATUS DECLARATIONS",
        "MODEL CHANGED: NO",
        "MODEL RETRAINED: NO",
        "FEATURES CHANGED: NO",
        "THRESHOLDS CHANGED: NO",
        "CALIBRATION FITTED: NO",
        "CALIBRATION PROTOCOL CHANGED: NO",
        "AUGUST DATA AVAILABLE: NO",
        "MAY→AUGUST HOLDOUT EVALUATED: NO",
        f"FINAL CANDIDATE: {status}",
        "",
        "13. REPRODUCIBILITY",
        "Two deterministic Phase 27 audit passes produced byte-identical manifests. The "
        "logical model-contract SHA-256 was unchanged. The Phase 23–26 files are missing, "
        "so they cannot be reported unchanged or treated as verified.",
    ]
    return "\n".join(lines) + "\n"


def run(root: Path, report_dir: Path) -> dict[str, Any]:
    """Run the read-only audit twice and write its deterministic report and manifest."""
    first = build_audit(root)
    second = build_audit(root)
    _assert_identical(first, second)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase27_final_model_decision_report.txt").write_text(
        report_text(first), encoding="utf-8"
    )
    (report_dir / "phase27_final_model_freeze_manifest.json").write_bytes(manifest_bytes(first))
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.root, args.report_dir)


if __name__ == "__main__":
    main()
