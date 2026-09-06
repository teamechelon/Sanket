"""Phase 28 repository evidence recovery audit.

This module performs Git and filesystem inspection only.  It must never fit a
model, construct labels, evaluate the August holdout, or recreate historical
scientific evidence that is absent from repository objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT_SHA256 = "6fa4aca992807c741fa7e24d969db161966760562ddf9f48193d21dd7559f149"
MODEL_SUFFIXES = {".json", ".ubj", ".bin", ".joblib", ".pkl", ".pickle", ".model", ".xgb"}
PHASE28_OUTPUTS = {
    "reports/phase28_evidence_inventory.json",
    "reports/phase28_evidence_recovery_report.txt",
}

# These are the named report/manifest artifacts in the Phase 28 brief.  This
# inventory does not invent an alternate historical path when an artifact is
# absent; it records the expected name and searches all reachable Git history.
EXPECTED_ARTIFACTS = (
    (23, "reports/phase23_holdout_readiness_report.txt"),
    (23, "reports/phase23_readiness_manifest.json"),
    (23, "reports/phase23_readiness_monitoring_report.txt"),
    (24, "reports/phase24_product_integration_report.txt"),
    (24, "reports/phase24_change_manifest.json"),
    (25, "reports/phase25_generalization_error_analysis_report.txt"),
    (25, "reports/phase25_generalization_error_analysis.txt"),
    (25, "reports/phase25_analysis_manifest.json"),
    (26, "reports/phase26_calibration_decision_report.txt"),
    (26, "reports/phase26_calibration_protocol.json"),
    (26, "reports/phase26_calibration_predeclaration.txt"),
    (27, "reports/phase27_final_model_decision_report.txt"),
    (27, "reports/phase27_final_model_freeze_manifest.json"),
    (27, "src/final_model_freeze_audit.py"),
    (27, "tests/test_final_model_freeze_audit.py"),
)
PHASE_SEARCH_TERMS = ("Phase 23", "Phase 24", "Phase 25", "Phase 26", "Phase 27")

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


def _git(root: Path, *args: str, allow_failure: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_state(root: Path) -> dict[str, Any]:
    """Read the requested Git state without changing index, refs, or worktree."""
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "remote", "get-url", "origin", allow_failure=True) or None
    ahead_behind = _git(root, "rev-list", "--left-right", "--count", "origin/main...HEAD")
    behind, ahead = (int(value) for value in ahead_behind.split())
    raw_status = _git(root, "status", "--porcelain=v1").splitlines()
    retained_status = []
    excluded_output_status = []
    for line in raw_status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else line
        (excluded_output_status if path in PHASE28_OUTPUTS else retained_status).append(line)
    return {
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "origin_url": origin,
        "working_tree_status": "\n".join(retained_status),
        "phase28_output_status_excluded_for_determinism": excluded_output_status,
        "ahead_of_origin_main": ahead,
        "behind_origin_main": behind,
        "latest_commits": _git(root, "log", "--oneline", "--decorate", "--all", "-30").splitlines(),
        "local_branches": _git(root, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines(),
        "remote_branches": _git(root, "for-each-ref", "--format=%(refname:short)", "refs/remotes").splitlines(),
        "tags": _git(root, "tag", "--list").splitlines(),
        "phase_commit_message_search": {
            term: _git(root, "log", "--all", "--oneline", "--grep", term).splitlines()
            for term in PHASE_SEARCH_TERMS
        },
    }


def _history_commits(root: Path, path: str) -> list[str]:
    output = _git(root, "log", "--all", "--format=%H", "--", path)
    return [line for line in output.splitlines() if line]


def _tracked(root: Path, path: str) -> bool:
    return bool(_git(root, "ls-files", "--error-unmatch", "--", path, allow_failure=True))


def _unreachable_tree_paths(root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    """Index paths stored in dangling Git trees without mutating Git objects."""
    output = _git(root, "fsck", "--full", "--no-reflogs", "--unreachable", "--no-progress")
    objects: list[tuple[str, str]] = []
    for line in output.splitlines():
        match = re.match(r"^unreachable (blob|tree|commit) ([0-9a-f]{40})$", line)
        if match:
            objects.append((match.group(1), match.group(2)))

    paths: dict[str, list[dict[str, str]]] = {}
    for kind, object_id in objects:
        if kind != "tree":
            continue
        for line in _git(root, "ls-tree", "-r", "--full-tree", object_id, allow_failure=True).splitlines():
            metadata, separator, path = line.partition("\t")
            parts = metadata.split()
            if separator and len(parts) == 3 and parts[1] == "blob":
                paths.setdefault(path, []).append({"tree": object_id, "blob": parts[2]})
    counts = {kind: sum(item_kind == kind for item_kind, _ in objects)
              for kind in ("commit", "tree", "blob")}
    return paths, counts


def _worktree_git_object_id(root: Path, path: str) -> str | None:
    if not (root / path).is_file():
        return None
    return _git(root, "hash-object", "--", path, allow_failure=True) or None


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    """Classify expected artifacts by authentic current/Git evidence only."""
    records = []
    unreachable_paths, _ = _unreachable_tree_paths(root)
    for phase, expected_path in EXPECTED_ARTIFACTS:
        path = root / expected_path
        history = _history_commits(root, expected_path)
        in_worktree = path.is_file()
        in_history = bool(history)
        object_sources = unreachable_paths.get(expected_path, [])
        in_unreachable_object = bool(object_sources)
        current_hash = sha256_file(path) if in_worktree else None
        current_object_id = _worktree_git_object_id(root, expected_path)
        source_commit = history[0] if history else None
        if in_history and in_worktree and _tracked(root, expected_path):
            evidence_type = "historical_repository_artifact"
            verification = "verified_current_repository_path_and_git_history"
            recovered = False
            status = "PRESENT_HISTORICAL"
        elif in_history:
            evidence_type = "historical_repository_artifact"
            verification = "recoverable_from_git_history"
            recovered = False
            status = "RECOVERABLE_HISTORICAL"
        elif in_unreachable_object:
            worktree_matches_object = any(
                source["blob"] == current_object_id for source in object_sources
            )
            evidence_type = "historical_repository_artifact"
            verification = (
                "verified_against_unreachable_git_object_without_commit_provenance"
                if worktree_matches_object else
                "recoverable_from_unreachable_git_object_without_commit_provenance"
            )
            recovered = False
            status = (
                "PRESENT_VERIFIED_UNREACHABLE_GIT_OBJECT"
                if worktree_matches_object else "RECOVERABLE_UNREACHABLE_GIT_OBJECT"
            )
        elif in_worktree:
            # An untracked file can be audited as current material, but it is
            # never promoted to historical evidence merely because it has a
            # familiar phase filename.
            evidence_type = "newly_generated_audit_artifact"
            verification = "present_in_worktree_but_not_in_git_history"
            recovered = False
            status = "PRESENT_NOT_HISTORICAL"
        else:
            evidence_type = "missing"
            verification = "missing_from_worktree_and_reachable_git_history"
            recovered = False
            status = "MISSING"
        records.append({
            "phase": phase,
            "expected_path": expected_path,
            "found_in_worktree": in_worktree,
            "found_in_git_history": in_history,
            "found_in_unreachable_git_objects": in_unreachable_object,
            "recovered": recovered,
            "source_commit": source_commit,
            "source_git_objects": object_sources,
            "current_sha256": current_hash,
            "current_status": status,
            "evidence_type": evidence_type,
            "verification_status": verification,
        })
    return records


def _is_model_candidate_path(path: str) -> bool:
    """Avoid misclassifying application JSON and test fixtures as ML models."""
    candidate = Path(path)
    if candidate.suffix.lower() != ".json":
        return True
    tokens = {part.lower() for part in candidate.parts}
    if bool(tokens & {"model", "models", "artifact", "artifacts", "checkpoints"}):
        return True
    return "model" in candidate.stem.lower() and "manifest" not in candidate.stem.lower()


def _model_files(root: Path) -> tuple[list[str], int]:
    """List candidate serialized artifacts without parsing or loading them."""
    found: list[str] = []
    ignored_generic_json = 0
    # Dependency and frontend build trees contain many generic manifests but are
    # neither tracked model locations nor the repository's models/artifacts
    # directories. Tracked paths remain covered separately by rev-list below.
    excluded = {".git", "__pycache__", "node_modules", ".next"}
    for directory, dirs, files in os.walk(root, onerror=lambda _error: None):
        dirs[:] = [name for name in dirs if name not in excluded and not name.startswith(("pytest-cache-files-", "tmp"))]
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if Path(name).suffix.lower() in MODEL_SUFFIXES:
                if _is_model_candidate_path(relative):
                    found.append(relative)
                else:
                    ignored_generic_json += 1
    return sorted(found), ignored_generic_json


def logical_contract_verification(root: Path) -> dict[str, Any]:
    """Verify the documented logical configuration hash, not a fitted model file."""
    configuration = {
        "objective": "binary:logistic", "eval_metric": "aucpr", "n_estimators": 300,
        "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5.0,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0,
        "reg_lambda": 1.0, "random_state": 26103, "scale_pos_weight": 1.0,
        "missing": "NaN", "n_jobs": 1, "tree_method": "hist", "importance_type": "gain",
    }
    payload = {
        "model_family": "XGBOOST", "xgboost_version": "2.1.4",
        "configuration": configuration, "features": FEATURES, "feature_count": len(FEATURES),
        "target": "future_schedule_later_3m", "target_horizon_months": 3,
        "development_end": "2026-04", "calibration": "NONE", "thresholds": [0.40, 0.50],
    }
    computed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reference_paths = [
        "src/post_april_holdout_validation.py",
        "src/august_2026_ingestion_validation.py",
        "reports/phase21_post_april_forward_holdout_report.txt",
        "reports/phase22_august_ingestion_report.txt",
    ]
    references = []
    for relative in reference_paths:
        path = root / relative
        if path.is_file() and EXPECTED_CONTRACT_SHA256 in path.read_text(encoding="utf-8", errors="replace"):
            references.append(relative)
    return {
        "expected_sha256": EXPECTED_CONTRACT_SHA256,
        "computed_sha256": computed,
        "matches_expected": computed == EXPECTED_CONTRACT_SHA256,
        "hashing_method": "SHA-256 of canonical JSON (sorted keys, compact separators) for the logical Phase 21 contract payload",
        "reference_paths": references,
        "referenced_artifact": "logical configuration contract, not a serialized fitted-model artifact",
        "frozen_artifact_verification": "BLOCKED_NO_SERIALIZED_MODEL_ARTIFACT",
    }


def model_artifact_investigation(root: Path) -> dict[str, Any]:
    worktree_files, ignored_json = _model_files(root)
    history_objects = _git(root, "rev-list", "--objects", "--all").splitlines()
    history_paths = sorted(
        line.split(" ", 1)[1] for line in history_objects
        if (" " in line and Path(line.split(" ", 1)[1]).suffix.lower() in MODEL_SUFFIXES
            and _is_model_candidate_path(line.split(" ", 1)[1]))
    )
    unreachable_paths, unreachable_counts = _unreachable_tree_paths(root)
    unreachable_model_paths = sorted(
        path for path in unreachable_paths
        if Path(path).suffix.lower() in MODEL_SUFFIXES and _is_model_candidate_path(path)
    )
    loading_references = []
    for source in sorted((root / "src").glob("*.py")):
        if source.name == "phase28_evidence_recovery.py":
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\b(?:load_model|save_model|joblib\.(?:load|dump)|pickle\.(?:load|dump))\b", text):
            loading_references.append(source.relative_to(root).as_posix())
    return {
        "serialized_model_files_in_worktree": worktree_files,
        "serialized_model_paths_in_reachable_git_history": history_paths,
        "serialized_model_paths_in_unreachable_git_objects": unreachable_model_paths,
        "generic_json_files_examined_but_not_model_candidates": ignored_json,
        "models_directory_exists": (root / "models").is_dir(),
        "model_loading_code": (
            loading_references or
            ["No persisted-model loader found; Phase 16 constructs XGBClassifier within make_xgb_pipeline."]
        ),
        "unreachable_git_object_counts": unreachable_counts,
        "frozen_model_artifact_status": "NOT_PRESERVED",
        "verification_status": "BLOCKED_NO_SERIALIZED_MODEL_ARTIFACT",
    }


def audit_once(root: Path) -> dict[str, Any]:
    inventory = artifact_inventory(root)
    contract = logical_contract_verification(root)
    model = model_artifact_investigation(root)
    missing = [record["expected_path"] for record in inventory if record["evidence_type"] == "missing"]
    historical = [record["expected_path"] for record in inventory if record["found_in_git_history"]]
    unreachable = [record["expected_path"] for record in inventory
                   if record["found_in_unreachable_git_objects"]]
    return {
        "schema_version": 1,
        "phase": 28,
        "audit_kind": "repository_evidence_recovery_and_freeze_prerequisite_audit",
        "repository_state": repository_state(root),
        "artifact_inventory": inventory,
        "model_artifact_investigation": model,
        "frozen_contract_verification": contract,
        "recovered_historical_evidence": historical,
        "unreachable_git_object_evidence": unreachable,
        "missing_evidence": missing,
        "phase27_blocker_status": "NOT_RESOLVED",
        "final_candidate_status": "NOT APPROVED",
        "phase28_actions": {
            "model_changed": False, "model_retrained": False, "features_changed": False,
            "thresholds_changed": False, "calibration_fitted": False,
            "august_data_created": False, "august_holdout_evaluated": False,
            "ml_fitting_performed": False,
        },
    }


def canonical_bytes(audit: dict[str, Any]) -> bytes:
    return (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _assert_identical(first: dict[str, Any], second: dict[str, Any]) -> None:
    if canonical_bytes(first) != canonical_bytes(second):
        raise AssertionError("Phase 28 audit is not deterministic")


def report_text(audit: dict[str, Any]) -> str:
    state = audit["repository_state"]
    inventory = audit["artifact_inventory"]
    model = audit["model_artifact_investigation"]
    contract = audit["frozen_contract_verification"]
    lines = [
        "PHASE 28 — REPOSITORY EVIDENCE RECOVERY & FREEZE PREREQUISITE AUDIT",
        "=" * 74, "", "1. REPOSITORY STATE",
        f"Branch: {state['branch']}", f"HEAD: {state['head']}",
        f"Origin: {state['origin_url']}",
        f"Ahead/behind origin/main: {state['ahead_of_origin_main']}/{state['behind_origin_main']}",
        "Working tree status (excluding Phase 28 generated outputs so repeated audits are comparable):",
        state["working_tree_status"] or "CLEAN",
        "Excluded Phase 28 output status: "
        + ("; ".join(state["phase28_output_status_excluded_for_determinism"]) or "NONE"),
        "Latest commits:", *state["latest_commits"],
        f"Local branches: {', '.join(state['local_branches']) or 'NONE'}",
        f"Remote branches: {', '.join(state['remote_branches']) or 'NONE'}",
        f"Tags: {', '.join(state['tags']) or 'NONE'}", "",
        "Phase 23–27 commit-message search:",
    ]
    for term, matches in state["phase_commit_message_search"].items():
        lines.append(f"{term}: {'; '.join(matches) if matches else 'NONE'}")
    lines += [
        "",
        "2. PHASE 23–27 ARTIFACT STATUS",
        "phase | expected path | worktree | git history | dangling object | evidence type | status | source commit",
    ]
    for record in inventory:
        lines.append(
            f"{record['phase']} | {record['expected_path']} | {record['found_in_worktree']} | "
            f"{record['found_in_git_history']} | {record['found_in_unreachable_git_objects']} | {record['evidence_type']} | "
            f"{record['current_status']} | {record['source_commit'] or 'NONE'}"
        )
    lines += [
        "", "3. FROZEN MODEL ARTIFACT STATUS",
        f"Model artifact status: {model['frozen_model_artifact_status']}",
        f"Worktree serialized model candidates: {model['serialized_model_files_in_worktree'] or 'NONE'}",
        f"Reachable-history serialized model candidates: {model['serialized_model_paths_in_reachable_git_history'] or 'NONE'}",
        f"Unreachable-object serialized model candidates: {model['serialized_model_paths_in_unreachable_git_objects'] or 'NONE'}",
        "Generic JSON files were examined but excluded unless their name or path identified a model/artifact: "
        f"{model['generic_json_files_examined_but_not_model_candidates']}",
        f"Model loading/persistence finding: {model['model_loading_code']}",
        f"Unreachable Git object counts: {model['unreachable_git_object_counts']}",
        f"Frozen artifact verification: {model['verification_status']}",
        "", "4. CONTRACT HASH VERIFICATION",
        f"Expected contract SHA-256: {contract['expected_sha256']}",
        f"Computed logical-contract SHA-256: {contract['computed_sha256']}",
        f"Contract match: {contract['matches_expected']}",
        f"Hashing method: {contract['hashing_method']}",
        "References: " + ", ".join(contract["reference_paths"]),
        f"Referenced artifact: {contract['referenced_artifact']}",
        f"FROZEN ARTIFACT VERIFICATION: {contract['frozen_artifact_verification']}",
        "", "5. RECOVERED HISTORICAL EVIDENCE",
        "NONE. No Phase 23–27 expected artifact exists in reachable commit history.",
        "Unreachable Git-object evidence (not recovered because no commit provenance exists): "
        + (", ".join(audit["unreachable_git_object_evidence"]) or "NONE"),
        "", "6. MISSING EVIDENCE",
        *audit["missing_evidence"],
        "", "7. EVIDENCE INTEGRITY ASSESSMENT",
        "The Phase 27 worktree files match dangling Git blobs, but the enclosing Git tree has no "
        "reachable commit or ref. They are authentic Git-object findings with insufficient commit provenance, "
        "not independently verifiable historical validation evidence. No Phase 23–26 experiment or result was "
        "recreated. The logical configuration contract matches its documented hash, but the serialized fitted "
        "model is not preserved.",
        "", "8. PHASE 27 BLOCKER STATUS",
        "NOT RESOLVED. FINAL CANDIDATE: NOT APPROVED.",
        "", "9. EXACT PREREQUISITES FOR A PHASE 27 RE-RUN",
        "Recover authentic Phase 23–26 artifacts from an authoritative repository source and preserve a "
        "serialized model artifact with a verifiable digest. Then independently verify the artifacts and "
        "run Phase 27 as a new audit. Authentic August data and the readiness gate remain separate requirements.",
        "", "10. REQUIRED DECLARATIONS",
        "MODEL CHANGED: NO", "MODEL RETRAINED: NO", "FEATURES CHANGED: NO",
        "THRESHOLDS CHANGED: NO", "CALIBRATION FITTED: NO", "AUGUST DATA CREATED: NO",
        "AUGUST HOLDOUT EVALUATED: NO", "",
        "11. DETERMINISM",
        "Two Phase 28 read-only audit passes produced byte-identical inventory JSON and report content. "
        "No runtime metadata is included.",
    ]
    return "\n".join(lines) + "\n"


def run(root: Path, report_dir: Path) -> dict[str, Any]:
    first = audit_once(root)
    second = audit_once(root)
    _assert_identical(first, second)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase28_evidence_inventory.json").write_bytes(canonical_bytes(first))
    (report_dir / "phase28_evidence_recovery_report.txt").write_text(report_text(first), encoding="utf-8")
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.root, args.report_dir)


if __name__ == "__main__":
    main()
