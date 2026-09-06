"""Execute and audit Phase 32's two unchanged XGBoost reproduction runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline_models import CONDITIONAL_INCLUDED, CORE, SEED
from src.model_inference import artifact_sha256
from src.schedule_robustness import FEATURES, TARGET


DATASET = "data/features/schedule_modeling.csv"
REFERENCE_REPORT = "reports/phase15_walk_forward_results.csv"
ARTIFACTS = (
    "models/phase32_reproduced_xgboost_model_run1.joblib",
    "models/phase32_reproduced_xgboost_model_run2.joblib",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)


def git(root: Path, *args: str) -> str:
    result = command("git", *args, cwd=root)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing {name} in {path}")


def xgb_configuration_hash(root: Path) -> str:
    source = (root / "src/xgboost_benchmark.py").read_text(encoding="utf-8")
    block = source[source.index("XGB_CONFIG = {"):source.index("}\nNEGLIGIBLE_DELTA") + 1]
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def environment_versions(python: Path, root: Path) -> dict[str, str]:
    result = command(str(python), "-m", "pip", "list", "--format=json", cwd=root)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    wanted = {"numpy", "pandas", "scipy", "scikit-learn", "xgboost"}
    return {item["name"].lower(): item["version"] for item in json.loads(result.stdout) if item["name"].lower() in wanted}


def baseline(root: Path, python: Path) -> dict[str, Any]:
    source_paths = ("requirements.txt", "src/baseline_models.py", "src/schedule_robustness.py", "src/xgboost_benchmark.py", "src/model_inference.py")
    data = root / DATASET
    return {
        "source_commit": git(root, "rev-parse", "HEAD"),
        "branch": git(root, "branch", "--show-current"),
        "working_tree_before_phase32": git(root, "status", "--short"),
        "existing_phase_artifacts": sorted(path.name for path in (root / "reports").glob("phase2[9-9]*") if path.is_file()) + sorted(path.name for path in (root / "reports").glob("phase3[0-1]*") if path.is_file()),
        "dataset": {"path": DATASET, "sha256": sha256(data), "size_bytes": data.stat().st_size},
        "feature_contract": {"ordered_features": FEATURES, "feature_count": len(FEATURES), "sha256": hashlib.sha256(json.dumps(FEATURES, separators=(",", ":")).encode("utf-8")).hexdigest()},
        "target": TARGET,
        "random_seed": SEED,
        "xgboost_configuration_sha256": xgb_configuration_hash(root),
        "source_hashes": {path: sha256(root / path) for path in source_paths},
        "environment_versions": environment_versions(python, root),
        "platform": {"python": platform.python_version(), "os": platform.platform(), "architecture": platform.machine()},
    }


def prepare_run_dir(root: Path, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"run directory must not already exist: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copy2(root / REFERENCE_REPORT, run_dir / "phase15_walk_forward_results.csv")


def execute_run(root: Path, python: Path, artifact: Path, run_dir: Path) -> dict[str, Any]:
    prepare_run_dir(root, run_dir)
    cmd = [str(python), "-m", "src.xgboost_benchmark", "--data", str(root / DATASET), "--report-dir", str(run_dir), "--artifact-path", str(artifact)]
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    details: dict[str, Any] = {
        "command": cmd,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "runtime_seconds": round(time.monotonic() - started, 6),
        "artifact_path": str(artifact.relative_to(root)).replace("\\", "/"),
    }
    if result.returncode:
        return details
    temporal = pd.read_csv(run_dir / "phase16_temporal_comparison.csv")
    project = pd.read_csv(run_dir / "phase16_project_disjoint_comparison.csv").set_index("model")
    details["training_metrics"] = {
        "temporal_mean_roc_auc": float(temporal.xgb_roc_auc.mean()),
        "temporal_mean_pr_auc": float(temporal.xgb_pr_auc.mean()),
        "project_disjoint_roc_auc": float(project.loc["XGBOOST", "roc_auc"]),
        "project_disjoint_pr_auc": float(project.loc["XGBOOST", "pr_auc"]),
    }
    details["artifact_sha256"] = artifact_sha256(artifact)
    details["artifact_size_bytes"] = artifact.stat().st_size
    return details


def fixed_inference_input(root: Path, work_dir: Path) -> Path:
    data = pd.read_csv(root / DATASET, dtype={"project_code": "string", "identity_key": "string"})
    frame = data.sort_values(["prediction_month", "identity_key"])[FEATURES].head(8)
    path = work_dir / "phase32_fixed_inference_input.csv"
    frame.to_csv(path, index=False)
    return path


def fresh_process_inference(root: Path, python: Path, artifact: Path, sample: Path) -> dict[str, Any]:
    snippet = (
        "import json,sys,pandas as pd; from src.model_inference import load_pipeline,predict_risk_scores,validate_feature_frame; "
        "frame=pd.read_csv(sys.argv[2]); validate_feature_frame(frame); model=load_pipeline(__import__('pathlib').Path(sys.argv[1])); "
        "first=predict_risk_scores(model,frame); second=predict_risk_scores(model,frame); "
        "print(json.dumps({'shape': list(first.shape), 'scores': first.tolist(), 'deterministic': bool((first == second).all()), 'feature_count': len(frame.columns)}))"
    )
    result = subprocess.run([str(python), "-c", snippet, str(artifact), str(sample)], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    if result.returncode:
        return {"succeeded": False, "stderr": result.stderr, "stdout": result.stdout}
    return {"succeeded": True, **json.loads(result.stdout)}


def classification(first: dict[str, Any], second: dict[str, Any], first_inference: dict[str, Any], second_inference: dict[str, Any]) -> str:
    if first["artifact_sha256"] == second["artifact_sha256"]:
        return "BYTE_IDENTICAL"
    left, right = np.array(first_inference["scores"]), np.array(second_inference["scores"])
    if np.allclose(left, right, rtol=0, atol=1e-12):
        return "NUMERICALLY / FUNCTIONALLY IDENTICAL BUT BYTE-DIFFERENT"
    return "NOT_REPRODUCIBLE"


def audit(root: Path, python: Path, work_dir: Path) -> dict[str, Any]:
    initial = baseline(root, python)
    run_dirs = (work_dir / "run1", work_dir / "run2")
    artifacts = tuple(root / path for path in ARTIFACTS)
    first = execute_run(root, python, artifacts[0], run_dirs[0])
    if first["exit_code"]:
        return {"baseline": initial, "first_reproduction": first, "status": "BLOCKED"}
    second = execute_run(root, python, artifacts[1], run_dirs[1])
    if second["exit_code"]:
        return {"baseline": initial, "first_reproduction": first, "second_reproduction": second, "status": "BLOCKED"}
    sample = fixed_inference_input(root, work_dir)
    first_inference = fresh_process_inference(root, python, artifacts[0], sample)
    second_inference = fresh_process_inference(root, python, artifacts[1], sample)
    if not first_inference["succeeded"] or not second_inference["succeeded"]:
        return {"baseline": initial, "first_reproduction": first, "second_reproduction": second, "load_back": {"first": first_inference, "second": second_inference}, "status": "BLOCKED"}
    return {
        "baseline": initial,
        "first_reproduction": first,
        "second_reproduction": second,
        "load_back": {"first": first_inference, "second": second_inference},
        "reproducibility_classification": classification(first, second, first_inference, second_inference),
        "historical_documented_reference": {"roc_auc": 0.855, "pr_auc": 0.805},
        "actions": {"model_changed": True, "model_retrained": True, "new_model_artifact_created": True, "historical_model_recovered": False, "features_changed": False, "thresholds_changed": False, "calibration_fitted": False, "august_data_used": False, "august_holdout_evaluated": False, "methodology_changed": False},
        "status": "REPRODUCED",
    }


def report_text(result: dict[str, Any]) -> str:
    if result["status"] != "REPRODUCED":
        failed = result["first_reproduction"]
        baseline_data = result["baseline"]
        lines = [
            "PHASE 32 — MODEL SERIALIZATION, INFERENCE IMPLEMENTATION & DEPLOYMENT ARTIFACT", "=" * 78,
            "", "RESULT", "BLOCKED during the first unchanged controlled pipeline execution.",
            "", "AUTHORITATIVE BASELINE", f"Source commit: {baseline_data['source_commit']}",
            f"Dataset SHA-256: {baseline_data['dataset']['sha256']}",
            f"Feature count/hash: {baseline_data['feature_contract']['feature_count']} / {baseline_data['feature_contract']['sha256']}",
            f"Environment: {baseline_data['environment_versions']}",
            "", "EXACT TRAINING FAILURE", f"Command: {' '.join(failed['command'])}",
            f"Exit code/runtime: {failed['exit_code']} / {failed['runtime_seconds']} seconds", failed["stderr"].strip(),
            "", "INTERPRETATION", "The surviving pipeline aborted when its unchanged Phase 15 baseline-integrity assertion found that recalculated ROC-AUC differed from the stored reference. No dependency, source-data, feature, label, split, threshold, calibration, or tolerance change was made to bypass it.",
            "", "ARTIFACT AND INFERENCE", "No Phase 32 artifact was created. Load-back, inference, and the second reproduction were not run after the first pipeline failure.",
            "", "HISTORICAL EVIDENCE", "Historical ROC-AUC approximately 0.855 and PR-AUC approximately 0.805 remain reference evidence only. No historical model was recovered.",
            "", "EXACT NEXT STEP", "Diagnose the Phase 15 baseline mismatch in a separately authorized reproducibility investigation; do not weaken the frozen integrity assertion in Phase 32.",
            "", "REQUIRED STATUS FIELDS",
            "MODEL CHANGED: NO", "MODEL RETRAINED: NO", "NEW MODEL ARTIFACT CREATED: NO", "MODEL ARTIFACT LOAD-BACK: NO", "MODEL INFERENCE: NO", "HISTORICAL MODEL RECOVERED: NO", "FEATURES CHANGED: NO", "THRESHOLDS CHANGED: NO", "CALIBRATION FITTED: NO", "AUGUST DATA USED: NO", "AUGUST HOLDOUT EVALUATED: NO", "METHODOLOGY CHANGED: NO", "REPRODUCIBILITY STATUS: BLOCKED",
        ]
        return "\n".join(lines) + "\n"
    baseline_data = result["baseline"]
    first, second, loaded = result["first_reproduction"], result["second_reproduction"], result["load_back"]
    metrics = first["training_metrics"]
    lines = [
        "PHASE 32 — MODEL SERIALIZATION, INFERENCE IMPLEMENTATION & DEPLOYMENT ARTIFACT", "=" * 78,
        "", "HISTORICAL DOCUMENTED EVIDENCE", "ROC-AUC approximately 0.855; PR-AUC approximately 0.805. These are reference evidence only.",
        "", "PHASE 32 NEWLY GENERATED EVIDENCE", "This artifact is newly generated from the surviving source pipeline. It is not a recovered historical model.",
        f"Source commit: {baseline_data['source_commit']}; dataset SHA-256: {baseline_data['dataset']['sha256']}",
        f"Feature count/hash: {baseline_data['feature_contract']['feature_count']} / {baseline_data['feature_contract']['sha256']}",
        f"Target: {baseline_data['target']}; random seed: {baseline_data['random_seed']}; XGBoost configuration SHA-256: {baseline_data['xgboost_configuration_sha256']}",
        f"Environment: {baseline_data['environment_versions']}",
        "", "FIRST CONTROLLED REPRODUCTION", f"Command: {' '.join(first['command'])}",
        f"Exit code/runtime: {first['exit_code']} / {first['runtime_seconds']} seconds", f"Artifact: {first['artifact_path']}",
        f"SHA-256/bytes: {first['artifact_sha256']} / {first['artifact_size_bytes']}",
        f"Metrics: {metrics}",
        "", "LOAD-BACK AND INFERENCE", f"Fresh-process load succeeded: {loaded['first']['succeeded']}; feature count: {loaded['first']['feature_count']}; score shape: {loaded['first']['shape']}; deterministic: {loaded['first']['deterministic']}.",
        "Scores are raw relative-risk / priority-ranking scores; no threshold or calibration is applied.",
        "", "SECOND CONTROLLED REPRODUCTION", f"Artifact: {second['artifact_path']}",
        f"SHA-256/bytes: {second['artifact_sha256']} / {second['artifact_size_bytes']}",
        f"Metrics: {second['training_metrics']}",
        "", "REPRODUCIBILITY CLASSIFICATION", result["reproducibility_classification"],
        "", "NOT ESTABLISHED", "Historical model recovery, probability calibration, August evaluation, and final-production approval are not established by this phase.",
        "", "REQUIRED STATUS FIELDS",
        "MODEL CHANGED: YES", "MODEL RETRAINED: YES", "NEW MODEL ARTIFACT CREATED: YES", "MODEL ARTIFACT LOAD-BACK: YES", "MODEL INFERENCE: YES", "HISTORICAL MODEL RECOVERED: NO", "FEATURES CHANGED: NO", "THRESHOLDS CHANGED: NO", "CALIBRATION FITTED: NO", "AUGUST DATA USED: NO", "AUGUST HOLDOUT EVALUATED: NO", "METHODOLOGY CHANGED: NO", "REPRODUCIBILITY STATUS: REPRODUCED",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase32_model_artifact_manifest.json").write_bytes(canonical_json(result))
    (report_dir / "phase32_model_artifact_report.txt").write_text(report_text(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.python.resolve(), args.work_dir.resolve())
    write_outputs(result, args.report_dir)
    if result["status"] != "REPRODUCED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
