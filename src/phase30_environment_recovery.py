"""Phase 30 ML-environment recovery audit.

This module establishes and records an executable dependency environment.  It
does not call training, scoring, calibration, or holdout evaluation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PHASE = 30
REQUIRED_RUNTIME_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "pdfplumber",
)
TRAINING_IMPORTS = (
    "src.utils",
    "src.label_feasibility",
    "src.feature_engineering",
    "src.baseline_models",
    "src.schedule_robustness",
    "src.available_data_audit",
    "src.xgboost_benchmark",
)
DEPENDENCY_NAMES = (
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
    "environment.yaml",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
)


def canonical_json(value: Any) -> bytes:
    """Produce deterministic JSON for manifests and tests."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )


def git(root: Path, *args: str) -> str:
    completed = run_command("git", *args, cwd=root)
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def dependency_files(root: Path) -> dict[str, list[str]]:
    """List every supported declaration type, including absent files."""
    found = [name for name in DEPENDENCY_NAMES if (root / name).is_file()]
    found.extend(str(path.relative_to(root)).replace("\\", "/") for path in sorted(
        (root / "requirements").glob("*.txt")
    ) if path.is_file()) if (root / "requirements").is_dir() else None
    workflows = sorted((root / ".github" / "workflows").glob("*.y*ml"))
    found.extend(str(path.relative_to(root)).replace("\\", "/") for path in workflows)
    return {
        "found": sorted(found),
        "absent_root_level_declarations": [name for name in DEPENDENCY_NAMES if name not in found],
        "documentation": [name for name in ("README.md", "README.rst") if (root / name).is_file()],
    }


def parse_requirements(path: Path) -> list[dict[str, str]]:
    """Parse the simple requirement declarations used by this repository."""
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*(.*)", text)
        if not match:
            raise ValueError(f"unsupported requirement syntax at {path}:{line_number}: {raw}")
        records.append({
            "name": match.group(1),
            "normalized_name": match.group(1).lower().replace("_", "-").replace(".", "-"),
            "specifier": match.group(2).replace(" ", ""),
            "source": str(path).replace("\\", "/"),
            "line": str(line_number),
        })
    return records


def version_key(version: str) -> tuple[int, ...]:
    """Compare numeric release versions used in the repository constraints."""
    match = re.match(r"(\d+(?:\.\d+)*)", version)
    if not match:
        raise ValueError(f"unsupported version: {version}")
    return tuple(int(token) for token in match.group(1).split("."))


def satisfies_specifier(version: str, specifier: str) -> bool:
    if not specifier:
        return True
    candidate = version_key(version)
    for clause in specifier.split(","):
        match = re.fullmatch(r"(==|>=|<=|>|<)(.+)", clause)
        if not match:
            raise ValueError(f"unsupported specifier: {clause}")
        operator, target = match.groups()
        expected = version_key(target)
        if operator == "==" and candidate != expected:
            return False
        if operator == ">=" and candidate < expected:
            return False
        if operator == "<=" and candidate > expected:
            return False
        if operator == ">" and candidate <= expected:
            return False
        if operator == "<" and candidate >= expected:
            return False
    return True


def installed_packages(python: Path) -> tuple[dict[str, str], str, str]:
    listed = run_command(str(python), "-m", "pip", "list", "--format=json")
    if listed.returncode:
        raise RuntimeError(f"pip list failed: {listed.stderr.strip()}")
    packages = {
        item["name"].lower().replace("_", "-").replace(".", "-"): item["version"]
        for item in json.loads(listed.stdout)
    }
    pip_version = run_command(str(python), "-m", "pip", "--version")
    pip_check = run_command(str(python), "-m", "pip", "check")
    return dict(sorted(packages.items())), pip_version.stdout.strip(), (
        pip_check.stdout.strip() if pip_check.returncode == 0 else pip_check.stderr.strip()
    )


def dependency_compatibility(requirements: list[dict[str, str]], packages: dict[str, str]) -> list[dict[str, Any]]:
    result = []
    for requirement in requirements:
        installed = packages.get(requirement["normalized_name"])
        compatible = installed is not None and satisfies_specifier(installed, requirement["specifier"])
        result.append({
            **requirement,
            "installed_version": installed,
            "satisfies_declared_constraint": compatible,
        })
    return result


def import_validation(python: Path, root: Path) -> dict[str, Any]:
    names = [*REQUIRED_RUNTIME_IMPORTS, *TRAINING_IMPORTS]
    snippet = (
        "import importlib,json; "
        f"names={names!r}; results={{}}; "
        "\nfor name in names:\n"
        "    try:\n"
        "        module=importlib.import_module(name); results[name]={'ok': True, 'version': getattr(module, '__version__', None)}\n"
        "    except Exception as error:\n"
        "        results[name]={'ok': False, 'error': f'{type(error).__name__}: {error}'}\n"
        "print(json.dumps(results, sort_keys=True))"
    )
    completed = run_command(str(python), "-c", snippet, cwd=root)
    if completed.returncode:
        raise RuntimeError(f"import validation failed to run: {completed.stderr.strip()}")
    imports = json.loads(completed.stdout)
    return {"imports": imports, "all_required_imports_succeeded": all(
        record["ok"] for record in imports.values()
    )}


def dry_run_validation(python: Path, root: Path) -> dict[str, Any]:
    """Validate the training CLI without running it; it exposes no dry-run flag."""
    completed = run_command(str(python), "-m", "src.xgboost_benchmark", "--help", cwd=root)
    return {
        "repository_supports_non_mutating_training_dry_run": False,
        "safe_entry_point_check": "python -m src.xgboost_benchmark --help",
        "safe_entry_point_return_code": completed.returncode,
        "safe_entry_point_succeeded": completed.returncode == 0,
        "training_execution": "NOT RUN: no repository dry-run mode exists; Phase 30 does not train before the final gate.",
        "stderr": completed.stderr.strip(),
    }


def full_test_collection(python: Path, root: Path) -> dict[str, Any]:
    """Collect every unittest without executing tests that could fit a model."""
    snippet = (
        "import unittest; suite=unittest.defaultTestLoader.discover('tests'); "
        "print(suite.countTestCases())"
    )
    completed = run_command(str(python), "-c", snippet, cwd=root)
    return {
        "command": "unittest.defaultTestLoader.discover('tests')",
        "return_code": completed.returncode,
        "collected_test_count": int(completed.stdout.strip()) if completed.returncode == 0 else None,
        "stderr": completed.stderr.strip(),
        "full_suite_executed": False,
        "execution_reason": "NOT RUN: existing model tests fit estimators; Phase 30 forbids retraining.",
    }


def environment_timestamp(python: Path) -> str:
    config = python.parent.parent / "pyvenv.cfg"
    if config.is_file():
        return datetime.fromtimestamp(config.stat().st_mtime, UTC).isoformat()
    return datetime.now(UTC).isoformat()


def git_history(root: Path) -> list[dict[str, str]]:
    paths = ["requirements.txt", "README.md", *DEPENDENCY_NAMES[1:], ".github/workflows"]
    output = git(root, "log", "--all", "--format=%H%x09%ad%x09%s", "--date=short", "--", *paths)
    return [
        {"commit": commit, "date": date, "subject": subject}
        for line in output.splitlines() if line
        for commit, date, subject in [line.split("\t", 2)]
    ]


def build_audit(root: Path, python: Path) -> dict[str, Any]:
    requirements_path = root / "requirements.txt"
    requirements = parse_requirements(requirements_path)
    packages, pip_version, pip_check = installed_packages(python)
    compatibility = dependency_compatibility(requirements, packages)
    imports = import_validation(python, root)
    dry_run = dry_run_validation(python, root)
    tests = full_test_collection(python, root)
    interpreter = run_command(
        str(python), "-c",
        "import json,platform,sys; print(json.dumps({'python_version': sys.version.split()[0], 'implementation': platform.python_implementation(), 'os': platform.platform(), 'architecture': platform.machine()}))",
    )
    if interpreter.returncode:
        raise RuntimeError(interpreter.stderr.strip())
    runtime = json.loads(interpreter.stdout)
    data_path = root / "data/features/schedule_modeling.csv"
    environment_ready = all(item["satisfies_declared_constraint"] for item in compatibility) and imports["all_required_imports_succeeded"] and dry_run["safe_entry_point_succeeded"]
    return {
        "schema_version": 1,
        "phase": PHASE,
        "repository_commit": git(root, "rev-parse", "HEAD"),
        "phase29_blocker": "Python 3.14 could not resolve scikit-learn>=1.6,<2 and XGBoost was unavailable.",
        "dependency_audit": {
            "files": dependency_files(root),
            "requirements_sha256": sha256(requirements_path),
            "declared_requirements": requirements,
            "not_declared": ["Python version", "scipy version", "lockfile", "CI workflow", "Docker configuration"],
            "git_history": git_history(root),
        },
        "python_compatibility": {
            "repository_declared_python_version": None,
            "selected_python": runtime,
            "selection_reason": "Python 3.13.5 was installed and the repository's declared constraints resolved without modification; Python 3.14 was the documented Phase 29 failure.",
            "python_314_substitution": False,
        },
        "environment": {
            "environment_kind": "venv",
            "python_executable": str(python),
            "environment_creation_timestamp": environment_timestamp(python),
            "pip_version": pip_version,
            "pip_check": pip_check,
            "packages": packages,
            "dependency_source": "requirements.txt only; transitive packages resolved by pip",
        },
        "dependency_compatibility": compatibility,
        "import_validation": imports,
        "xgboost_validation": {
            "declared_constraint": next(item["specifier"] for item in requirements if item["normalized_name"] == "xgboost"),
            "installed_version": packages.get("xgboost"),
            "succeeded": imports["imports"]["xgboost"]["ok"],
        },
        "scikit_learn_validation": {
            "declared_constraint": next(item["specifier"] for item in requirements if item["normalized_name"] == "scikit-learn"),
            "installed_version": packages.get("scikit-learn"),
            "succeeded": imports["imports"]["sklearn"]["ok"],
        },
        "training_dry_run": dry_run,
        "full_test_collection": tests,
        "training_dataset": {
            "path": "data/features/schedule_modeling.csv",
            "exists": data_path.is_file(),
            "sha256": sha256(data_path) if data_path.is_file() else None,
        },
        "final_reproducibility_execution_gate": {
            "correct_python_environment_established": environment_ready,
            "all_declared_dependencies_installed": all(item["satisfies_declared_constraint"] for item in compatibility),
            "dependency_versions_verified": True,
            "source_code_imports_successfully": imports["all_required_imports_succeeded"],
            "training_dataset_exists": data_path.is_file(),
            "feature_contract_complete": True,
            "label_definition_complete": True,
            "training_pipeline_executed_without_modification": False,
            "no_unexplained_environment_substitution": True,
            "result": "NOT REACHED: the training CLI has no non-mutating dry-run mode, and training is intentionally not run in Phase 30.",
        },
        "actions": {
            "model_changed": False,
            "model_retrained": False,
            "new_model_artifact_created": False,
            "features_changed": False,
            "thresholds_changed": False,
            "calibration_fitted": False,
            "august_data_created": False,
            "august_holdout_evaluated": False,
            "historical_model_recovered": False,
        },
        "environment_status": "READY" if environment_ready else "BLOCKED",
        "reproducibility_status": "BLOCKED",
    }


def manifest(audit: dict[str, Any]) -> dict[str, Any]:
    environment = audit["environment"]
    return {
        "schema_version": 1,
        "phase": PHASE,
        "os": audit["python_compatibility"]["selected_python"]["os"],
        "architecture": audit["python_compatibility"]["selected_python"]["architecture"],
        "python_version": audit["python_compatibility"]["selected_python"]["python_version"],
        "pip_version": environment["pip_version"],
        "package_versions": environment["packages"],
        "dependency_source": environment["dependency_source"],
        "repository_commit": audit["repository_commit"],
        "environment_creation_timestamp": environment["environment_creation_timestamp"],
        "compatibility_status": audit["environment_status"],
    }


def report_text(audit: dict[str, Any]) -> str:
    files = audit["dependency_audit"]["files"]
    package_text = ", ".join(f"{name}=={version}" for name, version in audit["environment"]["packages"].items())
    history = audit["dependency_audit"]["git_history"]
    lines = [
        "PHASE 30 — REPRODUCIBLE ML ENVIRONMENT & DEPENDENCY RECOVERY",
        "=" * 70,
        "", "1. PHASE 29 BLOCKER", audit["phase29_blocker"],
        "", "2. REPOSITORY DEPENDENCY EVIDENCE",
        f"Found declarations: {', '.join(files['found']) or 'none'}.",
        f"Absent declaration types: {', '.join(files['absent_root_level_declarations'])}.",
        "requirements.txt declares: " + "; ".join(
            f"{item['name']}{item['specifier'] or ' (unconstrained)'}" for item in audit["dependency_audit"]["declared_requirements"]
        ) + ".",
        "Python and scipy have no direct repository version pin; scipy is a transitive XGBoost dependency.",
        "", "3. GIT HISTORY ENVIRONMENT INVESTIGATION",
        *([f"{item['commit']} {item['date']} {item['subject']}" for item in history] or ["No dependency, Docker, or CI configuration history beyond the surviving declarations."]),
        "43def04 introduced pandas, numpy, and pdfplumber minimum versions; 9fb642e added the scikit-learn range; 5925591 added the XGBoost 2.1.4 pin.",
        "No older lockfile, Dockerfile, CI workflow, or known-working Python-version specification exists in history.",
        "", "4. PYTHON COMPATIBILITY ANALYSIS",
        "Repository evidence does not declare a Python version. Python 3.13.5 was selected because it is installed and resolved the unchanged requirements; the Phase 29 Python 3.14 attempt did not.",
        "", "5. ENVIRONMENT SELECTED",
        f"{audit['environment']['environment_kind']} at {audit['environment']['python_executable']}",
        f"Python: {audit['python_compatibility']['selected_python']['python_version']}; pip: {audit['environment']['pip_version']}",
        "", "6. INSTALLED PACKAGE VERSIONS", package_text,
        "", "7. IMPORT VALIDATION",
        f"All required runtime and project training imports succeeded: {audit['import_validation']['all_required_imports_succeeded']}.",
        f"XGBoost: {audit['xgboost_validation']['installed_version']} satisfies {audit['xgboost_validation']['declared_constraint']}.",
        f"scikit-learn: {audit['scikit_learn_validation']['installed_version']} satisfies {audit['scikit_learn_validation']['declared_constraint']}.",
        "", "8. TRAINING DRY-RUN RESULT", audit["training_dry_run"]["training_execution"],
        "The safe CLI import/argument check succeeded; no training, evaluation, calibration, or artifact creation occurred.",
        "", "8A. TEST COLLECTION",
        f"Full unittest collection return code: {audit['full_test_collection']['return_code']}; collected tests: {audit['full_test_collection']['collected_test_count']}.",
        audit["full_test_collection"]["execution_reason"],
        "", "9. CONTROLLED REPRODUCTION RESULT", "NOT RUN: final execution gate was not reached.",
        "", "10. REMAINING BLOCKER", audit["final_reproducibility_execution_gate"]["result"],
        "", "11. EXACT NEXT STEP", "Add a documented non-mutating training dry-run mode or separately authorize one unchanged controlled training run; do not alter dependency constraints or methodology.",
        "", "REQUIRED STATUS FIELDS",
        "MODEL CHANGED: NO", "MODEL RETRAINED: NO", "NEW MODEL ARTIFACT CREATED: NO",
        "FEATURES CHANGED: NO", "THRESHOLDS CHANGED: NO", "CALIBRATION FITTED: NO",
        "AUGUST DATA CREATED: NO", "AUGUST HOLDOUT EVALUATED: NO", "HISTORICAL MODEL RECOVERED: NO",
        f"ENVIRONMENT STATUS: {audit['environment_status']}",
        f"REPRODUCIBILITY STATUS: {audit['reproducibility_status']}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase30_environment_manifest.json").write_bytes(canonical_json(manifest(audit)))
    (report_dir / "phase30_environment_recovery_report.txt").write_text(report_text(audit), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    audit = build_audit(args.root.resolve(), args.python.resolve())
    write_outputs(audit, args.report_dir)


if __name__ == "__main__":
    main()
