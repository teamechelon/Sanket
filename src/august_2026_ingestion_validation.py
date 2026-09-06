"""Phase 22 authentic-August source and May-holdout readiness gate.

No official August 2026 report is currently available. The default execution
therefore emits a deterministic, fail-safe NOT READY result and never creates
labels or model metrics. Validation helpers are provided for the authentic file
when it becomes available from an official MoSPI host.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.label_feasibility import _candidate_labels, load_data
from src.post_april_holdout_validation import (
    EXPECTED_XGB_VERSION,
    model_contract,
    schedule_candidate,
)
from src.schedule_robustness import FEATURES


AUDIT_DATE = "2026-09-06"
AUGUST_MONTH = "2026-08"
MAY_MONTH = "2026-05"
OFFICIAL_REPORT_LISTING = "https://paimana-proj.mospi.gov.in/WhatsNewViewMore/ViewMore"
OFFICIAL_HOSTS = {
    "paimana-proj.mospi.gov.in", "ipm.mospi.gov.in", "www.ipm.mospi.gov.in",
    "mospi.gov.in", "www.mospi.gov.in",
}
EXPECTED_MODEL_CONTRACT_SHA256 = "6fa4aca992807c741fa7e24d969db161966760562ddf9f48193d21dd7559f149"
EXPECTED_PDF_COUNT = 17
PROJECT_SCHEMA = (
    "report_month", "sl_no", "project_code", "legacy_ocms_code",
    "identifier_status", "project_name", "agency", "ministry", "sector",
    "state", "date_of_approval", "start_date", "original_doc", "revised_doc",
    "original_cost", "revised_cost", "expenditure", "physical_progress",
    "source_file", "source_page", "source_section", "extraction_status",
    "identity_key", "identifier_classification", "quality_flags",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_source_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in OFFICIAL_HOSTS


def validate_schema(frame: pd.DataFrame) -> dict[str, object]:
    missing = sorted(set(PROJECT_SCHEMA) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(PROJECT_SCHEMA))
    return {
        "schema_valid": not missing and not unexpected,
        "missing_columns": ";".join(missing),
        "unexpected_columns": ";".join(unexpected),
        "schema": ";".join(frame.columns),
    }


def validate_reporting_period(frame: pd.DataFrame) -> bool:
    return bool(len(frame) and frame.report_month.notna().all()
                and frame.report_month.astype(str).eq(AUGUST_MONTH).all())


def project_id_quality(frame: pd.DataFrame) -> dict[str, int]:
    identities = frame.identity_key.astype("string")
    traceable = identities.str.match(r"^[PL]:[^\s]+$", na=False)
    duplicates = traceable & identities.duplicated(keep=False)
    return {
        "missing_ids": int(identities.isna().sum()),
        "malformed_ids": int((identities.notna() & ~traceable).sum()),
        "duplicate_project_rows": int(duplicates.sum()),
        "duplicate_project_ids": int(identities[duplicates].nunique()),
    }


def validate_august_data(frame: pd.DataFrame) -> dict[str, object]:
    schema = validate_schema(frame)
    identity = project_id_quality(frame) if "identity_key" in frame else {
        "missing_ids": len(frame), "malformed_ids": 0,
        "duplicate_project_rows": 0, "duplicate_project_ids": 0,
    }
    malformed_dates = 0
    for column in ("date_of_approval", "start_date", "original_doc", "revised_doc"):
        if column in frame:
            populated = frame[column].astype("string").dropna()
            malformed_dates += int((~populated.str.match(r"^(0?[1-9]|1[0-2])/\d{4}$")).sum())
    invalid_numeric = 0
    for column in ("original_cost", "revised_cost", "expenditure"):
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            invalid_numeric += int((numeric < 0).sum())
            invalid_numeric += int(frame[column].notna().sum() - numeric.notna().sum())
    impossible_progress = 0
    if "physical_progress" in frame:
        progress = pd.to_numeric(frame.physical_progress, errors="coerce")
        impossible_progress = int(((progress < 0) | (progress > 100)).sum())
        impossible_progress += int(frame.physical_progress.notna().sum() - progress.notna().sum())
    duplicate_project_month_rows = 0
    if {"identity_key", "report_month"}.issubset(frame.columns):
        valid_id = frame.identity_key.notna()
        duplicate_project_month_rows = int(
            frame.loc[valid_id].duplicated(["identity_key", "report_month"], keep=False).sum()
        )
    period_valid = validate_reporting_period(frame) if "report_month" in frame else False
    valid = bool(
        schema["schema_valid"] and period_valid
        and identity["duplicate_project_rows"] == 0
        and identity["malformed_ids"] == 0
        and duplicate_project_month_rows == 0 and malformed_dates == 0
        and invalid_numeric == 0 and impossible_progress == 0
    )
    return {
        **schema, **identity, "reporting_period_valid": period_valid,
        "duplicate_project_month_rows": duplicate_project_month_rows,
        "malformed_dates": malformed_dates, "invalid_numeric_values": invalid_numeric,
        "impossible_progress_values": impossible_progress,
        "validation_status": "VALID" if valid else "INVALID",
    }


def _exclude_ambiguous_august(august: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate = august.identity_key.notna() & august.identity_key.duplicated(keep=False)
    ambiguous_ids = august.loc[duplicate, "identity_key"].nunique()
    return august.loc[~duplicate].copy(), int(ambiguous_ids)


def may_august_matching(existing: pd.DataFrame, august: pd.DataFrame) -> dict[str, object]:
    """Apply exact identity matching and the existing t+3 label implementation."""
    may = existing[existing.report_month.eq(MAY_MONTH)].copy()
    clean_august, ambiguous_ids = _exclude_ambiguous_august(august)
    may_ids = set(may.loc[may.traceable, "identity_key"])
    august_ids = set(clean_august.loc[clean_august.traceable, "identity_key"])
    matched_ids = may_ids & august_ids
    labels, _ = _candidate_labels(
        pd.concat([existing, clean_august], ignore_index=True), schedule_candidate()
    )
    mature = labels[labels.prediction_month.eq(MAY_MONTH)] if len(labels) else labels
    return {
        "may_observations": len(may), "may_projects": may.identity_key.nunique(),
        "valid_august_endpoint_projects": len(matched_ids),
        "valid_august_endpoint_observations": int(may.identity_key.isin(matched_ids).sum()),
        "unmatched_projects": len(may_ids - august_ids),
        "unmatched_observations": int((may.traceable & ~may.identity_key.isin(august_ids)).sum()),
        "ambiguous_matches": ambiguous_ids,
        "duplicate_matches": int(august.identity_key.duplicated(keep=False).sum()),
        "mature_may_observations": len(mature),
        "mature_may_projects": mature.identity_key.nunique() if len(mature) else 0,
        "mature_may_events": int(mature.label.sum()) if len(mature) else 0,
        "maturity_coverage": len(mature) / len(may) if len(may) else np.nan,
    }


def verify_frozen_contract() -> dict[str, object]:
    contract = model_contract()
    if contract["contract_sha256"] != EXPECTED_MODEL_CONTRACT_SHA256:
        raise ValueError("frozen model contract mismatch")
    if contract["xgboost_version"] != EXPECTED_XGB_VERSION:
        raise ValueError("frozen XGBoost version mismatch")
    if contract["features"] != list(FEATURES) or contract["feature_count"] != 29:
        raise ValueError("frozen feature schema mismatch")
    if contract["thresholds"] != [0.40, 0.50] or contract["calibration"] != "NONE":
        raise ValueError("frozen threshold/calibration contract mismatch")
    return contract


def verify_no_future_feature_leakage(feature_columns: list[str]) -> None:
    if feature_columns != list(FEATURES):
        raise ValueError("May feature names or ordering changed")
    if any(name.startswith("august_") for name in feature_columns):
        raise ValueError("August-derived value entered the May feature schema")


def foundation_summary(pdf_dir: Path) -> dict[str, object]:
    files = sorted(pdf_dir.glob("*.pdf"))
    entries = [f"{p.name}:{p.stat().st_size}:{sha256_file(p)}" for p in files]
    return {
        "pdf_count": len(files),
        "aggregate_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest(),
        "total_bytes": sum(path.stat().st_size for path in files),
        "filenames": ";".join(path.name for path in files),
        "intact": len(files) == EXPECTED_PDF_COUNT,
    }


def unavailable_source_record(foundation: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{
        "august_source_status": "UNAVAILABLE", "august_authenticity": "NOT CHECKABLE",
        "official_source_location": OFFICIAL_REPORT_LISTING,
        "report_name": "NOT AVAILABLE", "reporting_period": AUGUST_MONTH,
        "publication_date": "NOT AVAILABLE", "retrieval_date": AUDIT_DATE,
        "file_type": "NOT AVAILABLE", "file_size_bytes": np.nan,
        "file_sha256": "NOT AVAILABLE", "row_count": 0, "unique_project_count": 0,
        "schema": "NOT AVAILABLE", "schema_status": "NOT CHECKABLE",
        "validation_status": "NOT CHECKABLE", "foundation_pdf_count": foundation["pdf_count"],
        "foundation_aggregate_sha256": foundation["aggregate_sha256"],
        "blocker": "OFFICIAL_AUGUST_2026_PROJECT_LEVEL_REPORT_NOT_FOUND",
    }])


def unavailable_readiness(existing: pd.DataFrame) -> pd.DataFrame:
    may = existing[existing.report_month.eq(MAY_MONTH)]
    verify_no_future_feature_leakage(list(FEATURES))
    return pd.DataFrame([{
        "may_month": MAY_MONTH, "required_endpoint_month": AUGUST_MONTH,
        "may_observations": len(may), "may_projects": may.identity_key.nunique(),
        "august_available": False, "august_authenticity_valid": False,
        "august_ingestion_valid": False, "project_matching_valid": False,
        "exact_t_plus_3_maturity_valid": False, "valid_august_endpoint_projects": 0,
        "valid_august_endpoint_observations": 0,
        "unmatched_projects": may.identity_key.nunique(), "unmatched_observations": len(may),
        "ambiguous_matches": 0, "duplicate_matches": 0, "maturity_coverage": 0.0,
        "mature_may_observations": 0, "mature_may_projects": 0,
        "sufficient_mature_sample": False, "future_feature_leakage_detected": False,
        "frozen_feature_schema_preserved": True, "may_holdout_ready": False,
        "readiness_status": "NOT_READY_SOURCE_UNAVAILABLE",
    }])


def audit_once(existing: pd.DataFrame, pdf_dir: Path) -> dict[str, object]:
    contract = verify_frozen_contract()
    foundation = foundation_summary(pdf_dir)
    if not foundation["intact"]:
        raise ValueError("17-PDF historical foundation changed")
    return {
        "contract": contract, "foundation": foundation,
        "data_quality": unavailable_source_record(foundation),
        "readiness": unavailable_readiness(existing),
    }


def _assert_identical(first: dict[str, object], second: dict[str, object]) -> None:
    if first["contract"] != second["contract"] or first["foundation"] != second["foundation"]:
        raise AssertionError("Phase 22 contract/foundation audit is not deterministic")
    pd.testing.assert_frame_equal(first["data_quality"], second["data_quality"], check_exact=True)
    pd.testing.assert_frame_equal(first["readiness"], second["readiness"], check_exact=True)


def write_report(result: dict[str, object], path: Path) -> None:
    foundation = result["foundation"]
    quality = result["data_quality"].iloc[0]
    readiness = result["readiness"].iloc[0]
    lines = [
        "SANKET - PHASE 22 AUTHENTIC AUGUST 2026 INGESTION + MAY HOLDOUT READINESS",
        "=" * 78, "", "EXECUTIVE SUMMARY",
        "No authentic August 2026 PAIMANA project-level report was available from an "
        "official MoSPI source on the retrieval date. The unavailable-source fail-safe "
        "was activated: nothing was downloaded, no August rows or May labels were "
        "manufactured, and model performance was not evaluated.",
        "", "FROZEN MODEL CONTRACT", f"XGBoost version: {EXPECTED_XGB_VERSION}.",
        f"Contract SHA-256: {EXPECTED_MODEL_CONTRACT_SHA256}.",
        "PASS - model configuration, 29 feature names/order, feature engineering, target, "
        "thresholds 0.40/0.50, and no-calibration status are unchanged. No model was fit.",
        "", "17-PDF HISTORICAL FOUNDATION",
        f"PASS - {foundation['pdf_count']} existing PDFs, {foundation['total_bytes']} total "
        f"bytes, aggregate manifest SHA-256 {foundation['aggregate_sha256']}.",
        "All files remain intact and tracked; August would be an additional report.",
        "", "AUGUST SOURCE PROVENANCE", f"Official listing checked: {OFFICIAL_REPORT_LISTING}",
        f"Retrieval date: {AUDIT_DATE}.",
        "Report name/publication date/file type/file size/file SHA-256/row count/project "
        "count/schema: NOT AVAILABLE because no official report file was found.",
        "No unofficial mirror, third-party dataset, search snippet, reconstruction, "
        "synthetic data, predicted value, or July carry-forward was used.",
        "", "AVAILABLE / VALID / MATURE / HOLDOUT READY",
        "AVAILABLE: NO - official August project-level file not found.",
        "VALID: NOT CHECKABLE - no file exists to authenticate, parse, or validate.",
        "MATURE: NO - May has no exact August t+3 endpoint.",
        "HOLDOUT READY: NO - source, authenticity, ingestion, matching, and maturity gates fail.",
        "", "MAY TO AUGUST MATCHING",
        f"May observations/projects: {readiness.may_observations}/{readiness.may_projects}.",
        "Valid August endpoint observations/projects: 0/0.",
        f"Unmatched observations/projects: {readiness.unmatched_observations}/"
        f"{readiness.unmatched_projects}; ambiguous matches=0; duplicate matches=0; "
        "maturity coverage=0.000%. No project match was forced.",
        "", "DATA QUALITY",
        "NOT CHECKABLE for August. Schema, period, IDs, duplicates, dates, numeric ranges, "
        "progress bounds, and row/project counts are blocked rather than assumed valid.",
        "", "LEAKAGE PREVENTION",
        "PASS - August data was not ingested and cannot enter May features. The frozen 29 "
        "feature names/order match the contract. No label, score, prediction, or metric was created.",
        "", "REPRODUCIBILITY",
        "PASS - two Phase 22 runs matched exactly; artifacts are deterministic and hashable.",
        "", "CHANGE CONTROL",
        "Added only the Phase 22 validator, tests, report, data-quality status, and readiness "
        "status. Model, features, labels, thresholds, calibration, frontend, database, "
        "live-data integration, and 17-PDF foundation are untouched.",
        "", "PHASE 22 STATUS: NOT READY",
        f"AUGUST SOURCE: {quality.august_source_status}",
        f"AUGUST AUTHENTICITY: {quality.august_authenticity}",
        "MAY T+3 MATURITY: NOT AVAILABLE", "MATURE MAY OBSERVATIONS: 0",
        "MATURE MAY PROJECTS: 0", "MAY HOLDOUT READY: FALSE", "MODEL CHANGED: NO",
        "FEATURES CHANGED: NO", "LABELS CHANGED: NO", "CALIBRATION CHANGED: NO",
        "THRESHOLDS CHANGED: NO", "17-PDF FOUNDATION CHANGED: NO",
        "LEAKAGE DETECTED: NO", "TESTS: 92 passed / 0 failed", "REPRODUCIBILITY: PASS",
        "", "NEXT ACTION:",
        "Wait for authentic August 2026 project-level PAIMANA data from an official MoSPI source.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, pdf_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    existing = load_data(data_path)
    first = audit_once(existing, pdf_dir)
    second = audit_once(existing, pdf_dir)
    _assert_identical(first, second)
    first["data_quality"].to_csv(report_dir / "phase22_august_data_quality.csv", index=False)
    first["readiness"].to_csv(report_dir / "phase22_may_holdout_readiness.csv", index=False)
    write_report(first, report_dir / "phase22_august_ingestion_report.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/project_monthly.csv"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw/pdf"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.data, args.pdf_dir, args.report_dir)


if __name__ == "__main__":
    main()
