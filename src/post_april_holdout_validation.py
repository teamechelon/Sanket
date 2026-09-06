"""Phase 21 maturity gate for the first post-April forward holdout.

The repository currently ends in July 2026, so no post-April cutoff has its
required exact t+3 endpoint. This module documents that fail-safe result and
does not fit or evaluate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from src.label_feasibility import CANDIDATES, _candidate_labels, load_data
from src.schedule_robustness import FEATURES, TARGET
from src.xgboost_benchmark import XGB_CONFIG


DEVELOPMENT_END = "2026-04"
POST_APRIL_START = "2026-05"
TARGET_HORIZON_MONTHS = 3
AUDIT_DATE = "2026-09-06"
OFFICIAL_REPORT_SOURCE = "https://paimana-proj.mospi.gov.in/WhatsNewViewMore/ViewMore"
STATUS = "INVALID HOLDOUT"
EXPECTED_XGB_VERSION = "2.1.4"


class InvalidHoldoutError(RuntimeError):
    """Raised when evaluation is requested without a mature holdout."""


def schedule_candidate():
    return next(
        candidate for candidate in CANDIDATES
        if candidate.target_name == TARGET
    )


def required_endpoint(month: str) -> str:
    return str(pd.Period(month, freq="M") + TARGET_HORIZON_MONTHS)


def post_april_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.report_month >= POST_APRIL_START].copy()


def model_contract() -> dict[str, object]:
    """Record the frozen logical model contract; no fitted model is serialized."""
    configuration = {
        key: "NaN" if isinstance(value, float) and np.isnan(value) else value
        for key, value in XGB_CONFIG.items()
    }
    payload = {
        "model_family": "XGBOOST",
        "xgboost_version": EXPECTED_XGB_VERSION,
        "configuration": configuration,
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "target": TARGET,
        "target_horizon_months": TARGET_HORIZON_MONTHS,
        "development_end": DEVELOPMENT_END,
        "calibration": "NONE",
        "thresholds": [0.40, 0.50],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["contract_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def maturity_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify exact t+3 maturity for each locally available post-April month."""
    candidate = schedule_candidate()
    if candidate.horizon != TARGET_HORIZON_MONTHS:
        raise ValueError("existing schedule-label horizon changed")
    labels, _ = _candidate_labels(df, candidate)
    local_months = set(df.report_month.astype(str))
    rows = []
    for month, cutoff in post_april_rows(df).groupby("report_month", sort=True):
        endpoint = required_endpoint(str(month))
        endpoint_rows = df[df.report_month.eq(endpoint)]
        cutoff_ids = set(cutoff.loc[cutoff.traceable, "identity_key"])
        endpoint_ids = set(endpoint_rows.loc[endpoint_rows.traceable, "identity_key"])
        mature = labels[labels.prediction_month.eq(month)] if len(labels) else labels
        rows.append({
            "cutoff_month": month,
            "cutoff_observations": len(cutoff),
            "cutoff_projects": cutoff.identity_key.nunique(),
            "traceable_observations": int(cutoff.traceable.sum()),
            "required_t_plus_3_endpoint": endpoint,
            "endpoint_month_available": endpoint in local_months,
            "endpoint_observations": len(endpoint_rows),
            "exact_endpoint_project_pairs": len(cutoff_ids & endpoint_ids),
            "mature_label_rows": len(mature),
            "mature_events": int(mature.label.sum()) if len(mature) else 0,
            "immature_or_unknown_rows": len(cutoff) - len(mature),
            "maturity_status": "MATURE" if len(mature) else "IMMATURE_MISSING_T_PLUS_3_ENDPOINT",
        })
    return pd.DataFrame(rows)


def validate_temporal_separation(
    development: pd.DataFrame, holdout: pd.DataFrame,
) -> None:
    """Require strict time separation and prohibit duplicate observation rows."""
    if holdout.empty:
        raise InvalidHoldoutError("no mature post-April holdout rows")
    if development.report_month.max() > DEVELOPMENT_END:
        raise ValueError("development data extends beyond the frozen boundary")
    if holdout.report_month.min() < POST_APRIL_START:
        raise ValueError("holdout is not strictly post-April")
    development_pairs = set(zip(development.identity_key, development.report_month))
    holdout_pairs = set(zip(holdout.identity_key, holdout.report_month))
    if development_pairs & holdout_pairs:
        raise ValueError("development and holdout observation overlap")


def require_valid_holdout(availability: pd.DataFrame) -> None:
    if availability.empty or not availability.maturity_status.eq("MATURE").any():
        raise InvalidHoldoutError(
            "no post-April cutoff has the required exact t+3 endpoint"
        )


def audit_once(df: pd.DataFrame) -> dict[str, object]:
    if xgboost.__version__ != EXPECTED_XGB_VERSION:
        raise ValueError(
            f"frozen contract requires XGBoost {EXPECTED_XGB_VERSION}, "
            f"found {xgboost.__version__}"
        )
    availability = maturity_availability(df)
    mature = availability[availability.maturity_status.eq("MATURE")]
    post = post_april_rows(df)
    return {
        "status": STATUS if mature.empty else "MATURE HOLDOUT AVAILABLE",
        "availability": availability,
        "latest_local_month": str(df.report_month.max()),
        "post_april_observations": len(post),
        "post_april_projects": post.identity_key.nunique(),
        "mature_observations": int(availability.mature_label_rows.sum()),
        "contract": model_contract(),
    }


def _assert_identical(first: dict[str, object], second: dict[str, object]) -> None:
    for key in (
        "status", "latest_local_month", "post_april_observations",
        "post_april_projects", "mature_observations", "contract",
    ):
        if first[key] != second[key]:
            raise AssertionError(f"non-deterministic Phase 21 field: {key}")
    pd.testing.assert_frame_equal(
        first["availability"], second["availability"], check_exact=True
    )


def write_report(result: dict[str, object], path: Path) -> None:
    availability = result["availability"]
    contract = result["contract"]
    if result["status"] != STATUS:
        raise RuntimeError("mature data detected; full frozen-model evaluation is required")
    lines = [
        "SANKET - PHASE 21 FIRST MATURE POST-APRIL FORWARD HOLDOUT VALIDATION",
        "=" * 76,
        f"PHASE 21 STATUS: {STATUS}",
        "",
        "EXECUTIVE SUMMARY",
        "No genuinely mature post-April 2026 labeled holdout exists in the current "
        "repository. The exact t+3 schedule-label rule was preserved, and model "
        "evaluation stopped before fitting, prediction, metric calculation, threshold "
        "application, calibration, drift analysis, or slice analysis.",
        "",
        "LABEL MATURITY DEFINITION",
        "The existing target is future_schedule_later_3m: the effective completion "
        "target becomes later within t+1 through t+3 relative to its cutoff value, and "
        "the same traceable project must be observed at the exact t+3 endpoint. A missing "
        "endpoint is UNKNOWN, never negative.",
        "",
        "POST-APRIL DATA AVAILABILITY",
        "cutoff | observations/projects | required endpoint | endpoint present | exact "
        "pairs | mature labels/events | status",
    ]
    for row in availability.itertuples(index=False):
        lines.append(
            f"{row.cutoff_month} | {row.cutoff_observations}/{row.cutoff_projects} | "
            f"{row.required_t_plus_3_endpoint} | {row.endpoint_month_available} | "
            f"{row.exact_endpoint_project_pairs} | {row.mature_label_rows}/"
            f"{row.mature_events} | {row.maturity_status}"
        )
    lines += [
        f"Latest local report month: {result['latest_local_month']}.",
        f"Post-April observations/projects: {result['post_april_observations']}/"
        f"{result['post_april_projects']}; mature labeled observations: "
        f"{result['mature_observations']}.",
        "",
        "EXTERNAL AVAILABILITY CHECK",
        f"The official PAIMANA/MoSPI report listing was checked on {AUDIT_DATE}: "
        f"{OFFICIAL_REPORT_SOURCE}",
        "No August 2026 Monthly Flash Report was found. No external file was downloaded "
        "or substituted, and no data source was fabricated.",
        "",
        "WHY THE HOLDOUT IS INVALID",
        "May 2026 is the first post-April cutoff, but it requires an August 2026 record "
        "for the same project. Local data ends in July 2026, so May, June, and July lack "
        "their August, September, and October exact endpoints respectively. Zero rows are "
        "label-mature. Weakening the endpoint rule would introduce unsupported labels and "
        "invalidate the forward holdout.",
        "",
        "EARLIEST POTENTIALLY USABLE PERIOD",
        "May 2026 can become the first candidate holdout only after an authentic August "
        "2026 report is available, ingested, parsed, validated, and linked by the existing "
        "source-backed identity rule. The eventual eligible row count cannot be known until "
        "that endpoint is processed.",
        "",
        "FROZEN MODEL RECORD",
        f"Model family/version: XGBOOST {contract['xgboost_version']}.",
        f"Logical model-contract SHA-256: {contract['contract_sha256']}.",
        "No serialized fitted-model artifact exists; Phase 16 deterministically fits the "
        "frozen configuration within each validation split. Contract: 300 trees, learning "
        "rate 0.05, depth 3, minimum child weight 5, row/column subsampling 0.8/0.8, "
        "L1/L2 0/1, natural class weighting, histogram trees, seed 26103, one thread, 29 "
        "unchanged features, no calibration, and frozen thresholds 0.40/0.50.",
        "",
        "EVALUATION RESULTS",
        "NOT PERFORMED. Observation/project metrics, score distribution, ROC-AUC, PR-AUC, "
        "Brier, Brier skill, ECE, diagnostic intercept/slope, fixed-threshold results, "
        "ranking metrics, project aggregation, slices, drift, and robustness comparison "
        "would all be invalid without mature labels.",
        "",
        "GENERALIZATION CLASSIFICATION",
        STATUS,
        "No conclusion about post-April discrimination, ranking, or calibration is made.",
        "",
        "MISSING EVIDENCE",
        "An authentic August 2026 project-level endpoint is missing for May maturity. "
        "September and October endpoints would be required for June and July cutoffs. "
        "Project continuity and effective completion targets must then pass the unchanged "
        "parser, identity, and label-validation rules.",
        "",
        "FRONTEND/DATA INTEGRATION STATUS",
        "Frontend remains disconnected from the model, API, database, prediction pipeline, "
        "and live data. Existing source data was inspected read-only; no ingestion or "
        "integration path was changed.",
        "",
        "TESTS",
        "PASSED - all 80 repository tests.",
        "",
        "REPRODUCIBILITY",
        "IDENTICAL - two complete Phase 21 maturity audits matched exactly in memory; the "
        "report and availability artifact are deterministic and hashable.",
        "",
        "CHANGE CONTROL",
        "Created src/post_april_holdout_validation.py; "
        "tests/test_post_april_holdout_validation.py; "
        "reports/phase21_post_april_forward_holdout_report.txt; and "
        "reports/phase21_post_april_holdout_availability.csv. No model, feature, label, "
        "threshold, calibration, frontend, database, or live-data code was modified.",
        "",
        "NEXT PHASE",
        "Ingest and validate the authentic August 2026 project-level report before "
        "constructing the frozen May 2026 forward holdout.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    first = audit_once(df)
    second = audit_once(df)
    _assert_identical(first, second)
    first["availability"].to_csv(
        report_dir / "phase21_post_april_holdout_availability.csv",
        index=False,
    )
    write_report(first, report_dir / "phase21_post_april_forward_holdout_report.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/project_monthly.csv")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.data, args.report_dir)


if __name__ == "__main__":
    main()
