"""Phase 9 semantic and temporal validation for the first ML targets.

This module validates labels; it deliberately does not train a model.  The
primary invariant is that a zero is assigned only when the project is present
at the exact horizon endpoint.  A missing project-month remains unknown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.label_feasibility import CANDIDATES, _candidate_labels, load_data
from src.utils import PROCESSED_DIR, REPORTS_DIR


def _candidate(name: str):
    return next(item for item in CANDIDATES if item.target_name == name)


def _timeline_by_number(df: pd.DataFrame) -> dict[tuple[str, int], object]:
    traceable = df[df["traceable"]].drop_duplicates(["identity_key", "month_number"])
    return {(r.identity_key, int(r.month_number)): r for r in traceable.itertuples(index=False)}


def _target_text(value: object) -> str:
    return "" if pd.isna(value) else pd.Period(value, freq="M").strftime("%m/%Y")


def _first_later_change_month(df: pd.DataFrame, horizon: int, minimum_shift: int = 1) -> pd.DataFrame:
    """Build labels without returning or retaining future feature values."""
    lookup = _timeline_by_number(df)
    rows: list[dict[str, object]] = []
    for cutoff in df[df["traceable"]].itertuples(index=False):
        month = int(cutoff.month_number)
        endpoint = lookup.get((cutoff.identity_key, month + horizon))
        if endpoint is None or pd.isna(cutoff.effective_end) or pd.isna(endpoint.effective_end):
            continue
        first = ""
        label = 0
        for offset in range(1, horizon + 1):
            future = lookup.get((cutoff.identity_key, month + offset))
            if future is None or pd.isna(future.effective_end):
                continue
            shift = int(future.effective_end.ordinal - cutoff.effective_end.ordinal)
            if shift >= minimum_shift:
                label, first = 1, future.report_month
                break
        rows.append({"identity_key": cutoff.identity_key, "prediction_month": cutoff.report_month,
                     "label": label, "first_change_month": first})
    return pd.DataFrame(rows)


def march_later_events(df: pd.DataFrame) -> pd.DataFrame:
    """All later revised-target transitions first published in March 2026.

    The comparison uses the latest prior observation (February for 441 rows and
    January for six rows), which reproduces the 447-event Phase 8 finding.
    """
    records: list[dict[str, object]] = []
    monthly = {m: g.set_index("identity_key") for m, g in df.groupby("report_month")}
    for identity, group in df[df["traceable"]].sort_values("month_number").groupby("identity_key"):
        march = group[group["report_month"] == "2026-03"]
        prior = group[group["report_month"] < "2026-03"]
        if march.empty or prior.empty:
            continue
        current, previous = march.iloc[-1], prior.iloc[-1]
        # Reproduce the Phase 8 anomaly definition: revised-to-revised changes.
        # First appearance of a revised value is not counted as a transition.
        if pd.isna(current.revised_end) or pd.isna(previous.revised_end) or current.revised_end <= previous.revised_end:
            continue
        def at(month: str) -> str:
            table = monthly.get(month)
            if table is None or identity not in table.index:
                return ""
            row = table.loc[identity]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return _target_text(row.effective_end)
        records.append({
            "project_code": current.project_code, "identity_key": identity,
            "project_name": current.project_name, "prediction_month": previous.report_month,
            "cutoff_target": _target_text(previous.revised_end), "future_target": _target_text(current.revised_end),
            "target_change_month": "2026-03", "shift_months": int(current.revised_end.ordinal - previous.revised_end.ordinal),
            "original_end_date": current.original_doc, "revised_end_date": current.revised_doc,
            "december_2025_target": at("2025-12"), "january_2026_target": at("2026-01"),
            "february_2026_target": at("2026-02"), "previous_report_target": _target_text(previous.revised_end),
            "march_2026_target": at("2026-03"), "april_2026_target": at("2026-04"),
            "next_report_target": at("2026-04"), "source_file": current.source_file,
            "source_page": current.source_page, "source_section": current.source_section, "label_value": 1,
        })
    return pd.DataFrame(records).sort_values(["project_code", "identity_key"]).reset_index(drop=True)


def schedule_candidates(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("future_schedule_later_3m", 3, 1, "Any effective target at least 1 month later within t+1..t+3; exact t+3 observation required."),
        ("future_schedule_later_6m", 6, 1, "Any effective target at least 1 month later within t+1..t+6; exact t+6 observation required."),
        ("future_schedule_overdue", 3, None, "Project remains listed at t+3 and cutoff-known target is earlier than t+3 report month."),
        ("future_schedule_shift_ge_3m", 3, 3, "Effective target moves at least 3 months later within t+1..t+3; exact t+3 observation required."),
    ]
    rows = []
    for name, horizon, shift, definition in specs:
        if shift is None:
            labels, unknown = _candidate_labels(df, _candidate("future_overdue_active_3m"))
            unknown += int((~df["traceable"]).sum())
        else:
            labels = _first_later_change_month(df, horizon, shift)
            unknown = len(df) - len(labels)
        positive = int(labels["label"].sum())
        march = int(labels.loc[(labels["label"] == 1) & (labels["first_change_month"] == "2026-03")].shape[0]) if "first_change_month" in labels else 0
        rows.append({"target_name": name, "definition": definition, "prediction_horizon": f"{horizon} months",
                     "eligible_rows": len(labels), "positive_rows": positive, "negative_rows": len(labels)-positive,
                     "unknown_rows": unknown, "positive_rate": round(positive/len(labels), 6),
                     "projects_covered": labels["identity_key"].nunique(), "march_2026_concentration": march,
                     "semantic_confidence": "LOW" if shift is None else "MEDIUM", "leakage_risk": "CONDITIONAL", "status": "CONDITIONAL"})
    return pd.DataFrame(rows)


def run(input_path: Path, output_dir: Path) -> None:
    """Recompute deterministic tabular validations; reviewed narratives are versioned separately."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    candidates = schedule_candidates(df)
    events = march_later_events(df)
    candidates.to_csv(output_dir / "schedule_target_validation.csv", index=False)
    review = output_dir / "schedule_label_march_2026_review.csv"
    if review.exists():
        prior_review = pd.read_csv(review, dtype={"project_code": "string"})
        audit = prior_review[["identity_key", "sampled_for_source_review", "source_months_verified", "review_status", "notes"]]
        events = events.merge(audit, on="identity_key", how="left")
    events.to_csv(review, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "project_monthly.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
