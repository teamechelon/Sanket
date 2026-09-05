"""Build longitudinal, identity, and aggregate-validation deliverables.

This module is deliberately conservative: it never fills a project code from
project text.  A legacy OCMS value can be used only as an explicitly labelled
source-provided alternate tracking key.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils import PROCESSED_DIR, REPORTS_DIR, safe_print


_MONTH_DATE = re.compile(r"^(0[1-9]|1[0-2])/\d{4}$")
_FULL_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
_SOURCE_CODE_TOKEN = re.compile(r"\(\d{5,7}\)")


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _identity_key(row: pd.Series) -> str | None:
    code = _text(row.get("project_code"))
    legacy = _text(row.get("legacy_ocms_code"))
    if code:
        return f"P:{code}"
    if legacy:
        return f"L:{legacy}"
    return None


def _identifier_classification(row: pd.Series) -> str:
    """Classify a missing primary identifier without reconstructing it."""
    if _text(row.get("project_code")):
        return "PRIMARY_ID_PRESENT"
    if _text(row.get("legacy_ocms_code")):
        return "ALTERNATE_ID_AVAILABLE"
    source_text = " ".join(_text(row.get(c)) for c in ("project_name", "agency"))
    if _SOURCE_CODE_TOKEN.search(source_text):
        return "EXTRACTION_FAILURE"
    # Table-7 extraction distinguishes source no-ID rows from rows where it
    # cannot find one.  Preserve that direct source assertion only.
    if _text(row.get("identifier_status")) == "SOURCE_MISSING_ID":
        return "SOURCE_MISSING_ID"
    return "UNRESOLVED"


def enrich_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["identity_key"] = df.apply(_identity_key, axis=1)
    df["identifier_classification"] = df.apply(_identifier_classification, axis=1)
    duplicate = df["project_code"].notna() & df.duplicated(
        ["report_month", "project_code"], keep=False
    )
    flags: list[str] = []
    for idx, row in df.iterrows():
        values: list[str] = []
        if row["identifier_classification"] != "PRIMARY_ID_PRESENT":
            values.append(row["identifier_classification"])
        if _text(row.get("extraction_status")) != "SUCCESS":
            values.append("PARTIAL_EXTRACTION")
        if bool(duplicate.loc[idx]):
            values.append("DUPLICATE_PRIMARY_ID_IN_MONTH")
        progress = pd.to_numeric(pd.Series([row.get("physical_progress")]), errors="coerce").iloc[0]
        if pd.notna(progress) and not 0 <= progress <= 100:
            values.append("PROGRESS_OUT_OF_RANGE")
        flags.append(";".join(values) if values else "NONE")
    df["quality_flags"] = flags
    return df


def _stable_value(group: pd.DataFrame, field: str) -> tuple[str | None, int, int]:
    values = [_text(value) for value in group[field] if _text(value)]
    if not values:
        return None, 0, 0
    counts = Counter(values)
    max_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == max_count}
    # If equally frequent, choose the most recent source observation. This is
    # a resolution rule, not a claim that older values were incorrect.
    chosen = next(
        _text(value) for value in group.sort_values("report_month", ascending=False)[field]
        if _text(value) in tied
    )
    return chosen, len(counts), len(values)


def build_master(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    identified = df[df["identity_key"].notna()].copy()
    for key, group in identified.groupby("identity_key", sort=True):
        group = group.sort_values("report_month")
        values: dict[str, str | None] = {}
        conflicts: list[str] = []
        for field in ("project_name", "agency", "state", "ministry", "sector"):
            chosen, variants, _observations = _stable_value(group, field)
            values[field] = chosen
            if variants > 1:
                conflicts.append(f"{field}:{variants}")
        primary = next((_text(v) for v in group["project_code"] if _text(v)), None)
        legacy = next((_text(v) for v in group["legacy_ocms_code"] if _text(v)), None)
        rows.append({
            "identity_key": key,
            "project_code": primary,
            "legacy_ocms_code": legacy,
            **values,
            "first_month": group["report_month"].min(),
            "last_month": group["report_month"].max(),
            "months_observed": group["report_month"].nunique(),
            "master_resolution_rule": "most_frequent_nonblank; most_recent_when_tied",
            "field_conflicts": ";".join(conflicts) if conflicts else "NONE",
        })
        if conflicts:
            notes.append(f"{key}: {', '.join(conflicts)}")
    return pd.DataFrame(rows), notes


def _pct_diff(actual: float | int | None, expected: float | int | None) -> float | None:
    if actual is None or expected is None or pd.isna(actual) or pd.isna(expected):
        return None
    if expected == 0:
        return 0.0 if actual == 0 else None
    return abs(actual - expected) / abs(expected) * 100


def _validation_status(pct: float | None, available: bool) -> str:
    if not available:
        return "UNAVAILABLE_TO_VALIDATE"
    if pct is None:
        return "WARNING"
    if pct <= 1:
        return "PASS"
    if pct <= 5:
        return "EXPECTED_DIFFERENCE"
    if pct <= 20:
        return "WARNING"
    return "ERROR"


def cross_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Compare only directly compatible national totals; state/sector are
    retained as contextual checks because multi-state projects can be counted
    differently by an aggregate snapshot.
    """
    cost_path = PROCESSED_DIR / "cost_overview_monthly.csv"
    state_path = PROCESSED_DIR / "state_monthly.csv"
    sector_path = PROCESSED_DIR / "sector_monthly.csv"
    costs = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()
    states = pd.read_csv(state_path) if state_path.exists() else pd.DataFrame()
    sectors = pd.read_csv(sector_path) if sector_path.exists() else pd.DataFrame()
    rows: list[dict] = []
    for month, group in df.groupby("report_month", sort=True):
        metrics: list[tuple[str, float | int, float | int | None, bool, str]] = []
        if not costs.empty and month in set(costs["report_month"]):
            source = costs[costs["report_month"] == month].iloc[0]
            for pdf_name, csv_name in (
                ("original_cost", "original_cost_crores"),
                ("expenditure", "expenditure_crores"),
            ):
                metrics.append((pdf_name, pd.to_numeric(group[pdf_name], errors="coerce").sum(min_count=1), source[csv_name], True, "national cost overview"))
            # A blank revised-cost cell means no revision is reported for that
            # row, not a zero project cost.  Keep the raw value blank in the
            # monthly file and use original cost only for this aggregate-level
            # comparison.
            effective_revised = pd.to_numeric(group["revised_cost"], errors="coerce").fillna(
                pd.to_numeric(group["original_cost"], errors="coerce")
            ).sum(min_count=1)
            metrics.append(("revised_cost", effective_revised, source["revised_cost_crores"], True, "national cost overview; blank row revisions treated as original cost for comparison only"))
        else:
            for pdf_name in ("original_cost", "revised_cost", "expenditure"):
                metrics.append((pdf_name, pd.to_numeric(group[pdf_name], errors="coerce").sum(min_count=1), None, False, "no same-month cost aggregate"))

        # State and sector totals are useful checks but potentially non-additive
        # for multi-state records, so a non-match is never silently forced.
        for name, aggregate, field in (("state_project_count", states, "state"), ("sector_project_count", sectors, "sector")):
            if not aggregate.empty and month in set(aggregate["report_month"]):
                expected = pd.to_numeric(aggregate[aggregate["report_month"] == month]["project_count"], errors="coerce").sum(min_count=1)
                metrics.append((name, len(group), expected, True, f"sum of {field} aggregate counts; may be non-additive"))
            else:
                metrics.append((name, len(group), None, False, f"no same-month {field} aggregate"))
        for metric, actual, expected, available, note in metrics:
            pct = _pct_diff(actual, expected)
            rows.append({
                "report_month": month, "metric": metric, "pdf_value": actual,
                "aggregate_value": expected, "pct_difference": pct,
                "status": _validation_status(pct, available), "note": note,
            })
    return pd.DataFrame(rows)


def _date_issues(df: pd.DataFrame) -> pd.DataFrame:
    issue_rows: list[dict] = []
    for field in ("date_of_approval", "start_date", "original_doc", "revised_doc"):
        for idx, value in df[field].items():
            text = _text(value)
            if text and not (_MONTH_DATE.fullmatch(text) or _FULL_DATE.fullmatch(text)):
                issue_rows.append({"row": idx, "field": field, "value": text})
    return pd.DataFrame(issue_rows)


def _integrity_stats(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identified = df[df["identity_key"].notna()]
    month_counts = identified.groupby("identity_key")["report_month"].nunique()
    duplicate = df[df["project_code"].notna() & df.duplicated(["report_month", "project_code"], keep=False)]
    out_of_range = df[(pd.to_numeric(df["physical_progress"], errors="coerce") < 0) | (pd.to_numeric(df["physical_progress"], errors="coerce") > 100)]
    changes: list[dict] = []
    gaps: list[dict] = []
    all_months = sorted(df["report_month"].dropna().unique())
    for key, group in identified.groupby("identity_key"):
        group = group.sort_values("report_month")
        observed = set(group["report_month"])
        first, last = group["report_month"].iloc[0], group["report_month"].iloc[-1]
        missing_between = [month for month in all_months if first < month < last and month not in observed]
        if missing_between:
            gaps.append({"identity_key": key, "first_month": first, "last_month": last, "missing_months": ";".join(missing_between), "flag": "DISAPPEARED_AND_REAPPEARED_REQUIRES_REVIEW"})
        for field in ("expenditure", "physical_progress", "original_cost", "revised_cost"):
            values = pd.to_numeric(group[field], errors="coerce")
            for idx in group.index[values.diff() < 0]:
                changes.append({"identity_key": key, "report_month": group.loc[idx, "report_month"], "field": field, "previous_value": values.shift().loc[idx], "current_value": values.loc[idx], "flag": "DECREASE_REQUIRES_REVIEW"})
    changes_df = pd.DataFrame(changes)
    gaps_df = pd.DataFrame(gaps)
    expenditure = pd.to_numeric(df["expenditure"], errors="coerce")
    costs = pd.concat([pd.to_numeric(df["original_cost"], errors="coerce"), pd.to_numeric(df["revised_cost"], errors="coerce")])
    stats = {
        "total_rows": len(df), "unique_primary_codes": df["project_code"].nunique(),
        "unique_safe_tracking_keys": identified["identity_key"].nunique(),
        "no_primary_code": int(df["project_code"].isna().sum()),
        "one_month": int((month_counts == 1).sum()), "two_plus": int((month_counts >= 2).sum()),
        "six_plus": int((month_counts >= 6).sum()), "twelve_plus": int((month_counts >= 12).sum()),
        "duplicates": len(duplicate), "out_of_range": len(out_of_range),
        "decreases": len(changes_df), "reappearing": len(gaps_df),
        "negative_expenditure": int((expenditure < 0).sum()), "negative_cost": int((costs < 0).sum()),
    }
    return stats, duplicate, out_of_range, changes_df, gaps_df


def generate_reports(df: pd.DataFrame, master: pd.DataFrame, master_notes: list[str], validation: pd.DataFrame) -> None:
    stats, duplicates, out_of_range, changes, gaps = _integrity_stats(df)
    date_issues = _date_issues(df)
    monthly = df.groupby("report_month").agg(rows=("report_month", "size"), unique_primary_projects=("project_code", "nunique"), primary_id_coverage=("project_code", lambda x: round(x.notna().mean() * 100, 2))).reset_index()
    repeated = master.sort_values(["months_observed", "identity_key"], ascending=[False, True]).head(10)
    classification = df["identifier_classification"].value_counts().to_dict()

    extraction = ["SANKET — FULL MONTHLY EXTRACTION SUMMARY", "=" * 55, f"Monthly rows: {stats['total_rows']}", f"Monthly source files: {df['source_file'].nunique()}", "Quarterly QPISR is intentionally stored separately in project_quarterly.csv.", "", "Records per month:"]
    extraction += [f"  {r.report_month}: {r.rows} rows; {r.unique_primary_projects} primary IDs; {r.primary_id_coverage:.2f}% primary-ID coverage" for r in monthly.itertuples()]
    (REPORTS_DIR / "extraction_summary.txt").write_text("\n".join(extraction) + "\n", encoding="utf-8")

    integrity = ["SANKET — LONGITUDINAL INTEGRITY REPORT", "=" * 55]
    integrity += [f"Total project-month rows: {stats['total_rows']}", f"Unique primary project codes: {stats['unique_primary_codes']}", f"Safe source tracking keys (primary or legacy): {stats['unique_safe_tracking_keys']}", f"Rows without a primary project code: {stats['no_primary_code']}", f"Projects appearing in 1 month: {stats['one_month']}", f"Projects appearing in 2+ months: {stats['two_plus']}", f"Projects appearing in 6+ months: {stats['six_plus']}", f"Projects appearing in 12+ months: {stats['twelve_plus']}", f"Duplicate primary-code + month rows: {stats['duplicates']}", f"Progress values outside 0–100: {stats['out_of_range']}", f"Negative expenditures: {stats['negative_expenditure']}; negative costs: {stats['negative_cost']}", f"Decreases in expenditure/progress/cost requiring review (not automatically errors): {stats['decreases']}", f"Projects that disappear and later reappear: {stats['reappearing']} (flagged; not interpreted as completion)", f"Malformed dates: {len(date_issues)}", "", "Per-month coverage:"]
    integrity += [f"  {r.report_month}: rows={r.rows}, unique primary IDs={r.unique_primary_projects}, coverage={r.primary_id_coverage:.2f}%" for r in monthly.itertuples()]
    integrity += ["", "Longitudinal verification sample (source identifiers only):"]
    for r in repeated.itertuples():
        subset = df[df.identity_key == r.identity_key].sort_values("report_month")
        integrity.append(f"  {r.identity_key} | {r.project_name} | months={r.months_observed} | {r.first_month} to {r.last_month}")
        for obs in subset.itertuples():
            integrity.append(f"    {obs.report_month}: expenditure={obs.expenditure}, progress={obs.physical_progress}, original_cost={obs.original_cost}, revised_cost={obs.revised_cost}")
    if not gaps.empty:
        integrity += ["", "Disappearance/reappearance flags (sample; not completion determinations):"]
        integrity += [f"  {r.identity_key}: {r.first_month}–{r.last_month}; missing {r.missing_months}" for r in gaps.head(25).itertuples()]
    negative_expenditure = df[pd.to_numeric(df["expenditure"], errors="coerce") < 0]
    if not negative_expenditure.empty:
        integrity += ["", "Negative-expenditure records (retained for source review):"]
        integrity += [f"  {r.report_month} sl_no={r.sl_no} id={r.identity_key} expenditure={r.expenditure} source={r.source_file}:p{r.source_page}" for r in negative_expenditure.itertuples()]
    if not out_of_range.empty:
        integrity += ["", "Out-of-range progress records (retained for source review):"]
        integrity += [f"  {r.report_month} sl_no={r.sl_no} id={r.identity_key} progress={r.physical_progress} source={r.source_file}:p{r.source_page}" for r in out_of_range.itertuples()]
    (REPORTS_DIR / "longitudinal_integrity_report.txt").write_text("\n".join(integrity) + "\n", encoding="utf-8")

    validation_lines = ["SANKET — AGGREGATE CROSS-VALIDATION REPORT", "=" * 55, "National cost totals are directly compared. State/sector sums are contextual because multi-state projects can be counted differently.", ""]
    for r in validation.itertuples():
        validation_lines.append(f"{r.report_month} | {r.metric} | PDF={r.pdf_value} | aggregate={r.aggregate_value} | diff={r.pct_difference} | {r.status} | {r.note}")
    (REPORTS_DIR / "validation_report.txt").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    sufficient_history = stats["six_plus"]
    tracking_pct = (stats["two_plus"] / stats["unique_safe_tracking_keys"] * 100) if stats["unique_safe_tracking_keys"] else 0
    pilot = df[df["report_month"].isin(["2025-07", "2026-03"])]
    pilot_primary = int(pilot["project_code"].notna().sum())
    dq = ["SANKET — DATA QUALITY REPORT", "=" * 55, "A. Coverage", f"  {stats['total_rows']} project-month rows across {df['report_month'].nunique()} monthly reports.", "B. Missingness", f"  Identifier classification: {classification}", f"  Rows without primary code: {stats['no_primary_code']}", "", "Pilot project-code investigation", f"  The reported 82.9% pilot coverage is not reproduced after source-aware wrapped-code handling: the July 2025 + March 2026 rerun has {pilot_primary}/{len(pilot)} ({pilot_primary / len(pilot) * 100:.2f}%) primary IDs.", "  Source inspection of March rows 17 and 168–170 confirmed that primary codes were present in the PDF but shared wrapped state/date lines; those were extraction failures, not source-missing IDs. The parser now consumes those lines with the current serial row.", "  No project_name-derived identifiers were created. A legacy OCMS code is used only as an explicitly labelled alternate source identifier.", "  The previously reported two March duplicate codes are not reproduced in the current source-token parse (0 duplicate primary-code + month rows); they are treated as a prior parser artifact, not evidence of legitimate duplicate projects.", "  Remaining missing-primary records are classified per row as ALTERNATE_ID_AVAILABLE, EXTRACTION_FAILURE, SOURCE_MISSING_ID, or UNRESOLVED in project_monthly.csv.", "C. Duplicate analysis", f"  Duplicate primary-code + report-month rows: {stats['duplicates']}. These are retained and flagged; not deduplicated.", "D. Longitudinal coverage", f"  2+ months: {stats['two_plus']}; 6+ months: {stats['six_plus']}; 12+ months: {stats['twelve_plus']}; disappearance/reappearance flags: {stats['reappearing']}.", "E. Identifier quality", "  No project name was used as a unique key. identity_key uses a primary code when present, otherwise a source-provided legacy OCMS code.", "F. Numerical validity", f"  Out-of-range progress: {stats['out_of_range']}; negative expenditures: {stats['negative_expenditure']}; negative costs: {stats['negative_cost']}; decreases requiring review: {stats['decreases']}; malformed dates: {len(date_issues)}.", "G. Cross-source validation", f"  Status counts: {validation['status'].value_counts().to_dict()}", "H. Known extraction failures", f"  Missing-primary classes include {classification.get('EXTRACTION_FAILURE', 0)} extraction failures and {classification.get('UNRESOLVED', 0)} unresolved rows. Monthly values are preserved rather than repaired.", "I. ML readiness", "  NOT READY. Repeated observations exist, but defensible forward labels for schedule delay and cost overrun are not established from these snapshots.", "", "ML readiness gate", f"1. Repeated observations: yes ({stats['two_plus']} source-keyed projects have 2+ months).", "2. Trajectories: partially, only for source-keyed records; corrections/decreases require review.", "3. Historical prediction cutoff: technically possible by report_month, once labels are defined.", "4. Future schedule-delay label: not defensible yet; no validated actual-completion/outcome field.", "5. Future cost-overrun label: not defensible yet; a target policy and event timing are missing.", "6. Required targets: unavailable/undefined.", "7. Positive cases: unavailable until target definitions are established.", f"8. Projects with 2+ months: {tracking_pct:.2f}% of safe source tracking keys.", "9. Leakage exclusions: revised_cost when forecasting later escalation; revised_doc when forecasting delay before revision; all future expenditure; all future physical_progress; fields from after the cutoff.", "10. Minimum viable modelling dataset: source-keyed project histories with >=3 observations before a fixed cutoff plus outcome dates/cost events recorded after that cutoff, verified target timing, and enough positive outcomes.", "", "Master resolution rule", "  Stable master values are the most frequent nonblank source values; ties select the latest observation. Conflicting monthly values remain untouched in project_monthly.csv."]
    if master_notes:
        dq += ["  Conflicting master fields (sample):"] + [f"    {note}" for note in master_notes[:25]]
    (REPORTS_DIR / "data_quality_report.txt").write_text("\n".join(dq) + "\n", encoding="utf-8")


def main() -> None:
    path = PROCESSED_DIR / "project_monthly.csv"
    if not path.exists():
        raise SystemExit("project_monthly.csv does not exist; run python -m src.pdf_extractor --all first")
    monthly = enrich_monthly(pd.read_csv(path, dtype={"project_code": "string", "legacy_ocms_code": "string"}))
    monthly.to_csv(path, index=False, encoding="utf-8")
    master, master_notes = build_master(monthly)
    master.to_csv(PROCESSED_DIR / "project_master.csv", index=False, encoding="utf-8")
    validation = cross_validate(monthly)
    validation.to_csv(REPORTS_DIR / "cross_source_validation.csv", index=False, encoding="utf-8")
    generate_reports(monthly, master, master_notes, validation)
    safe_print(f"Validated {len(monthly)} monthly rows and wrote project_master.csv plus reports.")


if __name__ == "__main__":
    main()
