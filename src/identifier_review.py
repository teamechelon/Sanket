"""Source-preserving identifier reconciliation reports.

This module deliberately separates a primary project code from the alternate
identifier printed by the source.  It reviews only observations which have
neither identifier; rows with an N/O-prefixed OCMS value remain traceable but
are not silently promoted to primary project codes.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.pdf_extractor import detect_report_format
from src.utils import PROCESSED_DIR, RAW_PDF_DIR, REPORTS_DIR, safe_print


PRIMARY_CODE = re.compile(r"\((\d{5,7})\)")


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"project_code": "string", "legacy_ocms_code": "string"},
    )


def _handler(row: pd.Series) -> str:
    report_format = detect_report_format(
        RAW_PDF_DIR / _text(row["source_file"]), _text(row["report_month"])
    )
    return {
        "EARLY_FORMAT": "extract_early_format/extract_table7",
        "QUARTERLY": "extract_quarterly_format/extract_table7",
        "NEW_FORMAT_B": "extract_new_format_b/extract_table6",
    }.get(report_format, "extract_table6")


def _review_population(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows lacking both a primary and a source alternate identifier."""
    return df[df["project_code"].isna() & df["legacy_ocms_code"].isna()].copy()


def _candidate_lookup(candidate: pd.DataFrame) -> pd.DataFrame:
    columns = ["report_month", "source_file", "sl_no", "project_code", "legacy_ocms_code"]
    available = candidate[columns].copy()
    # A serial number is an address within a source report.  Do not use name
    # matching: similar project names are not proof of identity.
    return available.drop_duplicates(["report_month", "source_file", "sl_no"])


def _source_token(row: pd.Series) -> str:
    return " ".join(_text(row.get(field)) for field in ("project_name", "agency"))


def build_review_queue(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    missing = _review_population(baseline)
    lookup = _candidate_lookup(candidate).rename(
        columns={
            "project_code": "candidate_project_code",
            "legacy_ocms_code": "candidate_legacy_ocms_code",
        }
    )
    review = missing.merge(lookup, on=["report_month", "source_file", "sl_no"], how="left")
    output: list[dict[str, object]] = []
    for row in review.itertuples(index=False):
        values = row._asdict()
        before_class = _text(values.get("identifier_classification"))
        candidate_code = _text(values.get("candidate_project_code"))
        candidate_alternate = _text(values.get("candidate_legacy_ocms_code"))
        token = _source_token(pd.Series(values))
        source_ref = (
            f"{values['source_file']}:p{values['source_page']}; "
            f"{values['source_section']}; sl_no={values['sl_no']}"
        )
        if candidate_code:
            review_status = "IDENTIFIER_FOUND"
            failure_type = "VERIFIED_PARSER_FAILURE_WRAPPED_OR_BOUNDARY_IDENTIFIER"
            notes = (
                f"Validation extraction recovered source primary code {candidate_code} "
                "at the same report/serial address; the baseline value was blank."
            )
        elif candidate_alternate:
            review_status = "IDENTIFIER_FOUND"
            failure_type = "VERIFIED_PARSER_FAILURE_ALTERNATE_IDENTIFIER"
            notes = (
                f"Validation extraction recovered source alternate identifier "
                f"{candidate_alternate}; it remains alternate and is not a primary code."
            )
        elif before_class == "EXTRACTION_FAILURE" or PRIMARY_CODE.search(token):
            review_status = "PARSER_FAILURE"
            failure_type = "SOURCE_PRIMARY_TOKEN_NOT_RECOVERED"
            notes = (
                "A numeric parenthetical token remains in extracted source text, "
                "but controlled validation did not safely bind it to this serial row."
            )
        elif _text(values.get("identifier_status")) == "SOURCE_MISSING_ID":
            review_status = "VERIFIED_SOURCE_MISSING"
            failure_type = "SOURCE_IDENTIFIER_ABSENT"
            notes = "The source-table handler marked this row as identifier-absent."
        else:
            review_status = "UNRESOLVED"
            failure_type = "AMBIGUOUS_OR_MALFORMED_SOURCE_LAYOUT"
            notes = (
                "No primary or alternate identifier was recovered in controlled "
                "validation; no identity was inferred from project text."
            )
        output.append({
            "report_month": values["report_month"],
            "source_file": values["source_file"],
            "source_page": values["source_page"],
            "source_section": values["source_section"],
            "sl_no": values["sl_no"],
            "project_name": values["project_name"],
            "agency": values["agency"],
            "state": values["state"],
            "legacy_ocms_code": values.get("legacy_ocms_code"),
            "alternate_id": values.get("legacy_ocms_code"),
            "current_project_code": values.get("project_code"),
            "review_status": review_status,
            "failure_type": failure_type,
            "raw_text_reference": source_ref,
            "notes": notes,
        })
    return pd.DataFrame(output)


def _coverage(df: pd.DataFrame) -> dict[str, float | int]:
    total = len(df)
    primary = int(df["project_code"].notna().sum())
    alternate = int(df["legacy_ocms_code"].notna().sum())
    unresolved = int((df["project_code"].isna() & df["legacy_ocms_code"].isna()).sum())
    parser_failure = int((df.get("identifier_classification", pd.Series(index=df.index)) == "EXTRACTION_FAILURE").sum())
    source_missing = int((df.get("identifier_classification", pd.Series(index=df.index)) == "SOURCE_MISSING_ID").sum())
    return {
        "rows": total, "primary": primary, "alternate": alternate,
        "unresolved": unresolved, "parser_failure": parser_failure,
        "source_missing": source_missing,
        "primary_pct": primary / total * 100 if total else 0,
        "alternate_pct": alternate / total * 100 if total else 0,
        "unresolved_pct": unresolved / total * 100 if total else 0,
    }


def _duplicates(df: pd.DataFrame, field: str) -> int:
    usable = df[df[field].notna()]
    return int(usable.duplicated(["report_month", field], keep=False).sum())


def _group_lines(df: pd.DataFrame, fields: list[str]) -> list[str]:
    values = df.groupby(fields, dropna=False).size().reset_index(name="rows")
    values = values.sort_values("rows", ascending=False)
    return [" | ".join(f"{field}={_text(getattr(row, field))}" for field in fields) + f" | rows={row.rows}" for row in values.itertuples(index=False)]


def write_summary(queue: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> None:
    before, after = _coverage(baseline), _coverage(candidate)
    candidate_missing = _review_population(candidate)
    lines = [
        "SANKET — MISSING PRIMARY IDENTIFIER REVIEW SUMMARY",
        "=" * 62,
        f"Review population: {len(queue)} rows lacking both a primary code and a source alternate identifier in the baseline.",
        f"Baseline primary-ID coverage: {before['primary']}/{before['rows']} ({before['primary_pct']:.2f}%).",
        f"Validation primary-ID coverage: {after['primary']}/{after['rows']} ({after['primary_pct']:.2f}%).",
        f"Baseline unresolved-without-any-source-ID: {before['unresolved']}; validation: {after['unresolved']}.",
        f"Baseline duplicate primary code + month: {_duplicates(baseline, 'project_code')}; validation: {_duplicates(candidate, 'project_code')}.",
        f"Baseline duplicate source tracking key + month: {_duplicates(baseline.assign(source_tracking_key=baseline['project_code'].fillna(baseline['legacy_ocms_code'])), 'source_tracking_key')}; validation: {_duplicates(candidate.assign(source_tracking_key=candidate['project_code'].fillna(candidate['legacy_ocms_code'])), 'source_tracking_key')}.",
        "",
        "Review classification:",
    ]
    lines += [f"  {status}: {count}" for status, count in queue["review_status"].value_counts().items()]
    lines += ["", "By report month:"] + [f"  {line}" for line in _group_lines(queue, ["report_month"])]
    lines += ["", "By source file / page:"] + [f"  {line}" for line in _group_lines(queue, ["source_file", "source_page"])[:80]]
    handler_queue = _review_population(baseline).copy()
    handler_queue["format"] = handler_queue.apply(lambda r: detect_report_format(RAW_PDF_DIR / _text(r.source_file), _text(r.report_month)), axis=1)
    handler_queue["handler"] = handler_queue.apply(_handler, axis=1)
    lines += ["", "By PDF format / handler:"] + [f"  {line}" for line in _group_lines(handler_queue, ["format", "handler"])]
    lines += ["", "By source section:"] + [f"  {line}" for line in _group_lines(queue, ["source_section"])]
    lines += ["", "By identifier/failure type:"] + [f"  {line}" for line in _group_lines(queue, ["failure_type"])]
    lines += ["", "Concentration finding:"]
    top = queue.groupby("report_month").size().sort_values(ascending=False).head(3)
    lines.append("  " + "; ".join(f"{month}: {count}" for month, count in top.items()))
    lines.append("  April–June 2025 N-prefixed values are not part of this queue: they are preserved as source alternate identifiers, never renamed as primary codes.")
    (REPORTS_DIR / "missing_id_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keyed = df.copy()
    keyed["project_id"] = keyed["project_code"].fillna(keyed["legacy_ocms_code"])
    keyed = keyed[keyed["project_id"].notna()].copy()
    for project_id, group in keyed.groupby("project_id"):
        group = group.sort_values("report_month")
        for field in ("expenditure", "physical_progress", "original_cost", "revised_cost"):
            values = pd.to_numeric(group[field], errors="coerce")
            for index in group.index[values.diff() < 0]:
                source = group.loc[index]
                rows.append({"report_month": source.report_month, "project_id": project_id, "anomaly_type": f"DECREASE_{field.upper()}", "value": values.loc[index], "previous_value": values.shift().loc[index], "severity": "REVIEW", "source_file": source.source_file, "source_page": source.source_page, "notes": "Source value retained unchanged; decrease is not treated as an error."})
        months = set(group.report_month)
        all_months = sorted(df.report_month.dropna().unique())
        first, last = group.report_month.iloc[0], group.report_month.iloc[-1]
        gaps = [month for month in all_months if first < month < last and month not in months]
        if gaps:
            source = group.iloc[-1]
            rows.append({"report_month": source.report_month, "project_id": project_id, "anomaly_type": "DISAPPEAR_REAPPEAR", "value": ";".join(gaps), "previous_value": "", "severity": "REVIEW", "source_file": source.source_file, "source_page": source.source_page, "notes": "Missing between observed months; not interpreted as completion."})
    for row in df.itertuples(index=False):
        project_id = _text(getattr(row, "project_code", "")) or _text(getattr(row, "legacy_ocms_code", ""))
        expenditure = pd.to_numeric(pd.Series([getattr(row, "expenditure", None)]), errors="coerce").iloc[0]
        progress = pd.to_numeric(pd.Series([getattr(row, "physical_progress", None)]), errors="coerce").iloc[0]
        if pd.notna(expenditure) and expenditure < 0:
            rows.append({"report_month": row.report_month, "project_id": project_id, "anomaly_type": "NEGATIVE_EXPENDITURE", "value": expenditure, "previous_value": "", "severity": "HIGH", "source_file": row.source_file, "source_page": row.source_page, "notes": "Raw source value retained unchanged."})
        if pd.notna(progress) and not 0 <= progress <= 100:
            rows.append({"report_month": row.report_month, "project_id": project_id, "anomaly_type": "PROGRESS_OUT_OF_RANGE", "value": progress, "previous_value": "", "severity": "HIGH", "source_file": row.source_file, "source_page": row.source_page, "notes": "Raw source value retained unchanged."})
        for field in ("date_of_approval", "start_date", "original_doc", "revised_doc"):
            value = _text(getattr(row, field, ""))
            if value and not re.fullmatch(r"(?:0[1-9]|1[0-2])/\d{4}|\d{1,2}/\d{1,2}/\d{4}", value):
                rows.append({"report_month": row.report_month, "project_id": project_id, "anomaly_type": f"MALFORMED_DATE_{field.upper()}", "value": value, "previous_value": "", "severity": "REVIEW", "source_file": row.source_file, "source_page": row.source_page, "notes": "Raw source value retained unchanged."})
    return pd.DataFrame(rows, columns=["report_month", "project_id", "anomaly_type", "value", "previous_value", "severity", "source_file", "source_page", "notes"])


def write_quality_report(queue: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> None:
    before, after = _coverage(baseline), _coverage(candidate)
    candidate_work = candidate.copy()
    candidate_work["format"] = candidate_work.apply(lambda r: detect_report_format(RAW_PDF_DIR / _text(r.source_file), _text(r.report_month)), axis=1)
    candidate_work["handler"] = candidate_work.apply(_handler, axis=1)
    lines = [
        "SANKET — IDENTIFIER QUALITY REPORT",
        "=" * 55,
        f"Total rows: {after['rows']}",
        f"Primary ID: {after['primary']} ({after['primary_pct']:.2f}%) [before {before['primary']} / {before['primary_pct']:.2f}%]",
        f"Alternate ID: {after['alternate']} ({after['alternate_pct']:.2f}%) [before {before['alternate']} / {before['alternate_pct']:.2f}%]",
        f"Unresolved without any source identifier: {after['unresolved']} ({after['unresolved_pct']:.2f}%) [before {before['unresolved']}]",
        f"Parser-failure candidates: {after['parser_failure']} [before {before['parser_failure']}]",
        f"Source-missing IDs: {after['source_missing']} [before {before['source_missing']}]",
        f"Duplicate primary code + month: {_duplicates(candidate, 'project_code')}",
        f"Duplicate source tracking key + month: {_duplicates(candidate.assign(source_tracking_key=candidate['project_code'].fillna(candidate['legacy_ocms_code'])), 'source_tracking_key')}",
        "",
        "Coverage by month:",
    ]
    coverage_month = candidate.groupby("report_month").agg(rows=("report_month", "size"), primary=("project_code", lambda s: int(s.notna().sum())), alternate=("legacy_ocms_code", lambda s: int(s.notna().sum()))).reset_index()
    for row in coverage_month.itertuples(index=False):
        lines.append(f"  {row.report_month}: rows={row.rows}, primary={row.primary} ({row.primary / row.rows * 100:.2f}%), alternate={row.alternate} ({row.alternate / row.rows * 100:.2f}%)")
    lines += ["", "Coverage by format:"]
    for line in _group_lines(candidate_work.assign(identifier_type=candidate_work.apply(lambda r: "PRIMARY" if _text(r.project_code) else ("ALTERNATE" if _text(r.legacy_ocms_code) else "NONE"), axis=1)), ["format", "identifier_type"]):
        lines.append(f"  {line}")
    lines += ["", "Coverage by handler:"]
    for line in _group_lines(candidate_work.assign(identifier_type=candidate_work.apply(lambda r: "PRIMARY" if _text(r.project_code) else ("ALTERNATE" if _text(r.legacy_ocms_code) else "NONE"), axis=1)), ["handler", "identifier_type"]):
        lines.append(f"  {line}")
    lines += [
        "", "Readiness assessment:",
        "  Identifier quality is acceptable for source-keyed longitudinal tracking only where a primary or alternate source identifier is present; unresolved records remain excluded from identity-dependent analysis.",
        "  ML label-definition work remains BLOCKED. Defensible labels need independently timestamped actual completion and cost-outcome events, plus a target policy fixed before model cutoffs.",
        "  Potential leakage fields: revised_cost, revised_doc, future expenditure, future physical progress, and future status information.",
        "  Future modelling must use temporal/project-aware evaluation, never random row splits.",
    ]
    (REPORTS_DIR / "identifier_quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global REPORTS_DIR
    parser = argparse.ArgumentParser(description="Build source-preserving Phase 7 identifier reports")
    parser.add_argument("--baseline", type=Path, default=PROCESSED_DIR / "project_monthly.csv")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    REPORTS_DIR = args.output_dir
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline, candidate = _read(args.baseline), _read(args.candidate)
    queue = build_review_queue(baseline, candidate)
    queue.to_csv(REPORTS_DIR / "missing_primary_id_review.csv", index=False, encoding="utf-8")
    build_anomalies(candidate).to_csv(REPORTS_DIR / "longitudinal_anomalies.csv", index=False, encoding="utf-8")
    write_summary(queue, baseline, candidate)
    write_quality_report(queue, baseline, candidate)
    safe_print(f"Wrote {len(queue)} identifier-review rows and Phase 7 reports.")


if __name__ == "__main__":
    main()
