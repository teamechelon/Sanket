"""
SANKET Data Audit Script
=========================
SIH 2026 · Problem Statement 26103
Predictive Early-Warning System for Infrastructure Projects

Purpose:
    Perform a non-destructive audit of all raw PAIMANA CSV files,
    producing a detailed report of schemas, missing values, duplicates,
    inferred semantics, and cross-month consistency.

Usage:
    python src/data_audit.py            # from project root
    python src/data_audit.py --raw-dir data/raw --output data/audit_report.txt

This module is intentionally modular so that the discovery, profiling,
and classification functions can be imported by future ingestion / ETL
code without re-implementing them.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ────────────────────────────────────────────────────────────────────
# 1. DISCOVERY — recursively find CSVs and infer metadata from path
# ────────────────────────────────────────────────────────────────────

# Regex to pull YYYY-MM (or YYYY_MM, YYYY/MM) from a path component
_MONTH_RE = re.compile(r"(20\d{2})[-_/](0[1-9]|1[0-2])")


def discover_csvs(raw_dir: str | Path) -> list[dict[str, Any]]:
    """
    Walk *raw_dir* recursively and return metadata dicts for every CSV file.

    Each dict contains:
        path          – absolute pathlib.Path
        relative_path – path relative to raw_dir
        inferred_month – 'YYYY-MM' string or None
        dataset_type  – stem of the filename (e.g. 'Cost-Wise-Report')
    """
    raw_dir = Path(raw_dir).resolve()
    results: list[dict[str, Any]] = []

    for root, _dirs, files in os.walk(raw_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(raw_dir)

            # Try to extract month from any part of the path
            month_match = _MONTH_RE.search(str(rel))
            inferred_month = (
                f"{month_match.group(1)}-{month_match.group(2)}"
                if month_match
                else None
            )

            dataset_type = fpath.stem  # e.g. 'State-Wise-Report'

            results.append(
                {
                    "path": fpath,
                    "relative_path": rel,
                    "inferred_month": inferred_month,
                    "dataset_type": dataset_type,
                }
            )

    return results


# ────────────────────────────────────────────────────────────────────
# 2. RAW READING — parse CSVs that may lack headers / have title rows
# ────────────────────────────────────────────────────────────────────


def _read_raw_csv(path: Path) -> tuple[list[str], list[list[str]], str | None]:
    """
    Read a PAIMANA-style CSV that typically has:
        line 1 : title (e.g. "Sector Wise Details")
        line 2 : blank
        lines 3+: data rows (no header)

    Returns:
        (title, rows, encoding_used)
    where *rows* is a list of lists of strings (no header row).
    """
    # Try common encodings
    content: str | None = None
    enc_used: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            content = path.read_text(encoding=enc)
            enc_used = enc
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if content is None:
        raise RuntimeError(f"Could not decode {path}")

    lines = content.splitlines()

    # Detect title line (first non-empty line that is a single quoted string)
    title: str | None = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # If line has only one field when CSV-parsed, treat as title
        parsed = list(csv.reader(io.StringIO(stripped)))
        if parsed and len(parsed[0]) == 1 and not parsed[0][0].replace(".", "").replace("-", "").replace(" ", "").isdigit():
            title = parsed[0][0]
            data_start = i + 1
            break
        else:
            # No title detected — data starts here
            data_start = i
            break

    # Parse remaining lines, skipping blanks
    rows: list[list[str]] = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parsed = list(csv.reader(io.StringIO(line)))
        if parsed:
            rows.append(parsed[0])

    return title, rows, enc_used


# ────────────────────────────────────────────────────────────────────
# 3. PROFILING — per-file statistics
# ────────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Statistics for a single column (positional)."""
    index: int
    inferred_name: str  # e.g. 'col_0', 'col_1' or a guessed name
    non_null_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    sample_values: list[str] = field(default_factory=list)
    inferred_dtype: str = "unknown"  # 'numeric', 'text', 'mixed', 'currency', 'progress_range'
    semantic_tags: list[str] = field(default_factory=list)


@dataclass
class FileProfile:
    """Complete audit profile for one CSV file."""
    path: Path
    relative_path: Path
    inferred_month: str | None
    dataset_type: str
    title_line: str | None
    encoding: str | None
    row_count: int = 0
    column_count: int = 0
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    duplicate_row_count: int = 0
    is_aggregate: bool | None = None  # True = aggregate, False = project-level, None = uncertain
    granularity_reason: str = ""
    possible_id_columns: list[int] = field(default_factory=list)


def _infer_dtype(values: list[str]) -> str:
    """Guess the data type from a list of non-empty string values."""
    if not values:
        return "unknown"

    num_count = 0
    currency_count = 0
    paren_count = 0  # values like "123.45 (678.90)"
    range_count = 0  # values like "0-10", "90-100"

    for v in values:
        v_stripped = v.strip()
        # Currency: starts with ₹ or contains ₹
        if "₹" in v_stripped:
            currency_count += 1
            continue
        # Parenthetical numeric: "123.45 (678.90)"
        if re.match(r"^[\d,]+\.?\d*\s*\([\d,]+\.?\d*\)$", v_stripped):
            paren_count += 1
            continue
        # Range: "0-10", "80-90", "100"
        if re.match(r"^\d+-\d+$", v_stripped):
            range_count += 1
            continue
        # Plain numeric
        cleaned = v_stripped.replace(",", "").replace(" ", "")
        try:
            float(cleaned)
            num_count += 1
        except ValueError:
            pass

    total = len(values)
    if currency_count > total * 0.5:
        return "currency"
    if paren_count > total * 0.5:
        return "numeric_parenthetical"  # e.g. "original (revised)"
    if range_count > total * 0.3:
        return "progress_range"
    if (num_count + paren_count) > total * 0.7:
        return "numeric"
    if num_count == 0 and paren_count == 0:
        return "text"
    return "mixed"


# Semantic keyword patterns for column classification (requirement #6)
_SEMANTIC_PATTERNS: dict[str, list[re.Pattern]] = {
    "cost": [
        re.compile(r"cost|₹|crore|lakh|budget|sanction", re.I),
    ],
    "expenditure": [
        re.compile(r"expend|spend|disburs|release|utiliz", re.I),
    ],
    "physical_progress": [
        re.compile(r"progress|physical|completion|%|percent|0-10|90-100", re.I),
    ],
    "timeline": [
        re.compile(r"date|deadline|schedule|target.*date|commissioning|start|end|completion.*date|year|month", re.I),
    ],
    "milestone": [
        re.compile(r"milestone|phase|stage|status|awarded|under.*construction", re.I),
    ],
    "ministry_department": [
        re.compile(r"ministry|department|govt|government|central|nodal", re.I),
    ],
    "implementing_agency": [
        re.compile(r"agency|implement|contractor|executing|company|corp|authority|nhai|ongc|ntpc", re.I),
    ],
    "state": [
        re.compile(r"state|pradesh|maharashtra|gujarat|bihar|delhi|assam|odisha|rajasthan|tamil|karnataka|bengal|punjab|kerala|ladakh|goa|manipur|mizoram|sikkim|meghalaya|tripura|nagaland|arunachal|chhattisgarh|uttarakhand|haryana|jharkhand|telangana|andhra|himachal|jammu|puducherry|andaman|dadra", re.I),
    ],
    "sector": [
        re.compile(r"sector|railways|coal|oil.*gas|electricity|power|telecom|roads|highway|aviation|steel|mining|port|shipping|water.*resource", re.I),
    ],
    "project_id": [
        re.compile(r"project.*code|project.*id|proj.*no|unique.*id|uid|code", re.I),
    ],
}


def _tag_semantics_from_values(values: list[str]) -> list[str]:
    """Apply semantic regex patterns against actual cell values."""
    tags: list[str] = []
    # Concatenate a sample of values for matching
    sample_text = " ".join(values[:50])
    for tag, patterns in _SEMANTIC_PATTERNS.items():
        for pat in patterns:
            if pat.search(sample_text):
                tags.append(tag)
                break
    return tags


def _tag_semantics_from_title_and_dtype(
    title: str | None, col_index: int, dtype: str, dataset_type: str
) -> list[str]:
    """
    Infer semantic tags from the file title, column position, inferred dtype,
    and dataset type — since PAIMANA CSVs have no header row.
    """
    tags: list[str] = []
    ds_lower = dataset_type.lower()
    title_lower = (title or "").lower()

    # Dataset-level inferences based on known PAIMANA report names
    if "state" in ds_lower or "state" in title_lower:
        if col_index == 1 and dtype == "text":
            tags.append("state")
    if "sector" in ds_lower or "sector" in title_lower:
        if col_index == 1 and dtype == "text":
            tags.append("sector")
    if "cost" in ds_lower or "cost" in title_lower:
        if dtype in ("currency", "numeric", "numeric_parenthetical"):
            tags.append("cost")
    if "physical" in ds_lower or "progress" in ds_lower or "progress" in title_lower:
        if dtype == "progress_range":
            tags.append("physical_progress")

    # Common positional patterns across PAIMANA reports:
    # col 0 = serial number, col 1 = name/category, col 2 = project count,
    # col 3 = cost (original (revised)), col 4 = expenditure
    if col_index == 2 and dtype == "numeric":
        tags.append("project_count")
    if col_index == 3 and dtype == "numeric_parenthetical":
        tags.extend(["cost", "original_and_revised_cost"])
    if col_index == 4 and dtype == "numeric":
        tags.append("expenditure")

    return list(set(tags))


def profile_file(meta: dict[str, Any]) -> FileProfile:
    """
    Build a complete FileProfile for a single CSV without modifying the file.
    """
    path: Path = meta["path"]
    title, rows, enc = _read_raw_csv(path)

    fp = FileProfile(
        path=path,
        relative_path=meta["relative_path"],
        inferred_month=meta["inferred_month"],
        dataset_type=meta["dataset_type"],
        title_line=title,
        encoding=enc,
        row_count=len(rows),
    )

    if not rows:
        return fp

    # Column count = max width across rows (handle ragged CSVs)
    col_count = max(len(r) for r in rows)
    fp.column_count = col_count

    # Build per-column profiles
    for ci in range(col_count):
        values = [r[ci] if ci < len(r) else "" for r in rows]
        non_empty = [v for v in values if v.strip()]
        null_count = len(values) - len(non_empty)

        dtype = _infer_dtype(non_empty)
        sem_tags_values = _tag_semantics_from_values(non_empty)
        sem_tags_positional = _tag_semantics_from_title_and_dtype(
            title, ci, dtype, fp.dataset_type
        )
        all_tags = sorted(set(sem_tags_values + sem_tags_positional))

        # Sample up to 5 unique values
        seen: set[str] = set()
        samples: list[str] = []
        for v in non_empty:
            if v not in seen and len(samples) < 5:
                seen.add(v)
                samples.append(v)

        cp = ColumnProfile(
            index=ci,
            inferred_name=f"col_{ci}",
            non_null_count=len(non_empty),
            null_count=null_count,
            unique_count=len(set(non_empty)),
            sample_values=samples,
            inferred_dtype=dtype,
            semantic_tags=all_tags,
        )
        fp.column_profiles.append(cp)

        # Check for possible ID column
        if "project_id" in all_tags:
            fp.possible_id_columns.append(ci)

    # Duplicate row detection
    row_tuples = [tuple(r) for r in rows]
    fp.duplicate_row_count = len(row_tuples) - len(set(row_tuples))

    # Granularity detection (requirement #7)
    fp.is_aggregate, fp.granularity_reason = _detect_granularity(fp, rows)

    return fp


def _detect_granularity(
    fp: FileProfile, rows: list[list[str]]
) -> tuple[bool | None, str]:
    """
    Heuristically determine whether the CSV is project-level or aggregate-level.

    Returns (is_aggregate, reason_string).
    """
    ds = fp.dataset_type.lower()
    title = (fp.title_line or "").lower()

    # Strong signals from known PAIMANA report types
    if "state" in ds or "state" in title:
        return True, "State-wise aggregation (rows represent states, not individual projects)"
    if "sector" in ds or "sector" in title:
        return True, "Sector-wise aggregation (rows represent sectors, not individual projects)"
    if "cost" in ds and fp.row_count <= 3:
        return True, "Cost-wise summary with very few rows — likely national-level aggregate"
    if "physical" in ds or "progress" in title:
        # Physical progress reports bucket projects into progress ranges
        has_ranges = any(
            cp.inferred_dtype == "progress_range" for cp in fp.column_profiles
        )
        if has_ranges:
            return True, "Physical-progress report with progress-range buckets (aggregate histogram)"

    # Generic heuristics
    # If unique values in col_1 look like entity names and row count is small,
    # it's likely aggregate
    if fp.row_count < 50 and fp.column_count >= 3:
        return True, f"Low row count ({fp.row_count}) with structured columns suggests aggregation"

    # If row count is very high and there appear to be unique IDs, likely project-level
    if fp.row_count > 500:
        return False, f"High row count ({fp.row_count}) suggests project-level data"

    return None, "Could not confidently determine granularity"


# ────────────────────────────────────────────────────────────────────
# 4. SCHEMA COMPARISON across months (requirement #4)
# ────────────────────────────────────────────────────────────────────

@dataclass
class SchemaComparison:
    """Result of comparing schemas across months for one dataset type."""
    dataset_type: str
    months: list[str]
    column_counts: dict[str, int]  # month → col count
    row_counts: dict[str, int]  # month → row count
    consistent_columns: bool = True
    notes: list[str] = field(default_factory=list)


def compare_schemas(profiles: list[FileProfile]) -> list[SchemaComparison]:
    """
    Group profiles by dataset_type and compare their schemas across months.
    """
    by_type: dict[str, list[FileProfile]] = defaultdict(list)
    for fp in profiles:
        by_type[fp.dataset_type].append(fp)

    comparisons: list[SchemaComparison] = []
    for ds_type, fps in sorted(by_type.items()):
        fps_sorted = sorted(fps, key=lambda f: f.inferred_month or "")
        months = [f.inferred_month or "unknown" for f in fps_sorted]
        col_counts = {
            (f.inferred_month or "unknown"): f.column_count for f in fps_sorted
        }
        row_counts = {
            (f.inferred_month or "unknown"): f.row_count for f in fps_sorted
        }

        sc = SchemaComparison(
            dataset_type=ds_type,
            months=months,
            column_counts=col_counts,
            row_counts=row_counts,
        )

        # Check column consistency
        unique_col_counts = set(col_counts.values())
        if len(unique_col_counts) > 1:
            sc.consistent_columns = False
            sc.notes.append(
                f"Column count varies across months: {dict(col_counts)}"
            )

        # Check dtype consistency per column position
        max_cols = max(col_counts.values()) if col_counts else 0
        for ci in range(max_cols):
            dtypes_by_month: dict[str, str] = {}
            for fp in fps_sorted:
                if ci < len(fp.column_profiles):
                    m = fp.inferred_month or "unknown"
                    dtypes_by_month[m] = fp.column_profiles[ci].inferred_dtype
            unique_dtypes = set(dtypes_by_month.values())
            if len(unique_dtypes) > 1:
                sc.consistent_columns = False
                sc.notes.append(
                    f"col_{ci} dtype varies: {dict(dtypes_by_month)}"
                )

        # Check row count trends
        rcs = list(row_counts.values())
        if rcs:
            min_rc, max_rc = min(rcs), max(rcs)
            if max_rc > 0 and (max_rc - min_rc) / max_rc > 0.3:
                sc.notes.append(
                    f"Row count varies significantly: min={min_rc}, max={max_rc}"
                )

        comparisons.append(sc)

    return comparisons


# ────────────────────────────────────────────────────────────────────
# 5. REPORT FORMATTING
# ────────────────────────────────────────────────────────────────────

_SEPARATOR = "=" * 80
_SUBSEP = "-" * 60


def _format_file_report(fp: FileProfile) -> str:
    """Format a single file's audit into readable text."""
    lines: list[str] = []
    lines.append(_SEPARATOR)
    lines.append(f"FILE: {fp.relative_path}")
    lines.append(_SUBSEP)
    lines.append(f"  Absolute path   : {fp.path}")
    lines.append(f"  Inferred month  : {fp.inferred_month or 'N/A'}")
    lines.append(f"  Dataset type    : {fp.dataset_type}")
    lines.append(f"  Title line      : {fp.title_line or '(none)'}")
    lines.append(f"  Encoding        : {fp.encoding or 'unknown'}")
    lines.append(f"  Row count       : {fp.row_count}")
    lines.append(f"  Column count    : {fp.column_count}")
    lines.append(f"  Duplicate rows  : {fp.duplicate_row_count}")

    # Granularity
    granularity_label = {
        True: "AGGREGATE",
        False: "PROJECT-LEVEL",
        None: "UNCERTAIN",
    }[fp.is_aggregate]
    lines.append(f"  Granularity     : {granularity_label}")
    lines.append(f"    Reason        : {fp.granularity_reason}")

    # Possible ID columns
    if fp.possible_id_columns:
        lines.append(f"  Possible ID cols: {fp.possible_id_columns}")
    else:
        lines.append("  Possible ID cols: (none detected)")

    # Column details
    lines.append("")
    lines.append("  COLUMNS:")
    for cp in fp.column_profiles:
        missing_pct = (
            f"{cp.null_count / (cp.null_count + cp.non_null_count) * 100:.1f}%"
            if (cp.null_count + cp.non_null_count) > 0
            else "N/A"
        )
        lines.append(f"    [{cp.index}] {cp.inferred_name}")
        lines.append(f"        dtype       : {cp.inferred_dtype}")
        lines.append(f"        non-null    : {cp.non_null_count}")
        lines.append(f"        missing     : {cp.null_count} ({missing_pct})")
        lines.append(f"        unique      : {cp.unique_count}")
        if cp.semantic_tags:
            lines.append(f"        semantics   : {', '.join(cp.semantic_tags)}")
        samples_display = [repr(s) for s in cp.sample_values[:5]]
        lines.append(f"        samples     : {', '.join(samples_display)}")

    lines.append("")
    return "\n".join(lines)


def _format_comparison_report(comps: list[SchemaComparison]) -> str:
    """Format cross-month schema comparison."""
    lines: list[str] = []
    lines.append(_SEPARATOR)
    lines.append("CROSS-MONTH SCHEMA COMPARISON")
    lines.append(_SEPARATOR)

    for sc in comps:
        lines.append("")
        lines.append(f"  Dataset: {sc.dataset_type}")
        lines.append(f"  Months covered: {', '.join(sc.months)}")
        lines.append(f"  Column counts by month: {sc.column_counts}")
        lines.append(f"  Row counts by month   : {sc.row_counts}")
        status = "CONSISTENT" if sc.consistent_columns else "INCONSISTENT"
        lines.append(f"  Schema consistency: {status}")
        if sc.notes:
            for note in sc.notes:
                lines.append(f"    ⚠  {note}")
        lines.append(_SUBSEP)

    lines.append("")
    return "\n".join(lines)


def _format_semantic_summary(profiles: list[FileProfile]) -> str:
    """Summarise detected semantic column categories across all files."""
    lines: list[str] = []
    lines.append(_SEPARATOR)
    lines.append("SEMANTIC COLUMN SUMMARY")
    lines.append(_SEPARATOR)

    tag_locations: dict[str, list[str]] = defaultdict(list)
    for fp in profiles:
        for cp in fp.column_profiles:
            for tag in cp.semantic_tags:
                loc = f"{fp.dataset_type}[{fp.inferred_month}].col_{cp.index}"
                tag_locations[tag].append(loc)

    if not tag_locations:
        lines.append("  No semantic tags detected.")
    else:
        for tag in sorted(tag_locations):
            locations = tag_locations[tag]
            lines.append(f"\n  {tag.upper()}")
            # Deduplicate and show up to 10 examples
            shown = []
            seen = set()
            for loc in locations:
                key = loc.split("[")[0] + ".col_" + loc.split("col_")[1]
                if key not in seen:
                    seen.add(key)
                    shown.append(loc)
                if len(shown) >= 10:
                    break
            for s in shown:
                lines.append(f"    - {s}")
            remaining = len(locations) - len(shown)
            if remaining > 0:
                lines.append(f"    ... and {remaining} more occurrences")

    lines.append("")
    return "\n".join(lines)


def _format_id_column_summary(profiles: list[FileProfile]) -> str:
    """Highlight possible project identifier columns."""
    lines: list[str] = []
    lines.append(_SEPARATOR)
    lines.append("PROJECT IDENTIFIER COLUMN CANDIDATES")
    lines.append(_SEPARATOR)

    found_any = False
    for fp in profiles:
        if fp.possible_id_columns:
            found_any = True
            cols = ", ".join(f"col_{c}" for c in fp.possible_id_columns)
            lines.append(f"  {fp.dataset_type} ({fp.inferred_month}): {cols}")

    if not found_any:
        lines.append(
            "  No explicit project identifier columns detected."
        )
        lines.append(
            "  NOTE: PAIMANA aggregate reports typically do not contain"
        )
        lines.append(
            "  per-project identifiers. Project-level data may need to be"
        )
        lines.append(
            "  obtained from the detailed PAIMANA portal or API."
        )

    lines.append("")
    return "\n".join(lines)


def build_report(profiles: list[FileProfile], comparisons: list[SchemaComparison]) -> str:
    """Assemble the full audit report."""
    parts: list[str] = []

    # Header
    parts.append(_SEPARATOR)
    parts.append("SANKET — DATA AUDIT REPORT")
    parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"Files audited: {len(profiles)}")
    parts.append(_SEPARATOR)
    parts.append("")

    # Per-file reports
    parts.append("INDIVIDUAL FILE PROFILES")
    parts.append("")
    for fp in profiles:
        parts.append(_format_file_report(fp))

    # Schema comparison
    parts.append(_format_comparison_report(comparisons))

    # Semantic summary
    parts.append(_format_semantic_summary(profiles))

    # ID column summary
    parts.append(_format_id_column_summary(profiles))

    # Footer
    parts.append(_SEPARATOR)
    parts.append("END OF AUDIT REPORT")
    parts.append(_SEPARATOR)

    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────

def run_audit(raw_dir: str | Path, output_path: str | Path | None = None) -> str:
    """
    Execute the full audit pipeline and return the report text.

    Parameters
    ----------
    raw_dir : path to the raw data directory
    output_path : if provided, the report is also saved to this file

    Returns
    -------
    The report as a string.
    """
    _safe_print(f"[SANKET Audit] Scanning: {Path(raw_dir).resolve()}")

    # Step 1 — discover
    csv_metas = discover_csvs(raw_dir)
    _safe_print(f"[SANKET Audit] Found {len(csv_metas)} CSV file(s)")

    if not csv_metas:
        msg = "No CSV files found. Aborting."
        _safe_print(f"[SANKET Audit] {msg}")
        return msg

    # Step 2 — profile each file
    profiles: list[FileProfile] = []
    for meta in csv_metas:
        _safe_print(f"  Profiling: {meta['relative_path']}")
        fp = profile_file(meta)
        profiles.append(fp)

    # Step 3 — cross-month comparison
    comparisons = compare_schemas(profiles)

    # Step 4 — build report
    report = build_report(profiles, comparisons)

    # Output
    _safe_print("")
    _safe_print(report)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        _safe_print(f"\n[SANKET Audit] Report saved to: {out.resolve()}")

    return report


def _safe_print(text: str) -> None:
    """Print text to stdout, handling encoding issues on Windows consoles."""
    try:
        # Attempt to reconfigure stdout to UTF-8 (Python 3.7+)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace unencodable characters
        print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        ))


def main() -> None:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SANKET Data Audit — inspect raw PAIMANA CSV files",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Path to the raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        default="data/audit_report.txt",
        help="Path for the output report file (default: data/audit_report.txt)",
    )
    args = parser.parse_args()

    run_audit(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
