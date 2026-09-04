"""
SANKET — PDF Format Audit
==========================
Systematically inspects every PDF in data/raw/pdf/ to classify
format generation, Table 6 structure, and parser compatibility.

Run:  python -m src.pdf_audit

Output:
  reports/pdf_format_audit.csv
  reports/pdf_format_audit.txt

This script is READ-ONLY: it never modifies raw files.
"""

from __future__ import annotations

import csv
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from src.utils import (
    RAW_PDF_DIR,
    REPORTS_DIR,
    get_logger,
    infer_month_from_path,
    infer_month_from_text,
    safe_print,
)

log = get_logger("pdf_audit")

# ────────────────────────────────────────────────────────────────────
# Regex patterns (reused from pdf_extractor plus new ones)
# ────────────────────────────────────────────────────────────────────

_SL_NO_RE = re.compile(r"^(\d+)\s+(.+)$")
_DATE_RE = re.compile(r"\d{2}/\d{4}")
_PROJECT_CODE_RE = re.compile(r"^\((\d{5,7})\)$")
_LEGACY_CODE_RE = re.compile(r"^\(([NO]\d{7,9})\)$")
_PAREN_VALUES_RE = re.compile(r"^\(([^)]*)\)\s*\(([^)]*)\)\s*\(([^)]*)\)$")
_MINISTRY_RE = re.compile(r"^Ministry of .+$|^Department of .+$")
_PAGE_NUMBER_RE = re.compile(r"^Page \d+$")
_MONTH_HEADER_RE = re.compile(
    r"^(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}$"
)

_SKIP_PATTERNS = [
    "Project Assessment, Infrastructure Monitoring",
    "Nation-building For details visit",
    "ipm.mospi.gov.in",
    "(PAIMANA)",
]

_HEADER_STARTS = (
    "All Ongoing Projects",
    "Orignal/Target", "Original/Target",
    "Date of Approval", "Date of Orignal", "Date of Original",
    "Sl.No", "Sl. No", "MM/YYYY",
    "in Rs.", "(Revised DoC)", "(Revised Cost)",
    "Project Name", "(Legacy OCMS Code)", "Legacy OCMS",
    "Cumulative", "Physical Progress", "Expenditure",
    "Table 6:", "Table 4:", "Table 3:",
)

_HEADER_SUBSTRINGS = (
    "Legacy OCMS Code", "in Rs. Crore",
    "Project Name (Agency)", "Approval Revised Cost",
    "(Project Code)",
)

_KNOWN_SECTORS = {
    "Aviation & Aviation Infrastructure", "Coal", "Oil & Gas",
    "Transmission & Distribution", "Water Resources",
    "Electricity Generation", "Waste & Water", "Education",
    "Urban Public Transport", "Steel", "Energy Storage",
    "Telecommunication", "Metals & Mining", "Real Estate",
    "Shipping", "Healthcare", "Inland Waterways", "Construction",
    "Tourism, Hospitality & Wellness", "Railways",
    "Roads & Highways", "Logistics Infrastructure",
}

_KNOWN_STATES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu & Kashmir", "Ladakh", "Puducherry",
    "Andaman & Nicobar", "Dadra & Nagar Haveli and Daman & Diu",
    "Chandigarh", "Lakshadweep", "Multi-States",
}


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _is_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    for pat in _SKIP_PATTERNS:
        if pat in stripped:
            return True
    return False


def _is_table_header(line: str) -> bool:
    stripped = line.strip()
    for prefix in _HEADER_STARTS:
        if stripped.startswith(prefix):
            return True
    for sub in _HEADER_SUBSTRINGS:
        if sub in stripped:
            return True
    if re.match(r"^(Orignal|Original|Revised)\s+(Cost|DoC)", stripped):
        return True
    return False


def _is_sector_or_ministry(line: str) -> bool:
    stripped = line.strip()
    if _MINISTRY_RE.match(stripped):
        return True
    if stripped in _KNOWN_SECTORS:
        return True
    return False


def _is_total_line(line: str) -> bool:
    return bool(re.match(r"^Total\s*\(\d+\)", line.strip()))


def _find_state_in_line(line: str) -> str | None:
    for state in sorted(_KNOWN_STATES, key=len, reverse=True):
        if state in line:
            return state
    return None


# ────────────────────────────────────────────────────────────────────
# PDFAuditResult
# ────────────────────────────────────────────────────────────────────

@dataclass
class PDFAuditResult:
    filename: str = ""
    report_month: str = ""
    report_type: str = ""        # MONTHLY | QUARTERLY | UNKNOWN
    pages: int = 0
    format_class: str = ""       # OLD_FORMAT_A, NEW_FORMAT_A, etc.
    table6_start: int = 0        # 1-indexed
    table6_end: int = 0          # 1-indexed
    table6_page_count: int = 0
    approx_project_count: int = 0
    legacy_code_present: str = ""  # YES | NO
    agency_layout: str = ""      # SAME_LINE | SEPARATE_LINE | MIXED | NONE
    record_line_pattern: str = "" # multi-line description
    page_break_pattern: str = "" # CLEAN | RECORDS_SPLIT | UNKNOWN
    sector_headers: str = ""     # YES | NO
    special_patterns: str = ""   # free-text
    risk_level: str = ""         # LOW | MEDIUM | HIGH
    parser_compatibility: str = ""  # PARSER_COMPATIBLE etc.
    notes: str = ""

    # ---- detail accumulators (not in CSV) ----
    toc_text: str = ""
    header_samples: list[str] = field(default_factory=list)
    first_table6_text: str = ""
    mid_table6_text: str = ""
    last_table6_text: str = ""
    legacy_code_samples: list[str] = field(default_factory=list)
    project_code_samples: list[str] = field(default_factory=list)
    paren_line_samples: list[str] = field(default_factory=list)
    sl_no_line_samples: list[str] = field(default_factory=list)
    sector_header_samples: list[str] = field(default_factory=list)
    agency_in_parens_count: int = 0
    agency_separate_line_count: int = 0
    page_break_records: list[str] = field(default_factory=list)
    column_headers_detected: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Main audit per-PDF
# ────────────────────────────────────────────────────────────────────

def audit_single_pdf(pdf_path: Path) -> PDFAuditResult:
    """Inspect a single PDF and return structured audit data."""
    result = PDFAuditResult(filename=pdf_path.name)
    log.info(f"Auditing: {pdf_path.name}")

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        result.notes = f"FAILED TO OPEN: {e}"
        result.risk_level = "HIGH"
        result.parser_compatibility = "UNKNOWN"
        return result

    result.pages = len(pdf.pages)

    # ---- Determine report type ----
    if "QPISR" in pdf_path.name or "QR" in pdf_path.name:
        result.report_type = "QUARTERLY"
    elif "Flash" in pdf_path.name or "FR" in pdf_path.name:
        result.report_type = "MONTHLY"
    else:
        result.report_type = "UNKNOWN"

    # ---- Infer report month ----
    result.report_month = infer_month_from_path(pdf_path) or ""
    if not result.report_month:
        for i in range(min(5, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            rm = infer_month_from_text(text)
            if rm:
                result.report_month = rm
                break

    # ---- Read TOC pages (first 4 pages) ----
    toc_texts = []
    for i in range(min(4, len(pdf.pages))):
        toc_texts.append(pdf.pages[i].extract_text() or "")
    result.toc_text = "\n".join(toc_texts)

    # ---- Detect format generation from TOC ----
    has_table6 = "Table 6:" in result.toc_text
    has_table5 = "Table 5:" in result.toc_text
    has_table4_ongoing = "Table 4: All Ongoing" in result.toc_text or "Table 4:All Ongoing" in result.toc_text
    has_table4_any = "Table 4:" in result.toc_text
    has_table3_any = "Table 3:" in result.toc_text
    has_legacy_ref = "Legacy OCMS" in result.toc_text or "OCMS Code" in result.toc_text

    # ---- Find Table 6 / Table 4 "All Ongoing Projects" pages ----
    table_pages = []
    table_label = ""
    for i, page in enumerate(pdf.pages):
        text = (page.extract_text() or "")[:500]
        if "All Ongoing Projects" in text:
            table_pages.append(i)
            # Identify which table label
            if "Table 6:" in text or "Table 6 :" in text:
                table_label = "Table 6"
            elif "Table 4:" in text or "Table 4 :" in text:
                table_label = "Table 4"
            elif not table_label:
                # Look for it in the broader text
                if "Table 6" in text:
                    table_label = "Table 6"
                elif "Table 4" in text:
                    table_label = "Table 4"

    if table_pages:
        result.table6_start = table_pages[0] + 1  # 1-indexed
        result.table6_end = table_pages[-1] + 1
        result.table6_page_count = len(table_pages)
    else:
        result.notes += "NO 'All Ongoing Projects' TABLE FOUND. "
        result.risk_level = "HIGH"
        result.parser_compatibility = "PARSER_NEEDS_NEW_HANDLER"
        pdf.close()
        return result

    # ---- Sample Table 6 pages: first, middle, last ----
    first_idx = table_pages[0]
    last_idx = table_pages[-1]
    mid_idx = table_pages[len(table_pages) // 2]

    result.first_table6_text = pdf.pages[first_idx].extract_text() or ""
    result.mid_table6_text = pdf.pages[mid_idx].extract_text() or ""
    result.last_table6_text = pdf.pages[last_idx].extract_text() or ""

    # ---- Deep inspection of ALL Table 6 pages ----
    sl_no_count = 0
    legacy_code_count = 0
    project_code_count = 0
    paren_triple_count = 0
    sector_count = 0
    ministry_count = 0
    total_line_count = 0
    agency_in_paren_on_sl_line = 0
    agency_on_separate_paren = 0
    dates_on_sl_line = 0
    dates_on_sl_line_count = 0
    records_at_page_boundary = 0
    last_sl_no_per_page: dict[int, int] = {}
    first_sl_no_per_page: dict[int, int | None] = {}
    column_header_lines: list[str] = []
    all_data_lines: list[str] = []

    for pi in table_pages:
        page_text = pdf.pages[pi].extract_text() or ""
        lines = page_text.split("\n")
        page_sl_nos = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Collect column header lines from first few Table 6 pages
            if _is_table_header(stripped) and pi in (first_idx, first_idx + 1 if first_idx + 1 <= last_idx else first_idx):
                column_header_lines.append(stripped)
            
            if _is_skip_line(stripped):
                continue
            if _MONTH_HEADER_RE.match(stripped):
                continue
            if _is_table_header(stripped):
                continue
            
            # Sector / ministry headers
            if _is_sector_or_ministry(stripped):
                sector_count += 1
                if len(result.sector_header_samples) < 10:
                    result.sector_header_samples.append(stripped)
                continue
            
            # Total lines
            if _is_total_line(stripped):
                total_line_count += 1
                continue
            
            all_data_lines.append(stripped)
            
            # Serial number lines
            sl_m = _SL_NO_RE.match(stripped)
            if sl_m:
                sl_val = int(sl_m.group(1))
                rest = sl_m.group(2)
                
                # Is this a plausible serial number?
                state_found = _find_state_in_line(rest)
                has_trailing_nums = bool(re.search(r"\d+\.?\d*\s*$", rest))
                
                if state_found and has_trailing_nums and sl_val <= 3000:
                    sl_no_count += 1
                    page_sl_nos.append(sl_val)
                    
                    if len(result.sl_no_line_samples) < 20:
                        result.sl_no_line_samples.append(f"[p{pi+1}] {stripped}")
                    
                    # Check if agency appears on this line (in parens)
                    if re.search(r"\([A-Za-z]", rest):
                        agency_in_paren_on_sl_line += 1
                    
                    # Check if dates appear on sl_no line (OLD format indicator)
                    dates = _DATE_RE.findall(rest)
                    if dates:
                        dates_on_sl_line += 1
                        dates_on_sl_line_count += len(dates)
            
            # Legacy OCMS code lines
            if _LEGACY_CODE_RE.match(stripped):
                legacy_code_count += 1
                if len(result.legacy_code_samples) < 5:
                    result.legacy_code_samples.append(stripped)
            
            # Project code lines
            if _PROJECT_CODE_RE.match(stripped):
                project_code_count += 1
                if len(result.project_code_samples) < 5:
                    result.project_code_samples.append(stripped)
            
            # Triple parenthetical lines
            if _PAREN_VALUES_RE.match(stripped):
                paren_triple_count += 1
                if len(result.paren_line_samples) < 5:
                    result.paren_line_samples.append(stripped)
            
            # Agency on separate parenthetical line
            if re.match(r"^\([A-Za-z].*\)$", stripped) and not _PROJECT_CODE_RE.match(stripped) and not _LEGACY_CODE_RE.match(stripped):
                # Looks like an agency-only paren line
                content = stripped[1:-1]
                if not content[0].isdigit() and content != "-":
                    agency_on_separate_paren += 1
        
        if page_sl_nos:
            last_sl_no_per_page[pi] = max(page_sl_nos)
            first_sl_no_per_page[pi] = min(page_sl_nos)

    # ---- Page break analysis ----
    prev_page = None
    for pi in table_pages:
        if prev_page is not None:
            # Check if a record might span pages
            page_text = pdf.pages[pi].extract_text() or ""
            first_lines = page_text.split("\n")[:10]
            # If first data line (non-header, non-skip) doesn't start with sl_no
            # it may be a continuation
            for fl in first_lines:
                fl = fl.strip()
                if not fl or _is_skip_line(fl) or _is_table_header(fl) or _MONTH_HEADER_RE.match(fl):
                    continue
                if _is_sector_or_ministry(fl) or _is_total_line(fl):
                    break
                # First data line - check if it starts with serial number
                sl_m = _SL_NO_RE.match(fl)
                if sl_m:
                    # Clean start
                    break
                else:
                    # Continuation from previous page
                    records_at_page_boundary += 1
                    if len(result.page_break_records) < 5:
                        result.page_break_records.append(
                            f"Page {pi+1} starts with continuation: '{fl[:80]}...'"
                        )
                break
        prev_page = pi

    # ---- Column header analysis ----
    result.column_headers_detected = list(dict.fromkeys(column_header_lines))[:10]

    # ---- Determine format class ----
    # Evidence-based classification
    fmt_signals = {
        "has_table6_label": has_table6,
        "has_table5_label": has_table5,
        "has_table4_ongoing": has_table4_ongoing,
        "legacy_codes_found": legacy_code_count > 0,
        "legacy_code_count": legacy_code_count,
        "triple_paren_lines": paren_triple_count,
        "dates_on_sl_line": dates_on_sl_line,
        "table_label": table_label,
        "project_code_count": project_code_count,
    }

    # Compute derived metrics for classification
    paren_triple_ratio = paren_triple_count / max(sl_no_count, 1) if sl_no_count > 0 else 0
    code_ratio = project_code_count / max(sl_no_count, 1) if sl_no_count > 0 else 0
    agency_sep_ratio = agency_on_separate_paren / max(sl_no_count, 1) if sl_no_count > 0 else 0
    
    if result.report_type == "QUARTERLY":
        if legacy_code_count > 0:
            result.format_class = "QUARTERLY_NEW"
        else:
            result.format_class = "QUARTERLY_FORMAT"
    elif (has_table6 or has_table5) and legacy_code_count > 0:
        # NEW_FORMAT_A: Sep 2025+ table numbering with legacy OCMS codes (Feb-Mar 2026)
        result.format_class = "NEW_FORMAT_A"
    elif table_label == "Table 6" and legacy_code_count == 0 and paren_triple_ratio >= 0.3:
        # TRANSITIONAL_FORMAT: Table 6 label, no legacy codes, but still has
        # the triple-parenthetical structure (Sep 2025 – Jan 2026 era)
        result.format_class = "TRANSITIONAL_FORMAT"
    elif table_label == "Table 6" and legacy_code_count == 0 and paren_triple_ratio < 0.3:
        # NEW_FORMAT_B: Table 6 label, no legacy codes, AND the triple-paren
        # structure is largely gone (Apr 2026+ era). Parenthetical values are
        # restructured into separate lines.
        result.format_class = "NEW_FORMAT_B"
    elif has_table4_ongoing and legacy_code_count == 0 and dates_on_sl_line > sl_no_count * 0.5:
        # OLD_FORMAT_A: Table 4 label, dates appear on the serial-number line
        # (Jul 2025 style)
        result.format_class = "OLD_FORMAT_A"
    elif has_table4_ongoing and legacy_code_count == 0:
        # OLD_FORMAT_B: Table 4 label, dates NOT on SL line, triple-paren
        # structure present (Aug 2025 style)
        result.format_class = "OLD_FORMAT_B"
    elif legacy_code_count > 0:
        result.format_class = "NEW_FORMAT_VARIANT"
    elif legacy_code_count == 0 and table_label == "Table 4":
        result.format_class = "OLD_FORMAT_A"
    else:
        result.format_class = "UNKNOWN"

    # ---- Populate remaining fields ----
    result.approx_project_count = sl_no_count

    result.legacy_code_present = "YES" if legacy_code_count > 0 else "NO"

    # Agency layout
    if agency_in_paren_on_sl_line > 0 and agency_on_separate_paren > 0:
        result.agency_layout = "MIXED"
    elif agency_in_paren_on_sl_line > 0:
        result.agency_layout = "SAME_LINE"
    elif agency_on_separate_paren > 0:
        result.agency_layout = "SEPARATE_LINE"
    else:
        result.agency_layout = "NONE_DETECTED"
    result.agency_in_parens_count = agency_in_paren_on_sl_line
    result.agency_separate_line_count = agency_on_separate_paren

    # Record line pattern
    if paren_triple_count > 0 and legacy_code_count > 0:
        result.record_line_pattern = (
            "MULTI_LINE: name_lines -> sl_no_line -> paren_triple -> project_code -> legacy_code"
        )
    elif paren_triple_count > 0:
        result.record_line_pattern = (
            "MULTI_LINE: name_lines -> sl_no_line -> paren_triple -> project_code"
        )
    elif dates_on_sl_line > 0:
        result.record_line_pattern = (
            "MULTI_LINE: name_lines -> sl_no_line(with dates) -> agency/revised_line -> project_code"
        )
    else:
        result.record_line_pattern = "MULTI_LINE: structure unclear"

    # Page break pattern
    if records_at_page_boundary > 0:
        result.page_break_pattern = f"RECORDS_SPLIT ({records_at_page_boundary} occurrences)"
    else:
        result.page_break_pattern = "CLEAN"

    # Sector headers
    result.sector_headers = "YES" if sector_count > 0 else "NO"

    # ---- Special patterns ----
    specials = []
    
    # Check for unusual page counts
    if result.table6_page_count < 10:
        specials.append(f"LOW_PAGE_COUNT({result.table6_page_count})")
    
    # Check for "Table 3" as All Ongoing (early format?)
    if "Table 3: All Ongoing" in result.toc_text or "Table 3:All Ongoing" in result.toc_text:
        specials.append("TABLE_3_ONGOING")
    
    # Check for "FR" prefix vs "FlashReport" prefix
    if pdf_path.name.startswith("FR"):
        specials.append("FR_NAMING")
    
    # Check total lines count
    if total_line_count > 0:
        specials.append(f"TOTAL_LINES({total_line_count})")
    
    # Check for very high or very low project count
    if sl_no_count > 2000:
        specials.append(f"HIGH_PROJECT_COUNT({sl_no_count})")
    elif sl_no_count < 100 and sl_no_count > 0:
        specials.append(f"LOW_PROJECT_COUNT({sl_no_count})")
    elif sl_no_count == 0:
        specials.append("ZERO_PROJECTS_DETECTED")
    
    result.special_patterns = "; ".join(specials) if specials else "NONE"

    # ---- Risk level ----
    risks = []
    if result.format_class == "UNKNOWN":
        risks.append("UNKNOWN_FORMAT")
    if sl_no_count == 0:
        risks.append("NO_PROJECTS")
    if records_at_page_boundary > 5:
        risks.append("MANY_PAGE_BREAKS")
    if result.report_type == "QUARTERLY":
        risks.append("QUARTERLY_FORMAT")
    if result.agency_layout == "NONE_DETECTED":
        risks.append("NO_AGENCIES")
    if sl_no_count > 0 and project_code_count / max(sl_no_count, 1) < 0.5:
        risks.append("LOW_CODE_RATIO")
    
    if len(risks) >= 2:
        result.risk_level = "HIGH"
    elif len(risks) == 1:
        result.risk_level = "MEDIUM"
    else:
        result.risk_level = "LOW"

    # ---- Parser compatibility ----
    if result.format_class in ("NEW_FORMAT_A",):
        result.parser_compatibility = "PARSER_COMPATIBLE"
    elif result.format_class in ("OLD_FORMAT_A",):
        result.parser_compatibility = "PARSER_COMPATIBLE"
    elif result.format_class in ("OLD_FORMAT_B",):
        # OLD_FORMAT_B is structurally like OLD_FORMAT_A but dates moved to
        # triple-paren lines. The existing OLD-format parser needs adaptation
        # to handle the absence of dates on the SL line.
        result.parser_compatibility = "PARSER_NEEDS_MINOR_ADAPTATION"
    elif result.format_class == "TRANSITIONAL_FORMAT":
        # Structurally very similar to OLD_FORMAT_B (triple-paren, no legacy)
        # but uses Table 6 label. The existing OLD-format parser can likely
        # handle these with the table-detection fix.
        result.parser_compatibility = "PARSER_NEEDS_MINOR_ADAPTATION"
    elif result.format_class == "NEW_FORMAT_B":
        # Fundamentally different parenthetical structure — triple parens are
        # nearly absent, replaced by separate-line parens. Needs new handler.
        result.parser_compatibility = "PARSER_NEEDS_NEW_HANDLER"
    elif result.format_class in ("NEW_FORMAT_VARIANT",):
        result.parser_compatibility = "PARSER_NEEDS_MINOR_ADAPTATION"
    elif result.format_class in ("QUARTERLY_FORMAT", "QUARTERLY_NEW"):
        result.parser_compatibility = "PARSER_NEEDS_NEW_HANDLER"
    else:
        result.parser_compatibility = "UNKNOWN"

    # ---- Notes ----
    notes_parts = []
    notes_parts.append(f"Table label: {table_label}")
    notes_parts.append(f"Serial numbers detected: {sl_no_count}")
    notes_parts.append(f"Project codes: {project_code_count}")
    notes_parts.append(f"Legacy codes: {legacy_code_count}")
    notes_parts.append(f"Triple paren lines: {paren_triple_count}")
    notes_parts.append(f"Dates on SL line: {dates_on_sl_line}")
    notes_parts.append(f"Sector/ministry headers: {sector_count}")
    notes_parts.append(f"Total lines: {total_line_count}")
    notes_parts.append(f"Agency on SL line: {agency_in_paren_on_sl_line}")
    notes_parts.append(f"Agency separate: {agency_on_separate_paren}")
    if risks:
        notes_parts.append(f"Risks: {', '.join(risks)}")
    result.notes = " | ".join(notes_parts)

    pdf.close()
    return result


# ────────────────────────────────────────────────────────────────────
# Write CSV
# ────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "filename", "report_month", "report_type", "pages", "format_class",
    "table6_start", "table6_end", "approx_project_count",
    "legacy_code_present", "agency_layout", "record_line_pattern",
    "page_break_pattern", "sector_headers", "special_patterns",
    "risk_level", "parser_compatibility", "notes",
]


def write_csv(results: list[PDFAuditResult], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {col: getattr(r, col, "") for col in CSV_COLUMNS}
            writer.writerow(row)
    log.info(f"CSV written: {out_path}")


# ────────────────────────────────────────────────────────────────────
# Write human-readable TXT
# ────────────────────────────────────────────────────────────────────

def write_txt(results: list[PDFAuditResult], out_path: Path) -> None:
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("SANKET — PDF FORMAT AUDIT REPORT")
    lines.append(sep)
    lines.append(f"Total PDFs audited: {len(results)}")
    lines.append("")

    # ---- Summary table ----
    lines.append("SUMMARY TABLE")
    lines.append("-" * 80)
    lines.append(f"{'Filename':<42} {'Month':<10} {'Fmt':<18} {'Pg':<5} {'Proj':<6} {'Parser':<30}")
    lines.append("-" * 80)
    for r in results:
        lines.append(
            f"{r.filename:<42} {r.report_month:<10} {r.format_class:<18} "
            f"{r.pages:<5} {r.approx_project_count:<6} {r.parser_compatibility:<30}"
        )
    lines.append("")

    # ---- Format class grouping ----
    format_groups: dict[str, list[PDFAuditResult]] = {}
    for r in results:
        format_groups.setdefault(r.format_class, []).append(r)

    lines.append(sep)
    lines.append("FORMAT CLASSES DISCOVERED")
    lines.append(sep)
    for fmt, members in sorted(format_groups.items()):
        lines.append(f"\n  {fmt} ({len(members)} PDFs)")
        lines.append(f"  {'─' * 40}")
        for m in members:
            lines.append(f"    - {m.filename} ({m.report_month})")
        # Describe characteristics
        sample = members[0]
        lines.append(f"    Characteristics:")
        lines.append(f"      Legacy codes: {sample.legacy_code_present}")
        lines.append(f"      Agency layout: {sample.agency_layout}")
        lines.append(f"      Record pattern: {sample.record_line_pattern}")
        lines.append(f"      Sector headers: {sample.sector_headers}")
        if sample.legacy_code_samples:
            lines.append(f"      Legacy code samples: {sample.legacy_code_samples[:3]}")
        if sample.paren_line_samples:
            lines.append(f"      Paren line samples: {sample.paren_line_samples[:3]}")

    # ---- Parser compatibility grouping ----
    compat_groups: dict[str, list[PDFAuditResult]] = {}
    for r in results:
        compat_groups.setdefault(r.parser_compatibility, []).append(r)

    lines.append("")
    lines.append(sep)
    lines.append("PARSER COMPATIBILITY")
    lines.append(sep)
    for compat, members in sorted(compat_groups.items()):
        lines.append(f"\n  {compat} ({len(members)} PDFs)")
        for m in members:
            lines.append(f"    - {m.filename}")

    # ---- Risk assessment ----
    lines.append("")
    lines.append(sep)
    lines.append("RISK ASSESSMENT")
    lines.append(sep)
    risk_groups: dict[str, list[PDFAuditResult]] = {}
    for r in results:
        risk_groups.setdefault(r.risk_level, []).append(r)
    for risk in ("HIGH", "MEDIUM", "LOW"):
        members = risk_groups.get(risk, [])
        if members:
            lines.append(f"\n  {risk} RISK ({len(members)} PDFs)")
            for m in members:
                lines.append(f"    - {m.filename}: {m.special_patterns}")

    # ---- Detailed per-PDF reports ----
    lines.append("")
    lines.append(sep)
    lines.append("DETAILED PER-PDF ANALYSIS")
    lines.append(sep)

    for r in results:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"FILE: {r.filename}")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Report Month:        {r.report_month}")
        lines.append(f"  Report Type:         {r.report_type}")
        lines.append(f"  Total Pages:         {r.pages}")
        lines.append(f"  Format Class:        {r.format_class}")
        lines.append(f"  Table 6 Range:       pages {r.table6_start}–{r.table6_end} ({r.table6_page_count} pages)")
        lines.append(f"  Approx Projects:     {r.approx_project_count}")
        lines.append(f"  Legacy Codes:        {r.legacy_code_present}")
        lines.append(f"  Agency Layout:       {r.agency_layout}")
        lines.append(f"    - On SL line:      {r.agency_in_parens_count}")
        lines.append(f"    - Separate line:   {r.agency_separate_line_count}")
        lines.append(f"  Record Pattern:      {r.record_line_pattern}")
        lines.append(f"  Page Break Pattern:  {r.page_break_pattern}")
        lines.append(f"  Sector Headers:      {r.sector_headers}")
        lines.append(f"  Special Patterns:    {r.special_patterns}")
        lines.append(f"  Risk Level:          {r.risk_level}")
        lines.append(f"  Parser Compat:       {r.parser_compatibility}")

        if r.column_headers_detected:
            lines.append(f"  Column Headers Detected:")
            for h in r.column_headers_detected[:5]:
                lines.append(f"    | {h[:100]}")

        if r.sl_no_line_samples:
            lines.append(f"  Serial Number Line Samples:")
            for s in r.sl_no_line_samples[:5]:
                lines.append(f"    | {s[:120]}")

        if r.legacy_code_samples:
            lines.append(f"  Legacy Code Samples:")
            for s in r.legacy_code_samples:
                lines.append(f"    | {s}")

        if r.paren_line_samples:
            lines.append(f"  Parenthetical Line Samples:")
            for s in r.paren_line_samples:
                lines.append(f"    | {s}")

        if r.project_code_samples:
            lines.append(f"  Project Code Samples:")
            for s in r.project_code_samples:
                lines.append(f"    | {s}")

        if r.sector_header_samples:
            lines.append(f"  Sector/Ministry Header Samples:")
            for s in r.sector_header_samples[:5]:
                lines.append(f"    | {s}")

        if r.page_break_records:
            lines.append(f"  Page Break Issues:")
            for s in r.page_break_records:
                lines.append(f"    | {s}")

        lines.append(f"  Notes: {r.notes}")

    # ---- Recommendations ----
    lines.append("")
    lines.append(sep)
    lines.append("RECOMMENDATIONS")
    lines.append(sep)

    compatible = [r for r in results if r.parser_compatibility == "PARSER_COMPATIBLE"]
    minor = [r for r in results if r.parser_compatibility == "PARSER_NEEDS_MINOR_ADAPTATION"]
    new_handler = [r for r in results if r.parser_compatibility == "PARSER_NEEDS_NEW_HANDLER"]
    unknown = [r for r in results if r.parser_compatibility == "UNKNOWN"]

    if compatible:
        lines.append(f"\n  READY TO EXTRACT ({len(compatible)} PDFs):")
        for r in compatible:
            lines.append(f"    ✓ {r.filename}")

    if minor:
        lines.append(f"\n  NEEDS MINOR ADAPTATION ({len(minor)} PDFs):")
        for r in minor:
            lines.append(f"    ~ {r.filename} — {r.format_class}")

    if new_handler:
        lines.append(f"\n  NEEDS NEW HANDLER ({len(new_handler)} PDFs):")
        for r in new_handler:
            lines.append(f"    ✗ {r.filename} — {r.format_class}")

    if unknown:
        lines.append(f"\n  UNKNOWN COMPATIBILITY ({len(unknown)} PDFs):")
        for r in unknown:
            lines.append(f"    ? {r.filename}")

    lines.append("")
    lines.append(sep)
    lines.append("END OF AUDIT REPORT")
    lines.append(sep)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"TXT report written: {out_path}")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    """Run the full audit on all PDFs."""
    safe_print("\n" + "=" * 60)
    safe_print("SANKET — PDF FORMAT AUDIT")
    safe_print("=" * 60 + "\n")

    pdf_dir = RAW_PDF_DIR
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    safe_print(f"Found {len(pdfs)} PDFs in {pdf_dir}\n")

    results: list[PDFAuditResult] = []
    for pdf_path in pdfs:
        result = audit_single_pdf(pdf_path)
        results.append(result)
        safe_print(
            f"  [{result.format_class:<18}] {result.filename:<42} "
            f"pages={result.pages:<4} projects≈{result.approx_project_count:<5} "
            f"{result.parser_compatibility}"
        )

    # Sort by report_month for consistent output
    results.sort(key=lambda r: r.report_month or "9999")

    # Write outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "pdf_format_audit.csv"
    txt_path = REPORTS_DIR / "pdf_format_audit.txt"
    write_csv(results, csv_path)
    write_txt(results, txt_path)

    # Print summary
    safe_print(f"\n{'=' * 60}")
    safe_print(f"AUDIT COMPLETE")
    safe_print(f"{'=' * 60}")
    safe_print(f"  CSV: {csv_path}")
    safe_print(f"  TXT: {txt_path}")

    # Quick stats
    format_counts = Counter(r.format_class for r in results)
    safe_print(f"\n  Format classes discovered:")
    for fmt, count in sorted(format_counts.items()):
        safe_print(f"    {fmt}: {count} PDFs")

    compat_counts = Counter(r.parser_compatibility for r in results)
    safe_print(f"\n  Parser compatibility:")
    for compat, count in sorted(compat_counts.items()):
        safe_print(f"    {compat}: {count} PDFs")

    risk_counts = Counter(r.risk_level for r in results)
    safe_print(f"\n  Risk levels:")
    for risk, count in sorted(risk_counts.items()):
        safe_print(f"    {risk}: {count} PDFs")


if __name__ == "__main__":
    main()
