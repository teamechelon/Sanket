"""
SANKET — PDF Extractor
=======================
Extract project-level and aggregate data from PAIMANA Flash Report PDFs.

The reports have several layout generations.  The original implementation
started with an OLD/NEW split; verified April--July 2026 pages need a
dedicated handler because their post-serial lines can contain project code,
legacy OCMS code, PMGID, dates, and costs in more than one arrangement.

The Table 6 ("All Ongoing Projects") parser uses a state-machine approach
to reconstruct multi-line project records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

from src.utils import (
    RAW_PDF_DIR,
    PROCESSED_DIR,
    get_logger,
    infer_month_from_path,
    infer_month_from_text,
    parse_numeric,
    safe_print,
)

log = get_logger("pdf_extractor")


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

# Lines containing these are page headers/footers to skip
_SKIP_PATTERNS = [
    "Project Assessment, Infrastructure Monitoring",
    "Nation-building For details visit",
    "ipm.mospi.gov.in",
    "(PAIMANA)",
]

_PAGE_NUMBER_RE = re.compile(r"^Page \d+$")

# Serial number at start of line: "1 ", "23 ", "1941 "
_SL_NO_RE = re.compile(r"^(\d+)\s+(.+)$")

# Date pattern: MM/YYYY
_DATE_RE = re.compile(r"\d{2}/\d{4}")
_FLEXIBLE_MONTH_RE = re.compile(
    r"\b(\d{1,2})[/-](\d{4})\b|\b"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ](\d{4})\b",
    re.IGNORECASE,
)

# Parenthetical line: "(612786)" or "(N04000106)" or "(01/2024) (05/2026) (265.91)"
_PROJECT_CODE_RE = re.compile(r"^\((\d{5,7})\)$")
_LEGACY_CODE_RE = re.compile(r"^\(([NO]\d{7,9})\)$")
_PAREN_VALUES_RE = re.compile(r"^\(([^)]*)\)\s*\(([^)]*)\)\s*\(([^)]*)\)$")
_SINGLE_PAREN_RE = re.compile(r"^\(([^)]+)\)$")
_PAREN_GROUP_RE = re.compile(r"\(([^)]*)\)")

# Known Indian states/UTs for detection
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

# Sector header detection — these appear as standalone lines before records
_MINISTRY_RE = re.compile(r"^Ministry of .+$|^Department of .+$")


# ────────────────────────────────────────────────────────────────────
# PDF Section Detection
# ────────────────────────────────────────────────────────────────────

@dataclass
class PDFSection:
    """A detected section within a Flash Report PDF."""
    name: str
    start_page: int  # 0-indexed
    end_page: int | None = None  # 0-indexed, inclusive


def detect_sections(pdf: pdfplumber.PDF) -> list[PDFSection]:
    """Detect major sections and their page ranges in a Flash Report PDF."""
    sections: list[PDFSection] = []
    table6_pages: list[int] = []

    for i, page in enumerate(pdf.pages):
        text = (page.extract_text() or "").strip()
        first_300 = text[:300]

        if "All Ongoing Projects" in first_300:
            table6_pages.append(i)

    if table6_pages:
        sections.append(PDFSection(
            name="Table 6: All Ongoing Projects",
            start_page=table6_pages[0],
            end_page=table6_pages[-1],
        ))

    return sections


def detect_format_generation(pdf: pdfplumber.PDF) -> str:
    """
    Determine whether a PDF uses OLD or NEW format.

    NEW format (Sep 2025+): Has "Table 5:" and "Table 6:" in TOC,
    and legacy OCMS codes in project records.
    OLD format: Has only "Table 3:" or "Table 4:" as last table.
    """
    # Check the TOC page (usually page 2)
    for i in range(min(3, len(pdf.pages))):
        text = (pdf.pages[i].extract_text() or "")
        if "Table 6:" in text or "Table 5:" in text:
            return "NEW"
        if "Table 4: All Ongoing" in text:
            return "OLD"
    return "UNKNOWN"


def detect_report_format(pdf_path: Path, report_month: str | None) -> str:
    """Route only the report periods whose layouts have been inspected.

    The document's table labels are not sufficient to distinguish every
    generation, so this deliberately uses verified report periods rather than
    guessing from incidental text.  Unknown periods retain the legacy route.
    """
    month = report_month or infer_month_from_path(pdf_path)
    if month == "2025-07":
        return "OLD_FORMAT_A"
    if month == "2025-08":
        return "OLD_FORMAT_B"
    if month in {"2025-09", "2025-10", "2025-11", "2025-12", "2026-01"}:
        return "TRANSITIONAL_FORMAT"
    if month in {"2026-02", "2026-03"}:
        return "NEW_FORMAT_A"
    if month in {"2026-04", "2026-05", "2026-06", "2026-07"}:
        return "NEW_FORMAT_B"
    if pdf_path.name.startswith(("FRApril", "FR_May", "FR_JUNE")):
        return "EARLY_FORMAT"
    if pdf_path.name.startswith("QPISR"):
        return "QUARTERLY"
    return "UNKNOWN"


# ────────────────────────────────────────────────────────────────────
# Line Classification
# ────────────────────────────────────────────────────────────────────

def _is_skip_line(line: str) -> bool:
    """Return True if this line is a header/footer to skip."""
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
    """Return True if this is a repeated table column header."""
    stripped = line.strip()
    _HEADER_STARTS = (
        "All Ongoing Projects",
        "Orignal/Target",
        "Original/Target",
        "Date of Approval",
        "Date of Orignal",
        "Date of Original",
        "Sl.No",
        "Sl. No",
        "MM/YYYY",
        "in Rs.",
        "(Revised DoC)",
        "(Revised Cost)",
        "Project Name",
        "(Legacy OCMS Code)",
        "Legacy OCMS",
        "Cumulative",
        "Physical Progress",
        "Expenditure",
        "Table 6:",
        "Table 4:",
    )
    for prefix in _HEADER_STARTS:
        if stripped.startswith(prefix):
            return True
    # Also catch combined header fragments like:
    # "Orignal Cost Cumulative" or "Revised Cost Expenditure"
    if re.match(r"^(Orignal|Original|Revised)\s+(Cost|DoC)", stripped):
        return True
    # Substring checks for fragments that appear mid-line when pdfplumber
    # joins column headers into a single text line
    _HEADER_SUBSTRINGS = (
        "Legacy OCMS Code",
        "in Rs. Crore",
        "Project Name (Agency)",
        "Approval Revised Cost",
        "(Project Code)",
    )
    for sub in _HEADER_SUBSTRINGS:
        if sub in stripped:
            return True
    return False


def _is_month_header(line: str) -> bool:
    """Return True for lines like 'MARCH 2026' or 'JULY 2025'."""
    return bool(re.match(
        r"^(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}$",
        line.strip(),
    ))


def _is_sector_or_ministry_header(line: str) -> bool:
    """Return True if this line is a sector or ministry group header."""
    stripped = line.strip()
    if _MINISTRY_RE.match(stripped):
        return True
    # Known sector names that appear as standalone headers
    if stripped in (
        "Aviation & Aviation Infrastructure", "Coal", "Oil & Gas",
        "Transmission & Distribution", "Water Resources",
        "Electricity Generation", "Waste & Water", "Education",
        "Urban Public Transport", "Steel", "Energy Storage",
        "Telecommunication", "Metals & Mining", "Real Estate",
        "Shipping", "Healthcare", "Inland Waterways", "Construction",
        "Tourism, Hospitality & Wellness", "Railways",
        "Roads & Highways", "Logistics Infrastructure",
    ):
        return True
    return False


def _is_total_line(line: str) -> bool:
    """Detect 'Total (N)' summary lines between sector groups."""
    return bool(re.match(r"^Total\s*\(\d+\)", line.strip()))


def _find_state_in_line(line: str) -> str | None:
    """Find a known Indian state name in a line."""
    for state in sorted(_KNOWN_STATES, key=len, reverse=True):
        if state in line:
            return state
    return None


def _extract_dates_from_line(line: str) -> list[str]:
    """Extract all MM/YYYY dates from a line."""
    return _DATE_RE.findall(line)


def _extract_trailing_numbers(line: str) -> list[float]:
    """Extract trailing numeric values from a line."""
    # Split line and take trailing numeric parts
    parts = line.strip().split()
    nums = []
    for p in reversed(parts):
        val = parse_numeric(p)
        if val is not None:
            nums.insert(0, val)
        else:
            break
    return nums


# ────────────────────────────────────────────────────────────────────
# Record Assembly — State Machine
# ────────────────────────────────────────────────────────────────────

@dataclass
class ProjectRecord:
    """A single project record extracted from Table 6."""
    sl_no: int | None = None
    project_name: str = ""
    agency: str = ""
    state: str = ""
    sector: str = ""
    ministry: str = ""
    date_of_approval: str = ""
    start_date: str = ""
    original_doc: str = ""
    revised_doc: str = ""
    original_cost: float | None = None
    revised_cost: float | None = None
    expenditure: float | None = None
    physical_progress: float | None = None
    project_code: str = ""
    legacy_ocms_code: str = ""
    identifier_status: str = "UNKNOWN"
    source_page: int = 0
    raw_lines: list[str] = field(default_factory=list)
    extraction_status: str = "SUCCESS"


def _parse_sl_no_line_new_format(line: str, record: ProjectRecord) -> None:
    """
    Parse the main data line in NEW format:
    'Sl_No [(Agency)] State Expenditure Progress%'
    
    Examples:
    '1 (Airport Authority of India [AAI]) Andhra Pradesh 120.19 60'
    '2 Andhra Pradesh 523.14 87.2'
    '7 (Airport Authority of India [AAI]) Bihar 1196.4 98.04'
    """
    m = _SL_NO_RE.match(line.strip())
    if not m:
        return
    
    record.sl_no = int(m.group(1))
    rest = m.group(2)
    
    # Extract trailing numbers (expenditure, progress)
    trailing = _extract_trailing_numbers(rest)
    if len(trailing) >= 2:
        record.expenditure = trailing[-2]
        record.physical_progress = trailing[-1]
    elif len(trailing) == 1:
        record.physical_progress = trailing[0]
    
    # Remove trailing numbers from rest
    parts = rest.strip().split()
    while parts and parse_numeric(parts[-1]) is not None:
        parts.pop()
    text_part = " ".join(parts)
    
    # Extract agency if present (in parentheses)
    agency_match = re.match(r"^\(([^)]+)\)\s*(.*)", text_part)
    if agency_match:
        record.agency = agency_match.group(1).strip()
        text_part = agency_match.group(2).strip()
    
    # Remaining text should contain state
    state = _find_state_in_line(text_part)
    if state:
        record.state = state
    elif text_part.strip():
        # Might be a state we don't recognize, or project name overflow
        record.state = text_part.strip()


def _parse_sl_no_line_old_format(line: str, record: ProjectRecord) -> None:
    """
    Parse the main data line in OLD format:
    'Sl_No [(Agency)] State Start_Date Expenditure Progress%'
    
    Examples:
    '1 Andhra Pradesh 01/2024 45.54 31'
    '4 (Adani Airport Holdings Limited) Assam 03/2018 0 94.1'
    '7 Bihar 10/2018 1046.48 94'
    """
    m = _SL_NO_RE.match(line.strip())
    if not m:
        return
    
    record.sl_no = int(m.group(1))
    rest = m.group(2)
    
    # Extract trailing numbers (expenditure, progress)
    trailing = _extract_trailing_numbers(rest)
    if len(trailing) >= 2:
        record.expenditure = trailing[-2]
        record.physical_progress = trailing[-1]
    elif len(trailing) == 1:
        record.physical_progress = trailing[0]
    
    # Remove trailing numbers
    parts = rest.strip().split()
    while parts and parse_numeric(parts[-1]) is not None:
        parts.pop()
    
    # Extract dates (start_date in old format)
    dates = _extract_dates_from_line(rest)
    if dates:
        record.start_date = dates[0]
        # Remove dates from parts
        for d in dates:
            if d in parts:
                parts.remove(d)
    
    text_part = " ".join(parts)
    
    # Extract agency
    agency_match = re.match(r"^\(([^)]+)\)\s*(.*)", text_part)
    if agency_match:
        record.agency = agency_match.group(1).strip()
        text_part = agency_match.group(2).strip()
    
    # Extract state
    state = _find_state_in_line(text_part)
    if state:
        record.state = state


def _parse_parenthetical_line_new(line: str, record: ProjectRecord) -> bool:
    """
    Parse '(start_date) (revised_doc) (revised_cost)' or
    '(project_code) (start_date) (revised_doc) (revised_cost)' lines.
    Returns True if it was a recognized parenthetical line.
    """
    stripped = line.strip()
    
    # Check for project code: "(612786)"
    m = _PROJECT_CODE_RE.match(stripped)
    if m:
        record.project_code = m.group(1)
        return True
    
    # Check for legacy code: "(N04000106)"
    m = _LEGACY_CODE_RE.match(stripped)
    if m:
        record.legacy_ocms_code = m.group(1)
        return True
    
    # Check for single parenthetical like "(-)"
    if stripped == "(-)":
        # Could be missing legacy code or missing revised_doc
        return True
    
    # Check for triple parenthetical: "(01/2024) (05/2026) (265.91)"
    m = _PAREN_VALUES_RE.match(stripped)
    if m:
        v1, v2, v3 = m.group(1), m.group(2), m.group(3)
        record.start_date = v1 if v1 != "-" else ""
        record.revised_doc = v2 if v2 != "-" else ""
        val = parse_numeric(v3)
        if val is not None:
            record.revised_cost = val
        return True
    
    # Check for quadruple: "(project_code) (start_date) (revised_doc) (revised_cost)"
    quad = re.match(r"^\(([^)]+)\)\s+\(([^)]*)\)\s+\(([^)]*)\)\s+\(([^)]*)\)$", stripped)
    if quad:
        record.project_code = quad.group(1)
        record.start_date = quad.group(2) if quad.group(2) != "-" else ""
        record.revised_doc = quad.group(3) if quad.group(3) != "-" else ""
        val = parse_numeric(quad.group(4))
        if val is not None:
            record.revised_cost = val
        return True
    
    return False


def _parse_parenthetical_line_new_format_b(line: str, record: ProjectRecord) -> bool:
    """Consume the flexible post-serial lines used from April 2026 onward.

    A representative record can place its fields on either of these lines::

        (612786) (01/2024) (07/2026) (265.91)
        (N04000106) (-)

    or put the agency in the first line and the project code in the second.
    Treating only four-parenthesis lines as valid caused legacy/PMGID lines to
    leak into the following project's name.
    """
    groups = [value.strip() for value in _PAREN_GROUP_RE.findall(line.strip())]
    if not groups or not re.fullmatch(r"\s*(?:\([^)]*\)\s*)+", line):
        return False

    dates: list[str] = []
    numeric_values: list[float] = []
    has_text_agency = False
    recognised = False

    for value in groups:
        if _PROJECT_CODE_RE.match(f"({value})"):
            record.project_code = value
            recognised = True
        elif _LEGACY_CODE_RE.match(f"({value})"):
            record.legacy_ocms_code = value
            recognised = True
        elif _DATE_RE.fullmatch(value):
            dates.append(value)
            recognised = True
        elif value in {"-", "", "NA", "N/A"}:
            recognised = True
        else:
            number = parse_numeric(value)
            if number is not None:
                numeric_values.append(number)
                recognised = True
            elif any(char.isalpha() for char in value):
                if not record.agency:
                    record.agency = value
                has_text_agency = True
                recognised = True

    if dates:
        record.start_date = dates[0]
        if len(dates) > 1:
            record.revised_doc = dates[1]

    # A two-group legacy/PMGID or placeholder/PMGID line must never be read as
    # a revised cost.  Cost appears only with dates or an agency in this layout.
    if numeric_values and (dates or has_text_agency) and len(groups) >= 3:
        record.revised_cost = numeric_values[-1]

    return recognised


def _parse_parenthetical_line_old(line: str, record: ProjectRecord) -> bool:
    """
    Parse OLD format parenthetical/agency lines.
    Old format has:
    '(Agency) Revised_DoC (Revised_Cost)'  or  '(-) (Revised_Cost)'
    '(Project_Code)'
    """
    stripped = line.strip()
    
    # Project code
    m = _PROJECT_CODE_RE.match(stripped)
    if m:
        record.project_code = m.group(1)
        return True
    
    # Single paren like "(-)"
    if stripped == "(-)":
        return True
    
    # Agency + revised values: "(Airport Authority of India [AAI]) 03/2026 (265.91)"
    # or "(-) (1712)"
    agency_revised = re.match(r"^\(([^)]+)\)\s+(\d{2}/\d{4})?\s*\(([^)]+)\)$", stripped)
    if agency_revised:
        agency_text = agency_revised.group(1)
        if agency_text != "-":
            if not record.agency:
                record.agency = agency_text
        if agency_revised.group(2):
            record.revised_doc = agency_revised.group(2)
        val = parse_numeric(agency_revised.group(3))
        if val is not None:
            record.revised_cost = val
        return True
    
    # Just revised values: "(-) (1712)" or "12/2025 (611.8)"
    rev_match = re.match(r"^(?:\(-\)|(\d{2}/\d{4}))\s+\(([^)]+)\)$", stripped)
    if rev_match:
        if rev_match.group(1):
            record.revised_doc = rev_match.group(1)
        val = parse_numeric(rev_match.group(2))
        if val is not None:
            record.revised_cost = val
        return True
    
    # Standalone agency line: "(Airport Authority of India [AAI])"
    m = _SINGLE_PAREN_RE.match(stripped)
    if m:
        content = m.group(1)
        if not content.replace("-", "").strip():
            return True  # just "(-)"-like
        if not content[0].isdigit():
            # Looks like agency name
            if not record.agency:
                record.agency = content
            return True
    
    return False


def _parse_name_date_cost_line(line: str, record: ProjectRecord) -> None:
    """
    Parse lines above the Sl.No line that contain project name
    and optionally trailing dates and cost.
    
    Examples:
    'Construction of New Domestic Terminal Building ...'           (name only)
    'including maintenance, operations and AICMC at Kadapa Airport'  (name continuation)
    '03/2023 01/2026 265.91'                                     (dates + cost only)
    'Airport. 06/2020 09/2022 611.8'                             (name fragment + dates + cost)
    '(Agency) 12/2022 08/2025 347.15'                            (agency + dates + cost)
    """
    stripped = line.strip()
    
    dates = _extract_dates_from_line(stripped)
    trailing_nums = _extract_trailing_numbers(stripped)
    
    # If line has dates and/or trailing numbers, extract them
    if dates or trailing_nums:
        # Remove the numeric/date portions to get the text
        text = stripped
        for d in dates:
            text = text.replace(d, "", 1)
        for n in trailing_nums:
            # Remove last occurrence of the number string
            for fmt in [str(int(n)), f"{n:.1f}", f"{n:.2f}", str(n)]:
                idx = text.rfind(fmt)
                if idx >= 0:
                    text = text[:idx] + text[idx+len(fmt):]
                    break
        text = text.strip()
        # Clean up any trailing numeric fragments (e.g. ".91" leftover)
        text = re.sub(r"\s*\.\d+\s*$", "", text)
        text = text.strip()
        
        # Assign dates based on count and context
        if len(dates) >= 2:
            record.date_of_approval = dates[0]
            record.original_doc = dates[1]
        elif len(dates) == 1:
            if not record.date_of_approval:
                record.date_of_approval = dates[0]
            else:
                record.original_doc = dates[0]
        
        # Trailing number is original cost
        if trailing_nums:
            record.original_cost = trailing_nums[-1]
        
        # Check if text starts with agency in parens
        if text.startswith("("):
            agency_m = re.match(r"^\(([^)]+)\)\s*(.*)$", text)
            if agency_m:
                if not record.agency:
                    record.agency = agency_m.group(1).strip()
                text = agency_m.group(2).strip()
        
        # Remaining text is part of project name
        if text and text not in (".", ""):
            if record.project_name:
                record.project_name += " " + text
            else:
                record.project_name = text
    else:
        # Pure text line — part of project name or agency
        if stripped.startswith("("):
            agency_m = re.match(r"^\(([^)]+)\)\s*(.*)$", stripped)
            if agency_m:
                content = agency_m.group(1).strip()
                remaining = agency_m.group(2).strip()
                if not content[0].isdigit() and content != "-":
                    if not record.agency:
                        record.agency = content
                    if remaining:
                        if record.project_name:
                            record.project_name += " " + remaining
                        else:
                            record.project_name = remaining
                    return
        
        # Append to project name
        if record.project_name:
            record.project_name += " " + stripped
        else:
            record.project_name = stripped


# ────────────────────────────────────────────────────────────────────
# Table 6 Extraction — Main State Machine
# ────────────────────────────────────────────────────────────────────

def extract_table6(
    pdf: pdfplumber.PDF,
    table6_pages: list[int],
    format_gen: str,
    report_month: str,
    source_file: str,
) -> list[ProjectRecord]:
    """
    Extract all project records from Table 6 pages using a state machine.
    
    The state machine accumulates lines between serial-number boundaries.
    When a new serial number is detected, the previous record is finalized.
    """
    records: list[ProjectRecord] = []
    
    # Collect all data lines from Table 6 pages, with page tracking
    all_lines: list[tuple[str, int]] = []  # (line_text, page_number)
    
    for pi in table6_pages:
        text = pdf.pages[pi].extract_text() or ""
        for line in text.split("\n"):
            if _is_skip_line(line):
                continue
            if _is_table_header(line):
                continue
            if _is_month_header(line):
                continue
            all_lines.append((line.strip(), pi + 1))
    
    # State machine variables
    current_record: ProjectRecord | None = None
    current_sector = ""
    current_ministry = ""
    pre_record_lines: list[tuple[str, int]] = []  # lines before sl_no is seen
    post_record_phase = 0  # 0=pre-name, 1=post-sl_no
    
    def finalize_record():
        nonlocal current_record
        if current_record and current_record.sl_no is not None:
            current_record.sector = current_sector
            current_record.ministry = current_ministry
            current_record.project_name = re.sub(r"\s+", " ", current_record.project_name).strip()
            current_record.agency = re.sub(r"\s+", " ", current_record.agency).strip()
            
            # Validate project_code: must be 5-7 digit numeric string
            if current_record.project_code:
                if not re.match(r"^\d{5,7}$", current_record.project_code):
                    # Invalid code — likely an agency name or state that leaked in
                    log.debug(
                        f"  Invalid project_code '{current_record.project_code}' "
                        f"for sl_no={current_record.sl_no} — clearing"
                    )
                    current_record.project_code = ""
            
            # Validate legacy_ocms_code: must match N/O + digits pattern
            if current_record.legacy_ocms_code:
                if not re.match(r"^[NO]\d{7,9}$", current_record.legacy_ocms_code):
                    current_record.legacy_ocms_code = ""
            
            # Set extraction status
            if not current_record.project_code:
                current_record.extraction_status = "PARTIAL"
            records.append(current_record)
        current_record = None
    
    # State: track the last confirmed serial number for continuity checks
    last_sl_no = 0
    
    i = 0
    while i < len(all_lines):
        line, page = all_lines[i]
        
        # Check for sector/ministry headers
        if _is_sector_or_ministry_header(line):
            if _MINISTRY_RE.match(line):
                current_ministry = line.strip()
            else:
                current_sector = line.strip()
            i += 1
            continue
        
        # Check for total lines
        if _is_total_line(line):
            finalize_record()
            pre_record_lines = []
            i += 1
            continue
        
        # Check if this line starts with a serial number
        sl_match = _SL_NO_RE.match(line)
        if sl_match:
            sl_no_candidate = int(sl_match.group(1))
            rest = sl_match.group(2)
            
            # Validate it's a plausible serial number
            # (not just a number that happens to start a project name,
            #  e.g. chainage "297700 to Km 308729..." or "47 in the State of...")
            state_found = _find_state_in_line(rest)
            trailing = _extract_trailing_numbers(rest)
            
            is_sl_no = False
            
            # Continuity check: sl_no should be close to the last known one
            # Allow some gap (projects can be skipped between sectors) but not
            # jumps like 1371 → 297700
            max_jump = max(200, last_sl_no * 0.5) if last_sl_no > 0 else 3000
            in_sequence = (
                last_sl_no == 0  # first record
                or abs(sl_no_candidate - last_sl_no) <= max_jump
            )
            
            if state_found and len(trailing) >= 1 and in_sequence:
                is_sl_no = True
            elif sl_no_candidate <= 3000 and len(trailing) >= 2 and in_sequence:
                # Even without recognized state, if there are trailing numbers 
                # and it's in sequence, it's likely a data line
                is_sl_no = True
            
            if is_sl_no:
                # Finalize previous record
                finalize_record()
                
                # Start new record
                current_record = ProjectRecord(source_page=page)
                current_record.raw_lines = [l for l, _ in pre_record_lines] + [line]
                
                # Parse the pre-record lines as project name / date / cost
                for pre_line, _ in pre_record_lines:
                    _parse_name_date_cost_line(pre_line, current_record)
                
                # Parse the sl_no line
                if format_gen in {"NEW", "NEW_FORMAT_A", "NEW_FORMAT_B"}:
                    _parse_sl_no_line_new_format(line, current_record)
                else:
                    _parse_sl_no_line_old_format(line, current_record)
                
                # Update continuity tracker
                last_sl_no = sl_no_candidate
                
                pre_record_lines = []
                post_record_phase = 1
                i += 1
                
                # Now consume the post-sl_no lines (parenthetical values, codes)
                while i < len(all_lines):
                    next_line, next_page = all_lines[i]
                    
                    if _is_sector_or_ministry_header(next_line):
                        break
                    if _is_total_line(next_line):
                        break
                    
                    # Check if next line is another sl_no
                    next_sl = _SL_NO_RE.match(next_line)
                    if next_sl:
                        next_sl_no = int(next_sl.group(1))
                        next_rest = next_sl.group(2)
                        next_state = _find_state_in_line(next_rest)
                        next_trailing = _extract_trailing_numbers(next_rest)
                        next_max_jump = max(200, last_sl_no * 0.5) if last_sl_no > 0 else 3000
                        next_in_seq = abs(next_sl_no - last_sl_no) <= next_max_jump
                        if next_in_seq and ((next_state and len(next_trailing) >= 1) or (next_sl_no <= 3000 and len(next_trailing) >= 2)):
                            break
                    
                    # Try to parse as parenthetical
                    if format_gen == "NEW_FORMAT_B":
                        parsed = _parse_parenthetical_line_new_format_b(next_line, current_record)
                    elif format_gen in {"NEW", "NEW_FORMAT_A"}:
                        parsed = _parse_parenthetical_line_new(next_line, current_record)
                    else:
                        parsed = _parse_parenthetical_line_old(next_line, current_record)
                    
                    if parsed:
                        current_record.raw_lines.append(next_line)
                        i += 1
                    else:
                        # This is the start of the next project's name
                        break
                
                # Remaining lines before next sl_no are the next record's name
                pre_record_lines = []
                continue
            
        # Not a serial number line — accumulate as pre-record (name) lines
        if current_record is None:
            pre_record_lines.append((line, page))
        else:
            # We're between records — this line is part of the next record's name
            pre_record_lines.append((line, page))
        
        i += 1
    
    # Finalize last record
    finalize_record()
    
    log.info(f"  Extracted {len(records)} project records from Table 6")
    return records


def extract_new_format_b(
    pdf: pdfplumber.PDF,
    table6_pages: list[int],
    report_month: str,
    source_file: str,
) -> list[ProjectRecord]:
    """Extract April--July 2026 project records with their verified layout."""
    return extract_table6(
        pdf=pdf,
        table6_pages=table6_pages,
        format_gen="NEW_FORMAT_B",
        report_month=report_month,
        source_file=source_file,
    )


# ────────────────────────────────────────────────────────────────────
# Table 7 extraction — April--June 2025 and quarterly reports
# ────────────────────────────────────────────────────────────────────

_MONTH_ABBREVIATIONS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalise_flexible_month(value: str) -> str | None:
    """Normalise a Table 7 month value to MM/YYYY without inventing a date."""
    match = _FLEXIBLE_MONTH_RE.search(value.replace("{", "").replace("}", ""))
    if not match:
        return None
    if match.group(1):
        return f"{int(match.group(1)):02d}/{match.group(2)}"
    return f"{_MONTH_ABBREVIATIONS[match.group(3).lower()[:3]]}/{match.group(4)}"


def _table7_original_and_revised_dates(value: str) -> tuple[str, str]:
    """Read original and revised dates, deliberately excluding anticipated dates.

    Table 7 uses ``{...}`` for anticipated values. They are useful provenance
    but must not be silently labelled as a revision.
    """
    original = ""
    revised = ""
    for raw in (value or "").splitlines():
        line = raw.strip()
        date = _normalise_flexible_month(line)
        if not date or line.startswith("{"):
            continue
        if line.startswith("("):
            revised = date
        elif not original:
            original = date
    return original, revised


def _table7_original_and_revised_costs(value: str) -> tuple[float | None, float | None]:
    """Read original/revised cost while deliberately excluding anticipated cost."""
    original: float | None = None
    revised: float | None = None
    for raw in (value or "").splitlines():
        line = raw.strip()
        if line.startswith("{"):
            continue
        number = parse_numeric(line.strip("() "))
        if number is None:
            continue
        if line.startswith("("):
            revised = number
        elif original is None:
            original = number
    return original, revised


def _parse_table7_project_cell(value: str) -> tuple[str, str, str]:
    """Return project name, agency, and source-supported alternate identifier."""
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    code_index = next(
        (index for index in range(len(lines) - 1, -1, -1) if _LEGACY_CODE_RE.match(lines[index])),
        None,
    )
    alternate_id = ""
    if code_index is not None:
        alternate_id = _SINGLE_PAREN_RE.match(lines[code_index]).group(1).strip()

    # The agency is the final parenthesised text immediately before the code.
    # Earlier parentheses can be part of a project name, e.g. a location.
    agency = ""
    agency_index: int | None = None
    if code_index is not None and code_index > 0:
        candidate = _SINGLE_PAREN_RE.match(lines[code_index - 1])
        if candidate:
            content = candidate.group(1).strip()
            if content not in {"-", "N.A.", "N/A"}:
                agency = content
                agency_index = code_index - 1

    name_lines = [
        line
        for index, line in enumerate(lines)
        if index not in {code_index, agency_index}
    ]
    return " ".join(name_lines), agency, alternate_id


def _is_table7_page(page: pdfplumber.page.Page) -> bool:
    text = page.extract_text() or ""
    return "Table:-7. Project List: Ongoing Projects" in text and "Sl No" in text


def extract_table7(
    pdf: pdfplumber.PDF,
    table7_pages: list[int],
    report_month: str,
    source_file: str,
) -> list[ProjectRecord]:
    """Extract the structured nine-column project list used by Table 7."""
    records: list[ProjectRecord] = []
    current_state = ""
    current_sector = ""

    for page_index in table7_pages:
        table = pdf.pages[page_index].extract_table()
        if not table:
            log.warning("  Table 7 grid not detected on page %s", page_index + 1)
            continue
        for row in table[1:]:  # first row is the repeated column header
            if len(row) < 9:
                continue
            state, sector, sl_no, project_cell, approval, commissioning, costs, expenditure, progress = row[:9]
            if state and state.strip():
                current_state = " ".join(state.split())
            if sector and sector.strip():
                current_sector = " ".join(sector.split())
            if not sl_no or not re.fullmatch(r"\d+", sl_no.strip()):
                continue

            project_name, agency, alternate_id = _parse_table7_project_cell(project_cell or "")
            original_doc, revised_doc = _table7_original_and_revised_dates(commissioning or "")
            original_cost, revised_cost = _table7_original_and_revised_costs(costs or "")
            expenditure_values = [parse_numeric((expenditure or "").strip())]
            progress_values = [parse_numeric((progress or "").strip())]
            record = ProjectRecord(
                sl_no=int(sl_no),
                project_name=project_name,
                agency=agency,
                state=current_state,
                sector=current_sector,
                date_of_approval=_normalise_flexible_month(approval or "") or "",
                original_doc=original_doc,
                revised_doc=revised_doc,
                original_cost=original_cost,
                revised_cost=revised_cost,
                expenditure=expenditure_values[0] if expenditure_values else None,
                physical_progress=progress_values[0] if progress_values else None,
                legacy_ocms_code=alternate_id,
                identifier_status="ALTERNATE_IDENTIFIER" if alternate_id else "SOURCE_MISSING_ID",
                source_page=page_index + 1,
                raw_lines=[cell or "" for cell in row[:9]],
                extraction_status="SUCCESS" if alternate_id else "PARTIAL",
            )
            records.append(record)

    log.info("  Extracted %s project records from Table 7", len(records))
    return records


def extract_early_format(
    pdf: pdfplumber.PDF, table7_pages: list[int], report_month: str, source_file: str
) -> list[ProjectRecord]:
    """Shared handler for the verified April--June 2025 Table 7 layout."""
    return extract_table7(pdf, table7_pages, report_month, source_file)


def extract_quarterly_format(
    pdf: pdfplumber.PDF, table7_pages: list[int], report_month: str, source_file: str
) -> list[ProjectRecord]:
    """Quarterly Table 7 handler; callers must store its observations separately."""
    return extract_table7(pdf, table7_pages, report_month, source_file)


# ────────────────────────────────────────────────────────────────────
# Records → DataFrame
# ────────────────────────────────────────────────────────────────────

def records_to_dataframe(
    records: list[ProjectRecord],
    report_month: str,
    source_file: str,
    source_section: str = "Table 6: All Ongoing Projects",
) -> pd.DataFrame:
    """Convert extracted ProjectRecords to a DataFrame."""
    rows = []
    for r in records:
        rows.append({
            "report_month": report_month,
            "sl_no": r.sl_no,
            "project_code": r.project_code or None,
            "legacy_ocms_code": r.legacy_ocms_code or None,
            "identifier_status": r.identifier_status,
            "project_name": r.project_name or None,
            "agency": r.agency or None,
            "ministry": r.ministry or None,
            "sector": r.sector or None,
            "state": r.state or None,
            "date_of_approval": r.date_of_approval or None,
            "start_date": r.start_date or None,
            "original_doc": r.original_doc or None,
            "revised_doc": r.revised_doc or None,
            "original_cost": r.original_cost,
            "revised_cost": r.revised_cost,
            "expenditure": r.expenditure,
            "physical_progress": r.physical_progress,
            "source_file": source_file,
            "source_page": r.source_page,
            "source_section": source_section,
            "extraction_status": r.extraction_status,
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# High-Level Extraction Interface
# ────────────────────────────────────────────────────────────────────

def extract_table6_from_pdf(pdf_path: Path) -> pd.DataFrame | None:
    """
    Extract Table 6 project records from a single Flash Report PDF.
    
    Returns a DataFrame or None if extraction fails.
    """
    log.info(f"Opening: {pdf_path.name}")
    
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        log.error(f"Failed to open {pdf_path.name}: {e}")
        return None
    
    total_pages = len(pdf.pages)
    
    # Infer report month
    report_month = infer_month_from_path(pdf_path)
    if not report_month:
        for i in range(min(5, total_pages)):
            text = pdf.pages[i].extract_text() or ""
            report_month = infer_month_from_text(text)
            if report_month:
                break
    
    if not report_month:
        log.error(f"  Could not determine report month for {pdf_path.name}")
        pdf.close()
        return None
    
    log.info(f"  Report month: {report_month}")

    # Retain the old detection for the existing handlers, while routing verified
    # April--July 2026 reports through their dedicated parser.
    format_gen = detect_format_generation(pdf)
    report_format = detect_report_format(pdf_path, report_month)
    log.info(f"  Format: {report_format} (legacy detector: {format_gen}) | Pages: {total_pages}")
    
    is_table7 = report_format in {"EARLY_FORMAT", "QUARTERLY"}
    if is_table7:
        project_pages = [i for i, page in enumerate(pdf.pages) if _is_table7_page(page)]
        section_name = "Table 7: Project List: Ongoing Projects"
    else:
        project_pages = []
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "")[:300]
            if "All Ongoing Projects" in text:
                project_pages.append(i)
        section_name = "Table 6: All Ongoing Projects"

    if not project_pages:
        log.warning(f"  No project-list pages found in {pdf_path.name}")
        pdf.close()
        return None
    
    log.info(
        "  %s: pages %s–%s (%s pages)",
        "Table 7" if is_table7 else "Table 6",
        project_pages[0] + 1,
        project_pages[-1] + 1,
        len(project_pages),
    )
    
    # Extract records
    if report_format == "NEW_FORMAT_B":
        records = extract_new_format_b(
            pdf=pdf,
            table6_pages=project_pages,
            report_month=report_month,
            source_file=pdf_path.name,
        )
    elif report_format == "EARLY_FORMAT":
        records = extract_early_format(pdf, project_pages, report_month, pdf_path.name)
    elif report_format == "QUARTERLY":
        records = extract_quarterly_format(pdf, project_pages, report_month, pdf_path.name)
    else:
        records = extract_table6(
            pdf=pdf,
            table6_pages=project_pages,
            format_gen=format_gen,
            report_month=report_month,
            source_file=pdf_path.name,
        )
    
    pdf.close()
    
    if not records:
        log.warning(f"  No records extracted from {pdf_path.name}")
        return None
    
    df = records_to_dataframe(records, report_month, pdf_path.name, section_name)
    return df


def discover_pdfs(pdf_dir: Path | None = None) -> list[Path]:
    """Discover all PDF files in the raw PDF directory."""
    pdf_dir = pdf_dir or RAW_PDF_DIR
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    log.info(f"Discovered {len(pdfs)} PDF files in {pdf_dir}")
    return pdfs


# ────────────────────────────────────────────────────────────────────
# CLI — Pilot mode
# ────────────────────────────────────────────────────────────────────

def main():
    """Run pilot extraction on specified PDFs."""
    import argparse
    parser = argparse.ArgumentParser(description="SANKET PDF Extractor")
    parser.add_argument("--pdf", type=str, help="Specific PDF file to process")
    parser.add_argument("--pilot", action="store_true", help="Run pilot on Mar 2026 + Jul 2025")
    parser.add_argument("--all", action="store_true", help="Process all Flash Report PDFs")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR), help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            pdf_path = RAW_PDF_DIR / args.pdf
        df = extract_table6_from_pdf(pdf_path)
        if df is not None:
            out = output_dir / "pilot_project_monthly.csv"
            df.to_csv(out, index=False, encoding="utf-8")
            safe_print(f"Saved {len(df)} records to {out}")
            _print_extraction_summary(df)
    
    elif args.pilot:
        # Pilot: process one new-format and one old-format PDF
        pilot_pdfs = [
            RAW_PDF_DIR / "FlashReport_March_2026.pdf",
            RAW_PDF_DIR / "FlashReport_July_2025.pdf",
        ]
        all_dfs = []
        for pdf_path in pilot_pdfs:
            if pdf_path.exists():
                df = extract_table6_from_pdf(pdf_path)
                if df is not None:
                    all_dfs.append(df)
            else:
                log.error(f"Pilot PDF not found: {pdf_path}")
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            out = output_dir / "pilot_project_monthly.csv"
            combined.to_csv(out, index=False, encoding="utf-8")
            safe_print(f"\nSaved {len(combined)} total records to {out}")
            _print_extraction_summary(combined)
    
    elif args.all:
        log.error(
            "Full extraction is blocked: not every report format has a "
            "validated handler and golden-record coverage. Run a per-PDF "
            "pilot instead. See reports/format_pilot_report.txt."
        )


def _print_extraction_summary(df: pd.DataFrame) -> None:
    """Print a summary of the extraction results."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("EXTRACTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total records: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")
    
    if "report_month" in df.columns:
        for month in sorted(df["report_month"].unique()):
            month_df = df[df["report_month"] == month]
            lines.append(f"\n  {month}:")
            lines.append(f"    Records: {len(month_df)}")
            lines.append(f"    Project codes present: {month_df['project_code'].notna().sum()}/{len(month_df)}")
            lines.append(f"    States present: {month_df['state'].notna().sum()}/{len(month_df)}")
            lines.append(f"    Agencies present: {month_df['agency'].notna().sum()}/{len(month_df)}")
            lines.append(f"    Original cost present: {month_df['original_cost'].notna().sum()}/{len(month_df)}")
            lines.append(f"    Expenditure present: {month_df['expenditure'].notna().sum()}/{len(month_df)}")
            lines.append(f"    Progress present: {month_df['physical_progress'].notna().sum()}/{len(month_df)}")
            lines.append(f"    Status: {dict(month_df['extraction_status'].value_counts())}")
    
    # Field-level missingness
    lines.append(f"\nField missingness (across all records):")
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            pct = null_count / len(df) * 100
            lines.append(f"  {col}: {null_count} ({pct:.1f}%)")
    
    lines.append("=" * 60)
    safe_print("\n".join(lines))


if __name__ == "__main__":
    main()
