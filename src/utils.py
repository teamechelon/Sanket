"""
SANKET — Shared Utilities
==========================
Common constants, parsing helpers, and I/O utilities
used across the entire ingestion pipeline.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any

# ────────────────────────────────────────────────────────────────────
# Project paths (all relative to the project root)
# ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV_DIR = RAW_DIR / "csv"
RAW_PDF_DIR = RAW_DIR / "pdf"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure output directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(levelname)s — %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ────────────────────────────────────────────────────────────────────
# Month inference
# ────────────────────────────────────────────────────────────────────

_MONTH_DIR_RE = re.compile(r"(20\d{2})[-_/](0[1-9]|1[0-2])")

_MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_MONTH_NAME_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[\s_]*(\d{4})",
    re.IGNORECASE,
)


def infer_month_from_path(path: Path) -> str | None:
    """
    Attempt to extract 'YYYY-MM' from a file path.
    Tries directory-name patterns first, then filename patterns.
    """
    full = str(path)
    # Try YYYY-MM in directory
    m = _MONTH_DIR_RE.search(full)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Try month name in filename
    m = _MONTH_NAME_RE.search(full)
    if m:
        month_num = _MONTH_NAMES[m.group(1).lower()]
        return f"{m.group(2)}-{month_num}"
    return None


def infer_month_from_text(text: str) -> str | None:
    """Extract 'YYYY-MM' from free text (e.g. 'MARCH 2026')."""
    m = _MONTH_NAME_RE.search(text)
    if m:
        month_num = _MONTH_NAMES[m.group(1).lower()]
        return f"{m.group(2)}-{month_num}"
    return None


# ────────────────────────────────────────────────────────────────────
# Number parsing
# ────────────────────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"[₹Rs.\s,]+")


def parse_indian_currency(value: str) -> float | None:
    """
    Parse Indian currency strings like '₹ 23,75,332' → 2375332.0.
    Returns None for unparseable values.
    """
    if not value or not value.strip():
        return None
    cleaned = _CURRENCY_RE.sub("", value.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_numeric(value: str) -> float | None:
    """
    Parse a numeric value, stripping commas and whitespace.
    Returns None for blank/missing/unparseable values.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if v in ("", "-", "NA", "N/A", "na", "n/a", "--"):
        return None
    # Remove commas and spaces
    v = v.replace(",", "").replace(" ", "")
    try:
        return float(v)
    except ValueError:
        return None


def parse_parenthetical_costs(value: str) -> tuple[float | None, float | None]:
    """
    Parse 'original (revised)' format like '381959.22 (382705.77)'.
    Returns (original_cost, revised_cost).
    """
    if not value or not isinstance(value, str):
        return None, None
    v = value.strip()
    m = re.match(r"^([\d,.]+)\s*\(([\d,.]+)\)$", v)
    if m:
        orig = parse_numeric(m.group(1))
        rev = parse_numeric(m.group(2))
        return orig, rev
    # Maybe just a plain number
    n = parse_numeric(v)
    return n, None


# ────────────────────────────────────────────────────────────────────
# CSV reading (PAIMANA aggregate CSVs)
# ────────────────────────────────────────────────────────────────────

def read_paimana_csv(path: Path) -> tuple[str | None, list[list[str]]]:
    """
    Read a PAIMANA-style CSV with:
        line 1: title (e.g. "Cost Wise Details")
        line 2: blank
        lines 3+: data (no header row)

    Returns (title, rows) where rows is a list of lists of strings.
    Never modifies the source file.
    """
    content: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            content = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if content is None:
        raise RuntimeError(f"Could not decode {path}")

    lines = content.splitlines()
    title: str | None = None
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = list(csv.reader(io.StringIO(stripped)))
        if parsed and len(parsed[0]) == 1:
            # Single-field line → title
            title = parsed[0][0]
            data_start = i + 1
            break
        else:
            data_start = i
            break

    rows: list[list[str]] = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parsed = list(csv.reader(io.StringIO(line)))
        if parsed:
            rows.append(parsed[0])

    return title, rows


# ────────────────────────────────────────────────────────────────────
# Safe printing (Windows console encoding)
# ────────────────────────────────────────────────────────────────────

import sys

def safe_print(text: str) -> None:
    """Print handling Windows cp1252 console encoding issues."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
