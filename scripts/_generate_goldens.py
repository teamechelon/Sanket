"""Generate golden-record candidates and pollution scans for format pilots."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.pdf_extractor import extract_table6_from_pdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdf"
OUT_DIR = ROOT / "tests" / "golden_records"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTION_RE = re.compile(
    r"^\d{4,7}\)|^\(-\)|^\([NO]?\d|^\([A-Z][a-z]+ [A-Z][a-z]+\)|^\d+\.\d+\)|Tamil Nadu\)|Telangana\)"
)


def sample_rows(df, serials):
    rows = []
    for sl in serials:
        hit = df[df.sl_no == sl]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "sl_no": int(r.sl_no),
                "project_code": None if (r.project_code != r.project_code or not r.project_code) else str(r.project_code),
                "legacy_ocms_code": None
                if (r.legacy_ocms_code != r.legacy_ocms_code or not r.legacy_ocms_code)
                else str(r.legacy_ocms_code),
                "project_name_contains": str(r.project_name)[:40],
                "agency": None if (r.agency != r.agency or not r.agency) else str(r.agency),
                "state": None if (r.state != r.state or not r.state) else str(r.state),
                "original_cost": None if r.original_cost != r.original_cost else float(r.original_cost),
                "revised_cost": None if r.revised_cost != r.revised_cost else float(r.revised_cost),
                "expenditure": None if r.expenditure != r.expenditure else float(r.expenditure),
                "physical_progress": None
                if r.physical_progress != r.physical_progress
                else float(r.physical_progress),
            }
        )
    return rows


def pollution_count(df) -> int:
    count = 0
    for name in df.project_name.fillna(""):
        if POLLUTION_RE.search(str(name).strip()):
            count += 1
        elif str(name).strip().startswith("(") and ")" in str(name)[:30]:
            count += 1
    return count


def coverage(df):
    primary = df.project_code.notna() & (df.project_code.astype(str).str.len() > 0)
    alt = df.legacy_ocms_code.notna() & (df.legacy_ocms_code.astype(str).str.len() > 0)
    return int(primary.sum()), int((~primary).sum()), int(alt.sum()), len(df)


CASES = [
    ("FRApril2025.pdf", "early_format_april_2025.json", [1, 2, 11, 12, 284, 355, 500, 1000, 1660, 1670]),
    ("FR_May2025.pdf", "early_format_may_2025.json", [1, 2, 11, 50, 200, 500, 800, 1200, 1600, 1637]),
    ("FR_JUNE_2025.pdf", "early_format_june_2025.json", [1, 2, 11, 50, 200, 500, 800, 1200, 1500, 1595]),
    ("QPISR_QR_1st_2025-26.pdf", "quarterly_q1_2025_26.json", [1, 2, 11, 50, 200, 500, 800, 1200, 1700, 1734]),
    ("FlashReport_April2026.pdf", None, [1, 2, 3, 1487, 1488, 1490]),
    ("FlashReport_May2026.pdf", "new_format_b_may_2026.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_June_2026.pdf", "new_format_b_june_2026.json", [1, 2, 3, 4, 5, 920, 921, 922, 923, 924]),
    ("FlashReport_July_2026-1.pdf", "new_format_b_july_2026.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_July_2025.pdf", "old_format_a_july_2025.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_August_2025.pdf", "old_format_b_august_2025.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_September_2025.pdf", "transitional_september_2025.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_January_2026.pdf", "transitional_january_2026.json", [1, 2, 3, 4, 5, 930, 931, 1275, 1276, 1277]),
    ("FlashReport_February_2026.pdf", "new_format_a_february_2026.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("FlashReport_March_2026.pdf", "new_format_a_march_2026.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
]

for pdf_name, golden_name, serials in CASES:
    path = PDF_DIR / pdf_name
    print(f"\n===== {pdf_name} =====")
    df = extract_table6_from_pdf(path)
    if df is None:
        print("FAILED EXTRACT")
        continue
    p, miss, alt, total = coverage(df)
    poll = pollution_count(df)
    print(f"records={total} primary={p} missing={miss} alt={alt} pollution={poll}")
    print(f"serials {df.sl_no.min()}..{df.sl_no.max()} unique={df.sl_no.nunique()}")
    rows = sample_rows(df, serials)
    for row in rows[:3]:
        print(row["sl_no"], row["state"], row["project_code"], row["project_name_contains"][:50])
    if golden_name:
        (OUT_DIR / golden_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("wrote", golden_name, "n=", len(rows))
