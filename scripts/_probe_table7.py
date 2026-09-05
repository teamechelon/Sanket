"""Temporary probe for Table 7 state-column extraction."""
from __future__ import annotations

import pdfplumber

with pdfplumber.open("data/raw/pdf/FRApril2025.pdf") as pdf:
    for pi in (42, 55, 80, 100):
        page = pdf.pages[pi]
        words = page.extract_words()
        left = [w for w in words if float(w["x0"]) < 95]
        print(f"===== PAGE {pi + 1} left-column words =====")
        for w in left[:80]:
            print(f"{w['top']:6.1f} {w['x0']:5.1f} {w['text']}")
        print()

        # Try table with looser settings
        table = page.extract_table(
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "intersection_tolerance": 5,
            }
        )
        if table:
            print(f"text-strategy rows={len(table)}")
            for i, row in enumerate(table[:8]):
                cells = [(c or "")[:35].replace("\n", "|") for c in (row or [])[:4]]
                print(i, cells)
            states = [
                (row[2], (row[0] or "")[:40].replace("\n", "|"))
                for row in table[1:]
                if row and row[0] and str(row[0]).strip()
            ]
            print("state cells:", states[:20])
        print()
