"""Find Sl No column x positions on Table 7 pages."""
from __future__ import annotations

import pdfplumber

with pdfplumber.open("data/raw/pdf/FRApril2025.pdf") as pdf:
    page = pdf.pages[42]
    for w in page.extract_words():
        if w["text"] in {"1", "2", "3", "4", "5", "Sl", "No"} or w["text"].startswith("ANDAMAN"):
            print(f"x0={float(w['x0']):6.1f} x1={float(w['x1']):6.1f} y={float(w['top']):6.1f} {w['text']}")
