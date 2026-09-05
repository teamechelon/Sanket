"""Probe GOA/GUJARAT dual-state page and Andhra page y-alignment."""
from __future__ import annotations

import pdfplumber

with pdfplumber.open("data/raw/pdf/FRApril2025.pdf") as pdf:
    for pi in (43, 90, 180):
        page = pdf.pages[pi]
        words = page.extract_words()
        print(f"===== PAGE {pi + 1} =====")
        for w in words:
            x0 = float(w["x0"])
            text = w["text"]
            if x0 < 70 or text.isdigit() and 1 <= int(text) <= 2000 and x0 < 130:
                if x0 < 70 or (text.isdigit() and float(w["x0"]) < 130):
                    print(f"x={x0:5.1f} y={float(w['top']):6.1f} {text}")
        print()
