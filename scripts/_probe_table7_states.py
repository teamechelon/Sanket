"""Probe where states appear vs serials in Table 7."""
from __future__ import annotations

import re
import pdfplumber

STATE_HINTS = (
    "ANDAMAN", "ANDHRA", "ARUNACHAL", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL", "JHARKHAND", "KARNATAKA",
    "KERALA", "MADHYA", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM",
    "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL",
    "TELANGANA", "TRIPURA", "UTTAR", "WEST", "DELHI", "JAMMU", "LADAKH",
    "PUDUCHERRY", "CHANDIGARH", "LAKSHADWEEP", "DADRA", "NICOBAR",
)

with pdfplumber.open("data/raw/pdf/FRApril2025.pdf") as pdf:
    found = []
    for pi in range(42, 268):
        page = pdf.pages[pi]
        words = page.extract_words()
        left = [w for w in words if float(w["x0"]) < 70]
        # Collect left-column tokens that look like state fragments
        tokens = [w["text"] for w in left if w["text"] not in {"State", "Sector"}]
        joined = " ".join(tokens)
        if any(h in joined.upper() for h in STATE_HINTS) or tokens:
            # also get sl nos from this page via default table
            table = page.extract_table()
            sls = []
            if table:
                for row in table[1:]:
                    if row and row[2] and re.fullmatch(r"\d+", str(row[2]).strip()):
                        sls.append(int(row[2]))
            if tokens:
                found.append((pi + 1, " ".join(tokens)[:80], sls[:3], sls[-1:] if sls else []))

    print(f"pages with left-col tokens: {len(found)}")
    for item in found[:60]:
        print(item)
    print("...")
    for item in found[-20:]:
        print(item)
