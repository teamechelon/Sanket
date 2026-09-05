import pdfplumber

with pdfplumber.open("data/raw/pdf/FR_May2025.pdf") as pdf:
    for pi in (42, 43, 44):
        page = pdf.pages[pi]
        words = page.extract_words()
        left = sorted(
            [w for w in words if float(w["x0"]) < 120],
            key=lambda w: (float(w["top"]), float(w["x0"])),
        )
        print(f"===== PAGE {pi + 1} =====")
        for w in left[:50]:
            print(f"{float(w['x0']):6.1f} {float(w['top']):6.1f} {w['text']}")
        print()
