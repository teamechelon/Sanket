"""Focused parser regressions backed by source-document structure."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pdfplumber

from src.pdf_extractor import (
    _normalise_flexible_month,
    _parse_table7_project_cell,
    _table7_original_and_revised_costs,
    _table7_original_and_revised_dates,
    detect_report_format,
    extract_table7,
    extract_table6_from_pdf,
)


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdf"


class FormatRoutingRegressionTest(unittest.TestCase):
    def test_previously_supported_formats_keep_their_routes(self) -> None:
        cases = {
            "FlashReport_July_2025.pdf": "OLD_FORMAT_A",
            "FlashReport_February_2026.pdf": "NEW_FORMAT_A",
            "FlashReport_March_2026.pdf": "NEW_FORMAT_A",
            "FlashReport_April2026.pdf": "NEW_FORMAT_B",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(detect_report_format(PDF_DIR / filename, None), expected)


class Table7ParsingRegressionTest(unittest.TestCase):
    def test_flexible_month_formats(self) -> None:
        self.assertEqual(_normalise_flexible_month("10-2013"), "10/2013")
        self.assertEqual(_normalise_flexible_month("(Jun-2023)"), "06/2023")
        self.assertEqual(_normalise_flexible_month("{6/2023}"), "06/2023")

    def test_anticipated_values_are_not_mislabeled_as_revisions(self) -> None:
        self.assertEqual(
            _table7_original_and_revised_dates("3/2025\n(N.A.)\n{12/2025}"),
            ("03/2025", ""),
        )
        self.assertEqual(
            _table7_original_and_revised_costs("235.72\n(N.A.)\n{235.72}"),
            (235.72, None),
        )

    def test_name_agency_and_identifier_are_kept_separate(self) -> None:
        name, agency, identifier = _parse_table7_project_cell(
            "LONG PROJECT NAME (PHASE II)\n(NHAI)\n(N24002208)"
        )
        self.assertEqual(name, "LONG PROJECT NAME (PHASE II)")
        self.assertEqual(agency, "NHAI")
        self.assertEqual(identifier, "N24002208")

    def test_wrapped_state_continuation_updates_the_source_record(self) -> None:
        header = ["State", "Sector", "Sl No", "Project", "Approval", "Dates", "Costs", "Exp", "Progress"]
        project = [
            "ANDAMAN AND", "CIVIL AVIATION", "1",
            "PORT PROJECT\n(AAI)\n(N04000073)", "10-2013",
            "9/2018\n(6/2023)\n{6/2023}", "417.23\n(707.73)\n{707.73}",
            "698.80", "100.00",
        ]
        continuation = ["NICOBAR ISLANDS", None, None, None, None, None, None, None, None]

        class Page:
            def extract_table(self):
                return [header, project, continuation]

            def extract_words(self):
                return []

        class PDF:
            pages = [Page()]

        records = extract_table7(PDF(), [0], "2025-04", "fixture.pdf")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, "ANDAMAN AND NICOBAR ISLANDS")
        self.assertEqual(records[0].source_page, 1)


class April2026SourceCountTest(unittest.TestCase):
    def test_printed_group_totals_reconcile_to_1981(self) -> None:
        path = PDF_DIR / "FlashReport_April2026.pdf"
        totals: list[int] = []
        first_serial = last_serial = None
        with pdfplumber.open(path) as pdf:
            # Table 6 is printed on PDF pages 54--162 (one-based).
            for page in pdf.pages[53:162]:
                for line in (page.extract_text() or "").splitlines():
                    total = re.match(r"^Total \((\d+)\)", line.strip())
                    if total:
                        totals.append(int(total.group(1)))
                    serial = re.match(r"^(\d+)\s+", line.strip())
                    if serial and int(serial.group(1)) <= 1981:
                        first_serial = int(serial.group(1)) if first_serial is None else first_serial
                        last_serial = max(last_serial or 0, int(serial.group(1)))
        self.assertEqual(len(totals), 31)
        self.assertEqual(sum(totals), 1981)
        self.assertEqual(first_serial, 1)
        self.assertEqual(last_serial, 1981)


class OldFormatWrappedIdentifierRegressionTest(unittest.TestCase):
    def test_july_wrapped_state_keeps_the_printed_primary_code(self) -> None:
        """A source code between wrapped state fragments belongs to its serial row."""
        path = PDF_DIR / "FlashReport_July_2025.pdf"
        records = extract_table6_from_pdf(path).set_index("sl_no")
        self.assertEqual(records.loc[10, "project_code"], "611047")
        self.assertEqual(records.loc[16, "project_code"], "611440")
        self.assertEqual(records.loc[17, "project_code"], "611570")


if __name__ == "__main__":
    unittest.main()
