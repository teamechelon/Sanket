"""Golden-record regression coverage for the verified April 2026 layout."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.pdf_extractor import extract_table6_from_pdf


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "pdf" / "FlashReport_April2026.pdf"
GOLDEN_PATH = ROOT / "tests" / "golden_records" / "new_format_b_april_2026.json"


class NewFormatBAprilGoldenRecordsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = extract_table6_from_pdf(PDF_PATH).set_index("sl_no")
        cls.expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    def test_source_records_match_golden_values(self) -> None:
        for expected in self.expected:
            with self.subTest(sl_no=expected["sl_no"]):
                actual = self.records.loc[expected["sl_no"]]
                self.assertEqual(actual["project_code"], expected["project_code"])
                if expected["legacy_ocms_code"] is None:
                    self.assertTrue(pd.isna(actual["legacy_ocms_code"]))
                else:
                    self.assertEqual(actual["legacy_ocms_code"], expected["legacy_ocms_code"])
                self.assertIn(expected["project_name_contains"], actual["project_name"])
                self.assertEqual(actual["agency"], expected["agency"])
                self.assertEqual(actual["state"], expected["state"])
                for field in ("original_cost", "revised_cost", "expenditure", "physical_progress"):
                    self.assertAlmostEqual(actual[field], expected[field], places=2)

    def test_serial_numbers_are_complete(self) -> None:
        self.assertEqual(len(self.records), 1981)
        self.assertEqual(self.records.index.tolist(), list(range(1, 1982)))


if __name__ == "__main__":
    unittest.main()
