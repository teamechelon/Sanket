"""
SANKET — Data Cleaner
======================
Deterministic, reproducible cleaning of extracted PAIMANA data.

Rules:
- NEVER modifies raw files.
- Reads from extraction output, writes to data/processed/.
- Distinguishes MISSING / ZERO / NOT_APPLICABLE / EXTRACTION_FAILURE.
- Logs all transformations.

Status: STUB — to be implemented in Phase 9.
"""

from __future__ import annotations

from src.utils import get_logger

log = get_logger("data_cleaner")
