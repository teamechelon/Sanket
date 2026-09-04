"""
SANKET — Ingestion Orchestrator
================================
Single entry point for the complete data pipeline:

    python -m src.ingest

Discovers raw PDFs/CSVs → Extracts → Cleans → Validates → Reports.
Idempotent: safe to run repeatedly without creating duplicates.

Status: PARTIAL — CSV ingestion implemented; PDF ingestion pending.
"""

from __future__ import annotations

from src.utils import get_logger, safe_print
from src.csv_processor import process_all_csvs, generate_csv_quality_report

log = get_logger("ingest")


def main():
    """Run the ingestion pipeline."""
    log.info("SANKET Ingestion Pipeline — Starting")

    # Phase 2: CSV Processing
    log.info("Phase 2: Processing aggregate CSVs")
    csv_results = process_all_csvs()
    report = generate_csv_quality_report(csv_results)
    safe_print(report)

    # Phase 4–8: PDF Processing (not yet implemented)
    log.info("Phase 4–8: PDF extraction not yet implemented")

    log.info("SANKET Ingestion Pipeline — Done")


if __name__ == "__main__":
    main()
