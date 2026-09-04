"""
SANKET — CSV Processor
=======================
Process the 40 aggregate PAIMANA CSV files into structured,
normalized DataFrames suitable for validation and analytics.

These CSVs are aggregate-level data (NOT project-level).
They serve as validation references for PDF extraction.

Usage:
    python -m src.csv_processor
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from src.utils import (
    RAW_CSV_DIR,
    PROCESSED_DIR,
    get_logger,
    infer_month_from_path,
    parse_indian_currency,
    parse_numeric,
    parse_parenthetical_costs,
    read_paimana_csv,
    safe_print,
)

log = get_logger("csv_processor")


# ────────────────────────────────────────────────────────────────────
# Discover CSVs
# ────────────────────────────────────────────────────────────────────

def discover_csvs(csv_dir: Path | None = None) -> list[dict]:
    """
    Walk csv_dir and return metadata for each CSV file found.
    """
    csv_dir = csv_dir or RAW_CSV_DIR
    results = []
    for root, _dirs, files in os.walk(csv_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = Path(root) / fname
            month = infer_month_from_path(fpath)
            dataset_type = fpath.stem  # e.g. "Cost-Wise-Report"
            results.append({
                "path": fpath,
                "inferred_month": month,
                "dataset_type": dataset_type,
            })
    return results


# ────────────────────────────────────────────────────────────────────
# Processors for each CSV type
# ────────────────────────────────────────────────────────────────────

def process_cost_wise(path: Path, month: str) -> pd.DataFrame:
    """
    Process a Cost-Wise-Report.csv.
    Structure: 1 data row, 4 columns:
        col_0: empty
        col_1: original cost (₹ currency)
        col_2: revised cost (₹ currency)
        col_3: expenditure (₹ currency)
    """
    _title, rows = read_paimana_csv(path)
    records = []
    for row in rows:
        if len(row) < 4:
            log.warning(f"Cost-Wise row with <4 columns in {path}: {row}")
            continue
        records.append({
            "report_month": month,
            "original_cost_crores": parse_indian_currency(row[1]),
            "revised_cost_crores": parse_indian_currency(row[2]),
            "expenditure_crores": parse_indian_currency(row[3]),
            "source_file": str(path.name),
            "source_type": "aggregate_csv",
        })
    return pd.DataFrame(records)


def process_physical_progress(path: Path, month: str) -> pd.DataFrame:
    """
    Process a Physical-Progress-Report.csv.
    Structure: ~11 data rows, 5 columns:
        col_0: serial number
        col_1: progress range (e.g. "0-10", "100")
        col_2: project count
        col_3: original cost (revised cost) — parenthetical
        col_4: expenditure
    """
    _title, rows = read_paimana_csv(path)
    records = []
    for row in rows:
        if len(row) < 5:
            log.warning(f"Physical-Progress row with <5 columns in {path}: {row}")
            continue
        orig_cost, rev_cost = parse_parenthetical_costs(row[3])
        records.append({
            "report_month": month,
            "progress_range": row[1].strip(),
            "project_count": parse_numeric(row[2]),
            "original_cost_crores": orig_cost,
            "revised_cost_crores": rev_cost,
            "expenditure_crores": parse_numeric(row[4]),
            "source_file": str(path.name),
            "source_type": "aggregate_csv",
        })
    return pd.DataFrame(records)


def process_sector_wise(path: Path, month: str) -> pd.DataFrame:
    """
    Process a Sector-Wise-Report.csv.
    Structure: ~20 data rows, 5 columns:
        col_0: serial number
        col_1: sector name
        col_2: project count
        col_3: original cost (revised cost) — parenthetical
        col_4: expenditure
    """
    _title, rows = read_paimana_csv(path)
    records = []
    for row in rows:
        if len(row) < 5:
            log.warning(f"Sector-Wise row with <5 columns in {path}: {row}")
            continue
        orig_cost, rev_cost = parse_parenthetical_costs(row[3])
        records.append({
            "report_month": month,
            "sector": row[1].strip(),
            "project_count": parse_numeric(row[2]),
            "original_cost_crores": orig_cost,
            "revised_cost_crores": rev_cost,
            "expenditure_crores": parse_numeric(row[4]),
            "source_file": str(path.name),
            "source_type": "aggregate_csv",
        })
    return pd.DataFrame(records)


def process_state_wise(path: Path, month: str) -> pd.DataFrame:
    """
    Process a State-Wise-Report.csv.
    Structure: ~34 data rows, 5 columns:
        col_0: serial number
        col_1: state name
        col_2: project count
        col_3: original cost (revised cost) — parenthetical
        col_4: expenditure
    """
    _title, rows = read_paimana_csv(path)
    records = []
    for row in rows:
        if len(row) < 5:
            log.warning(f"State-Wise row with <5 columns in {path}: {row}")
            continue
        orig_cost, rev_cost = parse_parenthetical_costs(row[3])
        records.append({
            "report_month": month,
            "state": row[1].strip(),
            "project_count": parse_numeric(row[2]),
            "original_cost_crores": orig_cost,
            "revised_cost_crores": rev_cost,
            "expenditure_crores": parse_numeric(row[4]),
            "source_file": str(path.name),
            "source_type": "aggregate_csv",
        })
    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────

_PROCESSORS = {
    "Cost-Wise-Report": ("cost_overview_monthly", process_cost_wise),
    "Physical-Progress-Report": ("physical_progress_monthly", process_physical_progress),
    "Sector-Wise-Report": ("sector_monthly", process_sector_wise),
    "State-Wise-Report": ("state_monthly", process_state_wise),
}


def process_all_csvs(csv_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Discover and process all aggregate CSVs into structured DataFrames.

    Returns a dict mapping output_name → DataFrame.
    Also saves each DataFrame to output_dir as a CSV.
    """
    csv_dir = csv_dir or RAW_CSV_DIR
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_metas = discover_csvs(csv_dir)
    log.info(f"Discovered {len(csv_metas)} CSV files in {csv_dir}")

    # Accumulate DataFrames by output type
    accumulators: dict[str, list[pd.DataFrame]] = {}
    skipped = 0

    for meta in csv_metas:
        ds_type = meta["dataset_type"]
        month = meta["inferred_month"]

        if ds_type not in _PROCESSORS:
            log.warning(f"Unknown dataset type: {ds_type} — skipping {meta['path']}")
            skipped += 1
            continue

        if month is None:
            log.warning(f"Could not infer month for {meta['path']} — skipping")
            skipped += 1
            continue

        output_name, processor_fn = _PROCESSORS[ds_type]
        df = processor_fn(meta["path"], month)

        if output_name not in accumulators:
            accumulators[output_name] = []
        accumulators[output_name].append(df)

    # Concatenate and save
    results: dict[str, pd.DataFrame] = {}
    for output_name, dfs in accumulators.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("report_month").reset_index(drop=True)
        out_path = output_dir / f"{output_name}.csv"
        combined.to_csv(out_path, index=False, encoding="utf-8")
        results[output_name] = combined
        log.info(f"  {output_name}.csv — {len(combined)} rows, {len(combined.columns)} columns")

    if skipped:
        log.warning(f"Skipped {skipped} files")

    return results


# ────────────────────────────────────────────────────────────────────
# Quality Report
# ────────────────────────────────────────────────────────────────────

def generate_csv_quality_report(results: dict[str, pd.DataFrame]) -> str:
    """Generate a readable quality report for the CSV processing results."""
    lines = []
    lines.append("=" * 70)
    lines.append("SANKET — CSV PROCESSING QUALITY REPORT")
    lines.append("=" * 70)

    for name, df in sorted(results.items()):
        lines.append(f"\n{'─' * 50}")
        lines.append(f"Dataset: {name}")
        lines.append(f"{'─' * 50}")
        lines.append(f"  Rows: {len(df)}")
        lines.append(f"  Columns: {list(df.columns)}")
        lines.append(f"  Months: {sorted(df['report_month'].unique())}")

        # Missing value analysis
        lines.append(f"\n  Missing values:")
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                pct = null_count / len(df) * 100
                lines.append(f"    {col}: {null_count} ({pct:.1f}%)")

        # Numeric column summaries
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) > 0:
            lines.append(f"\n  Numeric summaries:")
            for col in numeric_cols:
                valid = df[col].dropna()
                if len(valid) > 0:
                    lines.append(
                        f"    {col}: min={valid.min():.2f}, max={valid.max():.2f}, "
                        f"mean={valid.mean():.2f}, nulls={df[col].isna().sum()}"
                    )

        # Duplicate check
        if "report_month" in df.columns:
            key_cols = [c for c in df.columns if c not in ("source_file", "source_type")]
            dupes = df.duplicated(subset=key_cols).sum()
            lines.append(f"\n  Duplicate rows: {dupes}")

        # Sample rows
        lines.append(f"\n  Sample (first 3 rows):")
        for _, row in df.head(3).iterrows():
            lines.append(f"    {dict(row)}")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF CSV QUALITY REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    """Run the CSV processor and print quality report."""
    log.info("Starting CSV processing pipeline")

    results = process_all_csvs()

    report = generate_csv_quality_report(results)
    safe_print(report)

    # Save report
    report_path = PROCESSED_DIR.parent.parent / "reports" / "csv_processing_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
