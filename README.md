# SANKET — AI for Infrastructure Project Monitoring

**SIH 2026 · Problem Statement 26103**

Predictive early-warning system for PAIMANA-monitored infrastructure projects.

## Architecture

```
PAIMANA Data → Extraction → Cleaning → Validation → Features → ML → Dashboard
```

## Data Sources

| Source | Type | Records |
|--------|------|---------|
| Flash Report PDFs (17 files) | Project-level + aggregates | ~1000–1900 projects/month |
| Aggregate CSVs (40 files) | State/sector/cost/progress aggregates | Validation references |

## Pipeline Modules

| Module | Status | Purpose |
|--------|--------|---------|
| `src/utils.py` | ✅ Done | Shared utilities, parsers, constants |
| `src/data_audit.py` | ✅ Done | CSV audit and profiling |
| `src/csv_processor.py` | ✅ Done | Aggregate CSV → structured data |
| `src/pdf_extractor.py` | 🔲 Stub | Flash Report PDF extraction |
| `src/data_cleaner.py` | 🔲 Stub | Deterministic data cleaning |
| `src/validation.py` | 🔲 Stub | Cross-validation engine |
| `src/ingest.py` | 🔲 Partial | Pipeline orchestrator |

## Usage

```bash
# Process aggregate CSVs
python -m src.csv_processor

# Run full pipeline (CSV processing + PDF extraction when implemented)
python -m src.ingest

# Audit raw CSV files
python src/data_audit.py --raw-dir data/raw/csv
```

## Requirements

```bash
pip install -r requirements.txt
```
