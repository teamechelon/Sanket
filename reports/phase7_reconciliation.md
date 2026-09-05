# Phase 7 identifier reconciliation

The promoted dataset is `data/processed/project_monthly.csv`. It is byte-for-byte
identical to the isolated controlled extraction at
`data/processed/phase7_final_validation/project_monthly.csv`.

`data/processed/phase7_pre_repair_baseline.csv` is the retained pre-repair
comparison snapshot. It was reconstructed only from the preserved 639-row
source-addressed ledger in `reports/phase7_smoketest/`: each address retains its
original source fields and identifier classification, while the 197 subsequently
recovered primary codes remain blank in the comparison copy. It is not used for
production analytics.

Regenerate the Phase 7 review artifacts with:

```powershell
python -m src.identifier_review `
  --baseline data/processed/phase7_pre_repair_baseline.csv `
  --candidate data/processed/project_monthly.csv `
  --output-dir reports
```

Expected reconciliation:

- 23,503 project-month records before and after.
- 18,009 to 18,206 primary identifiers (197 source-printed codes recovered).
- 639 baseline no-identifier rows: 197 `IDENTIFIER_FOUND`, 242 remaining
  `PARSER_FAILURE`, and 200 `UNRESOLVED`.
- Zero duplicate primary-code/month and source-tracking-key/month pairs.

No project-name match, synthetic identifier, or promotion of an alternate OCMS
identifier to a primary project code is used in this reconciliation.
