# Repository Evidence Index

## A. Objective source and build artifacts included in this release

| Artifact | Purpose |
|---|---|
| `SOURCE_INTEGRITY.sha256` | Fixed checksum of unchanged `app.py` |
| `evidence/source_integrity.json` | Structured source provenance and verification status |
| `evidence/syntax_validation.txt` | Recorded syntax-compilation check |
| `evidence/source_function_inventory.csv` | AST-derived inventory of classes/functions in `app.py` |
| `evidence/dependency_inventory.csv` | Dependency-to-source mapping |
| `evidence/repository_inventory.csv` | File inventory and SHA-256 values |
| `sample_data/synthetic_telecom_equipment_demand.csv` | Reproducible, non-confidential telecom-oriented input |

## B. Protocols included for reproducible future evidence

- `docs/REPRODUCIBILITY.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/INDEPENDENT_REVIEW_PROTOCOL.md`
- `evidence/run_manifest_template.json`
- `evidence/validation_summary_template.md`

## C. Evidence that must come from outside the repository

The repository does not create or substitute for:

- independent first-hand corroboration of historical employer metrics;
- proof of the author's individual role in employer projects;
- third-party adoption or pilot interest;
- independent expert validation;
- government interest or endorsement;
- patents/licenses;
- media coverage or citations.

Those items should be obtained from authentic sources rather than generated as repository files.
