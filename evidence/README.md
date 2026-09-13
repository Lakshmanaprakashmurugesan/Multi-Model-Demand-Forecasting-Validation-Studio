# Evidence Directory

This directory contains repository-verification artifacts, measured forecasting execution evidence, and templates used to support reproducibility.

## 1. Repository Verification Artifacts

The following files document source integrity, repository structure, dependencies, syntax validation, and sample-data characteristics:

- `source_integrity.json`
- `syntax_validation.txt`
- `source_function_inventory.csv`
- `dependency_inventory.csv`
- `repository_inventory.csv`
- `sample_data_profile.json`

Additional execution and repository-integrity validation records are available under:

- `execution/`

These artifacts support verification of the repository structure and source used for the documented evidence package.

## 2. Measured Five-Model Execution Evidence

The directory:

`model_runs/full_run/`

contains preserved results from an actual execution of the forecasting workflow across five forecasting approaches:

- Prophet Baseline
- Prophet + U.S. Holidays
- XGBoost
- LSTM
- Holt-Winters

Key evidence files include:

- `model_runs/full_run/run_manifest.json`
- `model_runs/full_run/model_leaderboard.csv`
- `model_runs/full_run/VALIDATION_REPORT.md`
- `model_runs/full_run/execution_status.csv`
- `model_runs/full_run/forecasting_intelligence_results.xlsx`
- `model_runs/full_run/combined_future_forecast.csv`

The directory also contains model-specific validation predictions and future-forecast CSV files for the successfully executed models.

These files are measured execution evidence from the identified run and are different from the templates listed below.

## 3. Templates Only — Not Measured Results

The following files are templates intended to support future documented runs:

- `run_manifest_template.json`
- `validation_summary_template.md`

A template becomes execution evidence only after it is completed from an identified reproducible run and linked to the source version and configuration used for that run.