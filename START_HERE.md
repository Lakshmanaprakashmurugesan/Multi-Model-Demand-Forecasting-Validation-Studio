# START HERE — Final Repository Package

This is the complete public technical repository package. It is not the separate private attorney/RFE support pack.

## Source-code preservation

`app.py` has not been modified. Its SHA-256 is:

`28dbcd30bd48df4f659ea4b44c87cb1de263989182d4316df7262a0b7c23a87f`

Verify it with:

```powershell
python run_validation.py
```

## Run the application

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Generate execution evidence without editing app.py

```powershell
python tools\model_execution_runner.py --models all --output-dir evidence\model_runs\full_run
```

Or run the Windows helper:

```powershell
.\run_validation.ps1
```

## Evidence already included

- source-integrity validation
- syntax compilation validation
- repository-structure validation
- deterministic synthetic telecom sample-data profile
- automated repository tests
- a builder-verified runtime execution of the exact `app.py` forecasting definitions for XGBoost and Holt-Winters
- actual retained validation predictions, future forecasts, model leaderboard, Excel export, execution status, and run manifest for that verified run

The builder-verified run is evidence of reproducible execution on synthetic/demo data only. It is not evidence of historical employer results, production deployment, adoption, or national-scale outcomes.
