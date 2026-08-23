$ErrorActionPreference = "Stop"
Write-Host "Running repository integrity/static validation..."
python .\run_validation.py
Write-Host ""
Write-Host "Running currently available forecasting engines from unchanged app.py..."
python .\tools\model_execution_runner.py --models "XGBoost,Holt-Winters" --output-dir evidence/model_runs/local_run
Write-Host ""
Write-Host "For all five engines after installing requirements:"
Write-Host 'python .\tools\model_execution_runner.py --models all --output-dir evidence/model_runs/full_run'
