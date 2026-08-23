@echo off
python run_validation.py
if errorlevel 1 exit /b %errorlevel%
python tools\model_execution_runner.py --models "XGBoost,Holt-Winters" --output-dir evidence\model_runs\local_run
if errorlevel 1 exit /b %errorlevel%
echo.
echo To run all five models after installing requirements:
echo python tools\model_execution_runner.py --models all --output-dir evidence\model_runs\full_run
