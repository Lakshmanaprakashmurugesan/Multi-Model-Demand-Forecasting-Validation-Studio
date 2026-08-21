# Validation Protocol

## Objective
Demonstrate what the current application actually does under reproducible conditions without treating a synthetic demo as evidence of employer deployment or business impact.

## Minimum validation sequence

1. Verify the exact release and `app.py` SHA-256.
2. Install dependencies in a fresh environment.
3. Confirm `python -m py_compile app.py` succeeds.
4. Start Streamlit and confirm the application loads.
5. Load the committed synthetic telecommunications sample.
6. Select `date` as timestamp and `demand` as target.
7. Optionally select approved numeric drivers such as `promotion`, `price_index`, `availability_index`, and `deployment_activity`.
8. Record data frequency, holdout percentage, forecast horizon, seasonal period, and model parameters.
9. Run each available model separately and then together.
10. Preserve the model leaderboard and validation predictions.
11. Confirm XGBoost and LSTM multi-step forecasts are generated recursively from available history rather than future target actuals.
12. Export CSV/Excel results and preserve them unchanged.
13. Record failures as failures. Do not omit a model solely because it performed poorly.

## Evidence quality rules

- The result is valid only for the dataset and configuration used.
- Synthetic-data results demonstrate software behavior, not operational business impact.
- A single holdout result is not equivalent to rolling-origin validation.
- Approximate residual-based forecast intervals are not represented as calibrated probability intervals.
- An independent reviewer should state exactly what was personally reproduced and what was merely reviewed from supplied records.
