# Methodology Positioning

## What is original to this repository versus what is established technology

Prophet, XGBoost, LSTM, Holt-Winters, MAE, RMSE, MAPE, and WMAPE are established methods. This repository does **not** claim invention of those algorithms or metrics.

The engineering contribution demonstrated by the current repository is the integration of multiple model families into one common, time-aware validation workflow with shared preprocessing, common holdout evaluation, model comparison, diagnostics, optional drivers, holiday-effect analysis, full-history refitting, and exportable results.

## Current implementation compared with a weaker evaluation workflow

| Evaluation concern | Weaker practice | Current repository |
|---|---|---|
| Model selection | Choose a preferred model in advance | Compare several model families on a common holdout |
| Time-series splitting | Random train/test split | Chronological holdout |
| Multi-step ML prediction | Use future target values implicitly | Recursive XGBoost/LSTM forecasting |
| Model comparison | Different metrics by model | Common MAE/RMSE/MAPE/WMAPE/Bias |
| Holiday effect | Assume holidays matter | Compare Prophet with and without U.S. holidays |
| Reproducibility | Narrative result only | Exportable outputs + run-manifest protocol |
| Data governance | Ambiguous sample data | Explicit synthetic/public/authorized-data rule |

## Important limitation
This table describes implementation structure, not proof that the repository is globally unique, superior to every commercial forecasting product, or nationally adopted. Independent comparison and field-use evidence would be needed for those broader claims.
