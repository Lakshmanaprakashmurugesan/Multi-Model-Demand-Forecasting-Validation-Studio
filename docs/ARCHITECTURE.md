# Technical Architecture

## Current implemented boundary

The existing application accepts one timestamped business time series, optional numeric drivers, a selected data frequency, and a common chronological holdout configuration. It then evaluates selected forecasting models, reports common metrics, visualizes diagnostics, refits successful models on full history, and exports future forecasts.

```mermaid
flowchart LR
    A[CSV / XLSX / Built-in Demo] --> B[Data Preparation]
    B --> C[Regular Time Index]
    C --> D[Chronological Train / Holdout Split]
    D --> P[Prophet]
    D --> PH[Prophet + US Holidays]
    D --> X[XGBoost]
    D --> L[LSTM]
    D --> H[Holt-Winters]
    P --> E[Holdout Metrics]
    PH --> E
    X --> E
    L --> E
    H --> E
    E --> R[Leaderboard + Diagnostics]
    R --> F[Refit on Full History]
    F --> O[Future Forecast]
    O --> Z[CSV / Excel Export]
```

## Implemented engineering controls

- chronological rather than random holdout;
- common evaluation metrics across model families;
- recursive multi-step forecasting for XGBoost and LSTM;
- optional numeric regressors;
- explicit U.S. holiday comparison;
- regular-frequency data preparation and duplicate handling;
- result exports for review outside the UI;
- fixed random-seed settings where supported.

## Not implemented in the unchanged source

- rolling-origin / walk-forward cross-validation;
- seasonal-naive benchmark;
- automatic multi-SKU / multi-region orchestration;
- hierarchical reconciliation;
- calibrated probabilistic forecasts;
- downstream MEIO execution interface;
- production monitoring / drift / automated retraining.

These are documented as future extensions rather than represented as current capabilities.
