"""Forecast Studio.

A production-style Streamlit application that compares five forecasting engines:
1. Prophet baseline
2. Prophet with aligned U.S. federal holiday effects
3. XGBoost autoregressive forecasting
4. LSTM neural-network forecasting
5. Holt-Winters exponential smoothing

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import USFederalHolidayCalendar
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from prophet import Prophet
except ImportError:  # The app still opens and explains the missing package.
    Prophet = None  # type: ignore[assignment]

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None  # type: ignore[assignment]

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[assignment]

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# PAGE AND DESIGN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecast Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #1769e0;
    --primary-soft: #eef5ff;
    --success: #159a63;
    --warning: #ee7c18;
    --text: #17233c;
    --muted: #667085;
    --line: #dfe5ec;
    --surface: #ffffff;
    --page: #f7f9fc;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--page); color: var(--text); }
[data-testid="stHeader"] { background: rgba(247,249,252,.92); }
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--text); }

.hero {
    padding: .35rem 0 1rem 0;
    margin: 0 0 .5rem 0;
    border: 0;
    background: transparent;
    box-shadow: none;
}
.hero-kicker {
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .72rem;
    font-weight: 700;
}
.hero h1 {
    margin: .35rem 0 .25rem;
    color: var(--text);
    font-size: 2rem;
    line-height: 1.15;
    letter-spacing: -.025em;
}
.hero p { color: var(--muted); font-size: .96rem; margin: 0; }

.model-card, .glass-card {
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: 0 2px 8px rgba(16,24,40,.04);
    min-height: 135px;
}
.model-card h4 { margin: 0 0 .35rem; color: var(--text); }
.model-card p, .glass-card p { color: var(--muted); font-size: .86rem; line-height: 1.5; }
.model-number {
    display: inline-flex; width: 28px; height: 28px; align-items: center; justify-content: center;
    border-radius: 8px; margin-bottom: .65rem; color: #fff; background: var(--primary); font-weight: 700;
}

[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .8rem 1rem;
    background: #fff;
    box-shadow: 0 2px 8px rgba(16,24,40,.04);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--text); }

.stButton > button, .stDownloadButton > button {
    border: 1px solid var(--primary);
    border-radius: 8px;
    font-weight: 600;
    background: var(--primary);
    color: white;
    box-shadow: none;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #0f5ac8; border-color: #0f5ac8; color: white;
}

[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    background: #fff;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem; background: transparent; padding: 0; border-bottom: 1px solid var(--line); border-radius: 0;
}
.stTabs [data-baseweb="tab"] { border-radius: 0; padding: .65rem .2rem; color: var(--muted); }
.small-note { color: var(--muted); font-size: .8rem; line-height: 1.5; }
.section-label { color: var(--muted); font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
hr { border-color: var(--line) !important; }

/* Cleaner form controls */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
.stNumberInput input, .stTextInput input {
    background: #fff !important;
    border-color: var(--line) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Time-Series Forecasting</div>
      <h1>Forecast Studio</h1>
      <p>
        Upload one business time series and compare five forecasting engines in a single guided workflow.
        The studio validates performance on an unseen holdout period, ranks the models, visualizes holiday
        effects, supports optional numeric demand drivers, and creates a downloadable future forecast package.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CONSTANTS AND DATA STRUCTURES
# -----------------------------------------------------------------------------
MODEL_ORDER = [
    "Prophet Baseline",
    "Prophet + US Holidays",
    "XGBoost",
    "LSTM",
    "Holt-Winters",
]

FREQUENCIES = {
    "Hourly": "h",
    "Daily": "D",
    "Weekly (Monday)": "W-MON",
    "Monthly (Month Start)": "MS",
}

DEFAULT_SEASONAL_PERIODS = {
    "h": 24,
    "D": 7,
    "W-MON": 52,
    "MS": 12,
}

MODEL_DESCRIPTIONS = {
    "Prophet Baseline": (
        "Trend + seasonality",
        "An interpretable baseline for trend changes and recurring calendar seasonality.",
    ),
    "Prophet + US Holidays": (
        "Holiday-aware forecasting",
        "Adds aligned U.S. federal holiday periods to estimate recurring holiday lift or decline.",
    ),
    "XGBoost": (
        "Machine-learning regression",
        "Uses autoregressive lags, calendar signals, holiday flags, and optional numeric drivers.",
    ),
    "LSTM": (
        "Deep sequence learning",
        "Learns nonlinear temporal patterns from rolling sequences and optional numeric drivers.",
    ),
    "Holt-Winters": (
        "Statistical benchmark",
        "A transparent level, trend, and seasonality benchmark for regularly spaced data.",
    ),
}


@dataclass
class ModelResult:
    name: str
    validation: pd.DataFrame
    future: pd.DataFrame
    metrics: dict[str, float]
    notes: list[str]
    model_object: Any | None = None


# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------
def format_number(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def safe_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mask = np.abs(actual_arr) > 1e-10
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100)


def safe_wmape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    denominator = np.sum(np.abs(actual_arr))
    if denominator <= 1e-10:
        return float("nan")
    return float(np.sum(np.abs(actual_arr - predicted_arr)) / denominator * 100)


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))
    mape = safe_mape(actual_arr, predicted_arr)
    wmape = safe_wmape(actual_arr, predicted_arr)
    bias = float(np.mean(predicted_arr - actual_arr))
    accuracy = float(max(0.0, 100.0 - wmape)) if np.isfinite(wmape) else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE %": mape,
        "WMAPE %": wmape,
        "Bias": bias,
        "WMAPE Accuracy Score %": accuracy,
    }


def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=900, freq="D")
    trend = np.linspace(820, 1_250, len(dates))
    weekly = 90 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    annual = 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    promo = (rng.random(len(dates)) < 0.08).astype(int)
    price_index = 100 + 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 120)
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    holiday_flag = dates.normalize().isin(holidays.normalize()).astype(int)
    demand = trend + weekly + annual + 180 * promo - 4.5 * (price_index - 100) + 120 * holiday_flag
    demand += rng.normal(0, 42, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "demand": np.maximum(demand, 0).round(2),
            "promotion": promo,
            "price_index": price_index.round(2),
        }
    )


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")


def normalize_dates_for_frequency(values: pd.Series, freq_code: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if freq_code == "h":
        return dates.dt.floor("h")
    if freq_code == "D":
        return dates.dt.floor("D")
    if freq_code == "W-MON":
        return dates.dt.to_period("W-SUN").dt.start_time
    if freq_code == "MS":
        return dates.dt.to_period("M").dt.start_time
    return dates


def prepare_time_series(
    raw_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    driver_cols: list[str],
    freq_code: str,
    target_aggregation: str,
    missing_strategy: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if date_col == target_col:
        raise ValueError("The timestamp and target columns must be different.")

    selected_cols = list(dict.fromkeys([date_col, target_col, *driver_cols]))
    working = raw_df[selected_cols].copy()
    working[date_col] = normalize_dates_for_frequency(working[date_col], freq_code)
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    for col in driver_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    invalid_dates = int(working[date_col].isna().sum())
    invalid_targets = int(working[target_col].isna().sum())
    working = working.dropna(subset=[date_col, target_col])
    if working.empty:
        raise ValueError("No valid timestamp/target rows remain after data cleaning.")

    agg_map: dict[str, str] = {target_col: "sum" if target_aggregation == "Sum" else "mean"}
    agg_map.update({col: "mean" for col in driver_cols})
    duplicate_count = int(working.duplicated(subset=[date_col]).sum())
    grouped = working.groupby(date_col, as_index=True).agg(agg_map).sort_index()

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=freq_code)
    reindexed = grouped.reindex(full_index)
    inserted_periods = int(reindexed[target_col].isna().sum())

    numeric_cols = [target_col, *driver_cols]
    if missing_strategy == "Time interpolation":
        reindexed[numeric_cols] = reindexed[numeric_cols].interpolate(method="time")
    elif missing_strategy == "Forward fill":
        reindexed[numeric_cols] = reindexed[numeric_cols].ffill()
    else:
        reindexed[numeric_cols] = reindexed[numeric_cols].fillna(0.0)

    reindexed[numeric_cols] = reindexed[numeric_cols].ffill().bfill()
    if driver_cols:
        unresolved_drivers = [col for col in driver_cols if reindexed[col].isna().any()]
        if unresolved_drivers:
            raise ValueError(
                "These driver columns contain no usable numeric values after cleaning: "
                + ", ".join(unresolved_drivers)
            )
    reindexed.index.name = "ds"
    clean = reindexed.reset_index().rename(columns={target_col: "y"})
    clean["y"] = clean["y"].astype(float)

    if clean["y"].isna().any():
        raise ValueError("The target still contains missing values after cleaning.")
    if not np.isfinite(clean["y"]).all():
        raise ValueError("The target contains infinite values.")

    audit = {
        "invalid_dates_removed": invalid_dates,
        "invalid_targets_removed": invalid_targets,
        "duplicate_timestamps_aggregated": duplicate_count,
        "missing_periods_inserted": inserted_periods,
    }
    return clean, audit


def generate_future_dates(last_date: pd.Timestamp, periods: int, freq_code: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date) + to_offset(freq_code)
    return pd.date_range(start=start, periods=periods, freq=freq_code)


def holiday_period_flags(dates: Iterable[pd.Timestamp], freq_code: str) -> np.ndarray:
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if date_index.empty:
        return np.array([], dtype=int)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=date_index.min() - pd.Timedelta(days=370),
        end=date_index.max() + pd.Timedelta(days=370),
    )

    if freq_code in {"D", "h"}:
        keys = date_index.normalize()
        holiday_keys = holidays.normalize()
    elif freq_code == "W-MON":
        keys = date_index.to_period("W-SUN")
        holiday_keys = holidays.to_period("W-SUN")
    elif freq_code == "MS":
        keys = date_index.to_period("M")
        holiday_keys = holidays.to_period("M")
    else:
        keys = date_index.normalize()
        holiday_keys = holidays.normalize()

    return np.asarray(pd.Index(keys).isin(pd.Index(holiday_keys)), dtype=int)


def aligned_prophet_holidays(dates: Iterable[pd.Timestamp], freq_code: str) -> pd.DataFrame:
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    flags = holiday_period_flags(date_index, freq_code)
    holiday_dates = date_index[flags == 1]
    return pd.DataFrame(
        {
            "ds": holiday_dates,
            "holiday": "US_Federal_Holiday_Period",
            "lower_window": 0,
            "upper_window": 0,
        }
    ).drop_duplicates(subset=["ds", "holiday"])


def future_driver_frame(
    history_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    driver_cols: list[str],
    strategy: str,
    seasonal_periods: int,
) -> pd.DataFrame:
    future = pd.DataFrame({"ds": future_dates})
    if not driver_cols:
        return future

    history = history_df[driver_cols].reset_index(drop=True)
    if strategy == "Repeat last seasonal cycle" and len(history) >= seasonal_periods:
        cycle = history.iloc[-seasonal_periods:].reset_index(drop=True)
        for col in driver_cols:
            future[col] = [cycle.loc[i % len(cycle), col] for i in range(len(future))]
    else:
        for col in driver_cols:
            future[col] = float(history[col].iloc[-1])
    return future


def time_features(dates: Iterable[pd.Timestamp], freq_code: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    iso_week = idx.isocalendar().week.to_numpy(dtype=float)
    day_of_week = idx.dayofweek.to_numpy(dtype=float)
    month = idx.month.to_numpy(dtype=float)
    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    hour = idx.hour.to_numpy(dtype=float)

    features = pd.DataFrame(
        {
            "time_ordinal": idx.asi8 / 86_400_000_000_000,
            "year": idx.year.astype(float),
            "quarter": idx.quarter.astype(float),
            "month": month,
            "iso_week": iso_week,
            "day_of_week": day_of_week,
            "day_of_month": idx.day.astype(float),
            "day_of_year": day_of_year,
            "hour": hour,
            "is_weekend": (idx.dayofweek >= 5).astype(float),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7.0),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7.0),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "is_us_holiday_period": holiday_period_flags(idx, freq_code).astype(float),
        }
    )
    return features


def exogenous_features(
    frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
) -> pd.DataFrame:
    features = time_features(frame["ds"], freq_code)
    for col in driver_cols:
        features[f"driver__{col}"] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return features


def empirical_intervals(predictions: np.ndarray, residual_std: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(residual_std) or residual_std <= 0:
        return predictions.copy(), predictions.copy()
    margin = 1.645 * residual_std  # Approximate 90% residual-based interval.
    return predictions - margin, predictions + margin


# -----------------------------------------------------------------------------
# PROPHET
# -----------------------------------------------------------------------------
def fit_prophet_predict(
    train_df: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    use_holidays: bool,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
) -> tuple[Any, pd.DataFrame]:
    if Prophet is None:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    holiday_df = None
    if use_holidays:
        combined_dates = pd.concat([train_df["ds"], prediction_frame["ds"]], ignore_index=True)
        holiday_df = aligned_prophet_holidays(combined_dates, freq_code)

    model = Prophet(
        holidays=holiday_df,
        interval_width=0.90,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        daily_seasonality="auto",
    )
    for col in driver_cols:
        model.add_regressor(col)

    fit_columns = ["ds", "y", *driver_cols]
    predict_columns = ["ds", *driver_cols]
    model.fit(train_df[fit_columns].copy())
    forecast = model.predict(prediction_frame[predict_columns].copy())
    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def run_prophet_model(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    future_drivers: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    use_holidays: bool,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
) -> ModelResult:
    validation_model, validation_forecast = fit_prophet_predict(
        train_df,
        test_df[["ds", *driver_cols]],
        driver_cols,
        freq_code,
        use_holidays,
        changepoint_prior_scale,
        seasonality_prior_scale,
    )
    validation = test_df[["ds", "y"]].merge(validation_forecast, on="ds", how="inner")
    if len(validation) != len(test_df):
        raise ValueError("Prophet did not return a prediction for every holdout timestamp.")

    final_model, future_forecast = fit_prophet_predict(
        full_df,
        future_drivers,
        driver_cols,
        freq_code,
        use_holidays,
        changepoint_prior_scale,
        seasonality_prior_scale,
    )
    metrics = calculate_metrics(validation["y"], validation["yhat"])
    notes = [
        "Future forecast was refit on the complete available history.",
        "Optional numeric drivers use the future-driver assumption selected in the sidebar.",
    ]
    if use_holidays:
        notes.append("Holiday dates are aligned to the selected data period before model fitting.")
    return ModelResult(name, validation, future_forecast, metrics, notes, final_model)


# -----------------------------------------------------------------------------
# XGBOOST
# -----------------------------------------------------------------------------
def make_xgb_training_data(
    frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lag_count: int,
) -> tuple[pd.DataFrame, pd.Series]:
    features = exogenous_features(frame, driver_cols, freq_code)
    for lag in range(1, lag_count + 1):
        features[f"lag_{lag}"] = frame["y"].shift(lag).to_numpy()
    features["rolling_mean_short"] = frame["y"].shift(1).rolling(min(7, lag_count), min_periods=1).mean().to_numpy()
    features["rolling_std_short"] = (
        frame["y"].shift(1).rolling(min(7, lag_count), min_periods=2).std().fillna(0.0).to_numpy()
    )
    valid = features.notna().all(axis=1)
    return features.loc[valid].reset_index(drop=True), frame.loc[valid, "y"].reset_index(drop=True)


def recursive_xgb_forecast(
    model: Any,
    history_values: list[float],
    prediction_frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lag_count: int,
    feature_columns: list[str],
) -> np.ndarray:
    history = [float(value) for value in history_values]
    predictions: list[float] = []

    exog = exogenous_features(prediction_frame, driver_cols, freq_code)
    for row_number in range(len(prediction_frame)):
        row = exog.iloc[row_number].to_dict()
        for lag in range(1, lag_count + 1):
            row[f"lag_{lag}"] = history[-lag]
        recent_window = history[-min(7, lag_count):]
        row["rolling_mean_short"] = float(np.mean(recent_window))
        row["rolling_std_short"] = float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0
        x_row = pd.DataFrame([row]).reindex(columns=feature_columns)
        prediction = float(model.predict(x_row)[0])
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions, dtype=float)


def fit_xgb(
    train_df: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lag_count: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[Any, list[str]]:
    if XGBRegressor is None:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")
    x_train, y_train = make_xgb_training_data(train_df, driver_cols, freq_code, lag_count)
    if len(x_train) < 20:
        raise ValueError(
            f"XGBoost has only {len(x_train)} supervised rows after creating {lag_count} lags. "
            "Use more history or reduce the lag window."
        )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.02,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    return model, list(x_train.columns)


def run_xgboost_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    future_drivers: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lag_count: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> ModelResult:
    validation_model, validation_columns = fit_xgb(
        train_df,
        driver_cols,
        freq_code,
        lag_count,
        n_estimators,
        max_depth,
        learning_rate,
    )
    validation_predictions = recursive_xgb_forecast(
        validation_model,
        train_df["y"].tolist(),
        test_df[["ds", *driver_cols]],
        driver_cols,
        freq_code,
        lag_count,
        validation_columns,
    )
    validation = test_df[["ds", "y"]].copy()
    validation["yhat"] = validation_predictions
    residual_std = float(np.std(validation["y"] - validation["yhat"], ddof=1)) if len(validation) > 1 else 0.0
    validation["yhat_lower"], validation["yhat_upper"] = empirical_intervals(validation_predictions, residual_std)

    final_model, final_columns = fit_xgb(
        full_df,
        driver_cols,
        freq_code,
        lag_count,
        n_estimators,
        max_depth,
        learning_rate,
    )
    future_predictions = recursive_xgb_forecast(
        final_model,
        full_df["y"].tolist(),
        future_drivers,
        driver_cols,
        freq_code,
        lag_count,
        final_columns,
    )
    future = future_drivers[["ds"]].copy()
    future["yhat"] = future_predictions
    future["yhat_lower"], future["yhat_upper"] = empirical_intervals(future_predictions, residual_std)
    metrics = calculate_metrics(validation["y"], validation["yhat"])
    notes = [
        "Validation predictions are recursive: the model cannot see future target values.",
        "Intervals are approximate 90% residual-based ranges, not probabilistic XGBoost intervals.",
        "Future model was refit on the complete available history.",
    ]
    return ModelResult("XGBoost", validation, future, metrics, notes, final_model)


# -----------------------------------------------------------------------------
# LSTM
# -----------------------------------------------------------------------------
def import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Install the packages in requirements.txt, or run: pip install tensorflow"
        ) from exc
    return tf


def make_lstm_raw_features(
    frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
) -> pd.DataFrame:
    exog = exogenous_features(frame, driver_cols, freq_code)
    result = pd.concat([frame[["y"]].reset_index(drop=True), exog.reset_index(drop=True)], axis=1)
    return result.astype(float)


def make_lstm_sequences(
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for index in range(lookback, len(scaled_features)):
        x_values.append(scaled_features[index - lookback:index])
        y_values.append(float(scaled_target[index, 0]))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def fit_lstm(
    train_df: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lookback: int,
    units: int,
    epochs: int,
    batch_size: int,
) -> tuple[Any, MinMaxScaler, MinMaxScaler, list[str]]:
    tf = import_tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    raw_features = make_lstm_raw_features(train_df, driver_cols, freq_code)
    if len(raw_features) < lookback + 12:
        raise ValueError(
            f"LSTM requires at least lookback + 12 rows. Current rows: {len(raw_features)}, lookback: {lookback}."
        )

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(raw_features)
    scaled_target = target_scaler.fit_transform(train_df[["y"]])
    x_train, y_train = make_lstm_sequences(scaled_features, scaled_target, lookback)
    if len(x_train) < 10:
        raise ValueError("Too few LSTM training sequences. Reduce lookback or add more history.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, x_train.shape[2])),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.LSTM(max(16, units // 2)),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    validation_split = 0.10 if len(x_train) >= 40 else 0.0
    monitor = "val_loss" if validation_split > 0 else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=max(3, min(8, epochs // 4)),
            restore_best_weights=True,
        )
    ]
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=min(batch_size, len(x_train)),
        validation_split=validation_split,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    return model, feature_scaler, target_scaler, list(raw_features.columns)


def recursive_lstm_forecast(
    model: Any,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    history_df: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lookback: int,
    feature_columns: list[str],
) -> np.ndarray:
    historical_raw = make_lstm_raw_features(history_df, driver_cols, freq_code)
    raw_history = historical_raw.to_numpy(dtype=float).tolist()
    future_exog = exogenous_features(prediction_frame, driver_cols, freq_code)
    predictions: list[float] = []

    for row_number in range(len(prediction_frame)):
        sequence_raw = np.asarray(raw_history[-lookback:], dtype=float)
        sequence_scaled = feature_scaler.transform(
            pd.DataFrame(sequence_raw, columns=feature_columns)
        ).astype(np.float32)
        x_input = np.expand_dims(sequence_scaled, axis=0)
        scaled_prediction = float(model.predict(x_input, verbose=0)[0, 0])
        prediction = float(target_scaler.inverse_transform([[scaled_prediction]])[0, 0])
        predictions.append(prediction)

        exog_row = future_exog.iloc[row_number].to_numpy(dtype=float).tolist()
        raw_history.append([prediction, *exog_row])
    return np.asarray(predictions, dtype=float)


def run_lstm_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    future_drivers: pd.DataFrame,
    driver_cols: list[str],
    freq_code: str,
    lookback: int,
    units: int,
    epochs: int,
    batch_size: int,
) -> ModelResult:
    validation_model, feature_scaler, target_scaler, feature_columns = fit_lstm(
        train_df,
        driver_cols,
        freq_code,
        lookback,
        units,
        epochs,
        batch_size,
    )
    validation_predictions = recursive_lstm_forecast(
        validation_model,
        feature_scaler,
        target_scaler,
        train_df,
        test_df[["ds", *driver_cols]],
        driver_cols,
        freq_code,
        lookback,
        feature_columns,
    )
    validation = test_df[["ds", "y"]].copy()
    validation["yhat"] = validation_predictions
    residual_std = float(np.std(validation["y"] - validation["yhat"], ddof=1)) if len(validation) > 1 else 0.0
    validation["yhat_lower"], validation["yhat_upper"] = empirical_intervals(validation_predictions, residual_std)

    final_model, final_feature_scaler, final_target_scaler, final_feature_columns = fit_lstm(
        full_df,
        driver_cols,
        freq_code,
        lookback,
        units,
        epochs,
        batch_size,
    )
    future_predictions = recursive_lstm_forecast(
        final_model,
        final_feature_scaler,
        final_target_scaler,
        full_df,
        future_drivers,
        driver_cols,
        freq_code,
        lookback,
        final_feature_columns,
    )
    future = future_drivers[["ds"]].copy()
    future["yhat"] = future_predictions
    future["yhat_lower"], future["yhat_upper"] = empirical_intervals(future_predictions, residual_std)
    metrics = calculate_metrics(validation["y"], validation["yhat"])
    notes = [
        "Validation and future forecasts are recursive and do not use unknown future target values.",
        "The LSTM is trained twice: once for holdout validation and once on all history for production forecasting.",
        "Intervals are approximate 90% residual-based ranges rather than Bayesian uncertainty intervals.",
    ]
    return ModelResult("LSTM", validation, future, metrics, notes, final_model)


# -----------------------------------------------------------------------------
# HOLT-WINTERS
# -----------------------------------------------------------------------------
def fit_holt_winters(series: pd.Series, seasonal_periods: int) -> tuple[Any, bool]:
    if ExponentialSmoothing is None:
        raise ImportError("statsmodels is not installed. Run: pip install statsmodels")
    use_seasonality = len(series) >= max(2 * seasonal_periods, seasonal_periods + 10)
    seasonal = "add" if use_seasonality else None
    model = ExponentialSmoothing(
        series.astype(float).to_numpy(),
        trend="add" if len(series) >= 8 else None,
        damped_trend=len(series) >= 8,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if use_seasonality else None,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=False)
    return model, use_seasonality


def run_holt_winters_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    future_drivers: pd.DataFrame,
    seasonal_periods: int,
) -> ModelResult:
    validation_model, validation_seasonality = fit_holt_winters(train_df["y"], seasonal_periods)
    validation_predictions = np.asarray(validation_model.forecast(len(test_df)), dtype=float)
    validation = test_df[["ds", "y"]].copy()
    validation["yhat"] = validation_predictions
    residual_std = float(np.std(validation["y"] - validation["yhat"], ddof=1)) if len(validation) > 1 else 0.0
    validation["yhat_lower"], validation["yhat_upper"] = empirical_intervals(validation_predictions, residual_std)

    final_model, final_seasonality = fit_holt_winters(full_df["y"], seasonal_periods)
    future_predictions = np.asarray(final_model.forecast(len(future_drivers)), dtype=float)
    future = future_drivers[["ds"]].copy()
    future["yhat"] = future_predictions
    future["yhat_lower"], future["yhat_upper"] = empirical_intervals(future_predictions, residual_std)
    metrics = calculate_metrics(validation["y"], validation["yhat"])
    notes = [
        "Holt-Winters uses only the historical target; optional external drivers are intentionally ignored.",
        "Additive seasonality was enabled." if final_seasonality else "Seasonality was disabled because the history was too short for a stable seasonal initialization.",
        "Intervals are approximate 90% residual-based ranges.",
    ]
    if validation_seasonality != final_seasonality:
        notes.append("The validation and final refit used different seasonality settings because their history lengths differ.")
    return ModelResult("Holt-Winters", validation, future, metrics, notes, final_model)


# -----------------------------------------------------------------------------
# VISUALIZATION AND EXPORT
# -----------------------------------------------------------------------------
def result_leaderboard(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append({"Model": name, **result.metrics})
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    leaderboard["Rank Score"] = leaderboard["WMAPE %"].where(
        np.isfinite(leaderboard["WMAPE %"]), leaderboard["MAE"]
    )
    leaderboard = leaderboard.sort_values(["Rank Score", "MAE"], ascending=True).reset_index(drop=True)
    leaderboard.insert(0, "Rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard.drop(columns="Rank Score")


def forecast_comparison_figure(
    full_df: pd.DataFrame,
    split_date: pd.Timestamp,
    results: dict[str, ModelResult],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=full_df["ds"],
            y=full_df["y"],
            name="Actual",
            mode="lines",
            line={"width": 2.5},
        )
    )
    for name, result in results.items():
        fig.add_trace(
            go.Scatter(
                x=result.validation["ds"],
                y=result.validation["yhat"],
                name=f"{name} • Holdout",
                mode="lines",
                line={"dash": "dash", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=result.future["ds"],
                y=result.future["yhat"],
                name=f"{name} • Future",
                mode="lines",
                line={"width": 2.2},
            )
        )
    fig.add_vline(x=split_date.timestamp() * 1000, line_dash="dot", annotation_text="Holdout begins")
    fig.update_layout(
        title="Actual History, Holdout Predictions, and Future Forecasts",
        xaxis_title="Time",
        yaxis_title="Target",
        hovermode="x unified",
        template="plotly_white",
        height=640,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 30, "r": 30, "t": 95, "b": 35},
    )
    return fig


def validation_diagnostics_figure(result: ModelResult) -> go.Figure:
    residuals = result.validation["y"] - result.validation["yhat"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Holdout: Actual vs Forecast", "Holdout Residuals"),
        vertical_spacing=0.16,
    )
    fig.add_trace(
        go.Scatter(x=result.validation["ds"], y=result.validation["y"], name="Actual", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.validation["ds"], y=result.validation["yhat"], name="Forecast", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=result.validation["ds"], y=residuals, name="Residual"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", row=2, col=1)
    fig.update_layout(template="plotly_white", height=650, hovermode="x unified", showlegend=True)
    return fig


def prophet_holiday_comparison_figure(results: dict[str, ModelResult]) -> go.Figure | None:
    baseline = results.get("Prophet Baseline")
    holiday = results.get("Prophet + US Holidays")
    if baseline is None or holiday is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline.validation["ds"], y=baseline.validation["y"], name="Actual", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=baseline.validation["ds"],
            y=baseline.validation["yhat"],
            name="Prophet Baseline",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=holiday.validation["ds"],
            y=holiday.validation["yhat"],
            name="Prophet + Holidays",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Prophet Holiday Effect on the Holdout Period",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Target",
    )
    return fig


def export_excel(results: dict[str, ModelResult], leaderboard: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        leaderboard.to_excel(writer, sheet_name="Model Leaderboard", index=False)
        for model_name, result in results.items():
            safe_name = model_name.replace("+", "plus").replace("-", " ")[:24]
            result.validation.to_excel(writer, sheet_name=f"{safe_name} Validation"[:31], index=False)
            result.future.to_excel(writer, sheet_name=f"{safe_name} Future"[:31], index=False)
    buffer.seek(0)
    return buffer.getvalue()


def combined_future_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for name, result in results.items():
        model_frame = result.future[["ds", "yhat"]].rename(columns={"yhat": name})
        combined = model_frame if combined is None else combined.merge(model_frame, on="ds", how="outer")
    return combined.sort_values("ds").reset_index(drop=True) if combined is not None else pd.DataFrame()


# -----------------------------------------------------------------------------
# MODEL CARDS
# -----------------------------------------------------------------------------
model_columns = st.columns(len(MODEL_ORDER))
for index, model_name in enumerate(MODEL_ORDER, start=1):
    subtitle, description = MODEL_DESCRIPTIONS[model_name]
    with model_columns[index - 1]:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-number">{index}</div>
                <h4>{model_name}</h4>
                <p><strong>{subtitle}</strong><br>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.markdown("## Forecast Settings")
st.sidebar.caption("Set up your data and forecast in a few simple steps.")

source_mode = st.sidebar.radio("1. Data source", ["Upload a file", "Use built-in demo"], horizontal=False)
raw_df: pd.DataFrame | None = None
source_label = ""

if source_mode == "Upload a file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, XLSX, or XLS",
        type=["csv", "xlsx", "xls"],
        help="The file must contain one timestamp column and one numeric target column.",
    )
    if uploaded_file is not None:
        try:
            raw_df = load_uploaded_data(uploaded_file.getvalue(), uploaded_file.name)
            source_label = uploaded_file.name
        except Exception as exc:
            st.sidebar.error(f"Could not read the file: {exc}")
else:
    raw_df = make_demo_data()
    source_label = "Built-in retail demand demo"

if raw_df is None:
    st.info("Upload a dataset in the sidebar, or choose the built-in demo to explore the complete application.")
    st.stop()

st.sidebar.success(f"Loaded: {source_label}")

columns = list(raw_df.columns)
date_default = next((i for i, col in enumerate(columns) if str(col).lower() in {"ds", "date", "timestamp", "datetime"}), 0)
target_candidates = [i for i, col in enumerate(columns) if str(col).lower() in {"y", "demand", "sales", "value", "target"}]
target_default = target_candidates[0] if target_candidates else min(1, len(columns) - 1)

date_col = st.sidebar.selectbox("2. Timestamp column", columns, index=date_default)
target_col = st.sidebar.selectbox("3. Target column", columns, index=target_default)

numeric_candidates = [
    col
    for col in columns
    if col not in {date_col, target_col}
    and (
        pd.api.types.is_numeric_dtype(raw_df[col])
        or pd.to_numeric(raw_df[col], errors="coerce").notna().mean() >= 0.80
    )
]
driver_cols = st.sidebar.multiselect(
    "4. Optional numeric drivers",
    numeric_candidates,
    help="Examples: promotion, price, weather, subscribers, media spend, or availability.",
)

frequency_label = st.sidebar.selectbox("5. Data frequency", list(FREQUENCIES.keys()), index=1)
freq_code = FREQUENCIES[frequency_label]
default_seasonal = DEFAULT_SEASONAL_PERIODS[freq_code]

with st.sidebar.expander("Data preparation", expanded=False):
    target_aggregation = st.selectbox("Duplicate timestamp aggregation", ["Sum", "Mean"])
    missing_strategy = st.selectbox("Missing-period treatment", ["Time interpolation", "Forward fill", "Fill with zero"])

try:
    clean_df, audit = prepare_time_series(
        raw_df,
        date_col,
        target_col,
        driver_cols,
        freq_code,
        target_aggregation,
        missing_strategy,
    )
except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()

if len(clean_df) < 20:
    st.error("At least 20 regularly spaced observations are required after cleaning.")
    st.stop()

max_horizon = min(365, max(1, len(clean_df) // 2))
default_horizon = min(30, max_horizon)
horizon = st.sidebar.slider("6. Forecast horizon", 1, max_horizon, default_horizon)
holdout_percent = st.sidebar.slider("7. Holdout size (%)", 10, 40, 20, 5)

max_seasonal = max(2, min(365, len(clean_df) // 2))
seasonal_periods = st.sidebar.number_input(
    "8. Seasonal cycle length",
    min_value=2,
    max_value=max_seasonal,
    value=min(default_seasonal, max_seasonal),
    step=1,
    help="Examples: 24 for hourly daily seasonality, 7 for daily weekly seasonality, 52 for weekly annual seasonality, 12 for monthly annual seasonality.",
)

max_lookback = max(2, min(120, len(clean_df) // 3))
default_lookback = min(max(default_seasonal, 7), max_lookback)
lookback = st.sidebar.slider("9. ML/LSTM lookback", 2, max_lookback, default_lookback)

future_driver_strategy = st.sidebar.selectbox(
    "10. Future driver assumption",
    ["Carry last observed value", "Repeat last seasonal cycle"],
    disabled=not bool(driver_cols),
    help="Future external-driver values are unknown unless you provide a separate future-driver plan. This assumption fills them for the forecast horizon.",
)

selected_models = st.sidebar.multiselect(
    "11. Forecast engines",
    MODEL_ORDER,
    default=MODEL_ORDER,
)

with st.sidebar.expander("Advanced model settings", expanded=False):
    st.markdown("**Prophet**")
    changepoint_prior_scale = st.slider("Trend flexibility", 0.001, 0.500, 0.050, 0.001, format="%.3f")
    seasonality_prior_scale = st.slider("Seasonality flexibility", 0.1, 20.0, 10.0, 0.1)
    st.markdown("**XGBoost**")
    xgb_estimators = st.slider("Trees", 100, 1_000, 400, 50)
    xgb_depth = st.slider("Tree depth", 2, 10, 5)
    xgb_learning_rate = st.slider("Learning rate", 0.01, 0.30, 0.05, 0.01)
    st.markdown("**LSTM**")
    lstm_units = st.slider("LSTM units", 16, 128, 48, 16)
    lstm_epochs = st.slider("Maximum epochs", 10, 150, 40, 10)
    lstm_batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)

# -----------------------------------------------------------------------------
# DATA OVERVIEW
# -----------------------------------------------------------------------------
preview_tab, guide_tab = st.tabs(["📊 Data Readiness", "🧭 Model Selection Guide"])

with preview_tab:
    st.markdown('<div class="section-label">Prepared dataset</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", f"{len(clean_df):,}")
    metric_cols[1].metric("Start", clean_df["ds"].min().strftime("%Y-%m-%d"))
    metric_cols[2].metric("End", clean_df["ds"].max().strftime("%Y-%m-%d"))
    metric_cols[3].metric("Target average", format_number(clean_df["y"].mean()))
    metric_cols[4].metric("Inserted periods", f"{audit['missing_periods_inserted']:,}")

    left_preview, right_preview = st.columns([1.15, 1.85])
    with left_preview:
        st.subheader("Clean data preview")
        st.dataframe(clean_df.head(12), width="stretch", height=330)
    with right_preview:
        preview_fig = go.Figure()
        preview_fig.add_trace(go.Scatter(x=clean_df["ds"], y=clean_df["y"], mode="lines", name="Target"))
        preview_fig.update_layout(
            template="plotly_white",
            height=330,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title=target_col,
        )
        st.plotly_chart(preview_fig, width="stretch")

    st.caption(
        f"Cleaning audit — invalid dates removed: {audit['invalid_dates_removed']:,}; "
        f"invalid targets removed: {audit['invalid_targets_removed']:,}; "
        f"duplicate timestamps aggregated: {audit['duplicate_timestamps_aggregated']:,}."
    )

    with st.expander("Detected U.S. federal holiday periods", expanded=False):
        preview_future_dates = generate_future_dates(clean_df["ds"].iloc[-1], horizon, freq_code)
        holiday_preview = aligned_prophet_holidays(
            pd.concat([clean_df["ds"], pd.Series(preview_future_dates)], ignore_index=True),
            freq_code,
        )
        if holiday_preview.empty:
            st.info("No aligned U.S. federal holiday periods were detected in the selected range.")
        else:
            st.dataframe(holiday_preview[["ds", "holiday"]], width="stretch", hide_index=True, height=260)

with guide_tab:
    guide = pd.DataFrame(
        [
            ["Prophet Baseline", "Trend, changepoints, calendar seasonality", "Yes", "Strong interpretable baseline"],
            ["Prophet + US Holidays", "Holiday-sensitive business demand", "Yes", "Direct holiday-effect comparison"],
            ["XGBoost", "Nonlinear effects, promotions, prices, lag interactions", "Yes", "Usually strong on structured business data"],
            ["LSTM", "Complex sequence patterns with enough history", "Yes", "Flexible but computationally heavier"],
            ["Holt-Winters", "Stable level/trend/seasonality", "No", "Transparent statistical benchmark"],
        ],
        columns=["Model", "Best suited for", "Uses selected drivers", "Primary value"],
    )
    st.dataframe(guide, width="stretch", hide_index=True)
    st.info(
        "Model quality depends on data volume, frequency, structural changes, leakage prevention, and realistic future-driver assumptions. "
        "The application therefore ranks models using an unseen chronological holdout rather than in-sample fit."
    )

# -----------------------------------------------------------------------------
# RUN MODELS
# -----------------------------------------------------------------------------
run_button = st.button("Run Forecast", type="primary", width="stretch")

if run_button:
    if not selected_models:
        st.error("Select at least one forecasting engine.")
        st.stop()

    split_index = int(len(clean_df) * (1 - holdout_percent / 100))
    split_index = min(max(split_index, 10), len(clean_df) - 5)
    train_df = clean_df.iloc[:split_index].reset_index(drop=True)
    test_df = clean_df.iloc[split_index:].reset_index(drop=True)
    future_dates = generate_future_dates(clean_df["ds"].iloc[-1], horizon, freq_code)
    future_drivers = future_driver_frame(
        clean_df,
        future_dates,
        driver_cols,
        future_driver_strategy,
        int(seasonal_periods),
    )

    runners: dict[str, Callable[[], ModelResult]] = {
        "Prophet Baseline": lambda: run_prophet_model(
            "Prophet Baseline",
            train_df,
            test_df,
            clean_df,
            future_drivers,
            driver_cols,
            freq_code,
            False,
            changepoint_prior_scale,
            seasonality_prior_scale,
        ),
        "Prophet + US Holidays": lambda: run_prophet_model(
            "Prophet + US Holidays",
            train_df,
            test_df,
            clean_df,
            future_drivers,
            driver_cols,
            freq_code,
            True,
            changepoint_prior_scale,
            seasonality_prior_scale,
        ),
        "XGBoost": lambda: run_xgboost_model(
            train_df,
            test_df,
            clean_df,
            future_drivers,
            driver_cols,
            freq_code,
            lookback,
            xgb_estimators,
            xgb_depth,
            xgb_learning_rate,
        ),
        "LSTM": lambda: run_lstm_model(
            train_df,
            test_df,
            clean_df,
            future_drivers,
            driver_cols,
            freq_code,
            lookback,
            lstm_units,
            lstm_epochs,
            lstm_batch_size,
        ),
        "Holt-Winters": lambda: run_holt_winters_model(
            train_df,
            test_df,
            clean_df,
            future_drivers,
            int(seasonal_periods),
        ),
    }

    progress = st.progress(0, text="Initializing forecasting engines...")
    status_area = st.empty()
    results: dict[str, ModelResult] = {}
    failures: dict[str, str] = {}

    for index, model_name in enumerate(selected_models, start=1):
        status_area.info(f"Running {model_name}...")
        try:
            results[model_name] = runners[model_name]()
        except Exception as exc:
            failures[model_name] = str(exc)
        progress.progress(index / len(selected_models), text=f"Completed {index} of {len(selected_models)} engines")

    progress.empty()
    status_area.empty()
    st.session_state["forecast_results"] = results
    st.session_state["forecast_failures"] = failures
    st.session_state["forecast_clean_df"] = clean_df
    st.session_state["forecast_split_date"] = test_df["ds"].iloc[0]
    st.session_state["forecast_configuration"] = {
        "source": source_label,
        "frequency": frequency_label,
        "horizon": horizon,
        "holdout_percent": holdout_percent,
        "drivers": driver_cols,
        "future_driver_strategy": future_driver_strategy,
    }

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
results = st.session_state.get("forecast_results", {})
failures = st.session_state.get("forecast_failures", {})
result_clean_df = st.session_state.get("forecast_clean_df")
split_date = st.session_state.get("forecast_split_date")

if failures:
    with st.expander("⚠️ Models that could not run", expanded=True):
        for model_name, error_message in failures.items():
            st.error(f"**{model_name}:** {error_message}")

if results and result_clean_df is not None and split_date is not None:
    leaderboard = result_leaderboard(results)
    winner_name = str(leaderboard.iloc[0]["Model"])
    winner = results[winner_name]

    st.markdown("---")
    st.markdown('<div class="section-label">Forecast results</div>', unsafe_allow_html=True)
    st.header("🏆 Model Performance Leaderboard")

    winner_cols = st.columns(5)
    winner_cols[0].metric("Best model", winner_name)
    winner_cols[1].metric("Best WMAPE", f"{format_number(winner.metrics['WMAPE %'])}%")
    winner_cols[2].metric("Best MAE", format_number(winner.metrics["MAE"]))
    winner_cols[3].metric("WMAPE accuracy score", f"{format_number(winner.metrics['WMAPE Accuracy Score %'])}%")
    winner_cols[4].metric("Models completed", len(results))

    display_leaderboard = leaderboard.copy()
    numeric_columns = ["MAE", "RMSE", "MAPE %", "WMAPE %", "Bias", "WMAPE Accuracy Score %"]
    for col in numeric_columns:
        display_leaderboard[col] = display_leaderboard[col].map(lambda value: round(value, 3) if np.isfinite(value) else np.nan)
    st.dataframe(display_leaderboard, width="stretch", hide_index=True)

    overview_tab, holiday_tab, details_tab, export_tab = st.tabs(
        ["📈 Forecast Comparison", "📅 Holiday Impact", "🧪 Model Diagnostics", "📦 Export Center"]
    )

    with overview_tab:
        comparison_fig = forecast_comparison_figure(result_clean_df, pd.Timestamp(split_date), results)
        st.plotly_chart(comparison_fig, width="stretch")
        st.caption(
            "Dashed model lines represent the unseen chronological holdout. Solid model lines after the historical series represent forecasts refit on all available observations."
        )

    with holiday_tab:
        baseline = results.get("Prophet Baseline")
        holiday_model = results.get("Prophet + US Holidays")
        holiday_fig = prophet_holiday_comparison_figure(results)
        if baseline is not None and holiday_model is not None and holiday_fig is not None:
            delta_cols = st.columns(4)
            wmape_delta = holiday_model.metrics["WMAPE %"] - baseline.metrics["WMAPE %"]
            mae_delta = holiday_model.metrics["MAE"] - baseline.metrics["MAE"]
            delta_cols[0].metric("Baseline WMAPE", f"{format_number(baseline.metrics['WMAPE %'])}%")
            delta_cols[1].metric(
                "Holiday WMAPE",
                f"{format_number(holiday_model.metrics['WMAPE %'])}%",
                delta=f"{wmape_delta:+.2f} pp",
                delta_color="inverse",
            )
            delta_cols[2].metric("Baseline MAE", format_number(baseline.metrics["MAE"]))
            delta_cols[3].metric(
                "Holiday MAE",
                format_number(holiday_model.metrics["MAE"]),
                delta=f"{mae_delta:+.2f}",
                delta_color="inverse",
            )
            st.plotly_chart(holiday_fig, width="stretch")
            if wmape_delta < 0:
                st.success(f"Holiday modeling improved holdout WMAPE by {abs(wmape_delta):.2f} percentage points.")
            elif wmape_delta > 0:
                st.warning(
                    f"Holiday modeling increased holdout WMAPE by {wmape_delta:.2f} percentage points. "
                    "This dataset may not contain a stable recurring holiday effect, or the model may require tuning."
                )
            else:
                st.info("The holiday model produced the same WMAPE as the baseline on this holdout.")
        else:
            st.info("Run both Prophet variants to unlock the direct holiday-effect comparison.")

    with details_tab:
        detail_tabs = st.tabs(list(results.keys()))
        for tab, (model_name, result) in zip(detail_tabs, results.items()):
            with tab:
                metric_cols = st.columns(6)
                metric_cols[0].metric("MAE", format_number(result.metrics["MAE"]))
                metric_cols[1].metric("RMSE", format_number(result.metrics["RMSE"]))
                metric_cols[2].metric("MAPE", f"{format_number(result.metrics['MAPE %'])}%")
                metric_cols[3].metric("WMAPE", f"{format_number(result.metrics['WMAPE %'])}%")
                metric_cols[4].metric("Bias", format_number(result.metrics["Bias"]))
                metric_cols[5].metric("WMAPE score", f"{format_number(result.metrics['WMAPE Accuracy Score %'])}%")
                st.plotly_chart(validation_diagnostics_figure(result), width="stretch")
                left_table, right_table = st.columns(2)
                with left_table:
                    st.subheader("Holdout predictions")
                    st.dataframe(result.validation, width="stretch", height=300)
                with right_table:
                    st.subheader("Future forecast")
                    st.dataframe(result.future, width="stretch", height=300)
                st.markdown("**Model notes**")
                for note in result.notes:
                    st.markdown(f"- {note}")

    with export_tab:
        combined_future = combined_future_table(results)
        st.subheader("Combined future forecast")
        st.dataframe(combined_future, width="stretch", hide_index=True)

        csv_bytes = combined_future.to_csv(index=False).encode("utf-8")
        excel_bytes = export_excel(results, leaderboard)
        export_cols = st.columns(2)
        export_cols[0].download_button(
            "⬇️ Download combined forecast CSV",
            data=csv_bytes,
            file_name="combined_future_forecast.csv",
            mime="text/csv",
            width="stretch",
        )
        export_cols[1].download_button(
            "⬇️ Download complete Excel package",
            data=excel_bytes,
            file_name="forecasting_intelligence_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        st.caption(
            "The Excel package contains the model leaderboard plus separate validation and future sheets for every successful engine."
        )

st.markdown("---")
st.markdown(
    """
    <div class="small-note">
      <strong>Responsible-use note:</strong> No forecasting model is universally correct. Always compare advanced models against naive benchmarks. Use chronological backtesting,
      business review, realistic future-driver scenarios, and monitoring for drift before operational deployment.
      LSTM may take substantially longer than the other models on CPU-only machines.
    </div>
    """,
    unsafe_allow_html=True,
)
