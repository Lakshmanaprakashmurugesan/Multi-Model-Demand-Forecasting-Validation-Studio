#!/usr/bin/env python3
"""Execute forecasting functions directly from app.py without modifying app.py.

The runner parses app.py and executes its imports/constants/classes/function definitions while
omitting Streamlit page-rendering statements. The forecasting function bodies therefore come
from the exact repository source file being evidenced.

Examples:
  python tools/model_execution_runner.py --models "XGBoost,Holt-Winters"
  python tools/model_execution_runner.py --models all
  python tools/model_execution_runner.py --models all --output-dir evidence/model_runs/local_full_run
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DEFAULT_OUT = ROOT / "evidence" / "model_runs" / "latest"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_source_namespace() -> dict:
    """Load definitions from app.py while excluding UI-only top-level statements."""
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP))
    kept = []
    for node in tree.body:
        # All forecasting constants/classes/functions are defined before the MODEL CARDS/UI section.
        # Excluding later top-level statements prevents any Streamlit UI state from executing.
        if getattr(node, "lineno", 0) >= 1150:
            continue
        if isinstance(node, ast.Import):
            names = [a for a in node.names if a.name != "streamlit"]
            if names:
                node.names = names
                kept.append(node)
        elif isinstance(node, ast.ImportFrom):
            kept.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
            kept.append(node)
        elif isinstance(node, ast.ClassDef):
            kept.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Retain constants / configuration; skip assignments involving Streamlit calls.
            try:
                text = ast.unparse(node)
            except Exception:
                text = ""
            if "st." not in text:
                kept.append(node)
        elif isinstance(node, ast.Try):
            # Retain optional dependency imports (Prophet, statsmodels, XGBoost).
            try:
                text = ast.unparse(node)
            except Exception:
                text = ""
            if any(x in text for x in ("prophet", "statsmodels", "xgboost")):
                kept.append(node)
        elif isinstance(node, ast.Expr):
            # Keep module docstring only.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                kept.append(node)

    module = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(module)
    import types
    module_name = "forecast_app_definitions"
    holder = types.ModuleType(module_name)
    holder.__file__ = str(APP)
    sys.modules[module_name] = holder
    ns = holder.__dict__
    exec(compile(module, str(APP), "exec"), ns, ns)
    return ns


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def save_dataframe(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="XGBoost,Holt-Winters",
                        help='Comma-separated model names or "all". Exact app names accepted.')
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--lstm-epochs", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    ns = build_source_namespace()
    available_names = list(ns["MODEL_ORDER"])
    if args.models.strip().lower() == "all":
        selected = available_names
    else:
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        # friendly aliases
        aliases = {
            "prophet": "Prophet Baseline",
            "prophet holidays": "Prophet + US Holidays",
            "prophet + holidays": "Prophet + US Holidays",
            "xgboost": "XGBoost",
            "lstm": "LSTM",
            "holt-winters": "Holt-Winters",
            "holt winters": "Holt-Winters",
        }
        selected = [aliases.get(x.lower(), x) for x in requested]
        unknown = [x for x in selected if x not in available_names]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Available: {available_names}")

    raw_df = ns["make_demo_data"]()
    clean_df, audit = ns["prepare_time_series"](
        raw_df,
        "date",
        "demand",
        ["promotion", "price_index"],
        "D",
        "Sum",
        "Time interpolation",
    )
    split_index = int(len(clean_df) * (1 - args.holdout_percent / 100))
    split_index = min(max(split_index, 10), len(clean_df) - 5)
    train_df = clean_df.iloc[:split_index].reset_index(drop=True)
    test_df = clean_df.iloc[split_index:].reset_index(drop=True)
    future_dates = ns["generate_future_dates"](clean_df["ds"].iloc[-1], args.horizon, "D")
    future_drivers = ns["future_driver_frame"](
        clean_df, future_dates, ["promotion", "price_index"],
        "Carry last observed value", 7,
    )

    runners = {
        "Prophet Baseline": lambda: ns["run_prophet_model"](
            "Prophet Baseline", train_df, test_df, clean_df, future_drivers,
            ["promotion", "price_index"], "D", False, 0.05, 10.0),
        "Prophet + US Holidays": lambda: ns["run_prophet_model"](
            "Prophet + US Holidays", train_df, test_df, clean_df, future_drivers,
            ["promotion", "price_index"], "D", True, 0.05, 10.0),
        "XGBoost": lambda: ns["run_xgboost_model"](
            train_df, test_df, clean_df, future_drivers,
            ["promotion", "price_index"], "D", 7,
            args.xgb_estimators, 5, 0.05),
        "LSTM": lambda: ns["run_lstm_model"](
            train_df, test_df, clean_df, future_drivers,
            ["promotion", "price_index"], "D", 7,
            48, args.lstm_epochs, 32),
        "Holt-Winters": lambda: ns["run_holt_winters_model"](
            train_df, test_df, clean_df, future_drivers, 7),
    }

    results = {}
    failures = {}
    execution_rows = []
    for model in selected:
        m_started = datetime.now(timezone.utc)
        try:
            result = runners[model]()
            results[model] = result
            status = "PASS"
            error = ""
            save_dataframe(result.validation, out / f"{model.replace(' ', '_').replace('+','plus')}_validation.csv")
            save_dataframe(result.future, out / f"{model.replace(' ', '_').replace('+','plus')}_future.csv")
        except Exception as exc:
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
            failures[model] = error
            (out / f"{model.replace(' ', '_').replace('+','plus')}_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
        m_ended = datetime.now(timezone.utc)
        execution_rows.append({
            "model": model,
            "status": status,
            "started_utc": m_started.isoformat(),
            "ended_utc": m_ended.isoformat(),
            "duration_seconds": round((m_ended - m_started).total_seconds(), 3),
            "error": error,
        })

    if results:
        leaderboard = ns["result_leaderboard"](results)
        save_dataframe(leaderboard, out / "model_leaderboard.csv")
        combined = ns["combined_future_table"](results)
        save_dataframe(combined, out / "combined_future_forecast.csv")
        try:
            excel_bytes = ns["export_excel"](results, leaderboard)
            (out / "forecasting_intelligence_results.xlsx").write_bytes(excel_bytes)
        except Exception as exc:
            failures["Excel export"] = f"{type(exc).__name__}: {exc}"

    with (out / "execution_status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(execution_rows[0].keys()) if execution_rows else ["model"])
        writer.writeheader()
        writer.writerows(execution_rows)

    versions = {p: package_version(p) for p in [
        "numpy", "pandas", "plotly", "scikit-learn", "prophet",
        "statsmodels", "xgboost", "tensorflow", "openpyxl", "streamlit"
    ]}
    ended = datetime.now(timezone.utc)
    manifest = {
        "run_type": "sidecar execution of exact forecasting definitions from unchanged app.py",
        "app_sha256": sha256(APP),
        "app_path": "app.py",
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "models_requested": selected,
        "models_passed": list(results),
        "failures": failures,
        "dataset": "app.py built-in deterministic demo data",
        "dataset_rows": int(len(clean_df)),
        "data_audit": audit,
        "holdout_percent": args.holdout_percent,
        "forecast_horizon": args.horizon,
        "frequency": "Daily",
        "random_seed": 42,
        "package_versions": versions,
        "source_modified_by_runner": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Model Execution Validation",
        "",
        f"- app.py SHA-256: `{manifest['app_sha256']}`",
        f"- Run started: {manifest['started_utc']}",
        f"- Dataset: {manifest['dataset']}",
        f"- Models requested: {', '.join(selected)}",
        f"- Models passed: {', '.join(results) if results else 'None'}",
        f"- Failures: {len(failures)}",
        "",
        "## Status",
    ]
    for row in execution_rows:
        lines.append(f"- {row['model']}: **{row['status']}**" + (f" — {row['error']}" if row['error'] else ""))
    lines += [
        "",
        "## Evidence Boundary",
        "This run demonstrates executable behavior of the repository's forecasting function definitions on the built-in synthetic demo dataset.",
        "It does not establish historical employer results, production deployment, third-party adoption, or national-scale outcomes.",
    ]
    (out / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Validation outputs written to: {out}")
    for row in execution_rows:
        print(f"{row['model']}: {row['status']}" + (f" ({row['error']})" if row['error'] else ""))
    # Return success if at least one requested model ran; dependency-specific failures remain documented.
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
