"""``garmin-ml``: small XGBoost forecast over ``daily_metrics``.

The first model is a training-readiness proxy for "how heavy will today sit on
my body": it predicts **today's average stress** (``avg_stress`` on day t)
from what you already know when you wake up:

* **last night's sleep** (sleep hours, deep sleep, sleep score — recorded on
  day t's row),
* **yesterday's and earlier state** (HRV level + trend, RHR, prior stress,
  load, body battery) — strictly from day t-1 and before,
* calendar (weekday, day-of-year).

No feature comes from day t itself — the target lives in the future. Readiness
is reported two ways. ``readiness_ms`` is raw: baseline − prediction (positive
means "today should sit lighter than your recent normal", i.e. readier to
train; negative means heavier). ``readiness_score`` is the headline 0-100
"how ready am I" number: the share of your last ~30 days whose stress is at
least as high as today's forecast (100 = lighter than all, 50 = typical,
0 = heavier than all) — a self-calibrating percentile, so it means the same
for anyone. Only a single row — today — is predicted.

All reads come straight from SQLite via pandas; the model is an
XGBoost regressor evaluated with a strictly chronological split (latest 20%
held out) against two honest baselines: *persistence* (yesterday's stress is
today's) and *rolling mean* (7-day average).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PROJECT_ROOT, load_dotenv

#: Columns pulled out of ``daily_metrics``; the parser fills what it can and
#: leaves the rest NULL, which pandas carries forward/ignores as needed.
_DAILY_COLS = (
    "calendar_date", "hrv_last_night_avg", "resting_hr", "sleep_score",
    "sleep_time_hours", "deep_sleep_hours", "avg_stress", "max_stress",
    "body_battery_at_wake", "moderate_intensity_minutes",
    "vigorous_intensity_minutes", "total_distance_m",
)

#: The target column: today's average physiological stress (0-100). It is a
#: stand-in for "how heavy today will sit on the body".
_TARGET_COL = "avg_stress"

#: Feature columns the model sees. Rolling means are computed over on-or-before
#: data only; ``_shift_feature_rows`` then moves every non-wake column back one
#: day so the model at date t has nothing from t in its inputs.
_FEATURE_COLS = (
    "hrv_last_night_avg",
    "hrv_7d_mean", "hrv_7d_std", "hrv_3d_delta",
    "resting_hr", "resting_hr_7d_mean",
    "sleep_score", "sleep_hours", "deep_sleep_hours",
    "avg_stress", "max_stress", "stress_7d_mean",
    "body_battery_at_wake",
    "moderate_intensity_minutes", "vigorous_intensity_minutes",
    "total_distance_m",
    "weekday_sin", "weekday_cos", "day_of_year_sin", "day_of_year_cos",
)

#: House at-wake-of-day-t read from day t's row: the just-completed night's
#: sleep and the calendar date itself. Everything else must shift to t-1.
_WAKE_COLS = frozenset({
    "sleep_score", "sleep_hours", "deep_sleep_hours",
    "weekday_sin", "weekday_cos", "day_of_year_sin", "day_of_year_cos",
})

_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def _db_path() -> Path:
    """Resolve the SQLite path without requiring Garmin credentials."""
    load_dotenv(PROJECT_ROOT / ".env")
    raw = os.getenv("GARMIN_DB_PATH", "")
    if not raw:
        return PROJECT_ROOT / "garmin.db"
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_daily_frame(db_path: str | Path) -> pd.DataFrame:
    """Read the parsed daily rows into a date-indexed frame, newest last."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {', '.join(_DAILY_COLS)} FROM daily_metrics ORDER BY calendar_date"
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame()
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df = df.set_index("calendar_date").sort_index()
    for col in _DAILY_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns; each row uses data on/before its own date only."""
    out = df.copy()
    hrv = out["hrv_last_night_avg"]
    out["hrv_7d_mean"] = hrv.rolling(7, min_periods=3).mean()
    out["hrv_7d_std"] = hrv.rolling(7, min_periods=3).std()
    out["hrv_3d_delta"] = hrv - hrv.shift(3)
    out["resting_hr_7d_mean"] = out["resting_hr"].rolling(7, min_periods=3).mean()
    out["stress_7d_mean"] = out["avg_stress"].rolling(7, min_periods=3).mean()
    out["sleep_hours"] = out["sleep_time_hours"]
    idx = out.index
    out["weekday_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    doy = idx.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for forecasting each row's *own-day* average stress.

    ``y`` is ``avg_stress`` on the row's date. Features for date t are derived
    strictly from what is known at wake that morning: the sleep columns stay on
    day t (the night just slept), every other column is shifted back one day so
    the model sees only day t-1 and earlier — no future leakage. Drop rows
    with a missing target or any NaN feature.
    """
    feats = build_features(df)
    feats["__target"] = feats[_TARGET_COL]
    rows: dict[str, pd.Series] = {}
    for col in _FEATURE_COLS:
        rows[col] = feats[col] if col in _WAKE_COLS else feats[col].shift(1)
    X = pd.DataFrame(rows, index=feats.index)
    y = feats["__target"]
    keep = y.notna() & X.notna().all(axis=1)
    return X[keep], y[keep]


def _fit_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**{**_MODEL_PARAMS}).fit(X, y)


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split on row order (dates ascend), so the test set is the latest days."""
    n_test = max(1, int(round(len(X) * test_fraction)))
    cut = len(X) - n_test
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> dict[str, Any]:
    """Train, hold out the latest days, score vs persistence + rolling mean.

    Returns metrics plus the fitted model so the caller can forecast forward.
    """
    X, y = build_dataset(df)
    if len(X) < 30:
        raise ValueError(f"need at least 30 usable rows, got {len(X)}")
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, test_fraction)
    model = _fit_model(X_tr, y_tr)
    pred = model.predict(X_te)

    stress = df["avg_stress"].dropna()
    pers_te = y_te.index.map(lambda d: stress.asof(d - dt.timedelta(days=1)))
    roll_te = y_te.index.map(
        lambda d: stress.loc[: d - dt.timedelta(days=1)].tail(7).mean()
    )

    return {
        "target": _TARGET_COL,
        "n_rows": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_start": str(X_te.index[0].date()),
        "test_end": str(X_te.index[-1].date()),
        "rmse": _rmse(y_te, pred),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "persistence_rmse": _rmse(y_te, np.asarray(pers_te, dtype=float)),
        "rolling_mean_rmse": _rmse(y_te, np.asarray(roll_te, dtype=float)),
        "feature_importances": dict(
            sorted(
                zip(_FEATURE_COLS, model.feature_importances_),
                key=lambda kv: kv[1], reverse=True,
            )
        ),
        "model": model,
    }


def _readiness_score(pred: float, history: pd.Series, window: int = 30) -> float | None:
    """Map a predicted stress onto a 0-100 readiness score via your own history.

    ``score = share of your recent days that are at least as heavy as today``.
    100 = today is lighter than any day you've had lately (maximally ready),
    0 = heavier than all of them. Percentile-based: self-calibrating, no
    arbitrary constants. ``None`` when under ``window`` recent days exist.
    """
    hist = history.dropna().tail(window)
    if len(hist) < 5:
        return None
    return round(100.0 * float((hist >= pred).mean()), 1)


def predict_today(df: pd.DataFrame, model: xgb.XGBRegressor) -> pd.DataFrame:
    """Predict **today's** average stress from wake-of-day knowledge (one row).

    The target date is the most recent calendar date in the frame — i.e. the
    morning after that night's sleep was recorded. Wake features (last night's
    sleep + calendar) come from that date's own row; every other column is
    taken at the previous day so the model sees nothing from the day it is
    being asked to predict.
    """
    feats = build_features(df)
    target_date = feats.index[-1]
    medians = feats[list(_FEATURE_COLS)].median()
    row: dict[str, Any] = {}
    for col in _FEATURE_COLS:
        src = feats[col] if col in _WAKE_COLS else feats[col].shift(1)
        row[col] = src.loc[target_date]
    feat = pd.DataFrame([row])[list(_FEATURE_COLS)].fillna(medians)
    pred = float(model.predict(feat)[0])
    before = feats.loc[: target_date - dt.timedelta(days=1)]
    baseline = before["avg_stress"].tail(7).mean()
    score = _readiness_score(pred, before["avg_stress"].tail(90))
    return pd.DataFrame([{
        "calendar_date": target_date.date().isoformat(),
        "predicted_stress": round(pred, 1),
        "baseline_stress": round(baseline, 1) if pd.notna(baseline) else None,
        # Positive = today is expected to sit LIGHTER than your recent stress
        # normal, i.e. readier to train; negative = heavier.
        "readiness_ms": round(baseline - pred, 1) if pd.notna(baseline) else None,
        "readiness_score": score,
    }])


def refresh_forecast(db_path: str | Path) -> int:
    """Train, predict today, and write the forecast (post-sync hook).

    This is the quiet entrypoint ``garmin-fetch`` calls at the end of a sync so
    the ``ml_forecast`` row is fresh without a separate ``garmin-ml --write``
    run. Like ``main``, the model is retrained from scratch on the current
    ``daily_metrics``. Thin data is not an error: it returns 0 and the caller
    logs it, it never raises.
    """
    try:
        df = load_daily_frame(db_path)
        if df.empty:
            return 0
    except sqlite3.OperationalError:
        # No daily_metrics yet (fresh DB / first sync) — nothing to train on.
        return 0
    try:
        result = evaluate(df)
    except ValueError:
        return 0
    model = result.pop("model")
    today = predict_today(df, model)
    return write_forecast(db_path, today, "xgb_stress_v1")


def write_forecast(
    db_path: str | Path, forecast: pd.DataFrame, model_name: str
) -> int:
    """Upsert the forecast rows into the ``ml_forecast`` table the agent reads."""
    with sqlite3.connect(str(db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ml_forecast)")}
        # forecasts are recomputed outputs — a table carrying obsolete columns
        # (e.g. the old HRV schema) is dropped, never migrated.
        if "calendar_date" in cols and "readiness_score" not in cols:
            conn.execute("DROP TABLE ml_forecast")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_forecast (
                calendar_date TEXT PRIMARY KEY,
                predicted_stress REAL,
                baseline_stress REAL,
                readiness_ms REAL,
                readiness_score REAL,
                model TEXT,
                created_at TEXT
            )
            """
        )
        created = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        conn.executemany(
            """
            INSERT INTO ml_forecast
                (calendar_date, predicted_stress, baseline_stress, readiness_ms, readiness_score, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(calendar_date) DO UPDATE SET
                predicted_stress = excluded.predicted_stress,
                baseline_stress = excluded.baseline_stress,
                readiness_ms = excluded.readiness_ms,
                readiness_score = excluded.readiness_score,
                model = excluded.model,
                created_at = excluded.created_at
            """,
            [
                (r.calendar_date, r.predicted_stress, r.baseline_stress,
                 r.readiness_ms, r.readiness_score, model_name, created)
                for r in forecast.itertuples(index=False)
            ],
        )
    return len(forecast)


def _print_metrics(result: dict[str, Any]) -> None:
    print(f"target: {result['target']}  rows: {result['n_rows']}  "
          f"train: {result['n_train']}  test: {result['n_test']} "
          f"({result['test_start']} - {result['test_end']})")
    print(f"model         rmse {result['rmse']:.2f}  mae {result['mae']:.2f}  "
          f"r2 {result['r2']:.2f}")
    print(f"persistence   rmse {result['persistence_rmse']:.2f}")
    print(f"rolling mean  rmse {result['rolling_mean_rmse']:.2f}")
    print("\ntop features:")
    for name, imp in list(result["feature_importances"].items())[:10]:
        print(f"  {name:24s} {imp:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="garmin-ml",
        description="Train/eval a small forecast over daily_metrics "
        "(first model: today's average stress as a training-readiness proxy).",
    )
    parser.add_argument(
        "--db", metavar="PATH", default=str(_db_path()),
        help="SQLite DB path (default: GARMIN_DB_PATH or ./garmin.db)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="latest fraction of days held out for eval (default 0.2)",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="write today's prediction into the ml_forecast table "
        "(readable by garmin-ask)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print eval metrics as JSON"
    )
    args = parser.parse_args(argv)

    df = load_daily_frame(args.db)
    if df.empty:
        print("no daily_metrics rows found")
        return 1
    try:
        result = evaluate(df, args.test_fraction)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    model = result.pop("model")

    if args.json:
        print(json.dumps(
            {k: v for k, v in result.items() if k != "feature_importances"},
            indent=2,
        ))
    else:
        _print_metrics(result)

    today = predict_today(df, model)
    if args.write:
        written = write_forecast(args.db, today, "xgb_stress_v1")
        print(f"\nwrote {written} forecast row to ml_forecast")
    else:
        print("\ntoday:")
        print(today.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())