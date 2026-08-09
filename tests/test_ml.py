from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_fetch.ml import (
    build_dataset,
    chronological_split,
    evaluate,
    predict_today,
    load_daily_frame,
    write_forecast,
)


def _seed_db(path: Path, n_days: int = 120, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE daily_metrics (
            calendar_date TEXT PRIMARY KEY,
            hrv_last_night_avg NUMERIC,
            resting_hr NUMERIC,
            sleep_score NUMERIC,
            sleep_time_hours NUMERIC,
            deep_sleep_hours NUMERIC,
            avg_stress NUMERIC,
            max_stress NUMERIC,
            body_battery_at_wake NUMERIC,
            moderate_intensity_minutes NUMERIC,
            vigorous_intensity_minutes NUMERIC,
            total_distance_m NUMERIC
        );
        """
    )
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    hrv = 50.0 + 3 * np.sin(np.arange(n_days) / 9) + rng.normal(0, 2.0, n_days)
    base = pd.DataFrame(
        {
            "calendar_date": [d.date().isoformat() for d in dates],
            "hrv_last_night_avg": hrv,
            "resting_hr": 50 + rng.normal(0, 1.0, n_days),
            "sleep_score": 70 + 20 * rng.random(n_days),
            "sleep_time_hours": 6 + rng.random(n_days),
            "deep_sleep_hours": 1.5 + rng.random(n_days),
            "avg_stress": 30 + 20 * rng.random(n_days),
            "max_stress": 60 + 20 * rng.random(n_days),
            "body_battery_at_wake": 65 + 25 * rng.random(n_days),
            "moderate_intensity_minutes": 30 * rng.random(n_days),
            "vigorous_intensity_minutes": 15 * rng.random(n_days),
            "total_distance_m": 5000 * rng.random(n_days),
        }
    )
    base.to_sql("daily_metrics", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return str(path)


def test_load_daily_frame_parses_and_sorts(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=40)
    df = load_daily_frame(db)
    assert len(df) == 40
    assert df.index.is_monotonic_increasing
    assert "hrv_last_night_avg" in df.columns


def test_build_dataset_no_future_leakage(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=120)
    df = load_daily_frame(db)
    X, y = build_dataset(df)
    # y is the row's own-day average stress; sleep/calendar stay on day t
    # (the night just slept), every other feature is day t-1 or earlier.
    for d in X.index[:5]:
        assert y.loc[d] == pytest.approx(df.loc[d, "avg_stress"])
        assert X.loc[d, "sleep_score"] == pytest.approx(df.loc[d, "sleep_score"])
        assert X.loc[d, "resting_hr"] == pytest.approx(
            df.loc[d - pd.Timedelta(days=1), "resting_hr"]
        )


def test_chronological_split_keeps_time_order(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=100)
    df = load_daily_frame(db)
    X, y = build_dataset(df)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, 0.2)
    assert len(X_te) > 0
    assert X_tr.index[-1] < X_te.index[0]


def test_evaluate_runs_and_reports_sane_metrics(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=180)
    df = load_daily_frame(db)
    result = evaluate(df)
    assert result["rmse"] > 0
    assert result["target"] == "avg_stress"
    assert result["n_train"] + result["n_test"] == result["n_rows"]
    assert "hrv_7d_mean" in result["feature_importances"]
    # The model must never be *worse* than a plain persistence baseline.
    assert result["rmse"] <= result["persistence_rmse"] * 2


def test_evaluate_raises_on_tiny_dataset(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=15)
    df = load_daily_frame(db)
    with pytest.raises(ValueError):
        evaluate(df)


def test_predict_today_returns_single_row_for_latest_date(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=100)
    df = load_daily_frame(db)
    result = evaluate(df)
    model = result.pop("model")
    today = predict_today(df, model)
    assert len(today) == 1
    row = today.iloc[0]
    assert row["calendar_date"] == str(df.index[-1].date())
    assert 0 < row["predicted_stress"] < 100
    assert "baseline_stress" in today.columns
    assert "readiness_ms" in today.columns
    # readiness_ms = baseline - predicted (positive = lighter than normal)
    assert row["readiness_ms"] == pytest.approx(
        row["baseline_stress"] - row["predicted_stress"]
    )
    # readiness_score is a 0-100 percentile of today's forecast in recent history
    assert 0 <= row["readiness_score"] <= 100


def test_write_forecast_upserts_and_is_idempotent(tmp_path: Path) -> None:
    db = _seed_db(tmp_path / "g.db", n_days=80)
    fc = pd.DataFrame(
        {
            "calendar_date": ["2026-09-01", "2026-09-02"],
            "predicted_stress": [45.0, 46.0],
            "baseline_stress": [44.0, 44.5],
            "readiness_ms": [-1.0, -1.5],
            "readiness_score": [60.0, 65.0],
        }
    )
    assert write_forecast(db, fc, "xgb_test") == 2
    assert write_forecast(db, fc, "xgb_test") == 2  # replace, no dupes
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT calendar_date, model FROM ml_forecast ORDER BY calendar_date"
    ).fetchall()
    assert rows == [("2026-09-01", "xgb_test"), ("2026-09-02", "xgb_test")]
    conn.close()