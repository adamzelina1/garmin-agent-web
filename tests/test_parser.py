from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from garmin_fetch.db import Database
from garmin_fetch.parser import (
    build_activity_details,
    build_activity_summaries,
    build_activity_weather,
    build_daily_rows,
    build_hr_zones,
    build_race_predictions,
    parse_activity_detail_series,
    parse_activity_details,
    parse_activity_summary,
    parse_activity_weather,
    parse_body_battery,
    parse_heart_rate,
    parse_hrv,
    parse_hr_zones,
    parse_lactate_threshold,
    parse_max_metrics,
    parse_race_predictions,
    parse_rhr,
    parse_sleep,
    parse_stats,
    parse_steps,
    parse_training_status,
    PARSERS,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "test.db"))


def test_parse_heart_rate() -> None:
    out = parse_heart_rate({
        "restingHeartRate": 54,
        "minHeartRate": 52,
        "maxHeartRate": 173,
        "lastSevenDaysAvgRestingHeartRate": 53,
    })
    assert out == {
        "resting_hr": 54,
        "min_hr": 52,
        "max_hr": 173,
        "last_7d_avg_resting_hr": 53,
    }


def test_parse_heart_rate_zones() -> None:
    out = parse_heart_rate({
        "restingHeartRate": 54,
        "heartRateZones": [
            {"zoneNumber": 1, "min": 0, "max": 124, "secondsInZone": 36120},
            {"zoneNumber": 2, "min": 125, "max": 140, "secondsInZone": 2713},
        ],
    })
    assert out["hr_zone_1_min"] == 0
    assert out["hr_zone_1_max"] == 124
    assert out["hr_zone_1_hours"] == pytest.approx(36120 / 3600, abs=1e-4)
    assert out["hr_zone_2_min"] == 125
    assert out["hr_zone_2_max"] == 140
    assert out["hr_zone_2_hours"] == pytest.approx(2713 / 3600, abs=1e-4)


def test_parse_steps_sums_buckets() -> None:
    buckets = [
        {"steps": 100, "pushes": 2},
        {"steps": 250, "pushes": 0},
        {"steps": 50, "pushes": 1},
    ]
    out = parse_steps({"calendarDate": "2026-08-01", "value": buckets})
    assert out == {"total_steps": 400, "pushes": 3}


def test_parse_steps_empty_returns_empty() -> None:
    assert parse_steps({"value": []}) == {}
    assert parse_steps({"value": "nope"}) == {}


def test_parse_sleep_flat_and_score() -> None:
    out = parse_sleep({
        "dailySleepDTO": {
            "sleepTimeSeconds": 28675,
            "deepSleepSeconds": 6060,
            "remSleepSeconds": 8580,
            "awakeSleepSeconds": 1740,
            "sleepStartTimestampLocal": 1737158617000,
            "sleepEndTimestampLocal": 1737169417000,
            "sleepScores": {"overall": {"value": 90, "qualifierKey": "EXCELLENT"}},
        },
        "restingHeartRate": 54,
        "hrvStatus": "BALANCED",
    })
    assert out["sleep_time_hours"] == pytest.approx(28675 / 3600, abs=1e-4)
    assert out["deep_sleep_hours"] == pytest.approx(6060 / 3600, abs=1e-4)
    assert out["sleep_score"] == 90
    assert out["sleep_score_qualifier"] == "EXCELLENT"
    assert out["resting_hr"] == 54
    assert out["hrv_status"] == "BALANCED"
    assert re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", out["sleep_start_local"])
    assert out["sleep_start_local"] == datetime.fromtimestamp(
        1737158617000 / 1000
    ).strftime("%H:%M")
    assert out["sleep_end_local"] == datetime.fromtimestamp(
        1737169417000 / 1000
    ).strftime("%H:%M")
    assert "sleep_time_seconds" not in out
    assert "avg_overnight_hrv" not in out


def test_parse_hrv_from_summary() -> None:
    out = parse_hrv({
        "hrvSummary": {
            "lastNightAvg": 42,
            "lastNight5MinHigh": 104,
            "weeklyAvg": 40,
            "baseline": 45,
            "status": "BALANCED",
        }
    })
    assert out["hrv_last_night_avg"] == 42
    assert out["hrv_status"] == "BALANCED"
    assert "hrv_weekly_avg" not in out


def test_parse_rhr_nested() -> None:
    out = parse_rhr({
        "allMetrics": {
            "metricsMap": {
                "WELLNESS_RESTING_HEART_RATE": [{"value": 62.0, "calendarDate": "x"}]
            }
        }
    })
    assert out == {"resting_hr": 62.0}


def test_parse_rhr_missing_returns_empty() -> None:
    assert parse_rhr({"allMetrics": {"metricsMap": {}}}) == {}


def test_parse_max_metrics_from_wrapped_list() -> None:
    out = parse_max_metrics({
        "value": [{"generic": {"vo2MaxValue": 43, "vo2MaxPreciseValue": 43.0}}]
    })
    assert out == {"vo2max": 43, "vo2max_precise": 43.0}


def test_parse_body_battery_net_change() -> None:
    out = parse_body_battery({
        "value": [
            {"event": {"eventType": "SLEEP", "bodyBatteryImpact": 35}},
            {"event": {"eventType": "ACTIVITY", "bodyBatteryImpact": -16}},
        ]
    })
    assert out["body_battery_net_change"] == 19.0
    assert out["body_battery_slept"] is True


def test_parse_stats_rich() -> None:
    out = parse_stats({
        "totalSteps": 1153,
        "totalDistanceMeters": 1012,
        "restingHeartRate": 62,
        "averageStressLevel": 10,
        "bodyBatteryChargedValue": 16,
        "highlyActiveSeconds": 1800,
        "activeSeconds": 7200,
        "sedentarySeconds": 10800,
        "activeKilocalories": 300,
        "latestRespirationValue": 14.0,
    })
    assert out["total_steps"] == 1153
    assert out["resting_hr"] == 62
    assert out["avg_stress"] == 10
    assert out["body_battery_charged"] == 16
    assert out["active_kcal"] == 300
    assert out["highly_active_hours"] == pytest.approx(0.5)
    assert out["active_hours"] == pytest.approx(2.0)
    assert out["sedentary_hours"] == pytest.approx(3.0)
    assert "latest_respiration" not in out


def test_parse_training_status() -> None:
    out = parse_training_status({
        "mostRecentVO2Max": {"generic": {"vo2MaxValue": 43}},
        "mostRecentTrainingStatus": {
            "latestTrainingStatusData": {
                "3505199781": {
                    "trainingStatus": 1,
                    "weeklyTrainingLoad": 255,
                    "trainingStatusFeedbackPhrase": "PRODUCTIVE",
                }
            }
        },
        "mostRecentTrainingLoadBalance": {"metricsTrainingLoadBalanceDTOMap": None},
    })
    assert out["vo2max"] == 43
    assert out["training_status"] == 1
    assert out["weekly_training_load"] == 255


def test_parse_lactate_threshold() -> None:
    out = parse_lactate_threshold({
        "heart_rate": [{"value": 165, "calendarDate": "2026-08-01"}],
        "speed": [{"value": 0.353, "calendarDate": "2026-08-01"}],
        "power": [{"value": 300}],
    })
    assert out == {
        "lactate_threshold_hr": 165,
        "lactate_threshold_speed": 0.353,
        "ftp_watts": 300,
    }


def test_parse_lactate_threshold_empty() -> None:
    assert parse_lactate_threshold({}) == {}
    assert parse_lactate_threshold(
        {"heart_rate": [], "speed": [], "power": []}
    ) == {}


def test_all_parsers_registered_for_data_types() -> None:
    # Every registered parser is callable with a payload and returns a dict.
    for name, parser in PARSERS.items():
        assert callable(parser)
        assert isinstance(PARSERS[name]({}), dict)


def test_merge_daily_creates_row_and_columns(db: Database) -> None:
    db.merge_daily("2026-08-01", {"resting_hr": 62, "total_steps": 900}, "t")
    db.merge_daily("2026-08-02", {"resting_hr": 60}, "t2")
    row = db.conn.execute(
        "SELECT resting_hr, total_steps FROM daily_metrics WHERE calendar_date='2026-08-01'"
    ).fetchone()
    assert row["resting_hr"] == 62
    assert row["total_steps"] == 900


def test_merge_daily_merges_across_calls(db: Database) -> None:
    db.merge_daily("2026-08-01", {"resting_hr": 64}, "t")
    db.merge_daily("2026-08-01", {"total_steps": 900}, "t2")
    row = db.conn.execute(
        "SELECT resting_hr, total_steps FROM daily_metrics WHERE calendar_date='2026-08-01'"
    ).fetchone()
    assert row["resting_hr"] == 64
    assert row["total_steps"] == 900


def test_build_daily_rows_from_metrics(db: Database) -> None:
    db.upsert_metric("heart_rate", {"calendarDate": "2026-08-01", "restingHeartRate": 62, "fetched_at": "t"})
    db.upsert_metric("steps", {"calendarDate": "2026-08-01", "value": [{"steps": 100}], "fetched_at": "t"})
    counts = build_daily_rows(db, ["heart_rate", "steps"])
    assert counts == {"heart_rate": 1, "steps": 1}
    row = db.conn.execute(
        "SELECT resting_hr, total_steps FROM daily_metrics WHERE calendar_date='2026-08-01'"
    ).fetchone()
    assert row["resting_hr"] == 62
    assert row["total_steps"] == 100


def test_build_daily_rows_forward_fills_lactate(db: Database) -> None:
    """Sparse types carry the last known value into dates with no data."""
    db.upsert_metric("lactate_threshold", {
        "calendarDate": "2026-08-01",
        "heart_rate": [{"value": 179}],
        "speed": [{"value": 0.5}],
        "power": [{"value": 300}],
        "fetched_at": "t1",
    })
    db.upsert_metric("lactate_threshold", {
        "calendarDate": "2026-08-02",
        "heart_rate": [], "speed": [], "power": [],
        "fetched_at": "t2",
    })
    db.upsert_metric("lactate_threshold", {
        "calendarDate": "2026-08-03",
        "heart_rate": [{"value": 172}],
        "fetched_at": "t3",
    })
    assert build_daily_rows(db, ["lactate_threshold"]) == {"lactate_threshold": 3}

    row1 = db.conn.execute(
        "SELECT * FROM daily_metrics WHERE calendar_date='2026-08-01'"
    ).fetchone()
    row2 = db.conn.execute(
        "SELECT * FROM daily_metrics WHERE calendar_date='2026-08-02'"
    ).fetchone()
    row3 = db.conn.execute(
        "SELECT * FROM daily_metrics WHERE calendar_date='2026-08-03'"
    ).fetchone()
    assert row1["lactate_threshold_hr"] == 179
    assert row2["lactate_threshold_hr"] == 179  # carried
    assert row3["lactate_threshold_hr"] == 172  # new value wins
    assert row3["lactate_threshold_speed"] == 0.5  # untouched by new value
    assert row3["ftp_watts"] == 300


def test_build_daily_rows_does_not_forward_fill_other_types(db: Database) -> None:
    """Non-sparse types keep NULL for genuinely missing days."""
    db.upsert_metric("heart_rate", {
        "calendarDate": "2026-08-01", "restingHeartRate": 62, "fetched_at": "t",
    })
    db.upsert_metric("heart_rate", {
        "calendarDate": "2026-08-02", "fetched_at": "t",
    })
    assert build_daily_rows(db, ["heart_rate"]) == {"heart_rate": 1}
    row = db.conn.execute(
        "SELECT 1 FROM daily_metrics WHERE calendar_date='2026-08-02'"
    ).fetchone()
    assert row is None


def test_parse_activity_summary_shares_curated_columns() -> None:
    out = parse_activity_summary({
        "activityId": 1001,
        "activityName": "Mountain Run",
        "activityType": {"typeKey": "running"},
        "startTimeLocal": "2026-08-01 07:00:00",
        "distance": 5000.0,
        "duration": 1800.0,
        "elapsedDuration": 2100.0,
        "movingDuration": 1790.0,
        "averageHR": 145,
        "maxHR": 170,
        "calories": 400.0,
        "hrTimeInZone_3": 600.0,
        "activityTrainingLoad": 120.5,
    })
    assert out["activity_name"] == "Mountain Run"
    assert out["activity_type"] == "running"
    assert out["start_date"] == "2026-08-01"
    assert out["distance_km"] == 5.0
    assert out["duration_hours"] == pytest.approx(0.5)
    assert out["elapsed_hours"] == pytest.approx(2100 / 3600, abs=1e-3)
    assert out["avg_hr"] == 145
    assert out["hr_time_zone_3_pct"] == pytest.approx(600 / 1800 * 100, abs=1e-1)
    assert out["training_load"] == 120.5


def test_parse_activity_summary_distance_hours_conversion() -> None:
    out = parse_activity_summary({
        "activityName": "Indoor Cycle",
        "activityType": {"typeKey": "indoor_cycling"},
        "distance": 0.0,
        "duration": 3600.0,
        "averageSpeed": 3.0,
    })
    # Indoor cycling has no distance; store 0.0 km and 1 hour duration.
    assert out["distance_km"] == 0.0
    assert out["duration_hours"] == pytest.approx(1.0)
    # Average speed in m/s converted to km/h.
    assert out["avg_speed_kmh"] == pytest.approx(3.0 * 3.6)


def test_build_activity_summaries_upserts_curated_row(db: Database) -> None:
    db.upsert_activity({
        "activityId": 1001,
        "activityName": "Morning Run",
        "activityType": {"typeKey": "running"},
        "startTimeLocal": "2026-01-01 07:00:00",
        "distance": 5000.0,
        "duration": 1800.0,
        "averageHR": 145,
        "maxHR": 170,
        "hrTimeInZone_2": 900.0,
        "fetched_at": "t",
    })
    counts = build_activity_summaries(db)
    assert counts == {"activities": 1}
    row = db.conn.execute(
        "SELECT * FROM activity_summaries WHERE activity_id=1001"
    ).fetchone()
    assert row["activity_name"] == "Morning Run"
    assert row["activity_type"] == "running"
    assert row["avg_hr"] == 145
    assert row["distance_km"] == pytest.approx(5.0)
    assert row["duration_hours"] == pytest.approx(0.5)
    assert row["hr_time_zone_2_pct"] == pytest.approx(900 / 1800 * 100)


def test_parse_activity_summary_extra_curated_fields() -> None:
    out = parse_activity_summary({
        "activityId": 1001,
        "activityName": "Hill Walk",
        "activityType": {"typeKey": "hiking"},
        "startTimeLocal": "2026-08-01 07:00:00",
        "elevationGain": 350.0,
        "elevationLoss": 120.0,
        "minElevation": 55.0,
        "maxElevation": 275.0,
        "averageSpeed": 3.0,
        "maxSpeed": 5.0,
        "minRespirationRate": 12.0,
        "avgRespirationRate": 18.0,
        "maxRespirationRate": 26.0,
        "differenceBodyBattery": -14.0,
        "waterEstimated": 1872.0,
        "isPR": True,
    })
    assert out["elevation_gain_m"] == 350.0
    assert out["elevation_loss_m"] == 120.0
    assert out["min_elevation_m"] == 55.0
    assert out["max_elevation_m"] == 275.0
    # Speeds are m/s -> km/h.
    assert out["avg_speed_kmh"] == pytest.approx(3.0 * 3.6)
    assert out["max_speed_kmh"] == pytest.approx(5.0 * 3.6)
    # Respiration rates pass through.
    assert out["min_respiration_rate"] == 12.0
    assert out["avg_respiration_rate"] == 18.0
    assert out["max_respiration_rate"] == 26.0
    # New curated fields.
    assert out["body_battery_change"] == -14.0
    assert out["water_estimated_ml"] == 1872.0
    assert out["is_pr"] is True
    # Removed: max_vertical_speed and lap_count are no longer parsed.
    assert "max_vertical_speed" not in out
    assert "lap_count" not in out


def _detail_payload() -> dict:
    """A running-style details payload with 3 ticks of aligned metrics."""
    return {
        "activityId": 1001,
        "metricDescriptors": [
            {"key": "directTimestamp", "unit": "ms"},
            {"key": "directHeartRate", "unit": "bpm"},
            {"key": "sumDistance", "unit": "meter"},
            {"key": "directRunCadence", "unit": "stepsPerMinute"},
            {"key": "directPower", "unit": "watt"},
            {"key": "directSpeed", "unit": "mps"},
            {"key": "directElevation", "unit": "meter"},
        ],
        "activityDetailMetrics": [
            {"metrics": [1780000000000.0, 150.0, 400.0, 84.0, 180.0, 3.1, 100.0]},
            {"metrics": [1780000001000.0, 165.0, 800.0, 88.0, 220.0, 3.4, 102.0]},
            {"metrics": [1780000002000.0, 172.0, 1200.0, 92.0, 260.0, 3.6, 105.0]},
        ],
    }


def test_parse_activity_details_aggregates() -> None:
    out = parse_activity_details(_detail_payload())
    assert out == {
        "avg_cadence": pytest.approx(88.0),
        "max_cadence": 92.0,
        "avg_power_w": pytest.approx(220.0),
        "max_power_w": 260.0,
    }


def test_parse_activity_details_skips_zero_and_nan() -> None:
    payload = {
        "metricDescriptors": [
            {"key": "directRunCadence"},
            {"key": "directPower"},
        ],
        "activityDetailMetrics": [
            {"metrics": [0.0, 100.0]},
            {"metrics": [84.0, 830.0]},
            {"metrics": [88.0, float("nan")]},
        ],
    }
    out = parse_activity_details(payload)
    assert out == {"avg_cadence": 86.0, "max_cadence": 88.0, "avg_power_w": 465.0, "max_power_w": 830.0}


def test_parse_activity_details_empty_payload() -> None:
    assert parse_activity_details({}) == {}
    assert parse_activity_details({"metricDescriptors": []}) == {}


def test_parse_activity_details_double_cadence_halved() -> None:
    """directDoubleCadence streams 2x spm; aggregate + series must halve it."""
    payload = {
        "metricDescriptors": [
            {"key": "directRunCadence"},
            {"key": "directDoubleCadence"},
        ],
        "activityDetailMetrics": [
            {"metrics": [float("nan"), 88.0]},
            {"metrics": [float("nan"), 92.0]},
        ],
    }
    out = parse_activity_details(payload)
    # directRunCadence present -> chosen first, NaN skipped entirely.
    assert out == {}
    agg = parse_activity_details({
        "metricDescriptors": [{"key": "directDoubleCadence"}],
        "activityDetailMetrics": [{"metrics": [176.0]}, {"metrics": [184.0]}],
    })
    assert agg == {"avg_cadence": 90.0, "max_cadence": 92.0}


def test_parse_activity_detail_series_rows() -> None:
    rows = parse_activity_detail_series(_detail_payload())
    assert len(rows) == 3
    assert [r["tick"] for r in rows] == [0, 1, 2]
    assert rows[0]["ts_ms"] == 1780000000000.0
    assert rows[0]["heart_rate"] == 150.0
    assert rows[0]["distance_m"] == 400.0
    assert rows[0]["cadence"] == 84.0
    assert rows[0]["power_w"] == 180.0
    assert rows[0]["elevation_m"] == 100.0
    # Every tick carries a full set of aligned metrics.
    assert set(rows[2]) == {
        "tick", "ts_ms", "heart_rate", "distance_m", "cadence",
        "power_w", "speed_mps", "elevation_m",
    }


def test_parse_activity_detail_series_ignores_unknown_descriptors() -> None:
    payload = {
        "metricDescriptors": [
            {"key": "directHeartRate"},
            {"key": "someFutureMetric"},
        ],
        "activityDetailMetrics": [
            {"metrics": [150.0, 42.0]},
        ],
    }
    rows = parse_activity_detail_series(payload)
    assert rows == [{"tick": 0, "heart_rate": 150.0}]


def test_parse_activity_detail_series_empty() -> None:
    assert parse_activity_detail_series({}) == []
    assert parse_activity_detail_series({"metricDescriptors": [{"key": "directHeartRate"}]}) == []


def test_build_activity_details_upserts_series_and_aggregates(db: Database) -> None:
    db.upsert_activity({
        "activityId": 1001,
        "activityName": "Track Run",
        "activityType": {"typeKey": "running"},
        "startTimeLocal": "2026-08-01 07:00:00",
        "distance": 5000.0,
        "duration": 1800.0,
        "fetched_at": "t",
    })
    db.set_activity_details(1001, _detail_payload())
    counts = build_activity_details(db)
    assert counts == {"activities": 1, "series": 3}
    summary = db.conn.execute(
        "SELECT avg_cadence, max_cadence, avg_power_w, max_power_w "
        "FROM activity_summaries WHERE activity_id=1001"
    ).fetchone()
    assert summary["avg_cadence"] == pytest.approx(88.0)
    assert summary["max_power_w"] == 260.0
    ticks = db.conn.execute(
        "SELECT * FROM activity_detail_series WHERE activity_id=1001 ORDER BY tick"
    ).fetchall()
    assert len(ticks) == 3
    assert ticks[0]["heart_rate"] == 150.0
    assert ticks[2]["power_w"] == 260.0


def test_build_activity_details_replaces_previous_series(db: Database) -> None:
    db.upsert_activity({
        "activityId": 1001,
        "activityName": "Track Run",
        "activityType": {"typeKey": "running"},
        "fetched_at": "t",
    })
    db.set_activity_details(1001, _detail_payload())
    build_activity_details(db)
    first = db.conn.execute(
        "SELECT COUNT(*) AS n FROM activity_detail_series WHERE activity_id=1001"
    ).fetchone()["n"]
    assert first == 3
    # Re-parse with fewer ticks: old ticks must not survive.
    payload = {
        "metricDescriptors": [{"key": "directHeartRate"}],
        "activityDetailMetrics": [{"metrics": [150.0]}],
    }
    db.set_activity_details(1001, payload)
    build_activity_details(db)
    remaining = db.conn.execute(
        "SELECT COUNT(*) AS n FROM activity_detail_series WHERE activity_id=1001"
    ).fetchone()["n"]
    assert remaining == 1


def test_build_activity_details_skips_activities_without_details(db: Database) -> None:
    db.upsert_activity({
        "activityId": 2002,
        "activityName": "Manual",
        "activityType": {"typeKey": "other"},
        "fetched_at": "t",
    })
    assert build_activity_details(db) == {"activities": 0, "series": 0}


def test_parse_hr_zones_derives_ranges() -> None:
    rows = parse_hr_zones([{
        "trainingMethod": "HR_RESERVE",
        "restingHeartRateUsed": 54,
        "lactateThresholdHeartRateUsed": 166,
        "zone1Floor": 121,
        "zone2Floor": 134,
        "zone3Floor": 148,
        "zone4Floor": 161,
        "zone5Floor": 175,
        "maxHeartRateUsed": 188,
        "restingHrAutoUpdateUsed": True,
        "sport": "DEFAULT",
        "changeState": "UNCHANGED",
    }])
    assert len(rows) == 1
    row = rows[0]
    assert row["sport"] == "DEFAULT"
    assert row["training_method"] == "HR_RESERVE"
    assert row["max_hr_used"] == 188
    assert row["resting_hr_used"] == 54
    # Zone N spans [floor_N, floor_(N+1)); zone 5 runs to the max HR used.
    assert (row["zone1_min"], row["zone1_max"]) == (121, 133)
    assert (row["zone2_min"], row["zone2_max"]) == (134, 147)
    assert (row["zone3_min"], row["zone3_max"]) == (148, 160)
    assert (row["zone4_min"], row["zone4_max"]) == (161, 174)
    assert (row["zone5_min"], row["zone5_max"]) == (175, 188)


def test_parse_hr_zones_multiple_sports_and_skips_bad_rows() -> None:
    rows = parse_hr_zones([
        {"sport": "RUNNING", "zone1Floor": 120, "zone2Floor": 130,
         "zone3Floor": 140, "zone4Floor": 150, "zone5Floor": 160,
         "maxHeartRateUsed": 190},
        {"sport": "CYCLING", "zone1Floor": 110, "zone2Floor": 125,
         "zone3Floor": 140, "zone4Floor": 155, "zone5Floor": 170,
         "maxHeartRateUsed": 200},
        {"sport": None},
        "not-a-dict",
    ])
    assert [r["sport"] for r in rows] == ["RUNNING", "CYCLING"]
    assert (rows[0]["zone2_min"], rows[0]["zone2_max"]) == (130, 139)
    assert (rows[1]["zone5_min"], rows[1]["zone5_max"]) == (170, 200)


def test_build_hr_zones_projects_profile(db: Database) -> None:
    db.upsert_profile("hr_zones", json.dumps([
        {"sport": "DEFAULT", "trainingMethod": "HR_RESERVE",
         "zone1Floor": 121, "zone2Floor": 134, "zone3Floor": 148,
         "zone4Floor": 161, "zone5Floor": 175, "maxHeartRateUsed": 188,
         "restingHeartRateUsed": 54, "lactateThresholdHeartRateUsed": 166,
         "restingHrAutoUpdateUsed": True},
        {"sport": "RUNNING", "zone1Floor": 120, "zone2Floor": 130,
         "zone3Floor": 140, "zone4Floor": 150, "zone5Floor": 160,
         "maxHeartRateUsed": 190},
    ]), fetched_at="t")
    assert build_hr_zones(db) == {"hr_zones": 2}
    row = db.conn.execute(
        "SELECT * FROM hr_zones WHERE sport='DEFAULT'"
    ).fetchone()
    assert row["zone2_min"] == 134
    assert row["zone2_max"] == 147
    assert row["training_method"] == "HR_RESERVE"
    assert row["fetched_at"] == "t"
    run_row = db.conn.execute(
        "SELECT * FROM hr_zones WHERE sport='RUNNING'"
    ).fetchone()
    assert run_row["zone5_max"] == 190


def test_build_hr_zones_no_profile_is_empty(db: Database) -> None:
    assert build_hr_zones(db) == {"hr_zones": 0}
    assert db.conn.execute("SELECT COUNT(*) AS n FROM hr_zones").fetchone()["n"] == 0


def test_build_hr_zones_replaces_previous_snapshot(db: Database) -> None:
    db.upsert_profile("hr_zones", json.dumps([
        {"sport": "DEFAULT", "zone1Floor": 100, "zone2Floor": 110,
         "zone3Floor": 120, "zone4Floor": 130, "zone5Floor": 140,
         "maxHeartRateUsed": 150},
    ]), fetched_at="t1")
    build_hr_zones(db)
    assert db.conn.execute("SELECT COUNT(*) AS n FROM hr_zones").fetchone()["n"] == 1

    db.upsert_profile("hr_zones", json.dumps([
        {"sport": "DEFAULT", "zone1Floor": 100, "zone2Floor": 110,
         "zone3Floor": 120, "zone4Floor": 130, "zone5Floor": 140,
         "maxHeartRateUsed": 150},
        {"sport": "CYCLING", "zone1Floor": 90, "zone2Floor": 100,
         "zone3Floor": 110, "zone4Floor": 120, "zone5Floor": 130,
         "maxHeartRateUsed": 160},
    ]), fetched_at="t2")
    build_hr_zones(db)
    sports = [
        r["sport"]
        for r in db.conn.execute("SELECT sport FROM hr_zones ORDER BY sport")
    ]
    assert sports == ["CYCLING", "DEFAULT"]


def test_parse_race_predictions_to_minutes() -> None:
    out = parse_race_predictions({
        "calendarDate": "2026-08-08",
        "time5K": 1424,
        "time10K": 3093,
        "timeHalfMarathon": 7008,
        "timeMarathon": 15825,
    })
    assert out == {
        "calendar_date": "2026-08-08",
        "time_5k_min": pytest.approx(1424 / 60, abs=0.01),
        "time_10k_min": pytest.approx(3093 / 60, abs=0.01),
        "time_half_marathon_min": pytest.approx(7008 / 60, abs=0.01),
        "time_marathon_min": pytest.approx(15825 / 60, abs=0.01),
    }


def test_parse_race_predictions_empty_on_no_times() -> None:
    assert parse_race_predictions({"calendarDate": "2026-08-08"}) == {}
    assert parse_race_predictions({}) == {}


def test_build_race_predictions_projects_snapshot(db: Database) -> None:
    db.upsert_profile("race_predictions", json.dumps({
        "calendarDate": "2026-08-08",
        "time5K": 1424,
        "time10K": 3093,
        "timeHalfMarathon": 7008,
        "timeMarathon": 15825,
    }), fetched_at="t")
    assert build_race_predictions(db) == {"race_predictions": 1}
    row = db.conn.execute("SELECT * FROM race_predictions").fetchone()
    assert row["time_5k_min"] == pytest.approx(1424 / 60, abs=0.01)
    assert row["time_marathon_min"] == pytest.approx(15825 / 60, abs=0.01)
    assert row["fetched_at"] == "t"


def test_build_race_predictions_replaces_and_empty(db: Database) -> None:
    assert build_race_predictions(db) == {"race_predictions": 0}
    db.upsert_profile("race_predictions", json.dumps({
        "calendarDate": "2026-08-08", "time5K": 1424,
    }), fetched_at="t1")
    build_race_predictions(db)
    db.upsert_profile("race_predictions", json.dumps({
        "calendarDate": "2026-08-09", "time5K": 1400,
    }), fetched_at="t2")
    build_race_predictions(db)
    rows = db.conn.execute("SELECT * FROM race_predictions").fetchall()
    assert len(rows) == 1
    assert rows[0]["calendar_date"] == "2026-08-09"
    assert rows[0]["time_5k_min"] == pytest.approx(1400 / 60, abs=0.01)


def test_parse_activity_weather_converts_to_metric() -> None:
    out = parse_activity_weather({
        "temp": 68.0,
        "apparentTemp": 66.0,
        "relativeHumidity": 60,
        "windSpeed": 10.0,
        "windGust": 15.0,
        "weatherStationDTO": {"id": "LZIB", "name": "Bratislava Ivanka"},
        "weatherTypeDTO": {"desc": "Fair"},
    })
    assert out["weather_temp_c"] == pytest.approx((68 - 32) * 5 / 9, rel=1e-2)
    assert out["weather_apparent_c"] == pytest.approx((66 - 32) * 5 / 9, rel=1e-2)
    assert out["weather_humidity"] == 60
    assert out["weather_wind_kmh"] == pytest.approx(10 * 1.609344, rel=1e-2)
    assert out["weather_wind_gust_kmh"] == pytest.approx(15 * 1.609344, rel=1e-2)
    assert out["weather_station"] == "Bratislava Ivanka"
    assert out["weather_description"] == "Fair"


def test_parse_activity_weather_empty_on_blank() -> None:
    assert parse_activity_weather({}) == {}
    assert parse_activity_weather({"temp": "not-a-number"}) == {}


def test_build_activity_weather_projects_summary_columns(db: Database) -> None:
    db.upsert_activity({
        "activityId": 1001,
        "activityName": "Morning Run",
        "activityType": {"typeKey": "running"},
        "fetched_at": "t",
    })
    db.set_activity_weather(1001, {
        "temp": 68.0,
        "relativeHumidity": 60,
        "windSpeed": 10.0,
        "weatherStationDTO": {"name": "LZIB"},
        "weatherTypeDTO": {"desc": "Fair"},
    }, fetched_at="wt")
    assert build_activity_weather(db) == {"activities": 1}
    row = db.conn.execute(
        "SELECT * FROM activity_summaries WHERE activity_id=1001"
    ).fetchone()
    assert row["weather_temp_c"] == pytest.approx((68 - 32) * 5 / 9, rel=1e-2)
    assert row["weather_wind_kmh"] == pytest.approx(10 * 1.609344, rel=1e-2)
    assert row["weather_station"] == "LZIB"
    assert row["weather_description"] == "Fair"
    assert row["fetched_at"] == "wt"


def test_build_activity_weather_ignores_missing(db: Database) -> None:
    db.upsert_activity({
        "activityId": 1002,
        "activityName": "No Weather",
        "activityType": {"typeKey": "cycling"},
        "fetched_at": "t",
    })
    assert build_activity_weather(db) == {"activities": 0}