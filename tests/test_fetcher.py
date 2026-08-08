from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pytest

from garmin_fetch.datatypes import DataType, DEFAULT_TYPES, resolve_types
from garmin_fetch.db import Database
from garmin_fetch.fetcher import (
    DataFetcher,
    _configured_types,
    _fetch_local_date,
    _nonfinal_dates,
    _resolve_activity_start,
    _resolve_start_date,
    _rolling_start,
    mark_activities_finalized,
)


class FakeClient:
    """Minimal stand-in for GarminConnect."""

    def __init__(self) -> None:
        self.logged_in = False
        self.calls: list[tuple[str, str]] = []

    def login(self, tokens_path=None) -> None:
        self.logged_in = True

    def get_heart_rates(self, cdate):
        self.calls.append(("heart_rate", cdate))
        return {
            "calendarDate": cdate,
            "restingHeartRate": 55,
            "maxHeartRate": 130,
            "heartRateZones": [
                {"zoneNumber": 1, "min": 0, "max": 114, "secondsInZone": 43200},
                {"zoneNumber": 2, "min": 115, "max": 130, "secondsInZone": 43200},
            ],
        }

    def get_steps_data(self, cdate):
        self.calls.append(("steps", cdate))
        return {"calendarDate": cdate, "totalSteps": 8000, "totalCalories": 200.0}

    def get_sleep_data(self, cdate):
        self.calls.append(("sleep", cdate))
        return {
            "dailySleepDTO": {
                "calendarDate": cdate,
                "sleepTimeSeconds": 28800,
                "deepSleepDuration": [{"durationInSeconds": 3600}],
                "lightSleepDuration": [{"durationInSeconds": 7200}],
                "remSleepDuration": [{"durationInSeconds": 5400}],
                "awakeSleepDuration": [{"durationInSeconds": 600}],
            },
            "sleepScoreOverall": 80,
        }

    def get_body_battery_events(self, cdate):
        self.calls.append(("body_battery", cdate))
        return [{"eventDate": cdate, "value": 75}]

    def get_hrv_data(self, cdate):
        self.calls.append(("hrv", cdate))
        return {"calendarDate": cdate, "dailyRmssd": 45, "lastNightAvg": 42}

    def get_all_day_stress(self, cdate):
        self.calls.append(("stress", cdate))
        return {"calendarDate": cdate, "stressQualifier": "balanced"}

    def get_respiration_data(self, cdate):
        self.calls.append(("respiration", cdate))
        return {"calendarDate": cdate, "respiratoryRate": 14.2}

    def get_spo2_data(self, cdate):
        self.calls.append(("spo2", cdate))
        return {"calendarDate": cdate, "avgSpo2": 97}

    def get_rhr_day(self, cdate):
        self.calls.append(("rhr", cdate))
        return {"calendarDate": cdate, "restingHeartRate": 52}

    def get_stats(self, cdate):
        self.calls.append(("stats", cdate))
        return {"calendarDate": cdate, "totalCalories": 2200.0}

    def get_intensity_minutes_data(self, cdate):
        self.calls.append(("intensity_minutes", cdate))
        return {"calendarDate": cdate, "intensityMinutes": 45}

    def get_floors(self, cdate):
        self.calls.append(("floors", cdate))
        return {"calendarDate": cdate, "floorsClimbed": 12.0}

    def get_max_metrics(self, cdate):
        self.calls.append(("max_metrics", cdate))
        return {"calendarDate": cdate, "maxHeartRate": 178}

    def get_training_readiness(self, cdate):
        self.calls.append(("training_readiness", cdate))
        return [
            {"date": cdate, "readinessScore": 85},
            {"date": cdate, "readinessScore": 90},
        ]

    def get_morning_training_readiness(self, cdate):
        self.calls.append(("morning_training_readiness", cdate))
        return {"calendarDate": cdate, "readinessScore": 88}

    def get_fitnessage_data(self, cdate):
        self.calls.append(("fitnessage", cdate))
        return {"calendarDate": cdate, "fitnessAge": 25}

    def get_hydration_data(self, cdate):
        self.calls.append(("hydration", cdate))
        return {"calendarDate": cdate, "waterIntakeInLiters": 2.5}

    def get_training_status(self, cdate):
        self.calls.append(("training_status", cdate))
        return {"calendarDate": cdate, "trainingStatus": "productive"}

    def get_lactate_threshold(self, *, latest=None, start_date=None, end_date=None,
                              aggregation=None):
        self.calls.append(("lactate_threshold", f"{start_date}..{end_date}"))
        return {
            "heart_rate": [{"value": 165, "calendarDate": start_date}],
            "speed": [{"value": 0.353, "calendarDate": start_date}],
            "power": [{"value": 300}],
        }

    def get_heart_rate_zones(self):
        self.calls.append(("hr_zones", ""))
        return [
            {
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
            },
            {
                "sport": "RUNNING",
                "zone1Floor": 120,
                "zone2Floor": 130,
                "zone3Floor": 140,
                "zone4Floor": 150,
                "zone5Floor": 160,
                "maxHeartRateUsed": 190,
            },
        ]

    def get_activities_by_date(self, startdate, enddate=None, activitytype=None, sortorder=None):
        self.calls.append(("activities", f"{startdate}..{enddate}"))
        return [
            {
                "activityId": 1001,
                "activityName": "Morning Run",
                "activityType": {"typeKey": "running"},
                "sportType": {"typeKey": "running"},
                "startTimeLocal": "2026-08-01 07:00:00",
                "duration": 1800.0,
                "distance": 5000.0,
                "averageHR": 145,
                "maxHR": 170,
                "calories": 350.0,
            },
            {
                "activityId": 1002,
                "activityName": "Evening Ride",
                "activityType": {"typeKey": "cycling"},
                "sportType": {"typeKey": "cycling"},
                "startTimeLocal": "2026-08-01 18:00:00",
                "duration": 5400.0,
                "distance": 30000.0,
                "averageHR": 120,
                "calories": 600.0,
            },
        ]

    def get_activity_details(self, activity_id):
        self.calls.append(("activity_details", str(activity_id)))
        return {
            "activityId": int(activity_id),
            "metricDescriptors": [{"key": "heart_rate", "unit": "bpm"}],
            "activityDetailMetrics": [],
            "heartRateDTOs": [{"heartRateValues": [80, 90, 100]}],
        }


def make_fetcher(client: FakeClient) -> DataFetcher:
    fetcher = DataFetcher("e", "p")
    fetcher.client = client
    fetcher.logged_in = True
    return fetcher


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def test_resolve_types_defaults() -> None:
    assert resolve_types(None) == list(DEFAULT_TYPES)


def test_resolve_types_subset() -> None:
    names = [d.name for d in resolve_types(["heart_rate", "sleep"])]
    assert names == ["heart_rate", "sleep"]


def test_resolve_types_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_types(["not_a_thing"])


def test_configured_types_empty_returns_all() -> None:
    types = _configured_types({"data_types": ""})
    assert types == list(DEFAULT_TYPES)


def test_configured_types_subset() -> None:
    types = _configured_types({"data_types": "heart_rate, hrv ,steps"})
    assert [t.name for t in types] == ["heart_rate", "hrv", "steps"]


def test_configured_types_excluded_removed() -> None:
    types = _configured_types(
        {"data_types": "", "excluded_data_types": "training_readiness, hrv"}
    )
    names = [t.name for t in types]
    assert "training_readiness" not in names
    assert "hrv" not in names
    assert "heart_rate" in names


def test_configured_types_excluded_applies_to_explicit() -> None:
    types = _configured_types(
        {"data_types": "hrv,steps", "excluded_data_types": "steps"}
    )
    assert [t.name for t in types] == ["hrv"]


def test_configured_types_all_excluded_returns_all() -> None:
    excluded = ",".join(d.name for d in DEFAULT_TYPES)
    types = _configured_types({"data_types": "", "excluded_data_types": excluded})
    assert types == list(DEFAULT_TYPES)


def test_configured_types_skips_unknown() -> None:
    types = _configured_types({"data_types": "heart_rate,bogus,hrv"})
    assert [t.name for t in types] == ["heart_rate", "hrv"]


def test_configured_types_all_unknown_returns_all() -> None:
    types = _configured_types({"data_types": "bogus,also_bogus"})
    assert types == list(DEFAULT_TYPES)


def test_all_registered_types_resolve() -> None:
    names = [d.name for d in DEFAULT_TYPES]
    assert resolve_types(names) == list(DEFAULT_TYPES)
    # Every registered type has a distinct fetch adapter backed by a client method.
    for data_type in DEFAULT_TYPES:
        client = FakeClient()
        result = data_type.fetch(client, "2026-08-01")
        assert result is not None


def test_none_response_is_stored_as_fetched(client: FakeClient, db: Database) -> None:
    """A day with no data (e.g. HRV None) is recorded so it isn't refetched."""
    def fetch_hrv_none(c, cdate):
        return None

    none_type = DataType("hrv", fetch_hrv_none)
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 1)

    assert fetcher.fetch_range(none_type, start, end, db) == 1
    assert db.stored_dates("hrv") == {"2026-08-01"}
    row = db.conn.execute(
        "SELECT raw_json FROM metrics WHERE data_type='hrv'"
    ).fetchone()
    assert json.loads(row["raw_json"])["value"] is None


def test_resolve_start_date_config_wins(client, db) -> None:
    assert _resolve_start_date(db, "steps", date(2021, 1, 1)) == date(2021, 1, 1)


def test_resolve_start_date_empty_returns_today(client, db) -> None:
    assert _resolve_start_date(db, "steps", None) == date.today()


def test_sync_all_types(client: FakeClient, db: Database) -> None:
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 2)

    for data_type in DEFAULT_TYPES:
        assert fetcher.fetch_range(data_type, start, end, db) == 2

    # Raw rows exist for every type/date.
    for name in [d.name for d in DEFAULT_TYPES]:
        assert db.stored_dates(name) == {"2026-08-01", "2026-08-02"}

    row = db.conn.execute(
        "SELECT * FROM metrics WHERE data_type='heart_rate' AND calendar_date='2026-08-01'"
    ).fetchone()
    assert json.loads(row["raw_json"])["restingHeartRate"] == 55


def test_sync_is_idempotent(client: FakeClient, db: Database) -> None:
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 3)

    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 3
    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 0
    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 0
    assert db.stored_dates("heart_rate") == {"2026-08-01", "2026-08-02", "2026-08-03"}
    # Garmin API only called once per date.
    assert len(client.calls) == 3


def test_fetch_range_refetches_incomplete_days(
    client: FakeClient, db: Database
) -> None:
    """Stored days marked incomplete are re-fetched on a rerun."""
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 7)
    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 7

    # Same range again, only the two stale days in the refetch set re-hit.
    n = fetcher.fetch_range(
        DEFAULT_TYPES[0], start, end, db, refetch={"2026-08-06", "2026-08-07"}
    )
    assert n == 2
    heart_calls = [d for t, d in client.calls if t == "heart_rate"]
    assert heart_calls.count("2026-08-01") == 1
    assert heart_calls.count("2026-08-06") == 2
    assert heart_calls.count("2026-08-07") == 2


def test_fetch_range_refetches_overwrites_stale_data(
    client: FakeClient, db: Database
) -> None:
    """A day fetched while still in progress gets its settled data overwritten."""
    fetcher = make_fetcher(client)
    day = date(2026, 8, 5)
    assert fetcher.fetch_range(DEFAULT_TYPES[0], day, day, db) == 1

    # Garmin has since finalized the day: later values are returned.
    client.get_heart_rates = lambda cdate: {
        "calendarDate": cdate,
        "restingHeartRate": 57,
        "maxHeartRate": 132,
        "heartRateZones": [],
    }
    fetched = fetcher.fetch_range(
        DEFAULT_TYPES[0], day, day, db, refetch={"2026-08-05"}
    )
    assert fetched == 1
    row = db.conn.execute(
        "SELECT raw_json FROM metrics WHERE data_type='heart_rate' "
        "AND calendar_date='2026-08-05'"
    ).fetchone()
    assert json.loads(row["raw_json"])["restingHeartRate"] == 57


def test_fetch_range_skips_stored_days_without_refetch(
    client: FakeClient, db: Database
) -> None:
    """Without a refetch set, every stored day is skipped (pure incremental)."""
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 3)
    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 3

    assert fetcher.fetch_range(DEFAULT_TYPES[0], start, end, db) == 0
    heart_calls = [d for t, d in client.calls if t == "heart_rate"]
    assert heart_calls.count("2026-08-01") == 1


def test_fetch_local_date_naive_is_local() -> None:
    assert _fetch_local_date("2026-08-07T22:30:00") == date(2026, 8, 7)
    assert _fetch_local_date("2026-08-08T00:30:00") == date(2026, 8, 8)


def test_nonfinal_dates_marks_days_fetched_while_in_progress() -> None:
    """A copy captured on (or before) its own day is incomplete; later is final."""
    stored = {
        # Fetched the day after the data -> settled.
        "2026-08-05": "2026-08-06T09:00:00",
        # Fetched during the data's own day -> still in progress.
        "2026-08-06": "2026-08-06T15:00:00",
        # Captured just after midnight of the next day -> settled.
        "2026-08-07": "2026-08-08T00:30:00",
    }
    assert _nonfinal_dates(stored) == {"2026-08-06"}


def test_resolve_start_date_uses_oldest_incomplete_day(
    client: FakeClient, db: Database
) -> None:
    """No configured start: resume from the oldest still-incomplete day."""
    fetcher = make_fetcher(client)
    fetcher.fetch_range(DEFAULT_TYPES[0], date(2026, 8, 1), date(2026, 8, 7), db)

    start = _resolve_start_date(
        db, "heart_rate", None, nonfinal={"2026-08-03", "2026-08-07"}
    )
    assert start == date(2026, 8, 3)


def test_resolve_start_date_no_incomplete_uses_max_plus_one(
    client: FakeClient, db: Database
) -> None:
    """No incomplete stored day -> normal incremental start wins."""
    fetcher = make_fetcher(client)
    fetcher.fetch_range(DEFAULT_TYPES[0], date(2026, 8, 1), date(2026, 8, 3), db)

    start = _resolve_start_date(db, "heart_rate", None, nonfinal=set())
    assert start == date(2026, 8, 4)


def test_sync_continues_from_last_incomplete_day(
    client: FakeClient, db: Database
) -> None:
    """A day captured mid-day is stamped same-day, so the next sync resumes at it."""
    db.upsert_metric("heart_rate", {
        "calendarDate": "2026-08-05", "restingHeartRate": 52,
        "fetched_at": "2026-08-06T09:00:00",
    })
    db.upsert_metric("heart_rate", {
        "calendarDate": "2026-08-06", "restingHeartRate": 53,
        "fetched_at": "2026-08-06T15:00:00",
    })

    nonfinal = _nonfinal_dates(db.stored_fetches("heart_rate"))
    assert nonfinal == {"2026-08-06"}

    start = _resolve_start_date(db, "heart_rate", None, nonfinal or None)
    assert start == date(2026, 8, 6)

    fetcher = make_fetcher(client)
    assert fetcher.fetch_range(
        DEFAULT_TYPES[0], start, date(2026, 8, 6), db, nonfinal
    ) == 1
    heart_calls = [d for t, d in client.calls if t == "heart_rate"]
    assert "2026-08-05" not in heart_calls
    assert heart_calls == ["2026-08-06"]

    # After the refresh the captured-at is a later day -> now final.
    assert _nonfinal_dates(db.stored_fetches("heart_rate")) == set()


def test_custom_data_type_is_registerable(client: FakeClient, db: Database) -> None:
    def fetch_hrv(c, cdate):
        return {"calendarDate": cdate, "rmssd": 42}

    hrv = DataType("hrv", fetch_hrv)
    fetcher = make_fetcher(client)
    start, end = date(2026, 8, 1), date(2026, 8, 1)

    assert fetcher.fetch_range(hrv, start, end, db) == 1
    assert db.stored_dates("hrv") == {"2026-08-01"}
    raw = db.conn.execute(
        "SELECT raw_json FROM metrics WHERE data_type='hrv'"
    ).fetchone()
    assert json.loads(raw["raw_json"])["rmssd"] == 42


def test_fetch_activities_stores_raw(client: FakeClient, db: Database) -> None:
    fetcher = make_fetcher(client)
    assert fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db) == 2
    assert db.stored_activity_ids() == {1001, 1002}

    row = db.conn.execute(
        "SELECT * FROM activities WHERE activity_id=1001"
    ).fetchone()
    # Raw-only: no typed projection columns, just the raw payload + details.
    assert list(row.keys()) == [
        "activity_id", "raw_json", "fetched_at", "details_json", "details_fetched_at"
    ]
    assert json.loads(row["raw_json"])["activityId"] == 1001
    assert json.loads(row["raw_json"])["activityName"] == "Morning Run"
    assert json.loads(row["details_json"])["activityId"] == 1001
    assert json.loads(row["details_json"])["heartRateDTOs"][0]["heartRateValues"] == [80, 90, 100]


def test_fetch_activities_details_dont_duplicate_calls(
    client: FakeClient, db: Database
) -> None:
    """Details are fetched once per activity: summary call + 2 detail calls."""
    fetcher = make_fetcher(client)
    fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db)
    fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db)
    assert db.stored_activity_ids() == {1001, 1002}
    detail_calls = [arg for t, arg in client.calls if t == "activity_details"]
    assert detail_calls == ["1001", "1002"]


def test_activity_details_columns_added_to_existing_table(tmp_path: Path) -> None:
    """Opening a DB whose activities table lacks details columns migrates them."""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE activities (activity_id INTEGER PRIMARY KEY, "
        "raw_json TEXT NOT NULL, fetched_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    cols = [r["name"] for r in db.conn.execute("PRAGMA table_info(activities)")]
    assert "details_json" in cols
    assert "details_fetched_at" in cols


def test_fetch_activities_is_idempotent(client: FakeClient, db: Database) -> None:
    fetcher = make_fetcher(client)
    window = (date(2026, 8, 1), date(2026, 8, 1))
    assert fetcher.fetch_activities(*window, db) == 2
    assert fetcher.fetch_activities(*window, db) == 0
    assert db.stored_activity_ids() == {1001, 1002}
    # Still makes the range API call each run (list not cached), but skips
    # re-storing ids it already has.
    assert sum(1 for t, _ in client.calls if t == "activities") == 2
    assert db.conn.execute("SELECT COUNT(*) AS n FROM activities").fetchone()["n"] == 2


def test_resolve_activity_start_no_marker_no_config_uses_rolling(
    client: FakeClient, db: Database
) -> None:
    fetcher = make_fetcher(client)
    # Stored activities but no marker/config: still just the rolling window,
    # since activities are raw-only and resume relies on the marker.
    fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db)
    start = _resolve_activity_start(db, None, 7, date(2026, 8, 7))
    assert start == date(2026, 8, 1)


def test_rolling_start_no_finalized() -> None:
    end = date(2026, 8, 7)
    assert _rolling_start(end, 7, None) == date(2026, 8, 1)


def test_rolling_start_respects_finalized() -> None:
    end = date(2026, 8, 7)
    finalized = date(2026, 8, 5)
    assert _rolling_start(end, 7, finalized) == date(2026, 8, 6)


def test_rolling_start_finalized_older_than_buffer() -> None:
    end = date(2026, 8, 7)
    finalized = date(2026, 7, 1)
    assert _rolling_start(end, 7, finalized) == date(2026, 8, 1)


def test_mark_activities_finalized_sets_state(db: Database) -> None:
    mark_activities_finalized(db, date(2026, 8, 7), 7)
    assert db.get_state("activities_last_finalized") == "2026-07-31"


def test_mark_activities_finalized_zero_freezes_nothing(db: Database) -> None:
    mark_activities_finalized(db, date(2026, 8, 7), 0)
    assert db.get_state("activities_last_finalized") is None


def test_resolve_activity_start_config_backfill(db: Database) -> None:
    """First run with GARMIN_START_DATE: full backfill from that date."""
    start = _resolve_activity_start(db, date(2025, 1, 1), 7, date(2026, 8, 7))
    assert start == date(2025, 1, 1)


def test_resolve_activity_start_after_finalized_is_rolling(db: Database) -> None:
    """Once finalized, window is max(configured_start, finalized+1..)."""
    db.set_state("activities_last_finalized", "2026-08-05")
    start = _resolve_activity_start(db, date(2025, 1, 1), 7, date(2026, 8, 7))
    assert start == date(2026, 8, 6)


def test_resolve_activity_start_no_marker_no_config_no_stored(db: Database) -> None:
    """Fresh DB, no marker, no stored activities: rolling window only."""
    start = _resolve_activity_start(db, None, 7, date(2026, 8, 7))
    assert start == date(2026, 8, 1)


def test_backfill_details_fills_missing_only(client: FakeClient, db: Database) -> None:
    """Activities fetched are stored with both summary and details."""
    fetcher = make_fetcher(client)
    assert fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db) == 2
    for activity_id in (1001, 1002):
        row = db.conn.execute(
            "SELECT details_json FROM activities WHERE activity_id=?", (activity_id,)
        ).fetchone()
        assert json.loads(row["details_json"])["activityId"] == activity_id


def test_fetch_activity_skipped_when_no_details(client: FakeClient, db: Database) -> None:
    """An activity whose details fetch fails is not stored at all."""
    def get_activity_details(activity_id):
        if int(activity_id) == 1002:
            raise RuntimeError("boom")
        return {"activityId": int(activity_id)}

    client.get_activity_details = get_activity_details
    fetcher = make_fetcher(client)

    assert fetcher.fetch_activities(date(2026, 8, 1), date(2026, 8, 1), db) == 1
    assert db.stored_activity_ids() == {1001}


def test_backfill_details_idempotent(client: FakeClient, db: Database) -> None:
    """Re-running the fetch makes no additional detail calls for stored ids."""
    window = (date(2026, 8, 1), date(2026, 8, 1))
    fetcher = make_fetcher(client)
    fetcher.fetch_activities(*window, db)
    fetcher.fetch_activities(*window, db)
    detail_calls = [arg for t, arg in client.calls if t == "activity_details"]
    assert detail_calls == ["1001", "1002"]


def test_fetch_profile_stores_hr_zones_raw(client: FakeClient, db: Database) -> None:
    fetcher = make_fetcher(client)
    assert fetcher.fetch_profile(db) is True

    profile = db.get_profile("hr_zones")
    assert profile is not None
    payload = json.loads(profile["raw_json"])
    assert [p["sport"] for p in payload] == ["DEFAULT", "RUNNING"]
    assert payload[0]["zone2Floor"] == 134
    assert profile["fetched_at"]


def test_fetch_profile_overwrites_previous_snapshot(
    client: FakeClient, db: Database
) -> None:
    fetcher = make_fetcher(client)
    fetcher.fetch_profile(db)
    client.get_heart_rate_zones = lambda: [
        {"sport": "DEFAULT", "zone1Floor": 99, "zone2Floor": 110,
         "zone3Floor": 121, "zone4Floor": 132, "zone5Floor": 143,
         "maxHeartRateUsed": 160}
    ]
    assert fetcher.fetch_profile(db) is True
    payload = json.loads(db.get_profile("hr_zones")["raw_json"])
    assert len(payload) == 1
    assert payload[0]["sport"] == "DEFAULT"


def test_fetch_profile_none_skips(client: FakeClient, db: Database) -> None:
    client.get_heart_rate_zones = lambda: None
    fetcher = make_fetcher(client)
    assert fetcher.fetch_profile(db) is False
    assert db.get_profile("hr_zones") is None
