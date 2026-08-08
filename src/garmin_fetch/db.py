from __future__ import annotations

import json
import sqlite3
from typing import Any

from .config import load_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    data_type TEXT NOT NULL,
    calendar_date TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (data_type, calendar_date)
);

CREATE TABLE IF NOT EXISTS activities (
    activity_id INTEGER PRIMARY KEY,
    raw_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    details_json TEXT,
    details_fetched_at TEXT,
    weather_json TEXT,
    weather_fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    calendar_date TEXT PRIMARY KEY,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS activity_summaries (
    activity_id INTEGER PRIMARY KEY,
    activity_name TEXT,
    activity_type TEXT,
    start_time_local TEXT,
    start_time_gmt TEXT,
    start_date TEXT,
    duration_hours REAL,
    elapsed_hours REAL,
    moving_hours REAL,
    distance_km REAL,
    avg_hr REAL,
    max_hr REAL,
    hr_time_zone_1_pct REAL,
    hr_time_zone_2_pct REAL,
    hr_time_zone_3_pct REAL,
    hr_time_zone_4_pct REAL,
    hr_time_zone_5_pct REAL,
    calories REAL,
    training_load REAL,
    aerobic_training_effect REAL,
    anaerobic_training_effect REAL,
    moderate_intensity_minutes REAL,
    vigorous_intensity_minutes REAL,
    elevation_gain_m REAL,
    elevation_loss_m REAL,
    min_elevation_m REAL,
    max_elevation_m REAL,
    avg_speed_kmh REAL,
    max_speed_kmh REAL,
    min_respiration_rate REAL,
    avg_respiration_rate REAL,
    max_respiration_rate REAL,
    body_battery_change REAL,
    water_estimated_ml REAL,
    is_pr INTEGER,
    avg_cadence REAL,
    max_cadence REAL,
    avg_power_w REAL,
    max_power_w REAL,
    weather_temp_c REAL,
    weather_apparent_c REAL,
    weather_humidity REAL,
    weather_wind_kmh REAL,
    weather_wind_gust_kmh REAL,
    weather_station TEXT,
    weather_description TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS activity_detail_series (
    activity_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    ts_ms INTEGER,
    heart_rate REAL,
    cadence REAL,
    power_w REAL,
    speed_mps REAL,
    elevation_m REAL,
    distance_m REAL,
    latitude REAL,
    longitude REAL,
    PRIMARY KEY (activity_id, tick)
);

CREATE TABLE IF NOT EXISTS user_profile (
    profile_type TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hr_zones (
    sport TEXT PRIMARY KEY,
    training_method TEXT,
    zone1_min REAL,
    zone1_max REAL,
    zone2_min REAL,
    zone2_max REAL,
    zone3_min REAL,
    zone3_max REAL,
    zone4_min REAL,
    zone4_max REAL,
    zone5_min REAL,
    zone5_max REAL,
    max_hr_used REAL,
    resting_hr_used REAL,
    lactate_threshold_hr_used REAL,
    resting_hr_auto_update INTEGER,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS race_predictions (
    calendar_date TEXT PRIMARY KEY,
    time_5k_min REAL,
    time_10k_min REAL,
    time_half_marathon_min REAL,
    time_marathon_min REAL,
    fetched_at TEXT
);
"""


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    """Add any missing columns to a table (idempotent schema evolution)."""
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for col in columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} NUMERIC")


class Database:
    """SQLite storage for raw Garmin data + typed daily projections.

    The ``metrics`` table stores the raw payloads for every day-keyed data
    type. ``daily_metrics`` holds the parsed scalar projection (one wide row
    per calendar date); ``activities`` stores raw activity summaries + details
    keyed by activity id.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        _ensure_columns(self.conn, "activities", ["details_json", "details_fetched_at"])
        _ensure_columns(self.conn, "activities", ["weather_json", "weather_fetched_at"])
        _ensure_columns(self.conn, "activity_summaries", list(self._ACTIVITY_SUMMARY_COLUMNS))
        self.conn.commit()

    # -- Generic raw storage -------------------------------------------------

    def upsert_metric(self, data_type: str, payload: dict[str, Any]) -> None:
        """Store the raw payload for one data type on one date (upsert)."""
        self.conn.execute(
            """
            INSERT INTO metrics (data_type, calendar_date, raw_json, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(data_type, calendar_date) DO UPDATE SET
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (
                data_type,
                payload["calendarDate"],
                json.dumps(payload),
                payload.get("fetched_at"),
            ),
        )
        self.conn.commit()

    def stored_dates(self, data_type: str) -> set[str]:
        """Calendar dates already stored for a data type."""
        return {
            row["calendar_date"]
            for row in self.conn.execute(
                "SELECT calendar_date FROM metrics WHERE data_type = ?", (data_type,)
            )
        }

    def stored_fetches(self, data_type: str) -> dict[str, str]:
        """Map each stored calendar date to its fetched_at timestamp."""
        return {
            row["calendar_date"]: row["fetched_at"]
            for row in self.conn.execute(
                "SELECT calendar_date, fetched_at FROM metrics WHERE data_type = ?",
                (data_type,),
            )
        }

    def max_calendar_date(self, data_type: str) -> str | None:
        """Highest stored calendar date for a data type, or None."""
        return self.conn.execute(
            "SELECT MAX(calendar_date) AS max_date FROM metrics WHERE data_type = ?",
            (data_type,),
        ).fetchone()["max_date"]

    # -- Activities -----------------------------------------------------------

    def upsert_activity(self, payload: dict[str, Any]) -> None:
        activity = payload.get("activity") or payload
        self.conn.execute(
            """
            INSERT INTO activities (activity_id, raw_json, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(activity_id) DO UPDATE SET
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (
                activity["activityId"],
                json.dumps(payload),
                payload.get("fetched_at"),
            ),
        )
        self.conn.commit()

    def stored_activity_ids(self) -> set[int]:
        return {
            row["activity_id"]
            for row in self.conn.execute("SELECT activity_id FROM activities")
        }

    def set_activity_details(
        self, activity_id: int, details: dict[str, Any], fetched_at: str | None = None
    ) -> None:
        """Store the raw details payload for an activity (upsert)."""
        self.conn.execute(
            "UPDATE activities SET details_json = ?, details_fetched_at = ? "
            "WHERE activity_id = ?",
            (json.dumps(details), fetched_at, activity_id),
        )
        self.conn.commit()

    def set_activity_weather(
        self, activity_id: int, weather: dict[str, Any], fetched_at: str | None = None
    ) -> None:
        """Store the raw weather payload for an activity (upsert)."""
        self.conn.execute(
            "UPDATE activities SET weather_json = ?, weather_fetched_at = ? "
            "WHERE activity_id = ?",
            (json.dumps(weather), fetched_at, activity_id),
        )
        self.conn.commit()

    def activities_missing_weather(self) -> list[int]:
        """Activity ids that have no stored weather payload yet."""
        rows = self.conn.execute(
            "SELECT activity_id FROM activities "
            "WHERE weather_json IS NULL OR weather_json = ''"
        ).fetchall()
        return [r["activity_id"] for r in rows]

    def close(self) -> None:
        self.conn.close()

    # -- Daily metrics --------------------------------------------------------

    def merge_daily(
        self, calendar_date: str, values: dict[str, Any],
        fetched_at: str | None = None,
    ) -> None:
        """Merge parsed scalar values into the daily_metrics row for a date.

        Non-null values are written (upserted) keyed by column name. Columns
        not present in every row are created on demand. Existing values for
        the same date/column are overwritten.
        """
        if not values:
            return
        _ensure_columns(self.conn, "daily_metrics", list(values))
        exists = self.conn.execute(
            "SELECT 1 FROM daily_metrics WHERE calendar_date = ?", (calendar_date,)
        ).fetchone()
        if not exists:
            self.conn.execute(
                "INSERT INTO daily_metrics (calendar_date) VALUES (?)",
                (calendar_date,),
            )
        for col, value in values.items():
            self.conn.execute(
                f"UPDATE daily_metrics SET {col} = ? WHERE calendar_date = ?",
                (value, calendar_date),
            )
        if fetched_at:
            self.conn.execute(
                "UPDATE daily_metrics SET fetched_at = ? WHERE calendar_date = ?",
                (fetched_at, calendar_date),
            )
        self.conn.commit()

    def daily_dates(self) -> set[str]:
        return {
            row["calendar_date"]
            for row in self.conn.execute("SELECT calendar_date FROM daily_metrics")
        }

    # -- Activity summaries ---------------------------------------------------

    _ACTIVITY_SUMMARY_COLUMNS = (
        "activity_name", "activity_type", "start_time_local", "start_time_gmt",
        "start_date", "duration_hours", "elapsed_hours", "moving_hours",
        "distance_km", "avg_hr", "max_hr", "hr_time_zone_1_pct", "hr_time_zone_2_pct",
        "hr_time_zone_3_pct", "hr_time_zone_4_pct", "hr_time_zone_5_pct",         "calories",
        "training_load", "aerobic_training_effect", "anaerobic_training_effect",
        "moderate_intensity_minutes", "vigorous_intensity_minutes",
        "elevation_gain_m", "elevation_loss_m", "min_elevation_m",
        "max_elevation_m", "avg_speed_kmh", "max_speed_kmh",
        "min_respiration_rate", "avg_respiration_rate", "max_respiration_rate",
        "body_battery_change", "water_estimated_ml", "is_pr",
        "avg_cadence", "max_cadence", "avg_power_w", "max_power_w",
        "weather_temp_c", "weather_apparent_c", "weather_humidity",
        "weather_wind_kmh", "weather_wind_gust_kmh", "weather_station",
        "weather_description",
    )

    def upsert_activity_summary(
        self, activity_id: int, values: dict[str, Any], fetched_at: str | None = None
    ) -> None:
        """Insert or replace the curated activity summary row for an id.

        Only non-None values are written; the fixed column set guarantees the
        row always carries the shared summary fields regardless of activity
        type.
        """
        cols = [c for c in self._ACTIVITY_SUMMARY_COLUMNS if c in values]
        vals = tuple(values[c] for c in cols)
        assignments = ", ".join(f"{c} = excluded.{c}" for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        self.conn.execute(
            f"""
            INSERT INTO activity_summaries (activity_id, fetched_at, {', '.join(cols)})
            VALUES (?, ?, {placeholders})
            ON CONFLICT(activity_id) DO UPDATE SET
                {assignments}, fetched_at = excluded.fetched_at
            """,
            (activity_id, fetched_at, *vals),
        )
        self.conn.commit()

    _ACTIVITY_SERIES_COLUMNS = (
        "tick", "ts_ms", "heart_rate", "cadence", "power_w", "speed_mps",
        "elevation_m", "distance_m", "latitude", "longitude",
    )

    def replace_activity_series(
        self, activity_id: int, rows: list[dict[str, Any]]
    ) -> int:
        """Replace the per-tick detail series for one activity.

        The details payload is a fixed schema per activity; re-parsing rebuilds
        the whole tick series, so stale ticks are removed before the new ones
        are written. Returns the number of ticks written.
        """
        self.conn.execute(
            "DELETE FROM activity_detail_series WHERE activity_id = ?",
            (activity_id,),
        )
        present = {
            col for row in rows for col in self._ACTIVITY_SERIES_COLUMNS
            if row.get(col) is not None
        }
        cols = [c for c in self._ACTIVITY_SERIES_COLUMNS if c in present]
        placeholders = ", ".join("?" for _ in cols)
        insert = f"""
            INSERT INTO activity_detail_series (activity_id, {', '.join(cols)})
            VALUES (?, {placeholders})
        """
        written = 0
        if not cols or not rows:
            self.conn.commit()
            return written
        for row in rows:
            vals = tuple(row.get(c) for c in cols)
            self.conn.execute(insert, (activity_id, *vals))
            written += 1
        self.conn.commit()
        return written

    # -- User profile ------------------------------------------------------

    def upsert_profile(
        self, profile_type: str, raw: str, fetched_at: str | None = None
    ) -> None:
        """Store (or overwrite) one profile payload, keyed by profile_type."""
        self.conn.execute(
            """
            INSERT INTO user_profile (profile_type, raw_json, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(profile_type) DO UPDATE SET
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (profile_type, raw, fetched_at),
        )
        self.conn.commit()

    def get_profile(self, profile_type: str) -> dict[str, Any] | None:
        """Return (raw_json, fetched_at) for a stored profile, or None."""
        row = self.conn.execute(
            "SELECT raw_json, fetched_at FROM user_profile WHERE profile_type = ?",
            (profile_type,),
        ).fetchone()
        return dict(row) if row else None

    def replace_hr_zones(self, rows: list[dict[str, Any]]) -> int:
        """Replace the entire hr_zones table with a fresh per-sport snapshot.

        The heartRateZones endpoint returns one object per sport (DEFAULT,
        RUNNING, CYCLING, ...), each carrying that sport's zone floors. The set
        is replaced as a whole because it represents the current device
        configuration. Returns the number of rows written.
        """
        self.conn.execute("DELETE FROM hr_zones")
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO hr_zones (
                    sport, training_method,
                    zone1_min, zone1_max, zone2_min, zone2_max,
                    zone3_min, zone3_max, zone4_min, zone4_max,
                    zone5_min, zone5_max,
                    max_hr_used, resting_hr_used, lactate_threshold_hr_used,
                    resting_hr_auto_update, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["sport"], row.get("training_method"),
                    row.get("zone1_min"), row.get("zone1_max"),
                    row.get("zone2_min"), row.get("zone2_max"),
                    row.get("zone3_min"), row.get("zone3_max"),
                    row.get("zone4_min"), row.get("zone4_max"),
                    row.get("zone5_min"), row.get("zone5_max"),
                    row.get("max_hr_used"), row.get("resting_hr_used"),
                    row.get("lactate_threshold_hr_used"),
                    row.get("resting_hr_auto_update"), row.get("fetched_at"),
                ),
            )
        self.conn.commit()
        return len(rows)

    def replace_race_predictions(self, row: dict[str, Any] | None) -> int:
        """Replace the race-prediction snapshot with a fresh one.

        Race predictions are a current-fitness snapshot (like hr_zones), so the
        single stored row is overwritten entirely. Returns 1 when a row was
        written, else 0.
        """
        self.conn.execute("DELETE FROM race_predictions")
        if row is None:
            self.conn.commit()
            return 0
        self.conn.execute(
            """
            INSERT INTO race_predictions (
                calendar_date, time_5k_min, time_10k_min,
                time_half_marathon_min, time_marathon_min, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("calendar_date"), row.get("time_5k_min"),
                row.get("time_10k_min"), row.get("time_half_marathon_min"),
                row.get("time_marathon_min"), row.get("fetched_at"),
            ),
        )
        self.conn.commit()
        return 1

    # -- Sync state -----------------------------------------------------------

    def get_state(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM sync_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()


def init_db() -> Database:
    """Create/open the database using configuration."""
    return Database(load_config()["db_path"])