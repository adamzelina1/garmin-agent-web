from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta

from garminconnect import Garmin

from .config import load_config
from .datatypes import DataType, DEFAULT_TYPES, DATA_TYPES, resolve_types
from .db import Database

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fetch_local_date(fetched_at: str) -> date:
    """The local calendar date a fetch completed, regardless of tz in the iso.

    Naive timestamps (stored historically without a tz) were written from the
    caller's wall clock, so they're already local. Offset timestamps are
    converted to the machine's local zone.
    """
    dt = datetime.fromisoformat(fetched_at)
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.date()


def _nonfinal_dates(stored: dict[str, str]) -> set[str]:
    """Calendar dates whose stored copy was fetched on (or before) that day.

    Such days were captured while still in progress and are thus incomplete:
    the next sync re-fetches them once their data has settled. The last of
    these is the frontier the sync resumes from.
    """
    return {
        cal for cal, fetched in stored.items()
        if _fetch_local_date(fetched) <= date.fromisoformat(cal)
    }


_PROFILE_HR_ZONES = "hr_zones"
"""user_profile key for the configured heart-rate zone boundaries."""

_PROFILE_RACE_PREDICTIONS = "race_predictions"
"""user_profile key for the current race-prediction snapshot."""


class DataFetcher:
    """Authenticated client that fetches daily data of several types."""

    def __init__(self, email: str, password: str, tokens_path: str = "") -> None:
        self.client = Garmin(
            email=email,
            password=password,
            is_cn=False,
            prompt_mfa=lambda: input("MFA code: "),
        )
        self.tokens_path = tokens_path or None
        self.logged_in = False

    def login(self) -> None:
        logger.info("Logging in to Garmin Connect")
        self.client.login(self.tokens_path)
        self.logged_in = True
        logger.info("Logged in")

    def _ensure_logged_in(self) -> None:
        if not self.logged_in:
            self.login()

    def fetch(self, data_type: DataType, calendar_date: str) -> dict:
        """Fetch one date for one data type, tagging the fetch time."""
        self._ensure_logged_in()
        payload = data_type.fetch(self.client, calendar_date)
        if not isinstance(payload, dict):
            payload = {"calendarDate": calendar_date, "value": payload}
        payload.setdefault("calendarDate", calendar_date)
        payload["fetched_at"] = _now_iso()
        return payload

    def fetch_activities(self, start: date, end: date, db: Database) -> int:
        """Fetch activity summaries + details in [start, end], skipping known ids.

        For a new activity, the per-activity detail payload is fetched before the
        summary is stored. If the details fetch fails, the activity is skipped
        entirely (no summary-only row), so every stored activity always carries
        both the summary and its details. Returns the number of activities
        stored.
        """
        self._ensure_logged_in()
        summaries = self.client.get_activities_by_date(
            start.isoformat(), end.isoformat()
        )
        stored = db.stored_activity_ids()
        fetched = 0
        for item in summaries:
            activity = item.get("activity") or item
            activity_id = activity.get("activityId")
            if activity_id is None:
                logger.warning("Activity without activityId, skipping: %s", item)
                continue
            if activity_id in stored:
                continue
            try:
                details = self.client.get_activity_details(activity_id)
            except Exception:
                logger.exception(
                    "Failed to fetch details for activity %s, skipping", activity_id
                )
                continue
            payload = dict(item)
            payload["fetched_at"] = _now_iso()
            db.upsert_activity(payload)
            db.set_activity_details(activity_id, details, _now_iso())
            self._fetch_activity_weather(activity_id, db)
            stored.add(activity_id)
            fetched += 1
            logger.info(
                "Stored activity %s (%s, %s) with details",
                activity_id, activity.get("activityName"), activity.get("startTimeLocal"),
            )
        return fetched

    def _fetch_activity_weather(self, activity_id: int, db: Database) -> bool:
        """Best-effort: fetch and store one activity's observed weather.

        Weather is an enrichment, not part of the atomic summary+details store:
        a failure only logs a warning and leaves the activity without weather
        (it can be backfilled later). Returns True when weather was stored.
        """
        try:
            weather = self.client.get_activity_weather(str(activity_id))
        except Exception:
            logger.debug(
                "No weather for activity %s (will backfill later)", activity_id
            )
            return False
        if not isinstance(weather, dict):
            return False
        db.set_activity_weather(activity_id, weather, _now_iso())
        return True

    def backfill_activity_weather(self, db: Database) -> int:
        """Fetch weather for every activity that still lacks it.

        Returns the number of activities whose weather was newly stored.
        """
        self._ensure_logged_in()
        missing = db.activities_missing_weather()
        fetched = 0
        for activity_id in missing:
            if self._fetch_activity_weather(activity_id, db):
                fetched += 1
                logger.info("Backfilled weather for activity %s", activity_id)
        return fetched

    def fetch_profile(self, db: Database) -> bool:
        """Fetch the configured heart-rate zones and store them raw.

        HR zone boundaries are a device/user profile setting, not per-date
        data, so this is a single no-arg call stored under the fixed
        ``user_profile`` key ``hr_zones`` (overwriting the previous snapshot).
        Returns True when a payload was stored.
        """
        self._ensure_logged_in()
        payload = self.client.get_heart_rate_zones()
        if payload is None:
            logger.warning("get_heart_rate_zones returned nothing; skipping")
            return False
        db.upsert_profile(_PROFILE_HR_ZONES, json.dumps(payload), _now_iso())
        logger.info("Stored heart-rate zone profile (%d sport(s))", len(payload))
        return True

    def fetch_race_predictions(self, db: Database) -> bool:
        """Fetch the current race-prediction snapshot and store it raw.

        Race predictions are a current-fitness snapshot (like HR zones), so this
        is a single no-arg call stored under the fixed ``user_profile`` key
        ``race_predictions`` (overwriting the previous snapshot). Returns True
        when a payload was stored.
        """
        self._ensure_logged_in()
        payload = self.client.get_race_predictions()
        if payload is None:
            logger.warning("get_race_predictions returned nothing; skipping")
            return False
        db.upsert_profile(_PROFILE_RACE_PREDICTIONS, json.dumps(payload), _now_iso())
        logger.info(
            "Stored race predictions (as of %s)", payload.get("calendarDate")
        )
        return True

    def fetch_range(
        self, data_type: DataType, start: date, end: date, db: Database,
        refetch: set[str] | None = None,
    ) -> int:
        """Fetch one data type across [start, end], skipping stored dates.

        Stored dates are skipped unless their copy is incomplete (in ``refetch``),
        so a day fetched while still in progress gets re-fetched (and overwritten)
        once its data settles. When ``refetch`` is None, every stored date is
        skipped (pure incremental behavior). Returns the count of dates newly
        fetched (including re-fetched incomplete days).
        """
        stored = db.stored_dates(data_type.name)
        fetched = 0
        day = start
        while day <= end:
            day_str = day.isoformat()
            already = day_str in stored
            if already and (refetch is None or day_str not in refetch):
                logger.debug("Skipping %s %s (already stored)", data_type.name, day_str)
                day += timedelta(days=1)
                continue
            try:
                payload = self.fetch(data_type, day_str)
                db.upsert_metric(data_type.name, payload)
                stored.add(day_str)
                fetched += 1
                logger.info(
                    "%s %s for %s",
                    "Refetched" if already else "Fetched",
                    data_type.name, day_str,
                )
            except Exception:
                logger.exception("Failed to fetch %s for %s", data_type.name, day_str)
            day += timedelta(days=1)
        return fetched


def sync_data(
    *,
    types: list[DataType] | None = None,
    start: date | None = None,
    end: date | None = None,
    include_activities: bool = True,
    include_profile: bool = True,
    parse: bool = True,
) -> dict[str, int]:
    """Incremental sync of one or more data types from Garmin into SQLite.

    For each type, fetches every missing date from the resolved start date to
    ``end`` (default today). If ``include_activities`` (default True), also
    backfills activity summaries for the same window. If ``include_profile``
    (default True), refreshes the heart-rate zone profile snapshot. After
    fetching, the newly stored raw metrics are parsed into ``daily_metrics``
    (unless ``parse`` is False). Returns {type_name: dates_fetched} plus
    optional {"activities": count} and {"hr_zones": rows}.
    """
    config = load_config()
    db = Database(config["db_path"])
    fetcher = DataFetcher(config["email"], config["password"], config["tokens_path"])
    end = end or date.today()
    freeze_days = config["activity_freeze_days"]
    configured_start = (
        date.fromisoformat(config["start_date"]) if config["start_date"] else None
    )

    counts: dict[str, int] = {}
    selected_types = types or _configured_types(config)
    try:
        fetcher.login()
        for data_type in selected_types:
            # Resumes from the oldest still-incomplete day so each gets
            # re-fetched once its data has settled.
            nonfinal = _nonfinal_dates(db.stored_fetches(data_type.name))
            sync_start = _resolve_start_date(
                db, data_type.name, configured_start, nonfinal or None
            )
            dates = fetcher.fetch_range(data_type, sync_start, end, db, nonfinal)
            counts[data_type.name] = dates
            logger.info(
                "synced %s: fetched %d missing dates (%s to %s)",
                data_type.name, dates, sync_start, end,
            )
        if include_activities:
            act_start = _resolve_activity_start(db, configured_start, freeze_days, end)
            counts["activities"] = fetcher.fetch_activities(act_start, end, db)
            mark_activities_finalized(db, end, freeze_days)
            logger.info(
                "synced activities: fetched %d new (%s to %s)",
                counts["activities"], act_start, end,
            )
            weather_count = fetcher.backfill_activity_weather(db)
            if weather_count:
                logger.info("backfilled weather for %d activity(ies)", weather_count)
        if include_profile:
            fetcher.fetch_profile(db)
            fetcher.fetch_race_predictions(db)
            logger.info("synced user profile: heart-rate zones + race predictions")
        if parse:
            from .parser import (
                build_activity_details,
                build_activity_summaries,
                build_activity_weather,
                build_daily_rows,
                build_hr_zones,
                build_race_predictions,
            )

            parsed = build_daily_rows(db, [t.name for t in selected_types])
            logger.info(
                "parsed %d date/type rows into daily_metrics (%s).",
                sum(parsed.values()), ", ".join(parsed) or "none",
            )
            if include_activities:
                act_parsed = build_activity_summaries(db)
                detail_parsed = build_activity_details(db)
                weather_parsed = build_activity_weather(db)
                logger.info(
                    "parsed %d activities into activity_summaries (+ %d detail "
                    "series, %d with weather) across %.0f ticks.",
                    act_parsed.get("activities", 0),
                    detail_parsed.get("activities", 0),
                    weather_parsed.get("activities", 0),
                    detail_parsed.get("series", 0),
                )
            if include_profile:
                zone_parsed = build_hr_zones(db)
                counts["hr_zones"] = zone_parsed.get("hr_zones", 0)
                race_parsed = build_race_predictions(db)
                counts["race_predictions"] = race_parsed.get("race_predictions", 0)
                logger.info(
                    "parsed %d sport(s) into hr_zones and %d row(s) into "
                    "race_predictions.",
                    zone_parsed.get("hr_zones", 0),
                    race_parsed.get("race_predictions", 0),
                )
    finally:
        db.close()
    return counts


def _configured_types(config: dict[str, str]) -> list[DataType]:
    """Resolve the data-type set for a sync.

    All registered types are used, minus ``GARMIN_EXCLUDED_DATA_TYPES``.
    Unknown names are logged and skipped so a misconfigured .env doesn't
    silently disable the sync.
    """
    excluded = {
        n.strip()
        for n in (config.get("excluded_data_types") or "").split(",")
        if n.strip()
    }

    resolved = [
        data_type
        for data_type in DEFAULT_TYPES
        if data_type.name not in excluded
    ]
    return resolved or list(DEFAULT_TYPES)


_ACTIVITY_FINALIZED_KEY = "activities_last_finalized"
"""sync_state key for the newest activity date considered settled.

Days up to and including this date are treated as finalized and never
re-synced; days after it are re-scanned each run so late/retroactive
uploads within the freeze buffer are still caught.
"""


def _resolve_activity_start(
    db: Database, configured_start: date | None, freeze_days: int, end: date
) -> date:
    """Earliest date to re-scan activities for a run.

    Steady state (a finalization marker exists): re-scan only the rolling
    freeze buffer, i.e. ``end - freeze_days + 1``, bounded below by the
    finalized marker. First run: full backfill from ``configured_start``,
    else just the rolling window.
    """
    finalized = db.get_state(_ACTIVITY_FINALIZED_KEY)
    if finalized:
        return _rolling_start(end, freeze_days, date.fromisoformat(finalized))
    if configured_start:
        return configured_start
    return _rolling_start(end, freeze_days, None)


def _rolling_start(end: date, freeze_days: int, finalized: date | None) -> date:
    """Rolling window start: ``end - freeze_days + 1``, never before finalized."""
    start = end - timedelta(days=freeze_days - 1)
    if finalized and finalized >= start:
        return finalized + timedelta(days=1)
    return start


def mark_activities_finalized(db: Database, end: date, freeze_days: int) -> None:
    """After a successful sync, freeze everything older than the buffer."""
    if freeze_days > 0:
        finalized = end - timedelta(days=freeze_days)
        db.set_state(_ACTIVITY_FINALIZED_KEY, finalized.isoformat())


def _resolve_start_date(
    db: Database, data_type: str, configured_start: date | None,
    nonfinal: set[str] | None = None,
) -> date:
    """Earliest date to sync for a type: config > last incomplete > max + 1 > today."""
    if configured_start:
        return configured_start
    if nonfinal:
        return min(date.fromisoformat(d) for d in nonfinal)
    max_stored = db.max_calendar_date(data_type)
    if max_stored:
        return date.fromisoformat(max_stored) + timedelta(days=1)
    return date.today()


# --- Backward-compatible heart-rate helpers ---------------------------------

_HEART_RATE = DATA_TYPES["heart_rate"]


class HeartRateFetcher(DataFetcher):
    """Back-compat alias: fetch only heart rate."""

    def fetch_daily_hr(self, calendar_date: str) -> dict:
        return self.fetch(_HEART_RATE, calendar_date)


def sync_heart_rates() -> dict[str, int]:
    """Back-compat: sync only heart_rate data."""
    return sync_data(types=[_HEART_RATE])


def fetch_heart_rates(
    start: date | None = None,
    end: date | None = None,
    days_back: int = 7,
) -> dict[str, int]:
    """Back-compat: sync heart rate for an explicit range."""
    end = end or date.today()
    start = start or (end - timedelta(days=days_back - 1))
    return sync_data(types=[_HEART_RATE], start=start, end=end)


def main() -> None:
    import argparse

    options = ", ".join(
        sorted(d.name for d in DEFAULT_TYPES)
        + ["activities", "profile", "race_predictions"]
    )
    parser = argparse.ArgumentParser(
        description="Fetch daily Garmin Connect data into SQLite. By default "
        "incrementally syncs all registered data types and activities "
        "(backfill from GARMIN_START_DATE to today, skipping stored data)."
    )
    parser.add_argument(
        "--type", dest="types", action="append", metavar="NAME",
        help=f"Data type to sync (repeatable). Options: {options}",
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), type=date.fromisoformat,
        help="Fetch an explicit date range START END (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--list-types", action="store_true", help="List available data types and exit",
    )
    parser.add_argument(
        "--parse", dest="parse_types", action="store", nargs="*", metavar="NAME",
        help="Parse stored raw metrics into daily_metrics (optional names; "
        "default: all parseable types). Incremental: only rows whose raw "
        "payload changed since last parse; combine with --full for a rebuild.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="With --parse: force a full re-parse of every stored row, "
        "ignoring incremental change markers.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.list_types:
        print("Available data types:")
        for data_type in DEFAULT_TYPES:
            print(f"  {data_type.name}")
        print("  activities")
        print("  profile (hr_zones + race_predictions)")
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.parse_types is not None:
        from .parser import (
            PARSERS,
            build_activity_details,
            build_activity_summaries,
            build_activity_weather,
            build_daily_rows,
            build_hr_zones,
            build_race_predictions,
        )

        names = [t.strip().lower() for t in args.parse_types]
        want_activities = "activities" in names or "activity" in names
        want_weather = "weather" in names
        want_profile = (
            "profile" in names or "hr_zones" in names or "race_predictions" in names
        )
        day_names = [
            n for n in names
            if n not in (
                "activities", "activity", "profile", "hr_zones",
                "race_predictions", "weather",
            )
        ]
        unknown = [n for n in day_names if n not in PARSERS]
        options = ", ".join(
            sorted(PARSERS) + ["activities", "profile", "weather"]
        )
        if unknown:
            parser.error(
                f"Unknown parse type(s): {', '.join(unknown)}. Options: {options}"
            )
        db = Database(load_config()["db_path"])
        try:
            counts: dict[str, int] = {}
            force = args.full
            if want_activities or not names:
                counts.update(build_activity_summaries(db, force=force))
            if want_activities or not names:
                counts.update(build_activity_details(db, force=force))
            if want_weather or want_activities or not names:
                counts.update(build_activity_weather(db, force=force))
            if want_profile or not names:
                counts.update(build_hr_zones(db))
            if want_profile or not names:
                counts.update(build_race_predictions(db))
            counts.update(build_daily_rows(db, day_names or None, force=force))
        finally:
            db.close()
        total = sum(counts.values())
        logger.info("Parsed %d rows (%s).", total, ", ".join(counts) or "none")
        return

    arg_types = [t.strip().lower() for t in (args.types or [])]
    want_activities = "activities" in arg_types
    want_profile = (
        "profile" in arg_types or "hr_zones" in arg_types
        or "race_predictions" in arg_types
    )
    type_names = [
        t for t in arg_types
        if t not in ("activities", "profile", "hr_zones", "race_predictions")
    ]
    types = resolve_types(type_names or None)

    include_activities = want_activities or not type_names or want_profile
    if args.range:
        counts = sync_data(
            types=types, start=args.range[0], end=args.range[1],
            include_activities=include_activities,
            include_profile=want_profile or not type_names,
        )
    else:
        counts = sync_data(
            types=types,
            include_activities=include_activities,
            include_profile=want_profile or not type_names,
        )

    total = sum(counts.values())
    logger.info(
        "Done. Fetched %d new rows across %s.",
        total, ", ".join(counts) or "none",
    )


if __name__ == "__main__":
    main()