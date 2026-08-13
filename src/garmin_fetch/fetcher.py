from __future__ import annotations

import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Callable

from garminconnect import Garmin

from .config import load_config
from .datatypes import DataType, DEFAULT_TYPES, DATA_TYPES, resolve_types
from .db import Database

logger = logging.getLogger(__name__)

#: garminconnect login strategy that must be skipped. The ``widget+cffi``
#: strategy scrapes an HTML widget page and cannot actually confirm whether
#: Garmin delivered an OTP, so it can falsely report "MFA required" for
#: accounts that have no two-step verification — and during a Cloudflare/429
#: window that false MFA gets shelved and returned as if real, blocking
#: signup. The API-based strategies (mobile/portal) detect genuine MFA
#: reliably, so skipping the widget flow loses nothing for real MFA accounts.
_SKIP_GARMIN_STRATEGIES = ["widget+cffi"]


def _no_interactive_mfa() -> str:
    """Stand-in for garminconnect's MFA prompt on a headless server.

    Interactive console MFA is not supported in the multi-user model — if a
    Garmin login hits MFA, it fails loudly rather than blocking on stdin.
    """
    raise RuntimeError(
        "Garmin MFA is required but this process cannot prompt for it; "
        "temporarily disable two-step verification in Garmin account settings"
    )


def _open_db(config: dict[str, str], user_id: int | None = None) -> Database:
    """Open the per-user database (Postgres via GARMIN_DB_URL)."""
    uid = user_id or config.get("local_user_id") or 1
    return Database.from_url(config["db_url"], user_id=uid)


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

_PROFILE_GEAR = "gear"
"""user_profile key for the current gear snapshot (bikes, shoes, ...)."""

_PROFILE_POWER_ZONES = "power_zones"
"""user_profile key for the configured cycling power-zone boundaries."""

_PROFILE_DEVICES = "devices"
"""user_profile key for the current device list (+ per-device settings)."""


class DataFetcher:
    """Authenticated client that fetches daily data of several types."""

    def __init__(
        self,
        email: str,
        password: str,
        tokens_path: str = "",
        *,
        sleep_sec: float = 0.0,
        prompt_mfa: Callable[[], str] | None = None,
        return_on_mfa: bool = False,
    ) -> None:
        self.sleep_sec = max(0.0, sleep_sec)
        self.client = Garmin(
            email=email,
            password=password,
            is_cn=False,
            prompt_mfa=prompt_mfa or _no_interactive_mfa,
            return_on_mfa=return_on_mfa,
        )
        self.client.client.skip_strategies = list(_SKIP_GARMIN_STRATEGIES)
        self.tokens_path = tokens_path or None
        self.logged_in = False

    def _pace(self) -> None:
        """Sleep before an API call so a slow sync throttles request rate.

        ``GARMIN_FETCH_SLEEP_SEC`` is a deliberate pacing knob: on a big
        backfill it spaces every Garmin call apart, which keeps a single
        account (or server IP) well under Garmin's informal rate limits. In
        steady state the incremental sync only fetches the missing frontier, so
        the added delay is small.
        """
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

    def login(self) -> None:
        logger.info("Logging in to Garmin Connect")
        self.client.login(self.tokens_path)
        self.logged_in = True
        logger.info("Logged in")

    def get_token_string(self) -> str:
        """Serialize the current Garmin auth tokens (for encrypted storage)."""
        return self.client.dumps()

    def is_authenticated(self) -> bool:
        client = self.client
        return bool(
            getattr(client, "di_token", None)
            or getattr(client, "jwt_web", None)
        )

    def _ensure_logged_in(self) -> None:
        if not self.logged_in:
            self.login()

    def fetch(self, data_type: DataType, calendar_date: str) -> dict:
        """Fetch one date for one data type, tagging the fetch time."""
        self._ensure_logged_in()
        self._pace()
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
        self._pace()
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
                self._pace()
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
            self._fetch_activity_splits(activity_id, db)
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
            self._pace()
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
        self._pace()
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
        self._pace()
        payload = self.client.get_race_predictions()
        if payload is None:
            logger.warning("get_race_predictions returned nothing; skipping")
            return False
        db.upsert_profile(_PROFILE_RACE_PREDICTIONS, json.dumps(payload), _now_iso())
        logger.info(
            "Stored race predictions (as of %s)", payload.get("calendarDate")
        )
        return True

    def fetch_gear(self, db: Database) -> bool:
        """Fetch the current gear snapshot (bikes, shoes, ...) and store it raw.

        Gear is a current "what's in your garage" snapshot (like HR zones), so
        this is stored under the fixed ``user_profile`` key ``gear`` (overwriting
        the previous snapshot). The gear *list* only carries identity fields
        (type/make/model/name/status) and its service distance — the actual
        cumulative stats (total distance, max speed, max power, activity count,
        last activity date) live on the per-gear stats endpoint, so each item is
        merged with ``get_gear_stats(uuid)`` before storing. A failing stats
        call degrades to the list fields only. Returns True when a payload was
        stored.
        """
        self._ensure_logged_in()
        self._pace()
        device = self.client.get_device_last_used()
        profile_number = (
            device.get("userProfileNumber") if isinstance(device, dict) else None
        )
        if not profile_number:
            logger.warning(
                "no userProfileNumber from get_device_last_used; skipping gear"
            )
            return False
        self._pace()
        payload = self.client.get_gear(profile_number)
        if not payload:
            logger.warning("get_gear returned nothing; skipping")
            return False
        items = (
            payload.get("gearList") or payload
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(items, list):
            logger.warning("unexpected gear payload; skipping")
            return False
        merged = []
        for item in items:
            if not isinstance(item, dict):
                merged.append(item)
                continue
            entry = dict(item)
            gear_uuid = entry.get("uuid")
            if gear_uuid:
                try:
                    self._pace()
                    stats = self.client.get_gear_stats(gear_uuid)
                    if isinstance(stats, dict):
                        entry.update(stats)
                except Exception:
                    logger.warning(
                        "gear stats failed for %s; keeping list fields",
                        gear_uuid,
                    )
            merged.append(entry)
        db.upsert_profile(_PROFILE_GEAR, json.dumps(merged), _now_iso())
        logger.info("Stored gear snapshot (%d item(s))", len(merged))
        return True

    def fetch_power_zones(self, db: Database) -> bool:
        """Fetch the configured power zones and store them raw.

        Power zones are a device/profile setting (like HR zones): a single
        no-arg call stored under the fixed ``user_profile`` key ``power_zones``
        (overwriting the previous snapshot). Returns True when a payload was
        stored.
        """
        self._ensure_logged_in()
        self._pace()
        payload = self.client.get_power_zones()
        if not payload:
            logger.warning("get_power_zones returned nothing; skipping")
            return False
        db.upsert_profile(_PROFILE_POWER_ZONES, json.dumps(payload), _now_iso())
        logger.info(
            "Stored power-zone profile (%d sport(s))",
            len(payload) if isinstance(payload, list) else 1,
        )
        return True

    def fetch_devices(self, db: Database) -> bool:
        """Fetch the current device list and store it raw.

        Only the identity info is kept (model name) — enough for the agent to
        say what the user records on, without the per-device
        ``get_device_settings`` calls (which would add N more Garmin requests
        per sync). Stored under the fixed ``user_profile`` key ``devices``
        (overwriting the previous snapshot). Returns True when a payload was
        stored.
        """
        self._ensure_logged_in()
        self._pace()
        devices = self.client.get_devices()
        if not isinstance(devices, list) or not devices:
            logger.warning("get_devices returned nothing; skipping")
            return False
        merged: list[dict[str, Any]] = []
        for device in devices:
            if isinstance(device, dict):
                merged.append(dict(device))
        db.upsert_profile(_PROFILE_DEVICES, json.dumps(merged), _now_iso())
        logger.info("Stored device snapshot (%d device(s))", len(merged))
        return True

    def _fetch_activity_splits(self, activity_id: int, db: Database) -> bool:
        """Best-effort: fetch and store one activity's splits payload.

        Splits are an enrichment (like weather): a failure only logs a debug
        line and leaves the activity without splits (it can be backfilled
        later). Returns True when splits were stored.
        """
        try:
            self._pace()
            splits = self.client.get_activity_splits(str(activity_id))
        except Exception:
            logger.debug("No splits for activity %s (will backfill later)", activity_id)
            return False
        if not isinstance(splits, dict):
            return False
        db.set_activity_splits(activity_id, splits, _now_iso())
        return True

    def backfill_activity_splits(self, db: Database) -> int:
        """Fetch splits for every activity that still lacks them.

        Returns the number of activities whose splits were newly stored.
        """
        self._ensure_logged_in()
        missing = db.activities_missing_splits()
        fetched = 0
        for activity_id in missing:
            if self._fetch_activity_splits(activity_id, db):
                fetched += 1
                logger.info("Backfilled splits for activity %s", activity_id)
        return fetched

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
    user_id: int = 1,
    config: dict[str, str] | None = None,
    types: list[DataType] | None = None,
    start: date | None = None,
    end: date | None = None,
    include_activities: bool = True,
    include_profile: bool = True,
    parse: bool = True,
    full_reparse: bool = False,
    db: Database | None = None,
    fetcher: DataFetcher | None = None,
) -> dict[str, int]:
    """Incremental sync of one or more data types from Garmin into Postgres.

    Runs for a single ``user_id``: every row is written with that account and
    the connection applies ``SET app.user_id`` so RLS scopes all writes. For
    each type, fetches every missing date from the resolved start date to
    ``end`` (default today). If ``include_activities`` (default True), also
    backfills activity summaries for the same window. If ``include_profile``
    (default True), refreshes the heart-rate zone profile snapshot. After
    fetching, the newly stored raw metrics are parsed into ``daily_metrics``
    (unless ``parse`` is False).
    Returns {type_name: dates_fetched} plus optional {"activities": count} and
    {"hr_zones": rows}, {"race_predictions": rows}, {"gear": rows}.

    ``full_reparse`` (default False) forces every stored row to be re-parsed
    from raw, ignoring the incremental ``parsed_at`` markers — useful after a
    parser bug fix to rebuild the typed tables.

    ``config`` defaults to ``load_config()``. Pass ``db``/``fetcher`` to drive
    the sync from the server layer with per-user credentials (already bound to
    the account); otherwise they are built from ``config``.
    """
    config = config or load_config()
    db = db or _open_db(config, user_id)
    fetcher = fetcher or DataFetcher(
        config["email"], config["password"], config["tokens_path"],
        sleep_sec=config.get("fetch_sleep_sec", 0.0),
    )
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
            splits_count = fetcher.backfill_activity_splits(db)
            if splits_count:
                logger.info("backfilled splits for %d activity(ies)", splits_count)
        if include_profile:
            fetcher.fetch_profile(db)
            fetcher.fetch_race_predictions(db)
            fetcher.fetch_gear(db)
            fetcher.fetch_power_zones(db)
            fetcher.fetch_devices(db)
            logger.info(
                "synced user profile: heart-rate zones + race predictions + gear "
                "+ power zones + devices"
            )
        if parse:
            from .parser import (
                build_activity_details,
                build_activity_splits,
                build_activity_summaries,
                build_activity_weather,
                build_daily_rows,
                build_devices,
                build_gear,
                build_hr_zones,
                build_power_zones,
                build_race_predictions,
            )

            parsed = build_daily_rows(db, [t.name for t in selected_types], force=full_reparse)
            logger.info(
                "parsed %d date/type rows into daily_metrics (%s).",
                sum(parsed.values()), ", ".join(parsed) or "none",
            )
            if include_activities:
                act_parsed = build_activity_summaries(db, force=full_reparse)
                detail_parsed = build_activity_details(db, force=full_reparse)
                weather_parsed = build_activity_weather(db, force=full_reparse)
                splits_parsed = build_activity_splits(db, force=full_reparse)
                logger.info(
                    "parsed %d activities into activity_summaries (+ %d detail "
                    "series, %d with weather, %d with splits) across %.0f ticks.",
                    act_parsed.get("activities", 0),
                    detail_parsed.get("activities", 0),
                    weather_parsed.get("activities", 0),
                    splits_parsed.get("activities", 0),
                    detail_parsed.get("series", 0),
                )
            if include_profile:
                zone_parsed = build_hr_zones(db)
                counts["hr_zones"] = zone_parsed.get("hr_zones", 0)
                race_parsed = build_race_predictions(db)
                counts["race_predictions"] = race_parsed.get("race_predictions", 0)
                gear_parsed = build_gear(db)
                counts["gear"] = gear_parsed.get("gear", 0)
                power_parsed = build_power_zones(db)
                counts["power_zones"] = power_parsed.get("power_zones", 0)
                device_parsed = build_devices(db)
                counts["devices"] = device_parsed.get("devices", 0)
                logger.info(
                    "parsed %d sport(s) into hr_zones, %d row(s) into "
                    "race_predictions, %d item(s) into gear, %d sport(s) into "
                    "power_zones, and %d device(s) into devices.",
                    zone_parsed.get("hr_zones", 0),
                    race_parsed.get("race_predictions", 0),
                    gear_parsed.get("gear", 0),
                    power_parsed.get("power_zones", 0),
                    device_parsed.get("devices", 0),
                )
            # Derived daily metrics (training readiness, ACWR, ...) are
            # recomputed from the now-parsed tables once per sync and stored
            # for the AI agent.
            from .derived import build_derived

            derived = build_derived(db)
            counts.update(derived)
            logger.info(
                "rebuilt derived metrics: readiness %s, acwr %s, %s rows",
                derived.get("readiness", 0),
                derived.get("acwr", 0),
                derived.get("derived", 0),
            )
        if types is None and start is None:
            # Config-driven full sync only: apply the account's start-date and
            # exclusion choices to already-stored data, so changing them in the
            # config actually removes rows (not just stops future collection).
            # An explicit --range fetch passes start/end and is left alone.
            from .datatypes import DAILY_TYPES

            enabled = {t.name for t in selected_types}
            excluded = [n for n in DAILY_TYPES if n not in enabled]
            if configured_start:
                pruned = db.prune_dates_before(configured_start.isoformat())
                if pruned:
                    logger.info(
                        "pruned %d row(s) before configured start date %s",
                        pruned, configured_start,
                    )
            if excluded:
                deleted = db.prune_excluded_types(excluded, enabled)
                logger.info(
                    "pruned excluded type(s): deleted %d raw row(s) for %s",
                    deleted, ", ".join(excluded),
                )
    finally:
        db.close()
    return counts


def _configured_types(config: dict[str, str]) -> list[DataType]:
    """Resolve the data-type set for a sync.

    All registered types are used, minus the per-account exclusion list that
    the server passes through ``config["excluded_data_types"]`` (there is no
    server-level env default). Unknown names are logged and skipped so a
    misconfigured exclusion list doesn't silently disable the sync.
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
        + ["activities", "profile", "race_predictions", "gear"]
    )
    parser = argparse.ArgumentParser(
        description="Fetch daily Garmin Connect data into Postgres. By default "
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
        print("  profile (hr_zones + race_predictions + gear)")
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.parse_types is not None:
        from .parser import (
            PARSERS,
            build_activity_details,
            build_activity_splits,
            build_activity_summaries,
            build_activity_weather,
            build_daily_rows,
            build_gear,
            build_hr_zones,
            build_race_predictions,
        )

        names = [t.strip().lower() for t in args.parse_types]
        want_activities = "activities" in names or "activity" in names
        want_weather = "weather" in names
        want_profile = (
            "profile" in names or "hr_zones" in names
            or "race_predictions" in names or "gear" in names
        )
        day_names = [
            n for n in names
            if n not in (
                "activities", "activity", "profile", "hr_zones",
                "race_predictions", "gear", "weather",
            )
        ]
        unknown = [n for n in day_names if n not in PARSERS]
        options = ", ".join(
            sorted(PARSERS) + ["activities", "profile", "gear", "weather"]
        )
        if unknown:
            parser.error(
                f"Unknown parse type(s): {', '.join(unknown)}. Options: {options}"
            )
        cfg = load_config()
        db = _open_db(cfg, cfg["local_user_id"])
        try:
            counts: dict[str, int] = {}
            force = args.full
            if want_activities or not names:
                counts.update(build_activity_summaries(db, force=force))
            if want_activities or not names:
                counts.update(build_activity_details(db, force=force))
            if want_weather or want_activities or not names:
                counts.update(build_activity_weather(db, force=force))
            if want_activities or not names:
                counts.update(build_activity_splits(db, force=force))
            if want_profile or not names:
                counts.update(build_hr_zones(db))
            if want_profile or not names:
                counts.update(build_race_predictions(db))
            if want_profile or not names:
                counts.update(build_gear(db))
            counts.update(build_daily_rows(db, day_names or None, force=force))
            from .derived import build_derived
            counts.update(build_derived(db))
        finally:
            db.close()
        total = sum(counts.values())
        logger.info("Parsed %d rows (%s).", total, ", ".join(counts) or "none")
        return

    arg_types = [t.strip().lower() for t in (args.types or [])]
    want_activities = "activities" in arg_types
    want_profile = (
        "profile" in arg_types or "hr_zones" in arg_types
        or "race_predictions" in arg_types or "gear" in arg_types
    )
    type_names = [
        t for t in arg_types
        if t not in ("activities", "profile", "hr_zones", "race_predictions", "gear")
    ]
    # None (no explicit --type) -> sync_data applies _configured_types, which
    # honours the per-account exclusion list from config. Passing the full
    # default list here would bypass the exclusion config.
    types = resolve_types(type_names) if type_names else None

    include_activities = want_activities or not type_names or want_profile
    uid = (load_config().get("local_user_id") or 1)
    if args.range:
        counts = sync_data(
            user_id=uid,
            types=types, start=args.range[0], end=args.range[1],
            include_activities=include_activities,
            include_profile=want_profile or not type_names,
        )
    else:
        counts = sync_data(
            user_id=uid,
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